# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hub quant method of the heterogeneous MoE (spec 6.2).

``HeteroMoEMethod`` is a mixin placed in front of the platform's W4A16 MoE
method (``CompressedTensorsWNA16MoEMethod`` with the gfx1100 backend, or
``CompressedTensorsWNA16RDNA2MoEMethod`` on gfx1030), so weight loading, repack
and the hub's fused MoE kernel stay exactly that method's. What changes:

* the layer holds a **slot pool**: the expert map manager sized the layer to the
  plan's hub-resident experts (``local_num_experts == n_slots``) and the layer's
  ``expert_map`` is the device ``slot_map[E]`` (slot or -1, int32);
* ``apply`` runs dispatch -> shared expert -> hub MoE (resident experts only,
  masked ids, ``-1`` skipped by the kernels) -> combine; the result is the
  island's partial sum and the runner's all-reduce (none in H1) does the rest;
* layers the plan marks hub-only (the MTP draft) and layers outside the plan
  keep the plain method: no ``hx`` call.

``maybe_wrap_hetero_method`` swaps the class of an already built method; it is
the only hook ``get_moe_method`` needs.
"""

import functools
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed.hetero_moe import runtime
from vllm.distributed.hetero_moe.plan import PlanError, plan_layer_id
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)

logger = init_logger(__name__)


@dataclass
class HeteroLayer:
    layer_name: str
    plan_layer: int
    rank: int  # TP/EP rank = index into the hub device list
    n_slots: int
    max_rows: int


class HeteroMoEMethod:
    """Mixin; see the module docstring. Never instantiated on its own."""

    _hetero: HeteroLayer

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        # The shared expert runs between dispatch and the hub MoE on the same
        # stream (spec 6.2 step 2); the runner must not fork it to an aux stream.
        return True

    def create_weights(
        self, layer: torch.nn.Module, num_experts: int, *args, **kwargs
    ) -> None:
        state = self._hetero
        if num_experts != state.n_slots:
            raise PlanError(
                f"hetero MoE layer {state.layer_name}: {num_experts} local experts, "
                f"the plan's slot pool has {state.n_slots}"
            )
        super().create_weights(layer, num_experts, *args, **kwargs)  # type: ignore[misc]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)  # type: ignore[misc]
        slot_map = layer.expert_map  # type: ignore[attr-defined]
        if (
            slot_map is None
            or slot_map.dtype != torch.int32
            or not slot_map.is_contiguous()
        ):
            raise PlanError(
                f"hetero MoE layer {self._hetero.layer_name}: expert_map must be the "
                "contiguous int32 slot_map"
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        state = self._hetero
        if x.dtype != torch.float16:
            raise TypeError(
                f"hetero MoE needs fp16 activations (the hub MoE and the hx wire "
                f"are fp16), got {x.dtype}"
            )
        inner_apply = super().apply  # type: ignore[misc]
        rt = runtime.get_runtime(
            rank=state.rank, hidden=x.shape[1], max_rows=state.max_rows, device=x.device
        )
        x = x.contiguous()
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()

        def shared() -> None:
            if shared_experts is not None:
                assert shared_experts_input is not None
                shared_experts(
                    shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
                )

        def hub_moe(hub_ids: torch.Tensor) -> torch.Tensor:
            # masked ids: the pairs that run on the servers (or nowhere) are -1
            return inner_apply(layer, x, topk_weights, hub_ids, None, None)

        return rt.apply_layer(
            state.plan_layer, x, topk_weights, topk_ids, hub_moe=hub_moe, shared=shared
        )


@functools.cache
def hetero_method_class(inner_cls: type[FusedMoEMethodBase]) -> type:
    return type(f"Hetero{inner_cls.__name__}", (HeteroMoEMethod, inner_cls), {})


def maybe_wrap_hetero_method(
    method: FusedMoEMethodBase, layer: torch.nn.Module, layer_name: str
) -> FusedMoEMethodBase:
    """Turn ``method`` into its hetero variant when the plan sends part of this
    layer's experts to the servers; otherwise return it unchanged."""
    if not envs.VLLM_HETERO_MOE:
        return method
    plan = runtime.get_plan()
    assert plan is not None
    plan_layer = plan_layer_id(layer_name, plan.num_layers)
    if plan_layer is None or not plan.has_layer(plan_layer):
        return method
    if plan.is_hub_only(plan_layer):
        return method  # MTP draft: resident experts only, no hx call
    vllm_config = get_current_vllm_config()
    if not vllm_config.use_v2_model_runner:
        raise PlanError(
            "hetero MoE announces every forward to the expert servers from the V2 "
            "model runner; set VLLM_USE_V2_MODEL_RUNNER=1"
        )
    config = layer.moe_config  # type: ignore[attr-defined]
    parallel = config.moe_parallel_config
    if parallel.tp_size > 1 and not parallel.use_ep:
        raise PlanError(
            "hetero MoE needs TP1 or expert parallelism: the hub's slot pool is "
            "whole experts (stage 1: --tensor-parallel-size 1)"
        )
    rank = runtime.moe_rank(parallel)
    hub = runtime.hub_device_index(plan, rank)
    n_slots = int(plan.hub_resident(hub, plan_layer).shape[0])
    if config.num_local_experts != n_slots:
        raise PlanError(
            f"hetero MoE layer {layer_name}: the expert map manager sized the layer "
            f"to {config.num_local_experts} experts, the plan's slot pool is "
            f"{n_slots} (VLLM_HETERO_MOE must be set before the model is built)"
        )
    max_rows = vllm_config.scheduler_config.max_num_batched_tokens
    method.__class__ = hetero_method_class(type(method))
    method._hetero = HeteroLayer(  # type: ignore[attr-defined]
        layer_name=layer_name,
        plan_layer=plan_layer,
        rank=rank,
        n_slots=n_slots,
        max_rows=max_rows,
    )
    logger.info_once(
        "Using %s: %s, hub %s, %d slots in layer %s",
        type(method).__name__,
        plan.describe(),
        plan.device_id(hub),
        n_slots,
        layer_name,
    )
    return method
