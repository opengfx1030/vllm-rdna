# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hub-side runtime of the heterogeneous MoE (spec 6.2).

One ``HubRuntime`` per hub process. It owns the plan tables, the connection to
the expert servers (``hx``) and the per-step protocol:

* ``pre_forward`` / ``post_forward`` are called by the model runner around every
  target-model forward. Before a graph replay they wake the servers
  (``post_step``: host code, outside any graph); ``post_forward`` polls the
  abort record.
* ``begin_step`` runs in the first MoE layer of a forward. Kernel launches that
  Python enqueues itself (eager steps, capture warm-ups) wake the servers from
  here unless the runner already did; during a graph capture nothing is posted
  (a capture executes nothing) and the capture width is baked in.
* ``apply_layer`` is one MoE layer: dispatch, shared expert, hub fused MoE on the
  resident experts, combine. The returned tensor is the island's partial sum;
  any all-reduce is the runner's.

The ``hx`` backend is a module with ``begin_step`` / ``dispatch`` / ``combine``
(the HIP ops of ``hetero_moe.hx``, or ``ref_hx`` in tests).
"""

import atexit
import importlib
import os
from collections.abc import Callable
from types import ModuleType
from typing import Protocol

import numpy as np
import torch

from .plan import HeteroPlan, PlanError, load_plan, plan_layer_id
from .tables import IslandTables, build_island_tables, parse_spill_k

_PLAN: HeteroPlan | None = None
_RUNTIME: "HubRuntime | None" = None


class HubLike(Protocol):
    """What the runtime needs of ``hx.HxHub`` (``ref_hx.RefHub`` has it too)."""

    def post_step(self, rows: int, bucket: int) -> None: ...

    def check(self) -> None: ...

    def close(self) -> None: ...


def _env():
    from vllm import envs

    return envs


def hetero_enabled() -> bool:
    return bool(_env().VLLM_HETERO_MOE)


def get_plan() -> HeteroPlan | None:
    """The plan of this process (``VLLM_HETERO_MOE=1``), loaded and validated once."""
    global _PLAN
    if not hetero_enabled():
        return None
    if _PLAN is None:
        path = _env().VLLM_HETERO_MOE_PLAN
        if not path:
            raise PlanError("VLLM_HETERO_MOE=1 needs VLLM_HETERO_MOE_PLAN=<plan.json>")
        _PLAN = load_plan(path)
    return _PLAN


def reset_for_tests() -> None:
    global _PLAN, _RUNTIME
    _PLAN = None
    if _RUNTIME is not None:
        _RUNTIME.close()
    _RUNTIME = None


def moe_rank(parallel) -> int:
    """Index of this process in the hub device list: the EP rank under expert
    parallelism, else the TP rank (``FusedMoEParallelConfig``)."""
    return parallel.ep_rank if parallel.use_ep else parallel.tp_rank


def hub_device_index(plan: HeteroPlan, rank: int) -> int:
    """Plan device of the hub with this TP/EP rank: ``VLLM_HETERO_MOE_DEVICE_ID``
    (comma list in rank order) or the plan's hubs in plan order."""
    ids = [s for s in _env().VLLM_HETERO_MOE_DEVICE_ID.split(",") if s]
    if ids:
        if rank >= len(ids):
            raise PlanError(
                f"VLLM_HETERO_MOE_DEVICE_ID lists {len(ids)} device(s), rank {rank}"
            )
        dev = plan.device_index(ids[rank])
    else:
        hubs = [d for k in range(plan.num_islands) for d in plan.hubs(k)]
        if rank >= len(hubs):
            raise PlanError(f"plan has {len(hubs)} hub(s), rank {rank}")
        dev = hubs[rank]
    if plan.role(dev) != "hub":
        raise PlanError(f"device {plan.device_id(dev)} is not a hub")
    return dev


def hub_slot_experts(
    layer_name: str, global_num_experts: int, rank: int
) -> list[int] | None:
    """Expert ids of this layer's slot pool on this hub, in slot order, or
    ``None`` when the layer follows the standard path (hetero off, a layer the
    plan does not list, or a hub-only layer that holds every expert)."""
    plan = get_plan()
    if plan is None:
        return None
    layer = plan_layer_id(layer_name, plan.num_layers)
    if layer is None or not plan.has_layer(layer):
        return None
    if plan.num_experts != global_num_experts:
        raise PlanError(
            f"plan {plan.plan_id} has {plan.num_experts} experts, layer "
            f"{layer_name} has {global_num_experts}"
        )
    resident = plan.hub_resident(hub_device_index(plan, rank), layer)
    identity = np.array_equal(resident, np.arange(global_num_experts))
    if plan.is_hub_only(layer) and identity:
        return None
    return [int(e) for e in resident]


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class HubRuntime:
    def __init__(
        self,
        plan: HeteroPlan,
        tables: IslandTables,
        hub: HubLike,
        hx: ModuleType,
        *,
        max_rows: int,
        device: torch.device | str,
    ):
        self.plan = plan
        self.tables = tables
        self.hub = hub
        self.hx = hx
        self.max_rows = max_rows
        self.hx_layers = sorted(li for li, lt in tables.layers.items() if lt.uses_hx)
        if not self.hx_layers:
            raise PlanError(f"plan {plan.plan_id} has no layer that uses the servers")
        self.first_layer = self.hx_layers[0]
        self._widths = frozenset(tables.widths)
        # One buffer for all layers: graph captures bake its address.
        self.hub_ids = torch.empty(
            (max_rows, tables.top_k), dtype=torch.int32, device=device
        )
        self._armed = False
        self._closed = False

    # ---- per-step protocol -----------------------------------------------------

    def pre_forward(self, rows: int, replay: bool) -> None:
        """Runner hook before a target forward of ``rows`` rows (the padded width
        when ``replay``). Wakes the servers; the forward must follow."""
        if replay and rows not in self._widths:
            raise RuntimeError(
                f"hetero MoE: graph replay of width {rows}, but the servers have "
                f"graphs for {sorted(self._widths)} only"
            )
        self.hub.post_step(rows, rows if replay else 0)
        self._armed = True

    def post_forward(self) -> None:
        self._armed = False
        self.hub.check()

    def check(self) -> None:
        self.hub.check()

    def begin_step(self, rows: int) -> None:
        if rows > self.max_rows:
            raise RuntimeError(
                f"hetero MoE: {rows} rows exceed the transport's {self.max_rows}"
            )
        if _capturing():
            if rows not in self._widths:
                raise RuntimeError(
                    f"hetero MoE: capture of width {rows}, but the servers have "
                    f"graphs for {sorted(self._widths)} only (align "
                    "cudagraph_capture_sizes, the spill K table and the server "
                    "--buckets)"
                )
            bucket = rows
        else:
            bucket = 0
            if self._armed:
                self._armed = False  # the runner woke the servers for this forward
            else:
                self.hub.post_step(rows, 0)
        self.hx.begin_step(self.hub, rows, bucket)

    # ---- one MoE layer ----------------------------------------------------------

    def apply_layer(
        self,
        layer: int,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        hub_moe: Callable[[torch.Tensor], torch.Tensor],
        shared: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Dispatch -> shared expert -> hub MoE(masked ids) -> combine."""
        rows = x.shape[0]
        if layer == self.first_layer:
            self.begin_step(rows)
        hub_ids = self.hub_ids[:rows]
        self.hx.dispatch(self.hub, x, topk_ids, topk_weights, hub_ids, layer)
        if shared is not None:
            shared()
        y_hub = hub_moe(hub_ids)
        return self.hx.combine(self.hub, y_hub, y_hub, layer)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.hub.close()


# ---- process singleton and the runner hooks ----------------------------------------


def get_runtime(*, rank: int, hidden: int, max_rows: int, device) -> "HubRuntime":
    """The process' ``HubRuntime``; the first call connects to the servers."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if _capturing():
        raise RuntimeError("hetero MoE: the hub must connect before a graph capture")
    plan = get_plan()
    assert plan is not None
    hub_dev = hub_device_index(plan, rank)
    override = _env().VLLM_HETERO_MOE_SPILL_K
    tables = build_island_tables(
        plan, hub_dev, spill_k=parse_spill_k(override) if override else None
    )
    hx = importlib.import_module("vllm.distributed.hetero_moe.hx")
    hub = hx.HxHub.create(
        plan,
        plan.device_id(hub_dev),
        tables=tables,
        max_rows=max_rows,
        top_k=plan.top_k,
        hidden=hidden,
        wait_ms=_env().VLLM_HETERO_MOE_WAIT_MS,
    )
    _RUNTIME = HubRuntime(plan, tables, hub, hx, max_rows=max_rows, device=device)
    atexit.register(_RUNTIME.close)
    return _RUNTIME


def install_runtime(runtime: "HubRuntime | None") -> None:
    """Test hook: use ``runtime`` as the process' runtime."""
    global _RUNTIME
    _RUNTIME = runtime


def current_runtime() -> "HubRuntime | None":
    return _RUNTIME


def runner_pre_forward(num_tokens: int, replay: bool) -> None:
    """Model-runner hook before a target forward (no-op until the first MoE layer
    has connected the runtime, and whenever hetero MoE is off)."""
    runtime = _RUNTIME
    if runtime is not None:
        runtime.pre_forward(num_tokens, replay)


def runner_post_forward() -> None:
    runtime = _RUNTIME
    if runtime is not None:
        runtime.post_forward()


def runner_check() -> None:
    runtime = _RUNTIME
    if runtime is not None:
        runtime.check()


def hetero_env_report() -> str:
    envs = _env()
    return (
        f"VLLM_HETERO_MOE={int(envs.VLLM_HETERO_MOE)} "
        f"plan={envs.VLLM_HETERO_MOE_PLAN or '-'} "
        f"device_id={envs.VLLM_HETERO_MOE_DEVICE_ID or '-'} "
        f"spill_k={envs.VLLM_HETERO_MOE_SPILL_K or 'default'} "
        f"wait_ms={envs.VLLM_HETERO_MOE_WAIT_MS} pid={os.getpid()}"
    )
