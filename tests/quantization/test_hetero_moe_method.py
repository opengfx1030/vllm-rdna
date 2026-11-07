# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hub quant method and its hooks (needs the vLLM stack).

* ``ExpertMapManager`` sizes a hetero layer to the plan's slot pool for any EP
  size and leaves hub-only / unlisted layers alone;
* ``maybe_wrap_hetero_method`` turns the platform's W4A16 method into its hetero
  variant only for layers that use the servers;
* ``HeteroMoEMethod.apply`` runs dispatch -> shared expert -> hub MoE (masked ids)
  -> combine and returns the full MoE (``ref_hx`` standing in for the servers).
"""

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm.envs as envs
from tests.quantization.hetero_moe_support import (
    EXACT,
    Island,
    full_moe,
    hetero,
    make_experts,
    routing,
    stripe_doc,
)
from vllm.distributed.hetero_moe import runtime
from vllm.distributed.hetero_moe.plan import PlanError
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe as ct_moe,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_wna16_hetero as hetero_method,
)

ref_hx = hetero("ref_hx")
E = 64
LAYERS = 3  # target layers 0..2; the MTP draft is layer 3


def _wrap(method: Any, layer: Any, layer_name: str) -> Any:
    return hetero_method.maybe_wrap_hetero_method(method, layer, layer_name)


def _parallel(tp=1, ep=1, use_ep=False, rank=0):
    return FusedMoEParallelConfig(
        tp_size=tp,
        pcp_size=1,
        dp_size=1,
        ep_size=ep,
        tp_rank=rank if not use_ep else 0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=rank if use_ep else 0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


@pytest.fixture
def plan_env(tmp_path, monkeypatch):
    """H1 stripe plan (3 target layers + hub-only MTP) selected through the env."""
    doc = stripe_doc(layers=LAYERS, mtp=True)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("VLLM_HETERO_MOE", "1")
    monkeypatch.setenv("VLLM_HETERO_MOE_PLAN", str(path))
    monkeypatch.delenv("VLLM_HETERO_MOE_DEVICE_ID", raising=False)
    runtime.reset_for_tests()
    yield runtime.get_plan()
    runtime.reset_for_tests()


def _manager(layer_name, parallel=None):
    return ExpertMapManager(
        max_num_batched_tokens=64,
        top_k=8,
        global_num_experts=E,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel or _parallel(),
        placement_strategy="linear",
        enable_eplb=False,
        layer_name=layer_name,
    )


# ---- expert map manager -------------------------------------------------------------


def test_envs_are_registered():
    for name in (
        "VLLM_HETERO_MOE",
        "VLLM_HETERO_MOE_PLAN",
        "VLLM_HETERO_MOE_ROLE",
        "VLLM_HETERO_MOE_DEVICE_ID",
        "VLLM_HETERO_MOE_SERVERS",
        "VLLM_HETERO_MOE_SPILL_K",
        "VLLM_HETERO_MOE_WAIT_MS",
        "VLLM_HETERO_MOE_STATS",
    ):
        assert hasattr(envs, name), name
        assert name in envs.environment_variables


def test_explicit_experts_override_the_ep1_shortcut():
    local, expert_map, _ = determine_expert_map(
        ep_size=1, ep_rank=0, global_num_experts=8, explicit_local_experts=[5, 1, 6]
    )
    assert local == 3
    assert expert_map is not None
    assert expert_map.tolist() == [-1, 1, -1, -1, -1, 0, 2, -1]
    assert expert_map.dtype == torch.int32
    # unchanged: ep_size 1 without a placement keeps every expert, no map
    assert determine_expert_map(ep_size=1, ep_rank=0, global_num_experts=8) == (
        8,
        None,
        None,
    )


def test_manager_builds_the_slot_pool_of_a_hetero_layer(plan_env):
    plan = plan_env
    mgr = _manager("model.layers.1.mlp.experts")
    resident = plan.hub_resident(plan.hubs(0)[0], 1)
    assert mgr.local_num_experts == len(resident) < E
    slot_map = mgr.expert_map
    assert slot_map.dtype == torch.int32 and slot_map.shape == (E,)
    assert slot_map[torch.from_numpy(resident).long()].tolist() == list(
        range(len(resident))
    )
    assert int((slot_map >= 0).sum()) == len(resident)
    tabs = hetero("tables").build_island_tables(plan, plan.hubs(0)[0])
    assert np.array_equal(slot_map.numpy(), tabs.layers[1].slot_map)


def test_manager_leaves_hub_only_and_unlisted_layers_alone(plan_env):
    mtp = _manager("mtp.layers.0.mlp.experts")  # hub-only: every expert resident
    assert mtp.local_num_experts == E and mtp.expert_map is None
    other = _manager("model.layers.40.mlp.experts")  # not in the plan
    assert other.local_num_experts == E and other.expert_map is None
    nameless = _manager("")
    assert nameless.local_num_experts == E and nameless.expert_map is None


def test_manager_is_untouched_when_hetero_is_off(plan_env, monkeypatch):
    monkeypatch.setenv("VLLM_HETERO_MOE", "0")
    mgr = _manager("model.layers.1.mlp.experts")
    assert mgr.local_num_experts == E and mgr.expert_map is None


def test_manager_rejects_a_plan_for_another_expert_count(plan_env):
    with pytest.raises(PlanError, match="experts"):
        ExpertMapManager(
            max_num_batched_tokens=64,
            top_k=8,
            global_num_experts=128,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=_parallel(),
            placement_strategy="linear",
            enable_eplb=False,
            layer_name="model.layers.1.mlp.experts",
        )


# ---- the method wrapper -------------------------------------------------------------


class _Inner:
    """Stands in for the platform's W4A16 MoE method."""

    def __init__(self, island=None):
        self.island = island
        self.events = []

    def create_weights(self, layer, num_experts, *args, **kwargs):
        self.events.append(("create_weights", num_experts))

    def process_weights_after_loading(self, layer):
        self.events.append(("process",))

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_in):
        assert shared_experts is None and shared_in is None
        self.events.append(("hub_moe", topk_ids.clone()))
        return self.island.hub_moe(layer.plan_layer, x, topk_weights)(topk_ids)


def _layer(plan, plan_layer, parallel=None, slot_map=None, num_local=None):
    tabs = hetero("tables").build_island_tables(plan, plan.hubs(0)[0])
    lt = tabs.layers[plan_layer]
    return SimpleNamespace(
        plan_layer=plan_layer,
        moe_config=SimpleNamespace(
            moe_parallel_config=parallel or _parallel(),
            num_local_experts=lt.n_slots if num_local is None else num_local,
        ),
        expert_map=torch.from_numpy(lt.slot_map.copy())
        if slot_map is None
        else slot_map,
    )


@pytest.fixture
def stub_config(monkeypatch):
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
    )
    monkeypatch.setattr(hetero_method, "get_current_vllm_config", lambda: cfg)
    return cfg


def test_wrapper_swaps_the_class_of_hx_layers_only(plan_env, stub_config):
    plan = plan_env
    inner = _Inner()
    wrapped = _wrap(inner, _layer(plan, 1), "model.layers.1.mlp.experts")
    assert wrapped is inner and isinstance(inner, _Inner)
    assert isinstance(inner, hetero_method.HeteroMoEMethod)
    assert type(inner).__mro__[1] is hetero_method.HeteroMoEMethod
    assert type(inner).__mro__[2] is _Inner
    assert inner._hetero.plan_layer == 1 and inner._hetero.rank == 0
    assert inner._hetero.n_slots == plan.hub_resident(plan.hubs(0)[0], 1).shape[0]
    assert inner._hetero.max_rows == 64
    assert inner.mk_can_overlap_shared_experts is True
    # one cached dynamic class per inner class
    again = _wrap(_Inner(), _layer(plan, 2), "model.layers.2.mlp.experts")
    assert type(again) is type(inner)
    # the MTP draft (hub-only) and unlisted layers keep the plain method
    for name in ("mtp.layers.0.mlp.experts", "model.layers.40.mlp.experts"):
        method = _Inner()
        out = _wrap(method, _layer(plan, 1), name)
        assert out is method and type(method) is _Inner


def test_wrapper_is_a_noop_without_the_env(plan_env, stub_config, monkeypatch):
    monkeypatch.setenv("VLLM_HETERO_MOE", "0")
    method = _Inner()
    out = ct_moe._hetero_wrap(method, _layer(plan_env, 1), "model.layers.1.mlp.experts")
    assert out is method and type(method) is _Inner


def test_get_moe_method_hook_wraps_through_the_env(plan_env, stub_config):
    method = ct_moe._hetero_wrap(
        _Inner(), _layer(plan_env, 1), "model.layers.1.mlp.experts"
    )
    assert isinstance(method, hetero_method.HeteroMoEMethod)


@pytest.mark.parametrize(
    "case",
    ["tp2_without_ep", "other_slot_count", "v1_runner"],
)
def test_wrapper_refuses_unsupported_setups(plan_env, stub_config, case):
    plan = plan_env
    layer = _layer(plan, 1)
    if case == "tp2_without_ep":
        layer = _layer(plan, 1, parallel=_parallel(tp=2, rank=0))
        match = "TP1 or expert parallelism"
    elif case == "other_slot_count":
        layer = _layer(plan, 1, num_local=E)
        match = "sized the layer"
    else:
        stub_config.use_v2_model_runner = False
        match = "V2 model runner"
    with pytest.raises(PlanError, match=match):
        _wrap(_Inner(), layer, "model.layers.1.mlp.experts")


def test_create_weights_demands_the_slot_pool(plan_env, stub_config):
    plan = plan_env
    method = _wrap(_Inner(), _layer(plan, 1), "model.layers.1.mlp.experts")
    with pytest.raises(PlanError, match="slot pool"):
        method.create_weights(object(), E)
    method.create_weights(object(), method._hetero.n_slots)
    assert method.events == [("create_weights", method._hetero.n_slots)]


def test_process_weights_checks_the_device_slot_map(plan_env, stub_config):
    plan = plan_env
    layer = _layer(plan, 1)
    method = _wrap(_Inner(), layer, "model.layers.1.mlp.experts")
    method.process_weights_after_loading(layer)
    assert method.events == [("process",)]
    layer.expert_map = layer.expert_map.to(torch.int64)
    with pytest.raises(PlanError, match="int32 slot_map"):
        method.process_weights_after_loading(layer)
    layer.expert_map = None
    with pytest.raises(PlanError, match="int32 slot_map"):
        method.process_weights_after_loading(layer)


# ---- apply: the call sequence and the partial-sum identity ---------------------------


class _Spy:
    """``ref_hx`` recording the op order; shared by the method's inner stub."""

    def __init__(self, events):
        self.events = events

    def begin_step(self, hub, rows, bucket):
        self.events.append(("begin", rows, bucket))
        ref_hx.begin_step(hub, rows, bucket)

    def dispatch(self, hub, x, ids, w, hub_ids, layer):
        self.events.append(("dispatch", layer, ids.dtype, w.dtype))
        ref_hx.dispatch(hub, x, ids, w, hub_ids, layer)

    def combine(self, hub, y_hub, out, layer):
        self.events.append(("combine", layer))
        return ref_hx.combine(hub, y_hub, out, layer)


@pytest.fixture
def hub_runtime(plan_env, stub_config):
    plan = plan_env
    experts = make_experts(LAYERS)
    island = Island(plan, plan.hubs(0)[0], experts, spill_k={3: 4, 12: 4})
    events: list[tuple] = []
    rt = runtime.HubRuntime(
        plan, island.tables, island.hub, _Spy(events), max_rows=64, device="cpu"
    )
    runtime.install_runtime(rt)
    return SimpleNamespace(plan=plan, island=island, events=events, rt=rt)


def _shared(events):
    def shared_experts(shared_input, order):
        events.append(("shared", order))
        return shared_input

    return shared_experts


def test_apply_runs_the_hx_sequence_and_matches_the_full_moe(hub_runtime):
    h = hub_runtime
    for layer in range(LAYERS):
        layer_name = f"model.layers.{layer}.mlp.experts"
        method = _wrap(_Inner(h.island), _layer(h.plan, layer), layer_name)
        method.events = h.events
        x, w, ids = routing(5, seed=layer)
        h.events.clear()
        out = method.apply(
            _layer(h.plan, layer), x, w, ids, _shared(h.events), x.clone()
        )
        torch.testing.assert_close(
            out, full_moe(h.island.experts, layer, x, w, ids), **EXACT
        )
        names = [e[0] for e in h.events]
        first = ["begin"] if layer == 0 else []
        assert names == first + ["dispatch", "shared", "hub_moe", "combine"]
        shared_event = next(e for e in h.events if e[0] == "shared")
        assert shared_event[1] is SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        dispatch_event = next(e for e in h.events if e[0] == "dispatch")
        assert dispatch_event[2:] == (torch.int32, torch.float32)
        # the hub MoE saw the masked ids: every other-server pair is -1
        hub_ids = next(e for e in h.events if e[0] == "hub_moe")[1]
        assert ((hub_ids == -1) | (hub_ids >= 0)).all()
        assert (hub_ids[(ids < 0) | (ids >= E)] == -1).all()


def test_apply_without_a_shared_expert_and_for_other_dtypes(hub_runtime):
    h = hub_runtime
    method = _wrap(_Inner(h.island), _layer(h.plan, 0), "model.layers.0.mlp.experts")
    x, w, ids = routing(3, seed=4)
    out = method.apply(_layer(h.plan, 0), x, w, ids, None, None)
    torch.testing.assert_close(out, full_moe(h.island.experts, 0, x, w, ids), **EXACT)
    with pytest.raises(TypeError, match="fp16"):
        method.apply(_layer(h.plan, 0), x.to(torch.bfloat16), w, ids, None, None)
    # int64 ids and a non-contiguous fp32 weight view are normalised, not rejected
    wide = torch.cat([w, w], dim=1)[:, ::2]
    out = method.apply(_layer(h.plan, 0), x, wide, ids.to(torch.int64), None, None)
    torch.testing.assert_close(out, full_moe(h.island.experts, 0, x, w, ids), **EXACT)


def test_runner_hooks_reach_the_installed_runtime(hub_runtime):
    h = hub_runtime
    runtime.runner_pre_forward(3, True)
    runtime.runner_post_forward()
    assert h.island.hub.posted == [(3, 3)]
    assert h.island.hub.checks == 1
    runtime.runner_check()
    assert h.island.hub.checks == 2
