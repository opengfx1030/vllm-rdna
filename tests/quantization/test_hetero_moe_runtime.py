# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hub's MoE call sequence against a W4A16 reference.

The hub partial (resident experts through ``slot_map``, masked ids) plus the
servers' partials (the in-process ``ref_hx`` transport) must equal the full MoE,
for H1 and for the H2 shapes, with and without spill.
"""

import numpy as np
import pytest

from tests.quantization.hetero_moe_support import (
    EXACT,
    HIDDEN,
    TOP_K,
    E,
    hetero,
    make_experts,
    stripe_doc,
)
from tests.quantization.hetero_moe_support import Island as _Island
from tests.quantization.hetero_moe_support import full_moe as _full
from tests.quantization.hetero_moe_support import invalid_pairs as _invalid_pairs
from tests.quantization.hetero_moe_support import routing as _routing
from tests.quantization.hetero_moe_support import stripe_plan as _stripe_plan

torch = pytest.importorskip("torch")

plan_mod = hetero("plan")
tables_mod = hetero("tables")
ref_hx = hetero("ref_hx")
runtime_mod = hetero("runtime")


@pytest.fixture(scope="module")
def experts():
    """Checkpoint experts per layer: {layer: {expert id: Expert}}."""
    return make_experts()


@pytest.mark.parametrize("rows", [1, 3, 12, 50])
@pytest.mark.parametrize("spill", ["none", "some", "all"])
def test_partial_sum_identity_h1(experts, rows, spill):
    """Hub residents + remote partials == the full MoE, whatever the spill."""
    k = {"none": 10**6, "some": 3, "all": 0}[spill]
    island = _Island(_stripe_plan(), 0, experts, spill_k={3: k, 12: k, 48: k})
    island.tables.prefill_k = k
    invalid = 0
    for layer in range(3):
        x, w, ids = _routing(rows, seed=rows * 7 + layer)
        invalid += _invalid_pairs(ids)
        out, hub_ids = island.forward(layer, x, w, ids)
        torch.testing.assert_close(out, _full(experts, layer, x, w, ids), **EXACT)
        kept = hub_ids[hub_ids >= 0]
        if spill == "all":
            assert kept.numel() == 0  # everything went to the servers
        if spill == "some":
            assert len(torch.unique(kept)) <= k  # the rule bounds the hub's experts
        assert (hub_ids[(ids < 0) | (ids >= E)] == -1).all()
    pairs = island.hub.dispatched_pairs
    assert pairs["hub"] + pairs["remote"] == 3 * rows * TOP_K - invalid
    assert pairs["dropped"] == 0  # H1: every expert has a copy
    if spill == "all":
        assert pairs["hub"] == 0


def test_fp16_wire_matches_within_rounding(experts):
    """With fp16 partials on the wire the result is within fp16 rounding."""
    island = _Island(_stripe_plan(), 0, experts, spill_k={3: 4, 12: 4}, exact=False)
    x, w, ids = _routing(12, seed=5)
    ref_hx.begin_step(island.hub, 12, 0)
    hub_ids = torch.empty_like(ids)
    ref_hx.dispatch(island.hub, x, ids, w, hub_ids, 1)
    y = island.hub_moe(1, x, w)(hub_ids).to(torch.float16)
    out = torch.empty_like(y)
    ref_hx.combine(island.hub, y, out, 1)
    full = _full(experts, 1, x, w, ids)
    err = (out.double() - full).abs().max().item()
    assert err < 2e-3 * full.abs().max().item() + 1e-4


def test_masked_ids_are_inert(experts):
    """A row whose ids are all -1 gets exactly zero from hub, servers and the sum."""
    island = _Island(_stripe_plan(), 0, experts, spill_k={3: 3, 12: 3})
    x, w, ids = _routing(3, seed=9, bad=False)
    ids[2, :] = -1
    out, hub_ids = island.forward(0, x, w, ids)
    assert (out[2] == 0).all() and (hub_ids[2] == -1).all()
    torch.testing.assert_close(out, _full(experts, 0, x, w, ids), **EXACT)


def test_h2_replicate_islands_agree(experts):
    """Two replicate islands (one engine each) compute the same output."""
    plan = _stripe_plan(islands=2)
    a = _Island(plan, plan.hubs(0)[0], experts, spill_k={3: 4, 12: 4})
    b = _Island(plan, plan.hubs(1)[0], experts, spill_k={3: 4, 12: 4})
    x, w, ids = _routing(7, seed=3)
    out_a, _ = a.forward(1, x, w, ids)
    out_b, _ = b.forward(1, x, w, ids)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    torch.testing.assert_close(out_a, _full(experts, 1, x, w, ids), **EXACT)


@pytest.mark.parametrize("k", [10**6, 2])
def test_h2_partition_partials_sum_to_full(experts, k):
    """Each island computes its own experts and drops the other island's pairs;
    the runner's reduce (here +) gives the full MoE."""
    plan = _stripe_plan(islands=2, island_mode="partition")
    x, w, ids = _routing(9, seed=11)
    parts = []
    for island_idx in range(2):
        island = _Island(plan, plan.hubs(island_idx)[0], experts, spill_k={3: k, 12: k})
        out, _ = island.forward(2, x, w, ids)
        parts.append(out)
        assert island.hub.dispatched_pairs["dropped"] > 0  # the other island's pairs
    assert not torch.equal(parts[0], parts[1])
    torch.testing.assert_close(
        parts[0] + parts[1], _full(experts, 2, x, w, ids), **EXACT
    )


def test_slot_map_swap_in_place(experts):
    """A slot swap (new expert in a slot, slot_map updated IN PLACE) is followed by
    the hub call without rebuilding anything; the total stays the full MoE."""
    island = _Island(
        _stripe_plan(), 0, experts, spill_k={3: 10**6, 12: 10**6, 48: 10**6}
    )
    layer = 1
    lt = island.tables.layers[layer]
    slot_map = island.slot_map[layer]
    handle = slot_map.data_ptr()
    x, w, ids = _routing(12, seed=21, bad=False)
    # every row selects an expert that is not in the slot pool
    old = int(lt.resident[0])
    new = next(e for e in range(E) if lt.slot_map[e] < 0 and lt.n_cold[e] > 0)
    ids[:, 0] = new
    before, hub_ids = island.forward(layer, x, w, ids)
    assert (hub_ids[:, 0] == -1).all()  # remote
    torch.testing.assert_close(before, _full(experts, layer, x, w, ids), **EXACT)
    # in-place swap of slot 0: its weights, the layer's slot_map and the
    # transport's copy of it (both must follow, S8's flip does the same)
    island.slots[layer][0] = island.experts[layer][new]
    for table in (slot_map, island.hub._dev[layer][0]):
        table[old] = -1
        table[new] = 0
    assert slot_map.data_ptr() == handle
    after, hub_ids = island.forward(layer, x, w, ids)
    assert (hub_ids[:, 0] == new).all()  # served by the hub now
    torch.testing.assert_close(after, before, **EXACT)


# ---- HubRuntime: call order and the per-step protocol --------------------------------


class _HxSpy:
    """``ref_hx`` that records the order of the ops."""

    def __init__(self):
        self.events = []

    def begin_step(self, hub, rows, bucket):
        self.events.append(("begin", rows, bucket))
        ref_hx.begin_step(hub, rows, bucket)

    def dispatch(self, hub, x, ids, w, hub_ids, layer):
        self.events.append(("dispatch", layer))
        ref_hx.dispatch(hub, x, ids, w, hub_ids, layer)

    def combine(self, hub, y_hub, out, layer):
        self.events.append(("combine", layer))
        return ref_hx.combine(hub, y_hub, out, layer)


def _runtime(experts, widths=None):
    plan = _stripe_plan()
    island = _Island(plan, 0, experts, spill_k=widths or {3: 4, 12: 4})
    spy = _HxSpy()
    rt = runtime_mod.HubRuntime(
        plan, island.tables, island.hub, spy, max_rows=64, device="cpu"
    )
    return rt, island, spy


def _layer_call(rt, island, spy, layer, x, w, ids):
    def shared():
        spy.events.append(("shared", layer))

    def hub_moe(hub_ids):
        spy.events.append(("hub", layer))
        return island.hub_moe(layer, x, w)(hub_ids)

    return rt.apply_layer(layer, x, w, ids, hub_moe=hub_moe, shared=shared)


def test_layer_sequence_and_begin_once_per_forward(experts):
    rt, island, spy = _runtime(experts)
    x, w, ids = _routing(3, seed=1)
    for layer in range(3):
        out = _layer_call(rt, island, spy, layer, x, w, ids)
        torch.testing.assert_close(out, _full(experts, layer, x, w, ids), **EXACT)
    layer_ops = ["dispatch", "shared", "hub", "combine"]
    assert [e[0] for e in spy.events] == ["begin"] + layer_ops * 3
    assert spy.events[0] == ("begin", 3, 0)
    assert [e[1] for e in spy.events if e[0] == "combine"] == [0, 1, 2]
    # an eager forward without the runner hook: the begin step wakes the servers
    assert island.hub.posted == [(3, 0)]


def test_hub_ids_buffer_is_one_persistent_allocation(experts):
    rt, island, spy = _runtime(experts)
    base = rt.hub_ids.data_ptr()
    seen = []

    def one_forward(rows):
        x, w, ids = _routing(rows, seed=rows)

        def hub_moe(hub_ids):
            seen.append(hub_ids.data_ptr())
            return island.hub_moe(0, x, w)(hub_ids)

        rt.apply_layer(0, x, w, ids, hub_moe=hub_moe)

    for rows in (3, 12, 3):  # three forwards, each a first-layer call
        one_forward(rows)
    assert seen == [base] * 3 and rt.hub_ids.shape == (64, TOP_K)


def test_runner_hook_arms_eager_forward_without_double_post(experts):
    rt, island, spy = _runtime(experts)
    x, w, ids = _routing(5, seed=2)
    rt.pre_forward(5, replay=False)
    for layer in range(3):
        _layer_call(rt, island, spy, layer, x, w, ids)
    rt.post_forward()
    assert island.hub.posted == [(5, 0)]  # the hook posted, begin_step did not
    assert island.hub.checks == 1
    # a second forward without the hook posts again from begin_step
    for layer in range(3):
        _layer_call(rt, island, spy, layer, x, w, ids)
    assert island.hub.posted == [(5, 0), (5, 0)]


def test_replay_posts_the_capture_width_and_runs_no_python(experts):
    rt, island, spy = _runtime(experts)
    rt.pre_forward(12, replay=True)  # the graph replays: no MoE Python runs
    rt.post_forward()
    assert island.hub.posted == [(12, 12)] and not spy.events
    with pytest.raises(RuntimeError, match=r"graphs for \[3, 12\]"):
        rt.pre_forward(6, replay=True)  # no server graph of this width
    # an eager forward after the replay is not affected by a stale arm
    x, w, ids = _routing(4, seed=6)
    for layer in range(3):
        _layer_call(rt, island, spy, layer, x, w, ids)
    assert island.hub.posted[-1] == (4, 0)


def test_capture_bakes_the_width_and_posts_nothing(experts, monkeypatch):
    rt, island, spy = _runtime(experts)
    monkeypatch.setattr(runtime_mod, "_capturing", lambda: True)
    x, w, ids = _routing(12, seed=4)
    for layer in range(3):
        _layer_call(rt, island, spy, layer, x, w, ids)
    assert spy.events[0] == ("begin", 12, 12)  # bucket = the capture width
    assert island.hub.posted == []  # a capture executes nothing
    x, w, ids = _routing(5, seed=4)
    with pytest.raises(RuntimeError, match="capture of width 5"):
        _layer_call(rt, island, spy, 0, x, w, ids)


def test_rows_above_the_transport_limit_are_refused(experts):
    rt, island, spy = _runtime(experts)
    rt.max_rows = 4
    x, w, ids = _routing(5, seed=8)
    with pytest.raises(RuntimeError, match="exceed the transport"):
        _layer_call(rt, island, spy, 0, x, w, ids)


def test_runtime_needs_a_layer_that_uses_the_servers():
    doc = stripe_doc(layers=1, mtp=True)
    doc["layers"] = [x for x in doc["layers"] if x.get("kind") == "mtp"]
    doc["plan_id"] = plan_mod.compute_plan_id(doc)
    plan = plan_mod.plan_from_doc(doc, validate=False)
    tabs = tables_mod.build_island_tables(plan, 0)
    with pytest.raises(plan_mod.PlanError, match="no layer that uses the servers"):
        runtime_mod.HubRuntime(plan, tabs, object(), object(), max_rows=8, device="cpu")


def test_loop_oracle_and_vectorised_rule_agree_on_random_layers():
    plan = _stripe_plan(islands=2)
    tabs = tables_mod.build_island_tables(plan, plan.hubs(1)[0])
    gen = torch.Generator().manual_seed(0)
    for layer, lt in tabs.layers.items():
        dev = [
            torch.from_numpy(a) for a in (lt.slot_map, lt.n_cold, lt.cold_srv, lt.owned)
        ]
        for rows in (1, 3, 8, 33):
            ids = torch.randint(-2, E + 3, (rows, TOP_K), generator=gen).to(torch.int32)
            for k in (0, 1, 2, 5, 16, 10**6):
                hub, dest, lost = ref_hx.split_pairs(ids, *dev, layer, k)
                hub_ref, dest_ref = tables_mod.split_pairs_loop(
                    ids.numpy(), lt, layer, k
                )
                assert np.array_equal(hub.numpy(), hub_ref)
                assert np.array_equal(dest.numpy(), dest_ref)
                assert not lost.any()


def test_vectorised_rule_on_the_partition_island_and_plan_errors():
    plan = _stripe_plan(islands=2, island_mode="partition")
    tabs = tables_mod.build_island_tables(plan, plan.hubs(0)[0])
    lt = tabs.layers[1]
    assert 0 < int(lt.owned.sum()) < E
    dev = [torch.from_numpy(a) for a in (lt.slot_map, lt.n_cold, lt.cold_srv, lt.owned)]
    gen = torch.Generator().manual_seed(1)
    for k in (0, 3, 10**6):
        ids = torch.randint(0, E, (20, TOP_K), generator=gen).to(torch.int32)
        hub, dest, lost = ref_hx.split_pairs(ids, *dev, 1, k)
        hub_ref, dest_ref = tables_mod.split_pairs_loop(ids.numpy(), lt, 1, k)
        assert np.array_equal(hub.numpy(), hub_ref)
        assert np.array_equal(dest.numpy(), dest_ref)
        foreign = lt.owned[ids.numpy()] == 0
        assert (hub.numpy()[foreign] == -1).all() and (
            dest.numpy()[foreign] == -1
        ).all()
        assert not lost.any()
    # drop the copy of an owned, non-resident expert: the pair is a plan error
    victim = next(e for e in range(E) if lt.owned[e] and lt.slot_map[e] < 0)
    lt.n_cold[victim] = 0
    ids = torch.tensor([[victim, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    _, _, lost = ref_hx.split_pairs(ids, *dev, 1, 10**6)
    assert lost.tolist()[0][0] is True
    hub = ref_hx.RefHub(tabs, {})
    ref_hx.begin_step(hub, 1, 0)
    with pytest.raises(RuntimeError, match="plan error"):
        ref_hx.dispatch(
            hub,
            torch.zeros(1, HIDDEN, dtype=torch.float16),
            ids,
            torch.zeros(1, TOP_K),
            torch.empty_like(ids),
            1,
        )
