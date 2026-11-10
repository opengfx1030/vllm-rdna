# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hub's MoE call sequence against a W4A16 reference.

The hub partial (resident experts through ``slot_map``, masked ids) plus the
servers' partials (the in-process ``ref_hx`` transport) must equal the full MoE,
for H1 and for the H2 shapes, with and without spill.
"""

import json
import logging
from types import SimpleNamespace

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


def _runtime(experts, widths=None, **kw):
    plan = _stripe_plan()
    island = _Island(plan, 0, experts, spill_k=widths or {3: 4, 12: 4})
    spy = _HxSpy()
    rt = runtime_mod.HubRuntime(
        plan,
        island.tables,
        island.hub,
        spy,
        max_rows=64,
        device="cpu",
        hidden=HIDDEN,
        **kw,
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


class _Batch:
    """The fields of the runner's ``InputBatch`` that the step log reads."""

    def __init__(self, req_ids, slots, scheduled):
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        self.idx_mapping_np = np.array(slots)
        self.num_scheduled_tokens = np.array(scheduled)


def _states(prompt_len, computed):
    """The runner's ``RequestState`` fields: host arrays indexed by request slot."""
    return SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array(prompt_len)),
        num_computed_tokens_np=np.array(computed),
    )


def test_step_log_has_one_line_per_announced_step(experts, tmp_path):
    rt, island, spy = _runtime(experts, step_log=str(tmp_path / "steps"))
    island.hub._epoch = 41  # the epoch counter of the real hub
    batch = _Batch(["cmpl-a", "cmpl-b"], [3, 0], [3, 3])
    states = _states([20, 0, 0, 7], [25, 0, 0, 11])  # slot 3: prompt 7; slot 0: 20
    rt.pre_forward(12, True, batch, states)  # the runner hook, a graph replay
    rt.post_forward()
    x, w, ids = _routing(3, seed=8)
    for layer in range(3):  # an eager forward the runner did not announce
        _layer_call(rt, island, spy, layer, x, w, ids)
    rt.close()
    (path,) = tmp_path.glob("steps.*")
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["who"] == "runner" and (first["rows"], first["bucket"]) == (12, 12)
    assert first["epoch"] == 41
    assert first["reqs"] == [["cmpl-a", 7, 11, 3], ["cmpl-b", 20, 25, 3]]
    assert second["who"] == "begin" and (second["rows"], second["bucket"]) == (3, 0)
    assert second["reqs"] == []
    assert 0 < first["ns"] <= second["ns"]


def test_step_log_is_off_by_default_and_never_fails_a_forward(experts, tmp_path):
    rt, island, spy = _runtime(experts)
    assert rt._step_log is None
    rt.pre_forward(12, True, _Batch(["a"], [0], [3]), _states([1], [2]))
    assert island.hub.posted == [(12, 12)] and not list(tmp_path.iterdir())
    rt, island, spy = _runtime(experts, step_log=str(tmp_path / "steps"))
    rt.pre_forward(12, True, object(), _states([1], [2]))  # not a batch
    assert island.hub.posted == [(12, 12)] and rt._step_log is None  # log turned off
    rt.pre_forward(12, True)  # the forwards go on
    assert island.hub.posted == [(12, 12), (12, 12)]


def test_step_log_is_queued_and_written_in_blocks(experts, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_mod, "STEP_LOG_EVERY", 2)
    rt, island, spy = _runtime(experts, step_log=str(tmp_path / "steps"))
    (path,) = tmp_path.glob("steps.*")
    for forward in range(3):
        rt.pre_forward(12, True)
        assert len(path.read_text().splitlines()) == (2 if forward >= 2 else 0)
        rt.post_forward()  # the block of two is written by the second one
        assert len(path.read_text().splitlines()) == (2 if forward >= 1 else 0)
    rt.close()
    assert len(path.read_text().splitlines()) == 3


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


def _layer_fns(island):
    return {
        li: (lambda x, w, ids, li=li: island.hub_moe(li, x, w)(ids))
        for li in island.tables.layers
    }


def test_boot_self_test_runs_one_step_through_every_layer(experts):
    rt, island, spy = _runtime(experts)
    summary = rt.self_test(_layer_fns(island))
    assert summary["layers"] == 3 and summary["max_rel_err"] < 1e-9
    assert rt.self_tested
    assert [e[0] for e in spy.events] == ["begin"] + ["dispatch", "combine"] * 3
    assert island.hub.posted == [(3, 0)]  # the self-test is one eager step
    pairs = island.hub.dispatched_pairs
    assert pairs["hub"] > 0 and pairs["remote"] > 0  # both paths were exercised
    assert island.hub.began == [(3, 0)]


def test_boot_self_test_catches_a_wrongly_mapped_server(experts):
    rt, island, spy = _runtime(experts)
    right = island.hub.servers[2]
    # server 2 answers with server 3's expert set (an id <-> slot mapping bug)
    island.hub.servers[2] = lambda layer, x, ids, w: island.hub.servers[3](
        layer, x, ids, w
    )
    with pytest.raises(RuntimeError, match="boot self-test failed in layer 0"):
        rt.self_test(_layer_fns(island))
    island.hub.servers[2] = right
    assert not rt.self_tested


def test_boot_self_test_treats_a_nan_reply_as_a_failure(experts):
    rt, island, spy = _runtime(experts)
    island.hub.servers[1] = lambda layer, x, ids, w: torch.full(
        (x.shape[0], x.shape[1]), float("nan"), dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="boot self-test failed"):
        rt.self_test(_layer_fns(island))


def test_boot_self_test_needs_every_layer_registered_and_skips_copyless_layers(
    experts,
):
    rt, island, spy = _runtime(experts)
    fns = _layer_fns(island)
    del fns[1]
    with pytest.raises(RuntimeError, match="layer 1 has no hub MoE"):
        rt.self_test(fns)
    rt, island, spy = _runtime(experts)
    for lt in island.tables.layers.values():
        lt.n_cold[:] = 0  # nothing resident has a server copy: nothing to compare
    assert rt.self_test(_layer_fns(island))["layers"] == 0
    assert rt.self_tested and not island.hub.posted


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


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_counters_are_logged_every_n_forwards_and_at_close(experts, monkeypatch):
    rt, island, spy = _runtime(experts)
    rt.stats = True
    monkeypatch.setattr(runtime_mod, "STATS_EVERY", 2)
    handler = _Lines()
    runtime_mod.logger.addHandler(handler)
    old_level = runtime_mod.logger.level
    runtime_mod.logger.setLevel(logging.INFO)
    try:
        x, w, ids = _routing(5, seed=3)
        at_second = None
        for forward in range(3):
            rt.pre_forward(5, False)
            for layer in range(3):
                _layer_call(rt, island, spy, layer, x, w, ids)
            if forward == 1:  # the counters line is logged by this post_forward
                at_second = [
                    sum(r[i] for r in island.hub.layer_stats()) for i in range(3)
                ]
            rt.post_forward()
            logged = [m for m in handler.lines if "counters" in m]
            assert len(logged) == (1 if forward >= 1 else 0)  # after forwards 2, 4, ...
        interval = [m for m in handler.lines if "counters" in m][0]
        assert "pairs per step" in interval and "% remote" in interval
        status = [m for m in handler.lines if "hx status" in m]
        assert len(status) == 1  # one per counters line, same cadence
        assert status[0].startswith("hetero MoE hx status: record 0x0, epoch ")
        assert ", dispatches " in status[0] and ", combines " in status[0]
        totals = [m for m in handler.lines if "pairs over all layers" in m][0]
        assert at_second is not None and at_second[2] > 0  # K = 4 spills residents
        assert totals == (
            f"hetero MoE pairs over all layers: hub {at_second[0]}, servers (not "
            f"resident) {at_second[1]}, servers (spill rule) {at_second[2]}"
        )
        pairs = island.hub.dispatched_pairs
        n_hub, n_cold, n_spilled = (
            sum(r[i] for r in island.hub.layer_stats()) for i in range(3)
        )
        assert n_hub == pairs["hub"] and n_cold + n_spilled == pairs["remote"]
        handler.lines.clear()
        rt.close()
        assert any("(final)" in m for m in handler.lines)
        assert any("hx status" in m for m in handler.lines)
        assert any("per-layer pairs (final)" in m for m in handler.lines)
        assert island.hub.closed
    finally:
        runtime_mod.logger.removeHandler(handler)
        runtime_mod.logger.setLevel(old_level)


def test_digest_counters_are_logged_with_the_stats_of_a_digest_hub(experts):
    rt, island, spy = _runtime(experts)
    rt.stats = True
    line = (
        "hub digest: 41 ok, 0 bad at the first read, 0 healed, 0 unhealed, "
        "0 late digest words, 3 rescans"
    )
    island.hub.describe_digest = lambda: line
    handler = _Lines()
    runtime_mod.logger.addHandler(handler)
    old_level = runtime_mod.logger.level
    runtime_mod.logger.setLevel(logging.INFO)
    try:
        island.hub.digest = False  # a hub without the digest mode logs no line
        rt.log_stats()
        assert not [m for m in handler.lines if "hx digest" in m]
        island.hub.digest = True
        rt.log_stats()
        assert [m for m in handler.lines if "hx digest" in m] == [
            f"hetero MoE hx digest: {line}"
        ]
    finally:
        runtime_mod.logger.removeHandler(handler)
        runtime_mod.logger.setLevel(old_level)


def test_counters_are_silent_unless_enabled(experts):
    rt, island, spy = _runtime(experts)
    handler = _Lines()
    runtime_mod.logger.addHandler(handler)
    old_level = runtime_mod.logger.level
    runtime_mod.logger.setLevel(logging.INFO)
    try:
        for _ in range(3):
            rt.pre_forward(3, False)
            rt.post_forward()
        rt.close()
        assert not [m for m in handler.lines if "counters" in m]
    finally:
        runtime_mod.logger.removeHandler(handler)
        runtime_mod.logger.setLevel(old_level)


def test_a_hub_without_layer_stats_logs_only_the_totals(experts):
    rt, island, spy = _runtime(experts)
    rt.stats = True
    island.hub.layer_stats = lambda: None  # HxHub created without layer_stats=True
    handler = _Lines()
    runtime_mod.logger.addHandler(handler)
    old_level = runtime_mod.logger.level
    runtime_mod.logger.setLevel(logging.INFO)
    try:
        rt.log_stats(final=True)
        assert any("counters (final)" in m for m in handler.lines)
        assert not [m for m in handler.lines if "all layers" in m or "per-layer" in m]
    finally:
        runtime_mod.logger.removeHandler(handler)
        runtime_mod.logger.setLevel(old_level)
