# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Suite plans of the pciescope probe on the captured par1-cs25 topology.

``fx_topo`` (conftest of this directory) is the real fabric of the box: GPUs
cb ce d3 d8 db (switch A, HIP 0-4) and 07 0a 0f 14 17 (switch B, HIP 5-9). The
tests check which commands a suite plans for a GPU selection, their syntax
against the grammar of ``p2p_probe``, and that infeasible steps are skipped
with a reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope.probe import plan as pl  # noqa: E402

SLOT_A = ["ce", "d3", "d8", "db"]
SLOT_B = ["0a", "0f", "14", "17"]
BOTH = SLOT_A + SLOT_B


def verbs(plan: pl.Plan) -> list[str]:
    return [s.cmd.split()[0] for s in plan.steps]


def test_list_suites_in_order_of_cost() -> None:
    suites = pl.list_suites()
    assert [s.name for s in suites] == ["quick", "standard", "full"]
    assert [s.est_seconds for s in suites] == sorted(s.est_seconds for s in suites)
    assert all(s.description for s in suites)


def test_unknown_suite_is_a_value_error(fx_topo) -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        pl.make_plan("nightly", SLOT_B, fx_topo)


def test_plan_orders_gpus_by_hip_and_sets_hip_visible(fx_topo) -> None:
    plan = pl.make_plan("quick", ["17", "0a", "14", "0f"], fx_topo)
    assert plan.buses == ("0a", "0f", "14", "17")  # HIP 6 7 8 9
    assert plan.hip == (6, 7, 8, 9)
    assert plan.hip_visible == "6,7,8,9"
    assert plan.switch == ("B", "B", "B", "B")
    assert plan.arch == ("gfx1030",) * 4
    assert plan.entities == ["gpu:0a", "gpu:0f", "gpu:14", "gpu:17"]


def test_quick_slot_b_has_one_class_and_says_what_it_skipped(fx_topo) -> None:
    plan = pl.make_plan("quick", SLOT_B, fx_topo)
    cmds = [s.cmd for s in plan.steps]
    # latency: flag only and 4 KiB on the intra-switch pair 0:1
    assert "lat 2000 0 0:1" in cmds
    assert "lat 2000 4096 0:1" in cmds
    # push, pull, sdma at 1 MiB and the sdma plateau at 64 MiB
    assert "bw 0.4 1048576 128 p:0>1" in cmds
    assert "bw 0.4 1048576 128 l:0>1" in cmds
    assert "bw 0.4 1048576 128 s:0>1" in cmds
    assert "bw 0.4 67108864 128 s:0>1" in cmds
    # host transfers on one GPU: pinned and pageable, both ways, plus both at once
    assert "hbw 0.4 67108864 128 h2d:0" in cmds
    assert "hbw 0.4 67108864 128 d2h_pg:0" in cmds
    assert "hbw 0.4 67108864 128 h2d:0 d2h:0" in cmds
    assert "hflag 1000 0" in cmds  # host to GPU doorbell round trip
    # ring and fan-in inside the switch
    assert "bw 0.5 1048576 128 p:0>1 p:1>2 p:2>3 p:3>0" in cmds
    assert "bw 0.5 1048576 128 p:1>0 p:2>0 p:3>0" in cmds
    skipped = dict(plan.skipped)
    assert "cross-root pair" in skipped
    assert "one switch only" in skipped["cross-root pair"]
    assert not any(c.startswith("curve") for c in cmds)
    assert plan.repeats == 1


def test_quick_both_slots_covers_every_path_class_within_30_seconds(fx_topo) -> None:
    plan = pl.make_plan("quick", BOTH, fx_topo)
    cmds = [s.cmd for s in plan.steps]
    assert plan.hip_visible == "1,2,3,4,6,7,8,9"
    assert "lat 2000 0 0:1 0:4" in cmds  # one intra-switch pair, one cross-root pair
    assert "hflag 1000 0" in cmds and "hflag 1000 4" in cmds  # one GPU per switch
    assert "bw 0.4 1048576 128 p:0>4" in cmds  # cross-root push
    assert "bw 0.4 1048576 128 l:0>4" in cmds  # cross-root pull
    assert "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6" in cmds  # 3 concurrent pairs
    assert plan.skipped == []
    assert plan.est_total_s <= 35
    assert plan.needs_peer


def test_host_only_selection_needs_no_peer_access(fx_topo) -> None:
    # one GPU: every device-to-device step is infeasible, host steps remain
    plan = pl.make_plan("quick", ["0a"], fx_topo)
    assert set(verbs(plan)) == {"hbw", "hflag"}
    assert not plan.needs_peer
    assert len(plan.skipped) >= 3
    assert plan.est_total_s < 15


def test_standard_adds_curves_sweeps_and_repeats_key_cells(fx_topo) -> None:
    plan = pl.make_plan("standard", BOTH, fx_topo)
    cmds = [s.cmd for s in plan.steps]
    sizes = ",".join(str(s) for s in pl.LAT_SIZES)
    assert pl.LAT_SIZES[0] == 4 and pl.LAT_SIZES[-1] == 64 << 20
    for mode in ("push", "pull", "sdma"):
        assert f"curve {mode} 1000 0 {sizes} 0:1" in cmds  # intra-switch pair
        assert f"curve {mode} 1000 0 {sizes} 0:4" in cmds  # cross-root pair
    assert any(c.startswith("hlat 500 0 ") for c in cmds)
    assert any(c.startswith("hlat 500 4 ") for c in cmds)
    # concurrency sweep: 1..4 cross-root pairs, A to B, B to A and bidirectional
    assert "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6 p:3>7" in cmds
    assert "bw 0.5 1048576 128 p:4>0 p:5>1 p:6>2 p:7>3" in cmds
    assert any(
        c.startswith("bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6 p:3>7 p:4>0") for c in cmds
    )
    # fan-in / fan-out across the root link and the interference run
    assert "bw 0.5 1048576 128 p:0>4 p:1>4 p:2>4 p:3>4" in cmds
    assert any(" bg:p:" in c for c in cmds)
    # one-shot all-to-all inside each switch and over all eight GPUs
    assert "a2a 2000 4096 0,1,2,3" in cmds
    assert "a2a 2000 4096 0,1,2,3,4,5,6,7" in cmds
    assert plan.repeats == 3
    keyed = [s for s in plan.steps if s.key]
    assert keyed and all(dict(s.tags)["reps"] == "3" for s in keyed)
    reps = {dict(s.tags)["rep"] for s in keyed}
    assert reps == {"1", "2", "3"}
    first = next(s for s in plan.steps if s.cmd == "bw 0.4 1048576 128 p:0>1")
    third = next(s for s in plan.steps if s.id == f"{first.id}#3")
    assert third.cmd == first.cmd and dict(third.tags)["rep"] == "3"


def test_full_is_a_superset_of_standard_with_every_pair(fx_topo) -> None:
    standard = {s.cmd for s in pl.make_plan("standard", BOTH, fx_topo).steps}
    full = pl.make_plan("full", BOTH, fx_topo)
    cmds = {s.cmd for s in full.steps}
    assert standard < cmds
    assert "hflag 1000 0 1 2 3 4 5 6 7" in cmds  # the doorbell of every GPU
    for a in range(8):
        for b in range(8):
            if a != b:
                assert f"bw 0.4 1048576 128 p:{a}>{b}" in cmds
                assert f"bw 0.4 1048576 128 s:{a}>{b}" in cmds
    assert any(
        c.startswith("lat 2000 65536 0:1 0:2") for c in cmds
    )  # 28 pairs, chunked
    longest = max(len(s.cmd.split()) for s in full.steps if s.cmd.startswith("lat "))
    assert longest <= 3 + pl.LAT_PAIRS_PER_STEP


@pytest.mark.parametrize("suite", ["quick", "standard", "full"])
@pytest.mark.parametrize("buses", [SLOT_B, BOTH, ["ce", "0a"], ["0a"]])
def test_every_command_has_the_probe_grammar(
    fx_topo, suite: str, buses: list[str]
) -> None:
    plan = pl.make_plan(suite, buses, fx_topo)
    n = len(buses)
    ids = [s.id for s in plan.steps]
    assert len(ids) == len(set(ids))
    assert len({(s.cmd, dict(s.tags).get("rep")) for s in plan.steps}) == len(ids)
    for s in plan.steps:
        tok = s.cmd.split()
        assert tok[0] == s.kind
        assert len(s.cmd) < pl.BATCH_LINE_MAX
        assert s.est_s > 0
        used = [int(x) for x in _device_ids(s.cmd)]
        assert all(0 <= d < n for d in used), s.cmd
        if s.kind in ("bw", "hbw"):
            float(tok[1])  # seconds
            assert int(tok[2]) in (4096, 65536, 1 << 20, 5 << 20, 64 << 20)
            assert int(tok[3]) == pl.BW_BLOCKS
        if s.kind == "bw":
            assert all(x.split(":", 1)[0] in ("p", "l", "s", "bg") for x in tok[4:])
        if s.kind == "hbw":
            modes = {"h2d", "d2h", "h2d_pg", "d2h_pg", "h2d_k", "d2h_k", "bg"}
            assert all(x.split(":", 1)[0] in modes for x in tok[4:])
        if s.kind == "hflag":
            assert int(tok[1]) == 1000 and len(tok) >= 3
        if s.kind in ("lat", "latc"):
            assert int(tok[1]) == 2000 and int(tok[2]) in (0, 4096, 65536)
            assert all(">" not in x and ":" in x for x in tok[3:])
        if s.kind == "curve":
            assert tok[1] in ("push", "pull", "sdma")
            assert [int(x) for x in tok[4].split(",")] == list(pl.LAT_SIZES)
    assert plan.est_total_s == pytest.approx(
        pl.STARTUP_S + 0.5 * n + sum(s.est_s for s in plan.steps)
    )


def _device_ids(cmd: str) -> list[str]:
    """Logical device ids named by a command (for the range check above)."""
    tok = cmd.split()
    verb = tok[0]
    out: list[str] = []
    if verb in ("bw", "hbw"):
        for x in tok[4:]:
            x = x.removeprefix("bg:")
            body = x.split(":", 1)[1]
            out += body.replace(">", " ").split()
    elif verb in ("lat", "latc"):
        for x in tok[3:]:
            out += x.split(":")
    elif verb == "curve":
        for x in tok[5:]:
            out += x.split(":")
    elif verb == "a2a":
        out += tok[3].split(",")
    elif verb == "hlat":
        out += tok[2].split(",")
    elif verb == "hflag":
        out += tok[2:]
    return out


def test_w7800_and_v620_mix_is_planned_in_hip_order(fx_topo) -> None:
    # bus cb (W7800, switch A) and 0a (V620, switch B): the probe treats both alike
    plan = pl.make_plan("quick", ["0a", "cb"], fx_topo)
    assert plan.buses == ("cb", "0a")
    assert plan.hip_visible == "0,6"
    assert plan.arch == ("gfx1100", "gfx1030")
    assert "bw 0.4 1048576 128 p:0>1" in [s.cmd for s in plan.steps]  # cross-root


def test_repeats_override(fx_topo) -> None:
    one = pl.make_plan("standard", SLOT_B, fx_topo, repeats=1)
    five = pl.make_plan("standard", SLOT_B, fx_topo, repeats=5)
    assert one.repeats == 1 and five.repeats == 5
    keyed = sum(1 for s in one.steps if s.key)
    assert len(five.steps) == len(one.steps) + 4 * keyed
    assert pl.make_plan("quick", SLOT_B, fx_topo, repeats=0).repeats == 1


def test_to_dict_is_json_ready(fx_topo) -> None:
    import json

    d = pl.make_plan("standard", SLOT_B, fx_topo).to_dict()
    text = json.dumps(d)
    assert json.loads(text)["suite"] == "standard"
    assert d["hip_visible_devices"] == "6,7,8,9"
    assert d["steps"][0]["cmd"].startswith("hbw ")
    assert {"what", "why"} == set(d["skipped"][0])


# --- the interim box: six V620 behind ONE switch, two leaves (lease slots) -------
# Real capture of 2026-11-10 (conftest fx_interim_topo): GPUs d1 d4 (slot A, HIP
# 0-1) and d9 dc df e2 (slot B, HIP 2-5), all on switch A.

INTERIM_A = ["d1", "d4"]
INTERIM_B = ["d9", "dc", "df", "e2"]
INTERIM_ALL = INTERIM_A + INTERIM_B
SUITE_NAMES = ["quick", "standard", "full"]


def _xfer_ids(cmd: str) -> list[int]:
    return [int(x) for x in _device_ids(cmd)]


def _leaf_of(plan: pl.Plan, topo, lid: int) -> str | None:
    return topo.gpu(plan.buses[lid]).slot


@pytest.mark.parametrize("suite", SUITE_NAMES)
@pytest.mark.parametrize("buses", [INTERIM_A, INTERIM_B, INTERIM_ALL])
def test_interim_plans_build_and_have_the_probe_grammar(
    fx_interim_topo, suite: str, buses: list[str]
) -> None:
    plan = pl.make_plan(suite, buses, fx_interim_topo)
    n = len(buses)
    assert plan.switch == ("A",) * n  # the chip letter per GPU, not the leaf
    assert len({s.id for s in plan.steps}) == len(plan.steps)
    for s in plan.steps:
        assert s.cmd.split()[0] == s.kind
        assert len(s.cmd) < pl.BATCH_LINE_MAX
        assert all(0 <= d < n for d in _xfer_ids(s.cmd)), s.cmd
    assert plan.est_total_s == pytest.approx(
        pl.STARTUP_S + 0.5 * n + sum(s.est_s for s in plan.steps)
    )


# (steps, est_total_s) of the plans as built on 2026-11-10 (A, B, all six V620)
INTERIM_COUNTS = {
    ("A", "quick"): (14, 14.7),
    ("A", "standard"): (88, 132.7),
    ("A", "full"): (109, 146.7),
    ("B", "quick"): (16, 17.3),
    ("B", "standard"): (101, 144.0),
    ("B", "full"): (235, 244.6),
    ("all", "quick"): (20, 21.4),
    ("all", "standard"): (230, 268.8),
    ("all", "full"): (539, 502.9),
}


@pytest.mark.parametrize(("name", "suite"), list(INTERIM_COUNTS))
def test_interim_step_counts_and_duration(
    fx_interim_topo, name: str, suite: str
) -> None:
    buses = {"A": INTERIM_A, "B": INTERIM_B, "all": INTERIM_ALL}[name]
    plan = pl.make_plan(suite, buses, fx_interim_topo)
    steps, est = INTERIM_COUNTS[(name, suite)]
    assert len(plan.steps) == steps
    assert round(plan.est_total_s, 1) == est


@pytest.mark.parametrize("suite", SUITE_NAMES)
def test_interim_one_leaf_plan_equals_the_four_gpu_plan_of_the_ten_gpu_box(
    fx_topo, fx_interim_topo, suite: str
) -> None:
    # B = four GPUs of one leaf: same commands, patterns and skips as slot B of
    # the ten-GPU box (only the GPU names in the scenario texts differ)
    new = pl.make_plan(suite, INTERIM_B, fx_interim_topo)
    old = pl.make_plan(suite, SLOT_B, fx_topo)
    assert [(s.cmd, s.kind, s.pattern, s.key) for s in new.steps] == [
        (s.cmd, s.kind, s.pattern, s.key) for s in old.steps
    ]
    assert new.skipped == old.skipped
    assert new.est_total_s == pytest.approx(old.est_total_s)
    assert "cross-leaf" not in " ".join(s.scenario for s in new.steps)


def test_interim_all_plans_a_cross_leaf_pair(fx_interim_topo) -> None:
    plan = pl.make_plan("quick", INTERIM_ALL, fx_interim_topo)
    assert plan.hip == (0, 1, 2, 3, 4, 5)
    by_cmd = {s.cmd: s for s in plan.steps}
    # HIP 0 (d1, leaf A) to HIP 2 (d9, leaf B): push, pull, sdma
    for verb in "pls":
        step = by_cmd[f"bw 0.4 1048576 128 {verb}:0>2"]
        assert "cross-leaf gpu:d1>gpu:d9" in step.scenario
    assert "lat 2000 0 0:1 0:2" in by_cmd  # intra-leaf pair and cross-leaf pair
    assert "bw 0.4 1048576 128 p:0>1" in by_cmd  # the intra-leaf pair of leaf A
    standard = pl.make_plan("standard", INTERIM_ALL, fx_interim_topo)
    cmds = {s.cmd for s in standard.steps}
    assert "bw 0.4 1048576 128 p:2>3" in cmds  # second leaf: its own intra pair
    assert "bw 0.5 1048576 128 p:0>2 p:1>3" in cmds  # 2 concurrent cross-leaf pairs
    assert "latc 2000 4096 0:2 1:3" in cmds


@pytest.mark.parametrize("suite", SUITE_NAMES)
def test_interim_all_rings_and_intra_patterns_stay_in_one_leaf(
    fx_interim_topo, suite: str
) -> None:
    plan = pl.make_plan(suite, INTERIM_ALL, fx_interim_topo)
    checked = 0
    for s in plan.steps:
        # (bidir-pair is also planned for every pair of the full suite)
        if s.pattern not in ("ring-intra", "a2a-intra", "fan-in", "fan-out"):
            continue
        leaves = {_leaf_of(plan, fx_interim_topo, d) for d in _xfer_ids(s.cmd)}
        assert len(leaves) == 1, (s.cmd, leaves)
        checked += 1
    assert checked >= 2
    ring = next(s for s in plan.steps if s.pattern == "ring-intra")
    assert ring.cmd == "bw 0.5 1048576 128 p:2>3 p:3>4 p:4>5 p:5>2"  # leaf B
    assert "leaf B of switch A" in ring.scenario
    for s in plan.steps:
        if s.kind == "a2a" and "all 6 GPUs" not in s.scenario:
            leaves = {_leaf_of(plan, fx_interim_topo, d) for d in _xfer_ids(s.cmd)}
            assert len(leaves) == 1, s.cmd


@pytest.mark.parametrize("suite", SUITE_NAMES)
def test_interim_all_has_no_cross_root_wording(fx_interim_topo, suite: str) -> None:
    plan = pl.make_plan(suite, INTERIM_ALL, fx_interim_topo)
    text = " ".join(s.scenario for s in plan.steps)
    skips = " ".join(f"{w} {y}" for w, y in plan.skipped)
    assert "cross-root" not in text + skips
    assert "cross-leaf" in text
    assert ("3 concurrent cross-leaf pairs", "needs 3 GPUs on each of two leaves") in (
        plan.skipped
    )


def test_interim_host_steps_pick_per_chip_not_per_leaf(fx_interim_topo) -> None:
    # one host uplink: the single-GPU steps use one GPU, not one per leaf
    plan = pl.make_plan("standard", INTERIM_ALL, fx_interim_topo)
    cmds = [s.cmd for s in plan.steps]
    assert "hbw 0.4 67108864 128 h2d:0" in cmds
    assert "hbw 0.4 67108864 128 h2d:2" not in cmds
    assert "hflag 1000 0" in cmds and "hflag 1000 2" not in cmds
    assert not any(c.startswith("hlat") and c.split()[2] == "2" for c in cmds)
    assert "hbw 0.5 67108864 128 h2d:0 h2d:1 h2d:2 h2d:3 h2d:4 h2d:5" in cmds
    assert not any("one GPU per switch" in s.scenario for s in plan.steps)
    assert ("host H2D on both switches at once", "only one switch selected") in (
        plan.skipped
    )
    # the full suite also times the doorbell of every GPU
    full = [s.cmd for s in pl.make_plan("full", INTERIM_ALL, fx_interim_topo).steps]
    assert "hflag 1000 0 1 2 3 4 5" in full


def test_ten_gpu_plans_are_unchanged_by_the_leaf_grouping(fx_topo) -> None:
    # step counts and durations of the plans before the interim-box change
    ten_gpus = BOTH + ["cb", "07"]
    expect = {
        ("quick", 28, 29.5): ten_gpus,
        ("standard", 265, 334.1): ten_gpus,
        ("full", 1206, 1041.1): ten_gpus,
        ("quick", 28, 28.5): BOTH,
        ("standard", 266, 332.8): BOTH,
        ("full", 854, 774.7): BOTH,
    }  # fmt: skip
    for (suite, steps, est), buses in expect.items():
        plan = pl.make_plan(suite, buses, fx_topo)
        assert (len(plan.steps), round(plan.est_total_s, 1)) == (steps, est)
        assert "cross-leaf" not in " ".join(s.scenario for s in plan.steps)
    both = pl.make_plan("standard", BOTH, fx_topo)
    texts = [s.scenario for s in both.steps]
    assert "3 concurrent cross-root pairs" in texts
    assert "rings of both switches at once" in texts
    assert "ring of 4 inside switch A" in texts
    assert "1 pair concurrent cross-root 1 MiB, A to B" in texts
    # W7800 cb has no lease slot and does not split switch A into leaves
    mixed = pl.make_plan("quick", ["cb", "ce", "d3", "0a"], fx_topo)
    assert mixed.hip_visible == "0,1,2,6"
    assert "bw 0.4 1048576 128 p:0>3" in [s.cmd for s in mixed.steps]
    assert mixed.switch == ("A", "A", "A", "B")
