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
    assert "bw 0.4 1048576 128 p:0>4" in cmds  # cross-root push
    assert "bw 0.4 1048576 128 l:0>4" in cmds  # cross-root pull
    assert "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6" in cmds  # 3 concurrent pairs
    assert plan.skipped == []
    assert plan.est_total_s <= 35
    assert plan.needs_peer


def test_host_only_selection_needs_no_peer_access(fx_topo) -> None:
    # one GPU: every device-to-device step is infeasible, host steps remain
    plan = pl.make_plan("quick", ["0a"], fx_topo)
    assert set(verbs(plan)) == {"hbw"}
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
