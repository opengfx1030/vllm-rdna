# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversion of probe output into ``Measurement`` records.

Real data: ``fixtures/probe/audit/s1_base.probe.log`` is the output of
``p2p_probe`` on the eight V620 of par1-cs25 (logical ids 0-3 = buses ce d3 d8
db on switch A, 4-7 = 0a 0f 14 17 on switch B) from the PCIe audit. The tests
convert its blocks with the captured topology (``fx_topo``) and check values
against the numbers printed in that file and the audit.

The ``SYNTHETIC_*`` constants are FORMAT EXAMPLES of the new tags (HBW, HLAT,
CURVE), written by hand; their numbers are not measurements.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope.model import Measurement  # noqa: E402
from pciescope.probe import effectiveness as eff  # noqa: E402
from pciescope.probe import parse, results  # noqa: E402
from pciescope.probe import plan as pl
from pciescope.probe.paths import FabricModel  # noqa: E402

AUDIT = Path(__file__).parent / "fixtures" / "probe" / "audit" / "s1_base.probe.log"
BOTH = ["ce", "d3", "d8", "db", "0a", "0f", "14", "17"]
X16 = 31.508  # 16 GT/s x16 raw rate in GB/s per direction

SYNTHETIC_HOST = """\
### hbw 0.4 67108864 128 h2d:0
HBW h2d:0 size=67108864 blocks=128 gbps=24.00 verify=ok secs=0.401
HAGG size=67108864 fg_gbps=24.00 bg_gbps=0.00 n=1
END rc=0 ms=812.3
### hbw 0.4 67108864 128 d2h_pg:4
HBW d2h_pg:4 size=67108864 blocks=128 gbps=9.50 verify=ok secs=0.410
HAGG size=67108864 fg_gbps=9.50 bg_gbps=0.00 n=1
END rc=0 ms=900.1
### hbw 0.5 67108864 128 h2d:0 h2d:1 h2d:2 h2d:3
HBW h2d:0 size=67108864 blocks=128 gbps=7.00 verify=ok secs=0.502
HBW h2d:1 size=67108864 blocks=128 gbps=7.00 verify=ok secs=0.502
HBW h2d:2 size=67108864 blocks=128 gbps=7.00 verify=ok secs=0.502
HBW h2d:3 size=67108864 blocks=128 gbps=7.00 verify=FAIL secs=0.502
HAGG size=67108864 fg_gbps=28.00 bg_gbps=0.00 n=4
END rc=0 ms=1500.0
### hlat 500 0 4096
HLAT dev=0 dir=h2d mem=pinned size=4096 n=500 min_us=9.000 p50_us=10.000 p99_us=14.000 max_us=30.000 mean_us=10.500
HLAT dev=0 dir=d2h mem=pinned size=4096 n=500 min_us=9.500 p50_us=10.500 p99_us=15.000 max_us=31.000 mean_us=11.000
END rc=0 ms=300.0
### hflag 1000 0 4
HFLAG dev=0 n=1000 rtt_min_us=8.000 rtt_p50_us=9.000 rtt_p99_us=13.000 rtt_max_us=40.000 rtt_mean_us=9.400
HFLAG dev=4 n=1000 rtt_min_us=8.200 rtt_p50_us=9.200 rtt_p99_us=14.000 rtt_max_us=45.000 rtt_mean_us=9.600
END rc=0 ms=250.0
"""  # noqa: E501

SYNTHETIC_CURVE = """\
### curve push 1000 0 4096,65536 0:4
CURVE mode=push src=0 dst=4 size=4096 blocks=1 n=1000 min_us=6.100 p50_us=6.700 p99_us=9.000 max_us=20.000 mean_us=6.900 ev_us=6.800
CURVE mode=push src=0 dst=4 size=65536 blocks=1 n=1000 min_us=8.100 p50_us=9.000 p99_us=12.000 max_us=25.000 mean_us=9.200 ev_us=9.100
END rc=0 ms=500.0
### curve sdma 1000 0 4096 0:1
CURVE mode=sdma src=0 dst=1 size=4096 blocks=0 n=500 min_us=20.000 p50_us=22.000 p99_us=40.000 max_us=90.000 mean_us=23.000
END rc=0 ms=300.0
### a2a 2000 4096 0,1,2,3
A2A payload=4096 group=0,1,2,3 step_us=10.40
END rc=0 ms=100.0
"""  # noqa: E501


@pytest.fixture(scope="module")
def plan(fx_topo) -> pl.Plan:
    return pl.make_plan("standard", BOTH, fx_topo)


@pytest.fixture(scope="module")
def ctx(fx_topo, plan) -> results.ConvertContext:
    return results.ConvertContext(
        plan=plan,
        topo=fx_topo,
        model=FabricModel(fx_topo),
        run_tags={"run": "r1", "suite": "standard", "owner": "tester"},
        probe_version="pciescope-probe/1",
    )


def step_of(block: parse.Block, plan: pl.Plan | None = None) -> pl.Step:
    """The plan step that issues ``block.cmd`` (duration ignored), else a bare one."""
    if plan is not None:
        for s in plan.steps:
            if s.kind == block.argv[0] and s.cmd.split()[2:] == block.cmd.split()[2:]:
                return replace(s, cmd=block.cmd)
    return pl.Step(
        id=f"t/{block.argv[0]}",
        cmd=block.cmd,
        kind=block.argv[0],
        scenario=block.cmd,
        est_s=1.0,
    )


def convert(text: str, ctx: results.ConvertContext, index: int) -> list[Measurement]:
    block = parse.parse_text(text).blocks[index]
    return results.convert_block(step_of(block), block, ctx, 1_000, 2_000)


def audit_blocks() -> list[parse.Block]:
    return parse.parse_text(AUDIT.read_text()).blocks


def audit_measurements(ctx: results.ConvertContext, cmd: str) -> list[Measurement]:
    """Convert the first audit block whose command is ``cmd``."""
    for block in audit_blocks():
        if block.cmd == cmd:
            return results.convert_block(step_of(block, ctx.plan), block, ctx, 1, 2)
    raise AssertionError(f"no block {cmd!r} in the audit log")


def test_single_push_flow_is_a_bw_record_with_theoretical_and_efficiency(ctx) -> None:
    (m,) = audit_measurements(ctx, "bw 0.5 65536 128 p:0>4")
    assert m.kind == "bw"
    assert (m.src, m.dst) == ("gpu:ce", "gpu:0a")
    assert m.direction == "d2d-push"
    assert m.size_bytes == 65536
    assert m.value == pytest.approx(25.19) and m.unit == "GB/s"
    assert m.path_class == "cross-root"
    assert m.theoretical == pytest.approx(X16, abs=0.01)
    assert m.efficiency == pytest.approx(25.19 / m.theoretical)
    assert m.gpus_used == ["gpu:ce", "gpu:0a"]
    assert m.ok and m.error is None
    assert m.tags["method"] == "kernel-store"
    assert m.tags["run"] == "r1" and m.tags["owner"] == "tester"
    assert m.tags["probe"] == "pciescope-probe/1"
    assert (m.started_ns, m.ended_ns) == (1, 2)


@pytest.mark.parametrize(
    ("cmd", "direction", "method", "value", "pclass"),
    [
        ("bw 0.5 65536 128 l:0>4", "d2d-pull", "kernel-load", 5.67, "cross-root"),
        ("bw 0.5 65536 128 l:0>1", "d2d-pull", "kernel-load", 24.44, "intra-switch"),
        ("bw 0.5 65536 128 s:0>4", "d2d-sdma", "sdma", 4.63, "cross-root"),
        ("bw 0.5 1048576 128 s:0>4", "d2d-sdma", "sdma", 20.02, "cross-root"),
    ],
)
def test_pull_and_sdma_rows_of_the_audit(
    ctx, cmd, direction, method, value, pclass
) -> None:
    (m,) = audit_measurements(ctx, cmd)
    assert (m.kind, m.direction, m.tags["method"]) == ("bw", direction, method)
    assert m.value == pytest.approx(value)
    assert m.path_class == pclass


def test_three_concurrent_cross_root_pairs_are_one_concurrency_record(ctx) -> None:
    (m,) = audit_measurements(ctx, "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6")
    assert m.kind == "concurrency"
    assert m.value == pytest.approx(18.83)  # the AGG row, not the sum of rounded rows
    assert m.src == "gpu:ce+gpu:d3+gpu:d8"
    assert m.dst == "gpu:0a+gpu:0f+gpu:14"
    assert m.path_class == "cross-root"
    assert m.direction == "d2d-push"
    # the three flows share the uplink of switch A and of switch B
    assert m.theoretical == pytest.approx(X16, abs=0.01)
    assert m.efficiency == pytest.approx(18.83 / m.theoretical)
    assert m.stats["min"] == m.stats["max"] == m.stats["p50"] == pytest.approx(6.28)
    assert m.stats["n"] == 3
    assert m.tags["flows"] == "3"
    assert m.tags["per_flow_gbps"] == "6.28,6.28,6.28"
    assert m.tags["ideal_per_flow_gbps"].count(",") == 2
    assert m.tags["flow_list"].split(",")[0] == "gpu:ce>gpu:0a"
    assert m.gpus_used == [
        "gpu:ce",
        "gpu:d3",
        "gpu:d8",
        "gpu:0a",
        "gpu:0f",
        "gpu:14",
    ]


def test_ring_of_four_inside_a_switch_scales_with_the_flow_count(ctx) -> None:
    (m,) = audit_measurements(ctx, "bw 0.5 1048576 128 p:0>1 p:1>2 p:2>3 p:3>0")
    assert m.kind == "concurrency"
    assert m.value == pytest.approx(105.14)
    assert m.path_class == "intra-switch"
    assert m.theoretical == pytest.approx(4 * X16, abs=0.05)  # one link per flow
    assert 0.8 < m.efficiency < 0.9


def test_latc_rows_become_one_lat_record_per_pair(ctx) -> None:
    ms = audit_measurements(ctx, "latc 2000 4096 0:4 1:5 2:6 3:7")
    assert [m.value for m in ms] == [6.82, 6.76, 6.73, 6.66]
    assert all(m.kind == "lat" and m.unit == "us" for m in ms)
    assert all(m.direction == "d2d-push" and m.size_bytes == 4096 for m in ms)
    assert all(m.path_class == "cross-root" for m in ms)
    assert all(m.theoretical is None and m.efficiency is None for m in ms)
    assert [m.tags["concurrent_pairs"] for m in ms] == ["4"] * 4
    assert ms[0].src == "gpu:ce" and ms[0].dst == "gpu:0a"
    assert ms[0].stats == {"mean": 6.82, "n": 2000.0}
    assert len({m.id for m in ms}) == 4


def test_one_shot_all_to_all_step_time(ctx) -> None:
    (m,) = audit_measurements(ctx, "a2a 2000 4096 0,1,2,3,4,5,6,7")
    assert m.kind == "a2a" and m.unit == "us"
    assert m.value == pytest.approx(16.28)
    assert m.path_class == "cross-root"  # spans both switches
    assert m.tags["ranks"] == "8"
    assert m.src == m.dst
    assert len(m.gpus_used) == 8


def test_every_block_of_the_audit_log_converts_without_an_unknown_kind(ctx) -> None:
    n = 0
    for block in audit_blocks():
        got = results.convert_block(step_of(block), block, ctx, 1, 2)
        assert got, block.cmd
        assert {m.kind for m in got} <= {"bw", "concurrency", "lat", "a2a"}
        n += len(got)
    assert n >= 90


def test_host_transfer_rows_are_host_class_bw_records(ctx) -> None:
    pinned = convert(SYNTHETIC_HOST, ctx, 0)[0]
    assert (pinned.kind, pinned.direction) == ("bw", "h2d")
    assert (pinned.src, pinned.dst) == ("host", "gpu:ce")
    assert pinned.path_class == "host"
    assert pinned.tags["method"] == "pinned"
    assert pinned.theoretical == pytest.approx(X16, abs=0.01)
    assert pinned.efficiency == pytest.approx(24.0 / pinned.theoretical)
    pageable = convert(SYNTHETIC_HOST, ctx, 1)[0]
    assert (pageable.direction, pageable.src, pageable.dst) == ("d2h", "gpu:0a", "host")
    assert pageable.tags["method"] == "pageable"
    assert pinned.tags["secs"] == "0.401" and pageable.tags["secs"] == "0.410"


def test_a_verify_failure_marks_the_record_invalid(ctx) -> None:
    (m,) = convert(SYNTHETIC_HOST, ctx, 2)
    assert m.kind == "concurrency" and m.path_class == "host"
    assert m.value == pytest.approx(28.0)
    assert m.ok is False and m.error == "verify FAIL"
    # four GPUs of one switch share its uplink: the ideal aggregate is one x16 link
    assert m.theoretical == pytest.approx(X16, abs=0.01)
    assert m.tags["per_flow_secs"] == "0.502,0.502,0.502,0.502"


def test_host_latency_rows(ctx) -> None:
    h2d, d2h = convert(SYNTHETIC_HOST, ctx, 3)
    assert (h2d.kind, h2d.direction, h2d.value) == ("lat", "h2d", 10.0)
    assert (h2d.src, h2d.dst) == ("host", "gpu:ce")
    assert (d2h.src, d2h.dst) == ("gpu:ce", "host")
    assert h2d.stats["p99"] == 14.0 and h2d.stats["n"] == 500.0
    assert h2d.tags["method"] == "pinned" and h2d.path_class == "host"


def test_host_doorbell_round_trips_are_host_latency_records(ctx) -> None:
    first, second = convert(SYNTHETIC_HOST, ctx, 4)
    assert (first.kind, first.direction, first.size_bytes) == ("lat", "h2d", 0)
    assert (first.src, first.dst) == ("host", "gpu:ce")
    assert (second.src, second.dst) == ("host", "gpu:0a")  # logical device 4
    assert first.value == 9.0 and first.unit == "us"  # the p50 round trip
    assert first.stats["p99"] == 13.0 and first.stats["n"] == 1000.0
    assert first.tags["method"] == "doorbell"
    assert "not halved" in first.tags["semantics"]
    assert first.path_class == "host" and first.theoretical is None
    assert first.gpus_used == ["gpu:ce"] and first.ok


def test_curve_rows_carry_percentiles_and_the_cross_check(ctx) -> None:
    ms = convert(SYNTHETIC_CURVE, ctx, 0)
    assert [m.size_bytes for m in ms] == [4096, 65536]
    first = ms[0]
    assert first.kind == "lat" and first.direction == "d2d-push"
    assert first.value == 6.7  # the median, not the mean
    assert first.stats == {
        "min": 6.1,
        "p50": 6.7,
        "p99": 9.0,
        "max": 20.0,
        "mean": 6.9,
        "n": 1000.0,
    }
    assert first.tags["ev_us"] == "6.800"
    assert first.tags["method"] == "kernel-store"
    assert first.path_class == "cross-root"
    sdma = convert(SYNTHETIC_CURVE, ctx, 1)[0]
    assert sdma.direction == "d2d-sdma" and sdma.path_class == "intra-switch"
    assert "ev_us" not in sdma.tags


def test_failed_block_yields_no_measurements(ctx) -> None:
    log = parse.parse_text(
        "### bw 0.4 1048576 128 p:0>1\nERR timeout push 0:1 size=1048576\nERR rc=1\n"
        "END rc=1 ms=10000.0\n"
    )
    block = log.blocks[0]
    assert results.convert_block(step_of(block), block, ctx) == []
    assert results.block_error(block) == "timeout push 0:1 size=1048576"


def test_invalid_context_marks_every_record(fx_topo, plan) -> None:
    bad = results.ConvertContext(
        plan=plan,
        topo=fx_topo,
        model=None,
        ok=False,
        error="noisy window: quiet file of Other",
    )
    (m,) = audit_measurements(bad, "bw 0.5 65536 128 p:0>4")
    assert m.ok is False and "noisy" in (m.error or "")
    assert m.theoretical is None and m.efficiency is None  # no model given


def test_aggregate_folds_repeats_into_the_median(ctx) -> None:
    runs: list[Measurement] = []
    for value in (24.0, 25.0, 26.0):
        (m,) = audit_measurements(ctx, "bw 0.5 1048576 128 p:0>1")
        m.value = value
        m.tags["rep"] = str(len(runs) + 1)
        runs.append(m)
    other = audit_measurements(ctx, "bw 0.5 1048576 128 p:0>4")[0]
    out = results.aggregate([*runs, other])
    assert len(out) == 2
    folded = out[0]
    assert folded.value == 25.0
    assert (folded.stats["min"], folded.stats["max"], folded.stats["n"]) == (
        24.0,
        26.0,
        3.0,
    )
    assert folded.tags["reps"] == "3" and "rep" not in folded.tags
    assert folded.efficiency == pytest.approx(25.0 / folded.theoretical)
    assert out[1] is other


def test_aggregate_drops_invalid_repeats_when_a_valid_one_exists(ctx) -> None:
    good = audit_measurements(ctx, "bw 0.5 1048576 128 p:0>1")[0]
    bad = audit_measurements(ctx, "bw 0.5 1048576 128 p:0>1")[0]
    bad.ok, bad.error, bad.value = False, "evicted_ms jump", 3.0
    (out,) = results.aggregate([good, bad])
    assert out.value == good.value and out.ok
    (only_bad,) = results.aggregate([bad])
    assert only_bad.ok is False


def test_efficiency_rows_map_measurements_to_the_rules(ctx) -> None:
    ms = []
    for cmd in (
        "bw 0.5 1048576 128 p:0>4",
        "bw 0.5 1048576 128 l:0>4",
        "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6",
        "bw 0.5 1048576 128 p:0>1 p:1>2 p:2>3 p:3>0",
        "latc 2000 4096 0:4 1:5 2:6 3:7",
        "a2a 2000 4096 0,1,2,3,4,5,6,7",
    ):
        ms += audit_measurements(ctx, cmd)
    rows = results.eff_rows(ms)
    by = {(r.kind, r.path_class, r.n_flows, r.unit) for r in rows}
    assert ("push", "cross-root", 1, "GB/s") in by
    assert ("pull", "cross-root", 1, "GB/s") in by
    assert ("push", "cross-root", 3, "GB/s") in by  # concurrent-push.cross.pairs=3
    assert ("ring", "intra-switch", 4, "GB/s") in by
    assert ("a2a-step", "cross-root", 8, "us") in by
    assert not any(r.kind == "lat" for r in rows)  # concurrent latency: no baseline


def test_effectiveness_report_has_class_summary_baseline_and_findings(
    ctx, fx_topo
) -> None:
    ms = []
    for cmd in (
        "bw 0.5 1048576 128 p:0>1",
        "bw 0.5 1048576 128 l:0>1",
        "bw 0.5 1048576 128 p:0>4",
        "bw 0.5 1048576 128 l:0>4",
        "bw 0.5 1048576 128 p:0>4 p:1>5",
        "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6",
        "bw 0.5 1048576 128 p:0>4 p:1>5 p:2>6 p:3>7",
    ):
        ms += audit_measurements(ctx, cmd)
    rep = results.effectiveness_report(ms, FabricModel(fx_topo))
    assert set(rep["by_class"]) == {"intra-switch", "cross-root"}
    assert rep["by_class"]["cross-root"]["best_gbps"] == pytest.approx(27.59)
    row = next(
        r
        for r in rep["rows"]
        if r["kind"] == "push" and r["path_class"] == "cross-root" and r["n_flows"] == 1
    )
    assert row["practical_ceiling"] == pytest.approx(28.0, abs=0.1)
    assert row["baseline"]["verdict"] == "ok"
    codes = {f["code"] for f in rep["findings"]}
    assert "cross-root-pull-penalty" in codes  # 5.65 GB/s pull, known from the audit
    assert "cross-root-collapse" in codes  # 3 and 4 pairs, known from the audit
    assert all(f["severity"] in ("info", "warn", "bad") for f in rep["findings"])


# --- interim layout: six V620 behind ONE switch, two leaves (lease slots A, B) -------
# Topology = the real capture of 2026-11-10 (``fx_interim_topo``). The probe
# output below is SYNTHETIC (hand-written format examples, not measurements).
# Logical ids: 0 d1, 1 d4 (slot A) and 2 d9, 3 dc, 4 df, 5 e2 (slot B).

INTERIM_BUSES = ["d1", "d4", "d9", "dc", "df", "e2"]
X8 = 15.754  # 16 GT/s x8 raw rate in GB/s per direction

SYNTHETIC_INTERIM = """\
### bw 0.5 1048576 128 p:2>3
X p:2>3 size=1048576 blocks=128 gbps=13.90 verify=ok
AGG size=1048576 fg_gbps=13.90 bg_gbps=0.00 n=1
END rc=0 ms=600.0
### bw 0.5 1048576 128 p:1>2
X p:1>2 size=1048576 blocks=128 gbps=13.80 verify=ok
AGG size=1048576 fg_gbps=13.80 bg_gbps=0.00 n=1
END rc=0 ms=600.0
### bw 0.5 1048576 128 p:0>1
X p:0>1 size=1048576 blocks=128 gbps=13.90 verify=ok
AGG size=1048576 fg_gbps=13.90 bg_gbps=0.00 n=1
END rc=0 ms=600.0
### bw 0.5 1048576 128 p:0>2 p:1>3 p:4>5
X p:0>2 size=1048576 blocks=128 gbps=13.00 verify=ok
X p:1>3 size=1048576 blocks=128 gbps=13.00 verify=ok
X p:4>5 size=1048576 blocks=128 gbps=13.00 verify=ok
AGG size=1048576 fg_gbps=39.00 bg_gbps=0.00 n=3
END rc=0 ms=600.0
### bw 0.5 1048576 128 p:0>1 p:2>3
X p:0>1 size=1048576 blocks=128 gbps=13.90 verify=ok
X p:2>3 size=1048576 blocks=128 gbps=13.90 verify=ok
AGG size=1048576 fg_gbps=27.80 bg_gbps=0.00 n=2
END rc=0 ms=600.0
### bw 0.5 1048576 128 p:0>1 p:1>2
X p:0>1 size=1048576 blocks=128 gbps=13.90 verify=ok
X p:1>2 size=1048576 blocks=128 gbps=13.90 verify=ok
AGG size=1048576 fg_gbps=27.80 bg_gbps=0.00 n=2
END rc=0 ms=600.0
### lat 2000 4096 0:1 1:2 2:3
LAT 0<->1 payload=4096 one_way_us=6.50
LAT 1<->2 payload=4096 one_way_us=6.60
LAT 2<->3 payload=4096 one_way_us=6.50
END rc=0 ms=300.0
### curve push 100 0 4096 1:2
CURVE mode=push src=1 dst=2 size=4096 blocks=1 n=100 min_us=6.0 p50_us=6.5 p99_us=7.0 max_us=8.0 mean_us=6.6
END rc=0 ms=100.0
### a2a 2000 4096 0,1,2,3
A2A payload=4096 group=0,1,2,3 step_us=11.00
END rc=0 ms=100.0
### a2a 2000 4096 2,3,4,5
A2A payload=4096 group=2,3,4,5 step_us=10.50
END rc=0 ms=100.0
### hbw 0.4 67108864 128 h2d:2
HBW h2d:2 size=67108864 blocks=128 gbps=13.90 verify=ok secs=0.401
HAGG size=67108864 fg_gbps=13.90 bg_gbps=0.00 n=1
END rc=0 ms=800.0
### hbw 0.5 67108864 128 h2d:0 h2d:2 h2d:4 h2d:5
HBW h2d:0 size=67108864 blocks=128 gbps=6.00 verify=ok secs=0.501
HBW h2d:2 size=67108864 blocks=128 gbps=6.00 verify=ok secs=0.501
HBW h2d:4 size=67108864 blocks=128 gbps=6.00 verify=ok secs=0.501
HBW h2d:5 size=67108864 blocks=128 gbps=6.00 verify=ok secs=0.501
HAGG size=67108864 fg_gbps=24.00 bg_gbps=0.00 n=4
END rc=0 ms=1000.0
"""  # noqa: E501


@pytest.fixture(scope="module")
def iplan(fx_interim_topo) -> pl.Plan:
    return pl.make_plan("standard", INTERIM_BUSES, fx_interim_topo)


@pytest.fixture(scope="module")
def ictx(fx_interim_topo, iplan) -> results.ConvertContext:
    return results.ConvertContext(
        plan=iplan,
        topo=fx_interim_topo,
        model=FabricModel(fx_interim_topo),
        run_tags={"run": "r2"},
    )


def interim(
    ictx: results.ConvertContext, index: int, pattern: str = ""
) -> list[Measurement]:
    block = parse.parse_text(SYNTHETIC_INTERIM).blocks[index]
    step = pl.Step(
        id=f"i/{index}",
        cmd=block.cmd,
        kind=block.argv[0],
        scenario=block.cmd,
        est_s=1.0,
        pattern=pattern,
    )
    return results.convert_block(step, block, ictx, 1, 2)


def test_interim_plan_is_one_switch_two_leaves(iplan, fx_interim_topo) -> None:
    assert iplan.buses == tuple(INTERIM_BUSES)
    assert {g.slot for g in fx_interim_topo.gpus.values()} == {"A", "B"}
    assert {g.switch for g in fx_interim_topo.gpus.values()} == {"A"}


@pytest.mark.parametrize(
    ("index", "scope"),
    [(0, "same-leaf"), (1, "cross-leaf"), (2, "same-leaf")],
)
def test_interim_single_flow_gets_the_leaf_scope(ictx, index: int, scope: str) -> None:
    (m,) = interim(ictx, index)
    assert m.kind == "bw" and m.path_class == "intra-switch"
    assert m.tags["leaf_scope"] == scope
    assert m.tags["method"] == "kernel-store"
    assert m.theoretical == pytest.approx(X8, abs=0.01)  # the x8 link of the route
    assert m.efficiency == pytest.approx(m.value / m.theoretical)
    assert 0.85 < m.efficiency < 0.9


def test_interim_concurrency_scope_is_the_comma_joined_set(ictx) -> None:
    cross = interim(ictx, 3, "cross-k3")[0]
    assert cross.kind == "concurrency" and cross.path_class == "intra-switch"
    # 0>2 and 1>3 cross the leaves, 4>5 stays in leaf B
    assert cross.tags["leaf_scope"] == "cross-leaf,same-leaf"
    same = interim(ictx, 4)[0]
    assert same.tags["leaf_scope"] == "same-leaf"
    assert same.tags["path_classes"] == "intra-switch"
    assert interim(ictx, 5)[0].tags["leaf_scope"] == "cross-leaf,same-leaf"
    # a set uses the same sorted, comma-joined style as 'path_classes'
    assert cross.tags["leaf_scope"] == ",".join(
        sorted(cross.tags["leaf_scope"].split(","))
    )


def test_interim_latency_curve_and_a2a_carry_the_scope(ictx) -> None:
    lat = interim(ictx, 6)
    assert [m.tags["leaf_scope"] for m in lat] == [
        "same-leaf",
        "cross-leaf",
        "same-leaf",
    ]
    (curve,) = interim(ictx, 7)
    assert (
        curve.tags["leaf_scope"] == "cross-leaf"
        and curve.tags["method"] == "kernel-store"
    )
    (mixed,) = interim(ictx, 8)
    assert mixed.kind == "a2a" and mixed.path_class == "intra-switch"
    assert mixed.tags["leaf_scope"] == "cross-leaf,same-leaf"
    (leaf_b,) = interim(ictx, 9)
    assert leaf_b.tags["leaf_scope"] == "same-leaf"


def test_interim_host_records_have_no_leaf_scope(ictx) -> None:
    (one,) = interim(ictx, 10)
    assert one.path_class == "host" and "leaf_scope" not in one.tags
    assert one.theoretical == pytest.approx(X8, abs=0.01)
    assert one.efficiency == pytest.approx(13.9 / one.theoretical)
    (four,) = interim(ictx, 11)
    assert four.kind == "concurrency" and "leaf_scope" not in four.tags
    # four GPUs share the x16 uplink: 31.508 in total, 7.877 each (not 4 x 15.754)
    assert four.theoretical == pytest.approx(X16, abs=0.01)
    assert four.tags["ideal_per_flow_gbps"] == "7.88,7.88,7.88,7.88"
    assert four.efficiency == pytest.approx(24.0 / four.theoretical)


def test_ten_gpu_results_have_no_leaf_scope_tag(ctx) -> None:
    # one lease slot per switch: byte-identical tags to before the leaf scope
    for block in audit_blocks():
        for m in results.convert_block(step_of(block), block, ctx, 1, 2):
            assert "leaf_scope" not in m.tags, block.cmd


def test_leaf_scope_helper_edge_cases(fx_interim_topo, fx_topo) -> None:
    scope = results.leaf_scope
    assert scope(fx_interim_topo, [("gpu:d9", "gpu:dc")]) == "same-leaf"
    assert scope(fx_interim_topo, [("gpu:d4", "gpu:d9")]) == "cross-leaf"
    assert scope(fx_interim_topo, [("gpu:d9", "gpu:d4"), ("gpu:d9", "gpu:dc")]) == (
        "cross-leaf,same-leaf"
    )
    assert scope(fx_interim_topo, []) is None
    assert scope(fx_interim_topo, [("host", "gpu:d9")]) is None
    assert scope(fx_interim_topo, [("gpu:d9", "gpu:zz")]) is None
    # ten-GPU capture: every switch carries one slot, even across the roots
    assert scope(fx_topo, [("gpu:ce", "gpu:d3"), ("gpu:ce", "gpu:0a")]) is None


def test_interim_cross_leaf_concurrency_stays_a_sensible_unbaselined_row(
    ictx,
) -> None:
    # The plan's cross-k patterns on the interim box have path_class
    # 'intra-switch' (both leaves under one chip): they must remain rows
    # (kind push, k flows) without any cross-root baseline or finding.
    (m,) = interim(ictx, 3, "cross-k3")
    (r,) = results.eff_rows([m])
    assert (r.kind, r.path_class, r.n_flows, r.direction) == (
        "push",
        "intra-switch",
        3,
        "fwd",
    )
    assert eff.row_baseline(r) is None
    bidir = results.eff_rows(interim(ictx, 3, "cross-k3-bidir"))
    assert [(x.kind, x.direction) for x in bidir] == [("push", "bidir")]
    assert results.eff_rows(interim(ictx, 3, "cross-k3-rev"))[0].direction == "fwd"
    assert eff.findings([r]) == []


def test_interim_effectiveness_report_judges_x8_rows_by_the_box_rules(ictx) -> None:
    ms = interim(ictx, 0) + interim(ictx, 1) + interim(ictx, 10)
    ms += interim(ictx, 3, "cross-k3") + interim(ictx, 11, "host-k4")
    ms += interim(ictx, 6)
    rep = results.effectiveness_report(ms, ictx.model)
    rows = {r["label"]: r for r in rep["rows"]}
    first = rows["bw 0.5 1048576 128 p:2>3"]
    assert first["theoretical"] == pytest.approx(X8, abs=0.01)
    assert first["baseline"]["verdict"] == "ok"
    assert first["baseline"]["key"] == "push.intra.1MiB.link=x8"
    assert first["baseline"]["baseline"] == 13.9
    assert first["baseline"]["source"].startswith("lease README 2026-11-10")
    assert rows["bw 0.5 1048576 128 p:1>2"]["baseline"]["verdict"] == "ok"
    assert rows["hbw 0.4 67108864 128 h2d:2"]["baseline"]["verdict"] == "ok"
    # the k = 3 cross-leaf push and the 4-GPU host aggregate: no baseline
    assert rows["bw 0.5 1048576 128 p:0>2 p:1>3 p:4>5"]["baseline"] is None
    assert rows["hbw 0.5 67108864 128 h2d:0 h2d:2 h2d:4 h2d:5"]["baseline"] is None
    # concurrent latency has no row; no finding mentions the cross-root rules
    assert {f["code"] for f in rep["findings"]} == set()
    assert set(rep["by_class"]) == {"intra-switch", "host"}


def test_interim_rows_never_get_an_audit_verdict(ictx) -> None:
    for index in (0, 1, 2, 10):
        for m in interim(ictx, index):
            (r,) = results.eff_rows([m])
            diff = eff.row_baseline(r)
            assert diff is not None and diff.key.endswith("link=x8")
            assert diff.verdict == "ok"


# --- link_raw_gbps: the width of the route, set at conversion -----------------------


def _pattern_text(xfers: list[str], fg_gbps: float, size: int = 1 << 20) -> str:
    """SYNTHETIC bw block (format example): ``xfers`` like ``p:0>1``."""
    each = fg_gbps / len(xfers)
    lines = [f"### bw 0.5 {size} 128 {' '.join(xfers)}"]
    for x in xfers:
        lines.append(f"X {x} size={size} blocks=128 gbps={each:.2f} verify=ok")
    lines.append(f"AGG size={size} fg_gbps={fg_gbps:.2f} bg_gbps=0.00 n={len(xfers)}")
    lines.append("END rc=0 ms=600.0")
    return "\n".join(lines) + "\n"


def _rows_of(cvt: results.ConvertContext, text: str, pattern: str) -> list[eff.EffRow]:
    block = parse.parse_text(text).blocks[0]
    step = pl.Step(
        id="p/0",
        cmd=block.cmd,
        kind="bw",
        scenario=block.cmd,
        est_s=1.0,
        pattern=pattern,
    )
    return results.eff_rows(results.convert_block(step, block, cvt, 1, 2))


ALL_TO_ALL_4 = [f"p:{a}>{b}" for a in range(4) for b in range(4) if a != b]
RING_4 = ["p:0>1", "p:1>2", "p:2>3", "p:3>0"]
FAN_IN_3 = ["p:1>0", "p:2>0", "p:3>0"]


def test_link_raw_tag_is_the_smallest_bottleneck_of_the_flows(ictx, ctx) -> None:
    (m,) = interim(ictx, 0)
    assert m.tags["link_raw_gbps"] == "15.754"
    for i in (1, 2, 3, 4, 5, 8, 9, 10, 11):  # cross-leaf, many flows, a2a, host
        assert all(x.tags["link_raw_gbps"] == "15.754" for x in interim(ictx, i)), i
    # latency and curve records carry no width tag
    for i in (6, 7):
        assert all("link_raw_gbps" not in x.tags for x in interim(ictx, i))
    # ten-GPU records carry the x16 width; everything else of the conversion is
    # unchanged (the other audit tests assert the values)
    for block in audit_blocks():
        for x in results.convert_block(step_of(block, ctx.plan), block, ctx, 1, 2):
            expected = None if x.kind == "lat" else "31.508"
            assert x.tags.get("link_raw_gbps") == expected, block.cmd
    (pinned,) = convert(SYNTHETIC_HOST, ctx, 0)
    assert pinned.tags["link_raw_gbps"] == "31.508"


def test_link_raw_tag_is_absent_without_a_model(iplan, fx_interim_topo) -> None:
    bare = results.ConvertContext(plan=iplan, topo=fx_interim_topo, model=None)
    (m,) = interim(bare, 0)
    assert "link_raw_gbps" not in m.tags
    (r,) = results.eff_rows([m])
    assert r.link_raw is None and r.theoretical is None


def test_eff_row_carries_link_raw_from_the_tag(ictx, ctx) -> None:
    (m,) = interim(ictx, 0)
    (r,) = results.eff_rows([m])
    assert r.link_raw == pytest.approx(15.754)
    (p,) = audit_measurements(ctx, "bw 0.5 65536 128 p:0>4")
    assert results.eff_rows([p])[0].link_raw == pytest.approx(31.508)
    m.tags["link_raw_gbps"] = "wide"  # a damaged tag is treated as absent
    assert results.eff_rows([m])[0].link_raw is None


@pytest.mark.parametrize(
    ("xfers", "pattern", "kind", "value"),
    [
        (ALL_TO_ALL_4, "a2a-intra", "a2a", 100.0),
        (RING_4, "ring-intra", "ring", 105.0),
        (FAN_IN_3, "fan-in", "fan-in", 25.0),
    ],
)
def test_patterns_on_x8_get_no_audit_verdict_but_on_x16_they_keep_it(
    ictx, ctx, xfers: list[str], pattern: str, kind: str, value: float
) -> None:
    text = _pattern_text(xfers, value)
    (x8,) = _rows_of(ictx, text, pattern)
    assert (x8.kind, x8.n_flows, x8.path_class) == (kind, len(xfers), "intra-switch")
    assert x8.link_raw == pytest.approx(15.754)
    assert eff.row_baseline(x8) is None
    (x16,) = _rows_of(ctx, text, pattern)
    assert x16.link_raw == pytest.approx(31.508)
    if kind == "fan-in":  # the audit has no intra-switch fan-in entry
        assert eff.row_baseline(x16) is None
    else:
        diff = eff.row_baseline(x16)
        assert diff is not None and diff.verdict == "ok"
        assert diff.key == f"{kind}.intra.flows={len(xfers)}"


def test_a_12_flow_all_to_all_is_caught_by_the_tag_not_by_the_theoretical(
    ictx,
) -> None:
    # the case the theoretical heuristic could not see: 12 flows on x8 links
    (r,) = _rows_of(ictx, _pattern_text(ALL_TO_ALL_4, 100.0), "a2a-intra")
    assert r.theoretical is not None
    assert r.theoretical != pytest.approx(12 * 15.754, rel=0.05)
    assert eff.row_baseline(replace(r, link_raw=None)) is not None  # old fallback
    assert eff.row_baseline(r) is None  # the tag decides


def test_host_aggregate_without_baseline_is_documented_as_skipped(ictx) -> None:
    # Four GPUs reading the host: theoretical 31.508 = the x16 uplink, link_raw
    # 15.754 (each GPU link is x8). Neither table has a host aggregate entry (the
    # ~25 GB/s of the box rules is not implemented, see effectiveness.py).
    (r,) = results.eff_rows(interim(ictx, 11, "host-k4"))
    assert (r.kind, r.n_flows) == ("host", 4)
    assert r.link_raw == pytest.approx(15.754)
    assert eff.row_baseline(r) is None
