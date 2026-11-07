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


def test_a_verify_failure_marks_the_record_invalid(ctx) -> None:
    (m,) = convert(SYNTHETIC_HOST, ctx, 2)
    assert m.kind == "concurrency" and m.path_class == "host"
    assert m.value == pytest.approx(28.0)
    assert m.ok is False and m.error == "verify FAIL"
    # four GPUs of one switch share its uplink: the ideal aggregate is one x16 link
    assert m.theoretical == pytest.approx(X16, abs=0.01)


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
