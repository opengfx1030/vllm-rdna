# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit baseline and findings of the pciescope probe (pure rules, no box).

The expected numbers are the ones printed in
``docs/rdna2/pcie-iommu-audit-2026-11-06.md``; ranges are checked against the
midpoint rule of ``effectiveness.py``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope import model as m  # noqa: E402
from pciescope.probe import effectiveness as eff  # noqa: E402
from pciescope.probe.paths import FabricModel  # noqa: E402

MIB = 1 << 20
KIB = 1 << 10
RAW = m.raw_gbs(16.0, 16)  # 31.508


def row(**kw: object) -> eff.EffRow:
    base = eff.EffRow(
        kind="push",
        direction="fwd",
        path_class="cross-root",
        size=MIB,
        value=25.2,
        unit="GB/s",
        n_flows=1,
        theoretical=RAW,
        src="gpu:ce",
        dst="gpu:0a",
        label="push ce>0a 1MiB",
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def codes(found: list[eff.Finding]) -> list[str]:
    return [f.code for f in found]


# --- keys and the baseline table -------------------------------------------------


def test_baseline_key_format() -> None:
    assert eff.baseline_key("push", "intra-switch", MIB) == "push.intra.1MiB"
    assert eff.baseline_key("pull", "cross-root", 64 * KIB) == "pull.cross.64KiB"
    assert eff.baseline_key("sdma", "cross", 5 * MIB) == "sdma.cross.5MiB"
    assert eff.baseline_key("lat", "intra-switch", 0) == "lat.intra.0B"
    assert (
        eff.baseline_key("concurrent-push", "cross-root", pairs=3)
        == "concurrent-push.cross.pairs=3"
    )
    assert (
        eff.baseline_key("a2a-step", "cross-root", 4096, gpus=8)
        == "a2a-step.cross.4KiB.gpus=8"
    )
    with pytest.raises(ValueError, match="path class"):
        eff.baseline_key("push", "elsewhere")


def test_table_shape() -> None:
    table = eff.AUDIT_BASELINE
    assert len(table) == 48
    for key, base in table.items():
        assert base.key == key
        assert base.unit in ("GB/s", "us")
        assert base.tolerance >= 0.10
        assert base.value > 0
        assert "pcie-iommu-audit-2026-11-06.md" in base.source
        assert " line" in base.source  # every entry cites a table line


def test_single_pair_midpoints() -> None:
    t = eff.AUDIT_BASELINE
    # intra push 64 KiB: audit range 24.6-27.0 (A and B columns)
    assert t["push.intra.64KiB"].value == pytest.approx(25.8)
    assert t["push.intra.64KiB"].tolerance == pytest.approx(0.10)
    assert "24.6-27.0" in t["push.intra.64KiB"].source
    assert "midpoint" in t["push.intra.64KiB"].source
    assert t["push.cross.1MiB"].value == pytest.approx((25.1 + 26.8) / 2)
    assert t["pull.intra.1MiB"].value == pytest.approx((24.4 + 27.0) / 2)
    assert t["pull.cross.64KiB"].value == pytest.approx(5.75)
    assert t["pull.cross.1MiB"].value == pytest.approx(5.7)
    assert t["sdma.intra.64KiB"].value == pytest.approx(6.0)
    assert t["sdma.cross.64KiB"].value == pytest.approx(4.7)
    assert t["sdma.cross.64MiB"].value == pytest.approx((25.5 + 26.4) / 2)
    for kind in ("push", "pull", "sdma"):
        for cls in ("intra", "cross"):
            for size in ("64KiB", "1MiB", "5MiB", "64MiB"):
                assert f"{kind}.{cls}.{size}" in t


def test_concurrent_and_pattern_numbers() -> None:
    t = eff.AUDIT_BASELINE
    agg = {1: 25.2, 2: 27.6, 3: 18.7, 4: 17.0}
    bidir = {1: 53.5, 2: 55.1, 3: 39.5, 4: 35.3}
    for pairs in (1, 2, 3, 4):
        assert t[f"concurrent-push.cross.pairs={pairs}"].value == agg[pairs]
        assert t[f"concurrent-push-bidir.cross.pairs={pairs}"].value == bidir[pairs]
    assert t["fan-in.cross.flows=4"].value == 25.6
    assert t["fan-out.cross.flows=3"].value == 27.7
    assert t["fan-out.cross.flows=4"].value == 27.7
    assert t["ring.intra.flows=4"].value == 105.0
    assert t["ring.intra.flows=8"].value == 210.0
    # all-to-all inside a switch: audit range 93-107, midpoint 100, 7 % half range
    assert t["a2a.intra.flows=12"].value == pytest.approx(100.0)
    assert t["a2a.intra.flows=12"].tolerance == pytest.approx(0.10)


def test_latency_numbers() -> None:
    t = eff.AUDIT_BASELINE
    assert t["lat.intra.0B"].value == pytest.approx(6.2)  # 6.1-6.3
    assert t["lat.cross.0B"].value == pytest.approx(6.85)  # 6.8-6.9
    assert t["lat.intra.64KiB"].value == 8.4
    assert t["lat.cross.64KiB"].value == 9.0
    assert t["lat.intra.16KiB"].value == 7.0
    assert t["lat.cross.4KiB"].value == 7.2
    assert t["lat.cross.16KiB"].value == pytest.approx(7.35)  # 7.1-7.6
    assert t["lat.intra.0B"].unit == "us"
    assert t["a2a-step.intra.4KiB.gpus=4"].value == 10.4


def test_tolerance_widens_with_the_audit_spread() -> None:
    flat = eff.AUDIT_BASELINE["a2a-step.cross.4KiB.gpus=8"]
    assert flat.value == 15.7
    # other audit states gave 14.0 us: 1.7 / 15.7 = 10.8 % > 10 %
    assert flat.tolerance == pytest.approx(1.7 / 15.7)
    assert "14.0" in flat.source


def test_rccl_midpoint_of_the_audit_range() -> None:
    rccl = eff.AUDIT_BASELINE["rccl-allreduce.cross.4KiB.tp=8"]
    assert rccl.value == pytest.approx(112.0)  # 109-115 us
    assert rccl.unit == "us"
    assert "109" in rccl.source
    assert "115" in rccl.source


# --- compare_to_baseline -----------------------------------------------------------


def test_compare_bandwidth_verdicts() -> None:
    key = "pull.cross.1MiB"  # baseline 5.7 GB/s, tolerance 10 %
    ok = eff.compare_to_baseline(key, 5.65, "GB/s")
    assert ok is not None
    assert ok.verdict == "ok"
    assert ok.ratio == pytest.approx(5.65 / 5.7)
    assert ok.baseline == pytest.approx(5.7)
    slow = eff.compare_to_baseline(key, 4.0, "GB/s")
    assert slow is not None
    assert slow.verdict == "slower"
    fast = eff.compare_to_baseline(key, 25.0, "GB/s")
    assert fast is not None
    assert fast.verdict == "faster"


def test_compare_latency_verdicts_are_reversed() -> None:
    key = "lat.intra.0B"  # baseline 6.2 us
    slow = eff.compare_to_baseline(key, 9.0, "us")
    assert slow is not None
    assert slow.verdict == "slower"
    fast = eff.compare_to_baseline(key, 3.0, "us")
    assert fast is not None
    assert fast.verdict == "faster"
    ok = eff.compare_to_baseline(key, 6.5, "us")
    assert ok is not None
    assert ok.verdict == "ok"


def test_compare_edge_cases() -> None:
    assert eff.compare_to_baseline("push.intra.123KiB", 25.0, "GB/s") is None
    with pytest.raises(ValueError, match="unit"):
        eff.compare_to_baseline("push.intra.1MiB", 25.0, "us")


def test_efficiency() -> None:
    assert eff.efficiency(25.0, 31.508) == pytest.approx(25.0 / 31.508)
    assert eff.efficiency(25.0, None) is None
    assert eff.efficiency(25.0, 0.0) is None


# --- findings ----------------------------------------------------------------------


def test_clean_rows_give_no_findings() -> None:
    rows = [
        row(),
        row(path_class="intra-switch", dst="gpu:d3", label="push ce>d3"),
        row(kind="lat", unit="us", size=0, value=6.9, theoretical=None, label="lat"),
    ]
    assert eff.findings(rows) == []
    assert eff.findings([]) == []


def test_single_flow_push_below_70_percent_of_the_ceiling() -> None:
    ceiling = FabricModel.ceiling(RAW)  # 28.0 GB/s
    just_ok = row(value=0.70 * ceiling + 0.01)
    low = row(value=0.70 * ceiling - 0.01, label="slow push")
    assert eff.findings([just_ok]) == []
    (f,) = eff.findings([low])
    assert (f.severity, f.code, f.rows) == (
        "warn",
        "push-below-ceiling",
        ("slow push",),
    )
    assert "slow push" in f.text
    # not a single flow, or both directions, or no theoretical value: no rule
    assert eff.findings([replace(low, n_flows=2)]) == []
    assert eff.findings([replace(low, direction="bidir", n_flows=2)]) == []
    assert eff.findings([replace(low, theoretical=None)]) == []
    assert eff.findings([replace(low, kind="pull")]) == []


def test_cross_root_pull_penalty_is_info_when_it_matches_the_audit() -> None:
    intra = row(kind="pull", path_class="intra-switch", value=24.4, label="pull intra")
    cross = row(kind="pull", value=5.7, label="pull cross", theoretical=None)
    (f,) = eff.findings([intra, cross])
    assert (f.severity, f.code, f.rows) == (
        "info",
        "cross-root-pull-penalty",
        ("pull cross",),
    )
    assert "known" in f.text


def test_cross_root_pull_penalty_is_bad_when_it_is_off_the_audit() -> None:
    intra = row(kind="pull", path_class="intra-switch", value=24.4, label="pull intra")
    cross = row(kind="pull", value=3.0, label="pull cross", theoretical=None)
    (f,) = eff.findings([intra, cross])
    assert (f.severity, f.code) == ("bad", "cross-root-pull-penalty")


def test_cross_root_pull_needs_a_same_size_intra_pull_below_half() -> None:
    intra = row(kind="pull", path_class="intra-switch", value=24.4, label="pull intra")
    cross = row(kind="pull", value=5.7, label="pull cross", theoretical=None)
    assert eff.findings([cross]) == []
    assert eff.findings([replace(intra, size=64 * KIB), cross]) == []
    ok_cross = replace(cross, value=12.3)  # 50.4 % of 24.4
    assert eff.findings([intra, ok_cross]) == []


def test_cross_root_collapse_info_and_bad() -> None:
    single = row(label="1 pair")
    three = row(n_flows=3, value=18.7, label="3 pairs", theoretical=None)
    (f,) = eff.findings([single, three])
    assert (f.severity, f.code, f.rows) == ("info", "cross-root-collapse", ("3 pairs",))
    worse = replace(three, value=10.0)
    (g,) = eff.findings([single, worse])
    assert (g.severity, g.code) == ("bad", "cross-root-collapse")
    four = replace(three, n_flows=4, value=17.0, label="4 pairs")
    (h,) = eff.findings([single, four])
    assert (h.severity, h.code) == ("info", "cross-root-collapse")


def test_cross_root_collapse_thresholds() -> None:
    single = row()
    three = row(n_flows=3, theoretical=None, label="3 pairs")
    # 75 % of 25.2 = 18.9
    assert eff.findings([single, replace(three, value=19.0)]) == []
    assert codes(eff.findings([single, replace(three, value=18.8)])) == [
        "cross-root-collapse"
    ]
    # two pairs never count; no single-flow reference, no verdict
    assert eff.findings([single, replace(three, n_flows=2, value=5.0)]) == []
    assert eff.findings([replace(three, value=5.0)]) == []
    assert (
        eff.findings([replace(single, size=64 * KIB), replace(three, value=5.0)]) == []
    )


def test_cross_root_collapse_bidirectional() -> None:
    one = row(direction="bidir", n_flows=2, value=53.5, theoretical=None, label="1")
    three = replace(one, n_flows=6, value=39.5, label="3 pairs both ways")
    (f,) = eff.findings([one, three])
    assert (f.severity, f.code) == ("info", "cross-root-collapse")
    # a unidirectional single flow is not the reference of a bidirectional row
    assert eff.findings([row(), three]) == []


def test_exceeds_theoretical_over_125_percent() -> None:
    limit = 1.25 * RAW
    assert eff.findings([row(kind="sdma", value=limit - 0.01)]) == []
    (f,) = eff.findings([row(kind="sdma", value=limit + 0.01, label="too fast")])
    assert (f.severity, f.code, f.rows) == (
        "warn",
        "exceeds-theoretical",
        ("too fast",),
    )
    assert eff.findings([row(kind="sdma", value=99.0, theoretical=None)]) == []


def test_latency_above_baseline_times_1_25() -> None:
    base = eff.AUDIT_BASELINE["lat.intra.0B"].value  # 6.2 us
    lat = row(
        kind="lat",
        path_class="intra-switch",
        size=0,
        unit="us",
        theoretical=None,
        label="flag",
    )
    assert eff.findings([replace(lat, value=base * 1.25 - 0.01)]) == []
    (f,) = eff.findings([replace(lat, value=base * 1.25 + 0.01)])
    assert (f.severity, f.code, f.rows) == ("warn", "latency-above-baseline", ("flag",))
    # a latency without a baseline (size not in the audit) is never flagged
    assert eff.findings([replace(lat, size=123, value=500.0)]) == []
    # a faster latency is not a finding
    assert eff.findings([replace(lat, value=1.0)]) == []


def test_rccl_and_a2a_step_latency_use_their_keys() -> None:
    rccl = row(
        kind="rccl-allreduce",
        direction="none",
        size=4 * KIB,
        unit="us",
        value=112.0 * 1.25 + 1.0,
        n_flows=8,
        theoretical=None,
        label="tp8",
    )
    assert codes(eff.findings([rccl])) == ["latency-above-baseline"]
    assert eff.findings([replace(rccl, value=120.0)]) == []
    assert eff.findings([replace(rccl, n_flows=4, value=999.0)]) == []
    step = replace(rccl, kind="a2a-step", n_flows=4, path_class="intra-switch")
    assert eff.findings([replace(step, value=10.4 * 1.25 + 0.1)]) != []
    assert eff.findings([replace(step, value=10.4)]) == []


def test_host_link_underused_below_60_percent() -> None:
    host = row(
        kind="host",
        direction="h2d",
        path_class="host",
        value=0.6 * RAW + 0.01,
        src="host",
        dst="gpu:ce",
        label="h2d ce",
    )
    assert eff.findings([host]) == []
    (f,) = eff.findings([replace(host, value=0.6 * RAW - 0.01)])
    assert (f.severity, f.code, f.rows) == ("warn", "host-link-underused", ("h2d ce",))
    assert eff.findings([replace(host, value=1.0, theoretical=None)]) == []
    assert eff.findings([replace(host, value=1.0, unit="us")]) == []


def _topology_with_link() -> m.Topology:
    topo = m.Topology(host="synthetic", captured_ns=0)
    lid = m.link_entity("08:00.0", "09:00.0")
    topo.links[lid] = m.Link(
        id=lid,
        up="08:00.0",
        down="09:00.0",
        kind=m.LINK_GPU,
        physical=True,
        max_speed_gts=16.0,
        max_width=16,
        max_gbs=RAW,
        gpu="gpu:0a",
    )
    return topo


def test_downtrained_link_is_bad() -> None:
    text = "8 GT/s x16 (max 16 GT/s x16)"
    rows = [
        row(src="gpu:ce", dst="gpu:0a", label="to 0a"),
        row(label="x", dst="gpu:0f"),
    ]
    (f,) = eff.findings(rows, degraded_links={"gpu:0a": text})
    assert (f.severity, f.code, f.rows) == ("bad", "link-downtrained", ("to 0a",))
    assert text in f.text
    assert "gpu:0a" in f.text


def test_suspended_gpu_gives_info_instead() -> None:
    text = "2.5 GT/s x1 (max 16 GT/s x16)"
    rows = [row(label="to 0a")]
    (f,) = eff.findings(
        rows, degraded_links={"gpu:0a": text}, suspended=["gpu:0a", "gpu:ce"]
    )
    assert (f.severity, f.code, f.rows) == ("info", "link-suspended", ("to 0a",))
    # another GPU asleep does not excuse this one
    (g,) = eff.findings(rows, degraded_links={"gpu:0a": text}, suspended={"gpu:0f"})
    assert (g.severity, g.code) == ("bad", "link-downtrained")


def test_link_entity_is_attributed_to_its_gpu_through_the_model() -> None:
    topo = _topology_with_link()
    (lid,) = topo.links
    model = FabricModel(topo)
    text = "8 GT/s x16 (max 16 GT/s x16)"
    rows = [row(label="to 0a")]
    (f,) = eff.findings(rows, model, degraded_links={lid: text})
    assert (f.severity, f.code, f.rows) == ("bad", "link-downtrained", ("to 0a",))
    (g,) = eff.findings(rows, model, degraded_links={lid: text}, suspended=["gpu:0a"])
    assert (g.severity, g.code, g.rows) == ("info", "link-suspended", ("to 0a",))
    # without a model the link has no GPU: still reported, no rows attached
    (h,) = eff.findings(rows, degraded_links={lid: text}, suspended=["gpu:0a"])
    assert (h.severity, h.code, h.rows) == ("bad", "link-downtrained", ())


def test_several_degraded_entries_are_sorted_and_independent() -> None:
    found = eff.findings(
        [],
        degraded_links={"gpu:0f": "x", "gpu:0a": "y", "gpu:14": "z"},
        suspended=["gpu:0f"],
    )
    assert [(f.code, f.text.split(":")[0:2]) for f in found] == [
        ("link-downtrained", ["gpu", "0a"]),
        ("link-suspended", ["gpu", "0f"]),
        ("link-downtrained", ["gpu", "14"]),
    ]


def test_findings_are_reported_in_rule_order() -> None:
    intra = row(kind="pull", path_class="intra-switch", value=24.4, label="pull intra")
    cross = row(kind="pull", value=5.7, label="pull cross", theoretical=None)
    fast = row(kind="sdma", value=60.0, label="fast")
    low = row(value=5.0, label="low")
    found = eff.findings([fast, cross, intra, low], degraded_links={"gpu:0a": "x"})
    assert codes(found) == [
        "push-below-ceiling",
        "cross-root-pull-penalty",
        "exceeds-theoretical",
        "link-downtrained",
    ]
    assert (
        eff.findings([fast, cross, intra, low], degraded_links={"gpu:0a": "x"}) == found
    )
