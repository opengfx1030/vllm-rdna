# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data model: BDF/entity helpers, link math, records, registry, Topology JSON."""

from __future__ import annotations

import json

import pytest
from pciescope import model


def test_bdf_helpers():
    assert model.short_bdf("0000:ce:00.0") == "ce:00.0"
    assert model.short_bdf("ce:00.0") == "ce:00.0"
    assert model.short_bdf("0001:ce:00.0") == "0001:ce:00.0"
    assert model.full_bdf("ce:00.0") == "0000:ce:00.0"
    assert model.full_bdf("0000:ce:00.0") == "0000:ce:00.0"
    assert model.bdf_bus("0000:ce:00.0") == "ce"
    assert model.bdf_bus("07:00.0") == "07"


def test_entity_helpers():
    assert model.gpu_entity("0A") == "gpu:0a"
    assert model.port_entity("0000:c8:10.0") == "port:c8:10.0"
    assert model.link_entity("0000:c8:10.0", "0000:cc:00.0") == "link:c8:10.0-cc:00.0"
    assert model.switch_entity("A") == "switch:A"
    assert model.root_entity("c0") == "root:c0"
    assert model.entity_kind("gpu:ce") == "gpu"
    assert model.entity_kind(model.HOST) == "host"
    assert model.entity_id("gpu:ce") == "ce"
    assert model.split_link("link:c8:10.0-cc:00.0") == ("c8:10.0", "cc:00.0")


def test_link_math_matches_documented_ceilings():
    # Gen4 x16 raw: 16 GT/s * 16 lanes * 128/130 / 8 bits.
    assert model.raw_gbs(16.0, 16) == pytest.approx(31.508, abs=1e-3)
    assert model.raw_gbs(8.0, 16) == pytest.approx(15.754, abs=1e-3)
    assert model.raw_gbs(16.0, 8) == pytest.approx(15.754, abs=1e-3)
    # Practical: raw * MPS/(MPS+26) * 0.98 -> 28.0 GB/s at MPS 256.
    assert model.practical_gbs(16.0, 16) == pytest.approx(28.03, abs=0.01)
    assert model.practical_gbs(16.0, 16, mps=512) == pytest.approx(29.4, abs=0.05)
    assert model.practical_gbs(16.0, 16) < model.raw_gbs(16.0, 16)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("16.0 GT/s PCIe", 16.0),
        ("8.0 GT/s PCIe", 8.0),
        ("2.5 GT/s PCIe", 2.5),
        ("Unknown", None),
        ("16.0 GT/s", 16.0),
        ("", None),
        (None, None),
    ],
)
def test_parse_speed(text, expected):
    assert model.parse_speed_gts(text) == expected


def test_record_roundtrip():
    s = model.Sample(1, "gpu:ce", "gpu.busy_pct", 12.5, "%", {"a": "b"})
    m = model.Measurement(
        kind="bw",
        src="gpu:ce",
        dst="gpu:d3",
        direction="d2d-push",
        size_bytes=1 << 28,
        value=24.5,
        unit="GB/s",
        stats={"p50": 24.4, "n": 5.0},
        path_class="intra-switch",
        theoretical=31.5,
        efficiency=0.78,
        started_ns=10,
        ended_ns=20,
        gpus_used=["gpu:ce", "gpu:d3"],
    )
    e = model.Event(5, "gpu:ce", "thermal", "warn", "hot", {"c": 91})
    for rec in (s, m, e):
        back = model.record_from_dict(json.loads(json.dumps(rec.to_dict())))
        assert back == rec
    assert model.record_from_dict({"type": "nonsense"}) is None
    assert s.key() == ("gpu:ce", "gpu.busy_pct", (("a", "b"),))


def test_metric_registry_is_consistent():
    assert len(model.METRICS) >= 25
    for name, md in model.METRICS.items():
        assert md.name == name
        assert md.kind in ("gauge", "counter")
        assert md.unit
        assert md.doc
        assert md.entity
    assert model.metric_kind("aer.correctable") == "counter"
    assert model.metric_kind("link.speed_gts") == "gauge"
    assert model.metric_kind("no.such.metric") == "gauge"


def test_topology_json_roundtrip(fx_topo):
    doc = json.loads(json.dumps(fx_topo.to_dict()))
    back = model.Topology.from_dict(doc)
    assert back.to_dict() == fx_topo.to_dict()
    assert [g.id for g in back.gpu_list()] == [g.id for g in fx_topo.gpu_list()]
    assert back.path("gpu:ce", "gpu:0a").path_class == "cross-root"


def test_topology_lookups(fx_topo):
    assert fx_topo.gpu("gpu:ce") is fx_topo.gpu("ce") is fx_topo.gpu(1)
    assert fx_topo.gpu("hip:3").id == "gpu:d8"
    assert fx_topo.gpu("ce:00.0").id == "gpu:ce"
    with pytest.raises(KeyError):
        fx_topo.gpu("gpu:zz")
    with pytest.raises(KeyError):
        fx_topo.gpu(42)
    assert [g.id for g in fx_topo.gpu_list("smi")][:2] == ["gpu:07", "gpu:0a"]
    assert [g.id for g in fx_topo.slot_gpus("A")] == [
        "gpu:ce",
        "gpu:d3",
        "gpu:d8",
        "gpu:db",
    ]
    assert [g.id for g in fx_topo.gpus_of_switch("B")][0] == "gpu:07"
    assert fx_topo.switch_of_gpu("gpu:0f").letter == "B"


def test_paths(fx_topo):
    intra = fx_topo.path("gpu:ce", "gpu:cb")  # same sub-switch
    assert intra.path_class == "intra-switch"
    assert intra.lca_role == model.ROLE_SWITCH_SUB
    assert len(intra.physical_links) == 4  # two cards: card link + on-card link each
    far = fx_topo.path("gpu:ce", "gpu:d3")  # same switch, other sub-switch
    assert far.path_class == "intra-switch"
    assert far.lca_role == model.ROLE_SWITCH_UP
    cross = fx_topo.path("gpu:ce", "gpu:0a")
    assert cross.path_class == "cross-root"
    assert cross.lca is None
    assert any(h.link is None for h in cross.hops)  # the CPU fabric crossing
    assert "link:c0:03.1-c5:00.0" in cross.physical_links
    assert "link:00:01.1-01:00.0" in cross.physical_links
    host = fx_topo.path("host", "gpu:ce")
    assert host.path_class == "host"
    assert "link:c0:03.1-c5:00.0" in host.physical_links
    assert [h.direction for h in host.hops if h.link] == ["down"] * 4
    assert fx_topo.path("gpu:ce", "gpu:ce").path_class == "same"
    assert fx_topo.path_class("gpu:ce", "gpu:d3") == "intra-switch"
    # every link kind used on a path is a real link of the topology
    for lid in cross.physical_links:
        assert fx_topo.links[lid].physical


def test_every_core_metric_is_in_the_doc():
    from pathlib import Path

    doc = Path(__file__).parents[4] / "docs/rdna2/pciescope-2026-11-07.md"
    if not doc.is_file():  # a sparse checkout without docs/
        pytest.skip("docs/ is not checked out")
    text = doc.read_text()
    # plx.* and port.* are defined (and documented) by the switch reader
    core = [n for n, m in model.METRICS.items() if m.source != "plx"]
    assert core
    assert [n for n in core if f"`{n}`" not in text] == []
