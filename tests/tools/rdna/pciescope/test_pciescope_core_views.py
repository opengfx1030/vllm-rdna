# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot tables (``views``) and the invocation environment (``env``)."""

from __future__ import annotations

import json
import shutil

import pytest
from pciescope import collectors, env, model, runner, store, views
from pciescope.model import Sample


def S(entity: str, metric: str, value: float, **tags: str) -> Sample:
    return Sample(1_000, entity, metric, value, "", dict(tags))


@pytest.fixture()
def one_tick(fx_topo, fx_root):
    """Samples of one real pass over the captured fabric."""
    e = env.Env(str(fx_root / "sys"), str(fx_root / "proc"), "par1-cs25")
    res = e.build(fx_topo)
    tick = runner.snapshot(fx_topo, res.collectors)
    return tick, res


def test_summary_of_the_captured_fabric(fx_topo, one_tick):
    tick, res = one_tick
    doc = views.snapshot_summary(
        fx_topo,
        tick.samples,
        costs_ms=tick.costs_ms,
        collectors=[c.name for c in res.collectors],
        skipped=dict(res.skipped),
        ts_ns=tick.ts_ns,
    )
    assert [g["id"] for g in doc["gpus"]][:2] == ["gpu:cb", "gpu:ce"]  # HIP order
    ce = doc["gpus"][1]
    assert (ce["hip"], ce["smi"], ce["arch"], ce["slot"]) == (1, 6, "gfx1030", "A")
    assert ce["card_link"] == "link:c8:10.0-cc:00.0"
    assert ce["gpu_link"]["speed_gts"] == 16
    assert [lk["id"] for lk in doc["links"][:2]] == [
        "link:c0:03.1-c5:00.0",
        "link:00:01.1-01:00.0",
    ]
    assert all(set(lk) >= {"speed_gts", "width", "degraded"} for lk in doc["links"])
    assert doc["n_samples"] == len(tick.samples)
    assert "other_metrics" not in doc  # every metric of the passive set is known
    assert doc["aer"]["devices_checked"] == 66
    assert doc["asleep"] == {"ports": [], "gpus": []}  # the capture has no PM files
    assert json.loads(json.dumps(doc)) == doc  # JSON-ready


def test_asleep_branches_and_gpus_are_reported(fx_topo, fx_root, tmp_path):
    sys_copy = tmp_path / "sys"
    shutil.copytree(fx_root / "sys", sys_copy, symlinks=True)
    for bdf in ("c6:0c.0", "dc:00.0", "ce:00.0"):  # idle branch (as on the box), a GPU
        power = sys_copy / "bus/pci/devices" / f"0000:{bdf}" / "power"
        power.mkdir(exist_ok=True)
        (power / "runtime_status").write_text("suspended\n")
    e = env.Env(str(sys_copy), str(fx_root / "proc"), "par1-cs25")
    res = e.build(fx_topo, ["sysfs_link", "amdgpu"])
    tick = runner.snapshot(fx_topo, res.collectors)
    doc = views.snapshot_summary(
        fx_topo, tick.samples, collectors=[c.name for c in res.collectors]
    )
    assert doc["asleep"] == {
        "ports": ["dc:00.0", "dd:14.0", "dd:15.0"],
        "gpus": ["gpu:ce"],
    }
    ce = next(g for g in doc["gpus"] if g["id"] == "gpu:ce")
    cb = next(g for g in doc["gpus"] if g["id"] == "gpu:cb")
    assert ce["asleep"] is True and ce["busy_pct"] is None
    assert cb["asleep"] is False and cb["busy_pct"] is not None
    assert "other_metrics" not in doc
    text = views.render_snapshot(doc)
    assert (
        "asleep (runtime-suspended, attributes not read): "
        "ports dc:00.0 dd:14.0 dd:15.0; GPUs gpu:ce"
    ) in text


def test_degraded_link_and_kfd_rows(fx_topo):
    card = fx_topo.gpus["gpu:ce"].link
    samples = [
        S(card, "link.speed_gts", 8),
        S(card, "link.width", 8),
        S(card, "link.capacity_gbs", 7.88),
        S(card, "link.degraded", 1),
        S("gpu:ce", "gpu.busy_pct", 97.5),
        S("gpu:ce", "gpu.temp_junction_c", 71),
        S("gpu:ce", "kfd.evicted_ms", 120, pid="42", comm="python"),
        S("gpu:ce", "kfd.evicted_ms", 0, pid="7", comm="vllm"),
        S("gpu:ce", "mystery.metric", 1),
    ]
    doc = views.snapshot_summary(
        fx_topo, samples, collectors=["x"], ts_ns=1_700_000_000
    )
    ce = next(g for g in doc["gpus"] if g["id"] == "gpu:ce")
    assert ce["busy_pct"] == 97.5 and ce["temp_junction_c"] == 71
    assert [k["pid"] for k in ce["kfd"]] == [7, 42]  # sorted by pid
    assert ce["kfd"][1] == {"pid": 42, "comm": "python", "evicted_ms": 120}
    link = next(lk for lk in doc["links"] if lk["id"] == card)
    assert link["degraded"] is True and link["speed_gts"] == 8
    assert doc["other_metrics"] == {"mystery.metric": 1}
    text = views.render_snapshot(doc)
    assert "DEGRADED" in text
    assert f"{card} (gpu_link) 8 GT/s x8 = 7.88 GB/s" in text
    assert "KFD queues:" in text and "gpu:ce pid 42 python evicted 120 ms" in text
    assert "other metrics: {'mystery.metric': 1}" in text


def test_aer_rows_carry_types_and_untagged_counters_stay_plain(fx_topo):
    port = "port:c0:03.1"
    samples = [
        S(port, "aer.correctable", 3),
        S(port, "aer.error", 2, **{"class": "correctable", "type": "BadTLP"}),
        S(port, "aer.error", 1, **{"class": "correctable", "type": "RxErr"}),
        S("port:c8:00.0", "aer.correctable", 0),
        S("port:c8:00.0", "aer.fatal", 0),
    ]
    aer = views.snapshot_summary(fx_topo, samples)["aer"]
    assert aer["devices_checked"] == 2
    (row,) = aer["nonzero"]
    assert row["entity"] == port and row["class"] == "correctable"
    assert row["count"] == 3
    assert row["types"] == {"BadTLP": 2, "RxErr": 1}
    assert row["label"]  # the topology label of the port
    text = views.render_snapshot(
        views.snapshot_summary(fx_topo, samples, collectors=["aer"], ts_ns=None)
    )
    assert "AER: 1 non-zero counter(s) since boot (2 devices checked)" in text
    quiet = views.snapshot_summary(fx_topo, [S("port:c8:00.0", "aer.correctable", 0)])
    assert "AER: all zero (1 devices checked)" in views.render_snapshot(quiet)


# -- env ----------------------------------------------------------------------


def test_parse_names():
    assert env.parse_names(None) is None
    assert env.parse_names("") is None
    assert env.parse_names("default") is None
    assert env.parse_names("aer, kfd") == ["aer", "kfd"]
    assert env.parse_names(["aer,kfd", "amdgpu"]) == ["aer", "kfd", "amdgpu"]
    assert env.parse_names("all") == collectors.names()


def test_env_build_skips_unknown_and_unavailable(fx_topo, fx_root):
    e = env.Env(str(fx_root / "sys"), str(fx_root / "proc"), "par1-cs25")
    res = e.build(fx_topo, ["aer", "nope"])
    assert [c.name for c in res.collectors] == ["aer"]
    assert "unknown collector" in dict(res.skipped)["nope"]
    default = e.build(fx_topo)
    assert [c.name for c in default.collectors] == [
        "sysfs_link",
        "aer",
        "amdgpu",
        "kfd",
    ]
    assert [n for n, _why in default.skipped] == ["plx"]


def test_env_session_root(tmp_path, monkeypatch):
    assert env.Env(store_root=str(tmp_path)).session_root() == tmp_path
    monkeypatch.setenv(store.ENV_HOME, str(tmp_path / "h"))
    assert env.Env().session_root() == tmp_path / "h"


def test_load_topology_from_file_and_session(fx_topo, tmp_path):
    p = tmp_path / "topo.json"
    p.write_text(json.dumps(fx_topo.to_dict()))
    assert env.load_topology(p).to_dict() == fx_topo.to_dict()
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"topology": fx_topo.to_dict()}))
    assert env.load_topology(wrapped).to_dict() == fx_topo.to_dict()
    sess = store.Session.create(tmp_path / "s", "with-topo", topo=fx_topo)
    sess.close()
    assert env.load_topology(sess.name, str(tmp_path / "s")).to_dict() == (
        fx_topo.to_dict()
    )
    bare = store.Session.create(tmp_path / "s", "no-topo")
    bare.close()
    with pytest.raises(FileNotFoundError, match="has no topology.json"):
        env.load_topology(bare.name, str(tmp_path / "s"))


def test_close_collectors_calls_close_where_present():
    closed: list[str] = []

    class WithClose:
        def close(self) -> None:
            closed.append("yes")

    class Without:
        pass

    env.close_collectors([WithClose(), Without(), WithClose()])
    assert closed == ["yes", "yes"]
    assert model.now_ns() > 0
