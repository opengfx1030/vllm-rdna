# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive collectors against the captured par1-cs25 sysfs."""

from __future__ import annotations

import shutil
import struct

import pytest
from pciescope import collectors, model
from pciescope.collectors import aer, amdgpu, kfd, sysfs_link
from pciescope.sysfs import Sysfs


def _index(samples, metric):
    return {s.entity: s for s in samples if s.metric == metric}


@pytest.fixture()
def sys_copy(fx_root, tmp_path):
    """A writable copy of the captured sysfs (symlinks kept)."""
    dest = tmp_path / "sys"
    shutil.copytree(fx_root / "sys", dest, symlinks=True)
    return dest


# -- registry --------------------------------------------------------------


def test_build_default_set(fx_topo, fx_sysfs, fx_procfs):
    res = collectors.build(
        fx_topo, sysfs=fx_sysfs, options={"kfd": {"procfs": fx_procfs}}
    )
    assert [c.name for c in res.collectors] == [
        n for n in collectors.DEFAULT_PASSIVE if n != "plx"
    ]
    assert [n for n, _why in res.skipped] == ["plx"]  # live switch registers only
    for c in res.collectors:
        assert isinstance(c, model.Collector)
        assert c.available() == (True, "")


def test_build_reports_unknown_and_unavailable(fx_topo, tmp_path):
    empty = Sysfs(tmp_path)
    res = collectors.build(fx_topo, ["nope", "aer", "amdgpu"], sysfs=empty)
    reasons = dict(res.skipped)
    assert "unknown collector" in reasons["nope"]
    assert "aer" in reasons
    assert "amdgpu" in reasons
    assert res.collectors == []


# -- sysfs_link ------------------------------------------------------------


def test_sysfs_link_samples(fx_topo, fx_sysfs):
    samples = sysfs_link.make(fx_topo, sysfs=fx_sysfs).collect()
    assert len({s.ts_ns for s in samples}) == 1
    speed = _index(samples, "link.speed_gts")
    width = _index(samples, "link.width")
    cap = _index(samples, "link.capacity_gbs")
    degr = _index(samples, "link.degraded")
    physical = [lk.id for lk in fx_topo.links.values() if lk.physical]
    assert sorted(speed) == sorted(width) == sorted(cap) == sorted(degr)
    assert sorted(speed) == sorted(physical)
    assert all(s.value == 16.0 for s in speed.values())
    assert all(s.value == 16.0 for s in width.values())
    assert all(s.value == pytest.approx(31.508, abs=1e-3) for s in cap.values())
    assert all(s.value == 0.0 for s in degr.values())
    own = _index(samples, "pcie.link_speed_gts")
    assert own["gpu:ce"].value == 16.0
    assert "port:c8:10.0" in own


def test_sysfs_link_detects_degraded_link(fx_topo, sys_copy):
    # both ends of the card link of gpu:ce report 8 GT/s x8
    for bdf in ("0000:c8:10.0", "0000:cc:00.0"):
        d = sys_copy / "bus/pci/devices" / bdf
        (d / "current_link_speed").write_text("8.0 GT/s PCIe\n")
        (d / "current_link_width").write_text("8\n")
    samples = sysfs_link.make(fx_topo, sysfs=Sysfs(sys_copy)).collect()
    lid = "link:c8:10.0-cc:00.0"
    assert _index(samples, "link.speed_gts")[lid].value == 8.0
    assert _index(samples, "link.width")[lid].value == 8.0
    assert _index(samples, "link.capacity_gbs")[lid].value == pytest.approx(
        7.877, abs=1e-3
    )
    assert _index(samples, "link.degraded")[lid].value == 1.0
    # the link above the neighbour is untouched
    assert _index(samples, "link.degraded")["link:d0:00.0-d1:00.0"].value == 0.0


# -- aer -------------------------------------------------------------------


def test_parse_aer():
    text = "RxErr 0\nBadTLP 2\nBadDLLP 0\nTOTAL_ERR_COR 2\n"
    assert aer.parse_aer(text) == {
        "RxErr": 0,
        "BadTLP": 2,
        "BadDLLP": 0,
        "TOTAL_ERR_COR": 2,
    }
    assert aer.parse_aer("garbage line here\n\n") == {}


def test_aer_samples_on_fixture(fx_topo, fx_sysfs):
    samples = aer.make(fx_topo, sysfs=fx_sysfs).collect()
    cor = _index(samples, "aer.correctable")
    assert len(cor) == 66
    assert {"aer.nonfatal", "aer.fatal"} <= {s.metric for s in samples}
    root = _index(samples, "aer.rootport_correctable")
    assert root["port:00:01.1"].value == 10.0
    assert root["port:c0:03.1"].value == 10.0
    # the captured box has one advisory-non-fatal correctable error on some ports
    typed = [s for s in samples if s.metric == "aer.error"]
    assert typed
    assert all(s.tags["class"] == "correctable" for s in typed)
    assert {s.tags["type"] for s in typed} == {"NonFatalErr"}
    assert all(s.value > 0 for s in typed)


def test_aer_counts_a_new_fatal_error(fx_topo, sys_copy):
    d = sys_copy / "bus/pci/devices/0000:ce:00.0"
    (d / "aer_dev_fatal").write_text(
        "Undefined 0\nDLP 1\nSDES 0\nTLP 0\nFCP 0\nCmpltTO 0\nCmpltAbrt 0\n"
        "RxOF 0\nMalfTLP 0\nECRC 0\nUnsupReq 0\nACSViol 0\n"
        "UncorrIntErr 0\nBlockedTLP 0\nAtomicOpBlocked 0\nTLPBlockedErr 0\n"
        "PoisonTLPBlocked 0\nTOTAL_ERR_FATAL 1\n"
    )
    samples = aer.make(fx_topo, sysfs=Sysfs(sys_copy)).collect()
    assert _index(samples, "aer.fatal")["gpu:ce"].value == 1.0
    kinds = [
        (s.tags["class"], s.tags["type"], s.value)
        for s in samples
        if s.metric == "aer.error" and s.entity == "gpu:ce"
    ]
    assert kinds == [("fatal", "DLP", 1.0)]


# -- amdgpu ----------------------------------------------------------------


def _metrics_v13(**kw: int) -> bytes:
    raw = bytearray(0x78)
    struct.pack_into("<HBB", raw, 0, 0x78, 1, 3)
    struct.pack_into("<HHH", raw, 4, kw.get("edge", 40), kw.get("hot", 45), 38)
    struct.pack_into("<HHHH", raw, 0x10, kw.get("gfx", 70), 12, 3, kw.get("power", 180))
    struct.pack_into("<I", raw, 0x44, 0)
    struct.pack_into("<HHH", raw, 0x48, 0, kw.get("width", 16), kw.get("speed", 160))
    struct.pack_into("<Q", raw, 0x70, kw.get("indep", 0))
    return bytes(raw)


def test_parse_gpu_metrics():
    m = amdgpu.parse_gpu_metrics(_metrics_v13(indep=0x20))
    assert m is not None
    assert m["temp_junction_c"] == 45.0
    assert m["busy_pct"] == 70.0
    assert m["mem_busy_pct"] == 12.0
    assert m["power_w"] == 180.0
    assert m["link_width"] == 16.0
    assert m["link_speed_gts"] == 16.0
    assert m["throttle_status"] == float(0x20)
    assert amdgpu.parse_gpu_metrics(None) is None
    assert amdgpu.parse_gpu_metrics(b"\x00\x01") is None
    bad = bytearray(_metrics_v13())
    bad[2] = 2  # format revision 2 (APU layout) is not decoded
    assert amdgpu.parse_gpu_metrics(bytes(bad)) is None
    # 0xFFFF marks an unavailable field and is dropped, not reported as 65535
    ff = amdgpu.parse_gpu_metrics(_metrics_v13(hot=0xFFFF, power=0xFFFF))
    assert ff is not None
    assert "temp_junction_c" not in ff
    assert "power_w" not in ff


def test_parse_pp_dpm_pcie_and_pcie_bw():
    text = "0: 2.5GT/s, x1 150Mhz \n1: 16.0GT/s, x16 619Mhz *\n"
    assert amdgpu.parse_pp_dpm_pcie(text) == (16.0, 16)
    assert amdgpu.parse_pp_dpm_pcie("0: 2.5GT/s, x1 *\n") == (2.5, 1)
    assert amdgpu.parse_pp_dpm_pcie("0: 2.5GT/s, x1\n") is None
    assert amdgpu.parse_pp_dpm_pcie(None) is None
    assert amdgpu.parse_pcie_bw("1000 2000 256\n") == (1000, 2000, 256)
    assert amdgpu.parse_pcie_bw("x") is None


def test_amdgpu_samples_on_fixture(fx_topo, fx_sysfs):
    c = amdgpu.make(fx_topo, sysfs=fx_sysfs)
    assert c.available() == (True, "")
    samples = c.collect()
    by_gpu: dict[str, dict[str, model.Sample]] = {}
    for s in samples:
        by_gpu.setdefault(s.entity, {})[s.metric] = s
    assert sorted(by_gpu) == sorted(g.id for g in fx_topo.gpu_list())
    ce = by_gpu["gpu:ce"]
    assert ce["gpu.vram_total_bytes"].value == 32195477504.0
    assert ce["gpu.link_speed_gts"].value == 16.0
    assert ce["gpu.link_width"].value == 16.0
    assert 0 < ce["gpu.temp_junction_c"].value < 125
    assert ce["gpu.temp_junction_c"].unit == "C"
    # pcie_bw is opt-in and absent on this hardware: never emitted by default
    assert not any("pcie_rx" in s.metric for s in samples)


def test_amdgpu_temperature_helpers(fx_topo, fx_sysfs):
    gpu = fx_topo.gpu("gpu:ce")
    temps = amdgpu.read_temps_c(fx_sysfs, gpu)
    assert set(temps) == {"edge", "junction", "mem"}
    assert amdgpu.junction_temp_c(fx_sysfs, gpu) == temps["junction"]
    nohw = model.Gpu(**{**gpu.to_dict(), "hwmon": None})
    assert amdgpu.read_temps_c(fx_sysfs, nohw) == {}
    assert amdgpu.junction_temp_c(fx_sysfs, nohw) is None


# -- kfd -------------------------------------------------------------------


def test_kfd_samples_and_helpers(fx_topo, fx_sysfs, fx_procfs):
    c = kfd.make(fx_topo, sysfs=fx_sysfs, procfs=fx_procfs)
    assert c.available() == (True, "")
    samples = c.collect()
    assert samples
    assert {s.metric for s in samples} == {"kfd.evicted_ms"}
    for s in samples:
        assert s.entity.startswith("gpu:")
        assert s.tags["pid"].isdigit()
        assert s.tags["comm"]
    gpu = fx_topo.gpu(samples[0].entity)
    pids = kfd.kfd_pids(fx_sysfs, gpu)
    assert int(samples[0].tags["pid"]) in pids
    assert set(kfd.evicted_ms(fx_sysfs, gpu).values()) == {0}
    assert kfd.evicted_ms(fx_sysfs, gpu, pids=[]) == {}
    pid = pids[0]
    assert kfd.process_name(pid, fx_procfs)
    assert kfd.process_name(999999999, fx_procfs) == ""


def test_kfd_eviction_is_reported(fx_topo, sys_copy, fx_procfs):
    sfs = Sysfs(sys_copy)
    base = sys_copy / "class/kfd/kfd/proc"
    pid_dir, stats = next(
        (p, s)
        for p in sorted(base.iterdir())
        if p.is_dir()
        for s in sorted(p.iterdir())
        if s.name.startswith("stats_")
    )  # some captured pids are empty dirs: listing order differs per filesystem
    (stats / "evicted_ms").write_text("1234\n")
    gpu_id = int(stats.name.split("_", 1)[1])
    gpu = next(g for g in fx_topo.gpus.values() if g.kfd_gpu_id == gpu_id)
    assert kfd.evicted_ms(sfs, gpu)[int(pid_dir.name)] == 1234
    samples = kfd.make(fx_topo, sysfs=sfs, procfs=fx_procfs).collect()
    hit = [s for s in samples if s.value == 1234.0]
    assert len(hit) == 1
    assert hit[0].entity == gpu.id
    assert hit[0].tags["pid"] == pid_dir.name


def test_kfd_unavailable_without_kfd(fx_topo, tmp_path):
    ok, why = kfd.make(fx_topo, sysfs=Sysfs(tmp_path)).available()
    assert not ok
    assert why
