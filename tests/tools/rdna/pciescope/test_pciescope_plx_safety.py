# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fences around the counter writes: policy, idle gate, health diff, ``perf run``.

The user approved the PerformancePAK writes for switch B only, on an idle switch,
with a health diff after every run.  These tests pin each of those conditions on
the capture of par1-cs25 and the stateful ``plx_fakes.FakeAtlas``: switch A is
never written, the idle gate refuses on any process, memory, busy GPU or foreign
lease, and ``perf run`` stops and undoes on the first link/error event.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import plx_fakes
import pytest
from pciescope.collectors import plx as plx_collector
from pciescope.plx import cli, health, helper, pcie, policy, preflight, regs
from pciescope.plx.access import LocalClient
from pciescope.sysfs import Sysfs

SW_A, SW_B = plx_fakes.SWITCH_A, plx_fakes.SWITCH_B
GPU_IDS_B = {"07": 28857, "0a": 18860, "0f": 6767, "14": 57637, "17": 4197}


@pytest.fixture(scope="module")
def capture():
    topo, chips, _ = plx_fakes.load_capture()
    return topo, chips


# -- the approved-switch policy -----------------------------------------------------


def test_parse_switches():
    assert policy.parse_switches("A,b") == frozenset("AB")
    assert policy.parse_switches(["b"]) == frozenset("B")
    assert policy.parse_switches("") == frozenset()
    with pytest.raises(ValueError, match="unknown switch letter"):
        policy.parse_switches("A,C")


def test_write_switches_precedence():
    env = {policy.ENV_WRITE_SWITCHES: "A"}
    assert policy.write_switches(None, {}) == frozenset("B")  # the approved default
    assert policy.DEFAULT_WRITE_SWITCHES == ("B",)
    assert policy.write_switches(None, env) == frozenset("A")
    assert policy.write_switches(["B"], env) == frozenset("B")
    assert policy.write_switches([], {}) == frozenset()


# -- the idle gate ------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_sysfs(root: Path, *, vram=None, busy=None, runtime=None, queues=None):
    """The amdgpu and KFD files of the five switch-B GPUs (idle unless told)."""
    vram, busy, runtime = vram or {}, busy or {}, runtime or {}
    for bus in GPU_IDS_B:
        dev = root / "bus" / "pci" / "devices" / f"0000:{bus}:00.0"
        if runtime.get(bus):
            _write(dev / "power" / "runtime_status", runtime[bus] + "\n")
            continue
        _write(dev / "power" / "runtime_status", "active\n")
        _write(dev / "mem_info_vram_used", f"{vram.get(bus, 40 << 20)}\n")
        _write(dev / "gpu_busy_percent", f"{busy.get(bus, 0)}\n")
    for pid, gpu_ids in (queues or {}).items():
        for i, gid in enumerate(gpu_ids):
            _write(
                root / "class/kfd/kfd/proc" / str(pid) / "queues" / str(i) / "gpuid",
                f"{gid}\n",
            )
    return Sysfs(root)


@pytest.fixture()
def idle_env(tmp_path, capture):
    """(topology, sysfs factory, lease dir, proc root) of an idle, leased switch B."""
    lease_dir = tmp_path / "lease"
    lease_dir.mkdir()
    holder = f"PlxSwitch 2026-11-07T15:00:00+01:00 test pid {os.getpid()}"
    (lease_dir / "PlxSwitch.slotB").write_text(holder + "\n")
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    (proc / "4242" / "cmdline").write_bytes(b"python\0-m\0vllm.entrypoints\0")

    def sysfs(**kw):
        return make_sysfs(tmp_path / "sys", **kw)

    return capture[0], sysfs, lease_dir, proc


def check(env, owner="PlxSwitch", **kw):
    topo, sysfs, lease_dir, proc = env
    return preflight.check_idle(
        topo,
        "B",
        owner,
        sysfs=kw.pop("sysfs", None) or sysfs(),
        proc_root=proc,
        lease_directory=lease_dir,
        busy_samples=2,
        sleep=lambda _s: None,
        **kw,
    )


def test_idle_leased_switch_passes(idle_env):
    rep = check(idle_env)
    assert rep.ok and rep.problems == ()
    assert [g.bus for g in rep.gpus] == ["07", "0a", "0f", "14", "17"]  # HIP 5..9
    assert [g.hip for g in rep.gpus] == [5, 6, 7, 8, 9]
    assert all(
        g.vram_used_mib == pytest.approx(40) and g.busy_pct == (0, 0) for g in rep.gpus
    )
    assert "IDLE" in rep.text() and "NOT" not in rep.text()
    assert rep.to_dict()["ok"] is True


def test_a_process_with_a_kfd_queue_blocks(idle_env):
    sysfs = idle_env[1](queues={4242: [GPU_IDS_B["0f"]]})
    rep = check(idle_env, sysfs=sysfs)
    assert not rep.ok
    (g,) = [g for g in rep.gpus if g.pids]
    assert g.bus == "0f" and g.pids == ((4242, "python -m vllm.entrypoints"),)
    assert any("KFD queues of pid 4242" in p and "gpu:0f" in p for p in rep.problems)
    assert "NOT IDLE" in rep.text() and "4242 python -m vllm" in rep.text()


def test_kfd_stats_directories_are_no_evidence_of_use(idle_env, tmp_path):
    # every KFD process lists a stats_<gpu_id> directory for all GPUs
    for gid in GPU_IDS_B.values():
        _write(
            tmp_path / "sys/class/kfd/kfd/proc/777" / f"stats_{gid}" / "evicted_ms",
            "0\n",
        )
    assert check(idle_env).ok
    owners = preflight.kfd_queue_owners(Sysfs(tmp_path / "sys"))
    assert owners == {}


def test_kfd_queue_owners_maps_gpu_ids_to_pids(idle_env):
    sysfs = idle_env[1](queues={30: [1, 2], 20: [2], 10: []})
    _write(Path(sysfs.root) / "class/kfd/kfd/proc/notapid/queues/0/gpuid", "9\n")
    assert preflight.kfd_queue_owners(sysfs) == {1: [30], 2: [20, 30]}


def test_gpu_memory_and_busy_block(idle_env):
    sysfs = idle_env[1](vram={"0a": 26 << 30}, busy={"14": 99})
    rep = check(idle_env, sysfs=sysfs)
    assert not rep.ok
    text = "\n".join(rep.problems)
    assert "gpu:0a (hip 6): 26624 MiB of GPU memory in use" in text
    assert "gpu:14 (hip 8): busy 99%" in text
    assert "gpu:0f" not in text


def test_unreadable_usage_files_refuse(idle_env, tmp_path):
    sysfs = idle_env[1]()
    (tmp_path / "sys/bus/pci/devices/0000:0a:00.0/gpu_busy_percent").unlink()
    (tmp_path / "sys/bus/pci/devices/0000:0f:00.0/mem_info_vram_used").unlink()
    text = "\n".join(check(idle_env, sysfs=sysfs).problems)
    assert "gpu:0a (hip 6): gpu_busy_percent unreadable" in text
    assert "gpu:0f (hip 7): memory use unreadable" in text


def test_a_suspended_gpu_is_not_read_and_counts_as_idle(idle_env):
    sysfs = idle_env[1](runtime={"17": "suspended"})
    rep = check(idle_env, sysfs=sysfs)
    assert rep.ok
    (g,) = [g for g in rep.gpus if g.bus == "17"]
    assert g.asleep and g.vram_used_mib is None and g.busy_pct == ()
    assert "asleep" in rep.text()


def test_foreign_or_missing_lease_blocks(idle_env):
    lease_dir = idle_env[2]
    (lease_dir / "ExpertStats.slotB").write_text(
        f"ExpertStats 2026-11-07T13:42:49+01:00 t10_run.sh pid {os.getpid()}\n"
    )
    text = "\n".join(check(idle_env).problems)
    assert "ExpertStats" in text and "slotB" in text
    (lease_dir / "PlxSwitch.slotB").unlink()
    (lease_dir / "ExpertStats.slotB").unlink()
    text = "\n".join(check(idle_env).problems)
    assert "PlxSwitch.slotB" in text  # the file the owner has to hold


def test_no_owner_refuses(idle_env, monkeypatch):
    monkeypatch.delenv("PCIESCOPE_OWNER", raising=False)
    rep = check(idle_env, owner=None)
    assert not rep.ok and any("no owner" in p for p in rep.problems)


def test_a_holder_of_the_w7800_lease_blocks(idle_env):
    (idle_env[2] / "Lead.w7800-07").write_text(
        f"Lead 2026-11-07T15:00:00+01:00 hip5 pid {os.getpid()}\n"
    )
    rep = check(idle_env)
    assert not rep.ok
    assert any("w7800-07 (bus 07) is held by Lead" in p for p in rep.problems)


def test_switch_without_gpus_is_not_idle(idle_env):
    topo, _, lease_dir, proc = idle_env
    rep = preflight.check_idle(
        topo,
        "C",
        "PlxSwitch",
        sysfs=Sysfs("/nonexistent"),
        proc_root=proc,
        lease_directory=lease_dir,
    )
    assert not rep.ok and "no GPU below switch C" in rep.problems[0]


# -- health snapshot and diff -------------------------------------------------------


def make_collector(capture, **kw):
    topo, windows = capture
    chips, fakes = {}, {}
    for bdf, win in windows.items():
        chips[bdf], fakes[bdf] = plx_fakes.chip_access(win, bdf)
    coll = plx_collector.PlxCollector(topo, client=LocalClient(chips), **kw)
    return coll, fakes


AER_TEXT = "ECRC 0\nUnsupReq 1\nTOTAL_ERR_NONFATAL 1\n"


def test_kernel_aer_reads_every_function_below_the_switch(capture, tmp_path):
    topo, _ = capture
    nodes = sorted(n.bdf for n in topo.nodes.values() if n.switch == "B")
    first, other = nodes[0], next(n.bdf for n in topo.nodes.values() if n.switch == "A")
    for bdf in (first, other):
        dev = tmp_path / "bus/pci/devices" / f"0000:{bdf}"
        _write(dev / "aer_dev_nonfatal", AER_TEXT)
        _write(dev / "aer_dev_correctable", "RxErr 0\nTOTAL_ERR_COR 0\n")
    got = health.kernel_aer(topo, "B", Sysfs(tmp_path))
    assert list(got) == [first]  # the switch A function is not part of switch B
    assert got[first]["aer_dev_nonfatal"] == {
        "ECRC": 0,
        "UnsupReq": 1,
        "TOTAL_ERR_NONFATAL": 1,
    }
    assert "aer_dev_fatal" not in got[first]


def snap_of(capture, switch="B"):
    topo, _ = capture
    coll, _ = make_collector(capture, switches=switch)
    coll.collect()
    return health.capture(coll.snapshot(), topo, switch, Sysfs("/nonexistent"))


def test_capture_holds_the_ports_of_one_switch_only(capture):
    snap = snap_of(capture)
    assert set(snap["ports"]) >= {"04:10.0", "0c:00.0", "11:00.0", "11:10.0", "01:00.0"}
    assert all(not b.startswith(("c5", "c8", "ce")) for b in snap["ports"])
    assert snap["ports"]["04:10.0"]["speed_gts"] == 16.0
    assert health.diff(snap, copy.deepcopy(snap)) == []


def test_diff_names_every_changed_field_and_counter(capture):
    before = snap_of(capture)
    before["kernel"] = {"04:10.0": {"aer_dev_correctable": {"RxErr": 3}}}
    after = copy.deepcopy(before)
    after["ports"]["04:10.0"]["width"] = 8
    after["ports"]["0c:00.0"]["dev_status"] = 0x8
    del after["ports"]["11:00.0"]
    after["kernel"]["04:10.0"]["aer_dev_correctable"]["RxErr"] = 5
    after["kernel"]["04:10.0"]["aer_dev_nonfatal"] = {"TLP": 1}
    got = health.diff(before, after)
    assert "port 04:10.0 width: 16 -> 8" in got
    old = before["ports"]["0c:00.0"]["dev_status"]
    assert f"port 0c:00.0 dev_status: {old:#x} -> 0x8" in got
    assert "port 11:00.0: vanished" in got
    assert "kernel 04:10.0 aer_dev_correctable RxErr: 3 -> 5" in got
    assert "kernel 04:10.0 aer_dev_nonfatal TLP: 0 -> 1" in got
    assert len(got) == 5


# -- perf run over fakes ------------------------------------------------------------


class FakeBox:
    """Both chips with persistent state behind a patched ``connect``."""

    def __init__(self, capture):
        self.topo, windows = capture
        self.fakes = {b: plx_fakes.FakeAtlas(w) for b, w in windows.items()}
        self.connects: list[tuple[dict, object]] = []

    def connect(self, chips, *, allow_writes=False, sysfs=None):
        self.connects.append((dict(chips), allow_writes))
        return LocalClient(
            {
                b: helper.ChipAccess(
                    b,
                    port,
                    self.fakes[b],
                    allow_writes=helper.chip_may_write(b, allow_writes),
                    poll_s=0.02,
                )
                for b, port in chips.items()
            }
        )


@pytest.fixture()
def box(capture, monkeypatch):
    fake_box = FakeBox(capture)
    monkeypatch.setattr(cli, "_topology", lambda args: (fake_box.topo, None))
    monkeypatch.setattr(plx_collector, "connect", fake_box.connect)
    monkeypatch.setattr(health, "kernel_aer", lambda *a, **k: {})  # no real /sys
    monkeypatch.delenv(policy.ENV_WRITE_SWITCHES, raising=False)
    return fake_box


def run(capsys, *argv):
    rc = cli.main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


RUN = ("perf", "run", "--allow-switch-writes", "--json", "--interval", "0.05")


def test_perf_run_enables_and_undoes_the_counters_on_switch_b(box, capsys, tmp_path):
    raw = tmp_path / "raw.jsonl"
    rc, out, _ = run(
        capsys, *RUN, "--switch", "B", "--duration", "0.5", "--raw", str(raw)
    )
    assert rc == 0
    doc = json.loads(out)
    assert doc["switch"] == "B" and doc["ports"] == [16, 32, 48, 64, 80]
    assert doc["anomaly"] is None and doc["health_changes"] == []
    assert doc["undo_mismatch"] == [] and doc["interrupted"] is False
    regs_doc = doc["registers"]
    assert regs_doc["0x768@16"]["before"] == regs_doc["0x768@16"]["after"]
    assert regs_doc["0x768@16"]["during"] & regs.PERF_INGRESS_BIT
    assert (
        regs_doc["0xf30@32"]["during"] == regs.PERF_EGRESS_ENABLE_BIT
    )  # bit 21 reads 0
    assert regs_doc["0x3e0@16"]["during"] == regs.PERF_CTRL_START
    assert regs_doc["0x3e0@16"]["after"] == regs.PERF_CTRL_STOP
    assert doc["rows"], "no rate sample in the run"
    assert box.fakes[SW_A].writes == []  # switch A was not written, not even once
    lo, hi = regs.port_window(0), regs.port_window(1)
    assert [w for w in box.fakes[SW_B].writes if lo <= w[0] < hi] == []  # nor port 0
    lines = [json.loads(x) for x in raw.read_text().splitlines()]
    assert lines and lines[0]["switch"] == "B"
    assert sorted(lines[-1]["counters"], key=int) == ["16", "32", "48", "64", "80"]
    assert set(lines[-1]["counters"]["16"]) == {"ts_ns", "rx", "tx"}
    stamps = [c["16"]["ts_ns"] for c in (x["counters"] for x in lines)]
    assert stamps == sorted(set(stamps))  # one line per new capture
    assert [c for c in box.connects if c[1]] == [({SW_B: 48}, frozenset({SW_B}))]


def test_perf_run_refuses_a_switch_the_user_has_not_approved(box, capsys):
    rc, out, err = run(capsys, *RUN, "--switch", "A", "--duration", "0.2")
    assert rc == 2 and out == ""
    assert "approved for switch B only, not A" in err
    assert box.connects == []  # no helper was even started
    assert box.fakes[SW_A].writes == [] and box.fakes[SW_B].writes == []


def test_perf_run_honours_the_env_override(box, capsys, monkeypatch):
    monkeypatch.setenv(policy.ENV_WRITE_SWITCHES, "A,B")
    rc, out, _ = run(capsys, *RUN, "--switch", "A", "--duration", "0.2")
    assert rc == 0 and json.loads(out)["ports"] == [0, 16, 32, 48, 64, 80]
    assert box.fakes[SW_A].writes and box.fakes[SW_B].writes == []


def test_perf_run_needs_one_switch(box, capsys):
    rc, _, err = run(capsys, *RUN, "--duration", "0.2")
    assert rc == 2 and "needs --switch A|B" in err
    assert box.connects == []


def test_perf_run_stages_the_ports(box, capsys):
    rc, out, _ = run(
        capsys, *RUN, "--switch", "B", "--ports", "16", "--duration", "0.3"
    )
    assert rc == 0 and json.loads(out)["ports"] == [16]
    writes = box.fakes[SW_B].writes
    stations = {
        (off - regs.PORT_WINDOW_BASE) // regs.PORT_WINDOW_SIZE // 16
        for off, _ in writes
    }
    assert stations == {1}  # only the station of port 16
    f30 = [
        (off - regs.PORT_WINDOW_BASE) // regs.PORT_WINDOW_SIZE
        for off, _ in writes
        if (off - regs.PORT_WINDOW_BASE) % regs.PORT_WINDOW_SIZE == regs.REG_PERF_EGRESS
    ]
    assert set(f30) == {16}


@pytest.mark.parametrize("ports", ["0", "16,7"])
def test_perf_run_rejects_ports_outside_the_default_set(box, capsys, ports):
    with pytest.raises(SystemExit, match="not measurable"):
        cli.main([*RUN, "--switch", "B", "--ports", ports])
    assert box.connects == []  # port 0 hosts the excluded GPU of HIP 5


def test_perf_plan_follows_the_staged_ports(capsys):
    rc, out, _ = run(
        capsys,
        "perf",
        "plan",
        "--fixture",
        str(plx_fakes.CAPTURE),
        "--switch",
        "B",
        "--ports",
        "16,32",
        "--json",
    )
    assert rc == 0 and json.loads(out)["B"]["ports"] == [16, 32]


def test_require_idle_refuses_before_anything_is_written(box, capsys, monkeypatch):
    seen: dict[str, Any] = {}

    def not_idle(topo, switch, owner, **kw):
        seen.update(switch=switch, owner=owner)
        return preflight.IdleReport(switch, owner, False, (), ("pid 1 uses gpu:0a",))

    monkeypatch.setattr(preflight, "check_idle", not_idle)
    rc, out, err = run(
        capsys,
        *RUN,
        "--switch",
        "B",
        "--require-idle",
        "--owner",
        "PlxSwitch",
        "--duration",
        "0.2",
    )
    assert rc == 3 and out == ""
    assert seen == {"switch": "B", "owner": "PlxSwitch"}
    assert "NOT IDLE" in err and "pid 1 uses gpu:0a" in err
    assert box.connects == []


def test_require_idle_lets_an_idle_switch_run(box, capsys, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "check_idle",
        lambda topo, switch, owner, **kw: preflight.IdleReport(
            switch, owner, True, (), ()
        ),
    )
    rc, out, err = run(
        capsys, *RUN, "--switch", "B", "--require-idle", "--duration", "0.2"
    )
    assert rc == 0 and "IDLE" in err and json.loads(out)["switch"] == "B"


def test_perf_preflight_verb(box, capsys, monkeypatch):
    calls = []

    def verdict(topo, switch, owner, **kw):
        calls.append((switch, owner))
        return preflight.IdleReport(switch, owner, owner == "PlxSwitch", (), ())

    monkeypatch.setattr(preflight, "check_idle", verdict)
    rc, out, _ = run(
        capsys, "perf", "preflight", "--switch", "b", "--owner", "PlxSwitch", "--json"
    )
    assert rc == 0 and json.loads(out)["ok"] is True
    rc, out, _ = run(capsys, "perf", "preflight", "--switch", "B", "--owner", "Other")
    assert rc == 3 and "NOT IDLE" in out
    assert calls == [("B", "PlxSwitch"), ("B", "Other")]
    rc, _, err = run(capsys, "perf", "preflight", "--owner", "PlxSwitch")
    assert rc == 2 and "needs --switch" in err
    assert box.connects == []  # the preflight never opens BAR0


def poke(fake, port, name, fn):
    win = fake.win[port]
    st = pcie.decode_static(lambda o: win.get(o, 0))
    off = dict(pcie.dynamic_offsets(st))[name]
    win[off] = fn(win.get(off, 0))


def test_a_link_change_ends_the_run_and_undoes_the_counters(box, capsys):
    # the link of port 16 drops to 8 GT/s while the counters run
    timer = threading.Timer(
        0.25, poke, (box.fakes[SW_B], 16, "lnk", lambda v: (v & ~(0xF << 16)) | 3 << 16)
    )
    timer.start()
    try:
        rc, out, err = run(capsys, *RUN, "--switch", "B", "--duration", "30")
    finally:
        timer.cancel()
    assert rc == 4  # anomaly
    doc = json.loads(out)
    assert doc["anomaly"]["name"] == "link_change"
    assert "event link_change port:04:10.0" in err
    assert any("lnk" in c or "speed" in c for c in doc["health_changes"])
    # the counters are off again: 0x768 restored, 0x3E0 at the SDK stop value
    r = doc["registers"]
    assert r["0x768@16"]["after"] == r["0x768@16"]["before"]
    assert r["0x3e0@16"]["after"] == regs.PERF_CTRL_STOP
    assert box.fakes[SW_A].writes == []


def test_sigterm_stops_the_run_and_restores_the_handler(box, capsys, monkeypatch):
    seen: dict[str, Any] = {}
    ticks: list[int] = []

    def sleep(_s):
        ticks.append(1)
        if len(ticks) == 2:
            seen["handler"] = signal.getsignal(signal.SIGTERM)
            raise KeyboardInterrupt  # what the SIGTERM handler does

    clock = SimpleNamespace(
        sleep=sleep, monotonic=time.monotonic, time=time.time, strftime=time.strftime
    )
    monkeypatch.setattr(cli, "time", clock)
    before = signal.getsignal(signal.SIGTERM)
    rc, out, _ = run(capsys, *RUN, "--switch", "B", "--duration", "60")
    assert seen["handler"].__name__ == "_raise_interrupt"
    assert signal.getsignal(signal.SIGTERM) is before
    doc = json.loads(out)
    assert rc == 0 and doc["interrupted"] is True
    assert doc["registers"]["0x3e0@16"]["after"] == regs.PERF_CTRL_STOP
    with pytest.raises(KeyboardInterrupt):
        cli._raise_interrupt()


def test_the_anomaly_events_are_the_link_and_error_ones():
    ev = SimpleNamespace
    assert cli._is_anomaly(ev(name="link_change", severity="info"))
    assert cli._is_anomaly(ev(name="dpc_triggered", severity="error"))
    assert cli._is_anomaly(ev(name="plx_perf_error", severity="error"))
    assert cli._is_anomaly(ev(name="plx_error_flags", severity="warn"))
    assert not cli._is_anomaly(ev(name="plx_error_flags", severity="info"))  # at start
    assert not cli._is_anomaly(ev(name="plx_perf_start", severity="info"))


def test_undo_check_exempts_only_the_strobe_and_the_stop_value():
    before = {"0x768@16": 0x09000000, "0xf30@16": 0, "0x3e0@16": 0, "0x3f0@16": 0x21}
    ok = dict(before) | {"0x3e0@16": regs.PERF_CTRL_STOP, "0x3f0@16": 0x25}
    assert cli._undo_mismatch(before, ok) == []
    bad = dict(ok) | {"0x768@16": 0x29000000, "0xf30@16": 0x40, "0x3e0@16": 0xD8000000}
    assert cli._undo_mismatch(before, bad) == [
        "0x768@16: 0x09000000 -> 0x29000000",
        "0xf30@16: 0x00000000 -> 0x00000040",
        "0x3e0@16: 0x00000000 -> 0xd8000000",
    ]
