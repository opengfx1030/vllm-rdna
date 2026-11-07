# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The probe runner: protocol, guards and event order against a probe double.

The binary is ``fixtures/probe/fake_p2p_probe.py`` (SYNTHETIC numbers, protocol
of the real ``p2p_probe batch``); the box is a writable copy of the captured
par1-cs25 sysfs, so temperatures and KFD ``evicted_ms`` counters can change in
the middle of a run. No GPU is touched.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pciescope.model import Event, Measurement, Record  # noqa: E402
from pciescope.probe import plan as pl  # noqa: E402
from pciescope.probe import runner as rn
from pciescope_probe_harness import FakeBox, make_box, wait_for  # noqa: E402

SLOT_B = ["0a", "0f", "14", "17"]


@dataclass(frozen=True)
class QuickStepLimits(rn.RunLimits):
    """Step limit of half a second (the default floor is ten seconds)."""

    def step_limit(self, step: pl.Step) -> float:
        return 0.5


@pytest.fixture
def box(
    tmp_path: Path, fx_root: Path, fx_topo, monkeypatch: pytest.MonkeyPatch
) -> FakeBox:
    monkeypatch.setenv("FAKE_PROBE_SPEC", str(tmp_path / "spec.json"))
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")  # must not reach the probe
    return make_box(tmp_path, fx_root, fx_topo)


class Capture:
    """Collects what a run reports through ``progress`` and ``sink``."""

    def __init__(self) -> None:
        self.ui: list[Event] = []
        self.sink: list[Record] = []
        self.on_ui: list = []

    def progress(self, event: Event) -> None:
        self.ui.append(event)
        for fn in self.on_ui:
            fn(event)

    def record(self, rec: Record) -> None:
        self.sink.append(rec)

    @property
    def ui_names(self) -> list[str]:
        return [e.name for e in self.ui]

    @property
    def sink_events(self) -> list[Event]:
        return [r for r in self.sink if isinstance(r, Event)]

    @property
    def sink_measurements(self) -> list[Measurement]:
        return [r for r in self.sink if isinstance(r, Measurement)]


def run(
    box: FakeBox,
    suite: str = "quick",
    buses: list[str] | None = None,
    *,
    cap: Capture | None = None,
    owner: str = "PcieProbe",
    limits: rn.RunLimits | None = None,
    force_noisy: bool = False,
    cancel: threading.Event | None = None,
    binary: Path | None = None,
) -> tuple[rn.RunOutcome, Capture, pl.Plan]:
    plan = pl.make_plan(suite, buses or SLOT_B, box.topo)
    cap = cap or Capture()
    outcome = rn.Runner(
        plan,
        box.topo,
        binary=binary or box.binary,
        owner=owner,
        run_id="r-test",
        env=box.env,
        limits=limits or box.limits(),
        force_noisy=force_noisy,
        progress=cap.progress,
        sink=cap.record,
        cancel=cancel,
    ).run()
    return outcome, cap, plan


# -- a clean run ---------------------------------------------------------------


def test_a_clean_run_executes_every_step_in_order(box: FakeBox) -> None:
    out, cap, plan = run(box)
    assert out.status == "ok" and out.valid and not out.error
    assert out.steps_done == out.steps_total == len(plan.steps)
    assert box.commands() == [s.cmd for s in plan.steps]
    assert out.measurements, "the double answers every command"
    assert out.probe_version == "fake/1"
    assert out.tick_mhz == [100.0] * 4
    assert out.windows == 1
    assert set(out.max_temp_c) == {"gpu:0a", "gpu:0f", "gpu:14", "gpu:17"}
    assert all(30 < t < 80 for t in out.max_temp_c.values())  # captured idle values
    assert out.degraded_links == {}


def test_the_probe_sees_exactly_the_planned_gpus(box: FakeBox) -> None:
    run(box)
    lines = box.log_lines()
    assert "argv batch" in lines  # peer plan: no --no-peer
    assert "HIP_VISIBLE_DEVICES=6,7,8,9" in lines
    assert "ROCR_VISIBLE_DEVICES=<unset>" in lines


def test_a_host_only_plan_runs_without_peer_access(box: FakeBox) -> None:
    out, _, plan = run(box, buses=["0a"])
    assert not plan.needs_peer
    assert out.status == "ok"
    lines = box.log_lines()
    assert "argv --no-peer batch" in lines
    assert "HIP_VISIBLE_DEVICES=6" in lines


def test_measurements_carry_run_tags_and_reach_the_sink(box: FakeBox) -> None:
    out, cap, plan = run(box)
    assert cap.sink_measurements == out.measurements
    for m in out.measurements:
        assert m.tags["run"] == "r-test"
        assert m.tags["suite"] == "quick" and m.tags["owner"] == "PcieProbe"
        assert m.tags["step"] in {s.id for s in plan.steps}
        assert m.ok and m.started_ns < m.ended_ns


def test_event_order_for_the_trace_and_the_tui(box: FakeBox) -> None:
    out, cap, plan = run(box)
    assert [e.name for e in cap.sink_events] == ["measure_start", "measure_end"]
    assert cap.sink[0].name == "measure_start" and cap.sink[-1].name == "measure_end"
    end = cap.sink_events[-1]
    assert end.data["status"] == "ok" and end.data["valid"] is True
    assert end.data["results"] == len(out.measurements)
    names = cap.ui_names
    assert names[0] == "plan" and names[-1] == "done"
    assert names.count("step_start") == names.count("step_done") == len(plan.steps)
    assert names.count("measurement") == len(out.measurements)
    first = next(e for e in cap.ui if e.name == "step_start")
    assert first.data["i"] == 1 and first.data["n"] == len(plan.steps)
    assert first.data["step_id"] == plan.steps[0].id and "temp_c" in first.data
    done = cap.ui[-1]
    assert done.data["status"] == "ok" and done.data["valid"] is True


def test_the_quiet_file_exists_during_the_run_and_is_removed_after(
    box: FakeBox,
) -> None:
    cap = Capture()
    seen: list[bool] = []
    cap.on_ui.append(
        lambda e: seen.append((box.quiet_dir / "PcieProbe").is_file())
        if e.name == "step_start"
        else None
    )
    run(box, cap=cap)
    assert seen and all(seen)
    assert not (box.quiet_dir / "PcieProbe").exists()


def test_stderr_of_the_probe_is_kept(box: FakeBox) -> None:
    box.set_spec(stderr="p2p_probe: synthetic diagnostic")
    out, _, _ = run(box)
    assert "p2p_probe: synthetic diagnostic" in out.stderr_tail


def test_a_verify_failure_marks_the_measurement_not_the_run(box: FakeBox) -> None:
    box.set_spec(verify_fail=True)
    out, _, _ = run(box)
    bad = [m for m in out.measurements if not m.ok]
    assert bad and all(m.error == "verify FAIL" for m in bad)
    assert out.valid and out.status == "ok"


def test_a_failed_command_makes_the_run_partial(box: FakeBox) -> None:
    box.add_event(at=2, do="err", text="timeout push 0:1 size=1048576", rc=3)
    out, cap, plan = run(box)
    assert out.status == "partial" and out.valid
    assert out.steps_done == len(plan.steps)
    assert [f["step"] for f in out.steps_failed] == [plan.steps[1].id]
    assert out.steps_failed[0]["error"].startswith("timeout push")
    done = [e for e in cap.ui if e.name == "step_done" and not e.data["ok"]]
    assert [e.data["step_id"] for e in done] == [plan.steps[1].id]
    assert not [m for m in out.measurements if m.tags["step"] == plan.steps[1].id]


# -- failures of the probe process ---------------------------------------------


def test_a_missing_binary_is_a_failed_run(box: FakeBox, tmp_path: Path) -> None:
    out, cap, _ = run(box, binary=tmp_path / "absent")
    assert out.status == "failed" and "cannot start" in (out.error or "")
    assert cap.sink_events[-1].name == "measure_end"
    assert cap.sink_events[-1].data["status"] == "failed"


def test_a_probe_that_never_says_ready_fails_after_the_start_limit(
    box: FakeBox,
) -> None:
    box.set_spec(no_ready=True)
    started = time.monotonic()
    out, _, _ = run(box, limits=box.limits(start_s=0.5))
    assert time.monotonic() - started < 8
    assert out.status == "failed" and "READY" in (out.error or "")
    assert out.steps_done == 0


def test_a_probe_that_dies_mid_run_fails_the_run_and_keeps_earlier_results(
    box: FakeBox,
) -> None:
    box.add_event(at=3, do="exit", rc=7)
    out, _, plan = run(box)
    assert out.status == "failed"
    assert "rc=7" in (out.error or "") and plan.steps[2].id in (out.error or "")
    assert out.steps_done == 2 and out.measurements


def test_a_step_over_its_time_limit_kills_the_probe(box: FakeBox) -> None:
    box.add_event(at=2, do="sleep", s=30)
    started = time.monotonic()
    out, _, _ = run(
        box, limits=QuickStepLimits(poll_s=0.05, temp_period_s=0.05, kill_grace_s=1.0)
    )
    assert time.monotonic() - started < 10
    assert out.status == "failed" and "timeout" in (out.error or "")
    assert out.steps_done == 1


def test_the_run_time_limit_stops_between_steps(box: FakeBox) -> None:
    box.add_event(at=1, do="sleep", s=0.6)
    out, _, _ = run(box, limits=box.limits(max_seconds=0.3))
    assert out.status == "aborted" and "run time limit" in out.abort_reason
    assert out.steps_done == 1


# -- the junction temperature guard --------------------------------------------


def test_a_hot_sample_aborts_the_run_and_marks_it_invalid(box: FakeBox) -> None:
    box.add_event(at=2, do="write", path=str(box.temp_path("0f")), text="97000\n")
    box.add_event(at=2, do="sleep", s=30)
    started = time.monotonic()
    out, cap, plan = run(box)
    assert time.monotonic() - started < 10, "the probe was killed, not waited for"
    assert out.status == "aborted"
    assert "junction temperature guard" in out.abort_reason
    assert "gpu:0f" in out.abort_reason and "97" in out.abort_reason
    assert not out.valid
    assert "run aborted by the junction temperature guard" in out.invalid_reasons
    assert out.max_temp_c["gpu:0f"] >= 97.0
    assert out.steps_done == 1 and out.measurements  # the first step is kept
    thermal = [e for e in cap.sink_events if e.name == "thermal"]
    assert len(thermal) == 1 and thermal[0].severity == "error"
    assert thermal[0].data["max_temp_c"]["gpu:0f"] >= 97.0
    assert "abort" in cap.ui_names and cap.ui[-1].name == "done"
    assert cap.ui[-1].data["status"] == "aborted"


def test_a_hot_five_sample_mean_aborts_the_run(box: FakeBox) -> None:
    box.add_event(at=2, do="write", path=str(box.temp_path("14")), text="91000\n")
    box.add_event(at=2, do="sleep", s=30)
    out, _, _ = run(box)
    assert out.status == "aborted"
    assert "mean" in out.abort_reason and "gpu:14" in out.abort_reason


def test_below_the_limits_the_run_continues(box: FakeBox) -> None:
    box.set_temp("0a", 89.0)  # mean below 90 and no sample at 95
    out, _, _ = run(box)
    assert out.status == "ok" and out.valid
    assert out.max_temp_c["gpu:0a"] == 89.0


def test_an_unreadable_sensor_refuses_to_run_without_the_guard(box: FakeBox) -> None:
    box.temp_path("0f").unlink()
    out, cap, _ = run(box)
    assert out.status == "failed"
    assert "gpu:0f" in (out.error or "") and "guard cannot run" in (out.error or "")
    assert box.commands() == [] and not box.log.exists()  # the probe never started
    assert cap.sink_events[-1].name == "measure_end"


# -- noise rules -----------------------------------------------------------------


def test_another_agents_fresh_quiet_file_aborts_after_the_wait(box: FakeBox) -> None:
    box.quiet("ExpertStats", age_s=10)
    started = time.monotonic()
    out, cap, _ = run(box, limits=box.limits(noisy_wait_s=0.4))
    assert time.monotonic() - started < 8
    assert out.status == "aborted"
    assert "box stayed noisy" in out.abort_reason and "ExpertStats" in out.abort_reason
    assert box.commands() == [], "nothing ran on a noisy box"
    assert any(e.name == "guard" and "quiet" in e.message for e in cap.ui)
    assert not (box.quiet_dir / "PcieProbe").exists()


def test_an_old_quiet_file_does_not_count(box: FakeBox) -> None:
    box.quiet("ExpertStats", age_s=7 * 60)  # older than 6 minutes: stale
    out, _, _ = run(box)
    assert out.status == "ok" and out.valid


def test_the_run_waits_while_the_box_is_noisy_and_goes_on_when_it_quiets(
    box: FakeBox,
) -> None:
    noisy = box.quiet("ExpertStats", age_s=10)
    threading.Timer(0.3, noisy.unlink).start()
    out, cap, plan = run(box, limits=box.limits(noisy_wait_s=10.0))
    assert out.status == "ok" and out.valid
    assert out.steps_done == len(plan.steps)
    assert sum(1 for e in cap.ui if e.name == "guard") == 1  # told once


def test_a_buildx_build_counts_as_noise(box: FakeBox) -> None:
    pid = box.proc_root / "4242"
    pid.mkdir()
    (pid / "cmdline").write_bytes(b"docker\0buildx\0build\0--push\0.\0")
    out, _, _ = run(box, limits=box.limits(noisy_wait_s=0.3))
    assert out.status == "aborted" and "buildx" in out.abort_reason


def test_force_noisy_runs_but_every_result_is_invalid(box: FakeBox) -> None:
    box.quiet("ExpertStats", age_s=10)
    out, cap, plan = run(box, force_noisy=True)
    assert out.status == "ok" and not out.valid
    assert any("ExpertStats" in r for r in out.invalid_reasons)
    assert out.measurements and all(not m.ok for m in out.measurements)
    assert all("ExpertStats" in (m.error or "") for m in out.measurements)
    assert out.steps_done == len(plan.steps)
    assert cap.sink_events[-1].data["valid"] is False


def test_a_boot_lock_held_by_another_process_counts_as_noise(box: FakeBox) -> None:
    import fcntl

    with box.boot_lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        out, _, _ = run(box, limits=box.limits(noisy_wait_s=0.3))
    assert out.status == "aborted" and "boot.lock" in out.abort_reason


def test_the_quiet_window_rotates_after_its_cap(box: FakeBox) -> None:
    box.add_event(at=1, do="sleep", s=0.35)
    out, cap, plan = run(box, limits=box.limits(window_cap_s=0.2, window_gap_s=0.1))
    assert out.status == "ok" and out.steps_done == len(plan.steps)
    assert out.windows >= 2
    assert any(e.name == "guard" and "quiet window ended" in e.message for e in cap.ui)
    assert not (box.quiet_dir / "PcieProbe").exists()


# -- evictions -------------------------------------------------------------------


def a_pid_on(box: FakeBox, bus: str) -> int:
    procs = box.kfd_procs(SLOT_B)
    for pid, per_bus in sorted(procs.items()):
        if bus in per_bus:
            return pid
    raise AssertionError(f"no KFD process on {bus} in the captured box")


def test_an_evicted_ms_jump_invalidates_only_the_step_it_happened_in(
    box: FakeBox,
) -> None:
    pid = a_pid_on(box, "0a")
    base = box.kfd_procs(SLOT_B)[pid]["0a"]
    box.add_event(
        at=2, do="write", path=str(box.evicted_path(pid, "0a")), text=f"{base + 500}\n"
    )
    out, cap, plan = run(box)
    assert out.status == "ok" and not out.valid
    assert any(
        "evicted_ms jump +500 ms" in r and "gpu:0a" in r for r in out.invalid_reasons
    )
    second = plan.steps[1].id
    bad = {m.tags["step"] for m in out.measurements if not m.ok}
    assert bad == {second}
    assert any(m.ok for m in out.measurements), "other steps stay valid"
    (entry,) = out.evicted
    assert entry["pid"] == pid and entry["evicted_ms_delta"] == {"gpu:0a": 500}
    jumps = [e for e in cap.sink_events if e.name == "evicted_jump"]
    assert len(jumps) == 1 and jumps[0].severity == "warn"
    assert jumps[0].data["step_id"] == second


def test_a_new_kfd_process_with_a_queue_on_the_probed_gpus_invalidates_the_step(
    box: FakeBox,
) -> None:
    gpu = box.topo.gpu("14")
    proc = box.root / "sys/class/kfd/kfd/proc/999001"
    (proc / f"stats_{gpu.kfd_gpu_id}").mkdir(parents=True)
    (proc / "queues/0").mkdir(parents=True)
    box.new_kfd_process(999001, "vllm-worker", [])
    box.add_event(
        at=3, do="write", path=str(proc / "queues/0/gpuid"), text=f"{gpu.kfd_gpu_id}\n"
    )
    stats = proc / f"stats_{gpu.kfd_gpu_id}/evicted_ms"
    box.add_event(at=3, do="write", path=str(stats), text="0\n")
    out, _, plan = run(box)
    assert not out.valid
    assert any("new KFD process" in r and "999001" in r for r in out.invalid_reasons)
    bad = {m.tags["step"] for m in out.measurements if not m.ok}
    assert bad == {plan.steps[2].id}


def test_a_new_kfd_process_without_a_queue_is_not_noise(box: FakeBox) -> None:
    # every KFD process has counters for every GPU: one that holds no queue on
    # the probed GPUs (a python that only initialised HIP) must not count
    gpu = box.topo.gpu("14")
    stats = box.root / "sys/class/kfd/kfd/proc/999002" / f"stats_{gpu.kfd_gpu_id}"
    stats.mkdir(parents=True)
    box.new_kfd_process(999002, "python", [])
    box.add_event(at=3, do="write", path=str(stats / "evicted_ms"), text="0\n")
    out, _, _ = run(box)
    assert out.valid and not out.invalid_reasons


# -- cancel ----------------------------------------------------------------------


def test_cancel_stops_the_run_and_kills_the_probe(box: FakeBox) -> None:
    cancel = threading.Event()
    cap = Capture()

    def on_ui(event: Event) -> None:
        if event.name == "step_start" and event.data["i"] == 2:
            cancel.set()

    cap.on_ui.append(on_ui)
    box.add_event(at=2, do="sleep", s=30)
    started = time.monotonic()
    out, _, _ = run(box, cap=cap, cancel=cancel)
    assert time.monotonic() - started < 10
    assert out.status == "aborted" and out.abort_reason == "cancelled"
    assert out.steps_done == 1
    assert cap.ui[-1].name == "done" and cap.ui[-1].data["status"] == "aborted"
    assert wait_for(lambda: not (box.quiet_dir / "PcieProbe").exists())


def test_cancel_before_the_start_runs_nothing(box: FakeBox) -> None:
    cancel = threading.Event()
    cancel.set()
    out, _, _ = run(box, cancel=cancel)
    assert out.status == "aborted" and out.abort_reason == "cancelled"
    assert out.steps_done == 0 and box.commands() == []


# -- helpers around the binary ---------------------------------------------------


def test_probe_version_reads_the_version_line(box: FakeBox) -> None:
    v = rn.probe_version(box.binary)
    assert v is not None and v.version == "fake/1"
    assert "batch" in v.modes
    assert rn.probe_version(box.binary.parent / "absent") is None


def test_probe_buses_runs_cpu_visible_devices_and_orders_the_plan(box: FakeBox) -> None:
    box.set_spec(buses=SLOT_B)
    devices, err = rn.probe_buses(box.binary, "6,7,8,9")
    assert err == "" and [d.bus for d in devices] == SLOT_B
    assert "HIP_VISIBLE_DEVICES=6,7,8,9" in box.log_lines()
    plan = pl.make_plan("quick", SLOT_B, box.topo)
    assert rn.check_bus_order(devices, plan) == ""
    swapped = [devices[1], devices[0], *devices[2:]]
    assert "logical device 0 is bus 0f" in rn.check_bus_order(swapped, plan)
    assert "sees 2 devices" in rn.check_bus_order(devices[:2], plan)


def test_environments() -> None:
    base = {"ROCR_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "2", "KEEP": "x"}
    cpu = rn.cpu_only_env(base)
    assert cpu == {"HIP_VISIBLE_DEVICES": "", "KEEP": "x"}
    gpu = rn.gpu_env("6,7", base)
    assert gpu == {"HIP_VISIBLE_DEVICES": "6,7", "KEEP": "x"}
    assert base["ROCR_VISIBLE_DEVICES"] == "1"  # inputs are not modified


def test_default_binary_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(rn.ENV_BINARY, raising=False)
    assert rn.default_binary().parts[-3:] == ("pciescope", "bin", "p2p_probe")
    monkeypatch.setenv(rn.ENV_BINARY, "/opt/x/probe")
    assert rn.default_binary() == Path("/opt/x/probe")


def test_run_limits_scale_with_the_plan(box: FakeBox) -> None:
    quick = pl.make_plan("quick", SLOT_B, box.topo)
    full = pl.make_plan("full", SLOT_B + ["ce", "d3", "d8", "db"], box.topo)
    limits = rn.RunLimits()
    assert limits.run_limit(quick) == 120.0  # floor
    assert limits.run_limit(full) <= 3600.0
    assert limits.run_limit(full) > limits.run_limit(quick)
    assert rn.RunLimits(max_seconds=5.0).run_limit(full) == 5.0
    step = quick.steps[0]
    assert limits.step_limit(step) >= 20.0


def test_describe_is_one_line_per_measurement(box: FakeBox) -> None:
    out, _, _ = run(box)
    for m in out.measurements:
        text = rn.describe(m)
        assert "\n" not in text and text
    bw = next(m for m in out.measurements if m.kind == "bw")
    assert "GB/s" in rn.describe(bw)
