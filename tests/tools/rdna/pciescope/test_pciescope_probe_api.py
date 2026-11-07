# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``run_suite`` and ``cli_run``: refusals, dry run, end to end with a trace.

The probe is the SYNTHETIC double of ``fixtures/probe/fake_p2p_probe.py`` and the
box a writable copy of the captured par1-cs25 sysfs (see
``pciescope_probe_harness``). The lease directory, quiet directory and boot lock
are temporary directories, so no check reads the real box.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pciescope import report as core_report  # noqa: E402
from pciescope.model import Event, Measurement, Record  # noqa: E402
from pciescope.probe import api  # noqa: E402
from pciescope.probe.render import render_text  # noqa: E402
from pciescope.store import Session  # noqa: E402
from pciescope_probe_harness import FakeBox, make_box  # noqa: E402

OWNER = "PcieProbe"


@pytest.fixture
def box(
    tmp_path: Path, fx_root: Path, fx_topo, monkeypatch: pytest.MonkeyPatch
) -> FakeBox:
    monkeypatch.setenv("FAKE_PROBE_SPEC", str(tmp_path / "spec.json"))
    monkeypatch.delenv("PCIESCOPE_OWNER", raising=False)
    monkeypatch.delenv("PCIESCOPE_LEASE_DIR", raising=False)
    monkeypatch.delenv("PCIESCOPE_QUIET_DIR", raising=False)
    return make_box(tmp_path, fx_root, fx_topo)


def call(
    box: FakeBox,
    suite: str = "quick",
    gpus: Any = "B",
    *,
    sink: Callable[[Record], None] | None = None,
    **kw: Any,
) -> api.MeasureResult:
    kw.setdefault("owner", OWNER)
    kw.setdefault("env", box.env)
    kw.setdefault("binary", box.binary)
    kw.setdefault("limits", box.limits())
    return api.run_suite(
        suite,
        gpus,
        topo=box.topo,
        sink=sink,
        **kw,
    )


def never_started(box: FakeBox) -> bool:
    """True when no ``batch`` process ran (``version`` is a CPU-only query)."""
    argv = [ln for ln in box.log_lines() if ln.startswith("argv ")]
    return not any(ln.endswith(" batch") for ln in argv)


# -- refusals ----------------------------------------------------------------


def test_refused_without_a_lease_names_the_file_to_hold(box: FakeBox) -> None:
    records: list[Record] = []
    res = call(box, sink=records.append)
    assert res.status == "refused" and res.exit_code == api.EXIT_REFUSED == 2
    assert res.needed == ["PcieProbe.slotB"]
    assert "no holder file" in res.refusal and "PcieProbe.slotB" in res.refusal
    assert res.lease is not None and not res.lease.ok
    assert not res.measurements and res.steps_done == 0
    assert not box.log.exists(), "a lease refusal comes before any binary runs"
    (event,) = [r for r in records if isinstance(r, Event)]
    assert event.name == "measure_refused" and event.severity == "warn"
    assert event.data["needed"] == ["PcieProbe.slotB"]
    assert event.data["gpus"] == ["gpu:0a", "gpu:0f", "gpu:14", "gpu:17"]


def test_refused_when_another_agent_holds_the_slot(box: FakeBox) -> None:
    box.lease(
        "ExpertStats", "slotB", "ExpertStats 2026-11-07T13:42:49+01:00 pid 626742"
    )
    res = call(box)
    assert res.status == "refused"
    assert "held by ExpertStats" in res.refusal
    assert res.needed == ["PcieProbe.slotB"]
    assert never_started(box)


def test_my_own_lease_is_not_enough_when_someone_else_holds_it_too(
    box: FakeBox,
) -> None:
    box.lease(OWNER, "slotB")
    box.lease("ExpertStats", "slotB")
    res = call(box)
    assert res.status == "refused" and "also held by ExpertStats" in res.refusal


def test_the_w7800_bus_07_is_never_runnable_with_a_slot_lease(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    for gpus in ("07", "hip:5", ["0a", "07"]):
        res = call(box, gpus=gpus)
        assert res.status == "refused", gpus
        assert "NEVER without the lead" in res.refusal
        assert res.needed == ["PcieProbe.w7800-07"]
    assert never_started(box)


def test_a_gpu_the_box_does_not_have_is_refused(box: FakeBox) -> None:
    res = call(box, gpus="zz")
    assert res.status == "refused" and "zz" in res.refusal
    res = call(box, gpus="hip:42")
    assert res.status == "refused" and "42" in res.refusal
    assert never_started(box)


def test_no_owner_is_refused(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    res = call(box, owner=None)
    assert res.status == "refused" and "no owner" in res.refusal


def test_the_owner_may_come_from_the_environment(
    box: FakeBox, monkeypatch: pytest.MonkeyPatch
) -> None:
    box.lease(OWNER, "slotB")
    monkeypatch.setenv("PCIESCOPE_OWNER", OWNER)
    res = call(box, owner=None, dry_run=True)
    assert res.owner == OWNER and res.blockers == []


def test_a_noisy_box_is_refused_and_force_noisy_does_not_override_the_lease(
    box: FakeBox,
) -> None:
    box.lease(OWNER, "slotB")
    box.quiet("ExpertStats", age_s=10)
    res = call(box)
    assert res.status == "refused" and "not quiet" in res.refusal
    assert any("ExpertStats" in n for n in res.noise)
    assert never_started(box)
    other = call(box, gpus="ce", force_noisy=True)  # slot A: no lease
    assert other.status == "refused" and "slotA" in other.refusal


def test_too_little_free_vram_is_refused_even_with_force_noisy(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    used = box.root / "sys/bus/pci/devices/0000:0f:00.0/mem_info_vram_used"
    used.write_text("31000000000\n", encoding="utf-8")  # ~1.2 GiB free of 30 GiB
    res = call(box, force_noisy=True)
    assert res.status == "refused"
    assert "gpu:0f" in res.refusal and "VRAM" in res.refusal
    assert never_started(box)


def test_unreadable_vram_is_refused(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    (box.root / "sys/bus/pci/devices/0000:14:00.0/mem_info_vram_total").unlink()
    res = call(box)
    assert res.status == "refused" and "gpu:14: cannot read its VRAM" in res.refusal


# -- environment problems are failures, not refusals ---------------------------------


def test_a_missing_binary_is_a_failed_run_with_the_build_hint(
    box: FakeBox, tmp_path: Path
) -> None:
    box.lease(OWNER, "slotB")
    res = call(box, binary=tmp_path / "absent")
    assert res.status == "failed" and res.exit_code == api.EXIT_FAILED
    assert "absent" in res.error and "build" in res.error
    assert res.binary is not None and not res.binary.exists


def test_a_binary_without_the_modes_of_the_plan_is_a_failed_run(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    box.set_spec(modes="version,buses,batch,bw,lat")
    res = call(box, suite="standard")
    assert res.status == "failed"
    assert "lacks mode(s)" in res.error and "curve" in res.error
    assert box.commands() == []


def test_a_probe_that_orders_devices_differently_is_a_failed_run(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    box.set_spec(buses=["0f", "0a", "14", "17"])
    res = call(box)
    assert res.status == "failed"
    assert "bus order check failed" in res.error and "logical device 0" in res.error
    assert box.commands() == []


# -- dry run -------------------------------------------------------------------


def test_dry_run_without_a_lease_plans_and_lists_the_blockers(box: FakeBox) -> None:
    res = call(box, dry_run=True)
    assert res.status == "dry-run" and res.exit_code == 0
    assert any("lease check REFUSED" in b for b in res.blockers)
    assert res.needed == ["PcieProbe.slotB"]  # the holder file a real run needs
    assert res.to_dict()["needed"] == ["PcieProbe.slotB"]
    assert res.plan is not None and res.steps_total == len(res.plan.steps)
    assert box.commands() == []  # no GPU work: the batch process never started
    doc = res.to_dict()
    assert doc["status"] == "dry-run" and doc["plan"]["steps"], (
        "dry runs list the steps"
    )
    assert doc["plan"]["hip_visible_devices"] == "6,7,8,9"
    json.dumps(doc, allow_nan=False)


def test_dry_run_with_a_lease_has_no_blocker(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    events: list[Event] = []
    res = call(box, dry_run=True, progress=events.append)
    assert res.status == "dry-run" and res.blockers == []
    assert res.lease is not None and res.lease.ok
    assert res.binary is not None and res.binary.exists and res.binary.problem == ""
    assert [e.name for e in events] == ["plan", "done"]
    assert "ready" in events[-1].message


def test_dry_run_reports_what_would_stop_a_real_run(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    box.quiet("ExpertStats", age_s=10)
    res = call(box, dry_run=True)
    assert res.status == "dry-run"
    assert any("not quiet" in b for b in res.blockers)


def test_dry_run_of_a_bad_suite_is_a_blocker_not_an_exception(box: FakeBox) -> None:
    res = call(box, suite="nightly", dry_run=True)
    assert res.status == "dry-run" and res.plan is None
    assert any("unknown suite" in b for b in res.blockers)
    real = call(box, suite="nightly")
    assert real.status == "refused" and "unknown suite" in real.refusal


def test_dry_run_leaves_a_runtime_suspended_gpu_alone(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    gpu = box.root / "sys/bus/pci/devices/0000:0f:00.0"
    (gpu / "power").mkdir(exist_ok=True)
    (gpu / "power/runtime_status").write_text("suspended\n", encoding="utf-8")
    # reading the VRAM counters would resume the GPU; here it would also be a blocker
    (gpu / "mem_info_vram_total").unlink()
    res = call(box, dry_run=True)
    assert res.asleep == ["gpu:0f"] and res.blockers == []
    assert res.to_dict()["guard"]["asleep"] == ["gpu:0f"]
    assert "asleep" in render_text(res) and "gpu:0f" in render_text(res)
    # a real run uses the GPU anyway: it is read, and the missing counter refuses it
    real = call(box)
    assert real.status == "refused" and "gpu:0f: cannot read its VRAM" in real.refusal
    assert real.asleep == []


# -- real runs against the double --------------------------------------------------


def test_end_to_end_run_lands_in_the_trace_and_the_report(
    box: FakeBox, tmp_path: Path, fx_topo
) -> None:
    box.lease(OWNER, "slotB")
    with Session.create(tmp_path / "sessions", "measure", topo=fx_topo) as live:
        res = call(box, sink=live.append, session_path=str(live.path))
    assert res.status == "ok" and res.valid and res.exit_code == 0
    assert res.session == str(live.path)
    reopened = Session.open(live.path)
    got = reopened.measurements()
    assert len(got) == len(res.measurements) > 10
    assert [m.kind for m in got] == [m.kind for m in res.measurements]
    names = [e.name for e in reopened.events()]
    assert names[0] == "measure_start" and names[-1] == "measure_end"
    rep = core_report.build_report(reopened, fx_topo)
    assert rep["probes"]["count"] == len(got)
    kinds = {
        (s["kind"], s["direction"], s["path_class"]) for s in rep["probes"]["summary"]
    }
    assert ("bw", "d2d-push", "intra-switch") in kinds
    assert ("bw", "h2d", "host") in kinds
    assert ("lat", "d2d-push", "intra-switch") in kinds
    assert rep["events"]["by_name"]["measure_end"] == 1
    text = core_report.render_text(rep)
    assert "d2d-push" in text


def test_to_dict_of_a_run_is_strict_json_with_summary_and_effectiveness(
    box: FakeBox,
) -> None:
    box.lease(OWNER, "slotB")
    res = call(box)
    doc = res.to_dict()
    text = json.dumps(doc, allow_nan=False)
    assert json.loads(text)["status"] == "ok"
    assert doc["exit_code"] == 0 and doc["owner"] == OWNER and doc["valid"] is True
    assert doc["steps"]["total"] == doc["steps"]["done"] == len(res.plan.steps)  # type: ignore[union-attr]
    assert (
        "steps" not in doc["plan"]
        and doc["plan"]["steps_total"] == doc["steps"]["total"]
    )
    assert len(doc["measurements"]) == len(res.measurements)
    assert len(doc["summary"]) == len(res.summary())
    eff = doc["effectiveness"]
    assert set(eff) == {"rows", "by_class", "findings"}
    assert eff["rows"] and "intra-switch" in eff["by_class"]
    assert set(doc["guard"]) == {
        "max_temp_c",
        "evicted",
        "degraded_links",
        "noise_at_start",
        "asleep",
        "windows",
    }
    assert doc["probe"]["tick_mhz"] == [100.0] * 4
    assert doc["probe"]["binary"]["exists"] is True
    assert doc["lease"] == {
        "ok": True,
        "owner": OWNER,
        "needed": [],
        "reason": "",
    }
    steps_doc = res.to_dict(steps=True)
    assert len(steps_doc["plan"]["steps"]) == len(res.plan.steps)  # type: ignore[union-attr]


def test_force_noisy_run_is_complete_but_invalid_and_exits_4(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    box.quiet("ExpertStats", age_s=10)
    res = call(box, force_noisy=True)
    assert res.status == "ok" and not res.valid
    assert res.exit_code == api.EXIT_INVALID == 4
    assert res.measurements and all(not m.ok for m in res.measurements)
    assert all(m.tags.get("force_noisy") == "1" for m in res.measurements)
    assert any("ExpertStats" in r for r in res.invalid_reasons)
    assert res.to_dict()["valid"] is False


def test_wait_quiet_waits_for_the_box_instead_of_refusing(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    noisy = box.quiet("ExpertStats", age_s=10)
    threading.Timer(0.3, noisy.unlink).start()
    res = call(box, wait_quiet_s=10.0)
    assert res.status == "ok" and res.valid
    assert res.noise == []


def test_an_aborted_run_has_exit_code_3(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    box.add_event(at=2, do="write", path=str(box.temp_path("0a")), text="96000\n")
    box.add_event(at=2, do="sleep", s=30)
    res = call(box)
    assert res.status == "aborted" and res.exit_code == api.EXIT_ABORTED == 3
    assert "junction temperature" in res.abort_reason
    assert res.steps_done == 1 and not res.valid
    assert res.max_temp_c["gpu:0a"] >= 96.0


def test_a_failed_step_gives_a_partial_run_with_exit_code_1(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    box.add_event(at=1, do="err", text="timeout hbw ready size=67108864 xfers=1", rc=3)
    res = call(box)
    assert res.status == "partial" and res.exit_code == api.EXIT_FAILED == 1
    assert res.steps_failed and res.measurements


def test_cancel_through_the_api(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    cancel = threading.Event()
    cancel.set()
    res = call(box, cancel=cancel)
    assert res.status == "aborted" and res.abort_reason == "cancelled"


def test_progress_events_reach_the_caller_in_ui_vocabulary(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    seen: list[Event] = []
    res = call(box, progress=seen.append)
    names = {e.name for e in seen}
    assert names == {"plan", "step_start", "step_done", "measurement", "done"}
    assert seen[0].name == "plan" and seen[-1].name == "done"
    assert seen[0].data["n"] == len(res.plan.steps)  # type: ignore[union-attr]


def test_repeats_and_suite_flow_through_to_the_plan(box: FakeBox) -> None:
    box.lease(OWNER, "slotB")
    res = call(box, suite="standard", repeats=2, dry_run=True)
    assert res.plan is not None and res.plan.repeats == 2
    assert res.plan.suite == "standard"


def test_repeated_cells_fold_into_the_summary_and_keep_every_raw_record(
    box: FakeBox,
) -> None:
    box.lease(OWNER, "slotB")
    res = call(box, repeats=3)
    assert res.status == "ok"
    raw = [m for m in res.measurements if m.tags.get("reps") == "3"]
    assert raw, "key cells run three times"
    folded = [m for m in res.summary() if m.tags.get("reps") == "3"]
    assert 0 < len(folded) < len(raw)
    assert all(m.stats["n"] == 3 for m in folded)


# -- command line ----------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pciescope measure")
    api.add_cli_arguments(p)
    p.add_argument("--json", action="store_true")
    return p


@pytest.fixture
def cli(box: FakeBox, monkeypatch: pytest.MonkeyPatch):
    """``cli_run`` with the box guards injected (the CLI has no such option)."""
    real = api.run_suite

    def patched(*a: Any, **kw: Any) -> api.MeasureResult:
        kw.setdefault("env", box.env)
        kw.setdefault("limits", box.limits())
        return real(*a, **kw)

    monkeypatch.setattr(api, "run_suite", patched)

    def run(argv: list[str], **kw: Any) -> tuple[int, list[str]]:
        out: list[str] = []
        args = parser().parse_args([*argv, "--quiet", "--binary", str(box.binary)])
        rc = api.cli_run(args, topo=box.topo, emit=out.append, **kw)
        return rc, out

    return run


def test_cli_lists_suites(cli) -> None:
    rc, out = cli(["--list-suites", "--json"])
    assert rc == 0
    doc = json.loads(out[0])
    assert [s["name"] for s in doc["suites"]] == ["quick", "standard", "full"]
    rc, out = cli(["--list-suites"])
    assert rc == 0 and "quick" in out[0] and "~" in out[0]


def test_cli_needs_gpus(cli) -> None:
    rc, out = cli(["--suite", "quick", "--json"])
    assert rc == 2 and json.loads(out[0])["status"] == "refused"
    rc, out = cli(["--suite", "quick"])
    assert rc == 2 and "--gpus" in out[0]


def test_cli_refuses_without_a_lease_and_says_which(box: FakeBox, cli) -> None:
    rc, out = cli(["--suite", "quick", "--gpus", "B", "--owner", OWNER, "--json"])
    assert rc == 2
    doc = json.loads(out[0])
    assert doc["status"] == "refused" and doc["needed"] == ["PcieProbe.slotB"]
    rc, out = cli(["--suite", "quick", "--gpus", "B", "--owner", OWNER])
    assert rc == 2 and "PcieProbe.slotB" in out[0]


def test_cli_json_run_end_to_end(box: FakeBox, cli, tmp_path: Path, fx_topo) -> None:
    box.lease(OWNER, "slotB")
    with Session.create(tmp_path / "s", "measure", topo=fx_topo) as session:
        rc, out = cli(
            [
                "--suite",
                "quick",
                "--gpus",
                "0a",
                "0f",
                "14",
                "17",
                "--owner",
                OWNER,
                "--json",
            ],
            session=session,
        )
    assert rc == 0 and len(out) == 1
    doc = json.loads(out[0])
    assert doc["status"] == "ok" and doc["gpus"] == [
        "gpu:0a",
        "gpu:0f",
        "gpu:14",
        "gpu:17",
    ]
    assert doc["session"] == str(session.path)
    assert len(Session.open(session.path).measurements()) == len(doc["measurements"])


def test_cli_text_run_prints_tables(box: FakeBox, cli) -> None:
    box.lease(OWNER, "slotB")
    rc, out = cli(["--suite", "quick", "--gpus", "B", "--owner", OWNER])
    assert rc == 0
    text = out[0]
    assert "BANDWIDTH" in text and "LATENCY" in text and "GB/s" in text


def test_cli_comma_separated_gpus_and_dry_run(box: FakeBox, cli) -> None:
    rc, out = cli(["--suite", "standard", "--gpus", "0a,0f", "--dry-run", "--json"])
    assert rc == 0
    doc = json.loads(out[0])
    assert doc["status"] == "dry-run" and doc["gpus"] == ["gpu:0a", "gpu:0f"]
    assert doc["blockers"], "no owner and no lease"


def test_cli_exit_code_of_an_invalid_run(box: FakeBox, cli) -> None:
    box.lease(OWNER, "slotB")
    box.quiet("ExpertStats", age_s=10)
    rc, out = cli(
        ["--suite", "quick", "--gpus", "B", "--owner", OWNER, "--force-noisy"]
    )
    assert rc == 4 and "INVALID" in out[0]


def test_cli_in_situ_without_counters_says_so(box: FakeBox, cli) -> None:
    rc, out = cli(["--in-situ", "0.2", "--interval", "0.1", "--json"])
    assert rc == 0
    doc = json.loads(out[0])
    assert doc["mode"] == "in-situ"


def test_measurements_of_the_cli_are_model_measurements(box: FakeBox, cli) -> None:
    box.lease(OWNER, "slotB")
    rc, out = cli(["--suite", "quick", "--gpus", "B", "--owner", OWNER, "--json"])
    assert rc == 0
    for raw in json.loads(out[0])["measurements"]:
        m = Measurement.from_dict(raw)
        assert m.kind in {"bw", "concurrency", "lat", "a2a"}
