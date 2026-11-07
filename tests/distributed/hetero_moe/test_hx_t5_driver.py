# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the pure parts of the T5 driver (tools/rdna/hetero_moe/t5_run.py)
and of the plans ``t5_plan.py`` writes. No GPU, no torch."""

import signal
import subprocess
import sys
import time

import pytest

from tools.rdna.hetero_moe import t5_run as R

# ---- argument parsing -----


def test_parser_subcommands_and_defaults():
    p = R.build_parser()
    a = p.parse_args(["golden", "--plan", "p.json", "--out", "d", "--model", "m"])
    assert a.cmd == "golden" and a.widths == "3,24,48,2048" and not a.parallel
    a = p.parse_args(
        ["run", "--plan", "p.json", "--out", "d", "--synthetic", "--hub-hip", "1"]
    )
    assert a.cmd == "run" and a.hub_hip == 1 and a.synthetic
    assert a.modes == "verify,graph,bench" and a.wait_ms == 2000
    a = p.parse_args(["faults", "--plan", "p.json", "--out", "d"])
    assert a.cases.split(",") == list(R.FAULT_CASES)
    assert a.limit == 5.0 and a.connect_timeout == 20.0 and a.victim == 1
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--out", "d"])  # --plan is required


def test_child_env_prepends_the_repo_and_pins_the_gpu(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    env = R.child_env(HIP_VISIBLE_DEVICES="3")
    assert env["PYTHONPATH"] == f"{R.REPO}:/somewhere"
    assert env["HIP_VISIBLE_DEVICES"] == "3" and env["PYTHONUNBUFFERED"] == "1"


def _unset_target_device(monkeypatch) -> None:
    """VLLM_TARGET_DEVICE absent, and restored to absent even if main() sets it
    (monkeypatch.delenv of an absent variable records nothing to undo)."""
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "x")
    monkeypatch.delenv("VLLM_TARGET_DEVICE")


def test_run_baseline_flag_and_the_synthetic_exclusion(capsys, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    _unset_target_device(monkeypatch)
    p = R.build_parser()
    a = p.parse_args(["run", "--plan", "p.json", "--out", "d"])
    assert a.baseline is None and not a.synthetic
    a = p.parse_args(["run", "--plan", "p.json", "--out", "d", "--baseline", "b.json"])
    assert a.baseline == "b.json" and not a.synthetic
    both = ["run", "--plan", "nope.json", "--out", "d", "--synthetic"]
    assert R.main([*both, "--baseline", "b.json"]) == R.C.EXIT_SETUP
    assert "a synthetic run is the baseline" in capsys.readouterr().err


def test_main_refuses_without_hip_visible_devices_and_runs_the_parent_on_cpu(
    capsys, monkeypatch
):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    _unset_target_device(monkeypatch)
    argv = ["run", "--plan", "nope.json", "--out", "d"]
    assert R.main(argv) == R.C.EXIT_SETUP
    assert "HIP_VISIBLE_DEVICES is not set" in capsys.readouterr().err
    assert "VLLM_TARGET_DEVICE" not in R.os.environ  # nothing ran
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    assert R.main([*argv, "--synthetic", "--baseline", "b"]) == R.C.EXIT_SETUP
    assert R.os.environ["VLLM_TARGET_DEVICE"] == "cpu"


def test_main_refuses_hub_hip_5_before_starting_anything(capsys, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    _unset_target_device(monkeypatch)
    argv = ["run", "--plan", "nope.json", "--out", "d", "--hub-hip", "5"]
    assert R.main(argv) == R.C.EXIT_SETUP
    assert "HIP index 5 refused" in capsys.readouterr().err


def test_spawn_gives_a_gpu_child_its_pin_and_no_cpu_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    seen = {}

    def popen(cmd, **kwargs):
        seen.update(kwargs["env"])
        return "proc"

    monkeypatch.setattr(R.subprocess, "Popen", popen)
    assert R.spawn(["x"], tmp_path / "l.log", 3) == "proc"
    assert seen["HIP_VISIBLE_DEVICES"] == "3"
    assert seen["PYTHONUNBUFFERED"] == "1" and str(R.REPO) in seen["PYTHONPATH"]
    assert "VLLM_TARGET_DEVICE" not in seen and "ROCR_VISIBLE_DEVICES" not in seen
    with pytest.raises(R.C.SetupError):
        R.spawn(["x"], tmp_path / "l.log", 5)
    assert "VLLM_TARGET_DEVICE" not in R.child_env()  # the launcher env too


# ---- timing helpers -----


def _sleeper(seconds: float, rc: int = 0) -> subprocess.Popen:
    code = f"import sys,time; time.sleep({seconds}); sys.exit({rc})"
    return subprocess.Popen([sys.executable, "-c", code])


def test_await_exits_measures_each_process_from_t0():
    quick, slow, never = _sleeper(0.3), _sleeper(1.0), _sleeper(60)
    try:
        t0 = time.monotonic()
        res = R.await_exits({"quick": quick, "slow": slow, "never": never}, t0, 2.5)
        assert 0.25 <= res["quick"] < res["slow"] < 2.0
        assert res["slow"] >= 0.9
        assert res["never"] is None  # still running at the deadline
        assert quick.returncode == 0
    finally:
        never.kill()
        never.wait()


def test_await_exits_sees_a_signal_death_quickly():
    proc = _sleeper(60)
    time.sleep(0.2)
    t0 = time.monotonic()
    proc.send_signal(signal.SIGKILL)
    res = R.await_exits({"p": proc}, t0, 5.0)
    assert res["p"] is not None and res["p"] < 1.0
    assert proc.returncode == R.SIGKILL_RC


def test_wait_for_text_finds_late_text_and_stops_when_the_process_exits(tmp_path):
    log = tmp_path / "hub.log"
    assert not R.wait_for_text(log, "STEP", 0.2)  # no file yet
    log.write_text("boot\n")
    code = f"import time; time.sleep(0.3); open({str(log)!r}, 'a').write('STEP 100\\n')"
    proc = subprocess.Popen([sys.executable, "-c", code])
    assert R.wait_for_text(log, "STEP 100", 10.0, proc)
    proc.wait()
    t0 = time.monotonic()
    assert not R.wait_for_text(log, "nope", 30.0, proc)  # proc is gone: no 30 s wait
    assert time.monotonic() - t0 < 2.0


def test_parse_abort_and_parse_waited():
    text = (
        "STEP 100\n"
        "HX_ABORT hub aborted in combine wait for server (phase 2): peer server 1, "
        "waited 2003 ms. phase=2 peer=1\n"
    )
    assert R.parse_abort(text) == (2, 1)
    assert R.parse_abort("HX_ABORT hx server exp1 (pid 5) died phase=0 peer=-1") == (
        0,
        -1,
    )
    assert R.parse_abort("no abort here") is None
    assert R.parse_abort("HX_ABORT without fields") is None
    line = "T5 connect FAIL TimeoutError: x (waited 20.1 s)\n"
    assert R.parse_waited(line) == 20.1
    assert R.parse_waited("T5 connect FAIL boom") is None


# ---- leftovers -----


def test_leftover_files_lists_only_the_hx_files(tmp_path):
    assert R.leftover_files("abc", tmp_path) == []  # no directory
    d = tmp_path / "hx_abc"
    d.mkdir()
    assert R.leftover_files("abc", tmp_path) == []
    for name in (
        "srv_exp0.json",
        "ack_exp0_hub0.json",
        "hub_hub0.json",
        "ring_hub0.bin",
    ):
        (d / name).write_text("x")
    (d / "notes.txt").write_text("x")
    (d / ".srv_exp1.json.123.tmp").write_text("x")
    assert R.leftover_files("abc", tmp_path) == [
        "ack_exp0_hub0.json",
        "hub_hub0.json",
        "ring_hub0.bin",
        "srv_exp0.json",
    ]
    assert R.leftover_files("other", tmp_path) == []


# ---- fault verdicts -----


def _obs(case, **kw):
    base = dict(
        case=case,
        victim="exp1",
        victim_index=1,
        server_rc=dict.fromkeys(("exp0", "exp1", "exp2", "exp3"), 3),
        server_exit_s=dict.fromkeys(("exp0", "exp1", "exp2", "exp3"), 1.0),
        server_logs={
            f"exp{i}": "aborted in server request wait (phase 3)" for i in range(4)
        },
    )
    base.update(kw)
    return R.Observation(**base)


ABORT = "HX_ABORT hub aborted ... phase=2 peer=1\n"


def test_judge_kill_hub():
    assert R.judge(_obs("kill-hub")) == []
    slow = _obs("kill-hub")
    slow.server_exit_s["exp2"] = 5.5
    assert "server exp2 exit after 5.5 s" in R.judge(slow)[0]
    wrong = _obs("kill-hub")
    wrong.server_rc["exp0"] = 0
    assert "server exp0 exit code 0" in R.judge(wrong)[0]
    hung = _obs("kill-hub")
    hung.server_exit_s["exp3"] = None
    assert R.judge(hung)


def test_judge_kill_server():
    ok = _obs(
        "kill-server",
        hub_rc=3,
        hub_exit_s=1.2,
        hub_log=ABORT,
        server_rc={"exp0": 0, "exp1": R.SIGKILL_RC, "exp2": 3, "exp3": 0},
    )
    assert R.judge(ok) == []
    ok.hub_log = "nothing"
    assert any("no HX_ABORT" in w for w in R.judge(ok))
    ok.hub_log = ABORT
    ok.hub_exit_s = 6.0
    assert any("hub exit after 6.0 s" in w for w in R.judge(ok))
    ok.hub_exit_s = 1.0
    ok.server_rc["exp1"] = 3  # the victim must have been killed
    assert any("victim exp1" in w for w in R.judge(ok))


def test_judge_freeze_server_needs_the_combine_phase_and_the_right_peer():
    ok = _obs(
        "freeze-server",
        hub_rc=3,
        hub_exit_s=2.4,
        hub_log=ABORT,
        server_rc={"exp0": 0, "exp1": R.SIGKILL_RC, "exp2": 0, "exp3": 3},
    )
    assert R.judge(ok) == []
    ok.hub_log = "HX_ABORT x phase=2 peer=2\n"
    assert any("expected (2, 1)" in w for w in R.judge(ok))
    ok.hub_log = "HX_ABORT x phase=8 peer=1\n"
    assert any("expected (2, 1)" in w for w in R.judge(ok))


def test_judge_freeze_hub_needs_phase_3_in_every_server_log():
    assert R.judge(_obs("freeze-hub", hub_rc=R.SIGKILL_RC)) == []
    bad = _obs("freeze-hub")
    bad.server_logs["exp2"] = "died"
    assert any("server exp2 log has no 'phase 3'" in w for w in R.judge(bad))


def test_judge_wrong_plan_and_leftover_processes():
    ok = _obs(
        "wrong-plan",
        hub_rc=4,
        hub_exit_s=40.0,
        hub_log="T5 connect FAIL TimeoutError: ... (waited 20.0 s)\n",
        servers_alive_after_hub=dict.fromkeys(("exp0", "exp1", "exp2", "exp3"), True),
        server_rc=dict.fromkeys(("exp0", "exp1", "exp2", "exp3"), 0),
    )
    assert R.judge(ok) == []  # start-up time is not part of the connect wait
    ok.hub_rc = 0
    ok.servers_alive_after_hub["exp1"] = False
    ok.server_rc["exp3"] = 3
    ok.leftover_procs = ["hub pid 7"]
    why = "\n".join(R.judge(ok))
    assert "expected a failure" in why and "servers ['exp1'] exited" in why
    assert "server exp3 exit code 3 after SIGTERM" in why and "hub pid 7" in why
    late = _obs("wrong-plan", hub_rc=4, hub_log="T5 connect FAIL x (waited 31.0 s)")
    late.servers_alive_after_hub = dict.fromkeys(late.server_rc, True)
    late.server_rc = dict.fromkeys(late.server_rc, 0)
    assert any("hub waited 31.0 s" in w for w in R.judge(late))
    assert R.judge(_obs("bogus"))[0].startswith("unknown case")


# ---- plans -----


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("regex")
    from tests.quantization.hetero_moe_support import hetero, stripe_doc

    return hetero, stripe_doc


@pytest.mark.parametrize("how", ["hip", "owners"])
def test_wrong_plan_variants_are_valid_plans_with_another_id(env, how):
    hetero, stripe_doc = env
    plan_mod = hetero("plan")
    doc = stripe_doc()
    wrong = R.wrong_plan_doc(doc, plan_mod.compute_plan_id, how)
    assert wrong["plan_id"] != doc["plan_id"]
    assert doc["plan_id"] == plan_mod.compute_plan_id(doc)  # the original is untouched
    plan = plan_mod.plan_from_doc(wrong)  # validates and checks the stored id
    assert plan.plan_id == wrong["plan_id"]
    with pytest.raises(ValueError, match="unknown variant"):
        R.wrong_plan_doc(doc, plan_mod.compute_plan_id, "x")


def test_the_driver_loads_the_plan_module_by_path(env):
    _, stripe_doc = env
    doc = stripe_doc()
    plan = R.plan_module().plan_from_doc(doc)
    assert plan.plan_id == doc["plan_id"]
    hub_dev, hub_id = R.hub_of(plan)
    assert hub_id == "hub0" and plan.device_id(hub_dev) == "hub0"


@pytest.mark.parametrize("kind", ["stripe", "coldonly"])
def test_t5_plans_validate_and_route_everything_to_the_servers(env, kind):
    hetero, _ = env
    from tools.rdna.hetero_moe import t5_common as C
    from tools.rdna.hetero_moe import t5_plan, validate

    doc = t5_plan.build_doc(kind, 0, [1, 2, 3, 4], "c0")
    assert validate.validate_plan(doc) == []
    plan = hetero("plan").plan_from_doc(doc)
    assert plan.target_layers == list(range(48)) and plan.layer_ids[-1] == 48
    assert plan.is_hub_only(48) and plan.layer_kind(48) == "mtp"
    assert plan.num_experts == 512 and plan.top_k == 10 and plan.cold_complete
    assert [d["id"] for d in plan.devices] == ["hub0", "exp0", "exp1", "exp2", "exp3"]
    assert [d["hip"] for d in plan.devices] == [0, 1, 2, 3, 4]
    assert plan.spill_k() == dict.fromkeys(t5_plan.CAPTURE_WIDTHS, 0)
    tables = hetero("tables").build_island_tables(plan, plan.hubs(0)[0])
    assert C.hx_layers(tables) == list(range(48))
    assert tables.layers[48].hub_only and tables.layers[48].n_slots == 512
    hub_experts = tables.layers[0].n_slots
    assert hub_experts == (128 if kind == "stripe" else 0)
    assert tables.spill_disabled is None
    for width in (3, 24, 48):
        assert tables.k_for(width) == 0
    ids = C.make_inputs(1, 24, 0, nlayers=1)
    hub_ids, dest = C.expected_split(tables, 0, ids.ids[0], tables.k_for(24))
    assert (hub_ids == -1).all() and (dest >= 0).all()  # all pairs remote at K = 0


def test_t5_stripe_plan_keeps_hub_pairs_above_the_widest_capture(env):
    hetero, _ = env
    from tools.rdna.hetero_moe import t5_common as C
    from tools.rdna.hetero_moe import t5_plan

    plan = hetero("plan").plan_from_doc(
        t5_plan.build_doc("stripe", 0, [1, 2, 3, 4], "c0")
    )
    tables = hetero("tables").build_island_tables(plan, plan.hubs(0)[0])
    ids = C.make_inputs(1, 2048, 0, nlayers=1)
    hub_ids, dest = C.expected_split(tables, 0, ids.ids[0], tables.k_for(2048))
    assert (hub_ids >= 0).any() and (dest >= 0).any()
    assert ((hub_ids >= 0) | (dest >= 0)).all()
    assert (hub_ids[hub_ids >= 0] % 4 == 0).all()  # the stripe-4 hub set
    custom = t5_plan.build_doc("stripe", 5, [6, 7, 8, 9], "00", {3: 10, 24: 20})
    assert [d["hip"] for d in custom["devices"]] == [5, 6, 7, 8, 9]
    assert custom["runtime"]["spill_k"] == {"3": 10, "24": 20}
    assert custom["plan_id"] != plan.plan_id
