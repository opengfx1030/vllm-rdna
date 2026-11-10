# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/tp8pp2: the hetero slots HA (hub W7800 + slot A, the layout before the
2026-11-08 case move) and HV (the V620 stand-in of the interim layout), CFG=h1 / h1v.

slots.sh is sourced with a stub venv python that prints the "hip:bus:name:GiB"
lines the real resolve_slot would get from torch, so slot_hip / slot_buses /
verify_slot are checked without a GPU. driver.sh is run only as far as its
start-up report (a missing exact-SHA tree stops it right after).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SLOTS = REPO / "tools/rdna/tp8pp2/slots.sh"
DRIVER = REPO / "tools/rdna/tp8pp2/driver.sh"
W7800 = "AMD_Radeon_PRO_W7800"
V620 = "AMD_Radeon_PRO_V620"
# par1-cs25 before the case move: HIP index -> "bus:name:GiB"
TOPOLOGY = {
    0: f"cb:{W7800}:44.9",
    1: f"ce:{V620}:31.9",
    2: f"d3:{V620}:31.9",
    3: f"d8:{V620}:31.9",
    4: f"db:{V620}:31.9",
    5: f"07:{W7800}:44.9",
    6: f"0a:{V620}:31.9",
    7: f"0f:{V620}:31.9",
    8: f"14:{V620}:31.9",
    9: f"17:{V620}:31.9",
}
# since the case move (box rules INTERIM): six V620s, HIP 0-5
INTERIM = {
    0: f"d1:{V620}:31.9",
    1: f"d4:{V620}:31.9",
    2: f"d9:{V620}:31.9",
    3: f"dc:{V620}:31.9",
    4: f"df:{V620}:31.9",
    5: f"e2:{V620}:31.9",
}
STUB = """#!/bin/sh
IFS=,
for h in $HIP_VISIBLE_DEVICES; do
  eval "line=\\$DEV_$h"
  echo "$h:$line"
done
"""


@pytest.fixture()
def home(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(STUB)
    python.chmod(0o755)
    return tmp_path


def base_env(home, topology=None, **extra):
    topology = {**TOPOLOGY, **(topology or {})}
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HUB_", "SLOT_", "STANDIN_", "DEV_", "HIP_", "VLLM_"))
        and k not in ("RUN_DIR", "LANES", "BOOT_LOCK", "VENV", "CFG", "RECIPE", "PLAN")
    }
    env.update(HOME=str(home), VENV=str(home / "venv"))
    env.update({f"DEV_{i}": v for i, v in topology.items()})
    env.update(extra)
    return env


def slots(home, script, topology=None, **extra):
    return subprocess.run(
        ["bash", "-c", f'source "{SLOTS}"; {script}'],
        capture_output=True,
        text=True,
        env=base_env(home, topology, **extra),
        timeout=30,
    )


def test_existing_slots_are_unchanged(home):
    out = slots(
        home,
        'for s in A B AB; do echo "$s|$(slot_hip $s)|$(slot_buses $s)"; done',
    ).stdout.splitlines()
    assert out == [
        "A|1,2,3,4|ce d3 d8 db",
        "B|6,7,8,9|0a 0f 14 17",
        "AB|1,2,3,4,6,7,8,9|ce d3 d8 db 0a 0f 14 17",
    ]
    assert slots(home, "slot_hip XX").returncode == 1
    assert slots(home, "slot_buses XX").returncode == 1


@pytest.mark.parametrize(
    ("slot", "hip", "bus"), [("A", 2, "d3"), ("B", 7, "0f"), ("AB", 8, "14")]
)
def test_existing_slots_verify_and_still_refuse_a_w7800(home, slot, hip, bus):
    result = slots(home, f"verify_slot {slot}")
    assert result.returncode == 0, result.stdout
    assert f"slot {slot} verified:" in result.stdout
    # a W7800 where a V620 is expected is refused with the unchanged message
    bad = slots(home, f"verify_slot {slot}", {hip: f"{bus}:{W7800}:44.9"})
    assert bad.returncode == 1
    assert bad.stdout.strip() == (
        f"REFUSE: HIP {hip} is bus {bus} ({W7800}), expected {bus} V620"
    )


def test_slot_ha_is_the_hub_then_slot_a(home):
    out = slots(home, 'echo "$(slot_hip HA)|$(slot_buses HA)"').stdout.strip()
    assert out == "0,1,2,3,4|cb ce d3 d8 db"
    result = slots(home, "verify_slot HA")
    assert result.returncode == 0, result.stdout
    assert "slot HA verified: 0:cb:" in result.stdout
    assert W7800 in result.stdout.splitlines()[0]


def test_slot_ha_follows_the_hub_overrides(home):
    out = slots(
        home,
        'echo "$(slot_hip HA)|$(slot_buses HA)"',
        HUB_HIP="5",
        HUB_BUSES="07",
        SLOT_A_HIP="6,7",
        SLOT_A_BUSES="0a 0f",
    ).stdout.strip()
    assert out == "5,6,7|07 0a 0f"


def test_slot_ha_refuses_a_v620_in_the_hub_position(home):
    result = slots(home, "verify_slot HA", {0: f"cb:{V620}:31.9"})
    assert result.returncode == 1
    assert result.stdout.strip() == (
        f"REFUSE: HIP 0 is bus cb ({V620}), expected cb W7800"
    )


def test_slot_ha_refuses_a_w7800_in_a_server_position(home):
    result = slots(home, "verify_slot HA", {3: f"d8:{W7800}:44.9"})
    assert result.returncode == 1
    assert result.stdout.strip() == (
        f"REFUSE: HIP 3 is bus d8 ({W7800}), expected d8 V620"
    )


def test_slot_ha_refuses_the_wrong_hub_bus_and_the_other_w7800(home):
    result = slots(home, "verify_slot HA", {0: f"07:{W7800}:44.9"})
    assert result.returncode == 1
    assert "expected cb W7800" in result.stdout
    # HIP 5 (bus 07) is not a hub position of slot HA by default
    result = slots(home, "verify_slot HA", {}, HUB_HIP="5")
    assert result.returncode == 1


def test_slot_ha_refuses_a_missing_device(home):
    result = slots(home, "verify_slot HA", {}, SLOT_A_BUSES="ce d3 d8 db 0a")
    assert result.returncode == 1


# --- slot HV: the V620 stand-in of the interim layout ----------------------------


def test_slot_hv_is_the_hub_then_four_servers(home):
    out = slots(
        home, 'echo "$(slot_hip HV)|$(slot_buses HV)|$STANDIN_LEASES"', INTERIM
    ).stdout.strip()
    assert out == "1,2,3,4,5|d4 d9 dc df e2|a2 b4"
    result = slots(home, "verify_slot HV", INTERIM)
    assert result.returncode == 0, result.stdout
    assert "slot HV verified: 1:d4:" in result.stdout
    assert " 5:e2:" in result.stdout


def test_slot_hv_follows_the_standin_overrides(home):
    out = slots(
        home,
        'echo "$(slot_hip HV)|$(slot_buses HV)|$STANDIN_LEASES"',
        STANDIN_HUB_HIP="0",
        STANDIN_HUB_BUSES="d1",
        STANDIN_SERVER_HIP="2,3",
        STANDIN_SERVER_BUSES="d9 dc",
        STANDIN_LEASES="a2 b2",
    ).stdout.strip()
    assert out == "0,2,3|d1 d9 dc|a2 b2"


def test_slot_hv_refuses_the_old_layout_and_misplaced_gpus(home):
    # before the case move HIP 1 was bus ce, not d4
    result = slots(home, "verify_slot HV")
    assert result.returncode == 1
    assert result.stdout.strip() == (
        f"REFUSE: HIP 1 is bus ce ({V620}), expected d4 V620"
    )
    # a W7800 (or any other GPU) in a server position
    bad = slots(home, "verify_slot HV", {**INTERIM, 4: f"07:{W7800}:44.9"})
    assert bad.returncode == 1
    assert bad.stdout.strip() == (
        f"REFUSE: HIP 4 is bus 07 ({W7800}), expected df V620"
    )
    # the hub is bus d4 (HIP 1), not d1 (HIP 0)
    swapped = slots(home, "verify_slot HV", {**INTERIM, 1: f"d1:{V620}:31.9"})
    assert swapped.returncode == 1
    assert "expected d4 V620" in swapped.stdout


def test_slot_hv_refuses_a_missing_device(home):
    result = slots(
        home, "verify_slot HV", INTERIM, STANDIN_SERVER_BUSES="d9 dc df e2 d1"
    )
    assert result.returncode == 1


# --- driver.sh start-up report -----------------------------------------------


def driver(home, cfg, **extra):
    env = base_env(
        home,
        OWNER="tester",
        SHA="0" * 40,
        CFG=cfg,
        RUN_DIR=str(home / "run"),
        TREE_T=str(home / "no-tree"),
        **extra,
    )
    return subprocess.run(
        ["bash", str(DRIVER)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def start_line(result):
    (line,) = [ln for ln in result.stdout.splitlines() if "driver start" in ln]
    return line


def test_cfg_h1_serves_the_hub_gpu_only(home):
    result = driver(home, "h1", ARM="h1-x")
    line = start_line(result)
    assert "cfg h1 arm h1-x slot HA HIP 0,1,2,3,4 buses cb ce d3 d8 db" in line
    out = result.stdout
    plan = f"{home}/no-tree/bench_results/2026-11-07_expert-stats-t10/H1.W1.plan.json"
    assert "hetero: hub engine sees HIP 0 only" in out
    assert f"plan {plan}" in out
    assert "(refused HIP 5)" in out
    assert f"VLLM_HETERO_MOE_PLAN={plan}" in line
    assert f"HETERO_LOG_DIR={home}/run/arms/h1-x/hetero" in line
    assert f"HETERO_PYTHON={home}/venv/bin/python" in line


def test_cfg_h1v_is_slot_hv_with_the_hub_on_hip_1(home):
    result = driver(home, "h1v", ARM="h1v-x")
    line = start_line(result)
    assert "cfg h1v arm h1v-x slot HV HIP 1,2,3,4,5 buses d4 d9 dc df e2" in line
    assert "hetero: hub engine sees HIP 1 only" in result.stdout
    # no W7800 in this layout: nothing is refused
    assert "(refused HIP none)" in result.stdout
    plan = (
        f"{home}/no-tree/bench_results/2026-11-10_hetero-hub-standin/H1v.W1.plan.json"
    )
    assert f"VLLM_HETERO_MOE_PLAN={plan}" in line


def test_extra_ov_comes_after_the_hetero_keys_and_wins(home):
    result = driver(home, "h1v", EXTRA_OV="HETERO_LOG_DIR=/elsewhere KV=auto")
    line = start_line(result)
    assert line.index("VLLM_HETERO_MOE_PLAN=") < line.index("HETERO_LOG_DIR=/elsewhere")
    assert line.rindex("HETERO_LOG_DIR=") == line.index("HETERO_LOG_DIR=/elsewhere")


def test_given_plan_replaces_the_default_one(home):
    for cfg in ("h1", "h1v"):
        result = driver(home, cfg, PLAN="/plans/mine.json")
        assert "VLLM_HETERO_MOE_PLAN=/plans/mine.json" in start_line(result)


@pytest.mark.parametrize(
    ("cfg", "slot", "hips", "overrides"),
    [
        ("tp8", "AB", "1,2,3,4,6,7,8,9", "TP=8"),
        ("pp2", "AB", "1,2,3,4,6,7,8,9", "TP=4 PP=2 PP_PARTITION=24,24"),
        ("tp4a", "A", "1,2,3,4", "TP=4"),
        ("tp4b", "B", "6,7,8,9", "TP=4"),
    ],
)
def test_existing_cfgs_have_no_hetero_line(home, cfg, slot, hips, overrides):
    result = driver(home, cfg, ARM="x")
    line = start_line(result)
    assert f"slot {slot} HIP {hips} " in line
    assert line.endswith(f"overrides {overrides}")
    assert "hetero:" not in result.stdout


# --- the leases and the plan check of CFG=h1v (driver.sh sourced, main not run) ----


def lib(home, script, **extra):
    env = {
        "OWNER": "tester",
        "SHA": "0" * 40,
        "RUN_DIR": str(home / "run"),
        "LANES": str(home / "lanes"),
        "TREE_T": str(REPO),
        "CFG": "h1v",
        **extra,
    }
    return subprocess.run(
        ["bash", "-c", f'source "{DRIVER}"; {script}'],
        capture_output=True,
        text=True,
        env=base_env(home, INTERIM, **env),
        timeout=60,
    )


def lease(home, *names):
    (home / "lanes" / "lease").mkdir(parents=True, exist_ok=True)
    for name in names:
        (home / "lanes" / "lease" / name).write_text("")


def test_h1v_takes_both_leases_of_the_interim_layout(home):
    lease(home, "tester.a2")
    assert lib(home, "need_leases").returncode == 1  # b4 is missing
    lease(home, "tester.b4")
    assert lib(home, "need_leases").returncode == 0
    # another owner's holder files are not mine
    assert lib(home, "need_leases", OWNER="other").returncode == 1


def test_h1v_refuses_a_foreign_holder_of_either_lease(home):
    lease(home, "tester.a2", "tester.b4", "tester.b4.note")
    for foreign in ("HxServer.a2", "HxServer.b4"):
        lease(home, foreign)
        result = lib(home, "need_leases")
        assert result.returncode == 1
        assert f"foreign lease file {foreign}" in result.stdout
        (home / "lanes" / "lease" / foreign).unlink()
    assert lib(home, "need_leases").returncode == 0


def test_h1v_ignores_requests_and_the_old_layout_leases(home):
    lease(home, "tester.a2", "tester.b4", "a2.request", "b4.request", "Other.slotA")
    assert lib(home, "need_leases").returncode == 0


def test_h1_lease_check_is_unchanged(home):
    lease(home, "tester.slotA")
    assert lib(home, "need_leases", CFG="h1").returncode == 1
    lease(home, "tester.w7800cb")
    assert lib(home, "need_leases", CFG="h1").returncode == 0
    # the interim leases do not count for it
    lease(home, "tester.a2", "tester.b4")
    assert lib(home, "need_leases", CFG="h1").returncode == 0
    lease(home, "Other.w7800cb")
    assert lib(home, "need_leases", CFG="h1").returncode == 1


@pytest.fixture()
def plan_venv(home):
    """The stub python runs the plan CLI with this interpreter (numpy + regex)."""
    python = home / "venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (home / "run" / "arms" / "h1v").mkdir(parents=True)
    return home


def test_check_plan_accepts_the_committed_standin_plan(plan_venv):
    result = lib(plan_venv, "check_plan")
    assert result.returncode == 0, result.stdout
    assert "H1v.W1.plan.json (plan 4f5011cdc2d89dcd (replicate): 1 island(s)" in (
        result.stdout
    )
    assert "4 expert device(s), 48 target + 1 MTP layer(s)" in result.stdout


def test_check_plan_refuses_a_missing_or_unusable_plan(plan_venv):
    result = lib(plan_venv, "check_plan", PLAN=str(plan_venv / "missing.json"))
    assert result.returncode == 1
    assert "is not a readable file" in result.stdout
    bad = plan_venv / "bad.json"
    bad.write_text("{}")
    result = lib(plan_venv, "check_plan", PLAN=str(bad))
    assert result.returncode == 1
    assert "plan check failed" in result.stdout


def test_check_plan_is_a_no_op_for_the_other_configs(plan_venv):
    result = lib(plan_venv, "check_plan", CFG="tp4a", ARM="tp4a")
    assert result.returncode == 0 and result.stdout == ""
