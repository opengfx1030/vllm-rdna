# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/tp8pp2: the hetero slot HA (hub W7800 + slot A) and CFG=h1 / h1v.

slots.sh is sourced with a stub venv python that prints the "hip:bus:name:GiB"
lines the real resolve_slot would get from torch, so slot_hip / slot_buses /
verify_slot are checked without a GPU. driver.sh is run only as far as its
start-up report (a missing exact-SHA tree stops it right after).
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SLOTS = REPO / "tools/rdna/tp8pp2/slots.sh"
DRIVER = REPO / "tools/rdna/tp8pp2/driver.sh"
W7800 = "AMD_Radeon_PRO_W7800"
V620 = "AMD_Radeon_PRO_V620"
# par1-cs25: HIP index -> "bus:name:GiB"
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
        if not k.startswith(("HUB_", "SLOT_", "DEV_", "HIP_", "VLLM_"))
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
    assert "hetero: hub engine sees HIP 0 only" in out
    assert f"plan {home}/run/arms/h1-x/plan.json" in out
    assert "(refused HIP 5)" in out
    assert f"VLLM_HETERO_MOE_PLAN={home}/run/arms/h1-x/plan.json" in line
    assert f"HETERO_LOG_DIR={home}/run/arms/h1-x/hetero" in line
    assert f"HETERO_PYTHON={home}/venv/bin/python" in line


def test_cfg_h1v_is_slot_a_with_the_first_v620_as_hub(home):
    result = driver(home, "h1v", ARM="h1v-x")
    line = start_line(result)
    assert "cfg h1v arm h1v-x slot A HIP 1,2,3,4 buses ce d3 d8 db" in line
    assert "hetero: hub engine sees HIP 1 only" in result.stdout
    assert "(refused HIP 0,5)" in result.stdout
    assert "VLLM_HETERO_MOE_PLAN=" in line


def test_extra_ov_comes_after_the_hetero_keys_and_wins(home):
    result = driver(home, "h1v", EXTRA_OV="HETERO_LOG_DIR=/elsewhere KV=auto")
    line = start_line(result)
    assert line.index("VLLM_HETERO_MOE_PLAN=") < line.index("HETERO_LOG_DIR=/elsewhere")
    assert line.rindex("HETERO_LOG_DIR=") == line.index("HETERO_LOG_DIR=/elsewhere")


def test_given_plan_replaces_the_generated_one(home):
    result = driver(home, "h1", PLAN="/plans/mine.json")
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
