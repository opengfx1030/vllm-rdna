# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test of the HI2 kit's slot / HIP -> PCI gate (AGENTS.md two-slot table).

The live table is faked: no GPU and no torch are needed, and a moved enumeration,
a W7800 or a check that would open a card outside the slot must be refused before
a launch.  The launcher-contract tests read the scripts (``INKLING2_KIT_ROOT``
points them at an older tree for the pre-fix run).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
KIT = Path(os.environ.get("INKLING2_KIT_ROOT", REPO_ROOT / "tools" / "inkling_hi2"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "inkling_hi2"))

import slot  # noqa: E402  (the sys.path line above finds the kit module)

V620 = "Radeon PRO V620"
TABLE = {
    0: ("d1", V620),
    1: ("d4", V620),
    2: ("d9", V620),
    3: ("dc", V620),
    4: ("df", V620),
    5: ("e2", V620),
}


def _read(name: str) -> str:
    return (KIT / name).read_text()


def test_the_two_slot_table_is_the_agents_one():
    assert slot.default_pair("a2") == ("0,1", "d1,d4")
    assert slot.default_pair("b4") == ("2,3,4,5", "d9,dc,df,e2")
    with pytest.raises(slot.SlotError, match="unknown SLOT"):
        slot.slot_spec("a4")


def test_declared_pairs_of_both_slots_pass():
    slot.check_declared("a2", "0,1", "d1,d4")
    slot.check_declared("b4", "2,3,4,5", "d9,dc,df,e2")
    # the same pair in the forms torch and operators print
    slot.check_declared("a2", "0, 1", "0xd1, 0000:d4:00.0")


def test_declared_mismatches_are_refused():
    with pytest.raises(slot.SlotError, match="SLOT=a2 is HIP 0,1"):
        slot.check_declared("a2", "0,2", "d1,d4")
    with pytest.raises(slot.SlotError, match="SLOT=a2 is PCI d1,d4"):
        slot.check_declared("a2", "0,1", "d1,dc")
    with pytest.raises(slot.SlotError, match="repeats"):
        slot.check_declared("a2", "0,0", "d1,d4")
    with pytest.raises(slot.SlotError, match="not a PCI bus"):
        slot.check_declared("a2", "0,1", "d1,zz")


def test_the_live_check_accepts_the_visible_pair():
    slot.check_visible("0,1", "d1,d4", {0: TABLE[0], 1: TABLE[1]})
    slot.check_visible(
        "2,3,4,5", "d9,dc,df,e2", {0: TABLE[2], 1: TABLE[3], 2: TABLE[4], 3: TABLE[5]}
    )


def test_the_live_check_refuses_a_moved_or_foreign_device():
    swapped = {0: ("d4", V620), 1: ("d1", V620)}
    with pytest.raises(slot.SlotError, match="shows PCI d4,d1"):
        slot.check_visible("0,1", "d1,d4", swapped)
    # six visible devices while the slot declares two: the check opened all cards
    with pytest.raises(slot.SlotError, match="sees 6 GPUs"):
        slot.check_visible("0,1", "d1,d4", TABLE)
    with pytest.raises(slot.SlotError, match="not V620"):
        slot.check_visible("0,1", "d1,d4", {0: ("d1", "W7800"), 1: ("d4", V620)})
    with pytest.raises(slot.SlotError, match="lists 2 GPUs"):
        slot.check_visible("0,1", "d1", {0: TABLE[0], 1: TABLE[1]})
    # the old 0/5 W7800 pair can never pass
    with pytest.raises(slot.SlotError):
        slot.check_visible("0,5", "d1,d4", {0: TABLE[0], 1: TABLE[5]})


def test_the_live_check_uses_only_the_server_environment(monkeypatch):
    """No visibility var is cleared: the check sees exactly the server's devices."""
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(slot.SlotError, match="set HIP_VISIBLE_DEVICES"):
        slot.visible_table()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3,4,5")
    with pytest.raises(slot.SlotError, match="unset ROCR_VISIBLE_DEVICES"):
        slot.visible_table()
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5")
    with pytest.raises(slot.SlotError, match="unset CUDA_VISIBLE_DEVICES"):
        slot.visible_table()
    assert "os.environ.pop" not in _read("slot.py")
    slot_source = _read("slot.py")
    assert "torch.cuda.init()" in slot_source  # count is 0 until init on 2.12+rocm
    assert "accelerator.device_count()" in slot_source


def test_the_launchers_unset_the_other_visibility_vars():
    for name in ("serve.sh", "smoke.sh"):
        text = _read(name)
        assert "unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES" in text, name
        assert "env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES" in text, name


def test_the_boot_lock_is_held_until_the_server_is_ready():
    serve = _read("serve.sh")
    assert "exec 9>" in serve
    assert "flock 9" in serve
    assert "exec 9>&-" in serve
    assert serve.index("exec 9>&-") > serve.index("READY_SECONDS")
    assert 'EXTRA_ARGS="$EXTRA" 9>&-' in serve  # the server child must not hold it
    assert 'flock -w 900 "$BOOT_LOCK" true' not in serve
    # the quiet directory is re-checked while the lock is held
    assert "quiet window opened under the boot lock" in serve


def test_the_launchers_check_the_lease_and_take_an_owner():
    for name in ("serve.sh", "smoke.sh"):
        text = _read(name)
        assert "OWNER=${OWNER:-HI2Inkling}" in text, name
        assert "LEASE_DIR" in text, name
        assert "holds the $SLOT lease" in text, name


def test_the_launchers_confirm_the_lease_before_touching_the_gpus():
    serve = _read("serve.sh")
    smoke = _read("smoke.sh")
    assert "\ncheck_devices\n" not in serve.split("Another owner's holder")[0]
    assert "\ncheck_devices\n" in serve.split("lease_check")[-1]
    assert "\ncheck_devices\n" not in smoke.split("Another owner's holder")[0]


def test_smoke_retouches_quiet_and_samples_the_window():
    smoke = _read("smoke.sh")
    assert "gpu_temps.py" in smoke
    assert "kfd_evictions.py" in smoke
    assert "95" in smoke
    assert "sleep 300" in smoke  # re-touch every 5 min
    assert 'timeout "$CELL_TIMEOUT"' in smoke  # <= 15 min per measured probe
    assert "READY_SECONDS" in smoke


def test_bench_samples_the_window():
    bench = _read("bench.py")
    assert "gpu_temps.py" in bench
    assert "kfd_evictions.py" in bench
    assert "95" in bench


def test_the_hi2_recipe_defaults_to_the_a2_pair():
    recipe = (
        REPO_ROOT / "tools" / "rdna" / "recipes" / "inkling2-hi2-tp2.env"
    ).read_text()
    assert "HIP_VISIBLE_DEVICES=0,1" in recipe
    assert "<slotA" not in recipe


def test_bus_names_normalise():
    assert slot.normalise_bus("d1") == "d1"
    assert slot.normalise_bus("0xD1") == "d1"
    assert slot.normalise_bus("0000:d1:00.0") == "d1"
    assert slot.normalise_bus(0x0A) == "0a"
    assert slot.normalise_bus(0xD1) == "d1"  # the int torch reports
    slot.check_visible("0,1", "d1,d4", {0: (0xD1, V620), 1: (0xD4, V620)})


def test_the_cli_checks_without_touching_torch(monkeypatch, capsys):
    for name in ("SLOT", "HIP_VISIBLE_DEVICES", "EXPECT_PCI_BUSES"):
        monkeypatch.delenv(name, raising=False)
    assert slot.main(["--slot", "a2", "--declared-only"]) == 0
    assert "SLOT=a2 HIP 0,1 -> PCI d1,d4" in capsys.readouterr().out
    assert (
        slot.main(
            [
                "--slot",
                "a2",
                "--hip",
                "0,2",
                "--expect-buses",
                "d1,d4",
                "--declared-only",
            ]
        )
        == 2
    )
    assert "SLOT=a2 is HIP 0,1" in capsys.readouterr().err
    assert slot.main(["--slot", "a2", "--print"]) == 0
    assert capsys.readouterr().out.split() == [
        "HIP_VISIBLE_DEVICES=0,1",
        "EXPECT_PCI_BUSES=d1,d4",
    ]
