# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test of the HI2 kit's slot / HIP -> PCI gate (AGENTS.md two-slot table).

The live table is faked: no GPU and no torch are needed, and a moved enumeration
or a W7800 must be refused before a launch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "inkling_hi2"))

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


def test_the_live_check_accepts_a2_and_b4():
    slot.check_physical("0,1", "d1,d4", TABLE)
    slot.check_physical("2,3,4,5", "d9,dc,df,e2", TABLE)


def test_the_live_check_refuses_a_moved_or_foreign_device():
    swapped = {**TABLE, 0: ("d4", V620), 1: ("d1", V620)}
    with pytest.raises(slot.SlotError, match="not the expected d1,d4"):
        slot.check_physical("0,1", "d1,d4", swapped)
    with pytest.raises(slot.SlotError, match="no GPU at HIP"):
        slot.check_physical("0,1", "d1,d4", {0: ("d1", V620)})
    with pytest.raises(slot.SlotError, match="not V620"):
        slot.check_physical("0,1", "d1,d4", {0: ("d1", "W7800"), 1: ("d4", V620)})
    with pytest.raises(slot.SlotError, match="lists 2 GPUs"):
        slot.check_physical("0,1", "d1", TABLE)
    # the old 0/5 W7800 pair can never pass
    with pytest.raises(slot.SlotError):
        slot.check_physical("0,5", "d1,d4", TABLE)


def test_bus_names_normalise():
    assert slot.normalise_bus("d1") == "d1"
    assert slot.normalise_bus("0xD1") == "d1"
    assert slot.normalise_bus("0000:d1:00.0") == "d1"
    assert slot.normalise_bus(0x0A) == "0a"
    assert slot.normalise_bus(0xD1) == "d1"  # the int torch reports
    slot.check_physical("0,1", "d1,d4", {0: (0xD1, V620), 1: (0xD4, V620)})


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
