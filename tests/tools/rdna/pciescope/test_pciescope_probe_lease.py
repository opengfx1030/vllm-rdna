# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease check of the pciescope probe on fake lease directories (no box access).

The holder files copy the real formats seen on par1-cs25 on 2026-11-07:
``ExpertStats.slotB`` with ``... pid 626742`` and ``Rdna3Port.w7800cb`` whose
content starts with a bare number and has no ``pid`` word.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope.probe import lease  # noqa: E402

REAL_SLOTB = "ExpertStats 2026-11-07T13:42:49+01:00 t10_run.sh pid 626742\n"
REAL_W7800CB = "686235 2026-11-07T14:12:40+01:00 G0-G7\n"
ALIVE = frozenset({626742})


def alive(pid: int) -> bool:
    return pid in ALIVE


@pytest.fixture
def real_dir(tmp_path: Path) -> Path:
    d = tmp_path / "lease"
    d.mkdir()
    (d / "ExpertStats.slotB").write_text(REAL_SLOTB)
    (d / "Rdna3Port.w7800cb").write_text(REAL_W7800CB)
    (d / "README.md").write_text("# GPU lease queue\n")
    (d / "README.pre-topology-2026-11-05.md").write_text("old\n")
    (d / "slotB.request").write_text(REAL_SLOTB)
    (d / "Dir.slotA").mkdir()  # a directory is never a holder
    return d


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCIESCOPE_OWNER", raising=False)
    monkeypatch.delenv("PCIESCOPE_LEASE_DIR", raising=False)


def test_lease_key_mapping() -> None:
    assert lease.lease_key("ce") == "slotA"
    assert lease.lease_key("gpu:DB") == "slotA"
    assert lease.lease_key("0000:0a:00.0") == "slotB"
    assert lease.lease_key("cb") == "w7800cb"
    assert lease.lease_key("07") == "w7800-07"
    assert lease.lease_key("ff") is None
    assert lease.lease_key("not a bus") is None
    assert lease.lease_key("") is None


def test_lease_dir_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert lease.lease_dir() == lease.DEFAULT_LEASE_DIR
    assert lease.DEFAULT_LEASE_DIR.parts[-3:] == ("w4a8_runs", "lanes", "lease")
    monkeypatch.setenv("PCIESCOPE_LEASE_DIR", "/somewhere/else")
    assert lease.lease_dir() == Path("/somewhere/else")


def test_resolve_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    assert lease.resolve_owner(None) is None
    assert lease.resolve_owner("  ") is None
    assert lease.resolve_owner(" Me ") == "Me"
    monkeypatch.setenv("PCIESCOPE_OWNER", "FromEnv")
    assert lease.resolve_owner(None) == "FromEnv"
    assert lease.resolve_owner("Arg") == "Arg"  # the argument wins


def test_list_holders_real_formats(real_dir: Path) -> None:
    holders = lease.list_holders(real_dir)
    assert [(h.owner, h.key) for h in holders] == [
        ("ExpertStats", "slotB"),
        ("Rdna3Port", "w7800cb"),
    ]
    stats, w7800 = holders
    assert stats.pid == 626742
    assert stats.text.startswith("ExpertStats 2026-11-07T13:42:49+01:00")
    assert stats.path == real_dir / "ExpertStats.slotB"
    assert stats.mtime == pytest.approx(
        (real_dir / "ExpertStats.slotB").stat().st_mtime
    )
    # a bare leading number is not trusted as a pid
    assert w7800.pid is None
    assert w7800.pid_alive is None


def test_list_holders_dotted_owner_and_odd_names(tmp_path: Path) -> None:
    (tmp_path / "HxServer.HxProbe.slotA").write_text("x pid=77 y")
    (tmp_path / ".slotA").write_text("no owner")  # empty owner: not a holder
    (tmp_path / "slotA").write_text("no dot")
    (tmp_path / "Me.slotC").write_text("unknown key")
    (tmp_path / "Me.w7800-07").write_text("lead file")
    holders = lease.list_holders(tmp_path)
    assert [(h.owner, h.key, h.pid) for h in holders] == [
        ("HxServer.HxProbe", "slotA", 77),
        ("Me", "w7800-07", None),
    ]


def test_list_holders_missing_directory(tmp_path: Path) -> None:
    assert lease.list_holders(tmp_path / "absent") == []


@pytest.mark.parametrize(
    ("text", "pid"),
    [
        ("run.sh pid 12", 12),
        ("run.sh PID: 7", 7),
        ("run.sh pid=99 more", 99),
        ("mypid 5", None),
        ("pid", None),
        ("pid abc", None),
        ("pid 0", None),
        ("", None),
        ("pid x pid 31", 31),
    ],
)
def test_pid_extraction(tmp_path: Path, text: str, pid: int | None) -> None:
    (tmp_path / "Me.slotA").write_text(text)
    (holder,) = lease.list_holders(tmp_path)
    assert holder.pid == pid


def test_default_pid_probe_alive_and_dead(tmp_path: Path) -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    (tmp_path / "Live.slotA").write_text(f"x pid {os.getpid()}")
    (tmp_path / "Dead.slotB").write_text(f"x pid {proc.pid}")
    dead, live = lease.list_holders(tmp_path)
    assert live.pid_alive is True
    assert dead.pid_alive is False


def test_ok_when_owner_holds_every_key(tmp_path: Path) -> None:
    (tmp_path / "Me.slotA").write_text("Me pid 1")
    (tmp_path / "Me.w7800cb").write_text("Me")
    res = lease.check_leases(["ce", "gpu:cb", "d3"], "Me", tmp_path, pid_alive=alive)
    assert res.ok
    assert res.owner == "Me"
    assert res.message() == ""
    assert res.reason == ""
    assert res.needed == ()
    assert res.unknown == ()
    assert [(s.key, s.buses, s.ok) for s in res.statuses] == [
        ("slotA", ("ce", "d3"), True),
        ("w7800cb", ("cb",), True),
    ]
    assert res.statuses[0].mine is not None
    assert res.statuses[0].mine.owner == "Me"


def test_other_owner_holds_key_is_refused(real_dir: Path) -> None:
    res = lease.check_leases(["0a", "0f"], "PcieProbe", real_dir, pid_alive=alive)
    assert not res.ok
    assert res.needed == ("PcieProbe.slotB",)
    (status,) = res.statuses
    assert status.key == "slotB"
    assert status.mine is None
    assert [h.owner for h in status.holders] == ["ExpertStats"]
    stamp = time.strftime("%H:%M:%S", time.localtime(status.holders[0].mtime))
    msg = res.message()
    assert "ExpertStats" in msg
    assert "ExpertStats.slotB" in msg
    assert stamp in msg
    assert "pid 626742 alive" in msg
    assert "PcieProbe.slotB" in msg
    assert "ask the lead for a window on slotB" in msg
    assert "do not create the holder file yourself unless the lead granted it" in msg


def test_conflict_when_both_hold_the_key(real_dir: Path) -> None:
    (real_dir / "PcieProbe.slotB").write_text("PcieProbe pid 5")
    res = lease.check_leases(["0a"], "PcieProbe", real_dir, pid_alive=alive)
    assert not res.ok
    assert res.statuses[0].mine is not None
    assert "also held by ExpertStats" in res.message()


def test_dead_holder_is_reported_stale_but_still_refuses(real_dir: Path) -> None:
    res = lease.check_leases(["0a"], "Me", real_dir, pid_alive=lambda pid: False)
    assert not res.ok
    assert "pid 626742 dead, stale?" in res.message()


def test_holder_without_pid_is_reported(real_dir: Path) -> None:
    (real_dir / "Rdna3Port.w7800cb").write_text("686235 2026-11-07T14:12:40+01:00")
    res = lease.check_leases(["cb"], "Me", real_dir, pid_alive=alive)
    assert not res.ok
    assert "Rdna3Port" in res.message()
    assert "no pid in file" in res.message()


def test_no_holder_file_names_the_needed_file(tmp_path: Path) -> None:
    res = lease.check_leases(["ce"], "Me", tmp_path)
    assert not res.ok
    assert res.needed == ("Me.slotA",)
    assert "no holder file" in res.message()
    assert "Me.slotA" in res.message()


def test_w7800_bus_07_never_without_the_lead(tmp_path: Path) -> None:
    res = lease.check_leases(["07"], "Me", tmp_path)
    assert not res.ok
    assert res.needed == ("Me.w7800-07",)
    assert "NEVER without the lead" in res.message()
    # a lease for the V620 of the same switch does not cover it
    (tmp_path / "Me.slotB").write_text("Me")
    res = lease.check_leases(["07", "0a"], "Me", tmp_path)
    assert not res.ok
    assert [s.key for s in res.statuses if s.ok] == ["slotB"]
    assert res.needed == ("Me.w7800-07",)
    (tmp_path / "Me.w7800-07").write_text("lead")
    assert lease.check_leases(["07", "0a"], "Me", tmp_path).ok


def test_w7800_bus_07_refused_for_another_owners_file(tmp_path: Path) -> None:
    (tmp_path / "Other.w7800-07").write_text("lead gave it to Other")
    res = lease.check_leases(["07"], "Me", tmp_path)
    assert not res.ok
    assert "Other" in res.message()


def test_w7800_cb_needs_its_own_file(tmp_path: Path) -> None:
    (tmp_path / "Me.slotA").write_text("Me")
    res = lease.check_leases(["ce", "cb"], "Me", tmp_path)
    assert not res.ok
    assert res.needed == ("Me.w7800cb",)


@pytest.mark.parametrize("token", ["ff", "zz", "gpu:zz", "99:99", "", "host"])
def test_unknown_bus_is_refused(tmp_path: Path, token: str) -> None:
    (tmp_path / "Me.slotA").write_text("Me")
    res = lease.check_leases(["ce", token], "Me", tmp_path)
    assert not res.ok
    assert len(res.unknown) == 1
    assert "unknown" in res.message()
    # the known part is still reported
    assert [s.key for s in res.statuses] == ["slotA"]


def test_no_owner_is_refused(tmp_path: Path) -> None:
    (tmp_path / "Me.slotA").write_text("Me")
    res = lease.check_leases(["ce"], None, tmp_path)
    assert not res.ok
    assert res.owner is None
    assert "pass --owner NAME or set PCIESCOPE_OWNER" in res.message()
    assert res.needed == ("<owner>.slotA",)


def test_owner_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "EnvOwner.slotA").write_text("x")
    monkeypatch.setenv("PCIESCOPE_OWNER", "EnvOwner")
    res = lease.check_leases(["ce"], None, tmp_path)
    assert res.ok
    assert res.owner == "EnvOwner"


@pytest.mark.parametrize("owner", ["a/b", "../x", "my agent", ".hidden", "a\\b"])
def test_malformed_owner_is_refused(tmp_path: Path, owner: str) -> None:
    res = lease.check_leases(["ce"], owner, tmp_path)
    assert not res.ok
    assert res.owner is None
    assert "invalid owner" in res.message()


def test_owner_with_dots_matches_exactly(tmp_path: Path) -> None:
    (tmp_path / "HxServer.HxProbe.slotA").write_text("x")
    assert lease.check_leases(["ce"], "HxServer.HxProbe", tmp_path).ok
    res = lease.check_leases(["ce"], "HxServer", tmp_path)
    assert not res.ok
    assert "HxServer.HxProbe" in res.message()  # named as the other holder


def test_owner_match_is_case_sensitive(tmp_path: Path) -> None:
    (tmp_path / "Me.slotA").write_text("x")
    assert not lease.check_leases(["ce"], "me", tmp_path).ok


def test_missing_lease_directory_is_refused(tmp_path: Path) -> None:
    res = lease.check_leases(["ce"], "Me", tmp_path / "absent")
    assert not res.ok
    assert "missing or unreadable" in res.message()


def test_default_directory_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Me.slotA").write_text("x")
    monkeypatch.setenv("PCIESCOPE_LEASE_DIR", str(tmp_path))
    assert lease.check_leases(["ce"], "Me").ok
    assert [h.owner for h in lease.list_holders()] == ["Me"]


def test_empty_request_is_refused(tmp_path: Path) -> None:
    res = lease.check_leases([], "Me", tmp_path)
    assert not res.ok
    assert "no GPUs requested" in res.message()


def test_status_order_follows_lease_keys_and_buses_are_sorted(tmp_path: Path) -> None:
    res = lease.check_leases(["17", "cb", "db", "0a", "ce", "07"], "Me", tmp_path)
    assert [(s.key, s.buses) for s in res.statuses] == [
        ("slotA", ("ce", "db")),
        ("slotB", ("0a", "17")),
        ("w7800cb", ("cb",)),
        ("w7800-07", ("07",)),
    ]
    assert res.needed == ("Me.slotA", "Me.slotB", "Me.w7800cb", "Me.w7800-07")


def test_duplicate_tokens_collapse(tmp_path: Path) -> None:
    (tmp_path / "Me.slotA").write_text("x")
    res = lease.check_leases(["ce", "gpu:ce", "0000:ce:00.0"], "Me", tmp_path)
    assert res.ok
    assert res.statuses[0].buses == ("ce",)


def test_message_lists_every_refused_key(real_dir: Path) -> None:
    res = lease.check_leases(["ce", "0a", "cb", "07"], "Me", real_dir, pid_alive=alive)
    msg = res.message()
    assert msg.startswith("GPU lease check REFUSED (owner: Me):")
    for needle in ("slotA", "slotB", "w7800cb", "w7800-07"):
        assert needle in msg
    assert "Rdna3Port" in msg
    assert "ExpertStats" in msg
    assert msg.count("\n") >= 4
