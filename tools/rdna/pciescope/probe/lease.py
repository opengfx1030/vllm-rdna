# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU lease check: refuse to probe a GPU nobody leased to the caller.

Real format on par1-cs25 (observed read-only on 2026-11-07, directory
``~/w4a8_runs/lanes/lease/``; keys of the interim layout from its README of
2026-11-10):

* holder files are named ``<owner>.<key>`` with ``<key>`` one of ``a2``, ``b4``
  (since the 2026-11-08 case move: HIP 0-1 and HIP 2-5), ``slotA``, ``slotB``,
  ``w7800cb``, ``w7800-07`` (pre-move: ``ExpertStats.slotB``,
  ``Rdna3Port.w7800cb``); owners may contain dots, so the key is matched as a
  suffix.
* the content is free text of one line: ``ExpertStats 2026-11-07T13:42:49+01:00
  t10_run.sh pid 626742`` (``pid N`` present) or ``686235 2026-11-07T14:12:40+01:00
  G0-G7`` (no ``pid`` word: the number is not trusted as a pid).
* other files (``README.md``, ``README.pre-topology-*.md``, ``slotB.request``,
  ``b4.request``) are not holders.

Every refusal is explicit: the check is a safety gate, so any doubtful case
(unknown bus, no owner, other owner holds the key, W7800 bus 07 without the
lead's file) refuses.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .hw import KNOWN_GPUS, LEAD_ONLY_KEYS, LEASE_KEYS
from .ids import normalize_bus

DEFAULT_LEASE_DIR: Path = Path("~/w4a8_runs/lanes/lease").expanduser()
ENV_LEASE_DIR = "PCIESCOPE_LEASE_DIR"
ENV_OWNER = "PCIESCOPE_OWNER"

_OWNER_EXTRA = frozenset("_.+-")
_READ_LIMIT = 4096
_GRANT_HINT = (
    "ask the lead for a window on {key}; do not create the holder file "
    "yourself unless the lead granted it"
)


def _find_pid(text: str) -> int | None:
    """Return N of the first word ``pid`` followed by ``[=:]`` and digits."""
    low = text.lower()
    start = 0
    while (at := low.find("pid", start)) != -1:
        start = at + 3
        if at > 0 and (low[at - 1].isalnum() or low[at - 1] == "_"):
            continue  # inside a longer word
        i = start
        while i < len(low) and low[i].isspace():
            i += 1
        if i < len(low) and low[i] in "=:":
            i += 1
            while i < len(low) and low[i].isspace():
                i += 1
        j = i
        while j < len(low) and low[j] in "0123456789":
            j += 1
        if j > i:
            pid = int(low[i:j])
            return pid if pid > 0 else None
    return None


def _valid_owner(name: str) -> bool:
    """Return True for a name usable as the owner part of a holder file."""
    return (
        name.isascii()
        and (name[0].isalnum() or name[0] == "_")
        and all(c.isalnum() or c in _OWNER_EXTRA for c in name)
    )


def lease_dir() -> Path:
    """Return the lease directory (env ``PCIESCOPE_LEASE_DIR`` overrides)."""
    env = os.environ.get(ENV_LEASE_DIR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_LEASE_DIR


def lease_key(bus: str) -> str | None:
    """Return the lease key of ``bus`` (``ce`` -> ``slotA``), None if unknown."""
    try:
        norm = normalize_bus(bus)
    except ValueError:
        return None
    spec = KNOWN_GPUS.get(norm)
    return spec.lease_key if spec else None


def resolve_owner(owner: str | None) -> str | None:
    """Return ``owner``, else env ``PCIESCOPE_OWNER``, else None."""
    for candidate in (owner, os.environ.get(ENV_OWNER)):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return None


@dataclass(frozen=True)
class Holder:
    """One holder file of the lease directory."""

    owner: str
    key: str
    path: Path
    mtime: float
    text: str
    pid: int | None
    pid_alive: bool | None


@dataclass(frozen=True)
class LeaseStatus:
    """Lease situation of one key for the requested buses."""

    key: str
    buses: tuple[str, ...]
    holders: tuple[Holder, ...]
    mine: Holder | None
    ok: bool
    needed: str  # file name the owner must hold, e.g. 'PcieProbe.slotB'
    reason: str  # empty when ok


@dataclass(frozen=True)
class LeaseCheck:
    """Result of :func:`check_leases`."""

    ok: bool
    owner: str | None
    statuses: tuple[LeaseStatus, ...]
    unknown: tuple[str, ...]
    needed: tuple[str, ...]
    reason: str  # one problem per line, empty when ok

    def message(self) -> str:
        """Return the multi-line refusal text, empty when the check passed."""
        if self.ok:
            return ""
        lines = [f"GPU lease check REFUSED (owner: {self.owner or 'none'}):"]
        lines.extend(f"  - {line}" for line in self.reason.splitlines())
        if self.needed:
            lines.append("  needed holder files: " + ", ".join(self.needed))
        return "\n".join(lines)


def _default_pid_alive(pid: int) -> bool:
    """Probe a pid on this host with signal 0 (EPERM means it exists)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, OSError):
        return False
    return True


def _split_name(name: str) -> tuple[str, str] | None:
    """Split ``<owner>.<key>`` into ``(owner, key)``; None for other files."""
    for key in LEASE_KEYS:
        suffix = "." + key
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)], key
    return None


def _read_holders(
    directory: Path, pid_alive: Callable[[int], bool]
) -> list[Holder] | None:
    """Read the holders; None when the directory itself is unreadable."""
    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except OSError:
        return None
    holders: list[Holder] = []
    for path in entries:
        split = _split_name(path.name)
        if split is None:
            continue
        try:
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
        except OSError:
            continue  # vanished between listing and stat: not a holder anymore
        try:
            with path.open("rb") as fh:
                text = fh.read(_READ_LIMIT).decode("utf-8", errors="replace").strip()
        except OSError:
            text = ""  # unreadable file still holds the key
        pid = _find_pid(text)
        alive = pid_alive(pid) if pid is not None else None
        holders.append(Holder(split[0], split[1], path, mtime, text, pid, alive))
    return holders


def list_holders(directory: Path | None = None) -> list[Holder]:
    """List every holder file (``<owner>.<key>``) of the lease directory.

    Sorted by file name; an unreadable or missing directory gives ``[]``.
    """
    found = _read_holders(directory or lease_dir(), _default_pid_alive)
    return found if found is not None else []


def _clock(mtime: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(mtime))


def _describe(holder: Holder) -> str:
    if holder.pid is None:
        pid = "no pid in file"
    elif holder.pid_alive:
        pid = f"pid {holder.pid} alive"
    else:
        pid = f"pid {holder.pid} dead, stale?"
    return f"{holder.owner} (file {holder.path.name}, {_clock(holder.mtime)}, {pid})"


def _status(
    key: str,
    buses: Sequence[str],
    holders: Sequence[Holder],
    owner: str | None,
) -> LeaseStatus:
    of_key = tuple(h for h in holders if h.key == key)
    mine = next((h for h in of_key if owner is not None and h.owner == owner), None)
    others = tuple(h for h in of_key if h is not mine)
    needed = f"{owner or '<owner>'}.{key}"
    where = f"{key} (buses {','.join(buses)})"
    problems: list[str] = []
    if mine is None:
        if key in LEAD_ONLY_KEYS:
            problems.append("W7800 bus 07 (HIP 5): NEVER without the lead")
        if others:
            who = "; ".join(_describe(h) for h in others)
            problems.append(f"held by {who}")
        else:
            problems.append("no holder file")
        problems.append(f"you need {needed}")
    elif others:
        who = "; ".join(_describe(h) for h in others)
        problems.append(f"you hold {needed} but it is also held by {who}")
    if not problems:
        return LeaseStatus(key, tuple(buses), of_key, mine, True, needed, "")
    problems.append(_GRANT_HINT.format(key=key))
    reason = f"{where}: " + "; ".join(problems)
    return LeaseStatus(key, tuple(buses), of_key, mine, False, needed, reason)


def check_leases(
    gpus: Sequence[str],
    owner: str | None = None,
    directory: Path | None = None,
    *,
    pid_alive: Callable[[int], bool] | None = None,
) -> LeaseCheck:
    """Check that ``owner`` holds a lease for every bus of ``gpus``.

    A bus is covered when ``<owner>.<key of the bus>`` exists and no other
    owner holds the same key. Refused: no GPUs, no or malformed owner, unknown
    buses, bus 07 without ``<owner>.w7800-07``, unreadable lease directory.
    ``pid_alive`` replaces the signal-0 probe (tests).
    """
    lines: list[str] = []
    resolved = resolve_owner(owner)
    if resolved is None:
        lines.append("no owner: pass --owner NAME or set PCIESCOPE_OWNER")
    elif not _valid_owner(resolved):
        lines.append(
            f"invalid owner {resolved!r}: use letters, digits, '_', '-', '+', "
            "'.' only, it becomes the holder file name"
        )
        resolved = None
    if not gpus:
        lines.append("no GPUs requested")

    by_key: dict[str, set[str]] = {}
    unknown: list[str] = []
    for token in gpus:
        try:
            bus = normalize_bus(token)
        except ValueError:
            unknown.append(token)
            lines.append(f"unknown GPU {token!r}: not a PCI bus id, refused")
            continue
        key = lease_key(bus)
        if key is None:
            unknown.append(bus)
            lines.append(
                f"unknown bus {bus!r}: not a GPU of this box "
                f"(known buses: {' '.join(sorted(KNOWN_GPUS))}), refused"
            )
            continue
        by_key.setdefault(key, set()).add(bus)

    directory = directory if directory is not None else lease_dir()
    probe = pid_alive if pid_alive is not None else _default_pid_alive
    holders = _read_holders(directory, probe)
    if holders is None:
        lines.append(f"lease directory {directory} is missing or unreadable")
        holders = []

    statuses = tuple(
        _status(key, sorted(by_key[key]), holders, resolved)
        for key in LEASE_KEYS
        if key in by_key
    )
    lines.extend(s.reason for s in statuses if not s.ok)
    needed = tuple(s.needed for s in statuses if not s.ok)
    ok = not lines
    return LeaseCheck(
        ok=ok,
        owner=resolved,
        statuses=statuses,
        unknown=tuple(unknown),
        needed=needed,
        reason="\n".join(lines),
    )


__all__ = [
    "DEFAULT_LEASE_DIR",
    "Holder",
    "LeaseCheck",
    "LeaseStatus",
    "check_leases",
    "lease_dir",
    "lease_key",
    "list_holders",
    "resolve_owner",
]
