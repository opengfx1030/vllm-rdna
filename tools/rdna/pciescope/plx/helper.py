# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Root-side BAR0 access to a PEX88096 for pciescope (stdlib only).

Reading the PEX vendor registers needs root (``/sys/bus/pci/devices/<usp>/resource0``
is 0600), so the unprivileged collector starts this module once as a long-lived
child::

    sudo -n python -m pciescope.plx.helper --chip 0000:c5:00.0:0 [--allow-writes]

and talks JSON lines over stdin/stdout.  The same ``ChipAccess`` class is used
in-process when pciescope already runs as root and by the tests (over a fake
register file).

Safety model
------------
* BAR0 is mapped ``PROT_READ`` unless ``--allow-writes`` is given; without the
  flag there is no code path that writes, and a write attempt raises.
* Only registers of ports whose clock is enabled (masks at 0x30C..0x318 of the
  USP window) or that are in the fixed virtual-port set are touched, and never the
  read-sensitive registers (``regs.READ_SENSITIVE``) except inside the perf
  sequence.
* The only writes ever issued are the PerformancePAK enable/disable/capture
  sequence of the Broadcom SDK (``ChipAccess.start_ops`` / ``stop_ops`` /
  ``RAM_CAPTURE_OP``), on the ports the caller names, and only after the caller
  opted in twice (``--allow-writes`` here, ``allow_writes`` in the collector).
  ``perf_plan`` shows them with the current register values and writes nothing.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import mmap
import os
import signal
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pciescope.plx import pcie, regs

PROTOCOL_VERSION = 1
SYSFS_DEVICES = "/sys/bus/pci/devices"
#: counter poll period of the perf sampler: well below the ~0.5 s a 32-bit DW
#: counter needs to wrap at x16 Gen4 line rate (2**32 DW / 7.9e9 DW/s)
PERF_POLL_S = 0.1
_U32 = 0xFFFFFFFF


class AccessError(Exception):
    """A request was refused or the device did not answer as expected."""


class Mmio(Protocol):
    """32-bit register access to BAR0."""

    def read32(self, off: int) -> int: ...

    def write32(self, off: int, val: int) -> None: ...


class Bar0:
    """``resource0`` of the USP mapped through ``mmap`` with 32-bit accesses."""

    def __init__(self, path: str, writable: bool = False) -> None:
        self.path = path
        self.writable = writable
        self._fd = os.open(path, os.O_RDWR if writable else os.O_RDONLY)
        try:
            size = os.fstat(self._fd).st_size or regs.BAR0_SIZE
            prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
            self._mm = mmap.mmap(self._fd, size, mmap.MAP_SHARED, prot)
        except Exception:
            os.close(self._fd)
            raise
        # an 'I' view: item access is one aligned 32-bit load/store; it is
        # read-only exactly when the mapping is
        self._u32 = memoryview(self._mm).cast("I")
        self.size = size

    def read32(self, off: int) -> int:
        return self._u32[off >> 2]

    def write32(self, off: int, val: int) -> None:
        if not self.writable:
            raise AccessError("BAR0 is mapped read-only")
        self._u32[off >> 2] = val & _U32

    def lock(self) -> bool:
        """Advisory exclusive lock on the BAR0 file (one perf owner at a time)."""
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def close(self) -> None:
        self._u32.release()
        self._mm.close()
        os.close(self._fd)


def enabled_ports(masks: Sequence[int]) -> list[int]:
    """Physical ports whose clock is enabled, from the four 0x30C.. masks."""
    out: list[int] = []
    for word, mask in enumerate(masks[:3]):
        out.extend(word * 32 + i for i in range(32) if mask >> i & 1)
    if len(masks) > 3:
        out.extend(96 + 20 + i for i in range(12) if masks[3] >> i & 1)
    return sorted(out)


@dataclass(frozen=True, slots=True)
class WriteOp:
    """One register write of the PerformancePAK sequence.

    ``value`` is a plain write; otherwise a read-modify-write that sets
    ``set_bits`` and clears ``clear_bits`` of the current content.
    """

    name: str
    port: int
    reg: int  # offset inside the port window
    value: int | None = None
    set_bits: int = 0
    clear_bits: int = 0
    note: str = ""

    @property
    def bar0(self) -> int:
        return regs.port_window(self.port) + self.reg

    def result(self, old: int) -> int:
        if self.value is not None:
            return self.value
        return (old | self.set_bits) & ~self.clear_bits & _U32


#: per poll, to every station base port, before reading its 224 counter dwords
RAM_CAPTURE_VALUE = regs.PERF_RAM_CAPTURE


class PerfState:
    """Accumulated PerformancePAK counters of one chip (64-bit, unwrapped)."""

    def __init__(self, ports: Sequence[int], owned: bool) -> None:
        self.ports = list(ports)
        self.stations = sorted({regs.station_of(p) for p in ports})
        self.owned = owned
        self.totals: dict[int, list[int]] = {}
        self.last: dict[int, list[int]] = {}
        self.polls = 0
        self.resets = 0
        self.ts_ns = 0
        self.poll_ns = 0  # duration of the last poll
        self.error: str | None = None


class ChipAccess:
    """Policy layer over a BAR0 mapping of one PEX88096 (see module docstring)."""

    def __init__(
        self,
        bdf: str,
        usp_port: int,
        mmio: Mmio,
        *,
        allow_writes: bool = False,
        poll_s: float = PERF_POLL_S,
    ) -> None:
        if usp_port % regs.PORTS_PER_STATION:
            raise AccessError(
                f"USP port {usp_port} is not a station base port; unsupported"
            )
        self.bdf = bdf
        self.usp_port = usp_port
        self.mmio = mmio
        self.allow_writes = allow_writes
        self.poll_s = poll_s
        ident = mmio.read32(regs.port_window(usp_port))
        if ident != regs.ID_DWORD:
            raise AccessError(
                f"{bdf}: window of port {usp_port} reads id {ident:#010x}, expected "
                f"{regs.ID_DWORD:#010x} (wrong BAR or port number)"
            )
        base = regs.port_window(usp_port)
        #: Device Serial Number of the chip (every port window shows it), the key of
        #: ``regs.KNOWN_CHIPS``
        self.serial = pcie.decode_static(lambda o: mmio.read32(base + o)).serial
        entry = regs.KNOWN_CHIPS.get(self.serial or "")
        #: the registry entry, only while the chip sits where it was derived
        self.known = entry if entry is not None and entry.usp_bdf == bdf else None
        self.switch = None if self.known is None else self.known.switch
        if allow_writes and self.known is None:
            listed = "; ".join(
                f"{s} at {k.usp_bdf}" for s, k in sorted(regs.KNOWN_CHIPS.items())
            )
            raise AccessError(
                f"{bdf} (serial {self.serial or '?'}) is not a known PEX88096 of "
                f"par1-cs25 or moved since its entry was derived ({listed}): refusing "
                "to map it writable (plx/regs.py KNOWN_CHIPS)"
            )
        #: stations no write may land in -> why (the registry entry of the chip)
        self.protected: dict[int, str] = (
            {} if self.known is None else dict(self.known.protected)
        )
        #: ports whose windows no write may land in: every port of a protected station
        self.protected_ports = frozenset(
            p
            for s in self.protected
            for p in range(regs.station_base_port(s), regs.station_base_port(s + 1))
        )
        self.masks = [mmio.read32(base + o) for o in regs.REG_PORT_ENABLE]
        self.enabled = frozenset(enabled_ports(self.masks))
        self._perf: PerfState | None = None
        self._perf_lock = threading.Lock()  # serialises MMIO of the perf sequence
        self._data_lock = threading.Lock()  # guards PerfState totals
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- policy ---------------------------------------------------------------
    def check_port(self, port: int) -> None:
        if port not in self.enabled and port not in regs.VIRTUAL_PORTS:
            raise AccessError(
                f"port {port} is neither clock-enabled ({sorted(self.enabled)}) nor "
                "a known virtual port"
            )

    @staticmethod
    def _check_off(off: int) -> None:
        if off & 3 or not 0 <= off < regs.PORT_WINDOW_SIZE:
            raise AccessError(f"bad register offset {off:#x}")
        if off in regs.READ_SENSITIVE:
            raise AccessError(f"offset {off:#x} has read side effects")

    # -- read-only operations --------------------------------------------------
    def hello(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "euid": os.geteuid(),
            "writes": self.allow_writes,
            "bdf": self.bdf,
            "usp_port": self.usp_port,
            "serial": self.serial,
            "switch": self.switch,
            "protected_ports": sorted(self.protected_ports),
            "masks": self.masks,
            "enabled": sorted(self.enabled),
        }

    def read(self, items: Iterable[Sequence[int]]) -> list[int]:
        """Read ``[(port, offset), ...]`` (dwords), in order."""
        out: list[int] = []
        checked: set[int] = set()
        rd = self.mmio.read32
        for port, off in items:
            if port not in checked:
                self.check_port(port)
                checked.add(port)
            self._check_off(off)
            out.append(rd(regs.port_window(port) + off))
        return out

    def dump(self, ports: Iterable[int]) -> dict[int, dict[int, int]]:
        """Non-zero dwords of the windows of ``ports`` (sensitive offsets skipped)."""
        out: dict[int, dict[int, int]] = {}
        rd = self.mmio.read32
        for port in ports:
            self.check_port(port)
            base = regs.port_window(port)
            win = {}
            for off in range(0, regs.PORT_WINDOW_SIZE, 4):
                if off in regs.READ_SENSITIVE:
                    continue
                v = rd(base + off)
                if v:
                    win[off] = v
            out[port] = win
        return out

    # -- PerformancePAK write sequence ------------------------------------------
    def _check_perf_ports(self, ports: Sequence[int]) -> list[int]:
        if not ports:
            raise AccessError("no ports named")
        protected = sorted(p for p in ports if p in self.protected_ports)
        if protected:
            why = "; ".join(
                dict.fromkeys(self.protected[regs.station_of(p)] for p in protected)
            )
            raise AccessError(
                f"port(s) {protected} of switch {self.switch} lie in a protected "
                f"station ({why}): nothing is ever written to that station"
            )
        for p in ports:
            self.check_port(p)
            if p > regs.MAX_PHYSICAL_PORT or regs.station_of(p) >= regs.PERF_STATIONS:
                raise AccessError(f"port {p} has no PerformancePAK counters")
        return sorted(set(ports))

    def start_ops(self, ports: Sequence[int]) -> list[WriteOp]:
        """The ordered writes that start the counters on ``ports``.

        Broadcom SDK ``PlxPciPerformanceMonitorControl(START)`` for an Atlas,
        restricted to the named ports (the SDK touches all 96).  Per-station
        registers live in the station base port window (16 * station).
        """
        ports = self._check_perf_ports(ports)
        stations = sorted({regs.station_of(p) for p in ports})
        first = regs.station_base_port(stations[0])
        ops: list[WriteOp] = []
        probe = self.mmio.read32(regs.port_window(first) + regs.REG_PERF_PROBE_MODE)
        if probe & regs.PERF_PROBE_MODE_BIT:  # reads 0 on both chips: no write
            ops.append(
                WriteOp(
                    "probe_mode_off",
                    first,
                    regs.REG_PERF_PROBE_MODE,
                    clear_bits=regs.PERF_PROBE_MODE_BIT,
                    note="clear bit 8: probe mode must be off",
                )
            )
        ops += [
            WriteOp(
                "ingress_enable",
                regs.station_base_port(s),
                regs.REG_PERF_INGRESS,
                set_bits=regs.PERF_INGRESS_BIT,
                note="set bit 29: ingress counters of the station",
            )
            for s in stations
        ]
        ops += [
            WriteOp(
                "egress_enable",
                p,
                regs.REG_PERF_EGRESS,
                set_bits=regs.PERF_EGRESS_CREDIT_BIT | regs.PERF_EGRESS_ENABLE_BIT,
                note="set bit 6 (egress counters) and bit 21 (egress credit enable: "
                "reads 0, the SDK always keeps it set)",
            )
            for p in ports
        ]
        ops += [
            WriteOp(
                "monitor_start",
                regs.station_base_port(s),
                regs.REG_PERF_CTRL,
                value=regs.PERF_CTRL_START,
                note="enable | reset | infinite sampling | start",
            )
            for s in stations
        ]
        return ops

    def stop_ops(self, ports: Sequence[int]) -> list[WriteOp]:
        """The ordered writes that undo ``start_ops`` (SDK STOP)."""
        ports = self._check_perf_ports(ports)
        stations = sorted({regs.station_of(p) for p in ports})
        ops = [
            WriteOp(
                "ingress_disable",
                regs.station_base_port(s),
                regs.REG_PERF_INGRESS,
                clear_bits=regs.PERF_INGRESS_BIT,
                note="clear bit 29",
            )
            for s in stations
        ]
        ops += [
            WriteOp(
                "egress_disable",
                p,
                regs.REG_PERF_EGRESS,
                set_bits=regs.PERF_EGRESS_CREDIT_BIT,
                clear_bits=regs.PERF_EGRESS_ENABLE_BIT,
                note="clear bit 6, keep bit 21 set",
            )
            for p in ports
        ]
        ops += [
            WriteOp(
                "monitor_stop",
                regs.station_base_port(s),
                regs.REG_PERF_CTRL,
                value=regs.PERF_CTRL_STOP,
                note="reset | disable",
            )
            for s in stations
        ]
        return ops

    def perf_plan(self, ports: Sequence[int], *, stop: bool = False) -> dict[str, Any]:
        """Describe the START (or STOP) writes with the current register values.

        Read-only: nothing is written, so it works without ``allow_writes``.
        """
        ops = self.stop_ops(ports) if stop else self.start_ops(ports)
        ctl = self.mmio.read32(
            regs.port_window(regs.station_base_port(regs.station_of(min(ports))))
            + regs.REG_PERF_CTRL
        )
        steps = []
        for op in ops:
            old = self.mmio.read32(op.bar0)
            steps.append(
                {
                    "step": op.name,
                    "port": op.port,
                    "bar0": op.bar0,
                    "reg": op.reg,
                    "old": old,
                    "new": op.result(old),
                    "note": op.note,
                }
            )
        per_poll = [
            {
                "step": "ram_capture",
                "port": regs.station_base_port(s),
                "bar0": regs.port_window(regs.station_base_port(s))
                + regs.REG_PERF_RAM_CTRL,
                "reg": regs.REG_PERF_RAM_CTRL,
                "new": RAM_CAPTURE_VALUE,
                "note": "every poll: reset the counter RAM read pointer, then read "
                f"{regs.PERF_DWORDS_PER_STATION} dwords from 0x3E4 of the station",
            }
            for s in sorted({regs.station_of(p) for p in ports})
        ]
        return {
            "stop": stop,
            "monitor_already_running": bool(ctl & regs.PERF_BIT_ENABLE),
            "steps": steps,
            "per_poll": per_poll,
        }

    def _need_writes(self) -> None:
        if not self.allow_writes:
            raise AccessError(
                "register writes are disabled (start with --allow-switch-writes)"
            )

    def _write(self, off: int, val: int) -> None:
        """The only store to the chip: inside a port window, never a protected one."""
        if off < regs.PORT_WINDOW_BASE:
            raise AccessError(f"refusing a write outside the port windows: {off:#x}")
        if off & 3:
            raise AccessError(f"refusing an unaligned write: {off:#x}")
        port = (off - regs.PORT_WINDOW_BASE) // regs.PORT_WINDOW_SIZE
        if port in self.protected_ports:
            raise AccessError(
                f"refusing a write to port {port} of switch {self.switch} (protected "
                f"station: {self.protected[regs.station_of(port)]}): {off:#x}"
            )
        self.mmio.write32(off, val)

    def _apply(self, ops: Iterable[WriteOp]) -> None:
        r = self.mmio.read32
        for op in ops:
            self._write(op.bar0, op.result(r(op.bar0)))

    def perf_start(self, ports: Sequence[int]) -> dict[str, Any]:
        """Enable the PerformancePAK counters on ``ports`` and start sampling.

        Runs ``start_ops`` unless another agent already runs the monitor (bit 31 of
        0x3E0 set): then nothing is written except the per-poll RAM capture, and
        ``perf_stop`` leaves the monitor alone.
        """
        self._need_writes()
        if self._perf is not None:
            raise AccessError("perf monitor already started by this helper")
        ports = self._check_perf_ports(ports)
        lock = getattr(self.mmio, "lock", None)
        if lock is not None and not lock():
            raise AccessError("another process holds the BAR0 lock (perf owner)")
        first = regs.port_window(regs.station_base_port(regs.station_of(ports[0])))
        owned = not self.mmio.read32(first + regs.REG_PERF_CTRL) & regs.PERF_BIT_ENABLE
        with self._perf_lock:
            if owned:
                self._apply(self.start_ops(ports))
            self._perf = PerfState(ports, owned)
            self._poll_once()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="plx-perf-poll", daemon=True
        )
        self._thread.start()
        return self.perf_info()

    def perf_stop(self) -> dict[str, Any]:
        """Run ``stop_ops`` for the counters this helper enabled."""
        self._need_writes()
        st = self._perf
        if st is None:
            return {"running": False}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._perf_lock:
            if st.owned:
                self._apply(self.stop_ops(st.ports))
            self._perf = None
        return {"running": False, "owned": st.owned}

    def perf_poll(self) -> None:
        """One capture of every counter now (the sampler thread calls this)."""
        with self._perf_lock:
            if self._perf is not None:
                self._poll_once()

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                if self._perf is None:
                    return
                self.perf_poll()
            except Exception as exc:  # keep serving reads; report in perf_info
                if self._perf is not None:
                    self._perf.error = f"{type(exc).__name__}: {exc}"
                return

    def _poll_once(self) -> None:
        """One capture: reset RAM pointers, read 224 dwords per station, unwrap."""
        st = self._perf
        assert st is not None
        t0 = time.perf_counter_ns()
        w = self.mmio
        bases = [regs.port_window(regs.station_base_port(s)) for s in st.stations]
        for b in bases:
            self._write(b + regs.REG_PERF_RAM_CTRL, RAM_CAPTURE_VALUE)
        raw: dict[int, list[int]] = {}
        for s, b in zip(st.stations, bases):
            fifo = b + regs.REG_PERF_FIFO
            raw[s] = [w.read32(fifo) for _ in range(regs.PERF_DWORDS_PER_STATION)]
        ts = time.time_ns()
        with self._data_lock:
            for s, vals in raw.items():
                last = st.last.get(s)
                if last is None:
                    st.totals[s] = list(vals)
                else:
                    tot = st.totals[s]
                    for i, v in enumerate(vals):
                        d = (v - last[i]) & _U32
                        if d >= 1 << 31:  # counter went backwards: reset elsewhere
                            st.resets += 1
                            d = 0
                        tot[i] += d
                st.last[s] = vals
            st.polls += 1
            st.ts_ns = ts
            st.poll_ns = time.perf_counter_ns() - t0

    def perf_info(self) -> dict[str, Any]:
        st = self._perf
        if st is None:
            return {"running": False}
        with self._data_lock:
            return {
                "running": True,
                "owned": st.owned,
                "ports": st.ports,
                "stations": st.stations,
                "polls": st.polls,
                "resets": st.resets,
                "ts_ns": st.ts_ns,
                "poll_us": st.poll_ns // 1000,
                "error": st.error,
            }

    def perf_totals(self) -> dict[str, Any]:
        """Unwrapped counter totals per station (224 dwords each) since start."""
        info = self.perf_info()
        st = self._perf
        if st is None:
            return info
        with self._data_lock:
            info["totals"] = {str(s): list(v) for s, v in st.totals.items()}
        return info

    def close(self) -> None:
        """Stop the sampler and, if we started the counters, switch them off."""
        if self._perf is not None and self.allow_writes:
            self.perf_stop()
        self._stop.set()


# -- JSON-lines server ------------------------------------------------------------


def _handle(chips: dict[str, ChipAccess], req: dict[str, Any]) -> dict[str, Any]:
    op = req.get("op")
    if op == "hello":
        return {"chips": {b: c.hello() for b, c in chips.items()}}
    if op == "quit":
        raise EOFError
    chip = chips.get(req.get("chip", ""))
    if chip is None:
        raise AccessError(f"unknown chip {req.get('chip')!r}")
    if op == "read":
        t0 = time.perf_counter_ns()
        vals = chip.read(req["regs"])
        return {"vals": vals, "us": (time.perf_counter_ns() - t0) // 1000}
    if op == "dump":
        win = chip.dump(req["ports"])
        return {
            "windows": {
                str(p): {str(o): v for o, v in w.items()} for p, w in win.items()
            }
        }
    if op == "perf_plan":
        return chip.perf_plan(req["ports"], stop=bool(req.get("stop")))
    if op == "perf_start":
        return chip.perf_start(req["ports"])
    if op == "perf_stop":
        return chip.perf_stop()
    if op == "perf_info":
        return chip.perf_info()
    if op == "perf_totals":
        return chip.perf_totals()
    raise AccessError(f"unknown op {op!r}")


def serve(chips: dict[str, ChipAccess], rfile: Any, wfile: Any) -> None:
    """Answer JSON-line requests until EOF or ``quit``."""
    for line in rfile:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            resp = _handle(chips, req)
            resp["ok"] = True
        except EOFError:
            break
        except Exception as exc:
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        resp["id"] = rid
        wfile.write(json.dumps(resp, separators=(",", ":")) + "\n")
        wfile.flush()


def chip_may_write(bdf: str, allow: bool | Iterable[str]) -> bool:
    """True when ``allow`` (``True``: every chip, else the chips named) has ``bdf``."""
    if isinstance(allow, bool):
        return allow
    return bdf in set(allow)


def open_chip(
    bdf: str,
    usp_port: int,
    *,
    allow_writes: bool = False,
    sysfs: str = SYSFS_DEVICES,
) -> ChipAccess:
    """Map ``resource0`` of ``bdf`` and validate it (see ``ChipAccess``)."""
    bar = Bar0(f"{sysfs}/{bdf}/resource0", writable=allow_writes)
    try:
        return ChipAccess(bdf, usp_port, bar, allow_writes=allow_writes)
    except Exception:
        bar.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--chip",
        action="append",
        required=True,
        metavar="BDF:PORT",
        help="USP BDF (domain form) and its PEX port number, e.g. 0000:c5:00.0:0",
    )
    ap.add_argument(
        "--allow-writes", action="store_true", help="map every chip read-write"
    )
    ap.add_argument(
        "--write-chip",
        action="append",
        default=[],
        metavar="BDF",
        help="map only this chip read-write (repeatable); the others stay read-only",
    )
    ap.add_argument("--sysfs", default=SYSFS_DEVICES)
    args = ap.parse_args(argv)
    # SIGTERM/SIGHUP end the serve loop like EOF so the counters are switched off
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(0))
    chips: dict[str, ChipAccess] = {}
    try:
        specs = [spec.rpartition(":") for spec in args.chip]
        unknown = sorted(set(args.write_chip) - {bdf for bdf, _, _ in specs})
        if unknown:
            raise AccessError(
                f"--write-chip names chips not given with --chip: {unknown}"
            )
        allow: bool | list[str] = True if args.allow_writes else args.write_chip
        for bdf, _, port in specs:
            chips[bdf] = open_chip(
                bdf,
                int(port),
                allow_writes=chip_may_write(bdf, allow),
                sysfs=args.sysfs,
            )
    except Exception as exc:
        for chip in chips.values():
            chip.close()
        sys.stdout.write(
            json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n"
        )
        sys.stdout.flush()
        return 2
    try:
        serve(chips, sys.stdin, sys.stdout)
    finally:
        for chip in chips.values():
            chip.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
