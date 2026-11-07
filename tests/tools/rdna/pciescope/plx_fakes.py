# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers of the pciescope PEX88096 (plx) tests (not a test module).

``plx_par1_cs25.json.gz`` is a read-only capture of both PEX88096 switches of
par1-cs25 (``python -m pciescope.plx capture``): the topology and the non-zero
dwords of the BAR0 window of every port.  ``plx_lspci_expect.json`` holds the
values ``lspci -vvv -d 1000:c010`` printed for the same functions, taken right
after the capture; the decoder tests compare against it.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path

from pciescope.model import Topology
from pciescope.plx import fixture, helper, regs
from pciescope.plx.access import AccessError, LocalClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CAPTURE = FIXTURES / "plx_par1_cs25.json.gz"
LSPCI = FIXTURES / "plx_lspci_expect.json"

#: USP BDF (domain form) and PEX port of the two switches in the capture
SWITCH_A = "0000:c5:00.0"
SWITCH_B = "0000:01:00.0"


def load_capture() -> tuple[Topology, dict[str, dict[int, dict[int, int]]], dict]:
    """(topology, {chip bdf: {port: {offset: dword}}}, raw document)."""
    doc = fixture.read_doc(CAPTURE)
    topo = Topology.from_dict(doc["topology"])
    chips = {
        bdf: {
            int(port): {int(off, 16): int(v, 16) for off, v in win.items()}
            for port, win in c["windows"].items()
        }
        for bdf, c in doc["chips"].items()
    }
    return topo, chips, doc


def load_lspci() -> dict[str, dict]:
    return json.loads(LSPCI.read_text())


class FakeAtlas:
    """A BAR0 fake: recorded windows plus a model of the PerformancePAK RAM.

    Behaviour modelled (Broadcom SDK semantics, the only description we have):

    * ``0x3E0`` of a station base port: bit 30 zeroes the live counters, bit 31
      enables counting; the value written is read back.
    * ``0x3F0`` with bit 2 set resets the station's FIFO read pointer and
      latches (captures) the live counters into the FIFO image.
    * ``0x3E4`` pops the next dword of the captured image (a read has a side
      effect, logged in ``fifo_reads``).
    * ``0xF30`` reads back its content without bit 21 (egress credit enable
      always reads 0 on Atlas).

    ``writes`` logs every write as ``(bar0 offset, value)`` in order.
    """

    def __init__(self, windows: dict[int, dict[int, int]]) -> None:
        self.win = {p: dict(w) for p, w in windows.items()}
        self.writes: list[tuple[int, int]] = []
        self.fifo_reads = 0
        self.live = {s: [0] * regs.PERF_DWORDS_PER_STATION for s in range(8)}
        self.image = {s: [0] * regs.PERF_DWORDS_PER_STATION for s in range(8)}
        self.ptr = dict.fromkeys(range(8), 0)

    # -- Mmio -----------------------------------------------------------------
    def _split(self, off: int) -> tuple[int, int]:
        port, reg = divmod(off - regs.PORT_WINDOW_BASE, regs.PORT_WINDOW_SIZE)
        if port not in self.win:
            raise AccessError(f"fake: no window for port {port}")
        return port, reg

    def read32(self, off: int) -> int:
        port, reg = self._split(off)
        if reg == regs.REG_PERF_FIFO and port % regs.PORTS_PER_STATION == 0:
            s = port // regs.PORTS_PER_STATION
            v = self.image[s][self.ptr[s]]
            self.ptr[s] += 1
            self.fifo_reads += 1
            return v & 0xFFFFFFFF
        v = self.win[port].get(reg, 0)
        if reg == regs.REG_PERF_EGRESS:
            v &= ~regs.PERF_EGRESS_CREDIT_BIT
        return v

    def write32(self, off: int, val: int) -> None:
        port, reg = self._split(off)
        self.writes.append((off, val))
        station_base = port % regs.PORTS_PER_STATION == 0
        s = port // regs.PORTS_PER_STATION
        if reg == regs.REG_PERF_CTRL and station_base and val & 1 << 30:
            self.live[s] = [0] * regs.PERF_DWORDS_PER_STATION
        elif reg == regs.REG_PERF_RAM_CTRL and station_base and val & 4:
            self.ptr[s] = 0
            self.image[s] = list(self.live[s])
        self.win[port][reg] = val

    # -- test hooks -------------------------------------------------------------
    def add_traffic(
        self, port: int, *, ingress: dict[str, int], egress: dict[str, int]
    ):
        """Bump the live counters of ``port`` (field name -> increment)."""
        s, p = divmod(port, regs.PORTS_PER_STATION)
        order = list(regs.PERF_TLP_FIELDS)
        for name, inc in ingress.items():
            if name == "dllp":
                self.live[s][regs.PERF_DLLP_IN_BASE + p] += inc
            else:
                self.live[s][regs.PERF_INGRESS_BASE + 6 * p + order.index(name)] += inc
        for name, inc in egress.items():
            if name == "dllp":
                self.live[s][regs.PERF_DLLP_OUT_BASE + p] += inc
            else:
                self.live[s][regs.PERF_EGRESS_BASE + 6 * p + order.index(name)] += inc
        self.live[s] = [v & 0xFFFFFFFF for v in self.live[s]]  # 32-bit hardware wrap


def chip_access(
    windows: dict[int, dict[int, int]],
    bdf: str = SWITCH_B,
    *,
    allow_writes: bool = False,
    poll_s: float = 3600.0,
) -> tuple[helper.ChipAccess, FakeAtlas]:
    """A ``ChipAccess`` over a ``FakeAtlas``; the sampler thread stays idle."""
    fake = FakeAtlas(windows)
    chip = helper.ChipAccess(
        bdf, regs.DEFAULT_USP_PORT, fake, allow_writes=allow_writes, poll_s=poll_s
    )
    return chip, fake


def local_client(chips: dict[str, helper.ChipAccess]) -> LocalClient:
    return LocalClient(chips)


def write_resource0(path: Path, windows: dict[int, dict[int, int]]) -> None:
    """A file shaped like ``resource0``: 16 MiB with the recorded windows in it."""
    with open(path, "wb") as f:
        f.truncate(regs.BAR0_SIZE)
    fd = os.open(path, os.O_RDWR)
    try:
        mm = mmap.mmap(fd, regs.BAR0_SIZE)
        for port, win in windows.items():
            for off, val in win.items():
                struct.pack_into("<I", mm, regs.port_window(port) + off, val)
        mm.flush()
        mm.close()
    finally:
        os.close(fd)
