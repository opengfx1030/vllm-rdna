# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PerformancePAK counter layout and arithmetic (pure functions, no hardware).

The counters only run after the gated enable sequence of ``helper.ChipAccess``;
this module turns the 224-dword station RAM image it returns into per-port TLP and
DLLP counters and into byte counts and rates.

Layout [SDK ``PlxPciPerformanceGetCounters``, Atlas, 6 stations x 16 ports]: per
station 224 dwords: ingress counters of port ``p`` at ``6 * p`` (posted header,
posted DW, non-posted header, non-posted DW, completion header, completion DW),
egress the same at ``96 + 6 * p``, ingress DLLP count at ``192 + p`` and egress
DLLP count at ``208 + p``.  "Ingress" is traffic received from the link partner
into the switch, "egress" traffic transmitted to the partner.

Arithmetic follows ``PlxPci_PerformanceCalcStatistics`` (SDK ``PlxApi.c``):

* total bytes = (posted DW + non-posted DW + completion DW) * 4 + DLLPs * 8
* payload bytes = DW * 4 - headers * overhead.  The SDK assumes 20 bytes (3 DW
  header + 2 DW) for every TLP.  Calibrated on par1-cs25 against transfers of known
  size (doc, "Calibration"): the DW counters include the header, the LCRC, one
  framing DW and the ECRC digest of a sender that generates one (the GPUs and the
  bridges on their cards do, the host does not), so a posted write of a GPU counts
  7 DW over its data (``POSTED_OVERHEAD_BYTES`` 28), a completion of the host 5 DW
  (20) and a completion of a GPU 6 DW (peer reads: the model is 1.6 % high at 256
  bytes there).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pciescope.plx import regs

DLLP_BYTES = 8
COMPLETION_OVERHEAD_BYTES = 20  # host completion: 3 DW header, LCRC, framing DW
POSTED_OVERHEAD_BYTES = 28  # GPU posted write: 4 DW header, ECRC, LCRC, framing DW


@dataclass(frozen=True, slots=True)
class DirCounters:
    """Cumulative counters of one direction of one port."""

    ph: int  # posted TLP headers
    pdw: int  # posted DW (incl. header and overhead DW)
    nph: int  # non-posted TLP headers
    npdw: int
    cplh: int  # completion TLP headers
    cpldw: int
    dllp: int

    def tlps(self) -> int:
        return self.ph + self.nph + self.cplh

    def tlp_bytes(self) -> int:
        """Bytes of all TLPs as counted by the chip (headers and overhead included)."""
        return (self.pdw + self.npdw + self.cpldw) * 4

    def total_bytes(self) -> int:
        """TLP bytes plus 8 bytes per DLLP (the SDK's TotalBytes)."""
        return self.tlp_bytes() + self.dllp * DLLP_BYTES

    def payload_bytes(self) -> int:
        """Estimated data bytes of posted writes and completions."""
        posted = self.pdw * 4 - self.ph * POSTED_OVERHEAD_BYTES
        cpl = self.cpldw * 4 - self.cplh * COMPLETION_OVERHEAD_BYTES
        return max(posted, 0) + max(cpl, 0)

    def minus(self, other: DirCounters) -> DirCounters:
        return DirCounters(
            self.ph - other.ph,
            self.pdw - other.pdw,
            self.nph - other.nph,
            self.npdw - other.npdw,
            self.cplh - other.cplh,
            self.cpldw - other.cpldw,
            self.dllp - other.dllp,
        )


def decode_port(
    station_dwords: Sequence[int], station_port: int
) -> tuple[DirCounters, DirCounters]:
    """(ingress, egress) counters of a station port from the 224-dword image."""
    if len(station_dwords) != regs.PERF_DWORDS_PER_STATION:
        raise ValueError(
            f"expected {regs.PERF_DWORDS_PER_STATION} dwords, got {len(station_dwords)}"
        )
    if not 0 <= station_port < regs.PORTS_PER_STATION:
        raise ValueError(f"station port {station_port} out of range")
    i = regs.PERF_INGRESS_BASE + 6 * station_port
    e = regs.PERF_EGRESS_BASE + 6 * station_port
    d = station_dwords
    ing = DirCounters(*d[i : i + 6], d[regs.PERF_DLLP_IN_BASE + station_port])
    egr = DirCounters(*d[e : e + 6], d[regs.PERF_DLLP_OUT_BASE + station_port])
    return ing, egr


@dataclass(frozen=True, slots=True)
class DirRates:
    """Rates of one direction over one interval."""

    payload_gbs: float  # estimated payload, GB/s (1e9 B/s)
    total_gbs: float  # TLP + DLLP bytes, GB/s
    util_pct: float  # total bytes over the raw link capacity, clamped to 100


def rates(
    prev: DirCounters, cur: DirCounters, dt_s: float, capacity_gbs: float
) -> DirRates:
    """Rates between two cumulative snapshots ``dt_s`` seconds apart."""
    if dt_s <= 0:
        return DirRates(0.0, 0.0, 0.0)
    d = cur.minus(prev)
    payload = d.payload_bytes() / dt_s / 1e9
    total = d.total_bytes() / dt_s / 1e9
    util = min(100.0, 100.0 * total / capacity_gbs) if capacity_gbs > 0 else 0.0
    return DirRates(payload, total, util)
