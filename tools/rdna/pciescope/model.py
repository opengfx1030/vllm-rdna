# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared data contract of pciescope (stdlib only, Python >= 3.11).

Everything that crosses a module boundary is defined here: ``Sample``,
``Measurement``, ``Event``, the ``Collector`` protocol and the topology value
types (``Topology``, ``PciNode``, ``Link``, ``Gpu``, ``Switch``, ``RootComplex``).
Producers (passive collectors, the PEX88096 reader, the active probe) emit these
records; ``store`` writes them as JSONL; the CLI, the MCP server and the TUI read
them.  Nothing here touches the machine: discovery lives in ``topology``.

Conventions
-----------
* ``ts_ns`` / ``started_ns`` / ``ended_ns``: wall clock, integer nanoseconds since
  the epoch (``time.time_ns()``), so records of different processes line up.
* Entity ids: ``host`` | ``root:<bus>`` (``root:c0``) | ``switch:<A|B>`` |
  ``port:<BDF>`` (``port:c8:10.0``) | ``link:<upBDF>-<downBDF>``
  (``link:c8:10.0-cc:00.0``) | ``gpu:<bus>`` (``gpu:ce``).  A BDF is the short
  ``bb:dd.f`` form (lower case); a non-zero PCI domain is kept as
  ``dddd:bb:dd.f``.  ``up`` is the device closer to the root complex.
* Units: ``GT/s`` ``lanes`` ``GB/s`` (1e9 B/s, per direction) ``MB/s`` ``us``
  ``ms`` ``C`` ``W`` ``%`` ``B`` ``count`` ``bool`` ``mask``.  Metric names are
  ``<domain>.<name>`` and listed in ``METRICS`` (gauge or monotonic counter).
* ``Sample.tags`` values are strings.  Everything is JSON-serialisable through
  ``to_dict()`` / ``record_from_dict()``.

Collector contract
------------------
A collector is passive (sysfs, config-space reads, metadata), read-only and
cheap (target < 5 ms per ``collect()``).  It is built for a ``Topology`` by
``pciescope.collectors.build`` (module-level ``NAME`` and
``make(topo, **opts) -> Collector``), declares ``available()`` and returns
``Sample`` objects; it MAY expose ``drain_events() -> list[Event]``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1
TOOL_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# time / BDF / entity helpers
# ---------------------------------------------------------------------------


def now_ns() -> int:
    """Wall-clock nanoseconds since the epoch (the ``ts_ns`` convention)."""
    return time.time_ns()


@lru_cache(maxsize=4096)
def short_bdf(bdf: str) -> str:
    """Normalise ``0000:ce:00.0`` / ``CE:00.0`` to ``ce:00.0`` (domain kept if != 0)."""
    s = bdf.strip()
    try:
        head, fn = s.rsplit(".", 1)
        parts = [int(p, 16) for p in head.split(":")]
        func = int(fn, 10)
        if len(parts) == 2:
            dom, (bus, dev) = 0, parts
        elif len(parts) == 3:
            dom, bus, dev = parts
        else:
            raise ValueError("bad field count")
        if not (0 <= dom < 0x10000 and 0 <= bus < 256 and 0 <= dev < 32):
            raise ValueError("field out of range")
        if not 0 <= func < 8:
            raise ValueError("bad function")
    except ValueError:
        raise ValueError(f"not a PCI address: {bdf!r}") from None
    out = f"{bus:02x}:{dev:02x}.{func}"
    return out if dom == 0 else f"{dom:04x}:{out}"


def full_bdf(bdf: str) -> str:
    """Return the sysfs form ``dddd:bb:dd.f`` of a (short) BDF."""
    s = short_bdf(bdf)
    return s if s.count(":") == 2 else f"0000:{s}"


def bdf_bus(bdf: str) -> str:
    """Return the two-digit bus of a BDF (``ce:00.0`` -> ``ce``)."""
    return short_bdf(bdf).rsplit(":", 2)[-2]


HOST = "host"


def gpu_entity(bus: str) -> str:
    """Entity id of the GPU whose function 0 is on ``bus`` (``gpu:ce``)."""
    return f"gpu:{bus.lower()}"


def port_entity(bdf: str) -> str:
    """Entity id of a bridge/port (``port:c8:10.0``)."""
    return f"port:{short_bdf(bdf)}"


def link_entity(up_bdf: str, down_bdf: str) -> str:
    """Entity id of the link between an upstream and a downstream device."""
    return f"link:{short_bdf(up_bdf)}-{short_bdf(down_bdf)}"


def switch_entity(letter: str) -> str:
    """Entity id of a switch (``switch:A``)."""
    return f"switch:{letter.upper()}"


def root_entity(bus: str) -> str:
    """Entity id of a root complex (``root:c0``)."""
    return f"root:{bus.lower()}"


def entity_kind(entity: str) -> str:
    """Return the kind part of an entity id (``gpu``, ``link``, ``host`` ...)."""
    return entity.split(":", 1)[0]


def entity_id(entity: str) -> str:
    """Return the identifier part of an entity id (``ce`` for ``gpu:ce``)."""
    return entity.split(":", 1)[1] if ":" in entity else ""


def split_link(entity: str) -> tuple[str, str]:
    """Split ``link:<up>-<down>`` into the two BDFs."""
    if not entity.startswith("link:"):
        raise ValueError(f"not a link entity: {entity!r}")
    up, down = entity[5:].split("-", 1)
    return up, down


# ---------------------------------------------------------------------------
# PCIe link arithmetic
# ---------------------------------------------------------------------------

#: line-code efficiency per PCIe generation (GT/s -> payload bits per wire bit)
ENCODING_EFFICIENCY: dict[float, float] = {
    2.5: 8 / 10,
    5.0: 8 / 10,
    8.0: 128 / 130,
    16.0: 128 / 130,
    32.0: 128 / 130,
}

#: bytes added to each TLP on the wire at Gen3+ (4DW header 16 + seq 2 + LCRC 4 +
#: STP framing 4); a 3DW header would be 22.
TLP_OVERHEAD_BYTES = 26
#: DLLP traffic (ACK/NAK, flow-control updates) that shares the wire with TLPs.
DLLP_OVERHEAD = 0.02
#: Max_Payload_Size measured on par1-cs25 (``lspci -vv`` DevCtl of the Broadcom
#: ports, the AMD bridges, the GPUs and the root ports, 2026-11-07): 256 bytes.
DEFAULT_MPS = 256


def raw_gbs(speed_gts: float, width: int) -> float:
    """Payload rate in GB/s (1e9 B/s, one direction) after line coding.

    16 GT/s x16 -> 31.508 GB/s: the "raw" ceiling used as ``theoretical``.
    """
    enc = ENCODING_EFFICIENCY.get(round(speed_gts, 1))
    if enc is None:
        enc = 128 / 130 if speed_gts >= 8.0 else 8 / 10
    return speed_gts * width * enc / 8.0


def practical_gbs(speed_gts: float, width: int, mps: int = DEFAULT_MPS) -> float:
    """Best sustained one-direction rate expected for posted writes, GB/s.

    ``raw * mps / (mps + 26) * (1 - 2 %)``: TLP header/sequence/LCRC/framing
    overhead of a max-size memory write plus DLLP traffic.  x16 Gen4, MPS 256:
    28.0 GB/s (MPS 512: 29.4 GB/s).  Reads are lower (completion headers per
    64 B-128 B read completion boundary and the read-request rate).
    """
    return (
        raw_gbs(speed_gts, width)
        * mps
        / (mps + TLP_OVERHEAD_BYTES)
        * (1.0 - DLLP_OVERHEAD)
    )


def parse_speed_gts(text: str | None) -> float | None:
    """Parse sysfs ``current_link_speed`` (``16.0 GT/s PCIe``) into GT/s."""
    if not text:
        return None
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sample:
    """One scalar observation of one metric of one entity at one instant."""

    ts_ns: int
    entity: str
    metric: str
    value: float
    unit: str
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "sample",
            "ts_ns": self.ts_ns,
            "entity": self.entity,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
        }
        if self.tags:
            d["tags"] = dict(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Sample:
        return cls(
            ts_ns=int(d["ts_ns"]),
            entity=str(d["entity"]),
            metric=str(d["metric"]),
            value=float(d["value"]),
            unit=str(d.get("unit", "")),
            tags={str(k): str(v) for k, v in (d.get("tags") or {}).items()},
        )

    def key(self) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        """Series key: (entity, metric, sorted tags)."""
        return (self.entity, self.metric, tuple(sorted(self.tags.items())))


@dataclass(slots=True)
class Measurement:
    """One active measurement result (latency / bandwidth / concurrency ...).

    ``src``/``dst`` are entity ids (``gpu:ce``, ``host``); ``gpus_used`` lists the
    ``gpu:<bus>`` entities touched; ``value`` is in ``unit`` (``GB/s`` for
    ``bw``, ``us`` for ``lat``); ``stats`` holds p50/p99/min/max/n (and mean/std
    when known); ``theoretical`` is the raw line-rate ceiling of the narrowest
    link of the path in the same unit (None for latency); ``efficiency`` is
    ``value / theoretical`` for bandwidth.  ``path_class``: ``intra-switch``,
    ``cross-root`` or ``host``.
    """

    kind: str
    src: str
    dst: str
    direction: str
    size_bytes: int
    value: float
    unit: str
    stats: dict[str, float] = field(default_factory=dict)
    path_class: str = ""
    theoretical: float | None = None
    efficiency: float | None = None
    started_ns: int = 0
    ended_ns: int = 0
    gpus_used: list[str] = field(default_factory=list)
    tool_version: str = TOOL_VERSION
    # optional extras (keyword use recommended)
    id: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "measurement"}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in ("id", "tags", "error") and not v:
                continue
            if f.name == "ok" and v:
                continue
            d[f.name] = list(v) if isinstance(v, list) else v
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Measurement:
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw.setdefault("stats", {})
        kw["gpus_used"] = list(kw.get("gpus_used") or [])
        kw["tags"] = {str(k): str(v) for k, v in (kw.get("tags") or {}).items()}
        return cls(**kw)


SEVERITIES = ("info", "warn", "error")


@dataclass(slots=True)
class Event:
    """A discrete occurrence (link retrain, error burst, session/probe markers).

    ``name`` examples: ``session_start`` ``watch_start`` ``watch_stop``
    ``link_change`` ``link_degraded`` ``aer_delta`` ``thermal`` ``evicted_jump``
    ``measure_start`` ``measure_end`` ``collector_error``.
    """

    ts_ns: int
    entity: str
    name: str
    severity: str = "info"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "event",
            "ts_ns": self.ts_ns,
            "entity": self.entity,
            "name": self.name,
            "severity": self.severity,
        }
        if self.message:
            d["message"] = self.message
        if self.data:
            d["data"] = dict(self.data)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Event:
        return cls(
            ts_ns=int(d["ts_ns"]),
            entity=str(d["entity"]),
            name=str(d["name"]),
            severity=str(d.get("severity", "info")),
            message=str(d.get("message", "")),
            data=dict(d.get("data") or {}),
        )


Record = Sample | Measurement | Event


def record_from_dict(d: Mapping[str, Any]) -> Record | None:
    """Parse one JSONL line into a record; unknown ``type`` returns None."""
    t = d.get("type")
    if t == "sample":
        return Sample.from_dict(d)
    if t == "measurement":
        return Measurement.from_dict(d)
    if t == "event":
        return Event.from_dict(d)
    return None


# ---------------------------------------------------------------------------
# metric registry (names, units, gauge vs counter): also drives the docs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricDef:
    """Static description of a metric name."""

    name: str
    unit: str
    kind: str  # "gauge" | "counter" (monotonic since boot / driver load)
    entity: str  # entity kind it is emitted for
    source: str  # collector
    doc: str


def _m(
    name: str, unit: str, kind: str, entity: str, source: str, doc: str
) -> MetricDef:
    return MetricDef(name, unit, kind, entity, source, doc)


METRICS: dict[str, MetricDef] = {
    m.name: m
    for m in (
        _m(
            "pcie.link_speed_gts",
            "GT/s",
            "gauge",
            "port|gpu",
            "sysfs_link",
            "Current link speed of this device's own link end (sysfs "
            "current_link_speed = LnkSta). 2.5/5/8/16 GT/s = Gen1..4.",
        ),
        _m(
            "pcie.link_width",
            "lanes",
            "gauge",
            "port|gpu",
            "sysfs_link",
            "Current negotiated link width of this device's own link end.",
        ),
        _m(
            "pcie.asleep",
            "bool",
            "gauge",
            "port|gpu",
            "sysfs_link",
            "1 while the parent bridge of this device is runtime-suspended (D3hot): "
            "its link attributes are not read, because the kernel would resume the "
            "parent for every read (10-35 ms and a power transition per poll).",
        ),
        _m(
            "link.speed_gts",
            "GT/s",
            "gauge",
            "link",
            "sysfs_link",
            "Link speed, minimum of the two link ends.",
        ),
        _m(
            "link.width",
            "lanes",
            "gauge",
            "link",
            "sysfs_link",
            "Link width, minimum of the two link ends.",
        ),
        _m(
            "link.capacity_gbs",
            "GB/s",
            "gauge",
            "link",
            "sysfs_link",
            "Raw payload capacity of the link per direction at the current speed "
            "and width (after 128b/130b): 16 GT/s x16 = 31.5 GB/s.",
        ),
        _m(
            "link.degraded",
            "bool",
            "gauge",
            "link",
            "sysfs_link",
            "1 if the current speed or width is below the link maximum "
            "(minimum of both ends' max_link_speed/width) or the two ends disagree.",
        ),
        _m(
            "aer.correctable",
            "count",
            "counter",
            "port|gpu",
            "aer",
            "AER correctable errors reported by this device (aer_dev_correctable "
            "TOTAL_ERR_COR).",
        ),
        _m(
            "aer.nonfatal",
            "count",
            "counter",
            "port|gpu",
            "aer",
            "AER non-fatal uncorrectable errors (aer_dev_nonfatal).",
        ),
        _m(
            "aer.fatal",
            "count",
            "counter",
            "port|gpu",
            "aer",
            "AER fatal uncorrectable errors (aer_dev_fatal).",
        ),
        _m(
            "aer.rootport_correctable",
            "count",
            "counter",
            "port",
            "aer",
            "Root-port total of correctable errors received from below "
            "(aer_rootport_total_err_cor).",
        ),
        _m(
            "aer.rootport_nonfatal",
            "count",
            "counter",
            "port",
            "aer",
            "Root-port total of non-fatal errors received from below.",
        ),
        _m(
            "aer.rootport_fatal",
            "count",
            "counter",
            "port",
            "aer",
            "Root-port total of fatal errors received from below.",
        ),
        _m(
            "aer.error",
            "count",
            "counter",
            "port|gpu",
            "aer",
            "One AER error type with a non-zero count; tags class="
            "correctable|nonfatal|fatal and type=BadTLP|RxErr|...",
        ),
        _m(
            "gpu.busy_pct",
            "%",
            "gauge",
            "gpu",
            "amdgpu",
            "gpu_busy_percent (graphics engine activity).",
        ),
        _m(
            "gpu.mem_busy_pct",
            "%",
            "gauge",
            "gpu",
            "amdgpu",
            "mem_busy_percent (memory-controller activity).",
        ),
        _m("gpu.vram_used_bytes", "B", "gauge", "gpu", "amdgpu", "mem_info_vram_used."),
        _m(
            "gpu.vram_total_bytes",
            "B",
            "gauge",
            "gpu",
            "amdgpu",
            "mem_info_vram_total.",
        ),
        _m(
            "gpu.gtt_used_bytes",
            "B",
            "gauge",
            "gpu",
            "amdgpu",
            "mem_info_gtt_used (host memory mapped for the GPU).",
        ),
        _m("gpu.temp_edge_c", "C", "gauge", "gpu", "amdgpu", "hwmon temp1 (edge)."),
        _m(
            "gpu.temp_junction_c",
            "C",
            "gauge",
            "gpu",
            "amdgpu",
            "hwmon temp2 (junction); the safety limit of the box rules.",
        ),
        _m("gpu.temp_mem_c", "C", "gauge", "gpu", "amdgpu", "hwmon temp3 (memory)."),
        _m("gpu.power_w", "W", "gauge", "gpu", "amdgpu", "hwmon power1_average."),
        _m(
            "gpu.link_speed_gts",
            "GT/s",
            "gauge",
            "gpu",
            "amdgpu",
            "Link speed as the GPU firmware (SMU gpu_metrics) reports it.",
        ),
        _m(
            "gpu.link_width",
            "lanes",
            "gauge",
            "gpu",
            "amdgpu",
            "Link width as the GPU firmware (SMU gpu_metrics) reports it.",
        ),
        _m(
            "gpu.pcie_dpm_speed_gts",
            "GT/s",
            "gauge",
            "gpu",
            "amdgpu",
            "Speed of the active pp_dpm_pcie level (the GPU downshifts the link "
            "when idle on hosts that allow it).",
        ),
        _m(
            "gpu.pcie_dpm_width",
            "lanes",
            "gauge",
            "gpu",
            "amdgpu",
            "Width of the active pp_dpm_pcie level.",
        ),
        _m(
            "gpu.throttle_status",
            "mask",
            "gauge",
            "gpu",
            "amdgpu",
            "SMU gpu_metrics throttle_status bit mask (0 = not throttled).",
        ),
        _m(
            "gpu.asleep",
            "bool",
            "gauge",
            "gpu",
            "amdgpu",
            "1 while the GPU is runtime-suspended: amdgpu sysfs is not read, a read "
            "would resume the GPU. 0 while active.",
        ),
        _m(
            "gpu.pcie_rx_bytes_per_s",
            "B/s",
            "gauge",
            "gpu",
            "amdgpu",
            "Received PCIe bytes/s from the amdgpu pcie_bw file: packets x max "
            "payload, an upper bound (opt-in; the file blocks ~1 s per read and "
            "RDNA2/RDNA3 here do not expose it).",
        ),
        _m(
            "gpu.pcie_tx_bytes_per_s",
            "B/s",
            "gauge",
            "gpu",
            "amdgpu",
            "Sent PCIe bytes/s from the amdgpu pcie_bw file: packets x max payload, "
            "an upper bound (same caveats).",
        ),
        _m(
            "kfd.evicted_ms",
            "ms",
            "counter",
            "gpu",
            "kfd",
            "KFD per-process queue eviction time (stats_<gpu_id>/evicted_ms); "
            "tags pid, comm. A jump invalidates a measured window.",
        ),
        _m(
            "collector.cost_ms",
            "ms",
            "gauge",
            "host",
            "runner",
            "Wall time of one collect() call, tag collector=<name>.",
        ),
        _m(
            "collector.samples",
            "count",
            "gauge",
            "host",
            "runner",
            "Number of samples one collect() call produced, tag collector.",
        ),
    )
}

try:  # the PEX88096 register collector contributes its own rows (plain tuples)
    from pciescope.plx.metrics import ROWS as _PLX_ROWS
except ImportError:  # collectors/plx.py and the plx package are optional
    _PLX_ROWS = ()
METRICS.update({row[0]: _m(*row) for row in _PLX_ROWS})


def metric_kind(metric: str) -> str:
    """Return ``counter`` or ``gauge`` (unknown metrics are gauges)."""
    d = METRICS.get(metric)
    return d.kind if d else "gauge"


@runtime_checkable
class Collector(Protocol):
    """Passive, read-only, cheap sampler of one data source."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Return (usable on this host, reason when not)."""
        ...

    def collect(self) -> list[Sample]:
        """Read the source once and return its samples (target < 5 ms)."""
        ...


# ---------------------------------------------------------------------------
# topology value types
# ---------------------------------------------------------------------------

# PciNode.role values
ROLE_ROOT_PORT = "root_port"  # AMD root port (parent is the root complex)
ROLE_SWITCH_UP = "switch_up"  # PEX880xx upstream port (behind a root port)
ROLE_SWITCH_FABRIC = "switch_fabric"  # DSP of the switch-level virtual bridge (02:xx.0)
ROLE_SWITCH_SUB = "switch_sub"  # USP of a sub-switch (03:00.0, 0b:00.0 ...)
ROLE_SWITCH_DOWN = "switch_down"  # DSP a card / nothing is attached to (04:00.0)
ROLE_SWITCH_MGMT = "switch_mgmt"  # switch management endpoint (class 0x0180)
ROLE_GPU_UP = "gpu_bridge_up"  # on-card bridge, upstream side (AMD 1478)
ROLE_GPU_DOWN = "gpu_bridge_down"  # on-card bridge, downstream side (AMD 1479)
ROLE_GPU = "gpu"  # the GPU function (class 0x03xx, amdgpu)
ROLE_GPU_AUDIO = "gpu_audio"  # HDMI audio function of the GPU
ROLE_OTHER = "other"

# PciNode.port_type values: where the node's own LnkSta points.  A link exists
# between a "root_port"/"dsp" parent and its child; a "usp" -> "dsp" edge is the
# switch's internal virtual bus and is not a link.
PORT_ROOT = "root_port"
PORT_USP = "usp"
PORT_DSP = "dsp"
PORT_ENDPOINT = "endpoint"

# Link.kind values
LINK_HOST_UPLINK = "host_uplink"  # root port <-> switch upstream port (physical)
LINK_FABRIC = "fabric"  # fabric DSP <-> sub-switch USP inside the switch (no lanes)
LINK_GPU = "gpu_link"  # switch downstream port <-> GPU card (physical x16)
LINK_ONBOARD = "onboard"  # on-card bridge down port <-> GPU function
LINK_MGMT = "mgmt"  # switch fabric port <-> management endpoint (no lanes)
LINK_OTHER = "other"


@dataclass(slots=True)
class PciNode:
    """One PCI function of the fabric (bridge, endpoint or root port)."""

    bdf: str
    vendor: int
    device: int
    pci_class: int
    role: str = ROLE_OTHER
    port_type: str = ""
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    root: str = ""
    depth: int = 0
    driver: str | None = None
    secondary_bus: int | None = None
    subordinate_bus: int | None = None
    link_speed_gts: float | None = None
    link_width: int | None = None
    max_link_speed_gts: float | None = None
    max_link_width: int | None = None
    numa_node: int | None = None
    serial: str | None = None
    switch: str | None = None  # 'A' | 'B'
    gpu: str | None = None  # gpu entity this node is / belongs to
    label: str = ""
    has_aer: bool = False

    @property
    def entity(self) -> str:
        """Entity id: ``gpu:<bus>`` for the GPU function, else ``port:<BDF>``."""
        if self.role == ROLE_GPU:
            return gpu_entity(bdf_bus(self.bdf))
        return port_entity(self.bdf)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PciNode:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class Link:
    """A PCIe link between an upstream device and a downstream device.

    ``max_*`` is the capability of the link (minimum of both ends);
    ``max_gbs`` the raw payload rate at that speed/width.  ``physical`` is False for
    the virtual links inside the switch.
    """

    id: str
    up: str
    down: str
    kind: str
    physical: bool
    max_speed_gts: float | None = None
    max_width: int | None = None
    max_gbs: float | None = None
    gpu: str | None = None
    switch: str | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Link:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class Gpu:
    """One GPU with every index scheme the box uses.

    ``hip`` = HIP runtime index (rank of the KFD topology node, matches the box
    rules table); ``smi`` = rocm-smi / amd-smi index (ascending PCI address);
    ``kfd_node`` = KFD topology node id (1..N, 0 is the CPU node).  ``port`` is the
    switch downstream port the card hangs off; ``link`` the physical link entity of
    the card; ``slot`` is the lease slot letter of a gfx1030 card (W7800 cards are in
    no slot): the lease unit of the layout the box rules document
    (``topology.DOCUMENTED_LAYOUTS``), which is a switch (A, B) until 2026-11-08 and a
    leaf of the single remaining switch since (A = HIP 0-1, B = HIP 2-5), so it can
    differ from ``switch``.
    """

    id: str
    bus: str
    bdf: str
    name: str
    arch: str
    device_id: int
    root: str
    switch: str | None
    slot: str | None
    port: str | None
    link: str | None
    hip: int | None = None
    smi: int | None = None
    kfd_node: int | None = None
    kfd_gpu_id: int | None = None
    card: str | None = None
    render: str | None = None
    vram_total_bytes: int | None = None
    unique_id: str | None = None
    hwmon: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Gpu:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class Switch:
    """One PEX88096 switch: everything below one root port."""

    id: str
    letter: str
    root: str
    root_port: str
    upstream: str
    model: str = "PEX88096"
    ports: list[str] = field(default_factory=list)  # downstream ports (GPU + empty)
    mgmt: str | None = None
    gpus: list[str] = field(default_factory=list)
    serial: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Switch:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class RootComplex:
    """One CPU root complex (PCI root bus) that carries a switch."""

    id: str
    bus: str
    root_port: str
    switch: str | None = None
    gpus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RootComplex:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class Hop:
    """One step of a path: traffic crosses ``link`` toward ``direction``.

    ``up`` = child to parent (toward the root complex), ``down`` = parent to
    child.  ``link`` is None for the root-complex pseudo hop between two root
    ports (CPU data fabric) or between a root port and the host.
    """

    link: str | None
    direction: str
    node: str  # BDF of the device the hop leaves (up) or enters (down)


@dataclass(slots=True)
class PathInfo:
    """Classification of the path between two entities."""

    src: str
    dst: str
    path_class: str  # intra-switch | cross-root | host | same
    lca: str | None  # BDF of the lowest common ancestor bridge (None: host fabric)
    lca_role: str | None
    hops: list[Hop]
    physical_links: list[str]  # links with lanes on the path (no fabric/virtual)
    bottleneck_gbs: float | None  # min raw capacity over the physical links


@dataclass(slots=True)
class Topology:
    """The discovered fabric (the ``topo --json`` document)."""

    host: str
    captured_ns: int
    nodes: dict[str, PciNode] = field(default_factory=dict)
    links: dict[str, Link] = field(default_factory=dict)
    gpus: dict[str, Gpu] = field(default_factory=dict)
    switches: dict[str, Switch] = field(default_factory=dict)
    roots: dict[str, RootComplex] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    hip_source: str = "unknown"
    schema: int = SCHEMA_VERSION

    # -- lookups -----------------------------------------------------------

    def gpu_list(self, order: str = "hip") -> list[Gpu]:
        """GPUs sorted by ``hip`` (default), ``smi`` or ``bus``."""
        key = {
            "hip": lambda g: (g.hip is None, g.hip if g.hip is not None else 0, g.bus),
            "smi": lambda g: (g.smi is None, g.smi if g.smi is not None else 0, g.bus),
            "bus": lambda g: (g.bus,),
        }[order]
        return sorted(self.gpus.values(), key=key)

    def gpu(self, key: str | int) -> Gpu:
        """Find a GPU by ``gpu:ce``, ``ce``, ``ce:00.0``, ``hip:3`` or a HIP int."""
        if isinstance(key, int):
            for g in self.gpus.values():
                if g.hip == key:
                    return g
            raise KeyError(f"no GPU with HIP index {key}")
        k = str(key).strip().lower()
        if k.startswith("hip:"):
            return self.gpu(int(k[4:]))
        if k.startswith("gpu:"):
            k = k[4:]
        if k in {g.bus for g in self.gpus.values()}:
            return self.gpus[gpu_entity(k)]
        if ":" in k:
            bus = bdf_bus(k)
            if gpu_entity(bus) in self.gpus:
                return self.gpus[gpu_entity(bus)]
        raise KeyError(f"unknown GPU {key!r}")

    def node(self, bdf: str) -> PciNode:
        return self.nodes[short_bdf(bdf)]

    def link_above(self, bdf: str) -> Link | None:
        """The link between ``bdf`` and its parent (None for root ports)."""
        n = self.node(bdf)
        if n.parent is None:
            return None
        return self.links.get(link_entity(n.parent, n.bdf))

    def switch_of_gpu(self, entity: str) -> Switch | None:
        g = self.gpu(entity)
        return self.switches.get(switch_entity(g.switch)) if g.switch else None

    def gpus_of_switch(self, letter: str) -> list[Gpu]:
        return [g for g in self.gpu_list() if g.switch == letter.upper()]

    def slot_gpus(self, slot: str) -> list[Gpu]:
        """GPUs of lease slot ``A`` or ``B`` (the V620 cards of that lease unit)."""
        return [g for g in self.gpu_list() if g.slot == slot.upper()]

    def entities(self) -> list[str]:
        """Every entity id of the fabric (host, roots, switches, ports, links, GPUs)."""
        out = [HOST]
        out += [r.id for r in self.roots.values()]
        out += sorted(self.switches)
        out += [n.entity for n in self.nodes.values() if n.role != ROLE_GPU]
        out += sorted(self.links)
        out += sorted(self.gpus)
        return out

    def links_ordered(self, *, physical_only: bool = False) -> list[Link]:
        """Links in reading order: host uplinks, then per GPU (HIP order) its card
        link and on-card link, then everything else (virtual links last).
        """
        hip = {g.id: (g.hip if g.hip is not None else 99) for g in self.gpus.values()}
        kind_rank = {
            LINK_HOST_UPLINK: 0,
            LINK_GPU: 1,
            LINK_ONBOARD: 2,
            LINK_FABRIC: 3,
            LINK_MGMT: 4,
        }

        def key(lk: Link) -> tuple[int, int, int, str, str]:
            first = 0 if lk.kind == LINK_HOST_UPLINK else 1
            order = hip.get(lk.gpu or "", 99)
            return (first, order, kind_rank.get(lk.kind, 9), lk.switch or "", lk.id)

        links = [lk for lk in self.links.values() if lk.physical or not physical_only]
        return sorted(links, key=key)

    # -- paths -------------------------------------------------------------

    def _ancestors(self, bdf: str) -> list[str]:
        """BDFs from ``bdf`` up to its root port (inclusive)."""
        out = [bdf]
        while True:
            p = self.nodes[out[-1]].parent
            if p is None:
                return out
            out.append(p)

    def _anchor(self, entity: str) -> str | None:
        """The BDF a path starts at (None for ``host``)."""
        if entity == HOST:
            return None
        if entity.startswith("gpu:"):
            return self.gpu(entity).bdf
        if entity.startswith("port:"):
            return short_bdf(entity[5:])
        raise ValueError(f"cannot route from entity {entity!r}")

    def _link_hops(self, chain: list[str], direction: str) -> list[Hop]:
        """Hops over the link above each node of ``chain``.

        Edges that are not links (root ports, the USP -> DSP internal bus of a
        switch) produce no hop.
        """
        out: list[Hop] = []
        for x in chain:
            p = self.nodes[x].parent
            if p is not None and (lid := link_entity(p, x)) in self.links:
                out.append(Hop(lid, direction, x))
        return out

    def _same_switch(self, a: str, b: str) -> bool:
        sw = self.nodes[a].switch
        return sw is not None and sw == self.nodes[b].switch

    def path(self, src: str, dst: str) -> PathInfo:
        """Route between two entities (GPUs, ports or ``host``) over the tree.

        Traffic between two devices of one root complex turns around at their
        lowest common ancestor bridge; between root complexes (or to ``host``) it
        leaves through the root port into the CPU fabric, a pseudo hop with
        ``link=None``.
        """
        a, b = self._anchor(src), self._anchor(dst)
        if a is not None and a == b:
            return PathInfo(src, dst, "same", a, self.nodes[a].role, [], [], None)
        chain_a = self._ancestors(a) if a else []
        chain_b = self._ancestors(b) if b else []
        lca: str | None = None
        if a and b and chain_a[-1] == chain_b[-1]:
            in_b = set(chain_b)
            lca = next(x for x in chain_a if x in in_b)
            up = self._link_hops(chain_a[: chain_a.index(lca)], "up")
            below = chain_b[: chain_b.index(lca)]
            down = self._link_hops(below[::-1], "down")
            pclass = "intra-switch" if self._same_switch(a, b) else "intra-root"
        else:
            up = self._link_hops(chain_a, "up")
            down = self._link_hops(chain_b[::-1], "down")
            if a:
                up.append(Hop(None, "up", chain_a[-1]))
            if b:
                down.insert(0, Hop(None, "down", chain_b[-1]))
            pclass = "host" if a is None or b is None else "cross-root"
        hops = up + down
        phys = [h.link for h in hops if h.link and self.links[h.link].physical]
        caps = [g for x in phys if (g := self.links[x].max_gbs)]
        return PathInfo(
            src,
            dst,
            pclass,
            lca,
            self.nodes[lca].role if lca else None,
            hops,
            phys,
            min(caps) if caps else None,
        )

    def path_class(self, src: str, dst: str) -> str:
        """``intra-switch`` | ``intra-root`` | ``cross-root`` | ``host`` | ``same``."""
        return self.path(src, dst).path_class

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "host": self.host,
            "captured_ns": self.captured_ns,
            "hip_source": self.hip_source,
            "notes": list(self.notes),
            "roots": {k: v.to_dict() for k, v in self.roots.items()},
            "switches": {k: v.to_dict() for k, v in self.switches.items()},
            "gpus": {k: v.to_dict() for k, v in self.gpus.items()},
            "links": {k: v.to_dict() for k, v in self.links.items()},
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Topology:
        return cls(
            host=str(d.get("host", "")),
            captured_ns=int(d.get("captured_ns", 0)),
            nodes={k: PciNode.from_dict(v) for k, v in d.get("nodes", {}).items()},
            links={k: Link.from_dict(v) for k, v in d.get("links", {}).items()},
            gpus={k: Gpu.from_dict(v) for k, v in d.get("gpus", {}).items()},
            switches={k: Switch.from_dict(v) for k, v in d.get("switches", {}).items()},
            roots={k: RootComplex.from_dict(v) for k, v in d.get("roots", {}).items()},
            notes=list(d.get("notes", [])),
            hip_source=str(d.get("hip_source", "unknown")),
            schema=int(d.get("schema", SCHEMA_VERSION)),
        )


def group_by_series(samples: Iterable[Sample]) -> dict[tuple, list[Sample]]:
    """Group samples into time-ordered series keyed by ``Sample.key()``."""
    out: dict[tuple, list[Sample]] = {}
    for s in samples:
        out.setdefault(s.key(), []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s.ts_ns)
    return out
