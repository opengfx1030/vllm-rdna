# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Broadcom PEX88096 switch telemetry (collector ``plx``).

Reads the PEX vendor register windows of each switch through BAR0 of its upstream
port (see ``pciescope.plx``): link state, equalization, AER/DPC/lane error status
and port settings of every port function (the host uplink, the GPU-facing
downstream ports and the virtual fan-out ports) with one batched read per switch.
The registers need root, so the reads go through a long-lived ``sudo -n`` helper
(or in-process when already root); ``available()`` says why when that is not
possible.

The traffic counters (``port.rx_*`` / ``port.tx_*`` and ``plx.rx_*`` / ``plx.tx_*``)
are off at boot and cannot be read without enabling them, which writes switch
registers: they are only started with ``allow_writes=True`` (CLI
``--allow-switch-writes``) and the exact write list is documented in
``pciescope.plx.helper.ChipAccess.start_ops``.  Without the flag this collector
never writes anything.

Options of ``make``: ``allow_writes`` (default False), ``perf`` (start the traffic
counters, default = ``allow_writes``), ``perf_exclude`` (``gpu:<bus>`` entities
whose switch port is left alone, default the W7800 of HIP 5), ``perf_ports``
(only these PEX port numbers), ``write_switches`` (letters whose counters may be
enabled, default the approved ones, see ``pciescope.plx.policy``), ``switches``
(only these switches are read at all), ``client`` (a ``PlxClient``, for tests and
replay).
"""

from __future__ import annotations

import atexit
import contextlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pciescope.model import (
    Event,
    MetricDef,
    Sample,
    Topology,
    full_bdf,
    now_ns,
    port_entity,
    raw_gbs,
    switch_entity,
)
from pciescope.plx import metrics as plx_metrics
from pciescope.plx import pcie, perf, policy, regs
from pciescope.plx.access import AccessError, PlxClient, connect
from pciescope.plx.ports import (
    KIND_DSP,
    KIND_MGMT,
    KIND_USP,
    ChipMap,
    PortInfo,
    build_port_maps,
    verify_headers,
)
from pciescope.sysfs import Sysfs

NAME = "plx"

METRICS: dict[str, MetricDef] = {r[0]: MetricDef(*r) for r in plx_metrics.ROWS}

#: switch ports whose GPU is never touched (box rules: HIP 5 = W7800 on bus 07);
#: ``perf_exclude`` can only add to this list
DEFAULT_PERF_EXCLUDE = policy.NEVER_WRITTEN_GPUS

_PHYSICAL_KINDS = (KIND_USP, KIND_DSP)
#: dynamic registers of the chip-level perf state, read next to the port registers
_PERF_STATE_REGS = (
    ("perf_ctl", regs.REG_PERF_CTRL),
    ("perf_in", regs.REG_PERF_INGRESS),
    ("perf_out", regs.REG_PERF_EGRESS),
)


@dataclass(slots=True)
class _Port:
    info: PortInfo
    static: pcie.PortStatic
    offsets: tuple[tuple[str, int], ...]
    tags: dict[str, str]
    state: pcie.LinkState | None = None


@dataclass(slots=True)
class _Chip:
    cmap: ChipMap
    bdf: str  # domain form, the helper's chip key
    ports: dict[int, _Port]
    plan: list[tuple[int, int]]
    slots: list[tuple[int, str]]
    serial: str | None = None
    perf_ports: list[int] = field(default_factory=list)
    perf_tried: bool = False
    perf_running: bool = False
    perf_ts_ns: int = 0
    prev_perf: dict[int, tuple[int, perf.DirCounters, perf.DirCounters]] = field(
        default_factory=dict
    )
    sticky: dict[tuple[int, str], int] = field(default_factory=dict)


class PlxCollector:
    """Per-port state of the PEX88096 switches (and traffic when enabled)."""

    name = NAME

    def __init__(
        self,
        topo: Topology,
        sysfs: Sysfs | None = None,
        *,
        allow_writes: bool = False,
        perf: bool | None = None,
        perf_exclude: Iterable[str] = DEFAULT_PERF_EXCLUDE,
        perf_ports: Iterable[int] | None = None,
        write_switches: Iterable[str] | None = None,
        switches: Iterable[str] | None = None,
        client: PlxClient | None = None,
        connector: Callable[..., PlxClient] | None = None,
    ) -> None:
        self.topo = topo
        self.sysfs = sysfs
        self.allow_writes = allow_writes
        self.want_perf = allow_writes if perf is None else (perf and allow_writes)
        self.perf_exclude = frozenset(perf_exclude) | frozenset(
            policy.NEVER_WRITTEN_GPUS
        )
        self.perf_only = None if perf_ports is None else frozenset(perf_ports)
        self.write_switches = policy.write_switches(write_switches)
        self.switches = None if switches is None else policy.parse_switches(switches)
        self._client = client
        self._injected = client is not None
        self._connector = connector or connect  # looked up per instance (patchable)
        self._ready: tuple[bool, str] | None = None
        self._chips: list[_Chip] = []
        self._events: list[Event] = []
        self._atexit = False

    # -- Collector protocol -----------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if self._ready is None:
            try:
                self._init()
            except (AccessError, OSError, ValueError, KeyError) as exc:
                self._teardown()
                self._ready = (False, f"{type(exc).__name__}: {exc}")
            else:
                on = ",".join(
                    sorted(
                        c.cmap.switch
                        for c in self._chips
                        if c.cmap.switch in self.write_switches
                    )
                )
                perf_note = (
                    f"will start on switch {on or 'none'} (writes allowed)"
                    if self.want_perf
                    else "off (need --allow-switch-writes)"
                )
                self._ready = (
                    True,
                    f"link/error/settings read-only; traffic counters {perf_note}",
                )
        return self._ready

    def collect(self) -> list[Sample]:
        if not self.available()[0]:
            return []
        ts = now_ns()
        out: list[Sample] = []
        try:
            for chip in self._chips:
                out.extend(self._collect_chip(chip, ts))
        except AccessError:
            self._reconnect()  # the next collect() starts a fresh helper
            raise
        return out

    def drain_events(self) -> list[Event]:
        ev, self._events = self._events, []
        return ev

    def snapshot(self) -> dict[str, Any]:
        """Last decoded state (after at least one ``collect``): chips and ports."""
        chips = []
        ports = []
        for chip in self._chips:
            chips.append(
                {
                    "switch": chip.cmap.switch,
                    "usp_bdf": chip.cmap.usp_bdf,
                    "serial": chip.serial,
                    "perf_running": chip.perf_running,
                    "perf_ports": list(chip.perf_ports),
                }
            )
            for _, p in sorted(chip.ports.items()):
                d = p.info.to_dict()
                d["state"] = None if p.state is None else asdict(p.state)
                d["perf"] = p.info.port in chip.perf_ports and chip.perf_running
                ports.append(d)
        return {"chips": chips, "ports": ports}

    # -- lifecycle ----------------------------------------------------------------
    def close(self) -> None:
        """Stop our traffic counters and the helper (idempotent)."""
        self._teardown()
        self._ready = None
        self._chips = []

    def _teardown(self) -> None:
        client, self._client = self._client, None
        if client is None or self._injected:
            self._client = client
            return
        for chip in self._chips:
            if chip.perf_running:
                with contextlib.suppress(AccessError):
                    client.perf_stop(chip.bdf)
                chip.perf_running = False
        with contextlib.suppress(OSError):
            client.close()

    def _reconnect(self) -> None:
        self.close()

    # -- initialisation -----------------------------------------------------------
    def _init(self) -> None:
        maps = [
            m
            for m in build_port_maps(self.topo)
            if self.switches is None or m.switch in self.switches
        ]
        if not maps:
            raise AccessError("no PEX88096 (1000:c010) in the topology")
        if self._client is None:
            extra: dict[str, Any] = {}
            root = getattr(self.sysfs, "root", "/sys")
            if root != "/sys":  # --sysfs-root: look for resource0 in that tree
                extra["sysfs"] = f"{root}/bus/pci/devices"
            writable = frozenset(
                full_bdf(m.usp_bdf) for m in maps if m.switch in self.write_switches
            )
            self._client = self._connector(
                {full_bdf(m.usp_bdf): m.usp_port for m in maps},
                allow_writes=writable if self.allow_writes else False,
                **extra,
            )
        self._chips = [self._init_chip(self._client, m) for m in maps]
        if not self._atexit and not self._injected:
            atexit.register(self.close)
            self._atexit = True

    def _init_chip(self, client: PlxClient, cmap: ChipMap) -> _Chip:
        bdf = full_bdf(cmap.usp_bdf)
        wins = client.dump(bdf, [p.port for p in cmap.ports])
        ports: dict[int, _Port] = {}
        headers: dict[int, tuple[int, int]] = {}
        problems: list[str] = []
        for info in cmap.ports:
            win = wins[info.port]
            static = pcie.decode_static(lambda o, w=win: w.get(o, 0))
            headers[info.port] = (win.get(0x08, 0), win.get(0x18, 0))
            if (static.vendor, static.device) != (regs.VENDOR_ID, regs.DEVICE_ID):
                problems.append(f"port {info.port}: window id is not a PEX88096")
            if static.port_number not in (None, info.port):
                problems.append(
                    f"port {info.port} ({info.bdf}): LnkCap port number "
                    f"{static.port_number}"
                )
            tags = {
                "switch": info.switch,
                "plx_port": str(info.port),
                "role": info.kind,
            }
            if info.peer:
                tags["peer"] = info.peer
            if info.hip is not None:
                tags["hip"] = str(info.hip)
            ports[info.port] = _Port(info, static, pcie.dynamic_offsets(static), tags)
        problems += verify_headers(cmap, self.topo, headers)
        if problems:
            raise AccessError(
                f"switch {cmap.switch}: topology and BAR0 windows disagree: "
                + "; ".join(problems[:4])
            )
        plan: list[tuple[int, int]] = []
        slots: list[tuple[int, str]] = []
        for p in ports.values():
            if p.info.kind == KIND_MGMT:
                continue
            names = list(p.offsets)
            station = p.info.station
            if (
                p.info.physical
                and p.info.station_port == 0
                and station is not None
                and station < regs.PERF_STATIONS
            ):
                names += _PERF_STATE_REGS
            for name, off in names:
                plan.append((p.info.port, off))
                slots.append((p.info.port, name))
        usp = ports[cmap.usp_port].static
        return _Chip(cmap, bdf, ports, plan, slots, usp.serial or cmap.serial)

    # -- sampling -------------------------------------------------------------------
    def _collect_chip(self, chip: _Chip, ts: int) -> list[Sample]:
        client = self._client
        assert client is not None
        vals = client.read(chip.bdf, chip.plan)
        per_port: dict[int, dict[str, int]] = {}
        for (port, name), v in zip(chip.slots, vals):
            per_port.setdefault(port, {})[name] = v
        out: list[Sample] = []
        ports_up = 0
        perf_on = False
        for port, p in chip.ports.items():
            if p.info.kind == KIND_MGMT:
                continue
            raw = per_port.get(port, {})
            st = pcie.decode_dynamic(p.static, raw)
            self._watch(chip, p, st, ts)
            p.state = st
            ports_up += p.info.kind in _PHYSICAL_KINDS and st.link_up
            perf_on |= bool(raw.get("perf_ctl", 0) & regs.PERF_BIT_ENABLE)
            out.extend(self._port_samples(ts, p, st))
        sw = switch_entity(chip.cmap.switch)
        tags = {"switch": chip.cmap.switch}
        out += [
            Sample(ts, sw, "plx.ports_up", float(ports_up), "count", dict(tags)),
            Sample(ts, sw, "plx.perf_enabled", float(perf_on), "bool", dict(tags)),
            Sample(
                ts,
                sw,
                "plx.perf_ports",
                float(len(chip.perf_ports)),
                "count",
                dict(tags),
            ),
        ]
        if self.want_perf and not chip.perf_tried:
            self._start_perf(client, chip)
        elif chip.perf_running:
            out.extend(self._perf_samples(client, chip))
        return out

    @staticmethod
    def _port_samples(ts: int, p: _Port, st: pcie.LinkState) -> list[Sample]:
        ent = port_entity(p.info.bdf)
        out: list[Sample] = []

        def add(metric: str, value: float, unit: str) -> None:
            out.append(Sample(ts, ent, metric, float(value), unit, dict(p.tags)))

        if p.info.kind in _PHYSICAL_KINDS:
            add("plx.link_speed_gts", st.speed_gts, "GT/s")
            add("plx.link_width", st.width, "lanes")
            add("plx.link_up", st.link_up, "bool")
            add("plx.link_training", st.training, "bool")
            add("plx.link_degraded", st.degraded, "bool")
            bw = int(st.bw_mgmt_status) | int(st.autonomous_bw_status) << 1
            add("plx.link_bw_status", bw, "mask")
            add("plx.target_speed_gts", st.target_speed_gts, "GT/s")
            if st.eq_complete is not None:
                add("plx.eq_complete", st.eq_complete, "bool")
            if st.lane_err is not None:
                add("plx.lane_err", st.lane_err, "mask")
            if st.dpc_status is not None:
                add("plx.dpc_status", st.dpc_status, "mask")
        add("plx.dev_status", st.dev_status, "mask")
        if st.uncorr_status is not None:
            add("plx.uncorr_status", st.uncorr_status, "mask")
        if st.corr_status is not None:
            add("plx.corr_status", st.corr_status, "mask")
        add("plx.mps_bytes", st.mps, "B")
        add("plx.mrrs_bytes", st.mrrs, "B")
        add("plx.aspm", st.aspm, "mask")
        if st.acs_ctl is not None:
            add("plx.acs_ctl", st.acs_ctl, "mask")
        return out

    # -- events -----------------------------------------------------------------------
    def _event(
        self,
        ts: int,
        ent: str,
        name: str,
        sev: str,
        msg: str,
        data: dict[str, Any],
    ) -> None:
        self._events.append(Event(ts, ent, name, sev, msg, data))

    def _watch(self, chip: _Chip, p: _Port, st: pcie.LinkState, ts: int) -> None:
        """Emit link-change, degraded, DPC and new-sticky-error events."""
        ent = port_entity(p.info.bdf)
        old = p.state
        info = p.info
        base: dict[str, Any] = {"plx_port": info.port, "switch": info.switch}
        if old is not None and info.kind in _PHYSICAL_KINDS:
            before = (old.speed_gts, old.width, old.link_up)
            after = (st.speed_gts, st.width, st.link_up)
            if before != after:
                worse = (
                    (old.link_up and not st.link_up)
                    or st.width < old.width
                    or st.speed_gts < old.speed_gts
                )
                self._event(
                    ts,
                    ent,
                    "link_change",
                    "warn" if worse else "info",
                    f"plx port {info.port}: {_fmt_link(old)} -> {_fmt_link(st)}",
                    {**base, "old": list(before), "new": list(after)},
                )
            if old.degraded != st.degraded:
                self._event(
                    ts,
                    ent,
                    "link_degraded" if st.degraded else "link_restored",
                    "warn" if st.degraded else "info",
                    f"plx port {info.port}: {_fmt_link(st)} (max "
                    f"{st.max_speed_gts:g} GT/s x{st.max_width})",
                    base,
                )
        if st.dpc_status and not (old and old.dpc_status):
            self._event(
                ts,
                ent,
                "dpc_triggered",
                "error",
                f"plx port {info.port}: {pcie.dpc_reason(st.dpc_status)}",
                {**base, "dpc_status": st.dpc_status},
            )
        masks = (
            ("dev_status", st.dev_status, pcie.DEVSTA_BITS),
            ("uncorr_status", st.uncorr_status or 0, pcie.UNCORR_BITS),
            ("corr_status", st.corr_status or 0, pcie.CORR_BITS),
        )
        for key, mask, table in masks:
            prev = chip.sticky.get((info.port, key))
            chip.sticky[(info.port, key)] = mask
            new = mask & ~(prev or 0)
            if not new:
                continue
            at_start = prev is None
            if at_start and info.kind not in _PHYSICAL_KINDS:
                continue  # virtual ports: report changes only
            names = ", ".join(pcie.bit_names(new, table)) or f"{new:#x}"
            fatal = key == "dev_status" and bool(mask & 4)
            self._event(
                ts,
                ent,
                "plx_error_flags",
                "info" if at_start else "error" if fatal else "warn",
                f"plx port {info.port} {key} {'set at start' if at_start else 'new'}: "
                f"{names}",
                {**base, "register": key, "mask": mask, "new": new},
            )

    # -- traffic counters (writes, gated) ---------------------------------------
    def _perf_ports_for(self, chip: _Chip) -> list[int]:
        out = []
        for p in chip.cmap.physical_ports():
            if p.station is None or p.station >= regs.PERF_STATIONS:
                continue
            if p.peer in self.perf_exclude:
                continue
            if self.perf_only is not None and p.port not in self.perf_only:
                continue
            st = chip.ports[p.port].state
            if st is not None and not st.link_up:
                continue
            out.append(p.port)
        return out

    def perf_raw(self) -> dict[str, dict[int, dict[str, Any]]]:
        """Last cumulative counters per switch and port, as the chip reports them.

        ``{switch: {port: {"ts_ns", "rx": {ph, pdw, nph, npdw, cplh, cpldw, dllp},
        "tx": {...}}}}`` (64-bit totals since the counters started; ``rx`` is the
        ingress side of the port).  Empty until the counters ran once.
        """
        out: dict[str, dict[int, dict[str, Any]]] = {}
        for chip in self._chips:
            if not chip.perf_running:
                continue
            out[chip.cmap.switch] = {
                port: {"ts_ns": ts, "rx": asdict(ing), "tx": asdict(egr)}
                for port, (ts, ing, egr) in sorted(chip.prev_perf.items())
            }
        return out

    def read_registers(
        self, switch: str, items: Sequence[tuple[int, int]]
    ) -> list[int]:
        """Dwords of ``[(port, offset)]`` of one switch (read-only; the four
        counter-RAM offsets stay refused by the helper)."""
        ok, why = self.available()
        if not ok or self._client is None:
            raise AccessError(why)
        for chip in self._chips:
            if chip.cmap.switch == switch:
                return self._client.read(chip.bdf, items)
        raise AccessError(f"switch {switch} is not read by this collector")

    def _start_perf(self, client: PlxClient, chip: _Chip) -> None:
        chip.perf_tried = True  # one attempt per connection, errors are events
        sw = switch_entity(chip.cmap.switch)
        if chip.cmap.switch not in self.write_switches:
            approved = ",".join(sorted(self.write_switches)) or "none"
            self._event(
                now_ns(),
                sw,
                "plx_perf_skipped",
                "info",
                f"traffic counters stay off on switch {chip.cmap.switch}: register "
                f"writes are approved for switch {approved} only "
                f"({policy.ENV_WRITE_SWITCHES} overrides)",
                {"approved": sorted(self.write_switches)},
            )
            return
        # the topology says which ports host a never-written GPU: the root helper
        # must refuse their stations on its own, else this table is stale
        guarded = set(
            client.hello()["chips"].get(chip.bdf, {}).get("protected_ports", ())
        )
        stray = [
            p.port
            for p in chip.cmap.physical_ports()
            if p.peer in policy.NEVER_WRITTEN_GPUS and p.port not in guarded
        ]
        if stray:
            self._event(
                now_ns(),
                sw,
                "plx_perf_error",
                "error",
                f"traffic counters not started: port(s) {stray} host a GPU that is "
                "never written but the helper does not protect them (stale "
                "KNOWN_CHIPS / PROTECTED_STATIONS in plx/regs.py?)",
                {"ports": stray},
            )
            return
        ports = self._perf_ports_for(chip)
        if not ports:
            return
        try:
            info = client.perf_start(chip.bdf, ports)
        except AccessError as exc:
            self._event(
                now_ns(),
                sw,
                "plx_perf_error",
                "error",
                f"cannot start traffic counters: {exc}",
                {"ports": ports},
            )
            return
        chip.perf_ports = list(ports)
        chip.perf_running = True
        owned = bool(info.get("owned", True))
        self._event(
            now_ns(),
            sw,
            "plx_perf_start",
            "info",
            f"traffic counters on ports {ports}"
            + ("" if owned else " (monitor already running, not ours)"),
            {"ports": ports, "owned": owned},
        )

    def _perf_samples(self, client: PlxClient, chip: _Chip) -> list[Sample]:
        tot = client.perf_totals(chip.bdf)
        sw = switch_entity(chip.cmap.switch)
        if not tot.get("running"):
            return []
        if tot.get("error"):
            chip.perf_running = False
            self._event(
                now_ns(),
                sw,
                "plx_perf_error",
                "error",
                f"counter sampler stopped: {tot['error']}",
                {},
            )
            return []
        ts = int(tot["ts_ns"])
        if ts <= chip.perf_ts_ns:  # no new capture since the last collect
            return []
        chip.perf_ts_ns = ts
        out: list[Sample] = []
        totals: Mapping[str, Sequence[int]] = tot["totals"]
        for port in chip.perf_ports:
            p = chip.ports[port]
            ing, egr = perf.decode_port(
                totals[str(p.info.station)], p.info.station_port or 0
            )
            ent = port_entity(p.info.bdf)
            prev = chip.prev_perf.get(port)
            chip.prev_perf[port] = (ts, ing, egr)
            for pre, cur in (("rx", ing), ("tx", egr)):
                for metric, value, unit in (
                    (f"port.{pre}_bytes", cur.total_bytes(), "B"),
                    (f"plx.{pre}_payload_bytes", cur.payload_bytes(), "B"),
                    (f"plx.{pre}_tlps", cur.tlps(), "count"),
                    (f"plx.{pre}_dllps", cur.dllp, "count"),
                ):
                    out.append(
                        Sample(ts, ent, metric, float(value), unit, dict(p.tags))
                    )
            st = p.state
            if prev is None or st is None:
                continue
            dt = (ts - prev[0]) / 1e9
            cap = raw_gbs(st.speed_gts, st.width) if st.width else 0.0
            for pre, a, b in (("rx", prev[1], ing), ("tx", prev[2], egr)):
                r = perf.rates(a, b, dt, cap)
                for metric, value, unit in (
                    (f"port.{pre}_gbs", r.total_gbs, "GB/s"),
                    (f"plx.{pre}_payload_gbs", r.payload_gbs, "GB/s"),
                    (f"plx.{pre}_util", r.util_pct, "%"),
                ):
                    out.append(Sample(ts, ent, metric, value, unit, dict(p.tags)))
        return out


def _fmt_link(st: pcie.LinkState) -> str:
    return f"{st.speed_gts:g} GT/s x{st.width}" + ("" if st.link_up else " (down)")


def make(topo: Topology, *, sysfs: Sysfs | None = None, **opts: Any) -> PlxCollector:
    """Build the collector for ``topo`` (registry entry point)."""
    return PlxCollector(topo, sysfs, **opts)
