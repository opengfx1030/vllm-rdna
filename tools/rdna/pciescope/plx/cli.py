# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``python -m pciescope.plx``: look at the PEX88096 switches.

Verbs (all read-only except ``perf run``, which needs ``--allow-switch-writes``):

``ports``      PEX port number <-> BDF <-> GPU map (no privileges)
``status``     link, equalization, error flags and settings of every port
``dump``       raw non-zero dwords of one port window (``--decode`` names caps)
``perf plan``  the exact register writes the traffic counters need, with the
               current values and the values they would get (writes nothing)
``perf preflight``  is the switch idle and its lease held? (reads only)
``perf run``   one switch: start the counters (optionally on ``--ports`` only),
               print per-port rates for ``--duration`` seconds, stop them again,
               then verify the undo values and diff the link/error state
               (writes registers: gated by ``--allow-switch-writes`` and by the
               approved switches of ``pciescope.plx.policy``)
``capture``    record the topology and every port window into a fixture file

``--fixture FILE`` replays a capture instead of touching the hardware; every verb
accepts ``--json``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from pciescope.model import Topology, full_bdf
from pciescope.plx import access, fixture, health, pcie, policy, preflight, regs
from pciescope.plx.ports import ChipMap, build_port_maps
from pciescope.sysfs import Sysfs

WRITE_REFUSAL = (
    "refused: this verb writes switch registers (PerformancePAK enable). Re-run "
    "with --allow-switch-writes after the write list shown by `perf plan` was "
    "approved."
)


def _topology(args: argparse.Namespace) -> tuple[Topology, access.PlxClient | None]:
    if args.fixture:
        topo, client = fixture.load(args.fixture)
        return topo, client
    from pciescope.topology import discover

    return discover(), None


def _pick(maps: list[ChipMap], switch: str | None) -> list[ChipMap]:
    if switch is None:
        return maps
    sel = [m for m in maps if m.switch == switch.upper()]
    if not sel:
        raise SystemExit(f"no switch {switch!r} (have {[m.switch for m in maps]})")
    return sel


def _client(
    args: argparse.Namespace, maps: list[ChipMap], preset: access.PlxClient | None
) -> access.PlxClient:
    if preset is not None:
        return preset
    return access.connect(
        {full_bdf(m.usp_bdf): m.usp_port for m in maps},
        allow_writes=getattr(args, "allow_switch_writes", False),
    )


def _print(args: argparse.Namespace, doc: Any, text: str) -> None:
    print(json.dumps(doc, indent=1) if args.json else text)


def _dash(v: object) -> str:
    return "-" if v is None else str(v)


# -- ports -------------------------------------------------------------------------
def cmd_ports(args: argparse.Namespace) -> int:
    topo, _ = _topology(args)
    maps = _pick(build_port_maps(topo), args.switch)
    lines = []
    for m in maps:
        lines.append(
            f"switch {m.switch}  root {m.root_bus}  upstream {m.usp_bdf} "
            f"(port {m.usp_port})  serial {m.serial or '?'}"
        )
        lines.append("  port stn bdf      kind     peer     hip smi slot lease    gpu")
        for p in m.ports:
            stn = "-" if p.station is None else f"{p.station}.{p.station_port}"
            lines.append(
                f"  {p.port:4d} {stn:>3} {p.bdf:8} {p.kind:8} {p.peer or '':8} "
                f"{_dash(p.hip):>3} {_dash(p.smi):>3} {_dash(p.slot):>4} "
                f"{_dash(p.lease):8} {p.gpu_name or ''}".rstrip()
            )
    _print(args, [m.to_dict() for m in maps], "\n".join(lines))
    return 0


# -- status ------------------------------------------------------------------------
def _flags(st: dict[str, Any]) -> str:
    out = pcie.bit_names(st["dev_status"], pcie.DEVSTA_BITS)
    if st.get("uncorr_status"):
        names = pcie.bit_names(st["uncorr_status"], pcie.UNCORR_BITS)
        out += [f"uncorr:{n}" for n in names]
    if st.get("corr_status"):
        out += [f"corr:{n}" for n in pcie.bit_names(st["corr_status"], pcie.CORR_BITS)]
    if st.get("dpc_status"):
        out.append("DPC")
    return ",".join(out) or "-"


def _event_switch(e: Any) -> str | None:
    """The switch letter an event belongs to (port events tag it, others name it)."""
    if "switch" in e.data:
        return str(e.data["switch"])
    return e.entity[len("switch:") :] if e.entity.startswith("switch:") else None


def cmd_status(args: argparse.Namespace) -> int:
    from pciescope.collectors.plx import PlxCollector

    topo, preset = _topology(args)
    coll = PlxCollector(topo, client=preset)
    ok, why = coll.available()
    if not ok:
        print(f"plx unavailable: {why}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    samples = coll.collect()
    ms = (time.perf_counter() - t0) * 1e3
    events = coll.drain_events()
    snap = coll.snapshot()
    coll.close()
    if args.switch:
        want = args.switch.upper()
        known = [c["switch"] for c in snap["chips"]]
        if want not in known:
            raise SystemExit(f"no switch {args.switch!r} (have {known})")
        snap = {
            "chips": [c for c in snap["chips"] if c["switch"] == want],
            "ports": [p for p in snap["ports"] if p["switch"] == want],
        }
        events = [e for e in events if _event_switch(e) in (None, want)]
    if args.json:
        doc = {
            **snap,
            "events": [e.to_dict() for e in events],
            "collect_ms": round(ms, 3),
            "samples": len(samples),
        }
        print(json.dumps(doc, indent=1))
        return 0
    chips = {c["switch"]: c for c in snap["chips"]}
    cur = None
    for d in snap["ports"]:
        if d["switch"] != cur:
            cur = d["switch"]
            c = chips[cur]
            print(
                f"switch {cur}  upstream {c['usp_bdf']}  serial {c['serial'] or '?'}  "
                f"traffic counters {'on' if c['perf_running'] else 'off'}"
            )
            print("  port bdf      kind     peer     hip link            eq  flags")
        st = d["state"]
        if st is None:
            continue
        if d["kind"] in ("usp", "dsp"):
            link = (
                f"{st['speed_gts']:g} GT/s x{st['width']}" if st["link_up"] else "down"
            )
            eq = {None: "-", True: "ok", False: "NO"}[st["eq_complete"]]
        else:
            link, eq = "virtual", "-"
        print(
            f"  {d['port']:4d} {d['bdf']:8} {d['kind']:8} {d['peer'] or '':8} "
            f"{_dash(d['hip']):>3} {link:15} {eq:3} {_flags(st)}"
        )
    print(f"\n{len(samples)} samples in {ms:.2f} ms")
    for e in events:
        print(f"event {e.name} {e.entity}: {e.message}")
    return 0


# -- dump --------------------------------------------------------------------------
def cmd_dump(args: argparse.Namespace) -> int:
    topo, preset = _topology(args)
    maps = _pick(build_port_maps(topo), args.switch)
    cm = maps[0]
    by_port = cm.by_port()
    if args.port not in by_port:
        raise SystemExit(f"switch {cm.switch} has no port {args.port}")
    client = _client(args, maps, preset)
    try:
        bdf = full_bdf(cm.usp_bdf)
        win = client.dump(bdf, [args.port])[args.port]
    finally:
        client.close()
    caps: dict[str, int] = {}
    if args.decode:
        caps = pcie.walk_caps(lambda o: win.get(o, 0)).names()
    doc = {
        "switch": cm.switch,
        "port": args.port,
        "bdf": by_port[args.port].bdf,
        "caps": caps,
        "dwords": {f"{o:03x}": f"{v:08x}" for o, v in sorted(win.items())},
    }
    text = [f"switch {cm.switch} port {args.port} ({by_port[args.port].bdf})"]
    if caps:
        text.append("caps: " + " ".join(f"{n}@{o:03x}" for n, o in caps.items()))
    text += [f"{o:03x}: {v:08x}" for o, v in sorted(win.items())]
    _print(args, doc, "\n".join(text))
    return 0


# -- perf --------------------------------------------------------------------------
def _perf_ports(
    cm: ChipMap, exclude: Sequence[str], only: str | None = None
) -> list[int]:
    """Ports whose counters would run; ``only`` ("16,32") narrows the default set.

    The default set is every physical port with a device behind it (a GPU or the
    host); empty slots have no traffic and their stations stay untouched.
    """
    ports = [
        p.port
        for p in cm.physical_ports()
        if p.station is not None
        and p.station < regs.PERF_STATIONS
        and p.peer is not None
        and p.peer not in (*exclude, *policy.NEVER_WRITTEN_GPUS)
    ]
    if only:
        want = sorted({int(x) for x in only.split(",") if x.strip()})
        bad = sorted(set(want) - set(ports))
        if bad:
            raise SystemExit(
                f"--ports {bad}: not measurable on switch {cm.switch} (default set "
                f"{ports}; the excluded GPU's port can never be added)"
            )
        ports = want
    return ports


def _perf_registers(ports: Sequence[int]) -> list[tuple[int, int]]:
    """The registers of the counter sequence as ``(port, offset)`` pairs."""
    items: list[tuple[int, int]] = []
    for s in sorted({regs.station_of(p) for p in ports}):
        base = regs.station_base_port(s)
        items += [
            (base, regs.REG_PERF_CTRL),
            (base, regs.REG_PERF_RAM_CTRL),
            (base, regs.REG_PERF_INGRESS),
        ]
    return items + [(p, regs.REG_PERF_EGRESS) for p in ports]


def _reg_rows(items: Sequence[tuple[int, int]], vals: Sequence[int]) -> dict[str, int]:
    return {f"{off:#05x}@{port}": v for (port, off), v in zip(items, vals)}


def cmd_perf_plan(args: argparse.Namespace) -> int:
    topo, preset = _topology(args)
    maps = _pick(build_port_maps(topo), args.switch)
    client = _client(args, maps, preset)
    doc = {}
    lines = []
    try:
        for cm in maps:
            ports = _perf_ports(cm, args.exclude, args.ports)
            plan = client.perf_plan(full_bdf(cm.usp_bdf), ports)
            doc[cm.switch] = {"ports": ports, **plan}
            lines.append(
                f"switch {cm.switch}: counters on ports {ports} "
                f"(monitor already running: {plan['monitor_already_running']})"
            )
            lines.append("  step              port  BAR0 offset  reg    old       new")
            for s in plan["steps"]:
                lines.append(
                    f"  {s['step']:17} {s['port']:4d}  {s['bar0']:#010x}  "
                    f"{s['reg']:#05x}  {s['old']:08x}  {s['new']:08x}  {s['note']}"
                )
            for s in plan["per_poll"]:
                lines.append(
                    f"  {s['step']:17} {s['port']:4d}  {s['bar0']:#010x}  "
                    f"{s['reg']:#05x}  {'':8}  {s['new']:08x}  {s['note']}"
                )
    finally:
        client.close()
    _print(args, doc, "\n".join(lines))
    return 0


_ANOMALY_EVENTS = ("link_change", "link_degraded", "dpc_triggered", "plx_perf_error")

#: the link events a runtime-PM sleep or wake of an endpoint produces
_PM_LINK_EVENTS = ("link_change", "link_degraded", "link_restored")


def _pm_ports(
    topo: Topology, cm: ChipMap, sysfs: Sysfs | None = None
) -> dict[int, str]:
    """``{PEX port: bdf}`` of the ports whose link may change by runtime PM alone.

    A GPU that the kernel may runtime-suspend sleeps, its link dropping to 2.5 GT/s
    x1, whenever no process holds it, and wakes to its full link when one touches it
    (any HIP/torch start does, ``HIP_VISIBLE_DEVICES`` or not: every ROCm process
    enumerates all KFD nodes).  That is the kernel's runtime power management, not a
    fault (the box rules say so).  A port counts only while the kernel has runtime PM
    enabled for its GPU (``power/control`` = ``auto``): the V620s of the leases stay
    on (``on``), so a link change there is never excused.
    """
    sysfs = sysfs or Sysfs()
    out: dict[int, str] = {}
    for g in topo.gpus.values():
        if g.switch != cm.switch:
            continue
        control = sysfs.read(f"bus/pci/devices/{full_bdf(g.bdf)}/power/control")
        if control != "auto":
            continue
        out.update({p.port: p.bdf for p in cm.ports if p.peer == g.id})
    return out


def _is_runtime_pm(e: Any, pm_ports: Mapping[int, str]) -> bool:
    """A link event of a runtime-PM port whose link stayed up (nothing else changed:
    a DPC, a new error flag or a link going down is still an anomaly)."""
    if e.name not in _PM_LINK_EVENTS or e.data.get("plx_port") not in pm_ports:
        return False
    new = e.data.get("new")
    return e.name != "link_change" or bool(new and new[2])


def _is_anomaly(e: Any, pm_ports: Mapping[int, str] | None = None) -> bool:
    """An event that ends a counter run: link change, DPC, new error, perf error.

    Except the link events of a port in ``pm_ports`` (see ``_pm_ports``)."""
    if pm_ports and _is_runtime_pm(e, pm_ports):
        return False
    return e.name in _ANOMALY_EVENTS or (
        e.name == "plx_error_flags" and e.severity != "info"
    )


def _raise_interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def _read_only_view(topo: Topology, switch: str) -> Any:
    from pciescope.collectors.plx import PlxCollector

    rd = PlxCollector(topo, switches=[switch])
    ok, why = rd.available()
    if not ok:
        raise access.AccessError(why)
    rd.collect()
    return rd


def cmd_perf_preflight(args: argparse.Namespace) -> int:
    from pciescope.probe import lease

    if args.fixture or not args.switch:
        print("perf preflight needs --switch A|B and the real box", file=sys.stderr)
        return 2
    topo, _ = _topology(args)
    cm = _pick(build_port_maps(topo), args.switch)[0]
    report = preflight.check_idle(topo, cm.switch, lease.resolve_owner(args.owner))
    _print(args, report.to_dict(), report.text())
    return 0 if report.ok else 3


def cmd_perf_run(args: argparse.Namespace) -> int:
    if not args.allow_switch_writes:
        print(WRITE_REFUSAL, file=sys.stderr)
        return 2
    if args.fixture:
        print("perf run needs the real hardware (no --fixture)", file=sys.stderr)
        return 2
    if not args.switch:
        print("perf run needs --switch A|B (one switch at a time)", file=sys.stderr)
        return 2
    from pciescope.collectors.plx import PlxCollector
    from pciescope.probe import lease

    topo, _ = _topology(args)
    cm = _pick(build_port_maps(topo), args.switch)[0]
    approved = policy.write_switches()
    if cm.switch not in approved:
        print(
            f"refused: register writes are approved for switch "
            f"{','.join(sorted(approved)) or 'none'} only, not {cm.switch} "
            f"({policy.ENV_WRITE_SWITCHES} overrides it, with the lead's go)",
            file=sys.stderr,
        )
        return 2
    ports = _perf_ports(cm, args.exclude, args.ports)
    if not ports:
        print(f"switch {cm.switch}: no port to measure", file=sys.stderr)
        return 2
    log = sys.stderr if args.json else sys.stdout
    if args.require_idle:
        report = preflight.check_idle(topo, cm.switch, lease.resolve_owner(args.owner))
        print(report.text(), file=log)
        if not report.ok:
            return 3
    items = _perf_registers(ports)
    pm = _pm_ports(topo, cm)
    rd = _read_only_view(topo, cm.switch)
    before = health.capture(rd.snapshot(), topo, cm.switch)
    regs_before = _reg_rows(items, rd.read_registers(cm.switch, items))
    rd.close()

    coll = PlxCollector(
        topo,
        allow_writes=True,
        perf=True,
        perf_exclude=tuple(args.exclude),
        perf_ports=ports,
        write_switches=[cm.switch],
        switches=[cm.switch],
    )
    ok, why = coll.available()
    if not ok:
        print(f"plx unavailable: {why}", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    pm_events: list[dict[str, Any]] = []
    anomaly: Any = None
    failure: access.AccessError | None = None
    interrupted = False
    regs_during: dict[str, int] = {}
    last_ts = 0
    raw = open(args.raw, "w") if args.raw else None  # noqa: SIM115 (closed below)
    old_term = signal.signal(signal.SIGTERM, _raise_interrupt)
    note = (
        f"; the runtime-PM link changes of port(s) {sorted(pm)} are logged only"
        if pm
        else ""
    )
    print(
        f"switch {cm.switch}: counters on ports {ports} for {args.duration:g} s "
        f"(stop: Ctrl-C or SIGTERM; any link/error event ends the run{note})",
        file=log,
    )
    try:
        end = time.monotonic() + args.duration
        while True:
            samples = coll.collect()
            for e in coll.drain_events():
                excused = _is_runtime_pm(e, pm)
                tag = "  [runtime PM of the port, not an anomaly]" if excused else ""
                print(f"event {e.name} {e.entity}: {e.message}{tag}", file=sys.stderr)
                if excused:
                    pm_events.append(e.to_dict())
                if anomaly is None and _is_anomaly(e, pm):
                    anomaly = e
            if not regs_during and coll.snapshot()["chips"][0]["perf_running"]:
                regs_during = _reg_rows(items, coll.read_registers(cm.switch, items))
            rates = {
                (s.entity, s.metric): s.value
                for s in samples
                if s.metric.endswith(("_gbs", "_util"))
            }
            if rates:
                rows.append(
                    {
                        "ts": time.time(),
                        "rates": {f"{e} {m}": v for (e, m), v in rates.items()},
                    }
                )
                if not args.json:
                    _print_rates(rates)
            counters = coll.perf_raw().get(cm.switch, {})
            ts = max((v["ts_ns"] for v in counters.values()), default=0)
            if raw is not None and ts > last_ts:
                last_ts = ts
                raw.write(
                    json.dumps(
                        {"t": time.time(), "switch": cm.switch, "counters": counters}
                    )
                    + "\n"
                )
            if anomaly is not None or time.monotonic() >= end:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        interrupted = True
    except access.AccessError as exc:
        failure = exc
    finally:
        coll.close()  # stops the counters
        signal.signal(signal.SIGTERM, old_term)
        if raw is not None:
            raw.close()

    rd = _read_only_view(topo, cm.switch)
    after = health.capture(rd.snapshot(), topo, cm.switch)
    regs_after = _reg_rows(items, rd.read_registers(cm.switch, items))
    rd.close()
    changes = health.diff(before, after, pm_bdfs=pm.values())
    undo = _undo_mismatch(regs_before, regs_after)
    verdict = {
        "switch": cm.switch,
        "ports": ports,
        "registers": {
            k: {
                "before": regs_before[k],
                "during": regs_during.get(k),
                "after": regs_after[k],
            }
            for k in regs_before
        },
        "undo_mismatch": undo,
        "health_changes": changes,
        "runtime_pm": {
            "ports": {str(p): bdf for p, bdf in pm.items()},
            "events": pm_events,
            "changes": health.pm_diff(before, after, pm.values()),
        },
        "anomaly": None if anomaly is None else anomaly.to_dict(),
        "interrupted": interrupted,
        "error": None if failure is None else str(failure),
    }
    if args.json:
        print(json.dumps({**verdict, "rows": rows}, indent=1))
    else:
        _print_verdict(verdict)
    if failure is not None:
        return 2
    if anomaly is not None:
        return 4
    return 5 if undo or changes else 0


def _print_rates(rates: dict[tuple[str, str], float]) -> None:
    print(f"t={time.strftime('%H:%M:%S')}  (GB/s: total / payload est.)")
    for ent in sorted({e for e, _ in rates}):
        r = {m: v for (e, m), v in rates.items() if e == ent}
        print(
            f"  {ent:14} rx {r.get('port.rx_gbs', 0):6.3f} / "
            f"{r.get('plx.rx_payload_gbs', 0):6.3f} {r.get('plx.rx_util', 0):5.1f}%   "
            f"tx {r.get('port.tx_gbs', 0):6.3f} / "
            f"{r.get('plx.tx_payload_gbs', 0):6.3f} {r.get('plx.tx_util', 0):5.1f}%"
        )


def _undo_mismatch(before: dict[str, int], after: dict[str, int]) -> list[str]:
    """Registers that did not return to their pre-run value.

    Two are exempt by design: ``0x3E0`` may read the SDK's stop value
    (``PERF_CTRL_STOP``) instead of its old one, and ``0x3F0`` is the capture
    strobe every poll rewrites.
    """
    out = []
    for key, was in before.items():
        now = after[key]
        off = int(key.partition("@")[0], 16)
        if off == regs.REG_PERF_RAM_CTRL or now == was:
            continue
        if off == regs.REG_PERF_CTRL and now == regs.PERF_CTRL_STOP:
            continue
        out.append(f"{key}: {was:#010x} -> {now:#010x}")
    return out


def _print_verdict(v: dict[str, Any]) -> None:
    print(f"\nregisters of switch {v['switch']} (before | during | after):")
    for k, r in v["registers"].items():
        during = "-" if r["during"] is None else f"{r['during']:08x}"
        print(f"  {k:10} {r['before']:08x} | {during} | {r['after']:08x}")
    print(
        "  (0x3e0 after a run reads the SDK stop value or the old one; 0x3f0 is the "
        "capture strobe of every poll; bit 21 of 0xf30 always reads 0)"
    )
    restored = "0x768 and 0xf30 read back as before"
    print("undo: " + ("NOT RESTORED" if v["undo_mismatch"] else restored))
    for line in v["undo_mismatch"]:
        print(f"  DIFFERS {line}")
    print("health: " + ("no change" if not v["health_changes"] else "CHANGED"))
    for line in v["health_changes"]:
        print(f"  {line}")
    pm = v["runtime_pm"]
    if pm["events"] or pm["changes"]:
        print(
            "runtime PM of a port with power/control=auto (kernel power management, "
            f"not an error; {len(pm['events'])} link event(s)):"
        )
        for line in pm["changes"]:
            print(f"  {line}")
    if v["anomaly"]:
        print(
            f"ANOMALY ended the run: {v['anomaly']['name']}: {v['anomaly']['message']}"
        )
    if v["interrupted"]:
        print("interrupted; counters stopped")
    if v["error"]:
        print(f"ERROR {v['error']}")


# -- capture -----------------------------------------------------------------------
def cmd_capture(args: argparse.Namespace) -> int:
    topo, preset = _topology(args)
    maps = build_port_maps(topo)
    client = _client(args, maps, preset)
    try:
        doc = fixture.capture(topo, client, maps)
    finally:
        client.close()
    fixture.save(doc, args.out)
    print(f"wrote {args.out}: {len(maps)} switches")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pciescope-plx", description=__doc__.split("\n\n")[0]
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fixture", metavar="FILE", help="replay a capture")
    common.add_argument("--json", action="store_true")
    common.add_argument("--switch", metavar="A|B", help="only this switch")
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser(
        "ports", parents=[common], help="port number / BDF / GPU map"
    ).set_defaults(fn=cmd_ports)
    sub.add_parser(
        "status", parents=[common], help="link and error state"
    ).set_defaults(fn=cmd_status)
    d = sub.add_parser("dump", parents=[common], help="raw window of one port")
    d.add_argument("port", type=int)
    d.add_argument("--decode", action="store_true", help="name the capabilities")
    d.set_defaults(fn=cmd_dump)
    p = sub.add_parser("perf", help="traffic counters")
    ps = p.add_subparsers(dest="perf_verb", required=True)
    pexcl = argparse.ArgumentParser(add_help=False)
    pexcl.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="gpu:BUS",
        help="also leave this GPU's switch port alone (policy.NEVER_WRITTEN_GPUS is "
        "always excluded)",
    )
    pexcl.add_argument(
        "--ports",
        metavar="N,N",
        help="only these PEX port numbers (staged start), e.g. 16 then 16,32,64,80",
    )
    pl = ps.add_parser("plan", parents=[common, pexcl], help="show the register writes")
    pl.set_defaults(fn=cmd_perf_plan)
    pf = ps.add_parser(
        "preflight", parents=[common], help="is the switch idle and leased? (reads)"
    )
    pf.add_argument("--owner", help="lease owner (default $PCIESCOPE_OWNER)")
    pf.set_defaults(fn=cmd_perf_preflight)
    r = ps.add_parser("run", parents=[common, pexcl], help="measure (writes registers)")
    r.add_argument("--allow-switch-writes", action="store_true")
    r.add_argument("--duration", type=float, default=10.0)
    r.add_argument("--interval", type=float, default=1.0)
    r.add_argument(
        "--require-idle",
        action="store_true",
        help="refuse (exit 3) unless `perf preflight` passes for the switch",
    )
    r.add_argument("--owner", help="lease owner for --require-idle")
    r.add_argument(
        "--raw", metavar="FILE", help="JSONL of the cumulative raw counters per tick"
    )
    r.set_defaults(fn=cmd_perf_run)
    c = sub.add_parser("capture", parents=[common], help="record a fixture")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_capture)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except access.AccessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
