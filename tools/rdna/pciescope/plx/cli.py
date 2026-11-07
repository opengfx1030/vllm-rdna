# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``python -m pciescope.plx``: look at the PEX88096 switches.

Verbs (all read-only except ``perf run``, which needs ``--allow-switch-writes``):

``ports``      PEX port number <-> BDF <-> GPU map (no privileges)
``status``     link, equalization, error flags and settings of every port
``dump``       raw non-zero dwords of one port window (``--decode`` names caps)
``perf plan``  the exact register writes the traffic counters need, with the
               current values and the values they would get (writes nothing)
``perf run``   start the counters, print per-port rates for ``--duration`` seconds,
               stop them again (writes registers: gated)
``capture``    record the topology and every port window into a fixture file

``--fixture FILE`` replays a capture instead of touching the hardware; every verb
accepts ``--json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from typing import Any

from pciescope.model import Topology, full_bdf
from pciescope.plx import access, fixture, pcie, regs
from pciescope.plx.ports import ChipMap, build_port_maps

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
        lines.append("  port stn bdf      kind     peer")
        for p in m.ports:
            stn = "-" if p.station is None else f"{p.station}.{p.station_port}"
            lines.append(f"  {p.port:4d} {stn:>3} {p.bdf:8} {p.kind:8} {p.peer or ''}")
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
            print("  port bdf      kind     peer     link            eq  flags")
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
            f"{link:15} {eq:3} {_flags(st)}"
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
def _perf_ports(cm: ChipMap, exclude: Sequence[str]) -> list[int]:
    return [
        p.port
        for p in cm.physical_ports()
        if p.station is not None
        and p.station < regs.PERF_STATIONS
        and p.peer not in exclude
    ]


def cmd_perf_plan(args: argparse.Namespace) -> int:
    topo, preset = _topology(args)
    maps = _pick(build_port_maps(topo), args.switch)
    client = _client(args, maps, preset)
    doc = {}
    lines = []
    try:
        for cm in maps:
            ports = _perf_ports(cm, args.exclude)
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


def cmd_perf_run(args: argparse.Namespace) -> int:
    if not args.allow_switch_writes:
        print(WRITE_REFUSAL, file=sys.stderr)
        return 2
    if args.fixture:
        print("perf run needs the real hardware (no --fixture)", file=sys.stderr)
        return 2
    from pciescope.collectors.plx import PlxCollector

    topo, _ = _topology(args)
    coll = PlxCollector(
        topo, allow_writes=True, perf=True, perf_exclude=tuple(args.exclude)
    )
    ok, why = coll.available()
    if not ok:
        print(f"plx unavailable: {why}", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    try:
        end = time.monotonic() + args.duration
        while True:
            samples = coll.collect()
            for e in coll.drain_events():
                print(f"event {e.name} {e.entity}: {e.message}", file=sys.stderr)
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
                    print(
                        f"t={time.strftime('%H:%M:%S')}  (GB/s: total / payload est.)"
                    )
                    for ent in sorted({e for e, _ in rates}):
                        r = {m: v for (e, m), v in rates.items() if e == ent}
                        print(
                            f"  {ent:14} rx {r.get('port.rx_gbs', 0):6.3f} / "
                            f"{r.get('plx.rx_payload_gbs', 0):6.3f} "
                            f"{r.get('plx.rx_util', 0):5.1f}%   "
                            f"tx {r.get('port.tx_gbs', 0):6.3f} / "
                            f"{r.get('plx.tx_payload_gbs', 0):6.3f} "
                            f"{r.get('plx.tx_util', 0):5.1f}%"
                        )
            if time.monotonic() >= end:
                break
            time.sleep(args.interval)
    finally:
        coll.close()  # stops the counters
    if args.json:
        print(json.dumps(rows, indent=1))
    return 0


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
        default=["gpu:07"],
        metavar="gpu:BUS",
        help="leave this GPU's switch port alone (default: gpu:07, HIP 5)",
    )
    pl = ps.add_parser("plan", parents=[common, pexcl], help="show the register writes")
    pl.set_defaults(fn=cmd_perf_plan)
    r = ps.add_parser("run", parents=[common, pexcl], help="measure (writes registers)")
    r.add_argument("--allow-switch-writes", action="store_true")
    r.add_argument("--duration", type=float, default=10.0)
    r.add_argument("--interval", type=float, default=1.0)
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
