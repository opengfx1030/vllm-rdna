# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test double of ``p2p_probe`` for the runner and API tests (no GPU).

SYNTHETIC: it speaks the batch protocol of the real probe (``READY``, one
``###`` echo, the result rows and ``END`` per command) with canned numbers. The
numbers are not measurements; the double exists to test the mechanics of the
runner (protocol, guards, time limits, event order) on a machine without GPUs.

Configuration: a JSON file named by ``$FAKE_PROBE_SPEC``:

``buses``        bus list printed by ``buses`` and counted by ``READY``
``version``      version string (default ``fake/1``)
``modes``        mode list printed by ``version`` (default: every mode)
``gbps``         bandwidth printed for every flow (default 20.0)
``lat_us``       one-way latency printed for every pair (default 6.5)
``verify_fail``  print ``verify=FAIL`` in every bandwidth row
``log``          file that receives argv, ``HIP_VISIBLE_DEVICES`` and every command
``no_ready``     never print ``READY`` (sleeps)
``stderr``       line written to stderr at start
``events``       list of ``{"at": k, "do": ...}`` applied when the k-th command
                 arrives (1-based, before its output): ``write`` (``path``,
                 ``text``), ``sleep`` (``s``), ``exit`` (``rc``) and ``err``
                 (``text``, ``rc``: the command fails with that ``ERR`` line)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SPEC: dict[str, Any] = {}
DEFAULT_MODES = "version,buses,batch,bw,hbw,lat,latc,a2a,hlat,hflag,curve"


def out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def note(text: str) -> None:
    log = SPEC.get("log")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def buses() -> list[str]:
    return [str(b) for b in SPEC.get("buses", ["0a", "0f", "14", "17"])]


def version() -> str:
    return str(SPEC.get("version", "fake/1"))


def stats_tail(prefix: str, value: float) -> str:
    return (
        f" {prefix}min_us={value * 0.9:.3f} {prefix}p50_us={value:.3f}"
        f" {prefix}p99_us={value * 1.5:.3f} {prefix}max_us={value * 3:.3f}"
        f" {prefix}mean_us={value * 1.05:.3f}"
    )


def rows_bw(tok: list[str]) -> list[str]:
    size, blocks = tok[2], tok[3]
    gbps = float(SPEC.get("gbps", 20.0))
    verify = "FAIL" if SPEC.get("verify_fail") else "ok"
    fg = bg = 0.0
    lines = []
    for label in tok[4:]:
        lines.append(
            f"X {label} size={size} blocks={blocks} gbps={gbps:.2f} verify={verify}"
        )
        if label.startswith("bg:"):
            bg += gbps
        else:
            fg += gbps
    lines.append(f"AGG size={size} fg_gbps={fg:.2f} bg_gbps={bg:.2f} n={len(tok) - 4}")
    return lines


def rows_hbw(tok: list[str]) -> list[str]:
    size, blocks = tok[2], tok[3]
    gbps = float(SPEC.get("gbps", 20.0))
    fg = bg = 0.0
    lines = []
    for label in tok[4:]:
        lines.append(
            f"HBW {label} size={size} blocks={blocks} gbps={gbps:.2f} "
            "verify=ok secs=0.400"
        )
        if label.startswith("bg:"):
            bg += gbps
        else:
            fg += gbps
    lines.append(f"HAGG size={size} fg_gbps={fg:.2f} bg_gbps={bg:.2f} n={len(tok) - 4}")
    return lines


def rows_lat(tok: list[str]) -> list[str]:
    tag = "LATC" if tok[0] == "latc" else "LAT"
    value = float(SPEC.get("lat_us", 6.5))
    lines = []
    for pair in tok[3:]:
        a, b = pair.split(":")
        lines.append(f"{tag} {a}<->{b} payload={tok[2]} one_way_us={value:.2f}")
    return lines


def rows_a2a(tok: list[str]) -> list[str]:
    return [f"A2A payload={tok[2]} group={tok[3]} step_us=10.00"]


def rows_hlat(tok: list[str]) -> list[str]:
    lines = []
    for dev in tok[2].split(","):
        for size in tok[3:]:
            for d, mem in (
                ("h2d", "pinned"),
                ("h2d", "pageable"),
                ("d2h", "pinned"),
                ("d2h", "pageable"),
            ):
                lines.append(
                    f"HLAT dev={dev} dir={d} mem={mem} size={size} n={tok[1]}"
                    + stats_tail("", 10.0)
                )
    return lines


def rows_hflag(tok: list[str]) -> list[str]:
    return [f"HFLAG dev={d} n={tok[1]}" + stats_tail("rtt_", 3.0) for d in tok[2:]]


def rows_curve(tok: list[str]) -> list[str]:
    mode, iters = tok[1], tok[2]
    lines = []
    for pair in tok[5:]:
        a, b = pair.split(":")
        for size in tok[4].split(","):
            lines.append(
                f"CURVE mode={mode} src={a} dst={b} size={size} blocks=1 n={iters}"
                + stats_tail("", float(SPEC.get("lat_us", 6.5)))
                + " ev_us=6.600"
            )
    return lines


ROWS = {
    "bw": rows_bw,
    "hbw": rows_hbw,
    "lat": rows_lat,
    "latc": rows_lat,
    "a2a": rows_a2a,
    "hlat": rows_hlat,
    "hflag": rows_hflag,
    "curve": rows_curve,
}


def event(at: int) -> dict[str, Any] | None:
    """Apply the side effects planned for command ``at``; return an ``err`` event."""
    failing = None
    for ev in SPEC.get("events", []):
        if ev.get("at") != at:
            continue
        what = ev["do"]
        if what == "write":
            with open(ev["path"], "w", encoding="utf-8") as fh:
                fh.write(ev["text"])
        elif what == "sleep":
            time.sleep(float(ev["s"]))
        elif what == "exit":
            sys.stdout.flush()
            os._exit(int(ev.get("rc", 1)))
        elif what == "err":
            failing = ev
    return failing


def batch() -> int:
    if SPEC.get("stderr"):
        sys.stderr.write(str(SPEC["stderr"]) + "\n")
        sys.stderr.flush()
    if SPEC.get("no_ready"):
        time.sleep(600)
    ticks = ",".join("100.000" for _ in buses())
    out(f"READY devices={len(buses())} tick_mhz={ticks} version={version()}")
    count = 0
    for raw in sys.stdin:
        tok = raw.split()
        if not tok or tok[0].startswith("#"):
            continue
        count += 1
        note("cmd " + " ".join(tok))
        out("### " + " ".join(tok))
        failing = event(count)
        t0 = time.monotonic()
        rc = 0
        if failing is not None:
            rc = int(failing.get("rc", 3))
            out("ERR " + str(failing["text"]))
        elif tok[0] in ROWS:
            for line in ROWS[tok[0]](tok):
                out(line)
        else:
            rc = 1
        if rc:
            out(f"ERR rc={rc}")
        out(f"END rc={rc} ms={(time.monotonic() - t0) * 1000:.1f}")
    return 0


def main(argv: list[str]) -> int:
    global SPEC
    spec_file = Path(os.environ["FAKE_PROBE_SPEC"])
    SPEC = json.loads(spec_file.read_text(encoding="utf-8"))
    args = list(argv)
    note("argv " + " ".join(args))
    note("HIP_VISIBLE_DEVICES=" + os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"))
    note("ROCR_VISIBLE_DEVICES=" + os.environ.get("ROCR_VISIBLE_DEVICES", "<unset>"))
    if args[:1] == ["--no-peer"]:
        args = args[1:]
    if args[:1] == ["version"]:
        modes = SPEC.get("modes", DEFAULT_MODES)
        out(f"VERSION {version()} modes={modes}")
        return 0
    if args[:1] == ["buses"]:
        for i, bus in enumerate(buses()):
            out(f"DEV {i} AMD Radeon Pro V620 gfx1030 bus=0000:{bus}:00.0")
        return 0
    if args[:1] == ["batch"]:
        return batch()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
