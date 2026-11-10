#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log DPM level changes of GPU endpoint functions (sysfs, read-only).

Used next to the hx soak runs to correlate rare ordering events with clock or
link-state transitions (MCLK / FCLK / SOCCLK / SCLK / PCIe DPM level).

    hx_dpm_log.py --out FILE --buses "d4 d9 dc" [--interval 0.1] [--wait-pid PID]

A change of the level INDEX of the starred (current) line of
pp_dpm_{sclk,mclk,fclk,socclk,pcie} is appended as
`<epoch seconds> <bus> <clock> <old> -> <new>` (the full starred text, e.g.
`1: 673Mhz`; the frequency of a middle level wobbles and is not a transition).
A `# <epoch> samples=N transitions=M` line is written every 60 s. The script
ends when --wait-pid is gone or on SIGTERM / SIGINT.
"""

import argparse
import os
import signal
import sys
import time

CLOCKS = ("sclk", "mclk", "fclk", "socclk", "pcie")
_stop = False


def _on_signal(signum: int, frame: object) -> None:
    global _stop
    _stop = True


def starred_level(path: str) -> str:
    """The starred line of a pp_dpm_* file, e.g. '1: 673Mhz'."""
    try:
        with open(path) as fh:
            for line in fh:
                text = line.rstrip()
                if text.endswith("*"):
                    return text[:-1].strip()
    except OSError as exc:
        return f"error:{exc.errno}"
    return "?"


def level_index(text: str) -> str:
    return text.split(":", 1)[0]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="log file (appended)")
    ap.add_argument("--buses", required=True, help='PCI buses, e.g. "d4 d9 dc"')
    ap.add_argument("--interval", type=float, default=0.1, help="sample period, s")
    ap.add_argument("--wait-pid", type=int, default=0, help="stop when pid is gone")
    args = ap.parse_args()
    buses = args.buses.split()
    if not buses:
        print("hx_dpm_log: --buses is empty", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    last: dict[tuple[str, str], str] = {}
    samples = 0
    transitions = 0
    next_summary = time.time() + 60.0
    with open(args.out, "a", buffering=1) as log:
        while not _stop and (args.wait_pid == 0 or pid_alive(args.wait_pid)):
            now = time.time()
            for bus in buses:
                for clk in CLOCKS:
                    path = f"/sys/bus/pci/devices/0000:{bus}:00.0/pp_dpm_{clk}"
                    cur = starred_level(path)
                    prev = last.get((bus, clk))
                    if prev is None or level_index(prev) != level_index(cur):
                        log.write(f"{now:.3f} {bus} {clk} {prev or '-'} -> {cur}\n")
                        if prev is not None:
                            transitions += 1
                    last[(bus, clk)] = cur
            samples += 1
            if now >= next_summary:
                log.write(f"# {now:.3f} samples={samples} transitions={transitions}\n")
                next_summary = now + 60.0
            time.sleep(args.interval)
        end = f"# {time.time():.3f} end samples={samples} transitions={transitions}\n"
        log.write(end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
