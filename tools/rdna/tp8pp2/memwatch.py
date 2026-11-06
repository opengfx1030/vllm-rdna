#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host + VRAM sampler for one server run (stdlib only).

  memwatch.py BUS,BUS,... PIDFILE OUT.tsv [PERIOD_S]

PIDFILE holds the process-group id of the server (serve_validate.sh serve.pid).
One tab-separated file, three row kinds:
  G t bus vram_used_MiB gtt_used_MiB            amdgpu sysfs, one row per bus
  P t pid title rss anon file shmem vmlck MiB    every process of the server's group
  S t memavail memfree cached shmem mlocked swapused MiB
Stops when the group has been seen and is gone again (or after 6 h).
"""

import glob
import os
import sys
import time
from pathlib import Path

MIB = 1 << 20


def card_dirs() -> dict[str, str]:
    d = {}
    for c in glob.glob("/sys/class/drm/card[0-9]*/device"):
        if "-" in c.split("/")[4]:
            continue
        addr = os.path.basename(os.path.realpath(c))  # 0000:ce:00.0
        d[addr.split(":")[1]] = c
    return d


def rd(path: str) -> int:
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError):
        return -1


def group_members(pgid: int) -> list[int]:
    pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            st = Path(f"/proc/{p}/stat").read_text()
            rest = st[st.rindex(")") + 2 :].split()
            if int(rest[2]) == pgid:  # field 5 = pgrp (rest[0] = state)
                pids.append(int(p))
        except (OSError, ValueError, IndexError):
            continue
    return pids


def title(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return raw.decode(errors="replace").strip()[:60].replace(" ", "_") or "?"
    except OSError:
        return "?"


def pstat(pid: int):
    v = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            k, _, rest = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck"):
                v[k] = int(rest.split()[0]) / 1024.0
    except (OSError, ValueError):
        return None
    return v


def meminfo() -> dict[str, float]:
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, rest = line.partition(":")
        m[k] = int(rest.split()[0]) / 1024.0
    return m


def main() -> int:
    buses = [b.removeprefix("0000:").lower() for b in sys.argv[1].split(",") if b]
    pidfile, out = sys.argv[2], sys.argv[3]
    period = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    cards = card_dirs()
    missing = [b for b in buses if b not in cards]
    if missing:
        print("memwatch: no drm card for", missing, file=sys.stderr)
        return 2
    seen = False
    t_end = time.time() + 6 * 3600
    with open(out, "a", buffering=1) as f:
        while time.time() < t_end:
            t = time.strftime("%H:%M:%S")
            try:
                pgid = int(Path(pidfile).read_text().strip())
            except (OSError, ValueError):
                pgid = -1
            for b in buses:
                c = cards[b]
                f.write(
                    f"G\t{t}\t{b}\t{rd(c + '/mem_info_vram_used') / MIB:.0f}"
                    f"\t{rd(c + '/mem_info_gtt_used') / MIB:.0f}\n"
                )
            members = group_members(pgid) if pgid > 0 else []
            for pid in members:
                s = pstat(pid)
                if s is None or "VmRSS" not in s:
                    continue
                f.write(
                    f"P\t{t}\t{pid}\t{title(pid)}\t{s['VmRSS']:.0f}"
                    f"\t{s.get('RssAnon', 0):.0f}\t{s.get('RssFile', 0):.0f}"
                    f"\t{s.get('RssShmem', 0):.0f}\t{s.get('VmLck', 0):.0f}\n"
                )
            mi = meminfo()
            swap_used = mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)
            f.write(
                f"S\t{t}\t{mi['MemAvailable']:.0f}\t{mi['MemFree']:.0f}"
                f"\t{mi['Cached']:.0f}\t{mi['Shmem']:.0f}\t{mi['Mlocked']:.0f}"
                f"\t{swap_used:.0f}\n"
            )
            if members:
                seen = True
            elif seen:
                break
            time.sleep(period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
