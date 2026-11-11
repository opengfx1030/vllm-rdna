#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Memory audit table for one arm directory.

  memsum.py ARM_DIR "ce d3 d8 db"   # buses in device-ordinal order of the arm

Reads torchmem/*.jsonl (memhook), memwatch.tsv and summary.txt. Per worker: torch
allocated / reserved (now, peak), driver used = total - free (now, peak), non-torch =
driver used - reserved. `now` = the last sample at or before READY + 30 s (the server
is idle then); `peak` = max over the run. memwatch.tsv adds the sysfs VRAM per bus and
the host RSS / mlocked pages per process. Run it with the venv python (regex).
"""

import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import regex as re

GIB = 1 << 30


def ready_epoch(arm: str):
    """Epoch of the READY line, from t0.txt (date) + summary.txt (HH:MM:SS)."""
    try:
        day = (Path(arm) / "t0.txt").read_text().split()[0]
        text = (Path(arm) / "summary.txt").read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        m = re.match(r"\[(\d\d:\d\d:\d\d)\] READY", line)
        if m:
            stamp = f"{day} {m.group(1)}"
            return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    return None


def worker_rows(arm: str, t_ready) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(arm, "torchmem", "torchmem.*.jsonl"))):
        samples = []
        for line in Path(path).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "alloc" in r:
                samples.append(r)
        if not samples or not samples[-1]["title"].startswith("VLLM::Worker"):
            continue
        cutoff = (t_ready + 30) if t_ready else samples[-1]["t"] - 20
        now = [s for s in samples if s["t"] <= cutoff][-1:] or samples[-1:]
        s = now[0]
        peak_used = max(x["total"] - x["free"] for x in samples)
        rows.append(
            dict(
                title=s["title"].split()[0],
                pid=s["pid"],
                dev=s["dev"],
                alloc=s["alloc"] / GIB,
                reserved=s["reserved"] / GIB,
                max_alloc=max(x["max_alloc"] for x in samples) / GIB,
                max_reserved=max(x["max_reserved"] for x in samples) / GIB,
                used=(s["total"] - s["free"]) / GIB,
                peak_used=peak_used / GIB,
                total=s["total"] / GIB,
            )
        )
    rows.sort(key=lambda r: (r["dev"], r["title"]))
    return rows


def print_workers(rows: list[dict], buses: list[str]) -> None:
    print(
        "\n| worker | dev | bus | torch alloc | torch reserved | peak alloc "
        "| peak reserved | driver used | peak driver used "
        "| non-torch (used-reserved) | free |"
    )
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        bus = buses[r["dev"]] if r["dev"] < len(buses) else "?"
        print(
            f"| {r['title']} | {r['dev']} | {bus} | {r['alloc']:.2f} "
            f"| {r['reserved']:.2f} | {r['max_alloc']:.2f} "
            f"| {r['max_reserved']:.2f} | {r['used']:.2f} | {r['peak_used']:.2f} "
            f"| {r['used'] - r['reserved']:.2f} | {r['total'] - r['used']:.2f} |"
        )


def read_memwatch(arm: str):
    gpu = defaultdict(list)
    proc = defaultdict(list)
    sysm = []
    t0 = Path(arm) / "t0.txt"
    day = t0.read_text().split()[0] if t0.exists() else None
    text = (Path(arm) / "memwatch.tsv").read_text(errors="replace")
    for line in text.splitlines():
        f = line.split("\t")
        try:
            t = time.mktime(time.strptime(f"{day} {f[1]}", "%Y-%m-%d %H:%M:%S"))
        except (ValueError, IndexError, TypeError):
            continue
        if f[0] == "G" and len(f) >= 5:
            gpu[f[2]].append((t, int(f[3]), int(f[4])))
        elif f[0] == "P" and len(f) >= 9:
            proc[(f[2], f[3])].append((t, *[int(x) for x in f[4:9]]))
        elif f[0] == "S" and len(f) >= 8:
            sysm.append((t, *[int(x) for x in f[2:8]]))
    return gpu, proc, sysm


def print_memwatch(arm: str, t_ready) -> None:
    gpu, proc, sysm = read_memwatch(arm)
    print(
        "\n| bus | sysfs VRAM at READY+30 s (MiB) | peak (MiB) "
        "| idle before boot (MiB) | GTT peak (MiB) |"
    )
    print("|---|---:|---:|---:|---:|")
    for bus, v in sorted(gpu.items()):
        cut = (t_ready + 30) if t_ready else v[-1][0] - 20
        at = [x for x in v if x[0] <= cut][-1:] or v[-1:]
        print(
            f"| {bus} | {at[0][1]} | {max(x[1] for x in v)} | {v[0][1]} "
            f"| {max(x[2] for x in v)} |"
        )
    print(
        "\n| pid | process | RSS now (MiB) | RSS peak | anon peak | file peak "
        "| shmem peak | VmLck peak |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for (pid, title), v in sorted(
        proc.items(), key=lambda kv: -max(x[1] for x in kv[1])
    ):
        cut = (t_ready + 30) if t_ready else v[-1][0] - 20
        at = [x for x in v if x[0] <= cut][-1:] or v[-1:]
        if max(x[1] for x in v) < 200:
            continue
        print(
            f"| {pid} | {title[:40]} | {at[0][1]} | {max(x[1] for x in v)} "
            f"| {max(x[2] for x in v)} | {max(x[3] for x in v)} "
            f"| {max(x[4] for x in v)} | {max(x[5] for x in v)} |"
        )
    if sysm:
        gib = 1024
        print(
            f"\nhost: MemAvailable min {min(x[1] for x in sysm) // gib} GiB "
            f"(start {sysm[0][1] // gib} GiB), "
            f"mlocked max {max(x[5] for x in sysm) // gib} GiB, "
            f"shmem max {max(x[4] for x in sysm) // gib} GiB, "
            f"cached max {max(x[3] for x in sysm) // gib} GiB, "
            f"swap used max {max(x[6] for x in sysm)} MiB"
        )


def main() -> None:
    arm = sys.argv[1]
    buses = sys.argv[2].split() if len(sys.argv) > 2 else []
    t_ready = ready_epoch(arm)
    print(f"READY epoch: {t_ready}")
    print_workers(worker_rows(arm, t_ready), buses)
    print_memwatch(arm, t_ready)


if __name__ == "__main__":
    main()
