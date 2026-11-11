#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Junction temperature of the leased GPUs, one CSV row every PERIOD seconds.

Log only: nothing reads it to decide anything during a run; ``analyze.py
report`` prints the mean and the maximum per arm. HIP ordinal ``i`` is the i-th
GPU node of the KFD topology (node ids in numeric order, ``simd_count`` > 0);
that node's ``drm_render_minor`` names the render node whose hwmon has the
sensor labelled ``junction`` (millidegrees C).

  gpu_temps.py --gpus 2,3,4,5 --out temps.csv [--period 15] [--count N]

Columns: ``time`` (ISO, local), ``epoch``, then ``hip<N>`` per GPU (blank while
its sensor cannot be read; the sensors are looked up again then).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")
DRM = Path("/sys/class/drm")


def render_minors(nodes: Path = KFD_NODES) -> list[int]:
    """drm_render_minor of each GPU node, in HIP ordinal order."""
    out = []
    for node in sorted(
        (p for p in nodes.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
    ):
        props = {}
        for line in (node / "properties").read_text().splitlines():
            key, _, val = line.partition(" ")
            props[key] = val.strip()
        if int(props.get("simd_count", "0")) > 0:
            out.append(int(props["drm_render_minor"]))
    return out


def junction_input(minor: int, drm: Path = DRM) -> Path | None:
    """The hwmon ``temp*_input`` labelled ``junction`` of one render node."""
    hwmons = sorted((drm / f"renderD{minor}" / "device" / "hwmon").glob("hwmon*"))
    for hwmon in hwmons:
        for label in sorted(hwmon.glob("temp*_label")):
            if label.read_text().strip() == "junction":
                return hwmon / label.name.replace("_label", "_input")
    return None


def read_c(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return int(path.read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def sensors(
    gpus: list[int], nodes: Path = KFD_NODES, drm: Path = DRM
) -> dict[int, Path | None]:
    """The junction sensor file of each HIP ordinal (None where there is none)."""
    try:
        minors = render_minors(nodes)
    except OSError as exc:
        print(f"gpu_temps: no KFD topology ({exc}): logging blanks", file=sys.stderr)
        minors = []
    return {
        g: junction_input(minors[g], drm) if g < len(minors) else None for g in gpus
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpus", required=True, help="HIP ordinals, comma separated")
    ap.add_argument("--out", type=Path, required=True, help="CSV, appended to")
    ap.add_argument("--period", type=float, default=15.0, help="seconds")
    ap.add_argument("--count", type=int, default=0, help="rows to write, 0 = no end")
    ap.add_argument("--kfd-nodes", type=Path, default=KFD_NODES)
    ap.add_argument("--drm", type=Path, default=DRM)
    args = ap.parse_args(argv)
    gpus = [int(g) for g in args.gpus.split(",")]
    paths = sensors(gpus, args.kfd_nodes, args.drm)
    fresh = not args.out.exists() or args.out.stat().st_size == 0
    rows = 0
    with open(args.out, "a", newline="") as f:
        out = csv.writer(f)
        if fresh:
            out.writerow(["time", "epoch", *[f"hip{g}" for g in gpus]])
        tick = time.monotonic()
        while True:
            now = time.time()
            temps = [read_c(paths[g]) for g in gpus]
            out.writerow(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                    f"{now:.1f}",
                    *["" if t is None else f"{t:.1f}" for t in temps],
                ]
            )
            f.flush()
            rows += 1
            if args.count and rows >= args.count:
                return 0
            if any(t is None for t in temps):
                paths = sensors(gpus, args.kfd_nodes, args.drm)
            tick += args.period
            time.sleep(max(0.0, tick - time.monotonic()))


if __name__ == "__main__":
    sys.exit(main())
