#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KFD queues of the vLLM workers of the running arm, sampled from outside.

The hardware queue budget (``GPU_MAX_HW_QUEUES``) cannot be read back from a
worker: vLLM retitles its processes (setproctitle), which overwrites
``/proc/<pid>/environ``. KFD lists every user queue of a process in
``/sys/class/kfd/kfd/proc/<pid>/queues/<id>/{gpuid,size,type}`` (type 0 is a
compute queue, 1 an SDMA queue): the compute queues of a worker are what the
budget really allowed it to create. ``slot.sh`` runs this for the whole lease and
``analyze.py`` reads ``queues.jsonl``.

  queue_probe.py --run RUN [--period 20] [--out RUN/queues.jsonl] [--count N]

The server of the running arm is the newest ``RUN/arms/*/serve.pid`` whose pid
is alive. One JSON row per sample:
``{"t", "arm", "root", "workers": {label: {"pid", "compute", "sdma", "gpus"}}}``.
Rows are only written while workers with KFD queues exist. Log only.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "port_v031"))
import kfd_evictions as kfd  # noqa: E402

KFD_PROC = kfd.KFD_PROC


def read_queues(pid: int, kfd_proc: str = KFD_PROC) -> dict:
    """Queue counts of one process: compute and SDMA queues, and the GPUs they
    are on. Empty for a process without a KFD context."""
    compute = sdma = 0
    gpus = set()
    for d in glob.glob(f"{kfd_proc}/{pid}/queues/*"):
        try:
            with open(f"{d}/type") as f:
                kind = int(f.read().split()[0])
            with open(f"{d}/gpuid") as f:
                gpus.add(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if kind == 0:
            compute += 1
        elif kind in (1, 4):
            sdma += 1
    return {"compute": compute, "sdma": sdma, "gpus": len(gpus)} if gpus else {}


def running_arm(run: Path, proc: str = kfd.PROC) -> tuple[str, int] | None:
    """(arm, pid of its server) of the newest serve.pid whose process is alive."""
    pids = sorted(run.glob("arms/*/serve.pid"), key=lambda p: p.stat().st_mtime)
    for path in reversed(pids):
        try:
            pid = int(path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if os.path.exists(f"{proc}/{pid}"):
            return path.parent.name, pid
    return None


def sample(run: Path, proc: str = kfd.PROC, kfd_proc: str = KFD_PROC) -> dict | None:
    found = running_arm(run, proc)
    if found is None:
        return None
    arm, root = found
    workers = {}
    for label, pid in kfd.worker_pids(root, proc).items():
        queues = read_queues(pid, kfd_proc)
        if queues:
            workers[label] = {"pid": pid, **queues}
    if not workers:
        return None
    return {"t": round(time.time(), 1), "arm": arm, "root": root, "workers": workers}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default RUN/queues.jsonl")
    ap.add_argument("--period", type=float, default=20.0, help="seconds")
    ap.add_argument("--count", type=int, default=0, help="samples, 0 = no end")
    ap.add_argument("--proc", default=kfd.PROC)
    ap.add_argument("--kfd-proc", default=KFD_PROC)
    args = ap.parse_args(argv)
    out = args.out or args.run / "queues.jsonl"
    taken = 0
    tick = time.monotonic()
    while True:
        row = sample(args.run, args.proc, args.kfd_proc)
        if row is not None:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        taken += 1
        if args.count and taken >= args.count:
            return 0
        tick += args.period
        time.sleep(max(0.0, tick - time.monotonic()))


if __name__ == "__main__":
    sys.exit(main())
