#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Evidence that GPU_MAX_HW_QUEUES took effect in the workers of the running arm.

  qprobe.py TREE ROOT_PID LABEL

Prints one line: per-worker KFD compute/SDMA queue count (CLR creates the budget's
queues plus one of its own: 3 at a budget of 2, 5 at 4) and whatever
/proc/<pid>/environ still says (vLLM's setproctitle usually garbles it).
"""

import glob
import os
import sys
from pathlib import Path

import regex as re

tree, root, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.join(tree, "tools/rdna/port_v031"))
import kfd_evictions as k  # noqa: E402


def environ_hwq(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as e:
        return f"unreadable({e.strerror})"
    m = re.search(rb"GPU_MAX_HW_QUEUES=(\S+?)(?:\x00|$)", data)
    return m.group(1).decode(errors="replace") if m else "absent"


def read_queues(pid: int) -> str:
    comp = sdma = 0
    for d in glob.glob(f"{k.KFD_PROC}/{pid}/queues/*"):
        try:
            kind = int(Path(f"{d}/type").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if kind == 0:
            comp += 1
        elif kind in (1, 4):
            sdma += 1
    return f"compute={comp} sdma={sdma}"


parts = [f"{label}: root {root} environ={environ_hwq(root)}"]
for name, pid in sorted(k.worker_pids(root).items()):
    parts.append(f"{name} pid {pid} {read_queues(pid)} environ={environ_hwq(pid)}")
print("; ".join(parts))
