# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Per-worker torch allocator sampler for memory audits (the tree under test is
# unchanged).
#
#   PYTHONPATH=<this dir> RDNA_MEMHOOK_DIR=<out dir> [RDNA_MEMHOOK_PERIOD=2] \
#       <server command>
#
# Every process whose title starts with "VLLM::" and has initialised the accelerator
# appends one JSON line per period to <out dir>/torchmem.<pid>.jsonl:
#   t, pid, title, dev, alloc, reserved, max_alloc, max_reserved, free, total (bytes)
# `free`/`total` come from the driver (get_memory_info), so
# non_torch = total - free - reserved.
#
# The sampler thread is not the thread that selected the worker's device (the
# current device is thread-local), so the device is the one whose caching allocator
# holds memory in THIS process (allocator statistics are plain reads and never
# create a context on another GPU). get_memory_info is only called for that device.
import contextlib
import os

_d = os.environ.get("RDNA_MEMHOOK_DIR")


def _title() -> str:
    try:
        with open("/proc/self/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _sample(torch, pid: int, title: str, dev: int):
    """One JSON row for `dev` (or the device holding memory when dev < 0)."""
    import time

    acc = torch.accelerator
    if dev < 0:
        best = 0
        for i in range(acc.device_count()):
            r = acc.memory_reserved(i)
            if r > best:
                dev, best = i, r
        if dev < 0:
            return dev, None
    free, total = acc.get_memory_info(dev)
    return dev, {
        "t": round(time.time(), 2),
        "pid": pid,
        "title": title,
        "dev": dev,
        "alloc": acc.memory_allocated(dev),
        "reserved": acc.memory_reserved(dev),
        "max_alloc": acc.max_memory_allocated(dev),
        "max_reserved": acc.max_memory_reserved(dev),
        "free": free,
        "total": total,
    }


def _run(out_dir: str, period: float) -> None:
    import json
    import sys
    import time

    pid = os.getpid()
    path = os.path.join(out_dir, f"torchmem.{pid}.jsonl")
    dev = -1
    with contextlib.ExitStack() as stack:
        f = None
        while True:
            time.sleep(period)
            torch = sys.modules.get("torch")
            if torch is None:
                continue
            title = _title()
            if not title.startswith("VLLM::"):
                continue
            try:
                if not torch.cuda.is_initialized():
                    continue
                dev, row = _sample(torch, pid, title, dev)
                if row is None:
                    continue
            except Exception as e:  # noqa: BLE001
                row = {
                    "t": round(time.time(), 2),
                    "pid": pid,
                    "title": title,
                    "err": repr(e)[:200],
                }
            if f is None:
                f = stack.enter_context(open(path, "a", buffering=1))
            f.write(json.dumps(row) + "\n")


if _d:
    import threading

    os.makedirs(_d, exist_ok=True)
    threading.Thread(
        target=_run,
        args=(_d, float(os.environ.get("RDNA_MEMHOOK_PERIOD", "2"))),
        daemon=True,
        name="rdna-memhook",
    ).start()
