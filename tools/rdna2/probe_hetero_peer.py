#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print the HIP peer-access matrix and a copy time when devices exist.

The script never fills in a bandwidth it did not just measure. With
fewer than two HIP devices it reports that bandwidth was not measured
and exits 0. Peer transport stays off unless this JSON has
``measured`` true, ``passed`` true, both directions set, and a positive
byte count and elapsed time.

Mixed gfx1100/gfx1030 RCCL is not probed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time


def _emit(report: dict, path: str | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")


def _hip_can_access(src: int, dst: int) -> bool | None:
    """Return hipDeviceCanAccessPeer, or None if HIP is not loadable."""
    try:
        lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return None
    can = ctypes.c_int()
    err = lib.hipDeviceCanAccessPeer(ctypes.byref(can), int(src), int(dst))
    if err != 0:
        return None
    return bool(can.value)


def _torch_can_access(torch, src: int, dst: int) -> bool | None:
    fn = getattr(torch.cuda, "can_device_access_peer", None)
    if fn is None:
        return None
    try:
        return bool(fn(src, dst))
    except Exception:
        return None


def _measure_copy(torch, src: int, dst: int, nbytes: int) -> tuple[int, float] | None:
    """Time one device-to-device copy. Return None if it cannot run."""
    if nbytes <= 0:
        return None
    try:
        source = torch.empty(nbytes // 2, dtype=torch.float16, device=f"cuda:{src}")
        dest = torch.empty_like(source, device=f"cuda:{dst}")
        source.fill_(1)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        started = time.perf_counter()
        dest.copy_(source)
        torch.cuda.synchronize(dst)
        elapsed = time.perf_counter() - started
    except Exception:
        return None
    copied = int(source.numel() * source.element_size())
    if elapsed <= 0 or copied <= 0:
        return None
    return copied, elapsed


def build_report(copy_bytes: int) -> dict:
    """Collect the matrix. Bandwidth is set only from a timed copy."""
    report: dict = {
        "devices": [],
        "access": [],
        "measured": False,
        "passed": False,
        "nbytes": None,
        "seconds": None,
        "bandwidth_bytes_per_sec": None,
        "note": "",
    }
    try:
        import torch
    except Exception as exc:
        report["note"] = f"torch unavailable ({exc}); bandwidth not measured"
        return report
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        report["note"] = "no HIP/CUDA devices; bandwidth not measured"
        return report
    count = int(torch.cuda.device_count())
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", "") or getattr(props, "name", "")
        report["devices"].append({"index": index, "arch": str(arch)})
    both = True
    for src in range(count):
        for dst in range(count):
            if src == dst:
                can = False
            else:
                can = _hip_can_access(src, dst)
                if can is None:
                    can = _torch_can_access(torch, src, dst)
                if can is None:
                    can = False
                    both = False
            report["access"].append({"src": src, "dst": dst, "can": bool(can)})
    if count < 2:
        report["note"] = "fewer than 2 devices; bandwidth not measured"
        return report
    timed = _measure_copy(torch, 0, 1, copy_bytes)
    if timed is None:
        report["note"] = "copy was not timed; bandwidth not measured"
        return report
    nbytes, seconds = timed
    report["measured"] = True
    report["nbytes"] = nbytes
    report["seconds"] = seconds
    report["bandwidth_bytes_per_sec"] = nbytes / seconds
    access = {(item["src"], item["dst"]): item["can"] for item in report["access"]}
    report["passed"] = bool(
        both and access.get((0, 1)) and access.get((1, 0)) and seconds > 0
    )
    if not report["passed"]:
        report["note"] = (
            "copy was timed but peer access is not true both ways, "
            "or HIP did not answer; peer transport stays off"
        )
    else:
        report["note"] = "peer access and one timed copy; still UNVERIFIED for RCCL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="",
        help="Also write the JSON report to this path.",
    )
    parser.add_argument(
        "--copy-bytes",
        type=int,
        default=1 << 20,
        help="Bytes to attempt to copy when two devices exist.",
    )
    args = parser.parse_args()
    report = build_report(args.copy_bytes)
    _emit(report, args.output or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
