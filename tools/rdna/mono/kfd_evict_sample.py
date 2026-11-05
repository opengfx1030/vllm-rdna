#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KFD eviction counters of one server's worker processes.

The kernel keeps, per process and GPU, how long that process's buffers were
evicted from VRAM (``/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms``,
cumulative). Another job taking VRAM or a driver reset shows up as a jump of a
worker's counter, and a window with a jump is not a measurement of this server:
it is INVALID and gets measured again.

    kfd_evict_sample.py snapshot --pid-file serve.pid --out before.json
    kfd_evict_sample.py verdict --before before.json --after after.json \\
        --min-procs 4                       # exit 1 on a jump / a lost worker
    kfd_evict_sample.py watch --pid-file serve.pid --out evict.jsonl \\
        --interval 2                        # time series until interrupted

The root pid is the server (``serve.pid`` of ``serve_validate.sh``); the workers
are its descendants that hold a KFD context. Library use: ``snapshot(root)`` and
``verdict(before, after)``. Stdlib only, read-only.
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

KFD_PROC = Path("/sys/class/kfd/kfd/proc")


def parse_stat(text: str) -> list[str]:
    """Fields after the command name of /proc/<pid>/stat (state, ppid, ...).

    The command name may hold spaces and parentheses: split after the last ')'.
    """
    return text[text.rindex(")") + 2 :].split()


def _stat_fields(pid: int) -> list[str] | None:
    try:
        return parse_stat(Path(f"/proc/{pid}/stat").read_text())
    except (OSError, ValueError):
        return None


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


def descends_from(pid: int, root: int) -> bool:
    for _ in range(64):
        if pid == root:
            return True
        fields = _stat_fields(pid)
        if fields is None or pid <= 1:
            return False
        pid = int(fields[1])  # ppid
    return False


def read_evicted(pid: int) -> dict[str, int]:
    """evicted_ms per GPU node of one process (empty if it has no KFD entry)."""
    out = {}
    for path in sorted((KFD_PROC / str(pid)).glob("stats_*/evicted_ms")):
        with contextlib.suppress(OSError, ValueError):
            out[path.parent.name.removeprefix("stats_")] = int(path.read_text())
    return out


def snapshot(root: int) -> dict:
    """Eviction counters of ``root`` and its descendants that hold a KFD context."""
    procs = {}
    try:
        entries = [int(p.name) for p in KFD_PROC.iterdir() if p.name.isdigit()]
    except OSError as exc:
        raise SystemExit(f"cannot read {KFD_PROC}: {exc}") from exc
    for pid in entries:
        if descends_from(pid, root):
            counters = read_evicted(pid)
            procs[str(pid)] = {
                "comm": _comm(pid),
                "evicted_ms": counters,
                "total_ms": sum(counters.values()),
            }
    return {"t": time.time(), "root": root, "procs": procs}


def verdict(
    before: dict, after: dict, min_procs: int = 1, tolerance_ms: int = 0
) -> dict:
    """Compare two snapshots: ``ok`` is False on any counter increase above the
    tolerance, a worker that disappeared, or fewer than ``min_procs`` workers
    seen (no coverage is no proof)."""
    jumps = []
    for pid, b in before["procs"].items():
        a = after["procs"].get(pid)
        if a is None:
            continue
        for gpu, was in b["evicted_ms"].items():
            now = a["evicted_ms"].get(gpu, was)
            if now - was > tolerance_ms:
                jumps.append(
                    {
                        "pid": int(pid),
                        "comm": a["comm"],
                        "gpu": gpu,
                        "before_ms": was,
                        "after_ms": now,
                        "delta_ms": now - was,
                    }
                )
    gone = sorted(set(before["procs"]) - set(after["procs"]))
    new = sorted(set(after["procs"]) - set(before["procs"]))
    covered = min(len(before["procs"]), len(after["procs"]))
    reasons = []
    if jumps:
        reasons.append(f"{len(jumps)} counter jump(s)")
    if gone:
        reasons.append(f"worker(s) gone: {gone}")
    if covered < min_procs:
        reasons.append(f"saw {covered} KFD process(es), need {min_procs}")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "jumps": jumps,
        "gone": gone,
        "new": new,
        "procs": covered,
        "window_s": round(after["t"] - before["t"], 3),
        "total_delta_ms": sum(j["delta_ms"] for j in jumps),
    }


def verdict_line(v: dict) -> str:
    if v["ok"]:
        return f"EVICT OK ({v['procs']} procs, {v['window_s']:.0f} s)"
    worst = max(v["jumps"], key=lambda j: j["delta_ms"], default=None)
    detail = (
        f"; worst pid {worst['pid']} gpu {worst['gpu']} +{worst['delta_ms']} ms"
        if worst
        else ""
    )
    return f"EVICT INVALID: {'; '.join(v['reasons'])}{detail}"


def _root(args: argparse.Namespace) -> int:
    if args.root_pid:
        return args.root_pid
    return int(Path(args.pid_file).read_text().split()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("snapshot", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--root-pid", type=int)
        p.add_argument("--pid-file")
        p.add_argument("--out", required=True)
        if name == "watch":
            p.add_argument("--interval", type=float, default=2.0)
            p.add_argument("--duration", type=float, default=0.0, help="0: forever")
    v = sub.add_parser("verdict")
    v.add_argument("--before", required=True)
    v.add_argument("--after", required=True)
    v.add_argument("--min-procs", type=int, default=1)
    v.add_argument("--tolerance-ms", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "verdict":
        res = verdict(
            json.loads(Path(args.before).read_text()),
            json.loads(Path(args.after).read_text()),
            args.min_procs,
            args.tolerance_ms,
        )
        print(verdict_line(res))
        print(json.dumps(res))
        return 0 if res["ok"] else 1
    if not (args.root_pid or args.pid_file):
        ap.error("--root-pid or --pid-file is required")
    root = _root(args)
    if args.cmd == "snapshot":
        Path(args.out).write_text(json.dumps(snapshot(root)))
        return 0
    end = time.time() + args.duration if args.duration else float("inf")
    with open(args.out, "a") as f:
        while time.time() < end:
            f.write(json.dumps(snapshot(root)) + "\n")
            f.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
