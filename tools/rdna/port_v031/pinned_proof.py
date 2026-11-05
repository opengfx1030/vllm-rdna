#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pieces of the pinned-host mlock GPU proof (mlock_pinned_proof.sh).

    pinned_proof.py neighbour [--workers 2 --shm-gib 47.68 --lock --read DIR]
    pinned_proof.py steady    --url URL --model NAME --out FILE [--duration S]
    pinned_proof.py hook      [options] -- <command of one neighbour load>
    pinned_proof.py verdict   --run DIR --arms "base fix fix2 base2"

``neighbour`` replays what a neighbouring TP server's weight load asks of the
host, with no GPU: per rank one shared-anonymous block the size of the PLE
table (written, mlocked like the real one), anonymous memory sized so that the
whole demand is ``--overshoot-gib`` above the free memory (the page cache then
has to give way, as it does when a real checkpoint loads), and a sequential
read of the checkpoint through the page cache. It refuses a demand that does
not fit MemAvailable minus ``--margin-gib`` and stops allocating when
MemAvailable falls below half of that margin.

``steady`` is a few streaming decode requests in a loop; it logs every gap
between two chunks of one stream that is at least ``--gap-ms`` and one record
per finished request.

``hook`` is the body of serve_validate.sh's LOAD_HOOK. It measures the booted
TP server while the neighbour loads: the root-run pinned sampler and the raw
KFD event reader record in the background, KFD ``evicted_ms`` is sampled every
0.5 s, and each of ``--loads`` replays runs under the box's quiet-window and
boot-lock protocol. It writes ``proof/windows.json`` and the sampler report,
and ends with PASS (the loads ran) or FAIL, not with the verdict.

``verdict`` reads every arm's ``proof/windows.json``, serve.log and bench
cells and says PROVEN, INCONCLUSIVE (the baseline did not evict, so there is
nothing to remove) or FAILED, with the numbers.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import importlib.util
import json
import mmap
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from http.client import HTTPConnection
from typing import Any
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30
CELLS = ("bench-1024x512-c1", "bench-1024x512-c8")
METRICS = (
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "output_throughput",
)
LOCK_LINE = "Locked "
QUIET_MAX_AGE_S = 6 * 60
CHUNK = 256 << 20


def sibling(name: str) -> Any:
    """Load a tool next to this one (they are scripts, not a package)."""
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- neighbour: a replayed weight load ----------------------------------------


def meminfo_kib(path: str = "/proc/meminfo") -> dict[str, int]:
    out = {}
    with open(path) as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0])
    return out


def plan_demand(
    mem_free_kib: int,
    mem_avail_kib: int,
    workers: int,
    shm_gib: float,
    anon_gib: float | None,
    overshoot_gib: float,
    margin_gib: float,
) -> dict[str, Any]:
    """Per-rank shm and anonymous GiB, and whether the host can afford them.

    ``anon_gib`` None sizes the anonymous part so the total is
    ``overshoot_gib`` above MemFree (which forces reclaim), capped at
    MemAvailable minus ``margin_gib``.
    """
    free, avail = mem_free_kib / (1 << 20), mem_avail_kib / (1 << 20)
    cap = max(avail - margin_gib, 0.0)
    if anon_gib is None:
        anon = max(min(free + overshoot_gib, cap) / workers - shm_gib, 0.0)
    else:
        anon = anon_gib
    total = workers * (shm_gib + anon)
    return {
        "workers": workers,
        "shm_gib": shm_gib,
        "anon_gib": round(anon, 3),
        "total_gib": round(total, 3),
        "free_gib": round(free, 3),
        "avail_gib": round(avail, 3),
        "affordable": total <= cap,
    }


class Floor:
    """Raise ``stop`` when MemAvailable falls below a floor (checked at 1 Hz)."""

    def __init__(self, floor_kib: int, stop: threading.Event) -> None:
        self.floor_kib = floor_kib
        self.stop = stop
        self.next = 0.0

    def check(self, what: str) -> None:
        now = time.monotonic()
        if now < self.next:
            return
        self.next = now + 1.0
        if meminfo_kib()["MemAvailable"] < self.floor_kib:
            log(f"MemAvailable is below the floor while {what}: stopping")
            self.stop.set()


def mlock_buffer(buf: Any, size: int) -> int:
    """Lock an mmap into memory: the errno, 0 on success."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return 0 if libc.mlock(addr, size) == 0 else ctypes.get_errno()


def touch_shm(size: int, lock: bool, floor: Floor) -> mmap.mmap:
    """A shared-anonymous block (what hipHostMalloc leaves), written and mlocked."""
    mm = mmap.mmap(-1, size)
    chunk = bytes(min(size, CHUNK))
    done = 0
    while done < size and not floor.stop.is_set():
        n = min(len(chunk), size - done)
        mm.write(chunk[:n])
        done += n
        floor.check(f"writing shm ({done / GIB:.1f} GiB)")
    if lock and not floor.stop.is_set():
        err = mlock_buffer(mm, size)
        if err:
            log(f"mlock of {size / GIB:.2f} GiB failed: {os.strerror(err)}")
    return mm


def touch_anon(size: int, floor: Floor) -> list[bytearray]:
    blocks: list[bytearray] = []
    done = 0
    while done < size and not floor.stop.is_set():
        blocks.append(bytearray(min(CHUNK, size - done)))  # zero-filled: touched
        done += len(blocks[-1])
        floor.check(f"allocating anonymous memory ({done / GIB:.1f} GiB)")
    return blocks


def read_files(paths: list[str], stop: threading.Event) -> int:
    """Sequentially read every file (a checkpoint); the bytes read."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in sorted(os.walk(p, followlinks=True)):
                files += [os.path.join(root, n) for n in sorted(names)]
        else:
            files.append(p)
    total = 0
    for path in files:
        with open(path, "rb", buffering=0) as f:
            while not stop.is_set():
                data = f.read(16 << 20)
                if not data:
                    break
                total += len(data)
    return total


def run_neighbour(args: argparse.Namespace) -> int:
    mi = meminfo_kib()
    plan = plan_demand(
        mi["MemFree"],
        mi["MemAvailable"],
        args.workers,
        args.shm_gib,
        None if args.anon_gib == "auto" else float(args.anon_gib),
        args.overshoot_gib,
        args.margin_gib,
    )
    log(f"neighbour plan {json.dumps(plan)}")
    if not plan["affordable"]:
        log("ABORT: the demand does not fit MemAvailable minus the margin")
        return 3
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    floor = Floor(int(args.margin_gib * (1 << 20) / 2), stop)
    held: list[Any] = []
    read_bytes = [0]

    def rank() -> None:
        held.append(touch_shm(int(plan["shm_gib"] * GIB), args.lock, floor))
        held.append(touch_anon(int(plan["anon_gib"] * GIB), floor))

    def reader() -> None:
        read_bytes[0] = read_files(args.read or [], stop)

    t0 = time.time()
    threads = [threading.Thread(target=rank) for _ in range(args.workers)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    low = mi["MemFree"]
    while any(t.is_alive() for t in threads):
        low = min(low, meminfo_kib()["MemFree"])
        time.sleep(0.5)
    t_alloc = time.time()
    log(f"allocated in {t_alloc - t0:.0f} s, read {read_bytes[0] / GIB:.1f} GiB")
    while time.time() < t_alloc + args.hold and not stop.is_set():
        low = min(low, meminfo_kib()["MemFree"])
        time.sleep(0.5)
    held.clear()
    summary = {
        **plan,
        "t0": round(t0, 3),
        "t_alloc": round(t_alloc, 3),
        "t1": round(time.time(), 3),
        "read_gib": round(read_bytes[0] / GIB, 3),
        "memfree_low_gib": round(low / (1 << 20), 3),
        "aborted": stop.is_set(),
    }
    print("NEIGHBOUR " + json.dumps(summary), flush=True)
    return 4 if stop.is_set() else 0


# ---- steady: a streaming decode load -----------------------------------------


def stream_request(
    host: str,
    port: int,
    path: str,
    body: dict[str, Any],
    on_chunk: Callable[[float], None],
) -> None:
    conn = HTTPConnection(host, port, timeout=600)
    try:
        conn.request(
            "POST", path, json.dumps(body), {"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()
            raise OSError(f"HTTP {resp.status}")
        for raw in resp:
            line = raw.strip()
            if line.startswith(b"data:") and line != b"data: [DONE]":
                on_chunk(time.monotonic())
    finally:
        conn.close()


def stream_records(
    stamps: list[float], t_req: float, w_req: float, idx: int, gap_ms: float
) -> list[dict[str, Any]]:
    """Records of one finished request: its slow gaps and one summary."""
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)]
    out: list[dict[str, Any]] = [
        {"t": round(w_req + (ts - t_req), 3), "stream": idx, "gap_ms": round(g, 1)}
        for g, ts in zip(gaps, stamps[1:], strict=False)
        if g >= gap_ms
    ]
    out.append(
        {
            "t": round(w_req, 3),
            "stream": idx,
            "ttft_ms": round((stamps[0] - t_req) * 1000, 1),
            "chunks": len(stamps),
            "itl_med_ms": round(sorted(gaps)[len(gaps) // 2], 1) if gaps else 0,
            "gap_max_ms": round(max(gaps), 1) if gaps else 0,
        }
    )
    return out


def run_steady(args: argparse.Namespace) -> int:
    url = urlparse(args.url)
    host, port = url.hostname or "127.0.0.1", url.port or 80
    path = url.path
    if not path.endswith("/completions"):
        path = path.rstrip("/") + "/v1/completions"
    body = {
        "model": args.model,
        "prompt": ("The quick brown fox jumps over the lazy dog. " * 24).strip(),
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    deadline = time.time() + args.duration
    out_lock = threading.Lock()
    with open(args.out, "a", buffering=1) as out:

        def emit(rec: dict[str, Any]) -> None:
            with out_lock:
                out.write(json.dumps(rec) + "\n")

        def worker(idx: int) -> None:
            while not stop.is_set() and time.time() < deadline:
                stamps: list[float] = []
                t_req, w_req = time.monotonic(), time.time()
                try:
                    stream_request(host, port, path, body, stamps.append)
                except OSError as exc:
                    emit({"t": time.time(), "stream": idx, "error": repr(exc)})
                    stop.wait(2.0)
                    continue
                if stamps:
                    for rec in stream_records(stamps, t_req, w_req, idx, args.gap_ms):
                        emit(rec)

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(args.streams)
        ]
        for t in threads:
            t.start()
        while not stop.is_set() and time.time() < deadline:
            time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=30)
    return 0


# ---- the quiet-window and boot-lock protocol -----------------------------------


def recent_quiet(
    quiet_dir: str, max_age_s: float, own_prefix: str, now: float
) -> list[str]:
    """Files in quiet_dir touched within max_age_s that are not ours."""
    out = []
    try:
        names = os.listdir(quiet_dir)
    except OSError:
        return out
    for name in names:
        if own_prefix and name.startswith(own_prefix):
            continue
        try:
            if now - os.stat(f"{quiet_dir}/{name}").st_mtime < max_age_s:
                out.append(name)
        except OSError:
            continue
    return sorted(out)


def take_boot_lock(
    lock_path: str,
    quiet_dir: str,
    own_prefix: str,
    say: Callable[[str], None] = log,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
    poll_s: float = 15.0,
    max_age_s: float = QUIET_MAX_AGE_S,
) -> int:
    """Wait for no recent quiet file, flock boot.lock, check again once held.

    Returns the open fd holding the lock (close it to release).
    """
    while True:
        while quiet := recent_quiet(quiet_dir, max_age_s, own_prefix, now()):
            say(f"quiet window ({', '.join(quiet)}): waiting")
            sleep(poll_s)
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            say(f"{lock_path} is held (a boot is in flight): waiting")
            sleep(poll_s)
            continue
        if recent_quiet(quiet_dir, max_age_s, own_prefix, now()):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            continue
        return fd


# ---- hook: measure a server while a neighbour loads ----------------------------


def vmlck_gib(workers: dict[str, int]) -> dict[str, float | None]:
    """VmLck of each worker in GiB (None if the process is gone)."""
    out: dict[str, float | None] = {}
    for label, pid in sorted(workers.items()):
        out[label] = None
        with (
            contextlib.suppress(OSError, ValueError, IndexError),
            open(f"/proc/{pid}/status") as f,
        ):
            for line in f:
                if line.startswith("VmLck:"):
                    out[label] = round(int(line.split()[1]) / (1 << 20), 3)
    return out


def spawn(cmd: list[str], out: str) -> subprocess.Popen[bytes] | None:
    """A background child with its own session; None if it dies at start."""
    with open(out, "ab") as sink:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=sink,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            log(f"cannot start {cmd[:4]}: {exc}")
            return None
    time.sleep(2)
    if proc.poll() is not None:
        log(f"{cmd[:5]} exited with {proc.returncode}: see {out}")
        return None
    return proc


def stop_child(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


def write_sampler_report(proof: str) -> None:
    report = sibling("pinned_report")
    meta, ticks = report.load_samples(f"{proof}/samples.jsonl")
    raw = f"{proof}/kfd_events.log"
    events = (
        report.parse_kfd_events(raw, meta.get("start")) if os.path.exists(raw) else []
    )
    text = report.build_report(
        meta, ticks, report.load_jsonl(f"{proof}/pagemap.jsonl"), events
    )
    with open(f"{proof}/report.md", "w") as f:
        f.write(text)


def run_hook(args: argparse.Namespace) -> int:
    out_dir = os.environ["OUT"]
    with open(f"{out_dir}/serve.pid") as f:
        spid = int(f.read().split()[0])
    kfd = sibling("kfd_evictions")
    workers = kfd.worker_pids(spid)
    proof = f"{out_dir}/proof"
    os.makedirs(proof, exist_ok=True)
    if not workers:
        log("no VLLM::Worker below the server pid")
        print("FAIL")
        return 1
    log(f"server pid {spid}, workers {workers}, VmLck {vmlck_gib(workers)}")
    # A root child that outlives the hook (sudo cannot relay a SIGKILL) ends
    # on its own after this long.
    hours = f"{args.max_minutes / 60:.3f}"
    sampler = spawn(
        [
            "sudo",
            "-n",
            args.python,
            f"{HERE}/pinned_sampler.py",
            "--out",
            proof,
            "--hours",
            hours,
            "--root-pid",
            str(spid),
        ],
        f"{proof}/sampler.out",
    )
    raw = None
    if args.kfd_raw:
        raw = spawn(
            [
                "sudo",
                "-n",
                *shlex.split(args.kfd_raw),
                f"{proof}/kfd_events.log",
                hours,
            ],
            f"{proof}/kfd_raw.out",
        )
    steady = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "steady",
            "--url",
            f"http://127.0.0.1:{os.environ['PORT']}",
            "--model",
            os.environ["SERVED"],
            "--out",
            f"{proof}/steady.jsonl",
            "--duration",
            str(args.max_minutes * 60),
            "--streams",
            str(args.streams),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    evictions = kfd.Sampler(spid, time.time, period_s=0.5)
    evictions.start()
    vmlck_start = vmlck_gib(workers)
    windows: list[dict[str, Any]] = []
    log(f"idle window {args.idle:.0f} s")
    t0 = time.time()
    time.sleep(args.idle)
    windows.append(
        {"name": "idle", "t0": t0, "t1": time.time(), "vmlck": vmlck_gib(workers)}
    )
    ok = True
    for i in range(1, args.loads + 1):
        if steady.poll() is not None or not os.path.exists(f"/proc/{spid}"):
            log("the server or the load client died")
            ok = False
            break
        fd = take_boot_lock(
            args.boot_lock, args.quiet_dir, args.quiet_own, poll_s=args.poll
        )
        t0 = time.time()
        log(f"load {i}/{args.loads}: {shlex.join(args.neighbour)}")
        rc = subprocess.call(args.neighbour, stdin=subprocess.DEVNULL)
        os.close(fd)
        t_end = time.time()
        log(f"load {i} rc={rc} after {t_end - t0:.0f} s; settling {args.settle:.0f} s")
        time.sleep(args.settle)
        windows.append(
            {
                "name": f"load{i}",
                "t0": t0,
                "t_end": t_end,
                "t1": time.time(),
                "rc": rc,
                "vmlck": vmlck_gib(workers),
            }
        )
        if rc != 0:
            ok = False
            break
        if i < args.loads:
            time.sleep(args.gap)
    evictions.stop()
    for w in windows:
        w["evicted_ms"] = evictions.overlap_ms(w["t0"], w["t1"])
        log(f"window {w['name']}: evicted_ms {w['evicted_ms']} vmlck {w['vmlck']}")
    steady.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        steady.wait(timeout=60)
    stop_child(raw)
    stop_child(sampler)
    with open(f"{proof}/windows.json", "w") as f:
        json.dump(
            {
                "server_pid": spid,
                "workers": workers,
                "vmlck_start": vmlck_start,
                "vmlck_end": vmlck_gib(workers),
                "windows": windows,
                "evicted_samples": evictions.samples,
            },
            f,
            indent=1,
        )
    try:
        write_sampler_report(proof)
        log(f"sampler report: {proof}/report.md")
    except (OSError, ValueError, KeyError) as exc:
        log(f"no sampler report: {exc!r}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---- verdict: compare the arms --------------------------------------------------


def arm_kind(name: str) -> str:
    return "fix" if name.rstrip("0123456789").startswith("fix") else "base"


def parse_lock_line(line: str) -> dict[str, Any] | None:
    """``Locked N of M pinned host mappings (X GiB); VmLck A -> B GiB.``"""
    _, sep, rest = line.partition(LOCK_LINE)
    if not sep:
        return None
    w = rest.split()
    try:
        if w[1] != "of" or w[3:6] != ["pinned", "host", "mappings"] or w[10] != "->":
            return None
        return {
            "locked": int(w[0]),
            "candidates": int(w[2]),
            "gib": float(w[6].lstrip("(")),
            "vmlck_before": float(w[9]),
            "vmlck_after": float(w[11]),
        }
    except (IndexError, ValueError):
        return None


def read_json(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_arm(run: str, name: str) -> dict[str, Any]:
    d = f"{run}/arms/{name}"
    arm: dict[str, Any] = {"name": name, "kind": arm_kind(name), "dir": d}
    arm["proof"] = read_json(f"{d}/proof/windows.json")
    lines: list[str] = []
    summary: list[str] = []
    with contextlib.suppress(OSError):
        with open(f"{d}/serve.log", errors="replace") as f:
            lines = [ln for ln in f if LOCK_LINE in ln and "pinned host" in ln]
        with open(f"{d}/summary.txt", errors="replace") as f:
            summary = f.readlines()
    arm["lock_lines"] = [r for r in map(parse_lock_line, lines) if r]
    arm["cells"] = {}
    for cell in CELLS:
        data = read_json(f"{d}/{cell}.json")
        if data:
            arm["cells"][cell] = {m: data.get(m) for m in METRICS}
    # serve_validate.sh prints "evicted_ms ... INVALID" under each bench cell.
    invalid: set[str] = set()
    current = ""
    for ln in summary:
        for cell in CELLS:
            if f"bench {cell.removeprefix('bench-')}" in ln:
                current = cell
        if "INVALID" in ln and current:
            invalid.add(current)
    arm["invalid_cells"] = sorted(invalid)
    return arm


def window_totals(arm: dict[str, Any], prefix: str) -> dict[str, int]:
    """Per worker, the evicted ms over the arm's windows named ``prefix*``."""
    out: dict[str, int] = {}
    for w in (arm["proof"] or {}).get("windows", []):
        if w["name"].startswith(prefix):
            for label, ms in w.get("evicted_ms", {}).items():
                out[label] = out.get(label, 0) + ms
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def eviction_rows(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for a in arms:
        if a["proof"]:
            loads = window_totals(a, "load")
            rows.append(
                {
                    "arm": a["name"],
                    "kind": a["kind"],
                    "loads_ms": loads,
                    "idle_ms": window_totals(a, "idle"),
                    "load_max_ms": max(loads.values(), default=0),
                    "vmlck_end": a["proof"].get("vmlck_end", {}),
                    "lock_lines": a["lock_lines"],
                }
            )
    return rows


def vmlck_problems(arms: list[dict[str, Any]], tol_gib: float) -> list[str]:
    """VmLck of a worker must not move inside an arm."""
    out = []
    for a in arms:
        per_worker: dict[str, list[float]] = {}
        for w in (a["proof"] or {}).get("windows", []):
            for label, g in w.get("vmlck", {}).items():
                if g is not None:
                    per_worker.setdefault(label, []).append(g)
        for label, vals in per_worker.items():
            if max(vals) - min(vals) > tol_gib:
                out.append(f"{a['name']} {label}: VmLck {min(vals)} -> {max(vals)} GiB")
    return out


def vmlck_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per worker: VmLck of the base arms, of the fix arms, and what the fix logged."""
    fix = [r for r in rows if r["kind"] == "fix"]
    base = [r for r in rows if r["kind"] == "base"]
    out = []
    if not fix or not base:
        return out
    logged = [
        ln["vmlck_after"] - ln["vmlck_before"] for r in fix for ln in r["lock_lines"]
    ]
    for label in sorted(fix[0]["vmlck_end"]):
        f = [r["vmlck_end"].get(label) for r in fix]
        b = [r["vmlck_end"].get(label) for r in base]
        f_ok = [v for v in f if v is not None]
        b_ok = [v for v in b if v is not None]
        if f_ok and b_ok:
            out.append(
                {
                    "worker": label,
                    "base": round(mean(b_ok), 3),
                    "fix": round(mean(f_ok), 3),
                    "delta": round(mean(f_ok) - mean(b_ok), 3),
                    "logged_mean": round(mean(logged), 3),
                }
            )
    return out


def cell_rows(arms: list[dict[str, Any]], tol_pct: float) -> list[dict[str, Any]]:
    """Bench cells of the fix arms against the base arms (valid cells only)."""

    def values(kind: str, cell: str, metric: str) -> list[float]:
        return [
            a["cells"][cell][metric]
            for a in arms
            if a["kind"] == kind
            and cell in a["cells"]
            and cell not in a["invalid_cells"]
            and a["cells"][cell][metric] is not None
        ]

    out = []
    for cell in CELLS:
        for metric in METRICS:
            b, f = values("base", cell, metric), values("fix", cell, metric)
            if not b or not f:
                continue
            spread = 100 * (max(b) - min(b)) / mean(b)
            diff = 100 * (mean(f) - mean(b)) / mean(b)
            out.append(
                {
                    "cell": cell,
                    "metric": metric,
                    "base": round(mean(b), 2),
                    "fix": round(mean(f), 2),
                    "diff_pct": round(diff, 2),
                    "base_spread_pct": round(spread, 2),
                    "ok": abs(diff) <= max(tol_pct, spread),
                }
            )
    return out


def compare(
    arms: list[dict[str, Any]],
    ok_ms: int = 100,
    base_min_ms: int = 500,
    tol_pct: float = 3.0,
    vmlck_tol_gib: float = 0.1,
) -> dict[str, Any]:
    """PROVEN, INCONCLUSIVE or FAILED, with the evidence rows."""
    failed = [
        f"{a['name']}: no proof/windows.json (the hook did not finish)"
        for a in arms
        if not a["proof"]
    ]
    rows = eviction_rows(arms)
    for r in rows:
        if r["kind"] == "fix":
            worst = max(r["load_max_ms"], max(r["idle_ms"].values(), default=0))
            if worst > ok_ms:
                failed.append(
                    f"{r['arm']}: a worker was evicted {worst} ms (> {ok_ms})"
                )
            if not r["lock_lines"]:
                failed.append(
                    f"{r['arm']}: no 'Locked N of M pinned host mappings' line"
                )
            failed += [
                f"{r['arm']}: locked {ln['locked']} of {ln['candidates']} mappings"
                for ln in r["lock_lines"]
                if ln["locked"] != ln["candidates"]
            ]
        elif r["lock_lines"]:
            failed.append(f"{r['arm']}: the base arm locked mappings (env not applied)")
    failed += vmlck_problems(arms, vmlck_tol_gib)
    reproduced = [
        r["arm"]
        for r in rows
        if r["kind"] == "base" and r["load_max_ms"] >= base_min_ms
    ]
    vm = vmlck_rows(rows)
    cells = cell_rows(arms, tol_pct)
    side_effects = [r for r in cells if not r["ok"]]
    side_effects += [
        r for r in vm if abs(r["delta"] - r["logged_mean"]) > vmlck_tol_gib
    ]
    if failed:
        verdict, reasons = "FAILED", failed
    elif not reproduced:
        verdict = "INCONCLUSIVE"
        reasons = [
            (
                f"no base arm reached {base_min_ms} ms of eviction in a worker: "
                "the neighbour loads did not evict the baseline, so the fix has "
                "nothing to remove"
            )
        ]
    elif side_effects:
        verdict = "FAILED"
        reasons = ["side effects: " + json.dumps(side_effects)]
    else:
        verdict = "PROVEN"
        reasons = [
            (
                f"the baseline evicted ({', '.join(reproduced)}); every fix arm "
                f"stayed under {ok_ms} ms, VmLck grew by what the fix logged, "
                f"bench cells are within {tol_pct}% or the base spread"
            )
        ]
    return {
        "verdict": verdict,
        "reasons": reasons,
        "rows": rows,
        "vmlck": vm,
        "cells": cells,
        "reproduced": reproduced,
        "thresholds": {"ok_ms": ok_ms, "base_min_ms": base_min_ms, "tol_pct": tol_pct},
    }


def verdict_text(res: dict[str, Any]) -> str:
    out = [f"# Pinned-host mlock proof: {res['verdict']}", ""]
    out += [f"- {r}" for r in res["reasons"]]
    out += ["", "## Evicted ms per worker", ""]
    out += [
        "| arm | kind | loads | idle | locked, of, GiB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in res["rows"]:
        locked = [(x["locked"], x["candidates"], x["gib"]) for x in r["lock_lines"]]
        out.append(
            f"| {r['arm']} | {r['kind']} | {r['loads_ms']} | {r['idle_ms']} | "
            f"{locked} |"
        )
    out += ["", "## VmLck at the end (GiB)", ""]
    out += [
        "| worker | base | fix | fix - base | logged by the fix |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in res["vmlck"]:
        out.append(
            f"| {r['worker']} | {r['base']} | {r['fix']} | {r['delta']} | "
            f"{r['logged_mean']} |"
        )
    out += ["", "## Bench cells, fix against base", ""]
    out += [
        "| cell | metric | base | fix | diff % | base spread % | ok |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in res["cells"]:
        out.append(
            f"| {r['cell']} | {r['metric']} | {r['base']} | {r['fix']} | "
            f"{r['diff_pct']} | {r['base_spread_pct']} | {'yes' if r['ok'] else 'NO'} |"
        )
    return "\n".join(out) + "\n"


def run_verdict(args: argparse.Namespace) -> int:
    arms = [load_arm(args.run, n) for n in args.arms.split()]
    res = compare(arms, args.ok_ms, args.base_min_ms, args.tol_pct)
    text = verdict_text(res)
    os.makedirs(f"{args.run}/results", exist_ok=True)
    with open(f"{args.run}/results/verdict.md", "w") as f:
        f.write(text)
    with open(f"{args.run}/results/verdict.json", "w") as f:
        json.dump(res, f, indent=1)
    sys.stdout.write(text)
    return {"PROVEN": 0, "INCONCLUSIVE": 3}.get(res["verdict"], 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("neighbour")
    n.add_argument("--workers", type=int, default=2)
    n.add_argument("--shm-gib", type=float, default=47.68, help="per rank (TP2 PLE)")
    n.add_argument("--anon-gib", default="auto", help="GiB per rank, or auto")
    n.add_argument("--overshoot-gib", type=float, default=16.0)
    n.add_argument("--margin-gib", type=float, default=60.0)
    n.add_argument("--hold", type=float, default=30.0, help="seconds after allocating")
    n.add_argument(
        "--lock", action="store_true", help="mlock the shm like the PLE table"
    )
    n.add_argument("--read", action="append", help="checkpoint file or directory")

    s = sub.add_parser("steady")
    s.add_argument("--url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--duration", type=float, default=3600.0)
    s.add_argument("--streams", type=int, default=4)
    s.add_argument("--max-tokens", type=int, default=256)
    s.add_argument("--gap-ms", type=float, default=250.0)

    h = sub.add_parser("hook")
    h.add_argument("--loads", type=int, default=3)
    h.add_argument("--idle", type=float, default=60.0)
    h.add_argument("--gap", type=float, default=90.0)
    h.add_argument("--settle", type=float, default=45.0)
    h.add_argument("--streams", type=int, default=4)
    h.add_argument(
        "--max-minutes",
        type=int,
        default=120,
        help="life span of the sampler, the raw reader and the load client",
    )
    h.add_argument("--python", default=sys.executable, help="python with numpy")
    h.add_argument("--kfd-raw", default="", help="command of the raw KFD reader")
    h.add_argument("--boot-lock", required=True)
    h.add_argument("--quiet-dir", required=True)
    h.add_argument("--quiet-own", default="EvictFix")
    h.add_argument("--poll", type=float, default=15.0)
    h.add_argument("neighbour", nargs=argparse.REMAINDER, help="-- command of one load")

    v = sub.add_parser("verdict")
    v.add_argument("--run", required=True)
    v.add_argument("--arms", required=True)
    v.add_argument("--ok-ms", type=int, default=100)
    v.add_argument("--base-min-ms", type=int, default=500)
    v.add_argument("--tol-pct", type=float, default=3.0)

    args = ap.parse_args(argv)
    if args.cmd == "hook":
        args.neighbour = [a for a in args.neighbour if a != "--"]
        if not args.neighbour:
            ap.error("hook needs the command of one neighbour load after --")
    handlers = {
        "neighbour": run_neighbour,
        "steady": run_steady,
        "hook": run_hook,
        "verdict": run_verdict,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
