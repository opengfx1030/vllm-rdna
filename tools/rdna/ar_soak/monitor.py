#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM115
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Background monitor of one long serve (the rdna_ar soak).

  monitor.py SERVE_PID SERVE_LOG OUTDIR [--period 2]

Writes under OUTDIR:
  events.log    timestamped, one line per event: rdna_ar status lines, KFD
                eviction jumps per worker, new dmesg lines, neighbour server
                processes appearing or leaving, boot.lock and lease changes
  counts.json   running counters (rewritten every period)
  meminfo.csv   host memory and reclaim counters every 5 s
  capture-N/    dmesg, process list and the serve-log tail, written at the
                first critical event of each kind
  FAIL.<kind>   marker for a critical event (wedge, disabled, fault, died)

Critical: rdna_ar WEDGED/wedged, "rdna_ar: disabled", amdgpu fault / reset /
AER lines in dmesg, a Traceback or Memory Fault in the serve log, the server
process gone. Warnings: late peer, eviction jumps, other rdna_ar lines.
Stdlib only; reads sudo -n dmesg (box policy).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.join(_HERE, "..", "port_v031"),
    os.path.join(os.environ.get("TREE_T", ""), "tools/rdna/port_v031"),
):
    sys.path.insert(0, _p)
import kfd_evictions as kfd  # noqa: E402

HOME = os.path.expanduser("~")
LEASE = f"{HOME}/w4a8_runs/lanes/lease"
BOOT_LOCK = f"{HOME}/w4a8_runs/boot.lock"

SERVE_PATTERNS = [
    ("rdna_ar_wedged", re.compile(r"rdna_ar: WEDGED|rdna_ar wedged"), True),
    ("rdna_ar_disabled", re.compile(r"rdna_ar: disabled"), True),
    ("rdna_ar_late_peer", re.compile(r"rdna_ar: late peer"), False),
    ("rdna_ar_evicted", re.compile(r"rdna_ar:.*evicted for"), False),
    ("rdna_ar_other", re.compile(r"rdna_ar:(?!.*active)"), False),
    ("traceback", re.compile(r"Traceback \(most recent"), True),
    (
        "gpu_fault",
        re.compile(r"Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault"),
        True,
    ),
    ("runlist", re.compile(r"Runlist is getting oversubscribed"), False),
    (
        "timeout",
        re.compile(r"timed out|TimeoutError|execute_model timeout", re.I),
        False,
    ),
    ("engine_dead", re.compile(r"EngineDeadError|EngineCore.*(died|failed)"), True),
]
DMESG_FAULT = re.compile(
    r"amdgpu.*(fault|timeout|reset|hang|UTCL2|uncorrectable)|"
    r"(ecc|ras)\b.*(error|uncorrectable)|AER:|PCIe Bus Error|"
    r"Hardware Error|pcieport.*error|Oops|BUG:|Kernel panic|NMI",
    re.I,
)
DMESG_BORING = re.compile(r"uses VM inv eng")
NEIGHBOUR_KEYS = re.compile(
    r"serve_rdna|vllm\.entrypoints|vllm serve|VLLM::|g0a|g0b|serve_validate|"
    r"EngineCore|from multiprocessing"
)


def now() -> str:
    return time.strftime("%H:%M:%S")


class Monitor:
    def __init__(
        self, serve_pid: int, serve_log: str, outdir: str, family: int
    ) -> None:
        self.t_start = time.time()
        self.family = family
        self.pid = serve_pid
        self.log_path = serve_log
        self.out = outdir
        os.makedirs(outdir, exist_ok=True)
        self.ev = open(f"{outdir}/events.log", "a", buffering=1)
        self.mem = open(f"{outdir}/meminfo.csv", "a", buffering=1)
        if self.mem.tell() == 0:
            self.mem.write(
                "time,memfree_kb,cached_kb,shmem_kb,pgscan_kswapd,pgscan_direct,"
                "pgmigrate_success,compact_stall,allocstall_normal\n"
            )
        self.counts: dict[str, int] = {}
        self.offset = 0
        self.evict_prev: dict[str, int] = {}
        self.evict_total: dict[str, int] = {}
        self.evict_max_jump: dict[str, int] = {}
        self.dmesg_seen: dict[str, int] = {}
        self.neigh: dict[int, str] = {}
        self.lock_state: str | None = None
        self.lease_state = ""
        self.captures = 0
        self.captured: set[str] = set()
        self.t_mem = 0.0
        self.t_dmesg = 0.0
        self.t_neigh = 0.0
        self.t_beat = 0.0
        self.boot_lock_ino = (
            os.stat(BOOT_LOCK).st_ino if os.path.exists(BOOT_LOCK) else -1
        )

    def event(self, kind: str, msg: str) -> None:
        self.ev.write(f"{time.strftime('%F %T')} {kind} {msg}\n")
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def capture(self, kind: str) -> None:
        if kind in self.captured:
            return
        self.captured.add(kind)
        self.captures += 1
        d = f"{self.out}/capture-{self.captures}-{kind}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/dmesg.txt", "w") as f:
            subprocess.run(
                ["sudo", "-n", "dmesg", "--time-format", "iso"], stdout=f, check=False
            )
        with open(f"{d}/ps.txt", "w") as f:
            subprocess.run(
                ["ps", "-eo", "pid,ppid,pgid,stat,etimes,rss,args"],
                stdout=f,
                check=False,
            )
        try:
            with (
                open(self.log_path, "rb") as src,
                open(f"{d}/serve-tail.txt", "wb") as dst,
            ):
                src.seek(max(0, os.path.getsize(self.log_path) - 400_000))
                dst.write(src.read())
        except OSError:
            pass
        with open(f"{d}/meminfo.txt", "w") as f:
            f.write(open("/proc/meminfo").read())
        with open(f"{d}/kfd.json", "w") as f:
            json.dump(
                {"snapshot": kfd.snapshot(self.pid), "total": self.evict_total}, f
            )
        self.event("CAPTURE", d)

    def critical(self, kind: str) -> None:
        open(f"{self.out}/FAIL.{kind}", "a").write(f"{time.strftime('%F %T')}\n")
        self.capture(kind)

    # --- serve log ---
    def scan_serve_log(self) -> None:
        try:
            size = os.path.getsize(self.log_path)
        except OSError:
            return
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return
        with open(self.log_path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        # only whole lines; the rest is read next period
        end = data.rfind(b"\n") + 1
        self.offset += end
        for raw in data[:end].decode(errors="replace").splitlines():
            for kind, rx, crit in SERVE_PATTERNS:
                if rx.search(raw):
                    self.event(kind, raw.strip()[:400])
                    if crit:
                        self.critical(kind)
                    break

    # --- KFD evictions ---
    def scan_evictions(self) -> None:
        snap = kfd.snapshot(self.pid)
        for w, ms in snap.items():
            prev = self.evict_prev.get(w, ms)
            if ms > prev:
                jump = ms - prev
                self.evict_total[w] = self.evict_total.get(w, 0) + jump
                self.evict_max_jump[w] = max(self.evict_max_jump.get(w, 0), jump)
                self.event("kfd_evict", f"{w} +{jump} ms (total {ms})")
            self.evict_prev[w] = ms
            self.evict_total.setdefault(w, 0)

    # --- dmesg ---
    def dmesg_old(self, ln: str) -> bool:
        """True for a line stamped before the monitor started."""
        try:
            ts = datetime.datetime.fromisoformat(ln[:32].replace(",", "."))
        except ValueError:
            return False
        return ts.timestamp() < self.t_start - 2

    def scan_dmesg(self) -> None:
        r = subprocess.run(
            ["sudo", "-n", "dmesg", "--time-format", "iso"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = r.stdout.splitlines()
        if r.returncode != 0 or not lines:
            self.event("monitor_error", f"dmesg rc={r.returncode} lines={len(lines)}")
            return
        first = not self.dmesg_seen
        # dmesg's iso stamps jitter by microseconds between calls (the monotonic
        # to wall conversion drifts): count messages without the stamp instead
        cur: dict[str, int] = {}
        for ln in lines:
            msg = ln.split(" ", 1)[-1]
            cur[msg] = cur.get(msg, 0) + 1
            if cur[msg] <= self.dmesg_seen.get(msg, 0):
                continue
            self.dmesg_seen[msg] = cur[msg]
            if first or self.dmesg_old(ln):
                continue
            tagged = bool(DMESG_FAULT.search(ln)) and not DMESG_BORING.search(ln)
            if tagged or re.search(
                r"amdgpu|kfd|nvme|oom|Out of memory|hung task", ln, re.I
            ):
                self.event("dmesg_fault" if tagged else "dmesg", ln[:300])
                if tagged:
                    self.critical("dmesg_fault")

    # --- neighbours ---
    def scan_neighbours(self) -> None:
        mine: set[int] = set()
        kids: dict[int, list[int]] = {}
        cmd: dict[int, str] = {}
        for e in os.listdir("/proc"):
            if not e.isdigit():
                continue
            try:
                stat = open(f"/proc/{e}/stat").read()
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
                c = (
                    open(f"/proc/{e}/cmdline", "rb")
                    .read()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                )
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(e))
            cmd[int(e)] = c
        todo = [self.pid, self.family]
        while todo:
            p = todo.pop()
            mine.add(p)
            todo.extend(kids.get(p, []))
        me = {os.getpid(), os.getppid()}
        cur = {}
        for p, c in cmd.items():
            if p in mine or p in me or not NEIGHBOUR_KEYS.search(c):
                continue
            if (
                "ar_soak" in c
                or "tempwatch" in c
                or "monitor.py" in c
                or "driver.sh" in c
            ):
                continue
            cur[p] = c
        for p, c in cur.items():
            if p not in self.neigh:
                hip = ""
                try:
                    env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
                    hip = next(
                        (
                            x.decode()
                            for x in env
                            if x.startswith(b"HIP_VISIBLE_DEVICES=")
                        ),
                        "",
                    )
                except OSError:
                    pass
                self.event("neighbour_up", f"pid {p} {hip} {c[:160]}")
        for p in list(self.neigh):
            if p not in cur:
                self.event("neighbour_gone", f"pid {p} {self.neigh[p][:100]}")
        self.neigh = cur

    def scan_locks(self) -> None:
        state = "free"
        try:
            for ln in open("/proc/locks"):
                if f":{self.boot_lock_ino} " in ln or ln.rstrip().endswith(
                    f":{self.boot_lock_ino} 0 EOF"
                ):
                    state = "held"
                    break
        except OSError:
            pass
        if state != self.lock_state:
            self.event("boot_lock", state)
            self.lock_state = state
        try:
            ls = " ".join(
                sorted(
                    f
                    for f in os.listdir(LEASE)
                    if f != "README.md" and "README" not in f
                )
            )
            q = " ".join(sorted(os.listdir(f"{HOME}/w4a8_runs/lanes/quiet")))
        except OSError:
            return
        st = f"lease[{ls}] quiet[{q}]"
        if st != self.lease_state:
            self.event("lease", st)
            self.lease_state = st

    def scan_mem(self) -> None:
        mi = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            mi[k] = int(v.split()[0])
        vm = {}
        for ln in open("/proc/vmstat"):
            k, v = ln.split()
            vm[k] = int(v)
        self.mem.write(
            f"{now()},{mi['MemFree']},{mi['Cached']},{mi['Shmem']},"
            f"{vm.get('pgscan_kswapd', 0)},"
            f"{vm.get('pgscan_direct', 0)},{vm.get('pgmigrate_success', 0)},"
            f"{vm.get('compact_stall', 0)},{vm.get('allocstall_normal', 0)}\n"
        )

    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    def run(self, period: float) -> None:
        self.event("monitor_start", f"pid {self.pid} log {self.log_path}")
        died = False
        while True:
            t = time.time()
            try:
                self.scan_serve_log()
                self.scan_evictions()
                if t - self.t_neigh >= 4:
                    self.scan_neighbours()
                    self.scan_locks()
                    self.t_neigh = t
                if t - self.t_dmesg >= 20:
                    self.scan_dmesg()
                    self.t_dmesg = t
                if t - self.t_mem >= 5:
                    self.scan_mem()
                    self.t_mem = t
                if t - self.t_beat >= 300:
                    self.event(
                        "heartbeat",
                        f"evict_total={self.evict_total} "
                        f"max_jump={self.evict_max_jump}",
                    )
                    self.t_beat = t
            except Exception as e:  # keep monitoring whatever happens
                self.event("monitor_error", repr(e))
            json.dump(
                {
                    "counts": self.counts,
                    "evict_total_ms": self.evict_total,
                    "evict_max_jump_ms": self.evict_max_jump,
                    "updated": time.strftime("%F %T"),
                },
                open(f"{self.out}/counts.json.tmp", "w"),
            )
            os.replace(f"{self.out}/counts.json.tmp", f"{self.out}/counts.json")
            if not self.alive() and not died:
                died = True
                self.scan_serve_log()
                self.event("server_gone", f"pid {self.pid}")
                self.critical("server_gone")
            if died:
                return
            if os.path.exists(f"{self.out}/STOP"):
                self.event("monitor_stop", "STOP file")
                return
            time.sleep(period)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("serve_pid", type=int)
    ap.add_argument("serve_log")
    ap.add_argument("outdir")
    ap.add_argument("--period", type=float, default=2.0)
    ap.add_argument(
        "--family",
        type=int,
        default=0,
        help="driver pid: its whole process tree is ours, not a neighbour",
    )
    a = ap.parse_args()
    Monitor(a.serve_pid, a.serve_log, a.outdir, a.family or a.serve_pid).run(a.period)
