#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G0b transport probe: one GPU to one GPU across two processes.

docs/rdna2/disagg-g0-2026-11-06.md. One process per role, started by g0b_run.sh with the
same --cells:

  g0b_xfer_probe.py --role src --run-dir D --cells WRITE:contig:249.1M:100,...
  g0b_xfer_probe.py --role dst --run-dir D --cells WRITE:contig:249.1M:100,...

``src`` stands for the prefill rank (HIP 4, root complex c0), ``dst`` for the decode
rank (HIP 6, root complex 00). A cell is ``OP:LAYOUT:SIZE:ITERS``:

  OP      WRITE  push, initiated by src: src's GPU writes dst's memory (posted writes)
          READ   pull, initiated by dst: dst's GPU reads src's memory (peer reads)
          NOTIF  control-plane latency only (a timestamped message src -> dst, echoed)
  LAYOUT  contig   one window of SIZE bytes
          scatter  256 windows of SIZE/256 at random distinct slots of a 4x larger
                   pool, different slots on the two sides (P and D block tables differ)
          strided  one strided row (stride 4 x window) that NIXL expands to 256 windows
  SIZE    4K 64K 1M 249.1M (16k-token per-rank state) 912.1M (64k-token per-rank state)

Every iteration: refresh the source pattern (XOR of a closed-form base with a
per-iteration salt, generated on the GPU), zero the destination, transfer, and compare
on the receiving GPU, window by window, plus a count of non-zero words over the whole
pool to catch writes outside the windows. The notification that signals completion is
the transport's own (NIXL notif); the fallbacks use files.

Engines: nixl (the vLLM connector's library, UCX backend: rocm_ipc / rocm_copy), hipipc
(hipIpcGetMemHandle + hipMemcpyAsync on the mapped pointer: fallback B), hostbounce
(device -> pinned host -> /dev/shm -> pinned host -> device: fallback D).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "port_v031"))
import g0_hip  # noqa: E402
import kfd_evictions  # noqa: E402

MIB = 1 << 20
M64 = (1 << 64) - 1
N_WINDOWS = 256
BOOT_LOCK = os.path.expanduser("~/w4a8_runs/boot.lock")
QUIET_DIR = os.path.expanduser("~/w4a8_runs/lanes/quiet")
DOCKER_BUILD = r"^(/usr/sbin/|/usr/bin/)?docker buildx build"
WINDOW_MAX_S = 12 * 60


def now() -> float:
    return time.perf_counter()


def parse_size(tok: str) -> int:
    mult = {"K": 1 << 10, "M": MIB, "G": 1 << 30}[tok[-1].upper()]
    return int(float(tok[:-1]) * mult) // 4096 * 4096


def parse_cells(spec: str) -> list[tuple[str, str, int, int]]:
    cells = []
    for tok in spec.split(","):
        op, layout, size, iters = tok.split(":")
        assert op in ("WRITE", "READ", "NOTIF"), tok
        assert layout in ("contig", "scatter", "strided", "-"), tok
        cells.append((op, layout, parse_size(size) if size != "0" else 0, int(iters)))
    return cells


# --- layouts ---


@dataclass
class Layout:
    kind: str
    payload: int
    pool: int
    windows: list[tuple[int, int]]  # (offset in the arena, length), payload order
    strided: (
        tuple[int, int, int] | None
    )  # (offset, stride, count) of the single Nx5 row

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "pool": self.pool,
            "windows": self.windows,
            "strided": self.strided,
        }

    @staticmethod
    def from_json(d: dict) -> Layout:
        return Layout(
            d["kind"],
            d["payload"],
            d["pool"],
            [tuple(w) for w in d["windows"]],
            tuple(d["strided"]) if d["strided"] else None,
        )


def window_len(size: int) -> int:
    return size // N_WINDOWS // 4096 * 4096


def pool_bytes(kind: str, size: int) -> int:
    if kind == "contig":
        return size
    return (4 * N_WINDOWS * window_len(size) + MIB - 1) // MIB * MIB


def build_layout(kind: str, size: int, seed: int) -> Layout:
    if kind == "contig":
        return Layout(kind, size, size, [(0, size)], None)
    ln = window_len(size)
    assert ln >= 4096, f"{kind} needs SIZE >= 1M"
    pool = pool_bytes(kind, size)
    if kind == "strided":
        stride = 4 * ln
        wins = [(j * stride, ln) for j in range(N_WINDOWS)]
        return Layout(kind, N_WINDOWS * ln, pool, wins, (0, stride, N_WINDOWS))
    slots = sorted(random.Random(seed).sample(range(pool // ln), N_WINDOWS))
    return Layout(kind, N_WINDOWS * ln, pool, [(s * ln, ln) for s in slots], None)


# --- patterns ---

_SM1 = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
_SM2 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_SM3 = -7723592293110705685  # 0x94D049BB133111EB


def make_base(nbytes: int, device: torch.device) -> torch.Tensor:
    """Closed-form payload: the same bytes on every device (hash of the index)."""
    t = torch.empty(nbytes, dtype=torch.uint8, device=device)
    v = t.view(torch.int64)
    n = v.numel()
    chunk = 1 << 22
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        x = torch.arange(s, e, dtype=torch.int64, device=device)
        x = x * _SM1
        x = x ^ (x >> 29)
        x = x * _SM2
        x = x ^ (x >> 32)
        v[s:e] = x * _SM3
    return t


def salt_of(i: int, seed: int) -> int:
    z = (seed + (i + 1) * 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    z ^= z >> 31
    return z - (1 << 64) if z >= 1 << 63 else z


def fill(arena: torch.Tensor, lay: Layout, base: torch.Tensor, salt: int) -> None:
    a64, b64 = arena.view(torch.int64), base.view(torch.int64)
    pos = 0
    for off, ln in lay.windows:
        torch.bitwise_xor(
            b64[pos // 8 : (pos + ln) // 8], salt, out=a64[off // 8 : (off + ln) // 8]
        )
        pos += ln


def verify(
    arena: torch.Tensor, lay: Layout, base: torch.Tensor, salt: int
) -> tuple[bool, int]:
    """Byte-exact compare of every window; True when every window matches and every
    word of the pool outside the windows is still zero. Returns (ok, bad windows)."""
    a64, b64 = arena.view(torch.int64), base.view(torch.int64)
    dev = arena.device
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    nz_windows = torch.zeros((), dtype=torch.int64, device=dev)
    pos = 0
    chunk = 4 * MIB  # int64 words per compare chunk (32 MiB)
    for off, ln in lay.windows:
        w = a64[off // 8 : (off + ln) // 8]
        e = b64[pos // 8 : (pos + ln) // 8]
        mism = torch.zeros((), dtype=torch.bool, device=dev)
        for s in range(0, w.numel(), chunk):
            mism |= (w[s : s + chunk] != (e[s : s + chunk] ^ salt)).any()
        bad += mism.to(torch.int64)
        nz_windows += torch.count_nonzero(w)
        pos += ln
    n_bad = int(bad)
    gaps_clean = True
    if lay.kind != "contig":
        gaps_clean = int(torch.count_nonzero(a64[: lay.pool // 8])) == int(nz_windows)
    return n_bad == 0 and gaps_clean, n_bad + (0 if gaps_clean else 1)


def device_sync(mem: str) -> None:
    if mem == "gpu":
        torch.accelerator.synchronize()


# --- control channel ---


class Ctl:
    """Files in the run dir: rendezvous, barriers, results, and the notif queue of the
    fallback engines."""

    def __init__(self, run_dir: Path, role: str, timeout: float) -> None:
        self.dir = run_dir
        self.role = role
        self.peer = "dst" if role == "src" else "src"
        self.timeout = timeout
        self._sent = 0
        self._recv = 0

    def put(self, name: str, data: bytes) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self.dir / name)

    def get(self, name: str, timeout: float | None = None) -> bytes:
        p = self.dir / name
        t0 = now()
        while not p.exists():
            if now() - t0 > (timeout or self.timeout):
                raise TimeoutError(f"{self.role}: {name} never appeared")
            time.sleep(0.01)
        return p.read_bytes()

    def put_json(self, name: str, obj: object) -> None:
        self.put(name, json.dumps(obj).encode())

    def get_json(self, name: str, timeout: float | None = None) -> object:
        return json.loads(self.get(name, timeout))

    def barrier(self, tag: str, timeout: float | None = None) -> None:
        self.put(f"bar.{tag}.{self.role}", b"1")
        self.get(f"bar.{tag}.{self.peer}", timeout)

    def msg_send(self, msg: bytes) -> None:
        self.put(f"m.{self.peer}.{self._sent:08d}", msg)
        self._sent += 1

    def msg_recv(self) -> list[bytes]:
        out = []
        while (self.dir / f"m.{self.role}.{self._recv:08d}").exists():
            out.append((self.dir / f"m.{self.role}.{self._recv:08d}").read_bytes())
            self._recv += 1
        return out


# --- engines ---


class FileNotify:
    """Notifications through the control channel (fallback engines)."""

    ctl: Ctl

    def notify(self, msg: bytes) -> None:
        self.ctl.msg_send(msg)

    def recv(self) -> list[bytes]:
        return self.ctl.msg_recv()

    def after_done(self, notif: bytes) -> None:
        self.ctl.msg_send(notif)

    def receive(self, lay: Layout) -> None:
        return None

    def telemetry(self, h: object) -> dict:
        return {}


class NixlEngine:
    name = "nixl"

    def __init__(
        self, role: str, ctl: Ctl, arena: torch.Tensor, dev_index: int, mem: str
    ) -> None:
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError:
            from nixl_rocm._api import nixl_agent, nixl_agent_config
        self.ctl = ctl
        self.memtype = "VRAM" if mem == "gpu" else "DRAM"
        cfg = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            capture_telemetry=True,
            backends=["UCX"],
        )
        self.agent = nixl_agent(f"g0b-{role}", cfg)
        self.dev_index = dev_index
        self.arena_addr = arena.data_ptr()
        self.reg = self.agent.register_memory(arena)
        ctl.put(f"md.{role}", self.agent.get_agent_metadata())
        ctl.put_json(f"arena.{role}", {"addr": self.arena_addr, "dev": dev_index})
        self.peer_name = self.agent.add_remote_agent(ctl.get(f"md.{ctl.peer}"))
        self.peer_arena = ctl.get_json(f"arena.{ctl.peer}")
        self.agent.make_connection(self.peer_name)
        ctl.barrier("connected")
        self.pending: list[bytes] = []
        self._dlists: list[object] = []

    @staticmethod
    def _rows(lay: Layout, addr: int, dev: int) -> np.ndarray:
        if lay.strided:
            off, stride, count = lay.strided
            ln = lay.windows[0][1]
            return np.array([[addr + off, ln, dev, stride, count]], dtype=np.uint64)
        return np.array(
            [[addr + off, ln, dev, ln, 1] for off, ln in lay.windows], dtype=np.uint64
        )

    def prepare(
        self, op: str, mine: Layout, theirs: Layout
    ) -> tuple[object, object, np.ndarray]:
        lh = self.agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            self._rows(mine, self.arena_addr, self.dev_index),
            self.memtype,
        )
        rh = self.agent.prep_xfer_dlist(
            self.peer_name,
            self._rows(theirs, self.peer_arena["addr"], self.peer_arena["dev"]),
            self.memtype,
        )
        self._dlists += [lh, rh]
        return lh, rh, np.arange(len(mine.windows), dtype=np.int32)

    def make(self, op: str, prep: tuple[object, object, np.ndarray]) -> object:
        lh, rh, ids = prep
        return self.agent.make_prepped_xfer(op, lh, ids, rh, ids)

    def post(self, h: object, notif: bytes) -> str:
        return self.agent.transfer(h, notif)

    def poll(self, h: object) -> str:
        return self.agent.check_xfer_state(h)

    def release(self, h: object) -> None:
        self.agent.release_xfer_handle(h)

    def after_done(self, notif: bytes) -> None:
        return None  # NIXL sends the notif itself after the transfer

    def receive(self, lay: Layout) -> None:
        return None

    def notify(self, msg: bytes) -> None:
        self.agent.send_notif(self.peer_name, msg)

    def recv(self) -> list[bytes]:
        out: list[bytes] = []
        for msgs in self.agent.get_new_notifs().values():  # one peer only
            out += msgs
        return out

    def telemetry(self, h: object) -> dict:
        try:
            t = self.agent.get_xfer_telemetry(h)
            return {
                "xfer_us": t.xferDuration,
                "post_us": t.postDuration,
                "bytes": t.totalBytes,
                "descs": t.descCount,
            }
        except Exception as e:  # noqa: BLE001 - telemetry is optional evidence
            return {"error": repr(e)}

    def backend(self, h: object) -> str:
        try:
            return self.agent.query_xfer_backend(h)
        except Exception as e:  # noqa: BLE001
            return repr(e)

    def close(self) -> None:
        for d in self._dlists:
            with contextlib.suppress(Exception):
                self.agent.release_dlist_handle(d)


class HipIpcEngine(FileNotify):
    """Fallback B: hipIpcGetMemHandle on the arena, hipMemcpyAsync on the pointer."""

    name = "hipipc"

    def __init__(
        self, role: str, ctl: Ctl, arena: torch.Tensor, dev_index: int, mem: str
    ) -> None:
        assert mem == "gpu"
        self.ctl = ctl
        self.arena_addr = arena.data_ptr()
        ctl.put(f"ipc.{role}", g0_hip.ipc_get_handle(self.arena_addr))
        self.peer_base = g0_hip.ipc_open(ctl.get(f"ipc.{ctl.peer}"))
        self.stream = g0_hip.stream_create()

    def prepare(
        self, op: str, mine: Layout, theirs: Layout
    ) -> list[tuple[int, int, int]]:
        if op == "WRITE":  # src: local window -> remote window
            return [
                (self.peer_base + to, self.arena_addr + mo, ln)
                for (mo, ln), (to, _) in zip(mine.windows, theirs.windows)
            ]
        return [  # READ, dst: remote window -> local window
            (self.arena_addr + mo, self.peer_base + to, ln)
            for (mo, ln), (to, _) in zip(mine.windows, theirs.windows)
        ]

    def make(
        self, op: str, prep: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]]:
        return prep

    def post(self, h: list[tuple[int, int, int]], notif: bytes) -> str:
        for dst, src, ln in h:
            g0_hip.memcpy_async(dst, src, ln, self.stream)
        return "PROC"

    def poll(self, h: object) -> str:
        return "DONE" if g0_hip.stream_done(self.stream) else "PROC"

    def release(self, h: object) -> None:
        return None

    def close(self) -> None:
        return None


class HostBounceEngine(FileNotify):
    """Fallback D: device -> pinned host -> /dev/shm -> pinned host -> device."""

    name = "hostbounce"

    def __init__(
        self, role: str, ctl: Ctl, arena: torch.Tensor, dev_index: int, mem: str
    ) -> None:
        self.ctl = ctl
        self.arena = arena
        self.mem = mem
        nbytes = int(os.environ["G0B_MAX_PAYLOAD"])
        self.wire_path = f"/dev/shm/g0b-{ctl.dir.name}-wire"
        if role == "src":
            with open(self.wire_path, "wb") as f:
                f.truncate(nbytes)
        ctl.barrier("wire")
        self.wire = np.memmap(
            self.wire_path, dtype=np.uint8, mode="r+", shape=(nbytes,)
        )
        self.pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=(mem == "gpu"))

    def prepare(self, op: str, mine: Layout, theirs: Layout) -> Layout:
        if op != "WRITE":
            raise NotImplementedError("hostbounce has no pull direction")
        return mine

    def make(self, op: str, prep: Layout) -> Layout:
        return prep

    def post(self, h: Layout, notif: bytes) -> str:  # src: gather -> pinned -> shm
        pos = 0
        for off, ln in h.windows:
            self.pinned[pos : pos + ln].copy_(
                self.arena[off : off + ln], non_blocking=True
            )
            pos += ln
        device_sync(self.mem)
        self.wire[: h.payload] = self.pinned[: h.payload].numpy()
        return "DONE"

    def poll(self, h: object) -> str:
        return "DONE"

    def receive(self, lay: Layout) -> None:  # dst: shm -> pinned -> scatter
        self.pinned[: lay.payload].numpy()[:] = self.wire[: lay.payload]
        pos = 0
        for off, ln in lay.windows:
            self.arena[off : off + ln].copy_(
                self.pinned[pos : pos + ln], non_blocking=True
            )
            pos += ln
        device_sync(self.mem)

    def release(self, h: object) -> None:
        return None

    def close(self) -> None:
        return None


ENGINES = {"nixl": NixlEngine, "hipipc": HipIpcEngine, "hostbounce": HostBounceEngine}


# --- evidence ---


def kfd_queues(pid: int) -> dict[str, int]:
    comp = sdma = 0
    for d in Path(f"{kfd_evictions.KFD_PROC}/{pid}/queues").glob("*"):
        try:
            kind = int((d / "type").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if kind == 0:
            comp += 1
        elif kind in (1, 4):
            sdma += 1
    return {"compute": comp, "sdma": sdma}


def evidence() -> dict:
    pid = os.getpid()
    return {
        "evicted_ms": kfd_evictions.evicted_ms(pid),
        "queues": kfd_queues(pid),
        "rss_mib": round(
            int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / MIB, 1
        ),
    }


def window_gate(owner: str, state: dict) -> None:
    """Measured-window protocol: quiet/<owner> touched (<= 15 min, re-touched), no
    docker buildx build, boot.lock free. A window older than 12 min is closed for 30 s
    so that waiting boots can proceed."""
    qpath = Path(QUIET_DIR) / owner
    if state.get("t_open") and now() - state["t_open"] > WINDOW_MAX_S:
        qpath.unlink(missing_ok=True)
        print(
            f"[window] closing the quiet window for 30 s after {WINDOW_MAX_S} s",
            flush=True,
        )
        time.sleep(30)
        state["t_open"] = 0.0
    waited = 0.0
    while True:
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.touch()
        if not state.get("t_open"):
            state["t_open"] = now()
        build = (
            subprocess.run(
                ["pgrep", "-f", DOCKER_BUILD], capture_output=True
            ).returncode
            == 0
        )
        fd = os.open(BOOT_LOCK, os.O_RDONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_free = True
        except BlockingIOError:
            lock_free = False
        finally:
            os.close(fd)
        if lock_free and not build:
            return
        if waited % 60 < 5:
            print(
                f"[window] waiting: boot.lock free={lock_free} docker build={build}",
                flush=True,
            )
        time.sleep(5)
        waited += 5


# --- cell loop ---


def stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]  # noqa: E731
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "p05": q(0.05),
        "p95": q(0.95),
        "p99": q(0.99),
        "min": s[0],
        "max": s[-1],
    }


class Peer:
    """Notification waits with a queue of messages that arrived early."""

    def __init__(self, engine: object, timeout: float) -> None:
        self.e = engine
        self.timeout = timeout
        self.q: list[bytes] = []

    def wait(self, prefix: bytes) -> tuple[bytes, float]:
        t0 = now()
        while True:
            for i, m in enumerate(self.q):
                if m.startswith(prefix):
                    return self.q.pop(i), now()
            self.q += self.e.recv()
            if self.q and any(m.startswith(prefix) for m in self.q):
                continue
            if now() - t0 > self.timeout:
                raise TimeoutError(
                    f"no notification {prefix!r} in {self.timeout} s "
                    f"(queued {self.q[:3]})"
                )


def run_xfer_cell(a, eng, peer: Peer, arena, base_cache, cell, k: int) -> dict:
    op, kind, size, iters = cell
    role, mem = a.role, a.mem
    seed = 1 if role == "src" else 2
    mine = build_layout(kind, size, seed)
    a.ctl.put_json(f"layout.{k}.{role}", mine.to_json())
    theirs = Layout.from_json(a.ctl.get_json(f"layout.{k}.{a.ctl.peer}"))
    if mine.payload not in base_cache:
        base_cache.clear()
        base_cache[mine.payload] = make_base(mine.payload, arena.device)
    base = base_cache[mine.payload]
    t_prep = now()
    prep = eng.prepare(op, mine, theirs)
    prep_ms = (now() - t_prep) * 1e3
    initiator = (op == "WRITE" and role == "src") or (op == "READ" and role == "dst")
    rec: dict = {
        "cell": f"{op}:{kind}:{size}:{iters}",
        "op": op,
        "layout": kind,
        "bytes": mine.payload,
        "windows": len(mine.windows),
        "initiator": initiator,
        "prep_ms": round(prep_ms, 3),
        "dur_ms": [],
        "post_ms": [],
        "oneway_ms": [],
        "make_ms": [],
        "verify_ok": 0,
        "verify_bad": 0,
        "iters": iters,
        "warmup": a.warmup,
        "telemetry": [],
    }
    ev0 = evidence()
    a.ctl.barrier(f"cell{k}.start")
    for i in range(-a.warmup, iters):
        tag = str(i).encode()
        salt = salt_of(i, 0x5EED0000 + k)
        keep = i >= 0
        if op == "WRITE":
            if role == "dst":
                arena[: mine.pool].zero_()
                device_sync(mem)
                eng.notify(b"R|" + tag + b"|")
                msg, t_arr = peer.wait(b"D|" + tag + b"|")
                eng.receive(mine)
                t_done = now()
                t0 = float(msg.split(b"|")[2])
                ok, bad = verify(arena, mine, base, salt)
                if keep:
                    rec["oneway_ms"].append((t_arr - t0) * 1e3)
                    rec["dur_ms"].append(
                        (t_done - t0) * 1e3
                    )  # receiver view: post -> data usable
                    rec["verify_ok"] += int(ok)
                    rec["verify_bad"] += bad
            else:
                peer.wait(b"R|" + tag + b"|")
                fill(arena, mine, base, salt)
                device_sync(mem)
                t_make = now()
                h = eng.make("WRITE", prep)
                make_ms = (now() - t_make) * 1e3
                t0 = now()
                st = eng.post(h, b"D|" + tag + b"|%.9f" % t0)
                t_post = now()
                while st == "PROC":
                    st = eng.poll(h)
                t1 = now()
                if st == "ERR":
                    raise RuntimeError("WRITE transfer returned ERR")
                eng.after_done(b"D|" + tag + b"|%.9f" % t0)
                if keep:
                    rec["dur_ms"].append((t1 - t0) * 1e3)
                    rec["post_ms"].append((t_post - t0) * 1e3)
                    rec["make_ms"].append(make_ms)
                    if i < 2:
                        rec["telemetry"].append(eng.telemetry(h))
                    if i == 0 and hasattr(eng, "backend"):
                        rec["backend"] = eng.backend(h)
                eng.release(h)
        else:  # READ: dst pulls
            if role == "src":
                fill(arena, mine, base, salt)
                device_sync(mem)
                eng.notify(b"R|" + tag + b"|")
                peer.wait(b"D|" + tag + b"|")
            else:
                peer.wait(b"R|" + tag + b"|")
                arena[: mine.pool].zero_()
                device_sync(mem)
                t_make = now()
                h = eng.make("READ", prep)
                make_ms = (now() - t_make) * 1e3
                t0 = now()
                st = eng.post(h, b"D|" + tag + b"|")
                t_post = now()
                while st == "PROC":
                    st = eng.poll(h)
                t1 = now()
                if st == "ERR":
                    raise RuntimeError("READ transfer returned ERR")
                eng.after_done(b"D|" + tag + b"|")
                ok, bad = verify(arena, mine, base, salt)
                if keep:
                    rec["dur_ms"].append((t1 - t0) * 1e3)
                    rec["post_ms"].append((t_post - t0) * 1e3)
                    rec["make_ms"].append(make_ms)
                    rec["verify_ok"] += int(ok)
                    rec["verify_bad"] += bad
                    if i < 2:
                        rec["telemetry"].append(eng.telemetry(h))
                    if i == 0 and hasattr(eng, "backend"):
                        rec["backend"] = eng.backend(h)
                eng.release(h)
        if a.window and keep and i % 20 == 0:
            Path(f"{QUIET_DIR}/{a.owner}").touch()
    a.ctl.barrier(f"cell{k}.end")
    ev1 = evidence()
    rec["evidence_before"], rec["evidence_after"] = ev0, ev1
    e0, e1 = ev0["evicted_ms"], ev1["evicted_ms"]
    rec["evicted_jump_ms"] = None if e0 is None or e1 is None else e1 - e0
    rec["valid"] = rec["evicted_jump_ms"] in (None, 0)
    d = stats(rec["dur_ms"])
    rec["dur_stats_ms"] = d
    if d and initiator:
        rec["gbps_median"] = mine.payload / 1e9 / (d["median"] / 1e3)
        rec["gbps_mean"] = mine.payload / 1e9 / (d["mean"] / 1e3)
        rec["gbps_at_p95_dur"] = mine.payload / 1e9 / (d["p95"] / 1e3)
        rec["gib_s_median"] = mine.payload / 2**30 / (d["median"] / 1e3)
    if rec["oneway_ms"]:
        rec["oneway_stats_ms"] = stats(rec["oneway_ms"])
    return rec


def run_notif_cell(a, eng, peer: Peer, cell, k: int) -> dict:
    op, kind, size, iters = cell
    rec = {
        "cell": f"NOTIF:-:0:{iters}",
        "op": "NOTIF",
        "oneway_us": [],
        "rtt_us": [],
        "iters": iters,
    }
    a.ctl.barrier(f"cell{k}.start")
    for i in range(-20, iters):
        tag = str(i).encode()
        if a.role == "src":
            t0 = now()
            eng.notify(b"P|" + tag + b"|%.9f" % t0)
            _, t_back = peer.wait(b"Q|" + tag + b"|")
            if i >= 0:
                rec["rtt_us"].append((t_back - t0) * 1e6)
        else:
            msg, t_arr = peer.wait(b"P|" + tag + b"|")
            if i >= 0:
                rec["oneway_us"].append((t_arr - float(msg.split(b"|")[2])) * 1e6)
            eng.notify(b"Q|" + tag + b"|")
    a.ctl.barrier(f"cell{k}.end")
    rec["oneway_stats_us"] = stats(rec["oneway_us"])
    rec["rtt_stats_us"] = stats(rec["rtt_us"])
    rec["valid"] = True
    return rec


def cell_line(role: str, rec: dict) -> str:
    head = f"[{role}] {rec.get('cell')}: "
    s = rec.get("dur_stats_ms")
    if not s:
        keep = ("error", "skipped", "oneway_stats_us", "rtt_stats_us")
        return head + json.dumps({k: v for k, v in rec.items() if k in keep})
    return head + (
        f"median {s['median']:.3f} ms p95 {s['p95']:.3f} ms "
        f"{rec.get('gbps_median', 0):.2f} GB/s "
        f"verify {rec.get('verify_ok')}/{rec.get('iters')} bad {rec.get('verify_bad')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["src", "dst"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="nixl")
    ap.add_argument("--mem", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--dev-index", type=int, default=0)
    ap.add_argument(
        "--expect-bus", default="", help="hex PCI bus of the device, e.g. db"
    )
    ap.add_argument("--cells", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--window", action="store_true", help="measured-window protocol (src side)"
    )
    ap.add_argument("--owner", default="Disagg")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    a.ctl = Ctl(run_dir, a.role, a.timeout)
    cells = parse_cells(a.cells)
    arena_bytes = max(
        (pool_bytes(kind, size) for op, kind, size, _ in cells if op != "NOTIF"),
        default=MIB,
    )
    arena_bytes = (arena_bytes + MIB - 1) // MIB * MIB
    max_payload = max((s for _, _, s, _ in cells), default=MIB)
    os.environ["G0B_MAX_PAYLOAD"] = str(max_payload)

    info: dict = {
        "role": a.role,
        "engine": a.engine,
        "mem": a.mem,
        "label": a.label,
        "pid": os.getpid(),
        "argv": sys.argv[1:],
        "env": {
            k: os.environ.get(k)
            for k in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_MAX_HW_QUEUES",
                "HSA_ENABLE_SDMA",
                "HSA_FORCE_FINE_GRAIN_PCIE",
                "HSA_ENABLE_IPC_MODE_LEGACY",
                "UCX_TLS",
                "UCX_LOG_LEVEL",
                "UCX_PROTO_INFO",
                "UCX_RNDV_THRESH",
                "UCX_RNDV_SCHEME",
                "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS",
                "UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS",
                "NIXL_PLUGIN_DIR",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "arena_mib": arena_bytes // MIB,
    }
    if a.mem == "gpu":
        torch.accelerator.set_device_index(a.dev_index)
        device = torch.device("cuda", a.dev_index)
        props = torch.cuda.get_device_properties(a.dev_index)
        bus = f"{props.pci_bus_id:02x}"
        info.update(
            {
                "device_name": props.name,
                "pci_bus": bus,
                "visible_devices": torch.accelerator.device_count(),
            }
        )
        if a.expect_bus and bus != a.expect_bus:
            print(
                f"REFUSE: device {a.dev_index} is bus {bus} ({props.name}), "
                f"expected {a.expect_bus}",
                flush=True,
            )
            return 2
        if "V620" not in props.name:
            print(f"REFUSE: {props.name} is not a V620", flush=True)
            return 2
    else:
        device = torch.device("cpu")
    print(f"[{a.role}] {json.dumps(info)}", flush=True)

    arena = torch.empty(arena_bytes, dtype=torch.uint8, device=device)
    arena.zero_()
    device_sync(a.mem)
    a.ctl.barrier("start")
    t_init = now()
    eng = ENGINES[a.engine](a.role, a.ctl, arena, a.dev_index, a.mem)
    info["engine_init_s"] = round(now() - t_init, 3)
    peer = Peer(eng, a.timeout)
    base_cache: dict[int, torch.Tensor] = {}
    wstate: dict = {}
    rc = 0
    with open(run_dir / f"{a.role}.jsonl", "a") as out:
        out.write(json.dumps({"info": info}) + "\n")
        out.flush()
        for k, cell in enumerate(cells):
            if a.window and a.role == "src":
                window_gate(a.owner, wstate)
            name = f"{cell[0]}:{cell[1]}:{cell[2]}:{cell[3]}"
            try:
                if cell[0] == "NOTIF":
                    rec = run_notif_cell(a, eng, peer, cell, k)
                else:
                    rec = run_xfer_cell(a, eng, peer, arena, base_cache, cell, k)
            except NotImplementedError as e:
                rec = {"cell": name, "skipped": repr(e)}
            except Exception as e:  # noqa: BLE001 - recorded; the peer times out
                rec = {"cell": name, "error": repr(e)}
                rc = 1
            rec["role"] = a.role
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(cell_line(a.role, rec), flush=True)
            if rc:
                break
    a.ctl.put(f"done.{a.role}", b"1")
    with contextlib.suppress(TimeoutError):
        a.ctl.get(f"done.{a.ctl.peer}", timeout=60)
    eng.close()
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    sys.exit(main())
