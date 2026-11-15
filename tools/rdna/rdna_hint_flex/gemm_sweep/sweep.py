# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, B023, SIM115
"""Sweep of dense fp16 GEMM designs for the HI2 prefill (gfx1030, v_dot2_f32_f16).

C[M, N] = A[M, K] W[N, K]^T at the 27B HI2 TP2 shapes, every design next to
rocBLAS (heuristic and TunableOp-tuned) and, with --fused, next to the existing
fused decode-in-GEMM kernel and the dequant + rocBLAS pair the server runs
today.

Method (the V620 is power capped at ~175 W: an idle card boosts to ~2.2 GHz and
a loaded one settles near 1.8 GHz within a second, so short or sequential
timings differ by 25 % for identical work):
  * weights rotate through enough sets to exceed ``--ws-mb`` (default 640 MB,
    five times the 128 MB Infinity Cache), one CUDA-graph launch per set;
  * a cell (shape, M) times every contender in interleaved rounds (forward,
    then backward order), after a warm-up that brings the card to its power
    limit; each window is >= ``--win-ms``; the table keeps the minimum and the
    median over rounds;
  * a background sampler logs the graphics clock and socket power of the cell.

  python gemm_sweep/sweep.py build --preset core            # one .so per design
  HIP_VISIBLE_DEVICES=5 python gemm_sweep/sweep.py run --preset core --csv out.csv --fused
  HIP_VISIBLE_DEVICES=5 python gemm_sweep/sweep.py check --preset core
  python gemm_sweep/sweep.py info --preset core             # VGPR / LDS / occupancy
  python gemm_sweep/sweep.py isa --preset base              # hot-loop instruction mix

Designs are ``Variant`` keyword sets; ``--preset a+b+bm=64,bn=64,...`` mixes
presets and inline designs. One CSV row per (kind, variant, shape, M).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import ctypes
import dataclasses
import fcntl
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import regex as re

HERE = Path(__file__).resolve().parent
HIPCC = os.environ.get("HIPCC", "/opt/rocm/core-7.14/bin/hipcc")
OUT = Path(os.environ.get("SWEEP_OUT", str(Path.home() / "w4a8_runs" / "hi2-gemm")))
BUILD_LOCK = Path.home() / "w4a8_runs" / "audit" / "build.lock"
AMDSMI_GPU = os.environ.get(
    "AMDSMI_GPU", "5"
)  # amd-smi index of the HIP device (par1-cs25: HIP 5 = e2)
LLVM_BIN = Path(
    os.environ.get("LLVM_BIN", str(Path(HIPCC).parent.parent / "lib" / "llvm" / "bin"))
)
GEMM_KERNEL_SYM = "_Z11gemm_kernelPKcS0_PDF16_Pfiiiiiii"
GEMM_KERNEL_PREFIX = "_Z11gemm_kernel"
HW_BUS: str | None = (
    None  # PCI bus of the device under test (set by device_check): selects its amdgpu hwmon dir
)

# 27B HI2, TP2, per rank (N = output features, K = input features).
SHAPES = {
    "gate_up": (17408, 5120),
    "down": (5120, 8704),
    "in_proj_qkvz": (8192, 5120),
    "out_proj": (5120, 3072),  # o_proj is the same shape
    "qkv_proj": (7168, 5120),
}
MS = (512, 1024, 2048)
ODD_MS = (700, 1500)  # timing off the tile grid
CHECK_MS = (49, 50, 64, 65, 100, 127, 129, 513, 700, 777, 1000, 1500, 2047)
# (N, K): two production shapes, N off the tile grid, tiny N, short K
CHECK_SHAPES = ((17408, 5120), (5120, 3072), (4104, 192), (130, 64), (8, 64))
CHECK_REL_TOL = 1e-3  # relative Frobenius error vs fp32 (rocBLAS itself: ~2e-4, the fp16 output rounding)


@dataclasses.dataclass(frozen=True)
class Variant:
    bm: int = 128
    bn: int = 128
    bk: int = 32
    tm: int = 8
    tn: int = 8
    nbuf: int = 2
    pad: int = 8
    rast: int = 0
    glw: int = 16
    bar: int = 0
    wpe: int = 0
    splitk: int = 1
    mode: int = 0
    cu: int = 0
    ku: int = 1
    lw: int = 16
    swz: int = 0
    brot: int = 0
    ord: int = 0

    @property
    def name(self) -> str:
        s = (
            f"{self.bm}x{self.bn}k{self.bk}t{self.tm}x{self.tn}n{self.nbuf}"
            f"p{self.pad}r{self.rast}g{self.glw}b{self.bar}w{self.wpe}"
            f"s{self.splitk}m{self.mode}u{self.ku}"
        )
        suffix = (
            (f"l{self.lw}" if self.lw != 16 else "")
            + ("z" if self.swz else "")
            + (f"q{self.brot}" if self.brot else "")
            + (f"o{self.ord}" if self.ord else "")
            + ("cu" if self.cu else "")
        )
        return s + suffix

    @property
    def threads(self) -> int:
        return (self.bm // self.tm) * (self.bn // self.tn)

    def defines(self) -> list[str]:
        d = dict(
            BM=self.bm, BN=self.bn, BK=self.bk, TM=self.tm, TN=self.tn,
            NBUF=self.nbuf, PAD=self.pad, RAST=self.rast, GLW=self.glw,
            BAR=self.bar, WPE=self.wpe, SPLITK=self.splitk, MODE=self.mode, KU=self.ku, LW=self.lw, SWZ=self.swz,
            BROT=self.brot, ORD=self.ord,
        )  # fmt: skip
        out = [f"-DCFG_{k}={v}" for k, v in d.items()]
        out.append(f'-DCFG_NAME="{self.name}"')
        return out

    def lib(self) -> Path:
        return OUT / "lib" / f"{self.name}.so"

    def valid(self) -> bool:
        t = self.threads
        lds = self.nbuf * (self.bm + self.bn) * (self.bk + self.pad) * 2
        return (
            self.bm % self.tm == 0
            and self.bn % self.tn == 0
            and 32 <= t <= 1024
            and t % 32 == 0
            and self.bk % 8 == 0
            and (self.bk + self.pad) % 8 == 0
            and self.bk % (self.glw // 2) == 0
            and self.lw in (8, 16)
            and self.bk % (self.lw // 2) == 0
            and (not self.swz or self.swizzle_ok())
            and (not self.brot or (self.glw == 16 and self.brot < self.lw // 4))
            and (self.ord in (0, 1) or (self.ord == 2 and self.lw == 16))
            and lds <= 65536
        )

    def swizzle_ok(self) -> bool:
        tx, ty = self.bn // self.tn, self.bm // self.tm
        return (
            self.pad == 0
            and self.glw == 16
            and self.bk in (16, 32, 64)
            and tx % 8 == 0
            and ty % 8 == 0
        )


def preset(name: str) -> list[Variant]:
    V = Variant
    if name == "base":
        return [V(), V(ku=2)]
    if name == "diag":  # which pipe limits the 128x128 8x8 design
        return [V(mode=m) for m in (0, 1, 2, 3, 4)]
    if name == "diag2":  # the same on other micro-tiles
        return [V(bm=bm, bn=bn, tm=tm, tn=tn, mode=m)
                for bm, bn, tm, tn in ((128, 128, 8, 4), (64, 64, 4, 4), (128, 128, 16, 8), (128, 64, 8, 8))
                for m in (0, 1, 2, 3)]  # fmt: skip
    if name == "tiles":  # block tile x micro-tile, BK 32, double buffer
        out = []
        for bm, bn in (
            (64, 64),
            (128, 64),
            (64, 128),
            (128, 128),
            (128, 256),
            (256, 128),
        ):
            for tm, tn in (
                (4, 4),
                (8, 4),
                (4, 8),
                (8, 8),
                (16, 4),
                (4, 16),
                (16, 8),
                (8, 16),
            ):
                v = V(bm=bm, bn=bn, tm=tm, tn=tn)
                if v.valid() and 64 <= v.threads <= 512:
                    out.append(v)
        return out
    if name == "core":
        return _core()
    if name == "occ":  # MODE 5: the same designs with the per-block occupancy probe
        return [
            dataclasses.replace(v, mode=5)
            for v in _core()
            if (v.bk, v.nbuf) in ((32, 2), (64, 1))
        ]
    if (
        name == "knobs"
    ):  # one change at a time from the 128x128 / 8x8 / BK 32 / 2-stage design
        return [V(pad=0), V(bar=1), V(ku=2), V(glw=8), V(glw=4), V(cu=1), V(wpe=3), V(wpe=2),
                V(bk=16), V(bk=64, nbuf=1), V(bk=64, nbuf=1, pad=0), V(nbuf=1)]  # fmt: skip
    if (
        name == "rast"
    ):  # block order: 0 = N fastest (default), 1 = M fastest, >= 2 = groups of that many M tiles
        return [V(rast=r) for r in (1, 2, 4, 8, 16)]
    if name == "splitk":
        return [V(splitk=s) for s in (2, 3, 4)] + [
            V(bm=64, bn=128, tm=4, tn=8, splitk=s) for s in (2, 3)
        ]
    if (
        name == "lw8"
    ):  # 8-byte LDS fragments: half the operand registers, twice the ds_read instructions
        return [dataclasses.replace(v, lw=8) for v in _core() if (v.bk, v.nbuf) in ((32, 2), (64, 1))] + [
            V(lw=8, ku=0), V(lw=8, ku=2), V(lw=8, bk=64, nbuf=1, ku=2), V(lw=8, bk=16),
        ]  # fmt: skip
    if (
        name == "swz"
    ):  # XOR-swizzled LDS rows, no pad: 32 KiB for the 128x128 / BK 32 / 2-stage tile
        out = []
        for v in _core():
            if (v.bk, v.nbuf) in ((32, 2), (64, 1)):
                out += [dataclasses.replace(v, pad=0, swz=1, wpe=w) for w in (0, 6)]
        return [v for v in out if v.valid()]
    if (
        name == "t168"
    ):  # the 128-accumulator thread tile (16x8 / 8x16: Tensile's TT8_16) and its neighbours
        out = []
        for bm, bn in (
            (128, 128),
            (256, 128),
            (128, 256),
            (64, 128),
            (128, 64),
            (256, 64),
        ):
            for tm, tn in ((16, 8), (8, 16)):
                for lw in (8, 16):
                    for ku in (1, 2, 4):
                        out.append(V(bm=bm, bn=bn, tm=tm, tn=tn, lw=lw, ku=ku))
        for bm, bn in ((128, 128), (256, 128), (128, 256)):
            for lw in (8, 16):
                for ku in (1, 2):
                    out.append(V(bm=bm, bn=bn, tm=16, tn=8, lw=lw, ku=ku, pad=0, swz=1))
        return [v for v in out if v.valid() and 64 <= v.threads <= 512]
    if (
        name == "fin"
    ):  # conflict-free B pairing (BROT) and chained dot2 order (ORD) on the strongest families
        out = []
        for bm, bn in ((128, 128), (256, 128), (128, 256)):
            for lw, brot in ((16, 1), (8, 0), (8, 1)):
                for ku in (1, 2):
                    for o in (0, 1):
                        out.append(
                            V(bm=bm, bn=bn, tm=16, tn=8, lw=lw, brot=brot, ku=ku, ord=o)
                        )
        for o in (0, 1):
            out.append(V(brot=1, ord=o))  # 128x128, 8x8, BK 32
            out.append(V(bk=64, nbuf=1, pad=0, swz=1, wpe=6, brot=1, ord=o))
            out.append(V(bm=128, bn=64, bk=64, nbuf=1, lw=8, ord=o))
            out.append(V(bm=128, bn=64, bk=64, nbuf=1, lw=8, brot=1, ord=o))
        return [v for v in out if v.valid()]
    if (
        name == "best256"
    ):  # neighbourhood of the best `fin` design: 256x128 tile, 16x8 per thread, 8-byte LDS reads
        out = []
        for rast in (0, 4, 8, 16):
            for ku in (1, 2, 4):
                out.append(V(bm=256, bn=128, tm=16, tn=8, lw=8, rast=rast, ku=ku))
        out += [V(bm=256, bn=128, tm=16, tn=8, lw=8, wpe=w) for w in (3, 4, 5)]
        out += [
            V(bm=256, bn=128, tm=16, tn=8, lw=8, bk=bk, nbuf=nb)
            for bk, nb in ((16, 2), (64, 1))
        ]
        out += [V(bm=256, bn=128, tm=16, tn=8, lw=8, pad=p) for p in (0, 16)]
        return [v for v in out if v.valid()]
    if (
        name == "rot"
    ):  # B k-pair rotation: do the two dot2 sources need different VGPR banks?
        out = [V(mode=m, brot=r) for m in (0, 1, 3) for r in (0, 1)]
        out += [V(mode=m, brot=r, lw=8) for m in (1, 3) for r in (0, 1)]
        out += [dataclasses.replace(v, brot=1) for v in preset("swz") if v.bk == 64 and v.bm == 128 and v.bn == 128
                and (v.tm, v.tn) == (8, 8)]  # fmt: skip
        return [v for v in out if v.valid()]
    raise SystemExit(f"unknown preset {name}")


def _core() -> list[Variant]:
    out = []
    for bm, bn, tm, tn in (
        (64, 64, 4, 4), (64, 64, 8, 4), (128, 64, 8, 4), (64, 128, 4, 8),
        (128, 64, 8, 8), (64, 128, 8, 8), (128, 128, 8, 8), (128, 128, 8, 4),
        (128, 128, 4, 8), (128, 256, 8, 8), (256, 128, 8, 8), (128, 128, 16, 8),
    ):  # fmt: skip
        for bk, nbuf in ((16, 2), (32, 2), (32, 1), (64, 1)):
            v = Variant(bm=bm, bn=bn, tm=tm, tn=tn, bk=bk, nbuf=nbuf)
            if v.valid():
                out.append(v)
    return out


VARIANT_NAME = re.compile(
    r"(\d+)x(\d+)k(\d+)t(\d+)x(\d+)n(\d+)p(\d+)r(\d+)g(\d+)b(\d+)w(\d+)s(\d+)m(\d+)u(\d+)(?:l(\d+))?(z)?(?:q(\d+))?(?:o(\d+))?(cu)?"
)


def parse_name(name: str) -> Variant | None:
    """The inverse of Variant.name (None when ``name`` is not a variant name)."""
    m = VARIANT_NAME.fullmatch(name)
    if not m:
        return None
    g = m.groups()
    bm, bn, bk, tm, tn, nbuf, pad, rast, glw, bar, wpe, splitk, mode, ku = (
        int(x) for x in g[:14]
    )
    return Variant(bm=bm, bn=bn, bk=bk, tm=tm, tn=tn, nbuf=nbuf, pad=pad, rast=rast, glw=glw, bar=bar, wpe=wpe,
                   splitk=splitk, mode=mode, ku=ku, lw=int(g[14]) if g[14] else 16, swz=1 if g[15] else 0,
                   brot=int(g[16]) if g[16] else 0, ord=int(g[17]) if g[17] else 0, cu=1 if g[18] else 0)  # fmt: skip


def load_variants(spec: str) -> list[Variant]:
    """``spec``: preset names, literal variant names, or inline ``bm=128,bn=128,...``, joined by '+'."""
    out: list[Variant] = []
    for part in spec.split("+"):
        if (named := parse_name(part)) is not None:
            out.append(named)
        elif "=" in part:  # inline: bm=128,bn=128,tm=8,...
            out.append(
                Variant(
                    **{k: int(v) for k, v in (kv.split("=") for kv in part.split(","))}
                )
            )
        else:
            out.extend(preset(part))
    seen, uniq = set(), []
    for v in out:
        if v.name not in seen:
            seen.add(v.name)
            uniq.append(v)
    return uniq


# ---------------------------------------------------------------- build ----
def compile_one(v: Variant, extra: list[str]) -> tuple[str, int, str]:
    lib = v.lib()
    lib.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nice", "-n", "19", HIPCC, "-O3", "-std=c++17", "--offload-arch=gfx1030",
        "-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage", *v.defines(),
        *(["-mcumode"] if v.cu else []), *extra, str(HERE / "gemm_variant.hip"),
        "-o", str(lib),
    ]  # fmt: skip
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    (lib.parent / f"{v.name}.remarks").write_text(p.stderr)
    return (
        v.name,
        p.returncode,
        f"{time.time() - t0:5.1f}s\n{p.stderr[-2000:] if p.returncode else ''}",
    )


@contextlib.contextmanager
def build_lock():
    """CPU builds take build.lock (AGENTS.md); hold it only while compiling."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def cmd_build(args) -> None:
    variants = [v for v in load_variants(args.preset) if v.valid()]
    todo = variants if args.force else [v for v in variants if not v.lib().exists()]
    print(
        f"{len(variants)} variants, {len(todo)} to compile, {args.jobs} jobs",
        flush=True,
    )
    if not todo:
        return
    with (
        build_lock(),
        concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, 8)) as ex,
    ):
        for name, rc, msg in ex.map(lambda v: compile_one(v, args.extra), todo):
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {name} {msg}", flush=True)


def lib_ws(torch, lib, m: int, n: int):
    """Workspace a design needs (split-K partials, occupancy records), or None."""
    wsb = lib.ws_bytes(m, n)
    return torch.empty(max(wsb, 16), dtype=torch.uint8, device="cuda") if wsb else None


def read_remarks(v: Variant) -> dict[str, int]:
    """VGPRs, spills, scratch, LDS and the compiler's waves/SIMD of the gemm
    kernel, from the resource-usage remarks ``build`` saved next to the .so."""
    p = v.lib().with_suffix(".remarks")
    got: dict[str, int] = {}
    if not p.exists():
        return got
    cur = None
    for ln in p.read_text().splitlines():
        m = re.search(r"remark: Function Name: (\S+)", ln)
        if m:
            cur = m.group(1)
        elif cur and cur.startswith(GEMM_KERNEL_PREFIX):
            m = re.search(r"remark:\s+([A-Za-z ]+?)(?: \[[^\]]*\])?: (\d+)", ln)
            if m:
                got[m.group(1).strip()] = int(m.group(2))
    return got


# ------------------------------------------------------------------ run ----
class Lib:
    def __init__(self, v: Variant):
        self.v = v
        self.lib = ctypes.CDLL(str(v.lib()))
        self.lib.variant_launch.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 6 + [ctypes.c_void_p] * 2
        )
        self.lib.variant_launch.restype = ctypes.c_int
        self.lib.variant_ws_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.variant_ws_bytes.restype = ctypes.c_size_t
        self.lib.variant_info.argtypes = [ctypes.POINTER(ctypes.c_int)]
        info = (ctypes.c_int * 8)()
        self.lib.variant_info(info)
        self.vgpr, self.lds, self.scratch, self.threads, self.blocks, *_ = list(info)

    def ws_bytes(self, m: int, n: int) -> int:
        return int(self.lib.variant_ws_bytes(m, n))

    def launch(self, a, w, c, ws) -> None:
        import torch

        m, k = a.shape
        n = w.shape[0]
        rc = self.lib.variant_launch(
            a.data_ptr(), w.data_ptr(), c.data_ptr(), m, n, k, k, k, c.stride(0),
            ws.data_ptr() if ws is not None else None,
            torch.cuda.current_stream().cuda_stream,  # the capture stream inside a graph
        )  # fmt: skip
        if rc != 0:
            raise RuntimeError(f"{self.v.name}: launch rc={rc}")

    def occupancy(self) -> tuple[int, int]:
        """(blocks per WGP from the hip occupancy API, waves per SIMD the compiler
        allows for this VGPR / LDS footprint). The API assumes 64 KiB of LDS per
        multiprocessor, the compiler 128 KiB per WGP: ``sweep.py occ`` measures
        which one the hardware honours."""
        return self.blocks, read_remarks(self.v).get("Occupancy", -1)


FIELDS = [
    "phase", "kind", "variant", "shape", "N", "K", "M", "us", "us_min", "tflops", "vs_rb",
    "max_abs", "rel_fro", "rel_fp32", "vgpr", "lds", "scratch", "threads", "blocks_per_wgp",
    "waves_per_simd", "gfx_mhz", "power_w", "note",
]  # fmt: skip


def device_check(torch) -> str:
    p = torch.cuda.get_device_properties(0)
    if not p.gcnArchName.startswith("gfx1030") or "V620" not in p.name:
        raise SystemExit(f"refusing {p.name} {p.gcnArchName}: not a V620")
    want = os.environ.get("EXPECT_BUS")
    bus = f"{p.pci_bus_id:02x}"
    if want and bus != want.lower():
        raise SystemExit(f"device bus {bus}, expected {want}")
    global HW_BUS  # noqa: PLW0603
    HW_BUS = bus
    return f"{p.name} {p.gcnArchName} bus {bus} CUs {p.multi_processor_count} (WGPs)"


def sets_for(n: int, k: int, ws_mb: float, elt: int = 2) -> int:
    return max(2, math.ceil(ws_mb * 2**20 / (n * k * elt)))


def make_data(torch, m: int, n: int, k: int, sets: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=gen)
    ws = []
    for _ in range(sets):
        w = torch.randn(n, k, dtype=torch.float16, device="cuda", generator=gen)
        w.mul_(0.05)
        ws.append(w)
    return x, ws


def errors(torch, c, ref):
    d = c.float() - ref.float()
    return float(d.abs().max()), float(d.norm() / ref.float().norm().clamp_min(1e-30))


class ClockSampler(threading.Thread):
    """Median graphics clock and socket power of the device while a cell runs
    (amd-smi, one sample per ~0.5 s)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()
        self.mhz: list[float] = []
        self.watts: list[float] = []

    def run(self) -> None:
        while not self.stop_ev.is_set():
            try:
                out = subprocess.run(
                    ["amd-smi", "metric", "-g", AMDSMI_GPU, "--clock", "--power", "--csv"],
                    capture_output=True, text=True, timeout=10,
                ).stdout  # fmt: skip
                rows = list(csv.DictReader(out.splitlines()))
                if rows:
                    self.mhz.append(float(rows[-1]["gfx_0_clk"]))
                    self.watts.append(float(rows[-1]["socket_power"]))
            except Exception:  # noqa: BLE001
                pass

    def result(self) -> tuple[float | None, float | None]:
        self.stop_ev.set()
        self.join(timeout=15)
        # drop the first samples (still ramping) when there are enough
        mhz, w = self.mhz[1:] or self.mhz, self.watts[1:] or self.watts
        return (
            statistics.median(mhz) if mhz else None,
            statistics.median(w) if w else None,
        )


class HwmonSampler(threading.Thread):
    """Shader clock and socket power from the amdgpu hwmon files at ~200 Hz,
    time-stamped, so that every timing window gets its own clock and power: a
    power-capped part runs a dot2-bound kernel at a lower clock than a
    memory-bound one, and a per-cell median hides that."""

    def __init__(self, bus: str) -> None:
        super().__init__(daemon=True)
        hw = next(Path(f"/sys/bus/pci/devices/0000:{bus}:00.0/hwmon").glob("hwmon*"))
        self.f_mhz, self.f_w = open(hw / "freq1_input"), open(hw / "power1_average")  # noqa: SIM115
        self.stop_ev = threading.Event()
        self.t: list[float] = []
        self.mhz: list[float] = []
        self.watts: list[float] = []

    def run(self) -> None:
        while not self.stop_ev.is_set():
            self.f_mhz.seek(0)
            self.f_w.seek(0)
            self.t.append(time.perf_counter())
            self.mhz.append(int(self.f_mhz.read()) / 1e6)
            self.watts.append(int(self.f_w.read()) / 1e6)
            time.sleep(0.004)

    def stop(self) -> None:
        self.stop_ev.set()
        self.join(timeout=5)

    def window_stats(
        self, windows: list[tuple[float, float]]
    ) -> tuple[float | None, float | None]:
        """Median clock (MHz) and median power (W) over the samples inside any window."""
        import bisect  # noqa: PLC0415

        mh, w = [], []
        for t0, t1 in windows:
            lo, hi = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
            mh += self.mhz[lo:hi]
            w += self.watts[lo:hi]
        return (
            statistics.median(mh) if mh else None,
            statistics.median(w) if w else None,
        )


class Contender:
    """One timed thing: launches (one per rotating weight set) captured as one
    CUDA graph. ``tun`` is the TunableOp state while capturing (a lookup bakes
    the chosen solution into the graph)."""

    def __init__(
        self,
        kind: str,
        variant: str,
        calls,
        flop: float,
        tun: bool | None = None,
        **extra,
    ):
        self.kind, self.variant, self.calls, self.flop, self.tun = (
            kind,
            variant,
            calls,
            flop,
            tun,
        )
        self.extra = extra
        self.err: tuple | None = None  # (max_abs, rel_fro) vs rocBLAS, rel vs fp32
        self.graph = None
        self.inner = 1
        self.times: list[float] = []
        self.windows: list[
            tuple[float, float]
        ] = []  # wall-clock (start, end) of every timing window
        self.clk: tuple[float | None, float | None] = (
            None,
            None,
        )  # (MHz, W) inside those windows

    def capture(self, torch) -> None:
        torch.accelerator.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for f in self.calls:
                f()
        self.graph.replay()
        torch.accelerator.synchronize()


def measure(
    torch, tunable, cons: list[Contender], rounds: int, warm_s: float, win_ms: float
):
    """Interleaved windows over all contenders (forward, then backward order)
    after a warm-up at the power limit; returns the sampled (MHz, W)."""
    for c in cons:
        if c.tun is not None:
            tunable.enable(c.tun)
            tunable.tuning_enable(False)
        c.capture(torch)
    tunable.enable(False)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for c in cons:  # window size: >= win_ms of work per timing
        ev[0].record()
        c.graph.replay()
        ev[1].record()
        ev[1].synchronize()
        c.inner = max(1, math.ceil(win_ms / max(ev[0].elapsed_time(ev[1]), 1e-3)))
    t0 = time.time()
    while time.time() - t0 < warm_s:
        for _ in range(4):
            cons[0].graph.replay()
        torch.accelerator.synchronize()
    sampler = ClockSampler()
    sampler.start()
    hw = HwmonSampler(HW_BUS) if HW_BUS else None
    if hw:
        hw.start()
    for r in range(rounds):
        for c in cons if r % 2 == 0 else reversed(cons):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            t_start = time.perf_counter()
            s.record()
            for _ in range(c.inner):
                c.graph.replay()
            e.record()
            e.synchronize()
            c.windows.append((t_start, time.perf_counter()))
            c.times.append(s.elapsed_time(e) / (c.inner * len(c.calls)) * 1e3)
    if hw:
        hw.stop()
        for c in cons:
            c.clk = hw.window_stats(c.windows)
    return sampler.result()


class Writer:
    def __init__(self, path: str | None):
        self.f = None
        if path:
            new = not Path(path).exists()
            self.f = open(path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                self.w.writeheader()

    def add(self, **row) -> None:
        row = {k: ("" if v is None else v) for k, v in row.items()}
        if self.f:
            self.w.writerow(row)
            self.f.flush()
        us = float(row["us"]) if row.get("us") not in ("", None) else float("nan")
        print(
            f"{row.get('kind', ''):13s} {row.get('variant', ''):36s} {row.get('shape', ''):13s} "
            f"M={row.get('M', ''):<5} {us:9.1f} us (min {row.get('us_min', '')!s:>7}) "
            f"{row.get('tflops', '')!s:>6} TF/s  vs_rb {row.get('vs_rb', '')!s:>6}  "
            f"err {row.get('max_abs', '')!s:>8}  {row.get('note', '')}",
            flush=True,
        )


def save_rows(tunable, path: Path) -> None:
    """TunableOp results + validators as the CSV torch reads back (the module
    has no write_file; tune_prod_shapes.py writes it the same way)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerows(("Validator", *v) for v in tunable.get_validators())
        w.writerows(tunable.get_results())


def tune_rocblas(torch, x, w, tunable, max_ms: float, iters: int, rot_mb: int) -> None:
    tunable.set_max_tuning_duration(int(max_ms))
    tunable.set_max_tuning_iterations(iters)
    tunable.set_rotating_buffer_size(
        rot_mb
    )  # candidates see cold copies, as in serving
    tunable.set_numerical_check_tolerances(True, 0.01, 0.01)
    tunable.enable(True)
    tunable.tuning_enable(True)
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float16, device="cuda")
    for _ in range(2):
        torch.mm(x, w.t(), out=out)
    torch.accelerator.synchronize()
    tunable.tuning_enable(False)


HI2_FIELDS = [
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
]


def fused_sets(torch, n: int, k: int, ws_mb: float):
    """Random HI2-format weights (50 % INT2 / 50 % INT4 rows, mixed blocks)
    in enough sets to exceed ws_mb."""
    sys.path.insert(0, str(HERE.parent))
    from bench_kernel import make

    one = make(n, k, int4_frac=0.5, seed=1)
    per_set = sum(one[f].numel() * one[f].element_size() for f in HI2_FIELDS)
    nsets = max(2, math.ceil(ws_mb * 2**20 / per_set))
    return [
        {f: one[f].cuda().clone() for f in HI2_FIELDS} for _ in range(nsets)
    ], per_set


def cmd_run(args) -> None:
    import torch

    print("device:", device_check(torch), flush=True)
    import torch.cuda.tunable as tunable

    os.makedirs(OUT, exist_ok=True)
    rows_file = Path(args.tunable_file or OUT / "tunableop" / "tunableop_results0.csv")
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    tunable.set_filename(str(rows_file), insert_device_ordinal=False)
    if rows_file.exists():
        tunable.read_file(str(rows_file))
    tunable.enable(False)
    variants = (
        [] if args.no_variants else [v for v in load_variants(args.preset) if v.valid()]
    )
    libs = []
    for v in variants:
        if not v.lib().exists():
            print(f"skip {v.name}: not built", flush=True)
            continue
        libs.append(Lib(v))
    ops = None
    if args.fused:
        from vllm import _custom_ops as ops  # noqa: PLC0415
    wr = Writer(args.csv)
    shapes = {
        k: s
        for k, s in SHAPES.items()
        if not args.shapes or k in args.shapes.split(",")
    }
    ms = [int(m) for m in args.m.split(",")]
    for name, (n, k) in shapes.items():
        nsets = sets_for(n, k, args.ws_mb)
        fsets, fper = fused_sets(torch, n, k, args.ws_mb) if args.fused else ([], 0)
        scratch = (
            torch.empty(n, k, dtype=torch.float16, device="cuda")
            if args.fused
            else None
        )
        for m in ms:
            x, wsets = make_data(torch, m, n, k, nsets)
            flop = 2.0 * m * n * k
            c = torch.empty(m, n, dtype=torch.float16, device="cuda")
            ref32 = None
            if not args.no_check:
                ref32 = torch.empty(m, n, dtype=torch.float32, device="cuda")
                for lo in range(0, m, 512):  # fp32 truth in row chunks
                    ref32[lo : lo + 512] = (
                        x[lo : lo + 512].float() @ wsets[0].float().t()
                    )
            common = dict(shape=name, N=n, K=k, M=m)

            def vs32(out):
                return errors(torch, out, ref32)[1] if ref32 is not None else None

            def mm_cold():
                return [lambda w=w: torch.mm(x, w.t(), out=c) for w in wsets]

            def mm_hot():
                return [lambda: torch.mm(x, wsets[0].t(), out=c)] * len(wsets)

            # --- rocBLAS: the heuristic (TunableOp off), then rows tuned on this cell
            tunable.enable(False)
            torch.mm(x, wsets[0].t(), out=c)
            rel32_heur = vs32(c)
            tune_rocblas(
                torch, x, wsets[0], tunable, args.tune_ms, args.tune_iters, args.rot_mb
            )
            save_rows(tunable, rows_file)
            sol = ";".join(
                r[2] for r in tunable.get_results() if f"_{n}_{m}_{k}_" in r[1]
            )
            tunable.enable(True)
            tunable.tuning_enable(False)
            torch.mm(x, wsets[0].t(), out=c)
            tunable.enable(False)
            c_ref = c.clone()
            rel32_tuned = vs32(c_ref)
            heur = Contender("rocblas_heur", "default", mm_cold(), flop, tun=False)
            tuned = Contender(
                "rocblas_tuned", "tunableop", mm_cold(), flop, tun=True, note=sol
            )
            heur.err, tuned.err = (None, None, rel32_heur), (None, None, rel32_tuned)
            cons = [
                heur, tuned,
                Contender("rocblas_heur_hot", "default", mm_hot(), flop, tun=False),
                Contender("rocblas_tuned_hot", "tunableop", mm_hot(), flop, tun=True),
            ]  # fmt: skip

            # --- dense variants (error against rocBLAS tuned and the fp32 truth on set 0)
            for lib in libs:
                v = lib.v
                wsbuf = lib_ws(torch, lib, m, n)
                try:
                    lib.launch(x, wsets[0], c, wsbuf)
                    torch.accelerator.synchronize()
                    err = None
                    if v.mode == 0:
                        d = errors(torch, c, c_ref)
                        err = (d[0], d[1], vs32(c))
                    bpw, wps = lib.occupancy()
                    con = Contender(
                        "dense", v.name,
                        [lambda w=w, lib=lib, wsbuf=wsbuf: lib.launch(x, w, c, wsbuf) for w in wsets],
                        flop, vgpr=lib.vgpr, lds=lib.lds, scratch=lib.scratch, threads=lib.threads,
                        blocks_per_wgp=bpw, waves_per_simd=wps,
                    )  # fmt: skip
                    con.err = err
                    cons.append(con)
                except Exception as ex:  # noqa: BLE001
                    wr.add(
                        phase=args.label,
                        kind="dense",
                        variant=v.name,
                        note=f"ERR {ex}",
                        **common,
                    )

            # --- the existing fused kernel and the dequant + rocBLAS pair (compressed cold sets)
            if args.fused:
                out_f = torch.empty(m, n, dtype=torch.float16, device="cuda")

                def pair(s, out_f=out_f):
                    def f():
                        ops.rdna_hint_flex_dequant_rdna2(
                            *(s[g] for g in HI2_FIELDS), scratch
                        )
                        torch.mm(x, scratch.t(), out=out_f)

                    return f

                cons += [
                    Contender("fused_ref", "gemm2_128x128",
                              [lambda s=s, out_f=out_f: ops.rdna_hint_flex_gemm_rdna2(x, s, out_f) for s in fsets],
                              flop, note=f"{len(fsets)} sets {fper / 2**20:.1f} MiB"),
                    Contender("dequant_only", "dequant",
                              [lambda s=s: ops.rdna_hint_flex_dequant_rdna2(*(s[f] for f in HI2_FIELDS), scratch)
                               for s in fsets], 0.0),
                    Contender("dequant_mm", "dequant+heur", [pair(s) for s in fsets], flop, tun=False),
                    Contender("dequant_mm", "dequant+tuned", [pair(s) for s in fsets], flop, tun=True),
                ]  # fmt: skip
                for pname in [p for p in args.pair.split(",") if p]:
                    plib = next((lb for lb in libs if lb.v.name == pname), None)
                    if plib is None:
                        continue
                    pws = lib_ws(torch, plib, m, n)

                    def dq_dense(s, plib=plib, pws=pws, out_f=out_f):
                        def f():
                            ops.rdna_hint_flex_dequant_rdna2(
                                *(s[g] for g in HI2_FIELDS), scratch
                            )
                            plib.launch(x, scratch, out_f, pws)

                        return f

                    cons.append(
                        Contender(
                            "dequant_dense", pname, [dq_dense(s) for s in fsets], flop
                        )
                    )

            clk = measure(torch, tunable, cons, args.rounds, args.warm_s, args.win_ms)
            best_us = min(statistics.median(heur.times), statistics.median(tuned.times))
            for con in cons:
                med = statistics.median(con.times)
                row = dict(
                    phase=args.label, kind=con.kind, variant=con.variant, us=round(med, 1),
                    us_min=round(min(con.times), 1),
                    tflops=round(con.flop / med / 1e6, 2) if con.flop else None,
                    vs_rb=round(best_us / med, 3) if con.flop else None,
                    gfx_mhz=con.clk[0] or clk[0], power_w=con.clk[1] or clk[1], **common,
                )  # fmt: skip
                if con.err:
                    row["max_abs"] = (
                        None if con.err[0] is None else round(con.err[0], 5)
                    )
                    row["rel_fro"] = None if con.err[1] is None else f"{con.err[1]:.2e}"
                    row["rel_fp32"] = (
                        None if con.err[2] is None else f"{con.err[2]:.2e}"
                    )
                row.update(con.extra)
                wr.add(**row)
            x = wsets = c = ref32 = cons = (
                None  # drop the cell's tensors before the next one
            )
            torch.cuda.memory.empty_cache()


# ---------------------------------------------------------------- probe ----
DOT2_TFLOPS_PER_GHZ = (
    4608 * 4 / 1e3
)  # 72 CUs x 2 SIMD32 x 32 lanes x 4 flop (v_dot2c_f32_f16), per GHz


def cmd_probe(args) -> None:
    """Each design alone for --seconds on one cell, the shader clock and the socket
    power sampled at ~200 Hz. On a power-capped part a dot2-bound kernel settles
    at a lower clock than a stalled one, so TF/s alone cannot tell 'slow' from
    'power-limited'; TF/s per GHz (18.4 = every lane of every SIMD retiring one
    dot2 per cycle) is reported next to the clock."""
    import torch
    import torch.cuda.tunable as tunable

    print("device:", device_check(torch), flush=True)
    n, k = SHAPES[args.shape]
    m = args.m
    rows_file = Path(args.tunable_file or OUT / "tunableop" / "tunableop_results0.csv")
    tunable.set_filename(str(rows_file), insert_device_ordinal=False)
    if rows_file.exists():
        tunable.read_file(str(rows_file))
    tunable.enable(False)
    x, wsets = make_data(torch, m, n, k, sets_for(n, k, args.ws_mb))
    c = torch.empty(m, n, dtype=torch.float16, device="cuda")
    torch.mm(
        x, wsets[0].t(), out=c
    )  # the hipBLAS handle must exist before a graph capture
    torch.accelerator.synchronize()
    flop = 2.0 * m * n * k
    cons: list[Contender] = []
    if not args.no_rocblas:
        cons += [
            Contender("rocblas_heur", "default", [lambda w=w: torch.mm(x, w.t(), out=c) for w in wsets], flop, tun=False),
            Contender("rocblas_tuned", "tunableop", [lambda w=w: torch.mm(x, w.t(), out=c) for w in wsets], flop, tun=True),
        ]  # fmt: skip
    for v in load_variants(args.preset) if args.preset else []:
        if v.valid() and v.lib().exists():
            lib = Lib(v)
            wsb = lib_ws(torch, lib, m, n)
            cons.append(Contender("dense", v.name,
                                  [lambda w=w, lib=lib, wsb=wsb: lib.launch(x, w, c, wsb) for w in wsets], flop))  # fmt: skip
    for con in cons:
        if con.tun is not None:
            tunable.enable(con.tun)
            tunable.tuning_enable(False)
        con.capture(torch)
    tunable.enable(False)
    wr = Writer(args.csv)
    hw = HwmonSampler(HW_BUS)
    hw.start()
    print(
        f"{'contender':46s} {'TF/s':>6s} {'MHz':>5s} {'W':>5s} {'TF/s per GHz':>12s} {'% of dot2 rate':>14s}",
        flush=True,
    )
    for rep in range(args.repeat):
        for con in cons if rep % 2 == 0 else reversed(cons):
            stamps: list[float] = []
            t_end = time.perf_counter() + args.seconds
            while time.perf_counter() < t_end:
                con.graph.replay()
                torch.accelerator.synchronize()
                stamps.append(time.perf_counter())
            tail = [i for i, t in enumerate(stamps) if t >= stamps[-1] - args.tail]
            i0 = max(tail[0] - 1, 0)
            tf = (
                (len(stamps) - 1 - i0)
                * len(con.calls)
                * flop
                / (stamps[-1] - stamps[i0])
                / 1e12
            )
            mhz, w = hw.window_stats([(stamps[i0], stamps[-1])])
            per_ghz = tf / (mhz / 1e3) if mhz else float("nan")
            print(f"{con.kind + ' ' + con.variant:46s} {tf:6.2f} {mhz or 0:5.0f} {w or 0:5.0f} {per_ghz:12.2f}"
                  f" {100 * per_ghz / DOT2_TFLOPS_PER_GHZ:14.1f}", flush=True)  # fmt: skip
            wr.add(phase=args.label, kind="probe", variant=f"{con.kind}:{con.variant}", shape=args.shape, N=n, K=k, M=m,
                   tflops=round(tf, 2), us=round(flop / (tf * 1e12) * 1e6, 1), gfx_mhz=mhz, power_w=w,
                   note=f"rep{rep} {per_ghz:.2f} TF/s per GHz")  # fmt: skip
    hw.stop()


# ---------------------------------------------------------------- check ----
def cmd_check(args) -> None:
    """Every MODE 0 variant against an fp32 reference on the production shapes
    and on awkward ones: odd M (smaller than one tile, not a multiple of a
    tile), N that is not a multiple of the tile and tiny N, short K."""
    import torch

    print("device:", device_check(torch), flush=True)
    libs = [
        Lib(v)
        for v in load_variants(args.preset)
        if v.valid() and v.mode == 0 and v.lib().exists()
    ]
    stat = {lib.v.name: {"abs": 0.0, "rel": 0.0, "bad": 0, "cases": 0} for lib in libs}
    for n, k in CHECK_SHAPES:
        for m in CHECK_MS:
            gen = torch.Generator(device="cuda").manual_seed(m * 7919 + n)
            x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=gen)
            w = (
                torch.randn(n, k, dtype=torch.float16, device="cuda", generator=gen)
                * 0.05
            )
            ref = torch.empty(m, n, dtype=torch.float32, device="cuda")
            for lo in range(0, m, 512):
                ref[lo : lo + 512] = x[lo : lo + 512].float() @ w.float().t()
            for lib in libs:
                v = lib.v
                if k % v.bk:
                    continue
                c = torch.full((m, n), float("nan"), dtype=torch.float16, device="cuda")
                wsb = lib.ws_bytes(m, n)
                ws = (
                    torch.empty(max(wsb, 16), dtype=torch.uint8, device="cuda")
                    if wsb
                    else None
                )
                lib.launch(x, w, c, ws)
                torch.accelerator.synchronize()
                st = stat[v.name]
                st["cases"] += 1
                if torch.isnan(c).any():
                    st["bad"] += 1
                    print(f"NAN  {v.name} M={m} N={n} K={k}", flush=True)
                    continue
                e = errors(torch, c, ref)
                st["abs"], st["rel"] = max(st["abs"], e[0]), max(st["rel"], e[1])
                if e[1] > CHECK_REL_TOL:
                    st["bad"] += 1
                    if st["bad"] <= 3:
                        print(
                            f"BAD  {v.name} M={m} N={n} K={k} abs={e[0]:.4f} rel={e[1]:.2e}",
                            flush=True,
                        )
            del x, w, ref
    rows = []
    for name, st in stat.items():
        ok = st["bad"] == 0 and st["cases"] > 0
        rows.append({"variant": name, "status": "ok" if ok else "FAIL", "cases": st["cases"], "failures": st["bad"],
                     "worst_abs": f"{st['abs']:.5f}", "worst_rel": f"{st['rel']:.2e}"})  # fmt: skip
        print(
            f"{'ok  ' if ok else 'FAIL'} {name:46s} cases {st['cases']:3d} worst abs {st['abs']:.4f} rel {st['rel']:.2e}"
        )
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    sys.exit(1 if any(r["status"] != "ok" for r in rows) else 0)


def cmd_info(args) -> None:
    import torch  # noqa: F401  (loads libamdhip64 the way the sweep does)

    print(f"{'variant':46s} vgpr   lds scratch thr blk/WGP waves/SIMD")
    for v in load_variants(args.preset):
        if not v.valid() or not v.lib().exists():
            continue
        lib = Lib(v)
        bpw, wps = lib.occupancy()
        print(
            f"{v.name:46s} {lib.vgpr:4d} {lib.lds:5d} {lib.scratch:7d} {lib.threads:3d} {bpw:7d} {wps:10.2f}"
        )


def cmd_remarks(args) -> None:
    """Resource usage from the compiler remarks saved by ``build`` (no GPU):
    VGPRs, spills, scratch, LDS and the compiler's waves/SIMD estimate."""
    print(f"{'variant':46s} thr  vgpr spill scratch    lds  waves/SIMD(compiler)")
    for v in load_variants(args.preset):
        got = read_remarks(v) if v.valid() else {}
        if got:
            print(
                f"{v.name:46s} {v.threads:3d} {got.get('VGPRs', -1):5d} {got.get('VGPRs Spill', -1):5d} "
                f"{got.get('ScratchSize', -1):6d} {got.get('LDS Size', -1):6d} {got.get('Occupancy', -1):6d}"
            )


def cmd_occ(args) -> None:
    """Occupancy as measured: MODE 5 kernels log (start, end) per block; the
    peak overlap of blocks over the whole GPU, divided by the WGP count, is the
    resident-block count per WGP the hardware really sustains for that VGPR /
    LDS footprint (s_getreg(HW_ID) reads 0 on this part, so no per-WGP split)."""
    import torch

    print("device:", device_check(torch), flush=True)
    n, k = SHAPES[args.shape]
    m = args.m
    for v in load_variants(args.preset):
        if v.mode != 5 or not v.lib().exists():
            continue
        lib = Lib(v)
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda")
        c = torch.empty(m, n, dtype=torch.float16, device="cuda")
        ws = torch.zeros(lib.ws_bytes(m, n), dtype=torch.uint8, device="cuda")
        for _ in range(3):
            lib.launch(x, w, c, ws)
        torch.accelerator.synchronize()
        rec = ws.view(torch.int64).reshape(-1, 4).cpu().tolist()
        peak = cur = 0
        for _, d in sorted([(r[0], 1) for r in rec] + [(r[1], -1) for r in rec]):
            cur += d
            peak = max(peak, cur)
        mid = (min(r[0] for r in rec) + max(r[1] for r in rec)) // 2
        at_mid = sum(1 for r in rec if r[0] <= mid < r[1])
        wgps = torch.cuda.get_device_properties(0).multi_processor_count
        wpb = math.ceil(lib.threads / 32)
        print(
            f"{v.name:44s} vgpr {lib.vgpr:3d} lds {lib.lds:5d} B  hip-API blk/WGP {lib.blocks}  "
            f"measured: grid {len(rec)} blocks, peak resident {peak} ({peak / wgps:.2f}/WGP), "
            f"mid-run {at_mid} ({at_mid / wgps:.2f}/WGP) -> waves/SIMD {peak / wgps * wpb / 4:.1f} (peak) "
            f"{at_mid / wgps * wpb / 4:.1f} (mid-run)",
            flush=True,
        )


def cmd_isa(args) -> None:
    """Hot-loop instruction mix of the *built* variants: the gfx1030 code object
    is pulled out of each .so (llvm-objdump --offloading) and disassembled, so
    the ISA is that of the binary that was measured, and nothing is compiled."""
    d = OUT / "isa"
    d.mkdir(parents=True, exist_ok=True)
    for v in load_variants(args.preset):
        if not v.valid() or not v.lib().exists():
            continue
        so = d / f"{v.name}.so"
        shutil.copy(v.lib(), so)
        subprocess.run(
            [str(LLVM_BIN / "llvm-objdump"), "--offloading", so.name],
            cwd=d,
            check=True,
            capture_output=True,
        )
        co = d / f"{so.name}.0.hipv4-amdgcn-amd-amdhsa--gfx1030"
        dis = d / f"{v.name}.dis"
        dis.write_text(subprocess.run(
            [str(LLVM_BIN / "llvm-objdump"), "-d", "--mcpu=gfx1030", f"--disassemble-symbols={GEMM_KERNEL_SYM}", str(co)],
            check=True, capture_output=True, text=True).stdout)  # fmt: skip
        for f in d.glob(f"{so.name}.0.*"):
            f.unlink()
        so.unlink()
        print(f"== {v.name}", flush=True)
        subprocess.run([sys.executable, str(HERE / "isa_mix.py"), str(dis), "--objdump",
                        "--dump", str(d / f"{v.name}.loop.txt")], check=False)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--preset", default="base")
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--force", action="store_true")
    b.add_argument("--extra", nargs="*", default=[])
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("run")
    r.add_argument("--preset", default="base")
    r.add_argument("--csv")
    r.add_argument("--label", default="")
    r.add_argument("--shapes", default="")
    r.add_argument("--m", default=",".join(str(m) for m in MS))
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--warm-s", type=float, default=2.0)
    r.add_argument("--win-ms", type=float, default=40.0)
    r.add_argument("--ws-mb", type=float, default=640.0)
    r.add_argument("--fused", action="store_true")
    r.add_argument("--no-variants", action="store_true")
    r.add_argument("--no-check", action="store_true")
    r.add_argument(
        "--pair",
        default="",
        help="comma list of designs to time behind the dequant kernel (needs --fused)",
    )
    r.add_argument("--tunable-file")
    r.add_argument("--tune-ms", type=float, default=25)
    r.add_argument("--tune-iters", type=int, default=10)
    r.add_argument("--rot-mb", type=int, default=640)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("check")
    c.add_argument("--preset", default="base")
    c.add_argument("--csv", help="write one status row per variant (report.py --check)")
    c.set_defaults(fn=cmd_check)
    i = sub.add_parser("info")
    i.add_argument("--preset", default="base")
    i.set_defaults(fn=cmd_info)
    s = sub.add_parser(
        "isa", help="compile to .s and print the hot-loop instruction mix"
    )
    s.add_argument("--preset", default="base")
    s.set_defaults(fn=cmd_isa)
    o = sub.add_parser("occ", help="measured resident blocks per WGP (MODE 5 variants)")
    o.add_argument("--preset", default="occ")
    o.add_argument("--shape", default="in_proj_qkvz")
    o.add_argument("--m", type=int, default=2048)
    o.set_defaults(fn=cmd_occ)
    pr = sub.add_parser(
        "probe",
        help="each design alone for N seconds: TF/s with the sampled shader clock and power",
    )
    pr.add_argument(
        "--preset",
        default="",
        help="variants to probe (names or presets); rocBLAS rows are always added",
    )
    pr.add_argument("--shape", default="in_proj_qkvz")
    pr.add_argument("--m", type=int, default=2048)
    pr.add_argument("--seconds", type=float, default=6.0)
    pr.add_argument(
        "--tail",
        type=float,
        default=3.0,
        help="average over the last N seconds of each run",
    )
    pr.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="passes over the list (reversed on odd passes)",
    )
    pr.add_argument("--ws-mb", type=float, default=640)
    pr.add_argument("--no-rocblas", action="store_true")
    pr.add_argument("--tunable-file")
    pr.add_argument("--csv")
    pr.add_argument("--label", default="")
    pr.set_defaults(fn=cmd_probe)
    rm = sub.add_parser(
        "remarks", help="VGPR / spill / LDS / occupancy from the build remarks (no GPU)"
    )
    rm.add_argument("--preset", default="base")
    rm.set_defaults(fn=cmd_remarks)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
