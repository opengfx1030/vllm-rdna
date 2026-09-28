# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and load the W4A8 explore C ABI (V620 box only).

The kernels live in ``csrc/rocm/explore`` and are deliberately outside the
vLLM build: this module compiles ``w4a8_sdot4_capi.cu`` with hipcc into a
standalone ``.so`` and calls it through ctypes with raw device pointers.
Build with the hipcc of the ROCm that torch uses (``torch.version.hip``),
so one HIP runtime ends up in the process.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXPLORE = ROOT / "csrc" / "rocm" / "explore"
CAPI = EXPLORE / "w4a8_sdot4_capi.cu"
HEADER = EXPLORE / "w4a8_sdot4.cuh"
ABI_VERSION = 1
PROBE_KINDS = {"sdot4": 0, "fdot2": 1, "fma_f32": 2}
PROBE_MACS = {"sdot4": 4, "fdot2": 2, "fma_f32": 1}


def find_hipcc(explicit: str | None = None) -> str:
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    for c in (explicit, os.environ.get("HIPCC"), f"{rocm}/bin/hipcc"):
        if c and Path(c).exists():
            return c
    found = shutil.which("hipcc")
    if found is None:
        raise FileNotFoundError("hipcc not found; set HIPCC or ROCM_PATH")
    return found


def build(
    out_dir: Path | None = None, hipcc: str | None = None, save_temps: bool = False
) -> Path:
    """Compiles the C ABI once per source hash and returns the .so path."""
    hipcc = find_hipcc(hipcc)
    flags = ["-x", "hip", "-O3", "-std=c++17", "--offload-arch=gfx1030"]
    flags += ["-fPIC", "-shared", f"-I{EXPLORE}"]
    digest = hashlib.sha256()
    for part in (CAPI.read_bytes(), HEADER.read_bytes(), " ".join(flags).encode()):
        digest.update(part)
    out_dir = out_dir or ROOT / "build" / "w4a8_sdot4_explore"
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / f"libw4a8_sdot4_explore-{digest.hexdigest()[:12]}.so"
    if so.exists() and not save_temps:
        return so
    cmd = [hipcc, *flags, "-o", str(so), str(CAPI)]
    if save_temps:
        cmd.append("-save-temps")
    subprocess.run(cmd, check=True, cwd=out_dir)
    return so


@dataclass(frozen=True)
class Config:
    id: int
    name: str
    m_tile: int
    n_tile: int


class W4A8Lib:
    """Typed ctypes wrapper; every call raises on a non-zero status."""

    def __init__(self, path: Path):
        lib = ctypes.CDLL(str(path))
        i, p, ll = ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong
        sigs = {
            "w4a8_abi_version": ([], i),
            "w4a8_num_configs": ([], i),
            "w4a8_config_name": ([i], ctypes.c_char_p),
            "w4a8_config_m_tile": ([i], i),
            "w4a8_config_n_tile": ([i], i),
            "w4a8_probe_chains": ([], i),
            "w4a8_error_str": ([i], ctypes.c_char_p),
            "w4a8_pick_split_k": ([i, i, i, i, i], i),
            "w4a8_act_quant": ([p, ll, p, p, p, i, i, i, i, p], i),
            "w4a8_gemm": ([p] * 7 + [i] * 8 + [p], i),
            "w4a8_probe": ([i, i, i, p, p], i),
        }
        for name, (args, res) in sigs.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
        self._lib = lib
        if lib.w4a8_abi_version() != ABI_VERSION:
            raise RuntimeError(f"{path} has ABI {lib.w4a8_abi_version()}")
        self.configs = [
            Config(
                c,
                lib.w4a8_config_name(c).decode(),
                lib.w4a8_config_m_tile(c),
                lib.w4a8_config_n_tile(c),
            )
            for c in range(lib.w4a8_num_configs())
        ]
        self.probe_chains = lib.w4a8_probe_chains()

    def _check(self, what: str, code: int) -> int:
        if code < 0 or (code > 0 and what != "pick_split_k"):
            msg = self._lib.w4a8_error_str(code).decode()
            raise RuntimeError(f"w4a8 {what} failed: {code} ({msg})")
        return code

    @staticmethod
    def _stream() -> int:
        return torch.cuda.current_stream().cuda_stream

    def config(self, name_or_id: str | int) -> Config:
        for c in self.configs:
            if name_or_id in (c.id, c.name):
                return c
        raise KeyError(name_or_id)

    def pick_split_k(self, m: int, n: int, k: int, group_size: int, cfg: int) -> int:
        return self._check(
            "pick_split_k", self._lib.w4a8_pick_split_k(m, n, k, group_size, cfg)
        )

    def act_quant(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        asum: torch.Tensor,
        group_size: int,
        m_tile: int,
    ) -> None:
        """x fp16 [M, K] -> a (tiled int8), a_scale [M] f32, asum (tiled)."""
        m, k = x.shape
        self._check(
            "act_quant",
            self._lib.w4a8_act_quant(
                x.data_ptr(),
                x.stride(0),
                a.data_ptr(),
                a_scale.data_ptr(),
                asum.data_ptr(),
                m,
                k,
                group_size,
                m_tile,
                self._stream(),
            ),
        )

    def gemm(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        a_scale: torch.Tensor,
        asum: torch.Tensor,
        out: torch.Tensor,
        k: int,
        group_size: int,
        zero_offset: int,
        cfg: int,
        split_k: int = 0,
    ) -> None:
        m, n = out.shape
        self._check(
            "gemm",
            self._lib.w4a8_gemm(
                a.data_ptr(),
                w.data_ptr(),
                qzeros.data_ptr(),
                scales.data_ptr(),
                a_scale.data_ptr(),
                asum.data_ptr(),
                out.data_ptr(),
                m,
                n,
                k,
                group_size,
                zero_offset,
                cfg,
                split_k,
                int(out.dtype == torch.float32),
                self._stream(),
            ),
        )

    def probe(self, kind: str, blocks: int, iters: int, out: torch.Tensor) -> None:
        self._check(
            "probe",
            self._lib.w4a8_probe(
                PROBE_KINDS[kind], blocks, iters, out.data_ptr(), self._stream()
            ),
        )


def load(hipcc: str | None = None, save_temps: bool = False) -> W4A8Lib:
    return W4A8Lib(build(hipcc=hipcc, save_temps=save_temps))
