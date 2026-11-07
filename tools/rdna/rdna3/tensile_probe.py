#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G16: do the gfx1100 rocBLAS and hipBLASLt Tensile libraries exist and work?

Static: the ROCm wheels keep the device code of the BLAS libraries in kpack
archives (``_rocm_sdk_libraries/.kpack/blas_lib_<arch>.kpack``) or, unpacked, in
``lib/rocblas/library`` and ``lib/hipblaslt/library`` files named after the arch;
the probe lists what exists and fails when there is nothing for the arch.
Dynamic: fp16 and bf16 GEMMs of the Flash-Next dense shapes (M = decode rows up to
a prefill chunk, N and K of the projections and the lm_head shard) through
``torch.matmul`` with both BLAS backends (``cublas`` = rocBLAS, ``cublaslt`` =
hipBLASLt), against a CPU fp32 reference on a column sample.

    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/tensile_probe.py --arch gfx1100
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

# (M, N, K): decode and verify rows, a prefill chunk, qkv / o / shared expert /
# lm_head shapes of Flash-Next at TP1 (hidden 2560, vocab 248320).
SHAPES = [
    (1, 2560, 2560),
    (3, 5120, 2560),
    (24, 2560, 640),
    (64, 12288, 2560),
    (2048, 2560, 2560),
    (2048, 12288, 2560),
    (3, 248320, 2560),
    (48, 248320, 2560),
]
BACKENDS = ("cublas", "cublaslt")


def static_report(arch: str) -> list[str]:
    spec = importlib.util.find_spec("_rocm_sdk_libraries")
    if spec is None or not spec.submodule_search_locations:
        return []
    root = Path(next(iter(spec.submodule_search_locations)))
    found = [p for p in (root / ".kpack").glob(f"*{arch}*") if p.is_file()]
    for library in ("rocblas", "hipblaslt"):
        found += list((root / "lib" / library / "library").glob(f"*{arch}*"))
    return sorted(str(p.relative_to(root)) for p in found)


def check_gemm(backend: str, dtype: torch.dtype, shape: tuple[int, int, int]) -> str:
    m, n, k = shape
    try:
        torch.backends.cuda.preferred_blas_library(backend)
    except Exception as e:  # noqa: BLE001
        return f"SKIP backend unavailable ({type(e).__name__})"
    gen = torch.Generator(device="cuda").manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=dtype, generator=gen)
    w = torch.randn(n, k, device="cuda", dtype=dtype, generator=gen) * 0.05
    out = torch.matmul(x, w.t())
    torch.accelerator.synchronize()
    sample = min(n, 64)
    ref = x.float().cpu() @ w[:sample].float().cpu().t()
    got = out[:, :sample].float().cpu()
    if not bool(torch.isfinite(got).all()):
        return "FAIL non-finite output"
    tolerance = 2e-2 * float(ref.abs().max()) + 1e-3
    error = float((got - ref).abs().max())
    verdict = "OK" if error <= tolerance else "FAIL"
    return f"{verdict} max_err {error:.3e} tol {tolerance:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()

    files = static_report(args.arch)
    print(f"static: {len(files)} BLAS device-code files for {args.arch}")
    for name in files[:12]:
        print(f"  {name}")
    if not files:
        print("FAIL: no rocBLAS/hipBLASLt device code for this arch", file=sys.stderr)
        return 1
    found_arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if found_arch != args.arch:
        print(f"FAIL: visible device is {found_arch}", file=sys.stderr)
        return 1

    failures = 0
    for backend in BACKENDS:
        for dtype in (torch.float16, torch.bfloat16):
            for shape in SHAPES:
                verdict = check_gemm(backend, dtype, shape)
                failures += verdict.startswith("FAIL")
                print(f"{backend:9} {str(dtype)[6:]:9} {shape}: {verdict}")
    torch.backends.cuda.preferred_blas_library("default")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
