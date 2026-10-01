#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-level W4A8 sdot4 GEMM correctness sweep (single GPU, gfx1030).

For each (m, k, n, group) shape, packs an IDENTICAL int4 weight buffer and
compares the W4A8 fast path (`w4a8_gemm_rdna2`) against the W4A16 prefill
reference (`gptq_gemm_rdna2_prefill`) via rel-L2, and against a true fp32
dequant reference to classify which of the two is wrong.

Runs BOTH quant variants on the same shape:
  * GPTQ  ``uint4b8``  -> use_v2_format=False (zero_offset = 1, kernel adds 1)
  * AWQ   ``uint4``    -> use_v2_format=True  (zero_offset = 0, literal zero)

The weight buffer the kernels read is byte-identical to what
RDNA2W4A16LinearKernel leaves behind (pack_k_major + gptq_shuffle for W,
pack_n_major for qzeros).  This is the production apples-to-apples
comparison.  Modes:

  --mode shapes         run the given (or default in-model) shape set
  --mode bisect-k       fix n,group; sweep k over [k_lo, k_hi) step k_step
  --mode bisect-group   fix m,n,k; sweep group over a list

Usage:
  python probe_w4a8_op.py --mode shapes
  python probe_w4a8_op.py --mode bisect-k --n 5120 --group 32 --k-lo 512 --k-hi 6144 --k-step 32
  python probe_w4a8_op.py --mode bisect-group --m 2001 --n 5120 --k 4352 --groups 32,64,128
"""
import argparse
import sys

import numpy as np
import torch

# The full distinct fast-path shape set extracted from the serve logs
# (grep VLLM_RDNA2_W4A8 /home/chenco_adm/w4a8_runs/*/serve.log).  All group=32.
DEFAULT_SHAPES = [
    (225, 1536, 5120, 32),
    (225, 4352, 5120, 32),
    (225, 5120, 3584, 32),
    (225, 5120, 4096, 32),
    (225, 5120, 8704, 32),
    (2001, 1536, 5120, 32),
    (2001, 4352, 5120, 32),
    (2001, 5120, 3584, 32),
    (2001, 5120, 4096, 32),
    (2001, 5120, 8704, 32),
    (2048, 1536, 5120, 32),
    (2048, 4352, 5120, 32),
    (2048, 5120, 3584, 32),
    (2048, 5120, 4096, 32),
    (2048, 5120, 8704, 32),
]

# gptq_shuffle slot permutation (from explore/reference.py, matches the .cuh).
SHUFFLE_SLOTS = (0, 2, 4, 6, 1, 3, 5, 7)
# kAPerm for the A tile (matches csrc/rocm/explore/w4a8_sdot4.cuh).
A_PERM = (0, 4, 1, 5, 2, 6, 3, 7)
MTILE = 8


def _pack_k_major(q_kn: np.ndarray) -> np.ndarray:
    """[K, N] int4 -> [K/8, N] uint32, nibble i = q[k0+i, n] << 4i."""
    k, n = q_kn.shape
    q = q_kn.astype(np.uint32).reshape(k // 8, 8, n)
    out = np.zeros((k // 8, n), dtype=np.uint32)
    for i in range(8):
        out |= q[:, i, :] << np.uint32(4 * i)
    return out


def _exllama_shuffle(packed: np.ndarray) -> np.ndarray:
    packed = packed.astype(np.uint32)
    out = np.zeros_like(packed)
    for slot, k_off in enumerate(SHUFFLE_SLOTS):
        nibble = (packed >> np.uint32(4 * k_off)) & np.uint32(0xF)
        out |= nibble << np.uint32(4 * slot)
    return out


def _pack_zeros_n_major(z_gn: np.ndarray) -> np.ndarray:
    """[G, N] int4 -> [G, N/8] uint32, nibble j = z[g, n0+j] << 4j."""
    g, n = z_gn.shape
    z = z_gn.astype(np.uint32).reshape(g, n // 8, 8)
    out = np.zeros((g, n // 8), dtype=np.uint32)
    for j in range(8):
        out |= z[:, :, j] << np.uint32(4 * j)
    return out


def _quantize_random(k: int, n: int, group: int, seed: int):
    """Deterministic random int4 weights, fp16 scales, int32 zeros."""
    rng = np.random.default_rng(seed)
    g = k // group
    q_int4 = rng.integers(0, 16, size=(k, n), dtype=np.int32)
    scales = (0.05 * rng.random((g, n)) + 0.01).astype(np.float32)
    zeros = rng.integers(0, 16, size=(g, n), dtype=np.int32)
    return q_int4, scales, zeros


def _pack_weights(q_int4, scales_gn, zeros_gn, group):
    """Produce the exact buffers the kernels consume."""
    w_q = _exllama_shuffle(_pack_k_major(q_int4)).astype(np.uint32)
    w_zp = _pack_zeros_n_major(zeros_gn).astype(np.uint32)
    w_s = scales_gn.astype(np.float16)
    return w_q, w_zp, w_s


def _fp32_reference(x_mk, q_int4, scales_gn, zeros_gn, group, gptq):
    """True dequant reference: w = (q - z) * scale, out = x @ w."""
    k, n = q_int4.shape
    s_full = np.repeat(scales_gn, group, axis=0).astype(np.float32)
    z_full = np.repeat(zeros_gn, group, axis=0).astype(np.float32)
    if gptq:
        z_full = z_full + 1.0  # GPTQv1: stored zero = actual - 1
    w_fp = (q_int4.astype(np.float32) - z_full) * s_full
    return x_mk.astype(np.float32) @ w_fp


def run_shape(m, k, n, group, seed=0):
    """Return dict with rel-L2 metrics for one (m,k,n,group), both variants."""
    q_int4, scales_gn, zeros_gn = _quantize_random(k, n, group, seed)
    w_q, w_zp, w_s = _pack_weights(q_int4, scales_gn, zeros_gn, group)
    x_mk = (0.25 * torch.randn((m, k), device="cuda", dtype=torch.float32)).to(
        torch.float16
    )
    w_q_t = torch.from_numpy(w_q).to("cuda")
    w_zp_t = torch.from_numpy(w_zp).to("cuda")
    w_s_t = torch.from_numpy(w_s).to("cuda")
    g_idx = torch.empty(0, dtype=torch.int32, device="cuda")

    res = {"shape": (m, k, n, group)}
    for variant, use_v2, gptq in (("AWQ", True, False), ("GPTQ", False, True)):
        out_w4a8 = torch.ops._rocm_C.w4a8_gemm_rdna2(
            x_mk, w_q_t, w_zp_t, w_s_t, g_idx, use_v2
        )
        out_gptq = torch.ops._rocm_C.gptq_gemm_rdna2_prefill(
            x_mk, w_q_t, w_zp_t, w_s_t, g_idx, use_v2
        )
        ref = torch.from_numpy(
            _fp32_reference(
                x_mk.cpu().numpy().astype(np.float32),
                q_int4,
                scales_gn,
                zeros_gn,
                group,
                gptq,
            )
        ).to("cuda")

        def rel_l2(a, b):
            den = b.to(torch.float32).norm()
            if den.item() == 0.0:
                return float("nan")
            return ((a.to(torch.float32) - b.to(torch.float32)).norm() / den).item()

        res[f"{variant}_w4a8_vs_gptq"] = rel_l2(out_w4a8, out_gptq)
        res[f"{variant}_w4a8_vs_ref"] = rel_l2(out_w4a8, ref)
        res[f"{variant}_gptq_vs_ref"] = rel_l2(out_gptq, ref)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="shapes", choices=["shapes", "bisect-k", "bisect-group"])
    ap.add_argument("--m", type=int, default=2001)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--k-lo", type=int, default=512)
    ap.add_argument("--k-hi", type=int, default=6144)
    ap.add_argument("--k-step", type=int, default=32)
    ap.add_argument("--k", type=int, default=4352)
    ap.add_argument("--groups", default="32,64,128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shapes", default=None, help="comma list of m,k,n,group")
    args = ap.parse_args()

    if args.mode == "shapes":
        shapes = DEFAULT_SHAPES
        if args.shapes:
            shapes = [
                tuple(int(x) for x in s.split(","))
                for s in args.shapes.split(";")
            ]
    elif args.mode == "bisect-k":
        shapes = [
            (args.m, kk, args.n, args.group)
            for kk in range(args.k_lo, args.k_hi, args.k_step)
            if kk % 32 == 0 and kk % args.group == 0
        ]
    else:  # bisect-group
        groups = [int(x) for x in args.groups.split(",")]
        shapes = [
            (args.m, args.k, args.n, g)
            for g in groups
            if args.k % g == 0
        ]

    hdr = f"{'shape':>22} | {'AWQ w4a8/gptq':>14} | {'AWQ w4a8/ref':>13} | {'AWQ gptq/ref':>13} | {'GPTQ w4a8/gptq':>15} | {'GPTQ w4a8/ref':>14} | {'GPTQ gptq/ref':>14}"
    print(hdr)
    print("-" * len(hdr))
    for (m, k, n, group) in shapes:
        try:
            r = run_shape(m, k, n, group, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            print(f"{f'({m},{k},{n},{group})':>22} | ERROR: {e!r}")
            continue
        def fmt(v):
            return f"{v:.4f}" if v == v else "  nan"
        print(
            f"{f'({m},{k},{n},{group})':>22} | {fmt(r['AWQ_w4a8_vs_gptq']):>14} | "
            f"{fmt(r['AWQ_w4a8_vs_ref']):>13} | {fmt(r['AWQ_gptq_vs_ref']):>13} | "
            f"{fmt(r['GPTQ_w4a8_vs_gptq']):>15} | {fmt(r['GPTQ_w4a8_vs_ref']):>14} | "
            f"{fmt(r['GPTQ_gptq_vs_ref']):>14}"
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
