#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused probe: W4A16 prefill (gptq_gemm_rdna2_prefill) output norm vs k.

Reproduces the k=4352 zero-output anomaly from probe_w4a8_op.py. Uses the
same numpy packing as probe_w4a8_op.py (no vLLM config needed).
"""
import numpy as np
import torch

SHUFFLE_SLOTS = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_k_major(q_kn):
    k, n = q_kn.shape
    q = q_kn.astype(np.uint32).reshape(k // 8, 8, n)
    out = np.zeros((k // 8, n), dtype=np.uint32)
    for i in range(8):
        out |= q[:, i, :] << np.uint32(4 * i)
    return out


def _exllama_shuffle(packed):
    packed = packed.astype(np.uint32)
    out = np.zeros_like(packed)
    for slot, k_off in enumerate(SHUFFLE_SLOTS):
        nibble = (packed >> np.uint32(4 * k_off)) & np.uint32(0xF)
        out |= nibble << np.uint32(4 * slot)
    return out


def _pack_zeros_n_major(z_gn):
    g, n = z_gn.shape
    z = z_gn.astype(np.uint32).reshape(g, n // 8, 8)
    out = np.zeros((g, n // 8), dtype=np.uint32)
    for j in range(8):
        out |= z[:, :, j] << np.uint32(4 * j)
    return out


def main():
    m, n, g = 225, 5120, 32
    rng = np.random.default_rng(0)
    for k in [1536, 2176, 2560, 4096, 4352, 5120, 8192]:
        if k % g != 0 or k % 32 != 0:
            continue
        x = (0.25 * torch.randn((m, k), device="cuda", dtype=torch.float32)).to(
            torch.float16
        )
        q_int4 = rng.integers(0, 16, size=(k, n), dtype=np.int32)
        scales = (0.05 * rng.random((k // g, n)) + 0.01).astype(np.float32)
        zeros = rng.integers(0, 16, size=(k // g, n), dtype=np.int32)
        wq = torch.from_numpy(
            _exllama_shuffle(_pack_k_major(q_int4)).astype(np.uint32)
        ).to("cuda")
        wzp = torch.from_numpy(_pack_zeros_n_major(zeros).astype(np.uint32)).to("cuda")
        ws = torch.from_numpy(scales.astype(np.float16)).to("cuda")
        gi = torch.empty(0, dtype=torch.int32, device="cuda")
        out = torch.ops._rocm_C.gptq_gemm_rdna2_prefill(x, wq, wzp, ws, gi, False)
        out8 = torch.ops._rocm_C.w4a8_gemm_rdna2(x, wq, wzp, ws, gi, False)
        print(
            f"k={k:5d} gptq norm={out.norm().item():10.4f} max={out.abs().max().item():8.4f} | "
            f"w4a8 norm={out8.norm().item():10.4f} max={out8.abs().max().item():8.4f}"
        )


if __name__ == "__main__":
    main()
