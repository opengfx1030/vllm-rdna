#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests for the ROCm RDNA2 GDN prefill fused
solve_tril + recompute_w_u kernel.

Compares ``torch.ops._rocm_C.gdn_prefill_solve_wy_rdna2``
(csrc/rocm/gdn_prefill_solve_wy_rdna2.cu) against the chained Triton
references
``vllm.third_party.flash_linear_attention.ops.solve_tril`` and
``vllm.third_party.flash_linear_attention.ops.wy_fast.recompute_w_u_fwd``
on synthetic inputs at real shapes.

The kernel fuses two Triton launches into one workgroup by keeping
A_inv = (I + A_lower)^-1 resident in LDS between phases. Both phases are
verified together because in production the A_inv tensor never leaves the
on-chip path; here we still write it to global so the test can compare.

The kernel is only built/registered for gfx1030; tests skip elsewhere.

Tolerance: the fp16 outputs (A_inv, w, u) are produced by the same fp32
accumulation in both implementations but with potentially different
reduction order (forward substitution: Triton tree vs HIP sequential;
the W/U dots: Triton's V_DOT2 lowering order vs the HIP kernel's
k-ascending pair order). At magnitudes ~ O(1-10) the fp16 ULP is ~1e-3,
so rtol/atol = 1e-3 leaves ~1-2 ULP headroom -- tight, as specified.

Run ``pytest tests/kernels/quantization/test_gdn_prefill_solve_wy_rdna2.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN prefill solve_wy kernel is ROCm-only",
                allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN prefill solve_wy kernel is gfx1030-only",
                allow_module_level=True)

from vllm import _rocm_C  # noqa: E402,F401  (registers torch.ops._rocm_C.*)

from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
)
from vllm.third_party.flash_linear_attention.ops.solve_tril import (  # noqa: E402
    solve_tril,
)
from vllm.third_party.flash_linear_attention.ops.wy_fast import (  # noqa: E402
    recompute_w_u_fwd,
)

device = "cuda"

BT = 64   # FLA_CHUNK_SIZE; the HIP kernel requires BT == 64
K = 128   # head_k_dim; HIP requires K == 128
V = 128   # head_v_dim; HIP requires V == 128


def _make_inputs(B: int, T: int, H: int, Hg: int, seed: int):
    """Build synthetic inputs at real shapes.

    A is built strictly lower-triangular with entries ~O(0.1); the inverse
    of (I + A_lower) stays O(1) so fp16 outputs land in a normal range.
    k, v, beta, g are otherwise unconstrained -- they feed the w/u recompute
    step which doesn't depend on how A was constructed.
    """
    assert H % Hg == 0
    gen = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(B, T, H, BT, device=device, dtype=torch.float32,
                    generator=gen) * 0.1
    # Strictly lower-triangular within each chunk: for global token t the
    # chunk-local row is t % BT, and column j is kept only when j < row.
    rows = torch.arange(T, device=device) % BT
    cols = torch.arange(BT, device=device)
    mask = cols[None, :] < rows[:, None]  # [T, BT]
    A = A * mask[None, :, None, :]
    k = torch.randn(B, T, Hg, K, device=device, dtype=torch.float16,
                    generator=gen)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16,
                    generator=gen)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32,
                      generator=gen)
    g = torch.randn(B, T, H, device=device, dtype=torch.float32,
                    generator=gen) * 0.3
    return A, k, v, beta, g


def _run_ref(A, k, v, beta, g, cu_seqlens, chunk_indices):
    # Golden = the two Triton references chained.
    A_inv = solve_tril(A=A, cu_seqlens=cu_seqlens,
                       chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, g_cumsum=g, A=A_inv, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices)
    return A_inv, w, u


def _run_hip(A, k, v, beta, g, cu_seqlens, chunk_indices):
    A_inv = torch.empty(A.shape, device=device, dtype=torch.float16)
    w = torch.empty(k.shape[0], k.shape[1], v.shape[2], K,
                    device=device, dtype=torch.float16)
    u = torch.empty(v.shape, device=device, dtype=torch.float16)
    torch.ops._rocm_C.gdn_prefill_solve_wy_rdna2(
        A, k, v, beta, g, A_inv, w, u, cu_seqlens, chunk_indices)
    return A_inv, w, u


@pytest.mark.parametrize("H,Hg", [(12, 4), (4, 2), (2, 1)])
@pytest.mark.parametrize("T", [64, 65, 130, 200])
def test_gdn_prefill_solve_wy_rdna2_parity(T, H, Hg):
    """Fixed-length path; covers exact-chunk, mid-chunk-tail, and multi-chunk
    shapes at the production (H=12, Hg=4) and smaller head configs.
    """
    B = 2
    torch.manual_seed(0)
    A, k, v, beta, g = _make_inputs(B, T, H, Hg, seed=0)

    A_ref, w_ref, u_ref = _run_ref(A, k, v, beta, g, None, None)
    A_hip, w_hip, u_hip = _run_hip(A, k, v, beta, g, None, None)

    torch.testing.assert_close(A_hip, A_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w_hip, w_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(u_hip, u_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("H,Hg", [(12, 4), (4, 2)])
@pytest.mark.parametrize(
    "lens",
    [[130, 200, 65], [64, 64, 64], [100, 70], [1, 63, 64], [128]],
)
def test_gdn_prefill_solve_wy_rdna2_varlen_parity(H, Hg, lens):
    """Variable-length multi-sequence batch (B == 1 flattened). Sequences of
    mixed lengths -- some exact multiples of BT, some tail chunks -- and a
    single-sequence degenerate case (one chunk, tail of length 1).
    """
    torch.manual_seed(1)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(),
        device=device, dtype=torch.int32,
    )
    T = int(cu_seqlens[-1])
    A, k, v, beta, g = _make_inputs(B=1, T=T, H=H, Hg=Hg, seed=1)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    A_ref, w_ref, u_ref = _run_ref(A, k, v, beta, g, cu_seqlens,
                                   chunk_indices)
    A_hip, w_hip, u_hip = _run_hip(A, k, v, beta, g, cu_seqlens,
                                   chunk_indices)

    torch.testing.assert_close(A_hip, A_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w_hip, w_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(u_hip, u_ref, rtol=1e-3, atol=1e-3)
