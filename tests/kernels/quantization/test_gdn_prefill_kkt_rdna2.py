#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests for the ROCm RDNA2 GDN prefill KKT kernel.

Compares ``torch.ops._rocm_C.gdn_prefill_kkt_rdna2``
(csrc/rocm/gdn_prefill_kkt_rdna2.cu) against the Triton reference
``chunk_scaled_dot_kkt_fwd`` on random tensors.

The kernel computes the strictly-lower-triangular matrix
``A[i, j] = beta[i] * exp(g[i] - g[j]) * (k[i] . k[j])`` for ``i > j``
(0 otherwise) per (chunk, head). On gfx1030 the reference keeps the
beta-scaled key in fp32 (``_CAST_DOT_TO_K_DTYPE=False``), so the dot is a
pure fp32 x fp32 FMA accumulation -- the HIP port matches this exactly.

Tolerance: the output A is fp32. Both kernels accumulate the K=128 dot with
sequential fp32 FMA, so the only divergence is summation order, bounded by
~128 * 2^-24 ~= 7.6e-6 relative error in the worst case (typically far less
for random-sign terms). rtol=atol=1e-4 leaves >10x headroom; the masked
(zero) regions match bit-exactly.

The kernel is only built/registered for gfx1030; tests skip elsewhere.

Run `pytest tests/kernels/quantization/test_gdn_prefill_kkt_rdna2.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN prefill KKT kernel is ROCm-only",
                allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN prefill KKT kernel is gfx1030-only",
                allow_module_level=True)

from vllm import _rocm_C  # noqa: E402,F401  (registers torch.ops._rocm_C.*)

from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (  # noqa: E402
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
)

device = "cuda"

K = 128  # head_k_dim; the HIP kernel requires K == 128
BT = 64  # FLA_CHUNK_SIZE; the HIP kernel requires BT == 64


def _run_ref(k, beta, g, cu_seqlens, chunk_indices):
    return chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        output_dtype=torch.float32,
    )


def _run_hip(k, beta, g, A, cu_seqlens, chunk_indices):
    torch.ops._rocm_C.gdn_prefill_kkt_rdna2(k, beta, g, A, cu_seqlens,
                                            chunk_indices)
    return A


@pytest.mark.parametrize("T", [64, 130, 200])
@pytest.mark.parametrize("H", [4, 2])
def test_gdn_prefill_kkt_rdna2_parity(T, H):
    # T=64 is an exact chunk; T=130/200 exercise the chunk-boundary tail
    # (last chunk shorter than BT=64).
    B, Hg = 2, H
    torch.manual_seed(0)
    k = torch.randn(B, T, Hg, K, device=device, dtype=torch.float16)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32)
    g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.5

    A_ref = _run_ref(k, beta, g, None, None)
    A_hip = torch.empty(B, T, H, BT, device=device, dtype=torch.float32)
    _run_hip(k, beta, g, A_hip, None, None)

    torch.testing.assert_close(A_hip, A_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("H", [4, 2])
@pytest.mark.parametrize("lens", [[130, 200, 65], [64, 64, 64], [100, 70]])
def test_gdn_prefill_kkt_rdna2_varlen_parity(H, lens):
    # Variable-length multi-sequence batch (B=1, flattened T). Each sequence
    # has its own chunk boundaries; some sequences are not multiples of BT.
    Hg = H
    torch.manual_seed(1)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    T = int(cu_seqlens[-1])
    B = 1
    k = torch.randn(B, T, Hg, K, device=device, dtype=torch.float16)
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32)
    g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.5
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    A_ref = _run_ref(k, beta, g, cu_seqlens, chunk_indices)
    A_hip = torch.empty(B, T, H, BT, device=device, dtype=torch.float32)
    _run_hip(k, beta, g, A_hip, cu_seqlens, chunk_indices)

    torch.testing.assert_close(A_hip, A_ref, rtol=1e-4, atol=1e-4)
