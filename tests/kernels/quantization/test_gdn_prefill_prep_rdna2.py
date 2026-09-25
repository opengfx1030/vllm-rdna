#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests for the ROCm RDNA2 GDN prefill prep + cumsum kernel.

Compares ``torch.ops._rocm_C.gdn_prefill_prep_rdna2``
(csrc/rocm/gdn_prefill_prep_rdna2.cu) against the Triton reference
``fused_post_conv_prep`` + ``chunk_local_cumsum_scalar`` (chunk_size=64,
reverse=False) in sequence on random tensors.

The kernel fuses two steps of the Qwen3.5/3.6 GDN prefill pipeline:

  1. ``fused_post_conv_prep`` -- split mixed_qkv into q/k/v, L2-normalise
     q and k, compute g and beta from a/b/A_log/dt_bias.
  2. ``chunk_local_cumsum_scalar`` -- per-chunk inclusive cumsum of g
     (resets at every BT=64 chunk boundary; identical to the cumsum kernel
     used inside ``chunk_gated_delta_rule_fwd``).

The fused HIP kernel writes q/k/v directly from mixed_qkv and writes
g_cumsum and beta in one launch (g and g_cumsum get the same shape/dtype
because the reference's cumsum is an in-place overwrite of g).

The kernel is only built/registered for gfx1030; tests skip elsewhere.

Run `pytest tests/kernels/quantization/test_gdn_prefill_prep_rdna2.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN prefill prep kernel is ROCm-only",
                allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN prefill prep kernel is gfx1030-only",
                allow_module_level=True)

from vllm import _rocm_C  # noqa: E402,F401  (registers torch.ops._rocm_C.*)

from vllm.third_party.flash_linear_attention.ops.cumsum import (  # noqa: E402
    chunk_local_cumsum,
)
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E402
    fused_post_conv_prep,
)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
)

device = "cuda"

K = 128   # head_k_dim; the HIP kernel requires K == 128
V = 128   # head_v_dim; the HIP kernel requires V == 128
BT = 64   # FLA_CHUNK_SIZE; the HIP kernel requires BT == 64


def _make_inputs(L: int, H: int, HV: int, seed: int,
                 a_log_dtype: torch.dtype = torch.float32,
                 dt_bias_dtype: torch.dtype = torch.float32):
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(L, qkv_dim, device=device,
                            dtype=torch.float16, generator=gen)
    # vLLM casts dt_bias to the model dtype on load while A_log stays
    # fp32, so all dtype combinations must work.
    a = torch.randn(L, HV, device=device, dtype=torch.float16, generator=gen)
    b = torch.randn(L, HV, device=device, dtype=torch.float16, generator=gen)
    A_log = torch.randn(HV, device=device, dtype=torch.float32,
                        generator=gen).to(a_log_dtype)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32,
                          generator=gen).to(dt_bias_dtype)
    return mixed_qkv, a, b, A_log, dt_bias


def _run_ref(mixed_qkv, a, b, A_log, dt_bias, num_k_heads, cu_seqlens,
             chunk_indices):
    # Step 1: fused_post_conv_prep.
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=num_k_heads,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    # Step 2: chunk_local_cumsum_scalar (reverse=False, head_first=False).
    # g is [L, HV]; the cumsum helper expects 3D [B, T, H]. Reshape to
    # [1, L, HV] to match the real prefill call site (chunk.py:37).
    g_cumsum = chunk_local_cumsum(
        g.unsqueeze(0),
        chunk_size=BT,
        reverse=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=False,
        output_dtype=torch.float32,
    ).squeeze(0)
    return q, k, v, g_cumsum, beta


def _run_hip(mixed_qkv, a, b, A_log, dt_bias, q, k_out, v, g_cumsum, beta,
             cu_seqlens, chunk_indices):
    torch.ops._rocm_C.gdn_prefill_prep_rdna2(
        mixed_qkv, a, b, A_log, dt_bias, q, k_out, v, g_cumsum, beta,
        cu_seqlens, chunk_indices,
    )


def _check_parity(q_hip, k_hip, v_hip, g_cumsum_hip, beta_hip,
                  q_ref, k_ref, v_ref, g_cumsum_ref, beta_ref):
    # q/k/v are fp16: rtol/atol 1e-3 (L2norm + fp16 round-trip).
    torch.testing.assert_close(q_hip, q_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_hip, k_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_hip, v_ref, rtol=1e-3, atol=1e-3)
    # g_cumsum and beta are fp32; reference cumsum is exact fp32 adds, so
    # the only divergence is expf/log1pf reduction order (negligible).
    torch.testing.assert_close(g_cumsum_hip, g_cumsum_ref, rtol=1e-5,
                               atol=1e-5)
    torch.testing.assert_close(beta_hip, beta_ref, rtol=1e-5, atol=1e-5)


def _alloc_outputs(L, H, HV):
    q = torch.empty(L, H, K, device=device, dtype=torch.float16)
    k_out = torch.empty(L, H, K, device=device, dtype=torch.float16)
    v = torch.empty(L, HV, V, device=device, dtype=torch.float16)
    g_cumsum = torch.empty(L, HV, device=device, dtype=torch.float32)
    beta = torch.empty(L, HV, device=device, dtype=torch.float32)
    return q, k_out, v, g_cumsum, beta


@pytest.mark.parametrize("L", [64, 130, 200])
@pytest.mark.parametrize("H,HV", [(4, 12), (2, 4)])
@pytest.mark.parametrize(
    "a_log_dtype,dt_bias_dtype",
    [(torch.float32, torch.float32), (torch.float16, torch.float16),
     (torch.float32, torch.float16)],
)
def test_gdn_prefill_prep_rdna2_parity(L, H, HV, a_log_dtype, dt_bias_dtype):
    # L=64 is an exact chunk; L=130/200 exercise the chunk-boundary tail
    # (last chunk shorter than BT=64).
    torch.manual_seed(0)
    mixed_qkv, a, b, A_log, dt_bias = _make_inputs(
        L, H, HV, seed=42, a_log_dtype=a_log_dtype,
        dt_bias_dtype=dt_bias_dtype)

    q_ref, k_ref, v_ref, g_cumsum_ref, beta_ref = _run_ref(
        mixed_qkv, a, b, A_log, dt_bias, H, None, None)
    q, k_out, v, g_cumsum, beta = _alloc_outputs(L, H, HV)
    _run_hip(mixed_qkv, a, b, A_log, dt_bias, q, k_out, v, g_cumsum, beta,
             None, None)

    _check_parity(q, k_out, v, g_cumsum, beta,
                  q_ref, k_ref, v_ref, g_cumsum_ref, beta_ref)


@pytest.mark.parametrize("H,HV", [(4, 12), (2, 4)])
@pytest.mark.parametrize("lens", [[130, 200, 65], [64, 64, 64], [100, 70],
                                 [1, 63, 128]])
def test_gdn_prefill_prep_rdna2_varlen_parity(H, HV, lens):
    # Variable-length multi-sequence batch (B=1, flattened L). Each
    # sequence has its own chunk boundaries; some sequences are not
    # multiples of BT=64. Last entry [1, 63, 128] exercises a 1-token
    # sequence (one cumsum chunk of length 1) and a 128-token sequence
    # (two full chunks).
    torch.manual_seed(1)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    L = int(cu_seqlens[-1])
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    mixed_qkv, a, b, A_log, dt_bias = _make_inputs(L, H, HV, seed=43)

    q_ref, k_ref, v_ref, g_cumsum_ref, beta_ref = _run_ref(
        mixed_qkv, a, b, A_log, dt_bias, H, cu_seqlens, chunk_indices)
    q, k_out, v, g_cumsum, beta = _alloc_outputs(L, H, HV)
    _run_hip(mixed_qkv, a, b, A_log, dt_bias, q, k_out, v, g_cumsum, beta,
             cu_seqlens, chunk_indices)

    _check_parity(q, k_out, v, g_cumsum, beta,
                  q_ref, k_ref, v_ref, g_cumsum_ref, beta_ref)


def test_gdn_prefill_prep_rdna2_zero_length_returns_empty():
    # Guard: L=0 must short-circuit and return without launching anything.
    torch.manual_seed(2)
    H, HV = 4, 12
    mixed_qkv, a, b, A_log, dt_bias = _make_inputs(0, H, HV, seed=44)
    q, k_out, v, g_cumsum, beta = _alloc_outputs(0, H, HV)
    _run_hip(mixed_qkv, a, b, A_log, dt_bias, q, k_out, v, g_cumsum, beta,
             None, None)
    assert q.shape == (0, H, K)
    assert k_out.shape == (0, H, K)
    assert v.shape == (0, HV, V)
    assert g_cumsum.shape == (0, HV)
    assert beta.shape == (0, HV)
