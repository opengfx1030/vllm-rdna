# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8 weight-only shadows of the dense projections (T45/T45b).

Ported from leapdragon/vllm-rdna2-qwen (5d54e58f0, 77d25862e). CPU: the
blocked dequant GEMM of the shadows-only mode equals one full dequant GEMM.
GPU (gfx10x, ops built): rdna_dense_gemm runs gemv_i8_rdna2 for decode-shaped
batches and fp16 for prefill-shaped ones, both close to the fp16 reference.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import rdna_dense_int8


def _quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w.abs().amax(dim=1).float().clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(w.float() / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.float16).contiguous()


@pytest.mark.parametrize("bias", [False, True])
def test_linear_released_blocks_match_full_dequant(bias):
    torch.manual_seed(0)
    n, k = 40, 64
    w = torch.randn(n, k, dtype=torch.float32) * 0.05
    q, s = _quant(w)
    b = torch.randn(n, dtype=torch.float16) if bias else None
    x = torch.randn(3, k, dtype=torch.float16)
    ref = F.linear(x, (q.float() * s.float()[:, None]).half(), b)
    # block_rows < n exercises the blocked path.
    out = rdna_dense_int8.linear_released(x, q, s, b, block_rows=16)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_weight_for_gemm_keeps_unreleased_weight():
    w = torch.randn(8, 16)
    q, s = _quant(w)
    assert rdna_dense_int8.weight_for_gemm(w, q, s) is w
    assert rdna_dense_int8.weight_for_gemm(w, None, None) is w


@pytest.mark.parametrize("m", [1, 4, 8, 64])
def test_rdna_dense_gemm_decode_and_prefill(m):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("ROCm only")
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("gfx10x only")
    import vllm._custom_ops  # noqa: F401  (loads _rocm_C)

    if not hasattr(torch.ops._rocm_C, "gemv_i8_rdna2"):
        pytest.skip("gemv_i8_rdna2 not built")
    from vllm.model_executor.layers import rdna_ops  # noqa: F401

    torch.manual_seed(0)
    n, k = 1536, 2560
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    q, s = _quant(w)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    out = torch.ops.vllm.rdna_dense_gemm(x, w, q, s, None)
    ref = F.linear(x, w)
    err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err.item() < 2e-2
