# SPDX-License-Identifier: Apache-2.0
"""Parity: HIP reshape_and_cache_flash_rdna2 vs Triton, including padded pages."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_reshape_and_cache_flash_rdna2_padded_hybrid():
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "reshape_and_cache_flash_rdna2"
    ):
        pytest.skip("reshape_and_cache_flash_rdna2 not registered")

    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_tokens, H, D, x = 4, 1, 256, 8
    block_size, num_blocks = 784, 8
    # Pad block stride past packed numel (Qwen3.5 GDN page alignment).
    packed_k = H * (D // x) * block_size * x
    packed_v = H * D * block_size
    pad = 64
    key_storage = torch.zeros(
        num_blocks, packed_k + pad, device=device, dtype=torch.float16
    )
    value_storage = torch.zeros(
        num_blocks, packed_v + pad, device=device, dtype=torch.float16
    )
    key_cache = key_storage[:, :packed_k].view(
        num_blocks, H, D // x, block_size, x
    )
    value_cache = value_storage[:, :packed_v].view(num_blocks, H, D, block_size)
    assert key_cache.stride(0) > packed_k - 1

    key = torch.randn(num_tokens, H, D, device=device, dtype=torch.float16)
    value = torch.randn(num_tokens, H, D, device=device, dtype=torch.float16)
    slot_mapping = torch.tensor([10, 11, 800, -1], device=device, dtype=torch.int32)
    k_scale = torch.ones(1, device=device, dtype=torch.float32)
    v_scale = torch.ones(1, device=device, dtype=torch.float32)

    key_cache_ref = key_cache.clone()
    value_cache_ref = value_cache.clone()
    triton_reshape_and_cache_flash(
        key, value, key_cache_ref, value_cache_ref, slot_mapping, "auto", k_scale, v_scale
    )
    torch.ops._rocm_C.reshape_and_cache_flash_rdna2(
        key, value, key_cache, value_cache, slot_mapping
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(key_cache, key_cache_ref, atol=0, rtol=0)
    torch.testing.assert_close(value_cache, value_cache_ref, atol=0, rtol=0)
