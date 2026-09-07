# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the RDNA2 fused MoE AWQ kernel.
"""

import pytest
import torch
from vllm._custom_ops import moe_awq_gemm_rdna2


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("block_size_m", [1, 2, 4, 8])
@pytest.mark.parametrize("size_k", [1024, 2048])
def test_moe_awq_rdna2_runs(top_k, block_size_m, size_k):
    """Verify the kernel runs, produces finite output, and is non-trivial."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    size_n = 1024
    num_experts = 4
    groups = size_k // 128
    num_tokens = 16
    M = num_tokens * top_k

    a = torch.empty((M, size_k), dtype=torch.float16, device="cuda")
    a.uniform_(-0.01, 0.01)

    b_q_weight = torch.randint(
        0,
        2**31 - 1,
        (num_experts, size_k, size_n // 8),
        dtype=torch.int32,
        device="cuda",
    )
    b_qzeros = torch.randint(
        0,
        2**31 - 1,
        (num_experts, groups, size_n // 8),
        dtype=torch.int32,
        device="cuda",
    )
    b_scales = torch.empty(
        (num_experts, groups, size_n), dtype=torch.float16, device="cuda"
    )
    b_scales.uniform_(0.01, 0.1)

    topk_weights = torch.empty(M, dtype=torch.float32, device="cuda")
    topk_weights.uniform_(0.1, 1.0)

    num_blocks = (M + block_size_m - 1) // block_size_m
    sorted_token_ids = torch.arange(M, dtype=torch.int32, device="cuda")
    expert_ids = torch.randint(
        0, num_experts, (num_blocks,), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_padded = torch.tensor([M], dtype=torch.int32, device="cuda")

    c = torch.zeros((M, size_n), dtype=torch.float16, device="cuda")

    moe_awq_gemm_rdna2(
        a,
        c,
        b_q_weight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        True,
        0,
    )

    assert not torch.isnan(c).any(), "Output contains NaN"
    assert not torch.isinf(c).any(), "Output contains Inf"
    assert c.abs().max().item() > 0, "Output is all zeros"


@pytest.mark.parametrize("block_size_m", [1, 8])
def test_moe_awq_rdna2_output_topk(block_size_m):
    """Test output_topk > 0 reduction path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    top_k = 2
    size_n = 1024
    size_k = 1024
    num_experts = 4
    groups = size_k // 128
    num_tokens = 8
    M = num_tokens * top_k

    a = torch.empty((M, size_k), dtype=torch.float16, device="cuda")
    a.uniform_(-0.01, 0.01)

    b_q_weight = torch.randint(
        0,
        2**31 - 1,
        (num_experts, size_k, size_n // 8),
        dtype=torch.int32,
        device="cuda",
    )
    b_qzeros = torch.randint(
        0,
        2**31 - 1,
        (num_experts, groups, size_n // 8),
        dtype=torch.int32,
        device="cuda",
    )
    b_scales = torch.empty(
        (num_experts, groups, size_n), dtype=torch.float16, device="cuda"
    )
    b_scales.uniform_(0.01, 0.1)

    topk_weights = torch.empty(M, dtype=torch.float32, device="cuda")
    topk_weights.uniform_(0.1, 1.0)

    num_blocks = (M + block_size_m - 1) // block_size_m
    sorted_token_ids = torch.arange(M, dtype=torch.int32, device="cuda")
    expert_ids = torch.randint(
        0, num_experts, (num_blocks,), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_padded = torch.tensor([M], dtype=torch.int32, device="cuda")

    c = torch.zeros((num_tokens, size_n), dtype=torch.float16, device="cuda")

    moe_awq_gemm_rdna2(
        a,
        c,
        b_q_weight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        True,
        top_k,
    )

    assert not torch.isnan(c).any(), "Output contains NaN"
    assert not torch.isinf(c).any(), "Output contains Inf"
    assert c.abs().max().item() > 0, "Output is all zeros"


def test_moe_awq_rdna2_no_topk_weight():
    """Test without topk_weights multiplication."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    top_k = 1
    size_n = 1024
    size_k = 1024
    num_experts = 4
    groups = size_k // 128
    num_tokens = 16
    M = num_tokens * top_k

    a = torch.empty((M, size_k), dtype=torch.float16, device="cuda")
    a.uniform_(-0.01, 0.01)

    b_q_weight = torch.randint(
        0,
        2**31 - 1,
        (num_experts, size_k, size_n // 8),
        dtype=torch.int32,
        device="cuda",
    )
    b_qzeros = torch.randint(
        0,
        2**31 - 1,
        (num_experts, groups, size_n // 8),
        dtype=torch.int32,
        device="cuda",
    )
    b_scales = torch.empty(
        (num_experts, groups, size_n), dtype=torch.float16, device="cuda"
    )
    b_scales.uniform_(0.01, 0.1)

    topk_weights = torch.empty((0,), dtype=torch.float32, device="cuda")

    num_blocks = M
    sorted_token_ids = torch.arange(M, dtype=torch.int32, device="cuda")
    expert_ids = torch.randint(
        0, num_experts, (num_blocks,), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_padded = torch.tensor([M], dtype=torch.int32, device="cuda")

    c = torch.zeros((M, size_n), dtype=torch.float16, device="cuda")

    moe_awq_gemm_rdna2(
        a,
        c,
        b_q_weight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        1,
        False,
        0,
    )

    assert not torch.isnan(c).any(), "Output contains NaN"
    assert not torch.isinf(c).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
