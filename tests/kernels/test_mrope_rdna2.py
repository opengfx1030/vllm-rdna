# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the gfx1030 HIP M-RoPE kernel.

The HIP op is compared against ``triton_mrope``, the kernel it replaces, over
decode- and prefill-shaped token counts. The comparison guards the per-token
work split: the kernel strides the rotary index across the block, so every
rotary index must be applied exactly once and in the same order no matter how
many tokens share the launch.

cos/sin are passed as the raw ``[3, num_tokens, rotary_dim // 2]`` T/H/W planes
from the cache, exactly as ``MRotaryEmbedding.forward_cuda`` / ``forward_hip``
do: both kernels select the plane per channel from the masks, so the
``apply_interleaved_rope`` / section-reorder that ``forward_native`` needs must
NOT be applied here.
"""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.mrope import (
    mrope_forward_rdna2,
    triton_mrope,
)
from vllm.platforms.rocm import on_gfx10x

pytestmark = pytest.mark.skipif(
    not on_gfx10x(), reason="RDNA2 HIP mrope requires gfx10x"
)

# (head_size, rotary_dim, num_q_heads, num_kv_heads, mrope_section)
SHAPES = [
    (128, 128, 4, 1, [16, 24, 24]),
    (256, 256, 24, 2, [32, 48, 48]),
]
TOKENS = [1, 2, 3, 7, 48]
RTOL, ATOL = 2e-3, 2e-3


def _inputs(num_tokens: int, head_size: int, n_qh: int, n_kh: int, rd: int):
    """Raw [3, T, rd/2] planes plus q/k, all in the [-1, 1] range cos/sin live in."""
    torch.manual_seed(num_tokens)
    planes = torch.rand(6, num_tokens, rd // 2, device="cuda") * 2.0 - 1.0
    cos, sin = planes[:3].to(torch.float16), planes[3:].to(torch.float16)
    q = torch.randn(num_tokens, n_qh * head_size, dtype=torch.float16, device="cuda")
    k = torch.randn(num_tokens, n_kh * head_size, dtype=torch.float16, device="cuda")
    return q, k, cos, sin


@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("head_size,rotary_dim,n_qh,n_kh,sections", SHAPES)
def test_mrope_rdna2_matches_triton(
    num_tokens: int,
    interleaved: bool,
    neox: bool,
    head_size: int,
    rotary_dim: int,
    n_qh: int,
    n_kh: int,
    sections: list[int],
) -> None:
    q, k, cos, sin = _inputs(num_tokens, head_size, n_qh, n_kh, rotary_dim)

    q_ref, k_ref = q.clone(), k.clone()
    triton_mrope(
        q_ref, k_ref, cos, sin, sections, head_size, rotary_dim, interleaved, neox
    )

    q_hip, k_hip = q.clone(), k.clone()
    assert mrope_forward_rdna2(
        q_hip, k_hip, cos, sin, head_size, rotary_dim, sections, interleaved, neox
    )

    torch.testing.assert_close(q_hip, q_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(k_hip, k_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("neox", [True, False])
def test_mrope_rdna2_is_per_token(interleaved: bool, neox: bool) -> None:
    """A token's result must not depend on the other tokens in the launch."""
    head_size, rotary_dim, n_qh, n_kh, sections = SHAPES[1]
    q, k, cos, sin = _inputs(3, head_size, n_qh, n_kh, rotary_dim)

    q_all, k_all = q.clone(), k.clone()
    assert mrope_forward_rdna2(
        q_all, k_all, cos, sin, head_size, rotary_dim, sections, interleaved, neox
    )

    for t in range(3):
        q_one, k_one = q[t : t + 1].clone(), k[t : t + 1].clone()
        assert mrope_forward_rdna2(
            q_one,
            k_one,
            cos[:, t : t + 1].contiguous(),
            sin[:, t : t + 1].contiguous(),
            head_size,
            rotary_dim,
            sections,
            interleaved,
            neox,
        )
        torch.testing.assert_close(q_all[t : t + 1], q_one, rtol=0, atol=0)
        torch.testing.assert_close(k_all[t : t + 1], k_one, rtol=0, atol=0)
