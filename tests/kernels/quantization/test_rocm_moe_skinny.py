#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch and (optional) kernel tests for gfx10x MoE skinny GEMV.

``moe_skinny_decode_supported`` is CPU-only. The HIP kernel test needs
gfx1030 and ``moe_skinny_int4_decode``. Sequential moe_wna16 packing only
— not the shuffled RDNA2 fused layout in test_rdna2_moe_w4a16.py.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.rocm_moe_skinny import (
    moe_skinny_decode_supported,
)

_BASE = dict(
    use_int4_w4a16=True,
    hidden_dtype=torch.float16,
    num_tokens=1,
    activation=MoEActivation.SILU,
    expert_map=None,
    apply_router_weight_on_input=False,
    w1_zp=None,
    w2_zp=None,
    w1_scale=object(),
    w2_scale=object(),
    block_shape=[0, 32],
    global_num_experts=-1,
    num_local_experts=8,
)


def test_moe_skinny_decode_supported_happy_path():
    assert moe_skinny_decode_supported(**_BASE) is True


@pytest.mark.parametrize(
    "override",
    [
        {"use_int4_w4a16": False},
        {"hidden_dtype": torch.bfloat16},
        {"num_tokens": 9},
        {"activation": MoEActivation.GELU},
        {"apply_router_weight_on_input": True},
        {"w1_zp": object()},
        {"w2_zp": object()},
        {"w1_scale": None},
        {"block_shape": None},
        {"global_num_experts": 4, "num_local_experts": 8},
    ],
)
def test_moe_skinny_decode_supported_rejects(override):
    kwargs = {**_BASE, **override}
    assert moe_skinny_decode_supported(**kwargs) is False


def test_moe_skinny_decode_supported_accepts_expert_map():
    kwargs = {**_BASE, "expert_map": torch.zeros(1, dtype=torch.int32)}
    assert moe_skinny_decode_supported(**kwargs) is True


@pytest.mark.parametrize("m", [1, 3, 6, 9, 12, 16])
def test_moe_skinny_opt_in_covers_concurrent_mtp_rows(monkeypatch, m):
    monkeypatch.setenv("VLLM_ROCM_MOE_SKINNY_MAX_M", "16")
    assert moe_skinny_decode_supported(**{**_BASE, "num_tokens": m})


@pytest.mark.parametrize("limit", [8, 16, 100])
@pytest.mark.parametrize("m", [0, 8, 9, 16, 17])
def test_moe_skinny_limit_preserves_native_bounds(monkeypatch, limit, m):
    monkeypatch.setenv("VLLM_ROCM_MOE_SKINNY_MAX_M", str(limit))
    expected = 1 <= m <= min(limit, 16)
    assert moe_skinny_decode_supported(**{**_BASE, "num_tokens": m}) == expected


def _pack_sequential_int4(w: torch.Tensor) -> torch.Tensor:
    """Pack K-sequential nibbles into int32: bits 4*j hold k=i*8+j."""
    rows, k = w.shape
    assert k % 8 == 0
    packed = torch.zeros(rows, k // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        packed |= (w[:, j::8] & 0xF) << (4 * j)
    return packed


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="HIP kernel test needs a GPU",
)
@pytest.mark.parametrize("scale_multiplier", [0.1, 1.0])
@pytest.mark.parametrize("m", [1, 3, 6, 9, 12, 16])
@pytest.mark.parametrize("mapped", [False, True])
def test_moe_skinny_int4_decode_matches_dequant_ref(scale_multiplier, m, mapped):
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx10x

    if not current_platform.is_rocm() or not on_gfx10x():
        pytest.skip("Requires gfx10x")
    if not (
        hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_skinny_int4_decode")
    ):
        pytest.skip("moe_skinny_int4_decode op not built")

    from vllm import _custom_ops as ops

    device = "cuda"
    e, k, n, topk, group = 4, 128, 64, 2, 32
    torch.manual_seed(0)

    # Symmetric uint4b8 codes in 0..15.
    w13_nibble = torch.randint(0, 16, (e, 2 * n, k), dtype=torch.int32, device=device)
    w2_nibble = torch.randint(0, 16, (e, k, n), dtype=torch.int32, device=device)
    w13 = _pack_sequential_int4(w13_nibble.reshape(e * 2 * n, k)).view(e, 2 * n, k // 8)
    w2 = _pack_sequential_int4(w2_nibble.reshape(e * k, n)).view(e, k, n // 8)
    s13 = (
        torch.randn(e, 2 * n, k // group, dtype=torch.float16, device=device)
        * scale_multiplier
    )
    s2 = (
        torch.randn(e, k, n // group, dtype=torch.float16, device=device)
        * scale_multiplier
    )

    x = torch.randn(m, k, dtype=torch.float16, device=device)
    topk_ids = torch.randint(0, e, (m, topk), dtype=torch.int32, device=device)
    expert_map = None
    if mapped:
        expert_map = torch.tensor([2, -1, 0, -1], dtype=torch.int32, device=device)
        # Include graph-padded rows which have no local expert contribution.
        topk_ids[-1].fill_(1)
    topk_w = torch.softmax(torch.randn(m, topk, device=device), dim=-1).to(
        torch.float32
    )

    act = torch.empty(m, topk, n, dtype=torch.float16, device=device)
    out = torch.empty(m, k, dtype=torch.float16, device=device)

    def launch():
        ops.moe_skinny_int4_decode(
            x, w13, s13, w2, s2, topk_w, topk_ids, act, out, group, expert_map
        )

    launch()

    def dequant(nibbles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return (nibbles.float() - 8.0) * scales.float().repeat_interleave(group, dim=-1)

    w13_fp = dequant(
        w13_nibble.reshape(e * 2 * n, k), s13.reshape(e * 2 * n, k // group)
    )
    w13_fp = w13_fp.view(e, 2 * n, k)
    w2_fp = dequant(w2_nibble.reshape(e * k, n), s2.reshape(e * k, n // group))
    w2_fp = w2_fp.view(e, k, n)

    ref = torch.zeros(m, k, dtype=torch.float32, device=device)
    xf = x.float()
    for mi in range(m):
        for s in range(topk):
            expert = int(topk_ids[mi, s])
            if expert_map is not None:
                expert = int(expert_map[expert])
            if expert < 0:
                continue
            gate = xf[mi] @ w13_fp[expert, :n].T
            up = xf[mi] @ w13_fp[expert, n:].T
            # The public workspace/output contract is FP16. Keep FP32 matrix
            # accumulation but round the inter-projection activation accordingly.
            hidden = (_silu(gate) * up).half().float()
            ref[mi] += topk_w[mi, s] * (w2_fp[expert] @ hidden)

    if scale_multiplier < 1:
        assert out.isfinite().all()
    # Unit-scale synthetic weights can exceed FP16's output range; the larger
    # case checks that overflow agrees, while the smaller case must stay finite.
    torch.testing.assert_close(out.float(), ref.half().float(), atol=2e-2, rtol=2e-2)

    # Verify capture uses the changing input and writes all output rows, including
    # nonlocal/padded rows. Compare identical arithmetic, without new tolerances.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    for factor in (0.5, -1.0, 0.0):
        x.mul_(factor)
        launch()
        eager = out.clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, eager, atol=0, rtol=0)
    graph.reset()
