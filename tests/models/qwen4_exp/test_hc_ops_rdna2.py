# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the gfx1030 HIP HyperConnection kernels.

Every HIP op is compared against the Triton kernel it replaces and, where the
result is not a plain elementwise expression, against an fp32 reference. The
sigmoid uses the hardware ``__expf``, so the comparison tolerance is one fp16
ulp rather than bit equality.
"""

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops import hc as hc_triton
from vllm.models.qwen4_exp.amd.ops import hc_rdna2
from vllm.platforms.rocm import on_gfx10x

pytestmark = pytest.mark.skipif(
    not on_gfx10x(), reason="RDNA2 HIP HC kernels require gfx10x"
)

HC = 4
HIDDEN = 2560
HYPER = HC * HIDDEN
EPS = 1e-6
TOKENS = [1, 2, 7, 48, 512, 2048]
RTOL, ATOL = 2e-3, 2e-3


def _inputs(num_tokens: int, hyper: int = HYPER, hidden: int = HIDDEN, hc: int = HC):
    torch.manual_seed(0)
    res = torch.randn(num_tokens, hyper, dtype=torch.float16, device="cuda")
    blk = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda")
    inj = torch.randn(num_tokens, hc, dtype=torch.float16, device="cuda")
    return res, blk, inj


def _gemma_norm(grouped: torch.Tensor, w: torch.Tensor, shared: bool) -> torch.Tensor:
    """Reference ``x * rsqrt(mean(x^2)) * (1 + w)`` for a [N, HC, HIDDEN] input."""
    rrms = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + EPS)
    wv = w.float() if shared else w.float().unflatten(-1, (HC, HIDDEN))
    return (grouped * rrms * (1.0 + wv)).flatten(-2)

@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("shared_weight", [False, True])
def test_hc_combine_norm(num_tokens: int, shared_weight: bool) -> None:
    res, blk, inj = _inputs(num_tokens)
    w = torch.randn(
        HIDDEN if shared_weight else HYPER, dtype=torch.float16, device="cuda"
    )

    t_out, t_y = hc_triton._hc_combine_norm(res, blk, inj, w, EPS, HC)
    h_out, h_y = hc_rdna2.hc_combine_norm(res, blk, inj, w, EPS, HC)

    scale = 2.0 * torch.sigmoid(inj.float() / HC)
    ref = res.float().unflatten(-1, (HC, HIDDEN)) + blk.float().unsqueeze(
        -2
    ) * scale.unsqueeze(-1)
    ref = ref.flatten(-2).to(torch.float16)
    torch.testing.assert_close(h_out, ref, rtol=RTOL, atol=ATOL)

    ref_y = _gemma_norm(ref.float().unflatten(-1, (HC, HIDDEN)), w, shared_weight)
    torch.testing.assert_close(h_y, ref_y.to(torch.float16), rtol=RTOL, atol=ATOL)

    torch.testing.assert_close(h_out, t_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(h_y, t_y, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("shared_weight", [False, True])
def test_grouped_gemma_rmsnorm(num_tokens: int, shared_weight: bool) -> None:
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HYPER, dtype=torch.float16, device="cuda")
    w = torch.randn(
        HIDDEN if shared_weight else HYPER, dtype=torch.float16, device="cuda"
    )

    t = hc_triton._grouped_gemma_rmsnorm(x, w, EPS, HC)
    h = hc_rdna2.grouped_gemma_rmsnorm(x, w, EPS, HC)

    grouped = x.float().unflatten(-1, (HC, HIDDEN))
    ref = _gemma_norm(grouped, w, shared_weight)
    torch.testing.assert_close(h, ref.to(torch.float16), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(h, t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_hc_silu(num_tokens: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HYPER, dtype=torch.float16, device="cuda")
    t = hc_triton._hc_silu(x, HC)
    h = hc_rdna2.hc_silu(x, HC)
    ref = x.float() / HC
    ref = (ref * torch.sigmoid(ref)).to(torch.float16)
    torch.testing.assert_close(h, ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(h, t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_hc_gate_mix(num_tokens: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HYPER, dtype=torch.float16, device="cuda")
    gate = torch.randn(num_tokens, HYPER, dtype=torch.float16, device="cuda")
    t = hc_triton._hc_gate_mix(x, gate, HC)
    h = hc_rdna2.hc_gate_mix(x, gate, HC)
    ref = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN)))
        * x.float().unflatten(-1, (HC, HIDDEN))
    ).mean(-2)
    torch.testing.assert_close(h, ref.to(torch.float16), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(h, t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_hc_combine(num_tokens: int) -> None:
    res, blk, inj = _inputs(num_tokens)
    t = hc_triton._hc_combine(res, blk, inj, HC)
    h = hc_rdna2.hc_combine(res, blk, inj, HC)
    iw = 2.0 * torch.sigmoid(inj.float() / HC)
    ref = res.float().unflatten(-1, (HC, HIDDEN)) + blk.float().unsqueeze(
        -2
    ) * iw.unsqueeze(-1)
    torch.testing.assert_close(h, ref.flatten(-2).to(torch.float16), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(h, t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_tokens", [1, 2, 5, 17, 48])
def test_hc_ops_unusual_hc_count(num_tokens: int) -> None:
    """Exercise a width that is neither HC*2560 nor a multiple of the vector."""
    hc, hidden = 2, 384
    hyper = hc * hidden
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hyper, dtype=torch.float16, device="cuda")
    gate = torch.randn(num_tokens, hyper, dtype=torch.float16, device="cuda")
    res, blk, inj = _inputs(num_tokens, hyper, hidden, hc)
    w = torch.randn(hyper, dtype=torch.float16, device="cuda")

    for h, t in (
        (hc_rdna2.hc_gate_mix(x, gate, hc), hc_triton._hc_gate_mix(x, gate, hc)),
        (hc_rdna2.hc_silu(x, hc), hc_triton._hc_silu(x, hc)),
        (
            hc_rdna2.grouped_gemma_rmsnorm(x, w, EPS, hc),
            hc_triton._grouped_gemma_rmsnorm(x, w, EPS, hc),
        ),
        (hc_rdna2.hc_combine(res, blk, inj, hc), hc_triton._hc_combine(res, blk, inj, hc)),
    ):
        torch.testing.assert_close(h, t, rtol=RTOL, atol=ATOL)

    h_out, _ = hc_rdna2.hc_combine_norm(res, blk, inj, w, EPS, hc)
    t_out, _ = hc_triton._hc_combine_norm(res, blk, inj, w, EPS, hc)
    torch.testing.assert_close(h_out, t_out, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("hyper", [1023, 1024, 1025, 1151])
def test_hc_ops_unaligned_fallback(hyper: int) -> None:
    """Odd inner widths must still work through the scalar instantiation."""
    num_tokens, hc = 3, 1
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hyper, dtype=torch.float16, device="cuda")
    gate = torch.randn(num_tokens, hyper, dtype=torch.float16, device="cuda")
    w = torch.randn(hyper, dtype=torch.float16, device="cuda")

    for h, t in (
        (hc_rdna2.hc_gate_mix(x, gate, hc), hc_triton._hc_gate_mix(x, gate, hc)),
        (hc_rdna2.hc_silu(x, hc), hc_triton._hc_silu(x, hc)),
        (
            hc_rdna2.grouped_gemma_rmsnorm(x, w, EPS, hc),
            hc_triton._grouped_gemma_rmsnorm(x, w, EPS, hc),
        ),
    ):
        torch.testing.assert_close(h, t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("num_tokens", [2, 7])
def test_compiled_strided_hc_graph_replay(num_tokens: int) -> None:
    torch.manual_seed(0)
    sources = [
        torch.randn(num_tokens, width + HC, dtype=torch.float16, device="cuda")
        for width in (HYPER, HIDDEN, HC, HYPER)
    ]
    residual, block, injection, gate = [
        source[:, :width]
        for source, width in zip(sources, (HYPER, HIDDEN, HC, HYPER))
    ]
    weight = torch.randn(HIDDEN, dtype=torch.float16, device="cuda")

    def forward(residual, block, injection, gate, weight):
        return (
            hc_triton.grouped_gemma_rmsnorm(residual, weight, EPS, HC),
            hc_triton.hc_silu(residual, HC),
            hc_triton.hc_gate_mix(residual, gate, HC),
            hc_triton.hc_combine(residual, block, injection, HC),
            *hc_triton.hc_combine_norm(residual, block, injection, weight, EPS, HC),
        )

    # Dynamo must preserve the functional op boundary even for split views.
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    inputs = (residual, block, injection, gate, weight)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compiled(*inputs)
        compiled(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            actual = compiled(*inputs)
        for _ in range(2):
            for source in sources:
                source.add_(0.25)
            graph.replay()
            packed = [tensor.contiguous() for tensor in inputs]
            res, blk, inj, gates, norm_weight = packed
            expected = (
                hc_triton._grouped_gemma_rmsnorm(res, norm_weight, EPS, HC),
                hc_triton._hc_silu(res, HC),
                hc_triton._hc_gate_mix(res, gates, HC),
                hc_triton._hc_combine(res, blk, inj, HC),
                *hc_triton._hc_combine_norm(res, blk, inj, norm_weight, EPS, HC),
            )
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, rtol=RTOL, atol=ATOL)
    finally:
        graph.reset()
