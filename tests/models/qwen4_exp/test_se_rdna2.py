# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused shared-expert kernels on gfx1030 (csrc/rocm/rdna_fused_glue.cu).

``rdna_shared_expert`` runs ``se_gate_up_silu_k`` (which also yields the
expert gate) and ``se_down_gated_k`` for decode and MTP verify batches, M <= 32
with fp16 weights. Rows above 8 run guarded instances (MT = 16 / 24 / 32) that
must produce the same bits as the exact-MT instances.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


def _gfx10x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


pytestmark = pytest.mark.skipif(not _gfx10x(), reason="gfx1030 HIP kernels")

# (I per rank, K = hidden in, H = hidden out); 160 is Flash-Next at TP4.
GEOMETRIES = [(160, 2560, 2560), (1152, 2560, 2560), (96, 1024, 512)]
M_VALUES = [1, 3, 6, 8, 9, 12, 16, 17, 18, 24, 25, 31, 32]


def _weights(i, k, h, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = (torch.randn(2 * i, k, device="cuda", generator=g) * 0.02).half()
    w2 = (torch.randn(h, i, device="cuda", generator=g) * 0.05).half()
    wg = (torch.randn(k, device="cuda", generator=g) * 0.05).half()
    return w1, w2, wg


def _se(x, w1, w2, wg):
    from vllm.model_executor.layers import rdna_ops  # noqa: F401

    return torch.ops.vllm.rdna_shared_expert(x, w1, None, None, w2, None, None, wg)


def _reference(x, w1, w2, wg):
    """fp32 math from the fp16 operands; ``act`` is rounded to fp16 like the kernel."""
    gu = x.float() @ w1.float().t()
    i = gu.shape[1] // 2
    act = (F.silu(gu[:, :i]) * gu[:, i:]).half()
    out = act.float() @ w2.float().t()
    gate = torch.sigmoid(x.float() @ wg.float())
    return (out * gate[:, None]).half()


@pytest.mark.parametrize("geom", GEOMETRIES)
@pytest.mark.parametrize("m", M_VALUES)
def test_matches_reference(geom, m):
    i, k, h = geom
    w1, w2, wg = _weights(i, k, h)
    x = torch.randn(m, k, device="cuda").half()
    out = _se(x, w1, w2, wg)
    assert out.shape == (m, h) and out.dtype == torch.float16
    torch.testing.assert_close(
        out.float(), _reference(x, w1, w2, wg).float(), atol=2e-3, rtol=5e-3
    )


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_rows_are_bitwise_independent_of_batch(geom):
    """Row m must not depend on M or on which MT instance ran."""
    i, k, h = geom
    w1, w2, wg = _weights(i, k, h, seed=1)
    x = torch.randn(32, k, device="cuda").half()
    full = _se(x, w1, w2, wg)
    for m in M_VALUES:
        assert torch.equal(_se(x[:m].contiguous(), w1, w2, wg), full[:m]), m


def test_matches_torch_fallback(monkeypatch):
    """The fused op and its own torch branch agree to fp16 noise."""
    from vllm.model_executor.layers import rdna_ops

    w1, w2, wg = _weights(160, 2560, 2560, seed=2)
    x = torch.randn(18, 2560, device="cuda").half()
    fused = _se(x, w1, w2, wg)
    monkeypatch.setattr(rdna_ops, "_SE_DECODE_MAX", 0)
    ref = _se(x, w1, w2, wg)
    torch.testing.assert_close(fused.float(), ref.float(), atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("m", [6, 18, 24])
def test_cuda_graph_replay(m):
    w1, w2, wg = _weights(160, 2560, 2560, seed=3)
    x = torch.randn(m, 2560, device="cuda").half()
    _se(x, w1, w2, wg)  # warm-up outside the capture
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _se(x, w1, w2, wg)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.testing.assert_close(
            out.float(), _reference(x, w1, w2, wg).float(), atol=2e-3, rtol=5e-3
        )


@pytest.mark.parametrize("m", [6, 18, 32])
def test_is_two_kernels(m):
    from torch.profiler import ProfilerActivity, profile

    w1, w2, wg = _weights(160, 2560, 2560, seed=4)
    x = torch.randn(m, 2560, device="cuda").half()
    _se(x, w1, w2, wg)
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _se(x, w1, w2, wg)
        torch.accelerator.synchronize()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not names:
        pytest.skip("profiler returned no kernel events")
    assert len(names) == 2, names
    assert "se_gate_up_silu_k" in names[0] and "se_down_gated_k" in names[1], names
