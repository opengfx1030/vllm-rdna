# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused shared-expert kernels on gfx1030.

``rdna_shared_expert`` runs a gate_up kernel (which also yields the expert gate)
and ``se_down_gated_k`` for decode and MTP verify batches, M <= 32 with fp16
weights. Up to ``VLLM_RDNA_SE_TILE_MIN - 1`` rows the gate_up is
``se_gate_up_silu_k`` (csrc/rocm/rdna_fused_glue.cu; rows above 8 run guarded
instances, MT = 16 / 24 / 32, that must produce the same bits as the exact-MT
instances); from there up it is the split-K tile GEMM with a silu*mul epilogue
(``rdna_se_gate_up_tile``, csrc/rocm/hc_prefill_rdna2.cu).
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


@pytest.mark.parametrize("tile_min", [0, 9])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_rows_are_bitwise_independent_of_batch(geom, tile_min, monkeypatch):
    """Row m must not depend on M, on which MT instance ran or on the tile batch.

    The tile threshold splits the M range into two kernel families; the rows of
    each family are bitwise equal across batch sizes (tile_min 0: one family).
    """
    from vllm.model_executor.layers import rdna_ops

    monkeypatch.setattr(rdna_ops, "_SE_TILE_MIN", tile_min)
    i, k, h = geom
    w1, w2, wg = _weights(i, k, h, seed=1)
    x = torch.randn(32, k, device="cuda").half()
    families = [
        [m for m in M_VALUES if not tile_min or m < tile_min],
        [m for m in M_VALUES if tile_min and m >= tile_min],
    ]
    for family in filter(None, families):
        full = _se(x[: max(family)].contiguous(), w1, w2, wg)
        for m in family:
            assert torch.equal(_se(x[:m].contiguous(), w1, w2, wg), full[:m]), m


@pytest.mark.parametrize("geom", [(160, 2560), (1152, 2560), (96, 1024), (16, 64)])
@pytest.mark.parametrize("m", [1, 8, 9, 17, 24, 32])
def test_tile_gate_up_matches_reference(geom, m):
    """The activation silu(g) * u (fp16) and the fp32 expert gate, all M."""
    i, k = geom
    w1, _, wg = _weights(i, k, 16, seed=5)
    x = torch.randn(m, k, device="cuda").half()
    act, gate = torch.ops._rocm_C.rdna_se_gate_up_tile(x, w1, wg)
    assert act.shape == (m, i) and act.dtype == torch.float16
    assert gate.shape == (m,) and gate.dtype == torch.float32
    gu = x.float() @ w1.float().t()
    torch.testing.assert_close(
        act.float(), F.silu(gu[:, :i]) * gu[:, i:], atol=2e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        gate, torch.sigmoid(x.float() @ wg.float()), atol=1e-4, rtol=1e-4
    )


def test_tile_gate_up_is_deterministic():
    w1, _, wg = _weights(160, 2560, 16, seed=6)
    x = torch.randn(24, 2560, device="cuda").half()
    a0, g0 = torch.ops._rocm_C.rdna_se_gate_up_tile(x, w1, wg)
    for _ in range(5):
        a1, g1 = torch.ops._rocm_C.rdna_se_gate_up_tile(x, w1, wg)
        assert torch.equal(a0, a1) and torch.equal(g0, g1)


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

    from vllm.model_executor.layers import rdna_ops

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
    tile = 0 < rdna_ops._SE_TILE_MIN <= m
    gate_up = "gemm_tn_splitk_k" if tile else "se_gate_up_silu_k"
    down = "gemm_tn_splitk_k" if tile and rdna_ops._SE_DOWN_TILE else "se_down_gated_k"
    assert len(names) == 2, names
    assert gate_up in names[0] and down in names[1], names


@pytest.mark.parametrize("geom", [(160, 2560), (320, 2560), (96, 512), (32, 64)])
@pytest.mark.parametrize("m", [1, 8, 9, 17, 24, 32])
def test_tile_down_matches_reference(geom, m):
    """gate[m] * (act[m] . w[h]) rounded once to fp16, all M."""
    i, h = geom
    g = torch.Generator(device="cuda").manual_seed(8)
    act = torch.randn(m, i, device="cuda", generator=g).half()
    w = (torch.randn(h, i, device="cuda", generator=g) * 0.05).half()
    gate = torch.rand(m, device="cuda", generator=g)
    out = torch.ops._rocm_C.rdna_se_down_tile(act, w, gate)
    assert out.shape == (m, h) and out.dtype == torch.float16
    ref = (act.float() @ w.float().t()) * gate[:, None]
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=5e-3)


def test_tile_down_is_deterministic():
    g = torch.Generator(device="cuda").manual_seed(9)
    act = torch.randn(24, 160, device="cuda", generator=g).half()
    w = (torch.randn(2560, 160, device="cuda", generator=g) * 0.05).half()
    gate = torch.rand(24, device="cuda", generator=g)
    o0 = torch.ops._rocm_C.rdna_se_down_tile(act, w, gate)
    for _ in range(5):
        assert torch.equal(o0, torch.ops._rocm_C.rdna_se_down_tile(act, w, gate))


def test_down_tile_knob_matches_wave_kernel(monkeypatch):
    """VLLM_RDNA_SE_DOWN_TILE on and off agree to fp16 noise."""
    from vllm.model_executor.layers import rdna_ops

    w1, w2, wg = _weights(160, 2560, 2560, seed=10)
    x = torch.randn(18, 2560, device="cuda").half()
    tile = _se(x, w1, w2, wg)
    monkeypatch.setattr(rdna_ops, "_SE_DOWN_TILE", False)
    wave = _se(x, w1, w2, wg)
    torch.testing.assert_close(tile.float(), wave.float(), atol=3e-3, rtol=1e-2)
