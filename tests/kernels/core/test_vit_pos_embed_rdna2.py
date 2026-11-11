# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity and dispatch tests for the RDNA2 HIP bilinear position-embedding op.

Compares ``rdna2_pos_embed_interpolate`` (csrc/rocm/vit_pos_embed_rdna2.cu)
against the Triton kernel and the pure-PyTorch implementation on the real
Qwen3-VL / Qwen4Exp serving geometry plus edge cases, and asserts that the
HIP path -- not Triton -- is the one taken on gfx1030.
"""

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not on_rdna2(), reason="RDNA2 (gfx10x) HIP path only"
)

if HAS_TRITON:
    import vllm.model_executor.models.qwen3_vl as qwen3_vl
    from vllm.model_executor.models.qwen3_vl import (
        pos_embed_interpolate_native,
        rdna2_pos_embed_interpolate,
    )

# Real serving geometry: Qwen4Exp vision tower in
# models/qwen38-flash-next-uncensored-w4a16 (num_position_embeddings 2304,
# hidden_size 1152, patch 16, spatial_merge_size 2, max_pixels 1638400).
NUM_GRID_PER_SIDE = 48
SPATIAL_MERGE_SIZE = 2
HIDDEN_DIM = 1152

# (t, h, w) in patch units. 64x100 is the max_pixels image (1024x1600 px);
# 46x62 and 2x32x32 are deliberately non-multiples of the spatial-merge
# block sizes and a multi-frame video grid.
GRIDS = [
    (1, 8, 8),  # below the grid, heavy interpolation
    (1, 46, 62),  # odd sizes
    (1, 48, 48),  # exactly the embedding table
    (1, 64, 100),  # max_pixels image
    (1, 80, 80),  # above the table, clamped ceil
    (2, 32, 32),  # video
    (8, 16, 16),  # video, t > 1
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# bf16 is deliberately absent: the Triton _bilinear_pos_embed_kernel aborts the
# AMD compiler on gfx1030 ("LLVM ERROR: Cannot select: intrinsic
# %llvm.amdgcn.fdot2.bf16.bf16"). Pre-existing, independent of this port --
# tests/kernels/core/test_vit_bilinear_pos_embed.py aborts the same way.
TRITON_DTYPES = [torch.float32, torch.float16]

ATOL = {torch.float32: 5e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
RTOL = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}


def _embed(hidden_dim=HIDDEN_DIM, dtype=torch.float16, device="cuda"):
    # Scale to the real pos_embed weight distribution (std ~0.23).
    torch.manual_seed(42)
    return (
        torch.randn(
            NUM_GRID_PER_SIDE * NUM_GRID_PER_SIDE,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("dtype", TRITON_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("t,h,w", GRIDS, ids=lambda v: str(v))
def test_hip_matches_triton(t, h, w, dtype):
    embed = _embed(dtype=dtype)
    ref = qwen3_vl.triton_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    out = rdna2_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    if dtype == torch.float16:
        # The HIP kernel reproduces the Triton expression (fp32 index math,
        # weights cast to fp16, packed half2 MACs), so at fp16 the two are
        # bit-identical -- the served dtype gets exact parity, not a tolerance.
        assert torch.equal(out, ref), (
            f"expected bit-exact fp16 parity, max|d|="
            f"{(out.float() - ref.float()).abs().max().item():.3e}"
        )
    else:
        torch.testing.assert_close(
            out,
            ref,
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
            msg=lambda m: f"{m}\n{t, h, w, dtype}",
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("t,h,w", GRIDS[:5], ids=lambda v: str(v))
def test_hip_matches_native(t, h, w, dtype):
    """Triton's own agreement with native is asserted in
    test_vit_bilinear_pos_embed.py; the HIP op must be at least as close."""
    embed = _embed(dtype=dtype)
    ref = pos_embed_interpolate_native(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    out = rdna2_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    torch.testing.assert_close(out, ref, atol=ATOL[dtype], rtol=RTOL[dtype])


def test_temporal_repeat_is_exact():
    embed = _embed()
    single = rdna2_pos_embed_interpolate(embed, 1, 16, 16, 48, 2, torch.float16)
    multi = rdna2_pos_embed_interpolate(embed, 3, 16, 16, 48, 2, torch.float16)
    torch.testing.assert_close(multi, single.repeat(3, 1), atol=0, rtol=0)


def test_dtype_mismatch_is_rejected():
    embed = _embed(dtype=torch.float16)
    with pytest.raises(Exception, match="share a dtype"):
        torch.ops._rocm_C.bilinear_pos_embed_rdna2(
            embed,
            torch.empty(64, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16),
            8,
            8,
            47 / 7.0,
            47 / 7.0,
            48,
            2,
        )


def test_non_multiple_hidden_dim_scalar_fallback():
    """hidden_dim % 8 != 0 forces the scalar kernel; it must still match."""
    embed = _embed(hidden_dim=1150)
    ref = pos_embed_interpolate_native(
        embed, 1, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, torch.float16
    )
    out = rdna2_pos_embed_interpolate(
        embed, 1, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, torch.float16
    )
    torch.testing.assert_close(out, ref, atol=ATOL[torch.float16], rtol=RTOL[torch.float16])


def test_misaligned_input_scalar_fallback():
    """An 8-byte-offset view is not 16B aligned: the vec8 kernel must not be
    used, and the result must still match."""
    n = 2304
    buf = torch.zeros(n * HIDDEN_DIM + 4, device="cuda", dtype=torch.float16)
    embed = buf[4:].view(n, HIDDEN_DIM)
    embed.copy_(_embed())
    assert embed.data_ptr() % 16 != 0
    ref = pos_embed_interpolate_native(
        embed, 1, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, torch.float16
    )
    out = rdna2_pos_embed_interpolate(
        embed, 1, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, torch.float16
    )
    torch.testing.assert_close(out, ref, atol=ATOL[torch.float16], rtol=RTOL[torch.float16])


def test_unit_grid_uses_zero_scale():
    """h == 1 (m_size 1) takes the `h_scale = 0.0` branch of both wrappers."""
    embed = _embed(hidden_dim=64)
    ref = pos_embed_interpolate_native(embed, 1, 1, 1, 48, 1, torch.float16)
    out = rdna2_pos_embed_interpolate(embed, 1, 1, 1, 48, 1, torch.float16)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


class _PoisonedTriton:
    """Stands in for the triton.JITFunction; any launch raises."""

    def __getitem__(self, grid):
        raise AssertionError(
            f"Triton _bilinear_pos_embed_kernel launched with grid {grid}"
        )


def test_fast_pos_embed_interpolate_takes_hip_path(monkeypatch):
    """Poison the Triton kernel and assert fast_pos_embed_interpolate still
    works -- i.e. that it dispatches to the HIP op, not to Triton."""
    monkeypatch.setattr(
        qwen3_vl, "_bilinear_pos_embed_kernel", _PoisonedTriton(), raising=True
    )
    monkeypatch.setattr(qwen3_vl, "_pos_embed_interpolate_impl", None)

    stub = type("Stub", (), {})()
    stub.pos_embed = type("E", (), {})()
    stub.pos_embed.weight = _embed()
    stub.dtype = torch.float16
    stub.spatial_merge_size = SPATIAL_MERGE_SIZE
    stub.num_grid_per_side = NUM_GRID_PER_SIDE

    out = qwen3_vl.Qwen3_VisionTransformer.fast_pos_embed_interpolate(
        stub, [[1, 16, 16], [1, 32, 48]]
    )
    assert out.shape == (16 * 16 + 32 * 48, HIDDEN_DIM)
    assert qwen3_vl._pos_embed_interpolate_impl is rdna2_pos_embed_interpolate


def test_missing_binding_raises(monkeypatch):
    """A missing _rocm_C op must raise, never fall back to Triton."""
    class _NoRoCM:
        pass

    class _TorchShim:
        ops = _NoRoCM()

    monkeypatch.setattr(qwen3_vl, "torch", _TorchShim())
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        qwen3_vl._resolve_pos_embed_interpolate()
