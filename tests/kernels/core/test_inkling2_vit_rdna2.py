# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests of the Inkling-2 vision HIP kernels (csrc/rocm/inkling2_vit_rdna2.cu).

Every kernel is compared against an fp64 CPU reference written here from the
architecture description (never against another kernel).  fp16 outputs must be
within one fp16 ulp of the exact value and equal to the correctly rounded value for
all but a rounding-tie fraction (fp32 accumulation order vs the fp64 sum), which
also separates an exact GELU from its tanh approximation.  Run on gfx1030 only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.platforms.rdna import on_rdna2

pytestmark = pytest.mark.skipif(not on_rdna2(), reason="RDNA2 (gfx10x) HIP path only")

DEV = "cuda"
ULP = 2.0**-10  # relative spacing of fp16 (one ulp <= ULP * |x|)
TIE_FRACTION = 0.01  # fp16 outputs that may differ from the correctly rounded value
D_HEAD = 64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _check_f16(got: torch.Tensor, want: torch.Tensor, floor: float, what: str) -> None:
    """``got`` (fp16, on the GPU) vs the exact ``want`` (fp64, CPU)."""
    got = got.cpu()
    err = (got.double() - want).abs()
    bound = ULP * want.abs().clamp_min(2.0**-14) + floor
    assert (err <= bound).all(), (
        f"{what}: max error {float(err.max()):.3e} exceeds one ulp, "
        f"worst at {int((err - bound).argmax())}"
    )
    ties = float((got != want.half()).double().mean())
    assert ties <= TIE_FRACTION, f"{what}: {ties:.3%} not correctly rounded"


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _image_tables(grids: list[tuple[int, int]]):
    """(meta, blk_prefix, tokens, patches) for images of ``(rows, cols)`` cells."""
    meta, blk, tok, patch = [], [0], 0, 0
    for rows, cols in grids:
        meta.append([rows, cols, tok, patch])
        length = 4 * rows * cols + 1
        tok += length
        patch += 4 * rows * cols
        blk.append(blk[-1] + (length + 63) // 64)
    return (
        torch.tensor(meta, dtype=torch.int32, device=DEV),
        torch.tensor(blk, dtype=torch.int32, device=DEV),
        tok,
        patch,
    )


def _patch_coordinates(rows: int, cols: int):
    """(y, x) of every patch of an image in kernel order: cells row-major, four
    patches per cell as top-left, top-right, bottom-left, bottom-right."""
    cell = torch.arange(rows * cols)
    sub = torch.arange(4)
    y = ((cell // cols)[:, None] * 2 + (sub // 2)[None]).reshape(-1)
    x = ((cell % cols)[:, None] * 2 + (sub % 2)[None]).reshape(-1)
    return y, x


# ------------------------------------------------------------------ gather

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD_PROCESSOR = (0.26862954, 0.2613026, 0.2757771)  # the processor's spelling
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)  # the encoder's spelling


def _pixels(raw: torch.Tensor, mode: int) -> torch.Tensor:
    """fp16 patch pixels of raw values (0..255 or -1, channels last).  Every operation
    is an IEEE fp32 one (numpy), so the kernel must match bit for bit."""
    f32 = np.float32
    x = raw.float().numpy()
    if mode == 0:  # the transformers processor: 2 v / 255 - 1
        return torch.from_numpy((f32(2) * x - f32(255)) / f32(255)).half()
    # CLIP normalisation rounded to bf16, then mapped back to [-1, 1] and rounded
    mean = np.array(CLIP_MEAN, f32)
    clip = (x / f32(255) - mean) / np.array(CLIP_STD_PROCESSOR, f32)
    z = torch.from_numpy(clip).bfloat16().float().numpy() * np.array(CLIP_STD, f32)
    return torch.from_numpy(f32(2) * (z + mean) - f32(1)).bfloat16().half()


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cells", [1, 5])
def test_gather_patches_is_exact_for_every_raw_value(mode, dtype, cells):
    # every value -1..255 meets every channel (7919 is coprime to 257)
    n = cells * 40 * 40 * 3
    raw = ((torch.arange(n) * 7919) % 257 - 1).view(cells, 40, 40, 3).to(dtype)
    out = torch.full((4 * cells, 1216), 5.0, dtype=torch.float16, device=DEV)
    ops.inkling2_vit_gather_patches(raw.to(DEV), out, mode)

    want = _pixels(raw, mode).view(cells, 2, 20, 2, 20, 3).permute(0, 1, 3, 2, 4, 5)
    want = want.reshape(4 * cells, 1200)
    got = out.cpu()
    assert torch.equal(got[:, :1200], want)
    assert not got[:, 1200:].any()  # the K pad of the patch GEMM is zero


def test_gather_pixel_modes_differ_and_the_mode_is_checked():
    raw = (torch.arange(4800) % 256).view(1, 40, 40, 3).half()
    outs = []
    for mode in (0, 1):
        out = torch.empty((4, 1216), dtype=torch.float16, device=DEV)
        ops.inkling2_vit_gather_patches(raw.to(DEV), out, mode)
        outs.append(out[:, :1200].cpu().float())
    # bf16 rounding of the CLIP value is visible, but the two stay within 2^-7
    diff = (outs[0] - outs[1]).abs()
    assert 0.0 < float(diff.max()) <= 2.0**-7
    with pytest.raises(RuntimeError, match="mode must be"):
        ops.inkling2_vit_gather_patches(raw.to(DEV), out, 2)


# -------------------------------------------------------------------- GEMM

GEMM_SHAPES = [
    (1, 128, 128),
    (5, 192, 512),
    (130, 128, 64),
    (257, 1024, 1216),
    (300, 3072, 1024),
    (129, 2304, 4096),
]


def _gemm_operands(m: int, n: int, k: int, seed: int):
    g = _gen(seed)
    a = (torch.randn(m, k, generator=g) * 0.5).half()
    w = (torch.randn(n, k, generator=g) * 0.05).half()
    bias = torch.randn(n, generator=g) * 0.3
    acc = a.double() @ w.double().T
    return a, w, bias, acc


@pytest.mark.parametrize(("m", "n", "k"), GEMM_SHAPES)
@pytest.mark.parametrize("epilogue", [0, 1, 2, 3, 4])
def test_gemm_epilogues(m, n, k, epilogue):
    a, w, bias, acc = _gemm_operands(m, n, k, seed=m + n + k)
    want = {
        0: acc,
        1: acc + bias.double(),
        2: _gelu(acc + bias.double()),
        3: _gelu(acc),
        4: acc,
    }[epilogue]
    out = torch.empty(
        m, n, dtype=torch.float32 if epilogue == 4 else torch.float16, device=DEV
    )
    ops.inkling2_vit_gemm(
        a.to(DEV),
        w.to(DEV),
        out,
        bias.to(DEV) if epilogue in (1, 2) else None,
        epilogue,
    )
    if epilogue == 4:
        err = float((out.cpu().double() - want).abs().max())
        assert err <= 1e-5 * max(1.0, float(want.abs().max())), err
    else:
        _check_f16(
            out, want, floor=2e-5 * math.sqrt(k / 1024), what=f"epilogue {epilogue}"
        )


@pytest.mark.parametrize(("m", "n", "k"), GEMM_SHAPES)
def test_gemm_residual_accumulates_scaled_branch_in_fp32(m, n, k):
    a, w, bias, acc = _gemm_operands(m, n, k, seed=3 * (m + n + k))
    g = _gen(99)
    res = torch.randn(m, n, generator=g)
    gamma = torch.rand(n, generator=g) + 0.01
    want = res.double() + gamma.double() * (acc + bias.double())
    res_dev = res.to(DEV)
    ops.inkling2_vit_gemm_residual(
        a.to(DEV), w.to(DEV), bias.to(DEV), gamma.to(DEV), res_dev
    )
    err = float((res_dev.cpu().double() - want).abs().max())
    assert err <= 1e-5 * max(1.0, float(want.abs().max())), err


def test_gemm_rejects_unsupported_shapes():
    a = torch.zeros(4, 100, dtype=torch.float16, device=DEV)
    w = torch.zeros(128, 100, dtype=torch.float16, device=DEV)
    out = torch.empty(4, 128, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError, match="32"):
        ops.inkling2_vit_gemm(a, w, out, None, 0)  # K is not a multiple of 32
    a = torch.zeros(4, 128, dtype=torch.float32, device=DEV)
    w = torch.zeros(128, 128, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError):
        ops.inkling2_vit_gemm(a, w, out, None, 0)  # operands must be fp16


# --------------------------------------------------------------- layernorm


@pytest.mark.parametrize(
    ("rows", "dim", "scale", "shift"),
    [
        (7, 128, 3.0, 1.0),
        (33, 1024, 3.0, 1.0),
        (1, 2048, 3.0, 1.0),
        (300, 1024, 3.0, 1.0),
        # variance 1e-6, far below the epsilon: only the epsilon sets the output.
        # Zero mean: with mean 1 the unmutated kernel misses one fp16 ulp (max error
        # 4.8e-4; the fp32 rounding of the mean amplified by 1 / sqrt(eps), inferred)
        (9, 256, 1e-3, 0.0),
    ],
)
def test_layernorm_fp32_rows_to_fp16(rows, dim, scale, shift):
    g = _gen(rows + dim)
    h = torch.randn(rows, dim, generator=g) * scale + shift
    w = torch.randn(dim, generator=g) * 0.2 + 1
    b = torch.randn(dim, generator=g) * 0.1
    out = torch.empty(rows, dim, dtype=torch.float16, device=DEV)
    ops.inkling2_vit_layernorm(h.to(DEV), w.to(DEV), b.to(DEV), out, 1e-5)
    hd = h.double()
    mean = hd.mean(-1, keepdim=True)
    var = ((hd - mean) ** 2).mean(-1, keepdim=True)
    want = (hd - mean) / torch.sqrt(var + 1e-5) * w.double() + b.double()
    _check_f16(out, want, floor=1e-5, what="layernorm")


def test_layernorm_rejects_unsupported_width():
    h = torch.zeros(2, 100, device=DEV)
    out = torch.empty(2, 100, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError):
        ops.inkling2_vit_layernorm(
            h, torch.ones(100, device=DEV), torch.zeros(100, device=DEV), out, 1e-5
        )


# ------------------------------------------------------------------- embed

GRIDS = [(3, 5), (7, 2), (1, 1), (9, 4)]


@pytest.mark.parametrize("grid", [32, 8])
def test_embed_adds_bilinear_position_and_layernorms(grid):
    # grid 8 is smaller than the patch grids of some images (14 and 18 rows, 10
    # columns), so the source coordinate is clamped at 0 on both axes
    meta, _, tokens, patches = _image_tables(GRIDS)
    dim = 256
    g = _gen(5)
    emb = torch.randn(patches, dim, generator=g)
    pos = torch.randn(grid * grid, dim, generator=g)
    cls = torch.randn(dim, generator=g)
    ln_w = torch.randn(dim, generator=g) * 0.1 + 1
    ln_b = torch.randn(dim, generator=g) * 0.1
    h = torch.empty(tokens, dim, device=DEV)
    ops.inkling2_vit_embed(
        emb.to(DEV), pos.to(DEV), cls.to(DEV), ln_w.to(DEV), ln_b.to(DEV), meta, h, 1e-5
    )

    pos64 = pos.double().view(grid, grid, dim).permute(2, 0, 1)[None]
    want, start = [], 0
    for rows, cols in GRIDS:
        n = rows * cols
        # bilinear, align_corners=False, source coordinate clamped at 0
        interp = F.interpolate(pos64, size=(2 * rows, 2 * cols), mode="bilinear")[0]
        y, x = _patch_coordinates(rows, cols)
        patch_rows = emb[start * 4 : (start + n) * 4].double() + interp[:, y, x].T
        seq = torch.cat([cls.double()[None], patch_rows])
        mean = seq.mean(-1, keepdim=True)
        var = ((seq - mean) ** 2).mean(-1, keepdim=True)
        want.append(
            (seq - mean) / torch.sqrt(var + 1e-5) * ln_w.double() + ln_b.double()
        )
        start += n
    err = float((h.cpu().double() - torch.cat(want)).abs().max())
    assert err <= 2e-5, err


# -------------------------------------------------------------------- RoPE


def _rope_angles(grids, inv):
    """Per token (y, x) -> angle per rotation pair: pairs 0-15 use x, 16-31 use y;
    positions are 1-based, the CLS token sits at (0, 0)."""
    pos = []
    for rows, cols in grids:
        y, x = _patch_coordinates(rows, cols)
        pos.append(torch.zeros(1, 2, dtype=torch.float64))
        pos.append(torch.stack([y, x], -1).double() + 1)
    pos = torch.cat(pos)
    return torch.cat([pos[:, 1:] * inv, pos[:, :1] * inv], -1)


def test_rope_tables_and_q_k_rotation():
    meta, _, tokens, _ = _image_tables(GRIDS)
    inv = 10000.0 ** (-torch.arange(16, dtype=torch.float64) / 16)
    cs = torch.empty(tokens, 32, 2, device=DEV)
    ops.inkling2_vit_rope_tables(meta, inv.float().to(DEV), cs)
    ang = _rope_angles(GRIDS, inv.float().double())
    cs = cs.cpu().double()
    assert float((cs[..., 0] - ang.cos()).abs().max()) <= 2e-6
    assert float((cs[..., 1] - ang.sin()).abs().max()) <= 2e-6

    dim = 1024
    qkv = (torch.randn(tokens, 3 * dim, generator=_gen(2)) * 2).half()
    rotated = qkv.to(DEV)
    ops.inkling2_vit_rope_qk(rotated, cs.float().to(DEV))
    rotated = rotated.cpu()

    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    for part, name in ((0, "q"), (1, "k")):
        x = (
            qkv[:, part * dim : (part + 1) * dim]
            .double()
            .view(tokens, dim // 64, 32, 2)
        )
        even, odd = x[..., 0], x[..., 1]  # interleaved pairs
        want = torch.stack([even * c - odd * s, even * s + odd * c], -1).reshape(
            tokens, dim
        )
        _check_f16(
            rotated[:, part * dim : (part + 1) * dim], want, 1e-4, f"rope {name}"
        )
    assert torch.equal(rotated[:, 2 * dim :], qkv[:, 2 * dim :])  # v is not rotated


# -------------------------------------------------------------------- pool


def test_pool_drops_each_images_cls_row():
    meta, _, tokens, _ = _image_tables(GRIDS)
    h = torch.randn(tokens, 128, generator=_gen(4)) * 5
    out = torch.empty(tokens - len(GRIDS), 128, dtype=torch.float16, device=DEV)
    ops.inkling2_vit_pool(h.to(DEV), meta, out)
    keep, start = torch.ones(tokens, dtype=torch.bool), 0
    for rows, cols in GRIDS:
        keep[start] = False
        start += 4 * rows * cols + 1
    assert torch.equal(out.cpu(), h[keep].half())


# --------------------------------------------------------------- attention


def _attention_reference(qkv: torch.Tensor, grids, dim: int) -> torch.Tensor:
    heads = dim // D_HEAD
    want, start = torch.empty(qkv.shape[0], dim, dtype=torch.float64), 0
    for rows, cols in grids:
        length = 4 * rows * cols + 1
        q, k, v = (
            qkv[start : start + length, i * dim : (i + 1) * dim]
            .double()
            .view(length, heads, D_HEAD)
            .transpose(0, 1)
            for i in range(3)
        )
        p = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D_HEAD), dim=-1)
        want[start : start + length] = (p @ v).transpose(0, 1).reshape(length, dim)
        start += length
    return want


def _attention(qkv: torch.Tensor, grids, dim: int) -> torch.Tensor:
    meta, blk, tokens, _ = _image_tables(grids)
    out = torch.full((tokens, dim), 7.0, dtype=torch.float16, device=DEV)
    ops.inkling2_vit_attention(qkv.to(DEV), out, meta, blk, int(blk[-1]))
    return out.cpu()


ATTN_GRIDS = [
    [(1, 1)],
    [(2, 2)],
    [(3, 5), (7, 2), (1, 1)],
    [(5, 3), (1, 2)],
    [(16, 16)],
]


@pytest.mark.parametrize("grids", ATTN_GRIDS, ids=str)
@pytest.mark.parametrize("dim", [128, 1024])
def test_attention_matches_fp64_softmax(grids, dim):
    tokens = sum(4 * r * c + 1 for r, c in grids)
    qkv = (torch.randn(tokens, 3 * dim, generator=_gen(tokens + dim)) * 1.5).half()
    got = _attention(qkv, grids, dim)
    want = _attention_reference(qkv, grids, dim)
    err = float((got.double() - want).abs().max())
    assert err <= 4e-3, err


def test_attention_softmax_is_stable_for_large_logits():
    grids, dim = [(4, 4)], 128
    tokens = 4 * 16 + 1
    qkv = (torch.randn(tokens, 3 * dim, generator=_gen(8)) * 6.0).half()
    got = _attention(qkv, grids, dim)
    want = _attention_reference(qkv, grids, dim)
    assert torch.isfinite(got).all()
    err = float((got.double() - want).abs().max())
    assert err <= 0.05 * float(want.abs().max()), err


def test_attention_images_never_see_each_other():
    dim = 128
    a, b, c = (3, 5), (7, 2), (2, 4)
    g = _gen(21)
    rows = {}
    for name, (r, cc) in (("a", a), ("b", b), ("c", c)):
        rows[name] = (torch.randn(4 * r * cc + 1, 3 * dim, generator=g) * 1.5).half()
    la = rows["a"].shape[0]
    alone_a = _attention(rows["a"], [a], dim)
    batch = _attention(torch.cat([rows["a"], rows["b"]]), [a, b], dim)
    assert torch.equal(batch[:la], alone_a)
    assert torch.equal(batch[la:], _attention(rows["b"], [b], dim))
    other = _attention(torch.cat([rows["a"], rows["c"]]), [a, c], dim)
    assert torch.equal(other[:la], alone_a)  # a different neighbour changes nothing


def test_attention_rejects_other_head_dims():
    tokens, dim = 5, 96  # head_dim 48 is not supported
    meta, blk, _, _ = _image_tables([(1, 1)])
    qkv = torch.zeros(tokens, 3 * dim, dtype=torch.float16, device=DEV)
    out = torch.empty(tokens, dim, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError):
        ops.inkling2_vit_attention(qkv, out, meta, blk, int(blk[-1]))
