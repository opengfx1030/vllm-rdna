# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp64 reference tests for the Inkling-2-Mini DSpark draft kernels (gfx1030).

Ops under test (``csrc/rocm/inkling2_dspark_rdna2.cu``, ``torch.ops._rocm_C``):
``inkling2_dspark_{store_taps, mix_rms, embed_block, add_rmsnorm, silu_mul,
gather_rows, gemv, gemm, qkv_prep, attention, markov_step, head_logits,
fold_bias}``.

Every oracle is torch/CPU float64 written from the model semantics (the same
formulas as ``tools/rdna/inkling2/dspark_check.py`` and the upstream
``Inkling2DSparkModel``); it never runs in serving.  The kernels compute in fp32
on fp16 operands, so a result may differ from the exact value by the final fp16
rounding (2^-11 relative) plus the fp32 accumulation error, and the tolerances
below are exactly that, not looser.  The paged worlds are adversarial: pages and
slots the kernel must not read hold NaN, so a read out of range or a missed mask
shows up as NaN instead of a small error, and a "sharp" attention world puts
dominant keys on both sides of the window boundary so an off-by-one in the window
changes the output by orders of magnitude more than the tolerance.

Mutations run by hand against these tests are listed in
``docs/rdna2/inkling2-dspark.md`` (each one makes at least one test below fail).

Run: ``pytest tests/kernels/rocm/test_inkling2_dspark_rdna2.py`` on a gfx1030 GPU.
"""

import math

import pytest
import torch

HIDDEN = 2304
HEAD_DIM = 128
KV_HEADS = 8
Q_PER_KV = 3
BLOCK = 8
RANK = 256
EPS = 1e-6
THETA = 1e6
FP16_REL = 2.0**-11  # half an fp16 ulp, relative


def _on_gfx1030() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx1030" in torch.cuda.get_device_properties(0).gcnArchName


requires_kernels = pytest.mark.skipif(
    not _on_gfx1030() or not hasattr(torch.ops._rocm_C, "inkling2_dspark_fold_bias"),
    reason="needs a gfx1030 GPU with the inkling2_dspark ops in _rocm_C",
)
pytestmark = requires_kernels


def _ops():
    from vllm import _custom_ops as ops

    return ops


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(gen, *shape, scale=1.0, dtype=torch.float32):
    return (torch.randn(*shape, generator=gen) * scale).to(dtype)


def _cuda(x: torch.Tensor) -> torch.Tensor:
    return x.to("cuda")


def _check_fp16_result(
    out: torch.Tensor, ref: torch.Tensor, *, floor: float, what: str, rel=FP16_REL
):
    """|out - ref| <= rel * |ref| + floor everywhere (ref is the exact fp64 value)."""
    out64 = out.detach().cpu().double()
    assert not torch.isnan(out64).any(), f"{what}: NaN in the output"
    err = (out64 - ref).abs()
    bound = rel * 1.0001 * ref.abs() + floor
    excess = (err - bound).max().item()
    assert excess <= 0, (
        f"{what}: max |err| {err.max().item():.3e} exceeds the fp16 rounding bound "
        f"by {excess:.3e} (max |ref| {ref.abs().max().item():.3e})"
    )


def _round_fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).double()


def _rms_scale(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ---------------------------------------------------------------------------
# bf16 tap capture
# ---------------------------------------------------------------------------
def _tie_values(n: int) -> torch.Tensor:
    """fp32 values whose bf16 rounding exercises ties, carries and the exponent step."""
    base = torch.tensor(
        [
            1.00390625,  # tie between 1.0 and 1 + 2^-7, rounds to even (1.0)
            1.01171875,  # tie between 1 + 2^-7 and 1 + 2^-6, rounds to even (up)
            1.00390626,  # just above a tie
            1.00390624,  # just below a tie
            -1.00390625,
            3.9999998,  # carries into the next exponent
            65504.0,
            1e-30,
            0.0,
            -0.0,
        ],
        dtype=torch.float32,
    )
    reps = -(-n // base.numel())
    return base.repeat(reps)[:n]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_store_taps_rounds_to_nearest_even(dtype):
    ops = _ops()
    rows = 13
    srcs, expect = [], []
    gen = _gen(1)
    for _ in range(4):
        x = _randn(gen, rows, HIDDEN, scale=30.0)
        x[0] = _tie_values(HIDDEN)
        x = x.to(dtype)
        srcs.append(x)
        expect.append(x.float().to(torch.bfloat16))
    dst = _cuda(torch.full((rows + 2, 4 * HIDDEN), float("nan"), dtype=torch.bfloat16))
    ops.inkling2_dspark_store_taps([_cuda(s) for s in srcs], dst, rows)
    got = dst.cpu()
    want = torch.cat(expect, dim=1)
    assert torch.equal(got[:rows].view(torch.int16), want.view(torch.int16))
    # rows past num_rows are not touched
    assert torch.isnan(got[rows:].float()).all()


def test_store_taps_strided_sources():
    """Taps are views into a wider buffer (row stride != hidden)."""
    ops = _ops()
    rows = 5
    gen = _gen(2)
    wide = _randn(gen, rows, 2 * HIDDEN, scale=3.0).to(torch.float16)
    srcs = [_cuda(wide[:, :HIDDEN]), _cuda(wide[:, HIDDEN:])]
    dst = _cuda(torch.zeros(rows, 2 * HIDDEN, dtype=torch.bfloat16))
    ops.inkling2_dspark_store_taps(srcs, dst, rows)
    want = torch.cat([wide[:, :HIDDEN], wide[:, HIDDEN:]], 1).float().to(torch.bfloat16)
    assert torch.equal(dst.cpu().view(torch.int16), want.view(torch.int16))


def test_store_taps_mixed_dtypes():
    """What the target hands over: three fp32 AttnRes partial blocks (some beyond the
    fp16 range) and the fp16 final output, in one call."""
    ops = _ops()
    rows = 9
    gen = _gen(4)
    big = _randn(gen, rows, HIDDEN, scale=2.0e5)  # not representable in fp16
    srcs = [
        big,
        _randn(gen, rows, HIDDEN, scale=40.0),
        _randn(gen, rows, HIDDEN, scale=3.0),
        _randn(gen, rows, HIDDEN, scale=3.0).to(torch.float16),
    ]
    dst = _cuda(torch.full((rows, 4 * HIDDEN), float("nan"), dtype=torch.bfloat16))
    ops.inkling2_dspark_store_taps([_cuda(s) for s in srcs], dst, rows)
    want = torch.cat([s.float().to(torch.bfloat16) for s in srcs], dim=1)
    assert torch.equal(dst.cpu().view(torch.int16), want.view(torch.int16))
    assert torch.isfinite(dst.cpu().float()).all()


# ---------------------------------------------------------------------------
# DFlare mix + RMSNorm (context path and f_rnn gather)
# ---------------------------------------------------------------------------
def _taps(gen, rows: int) -> torch.Tensor:
    """bf16 taps with outlier channels like real target block states."""
    x = _randn(gen, rows, 4 * HIDDEN, scale=2.0)
    x[:, ::97] *= 40.0
    return x.to(torch.bfloat16)


def _mix_rms_ref(taps, fuse_weight, norm_w, row0):
    n = taps.shape[0]
    layers = norm_w.shape[0]
    alpha = torch.softmax(fuse_weight[row0 : row0 + layers].double(), dim=-1)
    x = taps.double().view(n, 4, HIDDEN)
    mixed = torch.einsum("lt,nth->lnh", alpha, x)
    return _rms_scale(mixed, norm_w.double()[:, None, :])


def test_mix_rms_three_layers_matches_fp64():
    ops = _ops()
    gen = _gen(3)
    rows = 37
    taps = _taps(gen, rows)
    fuse = _randn(gen, 4, 4, scale=1.5)
    norm_w = _randn(gen, 3, HIDDEN, scale=1.0).abs().to(torch.float16) + 0.25
    out = _cuda(torch.full((3, rows + 3, HIDDEN), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_mix_rms(
        _cuda(taps), _cuda(fuse), _cuda(norm_w), out, None, None, rows, 0, EPS
    )
    ref = _mix_rms_ref(taps, fuse, norm_w, 0)
    # the layers are stacked [3, rows, H] over the first 3 * rows * H elements
    flat = out.flatten()
    got = flat[: 3 * rows * HIDDEN].view(3, rows, HIDDEN)
    _check_fp16_result(got, ref, floor=1e-5, what="mix_rms[3 layers]")
    assert torch.isnan(flat[3 * rows * HIDDEN :].cpu()).all(), "wrote past the output"


def test_mix_rms_gather_selects_last_committed_row():
    ops = _ops()
    gen = _gen(4)
    n_tokens = 64
    taps = _taps(gen, n_tokens)
    fuse = _randn(gen, 4, 4, scale=1.5)
    norm_w = (_randn(gen, 1, HIDDEN).abs() + 0.25).to(torch.float16)
    # four requests: an empty one first (its row index is -1 and clamps to 0), then
    # 20 context rows with the last 4 rejected, a single row, and 43 rows
    qsl = torch.tensor([0, 0, 20, 21, 64], dtype=torch.int32)
    rej = torch.tensor([0, 4, 0, 0], dtype=torch.int32)
    out = _cuda(torch.full((1, 4, HIDDEN), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_mix_rms(
        _cuda(taps),
        _cuda(fuse),
        _cuda(norm_w),
        out,
        _cuda(qsl),
        _cuda(rej),
        4,
        3,
        EPS,
    )
    src = (qsl[1:] - rej - 1).clamp(min=0).long()
    assert src.tolist() == [0, 15, 20, 63]
    ref = _mix_rms_ref(taps[src], fuse, norm_w, 3)
    _check_fp16_result(out, ref, floor=1e-5, what="mix_rms[f_rnn gather]")


# ---------------------------------------------------------------------------
# block embedding, residual add + RMSNorm, silu_mul, gather
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("norm_dtype", [torch.float16, torch.float32])
def test_embed_block_matches_fp64(id_dtype, norm_dtype):
    ops = _ops()
    gen = _gen(5)
    vocab, reqs = 301, 5
    rows = reqs * BLOCK
    embed = _randn(gen, vocab, HIDDEN, scale=0.7).to(torch.float16)
    enw = (_randn(gen, HIDDEN).abs() + 0.5).to(norm_dtype)
    mask = _randn(gen, HIDDEN, scale=0.4).to(torch.float16)
    inw = (_randn(gen, HIDDEN).abs() + 0.5).to(torch.float16)
    ids = torch.randint(0, vocab, (rows + 4,), generator=gen).to(id_dtype)
    resid = _cuda(torch.full((rows + 4, HIDDEN), float("nan")))
    xn = _cuda(torch.full((rows + 4, HIDDEN), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_embed_block(
        _cuda(ids),
        _cuda(embed),
        _cuda(enw),
        _cuda(mask),
        _cuda(inw),
        resid,
        xn,
        rows,
        BLOCK,
        EPS,
    )
    ref_resid = torch.empty(rows, HIDDEN, dtype=torch.float64)
    for r in range(rows):
        if r % BLOCK == 0:
            e = embed[int(ids[r])].double()
            ref_resid[r] = _rms_scale(e, enw.double())
        else:
            ref_resid[r] = mask.double()
    ref_xn = _rms_scale(ref_resid, inw.double())
    got = resid.cpu().double()
    assert not torch.isnan(got[:rows]).any()
    assert torch.isnan(got[rows:]).all(), "rows past num_rows must be untouched"
    torch.testing.assert_close(got[:rows], ref_resid, rtol=2e-6, atol=2e-6)
    _check_fp16_result(xn[:rows], ref_xn, floor=1e-5, what="embed_block xn")


def test_add_rmsnorm_fp16_delta_matches_fp64():
    ops = _ops()
    gen = _gen(6)
    rows = 21
    resid0 = _randn(gen, rows + 2, HIDDEN, scale=20.0)
    delta = _randn(gen, rows + 2, HIDDEN, scale=3.0).to(torch.float16)
    w = (_randn(gen, HIDDEN).abs() + 0.25).to(torch.float16)
    resid = _cuda(resid0.clone())
    out = _cuda(torch.full((rows + 2, HIDDEN), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_add_rmsnorm(resid, _cuda(delta), _cuda(w), out, rows, 0, EPS)
    ref_resid = resid0[:rows].double() + delta[:rows].double()
    torch.testing.assert_close(
        resid.cpu()[:rows].double(), ref_resid, rtol=1e-6, atol=1e-6
    )
    assert torch.equal(resid.cpu()[rows:], resid0[rows:]), "extra rows modified"
    _check_fp16_result(
        out[:rows], _rms_scale(ref_resid, w.double()), floor=1e-5, what="add_rmsnorm"
    )


def test_add_rmsnorm_split_partials_sum_in_order():
    ops = _ops()
    gen = _gen(7)
    rows, splits = 17, 4
    resid0 = _randn(gen, rows, HIDDEN, scale=20.0)
    part = _randn(gen, splits, rows + 3, HIDDEN, scale=5.0)
    w = (_randn(gen, HIDDEN).abs() + 0.25).to(torch.float16)
    resid = _cuda(resid0.clone())
    out = _cuda(torch.zeros(rows, HIDDEN, dtype=torch.float16))
    ops.inkling2_dspark_add_rmsnorm(
        resid, _cuda(part), _cuda(w), out, rows, splits, EPS
    )
    ref_resid = resid0.double() + part[:, :rows].double().sum(0)
    torch.testing.assert_close(resid.cpu().double(), ref_resid, rtol=1e-6, atol=1e-5)
    _check_fp16_result(
        out, _rms_scale(ref_resid, w.double()), floor=1e-5, what="add_rmsnorm[split]"
    )


def test_silu_mul_interleaved_gate_up():
    ops = _ops()
    gen = _gen(8)
    rows, inter = 9, 9216
    gu = _randn(gen, rows + 1, 2 * inter, scale=2.5).to(torch.float16)
    out = _cuda(torch.full((rows + 1, inter), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_silu_mul(_cuda(gu), out, rows)
    g = gu[:rows, 0::2].double()
    u = gu[:rows, 1::2].double()
    ref = g * torch.sigmoid(g) * u
    _check_fp16_result(out[:rows], ref, floor=1e-5, what="silu_mul")
    assert torch.isnan(out[rows:].cpu()).all()


def test_gather_rows_is_exact():
    ops = _ops()
    gen = _gen(9)
    src = _randn(gen, 40, HIDDEN).to(torch.float16)
    idx = torch.randperm(40, generator=gen)[:23].to(torch.int64)
    dst = _cuda(torch.zeros(23, HIDDEN, dtype=torch.float16))
    ops.inkling2_dspark_gather_rows(_cuda(src), _cuda(idx), dst, 23)
    assert torch.equal(dst.cpu(), src[idx])


# ---------------------------------------------------------------------------
# skinny and dense GEMMs
# ---------------------------------------------------------------------------
def _gemm_inputs(gen, m, n, k):
    x = _randn(gen, m, k, scale=1.0).to(torch.float16)
    w = _randn(gen, n, k, scale=0.05).to(torch.float16)
    return x, w, x.double() @ w.double().T


@pytest.mark.parametrize("m", [1, 3, 8, 9, 16, 17, 24, 32])
@pytest.mark.parametrize(
    "n,k", [(2304, 2304), (5120, 2304), (1000, 9216), (3072, 3072)]
)
def test_gemv_fp16_epilogue_matches_fp64(m, n, k):
    ops = _ops()
    x, w, ref = _gemm_inputs(_gen(10 + m), m, n, k)
    out = _cuda(torch.full((m + 2, n), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_gemv(_cuda(x), _cuda(w), out, None, m)
    _check_fp16_result(
        out[:m], ref, floor=1e-5 * ref.abs().max().item(), what=f"gemv m={m} n={n}"
    )
    assert torch.isnan(out[m:].cpu()).all(), "rows past num_rows must be untouched"


@pytest.mark.parametrize(
    "m,variant", [(8, 2), (8, 4), (16, 1), (16, 2), (32, 1), (5, 4)]
)
def test_gemv_forced_variants_agree_with_fp64(m, variant):
    ops = _ops()
    n, k = 1111, 2304  # N not a multiple of any rows-per-block
    x, w, ref = _gemm_inputs(_gen(20 + m), m, n, k)
    out = _cuda(torch.zeros(m, n, dtype=torch.float16))
    ops.inkling2_dspark_gemv(_cuda(x), _cuda(w), out, None, m, 0, variant)
    _check_fp16_result(out, ref, floor=1e-5 * ref.abs().max().item(), what="variant")


@pytest.mark.parametrize("splits,k", [(2, 3072), (4, 9216), (9, 2304), (1, 2304)])
def test_gemv_split_partials_sum_to_fp64(splits, k):
    ops = _ops()
    m, n = 8, 2304
    x, w, ref = _gemm_inputs(_gen(30 + splits), m, n, k)
    part = _cuda(torch.full((splits + 1, m + 1, n), float("nan")))
    ops.inkling2_dspark_gemv(_cuda(x), _cuda(w), part, None, m, splits)
    got = part.cpu()[:splits, :m]
    assert not torch.isnan(got).any()
    # fp32 accumulation only: well below the fp16 rounding of the outputs
    torch.testing.assert_close(
        got.double().sum(0), ref, rtol=2e-6, atol=2e-6 * ref.abs().max().item()
    )
    if splits > 1:
        # every split carries only its K slice
        kps = k // splits
        for s in range(splits):
            ref_s = (
                x[:, s * kps : (s + 1) * kps].double()
                @ w[:, s * kps : (s + 1) * kps].double().T
            )
            torch.testing.assert_close(
                got[s].double(), ref_s, rtol=2e-6, atol=2e-6 * ref.abs().max().item()
            )


def test_gemv_gathers_rows_through_x_index():
    ops = _ops()
    gen = _gen(40)
    n, k, m = 2304, 2304, 14
    x_all = _randn(gen, 64, k).to(torch.float16)
    w = _randn(gen, n, k, scale=0.05).to(torch.float16)
    idx = torch.randperm(64, generator=gen)[:m].to(torch.int64)
    out = _cuda(torch.zeros(m, n, dtype=torch.float16))
    ops.inkling2_dspark_gemv(_cuda(x_all), _cuda(w), out, _cuda(idx), m)
    ref = x_all[idx].double() @ w.double().T
    _check_fp16_result(out, ref, floor=1e-5 * ref.abs().max().item(), what="x_index")


def test_gemv_rejects_unsupported_shapes():
    ops = _ops()
    x = _cuda(torch.zeros(40, 2304, dtype=torch.float16))
    w = _cuda(torch.zeros(64, 2304, dtype=torch.float16))
    out = _cuda(torch.zeros(40, 64, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="at most 32 rows"):
        ops.inkling2_dspark_gemv(x, w, out, None, 33)
    x2 = _cuda(torch.zeros(8, 2312, dtype=torch.float16))
    w2 = _cuda(torch.zeros(64, 2312, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="multiple of 256"):
        ops.inkling2_dspark_gemv(x2, w2, _cuda(torch.zeros(8, 64).half()), None, 8)


@pytest.mark.parametrize(
    "design,m", [(0, 300), (1, 300), (2, 129), (3, 300), (4, 32), (5, 64), (5, 7)]
)
def test_dense_gemm_designs_match_fp64(design, m):
    ops = _ops()
    n, k = 2304, 2304
    x, w, ref = _gemm_inputs(_gen(50 + design), m, n, k)
    out = _cuda(torch.full((m, n), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_gemm(_cuda(x), _cuda(w), out, m, design)
    _check_fp16_result(
        out, ref, floor=1e-5 * ref.abs().max().item(), what=f"dense design {design}"
    )


def test_dense_gemm_refuses_what_the_design_cannot_run():
    ops = _ops()
    x = _cuda(torch.zeros(64, 2312, dtype=torch.float16))
    w = _cuda(torch.zeros(128, 2312, dtype=torch.float16))
    out = _cuda(torch.zeros(64, 128, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="cannot run"):
        ops.inkling2_dspark_gemm(x, w, out, 64, 0)  # K not a multiple of 32 * ...


# ---------------------------------------------------------------------------
# per-head Q/K RMSNorm + RoPE + paged KV write
# ---------------------------------------------------------------------------
def _inv_freq() -> torch.Tensor:
    return THETA ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)


def _rope_ref(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Neox rope in float64 with exact angles; x [T, heads, 128], pos [T]."""
    ang = pos.double()[:, None] * _inv_freq()[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :64], x[..., 64:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _head_norm_ref(x, w):
    return _rms_scale(x, w.double())


POSITIONS = [0, 1, 17, 1023, 65535, 262_145, 1_000_003, 1_048_575]


@pytest.mark.parametrize("bs", [16, 64])
def test_qkv_prep_block_matches_fp64(bs):
    ops = _ops()
    gen = _gen(60)
    tokens = 16
    qh, kh = KV_HEADS * Q_PER_KV, KV_HEADS
    qkv = _randn(gen, tokens, (qh + 2 * kh) * HEAD_DIM, scale=2.0).to(torch.float16)
    qw = (_randn(gen, HEAD_DIM).abs() + 0.3).to(torch.float16)
    kw = (_randn(gen, HEAD_DIM).abs() + 0.3).to(torch.float16)
    pos = torch.tensor(POSITIONS + POSITIONS, dtype=torch.int64)
    nblocks = 9
    # unique slots in a shuffled pool, two PAD tokens
    slots = (
        torch.randperm(nblocks * bs - bs, generator=gen)[:tokens].to(torch.int64) + bs
    )
    slots[3] = -1
    slots[11] = -1
    cache = _cuda(
        torch.full((nblocks, kh, bs, 2 * HEAD_DIM), float("nan"), dtype=torch.float16)
    )
    q_out = _cuda(
        torch.full((tokens, qh * HEAD_DIM), float("nan"), dtype=torch.float16)
    )
    ops.inkling2_dspark_qkv_prep(
        _cuda(qkv),
        _cuda(pos),
        _cuda(slots),
        _cuda(qw),
        _cuda(kw),
        _cuda(_inv_freq()),
        q_out,
        cache,
        tokens,
        qh,
        kh,
        EPS,
    )
    q_in = qkv[:, : qh * HEAD_DIM].double().view(tokens, qh, HEAD_DIM)
    k_in = (
        qkv[:, qh * HEAD_DIM : (qh + kh) * HEAD_DIM].double().view(tokens, kh, HEAD_DIM)
    )
    v_in = qkv[:, (qh + kh) * HEAD_DIM :].double().view(tokens, kh, HEAD_DIM)
    q_ref = _rope_ref(_head_norm_ref(q_in, qw), pos)
    k_ref = _rope_ref(_head_norm_ref(k_in, kw), pos)
    _check_fp16_result(
        q_out.view(tokens, qh, HEAD_DIM), q_ref, floor=2e-5, what="rope(norm(q))"
    )
    cache_cpu = cache.cpu()
    touched = torch.zeros(nblocks, bs, dtype=torch.bool)
    for t in range(tokens):
        slot = int(slots[t])
        if slot < 0:
            continue
        blk, off = divmod(slot, bs)
        touched[blk, off] = True
        _check_fp16_result(
            cache_cpu[blk, :, off, :HEAD_DIM],
            k_ref[t],
            floor=2e-5,
            what=f"rope(norm(k)) token {t} pos {int(pos[t])}",
        )
        assert torch.equal(cache_cpu[blk, :, off, HEAD_DIM:].double(), v_in[t]), (
            f"v of token {t} must be stored raw"
        )
    # nothing else was written (PAD slots, other slots)
    untouched = cache_cpu.permute(0, 2, 1, 3)[~touched]
    assert torch.isnan(untouched).all(), "the kernel wrote outside its slots"


def test_qkv_prep_context_path_has_no_q_and_skips_pad_slots():
    ops = _ops()
    gen = _gen(61)
    tokens, bs, kh = 12, 16, KV_HEADS
    kv = _randn(gen, tokens, 2 * kh * HEAD_DIM, scale=2.0).to(torch.float16)
    kw = (_randn(gen, HEAD_DIM).abs() + 0.3).to(torch.float16)
    pos = torch.tensor(POSITIONS + POSITIONS[:4], dtype=torch.int64)
    slots = torch.arange(tokens, dtype=torch.int64) + 16
    slots[[0, 5, 6]] = -1  # rejected / padding context rows
    cache = _cuda(
        torch.full((4, kh, bs, 2 * HEAD_DIM), float("nan"), dtype=torch.float16)
    )
    ops.inkling2_dspark_qkv_prep(
        _cuda(kv),
        _cuda(pos),
        _cuda(slots),
        _cuda(kw),  # q norm weight is unused without q heads
        _cuda(kw),
        _cuda(_inv_freq()),
        None,
        cache,
        tokens,
        0,
        kh,
        EPS,
    )
    k_in = kv[:, : kh * HEAD_DIM].double().view(tokens, kh, HEAD_DIM)
    v_in = kv[:, kh * HEAD_DIM :].double().view(tokens, kh, HEAD_DIM)
    k_ref = _rope_ref(_head_norm_ref(k_in, kw), pos)
    cache_cpu = cache.cpu()
    for t in range(tokens):
        slot = int(slots[t])
        if slot < 0:
            continue
        blk, off = divmod(slot, bs)
        _check_fp16_result(
            cache_cpu[blk, :, off, :HEAD_DIM], k_ref[t], floor=2e-5, what=f"ctx k {t}"
        )
        assert torch.equal(cache_cpu[blk, :, off, HEAD_DIM:].double(), v_in[t])
    written = (~torch.isnan(cache_cpu[..., 0])).sum().item()
    assert written == (tokens - 3) * kh, "PAD context rows must not be written"


# ---------------------------------------------------------------------------
# bidirectional block attention over the paged cache
# ---------------------------------------------------------------------------
class AttnWorld:
    """Requests with ``seq_lens`` (= context + 8) in a shuffled, NaN-poisoned cache.

    The cache holds only what the layer may read: keys [start, seq_len) of each
    request (``window`` = 0 for a full layer).  Everything else -- unreferenced
    pages, slots past the sequence end and every position before the window --
    is NaN, and block-table entries for pages fully before the window point to the
    null block 0 (also NaN), as the sliding-window manager does.  ``sharp`` makes
    the first window key and the block's last key dominant (see ``_sharpen``).
    """

    def __init__(self, seq_lens, *, window, bs=16, q_scale=1.0, seed=0, sharp=False):
        gen = _gen(seed)
        self.seq_lens = list(seq_lens)
        self.window, self.bs, self.R = window, bs, len(seq_lens)
        nb = [-(-L // bs) if L > 0 else 0 for L in seq_lens]
        total = sum(nb) + 3
        pool = (torch.randperm(total - 1, generator=gen) + 1).tolist()
        self.table = torch.zeros(self.R, max(max(nb), 1) + 2, dtype=torch.int32)
        cache = torch.full((total, KV_HEADS, bs, 2 * HEAD_DIM), float("nan"))
        q = _randn(gen, self.R * BLOCK, KV_HEADS * Q_PER_KV * HEAD_DIM, scale=q_scale)
        self.starts, self.keys, self.values = [], [], []
        for r, length in enumerate(seq_lens):
            start = self.start(length)
            k = _randn(gen, length, KV_HEADS, HEAD_DIM)
            v = _randn(gen, length, KV_HEADS, HEAD_DIM)
            if sharp:
                self._sharpen(gen, q, k, v, r, start, length)
            for b in range(nb[r]):
                page = pool.pop()
                if (b + 1) * bs <= start:
                    continue  # fully before the window: the null block
                self.table[r, b] = page
                lo, hi = max(start, b * bs), min(length, (b + 1) * bs)
                cache[page, :, lo - b * bs : hi - b * bs, :HEAD_DIM] = k[
                    lo:hi
                ].transpose(0, 1)
                cache[page, :, lo - b * bs : hi - b * bs, HEAD_DIM:] = v[
                    lo:hi
                ].transpose(0, 1)
            self.starts.append(start)
            self.keys.append(k)
            self.values.append(v)
        self.cache = cache.to(torch.float16)
        self.q = q.to(torch.float16)

    def start(self, seq_len):
        p = seq_len - BLOCK
        return max(0, p - self.window) if self.window > 0 else 0

    @staticmethod
    def _sharpen(gen, q, k, v, r, start, length):
        """The first window key (values ~ +10) and the block's last key (values
        ~ -10) get logit ~ +24 for every query of the request, all other keys ~ 0,
        so the output is the mean of the two and dropping either one (an off-by-one
        in the window start or the key range) moves it by ~5 per element, far past
        any tolerance.  Keys before the window are NaN in the cache, so reading one
        shows up as NaN."""
        dirs = _randn(gen, KV_HEADS, HEAD_DIM)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        for h in range(KV_HEADS):
            for g in range(Q_PER_KV):
                col = (h * Q_PER_KV + g) * HEAD_DIM
                for row in range(BLOCK):
                    q[r * BLOCK + row, col : col + HEAD_DIM] = dirs[h] * math.sqrt(
                        HEAD_DIM
                    ) + 0.01 * _randn(gen, HEAD_DIM)
            k[start, h] = dirs[h] * 24.0
            k[length - 1, h] = dirs[h] * 24.0
            v[start, h] = _randn(gen, HEAD_DIM) + 10.0
            v[length - 1, h] = _randn(gen, HEAD_DIM) - 10.0

    def reference(self, start_delta=0, end_delta=0) -> torch.Tensor:
        """fp64 attention of every request: [R * 8, 24 * 128].

        ``start_delta`` / ``end_delta`` move the key range (the oracle's own
        off-by-one mutations, see ``test_attention_window_boundaries_are_exact``).
        """
        out = torch.zeros(
            self.R * BLOCK, KV_HEADS * Q_PER_KV * HEAD_DIM, dtype=torch.float64
        )
        for r, length in enumerate(self.seq_lens):
            start = self.starts[r] + start_delta
            length = length + end_delta
            if length <= 0 or length - start <= 0:
                continue
            # the cache is fp16: the oracle sees exactly the same fp16 values
            k = self.keys[r][start:length].to(torch.float16).double()
            v = self.values[r][start:length].to(torch.float16).double()
            q = self.q[r * BLOCK : (r + 1) * BLOCK].double()
            q = q.view(BLOCK, KV_HEADS * Q_PER_KV, HEAD_DIM)
            for h in range(KV_HEADS):
                for g in range(Q_PER_KV):
                    s = q[:, h * Q_PER_KV + g] @ k[:, h].T / math.sqrt(HEAD_DIM)
                    o = torch.softmax(s, dim=-1) @ v[:, h]  # [8, 128]
                    col = (h * Q_PER_KV + g) * HEAD_DIM
                    out[r * BLOCK : (r + 1) * BLOCK, col : col + HEAD_DIM] = o
        return out


def _run_attention(world: AttnWorld, splits: int) -> torch.Tensor:
    ops = _ops()
    seq = _cuda(torch.tensor(world.seq_lens, dtype=torch.int32))
    out = _cuda(torch.full_like(world.q, float("nan")))
    part = _cuda(
        torch.full(
            (ops.inkling2_dspark_attention_part_numel(world.R, KV_HEADS, splits),),
            float("nan"),
        )
    )
    ops.inkling2_dspark_attention(
        _cuda(world.q),
        _cuda(world.cache),
        _cuda(world.table),
        seq,
        out,
        part,
        world.R,
        world.window,
        splits,
    )
    return out.cpu()


ATTN_FLOOR = 2e-5


@pytest.mark.parametrize("splits", [1, 3, 8])
@pytest.mark.parametrize("window", [0, 1024])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [8 + 5],  # five context tokens: the block sees 13 keys
        [8],  # no context at all: only the block's own keys
        [8 + 1023, 8 + 1024, 8 + 1025, 8 + 3000],  # around the window edge
        [8 + 63, 8 + 64, 8 + 65, 8 + 200, 8 + 4091],  # tile edges, long context
    ],
)
def test_attention_matches_fp64(seq_lens, window, splits):
    world = AttnWorld(seq_lens, window=window, seed=70 + splits)
    got = _run_attention(world, splits)
    _check_fp16_result(
        got,
        world.reference(),
        floor=ATTN_FLOOR,
        what=f"attention seq={seq_lens} window={window} splits={splits}",
    )


@pytest.mark.parametrize("splits", [1, 5])
def test_attention_window_boundaries_are_exact(splits):
    """Sharp keys at the first window key and the block's last key: an off-by-one in
    the window start or the key range changes the result by ~10, not by 1e-3."""
    world = AttnWorld(
        [8 + 1024 + 17, 8 + 1024, 8 + 300, 8 + 2500],
        window=1024,
        seed=90,
        sharp=True,
    )
    got = _run_attention(world, splits)
    ref = world.reference()
    _check_fp16_result(got, ref, floor=1e-4, what="sharp window attention")
    # the oracle is sensitive to the two boundaries: moving either by one key
    # changes the expected output by ~5 per element, 5 orders above the tolerance
    for name, shift in (
        ("first window key dropped", {"start_delta": 1}),
        ("block's last key dropped", {"end_delta": -1}),
    ):
        moved = (world.reference(**shift) - ref).abs().max().item()
        assert moved > 1.0, f"{name}: the oracle moves by only {moved:.3e}"


@pytest.mark.parametrize("window", [0, 1024])
def test_attention_nonpositive_context_gives_zeros(window):
    world = AttnWorld([8 + 40, 0], window=window, seed=91)
    got = _run_attention(world, 2)
    assert not torch.isnan(got).any()
    assert (got[BLOCK:] == 0).all(), "a request without keys must output zeros"
    ref = world.reference()
    _check_fp16_result(got, ref, floor=ATTN_FLOOR, what="mixed zero/non-zero requests")


def test_attention_splits_agree_with_each_other():
    """Different split counts reorder the fp32 merge only: results agree to fp16."""
    world = AttnWorld([8 + 4000, 8 + 700], window=0, seed=92)
    a = _run_attention(world, 1)
    b = _run_attention(world, 16)
    _check_fp16_result(a, b.double(), floor=ATTN_FLOOR, what="splits 1 vs 16", rel=2e-3)


# ---------------------------------------------------------------------------
# Markov head
# ---------------------------------------------------------------------------
def _markov_ref_step(prev_row, state, w1, wt, bias, ctx_h, ctx_f):
    x = torch.cat([state, w1[prev_row].double()])  # [512]
    raw = x @ wt.double() + bias.double() + ctx_h.double() + ctx_f.double()
    gate, cand, out = raw.chunk(3)
    g = torch.sigmoid(gate)
    new_state = g * state + (1 - g) * torch.tanh(cand)
    return new_state, torch.tanh(out)


@pytest.mark.parametrize("prev_dtype", [torch.int32, torch.int64])
def test_markov_chain_matches_fp64(prev_dtype):
    ops = _ops()
    gen = _gen(100)
    reqs, steps, vocab = 5, 7, 3001
    w1 = _randn(gen, vocab, RANK, scale=0.8).to(torch.float16)
    wt = _randn(gen, 2 * RANK, 3 * RANK, scale=0.12).to(torch.float16)
    bias = _randn(gen, 3 * RANK, scale=0.3)
    ctx_h = _randn(gen, reqs * steps, 3 * RANK, scale=0.6)
    ctx_f = _randn(gen, reqs, 3 * RANK, scale=0.6)
    anchor = torch.randint(0, vocab, (reqs * BLOCK,), generator=gen).to(prev_dtype)
    drafted = torch.zeros(reqs + 1, steps, dtype=torch.int64)
    state = _cuda(torch.full((reqs, RANK), 123.0))  # garbage: step 0 must ignore it
    latent = _cuda(torch.zeros(reqs, RANK, dtype=torch.float16))
    ref_state = torch.zeros(reqs, RANK, dtype=torch.float64)
    anchor_g, drafted_g = _cuda(anchor), _cuda(drafted)
    for step in range(steps):
        if step == 0:
            prev, stride = anchor_g, BLOCK
            prev_tok = anchor[::BLOCK].long()
        else:
            prev, stride = drafted_g[:, step - 1], drafted_g.stride(0)
            prev_tok = drafted[:reqs, step - 1]
        ops.inkling2_dspark_markov_step(
            prev,
            stride,
            state,
            _cuda(w1),
            _cuda(wt),
            _cuda(bias),
            _cuda(ctx_h),
            _cuda(ctx_f),
            latent,
            reqs,
            step,
            steps,
            step == 0,
        )
        ref_lat = torch.empty(reqs, RANK, dtype=torch.float64)
        for r in range(reqs):
            ref_state[r], ref_lat[r] = _markov_ref_step(
                int(prev_tok[r]),
                ref_state[r],
                w1,
                wt,
                bias,
                ctx_h[r * steps + step],
                ctx_f[r],
            )
        torch.testing.assert_close(
            state.cpu().double(), ref_state, rtol=2e-5, atol=2e-5
        )
        _check_fp16_result(latent, ref_lat, floor=1e-5, what=f"latent step {step}")
        # any deterministic tokens stand in for the runner's sampler
        drafted[:reqs, step] = (prev_tok * 7 + step * 13 + 5) % vocab
        drafted_g.copy_(drafted)


def _head_ref(base_rows, latent, fold_w, fold_b, cap):
    x = base_rows.double() + fold_b.double() + latent.double() @ fold_w.double().T
    return cap * torch.tanh(x / cap)


@pytest.mark.parametrize("reqs", [1, 2, 3, 5, 8, 17, 32])
def test_head_logits_matches_fp64(reqs):
    ops = _ops()
    gen = _gen(110 + reqs)
    vocab, steps, step, cap = 4099, 7, 3, 30.0  # vocab not a multiple of 32
    base = _randn(gen, reqs * steps, vocab + 5, scale=8.0).to(torch.float16)
    latent = torch.tanh(_randn(gen, reqs, RANK, scale=1.5)).to(torch.float16)
    fold_w = _randn(gen, vocab, RANK, scale=0.4).to(torch.float16)
    fold_b = _randn(gen, vocab, scale=2.0)
    logits = _cuda(torch.full((reqs, vocab + 5), float("nan"), dtype=torch.float16))
    ops.inkling2_dspark_head_logits(
        _cuda(base),
        _cuda(latent),
        _cuda(fold_w),
        _cuda(fold_b),
        logits,
        reqs,
        step,
        steps,
        cap,
    )
    rows = base.view(reqs, steps, -1)[:, step, :vocab]
    ref = _head_ref(rows, latent, fold_w, fold_b, cap)
    got = logits.cpu()
    _check_fp16_result(got[:, :vocab], ref, floor=3e-5, what=f"head_logits R={reqs}")
    assert torch.isnan(got[:, vocab:]).all(), "columns past the vocabulary written"
    assert got[:, :vocab].abs().max() <= cap, "softcap exceeded"


def test_fold_bias_matches_fp64():
    ops = _ops()
    gen = _gen(120)
    vocab = 1003
    head = _randn(gen, vocab, HIDDEN, scale=0.5).to(torch.float16)
    b = _randn(gen, HIDDEN, scale=0.3).to(torch.float16)
    out = _cuda(torch.full((vocab + 3,), float("nan")))
    ops.inkling2_dspark_fold_bias(_cuda(head), _cuda(b), out)
    ref = head.double() @ b.double()
    got = out.cpu()
    # fp32 accumulation: each lane adds 9 products and the wave 5 more, so the
    # error is a few fp32 ulps of the sum of magnitudes (an fp16 accumulator would
    # be off by 2^-11 of it, 4 orders larger)
    magnitude = head.double().abs() @ b.double().abs()
    err = (got[:vocab].double() - ref).abs()
    assert bool((err <= 1e-7 * magnitude).all()), (err / magnitude).max().item()
    assert torch.isnan(got[vocab:]).all()


def _argmax_ref(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(logits.isnan(), float("-inf")).argmax(dim=-1)


@pytest.mark.parametrize("rows,vocab,ld", [(1, 200058, 200064), (7, 4099, 4112)])
def test_argmax_matches_reference(rows, vocab, ld):
    """The first maximal index, NaN as -inf; ties, all -inf rows and a strided
    output column (a token table) included."""
    ops = _ops()
    gen = _gen(130 + rows)
    logits = _randn(gen, rows, ld, scale=3.0).to(torch.float16)
    # ties: the maximum sits at three positions; the lowest must win
    logits[0, [vocab - 1, 11, 4097 % vocab]] = 20.0
    if rows > 2:
        logits[1, :] = float("-inf")  # all -inf: index 0
        logits[2, :vocab] = float("nan")  # NaN never wins over a finite value
        logits[2, 5] = -3.0
    if rows > 3:
        logits[3, 100] = float("nan")  # NaN is never the maximum
        logits[3, vocab - 3] = 9.0
        logits[3, vocab:] = 1e4  # past the vocabulary: ignored
    dev = _cuda(logits)
    table = torch.full((rows + 2, 9), -7, dtype=torch.int64, device="cuda")
    out = table[1 : rows + 1, 3]  # stride 9
    ops.inkling2_dspark_argmax(dev, out, rows, vocab)
    ref = _argmax_ref(logits[:, :vocab])
    assert torch.equal(out.cpu(), ref), (out.cpu().tolist(), ref.tolist())
    assert int(out.cpu()[0]) == 11
    if rows > 2:
        assert out[1].item() == 0 and out[2].item() == 5
    # nothing but the requested column was written
    mask = torch.full(table.shape, True, device="cuda")
    mask[1 : rows + 1, 3] = False
    assert bool((table[mask] == -7).all())


def test_argmax_unaligned_rows_use_the_scalar_path():
    """A row stride that is not a multiple of 8 elements cannot be read in 16-byte
    vectors; the result must not change."""
    ops = _ops()
    gen = _gen(140)
    rows, vocab, ld = 3, 1001, 1003
    logits = _randn(gen, rows, ld, scale=2.0).to(torch.float16)
    out = torch.empty(rows, dtype=torch.int64, device="cuda")
    ops.inkling2_dspark_argmax(_cuda(logits), out, rows, vocab)
    assert torch.equal(out.cpu(), _argmax_ref(logits[:, :vocab]))


# ---------------------------------------------------------------------------
# DFlash / DSpark input preparation (port of the runner's Triton kernel)
# ---------------------------------------------------------------------------
PAD_SLOT = -1
PREP_OUTPUTS = (
    "input_ids",
    "positions",
    "is_padding",
    "query_start_loc",
    "seq_lens",
    "query_slot",
    "ctx_positions",
    "ctx_slot",
    "sample_indices",
    "sample_pos",
    "sample_idx_mapping",
    "out_temperature",
    "out_seeds",
)


class PrepWorld:
    """A target batch (rejected suffixes, chunked prefill, evicted/null blocks, a
    block table narrower than the positions, positions past ``max_model_len``) and
    output buffers pre-filled with values the kernel must overwrite or leave."""

    def __init__(
        self, seed, num_reqs, max_reqs, anchor, idx_dtype, long_first=False
    ) -> None:
        gen = _gen(seed)
        self.num_reqs, self.max_reqs, self.anchor = num_reqs, max_reqs, anchor
        self.steps, self.block_size = 7, 16
        # the runner's layout: one sample row per step; with ``sample_from_anchor`` the
        # anchor row is sampled too, so the block is ``steps`` rows, else ``steps + 1``
        self.nq = self.steps + (0 if anchor else 1)
        self.stride, self.max_model_len, self.parallel_token = 40, 700, 151
        rnd = lambda lo, hi, n: torch.randint(lo, hi, (n,), generator=gen)  # noqa: E731
        lens = rnd(1, 14, num_reqs)
        if long_first:
            lens[0] = 200  # two 128-column programs for one request
        self.rejected = torch.stack(
            [
                rnd(0, int(n), 1)[0] if i % 2 else torch.tensor(0)
                for i, n in enumerate(lens)
            ]
        )
        if long_first:
            # 100 valid rows, a 100-row rejected suffix straddling column 128
            self.rejected[0] = 100
        self.num_sampled = rnd(0, 4, num_reqs)
        self.num_sampled[0] = 0  # chunked prefill: the next prompt token is the bonus
        self.num_sampled[-1] = 2
        base = rnd(0, 900, num_reqs)
        self.qsl = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).int()
        self.target_positions = torch.cat(
            [b + torch.arange(int(n)) for b, n in zip(base, lens)]
        )
        self.max_tokens = int(lens.sum()) + max_reqs * self.nq + 37
        self.lens = lens
        self.idx_mapping = torch.randperm(max_reqs, generator=gen)[:num_reqs]
        self.last_sampled = rnd(1, 200000, max_reqs).view(max_reqs, 1)
        self.next_prefill = torch.randint(
            1, 200000, (2, max_reqs), generator=gen, dtype=torch.int32
        )
        self.temperature = torch.rand(max_reqs, generator=gen)
        self.seeds = torch.randint(0, 1 << 40, (max_reqs,), generator=gen)
        self.block_table = torch.randint(
            1, 5000, (max_reqs, self.stride), generator=gen, dtype=torch.int32
        )
        self.block_table[torch.rand(max_reqs, self.stride, generator=gen) < 0.2] = 0
        self.idx_dtype = idx_dtype
        self.fresh = {
            "input_ids": torch.full((self.max_tokens,), -77, dtype=torch.int32),
            "positions": torch.full((self.max_tokens,), -88, dtype=torch.int64),
            "is_padding": torch.rand(self.max_tokens, generator=gen) < 0.5,
            "query_start_loc": torch.full((max_reqs + 1,), -5, dtype=torch.int32),
            "seq_lens": torch.full((max_reqs,), -6, dtype=torch.int32),
            "query_slot": torch.full((self.max_tokens,), -99, dtype=torch.int64),
            "ctx_positions": torch.full((self.max_tokens,), -111, dtype=torch.int64),
            "ctx_slot": torch.full((self.max_tokens,), -222, dtype=torch.int64),
            "sample_indices": torch.full(
                (max_reqs * self.steps,), -333, dtype=torch.int64
            ),
            "sample_pos": torch.full((max_reqs * self.steps,), -444, dtype=torch.int64),
            "sample_idx_mapping": torch.full(
                (max_reqs * self.steps,), -555, dtype=idx_dtype
            ),
            "out_temperature": torch.full((max_reqs,), -1.5),
            "out_seeds": torch.full((max_reqs,), -666, dtype=torch.int64),
        }

    def reference(self) -> dict[str, torch.Tensor]:
        """The Triton kernel, program by program, in plain Python."""
        out = {k: v.clone() for k, v in self.fresh.items()}
        nq, steps, bs = self.nq, self.steps, self.block_size
        for req in range(self.num_reqs):
            state = int(self.idx_mapping[req])
            start, end = int(self.qsl[req]), int(self.qsl[req + 1])
            valid_end = end - int(self.rejected[req])
            num_valid = valid_end - start
            bonus = (
                int(self.last_sampled[state, 0])
                if int(self.num_sampled[req]) > 0
                else int(self.next_prefill[0, state])
            )
            last_pos = int(self.target_positions[valid_end - 1])
            for j in range(end - start):
                pos, slot = 0, PAD_SLOT
                if j < num_valid:
                    pos = int(self.target_positions[start + j])
                    block = int(self.block_table[req, min(pos // bs, self.stride - 1)])
                    if block != 0:
                        slot = block * bs + pos % bs
                out["ctx_positions"][start + j] = pos
                out["ctx_slot"][start + j] = slot
            for off in range(nq):
                qpos, idx = last_pos + 1 + off, req * nq + off
                block = int(self.block_table[req, min(qpos // bs, self.stride - 1)])
                out["input_ids"][idx] = bonus if off == 0 else self.parallel_token
                out["is_padding"][idx] = False
                out["positions"][idx] = min(qpos, self.max_model_len - 1)
                out["query_slot"][idx] = (
                    block * bs + qpos % bs if block != 0 else PAD_SLOT
                )
                first = 0 if self.anchor else 1
                if off >= first:
                    row = req * steps + off - first
                    out["sample_indices"][row] = idx
                    out["sample_pos"][row] = qpos + 1 if self.anchor else qpos
                    out["sample_idx_mapping"][row] = state
            out["query_start_loc"][req] = req * nq
            out["seq_lens"][req] = min(last_pos + 1 + nq, self.max_model_len)
            out["out_temperature"][state] = self.temperature[state]
            out["out_seeds"][state] = self.seeds[state]
        n, last_end = self.num_reqs, self.num_reqs * nq
        out["query_start_loc"][n : self.max_reqs + 1] = last_end
        out["seq_lens"][n : self.max_reqs] = 0
        out["sample_indices"][n * steps :] = 0
        out["sample_pos"][n * steps :] = 0
        out["sample_idx_mapping"][n * steps :] = -1
        out["input_ids"][last_end:] = 0
        out["positions"][last_end:] = 0
        out["is_padding"][last_end:] = True
        out["query_slot"][last_end:] = PAD_SLOT
        return out

    def run(self) -> dict[str, torch.Tensor]:
        ops = _ops()
        out = {k: _cuda(v.clone()) for k, v in self.fresh.items()}
        idt = self.idx_dtype
        ops.inkling2_dspark_prepare_inputs(
            out["input_ids"],
            out["positions"],
            out["is_padding"],
            out["query_start_loc"],
            out["seq_lens"],
            out["query_slot"],
            out["ctx_positions"],
            out["ctx_slot"],
            out["sample_indices"],
            out["sample_pos"],
            out["sample_idx_mapping"],
            out["out_temperature"],
            out["out_seeds"],
            _cuda(self.target_positions),
            _cuda(self.qsl),
            _cuda(self.idx_mapping.to(idt)),
            _cuda(self.last_sampled),
            _cuda(self.next_prefill),
            _cuda(self.num_sampled.to(idt)),
            _cuda(self.rejected.to(idt)),
            _cuda(self.temperature),
            _cuda(self.seeds),
            _cuda(self.block_table),
            self.block_size,
            self.parallel_token,
            self.nq,
            self.steps,
            self.max_reqs,
            self.max_tokens,
            self.max_model_len,
            self.num_reqs,
            int(self.lens.max()) + self.nq,
            self.anchor,
        )
        return {k: v.cpu() for k, v in out.items()}


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_reqs,max_reqs", [(3, 8), (5, 40), (1, 1), (6, 6)])
def test_prepare_inputs_matches_the_triton_kernel(
    num_reqs, max_reqs, idx_dtype, anchor
):
    """Every output buffer, element by element: written rows, inert padding past the
    batch, rejected suffixes (position 0, PAD slot), null blocks (PAD slot), the
    block-table clamp, the ``max_model_len`` clamps and the chunked-prefill bonus."""
    world = PrepWorld(
        1000 + 10 * num_reqs + int(anchor), num_reqs, max_reqs, anchor, idx_dtype
    )
    got, ref = world.run(), world.reference()
    for name in PREP_OUTPUTS:
        assert torch.equal(got[name], ref[name]), name


def test_prepare_inputs_reference_reaches_every_branch():
    """The comparison above can fail in every branch: a big batch makes the
    reference contain null-block PAD slots (context and query), rejected-suffix
    rows, clamped positions, the block-table clamp and the prompt-token bonus."""
    world = PrepWorld(5, 40, 40, False, torch.int32)
    ref = world.reference()
    nq = world.nq
    assert (ref["ctx_slot"][: int(world.qsl[-1])] == PAD_SLOT).any()
    assert (ref["query_slot"][: 40 * nq] == PAD_SLOT).any()
    assert (ref["query_slot"][: 40 * nq] != PAD_SLOT).any()
    assert int(ref["positions"][: 40 * nq].max()) == world.max_model_len - 1
    # a position past the block table's reach (stride * block_size) occurred
    assert int(world.target_positions.max()) >= world.stride * world.block_size
    assert int(world.rejected.max()) > 0 and int(world.rejected.min()) == 0
    # request 0 has no sampled token: its bonus is the next prompt token
    state = int(world.idx_mapping[0])
    assert int(ref["input_ids"][0]) == int(world.next_prefill[0, state])
    assert int(ref["input_ids"][1]) == world.parallel_token
    assert world.next_prefill[0, state] != world.next_prefill[1, state]


def test_prepare_inputs_spans_several_programs_for_a_long_context():
    """A request with 200 context rows needs two 128-column programs; its rejected
    suffix straddles the boundary between them."""
    world = PrepWorld(77, 4, 8, False, torch.int32, long_first=True)
    got, ref = world.run(), world.reference()
    assert int(world.lens[0]) == 200 and int(world.rejected[0]) == 100
    for name in PREP_OUTPUTS:
        assert torch.equal(got[name], ref[name]), name
