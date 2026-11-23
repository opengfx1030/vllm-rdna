# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini dense linears on RDNA2 (gfx1030): fp64 reference tests.

Covers ``inkling2_linear_grouped`` (fp16 W, GEMV for M <= 16 and the tiled GEMM
above), ``inkling2_linear_f32w`` (router, controller gates),
``inkling2_linear_lm_head`` and the argmax partial / reduce ops, over the model
shapes at TP = 1, 2, 4, M in {1, 2, 3, 4, 8, 9, 16, 64, 257, 1024}, K / N tails,
strided activations, every epilogue and every forced kernel variant.

The reference is fp64 on random fp16 data. fp16 outputs must match the fp16
rounding of the fp64 result to one ulp (plus an fp32 accumulation allowance of
5e-6 of the largest output); fp32 outputs to 5e-6 of the largest output.

Run ``pytest tests/kernels/quantization/test_inkling2_linear_rdna2.py``.
"""

import pytest
import torch

from vllm.models.inkling2_loop.amd import linear_pack as lp
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("Inkling-2 linear kernels are ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.platforms.rocm import on_gfx10x  # noqa: E402

_HAVE_OPS = (
    hasattr(torch.ops, "_rocm_C")
    and hasattr(torch.ops._rocm_C, "inkling2_linear_grouped")
    and hasattr(torch.ops._rocm_C, "inkling2_linear_argmax_reduce")
)
gfx1030_only = pytest.mark.skipif(
    not (on_gfx10x() and _HAVE_OPS),
    reason="Requires gfx1030 with the inkling2_linear ops",
)

H = 2304
HEAD_DIM = 128
D_REL = 16
CONV_K = 4
SOFTCAP = 30.0
MS = [1, 2, 3, 4, 8, 9, 16, 64, 257, 1024]
GEMV_VARIANTS = list(range(6))
GEMM_VARIANTS = list(range(100, 106))
DEV = "cuda"


_GEN: list[torch.Generator] = []


def _randn(*shape, dtype=torch.float16, scale=1.0):
    """Fresh values on every call (one advancing generator, reseeded per test)."""
    if not _GEN:
        _GEN.append(torch.Generator(device=DEV))
    return (torch.randn(*shape, device=DEV, generator=_GEN[0]) * scale).to(dtype)


@pytest.fixture(autouse=True)
def _reseed():
    if _GEN:
        _GEN[0].manual_seed(1234)
    yield


def _wscale(k):
    return 1.0 / (k**0.5)


def _ref_linear(x, w, act=0, p=1.0, n_valid=None):
    """fp64 act(x @ w^T) with rows of w processed in chunks."""
    xd = x.double()
    cols = []
    for lo in range(0, w.shape[0], 16384):
        cols.append(xd @ w[lo : lo + 16384].double().T)
    y = torch.cat(cols, dim=1)
    if act == lp.ACT_SIGMOID:
        y = torch.sigmoid(y)
    elif act == lp.ACT_TANH:
        y = torch.tanh(y)
    elif act == lp.ACT_SOFTCAP:
        y = torch.tanh(y / p) * p
    if n_valid is not None and n_valid < y.shape[1]:
        y[:, n_valid:] = float("-inf")
    return y


def _ulp16(r16):
    e = torch.floor(torch.log2(r16.abs().float().clamp_min(2.0**-14)))
    return torch.pow(2.0, e - 10).double()


def _check(out, ref, what="", pre_scale=None):
    """Out (fp16 or fp32) against the fp64 reference.

    ``pre_scale`` is the magnitude of the pre-activation accumulator when ``ref``
    went through an activation: the fp32 accumulation noise scales with it (a
    tanh near 0 passes it through 1:1), not with the bounded output.
    """
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out.double())), f"{what}: -inf mask"
    assert torch.equal(ref[~finite], out.double()[~finite]), (
        f"{what}: masked values differ"
    )
    if not finite.any():
        return
    r, o = ref[finite], out.double()[finite]
    scale = r.abs().max().item()
    if out.dtype == torch.float16:
        r16 = r.half()
        tol = _ulp16(r16) * 1.0001 + 5e-6 * scale
        err = (o - r16.double()).abs()
    else:
        tol = 5e-6 * scale if pre_scale is None else 1e-6 * max(scale, pre_scale)
        err = (o - r).abs()
    bad = err > tol
    assert not bad.any(), (
        f"{what}: {int(bad.sum())}/{r.numel()} elements out of tolerance, "
        f"max err {err.max().item():.3e} (scale {scale:.3e})"
    )


def _is_invalid_variant(e: Exception) -> bool:
    return "<= 32" in str(e) or "take M <= 16" in str(e)


def _run_variants(call, M, variants):
    """Auto dispatch plus every forced variant that applies to M."""
    call(-1)
    for v in variants:
        if v < 100 and M > 16:
            continue
        try:
            call(v)
        except RuntimeError as e:
            if _is_invalid_variant(e):
                continue
            raise


# --------------------------------------------------------------------------
# fp16 weights: single and grouped segments
# --------------------------------------------------------------------------
@gfx1030_only
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize(
    "N,K",
    [
        (3072, H),  # q / gate_attn
        (1024, H),  # k / v
        (384, H),  # rel
        (24, H),  # sink
        (2304, 3072),  # o_proj
        (64, 4608),  # controller down_proj
        (1536, H),  # q at TP=2
        (768, H),  # q at TP=4
        (2304, 1536),  # o_proj at TP=2
        (2304, 768),  # o_proj at TP=4
        (12, H),  # sink at TP=2
        (6, H),  # sink at TP=4
        (8, H),  # dyn taps at TP=4
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_single_segment(M, N, K, out_dtype):
    x = _randn(M, K)
    w = _randn(N, K, scale=_wscale(K))
    out = torch.empty(M, N, dtype=out_dtype, device=DEV)

    ref = _ref_linear(x, w)

    def call(v):
        out.fill_(7.0)
        ops.inkling2_linear_grouped([x], w, [out], [0], [N], variant=v)
        _check(out, ref, f"N={N} K={K} M={M} v={v}")

    _run_variants(call, M, GEMV_VARIANTS + GEMM_VARIANTS)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 4, 16])
@pytest.mark.parametrize("N,K", [(3072, H), (2304, 3072), (64, 4608)])
def test_gemv_blocks_per_cu(M, N, K):
    """Variant ``cfg + 10 * s`` only changes the size of the persistent grid."""
    x = _randn(M, K)
    w = _randn(N, K, scale=_wscale(K))
    out = torch.empty(M, N, dtype=torch.float16, device=DEV)

    ref = _ref_linear(x, w)
    for cfg in (0, 1, 4):
        for s in range(1, 6):
            out.fill_(7.0)
            ops.inkling2_linear_grouped([x], w, [out], [0], [N], variant=cfg + 10 * s)
            _check(out, ref, f"N={N} K={K} M={M} v={cfg + 10 * s}")


@gfx1030_only
@pytest.mark.parametrize("M", [1, 3, 9, 17, 64, 257])
@pytest.mark.parametrize(
    "N,K",
    [
        (1, 8),
        (7, 64),
        (130, 72),
        (257, 200),
        (4097, 1200),  # vision patch K, N tail
        (129, 2312),  # K % 64 != 0
        (33, 40),
        (300, 1000),
        (1000, 24),
    ],
)
@pytest.mark.parametrize("pad", [0, 16])
def test_tails(M, N, K, pad):
    """Pad > 0: NaN follows every activation row, so a read past K shows."""
    buf = torch.full((M, K + pad), float("nan"), dtype=torch.float16, device=DEV)
    buf[:, :K] = _randn(M, K)
    x = buf[:, :K]
    w = _randn(N, K, scale=_wscale(K))
    out = torch.empty(M, N, dtype=torch.float16, device=DEV)

    ref = _ref_linear(x, w)

    def call(v):
        out.fill_(7.0)
        ops.inkling2_linear_grouped([x], w, [out], [0], [N], variant=v)
        _check(out, ref, f"N={N} K={K} M={M} pad={pad} v={v}")

    _run_variants(call, M, GEMV_VARIANTS + GEMM_VARIANTS)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 4, 9, 64, 257])
@pytest.mark.parametrize("pad,off", [(64, 0), (8, 0), (0, 8), (136, 24)])
def test_strided_activations(M, pad, off):
    """Row stride != K, base offset not a multiple of the row stride."""
    K, N = H, 1030
    buf = _randn(M, K + pad + off)
    x = buf[:, off : off + K]
    assert x.stride(0) == K + pad + off and x.data_ptr() % 16 == 0
    w = _randn(N, K, scale=_wscale(K))
    out = torch.empty(M, N, dtype=torch.float16, device=DEV)

    ref = _ref_linear(x, w)

    def call(v):
        out.fill_(7.0)
        ops.inkling2_linear_grouped([x], w, [out], [0], [N], variant=v)
        _check(out, ref, f"strided M={M} v={v}")

    _run_variants(call, M, GEMV_VARIANTS + GEMM_VARIANTS)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 4, 17, 130])
def test_strided_outputs(M):
    """Outputs written into column slices of wider buffers."""
    K, N = 512, 200
    x = _randn(M, K)
    w = _randn(N, K, scale=_wscale(K))
    for dt in (torch.float16, torch.float32):
        big = torch.full((M, N + 40), 3.0, dtype=dt, device=DEV)
        ops.inkling2_linear_grouped([x], w, [big[:, 16 : 16 + N]], [0], [N])
        _check(big[:, 16 : 16 + N], _ref_linear(x, w), f"strided out {dt} M={M}")
        assert torch.all(big[:, :16] == 3.0) and torch.all(big[:, 16 + N :] == 3.0)


@gfx1030_only
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("dyn", [False, True])
@pytest.mark.parametrize("same_input", [False, True])
def test_attn_in_pack(M, tp, dyn, same_input):
    """q|gate|rel|sink / k|dyn_k / v|dyn_v in one launch, per-stream inputs."""
    nh, nkv = 24 // tp, 8 // tp
    shapes = {
        "q": nh * HEAD_DIM,
        "gate": nh * HEAD_DIM,
        "rel": nh * D_REL,
        "sink": nh,
        "k": nkv * HEAD_DIM,
        "v": nkv * HEAD_DIM,
    }
    ws = {n: _randn(r, H, scale=_wscale(H)) for n, r in shapes.items()}
    dk = _randn(nkv * CONV_K, H, scale=_wscale(H)) if dyn else None
    dv = _randn(nkv * CONV_K, H, scale=_wscale(H)) if dyn else None
    pack = lp.inkling2_linear_pack_attn_in(
        ws["q"], ws["gate"], ws["rel"], ws["sink"], ws["k"], ws["v"], dk, dv
    )
    hq = _randn(M, H)
    xs = [hq, hq, hq] if same_input else [hq, _randn(M, H), _randn(M, H)]
    outs = pack.alloc_outputs(M)
    names = list(pack.names)
    ref_w = {"q": ws["q"], "gate": ws["gate"], "rel": ws["rel"], "sink": ws["sink"]}
    ref_w.update({"k": ws["k"], "v": ws["v"], "dyn_k": dk, "dyn_v": dv})
    refs = [_ref_linear(xs[sx], ref_w[n]) for n, sx in zip(names, pack.seg_x)]

    def call(v):
        for o in outs:
            o.fill_(7.0)
        pack(xs, outs, variant=v)
        for name, o, ref in zip(names, outs, refs):
            _check(o, ref, f"{name} M={M} tp={tp} v={v}")

    _run_variants(call, M, [0, 1, 3, 4, 5] + GEMM_VARIANTS[:3])


# --------------------------------------------------------------------------
# epilogues, -inf mask, no allocation
# --------------------------------------------------------------------------
@gfx1030_only
@pytest.mark.parametrize("M", [1, 5, 16, 33, 300])
@pytest.mark.parametrize("act", [lp.ACT_SIGMOID, lp.ACT_TANH, lp.ACT_SOFTCAP])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_epilogues_and_mask(M, act, out_dtype):
    K, N, nv = 640, 777, 700
    x = _randn(M, K, scale=2.0)
    w = _randn(N, K, scale=_wscale(K) * 4.0)
    out = torch.empty(M, N, dtype=out_dtype, device=DEV)
    p = 12.5
    ref = _ref_linear(x, w, act, p, nv)
    pre_scale = _ref_linear(x, w).abs().max().item()

    def call(v):
        out.fill_(7.0)
        ops.inkling2_linear_grouped(
            [x], w, [out], [0], [N], [act], [p], [nv], variant=v
        )
        _check(out, ref, f"act={act} M={M} v={v}", pre_scale)

    _run_variants(call, M, [0, 1, 4] + GEMM_VARIANTS)


@gfx1030_only
def test_no_allocation():
    M = 4
    x = _randn(M, H)
    w = _randn(3072, H, scale=_wscale(H))
    out = torch.empty(M, 3072, dtype=torch.float16, device=DEV)
    ops.inkling2_linear_grouped([x], w, [out], [0], [3072])
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_stats()["allocated_bytes.all.allocated"]
    for _ in range(2):
        ops.inkling2_linear_grouped([x], w, [out], [0], [3072])
    torch.accelerator.synchronize()
    after = torch.accelerator.memory_stats()["allocated_bytes.all.allocated"]
    assert before == after, "the op allocated device memory"


@gfx1030_only
def test_rejects_bad_arguments():
    x = _randn(4, 64)
    w = _randn(32, 64)
    out = torch.empty(4, 32, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError):
        ops.inkling2_linear_grouped([x], w[:, :60], [out], [0], [32])
    with pytest.raises(RuntimeError):
        ops.inkling2_linear_grouped([x], w, [out], [0], [33])
    with pytest.raises(RuntimeError):
        ops.inkling2_linear_grouped([x[:, :63]], w, [out], [0], [32])
    with pytest.raises(RuntimeError):
        ops.inkling2_linear_grouped([x], w, [out.float()], [0], [32], seg_act=[9])


# --------------------------------------------------------------------------
# fp32 weights (router, controller gates)
# --------------------------------------------------------------------------
@gfx1030_only
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("E,K", [(128, H), (48, 64), (130, 72), (5, 8)])
def test_f32w_router(M, E, K):
    x = _randn(M, K)
    w = _randn(E, K, dtype=torch.float32, scale=_wscale(K))
    pack = lp.inkling2_linear_pack_router(w)
    out = pack.alloc_outputs(M)

    ref = _ref_linear(x, w)

    def call(v):
        out[0].fill_(7.0)
        pack([x], out, variant=v)
        _check(out[0], ref, f"router E={E} K={K} M={M} v={v}")

    _run_variants(call, M, [0, 1, 2, 4, 100])


@gfx1030_only
@pytest.mark.parametrize("M", MS)
def test_f32w_controller(M):
    rank = 64
    gh = _randn(M, rank)
    wf = _randn(H, rank, dtype=torch.float32, scale=_wscale(rank))
    wg = _randn(H, rank, dtype=torch.float32, scale=_wscale(rank))
    pack = lp.inkling2_linear_pack_controller(wf, wg)
    outs = pack.alloc_outputs(M)

    rf = _ref_linear(gh, wf, lp.ACT_TANH)
    rg = _ref_linear(gh, wg, lp.ACT_TANH)

    def call(v):
        for o in outs:
            o.fill_(7.0)
        pack([gh], outs, variant=v)
        _check(outs[0], rf, f"forget M={M} v={v}")
        _check(outs[1], rg, f"gain M={M} v={v}")

    _run_variants(call, M, [0, 1, 2, 4, 100])


# --------------------------------------------------------------------------
# lm_head: softcap, vocab shards, argmax partials
# --------------------------------------------------------------------------
LM_CASES = [
    # (vocab, tp, rank)
    (200058, 1, 0),
    (200058, 2, 0),
    (200058, 2, 1),
    (200058, 4, 3),
    (5000, 4, 3),
]


@gfx1030_only
@pytest.mark.parametrize("M", [1, 2, 3, 4, 8, 9, 16, 17, 64])
@pytest.mark.parametrize("vocab,tp,rank", LM_CASES)
def test_lm_head(M, vocab, tp, rank):
    full = _randn(vocab, H, scale=_wscale(H))
    head = lp.inkling2_linear_pack_lm_head(full, vocab, tp, rank)
    x = _randn(M, H)
    logits = torch.empty(M, head.rows, dtype=torch.float32, device=DEV)
    ref = _ref_linear(x, head.weight, lp.ACT_SOFTCAP, SOFTCAP, head.n_valid)
    nparts = 61
    pv = torch.empty(M, nparts, dtype=torch.float32, device=DEV)
    pi = torch.empty(M, nparts, dtype=torch.int32, device=DEV)
    am = torch.empty(M, dtype=torch.int64, device=DEV)
    want = ref.argmax(dim=1) + head.col_base

    for v in [-1] + ([0, 1] if M <= 16 else GEMM_VARIANTS[:2]):
        logits.fill_(7.0)
        head(x, logits, pv, pi, SOFTCAP, variant=v)
        _check(logits, ref, f"lm_head M={M} tp={tp} v={v}")
        ops.inkling2_linear_argmax_reduce(pv, pi, am)
        assert torch.equal(am, want), f"argmax M={M} tp={tp} v={v}"
        if M <= 16 and v in (-1, 0, 1):
            pv.fill_(0)
            pi.fill_(-1)
            head(x, None, pv, pi, SOFTCAP, variant=v)  # partials only
            ops.inkling2_linear_argmax_reduce(pv, pi, am)
            assert torch.equal(am, want), f"fused argmax M={M} tp={tp} v={v}"


@gfx1030_only
def test_lm_head_no_softcap():
    M = 3
    full = _randn(1000, H, scale=_wscale(H))
    x = _randn(M, H)
    logits = torch.empty(M, 1000, dtype=torch.float32, device=DEV)
    ops.inkling2_linear_lm_head(x, full, logits, None, None, 0.0, 1000, 0)
    _check(logits, _ref_linear(x, full), "lm_head plain")


@gfx1030_only
@pytest.mark.parametrize("M", [1, 3, 16])
def test_lm_head_argmax_ties(M):
    """Adjacent duplicated rows tie bit-exactly: the lowest vocab index wins."""
    half = _randn(3001, H, scale=_wscale(H))
    full = half.repeat_interleave(2, dim=0)
    head = lp.inkling2_linear_pack_lm_head(full, full.shape[0])
    x = _randn(M, H)
    ref = _ref_linear(x, full, lp.ACT_SOFTCAP, SOFTCAP)
    want = ref[:, ::2].argmax(dim=1) * 2
    pv = torch.empty(M, 29, dtype=torch.float32, device=DEV)
    pi = torch.empty(M, 29, dtype=torch.int32, device=DEV)
    am = torch.empty(M, dtype=torch.int64, device=DEV)
    for v in (-1, 0, 1, 3, 4):
        head(x, None, pv, pi, SOFTCAP, variant=v)
        ops.inkling2_linear_argmax_reduce(pv, pi, am)
        assert torch.equal(am, want), f"fused ties M={M} v={v}"


@gfx1030_only
@pytest.mark.parametrize("M", [1, 7, 64])
@pytest.mark.parametrize("n", [100, 4096, 200058])
def test_argmax_partial_reduce(M, n):
    g = torch.Generator(device=DEV).manual_seed(5)
    logits = torch.randn(M, n + 8, device=DEV, generator=g)
    logits[0, n // 2] = 10.0  # unique max
    if M > 1:
        logits[1, 3] = 10.0  # ties: the lowest index must win
        logits[1, n - 1] = 10.0
        logits[1, 5] = float("nan")  # NaN counts as -inf
    for parts in (1, 7, 64):
        pv = torch.empty(M, parts, dtype=torch.float32, device=DEV)
        pi = torch.empty(M, parts, dtype=torch.int32, device=DEV)
        am = torch.empty(M, dtype=torch.int64, device=DEV)
        ops.inkling2_linear_argmax_partial(logits, pv, pi, n, 1000)
        ops.inkling2_linear_argmax_reduce(pv, pi, am)
        clean = torch.where(torch.isnan(logits[:, :n]), -torch.inf, logits[:, :n])
        assert torch.equal(am, clean.argmax(dim=1) + 1000), f"M={M} n={n} P={parts}"


@gfx1030_only
@pytest.mark.parametrize("n,parts", [(4096, 1), (4096, 7), (200058, 64)])
def test_argmax_partial_ties(n, parts):
    """Equal maxima 256 columns apart share a thread: the lowest index wins."""
    g = torch.Generator(device=DEV).manual_seed(11)
    logits = torch.randn(4, n, device=DEV, generator=g).clamp_(max=4.0)
    c = min(2, parts - 1) * -(-n // parts) + 5
    ties = [[c, c + 256], [c + 512, c + 256], [c, c + 256, c + 512]]
    ties.append([c + 257, c + 256, c + 512])
    for row, cols in enumerate(ties):
        logits[row, cols] = 9.0
    want = torch.tensor([c, c + 256, c, c + 256], device=DEV) + 1000
    pv = torch.empty(4, parts, dtype=torch.float32, device=DEV)
    pi = torch.empty(4, parts, dtype=torch.int32, device=DEV)
    am = torch.empty(4, dtype=torch.int64, device=DEV)
    ops.inkling2_linear_argmax_partial(logits, pv, pi, n, 1000)
    ops.inkling2_linear_argmax_reduce(pv, pi, am)
    assert torch.equal(am, want)


@gfx1030_only
@pytest.mark.parametrize("nparts", [2, 64, 300, 1000])
def test_argmax_reduce_ties(nparts):
    """Equal partial maxima: the lowest vocab index wins wherever it sits."""
    g = torch.Generator(device=DEV).manual_seed(13)
    pv = torch.randn(4, nparts, device=DEV, generator=g).clamp_(max=4.0)
    pi = (torch.arange(nparts, device=DEV, dtype=torch.int32) * 7 + 100).repeat(4, 1)
    last = nparts - 1
    ties = [[0, 1], [0, last], [1, min(257, last)], [0, 1, min(256, last)]]
    for row, cols in enumerate(ties):
        cols = sorted(set(cols))
        pv[row, cols] = 9.0
        # the later part holds the smaller index, so the lowest index != first part
        pi[row, cols] = 5000 - 3 * torch.arange(
            len(cols), device=DEV, dtype=torch.int32
        )
    am = torch.empty(4, dtype=torch.int64, device=DEV)
    ops.inkling2_linear_argmax_reduce(pv, pi, am)
    mx = pv.max(dim=1, keepdim=True).values
    want = torch.where(pv == mx, pi, 2**31 - 1).amin(dim=1).long()
    assert torch.equal(am, want)


# --------------------------------------------------------------------------
# graph capture: static pointers, no host sync
# --------------------------------------------------------------------------
@gfx1030_only
@pytest.mark.parametrize("M", [1, 4, 64])
def test_cuda_graph_replay(M):
    nh, nkv = 24, 8
    ws = [
        _randn(r, H, scale=_wscale(H))
        for r in (nh * HEAD_DIM, nh * HEAD_DIM, nh * D_REL, nh, nkv * HEAD_DIM)
    ]
    wv = _randn(nkv * HEAD_DIM, H, scale=_wscale(H))
    pack = lp.inkling2_linear_pack_attn_in(ws[0], ws[1], ws[2], ws[3], ws[4], wv)
    head = lp.inkling2_linear_pack_lm_head(_randn(3000, H, scale=0.02), 3000)
    xs = [_randn(M, H) for _ in range(3)]
    outs = pack.alloc_outputs(M)
    logits = torch.empty(M, 3000, dtype=torch.float32, device=DEV)
    pv = torch.empty(M, 16, dtype=torch.float32, device=DEV)
    pi = torch.empty(M, 16, dtype=torch.int32, device=DEV)
    am = torch.empty(M, dtype=torch.int64, device=DEV)
    xh = _randn(M, H)

    def work():
        pack(xs, outs)
        head(xh, logits, pv, pi)
        ops.inkling2_linear_argmax_reduce(pv, pi, am)

    work()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work()
    for x in xs + [xh]:
        x.copy_(_randn(M, H))
    for o in outs + [logits]:
        o.fill_(7.0)
    graph.replay()
    torch.accelerator.synchronize()
    names = list(pack.names)
    wmap = dict(zip(names, ws + [wv]))
    for name, o, sx in zip(names, outs, pack.seg_x):
        _check(o, _ref_linear(xs[sx], wmap[name]), f"graph {name}")
    ref = _ref_linear(xh, head.weight, lp.ACT_SOFTCAP, SOFTCAP)
    _check(logits, ref, "graph lm_head")
    assert torch.equal(am, ref.argmax(dim=1))


# --------------------------------------------------------------------------
# packing helpers (no GPU kernels)
# --------------------------------------------------------------------------
def test_pack_attn_in_order_and_shapes():
    g = torch.Generator().manual_seed(0)

    def mk(r):
        return torch.randn(r, 16, generator=g).half()

    q, gate, rel, sink, k, v, dk, dv = (mk(r) for r in (48, 48, 6, 3, 16, 16, 8, 8))
    pack = lp.inkling2_linear_pack_attn_in(q, gate, rel, sink, k, v, dk, dv)
    assert pack.names == ("q", "gate", "rel", "sink", "k", "dyn_k", "v", "dyn_v")
    assert pack.seg_n == (48, 48, 6, 3, 16, 8, 16, 8)
    assert pack.seg_x == (0, 0, 0, 0, 1, 1, 2, 2)
    assert torch.equal(pack.weight, torch.cat([q, gate, rel, sink, k, dk, v, dv]))
    assert all(dt == torch.float16 for dt in pack.out_dtypes)
    plain = lp.inkling2_linear_pack_attn_in(q, gate, rel, sink, k, v)
    assert plain.names == ("q", "gate", "rel", "sink", "k", "v")
    with pytest.raises(ValueError):
        lp.inkling2_linear_pack_attn_in(q, gate, rel, sink, k, v, dk, None)
    with pytest.raises(ValueError):
        lp.inkling2_linear_pack_attn_in(q.float(), gate, rel, sink, k, v)


@pytest.mark.parametrize(
    "tp,rank,rows,n_valid,base",
    [
        (1, 0, 200058, 200058, 0),
        (2, 0, 100032, 100032, 0),
        (2, 1, 100032, 100026, 100032),
        (4, 0, 50016, 50016, 0),
        (4, 3, 50016, 50010, 150048),
    ],
)
def test_pack_lm_head_shards(tp, rank, rows, n_valid, base):
    vocab = 200058
    w = torch.arange(201024, dtype=torch.float32)[:, None].expand(201024, 8).half()
    pk = lp.inkling2_linear_pack_lm_head(w, vocab, tp, rank)
    assert (pk.rows, pk.n_valid, pk.col_base) == (rows, n_valid, base)
    assert torch.equal(pk.weight[: pk.n_valid], w[base : base + pk.n_valid])
    assert not pk.weight[pk.n_valid :].any()
