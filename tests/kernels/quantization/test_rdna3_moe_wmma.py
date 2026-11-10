# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests of ``moe_gptq_gemm_rdna3_wmma`` (WMMA grouped W4A16 GEMM, gfx1100).

Skipped everywhere but on a gfx1100 device whose ``_rocm_C`` carries the op, and
collectable on any machine. The reference is a dense dequantization of the SAME
packed tensors the kernel reads: the packed words and zero words are random bit
patterns (every nibble value and every zero 1..16 occurs), decoded with the
documented format (``dequantize_expert`` of the CPU model test, which checks
that format against the GPTQ packing) and rounded ONCE to the activation dtype,
as the kernel feeds the matrix unit; products and sums are formed in fp32.

Tolerances follow the error structure of the kernel: the direct store is one
fp32 chain of K / 16 WMMA steps followed by one rounding to the output dtype, so
it must agree with the reference to the output rounding plus fp32 accumulation
noise measured against the sum of the absolute terms.
"""

from dataclasses import dataclass
from functools import lru_cache

import pytest
import torch

from tests.kernels.quantization.test_rdna3_moe_wmma_model import (
    BLOCK_MS,
    Q32_RANGE,
    Routing,
    align_block_size,
    dequantize_expert,
    to_int32,
)

DEVICE = "cuda"
DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]
BM_PARAMS = [pytest.param(bm, id=f"bm{bm}") for bm in BLOCK_MS]
TOP_K = 10
# fp32 accumulation noise, relative to the sum of the absolute terms of a value
# (K <= 2560 products: a random-walk error of a few 1e-6, bounded far below).
ACC_TOL = 5e-5


def _op_available() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1100

    return (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3_wmma")
    )


pytestmark = pytest.mark.skipif(
    not _op_available(), reason="needs gfx1100 and the moe_gptq_gemm_rdna3_wmma op"
)


def _ops():
    from vllm import _custom_ops

    return _custom_ops


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


@dataclass
class Packed:
    b_q: torch.Tensor  # [E, K/8, N] int32
    b_z: torch.Tensor  # [E, groups, N/8] int32
    scales: torch.Tensor  # [E, groups, N]


@lru_cache(maxsize=6)
def make_packed(E, K, N, group_size, dtype, seed=0) -> Packed:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    words = torch.randint(
        0, 1 << 32, (E, K // 8, N), generator=gen, device=DEVICE, dtype=torch.int64
    )
    zwords = torch.randint(
        0,
        1 << 32,
        (E, K // group_size, N // 8),
        generator=gen,
        device=DEVICE,
        dtype=torch.int64,
    )
    scales = torch.rand(E, K // group_size, N, generator=gen, device=DEVICE)
    return Packed(
        b_q=to_int32(words),
        b_z=to_int32(zwords),
        scales=(scales * 0.09 + 0.01).to(dtype),
    )


def make_topk_ids(M, top_k, E, seed, style="uniform") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if style == "uniform":
        return torch.randint(0, E, (M, top_k), generator=gen)
    if style == "skewed":  # ragged counts: low expert ids are hot, high ones empty
        return (torch.rand(M, top_k, generator=gen) ** 3 * E).long().clamp(max=E - 1)
    if style == "one_expert":
        return torch.zeros(M, top_k, dtype=torch.int64)
    if style == "sparse":  # two experts only
        return torch.randint(0, 2, (M, top_k), generator=gen)
    raise ValueError(style)


def routing_to_device(r: Routing) -> Routing:
    return Routing(
        sorted_ids=r.sorted_ids.to(DEVICE),
        expert_ids=r.expert_ids.to(DEVICE),
        ntpp=r.ntpp.to(DEVICE),
        block_m=r.block_m,
        num_pairs=r.num_pairs,
    )


def pad_routing(r: Routing, blocks: int) -> Routing:
    """Same routing in a larger capacity: sentinel ids, expert -1 (graph tests
    need identical shapes for every routing they replay)."""
    bm = r.block_m
    extra = blocks - r.expert_ids.numel()
    assert extra >= 0
    return Routing(
        sorted_ids=torch.cat(
            [r.sorted_ids, torch.full((extra * bm,), r.num_pairs, dtype=torch.int32)]
        ),
        expert_ids=torch.cat(
            [r.expert_ids, torch.full((extra,), -1, dtype=torch.int32)]
        ),
        ntpp=r.ntpp,
        block_m=bm,
        num_pairs=r.num_pairs,
    )


def scheduled_pairs(r: Routing, top_k: int, size_m: int, num_experts: int):
    """(pair ids, expert ids) of every row the op contract defines: blocks that
    are neither past num_tokens_post_padded nor expert -1, minus padding."""
    bm = r.block_m
    blocks = r.expert_ids.numel()
    ids = r.sorted_ids.view(blocks, bm).long()
    experts = r.expert_ids.long()
    live = (
        (torch.arange(blocks) * bm < int(r.ntpp[0]))
        & (experts >= 0)
        & (experts < num_experts)
    )
    valid = (ids >= 0) & (ids // top_k < size_m) & live[:, None]
    return ids[valid], experts[:, None].expand(blocks, bm)[valid]


def dense_reference(a, pk: Packed, pair, expert, top_k, topk_w, dtype):
    """fp32 [S, N] pair values and the [S, N] sums of |terms| of every scheduled
    pair (in ``pair`` order), from once-rounded dequantized weights."""
    N = pk.b_q.shape[2]
    values = torch.empty(pair.numel(), N, dtype=torch.float32, device=DEVICE)
    bound = torch.empty_like(values)
    pair_d, expert_d = pair.to(DEVICE), expert.to(DEVICE)
    for e in expert.unique().tolist():
        sel = (expert_d == e).nonzero().flatten()
        act = a[pair_d[sel] // top_k].float()
        w = dequantize_expert(pk.b_q, pk.b_z, pk.scales, e).to(dtype).float()
        values[sel] = act @ w
        bound[sel] = act.abs().nan_to_num(posinf=0.0) @ w.abs()
    if topk_w is not None:
        weight = topk_w[pair_d][:, None]
        values *= weight
        bound *= weight.abs()
    return values, bound


def output_rtol(dtype) -> float:
    return 2.0**-10 if dtype == torch.float16 else 2.0**-7


def assert_close_to_reference(got, values, bound, dtype, what="output"):
    err = (got.float() - values).abs()
    tol = output_rtol(dtype) * values.abs() + ACC_TOL * bound + 1e-6
    bad = ~(err <= tol)  # NaN-safe
    assert not bool(bad.any()), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements differ, max error "
        f"{err.nan_to_num(nan=float('inf')).max().item():.3e}"
    )


@dataclass
class Problem:
    a: torch.Tensor
    pk: Packed
    topk_ids: torch.Tensor
    topk_w: torch.Tensor  # [pairs] fp32
    top_k: int  # of the routing; the op's top_k is 1 for the down projection


def make_problem(
    E,
    K,
    N,
    M,
    *,
    dtype,
    top_k=TOP_K,
    group_size=128,
    seed=0,
    style="uniform",
    rows=None,
):
    """``rows``: rows of ``a`` (M for the gate/up layout, M * top_k for the
    down layout)."""
    pk = make_packed(E, K, N, group_size, dtype, seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    a = torch.randn(rows or M, K, generator=gen, device=DEVICE).to(dtype)
    topk_ids = make_topk_ids(M, top_k, E, seed + 2, style)
    topk_w = torch.rand(M * top_k, generator=gen, device=DEVICE) + 0.25
    return Problem(a, pk, topk_ids, topk_w, top_k)


def call_wmma(p: Problem, c, routing: Routing, *, op_top_k, bm, mul, output_topk=0):
    _ops().moe_gptq_gemm_rdna3_wmma(
        p.a,
        c,
        p.pk.b_q,
        p.pk.scales,
        p.pk.b_z,
        p.topk_w if mul else torch.empty(0, dtype=torch.float32, device=DEVICE),
        routing.sorted_ids,
        routing.expert_ids,
        routing.ntpp,
        op_top_k,
        bm,
        mul,
        output_topk,
    )


def call_scalar(p: Problem, c, routing: Routing, *, op_top_k, bm, mul, output_topk):
    _ops().moe_gptq_gemm_rdna3(
        p.a,
        c,
        p.pk.b_q,
        p.pk.scales,
        p.pk.b_z,
        p.topk_w if mul else torch.empty(0, dtype=torch.float32, device=DEVICE),
        routing.sorted_ids,
        routing.expert_ids,
        routing.ntpp,
        op_top_k,
        bm,
        mul,
        output_topk,
    )


def poisoned_output(pairs, N, dtype):
    return torch.full((pairs, N), float("nan"), dtype=dtype, device=DEVICE)


def run_direct(
    E, K, N, M, bm, dtype, *, mul=True, style="uniform", drop=(), stale=0, seed=0
):
    """One direct-store call on NaN-poisoned output against the dense
    reference; returns (problem, routing, c, scheduled pair ids)."""
    p = make_problem(E, K, N, M, dtype=dtype, seed=seed, style=style)
    routing = align_block_size(p.topk_ids, E, bm, drop_experts=drop, stale_blocks=stale)
    r_dev = routing_to_device(routing)
    c = poisoned_output(M * TOP_K, N, dtype)
    call_wmma(p, c, r_dev, op_top_k=TOP_K, bm=bm, mul=mul)
    pair, expert = scheduled_pairs(routing, TOP_K, M, E)
    values, bound = dense_reference(
        p.a, p.pk, pair, expert, TOP_K, p.topk_w if mul else None, dtype
    )
    pair_d = pair.to(DEVICE)
    assert_close_to_reference(c[pair_d], values, bound, dtype)
    untouched = torch.ones(M * TOP_K, dtype=torch.bool, device=DEVICE)
    untouched[pair_d] = False
    assert bool(torch.isnan(c[untouched]).all()), "a non-scheduled row was written"
    return p, routing, c, pair


# ---------------------------------------------------------------------------
# Direct store (output_topk == 0): fp16 and bf16
# ---------------------------------------------------------------------------

DIRECT_CASES = [
    # E, K, N, M
    pytest.param(8, 128, 16, 16, id="e8-k128-n16-m16"),
    pytest.param(8, 128, 16, 255, id="e8-k128-n16-m255"),
    pytest.param(8, 2560, 1280, 16, id="e8-gate_up-m16"),
    pytest.param(8, 2560, 1280, 17, id="e8-gate_up-m17"),
    pytest.param(8, 2560, 1280, 48, id="e8-gate_up-m48"),
    pytest.param(8, 2560, 1280, 255, id="e8-gate_up-m255"),
    pytest.param(8, 2560, 1280, 2048, id="e8-gate_up-m2048"),
    pytest.param(8, 640, 2560, 16, id="e8-down-m16"),
    pytest.param(8, 640, 2560, 255, id="e8-down-m255"),
    pytest.param(8, 640, 2560, 2048, id="e8-down-m2048"),
    pytest.param(64, 2560, 1280, 48, id="e64-gate_up-m48"),
    pytest.param(64, 2560, 1280, 255, id="e64-gate_up-m255"),
    pytest.param(64, 2560, 1280, 4096, id="e64-gate_up-m4096"),
    pytest.param(64, 640, 2560, 17, id="e64-down-m17"),
    pytest.param(64, 640, 2560, 2048, id="e64-down-m2048"),
    pytest.param(64, 640, 2560, 4096, id="e64-down-m4096"),
]


@pytest.mark.parametrize("E,K,N,M", DIRECT_CASES)
@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_direct_store_matches_the_dense_reference(E, K, N, M, bm, dtype):
    run_direct(E, K, N, M, bm, dtype, seed=M)


@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("K,N", [(2560, 1280), (640, 2560), (128, 16)])
def test_plain_store_without_the_topk_weight(K, N, bm, dtype):
    run_direct(8, K, N, 255, bm, dtype, mul=False, seed=3)


RAGGED = [
    pytest.param(dict(style="skewed"), id="skewed"),
    pytest.param(dict(style="one_expert"), id="one_expert_gets_all_pairs"),
    pytest.param(dict(style="sparse"), id="empty_experts"),
    pytest.param(dict(drop=(1, 5)), id="expert_minus_one_blocks"),
    pytest.param(dict(stale=4), id="stale_capacity_past_ntpp"),
    pytest.param(dict(style="skewed", drop=(0,), stale=2), id="mixed"),
]


@pytest.mark.parametrize("routing", RAGGED)
@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ragged_routing_and_n_tail(routing, bm, dtype):
    """N = 272 is not a multiple of any tile width: the last tile has a single
    live fragment."""
    run_direct(16, 640, 272, 255, bm, dtype, seed=5, **routing)


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_one_token_and_padding_only_blocks(bm):
    for M in (1, 2):
        run_direct(64, 640, 2560, M, bm, torch.float16, seed=M)


# ---------------------------------------------------------------------------
# Determinism, tile independence, special values
# ---------------------------------------------------------------------------


def _bits(t):
    return t.view(torch.int16) if t.dtype != torch.int64 else t


@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_direct_store_is_bit_identical_over_50_repeats(bm, dtype):
    E, K, N, M = 16, 640, 2560, 48
    p = make_problem(E, K, N, M, dtype=dtype, seed=11)
    routing = routing_to_device(align_block_size(p.topk_ids, E, bm))
    first = None
    for _ in range(50):
        c = poisoned_output(M * TOP_K, N, dtype)
        call_wmma(p, c, routing, op_top_k=TOP_K, bm=bm, mul=True)
        if first is None:
            first = c
        else:
            assert torch.equal(_bits(c), _bits(first))


@pytest.mark.parametrize("dtype", DTYPES)
def test_direct_store_is_bitwise_independent_of_the_block_size(dtype):
    """Each output element is the same fp32 chain of K / 16 WMMA steps whatever
    the tile shape and the padding of the routing."""
    E, K, N, M = 16, 2560, 1280, 255
    p = make_problem(E, K, N, M, dtype=dtype, seed=13)
    results = {}
    for bm in BLOCK_MS:
        routing = align_block_size(p.topk_ids, E, bm)
        c = poisoned_output(M * TOP_K, N, dtype)
        call_wmma(p, c, routing_to_device(routing), op_top_k=TOP_K, bm=bm, mul=True)
        results[bm] = c
    for bm in BLOCK_MS[1:]:
        assert torch.equal(_bits(results[bm]), _bits(results[BLOCK_MS[0]]))


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_non_finite_activation_poisons_only_its_own_rows(bm):
    E, K, N, M, bad = 16, 640, 2560, 64, 21
    dtype = torch.float16
    p = make_problem(E, K, N, M, dtype=dtype, seed=17)
    p.a[bad, 7] = float("nan")
    routing = align_block_size(p.topk_ids, E, bm)
    c = poisoned_output(M * TOP_K, N, dtype)
    call_wmma(p, c, routing_to_device(routing), op_top_k=TOP_K, bm=bm, mul=True)
    pair, expert = scheduled_pairs(routing, TOP_K, M, E)
    values, bound = dense_reference(p.a, p.pk, pair, expert, TOP_K, p.topk_w, dtype)
    token_pairs = pair.to(DEVICE) // TOP_K == bad
    got = c[pair.to(DEVICE)]
    assert bool(torch.isnan(got[token_pairs]).all())
    assert bool(torch.isfinite(got[~token_pairs]).all())
    assert_close_to_reference(
        got[~token_pairs], values[~token_pairs], bound[~token_pairs], dtype
    )


# ---------------------------------------------------------------------------
# Q32 (output_topk > 0): the down projection with the top-k reduction
# ---------------------------------------------------------------------------

Q32_CASES = [
    pytest.param(8, 640, 2560, 16, id="e8-down-m16"),
    pytest.param(8, 640, 2560, 48, id="e8-down-m48"),
    pytest.param(8, 640, 2560, 255, id="e8-down-m255"),
    pytest.param(8, 640, 2560, 2048, id="e8-down-m2048"),
    pytest.param(64, 640, 2560, 255, id="e64-down-m255"),
    pytest.param(64, 640, 2560, 2048, id="e64-down-m2048"),
    pytest.param(8, 2560, 1280, 48, id="e8-gate_up-m48"),
    pytest.param(64, 2560, 1280, 255, id="e64-gate_up-m255"),
    pytest.param(8, 128, 16, 255, id="e8-k128-n16-m255"),
]


def q32_problem(E, K, N, M, seed=0):
    """The down-projection layout: a is one fp16 row per (token, expert) pair,
    the op's top_k is 1 and the pairs of one token reduce into one row."""
    return make_problem(
        E, K, N, M, dtype=torch.float16, seed=seed, rows=M * TOP_K, group_size=128
    )


def finish_q32(accumulator):
    out = torch.empty(
        accumulator.shape[0],
        accumulator.shape[1] - 1,
        dtype=torch.float16,
        device=DEVICE,
    )
    _ops().moe_fixedpoint_to_half_rdna3(accumulator, out)
    return out


def q32_reference(p: Problem, routing: Routing, E, M):
    """fp32 [M, N] reduced rows and sum of |terms| from the dense reference."""
    pair, expert = scheduled_pairs(routing, 1, M * TOP_K, E)
    values, bound = dense_reference(p.a, p.pk, pair, expert, 1, p.topk_w, torch.float16)
    N = values.shape[1]
    rows = (pair.to(DEVICE) // TOP_K)[:, None].expand(-1, N)
    ref = torch.zeros(M, N, dtype=torch.float32, device=DEVICE)
    ref_bound = torch.zeros_like(ref)
    ref.scatter_add_(0, rows, values)
    ref_bound.scatter_add_(0, rows, bound)
    return ref, ref_bound


def assert_q32_close(out, ref, ref_bound, rtol_bound, what):
    err = (out.float() - ref).abs()
    tol = 2.0**-10 * ref.abs() + rtol_bound * ref_bound + 1e-3
    bad = ~(err <= tol)
    assert not bool(bad.any()), f"{what}: {int(bad.sum())} elements differ"


@pytest.mark.parametrize("E,K,N,M", Q32_CASES)
@pytest.mark.parametrize("bm", BM_PARAMS)
def test_q32_reduced_mode_matches_dense_reference_and_the_scalar_op(E, K, N, M, bm):
    p = q32_problem(E, K, N, M, seed=M)
    rows = M  # ceil(M * TOP_K / TOP_K)
    routing = align_block_size(p.topk_ids, E, bm)
    c = torch.zeros(rows, N + 1, dtype=torch.int64, device=DEVICE)
    call_wmma(
        p, c, routing_to_device(routing), op_top_k=1, bm=bm, mul=True, output_topk=TOP_K
    )
    ref, ref_bound = q32_reference(p, routing, E, M)

    out = finish_q32(c)
    assert bool((c[:, N] == 0).all()), "no row may be flagged on regular data"
    assert_q32_close(out, ref, ref_bound, ACC_TOL, "WMMA Q32 vs dense reference")

    # The scalar kernel (block sizes 1..8) on the same pairs.
    scalar_routing = routing_to_device(align_block_size(p.topk_ids, E, 8))
    c_s = torch.zeros(rows, N + 1, dtype=torch.int64, device=DEVICE)
    call_scalar(p, c_s, scalar_routing, op_top_k=1, bm=8, mul=True, output_topk=TOP_K)
    out_s = finish_q32(c_s)
    assert torch.equal(c[:, N], c_s[:, N]), "sticky status differs from the scalar op"
    # The scalar kernel uses the exact factored dequant; this one rounds each
    # weight once, which moves a pair value by up to ~1e-4 of its |terms|.
    assert_q32_close(out_s, ref, ref_bound, 3e-4, "scalar Q32 vs dense reference")
    assert_q32_close(out, out_s, ref_bound, 3e-4, "WMMA Q32 vs scalar Q32")


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_q32_reduced_mode_is_bit_identical_over_50_repeats(bm):
    E, K, N, M = 16, 640, 2560, 48
    p = q32_problem(E, K, N, M, seed=19)
    routing = routing_to_device(align_block_size(p.topk_ids, E, bm))
    first = None
    for _ in range(50):
        c = torch.zeros(M, N + 1, dtype=torch.int64, device=DEVICE)
        call_wmma(p, c, routing, op_top_k=1, bm=bm, mul=True, output_topk=TOP_K)
        if first is None:
            first = c
        else:
            assert torch.equal(c, first)


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_q32_non_finite_activation_flags_only_its_own_row(bm):
    E, K, N, M, bad_pair = 16, 640, 2560, 64, 21 * TOP_K + 3
    p = q32_problem(E, K, N, M, seed=23)
    p.a[bad_pair, 11] = float("nan")
    routing = align_block_size(p.topk_ids, E, bm)
    c = torch.zeros(M, N + 1, dtype=torch.int64, device=DEVICE)
    call_wmma(
        p, c, routing_to_device(routing), op_top_k=1, bm=bm, mul=True, output_topk=TOP_K
    )
    flagged = c[:, N].nonzero().flatten().tolist()
    assert flagged == [bad_pair // TOP_K]
    out = finish_q32(c)
    assert bool(torch.isnan(out[bad_pair // TOP_K]).all())
    others = torch.arange(M, device=DEVICE) != bad_pair // TOP_K
    assert bool(torch.isfinite(out[others]).all())
    # The scalar op flags the same row.
    c_s = torch.zeros(M, N + 1, dtype=torch.int64, device=DEVICE)
    scalar_routing = routing_to_device(align_block_size(p.topk_ids, E, 8))
    call_scalar(p, c_s, scalar_routing, op_top_k=1, bm=8, mul=True, output_topk=TOP_K)
    assert torch.equal(c[:, N], c_s[:, N])


@pytest.mark.parametrize("bm", BM_PARAMS)
def test_q32_partial_beyond_the_limit_sets_the_sticky_status(bm):
    """Expert 0 gets huge scales: its pairs reach the Q32 limit 2^30 / top_k
    (far beyond fp16). Rows with a pair clearly above the limit must be flagged,
    rows whose pairs are all clearly below must not; the band in between is
    left alone (the scalar kernel derives its limit from K slices)."""
    E, K, N, M = 8, 640, 256, 64
    p = q32_problem(E, K, N, M, seed=29)
    # The packed tensors are shared through the cache: scale a private copy.
    p.pk = Packed(p.pk.b_q, p.pk.b_z, p.pk.scales.clone())
    p.pk.scales[0] = 3000.0
    p.a *= 1500.0
    routing = align_block_size(p.topk_ids, E, bm)
    c = torch.zeros(M, N + 1, dtype=torch.int64, device=DEVICE)
    call_wmma(
        p, c, routing_to_device(routing), op_top_k=1, bm=bm, mul=True, output_topk=TOP_K
    )
    pair, expert = scheduled_pairs(routing, 1, M * TOP_K, E)
    values, _ = dense_reference(p.a, p.pk, pair, expert, 1, p.topk_w, torch.float16)
    limit = Q32_RANGE / TOP_K
    peak = values.abs().max(dim=1).values  # per scheduled pair
    rows = pair.to(DEVICE) // TOP_K
    over = torch.zeros(M, dtype=torch.bool, device=DEVICE)
    near = torch.zeros(M, dtype=torch.bool, device=DEVICE)
    over[rows[peak >= 4 * limit]] = True
    near[rows[(peak >= limit / 4) & (peak < 4 * limit)]] = True
    status = c[:, N] != 0
    assert bool(status[over].all()), "a row beyond the limit was not flagged"
    clean = ~over & ~near
    assert not bool(status[clean].any()), "a row far below the limit was flagged"
    assert bool(over.any()) and bool(clean.any()), "the scenario must mix both"


# ---------------------------------------------------------------------------
# CUDA graphs, early exits and argument checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bm", BM_PARAMS)
@pytest.mark.parametrize("mode", ["direct", "q32"])
def test_graph_capture_replays_with_new_inputs(bm, mode):
    """Capture the op once, then change activations, routing and the router
    weights in place: the replay must follow them (no host reads, no shapes
    taken from data)."""
    from vllm.platforms import current_platform

    E, K, N, M = 16, 640, 2560, 64
    q32 = mode == "q32"
    dtype = torch.float16
    p = (
        q32_problem(E, K, N, M, seed=31)
        if q32
        else make_problem(E, K, N, M, dtype=dtype, seed=31)
    )
    ids_a = make_topk_ids(M, TOP_K, E, 41)
    ids_b = make_topk_ids(M, TOP_K, E, 42, style="skewed")
    r_a, r_b = (align_block_size(ids, E, bm) for ids in (ids_a, ids_b))
    cap = max(r_a.expert_ids.numel(), r_b.expert_ids.numel())
    routing = routing_to_device(pad_routing(r_a, cap))
    other = routing_to_device(pad_routing(r_b, cap))
    # Capture and replay below share the tensors of ``routing`` (same shapes).
    op_top_k, output_topk = (1, TOP_K) if q32 else (TOP_K, 0)

    def make_c():
        if q32:
            return torch.zeros(M, N + 1, dtype=torch.int64, device=DEVICE)
        return poisoned_output(M * TOP_K, N, dtype)

    c = make_c()

    def run():
        call_wmma(
            p, c, routing, op_top_k=op_top_k, bm=bm, mul=True, output_topk=output_topk
        )

    run()
    current_platform.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    gen = torch.Generator(device=DEVICE).manual_seed(43)
    p.a.copy_(torch.randn(p.a.shape, generator=gen, device=DEVICE).to(dtype))
    p.topk_w.copy_(torch.rand(M * TOP_K, generator=gen, device=DEVICE) + 0.25)
    for dst, src in zip(
        (routing.sorted_ids, routing.expert_ids, routing.ntpp),
        (other.sorted_ids, other.expert_ids, other.ntpp),
    ):
        dst.copy_(src)
    c.copy_(make_c())
    graph.replay()
    current_platform.synchronize()

    expected = make_c()
    call_wmma(
        p, expected, other, op_top_k=op_top_k, bm=bm, mul=True, output_topk=output_topk
    )
    current_platform.synchronize()
    if q32:
        assert torch.equal(c, expected)
    else:
        assert torch.equal(_bits(c), _bits(expected))


def test_nothing_routed_returns_without_touching_c():
    E, K, N, M = 8, 128, 16, 16
    p = make_problem(E, K, N, M, dtype=torch.float16, seed=1)
    c = poisoned_output(M * TOP_K, N, torch.float16)
    empty = Routing(
        sorted_ids=torch.empty(0, dtype=torch.int32, device=DEVICE),
        expert_ids=torch.empty(0, dtype=torch.int32, device=DEVICE),
        ntpp=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        block_m=16,
        num_pairs=M * TOP_K,
    )
    call_wmma(p, c, empty, op_top_k=TOP_K, bm=16, mul=True)
    assert bool(torch.isnan(c).all())


def _valid_call_args(dtype=torch.float16):
    """A small valid direct-store call as a dict of the op's arguments."""
    E, K, N, M = 4, 128, 32, 8
    p = make_problem(E, K, N, M, dtype=dtype, seed=2)
    routing = routing_to_device(align_block_size(p.topk_ids, E, 16))
    return dict(
        a=p.a,
        c=poisoned_output(M * TOP_K, N, dtype),
        b_q_weight=p.pk.b_q,
        b_scales=p.pk.scales,
        b_qzeros=p.pk.b_z,
        topk_weights=p.topk_w,
        sorted_token_ids=routing.sorted_ids,
        expert_ids=routing.expert_ids,
        num_tokens_post_padded=routing.ntpp,
        top_k=TOP_K,
        block_size_m=16,
        mul_topk_weight=True,
        output_topk=0,
    )


def _call(**args):
    return torch.ops._rocm_C.moe_gptq_gemm_rdna3_wmma(**args)


def test_valid_call_runs():
    _call(**_valid_call_args())


@pytest.mark.parametrize("block_size_m", [0, 1, 8, 24, 128])
def test_unsupported_block_size_m_is_rejected(block_size_m):
    args = _valid_call_args()
    args["block_size_m"] = block_size_m
    with pytest.raises(RuntimeError, match="block_size_m must be 16, 32, or 64"):
        _call(**args)


def test_reduction_needs_the_q32_accumulator():
    args = _valid_call_args()
    args["output_topk"] = TOP_K
    with pytest.raises(RuntimeError, match="int64 Q32 accumulator"):
        _call(**args)


def test_q32_accumulator_is_fp16_only():
    args = _valid_call_args(torch.bfloat16)
    args["c"] = torch.zeros(8, 33, dtype=torch.int64, device=DEVICE)
    args["top_k"] = 1
    args["output_topk"] = TOP_K
    with pytest.raises(RuntimeError, match="fp16 activations only"):
        _call(**args)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda a: a.update(output_topk=-1), "output_topk must be >= 0"),
        (lambda a: a.update(top_k=0), "top_k must be >= 1"),
        (lambda a: a.update(c=a["c"][:-1]), "c must be a contiguous"),
        (lambda a: a.update(c=a["c"].to(torch.float32)), "dtype of a"),
        (lambda a: a.update(topk_weights=a["topk_weights"][:5]), "mul_topk_weight"),
        (
            lambda a: a.update(topk_weights=a["topk_weights"].to(torch.float16)),
            "mul_topk_weight",
        ),
        (
            lambda a: a.update(sorted_token_ids=a["sorted_token_ids"][:-1]),
            "whole token blocks",
        ),
        (
            lambda a: a.update(expert_ids=a["expert_ids"][:0]),
            "one entry per token block",
        ),
        (
            lambda a: a.update(b_scales=a["b_scales"][:, :, :16].contiguous()),
            "b_scales must be",
        ),
        (
            lambda a: a.update(b_qzeros=a["b_qzeros"][:, :, :3].contiguous()),
            "b_qzeros must be",
        ),
        (
            lambda a: a.update(sorted_token_ids=a["sorted_token_ids"].to(torch.int64)),
            "must be int32",
        ),
    ],
    ids=[
        "negative_output_topk",
        "zero_top_k",
        "short_c",
        "c_dtype",
        "short_topk_weights",
        "topk_weights_dtype",
        "ragged_sorted_ids",
        "short_expert_ids",
        "scales_shape",
        "zeros_shape",
        "ids_dtype",
    ],
)
def test_malformed_arguments_are_rejected(mutate, message):
    args = _valid_call_args()
    mutate(args)
    with pytest.raises(RuntimeError, match=message):
        _call(**args)


def test_tile_and_group_constraints_are_rejected():
    dtype = torch.float16
    # N = 24 is not a multiple of 16.
    p = make_packed(2, 128, 24, 128, dtype)
    base = _valid_call_args()
    args = dict(base, b_q_weight=p.b_q, b_scales=p.scales, b_qzeros=p.b_z)
    args["c"] = poisoned_output(80, 24, dtype)
    with pytest.raises(RuntimeError, match="multiple of 16"):
        _call(**args)
    # Group size 48 is not a multiple of 32 (K = 96, two groups).
    p = make_packed(2, 96, 32, 48, dtype)
    args = dict(base, b_q_weight=p.b_q, b_scales=p.scales, b_qzeros=p.b_z)
    args["a"] = torch.randn(8, 96, device=DEVICE).to(dtype)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        _call(**args)
