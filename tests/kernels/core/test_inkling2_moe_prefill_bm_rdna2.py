# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the Inkling-2-Mini prefill expert GEMMs (csrc/rocm/moe_prefill_bm_rdna2.cu).

``moe_nvfp4_bm_rdna2`` (layers 0-19), its W4A8 variant ``moe_nvfp4_bm_i8_rdna2`` and
``moe_mxfp8_bm_rdna2`` (layers 20-23) run on the work list of
``moe_prefill_cta_rdna2`` (blocks of up to 32 rows of one expert, built from the
router's 8-padded layout) and are compared against an fp64 reference that
dequantises the packed weights of the experts the routing touches, with the one fp16
rounding of a dequantised weight that the decode kernels
(``test_inkling2_moe_rdna2.py``) apply (the W4A8 kernel: against the int8 activations
its quantiser produced).  The
routing is either uniform or skewed (three experts get every token: many 32-row work
blocks and a partial last one; four experts get none), the token counts (1 .. 4096)
include counts that are no multiple of 8 (padded rows inside the last 8-row group), a
K tail past one 256 K slice (MXFP8: K = 288), an N tail past a 512-column tile
(N = 1056), and both stages of the MoE (the up projection writes one row per routed
pair; the down projection scales by the router weight and adds the top-k rows of a
token with fp16 atomics).  The rows past the output are canaries: a padded row that
leaks into the output is caught.  The work list itself is compared with a Python
reference (coverage of every valid row exactly once, full blocks first, partial blocks
longest first).

``I2_MOE_PREFILL_JIT=<path to a .cu>`` runs the same tests on a JIT build of that file
(this is how the mutants are checked, see ``tools/rdna/inkling2/moe_prefill_bm_jit.py``
and ``moe_prefill_bm_mutants.py``).
"""

import functools
import os
import pathlib
import sys

import pytest
import torch

from vllm.platforms.rocm import on_gfx10x

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_gfx10x()),
    reason="needs gfx1030",
)

DEV = "cuda"
E, TOP_K_ROUTE = 16, 4
K_FP4, K_FP8, N = 384, 288, 1056
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CANARY = 3.0
CANARY_ROWS = 16


def _ops():
    jit = os.environ.get("I2_MOE_PREFILL_JIT")
    if jit:
        import hashlib

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
        from rdna.inkling2 import moe_prefill_bm_jit

        tag = hashlib.md5(pathlib.Path(jit).read_bytes()).hexdigest()[:10]
        return moe_prefill_bm_jit.build("i2_moe_prefill_test_" + tag, jit)
    from vllm import _custom_ops as ops

    if not hasattr(torch.ops._rocm_C, "moe_prefill_cta_rdna2"):
        pytest.skip("the build has no moe_prefill_bm_rdna2.cu")
    return ops


@pytest.fixture(scope="module")
def ops():
    return _ops()


def _e4m3(b: torch.Tensor) -> torch.Tensor:
    """e4m3fn bytes -> float64; 0x7F / 0xFF are +-480 like in the kernels."""
    b = b.to(torch.int64)
    e, m = (b >> 3) & 15, b & 7
    v = torch.where(
        e == 0,
        m.double() * 2.0**-9,
        (1 + m.double() / 8) * torch.pow(2.0, (e - 7).double()),
    )
    return torch.where((b >> 7) & 1 == 1, -v, v)


def _routing(tokens: int, seed: int, skew: bool) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, E, generator=g)
    if skew:
        scores[:, 3] += 2.0
        scores[:, 5] += 1.5
        scores[:, 9] += 1.0
        scores[:, 12:] -= 5.0
    return scores.topk(TOP_K_ROUTE, dim=-1).indices.to(torch.int32)


@functools.cache
def _nvfp4(seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(
        -(2**31), 2**31 - 1, (E, K_FP4 // 8, N), generator=g, dtype=torch.int32
    )
    scales = torch.randint(
        0x18, 0x50, (E, K_FP4 // 16, N), generator=g, dtype=torch.int32
    ).to(torch.uint8)
    glob = (0.02 + 0.04 * torch.rand(E, generator=g)).float()
    e, kw, n = q.shape
    nib = torch.stack([(q.to(torch.int64) >> (4 * i)) & 0xF for i in range(8)], dim=2)
    table = torch.tensor(FP4_VALUES, dtype=torch.float64)
    mag = table[nib & 7] * torch.where((nib >> 3) & 1 == 1, -1.0, 1.0)
    mag = mag.reshape(e, kw * 8, n)
    sc = (_e4m3(scales).float() * glob.view(e, 1, 1) * 16384.0).half()
    sc = sc.double().repeat_interleave(16, dim=1)
    # kernel: (value * 2^-14 as fp16, exact) * sc in fp16, i.e. one rounding
    w = ((mag * 2.0**-14).half().double() * sc).half().double()
    return q, scales, glob, w


@functools.cache
def _mxfp8(seed: int):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (E, K_FP8 // 4, N, 4), generator=g, dtype=torch.int64)
    q = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16) | (raw[..., 3] << 24)
    q = ((q + 2**31) % 2**32 - 2**31).to(torch.int32)
    scales = torch.randint(118, 125, (E, K_FP8 // 32, N), generator=g).to(torch.uint8)
    e, kw, n, _ = raw.shape
    v = _e4m3(raw).permute(0, 1, 3, 2).reshape(e, kw * 4, n)
    f = torch.pow(2.0, scales.double() - 127).repeat_interleave(32, dim=1)
    return q, scales, (v * f).float().half().double()


def _align(ids: torch.Tensor, block_m: int):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(ids.to(DEV), block_m, E)


def _per_pair(a_rows: torch.Tensor, flat_ids: torch.Tensor, w: torch.Tensor):
    out = torch.zeros(a_rows.shape[0], w.shape[2], dtype=torch.float64)
    for e in flat_ids.unique().tolist():
        sel = flat_ids == e
        out[sel] = a_rows[sel] @ w[e]
    return out


# natural K of the 8 byte positions of a permuted 8-K word
_PERM = [0, 2, 4, 6, 1, 3, 5, 7]


def _unpermute(a8: torch.Tensor) -> torch.Tensor:
    """int8 [rows, K] in the kernel's K order -> natural K order."""
    rows, k = a8.shape
    x = a8.reshape(rows, k // 8, 8)
    nat = torch.empty_like(x)
    nat[..., _PERM] = x
    return nat.reshape(rows, k)


def _act_dequant(a8: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
    """fp64 [rows, K] value the W4A8 kernel multiplies: int8 * its window's scale."""
    rows, k = a8.shape
    x = _unpermute(a8).double().reshape(rows, k // 32, 32)
    return (x * sa.double()[..., None]).reshape(rows, k)


def _run_stage(ops, fmt, stage, tokens, skew, seed):
    """Run one stage on the router layout of block size 8; (kernel out, fp64 ref).
    ``nvfp4_i8``: the W4A8 kernel; the reference multiplies the dequantised int8
    activations the quantiser produced (its own error is tested separately)."""
    g = torch.Generator().manual_seed(seed)
    ids = _routing(tokens, seed, skew)
    pairs = tokens * TOP_K_ROUTE
    sorted_ids, expert_ids, ntpp = _align(ids, 8)
    cta = ops.moe_prefill_cta_rdna2(expert_ids, ntpp, E, 8)
    if fmt in ("nvfp4", "nvfp4_i8"):
        q, scales, glob, w = _nvfp4(seed % 3)
        k = K_FP4
    else:
        q, scales, w = _mxfp8(seed % 3)
        glob, k = None, K_FP8
    a = torch.randn(tokens if stage == "up" else pairs, k, generator=g).half()
    a_ref, a_args = a.double(), (a.to(DEV),)
    if fmt == "nvfp4_i8":
        a8, sa = ops.moe_prefill_act_quant_rdna2(a.to(DEV))
        a_ref, a_args = _act_dequant(a8.cpu(), sa.cpu()), (a8, sa)
    if stage == "up":
        rows = pairs
        tw = torch.empty(0, dtype=torch.float32, device=DEV)
        top_k, mul, otk = TOP_K_ROUTE, False, 0
        ref = _per_pair(
            a_ref[torch.arange(pairs) // TOP_K_ROUTE], ids.reshape(-1).long(), w
        )
        buf = torch.full(
            (rows + CANARY_ROWS, N), CANARY, dtype=torch.float16, device=DEV
        )
        buf[:rows] = float("nan")  # a row the kernel does not write stays NaN
    else:
        rows = tokens
        weights = torch.rand(pairs, generator=g)
        tw = weights.to(DEV)
        top_k, mul, otk = 1, True, TOP_K_ROUTE
        per_pair = _per_pair(a_ref, ids.reshape(-1).long(), w)
        ref = (
            (per_pair * weights.double()[:, None])
            .reshape(tokens, TOP_K_ROUTE, N)
            .sum(1)
        )
        buf = torch.full(
            (rows + CANARY_ROWS, N), CANARY, dtype=torch.float16, device=DEV
        )
        buf[:rows] = 0.0  # the output of the down stage is accumulated into zeros
    out = buf[:rows]
    args = (*a_args, out, q.to(DEV), scales.to(DEV))
    if fmt != "mxfp8":
        args += (glob.to(DEV),)
    args += (tw, sorted_ids, cta, top_k, mul, otk)
    fn = {
        "nvfp4": ops.moe_nvfp4_bm_rdna2,
        "nvfp4_i8": ops.moe_nvfp4_bm_i8_rdna2,
        "mxfp8": ops.moe_mxfp8_bm_rdna2,
    }[fmt]
    fn(*args)
    torch.accelerator.synchronize()
    assert torch.all(buf[rows:] == CANARY), "the kernel wrote past the output rows"
    return out.double().cpu(), ref


def _check(out, ref, what):
    rms = ref.pow(2).mean().sqrt().item()
    err = (out - ref).abs()
    assert torch.isfinite(out).all(), what
    # w13: one fp16 rounding of an fp32 sum; w2: eight fp16 CAS adds (a few ulps, 2^-11)
    assert err.max().item() <= 1.2e-2 * rms + 1e-2, (what, err.max().item(), rms)
    assert err.mean().item() <= 1.5e-3 * rms + 1e-3, (what, err.mean().item(), rms)


# tokens: fewer rows than a group of 8, one block, padded rows inside a group, experts
# with more rows than one 32-row work block holds, up to the largest prefill chunk
@pytest.mark.parametrize("tokens", [1, 5, 8, 9, 33, 100, 257, 1024, 4096])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("stage", ["up", "down"])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8", "nvfp4_i8"])
def test_prefill_gemm_matches_fp64(ops, fmt, stage, skew, tokens):
    seed = 2000 + 31 * tokens + (7 if skew else 0)
    out, ref = _run_stage(ops, fmt, stage, tokens, skew, seed)
    _check(out, ref, f"{fmt}/{stage}/skew{int(skew)}/T{tokens}")


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8", "nvfp4_i8"])
def test_up_stage_is_bitwise_repeatable(ops, fmt, skew):
    """The w13 stage has no atomics: three launches on the same inputs return the same
    bits (the w2 stage adds the top-k rows of a token with fp16 CAS atomics, like the
    shipped kernels, so its repeat runs differ in the last bits)."""
    outs = [_run_stage(ops, fmt, "up", 1024, skew, 77)[0] for _ in range(3)]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


# every token count of the small chunks, then the neighbourhood of every power of two up
# to the largest chunk (blocks of 8 / 32 rows fill exactly, by one row, by one group)
SWEEP = [
    *range(1, 13),
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1000,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
]


@pytest.mark.parametrize("tokens", SWEEP)
@pytest.mark.parametrize("stage", ["up", "down"])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_token_count_sweep(ops, fmt, stage, tokens):
    """T = 1 .. 4096 (odd counts, one row past a block, empty experts at small T; the
    skewed routing on the odd counts) against the fp64 reference."""
    skew = tokens % 2 == 1
    out, ref = _run_stage(ops, fmt, stage, tokens, skew, 5000 + tokens)
    _check(out, ref, f"{fmt}/{stage}/skew{int(skew)}/T{tokens}")


def _quant_ref(a: torch.Tensor):
    """fp32 reference of ``moe_prefill_act_quant_rdna2`` in natural K order."""
    rows, k = a.shape
    f = a.float().reshape(rows, k // 32, 32)
    amax = f.abs().amax(-1)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    q = torch.round(f * inv[..., None]).clamp(-127, 127)  # round half to even
    return q.reshape(rows, k), amax * (1.0 / 127.0)


@pytest.mark.parametrize("rows,k", [(1, 32), (7, 384), (130, 1152)])
def test_act_quant_matches_reference(ops, rows, k):
    g = torch.Generator().manual_seed(11)
    a = torch.randn(rows, k // 32, 32, generator=g)
    a = a * torch.pow(
        2.0, torch.randint(-6, 7, (rows, k // 32, 1), generator=g).float()
    )
    a[0, 0] = 0.0  # an all-zero window: scale 0, int8 0
    a = a.reshape(rows, k).half()
    a8, sa = ops.moe_prefill_act_quant_rdna2(a.to(DEV))
    assert a8.dtype == torch.int8 and a8.shape == (rows, k)
    assert sa.dtype == torch.float32 and sa.shape == (rows, k // 32)
    q_ref, s_ref = _quant_ref(a)
    assert torch.equal(sa.cpu(), s_ref)
    diff = (_unpermute(a8.cpu()).float() - q_ref).abs()
    # the same rounding except where a division / product lands within 1 ulp of a tie
    assert diff.max().item() <= 1.0
    assert (diff > 0).float().mean().item() < 1e-3
    assert a8.cpu().abs().max().item() <= 127


def _ref_work_list(expert_ids: list[int], ntpp: int, ba: int):
    r = 32 // ba
    ids = expert_ids[: ntpp // ba]
    first: dict[int, int] = {}
    cnt: dict[int, int] = {}
    for b, e in enumerate(ids):
        first.setdefault(e, b)
        cnt[e] = cnt.get(e, 0) + 1
    full, part = [], []
    for e in sorted(cnt):
        f, p = divmod(cnt[e], r)
        for i in range(f):
            full.append((e, (first[e] + i * r) * ba, r * ba, 0))
        if p:
            part.append((e, (first[e] + f * r) * ba, p * ba, 0))
    part.sort(key=lambda t: (-t[2], t[0]))
    return full + part


@pytest.mark.parametrize("ba", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("tokens", [3, 40, 301])
def test_work_list(ops, tokens, skew, ba):
    ids = _routing(tokens, 77 + tokens, skew)
    _, expert_ids, ntpp = _align_ba(ids, ba)
    cta = ops.moe_prefill_cta_rdna2(expert_ids, ntpp, E, ba).cpu()
    ref = _ref_work_list(expert_ids.cpu().tolist(), int(ntpp.item()), ba)
    assert cta.shape == (((expert_ids.numel() + 32 // ba - 1) // (32 // ba)) + E, 4)
    assert [tuple(r) for r in cta[: len(ref)].tolist()] == ref
    assert torch.all(cta[len(ref) :, 2] == 0), (
        "entries past the last block must be empty"
    )
    assert torch.all(cta[len(ref) :, 0] == -1)


def _align_ba(ids: torch.Tensor, ba: int):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(ids.to(DEV), ba, E)


@pytest.mark.parametrize("k", [32, 64, 96, 288, 576])
@pytest.mark.parametrize("stage", ["up", "down"])
def test_nvfp4_k_tails_match_fp64(ops, k, stage):
    """NVFP4 32-K tail shapes including TP=4 (K=288) and TP=2 (K=576) down proj."""
    seed = 7000 + k + (1 if stage == "down" else 0)
    g = torch.Generator().manual_seed(seed)
    n = 576 if k == 288 else 1152
    q = torch.randint(
        -(2**31), 2**31 - 1, (E, k // 8, n), generator=g, dtype=torch.int32
    )
    scales = torch.randint(
        0x18, 0x50, (E, k // 16, n), generator=g, dtype=torch.int32
    ).to(torch.uint8)
    glob = (0.02 + 0.04 * torch.rand(E, generator=g)).float()
    nib = torch.stack([(q.to(torch.int64) >> (4 * i)) & 0xF for i in range(8)], dim=2)
    table = torch.tensor(FP4_VALUES, dtype=torch.float64)
    mag = (table[nib & 7] * torch.where((nib >> 3) & 1 == 1, -1.0, 1.0)).reshape(
        E, k, n
    )
    sc = (
        (_e4m3(scales).float() * glob.view(E, 1, 1) * 16384.0)
        .half()
        .double()
        .repeat_interleave(16, dim=1)
    )
    w = ((mag * 2.0**-14).half().double() * sc).half().double()

    tokens = 130
    ids = _routing(tokens, seed, True)
    pairs = tokens * TOP_K_ROUTE
    sorted_ids, expert_ids, ntpp = _align(ids, 8)
    cta = ops.moe_prefill_cta_rdna2(expert_ids, ntpp, E, 8)
    a = torch.randn(tokens if stage == "up" else pairs, k, generator=g).half()
    if stage == "up":
        rows, top_k, mul, otk = pairs, TOP_K_ROUTE, False, 0
        tw = torch.empty(0, dtype=torch.float32, device=DEV)
        ref = _per_pair(
            a.double()[torch.arange(pairs) // TOP_K_ROUTE], ids.reshape(-1).long(), w
        )
        buf = torch.full(
            (rows + CANARY_ROWS, n), CANARY, dtype=torch.float16, device=DEV
        )
        buf[:rows] = float("nan")
    else:
        rows, top_k, mul, otk = tokens, 1, True, TOP_K_ROUTE
        weights = torch.rand(pairs, generator=g)
        tw = weights.to(DEV)
        per_pair = _per_pair(a.double(), ids.reshape(-1).long(), w)
        ref = (
            (per_pair * weights.double()[:, None])
            .reshape(tokens, TOP_K_ROUTE, n)
            .sum(1)
        )
        buf = torch.full(
            (rows + CANARY_ROWS, n), CANARY, dtype=torch.float16, device=DEV
        )
        buf[:rows] = 0.0
    out = buf[:rows]
    ops.moe_nvfp4_bm_rdna2(
        a.to(DEV),
        out,
        q.to(DEV),
        scales.to(DEV),
        glob.to(DEV),
        tw,
        sorted_ids,
        cta,
        top_k,
        mul,
        otk,
    )
    torch.accelerator.synchronize()
    assert torch.all(buf[rows:] == CANARY), "the kernel wrote past the output rows"
    _check(out.double().cpu(), ref, f"nvfp4_ktail/{stage}/K{k}/N{n}")


def test_unsupported_shapes_are_rejected(ops):
    ids = _routing(8, 5, False)
    sorted_ids, expert_ids, ntpp = _align(ids, 8)
    cta = ops.moe_prefill_cta_rdna2(expert_ids, ntpp, E, 8)
    k = 144  # a multiple of 16 but not of 32
    q = torch.zeros(E, k // 8, N, dtype=torch.int32, device=DEV)
    scales = torch.zeros(E, k // 16, N, dtype=torch.uint8, device=DEV)
    glob = torch.ones(E, dtype=torch.float32, device=DEV)
    a = torch.zeros(8, k, dtype=torch.float16, device=DEV)
    out = torch.zeros(32, N, dtype=torch.float16, device=DEV)
    tw = torch.empty(0, dtype=torch.float32, device=DEV)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        ops.moe_nvfp4_bm_rdna2(
            a, out, q, scales, glob, tw, sorted_ids, cta, TOP_K_ROUTE, False, 0
        )
    with pytest.raises(RuntimeError, match="must divide"):  # 32 % 5 != 0
        ops.moe_prefill_cta_rdna2(expert_ids, ntpp, E, 5)
    with pytest.raises(RuntimeError, match="multiple of 32"):
        ops.moe_prefill_act_quant_rdna2(
            torch.zeros(4, 48, dtype=torch.float16, device=DEV)
        )
