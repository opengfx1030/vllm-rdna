# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the Inkling-2-Mini expert GEMMs (csrc/rocm/moe_lowp_dot2_rdna2.cu).

``moe_nvfp4_gemm_rdna2`` (layers 0-19) and ``moe_mxfp8_gemm_rdna2`` (layers 20-23) are
compared against an fp64 reference that dequantises the packed weights of the experts
the routing touches.  The reference mimics the one fp16 rounding the kernels apply to a
dequantised weight (NVFP4: the combined scale is rounded to fp16 and multiplies the
E2M1 value; MXFP8: the E4M3 value times the power-of-two scale, exact in fp16).  Every
supported block size (1, 2, 4, 8 routed rows per expert block) is run with 1 .. M real
rows in a block (the padded rows of a block must be ignored), with experts that fill
several blocks, a K tail (K = 288 = 256 + 32), an N tail, and both stages of the MoE
(the up projection that writes one row per routed pair, and the down projection that
scales by the router weight and sums the top-k rows into the token).  The MXFP8
dequantisation is additionally checked bit for bit over all 256 byte values and the
scale bytes the kernel supports (95 .. 134).

The output accumulates in packed fp16 with CAS atomics, one add per 256 K, so the bound
is a few fp16 ulps of the largest partial sum, not an fp32 bound.

``I2_MOE_JIT=<path to a .cu>`` runs the same tests on a JIT build of that file (this is
how the mutants of the kernel are checked, see ``tools/rdna/inkling2/moe_jit.py``).
"""

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
K, N = 288, 1056  # K tail of 32 past one 256 slice, N tail past one 1024 tile
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _ops():
    jit = os.environ.get("I2_MOE_JIT")
    if jit:
        import hashlib

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
        from rdna.inkling2 import moe_jit

        tag = hashlib.md5(pathlib.Path(jit).read_bytes()).hexdigest()[:10]
        return moe_jit.build("i2_moe_test_" + tag, jit)
    from vllm import _custom_ops as ops

    if not hasattr(torch.ops._rocm_C, "moe_mxfp8_gemm_rdna2"):
        pytest.skip("the build has no moe_lowp_dot2_rdna2.cu")
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


def _routing(tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, E, generator=g)
    return scores.topk(TOP_K_ROUTE, dim=-1).indices.to(torch.int32)


def _nvfp4_weights(seed: int, k: int = K, n: int = N):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(
        -(2**31), 2**31 - 1, (E, k // 8, n), generator=g, dtype=torch.int32
    )
    scales = torch.randint(0x18, 0x50, (E, k // 16, n), generator=g, dtype=torch.int32)
    glob = 0.02 + 0.04 * torch.rand(E, generator=g)
    return q, scales.to(torch.uint8), glob.float()


def _nvfp4_dequant(q, scales, glob) -> torch.Tensor:
    """[E, K, N] float64: value * (e4m3 scale * global * 16384 as fp16) * 2^-14."""
    e, kw, n = q.shape
    nib = torch.stack([(q.to(torch.int64) >> (4 * i)) & 0xF for i in range(8)], dim=2)
    table = torch.tensor(FP4_VALUES, dtype=torch.float64)
    mag = table[nib & 7] * torch.where((nib >> 3) & 1 == 1, -1.0, 1.0)  # [E, K/8, 8, N]
    mag = mag.reshape(e, kw * 8, n)
    sc = (_e4m3(scales).float() * glob.view(e, 1, 1) * 16384.0).half()  # [E, K/16, N]
    sc = sc.double().repeat_interleave(16, dim=1)
    # kernel: (value * 2^-14 as fp16, exact) * sc in fp16, i.e. one rounding
    w = (mag * 2.0**-14).half().double() * sc
    return w.half().double()


def _mxfp8_weights(seed: int, k: int = K, n: int = N, scale_lo=118, scale_hi=124):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (E, k // 4, n, 4), generator=g, dtype=torch.int64)
    q = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16) | (raw[..., 3] << 24)
    q = ((q + 2**31) % 2**32 - 2**31).to(torch.int32)
    scales = torch.randint(scale_lo, scale_hi + 1, (E, k // 32, n), generator=g)
    return q, scales.to(torch.uint8), raw


def _mxfp8_dequant(raw, scales) -> torch.Tensor:
    """[E, K, N] float64 = fp16(e4m3 * 2^(scale - 127))."""
    e, kw, n, _ = raw.shape
    v = _e4m3(raw)  # [E, K/4, N, 4]
    v = v.permute(0, 1, 3, 2).reshape(e, kw * 4, n)
    f = torch.pow(2.0, scales.double() - 127).repeat_interleave(32, dim=1)
    return (v * f).float().half().double()


def _align(ids: torch.Tensor, block_m: int):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(ids.to(DEV), block_m, E)


def _per_pair(a_rows: torch.Tensor, flat_ids: torch.Tensor, w: torch.Tensor):
    """[pairs, N]: row p of ``a_rows`` against the dequantised weights of expert
    ``flat_ids[p]``."""
    out = torch.empty(a_rows.shape[0], w.shape[2], dtype=torch.float64)
    for e in flat_ids.unique().tolist():
        sel = flat_ids == e
        out[sel] = a_rows[sel] @ w[e]
    return out


def _run_stage(ops, fmt, stage, q, scales, glob, w, tokens, block_m, seed):
    """Run one stage, return (kernel output, fp64 reference)."""
    g = torch.Generator().manual_seed(seed)
    ids = _routing(tokens, seed)
    pairs = tokens * TOP_K_ROUTE
    sorted_ids, expert_ids, ntpp = _align(ids, block_m)
    if stage == "up":
        a = torch.randn(tokens, K, generator=g).half()
        out = torch.zeros(pairs, N, dtype=torch.float16, device=DEV)
        tw = torch.empty(0, dtype=torch.float32, device=DEV)
        top_k, mul, otk = TOP_K_ROUTE, False, 0
        a_rows = a.double()[torch.arange(pairs) // TOP_K_ROUTE]
        ref = _per_pair(a_rows, ids.reshape(-1).long(), w)
    else:
        a = torch.randn(pairs, K, generator=g).half()
        out = torch.zeros(tokens, N, dtype=torch.float16, device=DEV)
        weights = torch.rand(pairs, generator=g)
        tw = weights.to(DEV)
        top_k, mul, otk = 1, True, TOP_K_ROUTE
        per_pair = _per_pair(a.double(), ids.reshape(-1).long(), w)
        ref = (
            (per_pair * weights.double()[:, None])
            .reshape(tokens, TOP_K_ROUTE, N)
            .sum(1)
        )
    args = (a.to(DEV), out, q.to(DEV), scales.to(DEV))
    if fmt == "nvfp4":
        args += (glob.to(DEV),)
    args += (tw, sorted_ids, expert_ids, ntpp, top_k, block_m, mul, otk)
    (ops.moe_nvfp4_gemm_rdna2 if fmt == "nvfp4" else ops.moe_mxfp8_gemm_rdna2)(*args)
    return out.double().cpu(), ref


def _check(out, ref, what):
    rms = ref.pow(2).mean().sqrt().item()
    err = (out - ref).abs()
    assert torch.isfinite(out).all(), what
    # one fp16 CAS add per 256 K: a few ulps (2^-11 relative) of the partial sums
    assert err.max().item() <= 1.2e-2 * rms + 1e-2, (what, err.max().item(), rms)
    assert err.mean().item() <= 1.5e-3 * rms + 1e-3, (what, err.mean().item(), rms)


# tokens: 1 row per expert block, fewer rows than a block, a full block, experts with
# more rows than one block holds
@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8, 13, 40])
@pytest.mark.parametrize("block_m", [1, 2, 4, 8])
@pytest.mark.parametrize("stage", ["up", "down"])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_moe_gemm_matches_fp64(ops, fmt, stage, block_m, tokens):
    seed = 1000 + 17 * tokens + block_m
    if fmt == "nvfp4":
        q, scales, glob = _nvfp4_weights(seed)
        w = _nvfp4_dequant(q, scales, glob)
    else:
        q, scales, raw = _mxfp8_weights(seed)
        glob, w = None, _mxfp8_dequant(raw, scales)
    out, ref = _run_stage(ops, fmt, stage, q, scales, glob, w, tokens, block_m, seed)
    _check(out, ref, f"{fmt}/{stage}/M{block_m}/T{tokens}")


@pytest.mark.parametrize("block_m", [1, 2, 4, 8])
def test_mxfp8_dequant_is_exact_for_every_byte_and_supported_scale(ops, block_m):
    """One expert, one-hot activations: the output is the dequantised weight itself."""
    k, n = 32, 8
    # 8 columns x 32 K = 256 bytes: every byte value once; one scale byte per column
    raw = torch.arange(256, dtype=torch.int64).reshape(n, k).t().contiguous()  # [K, N]
    scale_bytes = torch.tensor([95, 104, 105, 118, 127, 130, 134, 120])
    words = raw.reshape(k // 4, 4, n)
    q = words[:, 0] | (words[:, 1] << 8) | (words[:, 2] << 16) | (words[:, 3] << 24)
    q = ((q + 2**31) % 2**32 - 2**31).to(torch.int32)[None].to(DEV)
    scales = scale_bytes.to(torch.uint8).reshape(1, 1, n).to(DEV)
    ids = torch.zeros(k, 1, dtype=torch.int32)  # token r -> expert 0
    sorted_ids, expert_ids, ntpp = moe_align_for(ids, block_m)
    a = torch.eye(k, dtype=torch.float16, device=DEV)  # token r reads K position r
    out = torch.zeros(k, n, dtype=torch.float16, device=DEV)
    tw = torch.empty(0, dtype=torch.float32, device=DEV)
    ops.moe_mxfp8_gemm_rdna2(
        a, out, q, scales, tw, sorted_ids, expert_ids, ntpp, 1, block_m, False, 0
    )
    expected = (
        (_e4m3(raw) * torch.pow(2.0, scale_bytes.double() - 127)[None]).float().half()
    )
    assert torch.equal(out.cpu(), expected)


def moe_align_for(ids: torch.Tensor, block_m: int):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(ids.to(DEV), block_m, 1)


def test_unsupported_block_size_is_rejected(ops):
    q, scales, _ = _mxfp8_weights(1)
    ids = _routing(2, 1)
    sorted_ids, expert_ids, ntpp = _align(ids, 4)
    a = torch.zeros(2, K, dtype=torch.float16, device=DEV)
    out = torch.zeros(2 * TOP_K_ROUTE, N, dtype=torch.float16, device=DEV)
    tw = torch.empty(0, dtype=torch.float32, device=DEV)
    with pytest.raises(RuntimeError):
        ops.moe_mxfp8_gemm_rdna2(
            a,
            out,
            q.to(DEV),
            scales.to(DEV),
            tw,
            sorted_ids,
            expert_ids,
            ntpp,
            TOP_K_ROUTE,
            3,
            False,
            0,
        )
