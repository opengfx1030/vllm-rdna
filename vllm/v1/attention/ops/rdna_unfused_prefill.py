# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unfused paged prefill attention for the gfx1030 family: GEMMs + softmax.

The register-tiled flash kernel behind ``RDNA_ATTN`` prefill
(``fa_prefill_paged_varlen_gqa_kernel`` in ``csrc/rocm/fa_rdna2.cu``) runs at
about 1.1 TFLOP/s at head_dim 256 on a V620 (measured with
``tools/rdna/rdna_hint_flex/bench_attn_prefill.py``: a 2048-token chunk takes
24.6 ms per layer at an empty context and 353 ms at a 14k context), while the
rocBLAS fp16 GEMMs of the same shapes reach 16 to 33 TFLOP/s. This module runs
the textbook attention

    softmax(Q K^T * scale + causal mask) V

as two rocBLAS GEMMs and one softmax per block of query rows, on K/V gathered
from the paged cache. The query heads of a GQA group are folded into the GEMM
rows (``group * rows`` rows against one KV head), so a block is one
``[group * rows, D] x [D, kv_len]`` and one ``[group * rows, kv_len] x
[kv_len, D]`` GEMM. Rows of a block only see keys up to their own position, so
the part of the score matrix to the right of the block's last row is never
computed; the diagonal block carries an additive ``-inf`` mask.

Numerics: the scores are stored in the input dtype (fp16: the GEMM accumulates
in fp32 and rounds once), the softmax accumulates in fp32 and writes P in the
input dtype, the PV GEMM accumulates in fp32. This is the eager-attention
recipe of Hugging Face transformers and of PyTorch's math SDPA; the flash
kernel keeps the scores in fp32. The error against an fp64 reference is
tabulated in ``docs/rdna2/hi2-attn-prefill-2026-11-14.md``.

Memory: the scratch of one call is the gathered K and V of the sequence
(``2 * kv_len * kv_heads * head_size`` elements) plus one score tile
(``group * qblock * kv_len`` elements, softmax in place) and the q/o staging
tiles. ``plan_query_block`` picks the largest block (<= 512 rows) whose scratch
fits ``scratch_bytes`` and returns 0 when none of at least 32 rows does; the
caller then keeps the flash kernel for that sequence.

No code is taken from outside: softmax(QK^T)V with the GQA group folded into the
GEMM rows is the standard formulation (Vaswani et al. 2017, "Attention is all
you need"; the folding is used by most GQA eager implementations).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

_MIN_QBLOCK = 32
_ENV_ENABLE = "VLLM_FA_RDNA2_UNFUSED_PREFILL"
_ENV_MIN_Q = "VLLM_FA_RDNA2_UNFUSED_PREFILL_MIN_Q"
_ENV_SCRATCH_MB = "VLLM_FA_RDNA2_UNFUSED_PREFILL_SCRATCH_MB"
_ENV_QBLOCK = "VLLM_FA_RDNA2_UNFUSED_PREFILL_QBLOCK"

_tri_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}


@dataclass(frozen=True)
class UnfusedPrefillConfig:
    """Routing knobs of the unfused prefill path (all from the environment)."""

    min_q: int
    scratch_bytes: int
    max_qblock: int


def config_from_env() -> UnfusedPrefillConfig | None:
    """None unless ``VLLM_FA_RDNA2_UNFUSED_PREFILL=1`` (the default is off)."""
    if os.environ.get(_ENV_ENABLE, "0") != "1":
        return None
    return UnfusedPrefillConfig(
        min_q=max(2, int(os.environ.get(_ENV_MIN_Q, "64"))),
        scratch_bytes=int(os.environ.get(_ENV_SCRATCH_MB, "768")) << 20,
        max_qblock=max(_MIN_QBLOCK, int(os.environ.get(_ENV_QBLOCK, "512"))),
    )


def scratch_bytes_required(
    n: int,
    kv_len: int,
    qblock: int,
    group: int,
    kv_heads: int,
    head_size: int,
    itemsize: int = 2,
) -> int:
    """Bytes ``unfused_paged_prefill`` allocates for one sequence."""
    rows = min(qblock, n)
    gather = 2 * kv_len * kv_heads * head_size * itemsize
    tile = group * rows * kv_len * itemsize
    staging = 2 * group * rows * head_size * itemsize
    return gather + tile + staging + rows * rows * itemsize


def plan_query_block(
    n: int,
    kv_len: int,
    group: int,
    kv_heads: int,
    head_size: int,
    scratch_bytes: int,
    max_qblock: int = 512,
    itemsize: int = 2,
) -> int:
    """Query rows per GEMM block for ``n`` rows over ``kv_len`` keys, 0 = no fit."""
    if n <= 0 or kv_len < n:
        return 0
    cap = min(n, max_qblock)
    gather = 2 * kv_len * kv_heads * head_size * itemsize
    if scratch_bytes <= gather:
        return 0
    per_row = group * (kv_len + 2 * head_size) * itemsize + cap * itemsize
    rows = (scratch_bytes - gather) // per_row
    if rows >= cap:
        return cap
    rows -= rows % _MIN_QBLOCK
    return rows if rows >= _MIN_QBLOCK else 0


def gather_kv(cache: torch.Tensor, blocks: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Paged ``[nb, H_kv, D/x, bs, x]`` cache (any strides) -> ``[H_kv, kv_len, D]``.

    ``blocks`` are the physical blocks of one sequence in logical order; the
    last block may be partial. The unpacked V view of ``RDNA_ATTN`` has the
    same logical shape, so one function serves K and V.
    """
    h_kv, bs = cache.shape[1], cache.shape[3]
    nblk = (kv_len + bs - 1) // bs
    picked = cache.permute(1, 0, 3, 2, 4).index_select(1, blocks[:nblk])
    return picked.reshape(h_kv, nblk * bs, -1)[:, :kv_len]


def _causal_bias(rows: int, like: torch.Tensor) -> torch.Tensor:
    """Additive causal mask of at least ``rows`` x ``rows``; slice ``[:r, :r]``.

    The leading ``r x r`` block of a larger strictly-upper ``-inf`` matrix is
    the ``r x r`` mask, so one matrix per (device, dtype) serves every block.
    """
    key = (str(like.device), like.dtype)
    tri = _tri_cache.get(key)
    if tri is None or tri.shape[0] < rows:
        size = max(rows, 512)
        tri = torch.triu(
            torch.full((size, size), float("-inf"), device=like.device), diagonal=1
        ).to(like.dtype)
        _tri_cache[key] = tri
    return tri


def unfused_paged_prefill(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    blocks: torch.Tensor,
    kv_len: int,
    scale: float,
    out: torch.Tensor,
    qblock: int,
) -> None:
    """Causal attention of one sequence's query rows over its paged KV.

    Args:
        q: ``[n, H_q, D]`` queries of the sequence (not modified).
        key_cache: ``[nb, H_kv, D/x, bs, x]`` paged K cache.
        value_cache: the same logical shape for V.
        blocks: physical block ids of the sequence, logical order, device int.
        kv_len: keys visible to the last row (prefix + n); the rows sit at
            positions ``kv_len - n .. kv_len - 1``.
        scale: softmax scale.
        out: ``[n, H_q, D]`` written in place (any strides).
        qblock: query rows per GEMM block, from ``plan_query_block``.

    """
    n, h_q, d = q.shape
    h_kv = key_cache.shape[1]
    group = h_q // h_kv
    ctx = kv_len - n
    if ctx < 0 or h_q != h_kv * group:
        raise ValueError(
            f"unfused prefill: n={n} kv_len={kv_len} H_q={h_q} H_kv={h_kv}"
        )
    qblock = min(qblock, n)
    k = gather_kv(key_cache, blocks, kv_len)
    v = gather_kv(value_cache, blocks, kv_len)
    tri = _causal_bias(qblock, q)
    dev, dt = q.device, q.dtype
    s_flat = torch.empty(group * qblock * kv_len, device=dev, dtype=dt)
    q_flat = torch.empty(group * qblock * d, device=dev, dtype=dt)
    o_flat = torch.empty(group * qblock * d, device=dev, dtype=dt)
    for h in range(h_kv):
        kh, vh = k[h], v[h]
        heads = slice(h * group, (h + 1) * group)
        for lo in range(0, n, qblock):
            hi = min(lo + qblock, n)
            rows = hi - lo
            s_len = ctx + hi
            qb = q_flat[: group * rows * d].view(group, rows, d)
            torch.mul(q[lo:hi, heads, :].transpose(0, 1), scale, out=qb)
            s = s_flat[: group * rows * s_len].view(group * rows, s_len)
            torch.mm(qb.view(group * rows, d), kh[:s_len].t(), out=s)
            s.view(group, rows, s_len)[:, :, s_len - rows :].add_(tri[:rows, :rows])
            torch.ops.aten._softmax.out(s, -1, False, out=s)
            o = o_flat[: group * rows * d].view(group * rows, d)
            torch.mm(s, vh[:s_len], out=o)
            out[lo:hi, heads, :].copy_(o.view(group, rows, d).transpose(0, 1))
