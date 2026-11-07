# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels of the routing-statistics collector.

The torch formulation in ``rdna_expert_stats.py`` costs about 18 device kernels
per MoE call for the counts and the row histogram and about 15 more for the
pair counts. Under a FULL cudagraph on gfx1030 each kernel node is a few
microseconds of the serial chain, and a decode step runs the collector 49 times
(48 target layers and the MTP layer), so that formulation adds milliseconds per
step (docs/rdna2/expert-stats-t10-2026-11-07.md). Here one kernel does the
counts, the invalid-pair slot, the call counter and (decode only) the row
histogram, and a second one the pair counts: 1 and 2 launches per call.

Semantics are exactly those of the torch reference (``record_reference``), which
the tests compare against, including the padding mask and out-of-range ids:

* ``counts[e]`` += 1 per valid (row, slot) with id ``e``; ``counts[E]`` takes
  the invalid pairs (id < 0, id >= E, or a padding row); ``counts[E + 1]`` the
  call counter;
* a row is valid when at least one of its pairs is;
* ``rows_hist[n]`` += 1 where ``n`` is the number of valid rows (decode only);
* ``pair_counts[min * E + max]`` += 1 per unordered pair of slots of a row,
  slot ``E * E`` when either id is invalid.
"""

import torch

from vllm.triton_utils import tl, triton

BLOCK_ROWS = 64  # decode_max_tokens must not exceed it (the histogram is one program)


# Nothing is specialised on a value (divisibility by 16, equality with 1): a variant
# first compiled inside a CUDA graph capture would load its module while the stream
# captures. Only the constexprs select a variant, and warm-up passes run before capture.
@triton.jit(
    do_not_specialize=[
        "ids_ptr",
        "pad_ptr",
        "counts_ptr",
        "hist_ptr",
        "rows",
        "ids_stride",
        "num_experts",
    ]
)
def _counts_kernel(
    ids_ptr,
    pad_ptr,
    counts_ptr,
    hist_ptr,
    rows,
    ids_stride,
    num_experts,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_PAD: tl.constexpr,
    HIST: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_k = tl.arange(0, BLOCK_K)
    in_rows = offs_r < rows
    in_range = in_rows[:, None] & (offs_k[None, :] < TOP_K)
    ids = tl.load(
        ids_ptr + offs_r[:, None] * ids_stride + offs_k[None, :],
        mask=in_range,
        other=-1,
    ).to(tl.int32)
    ok = in_range & (ids >= 0) & (ids < num_experts)
    if HAS_PAD:
        pad = tl.load(pad_ptr + offs_r, mask=in_rows, other=1).to(tl.int32)
        ok = ok & (pad[:, None] == 0)
    tl.atomic_add(counts_ptr + ids, 1, mask=ok)
    bad = (in_range & ~ok).to(tl.int32)
    tl.atomic_add(counts_ptr + num_experts, tl.sum(tl.sum(bad, axis=1), axis=0))
    if pid == 0:
        tl.atomic_add(counts_ptr + num_experts + 1, 1)
    if HIST:
        row_ok = (tl.sum(ok.to(tl.int32), axis=1) > 0).to(tl.int32)
        tl.atomic_add(hist_ptr + tl.sum(row_ok, axis=0), 1)


@triton.jit(
    do_not_specialize=["ids_ptr", "pad_ptr", "pair_ptr", "ids_stride", "num_experts"]
)
def _pairs_kernel(
    ids_ptr,
    pad_ptr,
    pair_ptr,
    ids_stride,
    num_experts,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs_i = tl.arange(0, BLOCK_K)
    offs_j = tl.arange(0, BLOCK_K)
    base = ids_ptr + row * ids_stride
    a = tl.load(base + offs_i, mask=offs_i < TOP_K, other=-1).to(tl.int32)
    b = tl.load(base + offs_j, mask=offs_j < TOP_K, other=-1).to(tl.int32)
    a_ok = (a >= 0) & (a < num_experts)
    b_ok = (b >= 0) & (b < num_experts)
    if HAS_PAD:
        row_ok = tl.load(pad_ptr + row).to(tl.int32) == 0
        a_ok = a_ok & row_ok
        b_ok = b_ok & row_ok
    a2 = a[:, None]
    b2 = b[None, :]
    both = a_ok[:, None] & b_ok[None, :]
    low = tl.minimum(a2, b2)
    high = tl.maximum(a2, b2)
    slot = tl.where(both, low * num_experts + high, num_experts * num_experts)
    pair = (
        (offs_i[:, None] < offs_j[None, :])
        & (offs_i[:, None] < TOP_K)
        & (offs_j[None, :] < TOP_K)
    )
    tl.atomic_add(pair_ptr + slot, 1, mask=pair)


def record_fused(
    ids: torch.Tensor,
    padding: torch.Tensor | None,
    counts: torch.Tensor,
    rows_hist: torch.Tensor | None,
    pair_counts: torch.Tensor | None,
    *,
    num_experts: int,
    top_k: int,
) -> None:
    """Count one call. ``ids`` is ``[rows, >= top_k]`` int; ``counts`` the
    ``[E + 2]`` row of the call's phase; ``rows_hist`` is given for decode calls
    (then ``rows <= BLOCK_ROWS``) and ``pair_counts`` for decode calls with pairs.
    """
    rows = ids.shape[0]
    assert ids.stride(1) == 1, ids.stride()
    block_k = triton.next_power_of_2(top_k)
    pad = padding if padding is not None else ids  # never read without HAS_PAD
    if rows_hist is not None:
        assert rows <= BLOCK_ROWS, rows
    _counts_kernel[(triton.cdiv(rows, BLOCK_ROWS),)](
        ids,
        pad,
        counts,
        rows_hist if rows_hist is not None else counts,
        rows,
        ids.stride(0),
        num_experts,
        TOP_K=top_k,
        BLOCK_K=block_k,
        BLOCK_R=BLOCK_ROWS,
        HAS_PAD=padding is not None,
        HIST=rows_hist is not None,
    )
    if pair_counts is not None:
        _pairs_kernel[(rows,)](
            ids,
            pad,
            pair_counts,
            ids.stride(0),
            num_experts,
            TOP_K=top_k,
            BLOCK_K=block_k,
            HAS_PAD=padding is not None,
        )
