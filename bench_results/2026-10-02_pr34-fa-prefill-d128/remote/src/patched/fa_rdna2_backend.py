# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python entry points for the FA-RDNA2 kernels (gfx1030).

Thin wrappers over the ``torch.ops._rocm_C.fa_rdna2_*`` ops built from
``csrc/rocm/fa_rdna2.cu``. Each op writes the attention output into ``out``
([num_tokens, H_q, D] fp16, contiguous); when ``out`` is omitted a new tensor
is allocated. ``scale`` defaults to ``D ** -0.5``.

Usage:
    from vllm.v1.attention.ops import fa_rdna2_backend as fa
    out = fa.fa_rdna2_decode_paged(Q, key_cache, value_cache, block_table,
                                   seq_lens, block_size, kv_splits=16)
"""

import torch


def _prepare(
    Q: torch.Tensor, scale: float | None, out: torch.Tensor | None
) -> tuple[float, torch.Tensor]:
    if scale is None:
        scale = Q.shape[-1] ** -0.5
    if out is None:
        out = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
    return float(scale), out


def fa_rdna2_decode_paged(
    Q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    kv_splits: int = 8,
    sliding_window: int = 0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    cu_query_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA2 split-K decode from the paged cache, one CTA group per query token.

    Args:
        Q: [num_tokens, H_q, D] fp16 queries; one per sequence unless
            cu_query_lens is given.
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache.
        value_cache: 5D fp16 view of the paged V cache
            (see rdna_attn._reinterpret_v_to_5d).
        block_table: [num_seqs, max_blocks] int32 block indices.
        seq_lens: [num_seqs] int32 KV length per sequence.
        block_size: Physical block size.
        kv_splits: Number of split-K CTAs per (token, head), 1..16.
        sliding_window: Window size (0 = none).
        scale: Softmax scale; defaults to D ** -0.5.
        out: Optional output buffer shaped like Q.
        cu_query_lens: Optional [num_seqs + 1] int32 cumulative query
            lengths for multi-token queries (spec-decode verify, short
            extends). Sequence s then owns queries cu[s]..cu[s+1]-1, which
            are its last positions (causal); tokens past cu[-1] get zeros.

    Returns:
        The attention output ([num_tokens, H_q, D] fp16), i.e. ``out``.
    """
    scale, out = _prepare(Q, scale, out)
    torch.ops._rocm_C.fa_rdna2_decode_paged(
        Q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        block_size,
        kv_splits,
        sliding_window,
        scale,
        out,
        cu_query_lens,
    )
    return out


def fa_rdna2_prefill_paged_varlen(
    Q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    causal: bool = True,
    sliding_window: int = 0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA2 varlen prefill (HEAD_DIM 128/256), one CTA per (q block, head).

    Args:
        Q: [num_tokens, H_q, D] fp16 queries of all sequences.
        key_cache: [num_blocks, H_kv, D/x, block_size, x] fp16 paged K cache.
        value_cache: 5D fp16 view of the paged V cache.
        block_table: [num_seqs, max_blocks] int32 block indices.
        cu_query_lens: [num_seqs + 1] int32 cumulative query lengths.
        seq_lens: [num_seqs] int32 KV length per sequence (prefix + query).
        block_size: Physical block size.
        causal: Apply the causal mask.
        sliding_window: Window size (0 = none).
        scale: Softmax scale; defaults to D ** -0.5.
        out: Optional output buffer shaped like Q.

    Returns:
        The attention output ([num_tokens, H_q, D] fp16), i.e. ``out``.
    """
    scale, out = _prepare(Q, scale, out)
    torch.ops._rocm_C.fa_rdna2_prefill_paged_varlen(
        Q,
        key_cache,
        value_cache,
        block_table,
        cu_query_lens,
        seq_lens,
        block_size,
        int(causal),
        sliding_window,
        scale,
        out,
    )
    return out


def fa_rdna2_prefill_paged_varlen_short(
    Q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    causal: bool = True,
    sliding_window: int = 0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """HEAD_DIM=128 varlen prefill tuned for KV < 4096 (BR=32, 256 threads).

    Arguments and return value as fa_rdna2_prefill_paged_varlen.
    """
    scale, out = _prepare(Q, scale, out)
    torch.ops._rocm_C.fa_rdna2_prefill_paged_varlen_short(
        Q,
        key_cache,
        value_cache,
        block_table,
        cu_query_lens,
        seq_lens,
        block_size,
        int(causal),
        sliding_window,
        scale,
        out,
    )
    return out


def fa_rdna2_prefill_paged_varlen_splitk(
    Q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    causal: bool = True,
    kv_splits: int = 4,
    sliding_window: int = 0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Varlen prefill with the KV range split over kv_splits CTAs (1..16).

    A reduction kernel merges the per-split partial O/M/L. Other arguments
    and the return value as fa_rdna2_prefill_paged_varlen.
    """
    scale, out = _prepare(Q, scale, out)
    torch.ops._rocm_C.fa_rdna2_prefill_paged_varlen_splitk(
        Q,
        key_cache,
        value_cache,
        block_table,
        cu_query_lens,
        seq_lens,
        block_size,
        int(causal),
        int(kv_splits),
        sliding_window,
        scale,
        out,
    )
    return out


def fa_rdna2_prefill_paged_varlen_gqa(
    Q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    causal: bool = True,
    sliding_window: int = 0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Varlen prefill (HEAD_DIM 128/256) with O and the softmax in registers.

    A CTA takes 16 query rows: 8 rows of 2 q-heads of one GQA group for even
    group sizes, 16 rows of one q-head otherwise. Arguments and return value
    as fa_rdna2_prefill_paged_varlen.
    """
    scale, out = _prepare(Q, scale, out)
    torch.ops._rocm_C.fa_rdna2_prefill_paged_varlen_gqa(
        Q,
        key_cache,
        value_cache,
        block_table,
        cu_query_lens,
        seq_lens,
        block_size,
        int(causal),
        sliding_window,
        scale,
        out,
    )
    return out
