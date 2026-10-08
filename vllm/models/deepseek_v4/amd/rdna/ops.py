# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers and fake (meta) impls for the DeepSeek-V4 RDNA HIP ops.

The kernels live in ``csrc/rocm/rdna/dsv4/`` and are registered in
``_rocm_C`` only when ``PYTORCH_ROCM_ARCH`` contains an RDNA target. Every
wrapper here calls straight into ``torch.ops._rocm_C``; the fakes let the ops
be traced by torch.compile and captured in cudagraphs.
"""

import contextlib

import torch

with contextlib.suppress(ImportError):
    import vllm._rocm_C  # noqa: F401

try:
    from torch.library import register_fake
except ImportError:  # older torch
    from torch.library import impl_abstract as register_fake


def _has_op(name: str) -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, name)


def has_sparse_mla_decode() -> bool:
    return _has_op("sparse_mla_decode_rdna2")


def has_sparse_mla_prefill() -> bool:
    return _has_op("sparse_mla_prefill_rdna2")


def has_paged_mqa_logits() -> bool:
    return _has_op("paged_mqa_logits_decode_rdna2")


# ── Sparse MLA decode (fp8_ds_mla cache) ───────────────────────────────────
def sparse_mla_decode(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_indptr: torch.Tensor,
    main_block_size: int,
    main_num_rows: int,
    extra_block_size: int,
    extra_num_rows: int,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops._rocm_C.sparse_mla_decode_rdna2(
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        main_block_size,
        main_num_rows,
        extra_block_size,
        extra_num_rows,
        scale,
        attn_sink,
        out,
    )


# ── Sparse MLA prefill (plain fp16/bf16 kv rows) ───────────────────────────
def sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    num_kv: int,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops._rocm_C.sparse_mla_prefill_rdna2(
        q, kv, indices, indptr, num_kv, scale, attn_sink, out
    )


# ── Lightning-indexer paged MQA logits ─────────────────────────────────────
def paged_mqa_logits_decode(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    block_flat: bool = False,
) -> torch.Tensor:
    """FP8 paged MQA logits ``[B * next_n, max_model_len]`` (fp32, -inf pad).

    ``block_flat`` selects the DeepSeek-V4 C4A indexer page layout
    (all values, then all fp32 scales) over the per-slot one.
    """
    return torch.ops._rocm_C.paged_mqa_logits_decode_rdna2(
        q_fp8, kv_cache, weights, context_lens, block_tables, max_model_len, block_flat
    )


if has_sparse_mla_decode():

    @register_fake("_rocm_C::sparse_mla_decode_rdna2")
    def _sparse_mla_decode_fake(
        q: torch.Tensor,
        main_cache: torch.Tensor,
        main_indices: torch.Tensor,
        main_indptr: torch.Tensor,
        extra_cache: torch.Tensor,
        extra_indices: torch.Tensor,
        extra_indptr: torch.Tensor,
        main_block_size: int,
        main_num_rows: int,
        extra_block_size: int,
        extra_num_rows: int,
        scale: float,
        attn_sink: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        return None


if has_sparse_mla_prefill():

    @register_fake("_rocm_C::sparse_mla_prefill_rdna2")
    def _sparse_mla_prefill_fake(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        indptr: torch.Tensor,
        num_kv: int,
        scale: float,
        attn_sink: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        return None


if has_paged_mqa_logits():

    @register_fake("_rocm_C::paged_mqa_logits_decode_rdna2")
    def _paged_mqa_logits_decode_fake(
        q_fp8: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        max_model_len: int,
        block_flat: bool = False,
    ) -> torch.Tensor:
        rows = q_fp8.size(0) * q_fp8.size(1)
        return q_fp8.new_empty((rows, max_model_len), dtype=torch.float32)


# ── q-norm + RoPE + fp8_ds_mla KV insert ───────────────────────────────────
def has_qnorm_rope_kv_insert() -> bool:
    return _has_op("dsv4_qnorm_rope_kv_insert_rdna")


def qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    block_size: int,
    apply_q_norm: bool = True,
    apply_q_rope: bool = True,
) -> torch.Tensor:
    """RDNA counterpart of ``_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``.

    Accepts fp16 or bf16 ``q`` ``[N, H, 512]`` / ``kv`` ``[N, 512]`` and writes
    the fp8_ds_mla V4 row (448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scale bytes)
    into the uint8 paged ``k_cache`` at ``slot_mapping``. Returns q (in q's
    dtype) after RMSNorm + RoPE, zero-padded to ``padded_heads`` heads, or an
    empty tensor when ``padded_heads == 0`` (KV insert only).
    """
    return torch.ops._rocm_C.dsv4_qnorm_rope_kv_insert_rdna(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
        apply_q_norm,
        apply_q_rope,
    )


if has_qnorm_rope_kv_insert():

    @register_fake("_rocm_C::dsv4_qnorm_rope_kv_insert_rdna")
    def _qnorm_rope_kv_insert_fake(
        q_in: torch.Tensor,
        kv: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_head_padded: int,
        eps: float,
        cache_block_size: int,
        apply_q_norm: bool = True,
        apply_q_rope: bool = True,
    ) -> torch.Tensor:
        if q_head_padded == 0:
            return q_in.new_empty((0,))
        return q_in.new_empty((q_in.size(0), q_head_padded, q_in.size(2)))
