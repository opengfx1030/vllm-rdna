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
) -> torch.Tensor:
    return torch.ops._rocm_C.paged_mqa_logits_decode_rdna2(
        q_fp8, kv_cache, weights, context_lens, block_tables, max_model_len
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
    ) -> torch.Tensor:
        rows = q_fp8.size(0) * q_fp8.size(1)
        return q_fp8.new_empty((rows, max_model_len), dtype=torch.float32)

