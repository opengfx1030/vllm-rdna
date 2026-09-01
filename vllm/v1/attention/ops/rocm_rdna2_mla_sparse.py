"""Sparse MLA decode kernel for gfx1030 (RDNA2).

This module mirrors `vllm.v1.attention.ops.rocm_aiter_mla_sparse` but is
isolated to the gfx1030 family and routes to our custom HIP sparse MLA
decode kernel compiled into `_rocm_C` (`sparse_mla_decode_rdna2`).

Activation:
  - VLLM_USE_RDNA2_MLA=1  → call the HIP kernel when it is registered
  - otherwise            → fall through to the existing Triton path

Architecture gate:
  - Only active on gfx1030/gfx1031/gfx1032/gfx1035. The dispatcher
    in deepseek_v4/amd/rocm.py only routes here when on_gfx10x() is True.

The Triton path remains the fallback for CDNA (AITER), for any RDNA arch
we have not ported, and when the HIP op is not registered.
"""
import os
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


_KERNEL_AVAILABLE: bool | None = None


def _kernel_available() -> bool:
    global _KERNEL_AVAILABLE
    if _KERNEL_AVAILABLE is None:
        try:
            from vllm import _rocm_C  # noqa: F401
            _KERNEL_AVAILABLE = hasattr(
                torch.ops._rocm_C, "sparse_mla_decode_rdna2")
        except Exception:
            _KERNEL_AVAILABLE = False
    return _KERNEL_AVAILABLE


def is_available() -> bool:
    """Return True if the HIP sparse MLA decode kernel can be used."""
    if os.environ.get("VLLM_USE_RDNA2_MLA") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from vllm.platforms.rocm import on_gfx10x
        if not on_gfx10x():
            return False
    except Exception:
        return False
    return _kernel_available()


def _hip_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    swa_ragged_indices: torch.Tensor,
    swa_ragged_indptr: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    output: torch.Tensor,
) -> None:
    logger.info_once(
        "RDNA2 sparse MLA decode using HIP kernel sparse_mla_decode_rdna2")
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    B, H, D = q.shape
    main_block_size = swa_k_cache.size(1)
    main_num_rows = swa_k_cache.size(0) * main_block_size

    if swa_only:
        extra_cache = torch.empty(0, device=q.device, dtype=torch.uint8)
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(B + 1, device=q.device, dtype=torch.int32)
        extra_block_size = 0
        extra_num_rows = 0
    else:
        assert kv_cache is not None
        assert topk_ragged_indices is not None and topk_ragged_indptr is not None
        extra_cache = kv_cache
        extra_indices = topk_ragged_indices
        extra_indptr = topk_ragged_indptr
        extra_block_size = kv_cache.size(1)
        extra_num_rows = kv_cache.size(0) * extra_block_size

    if attn_sink is None:
        sink = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        sink = attn_sink[:H].to(torch.float32).contiguous()

    out = torch.empty(B, H, D, device=q.device, dtype=torch.bfloat16)
    torch.ops._rocm_C.sparse_mla_decode_rdna2(
        q,
        swa_k_cache,
        swa_ragged_indices.to(torch.int32).contiguous(),
        swa_ragged_indptr.to(torch.int32).contiguous(),
        extra_cache,
        extra_indices.to(torch.int32).contiguous(),
        extra_indptr.to(torch.int32).contiguous(),
        main_block_size,
        main_num_rows,
        extra_block_size,
        extra_num_rows,
        float(scale),
        sink,
        out,
    )
    output.copy_(out.to(output.dtype))


def rocm_rdna2_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    """Sparse MLA decode for gfx1030: HIP kernel when available, else Triton."""
    if (
        _kernel_available()
        and swa_ragged_indices is not None
        and swa_ragged_indptr is not None
    ):
        _hip_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_only=swa_only,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            swa_ragged_indices=swa_ragged_indices,
            swa_ragged_indptr=swa_ragged_indptr,
            attn_sink=attn_sink,
            scale=scale,
            output=output,
        )
        return

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_sparse_attn_decode,
    )
    rocm_sparse_attn_decode(
        q=q,
        kv_cache=kv_cache,
        swa_k_cache=swa_k_cache,
        swa_only=swa_only,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        swa_ragged_indices=swa_ragged_indices,
        swa_ragged_indptr=swa_ragged_indptr,
        topk_ragged_indices=topk_ragged_indices,
        topk_ragged_indptr=topk_ragged_indptr,
        attn_sink=attn_sink,
        scale=scale,
        head_dim=head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        output=output,
    )
