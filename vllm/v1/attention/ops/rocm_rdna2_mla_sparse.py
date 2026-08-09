"""Sparse MLA decode kernel for gfx1030 (RDNA2) — stub.

This module mirrors `vllm.v1.attention.ops.rocm_aiter_mla_sparse` but is
isolated to the gfx1030 family and is the home for our custom HIP sparse
MLA decode kernel.

When the HIP kernel lands, this file will lazy-compile it via
torch.utils.cpp_extension.load_inline and route the call to it. Until
then, this module falls through to the upstream Triton path
(`rocm_sparse_attn_decode`) so behaviour is unchanged.

Activation:
  - VLLM_USE_RDNA2_MLA=1  → call our HIP kernel (once implemented)
  - otherwise            → fall through to the existing Triton path

Architecture gate:
  - Only active on gfx1030/gfx1031/gfx1032/gfx1035. The dispatcher
    in deepseek_v4/amd/rocm.py only routes here when on_gfx10x() is True.

This module is intentionally gfx1030-only. The Triton path remains the
fallback for CDNA (AITER) and for any RDNA arch we have not ported.
"""
import os
import torch


def is_available() -> bool:
    """Return True if our HIP sparse MLA decode kernel is available."""
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
    return _kernel_loaded()


_KERNEL_LOADED = False


def _kernel_loaded() -> bool:
    return _KERNEL_LOADED


def _load_kernel():
    """Lazy-compile the HIP kernel via load_inline. Not yet implemented."""
    raise NotImplementedError(
        "rocm_rdna2_sparse_mla kernel not yet implemented; "
        "fall through to the Triton path.")


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
    """Sparse MLA decode for gfx1030. Falls through to the Triton path until
    the HIP kernel lands.
    """
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
