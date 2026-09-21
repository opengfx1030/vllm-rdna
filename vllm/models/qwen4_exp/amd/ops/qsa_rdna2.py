# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP-side QSA (Qwen Sparse Attention) decode kernels for Qwen4Exp on gfx1030.

Opt-in replacement for the QSA decode Triton kernels in
``vllm/models/qwen4_exp/amd/ops/qsa.py``:

- ``qsa_store_cache_rows_rdna2`` mirrors ``_store_qsa_rows_kernel``.
- ``qsa_compress_groups_rdna2`` mirrors ``_compress_qsa_groups_kernel``.
- ``qsa_mqa_paged_rdna2`` reuses the existing
  ``paged_mqa_logits_decode_rdna2`` indexer kernel when the shape
  contract matches (single Q head, single KV head, paged).

The sparse splitk attention (``_qsa_sparse_paged_gqa_splitk_kernel`` and
``_qsa_merge_splitk_kernel``) is left to the Triton path for now; it
only fires at prefill (M >> 8) and the prefill cudagraph story on
gfx1030 is still being worked through.

Gated by ``VLLM_RDNA_QSA_HIP=1`` and ``on_gfx10x()``. Default off
(Triton path stays the source of truth until verified).
"""

import os as _os

import torch

from vllm import _custom_ops as ops
from vllm.platforms.rocm import on_gfx10x

from vllm.envs import VLLM_RDNA_QSA_HIP


def qsa_use_rdna2() -> bool:
    """True iff the HIP QSA decode path is enabled for this process."""
    return bool(VLLM_RDNA_QSA_HIP) and on_gfx10x()


# ---------------------------------------------------------------------------
# HIP-side implementations. Forward to the matching ``_rocm_C::`` op.
# Output buffers are allocated here (mirror the Triton helpers).
# ---------------------------------------------------------------------------


def qsa_store_cache_rows(
    rows: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor,
    page_size: int, width: int,
) -> None:
    """Scatter ``rows`` into the paged ``cache`` at ``slots``.

    Mirrors ``_store_qsa_rows_kernel`` (out-of-place via the same
    in-place ``cache`` arg). ``cache`` is mutated.
    """
    ops.qsa_store_cache_rows_rdna2(rows, slots, cache, page_size, width)


def qsa_compress_groups(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    compressor_state_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    pooled: torch.Tensor,
    first_positions: torch.Tensor,
    compress_ratio: int,
    compressor_state_size: int,
    head_dim: int,
    load_rope_positions: bool,
) -> None:
    """Average ``compress_ratio`` consecutive keys into one pooled key.

    Writes ``pooled`` and ``first_positions`` (the RoPE-position tail of
    the leading member of each group). Mirrors
    ``_compress_qsa_groups_kernel``.
    """
    ops.qsa_compress_groups_rdna2(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        compress_ratio,
        compressor_state_size,
        head_dim,
        bool(load_rope_positions),
    )


def qsa_mqa_paged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """MQA paged indexer logits (single Q head, single KV head).

    Returns a logits tensor of shape ``[num_rows, max_kv_per_request]``.
    Wraps the existing ``paged_mqa_logits_decode_rdna2`` indexer kernel
    when the shape contract matches.
    """
    return ops.qsa_mqa_paged_rdna2(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


__all__ = [
    "qsa_compress_groups",
    "qsa_compress_groups_with_ratio_compat",
    "qsa_mqa_paged",
    "qsa_store_cache_rows",
    "qsa_store_cache_rows_compat",
    "qsa_use_rdna2",
]


# ---------------------------------------------------------------------------
# Drop-in replacements for the Triton helpers in ``ops/qsa.py``. These expose
# the same signatures as the Triton entry points so callers can route via the
# ``qsa_use_rdna2()`` gate without changing call sites. The shape checks are
# the RDNA2 kernel's responsibility (TORCH_CHECK inside the .cu).
# ---------------------------------------------------------------------------


def qsa_store_cache_rows_compat(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Drop-in replacement for ``ops.qsa.qsa_store_cache_rows``.

    Bridges the RDNA2 wrapper's ``(rows, slots, cache, page_size, width)``
    ordering to the Triton's ``(cache, slot_mapping, rows)``. The
    3D->2D head flatten matches the Triton helper so callers can pass
    either ``[N, D]`` or ``[N, 1, D]`` rows.
    """
    if rows.ndim == 3:
        rows = rows[:, 0]
    if not rows.shape[0]:
        return
    page_size = cache.shape[1]
    width = cache.shape[3]
    ops.qsa_store_cache_rows_rdna2(rows, slot_mapping, cache, page_size, width)


def qsa_compress_groups_with_ratio_compat(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``ops.qsa.qsa_compress_groups_with_ratio``.

    Allocates ``pooled`` and ``first_positions`` (the Triton helper does
    this internally) and forwards to ``qsa_compress_groups_rdna2``.
    Returns ``(pooled, first_positions)`` to match the Triton contract.
    """
    rows = token_to_req.numel()
    head_dim = raw_keys.shape[2]
    pooled = torch.empty(
        (rows, 1, head_dim),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty(
        (rows, 3), dtype=torch.int64, device=raw_keys.device
    )
    if not rows:
        return pooled, first_positions

    if rope_cache is None:
        rope_cache_arg = compressor_state_cache
        load_rope_positions = False
    else:
        rope_cache_arg = rope_cache
        load_rope_positions = True

    compressor_state_size = compressor_state_cache.shape[1]
    ops.qsa_compress_groups_rdna2(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache_arg,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        compress_ratio,
        compressor_state_size,
        head_dim,
        bool(load_rope_positions),
    )
    if _os.environ.get("VLLM_QSA_RDNA2_DEBUG", "0") == "1":
        torch.cuda.synchronize()
        print(
            "[QSA-DBG] rows=%d fp=[%d] min=%d max=%d pooled_absmax=%.3f "
            "rp_min=%d rp_max=%d cs=%d hd=%d load_rope=%s "
            "s_cs=%s s_rope=%s s_rk=%s s_rp=%s s_pooled=%s s_fp=%s"
            % (
                rows,
                first_positions.shape[1],
                int(first_positions.min()),
                int(first_positions.max()),
                float(pooled.abs().max()),
                int(raw_positions.min()),
                int(raw_positions.max()),
                compressor_state_size,
                head_dim,
                load_rope_positions,
                list(compressor_state_cache.stride()),
                list(rope_cache_arg.stride()),
                list(raw_keys.stride()),
                list(raw_positions.stride()),
                list(pooled.stride()),
                list(first_positions.stride()),
            ),
            flush=True,
        )
    return pooled, first_positions
