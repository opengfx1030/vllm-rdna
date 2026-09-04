# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM5_DSA_ATTN: torch-scan attention backend for GLM-5.3-Flash DSA layers.

DeepSeek Sparse Attention (DSA) on gfx1030, written blind (no runtime). The
DEFAULT path is a pure-torch "scan" implementation over two paged KV-cache
groups; HIP kernels (``csrc/rocm/glm5_dsa_indexer_rdna2.cu`` and
``glm5_dsa_mla_decode_rdna2.cu``) are hooked but gated OFF behind
``envs.VLLM_GLM5_DSA_HIP`` (declared by the orchestrator in ``envs.py``).

Two paged cache groups back each DSA layer (registered by
``vllm/model_executor/layers/attention/glm5_dsa_attention.py`` as two
``AttentionLayerBase`` sublayers):

1. **KV latent cache** — ``FullAttentionSpec(num_kv_heads=1, head_size=512,
   head_size_v=0, dtype=fp16)``. Stores the raw ``kv_a`` latent per token
   (1 KiB/token). Expansion via ``kv_b_proj`` happens at attend time in the
   scan path (UNC-36 HIP follow-up keeps the latent cache).
2. **Indexer cache** — ``FullAttentionSpec(num_kv_heads=1, head_size=257,
   head_size_v=0, dtype=fp16)``. Stores a packed row
   ``[k(128) | gate_scores(128) | valid(1)]`` per token, written every forward.

The backend is deliberately minimal and standard: ``num_kv_heads == 1`` and
``head_size_v == 0`` mean every token stores ONE row, so the bound per-layer
cache view is logically ``[num_blocks, 1, block_size, head_size]``. Both cache
writes and reads index that view directly with the block table, which makes the
scan path independent of the physical KV-cache layout (we still publish a layout
preference for the allocator).

Cudagraph policy mirrors ``rdna_attn.py``: ``UNIFORM_SINGLE_TOKEN_DECODE`` only.
Multi-query batches (MTP verify, ``q > 1``) are NEVER faked — the builder/impl
raise so the engine falls back to the piecewise/eager path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import KVCacheLayout

logger = init_logger(__name__)

# Packed indexer row width: [k(128) | gate_scores(128) | valid(1)].
INDEXER_HEAD_SIZE = 257
# DSA topk width: index_topk (2048) + kpool-1 (3) tail slots.
TOPK_WIDTH = 2051

# Sentinel for "no token" in topk index buffers.
INVALID_INDEX = -1


def hip_enabled() -> bool:
    """True only when the orchestrator-declared env flag is on. Default OFF."""
    return bool(getattr(envs, "VLLM_GLM5_DSA_HIP", False))


def rocm_c_has(op_name: str) -> bool:
    """True when ``torch.ops._rocm_C`` exposes ``op_name`` (built .so present)."""
    try:
        return hasattr(torch.ops._rocm_C, op_name)
    except Exception:
        return False


def assemble_paged_rows(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather every cached row of each sequence into a dense tensor.

    This is the torch-scan way to "read all past packed rows for each sequence"
    without a paged kernel. It is O(B * T) gather work, which the scan path
    explicitly accepts.

    Args:
        kv_cache: per-layer cache view ``[num_blocks, 1, block_size, head_size]``.
        block_table: ``[num_reqs, max_blocks_per_req]`` block ids.
        seq_lens: ``[num_reqs]`` current (post-write) sequence lengths.
        max_seq_len: ``T`` to pad every sequence to.

    Returns:
        rows: ``[num_reqs, T, head_size]`` fp rows; padded slots zeroed.
        valid: ``[num_reqs, T]`` bool; True for real (in-range) tokens.
    """
    device = kv_cache.device
    num_reqs = seq_lens.shape[0]
    block_size = kv_cache.shape[2]

    pos = torch.arange(max_seq_len, device=device)              # [T]
    valid = pos[None, :] < seq_lens[:, None]                    # [B, T]
    block_idx = pos[None, :] // block_size                      # [1, T]
    offset = pos[None, :] % block_size                          # [1, T]
    # block_table [B, max_blocks] indexed by [1, T] -> [B, T]
    blocks = block_table[:, block_idx.reshape(-1)].reshape(
        num_reqs, max_seq_len)
    rows = kv_cache[blocks, 0, offset.expand(num_reqs, max_seq_len), :]
    rows = torch.where(valid[..., None], rows, rows.new_zeros(()))
    return rows, valid


def gather_selected_rows(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather cache rows at per-query-token selected positions.

    Args:
        kv_cache: ``[num_blocks, 1, block_size, head_size]``.
        block_table: ``[num_reqs, max_blocks_per_req]``.
        indices: ``[M, W]`` absolute in-sequence token positions, -1 invalid.
        token_to_req: ``[M]`` mapping each query token to its sequence.

    Returns:
        rows: ``[M, W, head_size]``; invalid slots zeroed.
        valid: ``[M, W]`` bool.
    """
    block_size = kv_cache.shape[2]
    valid = indices >= 0
    safe = indices.clamp(min=0)
    m, w = safe.shape
    req = token_to_req[:, None].expand(m, w)
    block_idx = safe // block_size
    offset = safe % block_size
    blocks = block_table[req, block_idx]                        # [M, W]
    rows = kv_cache[blocks, 0, offset, :]                       # [M, W, Hs]
    rows = torch.where(valid[..., None], rows, rows.new_zeros(()))
    return rows, valid


@dataclass
class GLM5DSAAttnMetadata:
    """Per-layer attention metadata for one DSA cache group.

    The topk index buffer is NOT carried here; it lives on the layer/impl as a
    pre-allocated, zeroed, cudagraph-stable tensor (see the layer module).
    """

    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class GLM5DSAAttnMetadataBuilder(
        AttentionMetadataBuilder[GLM5DSAAttnMetadata]):
    # The torch scan decode path is single-token-only; MTP-verify batches
    # (q > 1) must stay on the piecewise/eager path and are never faked.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Decode tokens (query_len == 1) are pulled to the front of the batch.
        self.reorder_batch_threshold = 1

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GLM5DSAAttnMetadata:
        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.all())
        return GLM5DSAAttnMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> GLM5DSAAttnMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        common_attn_metadata.query_start_loc.zero_()
        return attn_metadata


class GLM5DSAAttnImpl(AttentionImpl):
    """Torch-scan sparse MLA-NoPE attention over the paged latent cache.

    ``do_kv_cache_update`` is the standard cache-write path (called through
    ``unified_kv_cache_update``) for BOTH cache groups: it scatters each token's
    single row (latent or packed indexer state) into its slot. ``forward`` runs
    the gathered sparse attention for the latent group, reading the pre-selected
    topk indices from ``self.topk_indices_buffer``.
    """

    forward_includes_kv_cache_update: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        # DSA extras (passed via extra_impl_args by the DSA layer):
        topk_indices_buffer: torch.Tensor | None = None,
        kv_b_proj: torch.nn.Module | None = None,
        qk_nope_head_dim: int = 256,
        v_head_dim: int = 256,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else 1
        self.attn_type = attn_type
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sliding_window = (-1, -1) if sliding_window is None else (
            sliding_window - 1, 0)
        # DSA state.
        self.topk_indices_buffer = topk_indices_buffer
        self.kv_b_proj = kv_b_proj
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

    # ------------------------------------------------------------------ #
    # Cache write (standard path, used by BOTH cache groups).
    # ------------------------------------------------------------------ #
    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Scatter each token's single cache row into its paged slot.

        ``key`` is ``[M, 1, head_size]`` (num_kv_heads == 1). ``value`` is
        accepted for interface compatibility but ignored: a DSA cache group
        stores exactly ONE row per token. Padding tokens are never present in
        ``slot_mapping`` (it only covers the actual tokens), so no masking is
        required and the write is cudagraph-capture-safe.
        """
        del value
        if kv_cache is None or kv_cache.numel() == 0 or slot_mapping is None:
            return
        if self.attn_type in (AttentionType.ENCODER_ONLY,
                              AttentionType.ENCODER):
            return
        num_tokens = key.shape[0]
        block_size = kv_cache.shape[2]
        slots = slot_mapping[:num_tokens].to(torch.long)
        blocks = slots // block_size
        offsets = slots % block_size
        rows = key.reshape(num_tokens, -1)
        kv_cache[blocks, 0, offsets, :] = rows.to(kv_cache.dtype)

    # ------------------------------------------------------------------ #
    # Sparse attention over selected tokens (torch scan).
    # ------------------------------------------------------------------ #
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: GLM5DSAAttnMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value, output_scale, output_block_scale
        if attn_metadata is None:
            return output.fill_(0)
        num_tokens = query.shape[0]
        if num_tokens == 0:
            return output
        # Profiling / dummy runs may reach here before the cache is bound.
        if kv_cache is None or kv_cache.numel() == 0:
            return output.fill_(0)

        # The scan path only runs when the layer produced topk indices.
        if self.topk_indices_buffer is None or self.kv_b_proj is None:
            raise NotImplementedError(
                "GLM5_DSA_ATTN: impl.forward requires topk_indices_buffer and "
                "kv_b_proj; it is only valid as the DSA latent-group attention.")

        # q > 1 (MTP verify / chunked prefill) is never faked inside the scan
        # kernel: the builder advertised UNIFORM_SINGLE_TOKEN_DECODE only.
        if attn_metadata.max_query_len > 1:
            raise NotImplementedError(
                "GLM5_DSA_ATTN: query_len > 1 is not supported by the torch "
                "scan decode path; fall back to the eager/piecewise path.")

        device = query.device
        num_local_heads = self.num_heads
        qk_dim = self.qk_nope_head_dim
        v_dim = self.v_head_dim

        topk_indices = self.topk_indices_buffer[:num_tokens]     # [M, W]
        width = topk_indices.shape[1]

        # token -> request mapping (capture-safe; no host sync).
        query_lens = (attn_metadata.query_start_loc[1:]
                      - attn_metadata.query_start_loc[:-1])
        token_to_req = torch.repeat_interleave(
            torch.arange(query_lens.shape[0], device=device, dtype=torch.long),
            query_lens,
            output_size=num_tokens,
        )

        # Gather the selected latent rows: [M, W, kv_lora_rank].
        lat_sel, sel_valid = gather_selected_rows(
            kv_cache, attn_metadata.block_table, topk_indices, token_to_req)

        # Expand the gathered latents to per-head k_nope / v at attend time.
        # kv_b_proj: [kv_lora_rank] -> [num_local_heads * (qk_dim + v_dim)].
        kv_lora_rank = lat_sel.shape[-1]
        expanded = self.kv_b_proj(lat_sel.reshape(-1, kv_lora_rank))[0]
        expanded = expanded.view(num_tokens, width, num_local_heads,
                                 qk_dim + v_dim)
        k_sel = expanded[..., :qk_dim].float()                  # [M,W,H,qk]
        v_sel = expanded[..., qk_dim:qk_dim + v_dim].float()    # [M,W,H,v]

        q = query.view(num_tokens, num_local_heads, qk_dim).float()

        # Optional HIP fast path (env-gated OFF by default). On any failure we
        # transparently fall back to the torch scan below.
        out = torch.empty(num_tokens, num_local_heads, v_dim,
                          device=device, dtype=torch.float32)
        if not self._try_hip_mla_decode(q, k_sel, v_sel, sel_valid, out,
                                        self.scale):
            # scores[m,h,w] = q[m,h,:] . k_sel[m,w,h,:] * scale
            scores = torch.einsum("mhd,mwhd->mhw", q, k_sel) * self.scale
            neg_inf = torch.finfo(torch.float32).min
            scores = torch.where(sel_valid[:, None, :], scores, neg_inf)
            # If a query token selected nothing, softmax over all -inf would
            # NaN; guard by leaving output zero for such rows (they cannot
            # occur for real tokens, which always see >= 1 visible token).
            any_valid = sel_valid.any(dim=1)                    # [M]
            probs = torch.softmax(scores, dim=-1)
            probs = torch.where(any_valid[:, None, None], probs,
                                probs.new_zeros(()))
            out = torch.einsum("mhw,mwhd->mhd", probs, v_sel)

        out_dtype = output.dtype
        output.view(num_tokens, num_local_heads, v_dim).copy_(out.to(out_dtype))
        return output

    # ------------------------------------------------------------------ #
    # HIP hook (OFF by default). Contract: DESIGN.md UNC-35/36 §D.
    # ------------------------------------------------------------------ #
    def _try_hip_mla_decode(
        self,
        q_nope: torch.Tensor,      # [M, H, 256] fp32 query
        k_sel: torch.Tensor,       # [M, W, H, 256] fp32 gathered keys
        v_sel: torch.Tensor,       # [M, W, H, 256] fp32 gathered values
        sel_valid: torch.Tensor,   # [M, W] bool
        out: torch.Tensor,         # [M, H, v_dim] fp32 (Tensor!)
        scale: float,
    ) -> bool:
        """Call ``glm5_dsa_mla_decode_rdna2`` if enabled and available.

        Returns True on success (``out`` filled), False to fall back to torch.
        The D-agent kernel fuses QK -> softmax(fp32) -> PV over the selected
        slots. We pass contiguous fp16 buffers per the kernel contract; the
        per-head keys/values are flattened into the slot dimension because the
        gathered layout is ``[M, W, H, D]`` while the kernel expects a single
        contiguous selected-keys tensor. Wrapped in try/except so a shape or
        ABI mismatch degrades to the (always-correct) torch scan path.
        """
        if not (hip_enabled() and rocm_c_has("glm5_dsa_mla_decode_rdna2")):
            return False
        try:
            m, w, h, qk_dim = k_sel.shape
            v_dim = v_sel.shape[-1]
            # The kernel writes fp16 into a 2-D [M, H*v_dim] out buffer;
            # stage it and widen back into the caller's fp32 tensor.
            out_fp16 = torch.empty(
                m, h * v_dim, device=out.device, dtype=torch.float16
            )
            torch.ops._rocm_C.glm5_dsa_mla_decode_rdna2(
                q_nope.to(torch.float16).contiguous(),
                k_sel.to(torch.float16).reshape(m, w, h * qk_dim).contiguous(),
                v_sel.to(torch.float16).reshape(m, w, h * v_dim).contiguous(),
                sel_valid.to(torch.uint8).contiguous(),
                out_fp16,
                scale,
            )
            out.view(m, h * v_dim).copy_(out_fp16.float())
            return True
        except Exception as exc:
            logger.warning_once(
                "GLM5_DSA_ATTN: HIP mla_decode failed (%s); using torch scan.",
                exc)
            return False


class GLM5DSAAttnBackend(AttentionBackend):
    """Backend class for the GLM-5.3-Flash DSA torch-scan path.

    Both DSA cache groups (latent + indexer) use this backend; they differ only
    in their KV-cache spec (head_size 512 vs 257). The layer module returns this
    class directly from ``get_attn_backend()``, so no global-selector edit is
    required.
    """

    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16"]

    @staticmethod
    def get_name() -> str:
        return "GLM5_DSA_ATTN"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return GLM5DSAAttnImpl

    @staticmethod
    def get_builder_cls():
        return GLM5DSAAttnMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The scan path indexes blocks by (slot // block_size, slot % block_size)
        # and works for any positive block size.
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Latent (512) and packed-indexer (257) row widths.
        return [257, 512]

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        # DSA is a top-k sparse attention path.
        return True

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The scan indexes the logical [B, H, N, C] view directly; prefer the
        # block-contiguous, layer-compact layouts for allocator friendliness.
        return (KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)
