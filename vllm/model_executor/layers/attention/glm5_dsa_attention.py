# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash DeepSeek Sparse Attention (DSA) layer for gfx1030.

Implements ``Glm5NextDSAAttention`` — MLA-NoPE (q_lora 1536, kv_lora 512,
qk/v 256, NO RoPE) plus a k-pool-compressed lightning indexer (32 heads x 128,
topk 2048, kpool 4, tail <= 3 -> width 2051) — as a vLLM V1 attention layer.

Cache plumbing (least-engine-modification design)
-------------------------------------------------
vLLM V1 assigns exactly ONE ``KVCacheSpec`` per ``AttentionLayerBase``
registered in ``compilation_config.static_forward_context`` (see
``GPUModelRunner.get_kv_cache_spec``). A DSA layer needs TWO per-layer cache
groups (expanded-KV latents + indexer packed states), so this module registers
TWO lightweight ``AttentionLayerBase`` sublayers per DSA layer:

* ``{prefix}.kv_cache_layer``      -> latent cache
  ``FullAttentionSpec(num_kv_heads=1, head_size=512, head_size_v=0, fp16)``
* ``{prefix}.indexer_cache_layer`` -> indexer cache
  ``FullAttentionSpec(num_kv_heads=1, head_size=257, head_size_v=0, fp16)``

Both use the ``GLM5_DSA_ATTN`` backend (``vllm/v1/attention/backends/
glm5_dsa_attn.py``). Because the two specs differ, the hybrid KV-cache manager
places them in separate cache groups automatically; ROCm's
``check_runner_kv_caches_multi_layer`` is a no-op, so two attention layers
sharing one decoder-layer index are supported (same mechanism encoder-decoder
models already use). No engine files are modified.

Cache writes go through the STANDARD attention-layer update path:
``torch.ops.vllm.unified_kv_cache_update`` -> ``impl.do_kv_cache_update``
(scatter into the paged view). Reads are torch paged gathers in the backend.

Tensor-parallel decision (documented per task spec)
---------------------------------------------------
Projections that feed the REPLICATED indexer are replicated across TP
(``ReplicatedLinear``): q_a_proj, kv_a_proj_with_mqa, and all indexer
projections. The 64-head attention is TP-split by heads: q_b_proj and
kv_b_proj are ``ColumnParallelLinear`` (each rank holds 64/tp heads) and
o_proj is ``RowParallelLinear`` (all-reduce gather). The latent + indexer
caches are replicated (num_kv_heads=1), so every rank gathers identical
latents and expands its own head slice; the indexer produces identical topk
indices on every rank. At TP=1 (the gfx1030 serving config) this collapses to
plain replicated linears.

HIP hooks exist but are env-gated OFF (``envs.VLLM_GLM5_DSA_HIP``, declared by
the orchestrator); the default path is pure torch and cudagraph-capture-safe
(no .item(), no D2H, no data-dependent host branching; topk buffer is
pre-allocated and sentinel-filled).
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config.vllm import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.glm5_dsa_attn import (
    INDEXER_HEAD_SIZE,
    INVALID_INDEX,
    GLM5DSAAttnBackend,
    assemble_paged_rows,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    KVQuantMode,
)

logger = init_logger(__name__)

__all__ = ["Glm5NextDSAAttention", "Glm5NextDSAIndexer"]


# --------------------------------------------------------------------------- #
# Indexer top-k math (torch, vectorized; exact port of
# /tmp/glm53_refs/modeling_glm5_next.py Glm5NextTextIndexer lines 773-1024).
# --------------------------------------------------------------------------- #


def glm5_dsa_topk_from_packed(
    q_idx: torch.Tensor,
    weights: torch.Tensor,
    packed_states: torch.Tensor,
    token_to_req: torch.Tensor,
    token_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    ape: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    head_dim: int,
    always_select_tail: bool = True,
) -> torch.Tensor:
    """Compute DSA top-k raw token indices per query token.

    Args:
        q_idx: ``[M, n_heads, head_dim]`` indexer queries (wq_b(q_resid)).
        weights: ``[M, n_heads]`` fp32 head weights
            (weights_proj(x) * n_heads**-0.5).
        packed_states: ``[B, T, 257]`` assembled indexer rows
            ``[k | gate_scores | valid]``; padded slots zeroed (valid=0).
        token_to_req: ``[M]`` long, query token -> sequence.
        token_positions: ``[M]`` long, absolute KV position of each query token.
        seq_lens: ``[B]`` long, per-sequence lengths.
        ape: ``[kpool, head_dim]`` fp32 pool-compression positional embedding.
        index_topk: raw-token budget (2048).
        index_kpool: pool size (4).
        head_dim: indexer head dim (128).
        always_select_tail: append the current incomplete tail pool (<=3).

    Returns:
        ``[M, index_topk + index_kpool - 1]`` int32 raw indices, -1 invalid.
    """
    device = q_idx.device
    num_tokens = q_idx.shape[0]
    n_heads = q_idx.shape[1]
    bsz, kv_len, _ = packed_states.shape
    kpool = index_kpool
    output_width = index_topk + (kpool - 1 if always_select_tail else 0)

    keys, gate_scores, valid_channel = torch.split(
        packed_states, [head_dim, head_dim, 1], dim=-1)
    valid_keys = valid_channel.bool().squeeze(-1)                 # [B, T]

    # ---- get_pooled_states (modeling lines 899-972), vectorized over B ----
    num_pools = (kv_len + kpool - 1) // kpool
    any_valid = valid_keys.any(dim=-1)                            # [B]
    first_key = torch.where(
        any_valid,
        valid_keys.long().argmax(dim=-1),
        torch.full((bsz,), kv_len, dtype=torch.long, device=device),
    )                                                             # [B]
    pool_offsets = torch.arange(
        num_pools * kpool, device=device).view(1, num_pools, kpool)
    pool_indices = first_key[:, None, None] + pool_offsets        # [B,P,kpool]
    safe_indices = pool_indices.clamp(0, kv_len - 1)
    batch_idx = torch.arange(bsz, device=device)[:, None, None]
    grouped_keys = keys[batch_idx, safe_indices]                  # [B,P,kpool,D]
    grouped_gate = gate_scores[batch_idx, safe_indices]
    grouped_valid = valid_keys[batch_idx, safe_indices] & (
        pool_indices < kv_len)                                    # [B,P,kpool]
    pool_valid = grouped_valid.all(dim=-1)                        # [B,P]
    pool_indices = pool_indices.masked_fill(~grouped_valid, INVALID_INDEX)

    # Pool compression: softmax(gate + ape) weighted keys, invalid -> -inf.
    logits = grouped_gate.float() + ape.float()[None, None, :, :]
    logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
    probabilities = torch.nan_to_num(
        logits.softmax(dim=2)).to(grouped_keys.dtype)
    pool_keys = (probabilities * grouped_keys).sum(dim=2)         # [B,P,D]
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)         # [B,P]

    # ---- per-query-token scoring (modeling lines 825-844) ----
    softmax_scale = head_dim ** -0.5
    req = token_to_req                                            # [M]
    pool_keys_sel = pool_keys[req]                                # [M,P,D]
    scores = torch.bmm(
        q_idx.float(), pool_keys_sel.float().transpose(1, 2))     # [M,H,P]
    scores = F.relu(scores * softmax_scale)
    index_scores = torch.bmm(
        weights.float().unsqueeze(1), scores).squeeze(1)          # [M,P]

    # Causal visibility: a pool is selectable iff its final token is visible.
    pool_end_sel = pool_end[req]                                  # [M,P]
    pool_valid_sel = pool_valid[req]                              # [M,P]
    pool_visible = pool_end_sel <= token_positions[:, None]       # [M,P]
    valid_candidates = pool_visible & pool_valid_sel
    index_scores = index_scores.masked_fill(
        ~valid_candidates, torch.finfo(index_scores.dtype).min)

    # ---- top-k pool selection -> raw indices (modeling lines 846-864) ----
    select_k = min(index_topk // kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices        # [M,K]
    selected_valid = valid_candidates.gather(-1, selected)        # [M,K]
    selected_indices = pool_indices[req[:, None], selected]       # [M,K,kpool]
    topk_indices = selected_indices.flatten(-2)                   # [M,K*kpool]
    topk_indices = topk_indices.masked_fill(
        ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
        INVALID_INDEX,
    )

    # ---- append_visible_tail (modeling lines 974-1024) ----
    if always_select_tail and kpool > 1:
        max_tail_width = kpool - 1
        first_key_tok = first_key[req]                            # [M]
        # All keys at positions <= token_positions are valid in vLLM's packed
        # per-sequence cache, so visible_count == position + 1.
        visible_count = token_positions + 1                       # [M]
        tail_count = visible_count.remainder(kpool)               # [M]
        tail_offsets = torch.arange(max_tail_width, device=device)
        tail_start = first_key_tok + visible_count - tail_count   # [M]
        tail_indices = tail_start[:, None] + tail_offsets         # [M, tw]
        kv_len_tok = seq_lens[req]                                # [M]
        tail_valid = (tail_offsets[None, :] < tail_count[:, None]) & (
            tail_indices < kv_len_tok[:, None])
        # Tail tokens sit below visible_count, hence are causally visible;
        # keep the explicit check for fidelity with the reference.
        tail_visible = tail_indices <= token_positions[:, None]
        tail_indices = tail_indices.masked_fill(
            ~(tail_valid & tail_visible), INVALID_INDEX)
        topk_indices = torch.cat([topk_indices, tail_indices], dim=-1)

    # ---- pad / clip to the fixed output width ----
    if topk_indices.shape[-1] < output_width:
        topk_indices = F.pad(
            topk_indices, (0, output_width - topk_indices.shape[-1]),
            value=INVALID_INDEX)
    topk_indices = topk_indices[..., :output_width]
    return topk_indices.to(torch.int32)


# --------------------------------------------------------------------------- #
# Registered custom op: assemble packed indexer rows from the paged indexer
# cache and compute top-k. Mirrors vllm/model_executor/layers/
# sparse_attn_indexer.py (opaque to torch.compile, eager under capture).
# --------------------------------------------------------------------------- #


@eager_break_during_capture
def glm5_dsa_indexer(
    q_idx: torch.Tensor,
    weights: torch.Tensor,
    indexer_layer_name: LayerNameType,
    ape: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    index_head_dim: int,
    always_select_tail: bool,
) -> torch.Tensor:
    """Fill ``topk_indices_buffer`` with DSA top-k raw indices.

    Reads the packed ``[k|gate|valid]`` rows back from the indexer cache group
    (paged gather), rebuilds k-pool candidates, scores them against the indexer
    queries and selects ``index_topk // index_kpool`` pools (+visible tail).
    All causality/validity semantics follow the HF reference exactly.
    """
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    num_tokens = q_idx.shape[0]
    layer_name = _resolve_layer_name(indexer_layer_name)

    if not isinstance(attn_metadata, dict):
        # Profiling / dummy run: no metadata yet. Leave the buffer sentinel-
        # filled so any accidental consumer sees "no tokens".
        topk_indices_buffer[:num_tokens] = INVALID_INDEX
        return topk_indices_buffer

    attn_layer = forward_context.no_compile_layers[layer_name]
    indexer_cache = attn_layer.kv_cache
    meta = attn_metadata[layer_name]

    if indexer_cache is None or indexer_cache.numel() == 0:
        topk_indices_buffer[:num_tokens] = INVALID_INDEX
        return topk_indices_buffer

    # Assemble [B, T, 257] packed states (paged gather in torch).
    packed_states, _ = assemble_paged_rows(
        indexer_cache, meta.block_table, meta.seq_lens, meta.max_seq_len)

    # Per-query-token request mapping and absolute KV positions (device-only).
    query_start_loc = meta.query_start_loc
    seq_lens = meta.seq_lens
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    token_to_req = torch.repeat_interleave(
        torch.arange(query_lens.shape[0], device=q_idx.device,
                     dtype=torch.long),
        query_lens,
        output_size=num_tokens,
    )
    num_computed = seq_lens - query_lens
    local_idx = torch.arange(num_tokens, device=q_idx.device,
                             dtype=torch.long) - query_start_loc[token_to_req]
    token_positions = num_computed[token_to_req] + local_idx

    topk = glm5_dsa_topk_from_packed(
        q_idx, weights, packed_states, token_to_req, token_positions,
        seq_lens, ape, index_topk, index_kpool, index_head_dim,
        always_select_tail)

    topk_indices_buffer[:num_tokens] = INVALID_INDEX
    topk_indices_buffer[:num_tokens, :topk.shape[-1]] = topk
    return topk_indices_buffer


def glm5_dsa_indexer_fake(
    q_idx: torch.Tensor,
    weights: torch.Tensor,
    indexer_layer_name: LayerNameType,
    ape: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    index_head_dim: int,
    always_select_tail: bool,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="glm5_dsa_indexer",
    op_func=glm5_dsa_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=glm5_dsa_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


# --------------------------------------------------------------------------- #
# Indexer submodule (weights + per-token projections).
# --------------------------------------------------------------------------- #


class Glm5NextDSAIndexer(nn.Module):
    """DSA indexer with k-pool compression (Glm5NextTextIndexer port).

    All projections are REPLICATED across TP: the indexer must produce
    identical top-k indices on every rank so the TP-split attention gathers
    consistently.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        n_heads: int,
        head_dim: int,
        index_topk: int,
        index_kpool: int,
        index_kpool_always_select_tail: bool,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.softmax_scale = head_dim ** -0.5

        self.wq_b = ReplicatedLinear(
            q_lora_rank, n_heads * head_dim, bias=False,
            params_dtype=params_dtype, prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(
            hidden_size, head_dim, bias=False,
            params_dtype=params_dtype, prefix=f"{prefix}.wk")
        # LayerNorm WITH bias (elementwise_affine=True), eps 1e-6 — matches
        # nn.LayerNorm defaults and the HF reference.
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size, n_heads, bias=False,
            params_dtype=params_dtype, prefix=f"{prefix}.weights_proj")

        # fp32 pool-compression parameters (per task spec).
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(index_kpool, head_dim, dtype=torch.float32))
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(head_dim, hidden_size, dtype=torch.float32))

    def compute_packed_row(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Build this step's packed indexer row ``[k | gate | valid]``.

        ``k = k_norm(wk(x))``, ``gate = x @ gate_w^T`` (fp32 math, cast to the
        cache dtype), ``valid = 1`` for every real vLLM token. Returned fp16
        ``[M, 257]`` for the cache write.
        """
        k = self.k_norm(self.wk(hidden_states)[0])
        gate = F.linear(hidden_states.to(torch.float32),
                        self.index_kpool_compress_gate)
        valid = torch.ones(
            hidden_states.shape[0], 1, device=hidden_states.device,
            dtype=k.dtype)
        return torch.cat([k, gate.to(k.dtype), valid], dim=-1)

    def compute_q_idx_and_weights(
        self,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer query heads and head-weights for the top-k op."""
        num_tokens = q_resid.shape[0]
        q_idx = self.wq_b(q_resid)[0].view(
            num_tokens, self.n_heads, self.head_dim)
        weights = self.weights_proj(hidden_states)[0].to(torch.float32) * (
            self.n_heads ** -0.5)
        return q_idx, weights


# --------------------------------------------------------------------------- #
# Registered cache sublayer (one KV-cache group each).
# --------------------------------------------------------------------------- #


class _Glm5DSACacheLayer(nn.Module, AttentionLayerBase):
    """One registered paged cache group of a DSA layer.

    Registers itself in ``static_forward_context`` under its own layer name so
    the runner allocates a dedicated KV-cache group + metadata builder for it.
    The latent group additionally runs the scan attention through the standard
    ``unified_attention_with_output`` path; the indexer group is write-only
    (its ``forward`` is never invoked — the top-k op reads its cache).
    """

    def __init__(
        self,
        layer_name: str,
        head_size: int,
        vllm_config: VllmConfig,
        backend_cls: type[AttentionBackend],
        num_heads: int = 0,
        impl_kwargs: dict | None = None,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.head_size = head_size
        self.backend_cls = backend_cls
        self.num_kv_heads = 1
        self.num_heads = num_heads
        self.attn_backend = backend_cls
        self.kv_cache_dtype = "auto"
        self.kv_cache_torch_dtype = torch.float16
        self.sliding_window = None
        self.attn_type = AttentionType.DECODER

        compilation_config = vllm_config.compilation_config
        if layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {layer_name}")
        compilation_config.static_forward_context[layer_name] = self

        # Placeholder replaced by bind_kv_cache.
        self.kv_cache = torch.tensor([])

        impl_cls = backend_cls.get_impl_cls()
        self.impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            **(impl_kwargs or {}),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # head_size_v=0 -> ONE row per token (MLA-style latent packing):
        # state_content_size_bytes == head_size * dtype_size.
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=KVQuantMode.NONE,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # No quant scales on these cache groups; keep the hook a no-op.
        return


# --------------------------------------------------------------------------- #
# The DSA attention layer.
# --------------------------------------------------------------------------- #


class Glm5NextDSAAttention(nn.Module):
    """GLM-5.3-Flash DSA layer: MLA-NoPE + k-pool indexer + scan attention.

    See module docstring for the cache-plumbing and TP decisions. Forward
    contract (per DESIGN.md): ``forward(hidden_states [M, 4096],
    prev_topk_indices) -> (output [M, 4096], topk_indices or None)``; the
    top-k return is non-None only when the NEXT layer is a "shared" indexer
    layer reusing this layer's selection.
    """

    supports_dcp: ClassVar[bool] = False

    def __init__(
        self,
        config,
        layer_idx: int,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 0 for GLM-5.3
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        assert self.qk_rope_head_dim == 0, (
            "GLM-5.3-Flash DSA is NoPE; this layer does not implement RoPE.")
        self.scaling = self.qk_head_dim ** -0.5

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_local_heads = self.num_heads // tp_size
        params_dtype = vllm_config.model_config.dtype

        # Indexer wiring ("full" runs its own indexer, "shared" reuses the
        # previous full layer's selection via prev_topk_indices).
        indexer_types = getattr(config, "indexer_types", None)
        self.skip_topk = bool(
            indexer_types is not None
            and indexer_types[layer_idx] == "shared")
        self.next_skip_topk = bool(
            not self.skip_topk
            and indexer_types is not None
            and indexer_types[min(layer_idx + 1,
                                  len(indexer_types) - 1)] == "shared")

        # ---- projections (TP decision: indexer feeds replicated, attention
        # heads TP-split, o_proj RowParallel-gathers) ----
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size, self.q_lora_rank,
            bias=False, params_dtype=params_dtype,
            prefix=f"{prefix}.q_a_proj")
        self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                     eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim,
            bias=False, params_dtype=params_dtype,
            prefix=f"{prefix}.q_b_proj")
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False, params_dtype=params_dtype,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        # Expansion weight, applied to GATHERED latents at attend time.
        # ColumnParallel splits the 64 heads across TP consistently with
        # q_b_proj; the replicated latent cache feeds every rank.
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, params_dtype=params_dtype,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim, self.hidden_size,
            bias=False, params_dtype=params_dtype,
            prefix=f"{prefix}.o_proj")

        # ---- indexer ----
        self.indexer: Glm5NextDSAIndexer | None
        if self.skip_topk:
            self.indexer = None
        else:
            self.indexer = Glm5NextDSAIndexer(
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                n_heads=config.index_n_heads,
                head_dim=config.index_head_dim,
                index_topk=config.index_topk,
                index_kpool=config.index_kpool,
                index_kpool_always_select_tail=bool(
                    getattr(config, "index_kpool_always_select_tail", True)),
                params_dtype=params_dtype,
                prefix=f"{prefix}.indexer",
            )

        # ---- pre-allocated top-k buffer (cudagraph-stable) ----
        self.topk_width = (config.index_topk + config.index_kpool - 1
                           if getattr(config, "index_kpool_always_select_tail",
                                      True) else config.index_topk)
        self.topk_indices_buffer = torch.full(
            (vllm_config.scheduler_config.max_num_batched_tokens,
             self.topk_width),
            INVALID_INDEX,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        # ---- the two registered cache groups ----
        latent_head_size = self.kv_lora_rank + self.qk_rope_head_dim
        self.kv_cache_layer = _Glm5DSACacheLayer(
            layer_name=f"{prefix}.kv_cache_layer",
            head_size=latent_head_size,
            vllm_config=vllm_config,
            backend_cls=GLM5DSAAttnBackend,
            num_heads=self.num_local_heads,
            impl_kwargs={
                "topk_indices_buffer": self.topk_indices_buffer,
                "kv_b_proj": self.kv_b_proj,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "v_head_dim": self.v_head_dim,
            },
        )
        self.indexer_cache_layer = _Glm5DSACacheLayer(
            layer_name=f"{prefix}.indexer_cache_layer",
            head_size=INDEXER_HEAD_SIZE,
            vllm_config=vllm_config,
            backend_cls=GLM5DSAAttnBackend,
            num_heads=0,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_tokens = hidden_states.shape[0]

        # 1) Query path: q_resid = q_a_layernorm(q_a_proj(x)); q = q_b_proj.
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(q_resid)[0].view(
            num_tokens, self.num_local_heads, self.qk_head_dim)

        # 2) KV latent: compress + norm; cache the RAW latent row (expansion
        # happens at attend time — UNC-36 keeps the latent cache).
        compressed = self.kv_a_proj_with_mqa(hidden_states)[0]
        if self.qk_rope_head_dim > 0:
            k_pass, _ = torch.split(
                compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        else:
            k_pass = compressed
        k_pass = self.kv_a_layernorm(k_pass)
        latent_row = k_pass.view(
            num_tokens, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        # Standard attention-layer cache-write path (unified_kv_cache_update ->
        # impl.do_kv_cache_update).
        torch.ops.vllm.unified_kv_cache_update(
            latent_row, latent_row,
            _encode_layer_name(self.kv_cache_layer.layer_name))

        # 3) Indexer: write packed row, then select top-k.
        if self.indexer is not None:
            packed_row = self.indexer.compute_packed_row(hidden_states)
            idx_row = packed_row.view(num_tokens, 1, INDEXER_HEAD_SIZE)
            torch.ops.vllm.unified_kv_cache_update(
                idx_row, idx_row,
                _encode_layer_name(self.indexer_cache_layer.layer_name))

            q_idx, weights = self.indexer.compute_q_idx_and_weights(
                q_resid, hidden_states)
            topk_indices = torch.ops.vllm.glm5_dsa_indexer(
                q_idx,
                weights,
                _encode_layer_name(self.indexer_cache_layer.layer_name),
                self.indexer.index_kpool_compress_ape,
                self.topk_indices_buffer,
                self.indexer.index_topk,
                self.indexer.index_kpool,
                self.indexer.head_dim,
                self.indexer.index_kpool_always_select_tail,
            )
        else:
            # Shared-indexer layer: reuse the previous full layer's selection.
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer.")
            topk_indices = prev_topk_indices
            # Publish into THIS layer's buffer for the attention impl. The
            # copy is capture-safe (fixed [M, W] region, no host branching).
            self.topk_indices_buffer[:num_tokens] = INVALID_INDEX
            self.topk_indices_buffer[
                :num_tokens, :topk_indices.shape[-1]] = topk_indices

        # 4) Gathered sparse attention over the selected tokens (torch scan in
        # the backend impl; reads topk_indices_buffer + latent cache).
        attn_output = torch.empty(
            num_tokens, self.num_local_heads * self.v_head_dim,
            dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops.vllm.unified_attention_with_output(
            q, None, None, attn_output,
            _encode_layer_name(self.kv_cache_layer.layer_name))

        # 5) Output projection (RowParallel all-reduce across TP).
        output = self.o_proj(attn_output)[0]

        # Propagate the selection only if the next layer reuses it.
        return output, (topk_indices if self.next_skip_topk else None)

    def extra_repr(self) -> str:
        return (f"layer_idx={self.layer_idx}, "
                f"num_heads={self.num_heads}, "
                f"q_lora_rank={self.q_lora_rank}, "
                f"kv_lora_rank={self.kv_lora_rank}, "
                f"qk_head_dim={self.qk_head_dim}, "
                f"v_head_dim={self.v_head_dim}, "
                f"skip_topk={self.skip_topk}")
