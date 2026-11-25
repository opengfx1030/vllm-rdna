# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini DSpark draft (``mtp.safetensors``) for RDNA2 / gfx1030.

The draft reads four taps of the target model (the AttnRes partial block at the
entry of layer visits 5 / 23 / 41 and the final normalized output), mixes them per
draft layer (DFlare), and drafts a block of eight positions (the last committed
token plus seven masks) through three bidirectional layers over a paged KV cache.
A recurrent Markov head then turns the seven position hidden states into
sequentially dependent logits: ``hidden_out`` of the head is folded into the
target's LM head, so each step costs one ``[V, 256]`` fp16 row sweep.

Everything the draft runs per proposal is a native HIP kernel
(``csrc/rocm/inkling2_dspark_rdna2.cu``): there is no Triton, no torch matmul /
SDPA and no rocBLAS in this module.  PyTorch only allocates, slices and loads
weights.  Every kernel launch is on the current stream with persistent scratch, so
the whole draft step records into a HIP graph.

Numerics (each is a deviation from the bf16 reference, see
``docs/rdna2/inkling2-dspark.md`` for the measured size):

* weights are stored fp16 (the checkpoint is bf16; every value in range is exact,
  values below 6e-8 flush to zero and the loader counts them);
* GEMM operands are fp16, accumulation (``V_DOT2_F32_F16``), the residual stream,
  RMSNorm statistics, softmax, the Markov recurrence and its state are fp32;
* the target taps are stored as bf16 bits, the precision the draft was trained on
  (the target's fp32 block states can exceed the fp16 range);
* RoPE angles are formed in float64 and reduced mod 2 pi before the fp32 sincos;
* ``hidden_out`` is folded into the LM head with fp32 accumulation and kept as an
  fp16 ``[V, 256]`` matrix plus an fp32 bias, where the reference keeps bf16;
* the cached draft logits are fp16 (the reference caches bf16).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import regex as re
import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

from ..configs import Inkling2DSparkConfig
from ..speculator_config import Inkling2DSparkSpeculatorConfig
from .parallel import current_tensor_parallel, lm_head_shard
from .state import Inkling2KVOwner

logger = init_logger(__name__)

# Rows per request of the draft block (the anchor plus seven masks).
BLOCK = 8
# Requests per backbone pass: bounds the scratch (rows = 8 * requests).
_BACKBONE_REQS = 256
# Context rows per K/V precompute pass: bounds the mixed-feature scratch.
_CTX_CHUNK = 2048
# Rows the skinny kernel takes; more rows run on the dense GEMM designs.
_GEMV_MAX_ROWS = 32
# Requests per Markov head pass (``head_logits`` takes at most 32).
_HEAD_REQS = 32
# K splits of the o_proj / down_proj partial sums on the skinny path.
_RESIDUAL_SPLITS = 4
# Attention blocks to aim for (requests * kv heads * key splits).
_ATTN_TARGET_BLOCKS = 1024
_ATTN_MAX_SPLITS = 512
# fp16 largest finite value.
_FP16_MAX = 65504.0


def _gemm_design(rows: int) -> int:
    """The dense GEMM design (``rdna_hint_flex`` 0..5) for more than 32 rows."""
    return 5 if rows <= 64 else 0


def _linear(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    rows: int,
    w_row_ids: torch.Tensor | None = None,
) -> None:
    """``out[:rows] = x[:rows] @ w[w_row_ids].T`` (or ``w.T``) with fp16 operands
    and fp32 accumulation."""
    if rows <= _GEMV_MAX_ROWS:
        ops.inkling2_dspark_gemv(x, w, out, None, rows, 0, 0, w_row_ids)
    else:
        ops.inkling2_dspark_gemm(x, w, out, rows, _gemm_design(rows), w_row_ids)


def scheduled_draft_tokens_by_reqs(
    vllm_config: VllmConfig, drafted: int
) -> tuple[int, tuple[int, ...] | None]:
    """(Global max Markov steps, optional per-``num_reqs`` Markov step lookup).

    ``num_speculative_tokens_per_batch_size`` makes the scheduler verify the first
    ``k(batch size)`` of the block's tokens.  The chain is sequential (token ``i``
    never depends on a later one), so at batch size ``b`` any Markov steps past
    ``k(b)`` (or past the largest ``k`` among batch sizes padding into the same
    captured draft graph) produce tokens nothing consumes.
    """
    schedule = getattr(
        vllm_config.speculative_config, "num_speculative_tokens_per_batch_size", None
    )
    if not schedule:
        return drafted, None
    max_seqs = vllm_config.scheduler_config.max_num_seqs
    lookup = build_dynamic_sd_schedule_lookup(
        schedule,
        vllm_max_batch_size=max_seqs,
        vllm_num_speculative_tokens=drafted,
    )
    raw = [max(1, int(k)) for k in lookup]
    cc = getattr(vllm_config, "compilation_config", None)
    req_sizes = getattr(cc, "decode_cudagraph_capture_request_sizes", None)
    if not req_sizes:
        from vllm.config.compilation import decode_cudagraph_capture_grid

        req_sizes = sorted(
            {*range(1, min(max_seqs, 8) + 1), *decode_cudagraph_capture_grid(max_seqs)}
        )
    else:
        req_sizes = sorted({*req_sizes, *range(1, min(max_seqs, 8) + 1)})
    prev_g = 0
    for g in req_sizes:
        if g <= 0 or g > max_seqs:
            continue
        active_in_bucket = [raw[b] for b in range(prev_g + 1, g + 1) if lookup[b] > 0]
        if active_in_bucket:
            raw[g] = max(raw[g], max(active_in_bucket))
        prev_g = g
    return max(1, max(lookup[1:])), tuple(raw)


def scheduled_draft_tokens(vllm_config: VllmConfig, drafted: int) -> int:
    """Most of the ``drafted`` tokens of a block that any batch size verifies."""
    return scheduled_draft_tokens_by_reqs(vllm_config, drafted)[0]


def attention_splits(num_reqs: int) -> int:
    """Key splits of the draft attention for ``num_reqs`` requests."""
    blocks = max(1, num_reqs * 8)
    return max(1, min(_ATTN_MAX_SPLITS, -(-_ATTN_TARGET_BLOCKS // blocks)))


# ---------------------------------------------------------------------------
# Attention metadata of the draft layers
# ---------------------------------------------------------------------------
@dataclass
class Inkling2DraftMetadata(AttentionMetadata):
    block_table: torch.Tensor  # [R, max_blocks] int32
    slot_mapping: torch.Tensor  # [T] int64, negative => no write
    seq_lens: torch.Tensor  # [R] int32, absolute length including the block
    num_reqs: int
    num_actual_tokens: int


class Inkling2DraftMetadataBuilder(AttentionMetadataBuilder[Inkling2DraftMetadata]):
    """References the persistent block table / slot mapping / sequence lengths.

    The draft decode step rewrites those buffers in place between proposals, so a
    FULL graph needs no metadata update (``update_draft_decode_metadata`` is a
    no-op) and every request of a replayed batch is a uniform 8-row block.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Inkling2DraftMetadata:
        c = common_attn_metadata
        return Inkling2DraftMetadata(
            block_table=c.block_table_tensor,
            slot_mapping=c.slot_mapping[: c.num_actual_tokens],
            seq_lens=c.seq_lens[: c.num_reqs],
            num_reqs=c.num_reqs,
            num_actual_tokens=c.num_actual_tokens,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Inkling2DraftMetadata:
        metadata = self.build(0, common_attn_metadata)
        # Capture reads every request: a single key keeps the kernel in range.
        metadata.seq_lens.fill_(1)
        return metadata

    def update_draft_decode_metadata(self, metadata: Inkling2DraftMetadata) -> None:
        pass


class Inkling2DraftBackend(AttentionBackend):
    """Metadata-only backend: the draft's attention runs inside the model."""

    @staticmethod
    def get_name() -> str:
        return "INKLING2_DSPARK"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("The Inkling-2 draft attention runs in the model.")

    @staticmethod
    def get_builder_cls() -> type[Inkling2DraftMetadataBuilder]:
        return Inkling2DraftMetadataBuilder


class Inkling2DraftKVOwner(Inkling2KVOwner):
    """K|V cache of one draft layer, page ``[nb, H, bs, 2 * D]`` (as the target's)."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2DraftBackend


def _param(*shape: int, dtype: torch.dtype = torch.float16) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class Inkling2DraftLayer(nn.Module):
    """One draft layer: fused projections, norms and its KV owner."""

    def __init__(self, config: Inkling2DSparkConfig, layer_idx: int, prefix: str):
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        heads, kv_heads, dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        local = config.is_local_layer(layer_idx)
        self.window = config.sliding_window_size if local else 0
        self.input_ln_w = _param(hidden)
        self.post_ln_w = _param(hidden)
        self.qkv_w = _param((heads + 2 * kv_heads) * dim, hidden)  # q | k | v rows
        self.kvctx_w = _param(2 * kv_heads * dim, hidden)  # k_proj_ctx | v_proj_ctx
        self.o_w = _param(hidden, heads * dim)
        self.q_norm_w = _param(dim)
        self.k_norm_w = _param(dim)
        self.kctx_norm_w = _param(dim)
        self.gate_up_w = _param(2 * inter, hidden)  # rows: gate_0, up_0, gate_1, ...
        self.down_w = _param(hidden, inter)
        self.kv_owner = Inkling2DraftKVOwner(
            num_kv_heads=kv_heads,
            head_dim=dim,
            sliding_window=config.sliding_window_size if local else None,
            prefix=f"{prefix}.self_attn.attn",
        )


class Inkling2MarkovHead(nn.Module):
    """Recurrent Markov head: weights, the derived layouts and the folded head."""

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.dflare_w = _param(1, hidden)
        self.hidden_out_w = _param(hidden, rank)
        self.hidden_out_b = _param(hidden)
        self.w1 = _param(config.padded_vocab_size, rank)
        self.rnn_proj_w = _param(3 * rank, 2 * rank + 2 * hidden)
        self.rnn_bias = _param(3 * rank, dtype=torch.float32)
        # Derived at load: rnn_proj columns [state, w1[prev]] transposed for the
        # recurrence, [h] and [f_rnn] as contiguous GEMM weights, hidden_out^T.
        self.register_buffer("rnn_wt", torch.empty(0, dtype=torch.float16), False)
        self.register_buffer("rnn_wh", torch.empty(0, dtype=torch.float16), False)
        self.register_buffer("rnn_wf", torch.empty(0, dtype=torch.float16), False)
        self.register_buffer("hidden_out_t", torch.empty(0, dtype=torch.float16), False)
        # Derived from the target's LM head in ``attach_target``.
        self.register_buffer("fold_w", torch.empty(0, dtype=torch.float16), False)
        self.register_buffer("fold_b", torch.empty(0, dtype=torch.float32), False)

    def ensure_checkpoint_storage(
        self, config: Inkling2DSparkConfig, device: torch.device
    ) -> None:
        hidden, rank = config.hidden_size, config.markov_rank
        if self.rnn_proj_w.numel() == 0:
            self.rnn_proj_w.data = torch.empty(
                3 * rank, 2 * rank + 2 * hidden, dtype=torch.float16, device=device
            )
        if self.hidden_out_w.numel() == 0:
            self.hidden_out_w.data = torch.empty(
                hidden, rank, dtype=torch.float16, device=device
            )
        if self.hidden_out_b.numel() == 0:
            self.hidden_out_b.data = torch.empty(
                hidden, dtype=torch.float16, device=device
            )

    @torch.no_grad()
    def derive(self, config: Inkling2DSparkConfig) -> None:
        rank, hidden = config.markov_rank, config.hidden_size
        w = self.rnn_proj_w
        dev = w.device
        self.rnn_wt = w[:, : 2 * rank].t().contiguous()
        self.rnn_wh = w[:, 2 * rank : 2 * rank + hidden].contiguous()
        self.rnn_wf = w[:, 2 * rank + hidden :].contiguous()
        self.hidden_out_t = self.hidden_out_w.t().contiguous()
        # Free repacked checkpoint originals (~8.63 MiB on Inkling-2-Mini).
        self.rnn_proj_w.data = torch.empty(0, dtype=torch.float16, device=dev)
        self.hidden_out_w.data = torch.empty(0, dtype=torch.float16, device=dev)

    @torch.no_grad()
    def free_unfolded_head(self) -> None:
        """Free ``hidden_out_t`` and ``hidden_out_b`` after ``attach_target`` folds
        them into ``fold_w`` / ``fold_b`` (~1.13 MiB on Inkling-2-Mini)."""
        dev = self.hidden_out_t.device
        self.hidden_out_t = torch.empty(0, dtype=torch.float16, device=dev)
        self.hidden_out_b.data = torch.empty(0, dtype=torch.float16, device=dev)


class Inkling2DSparkDecoder(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig, prefix: str):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = nn.ModuleList(
            Inkling2DraftLayer(config, i, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.norm_w = _param(hidden)
        self.dflare_w = _param(config.num_hidden_layers, hidden)
        self.fuse_weight = _param(
            config.num_hidden_layers + 1,
            len(config.target_layer_ids),
            dtype=torch.float32,
        )
        self.mask_embed = _param(hidden)
        self.markov_head = Inkling2MarkovHead(config)
        angles = torch.arange(0, config.head_dim, 2, dtype=torch.float64)
        self.register_buffer(
            "inv_freq", config.rope_theta ** (-angles / config.head_dim), False
        )
        # Shared with the target model (``attach_target``).
        self.embed_tokens: nn.Module | None = None
        self.embed_norm: nn.Module | None = None
        self.confidence_head = None


@dataclass
class _DraftVocab:
    """The token ids the greedy draft ranks on this rank (an explicit opt-in, see
    ``speculator_options["draft_vocab"]``) with their local LM-head row indices and
    folded-head rows (zero fp16 LM-head weight copy).  A token outside it is never
    drafted, which can only cost acceptance: the target verifies every draft."""

    ids: torch.Tensor  # int64 [N], strictly ascending global token ids on this rank
    local_idx: torch.Tensor  # int32 [N], row indices into this rank's lm_head.weight
    fold_w: torch.Tensor  # fp16 [N, rank]
    fold_b: torch.Tensor  # fp32 [N]


class _Scratch:
    """Persistent buffers: every launch of a proposal reads and writes these."""

    def __init__(
        self, config: Inkling2DSparkConfig, max_reqs: int, vocab: int, device
    ) -> None:
        hidden, inter = config.hidden_size, config.intermediate_size
        heads, kv_heads, dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        chunk = min(max_reqs, _BACKBONE_REQS)
        rows = BLOCK * chunk
        f16, f32 = torch.float16, torch.float32

        def new(*shape: int, dtype: torch.dtype = f16) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=device)

        self.chunk_reqs = chunk
        self.resid = new(rows, hidden, dtype=f32)
        self.xn = new(rows, hidden)
        self.qkv = new(rows, (heads + 2 * kv_heads) * dim)
        self.q = new(rows, heads * dim)
        self.attn = new(rows, heads * dim)
        self.delta = new(rows, hidden)
        self.partial = new(_RESIDUAL_SPLITS, rows, hidden, dtype=f32)
        self.gate_up = new(rows, 2 * inter)
        self.act = new(rows, inter)
        part = max(
            ops.inkling2_dspark_attention_part_numel(r, kv_heads, attention_splits(r))
            for r in range(1, chunk + 1)
        )
        self.attn_part = new(part, dtype=f32)
        self.out = new(BLOCK * max_reqs, hidden)
        # context K/V precompute
        self.ctx_mix = new(config.num_hidden_layers * _CTX_CHUNK, hidden)
        self.ctx_kv = new(_CTX_CHUNK, 2 * kv_heads * dim)
        # Markov head: a chunk of requests at a time, sized to active head_vocab
        steps = config.draft_block_size - 1
        head_reqs = min(max(max_reqs, 1), _HEAD_REQS)
        head_rows = head_reqs * steps
        vocab_pad = -(-vocab // 8) * 8
        self.head_reqs = head_reqs
        self.head_rows = new(head_rows, hidden)
        self.base = new(head_rows, vocab_pad)
        self.logits = new(head_reqs, vocab_pad)
        self.ctx_h = new(1, head_rows, 3 * config.markov_rank, dtype=f32)
        self.ctx_f = new(1, head_reqs, 3 * config.markov_rank, dtype=f32)
        self.state = new(head_reqs, config.markov_rank, dtype=f32)
        self.latent = new(head_reqs, config.markov_rank)
        # tensor parallelism: this rank's (value, id) partial of the vocabulary shard
        # and the gathered argmax; the anchor embedding rows of a chunk (looked up in
        # the sharded table, then all-reduced) and the row -> anchor index the block
        # embed kernel reads them with
        self.part = new(2 * head_reqs, dtype=torch.float32)
        self.tok = new(head_reqs, dtype=torch.int64)
        # Compact sample_indices buffer when markov_steps < steps so the base
        # LM-head GEMM and rnn_wh GEMV only compute rows that the Markov loop reads.
        self.active_idx = new(head_rows, dtype=torch.int64)
        self.step_pick: dict[int, torch.Tensor] = {
            m: (
                torch.arange(head_reqs, dtype=torch.int64, device=device)[:, None]
                * steps
                + torch.arange(m, dtype=torch.int64, device=device)[None, :]
            ).reshape(-1)
            for m in range(1, steps)
        }
        self.anchor_ids = new(chunk, dtype=torch.int32)
        self.anchor_rows = new(chunk, hidden)
        self.row_anchor = torch.div(
            torch.arange(rows, dtype=torch.int32, device=device),
            BLOCK,
            rounding_mode="floor",
        )

    def ensure_vocab(self, vocab: int) -> None:
        vocab_pad = -(-vocab // 8) * 8
        if self.base.shape[1] < vocab_pad:
            dev = self.base.device
            self.base = torch.empty(
                self.base.shape[0], vocab_pad, dtype=torch.float16, device=dev
            )
            self.logits = torch.empty(
                self.logits.shape[0], vocab_pad, dtype=torch.float16, device=dev
            )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
# checkpoint suffix of a layer tensor -> (layer attribute, first row, row stride)
# for the stacked / interleaved projections; ``None`` rows copy the whole tensor.
_LAYER_KEYS: dict[str, str] = {
    "input_layernorm.weight": "input_ln_w",
    "post_attention_layernorm.weight": "post_ln_w",
    "self_attn.o_proj.weight": "o_w",
    "self_attn.q_norm.weight": "q_norm_w",
    "self_attn.k_norm.weight": "k_norm_w",
    "self_attn.k_norm_ctx.weight": "kctx_norm_w",
    "mlp.down_proj.weight": "down_w",
}
_TOP_KEYS: dict[str, str] = {
    "fuse_weight": "model.fuse_weight",
    "mask_embed": "model.mask_embed",
    "norm.weight": "model.norm_w",
    "markov_head.hidden_out.weight": "model.markov_head.hidden_out_w",
    "markov_head.hidden_out.bias": "model.markov_head.hidden_out_b",
    "markov_head.markov_w1.weight": "model.markov_head.w1",
    "markov_head.rnn_proj.weight": "model.markov_head.rnn_proj_w",
    "markov_head.rnn_proj.bias": "model.markov_head.rnn_bias",
}


class _Fp16Report:
    """Counts what the bf16 -> fp16 conversion of the checkpoint costs."""

    def __init__(self) -> None:
        self.max_abs = 0.0
        self.flushed = 0
        self.values = 0

    def convert(self, name: str, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy ``src`` (bf16) into ``dst`` (fp16 or fp32), checking the range."""
        wide = src.to(device=dst.device, dtype=torch.float32)
        if dst.dtype == torch.float16:
            top = float(wide.abs().max()) if wide.numel() else 0.0
            if not math.isfinite(top) or top > _FP16_MAX:
                raise ValueError(
                    f"{name}: max |w| = {top} does not fit fp16 (max {_FP16_MAX})"
                )
            narrow = wide.to(torch.float16)
            self.max_abs = max(self.max_abs, top)
            self.flushed += int(((wide != 0) & (narrow == 0)).sum())
            self.values += wide.numel()
            dst.copy_(narrow.view_as(dst))
        else:
            dst.copy_(wide.view_as(dst))


class Inkling2DSparkModel(nn.Module):
    """DSpark draft of Inkling-2-Mini, plugged in by ``Inkling2DSparkSpeculator``."""

    speculator_config_class = Inkling2DSparkSpeculatorConfig
    allow_patterns_overrides: ClassVar[list[str]] = ["mtp.safetensors"]
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        assert isinstance(config, Inkling2DSparkConfig), type(config)
        if spec.num_speculative_tokens != config.draft_block_size - 1:
            raise ValueError(
                "Inkling2 DSpark requires num_speculative_tokens = "
                f"{config.draft_block_size - 1} (the checkpoint's block size - 1), "
                f"got {spec.num_speculative_tokens}"
            )
        if vllm_config.model_config.max_model_len > config.max_position_embeddings:
            raise ValueError(
                "Inkling2 DSpark drafts up to max_position_embeddings="
                f"{config.max_position_embeddings} positions; set --max-model-len to "
                "at most that"
            )
        tp = current_tensor_parallel()
        if tp.enabled and spec.draft_sample_method != "greedy":
            raise NotImplementedError(
                "Inkling2 DSpark under tensor parallelism drafts greedily "
                f"(draft_sample_method='greedy'), got {spec.draft_sample_method!r}: "
                "the sampled draft needs the full-vocabulary logits on every rank"
            )
        self.config = config
        self._tp = tp
        self.model = Inkling2DSparkDecoder(config, f"{prefix}model")
        self.lm_head: nn.Module | None = None
        self._greedy_only = getattr(spec, "draft_sample_method", "greedy") == "greedy"
        self._draft_vocab_path = (spec.speculator_options or {}).get("draft_vocab")
        self._draft_vocab: _DraftVocab | None = None
        # this rank's share of the vocabulary: the first id, the rows of its LM head
        # and the rows holding real ids (the rest is the zero padding of the last
        # shard, which must never win an argmax); all of it at TP=1
        self._head_first, self._head_rows, self._head_take = lm_head_shard(
            config.vocab_size, tp.size, tp.rank
        )
        # first id of this rank's rows of the (padded) embedding table
        self._embed_first = lm_head_shard(config.padded_vocab_size, tp.size, tp.rank)[0]
        device = torch.get_default_device()
        if self._draft_vocab_path is not None and self._greedy_only:
            rank_ids, _ = self._load_draft_vocab_ids(
                self._draft_vocab_path, device, log=False
            )
            scratch_vocab = rank_ids.numel()
        else:
            scratch_vocab = self._head_rows
        self._scratch = _Scratch(
            config,
            vllm_config.scheduler_config.max_num_seqs,
            scratch_vocab,
            device,
        )
        self._eps = config.rms_norm_eps
        # Markov steps per draft pass and per-batch-size schedule lookup
        self._markov_steps, self._markov_steps_by_reqs = scheduled_draft_tokens_by_reqs(
            vllm_config, spec.num_speculative_tokens
        )
        # Optional temporary cap used during K-bucketed draft graph capture.
        self._active_markov_cap: int | None = None

    def steps_for_reqs(self, num_reqs: int) -> int:
        """Markov steps needed for a batch of ``num_reqs`` requests."""
        table = self._markov_steps_by_reqs
        if table is None or num_reqs <= 0:
            steps = self._markov_steps
        else:
            steps = min(self._markov_steps, table[min(num_reqs, len(table) - 1)])
        if self._active_markov_cap is not None:
            steps = min(steps, self._active_markov_cap)
        return steps

    # -- weights -----------------------------------------------------------
    def _route(self, name: str) -> tuple[str, slice | None]:
        """Parameter name and row slice receiving checkpoint tensor ``name``."""
        if name in _TOP_KEYS:
            return _TOP_KEYS[name], None
        if name == "markov_head.dflare_norm.weight":
            return "model.markov_head.dflare_w", None
        m = _LAYER_RE.match(name)
        if m is None:
            raise KeyError(f"unmapped Inkling-2 DSpark tensor {name!r}")
        i, key = int(m.group(1)), m.group(2)
        cfg = self.config
        if i >= cfg.num_hidden_layers:
            raise KeyError(f"{name!r}: no layer {i} in the draft")
        base = f"model.layers.{i}."
        q_rows = cfg.num_attention_heads * cfg.head_dim
        kv_rows = cfg.num_key_value_heads * cfg.head_dim
        if key == "dflare_norm.weight":
            return "model.dflare_w", slice(i, i + 1)
        if key in _LAYER_KEYS:
            return base + _LAYER_KEYS[key], None
        stacked = {
            "self_attn.q_proj.weight": ("qkv_w", slice(0, q_rows)),
            "self_attn.k_proj.weight": ("qkv_w", slice(q_rows, q_rows + kv_rows)),
            "self_attn.v_proj.weight": (
                "qkv_w",
                slice(q_rows + kv_rows, q_rows + 2 * kv_rows),
            ),
            "self_attn.k_proj_ctx.weight": ("kvctx_w", slice(0, kv_rows)),
            "self_attn.v_proj_ctx.weight": ("kvctx_w", slice(kv_rows, 2 * kv_rows)),
            "mlp.gate_proj.weight": ("gate_up_w", slice(0, None, 2)),
            "mlp.up_proj.weight": ("gate_up_w", slice(1, None, 2)),
        }
        if key not in stacked:
            raise KeyError(f"unmapped Inkling-2 DSpark tensor {name!r}")
        attr, rows = stacked[key]
        return base + attr, rows

    def _pieces(self) -> dict[str, int]:
        """Checkpoint tensors each parameter receives (stacked ones take several)."""
        pieces = {name: 1 for name, _ in self.named_parameters()}
        for i in range(self.config.num_hidden_layers):
            base = f"model.layers.{i}."
            pieces[base + "qkv_w"] = 3
            pieces[base + "kvctx_w"] = 2
            pieces[base + "gate_up_w"] = 2
        pieces["model.dflare_w"] = self.config.num_hidden_layers
        return pieces

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.model.markov_head.ensure_checkpoint_storage(
            self.config, self.model.norm_w.device
        )
        params = dict(self.named_parameters())
        report = _Fp16Report()
        received: dict[str, int] = {}
        for name, tensor in weights:
            dst_name, rows = self._route(name)
            view = params[dst_name] if rows is None else params[dst_name][rows]
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"{name}: expected bf16, got {tensor.dtype}")
            if tensor.numel() != view.numel() or (
                tensor.dim() > 1 and tensor.shape != view.shape
            ):
                raise ValueError(
                    f"{name}: checkpoint {tuple(tensor.shape)} does not fit "
                    f"{dst_name} {tuple(view.shape)}"
                )
            report.convert(name, tensor, view)
            received[dst_name] = received.get(dst_name, 0) + 1
        wrong = sorted(
            n for n, count in self._pieces().items() if received.get(n, 0) != count
        )
        if wrong:
            raise ValueError(
                "the Inkling-2 DSpark checkpoint is incomplete or repeats tensors "
                f"for: {wrong[:8]}"
            )
        self.model.markov_head.derive(self.config)
        logger.info(
            "Inkling-2 DSpark weights: bf16 -> fp16, max |w| %.3f, %d of %d values "
            "flushed to zero (below the fp16 range)",
            report.max_abs,
            report.flushed,
            report.values,
        )
        return set(received)

    # -- the target's embedding / LM head ------------------------------------
    @torch.no_grad()
    def attach_target(self, target_model: nn.Module) -> None:
        """Finish sharing the target's embedding norm and LM head (``embed_tokens``
        and ``lm_head`` are aliased by ``load_dspark_model``) and fold ``hidden_out``
        into the head."""
        assert self.lm_head is not None, "the target LM head was not shared"
        assert self.model.embed_tokens is not None, "the target embedding is missing"
        rows = self.lm_head.weight.shape[0]
        if rows != self._head_rows:
            raise ValueError(
                f"the target's LM head has {rows} rows on this rank, the draft's "
                f"vocabulary shard {self._head_rows}"
            )
        self.model.embed_norm = target_model.model.embed_norm
        self.refresh_derived_weights()
        self.model.markov_head.free_unfolded_head()

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        """Re-derive the weights computed from the target's LM head (also after the
        target's weights are replaced): ``fold_w = lm_head @ hidden_out.W`` and
        ``fold_b = lm_head @ hidden_out.b``, fp32 accumulation.  Under greedy
        sampling with ``draft_vocab``, only the subset rows of ``fold_w`` / ``fold_b``
        are materialized (~16.1 MiB instead of ~114.6 MiB on Inkling-2-Mini TP=1)."""
        assert self.lm_head is not None
        head = self.model.markov_head
        lm = self.lm_head.weight
        vocab = lm.shape[0]
        if self._draft_vocab_path is not None and self._greedy_only:
            if head.fold_w.numel() != 0:
                head.fold_w = torch.empty(0, dtype=torch.float16, device=lm.device)
                head.fold_b = torch.empty(0, dtype=torch.float32, device=lm.device)
            self._draft_vocab = self._derive_draft_vocab(self._draft_vocab_path)
            self._scratch.ensure_vocab(self._draft_vocab.ids.numel())
            return

        self._draft_vocab = None
        self._scratch.ensure_vocab(vocab)
        if head.fold_w.shape[0] != vocab:
            head.fold_w = torch.empty(
                vocab, self.config.markov_rank, dtype=torch.float16, device=lm.device
            )
            head.fold_b = torch.empty(vocab, dtype=torch.float32, device=lm.device)
        if lm.is_cuda:
            ops.inkling2_dspark_gemm(lm, head.hidden_out_t, head.fold_w, vocab, 0)
            ops.inkling2_dspark_fold_bias(lm, head.hidden_out_b, head.fold_b)
        else:
            head.fold_w.copy_(
                (lm.float() @ head.hidden_out_t.float().t()).to(torch.float16)
            )
            head.fold_b.copy_(lm.float() @ head.hidden_out_b.float())
        if self._draft_vocab_path is not None:
            self._draft_vocab = self._derive_draft_vocab(self._draft_vocab_path)

    def _load_draft_vocab_ids(
        self, path: str, device: torch.device, *, log: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate ``path`` and return ``(rank_ids [int64], local_idx [int32])``."""
        ids = torch.from_numpy(np.load(path).astype(np.int64)).to(device)
        vocab_limit = self.config.vocab_size
        if (
            ids.dim() != 1
            or ids.numel() == 0
            or int(ids[0]) < 0
            or int(ids[-1]) >= vocab_limit
            or not bool((ids[1:] > ids[:-1]).all())
        ):
            raise ValueError(
                f"{path}: the draft vocabulary must be a vector of strictly "
                f"ascending token ids below {vocab_limit}"
            )
        lo = self._head_first
        hi = lo + self._head_take
        if self._tp.enabled:
            rank_ids = ids[(ids >= lo) & (ids < hi)]
            local_idx = (rank_ids - lo).to(torch.int32)
            if log:
                logger.info(
                    "Inkling-2 DSpark draft vocabulary (TP rank %d/%d): %d local of %d "
                    "listed ids (shard [%d, %d), %s)",
                    self._tp.rank,
                    self._tp.size,
                    rank_ids.numel(),
                    ids.numel(),
                    lo,
                    hi,
                    path,
                )
        else:
            rank_ids = ids
            local_idx = ids.to(torch.int32)
            if log:
                logger.info(
                    "Inkling-2 DSpark draft vocabulary: %d of %d token ids (%s)",
                    ids.numel(),
                    vocab_limit,
                    path,
                )
        return rank_ids, local_idx

    @torch.no_grad()
    def _derive_draft_vocab(self, path: str) -> _DraftVocab:
        """The local LM-head row indices and folded-head rows of the ids listed in
        ``path`` (an ``.npy`` of strictly ascending token ids, made offline by
        ``tools/rdna/inkling2/draft_vocab.py``), filtered to this TP rank's shard."""
        assert self.lm_head is not None
        head, lm = self.model.markov_head, self.lm_head.weight
        rank_ids, local_idx = self._load_draft_vocab_ids(path, lm.device, log=True)
        n_sub = rank_ids.numel()
        rank = self.config.markov_rank
        if n_sub == 0:
            fold_w = torch.empty((0, rank), dtype=torch.float16, device=lm.device)
            fold_b = torch.empty((0,), dtype=torch.float32, device=lm.device)
        elif head.fold_w.shape[0] == lm.shape[0]:
            local_i64 = local_idx.to(torch.int64)
            fold_w = head.fold_w.index_select(0, local_i64)
            fold_b = head.fold_b.index_select(0, local_i64)
        else:
            fold_w = torch.empty((n_sub, rank), dtype=torch.float16, device=lm.device)
            fold_b = torch.empty((n_sub,), dtype=torch.float32, device=lm.device)
            chunk = 4096
            for i0 in range(0, n_sub, chunk):
                n = min(chunk, n_sub - i0)
                lm_sub = lm.index_select(0, local_idx[i0 : i0 + n].to(torch.int64))
                if lm.is_cuda:
                    ops.inkling2_dspark_gemm(
                        lm_sub, head.hidden_out_t, fold_w[i0 : i0 + n], n, 0
                    )
                    ops.inkling2_dspark_fold_bias(
                        lm_sub, head.hidden_out_b, fold_b[i0 : i0 + n]
                    )
                else:
                    fold_w[i0 : i0 + n].copy_(
                        (lm_sub.float() @ head.hidden_out_t.float().t()).to(
                            torch.float16
                        )
                    )
                    fold_b[i0 : i0 + n].copy_(
                        lm_sub.float() @ head.hidden_out_b.float()
                    )
        return _DraftVocab(rank_ids, local_idx, fold_w, fold_b)

    # -- speculator interface ------------------------------------------------
    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.kv_owner.prefix for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.model.layers)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def compute_f_rnn(
        self,
        taps: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor,
        out: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """``out[i]`` = Markov-head DFlare mix of the last committed row of request
        ``i`` (``query_start_loc[i + 1] - num_rejected[i] - 1``)."""
        if num_rejected.dtype != torch.int32:
            num_rejected = num_rejected.to(torch.int32)
        ops.inkling2_dspark_mix_rms(
            taps,
            self.model.fuse_weight,
            self.model.markov_head.dflare_w,
            out,
            query_start_loc,
            num_rejected,
            num_reqs,
            self.config.num_hidden_layers,
            self._eps,
        )

    @torch.no_grad()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | torch.Tensor | None = None,
    ) -> None:
        """Write the K/V of the committed context rows into every draft layer's
        cache: per layer the DFlare mix of the taps, ``k_proj_ctx`` / ``v_proj_ctx``,
        ``k_norm_ctx`` and RoPE.  Rows whose slot is negative write nothing."""
        if context_slot_mappings is None:  # profiling: no cache is bound
            return
        dec, cfg, s = self.model, self.config, self._scratch
        layers = len(dec.layers)
        for t0 in range(0, context_states.shape[0], _CTX_CHUNK):
            n = min(_CTX_CHUNK, context_states.shape[0] - t0)
            mixed = s.ctx_mix[: layers * n]
            ops.inkling2_dspark_mix_rms(
                context_states[t0 : t0 + n],
                dec.fuse_weight,
                dec.dflare_w,
                mixed,
                None,
                None,
                n,
                0,
                self._eps,
            )
            for i, layer in enumerate(dec.layers):
                slots = (
                    context_slot_mappings[i]
                    if isinstance(context_slot_mappings, list)
                    else context_slot_mappings
                )
                assert slots is not None
                kv = s.ctx_kv[:n]
                _linear(mixed[i * n : (i + 1) * n], layer.kvctx_w, kv, n)
                ops.inkling2_dspark_qkv_prep(
                    kv,
                    context_positions[t0 : t0 + n],
                    slots[t0 : t0 + n],
                    layer.q_norm_w,
                    layer.kctx_norm_w,
                    dec.inv_freq,
                    None,
                    layer.kv_owner.kv_cache,
                    n,
                    0,
                    cfg.num_key_value_heads,
                    self._eps,
                )

    def _layer_metadata(self) -> list[Inkling2DraftMetadata | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return [None] * len(self.model.layers)
        return cast(
            "list[Inkling2DraftMetadata | None]",
            [attn_metadata[layer.kv_owner.prefix] for layer in self.model.layers],
        )

    def _residual_projection(
        self, x: torch.Tensor, w: torch.Tensor, rows: int
    ) -> tuple[torch.Tensor, int]:
        """The projection feeding a residual add: fp32 split-K partials (summed by
        ``add_rmsnorm``) on the skinny path, an fp16 tensor on the dense path."""
        s = self._scratch
        if rows <= _GEMV_MAX_ROWS:
            ops.inkling2_dspark_gemv(
                x, w, s.partial[:_RESIDUAL_SPLITS], None, rows, _RESIDUAL_SPLITS, 0
            )
            return s.partial[:_RESIDUAL_SPLITS], _RESIDUAL_SPLITS
        ops.inkling2_dspark_gemm(x, w, s.delta, rows, _gemm_design(rows))
        return s.delta, 0

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hidden states ``[T, H]`` (fp16, final norm applied) of the draft blocks.

        ``T`` is a multiple of 8: request ``i`` owns rows ``8 i .. 8 i + 7`` (the
        anchor token, then seven masks).
        """
        assert inputs_embeds is None, "the draft embeds the block itself"
        num_tokens = input_ids.shape[0]
        assert num_tokens % BLOCK == 0, num_tokens
        num_reqs = num_tokens // BLOCK
        metadata = self._layer_metadata()
        chunk = self._scratch.chunk_reqs
        for r0 in range(0, num_reqs, chunk):
            self._backbone(
                input_ids, positions, metadata, r0, min(chunk, num_reqs - r0)
            )
        return self._scratch.out[:num_tokens]

    def _backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: list[Inkling2DraftMetadata | None],
        r0: int,
        nr: int,
    ) -> None:
        dec, cfg, s, eps = self.model, self.config, self._scratch, self._eps
        assert dec.embed_tokens is not None and dec.embed_norm is not None
        t0, rows = r0 * BLOCK, nr * BLOCK
        pos = positions[t0 : t0 + rows]
        resid, xn = s.resid[:rows], s.xn[:rows]
        layers = dec.layers
        ids, table = input_ids[t0 : t0 + rows], dec.embed_tokens.weight
        if self._tp.enabled:
            # The table is vocabulary-sharded: look up the anchors this rank owns
            # (zero rows for the ids of other ranks), one all-reduce adds the owner's
            # row back (x + 0 is exact, so the rows equal the TP=1 ones), and the
            # block embed reads row ``i`` of the reduced table for the anchor of
            # request ``i``.
            anchors = s.anchor_ids[:nr]
            anchors.copy_(input_ids[t0 : t0 + rows : BLOCK])
            ops.inkling2_glue_embed_lookup(
                anchors,
                table,
                s.anchor_rows[:nr],
                self._embed_first,
                cfg.padded_vocab_size,
            )
            table = self._tp.all_reduce(s.anchor_rows[:nr])
            ids = s.row_anchor[:rows]
        ops.inkling2_dspark_embed_block(
            ids,
            table,
            dec.embed_norm.weight,
            dec.mask_embed,
            layers[0].input_ln_w,
            resid,
            xn,
            rows,
            BLOCK,
            eps,
        )
        splits = attention_splits(nr)
        for i, layer in enumerate(layers):
            md = metadata[i]
            if md is None:  # profiling / dummy run: no cache to attend over
                s.attn[:rows].zero_()
            else:
                _linear(xn, layer.qkv_w, s.qkv, rows)
                cache = layer.kv_owner.kv_cache
                ops.inkling2_dspark_qkv_prep(
                    s.qkv,
                    pos,
                    md.slot_mapping[t0 : t0 + rows],
                    layer.q_norm_w,
                    layer.k_norm_w,
                    dec.inv_freq,
                    s.q,
                    cache,
                    rows,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    eps,
                )
                ops.inkling2_dspark_attention(
                    s.q,
                    cache,
                    md.block_table[r0 : r0 + nr],
                    md.seq_lens[r0 : r0 + nr],
                    s.attn,
                    s.attn_part,
                    nr,
                    layer.window,
                    splits,
                )
            delta, n_splits = self._residual_projection(s.attn, layer.o_w, rows)
            ops.inkling2_dspark_add_rmsnorm(
                resid, delta, layer.post_ln_w, xn, rows, n_splits, eps
            )
            _linear(xn, layer.gate_up_w, s.gate_up, rows)
            ops.inkling2_dspark_silu_mul(s.gate_up, s.act, rows)
            delta, n_splits = self._residual_projection(s.act, layer.down_w, rows)
            last = i + 1 == len(layers)
            ops.inkling2_dspark_add_rmsnorm(
                resid,
                delta,
                dec.norm_w if last else layers[i + 1].input_ln_w,
                s.out[t0 : t0 + rows] if last else xn,
                rows,
                n_splits,
                eps,
            )

    # -- the Markov head -------------------------------------------------------
    @torch.no_grad()
    def sample_sequential(
        self, spec: Inkling2DSparkSpeculator, num_reqs: int, head_hidden: torch.Tensor
    ) -> bool:
        """Draft the block's tokens left to right with the recurrent Markov head.

        Per chunk of requests: one LM-head GEMM for the base logits of all seven
        positions, the context terms of the recurrence (``h`` per position,
        ``f_rnn`` per request), then seven times: recurrence step, folded-head
        logits (softcapped, fp16) and sampling.  Greedy drafting takes the argmax in
        a native kernel; probabilistic drafting uses the runner's sampler (which also
        caches the logits for rejection).

        Returns:
            True: the whole chain ran here.

        """
        assert self.lm_head is not None
        cfg, s = self.config, self._scratch
        head = self.model.markov_head
        steps = spec.num_speculative_steps
        markov_steps = min(steps, self.steps_for_reqs(num_reqs))
        stride = spec.num_query_per_req
        weight = self.lm_head.weight
        vocab, take = weight.shape[0], self._head_take
        w_row_ids: torch.Tensor | None = None
        greedy = spec.draft_logits is None
        # with the draft vocabulary the greedy chain sweeps only its rows in place
        ranked = self._draft_vocab if greedy else None
        if ranked is not None:
            vocab = ranked.ids.numel()
            w_row_ids = ranked.local_idx
            fold_w, fold_b = ranked.fold_w, ranked.fold_b
        else:
            fold_w, fold_b = head.fold_w[:take], head.fold_b[:take]
        sample_rows = num_reqs * steps
        idx_map = spec.sample_idx_mapping[:sample_rows].view(num_reqs, steps)
        sample_pos = spec.sample_pos[:sample_rows].view(num_reqs, steps)
        head_chunk = s.head_reqs
        for c0 in range(0, num_reqs, head_chunk):
            rc = min(head_chunk, num_reqs - c0)
            rows = rc * markov_steps
            if markov_steps == steps:
                idx = spec.sample_indices[c0 * steps : c0 * steps + rows]
            elif rc == 1:
                idx = spec.sample_indices[c0 * steps : c0 * steps + markov_steps]
            else:
                idx_full = spec.sample_indices[c0 * steps : (c0 + rc) * steps]
                idx = s.active_idx[:rows]
                torch.index_select(
                    idx_full, 0, s.step_pick[markov_steps][:rows], out=idx
                )
            ops.inkling2_dspark_gather_rows(head_hidden, idx, s.head_rows, rows)
            if vocab > 0:
                _linear(s.head_rows, weight, s.base[:rows, :vocab], rows, w_row_ids)
            for r0 in range(0, rows, _GEMV_MAX_ROWS):
                m = min(_GEMV_MAX_ROWS, rows - r0)
                ops.inkling2_dspark_gemv(
                    head_hidden,
                    head.rnn_wh,
                    s.ctx_h[:, r0 : r0 + m],
                    idx[r0 : r0 + m],
                    m,
                    1,
                    0,
                )
            ops.inkling2_dspark_gemv(
                spec.f_rnn[c0 : c0 + rc], head.rnn_wf, s.ctx_f[:, :rc], None, rc, 1, 0
            )
            for i in range(markov_steps):
                if i == 0:  # the anchor token of each block
                    prev, prev_stride = (
                        spec.input_buffers.input_ids[c0 * stride :],
                        stride,
                    )
                else:
                    prev = spec.draft_tokens[c0:, i - 1]
                    prev_stride = spec.draft_tokens.stride(0)
                ops.inkling2_dspark_markov_step(
                    prev,
                    prev_stride,
                    s.state,
                    head.w1,
                    head.rnn_wt,
                    head.rnn_bias,
                    s.ctx_h[0],
                    s.ctx_f[0],
                    s.latent,
                    rc,
                    i,
                    markov_steps,
                    i == 0,
                )
                if vocab > 0:
                    ops.inkling2_dspark_head_logits(
                        s.base,
                        s.latent,
                        fold_w,
                        fold_b,
                        s.logits,
                        rc,
                        i,
                        markov_steps,
                        cfg.final_logit_softcapping,
                    )
                out = spec.draft_tokens[c0 : c0 + rc, i]
                if not greedy:
                    out.copy_(
                        spec._sample_logits(
                            s.logits[:rc, :vocab],
                            idx_map[c0 : c0 + rc, i],
                            sample_pos[c0 : c0 + rc, i],
                            i,
                        )
                    )
                elif self._tp.enabled:
                    out.copy_(self._argmax_across_ranks(rc, ranked))
                elif ranked is not None:
                    ops.inkling2_dspark_argmax(s.logits, out, rc, vocab, ranked.ids)
                else:
                    ops.inkling2_dspark_argmax(s.logits, out, rc, vocab)
        return True

    def _argmax_across_ranks(
        self, rows: int, ranked: _DraftVocab | None = None
    ) -> torch.Tensor:
        """Global argmax of the first ``rows`` rows of the logits (int64 ``[rows]``).

        This rank's logits cover its vocabulary shard (or its shard of
        ``draft_vocab`` when ``ranked`` is set): its ``(value, global id)`` partial
        over the real rows is all-gathered with the other ranks' and reduced by the
        kernel ``greedy_token_ids`` uses, so the tie rule (lowest id) is the
        single-rank one and every rank ends with the same token.
        """
        s = self._scratch
        part = s.part[: 2 * rows]
        if ranked is None:
            ops.inkling2_dspark_argmax_partial(
                s.logits, part, rows, self._head_take, self._head_first
            )
        elif ranked.ids.numel() == 0:
            part[:rows].fill_(float("-inf"))
            part[rows : 2 * rows].view(torch.int32).fill_(0x7FFFFFFF)
        else:
            ops.inkling2_dspark_argmax_partial(
                s.logits, part, rows, ranked.ids.numel(), 0, ranked.ids
            )
        gathered = self._tp.all_gather(part.view(2 * rows, 1))  # [2 * rows, tp]
        out = s.tok[:rows]
        ops.inkling2_linear_argmax_reduce(
            gathered[:rows], gathered[rows:].view(torch.int32), out
        )
        return out


class Inkling2DSparkSpeculator(DSparkSpeculator):
    """DSpark speculator of the Inkling-2 draft.

    Stores the raw target taps (each draft layer mixes them itself) as bf16 bits,
    gathers the last committed row for the Markov head and runs the draft's own
    sequential sampling.
    """

    _speculator_name = "Inkling2 DSpark"
    uses_hip_prepare_inputs = True  # the HIP port of the runner's Triton input prep

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        if self._draft_topk is not None or self.enable_adaptive_verification:
            raise NotImplementedError(
                "the Inkling-2 DSpark draft supports neither dspark_draft_topk nor "
                "enable_adaptive_verification (its Markov head keeps no per-position "
                "draft distribution); use speculator_options={'adaptive_k': {}}"
            )
        taps = len(config.target_layer_ids)
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            taps * config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        # The taps are stored eagerly, so the FULL graph stages nothing.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=torch.float16, device=device
        )
        # Secondary b=1 FULL draft graph capped at K=3 Markov steps when
        # adaptive_k is enabled (saves -0.53 ms full vocab / -0.20 ms 32k vocab
        # on K<=3 steps with 0-ULP identical draft tokens 0..K-1, ~2 MiB VRAM).
        self._b1_short_cap: int = 3
        self._b1_short_graph: torch.cuda.CUDAGraph | None = None

    def capture(self) -> None:
        super().capture()
        self._b1_short_graph = None
        spec_cfg = self.vllm_config.speculative_config
        mgr = self.query_cudagraph_manager
        if (
            spec_cfg is None
            or not getattr(spec_cfg, "uses_adaptive_num_tokens", False)
            or mgr is None
            or mgr.cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
            or self.model.steps_for_reqs(1) <= self._b1_short_cap
            or self.dp_size > 1
        ):
            return
        from vllm.distributed.parallel_state import graph_capture
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID
        from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
            _prepare_dflash_inputs_to_capture,
        )

        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        self._context_slot_mappings.fill_(PAD_SLOT_ID)
        attn_metadata, slot_mappings = _prepare_dflash_inputs_to_capture(
            1,
            self.num_query_per_req,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            skip_attn=False,
            causal=self._group_causal,
        )

        def _run_short() -> None:
            self._prepare_graph_forward(1, attn_metadata)
            self._generate_draft(
                1,
                self.num_query_per_req,
                attn_metadata,
                slot_mappings,
                None,
                CUDAGraphMode.FULL,
            )

        self.model._active_markov_cap = self._b1_short_cap
        try:
            with graph_capture(self.device) as gc_ctx:
                stream = mgr.capture_stream or gc_ctx.stream
                with torch.cuda.stream(stream):
                    _run_short()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=mgr.pool, stream=stream):
                        _run_short()
            self._b1_short_graph = g
            logger.info(
                "Inkling-2 DSpark: captured b=1 K<=%d draft graph bucket",
                self._b1_short_cap,
            )
        finally:
            self.model._active_markov_cap = None

    def _replay_fullgraph(
        self,
        batch_desc: Any,
        num_speculative_tokens: int | None = None,
    ) -> None:
        if (
            self._b1_short_graph is not None
            and batch_desc.num_reqs == 1
            and num_speculative_tokens is not None
            and 0 < num_speculative_tokens <= self._b1_short_cap
        ):
            self._b1_short_graph.replay()
            return
        super()._replay_fullgraph(batch_desc, num_speculative_tokens)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # The folded head writes fp16 logits (the target's head is fp32).
        return torch.float16, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        model.attach_target(language_model)
        return model

    def _store_hidden_states(
        self,
        num_tokens: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_graph_tokens: int = 0,
    ) -> None:
        if not aux_hidden_states:
            raise RuntimeError(
                "the Inkling-2 DSpark draft needs the target's four taps; the "
                "target did not return auxiliary hidden states"
            )
        ops.inkling2_dspark_store_taps(
            list(aux_hidden_states), self.hidden_states, num_tokens
        )

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        self.model.compute_f_rnn(
            self.hidden_states, query_start_loc, num_rejected, self.f_rnn, num_reqs
        )
