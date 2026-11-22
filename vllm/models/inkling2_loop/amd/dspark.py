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
from typing import ClassVar, cast

import regex as re
import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
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


def _linear(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, rows: int) -> None:
    """``out[:rows] = x[:rows] @ w.T`` with fp16 operands and fp32 accumulation."""
    if rows <= _GEMV_MAX_ROWS:
        ops.inkling2_dspark_gemv(x, w, out, None, rows, 0, 0)
    else:
        ops.inkling2_dspark_gemm(x, w, out, rows, _gemm_design(rows))


def scheduled_draft_tokens(vllm_config: VllmConfig, drafted: int) -> int:
    """Most of the ``drafted`` tokens of a block that any batch size verifies.

    ``num_speculative_tokens_per_batch_size`` makes the scheduler verify the first
    ``k(batch size)`` of the block's tokens.  The chain is sequential (token ``i``
    never depends on a later one), so the Markov steps beyond the schedule's
    largest ``k`` produce tokens nothing consumes.  Without a schedule every drafted
    token can be verified.
    """
    schedule = getattr(
        vllm_config.speculative_config, "num_speculative_tokens_per_batch_size", None
    )
    if not schedule:
        return drafted
    lookup = build_dynamic_sd_schedule_lookup(
        schedule,
        vllm_max_batch_size=vllm_config.scheduler_config.max_num_seqs,
        vllm_num_speculative_tokens=drafted,
    )
    return max(1, max(lookup[1:]))


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

    @torch.no_grad()
    def derive(self, config: Inkling2DSparkConfig) -> None:
        rank, hidden = config.markov_rank, config.hidden_size
        w = self.rnn_proj_w
        self.rnn_wt = w[:, : 2 * rank].t().contiguous()
        self.rnn_wh = w[:, 2 * rank : 2 * rank + hidden].contiguous()
        self.rnn_wf = w[:, 2 * rank + hidden :].contiguous()
        self.hidden_out_t = self.hidden_out_w.t().contiguous()


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
        # Markov head: a chunk of requests at a time
        steps = config.draft_block_size - 1
        head_rows = _HEAD_REQS * steps
        vocab_pad = -(-vocab // 8) * 8
        self.head_rows = new(head_rows, hidden)
        self.base = new(head_rows, vocab_pad)
        self.logits = new(_HEAD_REQS, vocab_pad)
        self.ctx_h = new(1, head_rows, 3 * config.markov_rank, dtype=f32)
        self.ctx_f = new(1, _HEAD_REQS, 3 * config.markov_rank, dtype=f32)
        self.state = new(_HEAD_REQS, config.markov_rank, dtype=f32)
        self.latent = new(_HEAD_REQS, config.markov_rank)
        # tensor parallelism: this rank's (value, id) partial of the vocabulary shard
        # and the gathered argmax; the anchor embedding rows of a chunk (looked up in
        # the sharded table, then all-reduced) and the row -> anchor index the block
        # embed kernel reads them with
        self.part = new(2 * _HEAD_REQS, dtype=f32)
        self.tok = new(_HEAD_REQS, dtype=torch.int64)
        self.anchor_ids = new(chunk, dtype=torch.int32)
        self.anchor_rows = new(chunk, hidden)
        self.row_anchor = torch.div(
            torch.arange(rows, dtype=torch.int32, device=device),
            BLOCK,
            rounding_mode="floor",
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
        # this rank's share of the vocabulary: the first id, the rows of its LM head
        # and the rows holding real ids (the rest is the zero padding of the last
        # shard, which must never win an argmax); all of it at TP=1
        self._head_first, self._head_rows, self._head_take = lm_head_shard(
            config.vocab_size, tp.size, tp.rank
        )
        # first id of this rank's rows of the (padded) embedding table
        self._embed_first = lm_head_shard(config.padded_vocab_size, tp.size, tp.rank)[0]
        self._scratch = _Scratch(
            config,
            vllm_config.scheduler_config.max_num_seqs,
            self._head_rows,
            torch.get_default_device(),
        )
        self._eps = config.rms_norm_eps
        # Markov steps per draft pass (the rest of the block's tokens are never used)
        self._markov_steps = scheduled_draft_tokens(
            vllm_config, spec.num_speculative_tokens
        )

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

    @torch.no_grad()
    def refresh_derived_weights(self) -> None:
        """Re-derive the weights computed from the target's LM head (also after the
        target's weights are replaced): ``fold_w = lm_head @ hidden_out.W`` and
        ``fold_b = lm_head @ hidden_out.b``, fp32 accumulation."""
        assert self.lm_head is not None
        head = self.model.markov_head
        lm = self.lm_head.weight
        vocab = lm.shape[0]
        if head.fold_w.shape[0] != vocab:
            head.fold_w = torch.empty(
                vocab, self.config.markov_rank, dtype=torch.float16, device=lm.device
            )
            head.fold_b = torch.empty(vocab, dtype=torch.float32, device=lm.device)
        ops.inkling2_dspark_gemm(lm, head.hidden_out_t, head.fold_w, vocab, 0)
        ops.inkling2_dspark_fold_bias(lm, head.hidden_out_b, head.fold_b)

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
        markov_steps = min(steps, self._markov_steps)  # ``steps`` stays the layout
        stride = spec.num_query_per_req
        weight = self.lm_head.weight
        vocab, take = weight.shape[0], self._head_take
        # the folded head of the rows holding real ids (all of them at TP=1)
        fold_w, fold_b = head.fold_w[:take], head.fold_b[:take]
        greedy = spec.draft_logits is None
        sample_rows = num_reqs * steps
        idx_map = spec.sample_idx_mapping[:sample_rows].view(num_reqs, steps)
        sample_pos = spec.sample_pos[:sample_rows].view(num_reqs, steps)
        for c0 in range(0, num_reqs, _HEAD_REQS):
            rc = min(_HEAD_REQS, num_reqs - c0)
            rows = rc * steps
            idx = spec.sample_indices[c0 * steps : c0 * steps + rows]
            ops.inkling2_dspark_gather_rows(head_hidden, idx, s.head_rows, rows)
            _linear(s.head_rows, weight, s.base[:rows, :vocab], rows)
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
                    steps,
                    i == 0,
                )
                ops.inkling2_dspark_head_logits(
                    s.base,
                    s.latent,
                    fold_w,
                    fold_b,
                    s.logits,
                    rc,
                    i,
                    steps,
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
                    out.copy_(self._argmax_across_ranks(rc))
                else:
                    ops.inkling2_dspark_argmax(s.logits, out, rc, vocab)
        return True

    def _argmax_across_ranks(self, rows: int) -> torch.Tensor:
        """Global argmax of the first ``rows`` rows of the logits (int64 ``[rows]``).

        This rank's logits cover its vocabulary shard: its ``(value, global id)``
        partial over the real rows is all-gathered with the other ranks' and reduced
        by the kernel ``greedy_token_ids`` uses, so the tie rule (lowest id) is the
        single-rank one and every rank ends with the same token.
        """
        s = self._scratch
        part = s.part[: 2 * rows]
        ops.inkling2_dspark_argmax_partial(
            s.logits, part, rows, self._head_take, self._head_first
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
        assert self._draft_topk is None and not self.enable_adaptive_verification
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
