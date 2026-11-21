# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini (``inkling2_loop``) text model for RDNA2 / ROCm.

Control flow mirrors ``Inkling2LoopTextModel`` of the transformers reference:
layers 0-5 run once, the looped core (layers 6-17) runs ``loop_num_passes``
times (passes >= 1 restart from the pre-loop state and receive per-slot
messages from the previous pass), layers 18-23 run once, then a final
attention-residual readout.  The residual stream and block states are kept in
fp32 between layers (the reference rounds them to bf16; fp16 cannot hold the
same range).  Attention and MoE inputs are cast to the serving dtype (fp16).

Quantised experts (NVFP4 layers 0-19, MXFP8 layers 20-23) run through the
RDNA2 ``V_DOT2`` kernels (W4A16 / W8A16); see ``moe.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..configs import Inkling2LoopConfig, Inkling2LoopTextConfig
from . import blocks, loader
from .attention import (
    Inkling2Attention,
    StaticSconv,
    get_metadata,
)
from .moe import Inkling2LoopMoE
from .state import (
    STREAM_ATTN,
    STREAM_MLP,
    Inkling2ConvOwner,
    Inkling2KVOwner,
    Inkling2Metadata,
    conv_taps,
)

logger = init_logger(__name__)


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        layer_idx: int,
        quant_format: loader.ExpertFormat,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        H = config.hidden_size
        self.is_full = not config.is_sliding_layer(layer_idx)
        self.is_loop = (
            config.loop_layer_index_start <= layer_idx < config.loop_layer_index_end
        )
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        eps = config.rms_norm_eps

        self.attention_input = (
            blocks.BlockAttentionResidual(config, 3 if self.is_full else 1)
            if layer_idx > 0
            else None
        )
        self.mlp_input = blocks.BlockAttentionResidual(config)
        self.self_attn = Inkling2Attention(config, layer_idx)
        self.mlp = Inkling2LoopMoE(config, quant_format, prefix=f"{prefix}.mlp")
        if self.is_full:
            self.q_input_layernorm = blocks.Inkling2RMSNorm(H, eps)
            self.k_input_layernorm = blocks.Inkling2RMSNorm(H, eps)
            self.v_input_layernorm = blocks.Inkling2RMSNorm(H, eps)
        else:
            self.input_layernorm = blocks.Inkling2RMSNorm(H, eps)
        self.post_attention_layernorm = blocks.Inkling2RMSNorm(H, eps)
        self.post_attention_convolution = StaticSconv(H, config.sconv_kernel_size)
        self.post_mlp_convolution = StaticSconv(H, config.sconv_kernel_size)
        if self.is_loop:
            self.q_message_gate = blocks.MessageGate(config)
            self.k_message_gate = blocks.MessageGate(config)
            self.v_message_gate = blocks.MessageGate(config)
            self.mlp_message_gate = blocks.MessageGate(config)

        # One KV owner and one conv owner per visit of this layer.
        visits = config.loop_num_passes if self.is_loop else 1
        nkv = self.self_attn.num_kv_heads
        D = self.self_attn.head_dim
        self.kv_owners = nn.ModuleList(
            Inkling2KVOwner(
                num_kv_heads=nkv,
                head_dim=D,
                sliding_window=self.self_attn.window,
                prefix=f"{prefix}.kv_p{p}",
            )
            for p in range(visits)
        )
        self.conv_owners = nn.ModuleList(
            Inkling2ConvOwner(
                num_kv_heads=nkv,
                stream_channels=(nkv * D, nkv * D, H, H),
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_p{p}",
            )
            for p in range(visits)
        )

    def _static_conv(
        self,
        conv: StaticSconv,
        x: torch.Tensor,
        stream: int,
        positions: torch.Tensor,
        owner: Inkling2ConvOwner,
        md: Inkling2Metadata | None,
    ) -> torch.Tensor:
        x = x.float()
        if md is None:  # profiling / dummy run: no history
            taps = torch.stack(
                [x] + [torch.zeros_like(x)] * (conv.kernel_size - 1), dim=0
            )
        else:
            taps = conv_taps(owner, stream, x, positions, md)
        return conv.forward_taps(taps)

    def forward(
        self,
        hidden: torch.Tensor,
        block_states: torch.Tensor,
        messages: torch.Tensor | None,
        positions: torch.Tensor,
        visit: int,
    ) -> torch.Tensor:
        dtype = self.self_attn.q_proj.linear.weight.dtype
        kv_owner = self.kv_owners[visit]
        conv_owner = self.conv_owners[visit]
        kv_md = get_metadata(kv_owner.prefix)
        conv_md = get_metadata(conv_owner.prefix)
        block_residual = hidden

        if self.is_full:
            assert self.attention_input is not None  # layer 0 is a sliding layer
            qh, kh, vh = self.attention_input(block_states, hidden)
            if messages is not None:
                qm, km, vm, mm = messages.unbind(1)
                qh = self.q_message_gate(qh, qm)
                kh = self.k_message_gate(kh, km)
                vh = self.v_message_gate(vh, vm)
            hq = self.q_input_layernorm(qh).to(dtype)
            hk = self.k_input_layernorm(kh).to(dtype)
            hv = self.v_input_layernorm(vh).to(dtype)
        else:
            h = (
                self.attention_input(block_states, hidden)[0]
                if self.attention_input is not None
                else hidden
            )
            if messages is not None:
                qm, km, vm, mm = messages.unbind(1)
                hq = self.input_layernorm(self.q_message_gate(h, qm)).to(dtype)
                hk = self.input_layernorm(self.k_message_gate(h, km)).to(dtype)
                hv = self.input_layernorm(self.v_message_gate(h, vm)).to(dtype)
            else:
                hq = hk = hv = self.input_layernorm(h).to(dtype)

        if kv_md is None:
            attn = torch.zeros_like(hq)
        else:
            attn = self.self_attn(
                hq, hk, hv, positions, kv_owner, conv_owner, kv_md, conv_md
            )
        attn = self._static_conv(
            self.post_attention_convolution,
            attn,
            STREAM_ATTN,
            positions,
            conv_owner,
            conv_md,
        )

        residual = h = attn.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(1)], dim=1)
        else:
            residual = h = block_residual.float() + h

        h = self.mlp_input(block_states, h)[0]
        if messages is not None:
            h = self.mlp_message_gate(h, mm)
        h = self.mlp(self.post_attention_layernorm(h).to(dtype))
        h = self._static_conv(
            self.post_mlp_convolution, h, STREAM_MLP, positions, conv_owner, conv_md
        )
        return residual + h


class Inkling2Model(nn.Module):
    def __init__(
        self,
        config: Inkling2LoopTextConfig,
        quant_formats: list[loader.ExpertFormat],
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size
        self.embed_tokens = nn.Module()
        self.embed_tokens.weight = nn.Parameter(
            torch.empty(config.padded_vocab_size, H), requires_grad=False
        )
        self.embed_norm = blocks.Inkling2RMSNorm(H, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, i, quant_formats[i], f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.norm = blocks.Inkling2RMSNorm(H, config.rms_norm_eps)
        self.final_input = blocks.BlockAttentionResidual(config)
        self.message_input = blocks.MessageAttentionResidual(config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.embed_tokens.weight)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        cfg = self.config
        ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        hidden = self.embed_norm(inputs_embeds).float()
        block_states = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[-1])

        def run(layers, hidden, block_states, messages_fn, visit):
            for j, layer in enumerate(layers):
                messages = messages_fn(j) if messages_fn is not None else None
                new = layer(hidden, block_states, messages, positions, visit)
                if layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden.unsqueeze(1)], dim=1)
                hidden = new
            return hidden, block_states

        hidden, block_states = run(self.layers[:ls], hidden, block_states, None, 0)
        pre_hidden, pre_blocks = hidden, block_states
        num_pre = pre_blocks.shape[1]

        loop_layers = self.layers[ls:le]
        hidden, block_states = run(loop_layers, hidden, block_states, None, 0)
        for visit in range(1, cfg.loop_num_passes):
            prev_blocks = block_states[:, num_pre + 1 :]
            prev_hidden = hidden
            hidden, block_states = pre_hidden, pre_blocks
            hidden, block_states = run(
                loop_layers,
                hidden,
                block_states,
                lambda j, pb=prev_blocks, ph=prev_hidden: self.message_input(pb, ph, j),
                visit,
            )

        hidden, block_states = run(self.layers[le:], hidden, block_states, None, 0)
        hidden = self.final_input(block_states, hidden)[0]
        return self.norm(hidden)


# checkpoint key (after the ``model.llm.`` prefix) -> parameter name in this model
_LAYER_KEYS: tuple[tuple[str, str], ...] = (
    ("attn_norm.weight", "input_layernorm.weight"),
    ("q_input_norm.weight", "q_input_layernorm.weight"),
    ("k_input_norm.weight", "k_input_layernorm.weight"),
    ("v_input_norm.weight", "v_input_layernorm.weight"),
    ("mlp_norm.weight", "post_attention_layernorm.weight"),
    ("attn_sconv.weight", "post_attention_convolution.conv1d.weight"),
    ("mlp_sconv.weight", "post_mlp_convolution.conv1d.weight"),
    ("attn.wq_du.weight", "self_attn.q_proj.linear.weight"),
    ("attn.q_l2_clip.scale", "self_attn.q_proj.norm.scale"),
    ("attn.wk_dv.weight", "self_attn.k_proj.linear.weight"),
    ("attn.wv_dv.weight", "self_attn.v_proj.linear.weight"),
    ("attn.wo_ud.weight", "self_attn.o_proj.linear.weight"),
    ("attn.gate_attn.weight", "self_attn.o_proj.gate_proj.weight"),
    ("attn.attn_sink_proj.weight", "self_attn.sink_proj.weight"),
    ("attn.wr_du.weight", "self_attn.relative_position_bias.linear.weight"),
    ("attn.rel_l2_clip.scale", "self_attn.relative_position_bias.norm.scale"),
    ("attn.rel_logits_proj.proj", "self_attn.relative_position_bias.proj"),
    ("attn.k_sconv.weight", "self_attn.k_proj.sconv.conv1d.weight"),
    ("attn.v_sconv.weight", "self_attn.v_proj.sconv.conv1d.weight"),
    ("attn.k_sconv.static_w", "self_attn.k_proj.sconv.static_weight"),
    ("attn.v_sconv.static_w", "self_attn.v_proj.sconv.static_weight"),
    ("attn.k_sconv.weight_proj", "self_attn.k_proj.sconv.weight_proj.weight"),
    ("attn.v_sconv.weight_proj", "self_attn.v_proj.sconv.weight_proj.weight"),
    ("mlp.gate.weight", "mlp.gate.weight"),
    ("mlp.gate.bias", "mlp.gate.e_score_correction_bias"),
)
_GATES = (
    ("q_message_gate", "attn"),
    ("k_message_gate", "k"),
    ("v_message_gate", "v"),
    ("mlp_message_gate", "mlp"),
)
_GATE_PARTS = (
    ("down_weight", "down_proj.weight"),
    ("forget_weight", "forget_gate_proj.weight"),
    ("gain_weight", "message_gate_proj.weight"),
)
_LAYER_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
_GATE_UP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.w13_weight(_scale)?$")


def copy_gate_up_deinterleaved(target: torch.Tensor, src: torch.Tensor) -> None:
    """Copy a checkpoint ``w13_weight`` / ``w13_weight_scale`` (``[E, 2I, ...]``)
    into ``target``, turning the on-disk row order into the model's.

    The checkpoint interleaves the gate and up projections row by row
    (``[g0, u0, g1, u1, ...]``); the model keeps ``[gate rows; up rows]`` so that
    the kernels' output columns split as ``chunk(2)`` -> ``gate, up``.  This is the
    transformers conversion ``Interleave(dim=1)`` of ``inkling2_loop`` (applied to
    ``mlp.experts.w13_weight``): ``out[:, j * I + i] = in[:, 2 * i + j]``.  The
    per-row block scales move with their rows; the per-expert ``amax`` is unaffected.
    """
    e, n2 = src.shape[:2]
    rest = tuple(src.shape[2:])
    rows = src.reshape(e, n2 // 2, 2, *rest).transpose(1, 2)  # [E, 2, I, ...]
    target.view(e, 2, n2 // 2, *rest).copy_(rows)


def _check_expert(name: str, src: torch.Tensor, target: torch.Tensor) -> None:
    """Expert tensors are stored bit-exact (packed NVFP4 / MXFP8 bytes and their
    scales), so a dtype or shape difference is a format mismatch, never a cast."""
    if src.dtype != target.dtype or tuple(src.shape) != tuple(target.shape):
        raise ValueError(
            f"{name}: checkpoint {src.dtype} {tuple(src.shape)}, "
            f"model {target.dtype} {tuple(target.shape)}"
        )


class Inkling2LoopForConditionalGeneration(nn.Module):
    """Text-only entry point (vision / MTP are not served yet)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, Inkling2LoopConfig), type(config)
        self.config = config.text_config
        dtype = vllm_config.model_config.dtype
        if dtype != torch.float16:
            raise ValueError(
                f"Inkling-2 on gfx1030 serves fp16 only (got {dtype}); "
                "pass --dtype float16"
            )
        quant_formats = loader.read_expert_formats(
            vllm_config.model_config.model,
            vllm_config.model_config.revision,
            self.config.num_hidden_layers,
        )
        self.model = Inkling2Model(self.config, quant_formats, prefix=prefix + "model")
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(
            torch.empty(self.config.vocab_size, self.config.hidden_size),
            requires_grad=False,
        )
        self._softcap = self.config.final_logit_softcapping
        self._msg_order = ("q", "k", "v", "m")
        self._moe_modules = [layer.mlp for layer in self.model.layers]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, positions, inputs_embeds)
        return hidden.to(self.lm_head.weight.dtype)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # ``lm_head`` holds exactly the ``vocab_size`` real ids (the checkpoint's
        # padding rows are dropped at load), as in the reference head.
        logits = F.linear(hidden_states, self.lm_head.weight).float()
        logits = torch.tanh(logits / self._softcap) * self._softcap
        return logits

    # -- weights ---------------------------------------------------------
    def _map_name(self, name: str) -> str | None:
        if name == "model.llm.embed.weight":
            return "model.embed_tokens.weight"
        if name == "model.llm.embed_norm.weight":
            return "model.embed_norm.weight"
        if name == "model.llm.norm.weight":
            return "model.norm.weight"
        if name == "model.llm.unembed.weight":
            return "lm_head.weight"
        if name == "model.llm.attn_res.output_query":
            return "model.final_input.query"
        m = re.match(r"^model\.llm\.attn_res\.mlp_queries\.(\d+)$", name)
        if m:
            return f"model.layers.{m.group(1)}.mlp_input.query"
        m = re.match(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$", name)
        if m:
            return f"model.layers.{int(m.group(1)) + 1}.attention_input.query"
        m = _LAYER_RE.match(name)
        if m:
            layer, rest = m.group(1), m.group(2)
            for src, dst in _LAYER_KEYS:
                if rest == src:
                    return f"model.layers.{layer}.{dst}"
            if rest.startswith("mlp.experts."):
                return f"model.layers.{layer}.mlp.{rest[len('mlp.experts.') :]}"
            cm = re.match(r"^controllers\.(\w+)\.(\w+)$", rest)
            if cm:
                for gate, ctl in _GATES:
                    if cm.group(1) == ctl:
                        for part, dst in _GATE_PARTS:
                            if cm.group(2) == part:
                                return f"model.layers.{layer}.{gate}.{dst}"
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        cross: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name.startswith(("mtp.", "model.visual.")):
                continue
            if name in (
                "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries",
            ):
                cross[name] = tensor
                continue
            dst = self._map_name(name)
            if dst is None:
                raise KeyError(f"unmapped Inkling-2 checkpoint tensor {name!r}")
            target = params.get(dst)
            if target is None:
                target = buffers.get(dst)
            if target is None:
                raise KeyError(f"{name!r} -> {dst!r}: no such parameter")
            with torch.no_grad():
                if dst == "model.embed_tokens.weight":
                    target.copy_(tensor.to(target.dtype))
                elif dst == "lm_head.weight":
                    # padded on disk; the reference head keeps the real ids only
                    target.copy_(tensor[: target.shape[0]].to(target.dtype))
                elif _GATE_UP_RE.match(dst):
                    _check_expert(name, tensor, target)
                    copy_gate_up_deinterleaved(target, tensor)
                elif ".mlp.w" in dst:
                    _check_expert(name, tensor, target)
                    target.copy_(tensor)
                else:
                    target.copy_(tensor.to(target.dtype).view_as(target))
            loaded.add(dst)
        if len(cross) == 2:
            q = cross["model.llm.middle_cycle_cross_pass_queries"].float()
            kv = cross["model.llm.middle_cycle_cross_pass_kv_queries"].float()
            n = self.config.num_loop_layers
            q = q.view(n, 2, -1)
            kv = kv.view(n, 2, -1)
            slots = {"q": q[:, 0], "m": q[:, 1], "k": kv[:, 0], "v": kv[:, 1]}
            msg = torch.stack([slots[c] for c in self._msg_order], dim=1)
            self.model.message_input.query.data.copy_(msg)
            loaded.add("model.message_input.query")
        for moe in self._moe_modules:
            moe.process_weights_after_loading()
        return loaded
