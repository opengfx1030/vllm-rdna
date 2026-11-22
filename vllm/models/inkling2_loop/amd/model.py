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
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal

from ..configs import Inkling2LoopConfig, Inkling2LoopTextConfig
from . import blocks, loader
from .attention import (
    Inkling2Attention,
    StaticSconv,
    get_metadata,
    static_conv_hip,
)
from .moe import Inkling2LoopMoE
from .multimodal import Inkling2MultiModalMixin, register_inkling2_processor
from .state import (
    STREAM_ATTN,
    STREAM_MLP,
    Inkling2ConvOwner,
    Inkling2KVOwner,
    Inkling2Metadata,
    conv_taps,
    use_hip_ops,
)

if TYPE_CHECKING:
    from .glue import Inkling2GlueWorkspace
    from .linear_pack import LMHeadPack

logger = init_logger(__name__)


def _gate_weights(w: torch.Tensor, fp16: bool) -> torch.Tensor:
    """Forget / gain projections ``[G, H, R]`` of ``G`` message gates, repacked for the
    glue kernels: fp16 pair-interleaved ``[G, R/2, H, 2]`` (the ``V_DOT2`` layout) or
    fp32 ``[G, R, H]`` (the FMA layout)."""
    if fp16:
        g, hidden, rank = w.shape
        return (
            w.to(torch.float16)
            .reshape(g, hidden, rank // 2, 2)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    return w.float().transpose(1, 2).contiguous()


def _stack_gates(
    gates: list[blocks.MessageGate],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(down [G, R, 2H] fp16, forget, gain)`` of ``G`` MessageGates; frees the
    checkpoint-layout weights.

    The glue kernels take one dtype for ``forget`` and ``gain``: fp16 when the
    checkpoint's bf16 values of both convert to fp16 exactly, else fp32 for both, so no
    gate ever sees a rounded weight."""
    down = torch.stack([g.down_proj.weight.detach() for g in gates]).contiguous()
    forget = torch.stack([g.forget_gate_proj.weight.detach() for g in gates])
    gain = torch.stack([g.message_gate_proj.weight.detach() for g in gates])
    exact = all(
        torch.equal(w.to(torch.float16).float(), w.float()) for w in (forget, gain)
    )
    for g in gates:
        del g.down_proj.weight, g.forget_gate_proj.weight, g.message_gate_proj.weight
    return down, _gate_weights(forget, exact), _gate_weights(gain, exact)


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
        """Eager torch reference of one layer visit (``forward_hip`` is the serving
        path); ``hidden`` is the fp32 partial block, the result is the layer's fp32
        output."""
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

    # -- HIP serving path --------------------------------------------------
    @torch.no_grad()
    def prepare_hip(self, ws: Inkling2GlueWorkspace) -> None:
        """Repack this layer for the HIP kernels (once, after loading) and free the
        checkpoint-layout originals: the packed attention projections, the stacked
        message-gate and layer-norm weights, the router."""
        H = self.config.hidden_size
        if self.is_full:
            norms = [
                self.q_input_layernorm,
                self.k_input_layernorm,
                self.v_input_layernorm,
            ]
        else:
            norms = [self.input_layernorm]
        self.attn_norm_w = torch.stack([n.weight.detach() for n in norms]).contiguous()
        for n in norms:
            del n.weight
        post = self.post_attention_layernorm
        self.post_attn_norm_w = post.weight.detach()[None].contiguous()
        del post.weight
        self._input_norm = self.attn_norm_w[0]  # layer 0 normalises the embedding
        self._attn_query = (
            None
            if self.attention_input is None
            else self.attention_input.query.view(-1, H)
        )
        self._mlp_query = self.mlp_input.query.view(1, H)
        if self.is_loop:
            self.qkv_wd, self.qkv_wf, self.qkv_wg = _stack_gates(
                [self.q_message_gate, self.k_message_gate, self.v_message_gate]
            )
            self.mlp_wd, self.mlp_wf, self.mlp_wg = _stack_gates(
                [self.mlp_message_gate]
            )
        self.self_attn.prepare_hip(ws)
        self.mlp.prepare_hip()

    def forward_hip(
        self,
        ws: Inkling2GlueWorkspace,
        v: Any,
        x0: torch.Tensor,
        x1: torch.Tensor | None,
        n_blocks: int,
        has_msgs: bool,
        positions: torch.Tensor,
        visit: int,
        tap: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """One layer visit on the native kernels, no allocation, no host sync.

        The partial block at the entry is the lazy fp32 sum ``x0 (+ x1)``; ``v`` holds
        the per-step views of the workspace (``Inkling2GlueWorkspace.bind``) and
        ``v.msgs`` the messages of this visit when ``has_msgs``.  Returns the lazy sum
        of the output as ``(x0, x1)`` and the new number of block states.
        """
        from vllm import _custom_ops as ops

        eps = self.config.rms_norm_eps
        G = 3 if self.is_full else 1
        blk = v.blk
        kv_owner, conv_owner = self.kv_owners[visit], self.conv_owners[visit]
        kv_md, conv_md = get_metadata(kv_owner.prefix), get_metadata(conv_owner.prefix)

        # ---- attention input: block residual (+ message gates) + layernorm -> fp16
        if self.attention_input is None:  # layer 0: the embedding is the partial block
            hid = x0
            ops.inkling2_glue_rms_norm(hid, v.hq, weight=self._input_norm, eps=eps)
        else:
            hid = v.hid
            if not has_msgs:
                ops.inkling2_glue_attn_res(
                    blk, n_blocks, x0, self._attn_query,
                    v.norm3 if G == 3 else v.norm1, x1=x1, partial_out=hid,
                    norm_w=self.attn_norm_w, eps=eps, norm_eps=eps,
                )  # fmt: skip
            else:
                raw = v.raw3 if G == 3 else v.raw1
                ops.inkling2_glue_attn_res(
                    blk, n_blocks, x0, self._attn_query, raw, x1=x1,
                    partial_out=hid, eps=eps,
                )  # fmt: skip
                ops.inkling2_glue_msg_gate_down(raw, v.msgs_qkv, self.qkv_wd, v.d, eps)
                ops.inkling2_glue_msg_gate_mix(
                    raw, v.msgs_qkv, v.d, self.qkv_wf, self.qkv_wg, v.x16,
                    norm_w=self.attn_norm_w, scratch=v.g_scratch, partials=v.g_part,
                    counters=ws.gate_counters, norm_eps=eps,
                )  # fmt: skip
        if tap is not None:
            ops.inkling2_glue_copy_states(hid, tap)
        # sliding layers without messages feed one stream to q, k and v
        one_stream = not has_msgs and not self.is_full
        hq = v.hq
        hk = hq if one_stream else v.hk
        hv = hq if one_stream else v.hv

        # ---- attention + post-attention short convolution -> attn_c (fp32 [T, H])
        # profiling / dummy run: no cache, no history
        if kv_md is None or conv_md is None:
            ops.inkling2_glue_rms_norm(hq, v.hk, sum_out=v.attn_c, eps=eps)
        else:
            attn = self.self_attn.forward_hip(
                hq, hk, hv, positions, kv_owner, conv_owner, kv_md, conv_md, v.o_out
            )
            static_conv_hip(
                self.post_attention_convolution,
                attn,
                v.attn_c,
                conv_owner,
                conv_md,
                STREAM_ATTN,
            )

        # ---- mlp input: boundary layers append their input, the others add attn_c
        if self.is_block_boundary:  # stack = states + hid + attn_c; residual = attn_c
            a0, a1, extra, new_x0 = v.attn_c, None, {"append": hid}, v.attn_c
        else:  # stack = states + (hid + attn_c); residual = hid + attn_c
            a0, a1, extra, new_x0 = hid, v.attn_c, {"partial_out": v.res}, v.res
        if not has_msgs:
            ops.inkling2_glue_attn_res(
                blk, n_blocks, a0, self._mlp_query, v.norm1, x1=a1,
                norm_w=self.post_attn_norm_w, eps=eps, norm_eps=eps, **extra,
            )  # fmt: skip
        else:
            ops.inkling2_glue_attn_res(
                blk, n_blocks, a0, self._mlp_query, v.raw1, x1=a1, eps=eps, **extra
            )
            ops.inkling2_glue_msg_gate_down(v.raw1, v.msgs_mlp, self.mlp_wd, v.d1, eps)
            ops.inkling2_glue_msg_gate_mix(
                v.raw1, v.msgs_mlp, v.d1, self.mlp_wf, self.mlp_wg, v.x16_g1,
                norm_w=self.post_attn_norm_w, scratch=v.g_scratch, partials=v.g_part,
                counters=ws.gate_counters, norm_eps=eps,
            )  # fmt: skip

        # ---- MoE (route, gemm1, swiglu, gemm2) + post-mlp short convolution
        moe = self.mlp.forward_hip(hq, ws, v.moe_out)
        if conv_md is None:
            ops.inkling2_glue_rms_norm(moe, v.hk, sum_out=v.mlp_c, eps=eps)
        else:
            static_conv_hip(
                self.post_mlp_convolution,
                moe,
                v.mlp_c,
                conv_owner,
                conv_md,
                STREAM_MLP,
            )
        return new_x0, v.mlp_c, n_blocks + int(self.is_block_boundary)


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
        # Global layer-visit indices (0..47) whose entry block states feed a DSpark /
        # EAGLE-style draft; empty = no taps and the plain forward below.
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # serving path (``Inkling2LoopTextForCausalLM`` switches it on after loading)
        self.serving = False
        self.glue_ws: Inkling2GlueWorkspace | None = None
        self._tap_slots: dict[int, int] = {}

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Text embedding rows with ``embed_norm`` applied (``[T, H]``, model dtype).

        This is what the reference ``NormedEmbedding`` returns: the vision mixin
        overwrites the image rows of this tensor and ``forward`` takes it as
        ``inputs_embeds`` unchanged.
        """
        if self.serving:
            from vllm import _custom_ops as ops

            out = torch.empty(
                input_ids.shape[0],
                self.config.hidden_size,
                dtype=self.embed_norm.weight.dtype,
                device=input_ids.device,
            )
            ops.inkling2_glue_embed_norm(
                input_ids,
                self.embed_tokens.weight,
                self.embed_norm.weight,
                out,
                self.config.rms_norm_eps,
            )
            return out
        return self.embed_norm(F.embedding(input_ids, self.embed_tokens.weight))

    def forward_reference(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        cfg = self.config
        ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        hidden = inputs_embeds.float()
        block_states = hidden.new_zeros(hidden.shape[0], 0, hidden.shape[-1])
        taps: list[torch.Tensor] = []
        visits_seen = 0

        def run(layers, hidden, block_states, messages_fn, visit):
            nonlocal visits_seen
            for j, layer in enumerate(layers):
                if visits_seen in self.aux_hidden_state_layers:
                    taps.append(hidden)  # the fp32 partial block at this visit's entry
                visits_seen += 1
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
        out = self.norm(hidden)
        if self.aux_hidden_state_layers:
            # the final normalized output is the last tap (its id is the visit count)
            return out, [*taps, out]
        return out

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """HIP kernels when serving, the torch reference otherwise (no fallback)."""
        if self.serving:
            return self.forward_hip(input_ids, positions, inputs_embeds)
        return self.forward_reference(input_ids, positions, inputs_embeds)

    # -- HIP serving path --------------------------------------------------
    @torch.no_grad()
    def prepare_hip(self, ws: Inkling2GlueWorkspace) -> None:
        cfg = self.config
        self.glue_ws = ws
        self._final_query = self.final_input.query.view(1, cfg.hidden_size)
        self._final_norm_w = self.norm.weight[None]
        self._msg_queries = [
            self.message_input.query[j] for j in range(cfg.num_loop_layers)
        ]
        self.configure_taps()

    def configure_taps(self) -> None:
        """Slots of the DSpark taps (one fp32 buffer per selected layer visit)."""
        visits = sorted(set(self.aux_hidden_state_layers))
        total = self.config.total_num_hidden_layers
        if visits and not 0 <= visits[0] <= visits[-1] < total:
            raise ValueError(f"tap visits {visits} outside [0, {total})")
        self._tap_slots = {visit: slot for slot, visit in enumerate(visits)}
        if self.glue_ws is not None:
            self.glue_ws.ensure_taps(len(visits))

    def forward_hip(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """The whole stack on the native kernels
        (``docs/rdna2/inkling2-glue-kernels.md``).

        Everything runs on the persistent buffers of the workspace: no allocation, no
        host synchronisation, shapes and pointers fixed by ``T`` (HIP-graph capturable).
        Returns the fp16 ``[T, H]`` input of the lm_head (a view of a workspace buffer,
        valid until the next forward) and, with taps, the fp32 taps as well.
        """
        from vllm import _custom_ops as ops

        ws = self.glue_ws
        if ws is None:
            raise RuntimeError(
                "Inkling-2 serving path used before load_weights() prepared it"
            )
        cfg = self.config
        eps, T = cfg.rms_norm_eps, positions.shape[0]
        ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        v = ws.bind(T)
        copy = ops.inkling2_glue_copy_states
        slots = self._tap_slots
        seen = 0

        if inputs_embeds is None:
            assert input_ids is not None
            ops.inkling2_glue_embed_norm(
                input_ids, self.embed_tokens.weight, self.embed_norm.weight, v.hid, eps
            )
        else:
            # the rows are final embeddings (``embed_input_ids`` applied embed_norm and
            # the vision mixin merged the image rows): widen them into the fp32 stream
            ops.inkling2_glue_rms_norm(inputs_embeds, v.hk, sum_out=v.hid, eps=eps)

        def run(layers, x0, x1, n, visit, fill=None):
            nonlocal seen
            for j, layer in enumerate(layers):
                slot = slots.get(seen)
                seen += 1
                if fill is not None:
                    fill(j)
                tap = None if slot is None else ws.taps[slot][:T]
                x0, x1, n = layer.forward_hip(
                    ws, v, x0, x1, n, fill is not None, positions, visit, tap
                )
            return x0, x1, n

        x0, x1, n = run(self.layers[:ls], v.hid, None, 0, 0)
        num_pre = n
        loop_layers = self.layers[ls:le]
        passes = cfg.loop_num_passes
        if passes > 1:  # the state every later pass restarts from
            copy(x0, v.pre_x0)
            copy(x1, v.pre_x1)
        x0, x1, n = run(loop_layers, x0, x1, n, 0)
        for visit in range(1, passes):
            # the previous pass's block states and output feed this pass's messages;
            # this pass rewrites those buffers while it still reads them
            n_prev = n - num_pre - 1
            if n_prev > 0:
                copy(v.blk[:, num_pre + 1 : n], v.prev_blk[:, :n_prev])
            copy(x0, v.prev_x0)
            copy(x1, v.prev_x1)

            def fill(j, n_prev=n_prev):
                ops.inkling2_glue_attn_res(
                    v.prev_blk, n_prev, v.prev_x0, self._msg_queries[j], v.msgs,
                    x1=v.prev_x1, eps=eps,
                )  # fmt: skip

            x0, x1, n = run(loop_layers, v.pre_x0, v.pre_x1, num_pre, visit, fill)
        x0, x1, n = run(self.layers[le:], x0, x1, n, 0)

        final16 = v.hq  # fp16 [T, H]: the input of the lm_head
        if not slots:
            ops.inkling2_glue_attn_res(
                v.blk, n, x0, self._final_query, final16.unsqueeze(1), x1=x1,
                norm_w=self._final_norm_w, eps=eps, norm_eps=eps,
            )  # fmt: skip
            return final16
        # DSpark: the fp32 normalised output is the last tap
        ops.inkling2_glue_attn_res(
            v.blk, n, x0, self._final_query, v.hid.unsqueeze(1), x1=x1, eps=eps
        )
        final32 = ws.final_tap(T)
        ops.inkling2_glue_rms_norm(v.hid, final32, weight=self.norm.weight, eps=eps)
        ops.inkling2_glue_rms_norm(v.hid, final16, weight=self.norm.weight, eps=eps)
        return final16, [*(ws.taps[i][:T] for i in range(len(slots))), final32]


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


class Inkling2LoopTextForCausalLM(nn.Module):
    """Text model of Inkling-2-Mini; the served class adds the vision tower (below).

    ``serving`` (default: on a GPU) runs the stack on the native HIP kernels of
    ``csrc/rocm`` from persistent buffers, with no torch compute and no fallback (a
    missing op raises); ``serving=False`` runs the eager torch reference that the tests
    and the CPU harness compare against.  ``VLLM_INKLING2_PATH=reference`` forces the
    latter.
    """

    # The DSpark draft (``mtp.safetensors``) reads taps of the target: the runner asks
    # for them through the EAGLE3 interface (see docs/rdna2/inkling2-dspark.md).
    supports_eagle3 = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        serving: bool | None = None,
    ) -> None:
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
        self.serving = use_hip_ops() if serving is None else serving
        self.model.serving = self.serving
        # sizes the persistent buffers of the serving path
        self._max_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens if self.serving else 0
        )
        self._head: LMHeadPack | None = None
        self._argmax_parts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Select the global layer visits whose entry block states are returned as
        auxiliary hidden states (the final normalized output is always appended)."""
        self.model.aux_hidden_state_layers = tuple(layers)
        self.model.configure_taps()

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # the Inkling-2-Mini draft's ``target_layer_ids`` without the final output
        return (5, 23, 41)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        out = self.model(input_ids, positions, inputs_embeds)
        dtype = self.lm_head.weight.dtype
        if isinstance(out, tuple):  # auxiliary hidden states were requested
            hidden, taps = out
            return hidden.to(dtype), taps
        return out.to(dtype)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Soft-capped fp32 logits ``[M, vocab]`` of the hidden states ``[M, H]``."""
        if self.serving:
            head = self._lm_head_pack()
            logits = torch.empty(
                hidden_states.shape[0],
                head.rows,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            head(hidden_states, logits, softcap=self._softcap)
            return logits
        # ``lm_head`` holds exactly the ``vocab_size`` real ids (the checkpoint's
        # padding rows are dropped at load), as in the reference head.
        logits = F.linear(hidden_states, self.lm_head.weight).float()
        logits = torch.tanh(logits / self._softcap) * self._softcap
        return logits

    def greedy_token_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``argmax`` of the soft-capped logits of every row, int64 ``[M]`` (the first
        index wins a tie).

        Up to 16 rows the lm_head GEMV reduces its own epilogue to one argmax partial
        per block and never writes the logits row of a token; more rows take the
        tiled GEMM and a partial argmax over its logits.
        """
        from vllm import _custom_ops as ops

        head = self._lm_head_pack()
        M, device = hidden_states.shape[0], hidden_states.device
        parts = 64 if M <= 4 else self._argmax_parts
        part_val = torch.empty(M, parts, dtype=torch.float32, device=device)
        part_idx = torch.empty(M, parts, dtype=torch.int32, device=device)
        out = torch.empty(M, dtype=torch.int64, device=device)
        logits = (
            None
            if M <= 16
            else torch.empty(M, head.rows, dtype=torch.float32, device=device)
        )
        head(hidden_states, logits, part_val, part_idx, softcap=self._softcap)
        ops.inkling2_linear_argmax_reduce(part_val, part_idx, out)
        return out

    def _lm_head_pack(self) -> LMHeadPack:
        if not self.serving:
            raise RuntimeError("the HIP lm_head needs the serving path")
        if self._head is None:
            raise RuntimeError("Inkling-2 serving path used before load_weights()")
        return self._head

    @torch.no_grad()
    def _prepare_serving(self) -> None:
        """Repack every dense weight for the HIP kernels and free the originals, once
        after loading: attention input / o_proj / router / message-gate / layer-norm
        packs per layer, the lm_head pack (a view of ``lm_head.weight``) and the
        persistent workspace."""
        from . import glue, linear_pack

        glue.check_kernel_shapes(self.config)
        device = self.lm_head.weight.device
        before = torch.accelerator.memory_allocated(device)
        ws = glue.Inkling2GlueWorkspace(self.config, self._max_tokens, device)
        workspace = torch.accelerator.memory_allocated(device) - before
        for layer in self.model.layers:
            layer.prepare_hip(ws)
        self.model.prepare_hip(ws)
        self._head = linear_pack.inkling2_linear_pack_lm_head(
            self.lm_head.weight, self.config.vocab_size
        )
        self._argmax_parts = (
            8 * torch.cuda.get_device_properties(device).multi_processor_count
        )
        layers = self.model.layers
        loop = [layer for layer in layers if layer.is_loop]
        logger.info(
            "Inkling-2 HIP serving path: workspace %.2f GiB for %d tokens per step; "
            "fp16 router weights in %d/%d layers, fp16 message-gate weights in %d/%d "
            "loop layers (the others keep fp32 weights)",
            workspace / 2**30,
            self._max_tokens,
            sum(layer.mlp.router_fp16 for layer in layers),
            len(layers),
            sum(layer.qkv_wf.dtype == torch.float16 for layer in loop),
            len(loop),
        )

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
            if name.startswith("mtp."):
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
        # the repack frees each layer's checkpoint-layout experts; the local dicts
        # (and `target`) would keep every one of them alive until load_weights
        # returns and double the expert memory
        del params, buffers, target, tensor
        for moe in self._moe_modules:
            moe.process_weights_after_loading()
        if self.serving:
            self._prepare_serving()
        return loaded


@register_inkling2_processor
class Inkling2LoopForConditionalGeneration(
    Inkling2MultiModalMixin, Inkling2LoopTextForCausalLM, SupportsMultiModal
):
    """Text and image entry point of ``Inkling2LoopForConditionalGeneration``."""
