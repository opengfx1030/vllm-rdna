# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The ZhipuAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GLM-5.3-Flash (Glm5Next) model.

Architecture (zai-org/GLM-5.3-Flash):
- 45 decoder layers. ``layer_types`` alternates 3 KDA (Kimi-style linear
  attention) layers followed by 1 DSA (DeepSeek Sparse Attention, NoPE MLA
  with a k-pool indexer) layer: DSA layers are [3, 7, ..., 43].
- Layers 0-2 use a dense SwiGLU MLP (clamped by ``swiglu_limit``), layers
  3-44 use MoE (288 routed experts + 1 shared expert, top-8, sigmoid
  routing with ``e_score_correction_bias``, ``routed_scaling_factor`` 2.5).
- mHC (manifold-constrained hyper-connections): the residual stream is carried
  as ``hc_mult`` (=4) parallel streams of ``hidden_size`` through the LM.
  Every decoder layer owns two ``Glm5NextHyperConnection`` modules (attn and
  ffn sites) that collapse the streams into the sublayer input and mix the
  sublayer output back into the streams (Sinkhorn-Knopp doubly-stochastic
  combine matrix). The final ``HyperHead`` is an unweighted mean over
  streams before the final RMSNorm.
- MTP: one extra predictor layer (checkpoint layer index 45), DeepSeek-style
  (enorm/hnorm/eh_proj + one DSA+MoE decoder layer + shared_head.norm). The
  MTP layer has NO mHC parameters in the checkpoint and therefore runs as a
  plain residual block.
- No RoPE anywhere in the language model (qk_rope_head_dim = 0).

The KDA sublayer is implemented in
``vllm.model_executor.layers.mamba.gdn.glm5_kda_linear_attn``
(``Glm5NextKDAAttention``, a ``GatedDeltaNetAttention`` subclass providing a
MambaSpec state cache) and the DSA sublayer in
``vllm.model_executor.layers.attention.glm5_dsa_attention``
(``Glm5NextDSAAttention``, providing paged attention specs for the expanded
KV and the indexer cache). This module wires them together with the mHC /
MoE / dense-MLP machinery, the vision tower, MTP and weight loading.
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention.glm5_dsa_attention import (
    Glm5NextDSAAttention,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.glm5_kda_linear_attn import (
    Glm5NextKDAAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from .qwen2_vl import _create_qwen2vl_field_factory
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    get_spec_layer_idx_from_weight_name,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# Upper bound used when budgeting dummy video inputs.
_MAX_FRAMES_PER_VIDEO = 600


# ---------------------------------------------------------------------------
# mHC (manifold-constrained hyper-connections)
# ---------------------------------------------------------------------------


class Glm5NextUnweightedRMSNorm(nn.Module):
    """RMSNorm without learnable parameters (mHC input norm)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        ).to(x.dtype)


class Glm5NextHyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection (mHC).

    Owns the learned (``fn``, ``base``, ``scale``) parameters that turn the
    incoming ``hc_mult`` residual streams into collapse / expand weights.
    Ported from ``Glm5NextTextHyperConnection`` in the HF reference
    (modeling_glm5_next.py); all routing math runs in fp32.
    """

    def __init__(self, config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextUnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(
            torch.empty(mix, self.hc_mult * config.hidden_size)
        )
        self.base = nn.Parameter(torch.empty(mix))
        # One learned scale per mHC output: pre / post / comb.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (post, comb, collapsed) from the mHC mapping.

        Args:
            hidden_streams: residual streams, ``[num_tokens, H, D]``.

        Returns:
            post: sublayer-output placement weights, ``[num_tokens, H]``.
            comb: doubly-stochastic stream mixer, ``[num_tokens, H, H]``.
            collapsed: streams collapsed into the sublayer input,
                ``[num_tokens, D]``.
        """
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=1).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
            [hc, hc, hc * hc], dim=-1
        )
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = (
            comb_w.view(-1, hc, hc) * comb_scale + comb_b.view(hc, hc)
        )
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        # Sinkhorn-Knopp: alternate row / column normalisation so ``comb``
        # lands on the doubly-stochastic manifold.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=1).to(
            hidden_streams.dtype
        )
        return post, comb, collapsed


class Glm5NextHyperHead(nn.Module):
    """Final HC-stream collapse: unweighted mean over the stream axis."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=1)


# ---------------------------------------------------------------------------
# Feed-forward blocks
# ---------------------------------------------------------------------------


class Glm5NextMLP(nn.Module):
    """Dense SwiGLU MLP with GLM's asymmetric gate/up clamping."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # clamp(gate, max=limit) * silu, clamp(up, -limit, limit):
        # SiluAndMulWithClamp with alpha=1, beta=0 is exactly this.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    """MoE block: sigmoid router with e_score_correction_bias + shared
    expert, wired through the stock FusedMoE machinery (glm4_moe pattern)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = (
            self.n_logical_experts + self.n_redundant_experts
        )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        # The HF router is a plain fp32 linear with a per-expert correction
        # bias (not an nn.Linear inside a quantized module), so quant never
        # touches it.
        self.gate = nn.Linear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            dtype=torch.float32,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts, dtype=torch.float32)
        )

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

        self.shared_experts = Glm5NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=(
                config.moe_intermediate_size * self.n_shared_experts
            ),
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            # aiter applies routed_scaling_factor internally; see glm4_moe.
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=not self.is_rocm_aiter_moe_enabled,
            swiglu_limit=config.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states.to(dtype=torch.float32))
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Glm5NextDecoderLayer(nn.Module):
    """One GLM-5.3-Flash block: mHC(attn) -> KDA|DSA -> mHC(ffn) -> MLP/MoE.

    Hidden states travel through the LM as flattened streams
    ``[num_tokens, hc_mult * hidden_size]``. With ``use_hc=False`` (the MTP
    layer, which has no hc_* tensors in the checkpoint) the layer degrades
    to plain pre-norm residual connections.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        use_hc: bool = True,
    ) -> None:
        super().__init__()
        config: Glm5NextTextConfig = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.layer_idx = extract_layer_index(prefix)
        self.block_type = layer_type
        self.use_hc = use_hc
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult

        if layer_type == "linear_attention":
            self.self_attn = Glm5NextKDAAttention(
                config, vllm_config, f"{prefix}.self_attn"
            )
        elif layer_type == "deepseek_sparse_attention":
            self.self_attn = Glm5NextDSAAttention(
                config, self.layer_idx, vllm_config, f"{prefix}.self_attn"
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        if config.mlp_layer_types[self.layer_idx] == "sparse":
            self.mlp = Glm5NextMoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=config.swiglu_limit,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if use_hc:
            self.attn_hc = Glm5NextHyperConnection(config)
            self.ffn_hc = Glm5NextHyperConnection(config)

    def _hc_apply(
        self,
        post: torch.Tensor,
        comb: torch.Tensor,
        sublayer_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the sublayer output back into the streams (HF lines 1316/1325):
        ``streams = post * out + comb^T @ residual``.
        """
        dtype = sublayer_out.dtype
        return post.to(dtype).unsqueeze(-1) * sublayer_out.unsqueeze(1) + (
            torch.matmul(comb.to(dtype).transpose(-1, -2), residual)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_tokens = hidden_states.shape[0]
        topk_indices: torch.Tensor | None = None

        if self.use_hc:
            residual = hidden_states.view(
                num_tokens, self.hc_mult, self.hidden_size
            )
            post, comb, x = self.attn_hc(residual)
        else:
            residual = hidden_states
            x = hidden_states

        x = self.input_layernorm(x)
        if self.block_type == "linear_attention":
            x = self.self_attn(hidden_states=x)
        else:
            x, topk_indices = self.self_attn(
                hidden_states=x, prev_topk_indices=prev_topk_indices
            )

        if self.use_hc:
            hidden_states = self._hc_apply(post, comb, x, residual).view(
                num_tokens, self.hc_mult * self.hidden_size
            )
            residual = hidden_states.view(
                num_tokens, self.hc_mult, self.hidden_size
            )
            post, comb, x = self.ffn_hc(residual)
        else:
            hidden_states = residual + x
            residual = hidden_states
            x = hidden_states

        x = self.post_attention_layernorm(x)
        x = self.mlp(x)

        if self.use_hc:
            hidden_states = self._hc_apply(post, comb, x, residual).view(
                num_tokens, self.hc_mult * self.hidden_size
            )
        else:
            hidden_states = residual + x

        return hidden_states, topk_indices


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------


@support_torch_compile
class Glm5NextTextModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # Flat checkpoint mHC names -> the vLLM submodules that own them.
            "hc_attn_fn": "attn_hc.fn",
            "hc_attn_base": "attn_hc.base",
            "hc_attn_scale": "attn_hc.scale",
            "hc_ffn_fn": "ffn_hc.fn",
            "hc_ffn_base": "ffn_hc.base",
            "hc_ffn_scale": "ffn_hc.scale",
        },
        orig_to_new_stacked={
            # Dense MLP (layers 0-2) and shared experts store gate/up
            # separately; routed experts are handled by the MoE loader.
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Glm5NextTextConfig = vllm_config.model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        def get_layer(layer_prefix: str) -> Glm5NextDecoderLayer:
            layer_idx = extract_layer_index(layer_prefix)
            return Glm5NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=layer_prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        self.hc_head = Glm5NextHyperHead()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError(
                "Glm5Next does not support pipeline parallelism."
            )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        # Expand the embeddings into hc_mult identical residual streams:
        # [M, D] -> [M, H*D].
        num_tokens = hidden_states.shape[0]
        hidden_states = (
            hidden_states.unsqueeze(1)
            .expand(num_tokens, self.hc_mult, self.hidden_size)
            .reshape(num_tokens, self.hc_mult * self.hidden_size)
        )

        topk_indices: torch.Tensor | None = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, topk_indices = layer(
                hidden_states, prev_topk_indices=topk_indices
            )

        # HyperHead collapse (mean over streams) + final norm.
        hidden_states = hidden_states.view(
            num_tokens, self.hc_mult, self.hidden_size
        )
        hidden_states = self.norm(self.hc_head(hidden_states))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# ---------------------------------------------------------------------------
# Vision tower (ported lean from the HF reference; weights load 1:1)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Glm5NextVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = (
            position_ids.float().unsqueeze(-1) * self.inv_freq
        ).flatten(1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(
            hidden_states.to(dtype=self.proj.weight.dtype)
        ).view(-1, self.embed_dim)
        return hidden_states


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=config.attention_bias,
        )
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = (
            qkv.view(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Varlen attention: process each image/video frame chunk separately
        # (the HF non-flash fallback). Vision runs eagerly, outside the LM
        # cudagraphs, so the chunked SDPA path is safe here.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        attn_outputs = []
        for q_c, k_c, v_c in zip(
            torch.split(q, lengths),
            torch.split(k, lengths),
            torch.split(v, lengths),
        ):
            attn_outputs.append(
                F.scaled_dot_product_attention(
                    q_c.transpose(0, 1).unsqueeze(0),
                    k_c.transpose(0, 1).unsqueeze(0),
                    v_c.transpose(0, 1).unsqueeze(0),
                    scale=self.scaling,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        attn_output = torch.cat(attn_outputs, dim=0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(self.gate_proj(hidden_states), max=self.swiglu_limit)
        up = torch.clamp(
            self.up_proj(hidden_states),
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, cos, sin
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Glm5NextVisionPatchMerger(nn.Module):
    """proj -> LayerNorm -> GELU -> clamped SwiGLU -> down."""

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        dim = config.out_hidden_size
        context_dim = config.projection_intermediate_size
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)
        self.act1 = nn.GELU()
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.act1(self.post_projection_norm(hidden_states))
        gate = torch.clamp(self.gate_proj(hidden_states), max=self.swiglu_limit)
        up = torch.clamp(
            self.up_proj(hidden_states),
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionModel(nn.Module):
    """GLM-5.3-Flash ViT: 24 RMSNorm pre-norm blocks, 2D vision RoPE,
    spatial-merge Conv2d downsample and a GELU+SwiGLU patch merger."""

    def __init__(
        self,
        config: Glm5NextVisionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size
        self.out_hidden_size = config.out_hidden_size

        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Glm5NextVisionBlock(config) for _ in range(config.depth)]
        )
        self.post_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm5NextVisionPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(h, w) position ids per patch token, ordered the same way the
        spatial-merge downsample consumes patches (Qwen2-VL convention)."""
        merge = self.spatial_merge_size
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos_ids = torch.arange(h, device=self.device).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w, device=self.device).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            )
        position_ids = torch.cat(pos_ids, dim=0)
        return self.rotary_pos_emb(position_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        cos, sin = self.rot_pos_emb(grid_thw)

        # Attention runs per frame chunk: each (t, h, w) item contributes t
        # chunks of h*w patches.
        chunk_sizes: list[int] = []
        for t, h, w in grid_thw.tolist():
            chunk_sizes.extend([h * w] * t)
        offsets = torch.tensor(chunk_sizes, dtype=torch.int32).cumsum(0)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                offsets,
            ]
        ).to(self.device)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, cos, sin)

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(
            -1, self.out_hidden_size
        )
        return self.merger(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# ---------------------------------------------------------------------------
# Multimodal processing (lean: token-id based placeholders)
# ---------------------------------------------------------------------------


class Glm5NextProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Glm5NextConfig:
        return self.ctx.get_hf_config(Glm5NextConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def _get_vision_grid(
        self,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
    ) -> int:
        vision_config = self.get_hf_config().vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        factor = patch_size * merge_size
        resized_height = (image_height // factor) * factor
        resized_width = (image_width // factor) * factor

        padded_num_frames = num_frames + (-num_frames % temporal_patch_size)
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        return grid_t * grid_h * grid_w // (merge_size**2)

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int
    ) -> int:
        return self._get_vision_grid(image_width, image_height, num_frames=1)

    def get_num_video_tokens(
        self, *, image_width: int, image_height: int, num_frames: int
    ) -> int:
        return self._get_vision_grid(image_width, image_height, num_frames)

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_config = self.get_hf_config().vision_config
        size = vision_config.image_size
        return ImageSize(width=size, height=size)

    def get_max_image_tokens(self) -> int:
        target = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target.width, image_height=target.height
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target = self.get_image_size_with_most_features()
        num_frames_with_most_features = 1
        max_video_tokens = 0
        for num_frames in range(1, _MAX_FRAMES_PER_VIDEO + 1):
            num_vision_tokens = self.get_num_video_tokens(
                image_width=target.width,
                image_height=target.height,
                num_frames=num_frames,
            )
            if (
                0 < num_vision_tokens <= max_tokens
                and num_vision_tokens > max_video_tokens
            ):
                num_frames_with_most_features = num_frames
                max_video_tokens = num_vision_tokens
        return num_frames_with_most_features

    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_image_tokens = self.get_max_image_tokens() * max_images
        return self._get_max_video_frames(seq_len - max_image_tokens)

    def get_max_video_tokens(self, seq_len: int) -> int:
        target = self.get_image_size_with_most_features()
        num_frames = self._get_max_video_frames(seq_len)
        return self.get_num_video_tokens(
            image_width=target.width,
            image_height=target.height,
            num_frames=num_frames,
        )

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len),
        }


class Glm5NextDummyInputsBuilder(
    BaseDummyInputsBuilder[Glm5NextProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Unused: get_dummy_processor_inputs builds the prompt directly from
        # the special token ids below.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        target = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )
        return {
            "image": self._get_dummy_images(
                width=target.width,
                height=target.height,
                num_images=num_images,
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=target.width,
                height=target.height,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=mm_options.get("video"),
            ),
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs:
        config = self.info.get_hf_config()
        num_image_tokens = self.info.get_max_image_tokens()
        num_video_tokens = self.info.get_max_video_tokens(seq_len)

        prompt: list[int] = []
        for _ in range(mm_counts.get("image", 0)):
            prompt.append(config.image_start_token_id)
            prompt.extend([config.image_token_id] * num_image_tokens)
            prompt.append(config.image_end_token_id)
        for _ in range(mm_counts.get("video", 0)):
            prompt.append(config.video_start_token_id)
            prompt.extend([config.image_token_id] * num_video_tokens)
            prompt.append(config.video_end_token_id)

        mm_items = self.info.parse_mm_data(
            self.get_dummy_mm_data(seq_len, mm_counts, mm_options),
            validate=False,
        )
        return ProcessorInputs(prompt=prompt, mm_data_items=mm_items)


class Glm5NextMultiModalProcessor(
    BaseMultiModalProcessor[Glm5NextProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        config = self.info.get_hf_config()
        merge_length = config.vision_config.spatial_merge_size**2

        def get_replacement(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            start_id, end_id = (
                (config.image_start_token_id, config.image_end_token_id)
                if modality == "image"
                else (config.video_start_token_id, config.video_end_token_id)
            )
            full = (
                [start_id] + [config.image_token_id] * num_tokens + [end_id]
            )
            image_token_id = config.image_token_id
            return PromptUpdateDetails(
                full=full,
                # Only the placeholder tokens carry vision embeddings; the
                # start/end delimiter tokens stay text embeddings.
                is_embed=lambda seq: torch.tensor(seq) == image_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[
                    config.image_start_token_id,
                    config.image_token_id,
                    config.image_end_token_id,
                ],
                replacement=partial(get_replacement, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=[
                    config.video_start_token_id,
                    config.image_token_id,
                    config.video_end_token_id,
                ],
                replacement=partial(get_replacement, modality="video"),
            ),
        ]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm5NextDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "model.visual.": "visual.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Glm5NextConfig = vllm_config.model_config.hf_config
        text_config = config.text_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        self.config = config
        self.quant_config = quant_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Glm5Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionModel(
                config.vision_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.model = Glm5NextTextModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(text_config.vocab_size)

    # -- IsHybrid: KDA (mamba-style) state specs ---------------------------

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        # conv state: q|k|v depthwise convs carried in one buffer of total
        # width 3 * (linear_num_heads * linear_head_dim); ssm state:
        # [slots, H/tp, D, D] fp32.
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    # -- Multimodal ----------------------------------------------------------

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> dict[str, Any] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "grid_thw": image_grid_thw}

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> dict[str, Any] | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return {"pixel_values": pixel_values_videos, "grid_thw": video_grid_thw}

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, Any]:
        modalities: dict[str, Any] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key == "pixel_values_videos"
                and "videos" not in modalities
            ):
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return modalities

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        merge_size = self.visual.spatial_merge_size
        for modality in modalities:
            mm_input = modalities[modality]
            if mm_input is None:
                continue
            embeds = self.visual(
                mm_input["pixel_values"], grid_thw=mm_input["grid_thw"]
            )
            grid_thw = mm_input["grid_thw"]
            sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
            multimodal_embeddings += tuple(embeds.split(sizes))
        return multimodal_embeddings

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    # -- Forward / logits / loading ------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# ---------------------------------------------------------------------------
# MTP (multi-token prediction) draft model
# ---------------------------------------------------------------------------


class Glm5NextSharedHead(nn.Module):
    def __init__(
        self,
        config: Glm5NextTextConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    """DeepSeek-style MTP layer: enorm/hnorm/eh_proj fusion + one decoder
    layer (checkpoint layer 45: DSA + MoE). Checkpoint layer 45 carries no
    hc_* tensors, so the decoder block runs with plain residuals."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config: Glm5NextTextConfig = vllm_config.model_config.hf_text_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.shared_head = Glm5NextSharedHead(
            config=config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = Glm5NextDecoderLayer(
            vllm_config,
            layer_type="deepseek_sparse_attention",
            prefix=prefix,
            use_hc=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # Mask the embedding at position 0; MTP never predicts it.
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0, 0, inputs_embeds
        )
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, _ = self.mtp_block(hidden_states)
        return hidden_states


class Glm5NextMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Glm5NextTextConfig = vllm_config.model_config.hf_text_config

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # Map the exact checkpoint layer index (45) into the module tree.
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Glm5NextMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[
            str(self.mtp_start_layer_idx + current_step_idx)
        ]
        return self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )


class Glm5NextMTP(nn.Module):
    """Registered as ``Glm5NextMTPModel`` for speculative decoding."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Add ``.mtp_block`` for transformer-block weights of the spec
        layer and hoist shared weights (embeddings) to the top level."""
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Normalize the composite checkpoint prefix so spec-layer detection
        # (model.layers.45.*) matches.
        def remap_prefix(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            lm_prefix = "model.language_model."
            for name, weight in weights:
                if name.startswith(lm_prefix):
                    name = "model." + name[len(lm_prefix) :]
                yield name, weight

        text_config: Glm5NextTextConfig = self.config.text_config

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=text_config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in remap_prefix(weights):
            if name == "lm_head.weight":
                spec_layer = self.model.mtp_start_layer_idx
                name = f"model.layers.{spec_layer}.shared_head.head.weight"
            elif name == "model.embed_tokens.weight":
                spec_layer = self.model.mtp_start_layer_idx
            else:
                spec_layer = get_spec_layer_idx_from_weight_name(
                    self.config, name
                )
                if spec_layer is None:
                    continue
                name = self._rewrite_spec_layer_name(spec_layer, name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts are handled by expert_params_mapping below.
                if "mlp.experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    name = name_mapped
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name
                    ):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
