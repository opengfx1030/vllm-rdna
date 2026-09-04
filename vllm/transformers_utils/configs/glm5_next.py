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
"""GLM-5.3-Flash (Glm5Next) model configuration.

Ports the text/vision/composite configs of ``zai-org/GLM-5.3-Flash``
(transformers ``configuration_glm5_next.py``) into the vLLM custom-config
style. Defaults mirror the real checkpoint's ``config.json``:

- 45 layers; ``layer_types`` = 3x ``linear_attention`` (KDA) followed by
  1x ``deepseek_sparse_attention`` (DSA), repeating. DSA layers are
  [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43].
- ``mlp_layer_types``: layers 0-2 dense (intermediate 12288), layers 3-44
  sparse MoE (288 routed + 1 shared expert, top-8, sigmoid routing with
  ``e_score_correction_bias``, ``routed_scaling_factor`` 2.5).
- KDA: ``linear_attn_config`` = {num_heads: 64, head_dim: 128,
  short_conv_kernel_size: 4, gate_lower_bound: -5.0}.
- DSA: q_lora_rank 1536, kv_lora_rank 512, qk_nope_head_dim 256,
  qk_rope_head_dim 0 (NoPE everywhere in the LM), v_head_dim 256.
- mHC (manifold-constrained hyper-connections): hc_mult 4, hc_eps 1e-6,
  hc_sinkhorn_iters 20.
- MTP: ``num_nextn_predict_layers`` 1; checkpoint layer 45 carries the
  MTP DSA decoder layer + MoE + ``enorm``/``hnorm``/``eh_proj``/
  ``shared_head.norm``.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Glm5NextTextConfig(PretrainedConfig):
    """Configuration of the GLM-5.3-Flash text (language) model."""

    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = 154820,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # Per-layer schedules.
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        first_k_dense_replace: int = 3,
        # DSA (DeepSeek Sparse Attention / NoPE MLA) geometry.
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        qk_head_dim: int = 256,
        v_head_dim: int = 256,
        mla_use_nope: bool = True,
        # DSA indexer.
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_kpool: int = 4,
        index_kpool_always_select_tail: bool = True,
        index_kpool_compress: bool = True,
        index_share_for_mtp_iteration: bool = True,
        indexer_rope_interleave: bool = True,
        # MoE routing.
        n_routed_experts: int = 288,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        moe_router_dtype: str = "float32",
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        # KDA (Kimi-style linear attention) geometry. Kept both as the raw
        # ``linear_attn_config`` dict (checkpoint format) and as derived flat
        # attributes (modeling format).
        linear_attn_config: dict | None = None,
        linear_num_heads: int = 64,
        linear_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float | None = -5.0,
        # mHC (manifold-constrained hyper-connections).
        mhc: bool = True,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        # SwiGLU clamping.
        swiglu_limit: float = 10.0,
        # MTP.
        num_nextn_predict_layers: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.first_k_dense_replace = first_k_dense_replace

        # DSA geometry. `head_dim` follows the HF convention of being the
        # RoPE-based dim, which is 0 for this NoPE model.
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.head_dim = qk_rope_head_dim

        # DSA indexer.
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.index_kpool_compress = index_kpool_compress
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration
        self.indexer_rope_interleave = indexer_rope_interleave

        # MoE routing.
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_router_dtype = moe_router_dtype
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # KDA geometry: the checkpoint nests these in `linear_attn_config`;
        # expose them as flat attributes for the model code.
        self.linear_attn_config = linear_attn_config
        if linear_attn_config is not None:
            self.linear_num_heads = linear_attn_config.get(
                "num_heads", linear_num_heads
            )
            self.linear_head_dim = linear_attn_config.get("head_dim", linear_head_dim)
            self.linear_conv_kernel_dim = linear_attn_config.get(
                "short_conv_kernel_size", linear_conv_kernel_dim
            )
            self.linear_lower_bound = linear_attn_config.get(
                "gate_lower_bound", linear_lower_bound
            )
            if linear_attn_config.get("safe_gate", True) and (
                self.linear_lower_bound is None
            ):
                self.linear_lower_bound = -5.0
        else:
            self.linear_num_heads = linear_num_heads
            self.linear_head_dim = linear_head_dim
            self.linear_conv_kernel_dim = linear_conv_kernel_dim
            self.linear_lower_bound = linear_lower_bound

        # mHC.
        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters

        self.swiglu_limit = swiglu_limit
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Per-layer MLP schedule: the first `first_k_dense_replace` layers
        # are dense, the rest are sparse MoE.
        if mlp_layer_types is None:
            mlp_layer_types = ["dense"] * min(
                first_k_dense_replace, num_hidden_layers
            ) + ["sparse"] * (num_hidden_layers - min(
                first_k_dense_replace, num_hidden_layers
            ))
        self.mlp_layer_types = mlp_layer_types

        # Per-layer attention schedule: 3 KDA layers followed by 1 DSA layer,
        # repeating. Older checkpoints may spell `full_attention` for the DSA
        # slots; normalize to the DSA name.
        if layer_types is None:
            layer_types = [
                "linear_attention" if (idx % 4) != 3 else "deepseek_sparse_attention"
                for idx in range(num_hidden_layers)
            ]
        self.layer_types = [
            "deepseek_sparse_attention" if lt == "full_attention" else lt
            for lt in layer_types
        ]

        # Per-layer DSA indexer mode. GLM-5.3-Flash ships all-full.
        if indexer_types is None:
            indexer_types = ["full"] * num_hidden_layers
        self.indexer_types = list(indexer_types)

        # The MTP layers (checkpoint indices
        # num_hidden_layers .. num_hidden_layers + num_nextn_predict_layers - 1)
        # reuse the same decoder-layer config. Extend the schedules so the
        # model code can index them uniformly; the MTP layer of
        # GLM-5.3-Flash is a DSA + MoE layer with a full indexer.
        total_layers = num_hidden_layers + num_nextn_predict_layers
        if len(self.layer_types) < total_layers:
            self.layer_types = self.layer_types + ["deepseek_sparse_attention"] * (
                total_layers - len(self.layer_types)
            )
        if len(self.mlp_layer_types) < total_layers:
            self.mlp_layer_types = self.mlp_layer_types + ["sparse"] * (
                total_layers - len(self.mlp_layer_types)
            )
        if len(self.indexer_types) < total_layers:
            self.indexer_types = self.indexer_types + ["full"] * (
                total_layers - len(self.indexer_types)
            )

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Glm5NextVisionConfig(PretrainedConfig):
    """Configuration of the GLM-5.3-Flash vision tower (24-layer ViT)."""

    model_type = "glm5_next_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        hidden_act: str = "silu",
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        num_heads: int = 16,
        in_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 4096,
        intermediate_size: int = 4096,
        initializer_range: float = 0.02,
        projection_intermediate_size: int = 10240,
        swiglu_limit: float = 10.0,
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.rms_norm_eps = rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.projection_intermediate_size = projection_intermediate_size
        self.swiglu_limit = swiglu_limit
        super().__init__(**kwargs)


class Glm5NextConfig(PretrainedConfig):
    """Composite configuration for GLM-5.3-Flash (vision + text)."""

    model_type = "glm5_next"
    sub_configs = {
        "vision_config": Glm5NextVisionConfig,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Initialize the base first so it does not clobber sub-config values
        # with PretrainedConfig defaults (mirrors other composite configs).
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # Flat (text-only) checkpoints store the text fields at the top
            # level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

    def __getattr__(self, key):
        # Transparently expose text-config attributes (hidden_size, ...) on
        # the composite config, as HF composite configs do. Only reached when
        # the normal attribute lookup has already failed.
        text_config = self.__dict__.get("text_config")
        if text_config is not None and key in text_config.__dict__:
            return getattr(text_config, key)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {key!r}"
        )


__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "Glm5NextVisionConfig"]
