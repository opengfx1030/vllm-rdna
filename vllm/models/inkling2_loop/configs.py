# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configs for thinkingmachines/Inkling-2-Mini (``inkling2_loop``).

Self-contained so serving does not depend on the installed ``transformers``
shipping ``inkling2_loop``.  Field names and defaults mirror the reference
(``transformers/models/inkling2_loop/configuration_inkling2_loop.py``); only the
text backbone is modelled, the vision tower config stays an opaque dict.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PreTrainedConfig


class Inkling2LoopTextConfig(PreTrainedConfig):
    model_type = "inkling2_loop_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        padded_vocab_size: int | None = None,
        hidden_size: int = 2304,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int = 24,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 128,
        sliding_window_size: int = 1024,
        local_layer_ids: list[int] | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 8192,
        log_scaling_alpha: float = 0.1,
        max_position_embeddings: int | None = None,
        model_max_length: int = 1048576,
        rms_norm_eps: float = 1e-6,
        sconv_kernel_size: int = 4,
        intermediate_size: int = 1152,
        n_routed_experts: int = 128,
        n_shared_experts: int = 0,
        num_experts_per_tok: int = 8,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        q_bias: bool = False,
        o_bias: bool = False,
        use_value_norm_no_gamma: bool = True,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        final_logit_softcapping: float = 30.0,
        scoring_func: str = "sigmoid",
        hidden_act: str = "silu",
        loop_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        loop_config = loop_config or {}
        # The checkpoint pads ``embed`` / ``unembed`` to ``vocab_size`` rows but only
        # the first ``unpadded_vocab_size`` ids exist (the reference head keeps just
        # those).  vLLM sizes the logits, the sampler state and the token-id
        # validation from ``vocab_size``, so the serving config exposes the real
        # vocabulary there and keeps the padded table height in
        # ``padded_vocab_size`` (idempotent across ``to_dict`` / ``from_dict``).
        self.padded_vocab_size = (
            padded_vocab_size if padded_vocab_size is not None else vocab_size
        )
        self.vocab_size = unpadded_vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        if local_layer_ids is None:
            # Reference default: every sixth layer is full attention.
            local_layer_ids = [i for i in range(num_hidden_layers) if (i + 1) % 6 != 0]
        self.local_layer_ids = list(local_layer_ids)
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.model_max_length = model_max_length
        self.max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else model_max_length
        )
        self.rms_norm_eps = rms_norm_eps
        self.sconv_kernel_size = sconv_kernel_size
        self.intermediate_size = intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.final_logit_softcapping = final_logit_softcapping
        self.scoring_func = scoring_func
        self.hidden_act = hidden_act
        self.loop_layer_index_start = int(loop_config.get("core_start", 6))
        self.loop_layer_index_end = int(loop_config.get("core_end", 18))
        self.loop_num_passes = int(loop_config.get("num_passes", 3))
        self.loop_gate_rank = int(loop_config.get("gate_rank", 64))
        self.loop_config = dict(loop_config)
        if self.loop_num_passes < 1:
            raise ValueError("loop_num_passes must be >= 1")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def num_loop_layers(self) -> int:
        return self.loop_layer_index_end - self.loop_layer_index_start

    @property
    def total_num_hidden_layers(self) -> int:
        """Decoder layers visited in one forward pass, looped layers counted
        once per pass.  This is also the number of attention KV caches."""
        return (
            self.num_hidden_layers + (self.loop_num_passes - 1) * self.num_loop_layers
        )

    def is_sliding_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids


class Inkling2LoopConfig(PreTrainedConfig):
    model_type = "inkling2_loop"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"text_config": Inkling2LoopTextConfig}

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | Inkling2LoopTextConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        mtp_config: dict[str, Any] | None = None,
        image_token_id: int = 200054,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config
            if isinstance(text_config, Inkling2LoopTextConfig)
            else Inkling2LoopTextConfig(**(text_config or {}))
        )
        self.vision_config = vision_config
        self.mtp_config = mtp_config
        self.image_token_id = image_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> Inkling2LoopTextConfig:
        return self.text_config

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim
