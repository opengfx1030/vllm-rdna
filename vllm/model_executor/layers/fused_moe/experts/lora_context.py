# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch


class LaneMoELoRAContextView:
    """A MoE LoRA context binding that resolves per calling lane.

    The MoE kernel keeps one context per layer, and both lanes forward through
    that layer. The context carries lane-mutable state (the aux stream and its
    events, and ``original_hidden_states`` written per forward), so lane mode
    keeps one context per lane and stashes this view; every read and write
    lands on the calling thread's lane context.
    """

    __slots__ = ("_contexts",)

    def __init__(self, contexts: dict[int, Any]):
        object.__setattr__(self, "_contexts", contexts)

    def _resolve(self) -> "MoELoRAContext":
        from vllm.v1.lane import current_lane, lane_mode_enabled

        contexts: dict[int, MoELoRAContext] = object.__getattribute__(self, "_contexts")
        if not lane_mode_enabled():
            return next(iter(contexts.values()))
        return contexts[int(current_lane())]

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __setattr__(self, name, value):
        if name == "_contexts":
            object.__setattr__(self, name, value)
            return
        setattr(self._resolve(), name, value)


@dataclass
class MoELoRAContext:
    """Carries all LoRA state for one MoE forward pass.

    Built by FusedMoEWithLoRA.forward() and propagated explicitly through the
    modular kernel path (FusedMoEKernel -> FusedMoEExpertsModular.apply) so
    that TritonExperts.apply() can compute the LoRA contribution inline,
    replacing the decorator-based monkey-patch approach.
    """

    # LoRA weight tensors (same shapes as FusedMoEWithLoRA attributes)
    w13_lora_a_stacked: tuple[torch.Tensor, ...]
    w13_lora_b_stacked: tuple[torch.Tensor, ...]
    w2_lora_a_stacked: tuple[torch.Tensor, ...]
    w2_lora_b_stacked: tuple[torch.Tensor, ...]

    # (max_loras + 1,) int32; slot 0 is the "no-adapter" sentinel
    adapter_enabled: torch.Tensor

    # Metadata
    max_loras: int
    top_k: int
    w13_num_slices: int  # 2 = gated (gate + up), 1 = non-gated or 3D-fused
    fully_sharded: bool
    tp_rank: int
    tp_size: int
    local_num_experts: int

    # The per-lane punica wrapper (a PunicaWrapperBase) or, in lane mode, the
    # view that resolves it per lane.
    punica_wrapper: Any

    # Whether VLLM_TUNED_CONFIG_FOLDER is set; selects get_lora_op_configs vs
    # try_get_optimal_moe_lora_config for Triton kernel tile configs.
    use_tuned_config: bool

    # Shared MoE LoRA: w13 lora_A and w2 lora_B are shared across all experts
    # (stored collapsed with expert-dim 1). When True, LoRAExpertsMixin expands
    # them to local_num_experts via a stride-0 view before the kernel.
    enable_moe_shared_loras: bool = False

    # Optional dual-stream support for overlapping each (base GEMM, LoRA)
    # pair. When aux_stream is None, the experts.apply() path runs the
    # original sequential schedule. When set, base GEMM runs on the default
    # stream and the LoRA fast-path writes the delta into a fresh buffer on
    # aux_stream, which the default stream sums in afterwards.
    # Events are paired one-per-overlap-pair: events[0,1] for w13,
    # events[2,3] for w2, so the two pairs do not race on the same event.
    aux_stream: torch.cuda.Stream | None = None
    events: tuple[torch.cuda.Event, ...] | None = None

    # Per-rank token→LoRA mapping after EP dispatch. Set by
    # FusedMoEPrepareAndFinalizeModular.prepare() when EP+LoRA is active, read
    # by LoRAExpertsMixin helpers in place of punica_wrapper's global mapping.
    # None means no dispatch happened (non-EP path), in which case callers
    # fall back to punica_wrapper.token_mapping_meta.
    local_token_lora_mapping: torch.Tensor | None = None

    # Original unquantized hidden states, stashed by the modular kernel
    # before the prepare step potentially quantizes them. Used by
    # apply_w13_lora so the LoRA kernel sees correct-magnitude activations
    # instead of raw quantized values that are missing the activation scale.
    # Set per forward pass; None until the modular kernel writes it.
    original_hidden_states: torch.Tensor | None = None
