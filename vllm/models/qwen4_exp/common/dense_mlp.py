# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense SwiGLU MLP for Qwen4 checkpoints without routed experts.

The module mirrors ``Qwen2MLP`` and keeps the parameter names the Qwen4
weight loader already expects: ``mlp.gate_up_proj`` (two shards) and
``mlp.down_proj``. ``Qwen4ExpForCausalLM.packed_modules_mapping`` maps the
HF names ``mlp.gate_proj`` / ``mlp.up_proj`` onto the two shards, so a dense
checkpoint loads through the existing ``AutoWeightsLoader`` path.

Selected only by config (see ``mlp_selection.uses_dense_mlp``); the
Flash-Next MoE path is untouched.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


class Qwen4DenseMLP(nn.Module):
    """SwiGLU MLP with the Qwen4 dense checkpoint's parameter names."""

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "mlp",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = getattr(config, "intermediate_size", None)
        if not intermediate_size:
            raise ValueError(
                "dense Qwen4 MLP needs config.intermediate_size; the leaked "
                "27B geometry implies ~14336 but no checkpoint confirms it"
            )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(f"unsupported dense Qwen4 MLP activation: {hidden_act!r}")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [int(intermediate_size)] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            int(intermediate_size),
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
