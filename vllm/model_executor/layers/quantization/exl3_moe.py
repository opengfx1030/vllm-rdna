# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 routed experts. K is whatever the trellis stores (1..8).

Gate and up are quantized separately, so each keeps its own suh/svh.
The forward runs the same Hadamard + trellis GEMM as a dense EXL3 linear,
once per expert that the router selected.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.exl3 import Exl3LinearMethod
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def _slot(shard_id: str) -> int:
    if shard_id == "w1":
        return 0
    if shard_id == "w3":
        return 1
    raise ValueError(f"EXL3 MoE shard {shard_id}")


_CB = {"3inst": 0, "mcg": 1, "mul1": 2}


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, hadamard: str = "both",
                 codebook: str = "3inst") -> None:
        super().__init__(moe)
        self.hadamard = hadamard
        self.cb = _CB.get(codebook, 0)
        self._linear = Exl3LinearMethod(bits=None, hadamard=hadamard)
        self.w13_bits: int | None = None
        self.w2_bits: int | None = None

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        loader = extra_weight_attrs.get("weight_loader")
        e = num_experts
        h = hidden_size // 16
        n = intermediate_size_per_partition // 16
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 MoE needs K and N divisible by 16, got "
                f"hidden={hidden_size} intermediate="
                f"{intermediate_size_per_partition}")
        # Words (16*K) are replaced on the first trellis. Gate and up are
        # stored side by side so their scales stay independent.
        device = torch.device("cuda")
        for name in ("w13_weight", "w2_weight"):
            dummy = torch.nn.Parameter(
                torch.empty(e, 1, device=device), requires_grad=False)
            if loader is not None:
                set_weight_attrs(dummy, {"weight_loader": loader})
            layer.register_parameter(name, dummy)
        layer.register_parameter(
            "w13_trellis",
            torch.nn.Parameter(
                torch.empty(e, 2, h, n, 16, dtype=torch.int16, device=device),
                requires_grad=False))
        layer.register_parameter(
            "w13_suh",
            torch.nn.Parameter(
                torch.empty(e, 2, hidden_size, dtype=torch.float16,
                            device=device),
                requires_grad=False))
        layer.register_parameter(
            "w13_svh",
            torch.nn.Parameter(
                torch.empty(e, 2, intermediate_size_per_partition,
                            dtype=torch.float16, device=device),
                requires_grad=False))
        layer.register_parameter(
            "w2_trellis",
            torch.nn.Parameter(
                torch.empty(e, n, h, 16, dtype=torch.int16, device=device),
                requires_grad=False))
        layer.register_parameter(
            "w2_suh",
            torch.nn.Parameter(
                torch.empty(e, intermediate_size_per_partition,
                            dtype=torch.float16, device=device),
                requires_grad=False))
        layer.register_parameter(
            "w2_svh",
            torch.nn.Parameter(
                torch.empty(e, hidden_size, dtype=torch.float16, device=device),
                requires_grad=False))
        self._hidden = hidden_size
        self._intermediate = intermediate_size_per_partition

    def _fit_words(self, layer, name: str, words: int) -> torch.nn.Parameter:
        param = getattr(layer, name)
        if param.shape[-1] == words:
            return param
        shape = param.shape[:-1] + (words,)
        fresh = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.int16, device=param.device),
            requires_grad=False)
        layer._parameters[name] = fresh
        return fresh

    def _slice_tp(self, tensor: torch.Tensor, shard_id: str, along_k: bool,
                  layer) -> torch.Tensor:
        tp = int(layer.moe_config.tp_size)
        rank = int(layer.moe_config.tp_rank)
        if tp <= 1:
            return tensor
        # Column-parallel gate/up shard N. Row-parallel down shards K.
        dim = 0 if along_k else 1 if tensor.dim() >= 2 else 0
        if tensor.dim() == 1:
            dim = 0
        n = tensor.shape[dim]
        local = n // tp
        if n == local:
            return tensor
        if n != local * tp:
            raise ValueError(
                f"EXL3 MoE cannot shard dim {n} across tp={tp}")
        sl = slice(rank * local, (rank + 1) * local)
        return tensor[sl] if dim == 0 else tensor[:, sl]

    def absorb(self, layer, loaded, weight_name, shard_id, expert_id) -> None:
        kind = weight_name.rsplit(".", 1)[-1]
        loaded = loaded.detach()
        if loaded.dim() >= 1 and loaded.shape[0] == 1 and kind != "trellis":
            loaded = loaded.squeeze(0)
        is_down = shard_id == "w2"
        if kind == "trellis":
            words = int(loaded.shape[-1])
            bits = words // 16
            if words % 16 or not 1 <= bits <= 8:
                raise ValueError(f"EXL3 MoE trellis width {words}")
            if is_down:
                loaded = self._slice_tp(loaded, shard_id, along_k=True,
                                        layer=layer)
                self.w2_bits = bits
                param = self._fit_words(layer, "w2_trellis", words)
                param.data[expert_id].copy_(loaded.to(param.device))
            else:
                loaded = self._slice_tp(loaded, shard_id, along_k=False,
                                        layer=layer)
                self.w13_bits = bits
                param = self._fit_words(layer, "w13_trellis", words)
                param.data[expert_id, _slot(shard_id)].copy_(
                    loaded.to(param.device))
            return
        scale = loaded.to(torch.float16).reshape(-1)
        if is_down:
            if scale.numel() == self._intermediate or (
                    scale.numel() % max(layer.moe_config.tp_size, 1) == 0
                    and scale.numel() != self._hidden):
                scale = self._slice_tp(scale, shard_id, True, layer)
                layer.w2_suh.data[expert_id].copy_(scale.to(layer.w2_suh.device))
            else:
                layer.w2_svh.data[expert_id].copy_(scale.to(layer.w2_svh.device))
        else:
            if scale.numel() == self._hidden:
                layer.w13_suh.data[expert_id, _slot(shard_id)].copy_(
                    scale.to(layer.w13_suh.device))
            else:
                scale = self._slice_tp(scale, shard_id, False, layer)
                layer.w13_svh.data[expert_id, _slot(shard_id)].copy_(
                    scale.to(layer.w13_svh.device))

    def _one(self, x, trellis, suh, svh, bits) -> torch.Tensor:
        m, k = x.shape
        n = svh.numel()
        xh = torch.zeros(m, k, dtype=torch.float16, device=x.device)
        mid = torch.zeros(m, n, dtype=torch.float16, device=x.device)
        out = torch.zeros(m, n, dtype=torch.float16, device=x.device)
        self._linear.bits = bits
        self._linear.cb = self.cb
        self._linear._project(
            x, xh, mid, out, suh, svh, trellis.contiguous(), bits, self.cb)
        return out

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        x = x.reshape(-1, x.shape[-1]).to(torch.float16)
        m = x.shape[0]
        topk_ids = topk_ids.reshape(m, -1)
        topk_weights = topk_weights.reshape(m, -1).to(torch.float16)
        hidden = x.shape[-1]
        out = torch.zeros(m, hidden, dtype=torch.float16, device=x.device)
        emap = getattr(layer, "expert_map", None)
        act = str(getattr(layer, "activation", "silu")).lower()
        experts = torch.unique(topk_ids)
        for e in experts.tolist():
            if e < 0:
                continue
            local = e
            if emap is not None:
                local = int(emap[e])
                if local < 0:
                    continue
            which = topk_ids == e
            token_idx = which.any(dim=1).nonzero(as_tuple=False).flatten()
            xe = x.index_select(0, token_idx)
            gate = self._one(
                xe, layer.w13_trellis[local, 0], layer.w13_suh[local, 0],
                layer.w13_svh[local, 0], int(self.w13_bits or 3))
            up = self._one(
                xe, layer.w13_trellis[local, 1], layer.w13_suh[local, 1],
                layer.w13_svh[local, 1], int(self.w13_bits or 3))
            if "silu" in act or "swiglu" in act:
                hidden_e = F.silu(gate) * up
            else:
                hidden_e = F.gelu(gate) * up
            down = self._one(
                hidden_e, layer.w2_trellis[local], layer.w2_suh[local],
                layer.w2_svh[local], int(self.w2_bits or self.w13_bits or 3))
            w = topk_weights[token_idx].masked_fill(
                ~which[token_idx], 0).sum(dim=1, keepdim=True)
            out.index_add_(0, token_idx, down * w)
        return out
