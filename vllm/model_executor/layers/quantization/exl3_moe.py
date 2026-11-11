# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 routed experts. K is whatever the trellis stores (1..8).

Gate and up are quantized separately, so each keeps its own suh/svh.
The forward runs the same Hadamard + trellis GEMM as a dense EXL3 linear,
once per expert that the router selected.
"""

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3LinearMethod,
    _exl3_hadamard,
    exl3_param_device,
)
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
# Debug / A-B: force the per-expert reference loop (eager only).
_MOE_REF = os.environ.get("VLLM_EXL3_MOE_REF") == "1"


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, hadamard: str = "both", codebook: str = "3inst") -> None:
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
                f"{intermediate_size_per_partition}"
            )
        # Words (16*K) are replaced on the first trellis. Gate and up are
        # stored side by side so their scales stay independent.
        device = exl3_param_device()
        for name in ("w13_weight", "w2_weight"):
            dummy = torch.nn.Parameter(
                torch.empty(e, 1, device=device), requires_grad=False
            )
            if loader is not None:
                set_weight_attrs(dummy, {"weight_loader": loader})
            layer.register_parameter(name, dummy)
        layer.register_parameter(
            "w13_trellis",
            torch.nn.Parameter(
                torch.empty(e, 2, h, n, 16, dtype=torch.int16, device=device),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_suh",
            torch.nn.Parameter(
                torch.empty(e, 2, hidden_size, dtype=torch.float16, device=device),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_svh",
            torch.nn.Parameter(
                torch.empty(
                    e,
                    2,
                    intermediate_size_per_partition,
                    dtype=torch.float16,
                    device=device,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_trellis",
            torch.nn.Parameter(
                torch.empty(e, n, h, 16, dtype=torch.int16, device=device),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_suh",
            torch.nn.Parameter(
                torch.empty(
                    e,
                    intermediate_size_per_partition,
                    dtype=torch.float16,
                    device=device,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_svh",
            torch.nn.Parameter(
                torch.empty(e, hidden_size, dtype=torch.float16, device=device),
                requires_grad=False,
            ),
        )
        self._hidden = hidden_size
        self._intermediate = intermediate_size_per_partition

    def _fit_words(self, layer, name: str, words: int) -> torch.nn.Parameter:
        param = getattr(layer, name)
        if param.shape[-1] == words:
            return param
        shape = param.shape[:-1] + (words,)
        fresh = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.int16, device=param.device),
            requires_grad=False,
        )
        layer._parameters[name] = fresh
        return fresh

    def _slice_tp(
        self, tensor: torch.Tensor, shard_id: str, along_k: bool, layer
    ) -> torch.Tensor:
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
            raise ValueError(f"EXL3 MoE cannot shard dim {n} across tp={tp}")
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
                loaded = self._slice_tp(loaded, shard_id, along_k=True, layer=layer)
                self.w2_bits = bits
                param = self._fit_words(layer, "w2_trellis", words)
                param.data[expert_id].copy_(loaded.to(param.device))
            else:
                loaded = self._slice_tp(loaded, shard_id, along_k=False, layer=layer)
                self.w13_bits = bits
                param = self._fit_words(layer, "w13_trellis", words)
                param.data[expert_id, _slot(shard_id)].copy_(loaded.to(param.device))
            return
        scale = loaded.to(torch.float16).reshape(-1)
        if is_down:
            if scale.numel() == self._intermediate or (
                scale.numel() % max(layer.moe_config.tp_size, 1) == 0
                and scale.numel() != self._hidden
            ):
                scale = self._slice_tp(scale, shard_id, True, layer)
                layer.w2_suh.data[expert_id].copy_(scale.to(layer.w2_suh.device))
            else:
                layer.w2_svh.data[expert_id].copy_(scale.to(layer.w2_svh.device))
        else:
            if scale.numel() == self._hidden:
                layer.w13_suh.data[expert_id, _slot(shard_id)].copy_(
                    scale.to(layer.w13_suh.device)
                )
            else:
                scale = self._slice_tp(scale, shard_id, False, layer)
                layer.w13_svh.data[expert_id, _slot(shard_id)].copy_(
                    scale.to(layer.w13_svh.device)
                )

    def _one(self, x, trellis, suh, svh, bits) -> torch.Tensor:
        m, k = x.shape
        n = svh.numel()
        xh = torch.zeros(m, k, dtype=torch.float16, device=x.device)
        mid = torch.zeros(m, n, dtype=torch.float16, device=x.device)
        out = torch.zeros(m, n, dtype=torch.float16, device=x.device)
        self._linear.bits = bits
        self._linear.cb = self.cb
        self._linear._project(
            x, xh, mid, out, suh, svh, trellis.contiguous(), bits, self.cb
        )
        return out

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        # The MoE kernel reads [E, K/16, N/16, words] per expert, so gate and
        # up must each be contiguous: [E, 2, ...] -> [2, E, ...] once.
        w13 = layer.w13_trellis
        if w13.dim() == 5 and w13.shape[1] == 2:
            layer._parameters["w13_trellis"] = torch.nn.Parameter(
                w13.data.transpose(0, 1).contiguous(), requires_grad=False
            )
            del w13

    @staticmethod
    def _w13(layer, slot: int) -> torch.Tensor:
        w13 = layer.w13_trellis
        return w13[slot] if w13.shape[0] == 2 else w13[:, slot]

    def _rows(
        self,
        a,
        suh_rows,
        svh_rows,
        trellis,
        sorted_ids,
        expert_ids,
        ntpp,
        block_m,
        bits,
    ):
        """Per-pair EXL3 projection: H(a * suh) @ W_e, then H(.) * svh."""
        a = (a * suh_rows).contiguous()
        xh = torch.empty_like(a)
        _exl3_hadamard(a, xh, None, None, 1.0)
        mid = torch.zeros(
            a.shape[0], svh_rows.shape[1], dtype=torch.float16, device=a.device
        )
        ops.moe_exl3_gemm_rdna2(
            xh,
            mid,
            trellis,
            torch.empty(0, dtype=torch.float32, device=a.device),
            sorted_ids,
            expert_ids,
            ntpp,
            1,
            block_m,
            False,
            0,
            bits,
            self.cb,
        )
        out = torch.empty_like(mid)
        _exl3_hadamard(mid, out, None, None, 1.0)
        return out * svh_rows

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        """Fixed-shape fused MoE (cudagraph-capturable).

        Every (token, expert) pair is one row: per-row suh/svh and Hadamards
        around moe_exl3_gemm_rdna2 with top_k=1, then the weighted top-k sum.
        """
        del shared_experts, shared_experts_input
        if self.hadamard != "both" or _MOE_REF:
            return self.apply_reference(layer, x, topk_weights, topk_ids)
        x = x.reshape(-1, x.shape[-1]).to(torch.float16)
        m, hidden = x.shape
        topk_ids = topk_ids.reshape(m, -1)
        top_k = topk_ids.shape[1]
        pairs = m * top_k
        flat_ids = topk_ids.reshape(-1)
        emap = getattr(layer, "expert_map", None)
        local = flat_ids.long() if emap is None else emap[flat_ids.long()].long()
        valid = local >= 0
        local = local.clamp_min(0)
        num_experts = (
            int(emap.shape[0]) if emap is not None else int(layer.w2_trellis.shape[0])
        )
        block_m = 8 if pairs >= 64 else 4 if pairs >= 16 else 1
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            flat_ids.view(pairs, 1), block_m, num_experts, expert_map=emap
        )
        w13_bits = int(self.w13_bits or 3)
        w2_bits = int(self.w2_bits or self.w13_bits or 3)
        xp = x.repeat_interleave(top_k, dim=0)
        gate = self._rows(
            xp,
            layer.w13_suh[local, 0],
            layer.w13_svh[local, 0],
            self._w13(layer, 0),
            sorted_ids,
            expert_ids,
            ntpp,
            block_m,
            w13_bits,
        )
        up = self._rows(
            xp,
            layer.w13_suh[local, 1],
            layer.w13_svh[local, 1],
            self._w13(layer, 1),
            sorted_ids,
            expert_ids,
            ntpp,
            block_m,
            w13_bits,
        )
        act = str(getattr(layer, "activation", "silu")).lower()
        if "silu" in act or "swiglu" in act:
            hidden_p = F.silu(gate) * up
        else:
            hidden_p = F.gelu(gate) * up
        down = self._rows(
            hidden_p,
            layer.w2_suh[local],
            layer.w2_svh[local],
            layer.w2_trellis,
            sorted_ids,
            expert_ids,
            ntpp,
            block_m,
            w2_bits,
        )
        w = topk_weights.reshape(-1).to(torch.float16) * valid.to(torch.float16)
        return (down * w[:, None]).view(m, top_k, hidden).sum(dim=1)

    def apply_reference(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        """Per-expert loop (host-synchronizing); the reference for apply."""
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
                xe,
                self._w13(layer, 0)[local],
                layer.w13_suh[local, 0],
                layer.w13_svh[local, 0],
                int(self.w13_bits or 3),
            )
            up = self._one(
                xe,
                self._w13(layer, 1)[local],
                layer.w13_suh[local, 1],
                layer.w13_svh[local, 1],
                int(self.w13_bits or 3),
            )
            if "silu" in act or "swiglu" in act:
                hidden_e = F.silu(gate) * up
            else:
                hidden_e = F.gelu(gate) * up
            down = self._one(
                hidden_e,
                layer.w2_trellis[local],
                layer.w2_suh[local],
                layer.w2_svh[local],
                int(self.w2_bits or self.w13_bits or 3),
            )
            w = (
                topk_weights[token_idx]
                .masked_fill(~which[token_idx], 0)
                .sum(dim=1, keepdim=True)
            )
            out.index_add_(0, token_idx, down * w)
        return out
