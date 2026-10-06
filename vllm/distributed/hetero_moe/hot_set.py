# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFU/EMA hot-expert placement under one per-device VRAM budget.

Re-placement is a host step between engine steps. It is refused while a
HIP/CUDA graph is capturing. The histogram is per MoE layer. One budget
covers every layer on that fast-tier device, so the same expert bytes
are not charged once per device and then again for a replica.
"""

from collections.abc import Mapping

import torch

from vllm.distributed.hetero_moe.capture import assert_outside_capture


class HotSetTracker:
    """Per-layer expert scores and the placed hot set.

    Args:
        num_experts: Routed experts in each layer.
        alpha: EMA rate in ``(0, 1]``. ``1`` keeps only the latest step.
    """

    def __init__(self, num_experts: int, alpha: float = 0.2) -> None:
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ema alpha must be in (0, 1]")
        self.num_experts = num_experts
        self.alpha = alpha
        self.count: dict[int, torch.Tensor] = {}
        self.ema: dict[int, torch.Tensor] = {}
        self.hot: dict[int, set[int]] = {}
        self.capturing = False

    def _ensure(self, layer: int) -> None:
        if layer not in self.ema:
            zeros = torch.zeros(self.num_experts, dtype=torch.float64)
            self.ema[layer] = zeros
            self.count[layer] = zeros.clone()
            self.hot[layer] = set()

    def observe(self, layer: int, topk_ids: torch.Tensor) -> None:
        """Fold one step of ``topk_ids`` into the LFU counts and the EMA.

        Args:
            layer: MoE layer index.
            topk_ids: Expert ids, any shape. Negatives are padding.

        Raises:
            RuntimeError: Called while capture is marked or live.
        """
        assert_outside_capture(self.capturing)
        self._ensure(layer)
        flat = topk_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
        flat = flat[flat >= 0]
        if flat.numel() == 0:
            return
        if int(flat.max()) >= self.num_experts:
            raise ValueError("expert id outside the layer expert count")
        bc = torch.bincount(flat, minlength=self.num_experts).to(torch.float64)
        freq = bc / bc.sum()
        self.count[layer] = self.count[layer] + bc
        self.ema[layer] = (1.0 - self.alpha) * self.ema[layer] + self.alpha * freq

    def place(
        self,
        budget_bytes: int,
        bytes_per_expert: int | Mapping[int, int],
    ) -> dict[int, set[int]]:
        """Choose the hot set that fits in ``budget_bytes``.

        Experts are ordered by EMA, then by LFU count, then by smaller
        id. An expert that was never routed is left cold. A candidate
        that does not fit is skipped so a later, smaller expert can.

        Args:
            budget_bytes: VRAM bytes this fast-tier device may spend on
                hot routed experts, summed across layers.
            bytes_per_expert: One size, or a size per layer index.

        Returns:
            ``layer -> hot expert ids``. Also stored on ``self.hot``.
        """
        assert_outside_capture(self.capturing)
        if budget_bytes < 0:
            raise ValueError("budget must be non-negative")
        chosen: dict[int, set[int]] = {layer: set() for layer in self.ema}
        if budget_bytes == 0 or not self.ema:
            self.hot = chosen
            return chosen
        candidates: list[tuple[float, float, int, int, int, int]] = []
        for layer, ema in self.ema.items():
            cost = _expert_cost(layer, bytes_per_expert)
            if cost < 0:
                raise ValueError("expert bytes must be non-negative")
            if cost == 0:
                continue
            counts = self.count[layer]
            for expert in range(self.num_experts):
                score = float(ema[expert])
                hits = float(counts[expert])
                if score <= 0.0 and hits <= 0.0:
                    continue
                candidates.append((score, hits, -expert, layer, expert, cost))
        candidates.sort(reverse=True)
        spent = 0
        for _score, _hits, _neg, layer, expert, cost in candidates:
            if spent + cost > budget_bytes:
                continue
            chosen[layer].add(expert)
            spent += cost
        self.hot = chosen
        return chosen


def _expert_cost(layer: int, bytes_per_expert: int | Mapping[int, int]) -> int:
    if isinstance(bytes_per_expert, int):
        return bytes_per_expert
    return int(bytes_per_expert[layer])
