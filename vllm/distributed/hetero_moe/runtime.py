# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide hetero runtime. Installed only after the flag is on."""

import torch

from vllm.distributed.hetero_moe.capture import assert_outside_capture
from vllm.distributed.hetero_moe.config import HeteroMoEConfig, load_config
from vllm.distributed.hetero_moe.hot_set import HotSetTracker
from vllm.distributed.hetero_moe.transport import build_transport

_RUNTIME = None


def _unset_cold_fn(hidden, expert_ids, router_weights):
    raise RuntimeError(
        "no cold runner is installed. The gfx1030 pool runs "
        "RDNA2W4A16MoEExperts / moe_gptq_gemm_rdna2 after the PR #37 "
        "residency prepare. This process will not fall back to another kernel."
    )


class HeteroRuntime:
    """Hot-set tracker plus the cold transport for one server process.

    Args:
        config: Serve-side config.
        transport: Built transport.
        num_experts: Routed experts per layer, if already known.
    """

    def __init__(self, config: HeteroMoEConfig, transport, num_experts: int | None):
        self.config = config
        self.transport = transport
        self.num_experts = num_experts
        self.tracker = (
            None
            if num_experts is None
            else HotSetTracker(num_experts, config.ema_alpha)
        )
        self._pending: list[tuple[int, torch.Tensor]] = []

    def bind_experts(self, num_experts: int) -> None:
        """Create the tracker on the first layer that reports a width."""
        if self.tracker is None:
            self.tracker = HotSetTracker(num_experts, self.config.ema_alpha)
            self.num_experts = num_experts
        elif self.tracker.num_experts != num_experts:
            raise RuntimeError("hetero hot set expert count changed")

    def note_routing(self, layer: int, topk_ids: torch.Tensor) -> None:
        """Remember ids for the between-step update. Does not sync."""
        self._pending.append((layer, topk_ids.detach()))

    def hot_experts(self, layer: int) -> set[int]:
        """Placed hot ids. Empty until :meth:`finish_step` runs."""
        if self.tracker is None:
            return set()
        return set(self.tracker.hot.get(layer, ()))

    def finish_step(
        self,
        bytes_per_expert: int | dict[int, int],
    ) -> dict[int, set[int]]:
        """Update the histogram and re-place. Call between engine steps.

        Args:
            bytes_per_expert: Bytes of one routed expert, or per layer.

        Returns:
            The new hot sets.

        Raises:
            RuntimeError: Called during graph capture, or before the
                expert width is known.
        """
        if self.tracker is None:
            raise RuntimeError("finish_step before any expert width was bound")
        assert_outside_capture(self.tracker.capturing)
        for layer, ids in self._pending:
            self.tracker.observe(layer, ids)
        self._pending.clear()
        return self.tracker.place(self.config.hot_budget_bytes, bytes_per_expert)


def install_runtime(runtime: HeteroRuntime) -> None:
    """Replace the process runtime. Tests use this."""
    global _RUNTIME
    _RUNTIME = runtime


def reset_runtime() -> None:
    """Drop the process runtime."""
    global _RUNTIME
    _RUNTIME = None


def get_runtime() -> HeteroRuntime:
    """Return the runtime, building a default one from the environment."""
    global _RUNTIME
    if _RUNTIME is None:
        config = load_config()
        transport = build_transport(config, _unset_cold_fn)
        _RUNTIME = HeteroRuntime(config, transport, num_experts=None)
    return _RUNTIME
