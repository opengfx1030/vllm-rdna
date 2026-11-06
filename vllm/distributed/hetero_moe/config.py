# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve-side configuration. Every knob defaults off or to host-staged."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class HeteroMoEConfig:
    """Knobs read from the server environment.

    Clients never see these. With ``enabled`` false the MoE forward does
    not consult the rest of the fields.
    """

    enabled: bool = False
    fast_tier: str = "gfx1100"
    cold_arch: str = "gfx1030"
    cold_devices: int = 8
    transport: str = "host_staged"
    hot_budget_bytes: int = 0
    ema_alpha: float = 0.2
    kv_tier: bool = False
    kv_capacity_blocks: int = 0
    peer_probe_path: str = ""
    host_link: str = "local"
    host_addr: str = ""

    def __post_init__(self) -> None:
        if self.transport not in ("loopback", "host_staged", "peer"):
            raise ValueError(f"unknown hetero transport {self.transport}")
        if self.host_link not in ("local", "shm", "tcp"):
            raise ValueError(f"unknown host link {self.host_link}")
        if self.cold_devices < 1:
            raise ValueError("cold pool needs at least one device")
        if self.hot_budget_bytes < 0 or self.kv_capacity_blocks < 0:
            raise ValueError("budgets must be non-negative")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema alpha must be in (0, 1]")


def _flag(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_config() -> HeteroMoEConfig:
    """Build a config from the current process environment."""
    return HeteroMoEConfig(
        enabled=_flag("VLLM_HETERO_MOE", "0") == "1",
        fast_tier=_flag("VLLM_HETERO_MOE_FAST_TIER", "gfx1100"),
        cold_devices=int(_flag("VLLM_HETERO_MOE_COLD_DEVICES", "8")),
        transport=_flag("VLLM_HETERO_MOE_TRANSPORT", "host_staged"),
        hot_budget_bytes=int(_flag("VLLM_HETERO_MOE_HOT_BUDGET_BYTES", "0")),
        ema_alpha=float(_flag("VLLM_HETERO_MOE_EMA_ALPHA", "0.2")),
        kv_tier=_flag("VLLM_HETERO_MOE_KV_TIER", "0") == "1",
        kv_capacity_blocks=int(_flag("VLLM_HETERO_MOE_KV_CAPACITY_BLOCKS", "0")),
        peer_probe_path=_flag("VLLM_HETERO_MOE_PEER_PROBE", ""),
        host_link=_flag("VLLM_HETERO_MOE_HOST_LINK", "local"),
        host_addr=_flag("VLLM_HETERO_MOE_HOST_ADDR", ""),
    )
