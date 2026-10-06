# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast-tier device class. gfx1100 is implemented; GB10 is only registered."""

from dataclasses import dataclass

# ROCm pin for the gfx1030 fatbin. The gfx1100 fatbin is a separate build.
ROCM_PIN = "7.14.0"
GFX1030_COMPILE = "hipcc --offload-arch=gfx1030 -O3 -mno-wavefrontsize64"
GFX1100_COMPILE = "hipcc --offload-arch=gfx1100 -O3"
FORBIDDEN_COMPILE_FLAG = "-ffp-contract=off"

# What stays on pool A. QSA main KV and its compressed-key / indexer heap
# are one layer-class heap (``qsa_heap``), not two movable objects.
FAST_TIER_OWNS = frozenset(
    {
        "embeddings",
        "lm_head",
        "norms",
        "router",
        "shared_experts",
        "hot_routed_experts",
        "attention",
        "qsa_heap",
        "gdn_rings",
        "ple",
        "hc",
        "dense_leftovers",
        "bf16_leftovers",
        "sampler",
        "scheduler",
        "primary_kv",
    }
)

# Pool B holds routed cold experts and the LRU full-attention KV blocks.
COLD_TIER_OWNS = frozenset({"cold_routed_experts", "lru_kv_blocks"})


@dataclass(frozen=True)
class FastTierDevice:
    """A pluggable fast-tier class.

    Attributes:
        name: Registry key.
        arch: GPU arch string used to reject the wrong fatbin.
        implemented: False reserves the name without a code path.
        product: Board the class was written for.
    """

    name: str
    arch: str
    implemented: bool
    product: str

    @property
    def owns(self) -> frozenset[str]:
        return FAST_TIER_OWNS


class Gfx1100FastTier(FastTierDevice):
    """2x W7800, gfx1100, 48GB. The implemented fast tier."""

    def __init__(self) -> None:
        super().__init__(
            name="gfx1100",
            arch="gfx1100",
            implemented=True,
            product="W7800",
        )


class SparkGB10FastTier(FastTierDevice):
    """Registered so a later fast tier can replace the W7800s.

    Resolving this class raises. No GB10 kernels or copies are stubbed in.
    """

    def __init__(self) -> None:
        super().__init__(
            name="gb10",
            arch="gb10",
            implemented=False,
            product="Spark GB10",
        )


_REGISTRY: dict[str, type[FastTierDevice]] = {
    "gfx1100": Gfx1100FastTier,
    "w7800": Gfx1100FastTier,
    "gb10": SparkGB10FastTier,
    "spark": SparkGB10FastTier,
}


def resolve_fast_tier(name: str) -> FastTierDevice:
    """Return the fast-tier class.

    Args:
        name: ``gfx1100`` or ``w7800`` today. ``gb10`` is reserved.

    Returns:
        The implemented device object.

    Raises:
        KeyError: The name is not registered.
        NotImplementedError: The class is registered and not implemented.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown fast tier {name}")
    tier = _REGISTRY[key]()
    if not tier.implemented:
        raise NotImplementedError(
            f"{tier.product} is registered as a fast tier and is not "
            "implemented. The tested fast tier is gfx1100 (W7800)."
        )
    return tier


@dataclass(frozen=True)
class ColdTier:
    """8x V620, gfx1030, 32GB. Routed cold experts and the KV block tier."""

    arch: str = "gfx1030"
    product: str = "V620"
    devices: int = 8
    kernel: str = "moe_gptq_gemm_rdna2"

    @property
    def owns(self) -> frozenset[str]:
        return COLD_TIER_OWNS
