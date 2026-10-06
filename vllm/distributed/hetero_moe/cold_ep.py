# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-parallel map for the cold pool, and the residency hand-off.

Cold experts are a contiguous shard across the V620s. Weight residency
is the dest PR #37 cache (``prepare`` / ``after_resident_gemm``). This
module does not allocate a second expert cache.
"""


def shard_sizes(num_experts: int, num_devices: int) -> list[int]:
    """Contiguous expert counts. The first devices absorb the remainder.

    Args:
        num_experts: Experts placed on the cold pool.
        num_devices: gfx1030 devices in the pool.

    Returns:
        One length per device, summing to ``num_experts``.
    """
    if num_experts < 0 or num_devices < 1:
        raise ValueError("need a non-negative expert count and one device")
    base = num_experts // num_devices
    extra = num_experts % num_devices
    return [base + (1 if index < extra else 0) for index in range(num_devices)]


def expert_owner(expert_id: int, num_experts: int, num_devices: int) -> int:
    """Device that owns ``expert_id`` under :func:`shard_sizes`.

    Args:
        expert_id: Global routed-expert id.
        num_experts: Experts in the cold map.
        num_devices: gfx1030 devices.

    Returns:
        Device index in ``[0, num_devices)``.
    """
    if expert_id < 0 or expert_id >= num_experts:
        raise ValueError("expert id out of range")
    start = 0
    for device, size in enumerate(shard_sizes(num_experts, num_devices)):
        if expert_id < start + size:
            return device
        start += size
    raise RuntimeError("shard map did not cover the expert")


class ColdPoolRunner:
    """Run cold pairs on the gfx1030 experts class after residency prepare.

    Args:
        arch: Must be ``gfx1030`` for a real launch.
        bridge: PR #37 adapter. Optional in tests that have no cache.
        num_experts: Width of the owner map.
        num_devices: Cold-pool device count.
    """

    def __init__(
        self,
        *,
        arch: str,
        bridge: object | None,
        num_experts: int,
        num_devices: int,
        apply_fn,
    ) -> None:
        self.arch = arch
        self.bridge = bridge
        self.num_experts = num_experts
        self.num_devices = num_devices
        self.apply_fn = apply_fn

    def owners(self, expert_ids) -> list[int]:
        """Map each cold expert id to a pool device."""
        return [
            expert_owner(int(expert), self.num_experts, self.num_devices)
            for expert in expert_ids
        ]

    def apply(self, layer, hidden, topk_weights, topk_ids, expert_map=None):
        """Prepare residency, then call the existing experts apply.

        Args:
            layer: Cold-pool layer. ``cold_kernel`` or the experts class
                name must select the gfx1030 W4A16 object.
            hidden: Token rows for the cold pairs.
            topk_weights: Router weights.
            topk_ids: Expert ids.
            expert_map: Forwarded to the PR #37 ``prepare``.

        Returns:
            Whatever ``apply_fn`` returns.

        Raises:
            RuntimeError: The layer would run a non-gfx1030 kernel, or
                the arch is not gfx1030.
        """
        from vllm.distributed.hetero_moe.kernels import (
            COLD_W4A16_KERNEL,
            assert_device_kernel,
        )

        kernel = getattr(layer, "cold_kernel", COLD_W4A16_KERNEL)
        experts = getattr(layer, "experts", None)
        if experts is not None:
            kernel = type(experts).__name__
        assert_device_kernel(self.arch, kernel)
        if self.arch != "gfx1030":
            raise RuntimeError(
                f"cold experts run on gfx1030 ({COLD_W4A16_KERNEL}); got {self.arch}"
            )
        cache = getattr(layer, "_expert_residency", None)
        if cache is not None:
            if self.bridge is None:
                raise RuntimeError(
                    "cold experts with a residency cache require the "
                    "PR #37 bridge; a second cache is not built here"
                )
            topk_ids = self.bridge.prepare(
                cache,
                topk_ids,
                expert_map,
                capturing=False,
            )
        out = self.apply_fn(
            layer=layer,
            x=hidden,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        if cache is not None:
            self.bridge.after_gemm(cache)
        return out
