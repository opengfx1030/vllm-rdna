# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Call the dest PR #37 routed-expert residency cache. Do not copy it.

The cache lives at
``vllm.model_executor.layers.fused_moe.rdna_expert_residency`` on
draft PR #37 (``ExpertResidencyCache.prepare`` and
``after_resident_gemm``). Until that module is on the branch, constructing
a bridge without an injected module raises. This file has no slot table,
no host pages, and no prefetch copier.
"""

import importlib

PR37_MODULE = "vllm.model_executor.layers.fused_moe.rdna_expert_residency"
REQUIRED_SYMBOLS = (
    "ExpertResidencyCache",
    "next_moe_index",
    "should_engage_expert_offload",
)


class ResidencyBridge:
    """Thin caller for the PR #37 cache API.

    Args:
        module: Injected module, used by tests. None imports PR #37.
    """

    def __init__(self, module: object | None = None) -> None:
        self._module = module

    def module(self):
        """Return the PR #37 module.

        Raises:
            RuntimeError: The module or a required symbol is missing.
        """
        if self._module is None:
            try:
                self._module = importlib.import_module(PR37_MODULE)
            except ImportError as exc:
                raise RuntimeError(
                    "Cold-tier expert residency is dest PR #37 "
                    f"({PR37_MODULE}). This tree does not vendor a "
                    "second expert cache."
                ) from exc
        missing = [name for name in REQUIRED_SYMBOLS if not hasattr(self._module, name)]
        if missing:
            raise RuntimeError(
                "PR #37 residency module is missing " + ", ".join(missing)
            )
        return self._module

    def prepare(self, cache, topk_ids, expert_map, capturing: bool = False):
        """Remap ids through ``cache.prepare``. The cache owns the slots.

        Args:
            cache: An ``ExpertResidencyCache`` from PR #37.
            topk_ids: Expert ids for this rank.
            expert_map: Global-to-local map, or None.
            capturing: Forwarded. The cache refuses a fetch in capture.

        Returns:
            Slot ids from the cache.
        """
        self.module()
        return cache.prepare(topk_ids, expert_map, capturing=capturing)

    def after_gemm(self, cache) -> None:
        """Record the GEMM and let the cache prefetch the next MoE layer."""
        self.module()
        cache.after_resident_gemm()

    def next_moe_index(self, kinds, index):
        """Delegate successor lookup, including the PR #37 skip rules."""
        return self.module().next_moe_index(kinds, index)

    def should_engage(self, *, enabled: bool, expert_bytes: int, vram_budget: int):
        """Delegate the engage-only-when-over-budget rule."""
        return self.module().should_engage_expert_offload(
            enabled=enabled,
            expert_bytes=expert_bytes,
            vram_budget=vram_budget,
        )
