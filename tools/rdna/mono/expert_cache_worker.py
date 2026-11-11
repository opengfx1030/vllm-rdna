# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker extension for quiescent expert-copy A/Bs and cache counters.

Load with --worker-extension-cls tools.rdna.mono.expert_cache_worker.ExpertCacheWorker.
The development collective_rpc endpoint can then call rdna_expert_cache_control
with an optional copy_mode string (tensor/native). Call only between workloads.
Captured compute graphs, expert residency and all payload pointers stay intact.
"""

import sys
from typing import Any

import torch


class ExpertCacheWorker:
    """Development-only mixin for the existing worker-extension interface."""

    def rdna_expert_cache_control(
        self: Any,
        copy_mode: str = "",
        finite_checks: str = "",
        post_step_checks: str = "",
    ) -> dict[str, Any]:
        """Inspect successful expert passes, optionally selecting their copy backend."""
        if copy_mode not in ("", "tensor", "native"):
            raise ValueError("copy_mode must be tensor or native")
        if finite_checks not in ("", "on", "off"):
            raise ValueError("finite_checks must be on or off")
        if post_step_checks not in ("", "collective", "wait", "off"):
            raise ValueError("post_step_checks must be collective, wait or off")
        options = getattr(
            sys.modules.get("vllm.models.qwen4_exp.amd.mono.rdna2"), "OPTS", None
        )
        if finite_checks and options is None:
            raise RuntimeError("Mono options are not loaded")
        failstop = sys.modules.get("vllm.models.qwen4_exp.amd.mono.rdna2.failstop")
        runner = getattr(self, "model_runner", None)
        if post_step_checks and (
            runner is None
            or failstop is None
            or not hasattr(failstop, "_USE_TP_COLLECTIVE")
        ):
            raise RuntimeError("worker has no post-step diagnostic controls")
        caches = []
        seen = set()
        for module in self.get_model().modules():
            cache = getattr(module, "_rdna_expert_cache", None)
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                caches.append(cache)
        if not caches:
            raise ValueError("this worker has no offloaded expert caches")
        op = None
        if copy_mode == "native":
            if torch.version.hip is None or any(
                cache.device.type != "cuda" for cache in caches
            ):
                raise ValueError("native expert copies require HIP")
            op = getattr(torch.ops._rocm_C, "rdna2_expert_cache_copy", None)
            if op is None:
                raise RuntimeError(
                    "rebuild the ROCm extension for native expert copies"
                )
        if copy_mode or finite_checks or post_step_checks:
            if any(cache.policy._active is not None for cache in caches):
                raise RuntimeError("expert-copy switching requires released leases")
            torch.accelerator.synchronize()
            if copy_mode:
                for cache in caches:
                    cache._copy_op = op
            if finite_checks:
                # Existing debug mode: excludes this run from performance claims.
                options.timing = finite_checks == "on"
            if post_step_checks:
                runner._mono_mtp_failstop = post_step_checks != "off"
                failstop._USE_TP_COLLECTIVE = post_step_checks != "wait"
        post_mode = None
        if runner is not None and failstop is not None:
            post_mode = (
                "collective"
                if getattr(failstop, "_USE_TP_COLLECTIVE", True)
                else "wait"
            )
            if not runner._mono_mtp_failstop:
                post_mode = "off"
        return {
            "caches": len(caches),
            "debug_timing": None if options is None else bool(options.timing),
            "post_step_checks": post_mode,
            "copy_modes": sorted(
                {
                    "native" if cache._copy_op is not None else "tensor"
                    for cache in caches
                }
            ),
            # Demands are distinct experts per successful bounded pass, not
            # routed token/expert pairs or a token-level cache hit rate.
            "expert_demands": sum(cache.expert_demands for cache in caches),
            "expert_misses": sum(cache.expert_misses for cache in caches),
            "committed_passes": sum(cache.committed_passes for cache in caches),
            "copied_bytes": sum(cache.copied_bytes for cache in caches),
            "native_copy_calls": sum(cache.native_copy_calls for cache in caches),
            "batch_copy_calls": sum(cache.batch_copy_calls for cache in caches),
            "map_uploads": sum(cache.map_uploads for cache in caches),
            "device_bytes": sum(cache.device_bytes for cache in caches),
            "host_bytes": sum(cache.host_bytes for cache in caches),
        }
