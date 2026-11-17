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

_TRACE_BYTES_LIMIT = 16 * 1024**2


class ExpertCacheWorker:
    """Development-only mixin for the existing worker-extension interface."""

    def rdna_cache_memory(self: Any) -> dict[str, Any]:
        """Read physical V2 KV storage sizes without synchronizing the device."""
        caches = getattr(self.model_runner, "kv_caches", None)
        if not caches:
            raise RuntimeError("V2 KV caches have not been initialized")
        storages = {}
        devices = set()
        for cache in caches:
            if not isinstance(cache, torch.Tensor):
                raise TypeError("expected V2 KV cache tensor views")
            storage = cache.untyped_storage()
            key = (str(cache.device), storage.data_ptr())
            storages[key] = storage.nbytes()
            devices.add(str(cache.device))
        return {
            "kv_cache_storage_bytes": sum(storages.values()),
            "kv_cache_storages": len(storages),
            "kv_cache_views": len(caches),
            "kv_cache_devices": sorted(devices),
        }

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
            if any(getattr(cache, "constant_qzeros", False) for cache in caches):
                raise ValueError("constant expert qzeros require tensor copies")
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

    def rdna_expert_cache_trace(
        self: Any, action: str, max_calls: int = 1024, layers: list[int] | None = None
    ) -> dict[str, Any]:
        """Record CPU-visible demands between quiescent diagnostic RPCs.

        This intentionally adds Python work: never use a traced window for a
        performance claim. All rows, including prefills and padding, are retained
        so an offline replay can start from the exported policy state.
        """
        from array import array

        import regex as re

        if action not in ("start", "stop"):
            raise ValueError("trace action must be start or stop")
        active = getattr(self, "_rdna_cache_trace", None)
        if action == "stop":
            if active is None:
                raise ValueError("expert-cache trace is not running")
            if any(cache.policy._active is not None for cache, _, _ in active):
                raise RuntimeError("trace controls require released expert leases")
            self._rdna_cache_trace = None
            result = []
            for cache, original, record in active:
                cache.policy.observe = original
                record["final_misses"] = cache.expert_misses
                record["final_demands"] = cache.expert_demands
                record.setdefault("captured_end_misses", cache.expert_misses)
                record.setdefault("captured_end_demands", cache.expert_demands)
                record["ids"] = [list(ids) for ids in record["ids"]]
                result.append(record)
            budget = self._rdna_cache_trace_budget
            return {
                "layers": result,
                "raw_bytes": budget["bytes"],
                "byte_limit_reached": budget["closed"],
            }
        if active is not None:
            raise ValueError("expert-cache trace is already running")
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or not (1 <= max_calls <= 4096)
        ):
            raise ValueError("max_calls must be an integer in [1, 4096]")
        selected = set(layers) if layers is not None else None
        targets = []
        seen = set()
        for name, module in self.get_model().named_modules():
            cache = getattr(module, "_rdna_expert_cache", None)
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if cache is None or id(cache) in seen or match is None:
                continue
            layer = int(match.group(1))
            if selected is not None and layer not in selected:
                continue
            seen.add(id(cache))
            if cache.policy._active is not None:
                raise RuntimeError("trace controls require released expert leases")
            targets.append((name, layer, cache))
        if not targets:
            raise ValueError("no matching expert caches")
        active = []
        budget = {"bytes": 0, "closed": False}
        for name, layer, cache in targets:
            policy = cache.policy
            record = {
                "name": name,
                "layer": layer,
                "local_experts": policy.num_experts,
                "capacity": policy.capacity,
                "policy": policy.policy,
                "decay_interval": policy.decay_interval,
                "bytes_per_miss": sum(
                    cache.host_payload[name][0].numel()
                    * cache.host_payload[name].element_size()
                    for name in cache._copied_names
                ),
                "initial_epoch": policy.epoch,
                "initial_mapping": policy.mapping,
                "initial_frequencies": list(policy.frequencies),
                "initial_last_used": list(policy.last_used),
                "initial_misses": cache.expert_misses,
                "initial_demands": cache.expert_demands,
                "seen_calls": 0,
                "ids": [],
                "misses_before": [],
                "demands_before": [],
            }
            original = policy.observe

            def observe(ids, *, _original=original, _record=record, _cache=cache):
                _original(ids)
                _record["seen_calls"] += 1
                size = 2 * len(ids)
                eligible = len(_record["ids"]) < max_calls
                if eligible and budget["bytes"] + size > _TRACE_BYTES_LIMIT:
                    budget["closed"] = True
                if eligible and not budget["closed"]:
                    _record["ids"].append(array("h", ids))
                    budget["bytes"] += size
                    _record["misses_before"].append(_cache.expert_misses)
                    _record["demands_before"].append(_cache.expert_demands)
                else:
                    _record.setdefault("captured_end_misses", _cache.expert_misses)
                    _record.setdefault("captured_end_demands", _cache.expert_demands)

            policy.observe = observe
            active.append((cache, original, record))
        self._rdna_cache_trace = active
        self._rdna_cache_trace_budget = budget
        return {"caches": len(active), "max_calls": max_calls}
