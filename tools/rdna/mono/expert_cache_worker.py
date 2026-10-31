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


def _packed_row_geometry(config: Any, page_bytes: Any) -> dict[str, Any]:
    """Describe defined bytes per owner, never combine mutually aliasing owners."""

    def payload_bytes(spec: Any) -> tuple[Any, Any]:
        # Avoid page-size properties: get_dtype_size creates a torch tensor.
        fields = getattr(spec, "__dict__", {})
        if "page_size_bytes" in fields and "real_page_size_bytes" in fields:
            return fields["page_size_bytes"], fields["real_page_size_bytes"]
        shapes, dtypes = fields.get("shapes"), fields.get("dtypes")
        real = None
        if shapes is not None and dtypes is not None:
            if len(shapes) != len(dtypes) or not shapes:
                return None, None
            real = 0
            for shape, dtype in zip(shapes, dtypes):
                size = getattr(dtype, "itemsize", None)
                if type(size) is not int or size <= 0:
                    return None, None
                for dim in shape:
                    if type(dim) is not int or dim < 0:
                        return None, None
                    size *= dim
                real += size
        elif "num_kv_heads" in fields and "head_size" in fields:
            heads = fields.get("num_head_slots")
            if heads is None:
                heads = fields["num_kv_heads"]
            # num_states only performs integer/Fraction arithmetic.
            states = getattr(spec, "num_states", None)
            content = fields.get("state_content_bytes")
            if content is None:
                head = fields["head_size"]
                value_head = fields.get("head_size_v", head)
                itemsize = getattr(fields.get("dtype"), "itemsize", None)
                if (
                    type(value_head) is not int
                    or value_head < 0
                    or any(type(v) is not int or v <= 0 for v in (head, itemsize))
                ):
                    return None, None
                content = (head + value_head) * itemsize
            if any(type(v) is not int or v <= 0 for v in (heads, states, content)):
                return None, None
            real = heads * states * content
        padded = fields.get("page_size_padded")
        return real if padded is None else padded, real

    unavailable = {"available": False, "groups": []}
    if type(page_bytes) is not int or page_bytes <= 0:
        return {**unavailable, "reason": "physical page size unavailable"}
    tensors = getattr(config, "kv_cache_tensors", None)
    groups = getattr(config, "kv_cache_groups", None)
    if not tensors or not groups:
        return {**unavailable, "reason": "cache placement metadata unavailable"}
    placements: dict[str, list[Any]] = {}
    for tensor in tensors:
        for index, name in enumerate(getattr(tensor, "layers", ())):
            placements.setdefault(name, []).append((tensor, index))
    reports = []
    for group_index, group in enumerate(groups):
        report: dict[str, Any] = {"group_index": group_index, "available": False}
        layers = []
        intervals = []
        reason = None
        names = getattr(group, "layer_names", ())
        if not names or getattr(group, "host_resident", False):
            reason = "missing device owner layers"
        for name in names:
            spec = getattr(group, "kv_cache_spec", None)
            specs = getattr(spec, "kv_cache_specs", None)
            if specs is not None:
                spec = specs.get(name)
            matches = placements.get(name, ())
            if len(matches) != 1 or spec is None:
                reason = "missing or ambiguous layer placement/spec"
                break
            tensor, index = matches[0]
            offset = getattr(tensor, "offset", None)
            stride = getattr(tensor, "layer_stride", None)
            block_stride = getattr(tensor, "block_stride", None)
            padded, real = payload_bytes(spec)
            size = getattr(tensor, "size", None)
            values = (offset, stride, block_stride, padded, real, size)
            if any(type(value) is not int for value in values):
                reason = "incomplete byte geometry"
                break
            if (
                getattr(tensor, "host_resident", False)
                or offset < 0
                or not 0 <= stride <= page_bytes
                or block_stride != page_bytes
                or not 0 < real <= padded
                or size < page_bytes
            ):
                reason = "unsupported packed-row geometry"
                break
            start = offset + index * stride
            if start + padded > page_bytes or start + padded > size:
                reason = "layer page outside physical row"
                break
            layers.append(
                {
                    "layer": name,
                    "offset": offset,
                    "layer_index": index,
                    "layer_stride": stride,
                    "block_stride": block_stride,
                    "row_offset": start,
                    "page_size_bytes": padded,
                    "real_page_size_bytes": real,
                }
            )
            intervals.append((start, start + real))
        if reason is not None:
            report["reason"] = reason
        else:
            merged: list[list[int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            defined = sum(end - start for start, end in merged)
            report.update(
                available=True,
                layers=layers,
                defined_content_intervals=merged,
                defined_bytes=defined,
                candidate_omitted_bytes=page_bytes - defined,
            )
        reports.append(report)
    return {
        "available": all(report["available"] for report in reports),
        "physical_page_bytes": page_bytes,
        "groups": reports,
    }


class ExpertCacheWorker:
    """Development-only mixin for the existing worker-extension interface."""

    def rdna_kv_snapshot_layout(self: Any) -> dict[str, Any]:
        """Read layout capabilities only; no transfer or device synchronization."""
        state = sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state")
        connector = getattr(state, "_KV_CONNECTOR_AGENT", None)
        wrapper = getattr(connector, "connector_worker", None)
        worker = getattr(wrapper, "worker", None)
        config = getattr(getattr(self, "model_runner", None), "kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", ())
        spec = getattr(wrapper, "spec", None)
        physical = bool(getattr(worker, "supports_physical_page_snapshots", False))
        page_bytes = (
            getattr(spec, "cpu_page_size_per_worker", None)
            if physical and getattr(spec, "blocks_per_chunk", None) == 1
            else None
        )
        gpu_pages = getattr(worker, "_physical_page_count", 0)
        cpu_chunks = getattr(worker, "_physical_cpu_chunk_count", 0)
        live_configured = bool(getattr(wrapper, "_live_snapshots_enabled", False))
        return {
            "packed_row_geometry": _packed_row_geometry(config, page_bytes),
            "connector": type(connector).__name__,
            "worker": type(worker).__name__,
            "physical_page_transport": physical,
            "physical_page_count": gpu_pages,
            "cpu_chunk_count": cpu_chunks,
            "physical_page_bytes": page_bytes,
            "gpu_physical_bytes": None
            if page_bytes is None
            else gpu_pages * page_bytes,
            "cpu_physical_bytes": None
            if page_bytes is None
            else cpu_chunks * page_bytes,
            "blocks_per_chunk": getattr(spec, "blocks_per_chunk", None),
            "groups": [
                {
                    "spec": type(group.kv_cache_spec).__name__,
                    "layers": list(group.layer_names),
                }
                for group in groups
            ],
            "live_request_paging_configured": live_configured,
            "live_request_paging": live_configured and physical,
        }

    def rdna_expert_cache_control(
        self: Any,
        copy_mode: str = "",
        finite_checks: str = "",
        post_step_checks: str = "",
        transfer_timing: str = "",
        overlap: str = "",
        qzeros_copy: str = "",
    ) -> dict[str, Any]:
        """Inspect successful expert passes, optionally selecting their copy backend."""
        if copy_mode not in ("", "tensor", "native"):
            raise ValueError("copy_mode must be tensor or native")
        if finite_checks not in ("", "on", "off"):
            raise ValueError("finite_checks must be on or off")
        if post_step_checks not in ("", "collective", "wait", "off"):
            raise ValueError("post_step_checks must be collective, wait or off")
        if transfer_timing not in ("", "on", "off"):
            raise ValueError("transfer_timing must be on or off")
        if overlap not in ("", "on", "off"):
            raise ValueError("overlap must be on or off")
        if qzeros_copy not in ("", "on", "off"):
            raise ValueError("qzeros_copy must be on or off")
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
        names = {}
        model = self.get_model()
        modules = (
            model.named_modules()
            if hasattr(model, "named_modules")
            else ((str(index), module) for index, module in enumerate(model.modules()))
        )
        for name, module in modules:
            cache = getattr(module, "_rdna_expert_cache", None)
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                caches.append(cache)
                names[id(cache)] = name
        if not caches:
            raise ValueError("this worker has no offloaded expert caches")
        if qzeros_copy == "off" and any(not cache._static_qzeros for cache in caches):
            raise ValueError("qzero copy elision requires verified static qzeros")
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
        if (
            copy_mode
            or finite_checks
            or post_step_checks
            or transfer_timing
            or overlap
            or qzeros_copy
        ):
            if any(
                cache.policy._active is not None or cache._preload is not None
                for cache in caches
            ):
                raise RuntimeError("expert-copy switching requires released leases")
            torch.accelerator.synchronize()
            if copy_mode:
                for cache in caches:
                    cache._copy_op = op
            if transfer_timing:
                for cache in caches:
                    cache.set_transfer_timing(transfer_timing == "on")
            if overlap:
                for cache in caches:
                    cache.overlap = overlap == "on"
            if qzeros_copy:
                for cache in caches:
                    cache.set_qzeros_copy(qzeros_copy == "on")
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
        rank = getattr(
            self, "rank", getattr(getattr(self, "model_runner", None), "rank", None)
        )
        layers = [
            {
                "module": names[id(cache)],
                "rank": rank,
                "expert_bytes": cache._expert_bytes,
                "copy_expert_bytes": cache._copy_expert_bytes,
                "qzeros_copy": cache._copy_qzeros,
                "map_bytes": cache.slot_map.numel() * cache.slot_map.element_size(),
                "slots": cache.slot_count,
                "overlap": cache.overlap,
                "prefetch_bytes": cache.prefetch_bytes,
                "adopted_prefetch_bytes": cache.adopted_prefetch_bytes,
                "unused_prefetch_bytes": cache.unused_prefetch_bytes,
                "prefetch_failures": cache.prefetch_failures,
                "expert_demands": cache.expert_demands,
                "expert_misses": cache.expert_misses,
                "committed_passes": cache.committed_passes,
                "copied_bytes": cache.copied_bytes,
                "map_uploads": cache.map_uploads,
                "native_copy_calls": cache.native_copy_calls,
                "batch_copy_calls": cache.batch_copy_calls,
                "transfer_timing": cache.transfer_timing(),
            }
            for cache in caches
        ]
        return {
            "layers": layers,
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
