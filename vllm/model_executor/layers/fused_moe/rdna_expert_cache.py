# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed host expert storage and event-fenced RDNA2 device slots."""

import os
from contextlib import contextmanager

import torch

from vllm.model_executor.layers.fused_moe.rdna_expert_cache_policy import (
    CacheLease,
    ExpertCachePolicy,
)

_TRACE_ROUTES = bool(os.environ.get("VLLM_RDNA_EXPERT_ROUTE_DIR"))

PAYLOAD_NAMES = (
    "w13_weight_packed",
    "w13_weight_scale",
    "w13_qzeros",
    "w2_weight_packed",
    "w2_weight_scale",
    "w2_qzeros",
)


def observe_routes(cache, ids):
    """Observe once; optional receipts reuse the existing CPU routing result."""
    cache.policy.observe(ids)
    if _TRACE_ROUTES:
        from vllm.model_executor.layers.fused_moe.rdna_expert_route_trace import (
            record_cache_routes,
        )

        record_cache_routes(cache, cache.layer_name, ids, cache.rank)


def is_capturing_expert_work() -> bool:
    """Distinguish raw capture from a breakable graph's eager/replay segments."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    active = BreakableCUDAGraphCapture.current()
    if active is not None:
        return active.capturing_segment
    return torch.cuda.is_current_stream_capturing()


@contextmanager
def cpu_weight_context(quant_method, layer):
    """Stage only opted-in, symmetric native CT experts on the host."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    extra = config.additional_config if config is not None else {}
    options = extra.get("rocm_mono_decode", {}) if isinstance(extra, dict) else {}
    slots = options.get("expert_cache_slots", 0)
    if not slots or type(quant_method).__name__ != (
        "CompressedTensorsWNA16RDNA2MoEMethod"
    ):
        yield
        return
    assert config is not None
    if not config.kernel_config.monokernel:
        raise ValueError("expert_cache_slots requires --monokernel")
    if not any(
        name.startswith("Qwen4Exp") for name in config.model_config.architectures
    ):
        raise ValueError("expert cache currently supports Qwen4Exp only")
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx10x

    if not current_platform.is_rocm() or not on_gfx10x():
        raise ValueError("expert_cache_slots requires native RDNA2 CT experts")
    if not quant_method.symmetric or quant_method.num_bits != 4:
        raise ValueError("expert cache requires symmetric INT4 experts")
    if quant_method.moe.in_dtype != torch.float16:
        raise ValueError("expert cache requires FP16 activations")
    if quant_method.moe.has_bias or quant_method.moe.is_lora_enabled:
        raise ValueError("expert cache does not support expert bias or LoRA")
    if quant_method.moe.moe_parallel_config.enable_eplb:
        raise ValueError("expert cache does not support EPLB")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise ValueError("expert_cache_slots must be a positive integer")
    if options.get("expert_cache_slot_plan"):
        from vllm.model_executor.layers.fused_moe.rdna_expert_cache_config import (
            select_slots,
        )

        slots = select_slots(
            slots,
            options.get("expert_cache_slot_plan", {}),
            quant_method.moe.moe_parallel_config.ep_rank,
            layer.layer_name,
        )
    device = torch.get_default_device()
    if device.type != "cuda":
        raise ValueError("expert cache requires a CUDA/HIP loading device")
    layer._rdna_expert_cache_slots = slots
    layer._rdna_expert_cache_device = device
    layer._rdna_expert_cache_policy = options.get("expert_cache_policy", "lfu_lru")
    layer._rdna_expert_cache_decay = options.get("expert_cache_decay", 128)
    layer._rdna_expert_cache_copy = options.get("expert_cache_copy", "tensor")
    layer._rdna_expert_cache_overlap = options.get("expert_cache_overlap", False)
    quant_method.requires_device_loading = False
    with torch.device("cpu"):
        yield


def prepare_host_payload(layer, group_size, device, shuffle):
    """Pack a CPU-loaded CT layer with at most one expert temporarily on HIP."""
    device = torch.device(device)
    pin = device.type == "cuda"
    host = {}
    for name in ("w13_weight_packed", "w2_weight_packed"):
        source = getattr(layer, name).detach()
        if source.device.type != "cpu":
            raise RuntimeError("offloaded experts must be loaded on CPU")
        shape = (source.shape[0], source.shape[2], source.shape[1])
        packed = torch.empty(shape, dtype=source.dtype, device="cpu", pin_memory=pin)
        for expert in range(source.shape[0]):
            row = source[expert].t().contiguous().to(device, copy=True)
            shuffle(row, 4)
            packed[expert].copy_(row)
            del row
        host[name] = packed
    for name in ("w13_weight_scale", "w2_weight_scale"):
        source = getattr(layer, name).detach()
        packed = torch.empty(
            (source.shape[0], source.shape[2], source.shape[1]),
            dtype=source.dtype,
            device="cpu",
            pin_memory=pin,
        )
        packed.copy_(source.transpose(1, 2))
        host[name] = packed
    for stem in ("w13", "w2"):
        weights = host[f"{stem}_weight_packed"]
        groups = weights.shape[1] * 8 // group_size
        host[f"{stem}_qzeros"] = torch.full(
            (weights.shape[0], groups, weights.shape[2] // 8),
            0x77777777,  # symmetric uint4b8 zero=8, GPTQv1 stores zero-1
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin,
        )
    return host


class PackedExpertCache:
    """Immutable packed CPU bank with a bounded, stable device allocation."""

    def __init__(
        self,
        host_payload,
        slot_count,
        device,
        policy="lfu_lru",
        decay_interval=128,
        copy_mode="tensor",
        static_qzeros=False,
    ):
        if set(host_payload) != set(PAYLOAD_NAMES):
            raise ValueError("expert cache requires all six packed payload tensors")
        self.device = torch.device(device)
        self.layer_name: str = ""
        self.rank: int = -1
        if copy_mode not in ("tensor", "native"):
            raise ValueError("expert_cache_copy must be tensor or native")
        self._copy_op = None
        if copy_mode == "native":
            if self.device.type != "cuda" or torch.version.hip is None:
                raise ValueError("native expert copies require a HIP device")
            self._copy_op = getattr(torch.ops._rocm_C, "rdna2_expert_cache_copy", None)
            if self._copy_op is None:
                raise RuntimeError(
                    "native expert copies require a rebuilt ROCm extension"
                )
        self.logical_num_experts = host_payload[PAYLOAD_NAMES[0]].shape[0]
        self.slot_count = min(slot_count, self.logical_num_experts)
        self.policy = ExpertCachePolicy(
            self.logical_num_experts, self.slot_count, policy, decay_interval
        )
        self.host_payload = {}
        self.payload = {}
        self._static_qzeros = static_qzeros
        for name in PAYLOAD_NAMES:
            source = host_payload[name].detach()
            if source.device.type != "cpu" or not source.is_contiguous():
                raise ValueError("expert host payload must be contiguous CPU storage")
            if source.shape[0] != self.logical_num_experts:
                raise ValueError("expert payload dimensions disagree")
            if (
                static_qzeros
                and name.endswith("_qzeros")
                and (
                    source.dtype != torch.int32
                    or not source.numel()
                    or not bool(torch.all(source == 0x77777777))
                )
            ):
                raise ValueError("static expert qzeros require packed symmetric zero=8")
            if self.device.type == "cuda" and not source.is_pinned():
                source = source.pin_memory()
            self.host_payload[name] = source
            self.payload[name] = torch.empty(
                (self.slot_count, *source.shape[1:]),
                dtype=source.dtype,
                device=self.device,
            )
            if static_qzeros and name.endswith("_qzeros"):
                self.payload[name].fill_(0x77777777)
        self.slot_map = torch.full(
            (self.logical_num_experts,), -1, dtype=torch.int32, device=self.device
        )
        self._host_slot_map = torch.full(
            (self.logical_num_experts,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        self._host_slot_map_array = self._host_slot_map.numpy()
        self._copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._consumer_event = None
        self._consumer_stream = None
        self._active_bindings = None
        self._map_upload_event = None
        self.map_uploads = 0
        self.copied_bytes = 0
        self.native_copy_calls = 0
        self.batch_copy_calls = 0
        self.expert_demands = 0
        self.expert_misses = 0
        self.committed_passes = 0
        self.overlap = False
        self._preload = None
        self.prefetch_bytes = 0
        self.adopted_prefetch_bytes = 0
        self.unused_prefetch_bytes = 0
        self.prefetch_failures = 0
        self._transfer_timing = None
        self._last_transfer_timing = None
        self._expert_bytes = sum(
            tensor[0].numel() * tensor.element_size()
            for tensor in self.host_payload.values()
        )
        self.set_qzeros_copy(not static_qzeros)
        if static_qzeros and self._copy_stream is not None:
            # The immutable zeros must be ready even when the first consumer
            # runs on a different stream from weight loading.
            self._consumer_stream = torch.cuda.current_stream(self.device)
            self._consumer_event = torch.cuda.Event()
            self._consumer_event.record(self._consumer_stream)
            self._copy_stream.wait_event(self._consumer_event)

    def set_qzeros_copy(self, enabled):
        """Quiescent diagnostic switch; immutable zero slots retain their storage."""
        if self.policy._active is not None or self._preload is not None:
            raise RuntimeError("expert-copy switching requires released leases")
        if not enabled and not self._static_qzeros:
            raise ValueError("qzero copy elision requires verified static qzeros")
        self._copy_qzeros = enabled
        self._copy_payload_names = tuple(
            name for name in PAYLOAD_NAMES if enabled or not name.endswith("_qzeros")
        )
        self._copy_sources = tuple(
            self.host_payload[name] for name in self._copy_payload_names
        )
        self._copy_destinations = tuple(
            self.payload[name] for name in self._copy_payload_names
        )
        self._copy_expert_bytes = sum(
            tensor[0].numel() * tensor.element_size() for tensor in self._copy_sources
        )

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.host_payload.values()) + (
            self._host_slot_map.numel() * self._host_slot_map.element_size()
        )

    @property
    def device_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.payload.values()) + (
            self.slot_map.numel() * self.slot_map.element_size()
        )

    def local_ids(self, ids, expert_map=None):
        """Convert global IDs once, preserving invalid and nonlocal padding."""
        if expert_map is None:
            valid = (ids >= 0) & (ids < self.logical_num_experts)
            return torch.where(valid, ids, -1)
        valid = (ids >= 0) & (ids < expert_map.numel())
        safe = ids.clamp(0, expert_map.numel() - 1).long()
        local = expert_map[safe].to(ids.dtype)
        valid &= (local >= 0) & (local < self.logical_num_experts)
        return torch.where(valid, local, -1)

    def set_transfer_timing(self, enabled):
        """Call only after worker quiescence; diagnostics never change residency."""
        if enabled:
            from time import perf_counter_ns

            self._transfer_timing = {
                "clock": perf_counter_ns,
                "pending": [],
                "limit": 64,
                "samples": 0,
                "dropped": 0,
                "cpu_passes": 0,
                "cpu_planning_us": 0.0,
                "cpu_map_source_wait_us": 0.0,
                "cpu_copy_enqueue_us": 0.0,
                "gpu_copy_dependency_wait_ms": 0.0,
                "gpu_copy_interval_ms": 0.0,
                "gpu_exposed_copy_wait_ms": 0.0,
            }
        else:
            if self._transfer_timing is not None:
                self._last_transfer_timing = self.transfer_timing()
                self._last_transfer_timing["enabled"] = False
            self._transfer_timing = None

    def transfer_timing(self):
        """Poll completed records; never synchronize for telemetry.

        GPU intervals include enqueue gaps and device scheduling. Dependency
        wait is on the copy stream; exposed wait is on the consumer stream.
        CPU planning excludes copy enqueue and the existing map-source wait.
        These diagnostic totals are not isolated DMA-engine service times.
        """
        state = self._transfer_timing
        if state is None:
            return self._last_transfer_timing or {"enabled": False}
        pending = state["pending"]
        while pending and pending[0][-1].query():
            dep, begin, end, exposed, ready = pending.pop(0)
            state["gpu_copy_dependency_wait_ms"] += dep.elapsed_time(begin)
            state["gpu_copy_interval_ms"] += begin.elapsed_time(end)
            state["gpu_exposed_copy_wait_ms"] += exposed.elapsed_time(ready)
            state["samples"] += 1
        return {
            "enabled": True,
            "pending": len(pending),
            **{
                key: value
                for key, value in state.items()
                if key not in ("clock", "pending", "limit")
            },
        }

    def preload_known(self, experts, protected_experts=()):
        """Copy known misses into disjoint slots without publishing a slot map."""
        if self.device.type == "cuda" and is_capturing_expert_work():
            raise RuntimeError("host expert planning must run outside graph capture")
        if self._preload is not None:
            raise RuntimeError("adopt or cancel the pending preload first")
        reservation = self.policy.reserve(experts, protected_experts)
        if reservation is None:
            return None
        copied = None
        events = None
        timing = self._transfer_timing
        try:
            if self._copy_stream is not None:
                if timing is not None:
                    self.transfer_timing()
                    if len(timing["pending"]) < timing["limit"]:
                        events = tuple(
                            torch.cuda.Event(enable_timing=True) for _ in range(5)
                        )
                    else:
                        timing["dropped"] += 1
                with torch.cuda.stream(self._copy_stream):
                    if events is not None:
                        events[0].record(self._copy_stream)
                    if self._consumer_event is not None:
                        self._copy_stream.wait_event(self._consumer_event)
                    if events is not None:
                        events[1].record(self._copy_stream)
                    if timing is not None:
                        started = timing["clock"]()
                    self._copy(reservation.bindings)
                    if timing is not None:
                        timing["cpu_copy_enqueue_us"] += (
                            timing["clock"]() - started
                        ) / 1000
                    copied = events[2] if events is not None else torch.cuda.Event()
                    copied.record(self._copy_stream)
            else:
                self._copy(reservation.bindings)
            # Count successfully submitted preloads. Failed/partial API calls
            # have unknown transferred bytes and are tracked as failures only.
            self.prefetch_bytes += len(reservation.bindings) * self._copy_expert_bytes
            self._preload = (reservation, copied, events, timing)
            return reservation
        except BaseException:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            reservation.cancel()
            self.prefetch_failures += 1
            raise

    def ensure_reserved(self, reservation) -> CacheLease:
        """Fence a completed/submitted preload and publish it without recopying."""
        if self._preload is None or self._preload[0] is not reservation:
            raise RuntimeError("expected this cache's pending preload")
        return self.ensure((), _reservation=reservation)

    def cancel_preload(self, reservation):
        """Drain pending copies before abandoning their invalidated slots."""
        if self._preload is None or self._preload[0] is not reservation:
            raise RuntimeError("expected this cache's pending preload")
        _, copied, events, timing = self._preload
        if copied is not None:
            copied.synchronize()
        if events is not None and timing is not None:
            timing["gpu_copy_dependency_wait_ms"] += events[0].elapsed_time(events[1])
            timing["gpu_copy_interval_ms"] += events[1].elapsed_time(events[2])
            timing["samples"] += 1
        reservation.cancel()
        self.unused_prefetch_bytes += (
            len(reservation.bindings) * self._copy_expert_bytes
        )
        self._preload = None

    @contextmanager
    def passes(self, ids, overlap=False):
        """Yield leased passes; closing the context releases/drains all work."""

        def iterate():
            if self._preload is not None:
                raise RuntimeError("adopt or cancel the pending preload first")
            try:
                planned = self.policy.plan_passes(ids)
                if overlap:
                    ordered = [expert for batch in planned for expert in batch]
                    mapping = self.policy.mapping
                    resident = [expert for expert in ordered if expert in mapping]
                    missing = [expert for expert in ordered if expert not in mapping]
                if overlap and resident and missing:
                    lease = self.ensure(resident)
                    try:
                        reservation = self.preload_known(
                            missing, protected_experts=resident
                        )
                        yield lease
                    finally:
                        self.release(lease)
                    if reservation is not None:
                        adopted = {expert for expert, _, _ in reservation.bindings}
                        lease = self.ensure_reserved(reservation)
                        try:
                            yield lease
                        finally:
                            self.release(lease)
                        missing = [
                            expert for expert in missing if expert not in adopted
                        ]
                    planned = self.policy.plan_passes(missing)
                for experts in planned:
                    lease = self.ensure(experts)
                    try:
                        yield lease
                    finally:
                        self.release(lease)
            finally:
                if self._preload is not None:
                    self.cancel_preload(self._preload[0])

        iterator = iterate()
        try:
            yield iterator
        finally:
            iterator.close()

    def ensure(self, experts, *, _reservation=None) -> CacheLease:
        """Acquire a bounded pass and establish copy completion before use."""
        if self.device.type == "cuda" and is_capturing_expert_work():
            raise RuntimeError("host expert planning must run outside graph capture")
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            if self._consumer_event is not None and current != self._consumer_stream:
                current.wait_event(self._consumer_event)
        timing = self._transfer_timing
        if timing is not None:
            self.transfer_timing()
            clock = timing["clock"]
            planning_start = clock()
        preload = self._preload if _reservation is not None else None
        lease = (
            self.policy.adopt(_reservation)
            if _reservation is not None
            else self.policy.acquire(experts)
        )
        if preload is not None:
            self._preload = None
        try:
            misses = [binding for binding in lease.bindings if binding[2]]
            if timing is not None:
                timing["cpu_planning_us"] += (clock() - planning_start) / 1000
                timing["cpu_passes"] += 1
            events = None
            if preload is not None:
                _, copied, preload_events, preload_timing = preload
                if copied is not None:
                    consumer = torch.cuda.current_stream(self.device)
                    if preload_events is not None:
                        preload_events[3].record(consumer)
                    consumer.wait_event(copied)
                    if preload_events is not None:
                        preload_events[4].record(consumer)
                        preload_timing["pending"].append(preload_events)
            if (
                preload is None
                and misses
                and self._copy_stream is not None
                and timing is not None
            ):
                if len(timing["pending"]) < timing["limit"]:
                    events = tuple(
                        torch.cuda.Event(enable_timing=True) for _ in range(5)
                    )
                else:
                    timing["dropped"] += 1
            if preload is None and misses and self._copy_stream is not None:
                with torch.cuda.stream(self._copy_stream):
                    if events is not None:
                        events[0].record(self._copy_stream)
                    if self._consumer_event is not None:
                        self._copy_stream.wait_event(self._consumer_event)
                    if events is not None:
                        events[1].record(self._copy_stream)
                    if timing is not None:
                        enqueue_start = clock()
                    self._copy(misses)
                    if timing is not None:
                        timing["cpu_copy_enqueue_us"] += (
                            clock() - enqueue_start
                        ) / 1000
                    copied = events[2] if events is not None else torch.cuda.Event()
                    copied.record(self._copy_stream)
                consumer = torch.cuda.current_stream(self.device)
                if events is not None:
                    events[3].record(consumer)
                consumer.wait_event(copied)
                if events is not None:
                    events[4].record(consumer)
                    assert timing is not None
                    timing["pending"].append(events)
            elif preload is None:
                self._copy(misses)
            if timing is not None:
                planning_start = clock()
            active = tuple(sorted((expert, slot) for expert, slot, _ in lease.bindings))
            if timing is not None:
                timing["cpu_planning_us"] += (clock() - planning_start) / 1000
            if active != self._active_bindings:
                # The pinned source must not change before its previous DMA
                # completes, even when the consumer switches streams.
                if self._map_upload_event is not None:
                    if timing is not None:
                        wait_start = clock()
                    self._map_upload_event.synchronize()
                    if timing is not None:
                        timing["cpu_map_source_wait_us"] += (
                            clock() - wait_start
                        ) / 1000
                if timing is not None:
                    planning_start = clock()
                self._host_slot_map_array.fill(-1)
                for expert, slot in active:
                    self._host_slot_map_array[expert] = slot
                if timing is not None:
                    timing["cpu_planning_us"] += (clock() - planning_start) / 1000
                self.slot_map.copy_(self._host_slot_map, non_blocking=True)
                if self.device.type == "cuda":
                    self._map_upload_event = torch.cuda.Event()
                    self._map_upload_event.record(
                        torch.cuda.current_stream(self.device)
                    )
                self.map_uploads += 1
                self._active_bindings = active
            lease.commit()
            self.expert_demands += len(lease.bindings)
            self.expert_misses += len(misses)
            self.committed_passes += 1
            if preload is not None:
                self.adopted_prefetch_bytes += len(misses) * self._copy_expert_bytes
            return lease
        except BaseException:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            lease.cancel()
            if preload is not None:
                self.unused_prefetch_bytes += (
                    len(lease.bindings) * self._copy_expert_bytes
                )
            raise

    def _copy(self, bindings):
        if not bindings:
            return
        if self._copy_op is not None:
            used_batch = self._copy_op(
                self._copy_sources,
                self._copy_destinations,
                [expert for expert, _, _ in bindings],
                [slot for _, slot, _ in bindings],
            )
            self.native_copy_calls += 1
            self.batch_copy_calls += int(used_batch)
            self.copied_bytes += len(bindings) * self._copy_expert_bytes
            return
        for expert, slot, _ in bindings:
            for name in self._copy_payload_names:
                self.payload[name][slot].copy_(
                    self.host_payload[name][expert], non_blocking=True
                )
                source = self.host_payload[name][expert]
                self.copied_bytes += source.numel() * source.element_size()

    def remap(self, local_ids, lease):
        """Mask other passes and map this pass's local expert IDs to slots."""
        safe = local_ids.clamp(0, self.logical_num_experts - 1).long()
        return torch.where(local_ids >= 0, self.slot_map[safe], -1)

    def release(self, lease):
        """Fence the last consumer before any future slot overwrite."""
        if self.device.type == "cuda":
            self._consumer_event = torch.cuda.Event()
            self._consumer_stream = torch.cuda.current_stream(self.device)
            self._consumer_event.record(self._consumer_stream)
        lease.release()
