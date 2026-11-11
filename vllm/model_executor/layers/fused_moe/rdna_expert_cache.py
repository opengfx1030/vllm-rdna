# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed host expert storage and event-fenced RDNA2 device slots."""

from contextlib import contextmanager

import torch

from vllm.model_executor.layers.fused_moe.rdna_expert_cache_policy import (
    CacheLease,
    ExpertCachePolicy,
)

PAYLOAD_NAMES = (
    "w13_weight_packed",
    "w13_weight_scale",
    "w13_qzeros",
    "w2_weight_packed",
    "w2_weight_scale",
    "w2_qzeros",
)


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
    device = torch.get_default_device()
    if device.type != "cuda":
        raise ValueError("expert cache requires a CUDA/HIP loading device")
    layer._rdna_expert_cache_slots = slots
    layer._rdna_expert_cache_device = device
    layer._rdna_expert_cache_policy = options.get("expert_cache_policy", "lfu_lru")
    layer._rdna_expert_cache_decay = options.get("expert_cache_decay", 128)
    layer._rdna_expert_cache_copy = options.get("expert_cache_copy", "tensor")
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
    ):
        if set(host_payload) != set(PAYLOAD_NAMES):
            raise ValueError("expert cache requires all six packed payload tensors")
        self.device = torch.device(device)
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
        # A cache that cannot hold every expert keeps one more, all-zero slot
        # past its own: a pass that maps an expert onto it still counts that
        # expert in its demand, yet adds exactly zero. The offload kernel sizes
        # its K slices by the number of experts a launch runs, so a demand split
        # over launches that each count all of it rounds like one full pass.
        self.dummy_slot = (
            self.slot_count if self.slot_count < self.logical_num_experts else None
        )
        self._absent = -1 if self.dummy_slot is None else self.dummy_slot
        self.policy = ExpertCachePolicy(
            self.logical_num_experts, self.slot_count, policy, decay_interval
        )
        self.host_payload = {}
        self.payload = {}
        self.k2_payload = {}
        rows = self.slot_count + (self.dummy_slot is not None)
        for name in PAYLOAD_NAMES:
            source = host_payload[name].detach()
            if source.device.type != "cpu" or not source.is_contiguous():
                raise ValueError("expert host payload must be contiguous CPU storage")
            if source.shape[0] != self.logical_num_experts:
                raise ValueError("expert payload dimensions disagree")
            if self.device.type == "cuda" and not source.is_pinned():
                source = source.pin_memory()
            self.host_payload[name] = source
            storage = torch.empty(
                (rows, *source.shape[1:]), dtype=source.dtype, device=self.device
            )
            if self.dummy_slot is not None:
                storage[self.dummy_slot].zero_()
            # The offload kernel takes the dummy slot with the others; the
            # modular path and the loads only ever see the real slots.
            self.k2_payload[name] = storage
            self.payload[name] = storage[: self.slot_count]
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
        self._consumer_event = None
        self._consumer_stream = None
        self._active_bindings = None
        self._map_upload_event = None
        # What the offload layer's expert passes run on. Its first pass starts
        # before the host has seen the routing, so it takes resident_map, the
        # residency of every expert (absent ones on the dummy slot): it runs
        # what is resident. The pass that follows the loads takes miss_map, the
        # loaded experts alone (the rest on the dummy slot). Both are published
        # together; slot_map stays the mask of a pass over exactly its experts.
        self._maps = torch.full(
            (2, self.logical_num_experts),
            self._absent,
            dtype=torch.int32,
            device=self.device,
        )
        self.resident_map, self.miss_map = self._maps.unbind(0)
        self._host_maps = torch.full(
            (2, self.logical_num_experts),
            self._absent,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        self._host_resident_array, self._host_miss_array = self._host_maps.numpy()
        self._slot_owner = [-1] * self.slot_count
        self._track = False
        self._maps_upload_event = None
        self.resident_uploads = 0
        self.map_uploads = 0
        self.copied_bytes = 0
        self.native_copy_calls = 0
        self.batch_copy_calls = 0
        self.expert_demands = 0
        self.expert_misses = 0
        self.committed_passes = 0
        self._copy_sources = tuple(self.host_payload[name] for name in PAYLOAD_NAMES)
        self._copy_destinations = tuple(self.payload[name] for name in PAYLOAD_NAMES)
        self._expert_bytes = sum(
            tensor[0].numel() * tensor.element_size() for tensor in self._copy_sources
        )

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.host_payload.values()) + (
            self._host_slot_map.numel() * self._host_slot_map.element_size()
        )

    @property
    def device_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k2_payload.values()) + (
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

    def ensure(self, experts, by_residency=False) -> CacheLease:
        """Acquire a bounded pass and establish copy completion before use.

        By default the pass runs on ``slot_map``, uploaded to mask in exactly
        this pass's experts. A pass that runs on ``resident_map`` and
        ``miss_map`` (see ``start_tracking``) reads the residency published
        here and needs no pass mask: with ``by_residency`` only the copies and
        that publication are issued.
        """
        if by_residency and not self._track:
            raise RuntimeError("a pass on the residency maps needs start_tracking()")
        cuda = self.device.type == "cuda"
        if cuda:
            if is_capturing_expert_work():
                raise RuntimeError(
                    "host expert planning must run outside graph capture"
                )
            current = torch.cuda.current_stream(self.device)
            if self._consumer_event is not None and current != self._consumer_stream:
                current.wait_event(self._consumer_event)
        lease = self.policy.acquire(experts)
        misses = [binding for binding in lease.bindings if binding[2]]
        try:
            # The copies go on the stream that runs the pass: stream order makes
            # them wait for the previous consumer on that stream (consumers on
            # other streams were fenced above) and the pass wait for the copies,
            # with no copy stream and no per-pass events.
            self._copy(misses)
            if not by_residency:
                active = tuple(
                    sorted((expert, slot) for expert, slot, _ in lease.bindings)
                )
                if active != self._active_bindings:
                    # The pinned source must not change before its previous DMA
                    # completes, even when the consumer switches streams.
                    if self._map_upload_event is not None:
                        self._map_upload_event.synchronize()
                    self._host_slot_map_array.fill(-1)
                    for expert, slot in active:
                        self._host_slot_map_array[expert] = slot
                    self.slot_map.copy_(self._host_slot_map, non_blocking=True)
                    if cuda:
                        if self._map_upload_event is None:
                            self._map_upload_event = torch.cuda.Event()
                        self._map_upload_event.record(current)
                    self.map_uploads += 1
                    self._active_bindings = active
            if misses and self._track:
                self._publish(misses)
            lease.commit()
            self.expert_demands += len(lease.bindings)
            self.expert_misses += len(misses)
            self.committed_passes += 1
            return lease
        except Exception:
            if cuda:
                torch.cuda.current_stream(self.device).synchronize()
            lease.cancel()
            if self._track:
                self._forget_overwritten(misses)
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
            self.copied_bytes += len(bindings) * self._expert_bytes
            return
        for expert, slot, _ in bindings:
            for name in PAYLOAD_NAMES:
                source = self.host_payload[name][expert]
                self.payload[name][slot].copy_(source, non_blocking=True)
                self.copied_bytes += source.numel() * source.element_size()

    def start_tracking(self):
        """Begin mirroring residency into ``resident_map`` (idempotent).

        Only the offload layer's passes read the maps, so the per-load upload
        is not paid by other users of the cache.
        """
        if self._track:
            return
        if self._maps_upload_event is not None:
            self._maps_upload_event.synchronize()
        resident, loaded = self._host_resident_array, self._host_miss_array
        resident.fill(self._absent)
        loaded.fill(self._absent)
        self._slot_owner = [-1] * self.slot_count
        for expert, slot in self.policy.mapping.items():
            resident[expert] = slot
            self._slot_owner[slot] = expert
        self._upload_maps()
        self._track = True

    def _publish(self, misses):
        """Record the experts a pass loads in ``resident_map`` and ``miss_map``.

        The pass's copies are already on the stream, so the upload lands after
        them and after every kernel that read the previous maps.
        """
        if self._maps_upload_event is not None:
            self._maps_upload_event.synchronize()
        resident, loaded = self._host_resident_array, self._host_miss_array
        absent = self._absent
        loaded.fill(absent)
        owner = self._slot_owner
        for expert, slot, _ in misses:
            victim = owner[slot]
            if victim >= 0:
                resident[victim] = absent
            owner[slot] = expert
            resident[expert] = slot
            loaded[expert] = slot
        self._upload_maps()

    def _forget_overwritten(self, misses):
        """After a failed pass: the slots it was loading hold no expert."""
        if self._maps_upload_event is not None:
            self._maps_upload_event.synchronize()
        resident, loaded = self._host_resident_array, self._host_miss_array
        owner = self._slot_owner
        loaded.fill(self._absent)
        for _, slot, _ in misses:
            if owner[slot] >= 0:
                resident[owner[slot]] = self._absent
                owner[slot] = -1
        self._upload_maps()

    def _upload_maps(self):
        self._maps.copy_(self._host_maps, non_blocking=True)
        if self.device.type == "cuda":
            if self._maps_upload_event is None:
                self._maps_upload_event = torch.cuda.Event()
            self._maps_upload_event.record(torch.cuda.current_stream(self.device))
        self.resident_uploads += 1

    def remap(self, local_ids, lease):
        """Mask other passes and map this pass's local expert IDs to slots."""
        safe = local_ids.clamp(0, self.logical_num_experts - 1).long()
        return torch.where(local_ids >= 0, self.slot_map[safe], -1)

    def release(self, lease):
        """Fence the last consumer before any future slot overwrite."""
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            if self._consumer_event is None:
                self._consumer_event = torch.cuda.Event()
            self._consumer_event.record(stream)
            self._consumer_stream = stream
        lease.release()
