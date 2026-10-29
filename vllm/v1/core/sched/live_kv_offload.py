# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in request-granularity KV spill/resume using native CPU snapshot tickets.

The scheduler owns logical tables and source allocations. The connector owns
CPU leases and rank acknowledgements. Model Runner V2 retains its
request slot, numerical selectors, accepted counts and RNG while unscheduled.
None of these ownership domains substitutes for the
others; ordinary prefix hits and recompute preemption remain separate paths.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)


@dataclass
class LiveKVRequest:
    request: Any
    ticket: Any
    phase: str = "quiescing"
    manifest: Any = None
    reservation: Any = None
    frontier: int = 0
    detached: bool = False
    started_at: float = field(default_factory=time.monotonic)
    phase_started_at: float = field(default_factory=time.monotonic)


class LiveKVOffloader:
    def __init__(
        self, scheduler: Any, watermark: float = 0.9, pause_target_s: float = 10.0
    ):
        self.scheduler = scheduler
        self.connector = scheduler.connector
        self.manager = scheduler.kv_cache_manager
        self.watermark = watermark
        self.pause_target_s = pause_target_s
        self.resumed_at: dict[str, float] = {}
        self.records: dict[str, LiveKVRequest] = {}
        self.ready_resumes: set[str] = set()
        self.resume_started_at: dict[str, float] = {}
        self.last_used: dict[str, int] = {}
        self.clock = 0
        self.spills = self.resumes = self.reclaimed_pages = self.rejections = 0

    def start(self, request: Any) -> bool:
        req_id = request.request_id
        if (
            req_id in self.records
            or request.num_computed_tokens < request.num_prompt_tokens
            or request.has_encoder_inputs
            or getattr(request, "lora_request", None) is not None
            or getattr(request, "resumable", False)
            or req_id not in self.last_used
        ):
            return False
        ticket = self.connector.begin_request_quiesce(req_id)
        if ticket is None:
            return False
        self.records[req_id] = LiveKVRequest(request, ticket)
        request.status = RequestStatus.WAITING_FOR_KV_SNAPSHOT
        # FCFS admission gives previously waiting requests the reclaimed space.
        self.scheduler.waiting.add_request(request)
        logger.info("Live KV quiesce %s at %d", req_id, request.num_computed_tokens)
        return True

    def evict_for_admission(self) -> None:
        scheduler = self.scheduler
        # A restore has priority over increasing the admitted working set.
        # Exchange existing V2 slots even when no fresh slot is available.
        overdue = self.oldest_overdue()
        if overdue is not None:
            scheduler.waiting.remove_requests((overdue.request,))
            scheduler.waiting.prepend_request(overdue.request)
            if not any(r.phase != "host" for r in self.records.values()):
                self.prepare_resume(overdue.request)
                if overdue.phase == "host":
                    self._evict_resident()
            return
        if (
            len(scheduler.running)
            + self.retained_waiting_slots()
            + scheduler.num_waiting_for_streaming_input
            >= scheduler.max_num_running_reqs
            or self.manager.usage < self.watermark
            or self.records
            or self.ready_resumes
            or not any(
                not self.has_worker_slot(req)
                and req.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
                and getattr(req, "lora_request", None) is None
                for queue in (scheduler.waiting, scheduler.kv_holding_waiting)
                for req in queue
            )
        ):
            return
        self._evict_resident()

    def oldest_overdue(self) -> LiveKVRequest | None:
        now = time.monotonic()
        return min(
            (
                record
                for record in self.records.values()
                if record.phase == "host"
                and not record.request.is_finished()
                and now - record.started_at >= self.pause_target_s
            ),
            key=lambda record: record.started_at,
            default=None,
        )

    def defer_prefills(self) -> bool:
        """Drain actual transfers; let incomplete residents become spillable."""
        return any(
            r.phase in ("quiescing", "storing", "loading")
            for r in self.records.values()
        )

    def allow_admission(self, request: Any) -> bool:
        return self.has_worker_slot(request) or not (
            self.defer_prefills() or self.oldest_overdue() is not None
        )

    def _evict_resident(self) -> None:
        scheduler = self.scheduler
        now = time.monotonic()
        candidates = sorted(
            scheduler.running,
            key=lambda req: (self.last_used.get(req.request_id, 0), req.arrival_time),
        )
        for request in candidates:
            resumed_at = self.resumed_at.get(request.request_id)
            if resumed_at is not None and now - resumed_at < self.pause_target_s:
                continue
            if self.start(request):
                scheduler.running.remove(request)
                return

    def _state(self, record: LiveKVRequest) -> str:
        value = self.connector.snapshot_status(record.ticket)
        return getattr(value, "value", value)

    def cancel(self, request_id: str) -> bool:
        record = self.records.get(request_id)
        if record is None:
            return False
        if record.detached and not record.request.is_finished():
            raise RuntimeError("cannot resume detached KV after cancelling its restore")
        self.connector.cancel_snapshot(record.ticket)
        record.phase = "cancelling"
        return True

    def advance(self) -> None:
        scheduler = self.scheduler
        for req_id, record in list(self.records.items()):
            request = record.request
            if request.is_finished() and record.phase != "cancelling":
                self.cancel(req_id)
            state = self._state(record)
            if record.phase == "cancelling":
                if state != "released":
                    continue
                if record.reservation is not None and record.reservation.active:
                    self.manager.cancel_snapshot_restore(record.reservation)
                del self.records[req_id]
                if request.is_finished():
                    scheduler._free_blocks(request)
                else:
                    request.status = RequestStatus.PREEMPTED
                    self.ready_resumes.add(req_id)
                continue
            if record.phase == "quiescing":
                if (
                    state != "quiesced"
                    or request.num_in_flight_tokens
                    or not scheduler._request_blocks_can_be_freed(request)
                ):
                    continue
                try:
                    manifest = self.manager.export_request_snapshot(req_id)
                except ValueError as error:
                    if getattr(error, "retryable", False):
                        continue
                    # Unsupported ownership never falls through to free.
                    logger.info("Live KV snapshot rejected %s: %s", req_id, error)
                    self.rejections += 1
                    self.cancel(req_id)
                    continue
                ticket = self.connector.begin_request_snapshot(
                    req_id, manifest.source_physical_page_ids, ticket=record.ticket
                )
                if ticket is None:
                    self.rejections += 1
                    self.cancel(req_id)
                    continue
                record.manifest = manifest
                record.frontier = request.num_computed_tokens
                record.phase = "storing"
                self._note_phase(record, "quiesce")
            elif record.phase == "storing":
                if state != "host_ready" or not self.connector.snapshot_sources_ready(
                    record.ticket, record.manifest.source_physical_page_ids
                ):
                    continue
                if request.num_computed_tokens != record.frontier:
                    raise RuntimeError("Live KV request advanced while frozen")
                if not self.connector.detach_snapshot_prefix_sources(record.ticket):
                    continue
                free_before = self.manager.block_pool.get_num_free_blocks()
                try:
                    self.manager.detach_request_snapshot(request, record.manifest)
                except ValueError as error:
                    if getattr(error, "retryable", False):
                        continue
                    logger.info("Live KV detach rejected %s: %s", req_id, error)
                    self.rejections += 1
                    self.cancel(req_id)
                    continue
                self.reclaimed_pages += (
                    self.manager.block_pool.get_num_free_blocks() - free_before
                )
                record.detached = True
                record.phase = "host"
                self._note_phase(record, "store")
                request.status = RequestStatus.WAITING_FOR_KV_RESTORE
                # Quiescing requests were classified as KV-holding waiters.
                # Once detached, park behind fresh admission instead of
                # immediately reserving the very pages this spill released.
                scheduler.kv_holding_waiting.remove_requests((request,))
                scheduler.waiting.remove_requests((request,))
                scheduler.waiting.add_request(request)
                self.spills += 1
                logger.info(
                    "Live KV spill %s: %d pages, frontier %d",
                    req_id,
                    len(record.manifest.pages),
                    record.frontier,
                )
            elif record.phase == "loading" and state == "restored":
                if request.num_computed_tokens != record.frontier:
                    raise RuntimeError("Live KV restore frontier changed")
                self.manager.commit_snapshot_restore(
                    request, record.manifest, record.reservation
                )
                self.connector.release_snapshot(record.ticket)
                request.status = RequestStatus.PREEMPTED
                self.ready_resumes.add(req_id)
                self.resume_started_at[req_id] = record.started_at
                del self.records[req_id]
                self.resumes += 1
                self._note_phase(record, "load")
                logger.info("Live KV resume %s at %d", req_id, record.frontier)
                logger.info(
                    "Live KV restore ready %s: %.3f seconds since quiesce",
                    req_id,
                    time.monotonic() - record.started_at,
                )

    @staticmethod
    def _note_phase(record: LiveKVRequest, phase: str) -> None:
        now = time.monotonic()
        logger.info(
            "Live KV phase %s %s: %.3f seconds",
            record.request.request_id,
            phase,
            now - record.phase_started_at,
        )
        record.phase_started_at = now

    def prepare_resume(self, request: Any) -> bool:
        record = self.records.get(request.request_id)
        if record is None:
            return True
        if record.phase != "host":
            return False
        reservation = self.manager.reserve_snapshot_restore(record.manifest)
        if reservation is None:
            return False
        if not self.connector.begin_snapshot_restore(
            record.ticket, reservation.destination_physical_page_ids
        ):
            self.manager.cancel_snapshot_restore(reservation)
            return False
        record.reservation = reservation
        record.phase = "loading"
        self._note_phase(record, "host_wait")
        return False

    def retained_v2_resumes(self, resumed: list[Any]) -> list[Any]:
        """Keep V2 slots, RNG, accepted counts and recurrent selectors alive."""
        retained = [req for req in resumed if req.request_id in self.ready_resumes]
        resumed[:] = [
            req for req in resumed if req.request_id not in self.ready_resumes
        ]
        return retained

    def has_worker_slot(self, request: Any) -> bool:
        return (
            request.request_id in self.records
            or request.request_id in self.ready_resumes
        )

    def retained_waiting_slots(self) -> int:
        running = {req.request_id for req in self.scheduler.running}
        return len(self.records) + len(self.ready_resumes - running)

    def note_schedule(self, output: Any) -> None:
        for req_id in output.num_scheduled_tokens:
            self.clock += 1
            self.last_used[req_id] = self.clock
            if req_id in self.ready_resumes:
                self.ready_resumes.remove(req_id)
                self.resumed_at[req_id] = time.monotonic()
                started = self.resume_started_at.pop(req_id, None)
                if started is not None:
                    logger.info(
                        "Live KV scheduled resume %s: %.3f seconds since quiesce",
                        req_id,
                        self.resumed_at[req_id] - started,
                    )
        for req_id in output.finished_req_ids:
            self.last_used.pop(req_id, None)
            self.ready_resumes.discard(req_id)
            self.resumed_at.pop(req_id, None)
            self.resume_started_at.pop(req_id, None)


def make_live_kv_offloader(scheduler: Any) -> LiveKVOffloader | None:
    config = scheduler.vllm_config
    additional = config.additional_config
    if not isinstance(additional, dict):
        return None
    options = additional.get("rocm_mono_decode", {})
    if not isinstance(options, dict):
        return None
    enabled = options.get("kv_cache_live_offload", False)
    if not isinstance(enabled, bool):
        raise ValueError("kv_cache_live_offload must be a boolean")
    if not enabled:
        return None
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import on_gfx10x

    if (
        not current_platform.is_rocm()
        or not on_gfx10x()
        or not config.kernel_config.monokernel
        or not any(
            name.startswith("Qwen4Exp") for name in config.model_config.architectures
        )
        or config.parallel_config.pipeline_parallel_size != 1
        or not scheduler.use_v2_model_runner
        or scheduler.cache_config.mamba_cache_mode != "align"
        or scheduler.policy.value != "fcfs"
        or not hasattr(scheduler.connector, "begin_request_quiesce")
    ):
        raise ValueError(
            "live KV offload requires native RDNA2 Qwen4Exp MonoKernel, aligned "
            "recurrent state, v2 runner, PP=1, FCFS and native CPU KV offloading"
        )
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    backend = scheduler.connector.scheduler
    if (
        not isinstance(backend.manager, CPUOffloadingManager)
        or backend.manager.store_threshold >= 2
        or backend.config.blocks_per_chunk != 1
    ):
        raise ValueError(
            "live KV offload requires unfiltered native one-page CPU chunks"
        )
    watermark = options.get("kv_cache_live_offload_watermark", 0.9)
    if (
        isinstance(watermark, bool)
        or not isinstance(watermark, (int, float))
        or not 0 <= watermark <= 1
    ):
        raise ValueError("kv_cache_live_offload_watermark must be in [0, 1]")
    pause_target_s = options.get("kv_cache_live_offload_pause_target_s", 10.0)
    if (
        isinstance(pause_target_s, bool)
        or not isinstance(pause_target_s, (int, float))
        or not 0 < pause_target_s <= 3600
    ):
        raise ValueError("kv_cache_live_offload_pause_target_s must be in (0, 3600]")
    return LiveKVOffloader(scheduler, float(watermark), float(pause_target_s))
