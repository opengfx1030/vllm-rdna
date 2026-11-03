# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lane-aware scheduler for the two-lane engine.

The decode lane (``Lane.DECODE``) schedules only graph-eligible decode rows of
decode-resident requests and never admits. The prefill lane
(``Lane.PREFILL``) schedules prompt chunks and admits new requests. A request
flips from P to D exactly once, after the engine processed the output of its
final prompt chunk (``_lane_after_output``), and is emitted by the next D
batch as a new request through the resumed-request payload.

All lane state lives here; ``Scheduler`` only dispatches to
``_schedule_lane`` when ``SchedulerConfig.two_lane`` is set, so the flag-off
path stays the baseline one (I9).
"""

import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import GrammarOutput, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.lane import NUM_LANES, Lane
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

_LANE_NAMES = {int(Lane.DECODE): "decode", int(Lane.PREFILL): "prefill"}


class TwoLaneScheduler(AsyncScheduler):
    """Scheduler that produces one lane's batch per ``schedule`` call."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lane_counters: dict[int, dict[str, int]] = {
            int(lane): {"batches": 0, "tokens": 0, "handoffs": 0}
            for lane in range(NUM_LANES)
        }
        # Rotating FCFS cursor for the prefill lane (R-S1-9).
        self._p_cursor = 0
        if self.lane_mode:
            logger.info_once(
                "Two-lane scheduler: decode budget %d tokens, prefill budget %d "
                "tokens, %d draft slots/request",
                self._lane_token_budget(Lane.DECODE),
                self._lane_token_budget(Lane.PREFILL),
                self.num_spec_tokens,
            )

    # ------------------------------------------------------------------
    # Public lane API
    # ------------------------------------------------------------------
    def lane_stats(self) -> dict[str, dict[str, int]]:
        """Per-lane counters for the engine trace (R-S1-12)."""
        stats: dict[str, dict[str, int]] = {}
        for lane in range(NUM_LANES):
            counters = self.lane_counters[lane]
            stats[_LANE_NAMES[lane]] = {
                "batches": counters["batches"],
                "tokens": counters["tokens"],
                "handoffs": counters["handoffs"],
                "deferred_free_depth": len(self.deferred_frees),
                "lane_seq": self.lane_seq[lane],
                "processed_seq": self.processed_seq[lane],
            }
        return stats

    def lane_has_pending_release(self, lane: Lane) -> bool:
        """Whether `lane`'s next output still carries release ids.

        The engine must keep submitting that lane's (possibly empty) batch
        until this turns False, or a worker slot would never be freed.
        """
        lane_id = int(lane)
        return bool(
            self.lane_finished_req_ids[lane_id] or self.lane_preempted_req_ids[lane_id]
        )

    def add_request(self, request: Request) -> None:
        if self.lane_mode:
            # Admission always happens in the prefill lane (I2).
            request.lane = int(Lane.PREFILL)
            request.awaiting_handoff = False
        super().add_request(request)

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput):
        """Per-lane grammar bitmask, constraining the bonus token only.

        Each lane's batch carries its own ``has_structured_output_requests``
        and its own scheduled request ids, so this is already per lane: the
        decode lane constrains every decode step, and the prefill lane
        constrains the first sampled token of a prompt (its final chunk is not
        a prefill chunk, so it is included). The engine core defers the call
        until the previous batch of the same lane has been applied, so the FSM
        always holds that lane's accepted tokens.

        Speculative drafts are not fed to the FSM here. At schedule time
        ``scheduled_spec_decode_tokens`` holds placeholders (-1) that the
        worker fills in later, and the bitmask fill treats a placeholder as
        padding: the bonus row would then copy an unconstrained row. Passing no
        spec tokens constrains the bonus token and reports zero acceptable
        drafts, so the verifier rejects every draft. That is the same bitmask
        the base class produces when speculative decoding is off; filling the
        real draft ids (as the async path does with
        ``update_draft_token_ids_in_output``) is a throughput follow-up, not a
        correctness one.
        """
        if not scheduler_output.has_structured_output_requests:
            return None
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None
        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests, structured_output_request_ids, {}
        )
        return GrammarOutput(
            structured_output_request_ids,
            bitmask,
            [0] * len(structured_output_request_ids),
        )

    # ------------------------------------------------------------------
    # Budgets
    # ------------------------------------------------------------------
    def _lane_token_budget(self, lane: Lane) -> int:
        if lane == Lane.DECODE:
            # One uniform 1 + k row per running sequence (R-S1-2).
            budget = self.max_num_running_reqs * (1 + self.num_spec_tokens)
        else:
            budget = (
                self.scheduler_config.two_lane_prefill_budget
                if self.scheduler_config.two_lane_prefill_budget is not None
                else self.max_num_scheduled_tokens
            )
        return min(budget, self.max_num_scheduled_tokens)

    def _lane_is_prefilling(self, request: Request) -> bool:
        return request.num_computed_tokens < max(
            request.num_prompt_tokens, request.num_tokens - 1
        )

    def _lane_prefill_share(self) -> int:
        """Per-request prefill cap: two or more prefilling requests share the
        prefill budget so no long prompt starves the others (R-S1-9)."""
        num_prefilling = 0
        for request in self.running:
            if (
                request.lane == int(Lane.PREFILL)
                and not request.is_finished()
                and self._lane_is_prefilling(request)
            ):
                num_prefilling += 1
        for request in list(self.kv_holding_waiting) + list(self.waiting):
            if not request.is_finished() and self._lane_is_prefilling(request):
                num_prefilling += 1
        if num_prefilling < 2:
            return 0
        block = self.mamba_state_block_size or self.cache_config.block_size
        return max(block, self._lane_token_budget(Lane.PREFILL) // num_prefilling)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def _schedule_lane(self, lane: Lane) -> SchedulerOutput:
        lane = Lane(lane)
        lane_id = int(lane)
        self.current_step += 1

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}

        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        encoder_budget = self.max_num_encoder_input_tokens
        # Pause zeroes both lanes' budgets (R-S1-11).
        paused = self._pause_state == PauseState.PAUSED_ALL
        token_budget = 0 if paused else self._lane_token_budget(lane)
        drafts_needed = not self.can_skip_drafts
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        if lane == Lane.DECODE:
            token_budget, drafts_needed = self._lane_schedule_decode(
                token_budget=token_budget,
                input_budget=input_budget,
                draft_slots=draft_slots,
                scheduled_timestamp=scheduled_timestamp,
                scheduled_resumed_reqs=scheduled_resumed_reqs,
                scheduled_running_reqs=scheduled_running_reqs,
                preempted_reqs=preempted_reqs,
                req_to_new_blocks=req_to_new_blocks,
                num_scheduled_tokens=num_scheduled_tokens,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=scheduled_encoder_inputs,
                drafts_needed=drafts_needed,
                encoder_budget=encoder_budget,
                paused=paused,
            )
        else:
            token_budget, drafts_needed = self._lane_schedule_prefill(
                token_budget=token_budget,
                input_budget=input_budget,
                draft_slots=draft_slots,
                p_share=0 if paused else self._lane_prefill_share(),
                scheduled_timestamp=scheduled_timestamp,
                scheduled_new_reqs=scheduled_new_reqs,
                scheduled_resumed_reqs=scheduled_resumed_reqs,
                scheduled_running_reqs=scheduled_running_reqs,
                preempted_reqs=preempted_reqs,
                req_to_new_blocks=req_to_new_blocks,
                num_scheduled_tokens=num_scheduled_tokens,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=scheduled_encoder_inputs,
                drafts_needed=drafts_needed,
                encoder_budget=encoder_budget,
                paused=paused,
            )

        return self._lane_build_output(
            lane_id=lane_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_resumed_reqs=scheduled_resumed_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            req_to_new_blocks=req_to_new_blocks,
            num_scheduled_tokens=num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            drafts_needed=drafts_needed,
        )

    # -- decode ---------------------------------------------------------
    def _lane_schedule_decode(
        self,
        *,
        token_budget: int,
        input_budget: int,
        draft_slots: int,
        scheduled_timestamp: float,
        scheduled_resumed_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        preempted_reqs: list[Request],
        req_to_new_blocks: dict,
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        scheduled_encoder_inputs: dict[str, list[int]],
        drafts_needed: bool,
        encoder_budget: int,
        paused: bool,
    ) -> tuple[int, bool]:
        lane_id = int(Lane.DECODE)
        if paused:
            return token_budget, drafts_needed
        for request in list(self.running):
            if token_budget <= 0:
                break
            if request not in self.running:
                # A preemption during this step may have removed it.
                continue
            if (
                request.lane != lane_id
                or request.status != RequestStatus.RUNNING
                or request.is_finished()
            ):
                # D never touches P-resident requests (I1).
                continue
            if self.current_step < request.next_decode_eligible_step:
                continue
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue

            handoff = request.awaiting_handoff
            if handoff:
                # Drafts proposed by the P drafter are discarded at handoff;
                # the first D row keeps the uniform 1 + k shape (R-S1-4).
                request.spec_token_ids = [-1] * self.num_spec_tokens

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            num_new_tokens = min(
                num_new_tokens,
                token_budget,
                input_budget - draft_slots,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            if num_new_tokens <= 0:
                continue
            if num_new_tokens > 1 + self.num_spec_tokens:
                # I1: a D row is at most 1 + k tokens.
                continue

            # Encoder inputs (multimodal decode of a partially encoded prompt).
            encoder_inputs_to_schedule = None
            external_encoder_input_ids: list[int] = []
            duplicate_encoder_input_ids: list[int] = []
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    encoder_budget,
                    external_encoder_input_ids,
                    duplicate_encoder_input_ids,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )
            if num_new_tokens == 0:
                continue

            new_blocks = self._lane_allocate(
                request=request,
                num_new_tokens=num_new_tokens,
                lane_id=lane_id,
                delay_cache_blocks=True,
                num_lookahead_tokens=self.num_lookahead_tokens,
                scheduled_timestamp=scheduled_timestamp,
                preempted_reqs=preempted_reqs,
                scheduled_ids=num_scheduled_tokens,
            )
            if new_blocks is None:
                # R-S1-8: skip this request instead of an illegal preemption.
                continue

            request_id = request.request_id
            if handoff:
                # Emit the request as a NEW worker request: prefill_token_ids
                # = prompt + [t1], num_computed_tokens = L (5.1, R-S1-3).
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                scheduled_resumed_reqs.append(request)
            else:
                req_to_new_blocks[request_id] = new_blocks
                scheduled_running_reqs.append(request)
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            drafts_needed = drafts_needed or not self._drafts_unused(
                request, request.num_computed_tokens, num_new_tokens
            )

            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request_id] = spec_token_ids
                request.spec_token_ids = []

            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            if encoder_inputs_to_schedule or external_encoder_input_ids:
                self._allocate_encoder_inputs(
                    request,
                    (encoder_inputs_to_schedule or []) + external_encoder_input_ids,
                    duplicate_encoder_input_ids,
                )

            if handoff:
                request.awaiting_handoff = False
                # The P worker slot release was queued when the request
                # flipped; the D worker now owns the request (R-S1-5).
                request.lane_resident |= 1 << lane_id
                self.lane_counters[lane_id]["handoffs"] += 1
        return token_budget, drafts_needed

    # -- prefill --------------------------------------------------------
    def _lane_schedule_prefill(
        self,
        *,
        token_budget: int,
        input_budget: int,
        draft_slots: int,
        p_share: int,
        scheduled_timestamp: float,
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        preempted_reqs: list[Request],
        req_to_new_blocks: dict,
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        scheduled_encoder_inputs: dict[str, list[int]],
        drafts_needed: bool,
        encoder_budget: int,
        paused: bool,
    ) -> tuple[int, bool]:
        lane_id = int(Lane.PREFILL)
        if paused:
            return token_budget, drafts_needed

        # FCFS order rotating through a cursor, so a budget too small for all
        # shares does not always serve the same requests first (R-S1-9).
        p_running = [
            request
            for request in self.running
            if request.lane == lane_id
            and request.status == RequestStatus.RUNNING
            and not request.is_finished()
        ]
        if p_running:
            offset = self._p_cursor % len(p_running)
            order = p_running[offset:] + p_running[:offset]
            self._p_cursor = (offset + 1) % len(p_running)
        else:
            order = []

        for request in order:
            if token_budget <= 0:
                break
            if request.status != RequestStatus.RUNNING or request.is_finished():
                # `order` is a snapshot; an earlier allocation may have
                # preempted this entry.
                continue
            if self.current_step < request.next_decode_eligible_step:
                continue
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            if num_new_tokens <= 0:
                continue
            request_token_budget = min(token_budget, input_budget - draft_slots)
            if (
                not self.scheduler_config.enable_chunked_prefill
                and num_new_tokens > request_token_budget
            ):
                break
            if p_share > 0:
                num_new_tokens = min(num_new_tokens, p_share)
            num_new_tokens = min(num_new_tokens, request_token_budget)
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            encoder_inputs_to_schedule = None
            external_encoder_input_ids: list[int] = []
            duplicate_encoder_input_ids: list[int] = []
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    encoder_budget,
                    external_encoder_input_ids,
                    duplicate_encoder_input_ids,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )
            num_new_tokens = self._reserve_prefill_lookahead(
                request, request.num_computed_tokens, num_new_tokens
            )
            if num_new_tokens == 0:
                continue

            num_encoder_tokens = 0
            if (
                self.is_encoder_decoder
                and request.has_encoder_inputs
                and encoder_inputs_to_schedule
            ):
                num_encoder_tokens = sum(
                    request.get_num_encoder_embeds(i)
                    for i in encoder_inputs_to_schedule
                )

            new_blocks = self._lane_allocate(
                request=request,
                num_new_tokens=num_new_tokens,
                lane_id=lane_id,
                delay_cache_blocks=False,
                num_lookahead_tokens=self.num_lookahead_tokens,
                scheduled_timestamp=scheduled_timestamp,
                preempted_reqs=preempted_reqs,
                scheduled_ids=num_scheduled_tokens,
                num_encoder_tokens=num_encoder_tokens,
            )
            if new_blocks is None:
                # R-S1-8: skip this request instead of an illegal preemption.
                continue

            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            drafts_needed = drafts_needed or not self._drafts_unused(
                request, request.num_computed_tokens, num_new_tokens
            )
            if request.spec_token_ids:
                # P rows carry prompt tokens only; drafts belong to D.
                request.spec_token_ids = []
            scheduled_running_reqs.append(request)
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            if encoder_inputs_to_schedule or external_encoder_input_ids:
                self._allocate_encoder_inputs(
                    request,
                    (encoder_inputs_to_schedule or []) + external_encoder_input_ids,
                    duplicate_encoder_input_ids,
                )

        # Admission: only the prefill lane admits (I2, R-S1-1).
        if self._pause_state != PauseState.PAUSED_ALL:
            step_skipped_waiting: deque[Request] = deque()
            step_skipped_kv_holding: deque[Request] = deque()

            def skip_request(from_queue) -> None:
                request_to_skip = from_queue.pop_request()
                self.deferred_waiting.add(request_to_skip)
                if self._holds_kv_blocks(request_to_skip):
                    step_skipped_kv_holding.appendleft(request_to_skip)
                else:
                    step_skipped_waiting.appendleft(request_to_skip)

            while token_budget > 0:
                request_queue = self.kv_holding_waiting
                if not request_queue and self._pause_state == PauseState.UNPAUSED:
                    request_queue = self.waiting
                if not request_queue or input_budget <= draft_slots:
                    break
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_active_reqs:
                    break

                request = request_queue.peek_request()
                request_id = request.request_id

                if not self._handle_blocked_waiting_request(request):
                    skip_request(request_queue)
                    continue
                if request.num_stale_output_tokens and not request.drop_stale_output:
                    skip_request(request_queue)
                    continue

                request.lane = lane_id
                request.awaiting_handoff = False
                did_prefix_cache_lookup = False
                num_replay_tokens = 0
                if request.num_computed_tokens == 0:
                    if request.resumable and self.num_hash_lookahead_tokens:
                        # Resident KV can retain a discarded streaming token's
                        # hash; recompute with the current sequence.
                        request.block_hashes.clear()
                        request.update_block_hashes()
                    did_prefix_cache_lookup = True
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                        _hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)
                    if self._ec_transfer_pending(
                        request, num_new_local_computed_tokens
                    ):
                        skip_request(request_queue)
                        continue
                    num_computed_tokens = num_new_local_computed_tokens
                    if request.prefill_stats and request.num_preemptions <= 0:
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=0,
                        )
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
                    if self._ec_transfer_pending(request, num_computed_tokens):
                        skip_request(request_queue)
                        continue

                encoder_inputs_to_schedule = None
                external_encoder_input_ids = []
                duplicate_encoder_input_ids = []
                if did_prefix_cache_lookup:
                    num_replay_tokens = self._mark_prefix_replay(
                        request, num_computed_tokens, False
                    )
                    num_computed_tokens -= num_replay_tokens

                request_token_budget = min(token_budget, input_budget - draft_slots)
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > request_token_budget
                ):
                    break
                if p_share > 0:
                    num_new_tokens = min(num_new_tokens, p_share)
                num_new_tokens = min(num_new_tokens, request_token_budget)
                if num_new_tokens <= 0:
                    skip_request(request_queue)
                    continue
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        0,
                    )
                    if num_new_tokens == 0:
                        skip_request(request_queue)
                        continue

                num_encoder_tokens = 0
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        encoder_budget,
                        external_encoder_input_ids,
                        duplicate_encoder_input_ids,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        encoder_budget,
                        shift_computed_tokens=self.num_prefill_lookahead,
                    )
                    if self.is_encoder_decoder and encoder_inputs_to_schedule:
                        num_encoder_tokens = sum(
                            request.get_num_encoder_embeds(i)
                            for i in encoder_inputs_to_schedule
                        )
                num_new_tokens = self._reserve_prefill_lookahead(
                    request, num_computed_tokens, num_new_tokens
                )
                if num_new_tokens == 0:
                    skip_request(request_queue)
                    continue

                # Replayed tokens already count in the adopted hit; a chunk
                # ending inside the replayed range needs no new slots.
                num_tokens_past_hit = max(num_new_tokens - num_replay_tokens, 0)
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_tokens_past_hit,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    has_scheduled_reqs=bool(self.running),
                )
                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )

                request = request_queue.pop_request()
                self.deferred_waiting.discard(request)
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                request.lane = lane_id
                request.lane_resident |= 1 << lane_id
                drafts_needed = drafts_needed or not self._drafts_unused(
                    request, num_computed_tokens, num_new_tokens
                )
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                if encoder_inputs_to_schedule or external_encoder_input_ids:
                    self._allocate_encoder_inputs(
                        request,
                        (encoder_inputs_to_schedule or []) + external_encoder_input_ids,
                        duplicate_encoder_input_ids,
                    )

            if step_skipped_kv_holding:
                self.kv_holding_waiting.prepend_requests(step_skipped_kv_holding)
            if step_skipped_waiting:
                self.waiting.prepend_requests(step_skipped_waiting)

        return token_budget, drafts_needed

    # -- allocation and preemption --------------------------------------
    def _lane_allocate(
        self,
        *,
        request: Request,
        num_new_tokens: int,
        lane_id: int,
        delay_cache_blocks: bool,
        num_lookahead_tokens: int,
        scheduled_timestamp: float,
        preempted_reqs: list[Request],
        scheduled_ids: dict[str, int],
        num_encoder_tokens: int = 0,
    ):
        while True:
            with record_function_or_nullcontext("schedule: allocate_slots"):
                blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=num_lookahead_tokens,
                    delay_cache_blocks=delay_cache_blocks,
                    num_encoder_tokens=num_encoder_tokens,
                    has_scheduled_reqs=bool(self.running),
                )
            if blocks is not None:
                return blocks
            victim = self._lane_pick_victim(request, scheduled_ids)
            if victim is None:
                return None
            # The base scheduler pops the victim before preempting it; leaving
            # it in self.running re-adds it on readmission and leaks a second
            # entry (review finding 3).
            self.running.remove(victim)
            self._preempt_request(victim, scheduled_timestamp, lane=lane_id)
            preempted_reqs.append(victim)

    def _lane_pick_victim(
        self, request: Request, scheduled_ids: dict[str, int]
    ) -> Request | None:
        """A victim with no in-flight batch in either lane (R-S1-8, 5.3)."""
        candidates = [
            candidate
            for candidate in self.running
            if candidate is not request
            and candidate.request_id not in scheduled_ids
            and candidate.status == RequestStatus.RUNNING
            and not candidate.is_finished()
            # 5.3: a request in the handoff window is not preemptible.
            and not candidate.awaiting_handoff
            and self._request_blocks_can_be_freed(candidate)
        ]
        if not candidates:
            return None
        if self.policy == SchedulingPolicy.PRIORITY:
            return max(candidates, key=lambda r: (r.priority, r.arrival_time))
        return candidates[-1]

    # -- output ---------------------------------------------------------
    def _lane_build_output(
        self,
        *,
        lane_id: int,
        scheduled_new_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        scheduled_running_reqs: list[Request],
        req_to_new_blocks: dict,
        num_scheduled_tokens: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
        scheduled_encoder_inputs: dict[str, list[int]],
        drafts_needed: bool,
    ) -> SchedulerOutput:
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        if scheduled_new_reqs or scheduled_resumed_reqs:
            # Lane mode runs the V2 model runner: the new-request payload
            # carries prefill_token_ids, which is what the handoff needs.
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    uses_mrope=self.model_uses_mrope,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = []

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # Mamba "align" boundary states must be handed off with exact ids.
        self.kv_cache_manager.take_boundary_state_offloads()
        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        if kv_cache_block_copies:
            # The copies run with this batch; its seq is the fence.
            fence = [self.lane_seq[index] for index in range(NUM_LANES)]
            fence[lane_id] += 1 if total_num_scheduled_tokens else 0
            self._free_cow_retained_blocks(cow_retained_blocks, (fence[0], fence[1]))
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        scheduled_encoder_input_stats = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs
            )

        lane_seq = self.lane_seq[lane_id] + 1 if total_num_scheduled_tokens else 0
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            preempted_req_ids=self.lane_preempted_req_ids[lane_id],
            finished_req_ids=self.lane_finished_req_ids[lane_id],
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            encoder_input_updates=self._get_encoder_input_updates(
                num_scheduled_tokens, new_reqs_data, scheduled_encoder_inputs
            ),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=False,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_connector_block_state=None,
            num_spec_tokens_to_schedule=self.num_spec_tokens,
            skip_drafts=not drafts_needed,
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
            lane=lane_id,
            lane_seq=lane_seq,
            lane_fence=(self.processed_seq[0], self.processed_seq[1]),
        )

        if total_num_scheduled_tokens:
            self.lane_seq[lane_id] = lane_seq
            self.lane_counters[lane_id]["batches"] += 1
            self.lane_counters[lane_id]["tokens"] += total_num_scheduled_tokens

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # ------------------------------------------------------------------
    # Handoff
    # ------------------------------------------------------------------
    def _lane_after_output(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        stopped: bool,
        output_is_stale: bool,
    ) -> None:
        """Flip P -> D after the final prompt-chunk output (I2, R-S1-3)."""
        if (
            int(scheduler_output.lane) != int(Lane.PREFILL)
            or request.lane != int(Lane.PREFILL)
            or stopped
            or output_is_stale
            or request.is_finished()
            or request.is_prefill_chunk
            or request.num_output_tokens <= 0
            or request.num_output_placeholders != 0
        ):
            return
        request.lane = int(Lane.DECODE)
        request.awaiting_handoff = True
        # The P drafter's proposals do not cross the handoff (5.1).
        request.spec_token_ids = [-1] * self.num_spec_tokens
        # Queue the P worker slot release; it rides the next P output (R-S1-5).
        if request.lane_resident & (1 << int(Lane.PREFILL)):
            self.lane_finished_req_ids[int(Lane.PREFILL)].add(request.request_id)
        request.lane_resident &= ~(1 << int(Lane.PREFILL))
        self.lane_counters[int(Lane.PREFILL)]["handoffs"] += 1
