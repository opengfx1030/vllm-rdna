# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S1 unit tests: lane-aware scheduling (contract 5.1-5.3, R-S1-1..R-S1-13).

CPU only: the scheduler never runs the model.
"""

import time
from unittest.mock import patch

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.two_lane_scheduler import TwoLaneScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.lane import Lane
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

MODEL = "facebook/opt-125m"
DECODE = Lane.DECODE
PREFILL = Lane.PREFILL


def make_lane_scheduler(
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 2048,
    block_size: int = 16,
    num_blocks: int = 2000,
    num_speculative_tokens: int = 0,
    enable_prefix_caching: bool = False,
    max_model_len: int | None = None,
    prefill_budget: int | None = None,
    two_lane: bool = True,
) -> TwoLaneScheduler:
    """Build a TwoLaneScheduler without touching the platform/executor."""
    max_model_len = max_model_len or max_num_batched_tokens
    model_config = ModelConfig(
        model=MODEL,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        async_scheduling=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        watermark=0.0,
        policy="fcfs",
        two_lane_prefill_budget=prefill_budget,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    speculative_config = (
        SpeculativeConfig(model="ngram", num_speculative_tokens=num_speculative_tokens)
        if num_speculative_tokens
        else None
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
        observability_config=ObservabilityConfig(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    # VllmConfig validates the flag against the platform/executor at
    # construction, which a CPU unit test cannot satisfy. The scheduler reads
    # the flag itself, so flip it after validation.
    scheduler_config.two_lane = two_lane
    register_all_kvcache_specs(vllm_config)
    scheduler = TwoLaneScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    # The two-lane engine requires the V2 model runner (S0 validation).
    scheduler.use_v2_model_runner = True
    return scheduler


def run_step(scheduler, lane: Lane, sampled=None, default_token: int = 1):
    """Schedule one lane batch and feed its (synthetic) worker output.

    Final-chunk rows get `default_token`; mid-prefill rows get no token.
    """
    output = scheduler.schedule(lane=lane)
    req_ids = list(output.num_scheduled_tokens)
    sampled = dict(sampled or {})
    for req_id in req_ids:
        if req_id in sampled:
            continue
        request = scheduler.requests.get(req_id)
        is_final = request is not None and not request.is_prefill_chunk
        sampled[req_id] = [default_token] if is_final else []
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[sampled[req_id] for req_id in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    return output, scheduler.update_from_output(output, model_output)


def prefill_to_handoff(scheduler, request_id: str, token: int = 100):
    """Run P batches until `request_id` has flipped to the decode lane."""
    for _ in range(256):
        output, _ = run_step(scheduler, PREFILL, default_token=token)
        request = scheduler.requests.get(request_id)
        if request is None or request.lane == int(DECODE):
            return output
    raise AssertionError("request never reached the decode lane")


# ---------------------------------------------------------------------------
# Lane split (I1, R-S1-1)
# ---------------------------------------------------------------------------
def test_lane_split_admission_only_in_prefill():
    scheduler = make_lane_scheduler(max_num_batched_tokens=256)
    requests = create_requests(
        num_requests=2, num_tokens=40, max_tokens=8, req_ids=["a", "b"]
    )
    for request in requests:
        scheduler.add_request(request)

    # D must not admit: with nothing D-resident the D batch is empty.
    d_output = scheduler.schedule(lane=DECODE)
    assert d_output.total_num_scheduled_tokens == 0
    assert d_output.scheduled_new_reqs == []
    assert d_output.lane == int(DECODE)
    assert scheduler.requests["a"].lane == int(PREFILL)

    # P admits and schedules prompt chunks only.
    p_output = scheduler.schedule(lane=PREFILL)
    assert set(p_output.num_scheduled_tokens) == {"a", "b"}
    assert all(
        scheduler.requests[req_id].num_output_tokens == 0
        for req_id in p_output.num_scheduled_tokens
    )
    assert p_output.lane == int(PREFILL)
    assert p_output.scheduled_spec_decode_tokens == {}


def test_decode_batch_holds_only_decode_rows():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=16, req_ids=["d"]
    )[0]
    scheduler.add_request(request)
    prefill_to_handoff(scheduler, "d")

    # Consume the handoff batch, then look at an ordinary decode batch.
    handoff_output, _ = run_step(scheduler, DECODE)
    assert [req.req_id for req in handoff_output.scheduled_new_reqs] == ["d"]
    assert handoff_output.num_scheduled_tokens == {"d": 1}

    # A second request arrives; D must never take it from the waiting queue.
    late = create_requests(
        num_requests=1, num_tokens=16, max_tokens=16, req_ids=["late"]
    )[0]
    scheduler.add_request(late)

    d_output = scheduler.schedule(lane=DECODE)
    assert set(d_output.num_scheduled_tokens) == {"d"}
    assert d_output.num_scheduled_tokens["d"] == 1  # no spec tokens: one row token
    assert d_output.scheduled_new_reqs == []  # no handoff pending
    assert late.lane == int(PREFILL)

    # The P lane keeps prefilling the newcomer and never touches the decoder.
    p_output = scheduler.schedule(lane=PREFILL)
    assert "d" not in p_output.num_scheduled_tokens
    assert "late" in p_output.num_scheduled_tokens


# ---------------------------------------------------------------------------
# Handoff (I2, R-S1-3, R-S1-4, 5.1)
# ---------------------------------------------------------------------------
def test_handoff_payload_carries_prompt_plus_sampled_token():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64, block_size=16)
    request = create_requests(
        num_requests=1, num_tokens=31, max_tokens=8, req_ids=["h"]
    )[0]
    scheduler.add_request(request)

    # Prefix caching off: the whole 31-token prompt is scheduled in one chunk.
    p_output, _ = run_step(scheduler, PREFILL, sampled={"h": [7]})
    assert p_output.num_scheduled_tokens == {"h": 31}
    p_block_ids = p_output.scheduled_new_reqs[0].block_ids
    assert request.awaiting_handoff
    assert request.lane == int(DECODE)

    d_output = scheduler.schedule(lane=DECODE)
    assert d_output.num_scheduled_tokens == {"h": 1}
    assert len(d_output.scheduled_new_reqs) == 1
    new_req = d_output.scheduled_new_reqs[0]
    assert new_req.req_id == "h"
    assert new_req.prefill_token_ids == list(request.prompt_token_ids) + [7]
    assert new_req.num_computed_tokens == 31
    # The worker must classify the row as a decode over the computed prompt,
    # not as a 1-token prefill continuation: prefill_len is the prompt length,
    # while prefill_token_ids also carries the sampled token.
    assert new_req.prefill_len == 31
    assert new_req.block_ids == p_block_ids
    assert not request.awaiting_handoff
    assert request.lane_resident == 1 << int(DECODE)
    # The P worker slot release rides the next P batch.
    assert "h" in scheduler.lane_finished_req_ids[int(PREFILL)]
    # F1: the D batch fences on the processed P sequence of the final chunk.
    assert d_output.lane_fence[int(PREFILL)] == scheduler.lane_seq[int(PREFILL)]


def test_admission_payload_has_no_prefill_len_override():
    """A normal admission computes its whole prefill_token_ids.

    Only the handoff sets prefill_len; every other new-request payload leaves
    it None so the worker keeps the baseline prefill_len == len(tokens).
    """
    scheduler = make_lane_scheduler(max_num_batched_tokens=64, block_size=16)
    request = create_requests(
        num_requests=1, num_tokens=31, max_tokens=8, req_ids=["a"]
    )[0]
    scheduler.add_request(request)

    # Schedule only: the payload aliases the request's token list, which
    # update_from_output would extend with the sampled token.
    p_output = scheduler.schedule(lane=PREFILL)
    new_req = p_output.scheduled_new_reqs[0]
    assert new_req.req_id == "a"
    assert new_req.num_computed_tokens == 0
    assert new_req.prefill_len is None
    assert new_req.prefill_token_ids == list(request.prompt_token_ids)


def test_handoff_row_is_uniform_decode_shape():
    scheduler = make_lane_scheduler(
        max_num_batched_tokens=64, num_speculative_tokens=2, block_size=16
    )
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=16, req_ids=["s"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"s": [11]})

    d_output = scheduler.schedule(lane=DECODE)
    assert d_output.num_scheduled_tokens == {"s": 3}
    assert d_output.scheduled_spec_decode_tokens == {"s": [-1, -1]}
    assert d_output.scheduled_new_reqs[0].num_computed_tokens == 16
    assert d_output.scheduled_new_reqs[0].prefill_token_ids[-1] == 11


def test_handoff_releases_prefill_slot_once():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["r"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"r": [3]})
    assert "r" in scheduler.lane_finished_req_ids[int(PREFILL)]

    # The next (empty) P batch delivers the release and clears it.
    p_idle, _ = run_step(scheduler, PREFILL)
    assert p_idle.total_num_scheduled_tokens == 0
    assert "r" in p_idle.finished_req_ids
    assert "r" not in scheduler.lane_finished_req_ids[int(PREFILL)]


# ---------------------------------------------------------------------------
# Per-lane sequence numbers and deferred frees (R-S1-6, 5.2)
# ---------------------------------------------------------------------------
def test_lane_sequences_and_fences():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["q"]
    )[0]
    scheduler.add_request(request)

    p1, _ = run_step(scheduler, PREFILL, sampled={"q": [4]})
    assert p1.lane == int(PREFILL)
    assert p1.lane_seq == 1
    assert p1.lane_fence == (0, 0)
    assert scheduler.processed_seq == [0, 1]

    d1 = scheduler.schedule(lane=DECODE)
    assert d1.lane_seq == 1
    assert d1.lane_fence == (0, 1)
    assert scheduler.lane_seq == [1, 1]
    assert scheduler.processed_seq == [0, 1]

    # Empty batches do not advance the counter.
    fresh = make_lane_scheduler()
    empty = fresh.schedule(lane=DECODE)
    assert empty.total_num_scheduled_tokens == 0
    assert empty.lane_seq == 0
    assert fresh.lane_seq == [0, 0]


def test_deferred_free_waits_for_both_lane_fences():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["f"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"f": [5]})

    request = scheduler.requests["f"]
    blocks = scheduler.kv_cache_manager.pop_blocks_for_free(request)
    assert blocks

    scheduler.lane_seq = [5, 3]
    scheduler.processed_seq = [4, 2]
    scheduler.deferred_frees.append(((5, 3), blocks))
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    scheduler._drain_deferred_frees()
    assert len(scheduler.deferred_frees) == 1
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks

    # One lane passing is not enough.
    scheduler.processed_seq = [4, 3]
    scheduler._drain_deferred_frees()
    assert len(scheduler.deferred_frees) == 1

    scheduler.processed_seq = [5, 3]
    scheduler._drain_deferred_frees()
    assert not scheduler.deferred_frees
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() > free_blocks


# ---------------------------------------------------------------------------
# Abort / finish / preempt matrix (5.3, R-S1-5, R-S1-8)
# ---------------------------------------------------------------------------
def test_abort_waiting_request_is_engine_only():
    scheduler = make_lane_scheduler()
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["w"]
    )[0]
    scheduler.add_request(request)
    scheduler.finish_requests("w", RequestStatus.FINISHED_ABORTED)
    assert "w" not in scheduler.requests
    assert not any(scheduler.lane_finished_req_ids.values())
    assert not scheduler.lane_preempted_req_ids[int(DECODE)]


def test_abort_prefill_in_flight_releases_prefill_only():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=64, max_tokens=8, req_ids=["p"]
    )[0]
    scheduler.add_request(request)
    p_output = scheduler.schedule(lane=PREFILL)
    assert p_output.num_scheduled_tokens == {"p": 64}
    assert request.is_prefill_chunk is False

    scheduler.finish_requests("p", RequestStatus.FINISHED_ABORTED)
    assert "p" in scheduler.lane_finished_req_ids[int(PREFILL)]
    assert "p" not in scheduler.lane_finished_req_ids[int(DECODE)]


def test_abort_prefill_idle_releases_prefill_only():
    scheduler = make_lane_scheduler(
        max_num_batched_tokens=32, block_size=16, max_model_len=256
    )
    request = create_requests(
        num_requests=1, num_tokens=64, max_tokens=8, req_ids=["pi"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL)  # 32 of 64 prompt tokens, processed
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 32

    # No batch in flight: preemptible (5.3) and abortable.
    other = create_requests(
        num_requests=1, num_tokens=8, max_tokens=8, req_ids=["other"]
    )[0]
    assert scheduler._lane_pick_victim(other, {}) is request

    scheduler.finish_requests("pi", RequestStatus.FINISHED_ABORTED)
    assert "pi" in scheduler.lane_finished_req_ids[int(PREFILL)]
    assert "pi" not in scheduler.lane_finished_req_ids[int(DECODE)]
    # Its blocks were not written by an in-flight batch: freed immediately.
    assert not scheduler.deferred_frees


def test_abort_in_handoff_window_releases_prefill_only():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["h"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"h": [6]})
    assert request.awaiting_handoff

    scheduler.finish_requests("h", RequestStatus.FINISHED_ABORTED)
    assert "h" in scheduler.lane_finished_req_ids[int(PREFILL)]
    assert "h" not in scheduler.lane_finished_req_ids[int(DECODE)]
    assert request.lane_resident == 0


def test_abort_decode_in_flight_defers_block_free():
    scheduler = make_lane_scheduler(
        max_num_batched_tokens=32, block_size=16, max_model_len=256
    )
    d_req = create_requests(num_requests=1, num_tokens=16, max_tokens=8, req_ids=["d"])[
        0
    ]
    # "p2" keeps prefilling across several P batches.
    p_req = create_requests(
        num_requests=1, num_tokens=64, max_tokens=8, req_ids=["p2"]
    )[0]
    for request in (d_req, p_req):
        scheduler.add_request(request)

    run_step(scheduler, PREFILL, sampled={"d": [1]})
    assert d_req.awaiting_handoff

    # Prefill batch for "p2" stays in flight, D batch carries the handoff.
    p_inflight = scheduler.schedule(lane=PREFILL)
    assert "p2" in p_inflight.num_scheduled_tokens
    d_inflight = scheduler.schedule(lane=DECODE)
    assert d_inflight.num_scheduled_tokens == {"d": 1}
    scheduler.finish_requests("d", RequestStatus.FINISHED_ABORTED)

    assert "d" in scheduler.lane_finished_req_ids[int(DECODE)]
    assert scheduler.deferred_frees, "a block freed with an in-flight batch"
    fence = scheduler.deferred_frees[0][0]
    assert fence[0] > scheduler.processed_seq[0]

    # P completes first: the D fence is still open.
    run_step(scheduler, PREFILL)
    assert scheduler.deferred_frees
    # Complete the D batch: the fence closes and the blocks return.
    scheduler.update_from_output(
        d_inflight,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert not scheduler.deferred_frees


def test_finish_in_decode_lane_releases_decode_slot():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=2, req_ids=["x"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"x": [1]})
    assert request.awaiting_handoff

    # D row samples the second (last) token: the request finishes in D.
    run_step(scheduler, DECODE, sampled={"x": [2]})
    assert "x" not in scheduler.requests
    assert scheduler.lane_finished_req_ids[int(DECODE)] == {"x"}


def test_preempt_decode_idle_requeues_to_prefill():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["v"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"v": [1]})
    run_step(scheduler, DECODE)  # processed: no in-flight batch
    assert request.lane == int(DECODE)

    # Never preempt the request being scheduled.
    assert scheduler._lane_pick_victim(request, {}) is None

    scheduler._preempt_request(request, time.monotonic(), lane=int(PREFILL))
    assert request.status == RequestStatus.PREEMPTED
    assert request.lane == int(PREFILL)
    assert request.lane_resident == 0
    assert "v" in scheduler.lane_finished_req_ids[int(DECODE)]
    assert "v" in scheduler.lane_preempted_req_ids[int(PREFILL)]
    assert scheduler.waiting[0] is request


def test_lane_allocate_removes_victim_before_preempt():
    """A preempted request must leave self.running before _preempt_request.

    The base scheduler pops it first; leaving it in re-adds it on readmission
    and leaks a duplicate entry (review finding 3).
    """
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    victim = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["victim"]
    )[0]
    scheduler.add_request(victim)
    run_step(scheduler, PREFILL, sampled={"victim": [1]})
    run_step(scheduler, DECODE)  # processed: no in-flight batch
    assert victim in scheduler.running

    request = create_requests(
        num_requests=1, num_tokens=8, max_tokens=8, req_ids=["new"]
    )[0]
    scheduler.add_request(request)
    preempted: list = []
    with (
        patch.object(
            scheduler.kv_cache_manager,
            "allocate_slots",
            side_effect=[None, None],
        ),
        patch.object(
            scheduler,
            "_lane_pick_victim",
            side_effect=[victim, None],
        ),
    ):
        blocks = scheduler._lane_allocate(
            request=request,
            num_new_tokens=1,
            lane_id=int(DECODE),
            delay_cache_blocks=True,
            num_lookahead_tokens=0,
            scheduled_timestamp=time.monotonic(),
            preempted_reqs=preempted,
            scheduled_ids={},
        )
    assert blocks is None
    assert victim not in scheduler.running
    assert preempted == [victim]
    assert victim.status == RequestStatus.PREEMPTED


def test_handoff_window_request_is_not_preemptible():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["hw"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"hw": [1]})
    assert request.awaiting_handoff

    other = create_requests(
        num_requests=1, num_tokens=8, max_tokens=8, req_ids=["other"]
    )[0]
    assert scheduler._lane_pick_victim(other, {}) is None


def test_preempt_refused_for_decode_in_flight():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["busy"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"busy": [1]})
    scheduler.schedule(lane=DECODE)  # in-flight, not processed

    other = create_requests(
        num_requests=1, num_tokens=8, max_tokens=8, req_ids=["other"]
    )[0]
    assert scheduler._lane_pick_victim(other, {}) is None


# ---------------------------------------------------------------------------
# Delayed prefix-cache registration (R-S1-7, H4)
# ---------------------------------------------------------------------------
def test_delayed_hashing_for_decode_allocations():
    scheduler = make_lane_scheduler(
        max_num_batched_tokens=64,
        num_speculative_tokens=2,
        enable_prefix_caching=True,
        block_size=16,
    )
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["c"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"c": [9]})

    cache_calls: list[int] = []
    coordinator = scheduler.kv_cache_manager.coordinator
    real_cache_blocks = coordinator.cache_blocks

    def spy(req, num_computed_tokens):
        cache_calls.append(num_computed_tokens)
        return real_cache_blocks(req, num_computed_tokens)

    with patch.object(coordinator, "cache_blocks", side_effect=spy):
        d_output = scheduler.schedule(lane=DECODE)
        assert cache_calls == [], "D allocations must not hash at schedule time"
        scheduler.update_from_output(
            d_output,
            ModelRunnerOutput(
                req_ids=list(d_output.num_scheduled_tokens),
                req_id_to_index={
                    req_id: index
                    for index, req_id in enumerate(d_output.num_scheduled_tokens)
                },
                sampled_token_ids=[[5] for _ in d_output.num_scheduled_tokens],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    assert cache_calls, "a D output must register its blocks at completion"

    # The prefill lane keeps schedule-time registration.
    cache_calls.clear()
    newcomer = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["n"]
    )[0]
    scheduler.add_request(newcomer)
    with patch.object(coordinator, "cache_blocks", side_effect=spy):
        scheduler.schedule(lane=PREFILL)
    assert cache_calls


# ---------------------------------------------------------------------------
# Pause and lane statistics (R-S1-11, R-S1-12)
# ---------------------------------------------------------------------------
def test_pause_zeroes_both_lane_budgets():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["pz"]
    )[0]
    scheduler.add_request(request)
    scheduler.set_pause_state(PauseState.PAUSED_ALL)

    assert scheduler.schedule(lane=PREFILL).num_scheduled_tokens == {}
    assert scheduler.schedule(lane=DECODE).num_scheduled_tokens == {}

    scheduler.set_pause_state(PauseState.UNPAUSED)
    assert scheduler.schedule(lane=PREFILL).num_scheduled_tokens == {"pz": 16}


def test_lane_stats_reports_per_lane_counters():
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["ls"]
    )[0]
    scheduler.add_request(request)
    run_step(scheduler, PREFILL, sampled={"ls": [1]})
    run_step(scheduler, DECODE)

    stats = scheduler.lane_stats()
    assert stats["prefill"]["batches"] == 1
    assert stats["prefill"]["tokens"] == 16
    assert stats["prefill"]["handoffs"] == 1
    assert stats["decode"]["batches"] == 1
    assert stats["decode"]["tokens"] == 1
    assert stats["decode"]["handoffs"] == 1
    assert stats["decode"]["lane_seq"] == scheduler.lane_seq[int(DECODE)]
    assert stats["decode"]["processed_seq"] == 1
    assert stats["decode"]["deferred_free_depth"] == 0


# ---------------------------------------------------------------------------
# Prefill fairness (R-S1-9)
# ---------------------------------------------------------------------------
def test_prefill_fairness_carries_every_unfinished_request(monkeypatch):
    # The 16k prompt exceeds opt-125m's declared context; the scheduler only
    # needs the configured cap.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    scheduler = make_lane_scheduler(
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        max_model_len=20000,
        block_size=16,
        enable_prefix_caching=False,
    )
    lengths = {"long": 16384, "mid": 1024, "short": 100}
    requests = []
    for req_id, num_tokens in lengths.items():
        requests.extend(
            create_requests(
                num_requests=1,
                num_tokens=num_tokens,
                max_tokens=8,
                req_ids=[req_id],
            )
        )
    for request in requests:
        scheduler.add_request(request)

    prefilled: set[str] = set()
    for step in range(64):
        deadline = {
            req_id
            for req_id, request in scheduler.requests.items()
            if request.num_computed_tokens < request.num_prompt_tokens
        }
        if not deadline:
            break
        output, _ = run_step(scheduler, PREFILL, default_token=step + 1)
        scheduled = set(output.num_scheduled_tokens)
        # Every request that still needs prompt tokens is served this batch.
        assert deadline <= scheduled, (
            f"step {step}: starved {deadline - scheduled}, "
            f"batches={output.num_scheduled_tokens}"
        )
        prefilled |= scheduled
    else:
        raise AssertionError("prefill did not finish")

    assert prefilled == set(lengths)
    # The short prompt was carried alongside the long one, not after it.
    assert scheduler.requests["short"].num_computed_tokens == 100


# ---------------------------------------------------------------------------
# Structured output (R-S1-10)
# ---------------------------------------------------------------------------
def _stub_grammar_bitmask(monkeypatch, scheduler):
    """Record every bitmask request and return a small fake mask."""
    calls: list[tuple[list[str], dict]] = []

    def fake_bitmask(requests, req_ids, spec_tokens):
        calls.append((list(req_ids), dict(spec_tokens)))
        return torch.zeros((len(req_ids), 4), dtype=torch.int32).numpy()

    monkeypatch.setattr(
        scheduler.structured_output_manager, "grammar_bitmask", fake_bitmask
    )
    # The fake request carries no real grammar; accept every sampled token.
    monkeypatch.setattr(
        scheduler.structured_output_manager,
        "accept_tokens",
        lambda request, new_token_ids: True,
    )
    return calls


def test_structured_output_is_admitted_and_constrained(monkeypatch):
    """A grammar request is admitted; its sampling rows are constrained."""
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["so"]
    )[0]
    # The scheduler only cares that a structured-output request is present.
    request.structured_output_request = object()
    scheduler.add_request(request)
    calls = _stub_grammar_bitmask(monkeypatch, scheduler)

    # The whole prompt fits one chunk, so this batch samples the first token.
    output, _ = run_step(scheduler, PREFILL)
    assert "so" in output.num_scheduled_tokens
    assert not scheduler.requests["so"].is_prefill_chunk
    assert output.has_structured_output_requests
    grammar = scheduler.get_grammar_bitmask(output)
    assert grammar is not None
    assert grammar.structured_output_request_ids == ["so"]
    # No spec tokens are fed to the FSM: the bonus row must stay constrained.
    assert calls == [(["so"], {})]
    assert grammar.num_acceptable_drafts == [0]


def test_structured_output_skips_mid_prefill_chunks(monkeypatch):
    """A chunk that samples no token is not constrained."""
    scheduler = make_lane_scheduler(max_num_batched_tokens=8)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["so"]
    )[0]
    request.structured_output_request = object()
    scheduler.add_request(request)
    calls = _stub_grammar_bitmask(monkeypatch, scheduler)

    output, _ = run_step(scheduler, PREFILL)
    assert "so" in output.num_scheduled_tokens
    assert scheduler.requests["so"].is_prefill_chunk
    assert not output.has_structured_output_requests
    assert scheduler.get_grammar_bitmask(output) is None
    assert calls == []


def test_structured_output_constrains_every_decode_step(monkeypatch):
    """After the handoff the decode lane constrains each step on its own."""
    scheduler = make_lane_scheduler(max_num_batched_tokens=64)
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["so"]
    )[0]
    request.structured_output_request = object()
    scheduler.add_request(request)
    calls = _stub_grammar_bitmask(monkeypatch, scheduler)

    prefill_to_handoff(scheduler, "so")
    assert scheduler.requests["so"].lane == int(DECODE)
    calls.clear()

    for _ in range(3):
        output, _ = run_step(scheduler, DECODE, default_token=7)
        assert output.has_structured_output_requests
        grammar = scheduler.get_grammar_bitmask(output)
        assert grammar is not None
        assert grammar.structured_output_request_ids == ["so"]

    assert calls == [(["so"], {})] * 3


# ---------------------------------------------------------------------------
# Flag-off equality (I9, R-S1-13)
# ---------------------------------------------------------------------------
def _trace_scheduler(scheduler) -> list[dict]:
    """Replay a fixed mixed workload and record the SchedulerOutput stream."""
    requests = create_requests(
        num_requests=3, num_tokens=40, max_tokens=6, req_ids=["0", "1", "2"]
    )
    for request in requests:
        scheduler.add_request(request)
    trace = []
    for step in range(10):
        lane = PREFILL if step < 5 else DECODE
        output = scheduler.schedule(lane=lane)
        trace.append(
            {
                "lane": output.lane,
                "lane_seq": output.lane_seq,
                "partitioned": output.partitioned,
                "lane_fence": output.lane_fence,
                "new": [
                    (
                        req.req_id,
                        req.block_ids,
                        req.num_computed_tokens,
                        req.prefill_token_ids,
                    )
                    for req in output.scheduled_new_reqs
                ],
                "cached": (
                    output.scheduled_cached_reqs.req_ids,
                    output.scheduled_cached_reqs.resumed_req_ids,
                    output.scheduled_cached_reqs.new_block_ids,
                    output.scheduled_cached_reqs.num_computed_tokens,
                    output.scheduled_cached_reqs.num_output_tokens,
                ),
                "tokens": dict(output.num_scheduled_tokens),
                "total": output.total_num_scheduled_tokens,
                "spec": dict(output.scheduled_spec_decode_tokens),
                "finished": set(output.finished_req_ids),
                "preempted": set(output.preempted_req_ids or ()),
                "skip_drafts": output.skip_drafts,
            }
        )
        req_ids = list(output.num_scheduled_tokens)
        sampled = {
            req_id: [step + 1]
            for req_id in req_ids
            if not scheduler.requests[req_id].is_prefill_chunk
        }
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
                sampled_token_ids=[sampled.get(req_id, []) for req_id in req_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    return trace


def test_flag_off_stream_is_identical_to_baseline():
    kwargs = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 128,
        "block_size": 16,
        "max_model_len": 256,
        "enable_prefix_caching": True,
        "async_scheduling": True,
        "use_v2_model_runner": True,
    }
    baseline = create_scheduler(**kwargs)
    assert type(baseline) is AsyncScheduler
    lane_off = create_scheduler(scheduler_cls=TwoLaneScheduler, **kwargs)
    assert type(lane_off) is TwoLaneScheduler
    assert lane_off.lane_mode is False

    baseline_trace = _trace_scheduler(baseline)
    lane_off_trace = _trace_scheduler(lane_off)

    assert len(baseline_trace) == len(lane_off_trace)
    for base_step, lane_step in zip(baseline_trace, lane_off_trace):
        # The lane fields keep their frozen defaults when the flag is off.
        assert lane_step["lane"] == 0
        assert lane_step["lane_seq"] == 0
        assert lane_step["partitioned"] is False
        assert lane_step["lane_fence"] == (0, 0)
        for field in (
            "new",
            "cached",
            "tokens",
            "total",
            "spec",
            "finished",
            "preempted",
            "skip_drafts",
        ):
            assert base_step[field] == lane_step[field], field


def test_flag_off_two_lane_scheduler_matches_plain_scheduler_directly():
    """`schedule(lane=...)` is additive: the default lane keeps the baseline."""
    scheduler = make_lane_scheduler(two_lane=False, max_num_batched_tokens=64)
    assert scheduler.lane_mode is False
    request = create_requests(
        num_requests=1, num_tokens=16, max_tokens=8, req_ids=["z"]
    )[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.lane == 0
    assert output.lane_seq == 0
    assert output.num_scheduled_tokens == {"z": 16}
    # No lane bookkeeping advanced.
    assert scheduler.lane_seq == [0, 0]
    assert not any(scheduler.lane_finished_req_ids.values())


# ---------------------------------------------------------------------------
# KV connector (per-lane metadata, external hits adopted by the prefill lane)
# ---------------------------------------------------------------------------
class _FakeKVConnector:
    """Minimal scheduler-side connector: one external hit, per-lane meta."""

    supports_divergent_local_hybrid_hits = False
    requires_kv_delivery = False

    def __init__(self, ext_tokens: int = 0) -> None:
        self.ext_tokens = ext_tokens
        self.matched_queries = 0
        self.updated: list[tuple[str, int]] = []
        self.meta_by_lane: list[int] = []

    def on_new_request(self, request) -> None:
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.matched_queries += 1
        return self.ext_tokens, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.updated.append((request.request_id, num_external_tokens))

    def get_loaded_kv_cache_group_ids(self, request):
        return ()

    def get_external_cache_hit_sources(self, request, num_hits):
        return None

    def build_connector_meta(self, scheduler_output):
        self.meta_by_lane.append(scheduler_output.lane)
        return {"lane": scheduler_output.lane}


def test_connector_external_hit_is_adopted_by_prefill():
    scheduler = make_lane_scheduler(max_num_batched_tokens=256, block_size=16)
    connector = _FakeKVConnector(ext_tokens=32)
    scheduler.connector = connector
    request = create_requests(
        num_requests=1, num_tokens=40, max_tokens=8, req_ids=["c"]
    )[0]
    scheduler.add_request(request)

    # D never admits, so it never asks the connector for a hit.
    scheduler.schedule(lane=DECODE)
    assert connector.matched_queries == 0

    output = scheduler.schedule(lane=PREFILL)
    # 32 tokens come from the connector, the remaining 8 are computed here.
    assert output.num_scheduled_tokens == {"c": 8}
    assert request.num_computed_tokens == 40
    # A synchronous load: the worker starts it before this batch's forward.
    assert output.has_sync_kv_loads is True
    assert connector.updated == [("c", 32)]
    # Every lane's output carries its own connector metadata.
    assert output.kv_connector_metadata == {"lane": int(PREFILL)}
    assert connector.meta_by_lane[-1] == int(PREFILL)


def test_connector_hit_never_consumes_the_whole_prompt():
    scheduler = make_lane_scheduler(max_num_batched_tokens=256, block_size=16)
    connector = _FakeKVConnector(ext_tokens=40)
    scheduler.connector = connector
    request = create_requests(
        num_requests=1, num_tokens=40, max_tokens=8, req_ids=["c"]
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule(lane=PREFILL)
    # One token must remain for the lane to compute and sample.
    assert output.num_scheduled_tokens == {"c": 1}
    assert connector.updated == [("c", 39)]
