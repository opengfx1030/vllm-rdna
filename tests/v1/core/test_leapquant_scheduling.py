# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission and decode progress with LeapQuant's smaller hybrid-cache pages.

The decode-growth policy also runs for FP32 state with --decode-growth-admission;
the tests at the end of the file cover that gate."""

import pytest
import torch
from transformers import OPTConfig

from vllm.platforms.cpu import CpuPlatform
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def make_scheduler(tmp_path, monkeypatch, request):
    async_scheduling = request.param

    class SchedulerOnlyPlatform(CpuPlatform):
        @classmethod
        def check_and_update_config(cls, vllm_config):
            super().check_and_update_config(vllm_config)
            # No model executes here. Keep the GPU scheduler's concurrency
            # metadata instead of CPU's unconditional async=False override.
            vllm_config.scheduler_config.async_scheduling = async_scheduling

    monkeypatch.setattr("vllm.platforms.current_platform", SchedulerOnlyPlatform())
    OPTConfig(
        architectures=["OPTForCausalLM"], max_position_embeddings=262144
    ).save_pretrained(tmp_path)

    def make(
        *,
        hybrid=False,
        block_size=16,
        force_growth=True,
        fp32_state=False,
        **kwargs,
    ):
        class HybridScheduler(Scheduler):
            def __init__(self, vllm_config, kv_cache_config, **scheduler_kwargs):
                vllm_config.cache_config.mamba_cache_mode = "align"
                vllm_config.cache_config.mamba_block_size = block_size
                page_bytes = block_size * 1024
                state = MambaSpec(
                    block_size=block_size,
                    # FP32 variant: the model's own [V, K] = [128, 128] FP32
                    # state per value head instead of the LeapQuant blob.
                    shapes=((2560, 5), (12, 128, 128) if fp32_state else (12, 28768)),
                    dtypes=(
                        (torch.float16, torch.float32)
                        if fp32_state
                        else (torch.float16, torch.uint8)
                    ),
                    page_size_padded=page_bytes,
                    mamba_cache_mode="align",
                    num_speculative_blocks=2,
                )
                kv_cache_config.kv_cache_groups = [
                    KVCacheGroupSpec(
                        ["attn"],
                        FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=2,
                            head_size=128,
                            dtype=torch.float16,
                        ),
                    ),
                    *[KVCacheGroupSpec([f"gdn{i}"], state) for i in range(3)],
                    KVCacheGroupSpec(
                        ["ring"],
                        CircularBufferSpec(
                            block_size=8,
                            num_kv_heads=2,
                            head_size=128,
                            dtype=torch.float16,
                            page_size_padded=page_bytes,
                        ),
                    ),
                    KVCacheGroupSpec(["ple"], state),
                ]
                super().__init__(
                    vllm_config=vllm_config,
                    kv_cache_config=kv_cache_config,
                    **scheduler_kwargs,
                )

        class HybridAsyncScheduler(HybridScheduler, AsyncScheduler):
            """Use the same cache layout with real async placeholder accounting."""

        hybrid_cls = HybridAsyncScheduler if async_scheduling else HybridScheduler

        scheduler = create_scheduler(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            enable_prefix_caching=True,
            block_size=block_size,
            scheduler_cls=hybrid_cls if hybrid else None,
            async_scheduling=async_scheduling,
            speculative_method=(
                "ngram_gpu" if kwargs.get("num_speculative_tokens") else None
            ),
            **kwargs,
        )
        # Exercise the policy with real allocators, without requiring a GPU or
        # model weights. Production enables it for RDNA LeapQuant or with
        # --decode-growth-admission; force_growth=False keeps the real gate.
        if force_growth:
            scheduler._decode_growth_admission = True
        if hybrid:
            scheduler.need_mamba_block_aligned_split = True
            scheduler.mamba_state_block_size = block_size
        return scheduler

    return make


def _update(scheduler, output, samples):
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[samples[rid] for rid in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _drain(scheduler, requests, *, strict=True):
    """Run `requests` (at most 8 in flight) to completion on synthetic outputs.

    Returns the prompt rows processed, repeats included. With `strict`, a step
    that schedules nothing, or a ready decoder skipping a step, fails."""
    pending = iter(requests)
    active = set()
    for _ in range(8):
        request = next(pending)
        scheduler.add_request(request)
        active.add(request.request_id)

    last_sample: dict[str, int] = {}
    prefill_rows = 0
    for step in range(20000):
        output = scheduler.schedule()
        if strict:
            assert output.num_scheduled_tokens, "admission must not deadlock"
        samples: dict[str, list[int]] = {}
        for rid, num_tokens in output.num_scheduled_tokens.items():
            request = scheduler.requests[rid]
            start = request.num_computed_tokens - num_tokens
            prefill_rows += max(0, min(num_tokens, request.num_prompt_tokens - start))
            if request.num_computed_tokens < request.num_tokens:
                samples[rid] = []
                continue
            count = 2 if rid in output.scheduled_spec_decode_tokens else 1
            samples[rid] = [100] * count
            if strict and rid in last_sample:
                assert step - last_sample[rid] == 1, "a ready decoder lost its step"
            last_sample[rid] = step
        _update(scheduler, output, samples)
        live = [rid for rid in active if rid in scheduler.requests]
        scheduler.update_draft_token_ids(
            DraftTokenIds(live, [[101, 102] for _ in live])
        )
        finished = active - set(scheduler.requests)
        active -= finished
        for _ in finished:
            request = next(pending, None)
            if request is not None:
                scheduler.add_request(request)
                active.add(request.request_id)
        if not active:
            return prefill_rows
    pytest.fail("bounded requests did not drain")


@pytest.mark.parametrize(("block_size", "num_blocks"), [(512, 556), (1024, 278)])
def test_hybrid_decode_growth_does_not_preempt_or_replay(
    make_scheduler, block_size, num_blocks
):
    scheduler = make_scheduler(
        hybrid=True,
        block_size=block_size,
        num_blocks=num_blocks,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        max_model_len=262144,
        num_speculative_tokens=2,
    )
    requests = create_requests(
        12, num_tokens=32768, max_tokens=2048, ignore_eos=True, block_size=block_size
    )
    prefill_rows = _drain(scheduler, requests)

    assert sum(request.num_preemptions for request in requests) == 0
    assert prefill_rows == 12 * 32768
    assert all(request.num_output_tokens == 2048 for request in requests)


def test_admission_includes_recurrent_boundary_peak(make_scheduler):
    scheduler = make_scheduler(
        hybrid=True,
        block_size=512,
        num_blocks=39,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        max_model_len=4096,
        num_speculative_tokens=2,
    )
    requests = create_requests(
        2, num_tokens=512, max_tokens=1024, ignore_eos=True, block_size=512
    )
    for request in requests:
        scheduler.add_request(request)
    # Initial state + two draft slots fits for both requests, but each also
    # needs a spare state at a block boundary. Their combined peaks do not fit.
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {requests[0].request_id}


@pytest.mark.parametrize("finish_reason", ["stop", "abort"])
def test_output_reservation_released_on_early_finish(make_scheduler, finish_reason):
    scheduler = make_scheduler(
        num_blocks=15, max_num_seqs=2, max_num_batched_tokens=64, max_model_len=256
    )
    requests = create_requests(
        2,
        num_tokens=32,
        max_tokens=128,
        stop_token_ids=[100] if finish_reason == "stop" else None,
    )
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {requests[0].request_id}
    _update(scheduler, output, {requests[0].request_id: [100]})
    if finish_reason == "abort":
        scheduler.finish_requests(
            requests[0].request_id, RequestStatus.FINISHED_ABORTED
        )
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {requests[1].request_id}
    assert requests[1].num_preemptions == 0


def test_ready_decode_gets_full_speculative_width_before_prefill(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=100,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        max_model_len=256,
        long_prefill_token_threshold=8,
        num_speculative_tokens=2,
    )
    (prefill,) = create_requests(1, num_tokens=64, req_ids=["prefill"])
    (decoder,) = create_requests(1, num_tokens=1, req_ids=["decoder"])
    scheduler.add_request(prefill)
    scheduler.add_request(decoder)
    output = scheduler.schedule()
    _update(scheduler, output, {"prefill": [], "decoder": [100]})
    scheduler.update_draft_token_ids(DraftTokenIds(["decoder"], [[101, 102]]))
    scheduler.scheduler_config.long_prefill_token_threshold = 0

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["decoder"] == 3
    assert output.scheduled_spec_decode_tokens["decoder"] == [101, 102]
    assert output.num_scheduled_tokens["prefill"] > 0
    _update(scheduler, output, {"prefill": [], "decoder": [100, 100]})
    assert decoder.num_output_tokens == 3


def test_admission_growth_clamps_to_context_limit(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=17, max_num_seqs=1, max_num_batched_tokens=32, max_model_len=256
    )
    (request,) = create_requests(1, num_tokens=32, max_tokens=1024)
    scheduler.add_request(request)
    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 32}


def test_admission_shares_cached_prompt_pages(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=13, max_num_seqs=3, max_num_batched_tokens=256, max_model_len=256
    )
    (warm,) = create_requests(1, num_tokens=64, stop_token_ids=[100], req_ids=["warm"])
    scheduler.add_request(warm)
    output = scheduler.schedule()
    _update(scheduler, output, {"warm": [100]})
    requests = create_requests(3, num_tokens=64, same_prompt=True)
    for request in requests:
        scheduler.add_request(request)
    # Three separate 80-token allocations do not fit. Shared cached prefixes
    # do, and must not be charged again as future growth.
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {r.request_id for r in requests}


def test_final_verify_reserves_rows_beyond_output_limit(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=6,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        max_model_len=256,
        num_speculative_tokens=2,
    )
    requests = create_requests(2, num_tokens=30, max_tokens=2, ignore_eos=True)
    for request in requests:
        scheduler.add_request(request)
    # Each final three-row verify crosses 32, even though only one of its
    # samples is needed. Five usable pages cannot promise both requests.
    output = scheduler.schedule()
    first_id = requests[0].request_id
    assert set(output.num_scheduled_tokens) == {first_id}
    _update(scheduler, output, {first_id: [100]})
    scheduler.update_draft_token_ids(DraftTokenIds([first_id], [[101, 102]]))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[first_id] == 3
    _update(scheduler, output, {first_id: [100, 100]})
    assert requests[0].num_output_tokens == 2
    assert requests[0].num_preemptions == 0
    assert requests[1].request_id in scheduler.schedule().num_scheduled_tokens


def test_async_inflight_verifies_keep_their_growth_promise(make_scheduler):
    scheduler = make_scheduler(
        num_blocks=6,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        max_model_len=256,
        num_speculative_tokens=2,
    )
    if not scheduler.scheduler_config.async_scheduling:
        pytest.skip("requires two concurrently scheduled batches")
    requests = create_requests(2, num_tokens=27, max_tokens=3, ignore_eos=True)
    for request in requests:
        scheduler.add_request(request)
    first_id = requests[0].request_id
    prefill = scheduler.schedule()
    # A single final verify fits in two pages, but two unresolved verifies
    # can reach row 33. Each request must be promised three pages, not two.
    assert set(prefill.num_scheduled_tokens) == {first_id}
    verify1 = scheduler.schedule()
    _update(scheduler, prefill, {first_id: [100]})
    verify2 = scheduler.schedule()
    assert verify2.num_scheduled_tokens[first_id] == 3
    assert requests[0].num_computed_tokens == 33
    _update(scheduler, verify1, {first_id: [100]})
    _update(scheduler, verify2, {first_id: [100]})
    assert requests[0].num_output_tokens == 3
    assert requests[0].num_preemptions == 0
    assert requests[1].request_id in scheduler.schedule().num_scheduled_tokens


def test_concurrent_prefill_chunks_reserve_live_recurrent_states(
    make_scheduler, monkeypatch
):
    from vllm.model_executor.layers.mamba.ops import leapquant_gdn

    monkeypatch.setattr(leapquant_gdn, "leapquant_gdn_enabled", lambda: True)
    scheduler = make_scheduler(
        hybrid=True,
        block_size=512,
        num_blocks=53,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        max_model_len=16384,
        num_speculative_tokens=2,
        long_prefill_token_threshold=1024,
    )
    requests = create_requests(
        2, num_tokens=4096, max_tokens=16, ignore_eos=True, block_size=512
    )
    for request in requests:
        scheduler.add_request(request)
    chunk1 = scheduler.schedule()
    if not scheduler.scheduler_config.async_scheduling:
        assert set(chunk1.num_scheduled_tokens) == {r.request_id for r in requests}
        return
    first_id = requests[0].request_id
    # Two sync requests fit, but two unresolved chunks retain one more state
    # per recurrent group. Promising both async requests would overcommit.
    assert set(chunk1.num_scheduled_tokens) == {first_id}
    chunk2 = scheduler.schedule()
    _update(scheduler, chunk1, {first_id: []})
    chunk3 = scheduler.schedule()
    state_manager = scheduler.kv_cache_manager.coordinator.single_type_managers[1]
    live_pages = sum(not b.is_null for b in state_manager.req_to_blocks[first_id])
    assert live_pages == 5
    spec = state_manager.kv_cache_spec
    assert spec.max_memory_usage_bytes(scheduler.vllm_config) >= (
        live_pages * spec.page_size_bytes
    )
    _update(scheduler, chunk2, {first_id: []})
    _update(scheduler, chunk3, {first_id: []})
    assert requests[0].num_preemptions == 0


# --- Decode-growth policy selected by --decode-growth-admission (FP32 state) ---

_FP32_POOL = dict(
    hybrid=True,
    fp32_state=True,
    force_growth=False,
    block_size=1024,
    num_blocks=278,
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    max_model_len=262144,
    num_speculative_tokens=2,
)


def test_fp32_hybrid_flag_on_does_not_preempt_or_replay(make_scheduler):
    scheduler = make_scheduler(decode_growth_admission=True, **_FP32_POOL)
    assert scheduler._decode_growth_admission
    requests = create_requests(
        12, num_tokens=32768, max_tokens=2048, ignore_eos=True, block_size=1024
    )
    prefill_rows = _drain(scheduler, requests)

    assert sum(request.num_preemptions for request in requests) == 0
    assert prefill_rows == 12 * 32768
    assert all(request.num_output_tokens == 2048 for request in requests)


def test_fp32_hybrid_both_off_keeps_preempting(make_scheduler):
    scheduler = make_scheduler(**_FP32_POOL)
    assert not scheduler._decode_growth_admission
    requests = create_requests(
        12, num_tokens=32768, max_tokens=2048, ignore_eos=True, block_size=1024
    )
    prefill_rows = _drain(scheduler, requests, strict=False)

    # The same pool and cap as the flag-on test above, upstream admission.
    assert sum(request.num_preemptions for request in requests) > 0
    assert prefill_rows > 12 * 32768


def test_flag_on_schedules_ready_decoders_before_running_prefills(make_scheduler):
    scheduler = make_scheduler(
        decode_growth_admission=True,
        **{**_FP32_POOL, "num_blocks": 300, "max_num_seqs": 2},
        long_prefill_token_threshold=1024,
    )
    (prefill,) = create_requests(
        1, num_tokens=8192, req_ids=["prefill"], block_size=1024
    )
    (decoder,) = create_requests(1, num_tokens=1, req_ids=["decoder"], block_size=1024)
    scheduler.add_request(prefill)
    scheduler.add_request(decoder)
    output = scheduler.schedule()
    _update(scheduler, output, {"prefill": [], "decoder": [100]})
    scheduler.update_draft_token_ids(DraftTokenIds(["decoder"], [[101, 102]]))
    scheduler.scheduler_config.long_prefill_token_threshold = 0

    output = scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == ["decoder", "prefill"]
    assert output.num_scheduled_tokens["decoder"] == 3
    assert output.scheduled_spec_decode_tokens["decoder"] == [101, 102]
    assert output.num_scheduled_tokens["prefill"] > 0


def test_both_off_leaves_running_order_and_lets_prefill_starve_decoder(
    make_scheduler,
):
    scheduler = make_scheduler(
        **{**_FP32_POOL, "num_blocks": 300, "max_num_seqs": 2},
        long_prefill_token_threshold=1024,
    )
    (prefill,) = create_requests(
        1, num_tokens=8192, req_ids=["prefill"], block_size=1024
    )
    (decoder,) = create_requests(1, num_tokens=1, req_ids=["decoder"], block_size=1024)
    scheduler.add_request(prefill)
    scheduler.add_request(decoder)
    output = scheduler.schedule()
    _update(scheduler, output, {"prefill": [], "decoder": [100]})
    scheduler.update_draft_token_ids(DraftTokenIds(["decoder"], [[101, 102]]))
    scheduler.scheduler_config.long_prefill_token_threshold = 0

    output = scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == ["prefill", "decoder"]
    assert output.num_scheduled_tokens["prefill"] == 2048
    assert "decoder" not in output.num_scheduled_tokens


@pytest.fixture
def gate_log(monkeypatch):
    from vllm.v1.core.sched import scheduler as scheduler_module

    messages: list[str] = []
    real_info = scheduler_module.logger.info

    def info(msg, *args, **kwargs):
        messages.append(msg % args if args else msg)
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(scheduler_module.logger, "info", info)
    return messages


def test_gate_flag_selects_policy_on_hybrid(make_scheduler, gate_log):
    scheduler = make_scheduler(decode_growth_admission=True, **_FP32_POOL)
    assert scheduler._decode_growth_admission
    assert any(
        "Decode-growth scheduling (selected by --decode-growth-admission)" in m
        for m in gate_log
    )


def test_gate_both_off_is_inactive_and_silent(make_scheduler, gate_log):
    scheduler = make_scheduler(**_FP32_POOL)
    assert not scheduler._decode_growth_admission
    assert not any("ecode-growth" in m for m in gate_log)


def test_gate_leapquant_selects_policy_without_flag(
    make_scheduler, gate_log, monkeypatch
):
    from vllm.model_executor.layers.mamba.ops import leapquant_gdn
    from vllm.v1.core.sched import scheduler as scheduler_module

    monkeypatch.setattr(leapquant_gdn, "leapquant_gdn_enabled", lambda: True)
    monkeypatch.setattr(scheduler_module, "leapquant_gdn_enabled", lambda: True)
    scheduler = make_scheduler(**_FP32_POOL)
    assert scheduler._decode_growth_admission
    assert any("(selected by LeapQuant)" in m for m in gate_log)


def test_gate_flag_without_mamba_layers_is_a_logged_no_op(make_scheduler, gate_log):
    scheduler = make_scheduler(
        decode_growth_admission=True, force_growth=False, num_blocks=100
    )
    assert not scheduler.kv_cache_config.has_mamba_layers
    assert not scheduler._decode_growth_admission
    assert any("has no effect" in m for m in gate_log)
    # Upstream ordering and admission: FCFS order is left alone.
    requests = create_requests(3, num_tokens=40, max_tokens=4)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == [r.request_id for r in requests]
    assert set(output.num_scheduled_tokens) == {r.request_id for r in requests}


def test_startup_peak_matches_admission_peak_with_flag(make_scheduler):
    scheduler = make_scheduler(decode_growth_admission=False, **_FP32_POOL)
    vllm_config = scheduler.vllm_config
    spec = next(
        g.kv_cache_spec
        for g in scheduler.kv_cache_config.kv_cache_groups
        if isinstance(g.kv_cache_spec, MambaSpec)
    )
    off = spec.max_memory_usage_bytes(vllm_config)
    vllm_config.scheduler_config.decode_growth_admission = True
    on = spec.max_memory_usage_bytes(vllm_config)
    assert on - off == spec.page_size_bytes * (vllm_config.max_concurrent_batches - 1)
