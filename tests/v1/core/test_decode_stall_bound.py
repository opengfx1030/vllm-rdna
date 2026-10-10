# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-stall bound (SchedulerConfig.decode_stall_budget_ms)."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.mixed_step import MixedStepController
from vllm.v1.outputs import ModelRunnerOutput


def _output(req_ids, sampled):
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _trained(
    budget_s=0.25, share=0.2, a=0.070, b=0.000423, max_tokens=2048, max_overhead=0.0
):
    c = MixedStepController(budget_s, share, max_tokens, max_overhead=max_overhead)
    for p in (128, 512, 1024, 2048):
        for _ in range(4):
            c.observe(p, 6, a + b * p)
    c._debt_s = 0.0
    return c


def test_cap_fits_budget():
    c = _trained()
    cap = c.prefill_cap()
    # (0.25 - 0.07) / 0.000423 = 425 -> 384 at 64-token granularity
    assert cap == 384
    assert 0.070 + 0.000423 * cap <= 0.25


def test_cap_tiles_recurrent_block():
    c = _trained(budget_s=0.29)
    # Raw cap 520: 1024 splits into two 512-token chunks, no short remainder.
    assert c.prefill_cap(block_size=1024) == 512
    c = _trained(budget_s=0.25)
    # Raw cap 425: three chunks per block -> 384 (384 + 384 + 256).
    assert c.prefill_cap(block_size=1024) == 384


def test_slow_model_gets_small_cap():
    # 27B-like: ~1 ms per prompt token.
    c = _trained(a=0.040, b=0.001)
    assert c.prefill_cap() == 192


def test_untrained_cap_is_conservative():
    c = MixedStepController(0.25, 0.2, 2048, initial_cap=256)
    assert c.prefill_cap() == 256


def test_efficiency_floor():
    # Large fixed step cost: the 250 ms budget alone would give 384 tokens;
    # the floor keeps the per-step cost within 25 % of the per-token work.
    c = _trained(max_overhead=0.25)
    # floor 0.070 / (0.25 * 0.000423) = 662 -> 704 (rounded up), and a whole
    # 1024-token recurrent block when chunks must tile it.
    assert c.prefill_cap() == 704
    assert c.prefill_cap(block_size=1024) == 1024
    # Small fixed cost (27B-like): the budget decides.
    c = _trained(a=0.040, b=0.001, max_overhead=0.25)
    assert c.prefill_cap() == 192


def test_decode_share_debt():
    c = _trained(share=0.2)
    assert not c.defer_prefill()
    c.observe(384, 6, 0.24)  # owes 0.25 * 0.24 = 60 ms of decode
    steps = 0
    while c.defer_prefill():
        c.observe(0, 6, 0.034)
        steps += 1
    assert steps == 2
    # Prefill-only steps (no decoders) owe nothing.
    c.observe(2048, 0, 0.94)
    assert not c.defer_prefill()


def test_single_bucket_can_shrink_below_initial_cap():
    # Flash-Next at the 512 initial cap: ~325 ms mixed steps vs a 250 ms
    # budget. The cap must shrink so a second bucket (and a fit) can exist.
    c = MixedStepController(0.25, 0.0, 2048, initial_cap=512)
    for _ in range(4):
        c.observe(512, 6, 0.325)
    assert c.prefill_cap() < 512
    assert c.prefill_cap(block_size=1024) <= 384


def test_prefill_only_steps_do_not_fit():
    # Flash-Next: prompt-logprob / deep-context prefill-only steps are slow
    # for their token count; fitting them pinned the cap at the full budget.
    c = _trained()
    cap = c.prefill_cap()
    for _ in range(8):
        c.observe(256, 0, 0.80)
        c.observe(2048, 0, 0.90)
    assert c.prefill_cap() == cap


def test_outlier_step_is_ignored():
    c = _trained()
    cap = c.prefill_cap()
    c.observe(512, 6, 25.0)  # a JIT compile, clipped to 1.5x the bucket mean
    c.observe(512, 6, 40.0)  # above the 30 s sanity bound: dropped
    assert abs(c.prefill_cap() - cap) <= 64


@pytest.fixture(autouse=True)
def _long_model_len(monkeypatch):
    # opt-125m derives max_model_len=2048; these tests need long prompts.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")


def _scheduler_with_bound(max_num_batched_tokens=2048):
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=32768,
        enable_chunked_prefill=True,
    )
    scheduler._mixed_step = _trained(max_tokens=max_num_batched_tokens)
    return scheduler


def test_prefill_chunk_bounded_while_decoding():
    scheduler = _scheduler_with_bound()
    decs = create_requests(
        num_requests=2, num_tokens=8, req_ids=["dec0", "dec1"], max_tokens=64
    )
    for r in decs:
        scheduler.add_request(r)
    out = scheduler.schedule()
    scheduler.update_from_output(out, _output(["dec0", "dec1"], [[0], [0]]))

    (long_req,) = create_requests(
        num_requests=1, num_tokens=8000, req_ids=["long"], max_tokens=4
    )
    scheduler.add_request(long_req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["dec0"] == 1
    assert out.num_scheduled_tokens["dec1"] == 1
    # Bounded to the controller's cap, not the 2048-token budget.
    assert out.num_scheduled_tokens["long"] == 384


def test_prefill_deferred_while_decode_share_owed():
    scheduler = _scheduler_with_bound()
    (dec,) = create_requests(num_requests=1, num_tokens=8, req_ids=["dec0"])
    scheduler.add_request(dec)
    out = scheduler.schedule()
    scheduler.update_from_output(out, _output(["dec0"], [[0]]))
    (long_req,) = create_requests(num_requests=1, num_tokens=8000, req_ids=["long"])
    scheduler.add_request(long_req)
    scheduler._mixed_step._debt_s = 1.0  # decode time owed
    out = scheduler.schedule()
    assert "long" not in out.num_scheduled_tokens
    assert out.num_scheduled_tokens["dec0"] == 1


def test_lone_prefill_keeps_full_budget():
    scheduler = _scheduler_with_bound()
    (long_req,) = create_requests(num_requests=1, num_tokens=8000, req_ids=["long"])
    scheduler.add_request(long_req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["long"] == 2048


def test_arrivals_admitted_next_to_running_prefill():
    scheduler = _scheduler_with_bound()
    (long_req,) = create_requests(num_requests=1, num_tokens=16000, req_ids=["long"])
    scheduler.add_request(long_req)
    out = scheduler.schedule()
    scheduler.update_from_output(out, _output(["long"], [[]]))
    arrivals = create_requests(
        num_requests=3, num_tokens=128, req_ids=["a0", "a1", "a2"]
    )
    for r in arrivals:
        scheduler.add_request(r)
    out = scheduler.schedule()
    # Room was left for the three short prompts in the same step.
    for rid in ("a0", "a1", "a2"):
        assert out.num_scheduled_tokens[rid] == 128
    assert out.num_scheduled_tokens["long"] == 2048 - 3 * 128


def test_off_by_default_in_config():
    # Off upstream; only the ROCm platform turns it on, and only on RDNA.
    from vllm.config import SchedulerConfig

    assert SchedulerConfig.default_factory().decode_stall_budget_ms is None
