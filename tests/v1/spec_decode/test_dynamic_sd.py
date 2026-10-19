# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule helpers."""

import logging

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert scheduler.dynamic_sd_lookup is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


def _feed(scheduler, out, sampled):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(out.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[sampled[r] for r in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_dynamic_sd_keeps_decode_rows_uniform_at_step_k():
    """Every decode row of a step carries that step's K drafts.

    The K of a step is the one the previous schedule() handed out. A new
    decode row (prefix-cache hit) is padded to it, drafts left over from a
    larger K are clipped to it, and K=0 pads nothing.
    """
    from vllm.v1.outputs import DraftTokenIds

    scheduler = create_scheduler(
        num_speculative_tokens=3,
        enable_prefix_caching=True,
        block_size=16,
        num_speculative_tokens_per_batch_size=[(1, 1, 3), (2, 2, 1), (3, 16, 0)],
    )
    r1, r2, r3, r4 = create_requests(
        num_requests=4, num_tokens=33, same_prompt=True, max_tokens=16
    )

    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_spec_tokens_to_schedule == 3
    _feed(scheduler, out, {r1.request_id: [100]})
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # K=3 step: r2 (whole prompt cached) is padded to 1 + 3.
    scheduler.add_request(r2)
    out = scheduler.schedule()
    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    assert out.num_scheduled_tokens[r2.request_id] == 4
    assert out.scheduled_spec_decode_tokens[r2.request_id] == [-1] * 3
    assert out.num_spec_tokens_to_schedule == 1
    _feed(scheduler, out, {r1.request_id: [101], r2.request_id: [102]})
    # The drafter may propose up to K_max; this step verifies K=1.
    scheduler.update_draft_token_ids(
        DraftTokenIds([r1.request_id, r2.request_id], [[4, 5, 6], [7, 8, 9]])
    )

    scheduler.add_request(r3)
    out = scheduler.schedule()
    assert out.scheduled_spec_decode_tokens[r1.request_id] == [4]
    assert out.scheduled_spec_decode_tokens[r2.request_id] == [7]
    assert out.num_scheduled_tokens[r3.request_id] == 2
    assert out.scheduled_spec_decode_tokens[r3.request_id] == [-1]
    assert set(out.num_scheduled_tokens.values()) == {2}
    assert out.num_spec_tokens_to_schedule == 0
    _feed(
        scheduler,
        out,
        {r1.request_id: [103], r2.request_id: [104], r3.request_id: [105]},
    )

    # K=0 step: nothing to pad, every row is one token.
    scheduler.add_request(r4)
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens.values()) == {1}
    assert not out.scheduled_spec_decode_tokens
