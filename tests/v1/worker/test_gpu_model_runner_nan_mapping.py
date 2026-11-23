# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for per-request NaN accounting with speculative logits.

The current runner samples spec-decode logits as one row per verification
token plus one bonus row per request. ``num_nans_in_logits`` is reported per
request, so every row must be summed into its request instead of being dropped
by a per-request zip over the row counts.
"""

import pytest
import torch

from vllm.v1.worker.gpu.async_utils import sum_nans_per_request
from vllm.v1.worker.utils import raise_if_nan_logits


def _row_mapping(draft_lens: list[int]) -> torch.Tensor:
    """Logit-row -> request index, matching the runner's expanded mapping."""
    repeats = torch.tensor([n + 1 for n in draft_lens], dtype=torch.long)
    return torch.arange(len(draft_lens), dtype=torch.int32).repeat_interleave(repeats)


def test_spec_nan_counts_include_late_verification_and_bonus_rows():
    # One request, two drafts: neither the second verification row nor the
    # bonus row may disappear behind the old one-row-per-request mapping.
    num_nans = torch.tensor([0, 2, 3], dtype=torch.int32)
    per_req = sum_nans_per_request(num_nans, _row_mapping([2]), 1)
    assert per_req.tolist() == [5]
    with pytest.raises(RuntimeError, match="req.*5"):
        raise_if_nan_logits(dict(zip(["req"], per_req.tolist())))


def test_spec_nan_counts_variable_lengths_and_later_requests():
    # c8 request order has variable drafts, including a request with no draft.
    draft_lens = [2, 0, 1, 2, 0, 2, 1, 2]
    counts = [0] * sum(n + 1 for n in draft_lens)
    counts[2] = 1  # bonus row of request 0
    counts[11] = 2  # a verification row of request 5
    counts[-1] = 3  # bonus row of request 7
    per_req = sum_nans_per_request(
        torch.tensor(counts, dtype=torch.int32), _row_mapping(draft_lens), 8
    )
    assert per_req.tolist() == [1, 0, 0, 0, 0, 2, 0, 3]


def test_spec_nan_counts_clean_and_mismatched_metadata():
    mapping = _row_mapping([2, 0])
    clean = sum_nans_per_request(
        torch.zeros(mapping.shape[0], dtype=torch.int32), mapping, 2
    )
    assert clean.tolist() == [0, 0]
    # A malformed speculative layout must not become an empty, successful
    # diagnostics dict.
    with pytest.raises(ValueError, match=f"Expected {mapping.shape[0]}"):
        sum_nans_per_request(torch.zeros(3, dtype=torch.int32), mapping, 2)


def test_non_spec_nan_counts_keep_missing_row_behavior():
    # Outside spec decoding there is exactly one row per request.
    mapping = torch.arange(2, dtype=torch.int32)
    assert sum_nans_per_request(
        torch.tensor([2], dtype=torch.int32), mapping[:1], 2
    ).tolist() == [2, 0]
    assert sum_nans_per_request(
        torch.empty(0, dtype=torch.int32), mapping[:0], 2
    ).tolist() == [0, 0]


def test_spec_nan_counts_own_their_rows():
    # The next model step reuses its metadata; the reported counts must not
    # observe later mutation.
    num_nans = torch.tensor([0, 2, 3], dtype=torch.int32)
    per_req = sum_nans_per_request(num_nans, _row_mapping([2]), 1)
    num_nans.zero_()
    assert per_req.tolist() == [5]


def _runner_with_sampler_output(sampler_output):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.batch_sharder = None
    runner.vocab_size = 8
    runner.structured_outputs_worker = None
    runner.rejection_sampler = lambda *args: sampler_output
    runner.speculator = SimpleNamespace(draft_logits=None)
    runner.sampler = lambda *args: sampler_output
    runner.model_state = SimpleNamespace(
        compute_sample_logits=lambda hidden: torch.zeros(3, runner.vocab_size)
    )
    return runner


def _sampler_output(num_nans: torch.Tensor):
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    return SamplerOutput(
        sampled_token_ids=torch.tensor([[0], [1], [2]], dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=num_nans,
        num_sampled=torch.tensor([3], dtype=torch.int32),
        num_rejected=torch.tensor([0], dtype=torch.int32),
    )


def test_current_runner_reduces_non_sharded_spec_nan_rows():
    # The current runner samples spec-decode logits as one row per
    # verification token plus one bonus row. Consumers zip ``num_nans`` onto
    # request ids, so sample() must reduce the rows before returning.
    from types import SimpleNamespace

    runner = _runner_with_sampler_output(
        _sampler_output(torch.tensor([0, 2, 3], dtype=torch.int32))
    )
    input_batch = SimpleNamespace(
        logits_indices=torch.arange(3),
        num_reqs=1,
        num_draft_tokens=2,
        expanded_idx_mapping=_row_mapping([2]),
    )

    output, _, _ = runner.sample(torch.zeros(3, 4), input_batch, None)

    assert output.num_nans.tolist() == [5]


def test_current_runner_keeps_per_request_nan_rows_without_drafts():
    # A non-spec batch already reports one count per request; reducing it again
    # would collapse or drop those counts.
    from types import SimpleNamespace

    runner = _runner_with_sampler_output(
        _sampler_output(torch.tensor([0, 2, 3], dtype=torch.int32))
    )
    runner.rejection_sampler = None
    input_batch = SimpleNamespace(
        logits_indices=torch.arange(3),
        num_reqs=3,
        num_draft_tokens=0,
        expanded_idx_mapping=torch.arange(3, dtype=torch.int32),
    )

    output, _, _ = runner.sample(torch.zeros(3, 4), input_batch, None)

    assert output.num_nans.tolist() == [0, 2, 3]
