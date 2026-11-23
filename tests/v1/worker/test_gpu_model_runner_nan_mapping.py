# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for per-request NaN accounting with speculative logits.

The current runner samples spec-decode logits as one row per verification
token plus one bonus row per request. ``num_nans_in_logits`` is reported per
request, so every row must be summed into its request instead of being dropped
by a per-request zip over the row counts. Rows belong to a request by batch
position (``cu_num_logits``); ``expanded_idx_mapping`` holds request state
indices, which are not batch positions and must never index the result.
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.async_utils import sum_nans_per_request
from vllm.v1.worker.utils import raise_if_nan_logits


def _layout(draft_lens: list[int]) -> tuple[torch.Tensor, np.ndarray]:
    """cu_num_logits as the runner builds it: n drafts + 1 bonus row per request."""
    cu_np = np.concatenate([[0], np.cumsum([n + 1 for n in draft_lens])]).astype(
        np.int32
    )
    return torch.from_numpy(cu_np.copy()), cu_np


def test_spec_nan_counts_include_late_verification_and_bonus_rows():
    # One request, two drafts: neither the second verification row nor the
    # bonus row may disappear behind the old one-row-per-request mapping.
    num_nans = torch.tensor([0, 2, 3], dtype=torch.int32)
    per_req = sum_nans_per_request(num_nans, *_layout([2]))
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
        torch.tensor(counts, dtype=torch.int32), *_layout(draft_lens)
    )
    assert per_req.tolist() == [1, 0, 0, 0, 0, 2, 0, 3]


def test_spec_nan_counts_clean_and_mismatched_metadata():
    cu, cu_np = _layout([2, 0])
    clean = sum_nans_per_request(torch.zeros(4, dtype=torch.int32), cu, cu_np)
    assert clean.tolist() == [0, 0]
    # A malformed speculative layout must not become an empty, successful
    # diagnostics dict.
    with pytest.raises(ValueError, match="Expected 4"):
        sum_nans_per_request(torch.zeros(3, dtype=torch.int32), cu, cu_np)


def test_spec_nan_counts_tolerate_empty_rows():
    cu, cu_np = _layout([])  # zero requests
    assert sum_nans_per_request(torch.empty(0, dtype=torch.int32), cu, cu_np).shape == (
        0,
    )


def test_spec_nan_counts_own_their_rows():
    # The next model step reuses its metadata; the reported counts must not
    # observe later mutation.
    num_nans = torch.tensor([0, 2, 3], dtype=torch.int32)
    per_req = sum_nans_per_request(num_nans, *_layout([2]))
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
    cu, cu_np = _layout([2])
    input_batch = SimpleNamespace(
        logits_indices=torch.arange(3),
        num_reqs=1,
        num_draft_tokens=2,
        cu_num_logits=cu,
        cu_num_logits_np=cu_np,
        # Request state indices: the only live request sits in slot 5.
        expanded_idx_mapping=torch.full((3,), 5, dtype=torch.int32),
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
