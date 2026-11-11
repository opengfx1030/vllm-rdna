# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for per-request NaN accounting with speculative logits."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.v1.worker.gpu_model_runner import (
    AsyncGPUModelRunnerOutput,
    GPUModelRunner,
    nans_to_dict,
)


def test_spec_nan_counts_include_late_verification_and_bonus_rows():
    # One request, two drafts: neither the second verification row nor the
    # bonus row may disappear behind the old one-row-per-request mapping.
    assert nans_to_dict([0, 2, 3], {"req": 0}, [2]) == {"req": 5}
    with pytest.raises(RuntimeError, match="req.*5"):
        runner_module.raise_if_nan_logits(nans_to_dict([0, 2, 3], {"req": 0}, [2]))


def test_spec_nan_counts_variable_lengths_and_later_requests():
    # c8 request order has variable drafts, including a request with no draft.
    draft_lens = [2, 0, 1, 2, 0, 2, 1, 2]
    counts = [0] * sum(n + 1 for n in draft_lens)
    counts[2] = 1  # bonus row of request 0
    counts[11] = 2  # a verification row of request 5
    counts[-1] = 3  # bonus row of request 7
    mapping = {f"req_{i}": i for i in range(8)}
    result = nans_to_dict(counts, mapping, draft_lens)
    assert result == {
        "req_0": 1,
        "req_1": 0,
        "req_2": 0,
        "req_3": 0,
        "req_4": 0,
        "req_5": 2,
        "req_6": 0,
        "req_7": 3,
    }


def test_spec_nan_counts_clean_and_mismatched_metadata():
    mapping = {"req_0": 0, "req_1": 1}
    assert nans_to_dict([0, 0, 0, 0], mapping, [2, 0]) == {
        "req_0": 0,
        "req_1": 0,
    }
    # Missing speculative rows on a sampling rank must fail closed.
    with pytest.raises(ValueError, match="Expected 4"):
        nans_to_dict([], mapping, [2, 0])
    # A malformed speculative layout must not become an empty, successful
    # diagnostics dict as the old IndexError catch did.
    with pytest.raises(ValueError, match="Expected 4"):
        nans_to_dict([0, 0, 0], mapping, [2, 0])


def test_non_spec_nan_counts_keep_missing_row_behavior():
    mapping = {"req_0": 0, "req_1": 1}
    assert nans_to_dict([2], mapping) == {"req_0": 2, "req_1": 0}
    assert nans_to_dict([], mapping) == {"req_0": 0, "req_1": 0}


def test_sync_spec_nan_check_raises_for_late_row(monkeypatch):
    monkeypatch.setattr(runner_module, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(runner_module.envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_id_to_index={"req": 0}))
    metadata = SimpleNamespace(num_draft_tokens=[2])
    logits = torch.tensor([[1.0], [2.0], [float("nan")]])
    with pytest.raises(RuntimeError, match="req.*1"):
        GPUModelRunner._get_nans_in_logits(runner, logits, metadata)
    # Non-last pipeline ranks have no logits to inspect.
    assert GPUModelRunner._get_nans_in_logits(runner, None, metadata) == {"req": 0}


@pytest.mark.parametrize("raise_on_nan", [False, True])
def test_async_output_owns_spec_lengths(monkeypatch, raise_on_nan):
    event = Mock()
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(runner_module.envs, "VLLM_RAISE_ON_LOGIT_NANS", raise_on_nan)

    output = SimpleNamespace(req_id_to_index={"req_0": 0, "req_1": 1})
    copy_stream = Mock()
    parse_output = Mock(return_value=([[1, 2], [4]], None))
    monkeypatch.setattr(runner_module.RejectionSampler, "parse_output", parse_output)
    draft_lens = [1, 0]
    async_output = AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=copy_stream,
        vocab_size=10,
        num_nans=torch.tensor([0, 2, 3]),
        num_draft_tokens=draft_lens,
    )
    draft_lens[:] = [0, 0]  # The next model step reuses its metadata.
    if raise_on_nan:
        with pytest.raises(RuntimeError, match="req_0.*2"):
            async_output.get_output()
    else:
        async_output.get_output()
    parse_output.assert_called_once()
    assert output.num_nans_in_logits == {
        "req_0": 2,
        "req_1": 3,
    }
