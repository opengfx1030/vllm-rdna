# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed decode+chunked-prefill steps always run (upstream semantics)."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
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


@pytest.mark.parametrize("no_mixed_env", ["1", "0", None])
def test_inflight_prefill_chunk_runs_alongside_decode(no_mixed_env, monkeypatch):
    if no_mixed_env is None:
        monkeypatch.delenv("VLLM_ROCM_NO_MIXED_BATCH", raising=False)
    else:
        monkeypatch.setenv("VLLM_ROCM_NO_MIXED_BATCH", no_mixed_env)

    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=50, enable_chunked_prefill=True
    )

    (decode_req,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(decode_req)
    output = scheduler.schedule()
    scheduler.update_from_output(output, _output(["dec0"], [[0]]))
    assert decode_req in scheduler.running and not decode_req.is_prefill_chunk

    (chunk_req,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk_req)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["chk0"] > 0
    scheduler.update_from_output(output, _output(["dec0", "chk0"], [[0], []]))
    assert chunk_req.is_prefill_chunk

    mixed = scheduler.schedule()
    assert "chk0" in mixed.num_scheduled_tokens, (
        "an in-flight prefill chunk must not be dropped because a decode is running"
    )
    assert "dec0" in mixed.num_scheduled_tokens
