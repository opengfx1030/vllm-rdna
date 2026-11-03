# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model runner's execute_model state is per lane.

The two lane threads share the lane's model runner in lane mode. With a single
state slot one lane's sample_tokens/pool consumed the other lane's executed
batch and returned None (every Flash-Next MTP2 split arm died at its first
probe in core.py:1141 "unexpected error").
"""

from vllm.v1.lane import Lane, lane_scope
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _bare_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.execute_model_state = {}
    return runner


def test_execute_model_state_is_keyed_by_lane():
    runner = _bare_runner()
    decode_state, prefill_state = object(), object()

    with lane_scope(Lane.DECODE):
        runner.execute_model_state[int(Lane.DECODE)] = decode_state
    with lane_scope(Lane.PREFILL):
        runner.execute_model_state[int(Lane.PREFILL)] = prefill_state

    with lane_scope(Lane.DECODE):
        assert runner._take_execute_model_state() is decode_state
    with lane_scope(Lane.PREFILL):
        assert runner._take_execute_model_state() is prefill_state

    assert runner.execute_model_state == {}


def test_execute_model_state_take_is_per_lane_and_pops():
    runner = _bare_runner()
    prefill_state = object()
    with lane_scope(Lane.PREFILL):
        runner.execute_model_state[int(Lane.PREFILL)] = prefill_state
        # D's sample must not see or consume P's executed batch.
        assert runner.execute_model_state.get(int(Lane.DECODE)) is None
        assert runner._take_execute_model_state() is prefill_state
        assert runner._take_execute_model_state() is None
