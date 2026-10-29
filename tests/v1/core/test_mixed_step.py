# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feedback-loop regressions for the optional mixed-step cost controller."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def controller():
    path = Path(__file__).resolve().parents[3] / "vllm/v1/core/sched/mixed_step.py"
    spec = importlib.util.spec_from_file_location("mixed_step_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MixedStepController


@pytest.mark.parametrize("block_size", [None, 512, 1024, 2048])
def test_expensive_expert_streaming_cannot_strand_cost_fit(controller, block_size):
    model = controller(
        budget_s=0.25, decode_share=0, max_prefill_tokens=2048, max_overhead=0.25
    )
    # Fixed expert-streaming overhead plus per-token compute. A proportional
    # one-point estimate asks for 64 tokens, which observe() deliberately skips.
    caps = []
    for _ in range(12):
        cap = model.prefill_cap(block_size)
        caps.append(cap)
        model.observe(cap, 4, 1.5 + 0.001 * cap)
    # The loop must collect a second informative bucket and recognize that
    # tiny chunks cannot amortize the fixed cost. It need not meet a hard SLA.
    assert model._fit() == pytest.approx((1.5, 0.001))
    assert caps[-1] > caps[0]
    assert all(128 <= cap <= 2048 for cap in caps)


def test_small_hard_token_budget_is_never_exceeded(controller):
    model = controller(budget_s=0.001, decode_share=0, max_prefill_tokens=64)
    for _ in range(5):
        cap = model.prefill_cap()
        assert cap == 64
        model.observe(cap, 1, 0.5)


def test_tiny_tail_does_not_pollute_prefill_cost_fit(controller):
    model = controller(budget_s=0.25, decode_share=0, max_prefill_tokens=2048)
    model.observe(512, 4, 2)
    model.observe(32, 4, 10)
    assert model._fit() is None


def test_learning_floor_preserves_nonstandard_block_tiling(controller):
    model = controller(budget_s=0.001, decode_share=0, max_prefill_tokens=2048)
    model.observe(512, 4, 2)
    assert model.prefill_cap(block_size=96) == 96


def test_decode_share_debt_is_paid_by_decode_steps(controller):
    model = controller(budget_s=0.25, decode_share=0.25, max_prefill_tokens=2048)
    model.observe(512, 4, 0.6)
    assert model.defer_prefill()
    model.observe(0, 4, 0.21)
    assert not model.defer_prefill()
