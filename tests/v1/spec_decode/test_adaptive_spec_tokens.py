# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The acceptance-driven count of verified drafts (``AdaptiveSpecTokens``).

The policy only moves a performance knob (the output does not depend on how many
drafts a step verifies), so what is asserted is that it finds the count that a
cost model prefers for a given acceptance, that it backs off to no drafting when
drafting cannot pay, and that it never strands the engine there.
"""

import random
from statistics import fmean

import pytest

from vllm.v1.spec_decode.dynamic.adaptive import AdaptiveSpecTokens, StepCostModel

MAX_K = 7
# A step of 16 ms for one row, 23 ms for three and 33 ms for eight (the expert union
# of a MoE target saturates), 2 ms of logits, a 4 ms draft.
_COST = StepCostModel(
    forward_ms=(15.90, 19.45, 22.99, 24.93, 26.87, 28.81, 30.75, 32.67),
    forward_row_ms=1.94,
    base_ms=2.23,
    draft_ms=4.15,
    draft_token_ms=0.13,
    draft_request_ms=0.35,
    host_ms=0.4,
)


def _policy(max_batch: int = 2) -> AdaptiveSpecTokens:
    return AdaptiveSpecTokens(_COST, [0] + [MAX_K] * max_batch)


def _drive(
    policy: AdaptiveSpecTokens,
    accept: float,
    steps: int,
    req_id: str = "r",
    rng: random.Random | None = None,
) -> tuple[list[int], int]:
    """Run ``steps`` steps of one request whose drafts are each accepted with
    probability ``accept`` (given the drafts before them were): the verified counts
    and the tokens committed."""
    rng = rng or random.Random(0)
    ks, tokens = [], 0
    k = policy.choose([req_id])
    for _ in range(steps):
        ks.append(k)
        accepted = 0
        while accepted < k and rng.random() < accept:
            accepted += 1
        tokens += 1 + accepted
        policy.observe(req_id, k, accepted)
        k = policy.choose([req_id])
    return ks, tokens


def test_step_cost_model():
    cost = _COST
    # Drafting is paid only by steps that draft; every extra row costs forward time.
    assert cost.step_ms(1, 1) - cost.step_ms(1, 0) > cost.draft_ms
    steps = [cost.step_ms(1, k) for k in range(MAX_K + 1)]
    assert steps == sorted(steps)
    # Beyond the measured table a row costs ``forward_row_ms``.
    rows = len(cost.forward_ms)
    assert cost.forward(rows + 3) == pytest.approx(
        cost.forward(rows) + 3 * cost.forward_row_ms
    )


@pytest.mark.parametrize(
    "accept, lo, hi",
    [
        (0.9, 6, 7),  # copy-like text (retrieval): the widest step
        (0.8, 5, 7),
        (0.6, 2, 4),  # chat
        (0.55, 1.5, 4),  # prose, where drafting just pays
    ],
)
def test_converges_to_the_cost_optimum(accept, lo, hi):
    policy = _policy()
    ks, _ = _drive(policy, accept, steps=300, rng=random.Random(1))
    assert lo <= fmean(ks[100:]) <= hi


def test_never_much_worse_than_not_drafting():
    cost = _COST
    for accept in (0.0, 0.1, 0.25, 0.4):
        policy = _policy()
        ks, tokens = _drive(policy, accept, steps=600, rng=random.Random(2))
        ms_per_token = sum(cost.step_ms(1, k) for k in ks) / tokens
        assert ms_per_token <= 1.03 * cost.step_ms(1, 0), (accept, ms_per_token)
        assert policy.chosen[0] > 0.8 * len(ks), accept


def test_probes_and_recovers_when_acceptance_returns():
    policy = _policy()
    ks, _ = _drive(policy, 0.05, steps=200)
    assert max(ks[30:]) <= 1
    ks, _ = _drive(policy, 0.9, steps=200, rng=random.Random(3))
    assert fmean(ks[-50:]) > 5


def test_follows_a_change_of_content():
    policy = _policy()
    _drive(policy, 0.9, steps=100)
    ks, _ = _drive(policy, 0.5, steps=150, rng=random.Random(4))
    assert fmean(ks[-50:]) < 3.5


def test_new_request_starts_from_recent_acceptance():
    policy = _policy()
    _drive(policy, 0.92, steps=100, req_id="a")
    policy.forget("a")
    assert policy.choose(["b"]) >= 5
    cold = _policy()
    assert 2 <= cold.choose(["b"]) <= 5


def test_a_new_request_drafts_before_the_engine_prior_may_stop_it():
    def first_stop(min_steps: int) -> int:
        policy = AdaptiveSpecTokens(_COST, [0, MAX_K], min_steps=min_steps)
        _drive(policy, 0.02, steps=300, req_id="a")  # what the engine saw is poor
        policy.forget("a")
        ks, _ = _drive(policy, 0.0, steps=12, req_id="b")
        return ks.index(0) if 0 in ks else len(ks)

    assert first_stop(min_steps=0) < 4
    assert first_stop(min_steps=4) >= 4


def test_batch_size_bounds_and_forget():
    policy = AdaptiveSpecTokens(_COST, [0, MAX_K, 3, 0])
    for req in ("a", "b"):
        for _ in range(30):
            policy.observe(req, MAX_K, MAX_K)
    assert policy.choose(["a"]) == MAX_K
    assert policy.choose(["a", "b"]) <= 3
    assert policy.choose(["a", "b", "c"]) == 0
    assert policy.choose([]) == 0
    policy.forget("a")
    policy.forget("a")
    assert "a" not in policy.requests


def test_observation_counts():
    policy = _policy()
    policy.observe("r", 3, 1)  # position 1 accepted, position 2 rejected
    counts = policy.requests["r"]
    assert (counts.first_n, counts.first_a) == (1.0, 1.0)
    assert (counts.rest_n, counts.rest_a) == (1.0, 0.0)
    policy.observe("r", 3, 3)  # all accepted: positions 2 and 3 reached and accepted
    d = policy.decay
    assert counts.first_n == pytest.approx(1 + d)
    assert counts.first_a == pytest.approx(1 + d)
    assert counts.rest_n == pytest.approx(2 + d)
    assert counts.rest_a == pytest.approx(2.0)
    policy.observe("r", 0, 0)  # a step without drafts teaches nothing
    assert counts.first_n == pytest.approx(1 + d)
