# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-aware dynamic SD policy (``dynamic_sd_policy="acceptance"``)."""

from vllm.v1.spec_decode.dynamic.policy import AcceptanceAwareKPolicy

TABLE = [(1, 1, 4), (2, 2, 2), (3, 4, 1), (5, 256, 0)]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _policy(clock=None, **kw) -> AcceptanceAwareKPolicy:
    return AcceptanceAwareKPolicy(
        TABLE, max_batch_size=8, num_speculative_tokens=4, clock=clock or _Clock(), **kw
    )


def _feed_alpha(policy: AcceptanceAwareKPolicy, alpha: float, n: int = 400) -> None:
    # Deterministic stream with the requested per-draft acceptance.
    acc = 0.0
    for _ in range(n):
        acc += alpha
        if acc >= 1.0:
            acc -= 1.0
            policy.observe_verify(1, 1)
        else:
            policy.observe_verify(1, 0)


def _time_shapes(policy, clock, times: dict[tuple[int, int], float]) -> None:
    for shape, dt in times.items():
        policy.observe_step(None)  # restart the timer
        clock.t += dt
        policy.observe_step(shape)


def test_candidates_follow_the_schedule_graph_bound():
    p = _policy()
    assert p.k_values == [0, 1, 2, 4]
    assert p.max_tokens == 8
    assert p.candidates(1) == [0, 1, 2, 4]
    assert p.candidates(2) == [0, 1, 2]
    assert p.candidates(3) == [0, 1]
    assert p.candidates(5) == [0]


def test_falls_back_to_the_table_without_estimates():
    clock = _Clock()
    p = _policy(clock)
    assert [p.choose(n) for n in range(1, 9)] == [4, 2, 1, 1, 0, 0, 0, 0]
    # Acceptance known but no step times: measure the table's K first.
    _feed_alpha(p, 0.7)
    assert p.choose(1) == 4
    assert p.choose(5) == 0  # single candidate


def test_low_acceptance_lowers_k_and_high_acceptance_keeps_it():
    clock = _Clock()
    times = {(1, 0): 0.020, (1, 1): 0.025, (1, 2): 0.026, (1, 4): 0.032}
    low = _policy(clock)
    _feed_alpha(low, 0.2)
    _time_shapes(low, clock, times)
    assert low.choose(1) == 0
    high = _policy(clock)
    _feed_alpha(high, 0.9)
    _time_shapes(high, clock, times)
    assert high.choose(1) == 4


def test_unmeasured_candidates_are_tried_once():
    clock = _Clock()
    p = _policy(clock)
    _feed_alpha(p, 0.6)
    _time_shapes(p, clock, {(2, 2): 0.03})
    k = p.choose(2)
    assert k in (0, 1)
    _time_shapes(p, clock, {(2, k): 0.025})
    assert p.choose(2) in (0, 1)


def test_verify_trials_count_the_first_rejection_only():
    p = _policy()
    for _ in range(10):
        p.observe_verify(4, 1)  # 1 accepted, then rejected: 2 trials
    assert abs(p.alpha - 0.5) < 1e-6
    q = _policy()
    for _ in range(10):
        q.observe_verify(2, 2)  # all accepted: 2 trials, 2 successes
    assert q.alpha > 0.99
