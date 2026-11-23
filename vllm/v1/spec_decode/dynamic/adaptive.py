# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-driven choice of the number of drafts verified per step.

Verifying fewer drafts than were proposed never changes the output (a draft is
accepted only when it equals what the target samples at that position), so the
count is a pure performance knob.  The scheduler asks this policy for the count of
every step.  It keeps, per request, a decayed estimate of the acceptance of the
first draft position and of the later ones (``p1`` and ``p``: the conditional
probability that a draft is accepted given that the drafts before it were), and
picks the ``k`` that maximizes

    expected tokens per step(k) / step time(k)

with ``expected tokens(k) = 1 + p1 + p1 * p + ... + p1 * p**(k - 1)`` and the step
time from a ``StepCostModel`` (the target forward grows with the verified rows, the
draft with the drafted tokens).  Pooling the later positions keeps the estimate
steady: a per-position estimate is too noisy for the few steps a request has, and
the throughput is flat around its optimum, so the loss of a stable ``k`` that is
one off is small while that of a noisy ``k`` is not.  A request starts from the
acceptance the engine saw lately (content comes in runs).  A step with ``k = 0``
verifies nothing and so teaches nothing: while the best ``k`` is 0, every
``probe_interval``-th step verifies one draft instead, which keeps the cost of
being wrong about low acceptance to a few percent.
"""

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepCostModel:
    """Milliseconds of one decode step of the whole batch.

    Only ratios matter.  ``forward_ms[r - 1]`` is the target forward of a step with
    ``r`` verified rows in total (one per request plus the drafts); beyond the table
    each row costs ``forward_row_ms``.
    """

    forward_ms: tuple[float, ...]
    forward_row_ms: float
    # logits, sampling and the gaps between steps, paid by every step
    base_ms: float
    # drafting (``draft_ms`` once per step plus ``draft_token_ms`` per drafted token,
    # ``draft_request_ms`` per request beyond the first) and the host time that a
    # drafting step exposes; none of them is paid by a step with ``k = 0``
    draft_ms: float
    draft_token_ms: float
    draft_request_ms: float = 0.0
    host_ms: float = 0.0

    def forward(self, rows: int) -> float:
        if rows <= len(self.forward_ms):
            return self.forward_ms[max(rows, 1) - 1]
        return self.forward_ms[-1] + (rows - len(self.forward_ms)) * self.forward_row_ms

    def step_ms(self, num_reqs: int, k: int) -> float:
        cost = self.forward(num_reqs * (1 + k)) + self.base_ms
        if k > 0:
            cost += (
                self.draft_ms
                + k * self.draft_token_ms
                + (num_reqs - 1) * self.draft_request_ms
                + self.host_ms
            )
        return cost


@dataclass
class _Counts:
    """Decayed numbers of times the first draft position and the later ones were
    verified (``*_n``) and accepted (``*_a``), and the steps seen."""

    first_n: float = 0.0
    first_a: float = 0.0
    rest_n: float = 0.0
    rest_a: float = 0.0
    steps: int = 0

    def decay(self, factor: float) -> None:
        self.first_n *= factor
        self.first_a *= factor
        self.rest_n *= factor
        self.rest_a *= factor

    def add(self, num_draft_tokens: int, num_accepted: int) -> None:
        self.steps += 1
        self.first_n += 1.0
        self.first_a += num_accepted >= 1
        # Position i + 1 is reached when the drafts before it were all accepted.
        reached = min(num_accepted + 1, num_draft_tokens)
        self.rest_n += max(reached - 1, 0)
        self.rest_a += max(num_accepted - 1, 0)


@dataclass
class AdaptiveSpecTokens:
    """Per-step number of drafts to verify, from the running acceptance.

    Args:
        cost: step time model.
        max_tokens: ``max_tokens[b]`` is the most drafts a step of ``b`` requests may
            verify (index 0 unused); a step of more requests than the table has
            verifies none.
        decay: weight kept of the older observations of a request at every step.
        prior_accept: acceptance assumed before anything was observed.
        prior_weight: observations of the engine-wide acceptance that a request's
            estimate starts from.
        engine_decay: like ``decay``, for the engine-wide acceptance.
        engine_weight: observations ``prior_accept`` is worth in the engine-wide
            acceptance.
        stop_margin: drafting stops only when not drafting is faster by this fraction.
        probe_interval: steps without drafts between two probes of one draft.
        min_steps: verified steps a request needs before drafting may stop for it
            (what the engine saw of the requests before says little about this one).
        k_tolerance: the smallest ``k`` whose rate is within this fraction of the
            best one is chosen (the rate is flat near its optimum, a smaller ``k``
            costs less when the estimate is too high).

    """

    cost: StepCostModel
    max_tokens: Sequence[int]
    decay: float = 0.90
    prior_accept: float = 0.6
    prior_weight: float = 6.0
    engine_decay: float = 0.995
    engine_weight: float = 20.0
    stop_margin: float = 0.08
    probe_interval: int = 32
    min_steps: int = 4
    k_tolerance: float = 0.01
    requests: dict[str, _Counts] = field(default_factory=dict)
    engine: _Counts = field(default_factory=_Counts)
    chosen: Counter = field(default_factory=Counter)
    _idle_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("decay", "engine_decay"):
            assert 0.0 < getattr(self, name) <= 1.0, name
        assert 0.0 < self.prior_accept < 1.0
        assert self.prior_weight > 0.0 and self.engine_weight > 0.0
        assert 0.0 <= self.k_tolerance < 1.0

    def forget(self, req_id: str) -> None:
        self.requests.pop(req_id, None)

    def observe(self, req_id: str, num_draft_tokens: int, num_accepted: int) -> None:
        """One verified step of a request: ``num_draft_tokens`` drafts, the first
        ``num_accepted`` of them accepted."""
        if num_draft_tokens <= 0:
            return
        counts = self.requests.setdefault(req_id, _Counts())
        counts.decay(self.decay)
        counts.add(num_draft_tokens, num_accepted)
        self.engine.decay(self.engine_decay)
        self.engine.add(num_draft_tokens, num_accepted)

    def _engine_acceptance(self) -> tuple[float, float]:
        engine, weight = self.engine, self.engine_weight
        return (
            (engine.first_a + weight * self.prior_accept) / (engine.first_n + weight),
            (engine.rest_a + weight * self.prior_accept) / (engine.rest_n + weight),
        )

    def acceptance(self, req_id: str) -> tuple[float, float]:
        """``(p1, p)``: estimated acceptance of the first draft position and of the
        later ones."""
        counts = self.requests.get(req_id) or _Counts()
        prior_first, prior_rest = self._engine_acceptance()
        weight = self.prior_weight
        first = (counts.first_a + weight * prior_first) / (counts.first_n + weight)
        # Later positions without data follow the first one as well as the prior.
        rest = (counts.rest_a + weight * (prior_rest + first) / 2) / (
            counts.rest_n + weight
        )
        return first, rest

    @staticmethod
    def expected_tokens(first: float, rest: float, k: int) -> float:
        """Expected tokens a step commits when it verifies ``k`` drafts."""
        tokens, survive = 1.0, 1.0
        for i in range(k):
            survive *= first if i == 0 else rest
            tokens += survive
        return tokens

    def _best_k(
        self,
        num_reqs: int,
        cap: int,
        estimates: Sequence[tuple[float, float]],
        may_stop: bool,
    ) -> int:
        rates = [
            sum(self.expected_tokens(first, rest, k) for first, rest in estimates)
            / self.cost.step_ms(num_reqs, k)
            for k in range(cap + 1)
        ]
        best_k = max(range(1, cap + 1), key=rates.__getitem__)
        # A noisy estimate makes stopping cost more than drafting in vain.
        if may_stop and rates[best_k] < rates[0] * (1.0 - self.stop_margin):
            return 0
        return next(
            k
            for k in range(1, best_k + 1)
            if rates[k] >= rates[best_k] * (1.0 - self.k_tolerance)
        )

    def choose(self, req_ids: Collection[str]) -> int:
        """Drafts to verify in the next step of the batch ``req_ids``."""
        num_reqs = len(req_ids)
        if num_reqs == 0:
            return 0
        cap = self.max_tokens[num_reqs] if num_reqs < len(self.max_tokens) else 0
        if cap <= 0:
            return 0
        estimates = [self.acceptance(r) for r in req_ids]
        may_stop = all(
            r in self.requests and self.requests[r].steps >= self.min_steps
            for r in req_ids
        )
        best_k = self._best_k(num_reqs, cap, estimates, may_stop)
        if best_k == 0:
            self._idle_steps += 1
            if self._idle_steps >= self.probe_interval:
                best_k = 1
        if best_k > 0:
            self._idle_steps = 0
        self.chosen[best_k] += 1
        return best_k
