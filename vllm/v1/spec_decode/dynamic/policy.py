# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-aware choice of the per-step draft count K for dynamic SD.

With per-draft acceptance ``alpha`` (i.i.d. model), a verify step with K drafts
emits ``(1 - alpha ** (K + 1)) / (1 - alpha)`` tokens per request on average and
costs ``step_time(num_reqs * (K + 1))``. The policy picks the K that maximizes
their ratio, among the K values of the configured batch-size schedule whose
uniform decode width keeps the step inside the schedule's token bound (the
shapes the model runner captures FULL graphs for). Until it has an acceptance
estimate and step times for the candidate widths, it returns the schedule's K.
"""

import time

from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup


class AcceptanceAwareKPolicy:
    def __init__(
        self,
        num_speculative_tokens_per_batch_size: object,
        max_batch_size: int,
        num_speculative_tokens: int,
        decay: float = 0.98,
        explore_interval: int = 64,
        clock=time.perf_counter,
    ) -> None:
        self.table = build_dynamic_sd_schedule_lookup(
            num_speculative_tokens_per_batch_size,
            vllm_max_batch_size=max_batch_size,
            vllm_num_speculative_tokens=num_speculative_tokens,
        )
        self.k_values = sorted(set(self.table[1:]))
        # Largest uniform decode step of the schedule (the FULL graph bound).
        self.max_tokens = max(
            num_reqs * (k + 1) for num_reqs, k in enumerate(self.table[1:], start=1)
        )
        self.decay = decay
        self.explore_interval = explore_interval
        self._clock = clock
        # Decayed acceptance counts: accepted drafts / drafts that were tried
        # (all accepted drafts plus the first rejected one).
        self._accepted = 0.0
        self._trials = 0.0
        # Decayed mean step time per uniform decode step (num_reqs, K).
        self._step_time: dict[tuple[int, int], float] = {}
        self._last_step_end: float | None = None
        self._num_choices = 0

    # ---- observations -------------------------------------------------------

    def observe_verify(self, num_drafts: int, num_accepted: int) -> None:
        """One request's verify result: ``num_accepted`` of ``num_drafts``."""
        if num_drafts <= 0:
            return
        trials = num_accepted + (1 if num_accepted < num_drafts else 0)
        self._accepted = self._accepted * self.decay + num_accepted
        self._trials = self._trials * self.decay + trials

    def observe_step(self, shape: tuple[int, int] | None) -> None:
        """Mark the end of a step. ``shape`` is its uniform decode shape
        (num_reqs, K); None for a step with prefill or mixed widths, which only
        restarts the timer. Steps are timed end to end, so back-to-back decode
        steps measure the steady-state step time."""
        now = self._clock()
        last, self._last_step_end = self._last_step_end, now
        if shape is None or last is None:
            return
        dt = now - last
        prev = self._step_time.get(shape)
        self._step_time[shape] = dt if prev is None else prev * 0.8 + dt * 0.2

    def reset_timer(self) -> None:
        """An idle gap (no work scheduled) must not count as a step."""
        self._last_step_end = None

    # ---- decision -----------------------------------------------------------

    @property
    def alpha(self) -> float | None:
        if self._trials < 8:
            return None
        return min(self._accepted / self._trials, 0.999)

    def candidates(self, num_reqs: int) -> list[int]:
        return [k for k in self.k_values if num_reqs * (k + 1) <= self.max_tokens]

    def choose(self, num_reqs: int) -> int:
        fallback = self.table[min(num_reqs, len(self.table) - 1)]
        cands = self.candidates(num_reqs)
        alpha = self.alpha
        if alpha is None or len(cands) < 2:
            return fallback
        self._num_choices += 1
        unmeasured = [k for k in cands if (num_reqs, k) not in self._step_time]
        if unmeasured:
            # Measure every candidate width once, starting from the table's K.
            return fallback if fallback in unmeasured else unmeasured[0]
        if self._num_choices % self.explore_interval == 0:
            # Re-measure a neighbour of the table's K now and then, so a stale
            # time cannot pin the choice.
            i = cands.index(fallback) if fallback in cands else 0
            return cands[(i + 1) % len(cands)]

        def tokens_per_second(k: int) -> float:
            expected = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
            return expected / self._step_time[(num_reqs, k)]

        return max(cands, key=tokens_per_second)
