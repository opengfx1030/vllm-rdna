# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound the prefill work of steps that also carry decodes.

A step that carries prefill tokens runs the prefill path for every request in
the batch, so each request that is already decoding waits for the whole step.
With a 2048-token budget that step is ~0.5 s on a 4x V620 Flash-Next and
~2 s on a 27B dense model, against a ~34 ms decode step: decoders freeze for
the whole prefill (docs/rdna2/mixed-batch-stall-probe.md).

This controller does two things while at least one request is decoding:

* **Chunk bound.** It caps the total prefill tokens of a step so that the
  step's predicted duration stays within ``budget_s``. The prediction is a
  line ``t = a + b * prefill_tokens`` fitted to the measured durations of
  earlier prefill-carrying steps, so the cap adapts to the model and the
  hardware (the 27B gets ~200 tokens, Flash-Next ~512) instead of being a
  fixed token count.
* **Decode share.** After each prefill-carrying step with decoders, it owes
  the decoders ``share / (1 - share)`` of that step's duration in pure decode
  steps, and defers prefills until that debt is paid. Decoders then get a
  guaranteed fraction of wall time, while prefill keeps the rest.

Without decoders nothing changes: a lone prefill keeps the full budget.
"""

from __future__ import annotations

import math
import os
import statistics
from collections import deque


class MixedStepController:
    def __init__(
        self,
        budget_s: float,
        decode_share: float,
        max_prefill_tokens: int,
        granularity: int = 64,
        initial_cap: int = 512,
        max_overhead: float = float(
            os.environ.get("VLLM_DECODE_STALL_MAX_OVERHEAD", "0.25")
        ),
        fit_all_steps: bool = False,
    ) -> None:
        self.budget_s = budget_s
        self.share_ratio = (
            decode_share / (1.0 - decode_share) if 0.0 < decode_share < 1.0 else 0.0
        )
        self.max_prefill_tokens = max_prefill_tokens
        self.granularity = max(1, granularity)
        self.initial_cap = initial_cap
        self.max_overhead = max_overhead
        # Recent durations per prefill-token bucket (bucket -> last 8 samples).
        # The fit uses each bucket's median, so a one-off step (a JIT compile,
        # a TunableOp miss, a slow first request) cannot move the cap.
        self._buckets: dict[int, deque[float]] = {}
        self._decode_s = 0.035
        self._debt_s = 0.0
        # Which steps feed the fit; see fit_all_steps_default().
        self._fit_all_steps = fit_all_steps

    # ------------------------------------------------------------ observation
    def observe(
        self, num_prefill_tokens: int, num_decode_reqs: int, elapsed_s: float
    ) -> None:
        if elapsed_s <= 0.0 or elapsed_s > 30.0:
            return
        if num_prefill_tokens <= 0:
            if num_decode_reqs > 0:
                self._decode_s += 0.2 * (elapsed_s - self._decode_s)
                self._debt_s = max(0.0, self._debt_s - elapsed_s)
            return
        if num_decode_reqs > 0 and self.share_ratio > 0.0:
            self._debt_s = min(
                self._debt_s + self.share_ratio * elapsed_s, 4.0 * self.budget_s
            )
        if num_decode_reqs == 0 and not self._fit_all_steps:
            # Mixed-steps-only fit (MoE default): prefill-only steps (deep-
            # context chunks of a long prompt, prompt-logprob steps,
            # prefix-cache tails) dragged Flash-Next's fitted fixed cost to
            # 330-800 ms, above the budget, which pinned the cap at 2048.
            return
        if num_prefill_tokens < 2 * self.granularity:
            # Tiny prefills (a short prompt, the tail of a long one) are
            # dominated by per-step overheads; they would distort the slope.
            return
        b = round(num_prefill_tokens / self.granularity)
        self._buckets.setdefault(b, deque(maxlen=8)).append(elapsed_s)

    @staticmethod
    def fit_all_steps_default(is_moe: bool) -> bool:
        """Whether every prefill-carrying step feeds the fit.

        Dense models fit all prefill-carrying steps: with mixed steps only,
        the 27B AWQ fit drifted up on deep-context mixed chunks and its 16k
        injection max decode gap went 394 -> 776 ms. MoE models (Flash-Next)
        fit mixed steps only: their prefill-only steps carry a large fixed
        cost that otherwise pins the cap at whole blocks (505 -> 341 ms with
        mixed-only). ``VLLM_DECODE_STALL_FIT_ALL=1/0`` forces either mode.
        """
        env = os.environ.get("VLLM_DECODE_STALL_FIT_ALL", "")
        if env in ("0", "1"):
            return env == "1"
        return not is_moe

    # ---------------------------------------------------------------- policy
    def _points(self) -> list[tuple[float, float, int]]:
        return [
            (b * self.granularity, statistics.median(v), len(v))
            for b, v in self._buckets.items()
        ]

    def _fit(self) -> tuple[float, float] | None:
        pts = self._points()
        if len(pts) < 2:
            return None
        sw = sum(n for _, _, n in pts)
        mx = sum(n * x for x, _, n in pts) / sw
        my = sum(n * y for _, y, n in pts) / sw
        sxx = sum(n * (x - mx) ** 2 for x, _, n in pts)
        if sxx <= 0.0:
            return None
        slope = sum(n * (x - mx) * (y - my) for x, y, n in pts) / sxx
        if slope <= 0.0:
            return None
        return max(0.0, my - slope * mx), slope

    def prefill_cap(self, block_size: int | None = None) -> int:
        """Max prefill tokens for a step that carries decodes."""
        fit = self._fit()
        floor = 0
        if fit is None:
            cap = self.initial_cap
            pts = self._points()
            if pts:
                # One bucket: assume the time is all per-token cost. This may
                # go below initial_cap; otherwise a model whose initial-cap
                # steps overrun the budget never sees a second bucket.
                x, y, _ = pts[0]
                cap = int(self.budget_s / max(y, 1e-6) * x)
        else:
            a, slope = fit
            cap = int((self.budget_s - a) / slope) if self.budget_s > a else 0
            # Efficiency floor: never cut chunks so small that the per-step
            # cost `a` exceeds `max_overhead` of the step's per-token work. On
            # a model whose prefill steps carry a large fixed cost (Flash-Next:
            # eager launch + MoE weight streaming per step) a tight budget
            # would otherwise trade most of the prefill throughput for decode
            # latency (measured: 16k/1k c=8 TTFT 16 s -> 180 s at a 256-token
            # cap); a model with a small fixed cost (27B dense) still gets
            # small chunks.
            if self.max_overhead > 0.0:
                floor = int(a / (self.max_overhead * slope))
        cap = self._tile(cap, block_size, up=False)
        if floor > cap:
            cap = self._tile(floor, block_size, up=True)
        return min(cap, self.max_prefill_tokens)

    def _tile(self, tokens: int, block_size: int | None, up: bool) -> int:
        g = self.granularity
        q = (tokens + g - 1) // g * g if up else tokens // g * g
        cap = max(g, min(q, self.max_prefill_tokens))
        if not block_size or g >= block_size:
            return cap
        if cap >= block_size:
            # Whole blocks, rounded down even for the efficiency floor: one
            # block per step already amortizes the per-step cost.
            return cap // block_size * block_size
        # Chunks stop at recurrent-state block boundaries. Split the block into
        # n near-equal chunks so no short remainder step is left before the
        # boundary (e.g. 1024 at cap 509 -> 384, 384, 256 rather than 506, 506,
        # 12).
        n = math.ceil(block_size / cap) if not up else max(1, block_size // cap)
        tile = math.ceil(block_size / n / g) * g
        if not up and tile > cap:
            tile -= g
        return max(g, tile)

    def defer_prefill(self) -> bool:
        """Whether this step should be decode-only (decode share owed)."""
        return self._debt_s > 0.5 * self._decode_s

    def stats(self) -> str:
        fit = self._fit()
        f = f"a={fit[0] * 1e3:.0f}ms b={fit[1] * 1e3:.3f}ms/tok" if fit else "nofit"
        return f"{f} dec={self._decode_s * 1e3:.1f}ms debt={self._debt_s * 1e3:.0f}ms"
