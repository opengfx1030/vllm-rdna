# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug: per-step GPU phase timing with CUDA events.

``VLLM_RDNA_PHASE_TIMING=<steps>`` records an event at named points of each
step (kineto records no ROCm kernels in this build). Every ``steps`` steps,
the mean time of each segment (previous mark -> this mark), grouped by the
batch token count, is logged. Segments include GPU idle time, so a large
segment can also mean the host was late to queue the next kernel.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PHASE_EVERY = int(os.environ.get("VLLM_RDNA_PHASE_TIMING", "0") or 0)


class PhaseTimer:
    def __init__(self, every: int) -> None:
        self.every = every
        self.cur: tuple[int, list[tuple[str, torch.cuda.Event]]] | None = None
        self.done: list[tuple[int, list[tuple[str, torch.cuda.Event]]]] = []

    def _event(self) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def start(self, num_tokens: int) -> None:
        self.cur = (num_tokens, [("start", self._event())])

    def mark(self, name: str) -> None:
        if self.cur is not None:
            self.cur[1].append((name, self._event()))

    def end(self, name: str) -> None:
        if self.cur is None:
            return
        self.mark(name)
        self.done.append(self.cur)
        self.cur = None
        if len(self.done) >= self.every:
            self.flush()

    def flush(self) -> None:
        done, self.done = self.done, []
        done[-1][1][-1][1].synchronize()
        # Per token count: name -> per-step totals (a name can repeat in a step,
        # e.g. one mark per layer).
        stats: dict[int, dict[str, list[float]]] = {}
        prev_end = None
        for n, evs in done:
            step: dict[str, float] = {}
            if prev_end is not None:
                step["gap"] = prev_end.elapsed_time(evs[0][1])
            for (_, e0), (name, e1) in zip(evs, evs[1:]):
                step[name] = step.get(name, 0.0) + e0.elapsed_time(e1)
            prev_end = evs[-1][1]
            seg = stats.setdefault(n, {})
            for name, v in step.items():
                seg.setdefault(name, []).append(v)
        for n, seg in sorted(stats.items()):
            k = max(len(v) for v in seg.values())
            parts = " ".join(
                f"{name}={sum(v) / len(v):.2f}ms" for name, v in seg.items()
            )
            logger.info("[PHASE tokens=%d n=%d] %s", n, k, parts)


phase_timer: PhaseTimer | None = PhaseTimer(PHASE_EVERY) if PHASE_EVERY > 0 else None
