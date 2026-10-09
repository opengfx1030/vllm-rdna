# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug: per-step GPU phase timing with CUDA events.

``VLLM_RDNA_PHASE_TIMING=<steps>`` records an event at named points of each
step (kineto records no ROCm kernels in this build). Every ``steps`` steps,
the mean time of each segment (previous mark -> this mark), grouped by the
batch token count, is logged. Segments include GPU idle time, so a large
segment can also mean the host was late to queue the next kernel.

Each segment is also reported as host time (``name@cpu``): the wall time the
host spent between the two marks. ``prep@cpu`` is the host time from the
start of ``execute_model`` to the forward. A segment whose host time is close
to its GPU time is host-bound.

``VLLM_RDNA_OP_TIMING=1`` (with the phase timer on) also times every vLLM
custom op (``direct_register_custom_op``) that runs outside a graph capture:
``op:<name>`` is the GPU span, ``op:<name>@cpu`` the host time, both summed
over the calls of one step, and ``op:<name>#`` the number of calls.
"""

import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PHASE_EVERY = int(os.environ.get("VLLM_RDNA_PHASE_TIMING", "0") or 0)
OP_TIMING = PHASE_EVERY > 0 and os.environ.get("VLLM_RDNA_OP_TIMING", "0") in ("1", "2")
# VLLM_RDNA_OP_TIMING=2 also keys each op by the shapes of its first two tensors.
OP_SHAPES = os.environ.get("VLLM_RDNA_OP_TIMING", "0") == "2"


class PhaseTimer:
    def __init__(self, every: int) -> None:
        self.every = every
        # (num_tokens, [(name, event, host_t)], [(op, ev0, ev1, host_dt)])
        self.cur: (
            tuple[
                int,
                list[tuple[str, torch.cuda.Event, float]],
                list[tuple[str, torch.cuda.Event, torch.cuda.Event, float]],
            ]
            | None
        ) = None
        self.done: list = []
        self.step_t0: float | None = None

    def _event(self) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def begin_step(self) -> None:
        """Host timestamp at the start of execute_model."""
        self.step_t0 = time.perf_counter()

    def start(self, num_tokens: int) -> None:
        now = time.perf_counter()
        ops: list = []
        marks = [("start", self._event(), now)]
        if self.step_t0 is not None:
            ops.append(("prep", None, None, now - self.step_t0))
            self.step_t0 = None
        self.cur = (num_tokens, marks, ops)

    def mark(self, name: str) -> None:
        if self.cur is not None:
            self.cur[1].append((name, self._event(), time.perf_counter()))

    def op_begin(self) -> tuple[torch.cuda.Event, float] | None:
        if self.cur is None or torch.cuda.is_current_stream_capturing():
            return None
        return self._event(), time.perf_counter()

    def op_end(self, name: str, token: tuple[torch.cuda.Event, float]) -> None:
        if self.cur is None:
            return
        ev1 = self._event()
        self.cur[2].append((name, token[0], ev1, time.perf_counter() - token[1]))

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
        for n, evs, ops in done:
            step: dict[str, float] = {}
            if prev_end is not None:
                step["gap"] = prev_end.elapsed_time(evs[0][1])
            for (_, e0, h0), (name, e1, h1) in zip(evs, evs[1:]):
                step[name] = step.get(name, 0.0) + e0.elapsed_time(e1)
                step[name + "@cpu"] = step.get(name + "@cpu", 0.0) + (h1 - h0) * 1e3
            step["total"] = evs[0][1].elapsed_time(evs[-1][1])
            step["total@cpu"] = (evs[-1][2] - evs[0][2]) * 1e3
            for name, e0, e1, hdt in ops:
                key = "op:" + name if e0 is not None else name
                if e0 is not None:
                    step[key] = step.get(key, 0.0) + e0.elapsed_time(e1)
                    step[key + "#"] = step.get(key + "#", 0.0) + 1
                step[key + "@cpu"] = step.get(key + "@cpu", 0.0) + hdt * 1e3
            prev_end = evs[-1][1]
            seg = stats.setdefault(n, {})
            for name, v in step.items():
                seg.setdefault(name, []).append(v)
        for n, seg in sorted(stats.items()):
            k = max(len(v) for v in seg.values())
            parts = " ".join(
                f"{name}={sum(v) / len(v):.2f}"
                + ("" if name.endswith("#") else "ms")
                for name, v in seg.items()
            )
            logger.info("[PHASE tokens=%d n=%d] %s", n, k, parts)


phase_timer: PhaseTimer | None = PhaseTimer(PHASE_EVERY) if PHASE_EVERY > 0 else None

# ``VLLM_RDNA_SYNC_DEBUG=<steps>``: report host-device synchronizing calls made
# during the forward of real (non-dummy) steps, deduplicated by call site,
# every ``steps`` steps (torch.cuda.set_sync_debug_mode).
SYNC_EVERY = int(os.environ.get("VLLM_RDNA_SYNC_DEBUG", "0") or 0)


class SyncDebug:
    def __init__(self, every: int) -> None:
        import collections

        self.every = every
        self.steps = 0
        self.sites: collections.Counter = collections.Counter()
        self.active = False

    def _show(self, message, category, filename, lineno, file=None, line=None):
        import traceback

        if not self.active or "synchroniz" not in str(message):
            return self._orig(message, category, filename, lineno, file, line)
        frames = [
            f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}"
            for f in traceback.extract_stack()[:-2]
            if "vllm" in f.filename and "phase_timer" not in f.filename
        ]
        self.sites[" < ".join(reversed(frames[-4:]))] += 1

    def begin(self) -> None:
        import warnings

        if not hasattr(self, "_orig"):
            self._orig = warnings.showwarning
            warnings.showwarning = self._show
            warnings.simplefilter("always")
        self.active = True
        torch.cuda.set_sync_debug_mode("warn")

    def end(self, num_tokens: int) -> None:
        torch.cuda.set_sync_debug_mode("default")
        self.active = False
        self.steps += 1
        if self.steps >= self.every:
            for site, n in self.sites.most_common(25):
                logger.info("[SYNC n=%d steps=%d] %s", n, self.steps, site)
            self.sites.clear()
            self.steps = 0


sync_debug: SyncDebug | None = SyncDebug(SYNC_EVERY) if SYNC_EVERY > 0 else None


def wrap_op_timing(op_name: str, fn):
    """Wrap a custom-op implementation with phase-timer op spans."""
    if not OP_TIMING:
        return fn

    def timed(*args, **kwargs):
        pt = phase_timer
        tok = pt.op_begin() if pt is not None else None
        if tok is None:
            return fn(*args, **kwargs)
        out = fn(*args, **kwargs)
        name = op_name
        if OP_SHAPES:
            dims = [
                "x".join(map(str, a.shape))
                for a in args[:2]
                if isinstance(a, torch.Tensor)
            ]
            name = f"{op_name}[{','.join(dims)}]"
        pt.op_end(name, tok)
        return out

    timed.__name__ = getattr(fn, "__name__", op_name)
    return timed
