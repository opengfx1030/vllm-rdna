# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-lane end-of-batch fences (S3 R-S3-4, section 5 rule F1).

A batch's fence event is recorded at the very end of ``sample_tokens`` -- after
the drafter, through the runner's ``lane_ctx.on_batch_done`` hook -- so it
covers every kernel the batch enqueued on its lane stream. ``AsyncOutput``'s
copy event is recorded before the drafter runs and must not be used.

Before a lane launches a batch it waits for the *other* lane's event named by
``SchedulerOutput.lane_fence``. Ranks complete at different times, so the event
may not exist yet on this rank: the wait is a host condition wait first, then
``stream.wait_event``. It is acyclic because a fence only names batches the
engine has already processed.
"""

import threading
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.lane import Lane

logger = init_logger(__name__)

# The engine keeps a lane queue at depth <= 2, so 64 events per lane is far more
# than any live fence can name.
KEEP_EVENTS = 64


def _new_cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event()


class LaneFences:
    """Per-lane ``seq -> event`` table with a host condition for late events."""

    def __init__(
        self,
        *,
        keep: int = KEEP_EVENTS,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._cond = threading.Condition()
        self._keep = keep
        self._event_factory = event_factory or _new_cuda_event
        self._events: dict[Lane, dict[int, Any]] = {lane: {} for lane in Lane}
        # Oldest -> newest, for pruning.
        self._order: dict[Lane, list[int]] = {lane: [] for lane in Lane}
        # Highest seq already pruned; a fence never names one of these, and if
        # one does the dependency is older than 64 recorded batches.
        self._pruned_to: dict[Lane, int] = {lane: 0 for lane in Lane}
        # Set by _mark_failed: a waiter must not block forever on a dead lane.
        self._aborted: BaseException | None = None

    def record(self, lane: int | Lane, seq: int, stream: torch.cuda.Stream) -> None:
        """Publish the end-of-batch event of ``seq`` on ``lane`` (lane thread)."""
        lane = Lane(lane)
        seq = int(seq)
        if seq <= 0:
            # lane_seq 0 means an empty batch or flag off: nothing to fence.
            return
        event = self._event_factory()
        event.record(stream)
        with self._cond:
            if seq in self._events[lane]:
                logger.warning(
                    "two-lane: fence %s seq %d recorded twice; overwriting",
                    lane.name,
                    seq,
                )
            else:
                self._order[lane].append(seq)
            self._events[lane][seq] = event
            while len(self._order[lane]) > self._keep:
                old = self._order[lane].pop(0)
                self._events[lane].pop(old, None)
                self._pruned_to[lane] = max(self._pruned_to[lane], old)
            self._cond.notify_all()

    def abort(self, exc: BaseException) -> None:
        """Wake every fence waiter because a lane failed (R-CRASH)."""
        with self._cond:
            if self._aborted is None:
                self._aborted = exc
            self._cond.notify_all()

    def wait(self, lane: int | Lane, seq: int, stream: torch.cuda.Stream) -> None:
        """Order ``stream`` after lane ``lane``'s batch ``seq`` (F1)."""
        lane = Lane(lane)
        seq = int(seq)
        if seq <= 0:
            return
        with self._cond:
            if self._aborted is not None:
                raise RuntimeError(
                    f"two-lane: fence wait on {lane.name} seq {seq} aborted"
                ) from self._aborted
            event = self._events[lane].get(seq)
            if event is None:
                if seq <= self._pruned_to[lane]:
                    # Older than the kept window: that lane has recorded 64
                    # newer events since, so its work is far behind this one.
                    logger.warning_once(
                        "two-lane: fence on %s seq %d is older than the kept "
                        "window (%d events); skipping the device wait",
                        lane.name,
                        seq,
                        self._keep,
                    )
                    return
                self._cond.wait_for(
                    lambda: seq in self._events[lane]
                    or seq <= self._pruned_to[lane]
                    or self._aborted is not None
                )
                if seq not in self._events[lane] and self._aborted is not None:
                    raise RuntimeError(
                        f"two-lane: fence wait on {lane.name} seq {seq} aborted"
                    ) from self._aborted
                event = self._events[lane].get(seq)
        if event is not None:
            stream.wait_event(event)

    def has(self, lane: int | Lane, seq: int) -> bool:
        with self._cond:
            return int(seq) in self._events[Lane(lane)]

    def latest(self, lane: int | Lane) -> int:
        with self._cond:
            order = self._order[Lane(lane)]
            return order[-1] if order else 0

    def pending_batches(self) -> int:
        """Number of published events (both lanes); for tests and tracing."""
        with self._cond:
            return sum(len(order) for order in self._order.values())
