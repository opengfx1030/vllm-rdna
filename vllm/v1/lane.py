# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane engine thread affinity: lane identifiers and the lane thread-local.

Leaf module: stdlib only. ``Lane`` values equal the workspace lane indices
used by ``vllm.v1.worker.workspace`` (0 = decode, 1 = prefill).
"""

import enum
import threading


class Lane(enum.IntEnum):
    DECODE = 0  # default for any thread that never entered a lane
    PREFILL = 1  # equals the workspace lane index


NUM_LANES = 2
_tls = threading.local()
_lane_mode = False  # set once per process by the worker/engine


def enable_lane_mode() -> None:
    """Mark this process as running the two-lane engine."""
    global _lane_mode
    _lane_mode = True


def lane_mode_enabled() -> bool:
    """Whether the two-lane engine was enabled in this process."""
    return _lane_mode


def current_lane() -> Lane:
    """The calling thread's lane; decode for threads that never entered one."""
    return getattr(_tls, "lane", Lane.DECODE)


class lane_scope:
    """Context manager that sets and restores the calling thread's lane."""

    __slots__ = ("_lane", "_had_previous", "_previous")

    def __init__(self, lane: Lane):
        self._lane = Lane(lane)
        self._had_previous = False
        self._previous: Lane | None = None

    def __enter__(self) -> "lane_scope":
        self._had_previous = hasattr(_tls, "lane")
        if self._had_previous:
            self._previous = _tls.lane
        _tls.lane = self._lane
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._had_previous:
            _tls.lane = self._previous
        else:
            del _tls.lane
