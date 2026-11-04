# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph pool routing for CUDA graph capture under sleep mode and two-lane."""

from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast

from vllm.config import VllmConfig
from vllm.device_allocator import get_mem_allocator_instance
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.platforms import current_platform
from vllm.v1.lane import Lane, current_lane, lane_mode_enabled

if TYPE_CHECKING:
    from vllm.device_allocator.cumem import CuMemAllocator

_outside_cumem: ContextVar[bool] = ContextVar("capture_outside_cumem", default=False)
# Two-lane: the graph pool of every lane except D, created on first capture.
_lane_pools: dict[int, tuple[int, int]] = {}


def _capturing_lane() -> int | None:
    """The calling thread's lane if it owns graphs apart from D's, else None."""
    if not lane_mode_enabled():
        return None
    lane = int(current_lane())
    return None if lane == Lane.DECODE else lane


def lane_graph_key(key: Hashable) -> Hashable:
    """Graph-entry key for the calling thread's lane.

    Flag off and lane D keep ``key``. Lane P shares D's compiled model, so its
    piecewise entries are keyed apart: they are captured from P's own input
    buffers into P's pool, and replaying D's graph from P would read D's
    buffers (R-S5-4).
    """
    lane = _capturing_lane()
    return key if lane is None else (lane, key)


def lane_graph_pool(pool: tuple[int, int] | None) -> tuple[int, int] | None:
    """The pool a capture on the calling thread's lane goes to.

    Flag off and lane D keep ``pool`` (the process-wide pool). Lane P captures
    into its own pool: the lanes replay concurrently on their own streams, so
    their graph memory must not overlap (R-S5-4).
    """
    lane = _capturing_lane()
    if lane is None:
        return pool
    lane_pool = _lane_pools.get(lane)
    if lane_pool is None:
        lane_pool = _lane_pools[lane] = current_platform.graph_pool_handle()
    return lane_pool


@contextmanager
def capture_outside_cumem_pool() -> Iterator[None]:
    """CUDA graphs captured here stay out of cuMem, so that memory profiling can
    free its throwaway graphs normally."""
    token = _outside_cumem.set(True)
    try:
        yield
    finally:
        _outside_cumem.reset(token)


@contextmanager
def capture_pool(
    pool: tuple[int, int] | None, vllm_config: VllmConfig
) -> Iterator[tuple[int, int] | None]:
    """Yield the pool to capture into, the cuMem graph pool when sleep offloads
    graph memory, and point NCCL's graph allocator at it. Two-lane routes lane
    P's captures to its own pool (``lane_graph_pool``)."""
    if vllm_config.use_cumem_cudagraph_pool and not _outside_cumem.get():
        if _capturing_lane() is not None:
            raise RuntimeError(
                "two-lane: lane P cannot capture into the shared cuMem graph "
                "pool; it runs without CUDA graphs under sleep-mode graph offload"
            )
        allocator = cast("CuMemAllocator", get_mem_allocator_instance())
        with allocator.cudagraph_pool() as cumem_pool:
            set_graph_pool_id(cumem_pool)
            yield cumem_pool
    else:
        pool = lane_graph_pool(pool)
        set_graph_pool_id(pool or current_platform.graph_pool_handle())
        yield pool
