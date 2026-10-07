# Watch the FULL-graph private pool (0,1) for post-freeze allocation
# escapes. Any growth after graph capture is the skip-compiled leak
# that poisons TRUE FULL decode on gfx1030. Env: VLLM_RDNA_POOLWATCH=1.
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_RDNA_POOLWATCH", "0") == "1"
_GRAPH_POOL = (0, 1)
_baseline: int | None = None
_history_on = False


def maybe_enable() -> None:
    global _history_on
    if not _ENABLED or _history_on:
        return
    try:
        # Full allocation history is tens of tok/s of overhead on gfx1030
        # (serve30 1k c=1 decode 0.68 vs 23). Only enable with
        # VLLM_RDNA_POOLWATCH_HISTORY=1 when dumping a snapshot.
        if os.environ.get("VLLM_RDNA_POOLWATCH_HISTORY", "0") == "1":
            torch.cuda.memory._record_memory_history(max_entries=200_000)
            _history_on = True
            logger.info("[poolwatch] memory history recording on")
        else:
            _history_on = True
            logger.info("[poolwatch] graph-pool byte watch on (no history)")
    except Exception as e:
        logger.warning("[poolwatch] record_memory_history failed: %s", e)


def graph_pool_alloc_bytes() -> int:
    total = 0
    try:
        for seg in torch.cuda.memory_snapshot():
            if tuple(seg.get("segment_pool_id", ())) == _GRAPH_POOL:
                total += int(seg.get("allocated_size", 0))
    except Exception:
        pass
    return total


def set_baseline(tag: str) -> None:
    global _baseline
    if not _ENABLED:
        return
    maybe_enable()
    _baseline = graph_pool_alloc_bytes()
    logger.info(
        "[poolwatch] baseline %s graph_pool%s alloc=%d bytes",
        tag,
        _GRAPH_POOL,
        _baseline,
    )


def check(tag: str) -> None:
    global _baseline
    if not _ENABLED or _baseline is None:
        return
    cur = graph_pool_alloc_bytes()
    if cur <= _baseline:
        return
    path = f"/tmp/poolwatch-{tag}-{os.getpid()}-{int(time.time())}.pkl"
    try:
        torch.cuda.memory._dump_snapshot(path)
    except Exception as e:
        path = f"dump-failed:{e}"
    logger.warning(
        "[poolwatch] graph pool%s GREW at %s: %d -> %d (+%d B); snapshot=%s",
        _GRAPH_POOL,
        tag,
        _baseline,
        cur,
        cur - _baseline,
        path,
    )
    # Ratchet the baseline so we log each distinct growth step once.
    _baseline = cur


_dumped = False


def maybe_dump_on_replay(tag: str) -> None:
    global _dumped
    if _dumped or not _ENABLED or not _history_on:
        return
    if os.environ.get("VLLM_RDNA_POOLWATCH_DUMP_ON_REPLAY", "0") != "1":
        return
    _dumped = True
    path = f"/tmp/poolwatch-replay-{os.getpid()}-{int(time.time())}.pkl"
    try:
        torch.cuda.memory._dump_snapshot(path)
        logger.warning("[poolwatch] replay dump %s at %s", path, tag)
    except Exception as e:
        logger.warning("[poolwatch] replay dump failed: %s", e)
