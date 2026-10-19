# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async micro-batcher. Model work runs on one worker thread."""

import asyncio
import contextlib
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from vllm.entrypoints.systemone.errors import SystemOneError
from vllm.logger import init_logger

logger = init_logger(__name__)

_T = TypeVar("_T")
_R = TypeVar("_R")


class MicroBatcher:
    """Coalesce ``submit`` calls and run each batch on a single worker thread.

    The event loop that serves ``/v1/chat/completions`` only waits on futures.
    A full queue returns 429. A request that outlives ``timeout_s`` returns 504.
    """

    def __init__(
        self,
        decide_batch: Callable[[Sequence[_T]], list[_R]],
        *,
        max_batch: int,
        max_wait_ms: float,
        max_queue: int,
        timeout_s: float,
    ) -> None:
        self._decide_batch = decide_batch
        self._max_batch = max_batch
        self._max_wait_s = max_wait_ms / 1000.0
        self._timeout_s = timeout_s
        self._queue: asyncio.Queue[tuple[_T, asyncio.Future[_R]]] = asyncio.Queue(
            maxsize=max_queue
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="systemone"
        )
        self._task: asyncio.Task[None] | None = None

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def shutdown(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def submit(self, item: _T) -> _R:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_R] = loop.create_future()
        try:
            self._queue.put_nowait((item, future))
        except asyncio.QueueFull as exc:
            raise SystemOneError("systemone queue is full; retry later", 429) from exc
        try:
            return await asyncio.wait_for(future, self._timeout_s)
        except TimeoutError as exc:
            raise SystemOneError("systemone request timed out", 504) from exc

    async def _loop(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            if self._max_batch > 1:
                batch.extend(self._drain(self._max_batch - 1))
            if self._max_batch > 1 and self._max_wait_s > 0:
                deadline = time.monotonic() + self._max_wait_s
                while len(batch) < self._max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._queue.get(), remaining)
                    except TimeoutError:
                        break
                    batch.append(nxt)
                    batch.extend(self._drain(self._max_batch - len(batch)))
            items = [item for item, _future in batch]
            futures = [future for _item, future in batch]
            try:
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._decide_batch, items
                )
            except Exception as exc:
                logger.exception("systemone batch failed")
                wrapped = (
                    exc
                    if isinstance(exc, SystemOneError)
                    else SystemOneError("systemone backend failed", 500)
                )
                for future in futures:
                    if not future.done():
                        future.set_exception(wrapped)
                continue
            if len(results) != len(futures):
                err = SystemOneError(
                    "systemone backend returned the wrong number of answers",
                    500,
                )
                for future in futures:
                    if not future.done():
                        future.set_exception(err)
                continue
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

    def _drain(self, limit: int) -> list[tuple[_T, asyncio.Future[_R]]]:
        drained: list[tuple[_T, asyncio.Future[_R]]] = []
        while limit > 0 and not self._queue.empty():
            drained.append(self._queue.get_nowait())
            limit -= 1
        return drained
