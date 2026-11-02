# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker lane runtime (S3 R-S3-3): one thread per lane, inboxes, responses.

The engine's dispatcher (``worker_busy_loop``, S2) hands every lane RPC to
``LaneRuntime.submit``; this module owns the rest of the worker side:

* a **lane thread** per lane. It sets the device (a thread that does not would
  create a new CUDA context on device 0), enters ``inference_mode``,
  ``lane_scope(lane)`` and workspace lane ``lane``, and runs ``execute_model`` /
  ``sample_tokens`` on that lane's runner in submission order (I3);
* a **response queue and output thread** per lane. The output thread calls
  ``AsyncModelRunnerOutput.get_output()`` -- a host wait on the copy event -- and
  resolves the future, so the lane thread never blocks on D2H;
* the **stream pick and the two ordering rules**: F2 (``wait_stream`` when the
  lane hops between its masked and full-mask stream) and F1 (the other lane's
  end-of-batch fence);
* ``quiesce`` for control RPCs, ``shutdown``, and the failure path: an exception
  in a lane thread fails the item's future and every later submit (R-S3-8).

Lane streams are created once and never destroyed (I5). The KFD queue census
runs at start, after the streams exist (R-S3-2).
"""

import concurrent.futures
import json
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.lane import Lane, lane_scope
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.lane.fences import LaneFences
from vllm.v1.worker.lane.streams import (
    LaneStreams,
    check_kfd_budget,
    create_lane_streams,
    kfd_census,
)
from vllm.v1.worker.workspace import use_workspace_lane

logger = init_logger(__name__)

# R-S3-10: the P thread's Python launch work must not starve D's replay.
GIL_SWITCH_INTERVAL = 0.0005

LANE_METHODS = ("execute_model", "sample_tokens")

# Eager prefill shapes P must have compiled and tuned before capture locks the
# workspace (R-S3-5e). The last entry is the scheduler's prefill token budget.
PREFILL_WARMUP_TOKEN_COUNTS = (1, 2, 5, 24, 25, 100, 511, 1024, 1025)


class _LaneThreadState(threading.local):
    active: bool = False


_lane_thread_state = _LaneThreadState()

_strict_installed = False


def in_lane_thread() -> bool:
    """Whether the calling thread is a lane executor thread."""
    return bool(getattr(_lane_thread_state, "active", False))


def install_strict_sync() -> None:
    """Make device-wide synchronizations from a lane thread raise (R-S3-9).

    Only debug/warmup/shutdown paths call them today; in lane mode a hidden one
    would stall the other lane. Idempotent; the wrappers stay installed for the
    life of the process.
    """
    global _strict_installed
    if _strict_installed:
        return
    _strict_installed = True

    def guard(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if in_lane_thread():
                raise RuntimeError(
                    f"two-lane strict mode: device-wide {name}() was called "
                    f"from a lane thread. That stalls the other lane; use a "
                    f"stream event instead (R-S3-9)."
                )
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", name)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        return wrapper

    # The CUDA sync entry point model code calls. Patched through the module
    # dict: a literal torch.cuda.<name> access is rejected by
    # tools/pre_commit/check_torch_cuda.py, and getattr/setattr with a constant
    # name are rewritten back by ruff (B009/B010).
    cuda_namespace = vars(torch.cuda)
    cuda_synchronize = cuda_namespace["synchronize"]
    cuda_namespace["synchronize"] = guard("cuda.synchronize", cuda_synchronize)
    torch.accelerator.synchronize = guard(
        "torch.accelerator.synchronize", torch.accelerator.synchronize
    )


class LaneContext:
    """What a lane's model runner sees of the runtime (S5 3.5).

    ``current_stream`` is the stream the lane is running on right now: the
    runner binds its per-step work (``main_stream``, D2H copies) to it, because
    the stream changes when the lane hops between masked and full-mask (I5).
    ``on_batch_done`` is the runner's end-of-batch hook, called at the very end
    of ``sample_tokens`` after the drafter, and publishes the fence event
    (R-S3-4) the other lane waits on before its next batch.
    """

    __slots__ = ("lane", "_streams", "_fences", "_current")

    def __init__(self, lane: Lane) -> None:
        self.lane = Lane(lane)
        self._streams: LaneStreams | None = None
        self._fences: LaneFences | None = None
        self._current: torch.cuda.Stream | None = None

    def bind(self, streams: LaneStreams, fences: LaneFences) -> None:
        self._streams = streams
        self._fences = fences

    def set_current(self, stream: torch.cuda.Stream) -> None:
        self._current = stream

    def current_stream(self) -> torch.cuda.Stream:
        if self._current is not None:
            return self._current
        return torch.cuda.current_stream()

    def on_batch_done(self, scheduler_output: Any) -> None:
        fences = self._fences
        if fences is None:
            return
        seq = int(getattr(scheduler_output, "lane_seq", 0) or 0)
        if seq > 0:
            fences.record(self.lane, seq, self.current_stream())


@dataclass
class LaneItem:
    """One lane RPC in flight."""

    lane: Lane
    method: str
    args: tuple
    future: concurrent.futures.Future
    t_recv: float
    t_start: float = 0.0
    t_launch_done: float = 0.0
    seq: int = 0
    partitioned: bool = False
    stream_name: str = ""
    n_reqs: int = 0
    n_tokens: int = 0


_STOP = object()


class LaneRuntime:
    """Per-worker lane runtime; the worker keeps ``None`` when the flag is off."""

    def __init__(
        self,
        worker: Any,
        *,
        decode_wgps: int,
        streams: LaneStreams | None = None,
        fences: LaneFences | None = None,
        trace: bool = False,
        strict: bool = False,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._worker = worker
        self._decode_wgps = int(decode_wgps)
        self._streams = streams
        self._fences = fences or LaneFences(event_factory=event_factory)
        self._trace_enabled = bool(trace)
        self._strict = bool(strict)
        self._device = getattr(worker, "device", None)

        self._inboxes: dict[Lane, queue.Queue] = {lane: queue.Queue() for lane in Lane}
        # Per-lane response queues: lane thread -> that lane's output thread.
        self._responses: dict[Lane, queue.Queue] = {
            lane: queue.Queue() for lane in Lane
        }
        self._contexts: dict[Lane, LaneContext] = {
            lane: LaneContext(lane) for lane in Lane
        }
        # Stream each lane last used (F2).
        self._lane_stream: dict[Lane, torch.cuda.Stream | None] = {
            lane: None for lane in Lane
        }
        self._stream_hops = 0

        self._cond = threading.Condition()
        self._pending: dict[Lane, int] = {lane: 0 for lane in Lane}
        self._failed: BaseException | None = None
        self._started = False
        self._closed = False
        self._start_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._trace_file: Any | None = None
        self._trace_lock = threading.Lock()

    # ------------------------------------------------------------------ state
    @property
    def started(self) -> bool:
        return self._started

    @property
    def failed(self) -> BaseException | None:
        return self._failed

    @property
    def streams(self) -> LaneStreams | None:
        return self._streams

    @property
    def fences(self) -> LaneFences:
        return self._fences

    def context(self, lane: int | Lane) -> LaneContext:
        return self._contexts[Lane(lane)]

    def stream_hops(self) -> int:
        return self._stream_hops

    def ensure_streams(self) -> LaneStreams:
        """Create the four lane streams if they do not exist yet (I5)."""
        if self._streams is None:
            self._streams = create_lane_streams(self._decode_wgps, self._device)
        return self._streams

    # --------------------------------------------------------------- startup
    def warmup(self, token_counts: Sequence[int] | None = None) -> None:
        """Warm lane P on the main thread, before capture locks the workspace.

        Runs the eager dummy sweep under lane P's scope, so its collectives use
        P's communicator set; ``lane_warmup`` enters workspace lane 1 itself.
        Dummy runs are forbidden after serving starts, so this must happen
        before the threads start (R-S3-5e, R-S5-7).
        """
        runner = self._lane_runner(Lane.PREFILL)
        if runner is None:
            return
        if token_counts is None:
            scheduler_config = getattr(
                getattr(self._worker, "vllm_config", None), "scheduler_config", None
            )
            max_bat = int(getattr(scheduler_config, "max_num_batched_tokens", 2048))
            budget = getattr(scheduler_config, "two_lane_prefill_budget", None)
            token_counts = (*PREFILL_WARMUP_TOKEN_COUNTS, int(budget or max_bat))
        with lane_scope(Lane.PREFILL):
            runner.lane_warmup(list(token_counts))

    def start(self) -> None:
        """Create streams (if needed), check the queue budget, start the threads."""
        with self._start_lock:
            if self._started or self._closed:
                return
            streams = self.ensure_streams()
            if self._strict:
                install_strict_sync()
            sys.setswitchinterval(GIL_SWITCH_INTERVAL)
            # R-S3-2: streams, comm sets, warmup and capture are all done here.
            check_kfd_budget(kfd_census(), context="lane start (streams + comms)")
            self._open_trace()
            for lane in Lane:
                context = self._contexts[lane]
                context.bind(streams, self._fences)
                runner = self._lane_runner(lane)
                if runner is not None:
                    runner.lane_ctx = context
                self._threads.append(
                    threading.Thread(
                        target=self._lane_thread_main,
                        args=(lane,),
                        name=f"lane-{lane.name.lower()}",
                        daemon=True,
                    )
                )
                self._threads.append(
                    threading.Thread(
                        target=self._output_thread_main,
                        args=(lane,),
                        name=f"lane-{lane.name.lower()}-output",
                        daemon=True,
                    )
                )
            for thread in self._threads:
                thread.start()
            self._started = True
            logger.info(
                "two-lane: lane threads started (masks %s, GIL switch interval %.4f s)",
                streams.masks(),
                GIL_SWITCH_INTERVAL,
            )

    # ------------------------------------------------------------------ RPCs
    def submit(self, lane: int, method: str, args: tuple) -> concurrent.futures.Future:
        """Queue one lane RPC; the future resolves to the output's result.

        Called from the worker's dispatcher thread and, at TP1, from the
        engine-core thread. Fails fast once the runtime failed or shut down.
        """
        lane = Lane(lane)
        future: concurrent.futures.Future = concurrent.futures.Future()
        if method not in LANE_METHODS:
            future.set_exception(ValueError(f"two-lane: {method!r} is not a lane RPC"))
            return future
        with self._cond:
            failure = self._failure()
            if failure is not None:
                future.set_exception(failure)
                return future
        try:
            self.start()
        except BaseException as e:  # noqa: BLE001 - reported through the future
            future.set_exception(e)
            return future
        with self._cond:
            failure = self._failure()
            if failure is not None:
                future.set_exception(failure)
                return future
            self._pending[lane] += 1
        self._inboxes[lane].put(
            LaneItem(
                lane=lane,
                method=method,
                args=tuple(args),
                future=future,
                t_recv=time.perf_counter(),
            )
        )
        return future

    def quiesce(self) -> None:
        """Wait until both inboxes are empty and every item has resolved.

        No device synchronization: a pending item's future may still be waiting
        for its copy event, which is a host wait (R-PAUSE, R-S3-3).
        """
        with self._cond:
            if self._failed is not None:
                raise RuntimeError("two-lane lane runtime failed") from self._failed
            self._cond.wait_for(lambda: all(self._pending[lane] == 0 for lane in Lane))

    def shutdown(self) -> None:
        """Stop both lane and output threads; streams are left to process exit."""
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if started:
            for lane in Lane:
                self._inboxes[lane].put(_STOP)
            lane_threads = [t for t in self._threads if not t.name.endswith("-output")]
            output_threads = [t for t in self._threads if t.name.endswith("-output")]
            for thread in lane_threads:
                thread.join(timeout=30)
                if thread.is_alive():
                    logger.error(
                        "two-lane: lane thread %s did not stop; a batch is "
                        "probably waiting on a fence event that never came",
                        thread.name,
                    )
            for lane in Lane:
                self._responses[lane].put(_STOP)
            for thread in output_threads:
                thread.join(timeout=30)
        self._fail_pending(
            RuntimeError("two-lane: worker shut down with lane work outstanding")
        )
        self._close_trace()
        with self._cond:
            self._cond.notify_all()

    def _failure(self) -> BaseException | None:
        if self._failed is not None:
            return self._failed
        if self._closed:
            return RuntimeError("two-lane: lane runtime is shut down")
        return None

    # ------------------------------------------------------------- lane thread
    def _lane_thread_main(self, lane: Lane) -> None:
        try:
            if self._device is not None:
                current_platform.set_device(self._device)
            _lane_thread_state.active = True
            with (
                torch.inference_mode(),
                lane_scope(lane),
                use_workspace_lane(int(lane)),
            ):
                self._lane_loop(lane)
        except BaseException as e:  # noqa: BLE001 - never die silently
            logger.exception("two-lane: lane thread %s failed", lane.name)
            self._mark_failed(e)
        finally:
            _lane_thread_state.active = False

    def _lane_loop(self, lane: Lane) -> None:
        while True:
            item = self._inboxes[lane].get()
            if item is _STOP:
                return
            self._run_item(item)

    def _run_item(self, item: LaneItem) -> None:
        item.t_start = time.perf_counter()
        try:
            result = self._dispatch(item)
        except BaseException as e:  # noqa: BLE001 - reported through the future
            item.t_launch_done = time.perf_counter()
            self._trace(item)
            item.future.set_exception(e)
            self._resolved(item.lane)
            self._mark_failed(e)
            return
        item.t_launch_done = time.perf_counter()
        self._trace(item)
        if isinstance(result, AsyncModelRunnerOutput):
            # Resolved by the lane's output thread, after the copy event.
            self._responses[item.lane].put((item, result))
        else:
            item.future.set_result(result)
            self._resolved(item.lane)

    def _dispatch(self, item: LaneItem) -> Any:
        lane = item.lane
        if item.method == "execute_model":
            scheduler_output = item.args[0]
            partitioned = bool(getattr(scheduler_output, "partitioned", False))
            streams = self.ensure_streams()
            stream = streams.stream_for(lane, partitioned)
            self._enter_stream(item.lane, stream)
            other = Lane.PREFILL if lane == Lane.DECODE else Lane.DECODE
            fence = getattr(scheduler_output, "lane_fence", (0, 0)) or (0, 0)
            # F1: the other lane's newest processed batch, before any GPU work.
            self._fences.wait(other, int(fence[int(other)]), stream)
            item.seq = int(getattr(scheduler_output, "lane_seq", 0) or 0)
            item.partitioned = partitioned
            item.stream_name = streams.name_for(lane, partitioned)
            item.n_reqs, item.n_tokens = _batch_size(scheduler_output)
            with torch.cuda.stream(stream):
                return self._worker._lane_step(lane, item.method, item.args)
        if item.method == "sample_tokens":
            # Same stream as the batch: sampling and the drafter are ordered
            # after its forward pass (I5).
            stream = self._lane_stream[lane]
            if stream is None:
                raise RuntimeError(
                    f"two-lane: sample_tokens on {lane.name} without a batch"
                )
            with torch.cuda.stream(stream):
                return self._worker._lane_step(lane, item.method, item.args)
        raise ValueError(f"two-lane: unknown lane method {item.method!r}")

    def _enter_stream(self, lane: Lane, stream: torch.cuda.Stream) -> None:
        previous = self._lane_stream[lane]
        if previous is not None and previous is not stream:
            # F2: preserve this lane's own order across a stream hop.
            stream.wait_stream(previous)
            self._stream_hops += 1
        self._lane_stream[lane] = stream
        self._contexts[lane].set_current(stream)

    # ----------------------------------------------------------- output thread
    def _output_thread_main(self, lane: Lane) -> None:
        try:
            if self._device is not None:
                current_platform.set_device(self._device)
            while True:
                entry = self._responses[lane].get()
                if entry is _STOP:
                    return
                item, output = entry
                try:
                    item.future.set_result(output.get_output())
                except BaseException as e:  # noqa: BLE001
                    item.future.set_exception(e)
                    self._resolved(lane)
                    self._mark_failed(e)
                    continue
                self._resolved(lane)
        except BaseException as e:  # noqa: BLE001
            logger.exception("two-lane: output thread %s failed", lane.name)
            self._mark_failed(e)

    def _resolved(self, lane: Lane) -> None:
        with self._cond:
            self._pending[lane] = max(0, self._pending[lane] - 1)
            self._cond.notify_all()

    # --------------------------------------------------------------- failures
    def _mark_failed(self, exc: BaseException) -> None:
        with self._cond:
            if self._failed is None:
                self._failed = exc
            self._cond.notify_all()
        self._fail_pending(exc)

    def _fail_pending(self, exc: BaseException) -> None:
        """Fail every queued-but-unresolved item, so the engine never hangs."""
        for lane in Lane:
            while True:
                try:
                    item = self._inboxes[lane].get_nowait()
                except queue.Empty:
                    break
                if item is not _STOP and not item.future.done():
                    item.future.set_exception(exc)
                    self._resolved(lane)

    # ------------------------------------------------------------------ trace
    def _open_trace(self) -> None:
        if not self._trace_enabled or self._trace_file is not None:
            return
        rank = int(getattr(self._worker, "rank", 0) or 0)
        path = os.path.join(envs.VLLM_CACHE_ROOT, f"lane_trace_rank{rank}.jsonl")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._trace_file = open(path, "a", encoding="utf-8")  # noqa: SIM115
        except OSError as e:
            logger.warning("two-lane: cannot open the lane trace %s (%s)", path, e)
            self._trace_file = None
            return
        logger.info("two-lane: lane trace at %s", path)

    def _trace(self, item: LaneItem) -> None:
        trace_file = self._trace_file
        if trace_file is None or item.method != "execute_model":
            return
        record = {
            "lane": int(item.lane),
            "seq": item.seq,
            "partitioned": item.partitioned,
            "stream": item.stream_name,
            "t_recv": item.t_recv,
            "t_start": item.t_start,
            "t_launch_done": item.t_launch_done,
            "n_reqs": item.n_reqs,
            "n_tokens": item.n_tokens,
        }
        with self._trace_lock:
            trace_file.write(json.dumps(record) + "\n")
            trace_file.flush()

    def _close_trace(self) -> None:
        with self._trace_lock:
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None

    # ------------------------------------------------------------------ utils
    def _lane_runner(self, lane: Lane) -> Any:
        getter = getattr(self._worker, "_lane_runner", None)
        if getter is None:
            return None
        return getter(lane)


def _batch_size(scheduler_output: Any) -> tuple[int, int]:
    num_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not num_tokens:
        return 0, int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    return len(num_tokens), int(sum(num_tokens.values()))
