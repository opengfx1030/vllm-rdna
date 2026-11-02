# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the executor half of the two-lane engine (S2, R-S2-3/4/5).

The worker lane runtime is the S3 contract from 3.4 and is mocked here; these
tests pin the wire format, the per-lane response FIFO, the quiesce-before-
control-RPC rule and the flag-off routing.
"""

import threading
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.lane import Lane


def _output(lane: int, tokens: int = 1) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req": tokens},
        total_num_scheduled_tokens=tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        lane=lane,
    )


class _FakeLaneRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, tuple]] = []
        self.events: list[tuple] = []
        self.futures: list[Future] = []
        self.quiesce_calls = 0

    def submit(self, lane: int, method: str, args: tuple) -> Future:
        self.calls.append((int(lane), method, args))
        self.events.append(("submit", int(lane), method))
        future: Future = Future()
        self.futures.append(future)
        return future

    def quiesce(self) -> None:
        self.quiesce_calls += 1
        self.events.append(("quiesce",))


class _RecordingMQ:
    def __init__(self) -> None:
        self.payloads: list = []
        self.shutdown_calls = 0

    def enqueue(self, payload) -> None:
        self.payloads.append(payload)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _ScriptedMQ:
    def __init__(self, items: list) -> None:
        self.items = list(items)

    def dequeue(self, indefinite: bool = False):
        assert indefinite
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _ScriptedBroadcastMQ:
    def __init__(self, requests: list) -> None:
        self.requests = list(requests)
        self.enqueued: list = []

    def dequeue(self, indefinite: bool = False):
        assert indefinite
        if not self.requests:
            raise _ExitLoop
        return self.requests.pop(0)

    def enqueue(self, payload) -> None:
        self.enqueued.append(payload)


class _ExitLoop(Exception):
    pass


def _uniproc(two_lane: bool = True) -> tuple[Any, _FakeLaneRuntime]:
    runtime = _FakeLaneRuntime()
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.two_lane = two_lane
    executor.driver_worker = SimpleNamespace(
        lane_runtime=runtime, control=lambda: "control-result"
    )
    return executor, runtime


def _multiproc(two_lane: bool = True) -> Any:
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.two_lane = two_lane
    executor.is_failed = False
    executor.rpc_broadcast_mq = _RecordingMQ()
    executor.output_rank = 1
    executor.kv_output_aggregator = None
    executor.ec_output_aggregator = None
    executor._lane_output_rank = 1
    executor._lane_futures = [deque(), deque()]
    executor._lane_future_locks = [threading.Lock(), threading.Lock()]
    executor._lane_response_mqs = [[], []]
    executor._lane_reader_threads = []
    return executor


def _worker_proc(runtime, requests: list, *, rank: int = 0) -> tuple[Any, list, list]:
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = rank
    worker_proc.worker = SimpleNamespace(
        lane_runtime=runtime, control=lambda: "control-result"
    )
    worker_proc.rpc_broadcast_mq = _ScriptedBroadcastMQ(requests)
    worker_proc.lane_response_mqs = [_RecordingMQ(), _RecordingMQ()]
    worker_proc._lane_response_locks = [threading.Lock(), threading.Lock()]
    outputs: list = []
    worker_proc.handle_output = outputs.append
    return worker_proc, outputs, worker_proc.lane_response_mqs


def test_supports_two_lane_matrix():
    assert Executor.supports_two_lane() is False
    assert UniProcExecutor.supports_two_lane() is True
    assert MultiprocExecutor.supports_two_lane() is True

    class _RayLike(MultiprocExecutor):
        uses_ray = True

    assert _RayLike.supports_two_lane() is False


def test_uniproc_execute_model_submits_to_lane_runtime():
    executor, runtime = _uniproc()
    scheduler_output = _output(int(Lane.PREFILL))

    future = executor.execute_model(scheduler_output, non_block=True)

    assert future is runtime.futures[0]
    assert runtime.calls == [(int(Lane.PREFILL), "execute_model", (scheduler_output,))]


def test_uniproc_sample_tokens_passes_lane():
    executor, runtime = _uniproc()

    future = executor.sample_tokens(None, non_block=True, lane=int(Lane.DECODE))

    assert future is runtime.futures[0]
    assert runtime.calls == [(int(Lane.DECODE), "sample_tokens", (None,))]


def test_uniproc_control_rpc_quiesces_first():
    executor, runtime = _uniproc()

    result = executor.collective_rpc("control")

    assert result == ["control-result"]
    assert runtime.events == [("quiesce",)]


def test_uniproc_flag_off_keeps_baseline_path():
    executor, runtime = _uniproc(two_lane=False)

    result = executor.collective_rpc("control")

    assert result == ["control-result"]
    assert runtime.quiesce_calls == 0
    assert runtime.calls == []


def test_multiproc_lane_rpc_wire_format():
    executor = _multiproc()
    scheduler_output = _output(int(Lane.PREFILL))

    future = executor.execute_model(scheduler_output, non_block=True)

    assert executor.rpc_broadcast_mq.payloads == [
        ("execute_model", (scheduler_output,), {"lane": int(Lane.PREFILL)}, 1)
    ]
    assert list(executor._lane_futures[int(Lane.PREFILL)]) == [future]
    assert not future.done()


def test_multiproc_sample_tokens_wire_format():
    executor = _multiproc()

    future = executor.sample_tokens(None, non_block=True, lane=int(Lane.DECODE))

    assert executor.rpc_broadcast_mq.payloads == [
        ("sample_tokens", (None,), {"lane": int(Lane.DECODE)}, 1)
    ]
    assert list(executor._lane_futures[int(Lane.DECODE)]) == [future]


def test_multiproc_rejects_invalid_lane():
    executor = _multiproc()
    with pytest.raises(ValueError, match="Invalid lane"):
        executor.sample_tokens(None, non_block=True, lane=7)


def test_multiproc_flag_off_keeps_collective_rpc(monkeypatch):
    executor = _multiproc(two_lane=False)
    executor.collective_rpc = lambda *a, **k: "baseline"

    assert executor.execute_model(_output(0)) == "baseline"
    assert executor.sample_tokens(None) == "baseline"
    assert executor.rpc_broadcast_mq.payloads == []


def test_lane_reader_fulfils_futures_in_fifo_order():
    executor = _multiproc()
    l0 = int(Lane.DECODE)
    executor._lane_output_rank = 0
    executor._lane_response_mqs[l0] = [
        _ScriptedMQ(
            [
                (WorkerProc.ResponseStatus.SUCCESS, "first"),
                (WorkerProc.ResponseStatus.SUCCESS, "second"),
                RuntimeError("cancelled"),
            ]
        )
    ]
    first: Future = Future()
    second: Future = Future()
    executor._lane_futures[l0].append(first)
    executor._lane_futures[l0].append(second)

    executor._lane_reader_loop(l0)

    assert first.result() == "first"
    assert second.result() == "second"
    assert not executor._lane_futures[l0]


def test_lane_reader_surfaces_worker_failure():
    executor = _multiproc()
    l0 = int(Lane.DECODE)
    executor._lane_output_rank = 0
    executor._lane_response_mqs[l0] = [
        _ScriptedMQ(
            [
                (WorkerProc.ResponseStatus.FAILURE, "boom"),
                RuntimeError("cancelled"),
            ]
        )
    ]
    pending: Future = Future()
    executor._lane_futures[l0].append(pending)

    executor._lane_reader_loop(l0)

    with pytest.raises(RuntimeError, match="boom"):
        pending.result()


def test_worker_dispatches_lane_rpc_and_quiesces_for_control():
    runtime = _FakeLaneRuntime()
    scheduler_output = _output(int(Lane.PREFILL))
    requests = [
        ("execute_model", (scheduler_output,), {"lane": int(Lane.PREFILL)}, 0),
        ("control", (), {}, None),
    ]
    worker_proc, outputs, lane_mqs = _worker_proc(runtime, requests)

    with pytest.raises(_ExitLoop):
        worker_proc.worker_busy_loop()

    # The lane RPC went to the lane runtime, not to the control path.
    assert runtime.calls == [(int(Lane.PREFILL), "execute_model", (scheduler_output,))]
    # The control RPC quiesced first, then ran inline.
    assert runtime.events == [
        ("submit", int(Lane.PREFILL), "execute_model"),
        ("quiesce",),
    ]
    assert outputs == ["control-result"]

    # The lane result is enqueued on that lane's queue only.
    runtime.futures[0].set_result("lane-result")
    lane_mq = lane_mqs[int(Lane.PREFILL)]
    assert lane_mq.payloads == [(WorkerProc.ResponseStatus.SUCCESS, "lane-result")]
    assert lane_mqs[int(Lane.DECODE)].payloads == []


def test_worker_lane_failure_enqueued_as_failure():
    runtime = _FakeLaneRuntime()
    requests = [("sample_tokens", (None,), {"lane": 0}, 0)]
    worker_proc, _, lane_mqs = _worker_proc(runtime, requests)

    with pytest.raises(_ExitLoop):
        worker_proc.worker_busy_loop()

    runtime.futures[0].set_exception(RuntimeError("lane exploded"))
    status, message = lane_mqs[0].payloads[0]
    assert status is WorkerProc.ResponseStatus.FAILURE
    assert "lane exploded" in message


def test_worker_lane_rpc_without_runtime_raises():
    requests = [("execute_model", (_output(0),), {"lane": 0}, None)]
    worker_proc, _, _ = _worker_proc(None, requests)

    with pytest.raises(RuntimeError, match="no lane runtime"):
        worker_proc.worker_busy_loop()


def test_shutdown_lane_transport_fails_pending_futures():
    executor = _multiproc()
    l0 = int(Lane.DECODE)
    mq = _RecordingMQ()
    executor._lane_response_mqs[l0] = [mq]
    pending: Future = Future()
    executor._lane_futures[l0].append(pending)

    executor._shutdown_lane_transport()

    assert mq.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="Executor shut down"):
        pending.result()
    assert executor._lane_response_mqs == []
