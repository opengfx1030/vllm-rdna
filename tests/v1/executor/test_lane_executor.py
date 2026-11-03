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
from vllm.v1.lane import NUM_LANES, Lane


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


def test_lane_transport_round_trips_through_real_message_queue():
    """The per-lane response MQs and reader threads, with real shm queues."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    writers = [
        MessageQueue(1, 1, max_chunk_bytes=1 << 20, max_chunks=4)
        for _ in range(NUM_LANES)
    ]
    executor = _multiproc()
    executor.world_size = 1
    executor.parallel_config = SimpleNamespace(
        nnodes_within_dp=1,
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    executor.workers = [
        SimpleNamespace(
            lane_response_handles=[writer.export_handle() for writer in writers]
        )
    ]
    init_errors: list[BaseException] = []

    def init_transport() -> None:
        try:
            executor._init_lane_transport()
        except BaseException as e:  # pragma: no cover - surfaced below
            init_errors.append(e)

    try:
        thread = threading.Thread(target=init_transport)
        thread.start()
        for writer in writers:
            writer.wait_until_ready()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert not init_errors

        future = executor._lane_rpc("execute_model", (_output(int(Lane.DECODE)),), 0)
        writers[int(Lane.DECODE)].enqueue(
            (WorkerProc.ResponseStatus.SUCCESS, "lane-output")
        )
        assert future.result(timeout=5.0) == "lane-output"
    finally:
        executor._shutdown_lane_transport()
        for writer in writers:
            writer.shutdown()


class _FakeMessageQueue:
    created: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).created.append((args, kwargs))

    @classmethod
    def create_from_handle(cls, handle, rank):
        cls.created.append(((handle, rank), {}))
        return cls()


def _lane_mq_config(two_lane: bool) -> Any:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(two_lane=two_lane),
        parallel_config=SimpleNamespace(nnodes_within_dp=1),
    )


def test_flag_off_creates_no_lane_response_queues(monkeypatch):
    """I9: the worker's lane response MQs exist only with the flag on."""
    import vllm.v1.executor.multiproc_executor as mp

    _FakeMessageQueue.created = []
    monkeypatch.setattr(mp, "MessageQueue", _FakeMessageQueue)
    worker: Any = SimpleNamespace(worker=SimpleNamespace(rank=0))

    WorkerProc._init_message_queues(worker, None, _lane_mq_config(False))

    assert worker.lane_response_mqs is None
    assert worker._lane_response_locks is None
    lane_mqs = [
        kwargs
        for _args, kwargs in _FakeMessageQueue.created
        if kwargs.get("max_chunks") == 8
    ]
    assert lane_mqs == []


def test_flag_on_creates_two_lane_response_queues(monkeypatch):
    import vllm.v1.executor.multiproc_executor as mp

    _FakeMessageQueue.created = []
    monkeypatch.setattr(mp, "MessageQueue", _FakeMessageQueue)
    worker: Any = SimpleNamespace(worker=SimpleNamespace(rank=0))

    WorkerProc._init_message_queues(worker, None, _lane_mq_config(True))

    assert len(worker.lane_response_mqs) == NUM_LANES
    assert len(worker._lane_response_locks) == NUM_LANES
    lane_mqs = [
        kwargs
        for _args, kwargs in _FakeMessageQueue.created
        if kwargs.get("max_chunks") == 8
    ]
    assert len(lane_mqs) == NUM_LANES


def test_flag_off_starts_no_lane_reader_threads(monkeypatch):
    executor = _multiproc(two_lane=False)
    calls: list[bool] = []
    monkeypatch.setattr(executor, "_init_lane_transport", lambda: calls.append(True))

    executor._maybe_init_lane_transport()

    assert calls == []
    assert executor._lane_reader_threads == []


def test_flag_on_starts_lane_transport_once(monkeypatch):
    executor = _multiproc(two_lane=True)
    calls: list[bool] = []
    monkeypatch.setattr(executor, "_init_lane_transport", lambda: calls.append(True))

    executor._maybe_init_lane_transport()

    assert calls == [True]


def test_flag_off_ready_payload_has_no_lane_key():
    import vllm.v1.executor.multiproc_executor as mp

    worker: Any = SimpleNamespace(
        worker_response_mq=SimpleNamespace(export_handle=lambda: "handle"),
        peer_response_handles=[],
        lane_response_mqs=None,
    )
    payload = mp._worker_ready_payload(worker)
    assert "lane_response_handles" not in payload

    worker.lane_response_mqs = [
        SimpleNamespace(export_handle=lambda i=i: f"lane{i}") for i in range(NUM_LANES)
    ]
    payload = mp._worker_ready_payload(worker)
    assert payload["lane_response_handles"] == ["lane0", "lane1"]
