# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker lane runtime with fake runners (S3 R-S3-3/4, I3, I5, R-PAUSE).

These are CPU tests: the runtime is given fake streams (no CUDA object is
created), a fake worker whose ``_lane_step`` records the lane, the stream and
the workspace lane the runtime entered, and the four lane streams are faked.
"""

import contextlib
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.lane import Lane
from vllm.v1.worker import workspace
from vllm.v1.worker.lane import runtime as runtime_mod
from vllm.v1.worker.lane.fences import LaneFences
from vllm.v1.worker.lane.runtime import LaneRuntime


class FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited: list[str] = []

    def wait_stream(self, other: "FakeStream") -> None:
        self.waited.append(other.name)

    def synchronize(self) -> None:
        pass


class FakeStreams:
    """``LaneStreams`` stand-in: the four streams as identity-only fakes."""

    total_wgps = 36

    def __init__(self) -> None:
        self._streams = {
            lane: {
                True: FakeStream(f"{lane.name}-partitioned"),
                False: FakeStream(f"{lane.name}-full"),
            }
            for lane in Lane
        }

    def stream_for(self, lane: int, partitioned: bool = True) -> FakeStream:
        return self._streams[Lane(lane)][bool(partitioned)]

    def name_for(self, lane: int, partitioned: bool = True) -> str:
        return self.stream_for(lane, partitioned).name

    def masks(self) -> str:
        return "fake masks"


class FakeOutput:
    """``AsyncModelRunnerOutput`` stand-in with only the host output side."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def get_output(self) -> Any:
        return self._value


class FakeWorker:
    """Records what the runtime dispatches, never touches a device."""

    device = None
    rank = 0

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.entered = threading.Event()
        self.block = False
        self.fail_lane: Lane | None = None

    def _lane_runner(self, lane: int) -> Any:
        return self

    def lane_warmup(self, token_counts: list[int]) -> None:
        raise AssertionError("the fake worker expects no warmup")

    def _lane_step(self, lane: int, method: str, args: tuple) -> FakeOutput:
        lane = Lane(lane)
        with self.lock:
            self.calls.append(
                (
                    lane,
                    method,
                    args,
                    threading.current_thread().name,
                    workspace._workspace_lane.get(),
                )
            )
        if self.block:
            self.entered.set()
            if not self.release.wait(timeout=30):
                raise RuntimeError("fake lane step was never released")
        if self.fail_lane is not None and lane == self.fail_lane:
            raise RuntimeError("fake lane failure")
        return FakeOutput((lane.name, args[0]))


@contextlib.contextmanager
def _patched_runtime(monkeypatch: pytest.MonkeyPatch):
    """A runtime whose startup gate, streams and output type are faked."""
    worker = FakeWorker()
    streams = FakeStreams()
    runtime = LaneRuntime(worker, decode_wgps=20, streams=streams, strict=False)
    monkeypatch.setattr(runtime_mod, "check_kfd_budget", lambda *a, **k: None)
    monkeypatch.setattr(runtime_mod, "kfd_census", lambda: None)
    monkeypatch.setattr(runtime_mod, "create_lane_streams", FakeStreams)
    monkeypatch.setattr(runtime_mod, "AsyncModelRunnerOutput", FakeOutput)
    monkeypatch.setattr(
        runtime_mod.torch.cuda,
        "stream",
        lambda stream=None: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime_mod.torch.accelerator,
        "synchronize",
        lambda *args, **kwargs: None,
    )
    try:
        yield runtime, worker, streams
    finally:
        runtime.shutdown()


def test_lane_runtime_inboxes_are_per_lane_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I3: one inbox and one thread per lane, each lane's order preserved."""
    with _patched_runtime(monkeypatch) as (runtime, worker, _streams):
        futures = [
            runtime.submit(Lane(i % 2), "execute_model", (i,)) for i in range(12)
        ]
        outputs = [future.result(timeout=10) for future in futures]
        assert outputs == [(Lane(i % 2).name, i) for i in range(12)], (
            "results came back out of submission order"
        )
        per_lane: dict[Lane, list[int]] = {lane: [] for lane in Lane}
        thread_names: dict[Lane, set] = {lane: set() for lane in Lane}
        workspace_lanes: dict[Lane, set] = {lane: set() for lane in Lane}
        for lane, method, args, name, ws_lane in worker.calls:
            assert method == "execute_model"
            per_lane[lane].append(args[0])
            thread_names[lane].add(name)
            workspace_lanes[lane].add(ws_lane)
        for lane in Lane:
            assert per_lane[lane] == list(range(int(lane), 12, 2))
            assert thread_names[lane] == {f"lane-{lane.name.lower()}"}
            assert len(thread_names[lane]) == 1, "a lane used two thread names"
            # R-S3 (workspace): each lane thread enters its own workspace lane.
            assert workspace_lanes[lane] == {int(lane)}


def test_lane_runtime_failure_propagates_and_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane failure reaches every waiter and later submits fail fast."""
    with _patched_runtime(monkeypatch) as (runtime, worker, _streams):
        worker.fail_lane = Lane.PREFILL
        prefill = runtime.submit(Lane.PREFILL, "execute_model", (0,))
        with pytest.raises(RuntimeError, match="fake lane failure"):
            prefill.result(timeout=10)
        assert runtime.failed is not None
        # Fast: no lane thread is involved any more.
        decode = runtime.submit(Lane.DECODE, "execute_model", (1,))
        with pytest.raises(RuntimeError):
            decode.result(timeout=1)
        with pytest.raises(RuntimeError):
            runtime.quiesce()


def test_lane_runtime_quiesce_waits_for_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-PAUSE: ``quiesce`` returns only once no lane item is outstanding."""
    with _patched_runtime(monkeypatch) as (runtime, worker, _streams):
        worker.block = True
        future = runtime.submit(Lane.DECODE, "execute_model", (7,))
        assert worker.entered.wait(timeout=10), "the lane never started the item"
        quiet = threading.Event()
        errors: list = []

        def quiesce() -> None:
            try:
                runtime.quiesce()
            except BaseException as e:  # noqa: BLE001 - asserted below
                errors.append(e)
            finally:
                quiet.set()

        thread = threading.Thread(target=quiesce, daemon=True)
        thread.start()
        assert not quiet.wait(timeout=0.3), "quiesce returned before the item did"
        worker.release.set()
        assert quiet.wait(timeout=20), "quiesce did not return after the item"
        assert not errors, errors
        assert future.result(timeout=10) == ("DECODE", 7)


def test_lane_runtime_hop_takes_f2_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I5/F2: a lane's stream hop waits on the stream it leaves."""
    with _patched_runtime(monkeypatch) as (runtime, worker, streams):

        class _Batch:
            partitioned = False
            lane_fence = (0, 0)
            lane_seq = 0
            num_scheduled_tokens = None
            total_num_scheduled_tokens = 1

        partitioned = _Batch()
        partitioned.partitioned = True
        full = _Batch()
        full.partitioned = False
        assert runtime.submit(Lane.DECODE, "execute_model", (partitioned,)).result(
            timeout=10
        ) == ("DECODE", partitioned)
        assert runtime.submit(Lane.DECODE, "execute_model", (full,)).result(
            timeout=10
        ) == ("DECODE", full)
        assert runtime.stream_hops() == 1
        assert streams.stream_for(Lane.DECODE, False).waited == [
            streams.name_for(Lane.DECODE, True)
        ]


def test_lane_runtime_shutdown_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """After shutdown a submit fails fast and quiesce returns (R-PAUSE)."""
    manager = _patched_runtime(monkeypatch)
    runtime, _worker, _streams = manager.__enter__()
    try:
        runtime.submit(Lane.DECODE, "execute_model", (0,)).result(timeout=10)
        runtime.shutdown()
        future = runtime.submit(Lane.DECODE, "execute_model", (1,))
        with pytest.raises(RuntimeError):
            future.result(timeout=1)
        # Quiesce after shutdown is a no-op: nothing is outstanding (R-PAUSE).
        runtime.quiesce()
    finally:
        manager.__exit__(None, None, None)


def test_lane_runtime_rejects_unknown_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _patched_runtime(monkeypatch) as (runtime, _worker, _streams):
        future = runtime.submit(Lane.DECODE, "not_a_lane_rpc", ())
        with pytest.raises(ValueError, match="not a lane RPC"):
            future.result(timeout=1)


def test_lane_rpc_is_wrapped_in_a_lane_named_profiler_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The torch profiler is process-wide; the range names the lane."""
    names: list[str] = []

    class FakeRange:
        def __init__(self, name: str) -> None:
            names.append(name)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(runtime_mod.torch.profiler, "record_function", FakeRange)
    with _patched_runtime(monkeypatch) as (runtime, _worker, _streams):
        runtime.submit(int(Lane.DECODE), "execute_model", (object(),)).result(5)
        runtime.submit(int(Lane.PREFILL), "execute_model", (object(),)).result(5)

    assert names == ["lane_decode_execute_model", "lane_prefill_execute_model"]


def test_annotate_profile_counts_decode_steps_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both lanes run forwards; an iteration is a decode step."""
    import vllm.v1.lane as lane_mod
    from vllm.v1.lane import lane_scope
    from vllm.v1.worker.gpu_worker import Worker

    monkeypatch.setattr(lane_mod, "_lane_mode", True, raising=False)

    steps: list[int] = []
    worker = SimpleNamespace(
        profiler=SimpleNamespace(step=lambda: steps.append(1), should_annotate=False),
    )

    with lane_scope(Lane.PREFILL), Worker.annotate_profile(worker, None):
        pass
    assert steps == []

    with lane_scope(Lane.DECODE), Worker.annotate_profile(worker, None):
        pass
    assert steps == [1]


def test_lane_fences_zero_fence_is_a_host_noop() -> None:
    """A zero fence names nothing: no event is created and nothing is waited."""

    def forbidden_event() -> Any:
        raise AssertionError("a fence event was created without a record")

    fences = LaneFences(event_factory=forbidden_event)
    stream = FakeStream("s")
    fences.wait(Lane.PREFILL, 0, stream)
    assert stream.waited == []


def test_lane_fences_abort_wakes_a_waiter() -> None:
    """A failed lane must fail the peer's fence wait instead of hanging."""
    fences = LaneFences()
    exc = RuntimeError("lane P died")
    fences.abort(exc)
    with pytest.raises(RuntimeError, match="aborted") as caught:
        fences.wait(Lane.PREFILL, 7, FakeStream("s"))
    assert caught.value.__cause__ is exc
