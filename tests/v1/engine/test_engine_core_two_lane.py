# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the two-lane engine loop (S2, R-S2-1/2/3/6).

The scheduler, the executor and the future completion times are fakes: these
tests exercise the lane queue discipline, the adaptive partition flag, the
pause drain and the flag-off step selection in ``EngineCore`` without a GPU.
"""

import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.lane import Lane
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

D = int(Lane.DECODE)
P = int(Lane.PREFILL)


def _output(
    lane: int,
    tokens: int = 1,
    reqs: dict[str, int] | None = None,
    finished: set[str] | None = None,
) -> SchedulerOutput:
    if reqs is None:
        reqs = {f"req{lane}": tokens} if tokens else {}
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=dict(reqs),
        total_num_scheduled_tokens=sum(reqs.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(finished or ()),
        free_encoder_mm_hashes=[],
        lane=lane,
        lane_seq=0,
        partitioned=False,
    )


class _FakeScheduler:
    def __init__(self) -> None:
        self.activity = [False, False]
        self.pending_release = [False, False]
        self.schedule_queue: dict[int, deque[SchedulerOutput]] = {
            D: deque(),
            P: deque(),
        }
        self.schedule_calls: list[int] = []
        self.updates: list[tuple[SchedulerOutput, ModelRunnerOutput]] = []
        self.added: list[object] = []
        self.running: list[object] = []
        self.waiting: list[object] = []
        # Defaults used when a test does not queue an explicit batch.
        self.default_tokens = [1, 1]
        self._lane_seq = [0, 0]

    def lane_activity(self) -> tuple[bool, bool]:
        return bool(self.activity[0]), bool(self.activity[1])

    def lane_has_pending_release(self, lane: int) -> bool:
        return bool(self.pending_release[int(lane)])

    def lane_stats(self) -> dict[str, dict[str, int]]:
        return {"decode": {"batches": len(self.updates)}}

    def schedule(
        self, lane: int = D, throttle_prefills: bool = False
    ) -> SchedulerOutput:
        lane = int(lane)
        self.schedule_calls.append(lane)
        if self.schedule_queue[lane]:
            output = self.schedule_queue[lane].popleft()
        else:
            output = _output(lane, tokens=self.default_tokens[lane])
        if output.total_num_scheduled_tokens:
            self._lane_seq[lane] += 1
        output.lane_seq = self._lane_seq[lane]
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        self.updates.append((scheduler_output, model_runner_output))
        outputs = [
            EngineCoreOutput(request_id=req_id, new_token_ids=[1])
            for req_id in scheduler_output.num_scheduled_tokens
        ]
        eco = EngineCoreOutputs(outputs=outputs)
        if scheduler_output.finished_req_ids:
            eco.finished_requests = set(scheduler_output.finished_req_ids)
        return {0: eco}

    def make_stats(self):
        return None

    def has_requests(self) -> bool:
        return bool(self.activity[0] or self.activity[1]) or bool(
            self.running or self.waiting
        )

    def add_request(self, request) -> None:
        self.added.append(request)
        self.activity[P] = True
        self.schedule_queue[P].append(
            _output(P, tokens=8, reqs={request.request_id: 8})
        )

    def finish_requests(self, request_ids, status) -> None:
        pass


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []
        self.rpc_calls: list[str] = []

    def execute_model(self, scheduler_output, non_block=False):
        self.calls.append(("execute_model", int(scheduler_output.lane), non_block))
        assert non_block
        return Future()

    def sample_tokens(self, grammar_output, non_block=False, lane=0):
        assert grammar_output is None
        assert non_block
        self.calls.append(("sample_tokens", int(lane), non_block))
        return Future()

    def collective_rpc(self, method, *args, **kwargs):
        self.rpc_calls.append(method)
        return None


class _TimedExecutor(_FakeExecutor):
    """Futures resolve on their own: D after 60 ms, P after 900 ms."""

    def __init__(self, d_delay: float = 0.06, p_delay: float = 0.9) -> None:
        super().__init__()
        self.d_delay = d_delay
        self.p_delay = p_delay

    @staticmethod
    def _resolve_later(future: Future, delay: float, value) -> None:
        timer = threading.Timer(delay, future.set_result, args=(value,))
        timer.daemon = True
        timer.start()

    def execute_model(self, scheduler_output, non_block=False):
        future = super().execute_model(scheduler_output, non_block=non_block)
        self._resolve_later(future, 0.0, None)
        return future

    def sample_tokens(self, grammar_output, non_block=False, lane=0):
        future = super().sample_tokens(grammar_output, non_block=non_block, lane=lane)
        delay = self.d_delay if int(lane) == D else self.p_delay
        self._resolve_later(future, delay, EMPTY_MODEL_RUNNER_OUTPUT)
        return future


def _make_core(
    *,
    two_lane: bool = True,
    decode_depth: int = 2,
    prefill_depth: int = 1,
    solo_full: bool = True,
    executor=None,
    scheduler=None,
    cls=EngineCore,
) -> EngineCore:
    core = cls.__new__(cls)
    core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            two_lane=two_lane,
            two_lane_decode_depth=decode_depth,
            two_lane_prefill_depth=prefill_depth,
            two_lane_solo_full=solo_full,
            two_lane_trace=False,
        ),
        observability_config=SimpleNamespace(enable_logging_iteration_details=False),
        model_config=SimpleNamespace(runner_type="generate"),
    )
    core.log_stats = False
    core.two_lane = two_lane
    core.lane_depths = {D: decode_depth, P: prefill_depth} if two_lane else None
    core.lane_queues = {int(lane): deque() for lane in Lane} if two_lane else None
    core._lane_wake = threading.Event() if two_lane else None
    core._lane_trace_file = None
    core._lane_trace_path = None
    core._lane_step_times = {int(lane): deque(maxlen=256) for lane in Lane}
    core._lane_completed = {int(lane): 0 for lane in Lane}
    core._lane_batches_total = 0
    core._lane_partitioned_total = 0
    core._lane_stats_last_log = time.monotonic()
    core._lane_solo_full = solo_full
    core.aborts_queue = queue.Queue()
    core.input_queue = queue.Queue()
    core._weight_version = "default"
    core._iteration_index = 0
    core.batch_queue = None
    core.engines_running = False
    core.scheduler = scheduler if scheduler is not None else _FakeScheduler()
    core.model_executor = executor if executor is not None else _FakeExecutor()
    # Diagnostics on the error path dump profiler state; not part of the
    # behaviour under test.
    core.log_error_detail = lambda scheduler_output: nullcontext()
    return core


def _complete_lane(core: EngineCore, lane: int, index: int = 0) -> None:
    batch = core.lane_queues[lane][index]
    batch.exec_future.set_result(None)
    assert batch.sample_future is not None
    batch.sample_future.set_result(EMPTY_MODEL_RUNNER_OUTPUT)


def test_flag_off_selects_baseline_step_fn():
    core = _make_core(two_lane=False)
    assert core._select_step_fn() == core.step

    core.batch_queue = deque(maxlen=2)
    assert core._select_step_fn() == core.step_with_batch_queue

    core = _make_core(two_lane=True)
    assert core._select_step_fn() == core.step_two_lane


def test_d_completes_and_reschedules_while_p_pending():
    core = _make_core()
    core.scheduler.activity = [True, True]

    core.step_two_lane()
    assert [len(core.lane_queues[D]), len(core.lane_queues[P])] == [1, 1]
    assert core.model_executor.calls == [
        ("execute_model", D, True),
        ("sample_tokens", D, True),
        ("execute_model", P, True),
        ("sample_tokens", P, True),
    ]

    _complete_lane(core, D)
    core.step_two_lane()

    # D was harvested and a new D batch submitted; P stayed in flight.
    assert [so.lane for so, _ in core.scheduler.updates] == [D]
    assert len(core.lane_queues[P]) == 1
    assert not core.lane_queues[P][0].sample_future.done()
    assert len(core.lane_queues[D]) == 1
    assert core.model_executor.calls[-2:] == [
        ("execute_model", D, True),
        ("sample_tokens", D, True),
    ]


def test_p_completes_and_reschedules_while_d_pending():
    core = _make_core()
    core.scheduler.activity = [True, True]
    core.step_two_lane()

    _complete_lane(core, P)
    core.step_two_lane()

    assert [so.lane for so, _ in core.scheduler.updates] == [P]
    # The P batch was replaced and D was topped back up to its depth.
    assert len(core.lane_queues[D]) == 2
    assert len(core.lane_queues[P]) == 1
    assert core.model_executor.calls[-2:] == [
        ("execute_model", P, True),
        ("sample_tokens", P, True),
    ]


def test_queue_depths_are_independent():
    core = _make_core(decode_depth=2, prefill_depth=1)
    core.scheduler.activity = [True, True]

    core.step_two_lane()
    core.step_two_lane()

    assert len(core.lane_queues[D]) == 2
    assert len(core.lane_queues[P]) == 1
    assert core.scheduler.schedule_calls.count(D) == 2
    assert core.scheduler.schedule_calls.count(P) == 1

    # D refills its own queue without touching the in-flight P batch.
    _complete_lane(core, D, 0)
    core.step_two_lane()
    assert len(core.lane_queues[D]) == 2
    assert len(core.lane_queues[P]) == 1
    assert core.scheduler.schedule_calls.count(D) == 3
    assert core.scheduler.schedule_calls.count(P) == 1


@pytest.mark.parametrize(
    "d_active,p_active,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_partition_truth_table(d_active, p_active, expected):
    core = _make_core()
    core.scheduler.activity = [d_active, p_active]
    assert core._partitioned() is expected


def test_solo_full_off_always_partitions():
    core = _make_core(solo_full=False)
    core.scheduler.activity = [False, False]
    assert core._partitioned() is True


def test_partition_flag_stamped_on_both_lanes():
    core = _make_core()
    core.scheduler.activity = [True, True]
    core.step_two_lane()

    for lane in (D, P):
        assert core.lane_queues[lane][0].scheduler_output.partitioned is True

    core = _make_core()
    # Only D has work: D runs full-GPU (solo full mask).
    core.scheduler.activity = [True, False]
    core.scheduler.pending_release[P] = True
    core.scheduler.schedule_queue[P].append(_output(P, tokens=0, finished={"x"}))
    core.step_two_lane()
    for lane in (D, P):
        assert core.lane_queues[lane][0].scheduler_output.partitioned is False


def test_lane_activity_falls_back_to_request_residency():
    core = _make_core()
    core.scheduler.lane_activity = None
    core.scheduler.running = [
        SimpleNamespace(lane=D, lane_resident=1 << D),
        SimpleNamespace(lane=P, lane_resident=1 << P),
    ]
    assert core._lane_activity() == (True, True)

    core.scheduler.running = [SimpleNamespace(lane=D, lane_resident=1 << D)]
    assert core._lane_activity() == (True, False)

    core.scheduler.running = []
    core.scheduler.waiting = [object()]
    assert core._lane_activity() == (False, True)


def test_release_only_empty_batch_is_submitted_and_applied():
    core = _make_core()
    core.scheduler.activity = [False, True]
    core.scheduler.pending_release[D] = True
    core.scheduler.schedule_queue[D].append(_output(D, tokens=0, finished={"dead"}))
    core.scheduler.schedule_queue[P].append(_output(P, tokens=0, finished={"pdead"}))

    core.step_two_lane()
    # No sampling for a 0-token batch; execute_model is still submitted.
    assert core.model_executor.calls == [
        ("execute_model", D, True),
        ("execute_model", P, True),
    ]

    for lane in (D, P):
        core.lane_queues[lane][0].exec_future.set_result(None)
    core.step_two_lane()

    assert len(core.scheduler.updates) == 2
    for _, model_output in core.scheduler.updates:
        assert model_output is EMPTY_MODEL_RUNNER_OUTPUT


@pytest.mark.parametrize("failed_lane", [D, P])
def test_lane_exception_propagates_without_touching_other_lane(failed_lane):
    other_lane = P if failed_lane == D else D
    core = _make_core()
    core.scheduler.activity = [True, True]
    core.step_two_lane()

    batch = core.lane_queues[failed_lane][0]
    batch.exec_future.set_result(None)
    batch.sample_future.set_exception(RuntimeError("lane failed"))
    with pytest.raises(RuntimeError, match="lane failed"):
        core.step_two_lane()
    # The failed lane's batch was consumed; the other lane is untouched.
    assert len(core.lane_queues[failed_lane]) == 0
    assert len(core.lane_queues[other_lane]) == 1
    assert not core.lane_queues[other_lane][0].sample_future.done()


def test_sample_none_reraises_execute_model_exception():
    core = _make_core()
    core.scheduler.activity = [True, False]
    core.step_two_lane()

    batch = core.lane_queues[D][0]
    batch.sample_future.set_result(None)
    batch.exec_future.set_exception(RuntimeError("exec failed"))
    with pytest.raises(RuntimeError, match="exec failed"):
        core.step_two_lane()


def test_pause_drains_both_lanes_before_sync():
    core = _make_core()
    core.output_queue = queue.Queue()
    core.scheduler.activity = [True, True]
    core.step_two_lane()
    _complete_lane(core, D)
    _complete_lane(core, P)

    core._finish_pause(clear_cache=False)

    assert [so.lane for so, _ in core.scheduler.updates] == [D, P]
    assert not any(core.lane_queues.values())
    assert core.model_executor.rpc_calls == ["synchronize_device"]
    assert core.output_queue.qsize() >= 1


def test_wake_on_new_request_while_blocked():
    core = _make_core(decode_depth=1)
    core.scheduler.activity = [True, False]
    core.step_two_lane()  # one D batch in flight, nothing scheduled after
    assert len(core.lane_queues[D]) == 1

    result: dict[str, Any] = {}
    thread = threading.Thread(target=lambda: result.update(out=core.step_two_lane()))
    thread.start()
    time.sleep(0.1)
    assert thread.is_alive()  # blocked on the lane future

    request = SimpleNamespace(
        request_id="late",
        pooling_params=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        abort_immediately=False,
    )
    core.add_request(request)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert result["out"] == (None, False)

    core.step_two_lane()
    assert core.scheduler.schedule_queue[P] == deque()
    assert core.lane_queues[P]


def test_has_work_includes_lane_queues():
    core = _make_core(cls=EngineCoreProc)
    core.scheduler.activity = [False, False]
    assert core.has_work() is False

    core.scheduler.activity = [True, False]
    core.step_two_lane()
    core.scheduler.activity = [False, False]
    assert core.has_work() is True


def test_d_batches_flow_while_a_long_prefill_is_in_flight():
    """The engine's done criterion: ~60 ms D steps during a 900 ms P step."""
    executor = _TimedExecutor(d_delay=0.06, p_delay=0.9)
    core = _make_core(executor=executor)
    core.scheduler.activity = [True, True]

    start = time.monotonic()
    deadline = start + 0.72
    while time.monotonic() < deadline:
        core.step_two_lane()

    d_updates = sum(1 for so, _ in core.scheduler.updates if so.lane == D)
    p_updates = sum(1 for so, _ in core.scheduler.updates if so.lane == P)
    assert d_updates >= 6, f"only {d_updates} D batches in 0.72 s"
    assert p_updates == 0, "the 900 ms prefill finished too early"

    # Every D batch was submitted while the single P batch was still pending.
    assert len(core.lane_queues[P]) == 1
    assert not core.lane_queues[P][0].sample_future.done()
