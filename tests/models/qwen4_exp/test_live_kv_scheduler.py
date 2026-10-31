# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No source free or request resume may precede the corresponding rank fence."""

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Status(Enum):
    RUNNING = 1
    WAITING_FOR_KV_SNAPSHOT = 2
    WAITING_FOR_KV_RESTORE = 3
    PREEMPTED = 4
    WAITING = 5
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 6
    WAITING_FOR_REMOTE_KVS = 7


@pytest.fixture
def setup(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setitem(
        sys.modules, "vllm.logger", SimpleNamespace(init_logger=lambda _: Mock())
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.request", SimpleNamespace(RequestStatus=Status)
    )
    spec = importlib.util.spec_from_file_location(
        "live_controller_test", root / "vllm/v1/core/sched/live_kv_offload.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    request = SimpleNamespace(
        request_id="a",
        num_computed_tokens=1025,
        num_prompt_tokens=1024,
        has_encoder_inputs=False,
        num_in_flight_tokens=0,
        status=Status.RUNNING,
        is_finished=Mock(return_value=False),
        arrival_time=1,
    )
    connector = Mock()
    ticket = object()
    connector.begin_request_quiesce.return_value = ticket
    connector.begin_request_snapshot.return_value = ticket
    connector.snapshot_status.return_value = "quiesce_pending"
    connector.snapshot_sources_ready.return_value = False
    connector.detach_snapshot_prefix_sources.return_value = True
    connector.begin_snapshot_restore.return_value = True
    manager = Mock()
    manager.export_request_snapshot.return_value = SimpleNamespace(
        source_physical_page_ids=(1, 2),
        pages=(1, 2),
        groups=(SimpleNamespace(block_ids=(1, 2, None, 1)),),
    )
    manager.block_pool.get_num_free_blocks.side_effect = [3, 5]
    manager.reserve_snapshot_restore.return_value = SimpleNamespace(
        destination_physical_page_ids=(3, 4), active=True
    )
    manager.get_blocks.return_value = SimpleNamespace(blocks=((),))
    scheduler = SimpleNamespace(
        connector=connector,
        kv_cache_manager=manager,
        prev_step_scheduled_req_ids={"a"},
        waiting=Mock(),
        kv_holding_waiting=Mock(),
        _request_blocks_can_be_freed=Mock(return_value=True),
        _free_blocks=Mock(),
    )
    controller = module.LiveKVOffloader(scheduler)
    controller.last_used["a"] = 1
    return controller, request, connector, manager, scheduler, module


def test_full_cycle_preserves_frontier_and_waits_for_every_ownership_fence(setup):
    controller, req, connector, manager, scheduler, module = setup
    assert controller.start(req)
    assert req.num_computed_tokens == 1025
    controller.advance()
    manager.export_request_snapshot.assert_not_called()
    connector.snapshot_status.return_value = "quiesced"
    req.num_in_flight_tokens = 1
    controller.advance()
    manager.export_request_snapshot.assert_not_called()
    req.num_in_flight_tokens = 0
    controller.advance()
    manager.detach_request_snapshot.assert_not_called()
    connector.snapshot_status.return_value = "host_ready"
    controller.advance()
    manager.detach_request_snapshot.assert_not_called()
    connector.snapshot_sources_ready.return_value = True
    controller.advance()
    manager.detach_request_snapshot.assert_called_once()
    module.logger.info.assert_any_call("Live KV owner pages %s: %s", "a", [2])
    assert req.status == Status.WAITING_FOR_KV_RESTORE
    assert controller.reclaimed_pages == 2
    assert not controller.prepare_resume(req)
    manager.commit_snapshot_restore.assert_not_called()
    controller.advance()
    manager.commit_snapshot_restore.assert_not_called()
    connector.snapshot_status.return_value = "restored"
    controller.advance()
    manager.commit_snapshot_restore.assert_called_once()
    connector.release_snapshot.assert_called_once()
    assert req.status == Status.PREEMPTED and req.num_computed_tokens == 1025
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 3},
        live_kv_resumed_req_ids=set(),
        finished_req_ids=set(),
    )
    controller.note_schedule(output)
    resumed = [req]
    controller.ready_resumes.add("a")
    assert controller.retained_v2_resumes(resumed) == [req]
    assert resumed == []
    controller.ready_resumes.clear()
    assert not controller.records and not controller.ready_resumes


def test_abort_retains_allocations_until_cancel_ack(setup):
    controller, req, connector, manager, scheduler, _ = setup
    assert controller.start(req)
    req.is_finished.return_value = True
    controller.advance()
    connector.cancel_snapshot.assert_called_once()
    scheduler._free_blocks.assert_not_called()
    connector.snapshot_status.return_value = "released"
    controller.advance()
    scheduler._free_blocks.assert_called_once_with(req)
    manager.detach_request_snapshot.assert_not_called()
    assert not controller.records


def test_unsupported_export_resumes_original_without_recompute_or_free(setup):
    controller, req, connector, manager, scheduler, _ = setup
    controller.start(req)
    connector.snapshot_status.return_value = "quiesced"
    manager.export_request_snapshot.side_effect = ValueError("pending CoW")
    controller.advance()
    connector.cancel_snapshot.assert_called_once()
    manager.detach_request_snapshot.assert_not_called()
    connector.snapshot_status.return_value = "released"
    controller.advance()
    assert req.status == Status.PREEMPTED and req.num_computed_tokens == 1025
    assert "a" in controller.ready_resumes
    scheduler._free_blocks.assert_not_called()


@pytest.mark.parametrize("additional", [{}, None, object()])
def test_disabled_factory_does_not_import_or_touch_device_runtime(setup, additional):
    *_, module = setup
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=additional)
    )
    assert module.make_live_kv_offloader(scheduler) is None


def test_no_proactive_spill_when_persistent_request_slots_are_full(setup):
    controller, req, connector, manager, scheduler, _ = setup
    scheduler.running = [req]
    scheduler.waiting = [SimpleNamespace(request_id="fresh")]
    scheduler.kv_holding_waiting = []
    scheduler.max_num_running_reqs = 1
    scheduler.num_waiting_for_streaming_input = 0
    manager.usage = 1.0
    controller.evict_for_admission()
    connector.begin_request_quiesce.assert_not_called()
    assert not controller.records


def test_v2_cached_resume_overwrites_tables_and_keeps_numerical_request_slot():
    import ast

    import numpy as np

    source = Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    functions: list[ast.stmt] = []
    for name in ("finish_requests", "add_requests", "update_requests"):
        fn = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        fn.decorator_list = []
        fn.returns = None
        for argument in fn.args.args:
            argument.annotation = None
        functions.append(fn)
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    tables = ([3, 4], [0, 5])
    states = SimpleNamespace(
        req_id_to_index={"a": 2},
        num_computed_tokens_np=np.array([0, 0, 1025]),
        prefill_len=SimpleNamespace(np=np.array([0, 0, 1024])),
        num_computed_prefill_tokens=np.zeros(3, dtype=np.int32),
    )
    runner = SimpleNamespace(
        req_states=states,
        block_tables=Mock(),
        pooling_runner=None,
        sampler=None,
        _remove_request=Mock(),
        kv_block_zeroer=Mock(),
    )
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        block_table_updates={"a": tables},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["a"],
            num_computed_tokens=[1025],
            new_block_ids=[tables],
            resumed_req_ids={"a"},
        ),
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )
    for name in ("finish_requests", "add_requests", "update_requests"):
        namespace[name](runner, output)
    runner._remove_request.assert_not_called()
    assert states.req_id_to_index == {"a": 2}
    runner.block_tables.append_block_ids.assert_called_once_with(
        2, tables, overwrite=True
    )


def test_transient_source_handoff_defers_without_cancel_or_gpu_free(setup):
    controller, req, connector, manager, scheduler, _ = setup

    class Busy(ValueError):
        retryable = True

    manifest = manager.export_request_snapshot.return_value
    manager.export_request_snapshot.side_effect = [Busy("queued CoW"), manifest]
    controller.start(req)
    connector.snapshot_status.return_value = "quiesced"
    controller.advance()
    assert controller.records["a"].phase == "quiescing"
    connector.cancel_snapshot.assert_not_called()
    manager.detach_request_snapshot.assert_not_called()
    controller.advance()
    assert controller.records["a"].phase == "storing"
    connector.begin_request_snapshot.assert_called_once()


def test_ready_restore_does_not_trigger_another_admission_spill(setup):
    controller, req, _, manager, scheduler, _ = setup
    controller.ready_resumes.add(req.request_id)
    victim = SimpleNamespace(request_id="other", arrival_time=2)
    scheduler.running = [victim]
    scheduler.waiting = [req]
    scheduler.kv_holding_waiting = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 8
    manager.usage = 1.0
    controller.start = Mock(return_value=True)
    controller.evict_for_admission()
    controller.start.assert_not_called()
    assert scheduler.running == [victim]
    assert controller.ready_resumes == {req.request_id}


def test_detached_snapshot_leaves_holding_queue_behind_fresh_admission(
    setup, monkeypatch
):
    controller, original, connector, _, scheduler, _ = setup
    root = Path(__file__).resolve().parents[3]
    request_module = sys.modules["vllm.v1.request"]
    monkeypatch.setattr(request_module, "Request", object, raising=False)
    spec = importlib.util.spec_from_file_location(
        "live_queue_test", root / "vllm/v1/core/sched/request_queue.py"
    )
    assert spec is not None and spec.loader is not None
    queues = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queues)

    class Request:
        request_id: str
        status: Status

    request = Request()
    request.__dict__.update(vars(original))
    fresh = Request()
    fresh.request_id = "fresh"
    scheduler.waiting = queues.FCFSRequestQueue([fresh])
    scheduler.kv_holding_waiting = queues.FCFSRequestQueue()
    assert controller.start(request)
    # The scheduler prioritizes this request while it still owns source pages.
    scheduler.waiting.remove_requests((request,))
    scheduler.kv_holding_waiting.add_request(request)
    connector.snapshot_status.return_value = "quiesced"
    controller.advance()
    connector.snapshot_status.return_value = "host_ready"
    connector.snapshot_sources_ready.return_value = True
    controller.advance()
    assert not scheduler.kv_holding_waiting
    assert list(scheduler.waiting) == [fresh, request]
    assert request.status == Status.WAITING_FOR_KV_RESTORE


@pytest.mark.parametrize(
    "status,lora",
    [
        (Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, None),
        (Status.WAITING_FOR_REMOTE_KVS, None),
        (Status.WAITING, object()),
    ],
)
def test_blocked_or_adapter_waiter_does_not_trigger_admission_spill(
    setup, status, lora
):
    controller, victim, _, manager, scheduler, _ = setup
    scheduler.running = [victim]
    scheduler.waiting = [
        SimpleNamespace(request_id="fresh", status=status, lora_request=lora)
    ]
    scheduler.kv_holding_waiting = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 8
    manager.usage = 1.0
    controller.start = Mock(return_value=True)
    controller.evict_for_admission()
    controller.start.assert_not_called()
    assert scheduler.running == [victim]


def test_adapter_request_keeps_ordinary_scheduler_ownership(setup):
    controller, request, connector, _, _, _ = setup
    request.lora_request = object()
    assert not controller.start(request)
    connector.begin_request_quiesce.assert_not_called()
    assert not controller.records


def test_ready_fresh_waiter_still_spills_the_lru_resident(setup):
    controller, oldest, _, manager, scheduler, _ = setup
    newer = SimpleNamespace(request_id="newer", arrival_time=2)
    controller.last_used[newer.request_id] = 2
    scheduler.running = [newer, oldest]
    scheduler.waiting = [
        SimpleNamespace(request_id="fresh", status=Status.WAITING, lora_request=None)
    ]
    scheduler.kv_holding_waiting = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 8
    manager.usage = 1.0
    controller.start = Mock(return_value=True)
    controller.evict_for_admission()
    controller.start.assert_called_once_with(oldest)
    assert scheduler.running == [newer]


def test_overdue_restore_exchanges_slots_even_when_all_worker_slots_are_held(
    setup, monkeypatch
):
    controller, request, _, manager, scheduler, module = setup
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    assert controller.start(request)
    record = controller.records[request.request_id]
    record.phase = "host"
    record.started_at = 80.0
    record.manifest = object()
    request.status = Status.WAITING_FOR_KV_RESTORE
    victim = SimpleNamespace(request_id="victim", arrival_time=2)
    scheduler.running = [victim]
    scheduler.max_num_running_reqs = 2
    manager.reserve_snapshot_restore.return_value = None
    controller.start = Mock(return_value=True)
    controller.evict_for_admission()
    manager.reserve_snapshot_restore.assert_called_once_with(record.manifest)
    controller.start.assert_called_once_with(victim)
    scheduler.waiting.prepend_request.assert_called_once_with(request)
    assert not scheduler.running


def test_overdue_restore_uses_free_pages_before_evicting_another_request(
    setup, monkeypatch
):
    controller, request, _, manager, scheduler, module = setup
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    assert controller.start(request)
    record = controller.records[request.request_id]
    record.phase = "host"
    record.started_at = 80.0
    record.manifest = object()
    controller.start = Mock()
    controller.evict_for_admission()
    assert record.phase == "loading"
    controller.start.assert_not_called()
    manager.reserve_snapshot_restore.assert_called_once()


def test_resumed_request_gets_a_compute_quantum_before_another_eviction(
    setup, monkeypatch
):
    controller, request, _, _, scheduler, module = setup
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler.running = [request]
    controller.resumed_at[request.request_id] = 95.0
    controller.start = Mock(return_value=True)
    controller._evict_resident()
    controller.start.assert_not_called()
    controller.resumed_at[request.request_id] = 90.0
    controller._evict_resident()
    controller.start.assert_called_once_with(request)


def test_snapshot_drain_blocks_new_admission_and_prefill_but_retains_decode(setup):
    controller, request, _, _, _, _ = setup
    assert controller.start(request)
    assert controller.defer_prefills()
    assert controller.allow_admission(request)
    assert not controller.allow_admission(SimpleNamespace(request_id="fresh"))
    controller.records.clear()
    assert not controller.defer_prefills()
    assert controller.allow_admission(SimpleNamespace(request_id="fresh"))


def test_young_host_snapshot_does_not_trigger_fresh_admission_spill(setup):
    controller, request, _, manager, scheduler, _ = setup
    assert controller.start(request)
    controller.records[request.request_id].phase = "host"
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 8
    manager.usage = 1.0
    controller.start = Mock()
    controller.evict_for_admission()
    controller.start.assert_not_called()


def test_overdue_restore_waits_for_an_existing_eviction_before_selecting_more(
    setup, monkeypatch
):
    controller, request, _, manager, scheduler, module = setup
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    assert controller.start(request)
    controller.records[request.request_id].phase = "host"
    controller.records[request.request_id].started_at = 80.0
    controller.records["pending"] = module.LiveKVRequest(
        SimpleNamespace(request_id="pending"), object()
    )
    controller.start = Mock()
    controller.evict_for_admission()
    controller.start.assert_not_called()
    manager.reserve_snapshot_restore.assert_not_called()
    scheduler.waiting.prepend_request.assert_called_once_with(request)
    assert controller.defer_prefills()


def test_residency_protection_starts_at_scheduling_and_is_freed_at_completion(setup):
    controller, request, _, _, _, _ = setup
    controller.ready_resumes.add(request.request_id)
    controller.note_schedule(
        SimpleNamespace(num_scheduled_tokens={}, finished_req_ids=set())
    )
    assert not controller.resumed_at
    controller.note_schedule(
        SimpleNamespace(
            num_scheduled_tokens={request.request_id: 3}, finished_req_ids=set()
        )
    )
    assert request.request_id in controller.resumed_at
    controller.note_schedule(
        SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={request.request_id})
    )
    assert not controller.resumed_at


def test_overdue_restore_cannot_freeze_the_only_incomplete_resident_prefill(
    setup, monkeypatch
):
    controller, request, connector, manager, scheduler, module = setup
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    assert controller.start(request)
    record = controller.records[request.request_id]
    record.phase = "host"
    record.started_at = 80.0
    record.manifest = object()
    prefill = SimpleNamespace(**vars(request))
    prefill.request_id = "prefill"
    prefill.num_computed_tokens = 1024
    prefill.num_prompt_tokens = 4096
    prefill.status = Status.RUNNING
    scheduler.running = [prefill]
    controller.last_used[prefill.request_id] = 2
    manager.reserve_snapshot_restore.return_value = None
    connector.begin_request_quiesce.reset_mock()

    controller.evict_for_admission()
    connector.begin_request_quiesce.assert_not_called()
    assert scheduler.running == [prefill]
    assert not controller.defer_prefills()
    assert not controller.allow_admission(SimpleNamespace(request_id="fresh"))

    # Existing work can finish prefill and become an eligible victim. A real
    # quiesce then defers further prefill until its transfer fences drain.
    prefill.num_computed_tokens = prefill.num_prompt_tokens
    controller.evict_for_admission()
    connector.begin_request_quiesce.assert_called_once_with(prefill.request_id)
    assert controller.defer_prefills()


@pytest.mark.parametrize(
    "null_flags,expected", [([], False), ([True], False), ([True, False], True)]
)
def test_overdue_restore_allows_only_real_gpu_kv_holders(setup, null_flags, expected):
    controller, request, _, manager, _, _ = setup
    controller.oldest_overdue = Mock(return_value=object())
    manager.get_blocks.return_value = SimpleNamespace(
        blocks=(tuple(SimpleNamespace(is_null=value) for value in null_flags),)
    )
    # Prefix replay can rewind computed count to zero while retaining pages.
    request.num_computed_tokens = 0
    assert controller.allow_admission(request) is expected
