# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute real reset methods without loading platform/runtime dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CPU = "vllm/v1/kv_offload/cpu/manager.py"
_TIER = "vllm/v1/kv_offload/tiering/manager.py"
_SCHEDULER = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"


def _method(path, owner, name):
    tree = ast.parse((_ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    function = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    function.decorator_list = []
    function.returns = None
    for argument in (*function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def _cpu(retained=True):
    cls = type(
        "CPU",
        (),
        {
            name: _method(_CPU, "CPUOffloadingManager", name)
            for name in ("preflight_reset_cache", "reset_cache")
        },
    )
    manager = cls()
    manager._leases = {1: object()} if retained else {}
    manager._cache_generation = 5
    manager._policy = Mock()
    manager._num_evictable_cache_chunks = 3
    manager._num_write_pending_chunks = 2
    manager._free_list = [7]
    manager._num_allocated_chunks = 8
    manager.on_request_finished = Mock()
    return manager


def _tier(primary, children):
    cls = type(
        "Tier",
        (),
        {
            name: _method(_TIER, "TieringOffloadingManager", name)
            for name in ("preflight_reset_cache", "reset_cache")
        },
    )
    manager = cls()
    manager.primary_tier, manager.secondary_tiers = primary, children
    manager._jobs = {9: object()}
    manager._process_finished_jobs = Mock(side_effect=manager._jobs.clear)
    manager._pending_load_submissions = {4: object()}
    manager._metrics = SimpleNamespace(assert_idle=Mock(), on_request_finished=Mock())
    manager._req_state = {
        "finished": SimpleNamespace(
            pending_primary_stores=1,
            pending_cascade_keys=[b"key"],
            is_finished=True,
            req_context=object(),
        )
    }
    manager._processed_jobs_this_step = True
    return manager


def _child():
    # Existing secondary-tier implementations need not define a preflight API.
    return SimpleNamespace(
        drain_jobs=Mock(),
        on_request_finished=Mock(),
        bp_detector=SimpleNamespace(reset=Mock()),
    )


def _scheduler(manager):
    group = SimpleNamespace(next_stored_chunk_idx=7)
    status = SimpleNamespace(
        req=SimpleNamespace(is_finished=lambda: True),
        finished_signaled=False,
        req_context=object(),
        group_states=[group],
        transfer_jobs={9},
        partial_tail_boundary=32,
    )
    return SimpleNamespace(
        manager=manager,
        _snapshots={},
        _released_snapshots={},
        _detached_snapshot_requests=set(),
        _snapshot_generation=0,
        _current_batch_load_jobs={},
        _current_batch_jobs_to_flush=set(),
        _current_batch_allocated_block_ids=set(),
        _jobs={9: object()},
        _req_status={"finished": status},
        _job_counter=10,
        _stale_job_threshold=0,
        _block_id_to_pending_jobs={4: {9}},
        _events_tracker=SimpleNamespace(reset=Mock()),
        _chunks_being_loaded={b"key"},
    )


def test_default_manager_preflight_is_a_noop():
    preflight = _method(
        "vllm/v1/kv_offload/base.py", "OffloadingManager", "preflight_reset_cache"
    )
    assert preflight(object()) is None


def test_cpu_preflight_and_direct_reset_refuse_without_state_changes():
    manager = _cpu()
    for operation in (manager.preflight_reset_cache, manager.reset_cache):
        with pytest.raises(RuntimeError, match="chunk leases are retained"):
            operation()
        assert manager._cache_generation == 5 and manager._free_list == [7]
        assert manager._num_allocated_chunks == 8 and manager._leases
        manager._policy.clear.assert_not_called()
    manager._leases.clear()
    manager.reset_cache()
    assert manager._cache_generation == 6 and manager._free_list == []
    assert manager._num_allocated_chunks == 0
    manager._policy.clear.assert_called_once()


@pytest.mark.parametrize("rejecting_child", ["primary", "secondary", "nested"])
def test_tiered_reset_preflights_every_child_before_draining(rejecting_child):
    primary, first, second = _cpu(retained=False), _child(), _child()
    if rejecting_child == "primary":
        primary._leases[1] = object()
    elif rejecting_child == "secondary":
        second.preflight_reset_cache = Mock(side_effect=RuntimeError("retained child"))
    else:
        second.preflight_reset_cache = _tier(_cpu(), []).preflight_reset_cache
    manager = _tier(primary, [first, second])
    jobs, pending, state = (
        dict(manager._jobs),
        dict(manager._pending_load_submissions),
        manager._req_state["finished"],
    )
    with pytest.raises(RuntimeError, match="retained"):
        manager.reset_cache()
    assert manager._jobs == jobs and manager._pending_load_submissions == pending
    assert manager._req_state == {"finished": state}
    assert state.pending_primary_stores == 1 and state.pending_cascade_keys == [b"key"]
    assert manager._processed_jobs_this_step is True
    manager._process_finished_jobs.assert_not_called()
    manager._metrics.assert_idle.assert_not_called()
    primary._policy.clear.assert_not_called()
    for child in (first, second):
        child.drain_jobs.assert_not_called()
        child.on_request_finished.assert_not_called()
        child.bp_detector.reset.assert_not_called()


def test_tiered_reset_preserves_existing_success_path_without_optional_preflight():
    primary, child = _cpu(retained=False), _child()
    manager = _tier(primary, [child])
    manager.reset_cache()
    child.drain_jobs.assert_called_once()
    child.on_request_finished.assert_called_once()
    child.bp_detector.reset.assert_called_once()
    assert manager._jobs == {} and manager._pending_load_submissions == {}
    assert manager._req_state == {} and manager._processed_jobs_this_step is False
    primary._policy.clear.assert_called_once()


@pytest.mark.parametrize("tiered", [False, True])
def test_scheduler_rejected_reset_leaves_jobs_requests_and_events_untouched(tiered):
    primary = _cpu()
    manager = _tier(primary, [_child()]) if tiered else primary
    scheduler = _scheduler(manager)
    reset = _method(_SCHEDULER, "OffloadingConnectorScheduler", "reset_cache")
    jobs, statuses = dict(scheduler._jobs), dict(scheduler._req_status)
    with pytest.raises(RuntimeError, match="chunk leases are retained"):
        reset(scheduler)
    assert scheduler._current_batch_jobs_to_flush == set()
    assert scheduler._jobs == jobs and scheduler._req_status == statuses
    assert scheduler._stale_job_threshold == 0
    assert scheduler._block_id_to_pending_jobs == {4: {9}}
    assert scheduler._chunks_being_loaded == {b"key"}
    status = statuses["finished"]
    assert status.group_states[0].next_stored_chunk_idx == 7
    assert status.transfer_jobs == {9} and status.partial_tail_boundary == 32
    scheduler._events_tracker.reset.assert_not_called()
    primary.on_request_finished.assert_not_called()
    primary._policy.clear.assert_not_called()


def test_scheduler_success_retains_normal_flush_and_request_cleanup():
    manager = _cpu(retained=False)
    scheduler = _scheduler(manager)
    reset = _method(_SCHEDULER, "OffloadingConnectorScheduler", "reset_cache")
    reset(scheduler)
    assert scheduler._current_batch_jobs_to_flush == {9}
    assert scheduler._jobs == {} and scheduler._req_status == {}
    assert scheduler._stale_job_threshold == 10
    assert scheduler._block_id_to_pending_jobs == {}
    assert scheduler._chunks_being_loaded == set()
    scheduler._events_tracker.reset.assert_called_once()
    manager.on_request_finished.assert_called_once()
    manager._policy.clear.assert_called_once()
