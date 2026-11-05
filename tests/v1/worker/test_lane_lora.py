# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lane-mode LoRA: per-lane mapping buffers and union adapter residency.

Both lanes forward through one model and one adapter set. Each lane needs its
own punica wrapper (the per-step token-to-adapter mapping buffers) so neither
lane's copy lands under the other lane's in-flight kernels, and the resident
adapters must be the union of the lanes' request sets so a lane finishing a
request cannot evict an adapter the other lane still reads. These tests cover
both without a GPU.
"""

from __future__ import annotations

import threading

import pytest

import vllm.v1.lane as lane_module
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.v1.lane import Lane, lane_scope


class _Layer(BaseLayerWithLoRA):
    """Minimal LoRA layer: only the punica binding matters here."""

    def __init__(self) -> None:
        super().__init__()
        import torch.nn as nn

        self.base_layer = nn.Identity()

    @classmethod
    def can_replace_layer(cls, *args, **kwargs) -> bool:
        return False


@pytest.fixture
def lane_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lane_module, "_lane_mode", True)
    yield
    monkeypatch.setattr(lane_module, "_lane_mode", False)


def test_layer_resolves_wrapper_by_calling_lane(lane_mode):
    layer = _Layer()
    decode_wrapper = object()
    prefill_wrapper = object()
    layer.set_mapping(
        {int(Lane.DECODE): decode_wrapper, int(Lane.PREFILL): prefill_wrapper}
    )

    with lane_scope(Lane.DECODE):
        assert layer.punica_wrapper is decode_wrapper
    with lane_scope(Lane.PREFILL):
        assert layer.punica_wrapper is prefill_wrapper
    # A thread that never entered a lane is lane D's.
    assert layer.punica_wrapper is decode_wrapper


def test_layer_resolution_is_thread_local(lane_mode):
    layer = _Layer()
    decode_wrapper = object()
    prefill_wrapper = object()
    layer.set_mapping(
        {int(Lane.DECODE): decode_wrapper, int(Lane.PREFILL): prefill_wrapper}
    )

    seen: list[object] = []

    def prefill_thread() -> None:
        with lane_scope(Lane.PREFILL):
            seen.append(layer.punica_wrapper)

    thread = threading.Thread(target=prefill_thread)
    thread.start()
    thread.join()
    assert seen == [prefill_wrapper]
    # The spawning thread stays lane D's wrapper.
    assert layer.punica_wrapper is decode_wrapper


def test_layer_single_wrapper_when_lane_mode_off(monkeypatch):
    monkeypatch.setattr(lane_module, "_lane_mode", False)
    layer = _Layer()
    wrapper = object()
    layer.set_mapping(wrapper)
    with lane_scope(Lane.PREFILL):
        assert layer.punica_wrapper is wrapper
    with lane_scope(Lane.DECODE):
        assert layer.punica_wrapper is wrapper


class _AdapterStub:
    def __init__(self) -> None:
        self.calls: list[tuple[Lane, object]] = []
        self.adapter_slots = 8

    def set_adapter_mapping(self, mapping: object) -> None:
        self.calls.append((lane_module.current_lane(), mapping))


class _ManagerStub:
    """The state ``set_active_adapters``/``release_lane`` touch."""

    def __init__(self) -> None:
        self._lane_lock = threading.RLock()
        self._lane_requests: dict[int, set[object]] = {}
        self._adapter_manager = _AdapterStub()
        self.applied: list[set[object]] = []

    def _apply_adapters(self, requests: set[object]) -> None:
        self.applied.append(set(requests))


class _Request:
    def __init__(self, adapter_id: int) -> None:
        self.adapter_id = adapter_id


def _request(adapter_id: int) -> _Request:
    return _Request(adapter_id)


def test_active_adapters_keep_both_lanes_resident(lane_mode):
    manager = _ManagerStub()
    decode_req = _request(1)
    prefill_req = _request(2)

    WorkerLoRAManager.set_active_adapters(
        manager, {decode_req}, "decode-map", Lane.DECODE
    )
    WorkerLoRAManager.set_active_adapters(
        manager, {prefill_req}, "prefill-map", Lane.PREFILL
    )

    # The resident set is the union: lane D's adapter stays while lane P's
    # request is active, even though D's latest set no longer lists it.
    assert manager.applied[-1] == {decode_req, prefill_req}
    # Each lane's mapping update landed in its own buffers.
    assert manager._adapter_manager.calls == [
        (Lane.DECODE, "decode-map"),
        (Lane.PREFILL, "prefill-map"),
    ]
    # Each lane recorded its own request set.
    assert manager._lane_requests == {0: {decode_req}, 1: {prefill_req}}


def test_release_lane_drops_only_that_lanes_residency(lane_mode):
    manager = _ManagerStub()
    decode_req = _request(1)
    prefill_req = _request(2)
    WorkerLoRAManager.set_active_adapters(manager, {decode_req}, None, Lane.DECODE)
    WorkerLoRAManager.set_active_adapters(manager, {prefill_req}, None, Lane.PREFILL)

    WorkerLoRAManager.release_lane(manager, Lane.PREFILL)

    # Lane P's adapters are gone from its set; lane D's request set is intact
    # and still drives the union.
    assert 1 not in manager._lane_requests
    assert manager.applied[-1] == {decode_req}


def test_active_adapters_single_lane_path_unchanged(monkeypatch):
    monkeypatch.setattr(lane_module, "_lane_mode", False)
    manager = _ManagerStub()
    req = _request(1)
    WorkerLoRAManager.set_active_adapters(manager, {req}, "map")

    assert manager.applied == [{req}]
    assert manager._adapter_manager.calls == [(Lane.DECODE, "map")]
    # The single-lane path does not track lane request sets.
    assert manager._lane_requests == {}


def test_stale_lane_adapter_does_not_kill_the_step(lane_mode):
    """A finished request lingering in the other lane's set is stale."""
    manager = _ManagerStub()
    manager._adapter_manager.adapter_slots = 2
    first = _request(1)
    second = _request(2)
    third = _request(3)
    WorkerLoRAManager.set_active_adapters(manager, {first, second}, None, Lane.DECODE)

    # Lane D still names the finished request's adapter; lane P needs a slot.
    WorkerLoRAManager.set_active_adapters(manager, {third}, None, Lane.PREFILL)

    assert manager.applied[-1] == {third, first} or manager.applied[-1] == {
        third,
        second,
    }
    assert len(manager.applied[-1]) == 2


def test_request_set_over_the_slot_budget_prunes_stale(lane_mode):
    """Two live sets over the budget cannot occur: the scheduler refuses it.

    If it does (a stale set the other lane has not refreshed yet), the step
    keeps this lane's set and drops the rest instead of killing the engine.
    """
    manager = _ManagerStub()
    manager._adapter_manager.adapter_slots = 1
    WorkerLoRAManager.set_active_adapters(manager, {_request(1)}, None, Lane.DECODE)

    WorkerLoRAManager.set_active_adapters(manager, {_request(2)}, None, Lane.PREFILL)

    assert manager.applied[-1] == {manager._lane_requests[int(Lane.PREFILL)].pop()}
    assert len(manager.applied[-1]) == 1


def test_adapter_mapping_buffers_are_per_lane(lane_mode):
    from vllm.lora.model_manager import LRUCacheLoRAModelManager

    class _MappingManager:
        def __init__(self) -> None:
            self.lora_index_to_id = [None, None]
            self._last_mapping = None
            self._last_mapping_by_lane: dict[int, object] = {}
            self._last_slot_layout = None
            self.calls: list[object] = []

        def _set_adapter_mapping(self, mapping: object) -> None:
            self.calls.append(mapping)

    manager = _MappingManager()
    mapping = object()
    LRUCacheLoRAModelManager.set_adapter_mapping(manager, mapping)
    with lane_scope(Lane.PREFILL):
        # The same mapping and slot layout, but the other lane's buffers are
        # still stale: it must get its own update.
        LRUCacheLoRAModelManager.set_adapter_mapping(manager, mapping)

    assert manager.calls == [mapping, mapping]


def test_moe_lora_context_view_resolves_per_lane(lane_mode):
    from vllm.model_executor.layers.fused_moe.experts.lora_context import (
        LaneMoELoRAContextView,
    )

    class _Context:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.original_hidden_states = None

    decode_context = _Context("decode")
    prefill_context = _Context("prefill")
    view = LaneMoELoRAContextView(
        {int(Lane.DECODE): decode_context, int(Lane.PREFILL): prefill_context}
    )

    with lane_scope(Lane.PREFILL):
        assert view.tag == "prefill"
        view.original_hidden_states = "prefill-hidden"
    with lane_scope(Lane.DECODE):
        assert view.tag == "decode"
        assert view.original_hidden_states is None
    # The kernel's per-forward write stayed on the writing lane's context.
    assert prefill_context.original_hidden_states == "prefill-hidden"
    assert decode_context.original_hidden_states is None


def test_punica_wrappers_are_per_lane_in_lane_mode(lane_mode, monkeypatch):
    from vllm.lora import model_manager

    created: list[object] = []

    def fake_get_punica_wrapper(*args, **kwargs):
        wrapper = object()
        created.append(wrapper)
        return wrapper

    monkeypatch.setattr(model_manager, "get_punica_wrapper", fake_get_punica_wrapper)

    wrappers = model_manager.LoRAModelManager._make_punica_wrappers(16, max_batches=4)
    assert set(wrappers) == {int(Lane.DECODE), int(Lane.PREFILL)}
    # One wrapper per lane: each owns its step's mapping buffers.
    assert wrappers[int(Lane.DECODE)] is not wrappers[int(Lane.PREFILL)]
    assert len(created) == 2


def test_single_punica_wrapper_when_lane_mode_off(monkeypatch):
    from vllm.lora import model_manager

    def fake_get_punica_wrapper(*args, **kwargs):
        return "single"

    monkeypatch.setattr(model_manager, "get_punica_wrapper", fake_get_punica_wrapper)
    monkeypatch.setattr(lane_module, "_lane_mode", False)
    assert model_manager.LoRAModelManager._make_punica_wrappers(16) == "single"
