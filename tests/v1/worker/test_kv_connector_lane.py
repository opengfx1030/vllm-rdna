# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane engine: one KV connector, one metadata view per lane.

The connector is a process singleton that binds one step's metadata at a
time. Two lane threads forward concurrently, so without a lane view a lane's
metadata would be replaced mid-forward by the other lane's step. These tests
run two threads that act as lane threads and assert each call sees its own
step.
"""

import threading
import time
from typing import Any, cast

import pytest

import vllm.v1.lane as lane_module
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.v1.lane import Lane, lane_scope
from vllm.v1.worker.lane import runtime as lane_runtime

pytestmark = pytest.mark.cpu_test


class _FakeConnector:
    """Records which metadata each connector call observed."""

    def __init__(self) -> None:
        self.metadata = None
        self.seen: list[tuple[str, str]] = []
        self.cleared = 0

    def bind_connector_metadata(self, metadata) -> None:
        self.metadata = metadata

    def clear_connector_metadata(self) -> None:
        self.metadata = None
        self.cleared += 1

    def start_load_kv(self, forward_context=None, **kwargs) -> None:
        # A real connector reads the bound metadata after this call.
        self.seen.append(("load", self.metadata))

    def save_kv_layer(self, layer_name: str) -> None:
        self.seen.append((layer_name, self.metadata))


@pytest.fixture
def fake_connector(monkeypatch):
    monkeypatch.setattr(lane_module, "_lane_mode", True)
    connector = _FakeConnector()
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    yield connector
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)


@pytest.fixture
def as_lane_thread():
    """Mark the calling thread as a lane thread, then restore it."""
    previous = lane_runtime._lane_thread_state.active
    lane_runtime._lane_thread_state.active = True
    yield
    lane_runtime._lane_thread_state.active = previous


def _run_lane_thread(lane: Lane, metadata: str, barrier, errors) -> None:
    try:
        lane_runtime._lane_thread_state.active = True
        with lane_scope(lane):
            barrier.wait(timeout=5)
            view = cast(Any, kv_transfer_state.get_kv_transfer_group())
            assert view.lane == int(lane)
            view.bind_connector_metadata(metadata)
            for step in range(5):
                view.start_load_kv()
                # A layer save later in the same forward must still see this
                # lane's step, even if the other lane bound in between.
                time.sleep(0.001)
                view.save_kv_layer(f"layer-{int(lane)}-{step}")
            view.clear_connector_metadata()
    except BaseException as e:  # noqa: BLE001 - reported to the main thread
        errors.append(e)


def test_lane_view_isolates_step_metadata(fake_connector):
    """Each lane thread's connector calls see that lane's metadata."""
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_run_lane_thread,
            args=(Lane.DECODE, "decode-step", barrier, errors),
        ),
        threading.Thread(
            target=_run_lane_thread,
            args=(Lane.PREFILL, "prefill-step", barrier, errors),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, errors
    assert len(fake_connector.seen) == 20
    for name, metadata in fake_connector.seen:
        if name == "load":
            continue
        expected = "decode-step" if name.startswith("layer-0-") else "prefill-step"
        assert metadata == expected, (name, metadata)


def test_no_lane_view_outside_a_lane_thread(fake_connector):
    """The worker's control path keeps the plain connector."""
    connector = kv_transfer_state.get_kv_transfer_group()
    assert connector is fake_connector


def test_lane_view_rebinds_before_each_call(fake_connector, as_lane_thread):
    """A lane's call rebinds its own step after another lane bound."""
    with lane_scope(Lane.DECODE):
        view = cast(Any, kv_transfer_state.get_kv_transfer_group())
        view.bind_connector_metadata("decode-step")
        fake_connector.metadata = "prefill-step"
        view.start_load_kv()
    assert fake_connector.seen[-1] == ("load", "decode-step")


def test_lane_view_clear_is_per_lane(fake_connector, as_lane_thread):
    """Clearing a lane's view leaves the other lane's binding intact."""
    with lane_scope(Lane.DECODE):
        view = cast(Any, kv_transfer_state.get_kv_transfer_group())
        view.bind_connector_metadata("decode-step")
        view.clear_connector_metadata()
        # No metadata is bound again by this lane, so a connector call made
        # after its own clear leaves the other lane's binding in place.
        fake_connector.metadata = "prefill-step"
        fake_connector.seen.clear()
        view.start_load_kv()
    assert fake_connector.seen == [("load", "prefill-step")]


def test_example_connector_is_hybrid_aware():
    """The example connector can run under the hybrid KV cache manager."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
    from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
        ExampleConnector,
    )

    assert supports_hma(ExampleConnector)
