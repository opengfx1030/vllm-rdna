# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer telemetry contracts without loading the platform-dependent package."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    directory = root / "vllm/model_executor/layers/fused_moe"
    for name, path in (
        (
            "vllm.model_executor.layers.fused_moe.rdna_expert_cache_policy",
            directory / "rdna_expert_cache_policy.py",
        ),
        ("cache_timing_test", directory / "rdna_expert_cache.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
    return result


def cache_for(module, device="cpu"):
    bank = {
        name: torch.arange(24, dtype=torch.int32).reshape(4, 2, 3)
        for name in module.PAYLOAD_NAMES
    }
    return module.PackedExpertCache(bank, 2, device)


def test_default_has_no_timing_state_or_events(module, monkeypatch):
    cache = cache_for(module)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda **_: pytest.fail("unexpected timing event")
    )
    lease = cache.ensure([0, 1])
    cache.release(lease)
    assert cache._transfer_timing is None
    assert cache.transfer_timing() == {"enabled": False}


def test_cpu_planning_and_disable_preserve_totals(module):
    cache = cache_for(module)
    cache.set_transfer_timing(True)
    lease = cache.ensure([0, 1])
    cache.release(lease)
    report = cache.transfer_timing()
    assert report["cpu_passes"] == 1
    assert report["cpu_planning_us"] >= 0
    assert report["samples"] == report["pending"] == 0
    cache.set_transfer_timing(False)
    assert cache._transfer_timing is None
    assert cache.transfer_timing()["cpu_passes"] == 1
    assert not cache.transfer_timing()["enabled"]


def test_event_poll_never_waits_and_keeps_interval_meanings(module):
    cache = cache_for(module)
    cache.set_transfer_timing(True)

    class Event:
        def __init__(self, timestamp, ready=True):
            self.timestamp, self.ready = timestamp, ready

        def query(self):
            return self.ready

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

        def synchronize(self):
            pytest.fail("telemetry polling must not synchronize")

    events = tuple(Event(t) for t in (1, 4, 9, 7, 10))
    events[-1].ready = False
    cache._transfer_timing["pending"].append(events)
    assert cache.transfer_timing()["pending"] == 1
    events[-1].ready = True
    report = cache.transfer_timing()
    assert report["samples"] == 1
    assert report["gpu_copy_dependency_wait_ms"] == 3
    assert report["gpu_copy_interval_ms"] == 5
    assert report["gpu_exposed_copy_wait_ms"] == 3
    assert cache.transfer_timing()["samples"] == 1


def test_worker_names_rank_and_quiescent_timing_control(module, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "timing_worker_test", root / "tools/rdna/mono/expert_cache_worker.py"
    )
    assert spec is not None and spec.loader is not None
    worker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_module)
    cache = cache_for(module)
    model = SimpleNamespace(
        named_modules=lambda: iter(
            [("layers.7.experts", SimpleNamespace(_rdna_expert_cache=cache))]
        )
    )
    worker = worker_module.ExpertCacheWorker()
    worker.get_model = lambda: model
    worker.rank = 1
    syncs: list[bool] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: syncs.append(True))
    lease = cache.ensure([0])
    with pytest.raises(RuntimeError, match="released leases"):
        worker.rdna_expert_cache_control(transfer_timing="on")
    assert not syncs
    cache.release(lease)
    report = worker.rdna_expert_cache_control(transfer_timing="on")
    layer = report["layers"][0]
    assert layer["module"] == "layers.7.experts"
    assert layer["rank"] == 1
    assert layer["slots"] == 2
    assert layer["map_bytes"] == 16
    assert layer["expert_bytes"] == 144
    assert layer["transfer_timing"]["enabled"]
    assert report["expert_misses"] == 1
    worker.rdna_expert_cache_control(transfer_timing="off")
    assert syncs == [True, True]
    worker.rdna_expert_cache_control(overlap="on")
    assert cache.overlap
    reservation = cache.preload_known([1])
    with pytest.raises(RuntimeError, match="released leases"):
        worker.rdna_expert_cache_control(overlap="off")
    assert cache.overlap
    cache.cancel_preload(reservation)
    worker.rdna_expert_cache_control(overlap="off")
    assert not cache.overlap
    wrapper = SimpleNamespace(
        worker=SimpleNamespace(
            supports_physical_page_snapshots=True,
            _physical_page_count=8,
            _physical_cpu_chunk_count=2,
        ),
        spec=SimpleNamespace(blocks_per_chunk=1, cpu_page_size_per_worker=4096),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_transfer_state",
        SimpleNamespace(_KV_CONNECTOR_AGENT=SimpleNamespace(connector_worker=wrapper)),
    )
    layout = worker.rdna_kv_snapshot_layout()
    assert layout["physical_page_transport"]
    assert layout["blocks_per_chunk"] == 1
    assert not layout["live_request_paging"]
    assert layout["physical_page_bytes"] == 4096
    assert layout["gpu_physical_bytes"] == 8 * 4096
    assert layout["cpu_physical_bytes"] == 2 * 4096
    wrapper._live_snapshots_enabled = True
    assert worker.rdna_kv_snapshot_layout()["live_request_paging"]
    wrapper.worker.supports_physical_page_snapshots = False
    layout = worker.rdna_kv_snapshot_layout()
    assert layout["live_request_paging_configured"]
    assert not layout["live_request_paging"]
    assert layout["physical_page_bytes"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_gpu_queue_is_bounded_without_step_synchronization(module):
    cache = cache_for(module, "cuda")
    cache.set_transfer_timing(True)
    cache._transfer_timing["limit"] = 2
    for expert in (0, 1, 2, 3, 0, 1):
        lease = cache.ensure([expert])
        cache.release(lease)
        assert len(cache._transfer_timing["pending"]) <= 2
    torch.accelerator.synchronize()
    report = cache.transfer_timing()
    assert report["samples"] + report["dropped"] == 6
    for key in (
        "gpu_copy_dependency_wait_ms",
        "gpu_copy_interval_ms",
        "gpu_exposed_copy_wait_ms",
    ):
        assert report[key] >= 0


def test_pending_limit_drops_samples_without_timing_synchronization(
    module, monkeypatch
):
    from contextlib import nullcontext

    cache = cache_for(module)
    cache.device = torch.device("cuda:0")
    cache.set_transfer_timing(True)
    cache._transfer_timing["limit"] = 2
    timing_events: list[object] = []

    class Event:
        def __init__(self, enable_timing=False):
            self.timing = enable_timing
            if enable_timing:
                timing_events.append(self)

        def record(self, stream):
            pass

        def query(self):
            return False

        def synchronize(self):
            assert not self.timing, "timing added a per-step wait"

    class Stream:
        def wait_event(self, event):
            pass

    consumer = Stream()
    cache._copy_stream = Stream()
    monkeypatch.setattr(module, "is_capturing_expert_work", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: consumer)
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    for expert in (0, 1, 2):
        lease = cache.ensure([expert])
        cache.release(lease)
    assert len(timing_events) == 10
    assert len(cache._transfer_timing["pending"]) == 2
    assert cache.transfer_timing()["dropped"] == 1
