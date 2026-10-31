# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer telemetry contracts without loading the platform-dependent package."""

import importlib.util
import sys
from pathlib import Path

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
        ("cache_preload_test", directory / "rdna_expert_cache.py"),
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


def test_preload_keeps_active_map_and_adopts_all_payloads_without_recopy(module):
    cache = cache_for(module)
    first = cache.ensure([0])
    before_map = cache.slot_map.clone()
    pointers = {name: t.data_ptr() for name, t in cache.payload.items()}
    reservation = cache.preload_known([1, 2, 3], protected_experts=[0])
    assert len(reservation.bindings) == 1
    assert torch.equal(cache.slot_map, before_map)
    assert cache.policy.mapping == {0: 0}
    copied = cache.copied_bytes
    cache.release(first)
    adopted = cache.ensure_reserved(reservation)
    assert cache.copied_bytes == copied
    for expert, slot, _ in adopted.bindings:
        for name in module.PAYLOAD_NAMES:
            assert torch.equal(
                cache.payload[name][slot], cache.host_payload[name][expert]
            )
    assert cache.prefetch_bytes == cache.adopted_prefetch_bytes == cache._expert_bytes
    assert cache.unused_prefetch_bytes == 0
    assert pointers == {name: t.data_ptr() for name, t in cache.payload.items()}
    cache.release(adopted)


def test_cancel_drops_reservation_without_reviving_evicted_payload(module):
    cache = cache_for(module)
    cache.release(cache.ensure([0, 1]))
    reservation = cache.preload_known([2])
    assert 0 not in cache.policy.mapping
    cache.cancel_preload(reservation)
    assert cache.unused_prefetch_bytes == cache.prefetch_bytes == cache._expert_bytes
    assert cache._preload is cache.policy._pending_reservation is None
    lease = cache.ensure([0])
    assert lease.bindings[0][2]
    cache.release(lease)


@pytest.mark.parametrize("abort", ["break", "raise"])
def test_pass_context_releases_hit_and_drains_unused_preload(module, abort):
    cache = cache_for(module)
    cache.release(cache.ensure([0]))
    try:
        with cache.passes([0, 1, 2], overlap=True) as passes:
            for lease in passes:
                assert lease.bindings == ((0, 0, False),)
                assert cache._preload is not None
                if abort == "raise":
                    raise ValueError("consumer failed")
                break
    except ValueError:
        assert abort == "raise"
    assert cache.policy._active is cache.policy._pending_reservation is None
    assert cache._preload is None
    assert cache.unused_prefetch_bytes == cache._expert_bytes


def test_overlap_passes_cover_demand_once_and_default_stays_combined(module):
    cache = cache_for(module)
    cache.release(cache.ensure([0]))
    seen: list[int] = []
    with cache.passes([0, 1, 2, 3], overlap=True) as passes:
        for lease in passes:
            seen.extend(expert for expert, _, _ in lease.bindings)
    assert seen == [0, 1, 2, 3]
    assert cache.adopted_prefetch_bytes == cache._expert_bytes
    assert cache.policy._active is cache._preload is None
    default = cache_for(module)
    default.release(default.ensure([0]))
    with default.passes([0, 1]) as passes:
        assert [lease.bindings for lease in passes] == [((0, 0, False), (1, 1, True))]
    assert default.prefetch_bytes == 0


def test_preload_failure_invalidates_and_retains_existing_active_lease(
    module, monkeypatch
):
    cache = cache_for(module)
    cache.release(cache.ensure([0, 1]))
    active = cache.ensure([1])

    def fail(bindings):
        cache.payload[module.PAYLOAD_NAMES[0]][bindings[0][1]].fill_(-8)
        raise RuntimeError("copy failed")

    monkeypatch.setattr(cache, "_copy", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        cache.preload_known([2])
    assert cache.policy.mapping == {1: 1}
    assert cache.policy._active is active
    assert cache._preload is cache.policy._pending_reservation is None
    assert cache.prefetch_failures == 1
    cache.release(active)


def test_preload_rejects_capture_and_foreign_adoption(module, monkeypatch):
    cache = cache_for(module)
    other = cache_for(module)
    reservation = other.preload_known([1])
    with pytest.raises(RuntimeError, match="pending preload"):
        cache.ensure_reserved(reservation)
    other.cancel_preload(reservation)
    cache.device = torch.device("cuda:0")
    monkeypatch.setattr(module, "is_capturing_expert_work", lambda: True)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        cache.preload_known([1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_preload_disjoint_consumer_graph_and_pool_pointer_reuse(module):
    cache = cache_for(module, "cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cache.release(cache.ensure([0]))
    stream.synchronize()
    out = torch.empty_like(cache.payload[module.PAYLOAD_NAMES[0]][0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out.copy_(cache.payload[module.PAYLOAD_NAMES[0]][0])
    pointers = tuple(t.data_ptr() for t in cache._copy_destinations)
    with torch.cuda.stream(stream):
        lease = cache.ensure([0])
        reservation = cache.preload_known([1], protected_experts=[0])
        graph.replay()
        cache.release(lease)
        lease = cache.ensure_reserved(reservation)
        cache.release(lease)
    stream.synchronize()
    assert torch.equal(out.cpu(), cache.host_payload[module.PAYLOAD_NAMES[0]][0])
    assert pointers == tuple(t.data_ptr() for t in cache._copy_destinations)
    assert cache.adopted_prefetch_bytes == cache._expert_bytes
