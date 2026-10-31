# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only residency contracts, loaded without initializing vLLM or torch."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py"
)
_SPEC = importlib.util.spec_from_file_location("rdna_expert_cache_policy", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ExpertCachePolicy = _MODULE.ExpertCachePolicy


def _run(cache, experts):
    lease = cache.acquire(experts)
    lease.commit()
    lease.release()
    return lease.bindings


def test_demanded_cold_hit_cannot_be_evicted_by_hot_miss():
    cache = ExpertCachePolicy(4, 2)
    _run(cache, [0, 1])
    cache.observe([1] * 10 + [2] * 20)
    lease = cache.acquire([2, 0])
    assert lease.bindings == ((2, 1, True), (0, 0, False))
    assert cache.mapping == {0: 0}
    lease.commit()
    assert cache.mapping == {0: 0, 2: 1}
    lease.release()


def test_overflow_passes_cover_each_valid_expert_once_and_keep_hot_misses_last():
    cache = ExpertCachePolicy(5, 2)
    ids = [3, -1, 1, 3, 7, 0, 2, 4, 1]
    assert cache.plan_passes(ids) == [(0, 1), (2, 3), (4,)]
    cache.observe(ids)
    assert cache.plan_passes(ids) == [(0, 2), (4, 1), (3,)]
    for experts in cache.plan_passes(ids):
        assert len(_run(cache, experts)) <= 2
    assert cache.frequencies == (1, 2, 1, 2, 1)
    assert cache.epoch == 1
    assert cache.plan_passes([-1, 5]) == []


@pytest.mark.parametrize("policy", ["lfu_lru", "lru"])
def test_overflow_consumes_late_resident_hits_before_any_miss_can_evict_them(policy):
    cache = ExpertCachePolicy(4, 2, policy=policy)
    _run(cache, [0, 1])
    demand = [2, 3, 0, 1, 0]
    cache.observe(demand)
    passes = cache.plan_passes(demand)
    assert passes == [(0, 1), (2, 3)]
    bindings = [_run(cache, experts) for experts in passes]
    assert not any(is_miss for _, _, is_miss in bindings[0])
    assert sum(is_miss for batch in bindings for _, _, is_miss in batch) == 2


def test_first_pass_uses_spare_capacity_for_misses_and_retains_hottest_overflow():
    cache = ExpertCachePolicy(5, 2)
    _run(cache, [0])
    demand = [4] * 20 + [3] * 10 + [2, 1, 0]
    cache.observe(demand)
    passes = cache.plan_passes(demand)
    assert passes == [(0, 1), (2, 3), (4,)]
    for experts in passes:
        _run(cache, experts)
    assert set(cache.mapping) == {3, 4}


def test_lru_overflow_orders_misses_by_recency_then_expert_id():
    cache = ExpertCachePolicy(5, 2, policy="lru")
    cache.observe([4])
    cache.observe([2])
    cache.observe([3, 1])
    assert cache.plan_passes([3, 2, 1, 4]) == [(4, 2), (1, 3)]
    for experts in cache.plan_passes([3, 2, 1, 4]):
        _run(cache, experts)
    assert set(cache.mapping) == {1, 3}


def test_frequency_keeps_hot_expert_while_lru_keeps_recent_expert():
    for policy, expected in [("lfu_lru", {0, 2}), ("lru", {1, 2})]:
        cache = ExpertCachePolicy(3, 2, policy=policy)
        _run(cache, [0, 1])
        cache.observe([0] * 10)
        cache.observe([1])
        _run(cache, [2])
        assert set(cache.mapping) == expected


def test_decay_allows_phase_change_and_lru_breaks_equal_frequency():
    cache = ExpertCachePolicy(3, 2, decay_interval=2)
    _run(cache, [0, 1])
    cache.observe([0] * 8)
    for _ in range(8):
        cache.observe([1])
    assert cache.frequencies[0] == 0
    _run(cache, [2])
    assert set(cache.mapping) == {1, 2}

    tied = ExpertCachePolicy(3, 2)
    _run(tied, [0, 1])
    tied.observe([1])
    tied.observe([0])
    _run(tied, [2])
    assert set(tied.mapping) == {0, 2}


def test_same_epoch_ties_use_expert_id_independently_of_slot_order():
    cache = ExpertCachePolicy(3, 2)
    _run(cache, [1, 0])
    cache.observe([1, 0])
    assert _run(cache, [2]) == ((2, 1, True),)


def test_failed_copy_invalidates_victim_without_publishing_partial_payload():
    cache = ExpertCachePolicy(3, 2)
    _run(cache, [0, 1])
    lease = cache.acquire([2, 1])
    assert cache.mapping == {1: 1}
    lease.cancel()
    assert cache.mapping == {1: 1}
    assert _run(cache, [0, 1]) == ((0, 0, True), (1, 1, False))


def test_slots_remain_leased_after_copy_commit_until_consumer_release():
    cache = ExpertCachePolicy(3, 1)
    lease = cache.acquire([0])
    with pytest.raises(RuntimeError, match="active lease"):
        cache.acquire([1])
    lease.commit()
    with pytest.raises(RuntimeError, match="active lease"):
        cache.acquire([1])
    with pytest.raises(RuntimeError, match="pending"):
        lease.cancel()
    lease.release()
    assert _run(cache, [1]) == ((1, 0, True),)
    with pytest.raises(RuntimeError, match="committed"):
        lease.release()


def test_foreign_lease_and_snapshot_mutation_cannot_change_bindings():
    first = ExpertCachePolicy(2, 1)
    second = ExpertCachePolicy(2, 1)
    lease = first.acquire([0])
    with pytest.raises(RuntimeError, match="pending"):
        second.commit(lease)
    lease.commit()
    snapshot = first.mapping
    snapshot.clear()
    assert first.mapping == {0: 0}
    lease.release()
    assert _run(first, [0, 0]) == ((0, 0, False),)


def test_tp2_local_expert_range_and_churn_preserve_unique_slot_ownership():
    cache = ExpertCachePolicy(256, 31, decay_interval=4)
    for offset in range(12):
        ids = [(expert + offset) % 256 for expert in range(256)]
        cache.observe(ids)
        passes = cache.plan_passes(ids + [-1, 256])
        scheduled = [expert for batch in passes for expert in batch]
        assert len(scheduled) == len(ids)
        assert set(scheduled) == set(ids)
        assert set(cache.mapping) <= set(passes[0])
        for batch in passes:
            lease = cache.acquire(batch)
            lease.commit()
            assert all(
                cache.mapping[expert] == slot for expert, slot, _ in lease.bindings
            )
            assert len(set(cache.mapping.values())) == len(cache.mapping)
            assert len(cache.mapping) <= 31
            lease.release()


@pytest.mark.parametrize("experts", [[-1], [3], [0, 1, 2]])
def test_invalid_acquisition_does_not_change_residency(experts):
    cache = ExpertCachePolicy(3, 2)
    _run(cache, [0, 1])
    with pytest.raises(ValueError):
        cache.acquire(experts)
    assert cache.mapping == {0: 0, 1: 1}
    assert _run(cache, [0]) == ((0, 0, False),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_experts": 0, "capacity": 1},
        {"num_experts": 2, "capacity": 0},
        {"num_experts": 2, "capacity": 3},
        {"num_experts": 2, "capacity": 1, "policy": "unknown"},
        {"num_experts": 2, "capacity": 1, "decay_interval": 0},
    ],
)
def test_reject_invalid_pool_configuration(kwargs):
    with pytest.raises(ValueError):
        ExpertCachePolicy(**kwargs)


def test_reservation_excludes_active_and_unconsumed_resident_demand():
    cache = ExpertCachePolicy(8, 4)
    _run(cache, [0, 1, 2, 3])
    active = cache.acquire([0])
    active.commit()
    reservation = cache.reserve([4, 5, 6], protected_experts=[1, 2])
    assert reservation.bindings == ((4, 3, True),)
    assert cache.mapping == {0: 0, 1: 1, 2: 2}
    assert cache._active is active
    with pytest.raises(RuntimeError, match="active lease"):
        reservation.adopt()
    assert cache._pending_reservation is reservation
    active.release()
    adopted = reservation.adopt()
    assert adopted.bindings == ((4, 3, True),)
    assert 4 not in cache.mapping
    adopted.commit()
    adopted.release()
    assert cache.mapping == {0: 0, 1: 1, 2: 2, 4: 3}


def test_reservation_excludes_uncommitted_active_miss_slots_and_experts():
    cache = ExpertCachePolicy(6, 3)
    active = cache.acquire([0, 1])
    reservation = cache.reserve([0, 1, 2, 3])
    assert reservation.bindings == ((2, 2, True),)
    assert cache.mapping == {}
    active.commit()
    active.release()
    reservation.cancel()
    assert cache.mapping == {0: 0, 1: 1}
    assert _run(cache, [2]) == ((2, 2, True),)


def test_reservation_cancel_invalidates_victim_and_restores_demand_priority():
    cache = ExpertCachePolicy(5, 2)
    _run(cache, [0, 1])
    reservation = cache.reserve([2])
    assert 0 not in cache.mapping and 2 not in cache.mapping
    with pytest.raises(RuntimeError, match="pending reservation"):
        cache.acquire([0])
    with pytest.raises(RuntimeError, match="pending reservation"):
        cache.reserve([3])
    reservation.cancel()
    assert 0 not in cache.mapping
    assert _run(cache, [0]) == ((0, 0, True),)
    with pytest.raises(RuntimeError, match="pending reservation"):
        reservation.adopt()


def test_reservation_skips_hits_deduplicates_and_uses_cold_victims():
    cache = ExpertCachePolicy(7, 3)
    _run(cache, [0, 1, 2])
    cache.observe([0] * 10 + [1] * 2)
    epoch, frequencies = cache.epoch, cache.frequencies
    reservation = cache.reserve([0, 3, 3, 4, 5, 6])
    assert reservation.bindings == ((3, 2, True), (4, 1, True))
    assert cache.epoch == epoch and cache.frequencies == frequencies
    assert cache.mapping == {0: 0}
    reservation.cancel()


def test_reservation_no_capacity_or_no_misses_has_no_side_effects():
    cache = ExpertCachePolicy(4, 2)
    _run(cache, [0, 1])
    before = cache.mapping
    assert cache.reserve([0, 1]) is None
    assert cache.reserve([2], protected_experts=[0, 1]) is None
    active = cache.acquire([0, 1])
    active.commit()
    assert cache.reserve([2]) is None
    assert cache.mapping == before
    assert cache._pending_reservation is None
    active.release()


@pytest.mark.parametrize(
    "experts,protected", [([2, 8], []), ([2], [8]), ([-1], []), ([2], [-1])]
)
def test_invalid_reservation_is_atomic(experts, protected):
    cache = ExpertCachePolicy(4, 2)
    _run(cache, [0, 1])
    before = cache.mapping
    with pytest.raises(ValueError):
        cache.reserve(experts, protected)
    assert cache.mapping == before
    assert cache._pending_reservation is None


def test_foreign_or_reused_reservation_cannot_change_ownership():
    cache = ExpertCachePolicy(4, 2)
    other = ExpertCachePolicy(4, 2)
    reservation = cache.reserve([0])
    with pytest.raises(RuntimeError, match="pending reservation"):
        other.adopt(reservation)
    with pytest.raises(RuntimeError, match="pending reservation"):
        other.cancel_reservation(reservation)
    lease = reservation.adopt()
    with pytest.raises(RuntimeError, match="pending reservation"):
        reservation.cancel()
    lease.cancel()
    assert cache.mapping == {}
