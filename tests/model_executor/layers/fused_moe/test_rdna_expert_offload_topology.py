# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2/4 expert-ownership and budget contract tests (CPU, no tensor I/O).

The policy module is pure Python (imported by the offline trace sweep tool),
so these tests exercise the real validator, not a copy of it. TP1 is refused
on purpose: 512 local experts exceeds the K2 kernel's stage capacity, which is
why expert-parallel serving is required for this offload path.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py"
)
_SPEC = importlib.util.spec_from_file_location("topology_policy", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
policy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = policy
_SPEC.loader.exec_module(policy)

validate = policy.validate_offload_topology


def test_tp4_owns_a_contiguous_disjoint_block_per_rank():
    assert validate(4, [0, 1, 2, 3]) == 128
    blocks = [range(r * 128, (r + 1) * 128) for r in range(4)]
    owned = [e for block in blocks for e in block]
    assert sorted(owned) == list(range(512))  # every expert owned exactly once
    assert all(len(b) == 128 for b in blocks)


def test_tp2_baseline_geometry_unchanged():
    assert validate(2, [6, 7]) == 256
    assert validate(2, [7, 6]) == 256  # order is an operator choice, not geometry


def test_slots_limits_track_the_per_rank_local_count():
    assert validate(4, [0, 1, 2, 3], slots=128) == 128
    assert validate(2, [6, 7], slots=96) == 256  # maintainer baseline
    with pytest.raises(ValueError, match=r"cache slots must be in \[1, 128\]"):
        validate(4, [0, 1, 2, 3], slots=129)
    with pytest.raises(ValueError, match="cache slots"):
        validate(2, [6, 7], slots=0)


def test_invalid_topologies_refused():
    with pytest.raises(ValueError, match="power of two"):
        validate(3, [0, 1, 2])
    with pytest.raises(ValueError, match="requires 4 GPUs"):
        validate(4, [0, 1, 2])
    with pytest.raises(ValueError, match="requires 2 GPUs"):
        validate(2, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="unique"):
        validate(2, [6, 6])
    with pytest.raises(ValueError, match="non-negative"):
        validate(2, [-1, 6])
    with pytest.raises(ValueError, match="qualified"):
        validate(1, [0])  # 512 local experts exceeds K2 stage capacity
    with pytest.raises(ValueError, match="qualified"):
        validate(8, list(range(8)))  # 64 local experts is unqualified
    with pytest.raises(ValueError, match="power of two"):
        validate(0, [])


def test_per_rank_policy_caches_are_independent_and_bounded():
    # Cross-rank accounting: four independent 128-expert caches never observe
    # each other's IDs, and each keeps slots <= its local expert count.
    caches = [
        policy.ExpertCachePolicy(validate(4, [0, 1, 2, 3]), 48) for _ in range(4)
    ]
    for rank, cache in enumerate(caches):
        base = rank * 128
        ids = [(base + i) % 128 for i in range(60)]  # local IDs, over capacity
        cache.observe(ids)
        for experts in cache.plan_passes(ids):
            assert all(0 <= e < 128 for e in experts)
            lease = cache.acquire(experts)
            assert all(0 <= slot < 48 for _, slot, _ in lease.bindings)
            lease.commit()
            lease.release()
    mapping = caches[0].mapping
    assert len(mapping) == 48
    assert len(set(mapping.values())) == 48  # one expert per slot


def test_global_to_local_shift_matches_ownership_rule():
    # The sweep tool maps a global route ID g on rank r to g - r*local; verify
    # that every globally owned ID lands in [0, local) under that rule.
    local = validate(4, [0, 1, 2, 3])
    shifted = sorted(g - (g // local) * local for g in range(512))
    assert shifted == sorted(list(range(local)) * 4)
    assert [g // local for g in (0, 127, 128, 511)] == [0, 0, 1, 3]
