# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local expert residency decisions, independent of tensor storage."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import index
from typing import Literal

Binding = tuple[int, int, bool]


@dataclass(eq=False)
class CacheLease:
    """Reserved slots; commit after copies, release after their last consumer.

    Each binding contains (local expert, slot, is_miss). A failed copy must
    cancel the lease. Slots selected for replacement are invalidated at acquire
    time, so partially overwritten victims cannot become hits after failure.
    """

    bindings: tuple[Binding, ...]
    _owner: "ExpertCachePolicy" = field(repr=False)
    _state: str = field(default="pending", init=False, repr=False)

    def commit(self) -> None:
        self._owner.commit(self)

    def cancel(self) -> None:
        self._owner.cancel(self)

    def release(self) -> None:
        self._owner.release(self)


class ExpertCachePolicy:
    """Deterministic LFU/LRU decisions with one live demand lease per cache.

    The storage manager owns copy and consumer events. It must commit only
    after copy completion is established for the consumer stream, and release
    only after all consumers finish or the next overwrite is event-fenced.
    Call observe once per actual invocation, not once per overflow pass.
    """

    def __init__(
        self,
        num_experts: int,
        capacity: int,
        policy: Literal["lfu_lru", "lru"] = "lfu_lru",
        decay_interval: int = 128,
    ) -> None:
        num_experts = index(num_experts)
        capacity = index(capacity)
        decay_interval = index(decay_interval)
        if not 0 < capacity <= num_experts:
            raise ValueError("capacity must be between one and num_experts")
        if policy not in ("lfu_lru", "lru"):
            raise ValueError("policy must be 'lfu_lru' or 'lru'")
        if decay_interval <= 0:
            raise ValueError("decay_interval must be positive")
        self.num_experts = num_experts
        self.capacity = capacity
        self.policy = policy
        self.decay_interval = decay_interval
        self.epoch = 0
        self._frequency = [0] * num_experts
        self._last_used = [0] * num_experts
        self._slot_experts: list[int | None] = [None] * capacity
        self._expert_slots: dict[int, int] = {}
        self._active: CacheLease | None = None

    @property
    def mapping(self) -> dict[int, int]:
        """Committed expert-to-slot mapping, excluding pending copies."""
        return dict(self._expert_slots)

    @property
    def frequencies(self) -> tuple[int, ...]:
        return tuple(self._frequency)

    @property
    def last_used(self) -> tuple[int, ...]:
        return tuple(self._last_used)

    def _valid_ids(self, ids: Sequence[int]) -> list[int]:
        return [expert for raw in ids if 0 <= (expert := index(raw)) < self.num_experts]

    def observe(self, ids: Sequence[int]) -> None:
        """Count routed pairs, ignoring padding and nonlocal expert IDs."""
        counts = Counter(self._valid_ids(ids))
        self.epoch += 1
        if self.epoch % self.decay_interval == 0:
            self._frequency = [count // 2 for count in self._frequency]
        for expert, count in counts.items():
            self._frequency[expert] += count
            self._last_used[expert] = self.epoch

    def plan_passes(self, ids: Sequence[int]) -> list[tuple[int, ...]]:
        """Consume resident demand first, then fill misses from cold to hot.

        The execution adapter retains original pair positions and weights;
        these expert sets alone do not describe token routing or reduction.
        Resident hits all fit in the first pass, so an earlier miss cannot
        evict a hit before its use. Hot misses run last to favor their reuse.
        """
        experts = list(dict.fromkeys(self._valid_ids(ids)))
        resident = [expert for expert in experts if expert in self._expert_slots]
        missing = [expert for expert in experts if expert not in self._expert_slots]
        missing.sort(
            key=lambda expert: (
                self._frequency[expert] if self.policy == "lfu_lru" else 0,
                self._last_used[expert],
                expert,
            )
        )
        experts = resident + missing
        return [
            tuple(experts[start : start + self.capacity])
            for start in range(0, len(experts), self.capacity)
        ]

    def _cold_key(self, slot: int) -> tuple[int, int, int]:
        expert = self._slot_experts[slot]
        assert expert is not None
        frequency = self._frequency[expert] if self.policy == "lfu_lru" else 0
        return frequency, self._last_used[expert], expert

    def acquire(self, experts: Sequence[int]) -> CacheLease:
        """Reserve a bounded demand set without evicting any demanded hit."""
        if self._active is not None:
            raise RuntimeError("release or cancel the active lease before acquiring")
        requested = list(dict.fromkeys(index(expert) for expert in experts))
        if any(expert < 0 or expert >= self.num_experts for expert in requested):
            raise ValueError("acquire requires valid rank-local expert IDs")
        if len(requested) > self.capacity:
            raise ValueError("demand exceeds capacity; use plan_passes first")
        protected = set(requested)
        free = [
            slot for slot, expert in enumerate(self._slot_experts) if expert is None
        ]
        victims = sorted(
            (
                slot
                for slot, expert in enumerate(self._slot_experts)
                if expert is not None and expert not in protected
            ),
            key=self._cold_key,
        )
        available = iter(free + victims)
        bindings: list[Binding] = []
        for expert in requested:
            slot = self._expert_slots.get(expert)
            if slot is None:
                bindings.append((expert, next(available), True))
            else:
                bindings.append((expert, slot, False))
        for _, slot, is_miss in bindings:
            if is_miss:
                victim = self._slot_experts[slot]
                if victim is not None:
                    del self._expert_slots[victim]
                self._slot_experts[slot] = None
        lease = CacheLease(tuple(bindings), self)
        self._active = lease
        return lease

    def _require(self, lease: CacheLease, state: str) -> None:
        if self._active is not lease or lease._state != state:
            raise RuntimeError(f"expected the active {state} lease")

    def commit(self, lease: CacheLease) -> None:
        """Publish copied experts; retain the lease until consumers are safe."""
        self._require(lease, "pending")
        for expert, slot, is_miss in lease.bindings:
            if is_miss:
                self._slot_experts[slot] = expert
                self._expert_slots[expert] = slot
        lease._state = "committed"

    def cancel(self, lease: CacheLease) -> None:
        """Abandon pending copies, leaving replacement destinations invalid.

        The manager must drain or fence any submitted copies before reusing
        these slots. Resident hits are unchanged.
        """
        self._require(lease, "pending")
        lease._state = "cancelled"
        self._active = None

    def release(self, lease: CacheLease) -> None:
        """Permit another acquisition after consumers are fenced or complete."""
        self._require(lease, "committed")
        lease._state = "released"
        self._active = None
