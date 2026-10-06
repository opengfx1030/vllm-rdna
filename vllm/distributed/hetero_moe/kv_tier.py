# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU second-tier store for full-attention KV blocks.

Blocks are keyed by block hash. Eviction copies the block to the cold
pool. Attention never remote-reads it: ``prepare_for_attention`` copies
it back to the fast tier first.

GDN, PLE, and other recurrent state groups are refused. QSA main KV and
the compressed-key heap move only together, as one ``qsa_heap`` unit,
and that unit is not a remote-read source either.

With automatic prefix caching off (the dest default after the
2026-09-15 prefix corruption), this store is a preemption swap and a
long-context spill. It does not advertise prefix hits.
"""

from collections import OrderedDict
from collections.abc import Iterable

REFUSED_GROUPS = frozenset(
    {
        "gdn",
        "ple",
        "recurrent",
        "mamba",
        "kda",
        "ssm",
    }
)
QSA_PARTS = frozenset(
    {
        "qsa_main_kv",
        "qsa_compressed_key",
        "qsa_indexer",
    }
)


class RefusedStateGroup(RuntimeError):
    """The KV tier will not store this state group."""


class RemoteReadRefused(RuntimeError):
    """Attention must not read QSA or full-attention pages across the link."""


class HeteroKVTier:
    """In-process LRU. One full-attention block costs one slot.

    A QSA heap costs two slots and is evicted as one unit. Capacity ``0``
    stores nothing.

    Args:
        capacity_blocks: Cold-pool block slots.
    """

    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity_blocks = capacity_blocks
        self._cold: OrderedDict[object, tuple[str, object, int]] = OrderedDict()
        self._fast: dict[object, object] = {}
        self.events: list[tuple[str, object]] = []

    def is_refused(self, group: str) -> bool:
        """True when ``group`` must not be stored, including a split QSA."""
        name = group.lower()
        return name in REFUSED_GROUPS or name in QSA_PARTS

    def reject(self, group: str) -> None:
        """Raise if ``group`` is recurrent or a split QSA heap.

        Args:
            group: ``full_attention``, ``qsa_heap``, or a refused name.

        Raises:
            RefusedStateGroup: The group cannot be spilled.
        """
        name = group.lower()
        if name in REFUSED_GROUPS:
            raise RefusedStateGroup(f"refusing to spill recurrent state group {group}")
        if name in QSA_PARTS:
            raise RefusedStateGroup(
                "QSA main KV and the compressed-key heap move only as one qsa_heap"
            )
        if name not in ("full_attention", "qsa_heap"):
            raise RefusedStateGroup(f"unknown KV group {group}")

    def _slots_used(self) -> int:
        return sum(slots for _group, _payload, slots in self._cold.values())

    def _make_room(self, needed: int) -> None:
        while self._slots_used() + needed > self.capacity_blocks and self._cold:
            key, _record = self._cold.popitem(last=False)
            self.events.append(("lru_evict", key))

    def evict(self, block_hash, payload, group: str = "full_attention") -> bool:
        """Move one full-attention block onto the cold tier.

        Args:
            block_hash: Block-hash key.
            payload: Block bytes or tensor. Stored as given.
            group: Must be ``full_attention``.

        Returns:
            False when the capacity cannot hold the block. True when the
            block is resident on the cold tier (and gone from the fast map).
        """
        self.reject(group)
        if self.capacity_blocks < 1:
            return False
        self._make_room(1)
        if self._slots_used() + 1 > self.capacity_blocks:
            return False
        self._cold[block_hash] = (group, payload, 1)
        self._cold.move_to_end(block_hash)
        self._fast.pop(block_hash, None)
        self.events.append(("spill", block_hash))
        return True

    def evict_qsa_heap(
        self,
        main_hash,
        main_payload,
        compressed_hash,
        compressed_payload,
    ) -> bool:
        """Spill a QSA main block and its compressed-key heap together.

        Args:
            main_hash: Hash of the main KV block.
            main_payload: Main block.
            compressed_hash: Hash of the compressed-key / indexer heap.
            compressed_payload: Matching heap block.

        Returns:
            False when two slots are not available, leaving both halves
            unstored. True when both are stored under one LRU key.
        """
        self.reject("qsa_heap")
        if main_hash == compressed_hash:
            raise RefusedStateGroup("QSA heap halves need distinct hashes")
        if self.capacity_blocks < 2:
            return False
        key = ("qsa_heap", main_hash, compressed_hash)
        self._make_room(2)
        if self._slots_used() + 2 > self.capacity_blocks:
            return False
        self._cold[key] = (
            "qsa_heap",
            (main_payload, compressed_payload),
            2,
        )
        self._cold.move_to_end(key)
        self._fast.pop(main_hash, None)
        self._fast.pop(compressed_hash, None)
        self.events.append(("spill", key))
        return True

    def on_cold(self, key) -> bool:
        """True when ``key`` is a cold full-attention block or a heap key."""
        return key in self._cold

    def prepare_for_attention(self, keys: Iterable) -> dict:
        """Copy requested blocks back to the fast tier.

        Args:
            keys: Block hashes or QSA heap keys needed before attention.

        Returns:
            The restored payloads, now in the fast map. No handle into
            the cold store is returned.
        """
        restored = {}
        for key in keys:
            if key in self._cold:
                _group, payload, _slots = self._cold.pop(key)
                self._fast[key] = payload
                restored[key] = payload
                self.events.append(("copy_back", key))
            elif key in self._fast:
                restored[key] = self._fast[key]
        return restored

    def remote_read(self, key) -> None:
        """Refuse a read that would touch cold pages during attention.

        Raises:
            RemoteReadRefused: Always.
        """
        raise RemoteReadRefused(
            "refusing remote read of QSA/FA pages across the link; "
            f"copy {key!r} back to the fast tier before attention"
        )

    def cold_hashes(self) -> list:
        """Current cold keys, oldest first."""
        return list(self._cold.keys())
