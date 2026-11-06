# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension that dumps RidgeFill cache-only vs stock QSA cache writes.

PR 155 study tool. ``VLLM_RDNA_RIDGEFILL_CACHE_ONLY=1`` routes filled rows of
a QSA layer through ``torch.ops.vllm.ridgefill_cache_only``: the filled rows
write their caches from ``Qwen4ExpQSAAttention.cache_only_forward`` and get a
zero attention output. The legacy fill (``CACHE_ONLY=0``) runs the stock
``self_attn(block_input, positions)`` on every row instead. This probe runs
both writers on the same block input, inside every TP worker, and saves what
each one left in the caches.

For the first ``forwards`` filled forwards (after ``skip``), every call of
the op in this worker does:

A.  ``cache_only_forward`` on the filled rows, exactly as the op does, then
    snapshot the filled rows' cache entries.
B.  restore the indexer raw-key ring to its pre-A content (the compress
    kernel reads it for a group that straddles the chunk start), run the
    stock call on the whole block input as the legacy decoder does, snapshot.
B2. restore the ring again, run the stock call again, snapshot. B2 - B is
    the run-to-run floor of the stock writers.

The op then returns B2's attention output: the captured forward continues as
the legacy fill for every QSA layer that reaches the op, and the caches it
leaves are the stock writers'. Forwards before ``skip`` and after the capture
run the original op unchanged (cache-only semantics).

Caches, per QSA layer, all gathered at the slots the full-batch metadata
gives the filled rows (the slots every later reader uses):

``main_k`` / ``main_v``
    ``attn.kv_cache.transpose(1, 2).split(head_dim, -1)``: fp16
    ``[blocks, block_size, kv_heads, head_dim]``, slot
    ``metadata[attn.layer_name].slot_mapping``. RoPE covers dims
    ``[0, rotary_dim)``.
``raw_k`` / ``raw_pos``
    ``attn.indexer.raw_key_cache.kv_cache``: the per-request ring
    ``[pages, ring, 1, key_width (+ 3 int64 M-RoPE positions)]``, slot
    ``metadata[raw_key_cache.prefix].slot_mapping`` (only a request's last
    ``ring`` rows of the batch get one; the rest are -1).
``compressed_k``
    ``attn.indexer.compressed_key_cache.kv_cache``
    ``[pages, block_size / ratio, 1, key_width]``, slot
    ``metadata[compressed_key_cache.prefix].slot_mapping`` (one per group
    closed in this batch, -1 elsewhere).

Main K/V are per TP rank (one KV head per rank at TP4); the indexer caches
are replicated. Each worker writes ``rank<rank>.pt``; see
``docs/explore/ridgefill/cache_dump.py`` for the driver and the comparison.
The probe needs no import of its own package, so a copy of this file can be
put on ``PYTHONPATH`` as a top-level module to probe a read-only tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def gather_slots(cache: Any, slots: Any) -> Any:
    """Rows of a paged ``[pages, page_size, heads, width]`` cache at ``slots``.

    The slot is ``page * page_size + offset`` for the flash K/V writer and the
    QSA row stores alike. Advanced indexing copies, so any strides work.
    """
    page_size = cache.shape[1]
    return cache[slots // page_size, slots % page_size]


def scatter_slots(cache: Any, slots: Any, rows: Any) -> None:
    """Write ``rows`` back at ``slots``; the inverse of ``gather_slots``."""
    page_size = cache.shape[1]
    cache[slots // page_size, slots % page_size] = rows


def _gather_positions(positions: Any, index: Any) -> Any:
    if positions is None:
        return None
    if positions.dim() == 1:
        return positions.index_select(0, index)
    return positions.index_select(-1, index)


def _row_rms(rows: Any) -> Any:
    return rows.float().pow(2).mean(-1).sqrt().cpu()


def _attn_metadata() -> Any:
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = context.attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0]
    return context, metadata


class _DumpState:
    """Capture state of one worker; the patched op closes over it."""

    def __init__(self, out_dir: str, forwards: int, skip: int, rank: int) -> None:
        self.out_dir = Path(out_dir)
        self.forwards = int(forwards)
        self.skip = int(skip)
        self.rank = int(rank)
        self.records: list[dict[str, Any]] = []
        self.seen = 0
        self.calls = 0
        self._context: Any = None
        self._metadata: Any = None

    def _new_forward(self, context: Any, metadata: Any) -> None:
        # The runner builds a new context and metadata per forward. Holding
        # both keeps their ids from being reused by a later forward.
        if context is not self._context or metadata is not self._metadata:
            self._context = context
            self._metadata = metadata
            self.seen += 1

    def call(
        self,
        original: Any,
        block_input: Any,
        positions: Any,
        filled_index: Any,
        exact_index: Any,
        layer_name: str,
    ) -> Any:
        context, metadata = _attn_metadata()
        self._new_forward(context, metadata)
        forward = self.seen - 1
        if not (self.skip <= forward < self.skip + self.forwards):
            return original(
                block_input, positions, filled_index, exact_index, layer_name
            )
        if not isinstance(metadata, dict):
            raise RuntimeError("ridgefill dump needs dict attention metadata")
        self.calls += 1
        return self._capture(
            forward, metadata, block_input, positions, filled_index, layer_name
        )

    def _slots(self, attn: Any, metadata: dict, filled_index: Any) -> dict:
        indexer = attn.indexer
        names = {
            "main": attn.layer_name,
            "raw": indexer.raw_key_cache.prefix,
            "compressed": indexer.compressed_key_cache.prefix,
        }
        slots = {}
        for kind, name in names.items():
            full = metadata[name].slot_mapping.index_select(0, filled_index).long()
            rows = (full >= 0).nonzero().squeeze(1)
            slots[kind] = {
                "all": full,
                "rows": rows,
                "valid": full.index_select(0, rows),
            }
        return slots

    def _snapshot(self, attn: Any, slots: dict) -> dict:
        import torch

        out: dict[str, Any] = {}
        kv = attn.kv_cache
        if kv.dtype in (torch.float16, torch.bfloat16) and kv.dim() >= 4:
            key_cache, value_cache = kv.transpose(1, 2).split(attn.head_dim, dim=-1)
            main = slots["main"]["valid"]
            out["main_k"] = gather_slots(key_cache, main).float().cpu()
            out["main_v"] = gather_slots(value_cache, main).float().cpu()
        raw = attn.indexer.raw_key_cache
        rows = gather_slots(raw.kv_cache, slots["raw"]["valid"])
        width = int(getattr(raw, "key_head_size", rows.shape[-1]))
        out["raw_k"] = rows[..., :width].float().cpu()
        if getattr(raw, "cache_rope_positions", False):
            offset = int(raw.rope_position_offset)
            out["raw_pos"] = rows[..., offset:].contiguous().view(torch.int64).cpu()
        compressed = attn.indexer.compressed_key_cache.kv_cache
        valid = slots["compressed"]["valid"]
        out["compressed_k"] = gather_slots(compressed, valid).float().cpu()
        return out

    def _capture(
        self,
        forward: int,
        metadata: dict,
        block_input: Any,
        positions: Any,
        filled_index: Any,
        layer_name: str,
    ) -> Any:
        from vllm.model_executor.layers.ridgefill import ops as rf_ops

        layer = rf_ops._LAYERS[layer_name]  # noqa: SLF001
        attn = layer.self_attn
        slots = self._slots(attn, metadata, filled_index)
        ring = attn.indexer.raw_key_cache.kv_cache
        ring_slots = slots["raw"]["valid"]
        ring_before = gather_slots(ring, ring_slots).clone()

        fallback = False
        snap_a = None
        try:
            attn.cache_only_forward(
                hidden_states=block_input.index_select(0, filled_index),
                positions=_gather_positions(positions, filled_index),
                row_index=filled_index,
            )
            snap_a = self._snapshot(attn, slots)
        except NotImplementedError:
            # The op would run the full stock call here: A is B.
            fallback = True

        scatter_slots(ring, ring_slots, ring_before)
        out_b = attn(hidden_states=block_input, positions=positions)
        snap_b = self._snapshot(attn, slots)
        scatter_slots(ring, ring_slots, ring_before)
        out_b2 = attn(hidden_states=block_input, positions=positions)
        snap_b2 = self._snapshot(attn, slots)

        session = rf_ops._SESSION  # noqa: SLF001
        split = getattr(getattr(session, "cfg", None), "split", None)
        layer_idx = int(getattr(layer, "layer_idx", -1))
        stock = out_b.index_select(0, filled_index)
        stock2 = out_b2.index_select(0, filled_index)
        self.records.append(
            {
                "forward": forward,
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "split": split,
                # Before the split the filled rows' attention output feeds
                # their residual, hence the split state and every map input.
                "consumed": split is not None and layer_idx < int(split),
                "rows": int(block_input.shape[0]),
                "filled": int(filled_index.numel()),
                "fallback": fallback,
                "rotary_dim": getattr(attn.rotary_emb, "rotary_dim", None),
                "positions": _gather_positions(positions, filled_index).cpu(),
                "slots": {k: v["all"].cpu() for k, v in slots.items()},
                "slot_rows": {k: v["rows"].cpu() for k, v in slots.items()},
                "A": snap_a,
                "B": snap_b,
                "B2": snap_b2,
                "out": {
                    "a_zero": not fallback,
                    "block_rms": _row_rms(block_input.index_select(0, filled_index)),
                    "stock_rms": _row_rms(stock),
                    "floor_rms": _row_rms(stock2.float() - stock.float()),
                },
            }
        )
        return out_b2

    def save(self) -> dict:
        import torch

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"rank{self.rank}.pt"
        torch.save({"rank": self.rank, "records": self.records}, path)
        return {
            "rank": self.rank,
            "path": str(path),
            "filled_forwards_seen": self.seen,
            "captured_calls": self.calls,
            "records": len(self.records),
            "layers": sorted({r["layer_idx"] for r in self.records}),
        }


class RidgeFillDumpProbe:
    """Extension with three RPC methods; it overrides no worker method.

    ``worker_extension_cls`` makes this a base of the worker, so the methods
    run with the worker as ``self`` (``self.rank`` is its TP rank)::

        llm.collective_rpc("ridgefill_dump_arm", args=(out_dir, 2, 0))
        llm.generate(prompt, params)
        llm.collective_rpc("ridgefill_dump_save")
    """

    def ridgefill_dump_arm(
        self, out_dir: str, forwards: int = 2, skip: int = 0
    ) -> None:
        """Patch the cache-only op in this worker for the next filled forwards.

        Filled forwards run eagerly and look the op up on ``torch.ops.vllm``
        at every call, so replacing the cached packet attribute reaches them.
        The compiled graph never holds the op: it is the flag-off graph.
        """
        import torch

        from vllm.model_executor.layers.ridgefill import ops as rf_ops  # noqa: F401

        self.ridgefill_dump_disarm()
        original = torch.ops.vllm.ridgefill_cache_only
        state = _DumpState(out_dir, forwards, skip, int(self.rank))  # type: ignore[attr-defined]

        def patched(block_input, positions, filled_index, exact_index, layer_name):
            return state.call(
                original, block_input, positions, filled_index, exact_index, layer_name
            )

        self._ridgefill_dump = (state, original)
        torch.ops.vllm.ridgefill_cache_only = patched

    def ridgefill_dump_save(self) -> dict:
        """Write ``rank<rank>.pt`` and return this rank's capture summary."""
        state, _ = self._ridgefill_dump
        return state.save()

    def ridgefill_dump_disarm(self) -> None:
        """Put the registered op packet back."""
        import torch

        armed = getattr(self, "_ridgefill_dump", None)
        if armed is not None:
            torch.ops.vllm.ridgefill_cache_only = armed[1]
