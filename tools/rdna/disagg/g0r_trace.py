# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step state tracer of the DisaggRework spike (docs/rdna2/disagg-g0-2026-11-06.md).

Wraps ``execute_model`` and ``sample_tokens`` of the V2 model runner on one TP rank
and records, for every prefill step of a request, which chunk ran (``num_computed``,
``num_scheduled``, the request's block ids per KV group) and a fingerprint of the cache
pages that hold its state:

* ``pre`` snapshot, first step of a request only: the state the step starts from. On a
  decode instance that is the state the connector loaded; after a prefix hit it is the
  state restored from the cache.
* ``post`` snapshot after every prefill step (after ``sample_tokens``, so the drafter's
  writes are in): the state the next chunk starts from.

State pages of Mamba groups (running block and the one before it) and the QSA raw ring
are also saved byte for byte when the plan asks for it (``raw``: the ``pre`` snapshot
and the ``post`` snapshot at ``raw_at``); of attention-type groups (QSA compressed
keys) only the block that holds the last position is saved (its unused tail is stale,
so its fingerprint does not compare across runs). Every block is fingerprinted. Pages
are addressed through the cache layout (``kv_cache_tensors``: ``offset + layer *
layer_stride + block * block_stride`` into the one backing allocation), not through the
connector, so engines without a connector are traced the same way and the keys are
comparable across arms: ``(group, column, layer index within the group)``.

Driven through ``collective_rpc`` (``g0a_ext.G0aWorkerExtension``): ``g0r_begin`` arms
the tracer for the next request, ``g0r_end`` writes ``rank<r>/<tag>/steps.json``.
Nothing runs while the tracer is not armed, and an error in the tracer is recorded,
never raised into the engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from typing import Any

import g0_rt
import numpy as np
import torch

_STATE_KINDS = ("MambaSpec", "CircularBufferSpec")
_CHUNK_BLOCKS = 16  # blocks fingerprinted per index_select (bounds the int64 temporary)


class StateTracer:
    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.runner = worker.model_runner
        self.rank = int(getattr(worker, "rank", 0))
        self.armed = False
        self.installed = False
        self.errors: list[str] = []
        self.tag = ""
        self.out_dir = ""
        self.plan: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.blocks: dict[str, list[list[int]]] = {}
        self.prompt_len: dict[str, int] = {}
        self.pending: list[dict[str, Any]] = []
        self.step_no = 0
        self._layout()

    # ------------------------------------------------------------------ layout
    def _layout(self) -> None:
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        cfg = self.runner.kv_cache_config
        self.num_blocks = int(cfg.num_blocks)
        group_of: dict[str, int] = {}
        spec_of: dict[str, Any] = {}
        self.kinds: list[str] = []
        self.group_block_size: list[int] = []
        for g, grp in enumerate(cfg.kv_cache_groups):
            spec = grp.kv_cache_spec
            first = None
            for name in grp.layer_names:
                s = (
                    spec.kv_cache_specs[name]
                    if isinstance(spec, UniformTypeKVCacheSpecs)
                    else spec
                )
                group_of[name] = g
                spec_of[name] = s
                first = first or s
            self.kinds.append(type(first).__name__)
            self.group_block_size.append(int(first.block_size))
        # per group: (layer name, byte address of block 0, block stride, page bytes)
        self.entries: list[list[tuple[str, int, int, int]]] = [
            [] for _ in cfg.kv_cache_groups
        ]
        for t in cfg.kv_cache_tensors:
            for i, name in enumerate(t.layers):
                s = spec_of[name]
                self.entries[group_of[name]].append(
                    (
                        name,
                        int(t.offset) + i * int(t.layer_stride),
                        int(t.block_stride),
                        int(s.page_size_bytes),
                    )
                )
        cache0 = self.runner.kv_caches[0]
        self.device = cache0.device
        self.raw = torch.empty(0, dtype=torch.uint8, device=self.device)
        self.raw.set_(cache0.untyped_storage())
        need = max(
            (a + (self.num_blocks - 1) * st + pg)
            for ents in self.entries
            for _, a, st, pg in ents
        )
        if need > self.raw.numel():
            raise RuntimeError(
                f"layout needs {need} bytes, backing storage has {self.raw.numel()}"
            )
        self._weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def probe(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "num_blocks": self.num_blocks,
            "raw_bytes": int(self.raw.numel()),
            "groups": [
                {
                    "group": g,
                    "kind": self.kinds[g],
                    "block_size": self.group_block_size[g],
                    "layers": [e[0] for e in ents],
                    "page_bytes": sorted({e[3] for e in ents}),
                }
                for g, ents in enumerate(self.entries)
            ],
        }

    # --------------------------------------------------------------- fingerprint
    def _w(self, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self._weights.get(width)
        if w is None:
            idx = torch.arange(width, device=self.device, dtype=torch.int64)
            w = (idx * 2654435761 + 1, (idx ^ 0x5BD1E995) * 40503 + 7)
            self._weights[width] = w
        return w

    def _fp(self, pages: torch.Tensor) -> torch.Tensor:
        """[nb, page] uint8 -> [nb, 2] int64 (two position-weighted wrapping sums)."""
        assert pages.shape[1] % 4 == 0, pages.shape
        y = pages.contiguous().view(torch.int32).to(torch.int64)
        w1, w2 = self._w(y.shape[1])
        return torch.stack(
            [(y * w1).sum(dim=1), ((y ^ 0x9E3779B9) * w2).sum(dim=1)], dim=1
        )

    def _pages(
        self, addr0: int, stride: int, page: int, bids: list[int]
    ) -> torch.Tensor:
        idx = torch.as_tensor(bids, device=self.device, dtype=torch.int64)
        view = self.raw.as_strided((self.num_blocks, page), (stride, 1), addr0)
        return view.index_select(0, idx)

    # ---------------------------------------------------------------- snapshots
    def _cols(self, g: int, pos: int, nblocks: int) -> list[int]:
        kind = self.kinds[g]
        if kind == "CircularBufferSpec":
            return [0] if nblocks else []
        if kind == "MambaSpec":
            if pos <= 0:
                return [0] if nblocks else []
            run = -(-pos // self.group_block_size[g]) - 1  # ceil(pos / bs) - 1
            return [c for c in (run - 1, run) if 0 <= c < nblocks]
        return list(range(nblocks))

    def _save_raw(self, g: int, col: int, pos: int) -> bool:
        """State groups: every fingerprinted column. Attention groups: the block that
        holds position ``pos - 1``, whose unused tail may be stale (blocks are not
        cleared), so its whole-page fingerprint compares more than the state."""
        if self.kinds[g] in _STATE_KINDS:
            return True
        return col == max(pos - 1, 0) // self.group_block_size[g]

    def snapshot(
        self, req_id: str, pos: int, raw: bool, raw_path: str
    ) -> dict[str, Any]:
        """Fingerprint (and optionally save) the pages holding ``req_id``'s state at
        token position ``pos``. Returns the JSON-able record."""
        g0_rt.sync()
        blocks = self.blocks[req_id]
        rec: dict[str, Any] = {
            "pos": pos,
            "fp": {},
            "kinds": self.kinds,
            "bs": self.group_block_size,
        }
        jobs: list[tuple[int, int, list[tuple[int, int]], torch.Tensor]] = []
        raw_items: list[dict[str, Any]] = []
        raw_buf: list[np.ndarray] = []
        raw_off = 0
        for g, ents in enumerate(self.entries):
            cols = [
                (c, blocks[g][c])
                for c in self._cols(g, pos, len(blocks[g]))
                if blocks[g][c] > 0
            ]
            if not cols or not ents:
                continue
            rec["fp"][str(g)] = {
                str(c): {"b": b, "layers": [None] * len(ents)} for c, b in cols
            }
            for li, (name, addr0, stride, page) in enumerate(ents):
                for lo in range(0, len(cols), _CHUNK_BLOCKS):
                    part = cols[lo : lo + _CHUNK_BLOCKS]
                    pages = self._pages(addr0, stride, page, [b for _, b in part])
                    jobs.append((g, li, part, self._fp(pages)))
                    if raw:
                        keep = [
                            k
                            for k, (c, _b) in enumerate(part)
                            if self._save_raw(g, c, pos)
                        ]
                        if keep:
                            host = pages[keep].cpu().numpy()
                        for j, k in enumerate(keep):
                            c, b = part[k]
                            buf = np.ascontiguousarray(host[j])
                            raw_items.append(
                                {
                                    "g": g,
                                    "col": c,
                                    "b": b,
                                    "layer": li,
                                    "name": name,
                                    "off": raw_off,
                                    "n": int(buf.nbytes),
                                    "h": hashlib.blake2b(
                                        memoryview(buf), digest_size=16
                                    ).hexdigest(),
                                }
                            )
                            raw_buf.append(buf)
                            raw_off += int(buf.nbytes)
        if jobs:
            flat = torch.cat([j[3] for j in jobs], dim=0).cpu().tolist()
            at = 0
            for g, li, part, _ in jobs:
                for (c, _b), h in zip(part, flat[at : at + len(part)]):
                    rec["fp"][str(g)][str(c)]["layers"][li] = h
                at += len(part)
        if raw_items:
            with open(raw_path + ".bin", "wb") as f:
                for buf in raw_buf:
                    f.write(memoryview(buf))
            with open(raw_path + ".json", "w") as f:
                json.dump(
                    {
                        "rank": self.rank,
                        "pos": pos,
                        "kinds": self.kinds,
                        "bs": self.group_block_size,
                        "items": raw_items,
                    },
                    f,
                )
            rec["raw"] = os.path.basename(raw_path)
            rec["raw_bytes"] = raw_off
        return rec

    # ------------------------------------------------------------------ control
    def begin(self, out_dir: str, tag: str, plan: dict[str, Any]) -> dict[str, Any]:
        self.out_dir = os.path.join(out_dir, tag, f"rank{self.rank}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.tag = tag
        self.plan = plan
        self.steps = []
        self.errors = []
        self.blocks = {}
        self.prompt_len = {}
        self.pending = []
        self.step_no = 0
        if not self.installed:
            self._install()
        self.armed = True
        return {"rank": self.rank, "armed": True, "dir": self.out_dir}

    def end(self) -> dict[str, Any]:
        self.armed = False
        with open(os.path.join(self.out_dir, "steps.json"), "w") as f:
            json.dump(
                {
                    "tag": self.tag,
                    "rank": self.rank,
                    "plan": self.plan,
                    "steps": self.steps,
                    "errors": self.errors,
                },
                f,
            )
        return {
            "rank": self.rank,
            "steps": len(self.steps),
            "errors": len(self.errors),
            "last_error": self.errors[-1][-400:] if self.errors else None,
        }

    def _err(self, where: str) -> None:
        self.errors.append(f"{where}: {traceback.format_exc()}")

    # ------------------------------------------------------------------- hooks
    def _install(self) -> None:
        runner = self.runner
        orig_exec = runner.execute_model
        orig_sample = runner.sample_tokens
        tracer = self

        def execute_model(scheduler_output, *args, **kwargs):
            if tracer.armed:
                try:
                    tracer._before(scheduler_output)
                except Exception:  # noqa: BLE001 - the tracer must never break serving
                    tracer._err("before")
            return orig_exec(scheduler_output, *args, **kwargs)

        def sample_tokens(*args, **kwargs):
            out = orig_sample(*args, **kwargs)
            if tracer.armed and tracer.pending:
                try:
                    tracer._after()
                except Exception:  # noqa: BLE001
                    tracer._err("after")
            return out

        runner.execute_model = execute_model
        runner.sample_tokens = sample_tokens
        self.installed = True

    def _before(self, so: Any) -> None:
        self.pending = []
        self.step_no += 1
        todo: list[tuple[str, int, bool]] = []  # (request, num_computed, first step)
        for nr in so.scheduled_new_reqs:
            self.blocks[nr.req_id] = [list(ids) for ids in nr.block_ids]
            self.prompt_len[nr.req_id] = int(nr.prompt_len)
            todo.append((nr.req_id, int(nr.num_computed_tokens), True))
        cached = so.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            if rid not in self.blocks:
                continue
            new_ids = cached.new_block_ids[i]
            if new_ids is not None:
                for g, ids in enumerate(new_ids):
                    self.blocks[rid][g].extend(ids)
            todo.append((rid, int(cached.num_computed_tokens[i]), False))
        for rid, computed, first in todo:
            n = int(so.num_scheduled_tokens.get(rid, 0))
            plen = self.prompt_len[rid]
            if n <= 0 or computed >= plen:
                continue  # not a prefill step
            rec: dict[str, Any] = {
                "step": self.step_no,
                "req": rid,
                "computed": computed,
                "sched": n,
                "prompt_len": plen,
                "blocks": {str(g): list(ids) for g, ids in enumerate(self.blocks[rid])},
            }
            if first:
                path = os.path.join(self.out_dir, f"raw_pre_{computed}")
                rec["pre"] = self.snapshot(
                    rid, computed, bool(self.plan.get("raw")), path
                )
            self.steps.append(rec)
            self.pending.append(rec)
        for rid in so.finished_req_ids:
            self.blocks.pop(rid, None)
            self.prompt_len.pop(rid, None)

    def _after(self) -> None:
        raw_at = self.plan.get("raw_at") or []
        for rec in self.pending:
            pos = rec["computed"] + rec["sched"]
            raw = bool(self.plan.get("raw")) and pos in raw_at
            rec["post"] = self.snapshot(
                rec["req"], pos, raw, os.path.join(self.out_dir, f"raw_post_{pos}")
            )
            try:
                idx = self.runner.req_states.req_id_to_index.get(rec["req"])
                if idx is not None and rec["req"] in self.blocks:
                    counts = self.runner.block_tables.num_blocks.np[:, idx].tolist()
                    mine = [len(x) for x in self.blocks[rec["req"]]]
                    rec["table_counts_ok"] = counts == mine
                    if counts != mine:
                        rec["table_counts"] = [counts, mine]
            except Exception:  # noqa: BLE001
                rec["table_counts_ok"] = None
        self.pending = []
