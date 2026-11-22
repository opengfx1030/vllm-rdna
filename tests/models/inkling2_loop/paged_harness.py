# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine emulation for the CPU tests of the paged Inkling-2 serving code.

``vllm.models.inkling2_loop.amd`` reads its caches through per-KV-cache-group block
tables and slot mappings produced by the engine.  This module reproduces what the engine
hands over, on torch CPU tensors:

* one :class:`Group` per KV-cache group (owners of one spec share block ids, the block
  table and the slot mapping; every group has its own block size);
* the sliding-window manager: before a step, blocks wholly below
  ``num_computed - window + 1`` are freed, and the forward block table shows them as the
  null block 0 (``BlockTables.gather_block_tables``);
* cache pages in the engine's block-outermost ``[B, bs, H, C]`` layout seen as the
  non-contiguous ``[B, H, bs, C]`` view the owners index (the conv owners' ring pages
  are ``[B, S, H, W]`` seen as ``[B, H, S, W]``);
* NaN poison: caches start as NaN, pages are re-poisoned when they are allocated and
  when they are freed, and the null block must stay NaN forever.  Reading anything this
  request did not write turns the result into NaN; writing through another group's
  metadata hits the null-block check.
* prefix-cache hits: ``PagedWorld.fork_prefix`` starts a request on blocks that
  another request computed (shared and reference counted; a windowed group keeps only
  the blocks its window still needs, the blocks before are the null block).  A hit never
  reads blocks another request completes in the same step: ``Inkling2ConvStateManager``
  defers it (``test_conv_state_manager.py``);
* speculative rollback: ``PagedWorld.commit(work, accepted)`` keeps only the accepted
  prefix of a verify step's tokens.
"""

from __future__ import annotations

import contextlib
import random
import types

import torch

from vllm.models.inkling2_loop.amd import state as state_mod
from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def kv_cache(
    num_blocks: int, heads: int, block_size: int, width: int, dtype, device="cpu"
):
    """``[B, H, bs, C]`` logical view of a ``[B, bs, H, C]`` page array (the
    engine's block-outermost ``LBNHC`` layout), NaN everywhere."""
    page = torch.full(
        (num_blocks, block_size, heads, width), float("nan"), dtype=dtype, device=device
    )
    return page.permute(0, 2, 1, 3)


class Group:
    """One KV-cache group."""

    def __init__(
        self, name: str, block_size: int, window: int | None, num_blocks: int, seed: int
    ) -> None:
        self.name = name
        self.bs = block_size
        self.window = window  # None: full attention (never frees)
        ids = list(range(1, num_blocks))  # block 0 is the null block
        random.Random(seed).shuffle(ids)
        self.free = ids
        self.owners: list = []
        self.tables: dict[int, list[int]] = {}
        self.refs: dict[int, int] = {}  # block -> requests holding it

    def _poison(self, blk: int) -> None:
        for owner in self.owners:
            owner.kv_cache[blk] = float("nan")

    def release(self, blk: int) -> None:
        """Drop one holder of ``blk``; the last one poisons the page and frees it."""
        self.refs[blk] -= 1
        if self.refs[blk] == 0:
            del self.refs[blk]
            self._poison(blk)
            self.free.append(blk)

    def remove_skipped(self, req: int, num_computed: int) -> None:
        """``SlidingWindowManager.remove_skipped_blocks``."""
        if self.window is None:
            return
        skipped_tokens = max(0, num_computed - self.window + 1)
        table = self.tables[req]
        for i in range(min(skipped_tokens // self.bs, len(table))):
            if table[i] != 0:
                self.release(table[i])
                table[i] = 0

    def allocate(self, req: int, num_tokens: int) -> None:
        table = self.tables.setdefault(req, [])
        while len(table) < cdiv(num_tokens, self.bs):
            blk = self.free.pop()
            self._poison(blk)
            self.refs[blk] = 1
            table.append(blk)

    def share(self, src: int, dst: int, num_tokens: int) -> None:
        """Prefix-cache hit: ``dst`` starts from the first ``num_tokens`` tokens held in
        the blocks of ``src`` (shared, reference counted).  A windowed group only keeps
        the blocks of the last ``window - 1`` tokens, the ones before are the null block
        (``SlidingWindowManager.find_longest_cache_hit``)."""
        assert num_tokens % self.bs == 0, "hits are block aligned"
        end = num_tokens // self.bs
        first = 0
        if self.window is not None:
            first = max(0, num_tokens - (self.window - 1)) // self.bs
        needed = self.tables[src][first:end]
        assert len(needed) == end - first and 0 not in needed, "the hit blocks are gone"
        for blk in needed:
            self.refs[blk] += 1
        self.tables[dst] = [0] * first + needed

    def row(self, req: int, width: int, first_pos: int) -> list[int]:
        """Block-table row of the forward pass: entries wholly below the window of
        the first scheduled token are the null block."""
        row = self.tables[req] + [0] * (width - len(self.tables[req]))
        if self.window is not None:
            num_null = max(first_pos - (self.window - 1), 0) // self.bs
            row[:num_null] = [0] * num_null
        return row

    def finish(self, req: int) -> None:
        for blk in self.tables.pop(req, []):
            if blk != 0:
                self.release(blk)


class PagedWorld:
    """Caches, block tables and per-step metadata of a set of cache owners.

    ``owners`` is a list of ``(owner, group_name)``; ``groups`` maps a group name to
    ``(block_size, window)``.
    """

    def __init__(
        self,
        owners,
        groups: dict[str, tuple[int, int | None]],
        kv_dtype,
        max_len: int = 256,
        num_blocks: int = 128,
        seed: int = 0,
        device="cpu",
        hip: bool = False,
        max_tokens: int = 1024,
    ) -> None:
        self.device = torch.device(device)
        self.groups = {
            name: Group(name, bs, window, num_blocks, seed + i)
            for i, (name, (bs, window)) in enumerate(groups.items())
        }
        self.max_len = max_len
        for owner, gname in owners:
            g = self.groups[gname]
            if isinstance(owner, state_mod.Inkling2KVOwner):
                owner.kv_cache = kv_cache(
                    num_blocks,
                    owner.num_kv_heads,
                    g.bs,
                    2 * owner.head_dim,
                    kv_dtype,
                    self.device,
                )
            else:
                # the engine's cache block size is the owner's token span per block
                owner.block_size = g.bs
                owner.kv_cache = kv_cache(
                    num_blocks,
                    owner.num_kv_heads,
                    owner.ring_rows,
                    owner.row_width,
                    kv_dtype,
                    self.device,
                )
            g.owners.append(owner)
        self.computed: dict[int, int] = {}
        # ``hip``: ``seq_idx`` comes from the persistent device scratch of the HIP
        # serving path (what ``Inkling2MetadataBuilder`` does on a GPU); otherwise torch
        self.scratch = state_mod.SeqIdxScratch(max_tokens, self.device) if hip else None

    def metadata(
        self, work: list[tuple[int, int]]
    ) -> dict[str, state_mod.Inkling2Metadata]:
        """Allocate for ``work = [(request, num_new_tokens)]`` (batch order) and
        build the per-owner :class:`Inkling2Metadata` like the engine does."""
        n_tok = [n for _, n in work]
        starts = [0]
        for n in n_tok:
            starts.append(starts[-1] + n)
        seq_lens = [self.computed.get(r, 0) + n for r, n in work]
        out: dict[str, state_mod.Inkling2Metadata] = {}
        for g in self.groups.values():
            width = cdiv(self.max_len, g.bs) + 1
            rows, slots = [], []
            for r, n in work:
                p0 = self.computed.get(r, 0)
                g.tables.setdefault(r, [])
                g.remove_skipped(r, p0)
                g.allocate(r, p0 + n)
                rows.append(g.row(r, width, p0))
                table = g.tables[r]
                slots += [table[p // g.bs] * g.bs + p % g.bs for p in range(p0, p0 + n)]
            common = types.SimpleNamespace(
                num_reqs=len(work),
                num_actual_tokens=starts[-1],
                query_start_loc=torch.tensor(
                    starts, dtype=torch.int32, device=self.device
                ),
                query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
                block_table_tensor=torch.tensor(
                    rows, dtype=torch.int32, device=self.device
                ),
                slot_mapping=torch.tensor(slots, dtype=torch.int64, device=self.device),
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
                max_query_len=max(n_tok),
                max_seq_len=max(seq_lens),
            )
            md = state_mod.make_metadata(common, self.scratch)
            for owner in g.owners:
                out[owner.prefix] = md
        return out

    def commit(
        self, work: list[tuple[int, int]], accepted: dict[int, int] | None = None
    ) -> None:
        """End of a step; ``accepted[r]`` of the ``n`` tokens of request ``r`` stay
        computed (a verify step whose draft tokens were rejected), all by default."""
        for r, n in work:
            self.computed[r] = self.computed.get(r, 0) + (accepted or {}).get(r, n)
        for g in self.groups.values():  # nothing may ever be written to the null block
            for owner in g.owners:
                assert torch.isnan(owner.kv_cache[0]).all(), (
                    f"{owner.prefix}: the null block was written"
                )

    def positions(self, work: list[tuple[int, int]]) -> torch.Tensor:
        return torch.cat(
            [
                torch.arange(self.computed.get(r, 0), self.computed.get(r, 0) + n)
                for r, n in work
            ]
        ).to(self.device)

    def finish(self, req: int) -> None:
        for g in self.groups.values():
            g.finish(req)
        self.computed.pop(req, None)

    def fork_prefix(self, src: int, dst: int, num_tokens: int) -> None:
        """``dst`` is a prefix-cache hit of ``num_tokens`` tokens on the blocks of
        ``src`` (call it before ``src`` frees them: its next ``metadata`` does)."""
        assert dst not in self.computed
        assert self.computed[src] >= num_tokens
        for g in self.groups.values():
            g.share(src, dst, num_tokens)
        self.computed[dst] = num_tokens


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def fake_vllm_config(
    model_dir,
    hf_config,
    dtype=torch.float16,
    max_num_batched_tokens: int = 1024,
    block_size: int = 64,
    num_speculative_tokens: int = 0,
):
    """The slice of ``VllmConfig`` that the owners and the model read."""
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=hf_config, dtype=dtype, model=str(model_dir), revision=None
        ),
        cache_config=types.SimpleNamespace(block_size=block_size),
        speculative_config=types.SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        )
        if num_speculative_tokens
        else None,
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens
        ),
    )


def owners_of(layers):
    """``[(owner, group_name)]`` of every layer visit, as the engine groups them."""
    owners = []
    for layer in layers:
        for kv, conv in zip(layer.kv_owners, layer.conv_owners):
            owners.append((kv, "swa" if kv.sliding_window is not None else "full"))
            owners.append((conv, "conv"))
    return owners


def split(rng: random.Random, n: int, sizes=(1, 1, 2, 3, 5, 16, 33)) -> list[int]:
    out: list[int] = []
    while sum(out) < n:
        out.append(min(rng.choice(sizes), n - sum(out)))
    return out


def schedule(rng: random.Random, lengths: list[int]):
    """Random engine steps ``[[(request, start, stop)]]``: every request's tokens in
    order and in random chunks, requests of a step in shuffled batch order."""
    chunks = [split(rng, n) for n in lengths]
    done = [0] * len(lengths)
    cur = [0] * len(lengths)
    steps = []
    while any(cur[i] < len(chunks[i]) for i in range(len(lengths))):
        ready = [i for i in range(len(lengths)) if cur[i] < len(chunks[i])]
        rng.shuffle(ready)
        step = []
        for i in ready[: rng.randint(1, len(ready))]:
            n = chunks[i][cur[i]]
            step.append((i, done[i], done[i] + n))
            done[i] += n
            cur[i] += 1
        steps.append(step)
    return steps


def text_config(**overrides) -> Inkling2LoopTextConfig:
    """Text config of the default tiny checkpoint."""
    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "rdna"
        / "inkling2"
        / "make_tiny_checkpoint.py"
    )
    name = "make_tiny_checkpoint"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
        spec.loader.exec_module(module)
    return Inkling2LoopTextConfig(
        **sys.modules[name].TinySpec(**overrides).text_config()
    )
