# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executable model of the ``hx`` protocol (numpy + ``hx_proto`` only, no GPU).

* ``dispatch_ref``: the hub's spill/copy rule, in two independent forms (a loop over
  pairs, and the 512-bit touched bitmap + prefix popcount of the device kernel).
* ``combine_ref``: the fixed-order fp32 sum of the partials, NaN for an aborted step.
* ``toy_moe``: a small dense MoE, to check that the pairs partition exactly.
* ``HxSim``: a deterministic simulator of one hub and ``nsrv`` expert servers sharing
  memory through posted stores, written as the documentation of the state machine of
  ``csrc/rocm/hetero_moe/hx_core.cuh`` (the kernels are the reference where
  ``docs/rdna2/hetero-moe-design-2026-11-07.md`` is silent).

The simulator runs the processes as generators on a virtual clock (ticks): an operation
that has to wait yields the time at which it wants to run again, so a test controls the
interleaving with ``delay`` calls, the order of the task dict and the optional seed.
Stores are posted: a store becomes visible ``link_ticks(nbytes)`` ticks after it was
issued, and a fence stalls its process until everything it posted is visible. With the
default adversarial order two stores issued without a fence between them may become
visible in either order (a small flag overtakes a large payload), so removing a fence
corrupts or aborts a run; ``order="fifo"`` keeps program order for the control run.
"""

import random
from bisect import bisect_left
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from vllm.distributed.hetero_moe import hx_proto

HX_BITMAP_WORDS = hx_proto.HX_MAX_EXPERTS // 32  # 512 slots in 16 u32 words
FENCES_ALL = frozenset({"hdr", "dispatch", "push"})


def _need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Dispatch reference
# ---------------------------------------------------------------------------
def _layer_table(a: np.ndarray, ndim: int, layer: int, name: str) -> np.ndarray:
    """The table of one layer: ``a`` itself, or ``a[layer]`` of a stacked table."""
    a = np.asarray(a)
    if a.ndim == ndim + 1:
        _need(0 <= layer < a.shape[0], f"{name} has {a.shape[0]} layers, not {layer}")
        return a[layer]
    _need(a.ndim == ndim, f"{name} needs {ndim} or {ndim + 1} dims, got {a.ndim}")
    return a


def _popcount32(v: np.ndarray) -> np.ndarray:
    """Set bits of every uint32 element (int64)."""
    raw = np.ascontiguousarray(v, dtype="<u4")[..., None].view(np.uint8)
    return np.unpackbits(raw, axis=-1).sum(axis=-1, dtype=np.int64)


def dispatch_ref(
    x: np.ndarray | None,
    ids: np.ndarray,
    w: np.ndarray | None,
    layer: int,
    slot_map: np.ndarray,
    n_cold: np.ndarray,
    cold_srv: np.ndarray,
    kmax: int,
    nsrv: int,
    owned: np.ndarray | None = None,
    *,
    form: str = "loop",
) -> tuple[np.ndarray, np.ndarray, int]:
    """The hub's dispatch rule of layer ``layer`` (design 6.2, equal to hx_core.cuh).

    ``ids`` is ``[T, K]`` ints; ``x`` (``[T, H]``) and ``w`` (``[T, K]``) are only shape
    checked, the hub copies them unchanged to every server. The tables are those of the
    layer (``slot_map[E]``, ``n_cold[E]``, ``cold_srv[E, MAXC]``, optional ``owned[E]``)
    or stacked over layers (one more leading dimension, indexed by ``layer``).

    A pair is valid iff ``0 <= id < E`` and the expert is owned; a pair of a non-owned
    expert is silently dropped. A valid pair is resident iff its expert has a hub slot.
    The touched slots are those of the resident pairs and a resident pair stays on the
    hub iff fewer than ``kmax`` touched slots are below its slot. Every other valid
    pair goes to server ``cold_srv[id][(row + layer) % n_cold[id]]``; with
    ``n_cold <= 0`` or a server outside ``[0, nsrv)`` it is a plan error: it goes
    nowhere and is counted.

    Returns ``(hub_ids[T, K] int32, srv_ids[nsrv, T, K] int16, plan_errors)``: the id
    where the pair goes to that destination, else -1. ``form`` selects one of two
    independent implementations: ``"loop"`` (plain Python over the pairs) or
    ``"bitmap"`` (a 512-bit touched bitmap in 16 u32 words and a prefix popcount, as
    the device kernel does); tests keep them equal.
    """
    ids = np.asarray(ids)
    _need(ids.ndim == 2 and np.issubdtype(ids.dtype, np.integer), "ids: int [T, K]")
    rows, topk = ids.shape
    _need(1 <= topk <= hx_proto.HX_MAX_TOPK, f"K={topk} outside [1, HX_MAX_TOPK]")
    for name, arr in (("x", x), ("w", w)):
        if arr is not None:
            _need(np.shape(arr)[:1] == (rows,), f"{name} must have {rows} rows")
    _need(w is None or np.shape(w) == ids.shape, "w must have the shape of ids")
    _need(0 <= layer < hx_proto.HX_MAX_LAYERS, f"layer {layer} outside [0, 64)")
    _need(1 <= nsrv <= hx_proto.HX_MAX_SERVERS, f"nsrv={nsrv} outside [1, 8]")
    _need(kmax >= 0, f"kmax={kmax} must not be negative")
    sm = _layer_table(slot_map, 1, layer, "slot_map").astype(np.int64)
    nc = _layer_table(n_cold, 1, layer, "n_cold").astype(np.int64)
    cs = _layer_table(cold_srv, 2, layer, "cold_srv").astype(np.int64)
    ow = None if owned is None else _layer_table(owned, 1, layer, "owned")
    n_exp = sm.shape[0]
    _need(1 <= n_exp <= hx_proto.HX_MAX_EXPERTS, f"E={n_exp} outside [1, 512]")
    _need(nc.shape == (n_exp,), "n_cold must be [E]")
    _need(cs.ndim == 2 and cs.shape[0] == n_exp, "cold_srv must be [E, MAXC]")
    _need(1 <= cs.shape[1] <= hx_proto.HX_MAX_COLD, "MAXC outside [1, HX_MAX_COLD]")
    _need(int(nc.max()) <= cs.shape[1], "n_cold exceeds the width of cold_srv")
    _need(int(sm.min()) >= -1 and int(sm.max()) < hx_proto.HX_MAX_EXPERTS, "bad slot")
    resident = sm[sm >= 0]
    _need(np.unique(resident).size == resident.size, "two experts share a hub slot")
    if ow is not None:
        _need(ow.shape == (n_exp,), "owned must be [E]")
        _need(bool(np.isin(ow, (0, 1)).all()), "owned must hold 0 or 1")
        ow = ow.astype(np.int64)
    if form not in _FORMS:
        raise ValueError(f"form must be 'loop' or 'bitmap', got {form!r}")
    return _FORMS[form](ids, layer, sm, nc, cs, int(kmax), int(nsrv), ow)


def _dispatch_loop(ids, layer, slot_map, n_cold, cold_srv, kmax, nsrv, owned):
    """The rule as the design states it: sets, ranks and one pass over the pairs."""
    rows, topk = ids.shape
    n_exp = slot_map.shape[0]

    def valid(e: int) -> bool:
        return 0 <= e < n_exp and (owned is None or int(owned[e]) == 1)

    touched = set()
    for r in range(rows):
        for k in range(topk):
            e = int(ids[r, k])
            if valid(e) and slot_map[e] >= 0:
                touched.add(e)
    slots = sorted(int(slot_map[e]) for e in touched)
    hub = np.full((rows, topk), -1, np.int32)
    srv = np.full((nsrv, rows, topk), -1, np.int16)
    errors = 0
    for r in range(rows):
        for k in range(topk):
            e = int(ids[r, k])
            if not valid(e):
                continue
            # rank = number of touched ids with a smaller slot
            if e in touched and bisect_left(slots, int(slot_map[e])) < kmax:
                hub[r, k] = e
                continue
            n = int(n_cold[e])
            s = int(cold_srv[e][(r + layer) % n]) if n > 0 else -1
            if 0 <= s < nsrv:
                srv[s, r, k] = e
            else:
                errors += 1
    return hub, srv, errors


def _dispatch_bitmap(ids, layer, slot_map, n_cold, cold_srv, kmax, nsrv, owned):
    """The rule as the device kernel computes it: a touched bitmap over hub slots in
    16 u32 words (LDS ``atomicOr``), a prefix popcount in slot order, then a masked
    gather of the cold copy."""
    rows, topk = ids.shape
    n_exp = slot_map.shape[0]
    ids64 = ids.astype(np.int64)
    valid = (ids64 >= 0) & (ids64 < n_exp)
    eid = np.where(valid, ids64, 0)
    if owned is not None:
        valid &= owned[eid] == 1
    slot = np.where(valid, slot_map[eid], -1)
    resident = slot >= 0
    words = np.zeros(HX_BITMAP_WORDS, np.uint32)
    s = slot[resident]
    np.bitwise_or.at(
        words, s >> 5, np.left_shift(np.uint32(1), (s & 31).astype(np.uint32))
    )
    per_word = _popcount32(words)
    base = np.cumsum(per_word) - per_word  # touched slots in the lower words
    word = np.where(resident, slot >> 5, 0)
    bit = np.where(resident, slot & 31, 0).astype(np.uint32)
    below = words[word] & (np.left_shift(np.uint32(1), bit) - np.uint32(1))
    rank = base[word] + _popcount32(below)
    keep = resident & (rank < kmax)
    remote = valid & ~keep
    ncold = n_cold[eid]
    row = np.arange(rows, dtype=np.int64)[:, None]
    dest = cold_srv[eid, (row + layer) % np.maximum(ncold, 1)]
    bad = remote & ((ncold <= 0) | (dest < 0) | (dest >= nsrv))
    srv = np.full((nsrv, rows, topk), -1, np.int16)
    r_idx, k_idx = np.nonzero(remote & ~bad)
    srv[dest[r_idx, k_idx], r_idx, k_idx] = ids64[r_idx, k_idx]
    return np.where(keep, ids64, -1).astype(np.int32), srv, int(bad.sum())


_FORMS = {"loop": _dispatch_loop, "bitmap": _dispatch_bitmap}


# ---------------------------------------------------------------------------
# Combine reference and a toy MoE
# ---------------------------------------------------------------------------
def combine_ref(
    y_hub: np.ndarray, y_servers: Sequence[np.ndarray], *, aborted: bool = False
) -> np.ndarray:
    """``fp16(((f32(y_hub) + f32(y_0)) + f32(y_1)) + ...)``: the hub partial first, then
    the servers in index order, left to right in fp32, one fp16 rounding. An aborted
    step writes NaN (fp16 0x7E00) everywhere."""
    y_hub = np.asarray(y_hub)
    _need(y_hub.dtype == np.float16, "y_hub must be fp16")
    if aborted:
        return np.full(y_hub.shape, hx_proto.HX_NAN_F16, np.uint16).view(np.float16)
    acc = y_hub.astype(np.float32)
    for y in y_servers:
        y = np.asarray(y)
        _need(
            y.dtype == np.float16 and y.shape == y_hub.shape, "partials: fp16, [T, H]"
        )
        acc = acc + y.astype(np.float32)
    with np.errstate(over="ignore"):
        return acc.astype(np.float16)


def toy_moe(
    x: np.ndarray, ids: np.ndarray, w: np.ndarray, W13: np.ndarray, W2: np.ndarray
) -> np.ndarray:
    """Dense fp32 toy MoE over the pairs of ``ids`` (``[T, K]``, -1 or >= E masked):
    ``out[r] = sum_k w[r, k] * W2[e] @ (silu(gate) * up)`` with ``[gate; up] =
    W13[e] @ x[r]``, ``W13 [E, 2I, H]``, ``W2 [E, H, I]``. The partial of a masked pair
    set is this function applied to the ids of that set."""
    x = np.asarray(x, np.float32)
    w = np.asarray(w, np.float32)
    ids = np.asarray(ids)
    n_exp, two_i, hidden = W13.shape
    inter = two_i // 2
    out = np.zeros((ids.shape[0], hidden), np.float32)
    with np.errstate(over="ignore"):
        for k in range(ids.shape[1]):
            e = ids[:, k]
            ok = (e >= 0) & (e < n_exp)
            g = np.where(ok, e, 0)
            h = np.einsum("toh,th->to", W13[g], x)
            act = h[:, :inter] / (1.0 + np.exp(-h[:, :inter])) * h[:, inter:]
            y = np.einsum("tho,to->th", W2[g], act)
            out += np.where(ok[:, None], w[:, k, None] * y, 0.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Protocol simulator
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Tables:
    """Dispatch tables of one island, stacked over layers like ``StackedTables``."""

    slot_map: np.ndarray  # [L, E] hub slot or -1
    n_cold: np.ndarray  # [L, E] copies on servers
    cold_srv: np.ndarray  # [L, E, MAXC] server index of each copy, -1 padded
    owned: np.ndarray | None = None  # [L, E] uint8; None: the hub owns every expert


class StepDesc(NamedTuple):
    """What the server host posts through the step ring before a step."""

    epoch: int
    T: int
    bucket: int


class _Posted(NamedTuple):
    t: int  # tick at which the store is visible
    n: int  # issue order
    dst: str
    off: int
    data: bytes


def _get_u64(mem: bytearray, off: int) -> int:
    return int.from_bytes(mem[off : off + 8], "little")


def _put_u64(mem: bytearray, off: int, value: int) -> None:
    mem[off : off + 8] = value.to_bytes(8, "little")


def _default_link(nbytes: int) -> int:
    """Visibility latency of a store: 1 tick + 1 per 8 B (a flag beats any payload)."""
    return 1 + nbytes // 8


class Ctx:
    """Device context of one process: status words, abort word, hdr and posted stores.

    ``st`` is the ``hx_status`` vector (``HX_ST_*``); ``abort`` the sticky abort word.
    ``magic``, ``opcode`` and ``version`` are what a hub writes into the header (fault
    injection); a server checks the header against its own ``plan``.
    """

    def __init__(self, name: str, epoch0: int, plan: int) -> None:
        self.name = name
        self.plan = plan
        self.st = [0] * hx_proto.HX_STATUS_WORDS
        self.st[hx_proto.HX_ST_EPOCH] = epoch0
        self.abort = 0
        self.last_epoch = epoch0  # server: epoch of the last adopted step
        self.cur_T = 0
        self.cur_bucket = 0
        self.cur_K = 0  # hub: spill K of the current step
        self.magic = hx_proto.HX_MAGIC
        self.opcode = hx_proto.HX_OPCODE_MOE
        self.version = hx_proto.HX_VERSION
        self.pending: list[_Posted] = []
        self.last_vis = 0

    @property
    def aborted(self) -> bool:
        return self.abort != 0

    @property
    def epoch(self) -> int:
        return self.st[hx_proto.HX_ST_EPOCH]

    @property
    def record(self) -> int:
        return self.st[hx_proto.HX_ST_RECORD]

    @property
    def live(self) -> int:
        return self.st[hx_proto.HX_ST_LIVE]


Op = Generator[int, None, None]  # an operation: yields wake-up times, no result
OpArray = Generator[int, None, np.ndarray]  # ... and one that returns an array


class HxSim:
    """One hub and ``nsrv`` servers of one island on a virtual clock.

    The six protocol operations are generators to be driven with ``yield from`` inside
    a task given to ``run``: ``begin_step`` / ``dispatch`` / ``combine`` (hub) and
    ``srv_begin`` / ``srv_wait`` / ``srv_push`` (server ``s``), plus ``srv_read`` (what
    the MoE chain reads from the inbox) and ``delay`` (compute time). Each follows
    ``hx_core.cuh``:

    * ``begin_step``: epoch++, header fields into every inbox, fence, ``hdr_seq`` last.
    * ``dispatch``: ``dispatch_ref``, x / ids / w into slot ``layer % nslots`` of every
      inbox, fence, request flags ``seq(epoch, layer)``. A plan error claims the abort
      (phase 7) but the layer is still published, so the servers answer it.
    * ``combine``: wait the reply flags in server order (each with its own timeout),
      then ``combine_ref``; NaN once the process is aborted.
    * ``srv_begin``: wait ``hdr_seq >= last_epoch + 1`` (``begin_wait_ms``), check the
      header against the host descriptor, adopt the epoch.
    * ``srv_wait``: wait the request flag (``wait_ms``), widen ids to int32
      (outside ``[0, E)`` -> -1); ``srv_push``: y into the reply slot, fence, flag.
    * A wait ends in the first of: flag reached; host abort word seen (read every 8th
      poll, claims phase 8); ``dt > budget`` (claims the waited phase). The first
      claimer of ``abort`` writes the record; every later operation of that process
      is a no-op (dispatch/push), writes -1 / NaN (srv_wait / combine) or returns.
    * While a wait lasts longer than 1 s the ``live`` word is refreshed every second.

    Knobs: ``fences`` (the fences issued), ``order`` ("adversarial" / "fifo"),
    ``reached`` (flag compare, wrap-safe by default), ``link_ticks`` (store latency),
    ``epoch0`` (the counters start there, flags just below the first sequence number),
    ``seed`` (random choice among ready tasks instead of least recently run). Fault
    injection: the ``Ctx`` fields, ``lose(src, dst, kind, layer) -> bool`` (drop a
    posted store; kinds hdr, hdr_seq, x, ids, w, req_flag, y, reply_flag),
    ``host_poke`` and tasks that stop early.
    """

    def __init__(
        self,
        tables: Tables,
        *,
        nsrv: int = 4,
        rows_cap: int = 8,
        topk: int = 4,
        hidden: int = 16,
        nslots: int = 2,
        kmax: int | Callable[[int], int] = 1 << 30,
        plan_id: str = "00000000000000a1",
        wait_ms: int = hx_proto.HX_DEFAULT_WAIT_MS,
        begin_wait_ms: int = hx_proto.HX_DEFAULT_BEGIN_WAIT_MS,
        ticks_per_ms: int = 1000,
        poll_ticks: int = 1,
        epoch0: int = 0,
        link_ticks: Callable[[int], int] = _default_link,
        order: str = "adversarial",
        fences: Iterable[str] = FENCES_ALL,
        reached: Callable[[int, int], bool] = hx_proto.seq_reached,
        seed: int | None = None,
    ) -> None:
        _need(order in ("adversarial", "fifo"), f"order {order!r}")
        self.fences = frozenset(fences)
        _need(self.fences <= FENCES_ALL, f"unknown fences {sorted(self.fences)}")
        self.tables = tables
        self.num_experts = tables.slot_map.shape[1]
        self.nsrv, self.nslots, self.topk = nsrv, nslots, topk
        self.ibx = hx_proto.inbox_layout(rows_cap, topk, hidden, nslots)
        self.rpl = hx_proto.reply_layout(rows_cap, hidden, nslots, nsrv)
        self.ktab = kmax if callable(kmax) else (lambda t: kmax)
        self.wait_ms, self.begin_wait_ms = wait_ms, begin_wait_ms
        self.tpm, self.poll_ticks = ticks_per_ms, poll_ticks
        self.link_ticks, self.order, self.reached = link_ticks, order, reached
        self.lose: Callable[[str, str, str, int], bool] | None = None
        self.rng = None if seed is None else random.Random(seed)
        self.now = 0
        self.version = 0  # bumps whenever memory changes (wakes blocked tasks)
        self._n = 0
        plan = hx_proto.plan_hash(plan_id)
        self.hub = Ctx("hub", epoch0, plan)
        self.srv = [Ctx(f"srv{s}", epoch0, plan) for s in range(nsrv)]
        self.ctxs = [self.hub, *self.srv]
        self.reply = bytearray(self.rpl.total_bytes)
        self.inbox = [bytearray(self.ibx.total_bytes) for _ in range(nsrv)]
        self._mem = {
            "hub": self.reply,
            **{f"srv{s}": m for s, m in enumerate(self.inbox)},
        }
        flag0 = (epoch0 * hx_proto.HX_MAX_LAYERS) & hx_proto.U64_MAX  # 0 when fresh
        for s in range(nsrv):
            _put_u64(self.inbox[s], hx_proto.HX_HDR_OFF_SEQ, epoch0)
            for j in range(nslots):
                _put_u64(self.inbox[s], self.ibx.req_flag_off(j), flag0)
                _put_u64(self.reply, self.rpl.flag_off(s, j), flag0)

    # -- memory model --------------------------------------------------------
    def _post(
        self, ctx: Ctx, dst: str, off: int, data: bytes, kind: str, layer: int
    ) -> None:
        """A posted store: visible ``link_ticks`` later (in order for ``fifo``)."""
        if self.lose is not None and self.lose(ctx.name, dst, kind, layer):
            return
        t = self.now + self.link_ticks(len(data))
        _need(t > self.now, "a store needs at least one tick to become visible")
        if self.order == "fifo":
            t = max(t, ctx.last_vis)
        ctx.last_vis = t
        ctx.pending.append(_Posted(t, self._n, dst, off, bytes(data)))
        self._n += 1

    def _deliver(self) -> None:
        due = [p for c in self.ctxs for p in c.pending if p.t <= self.now]
        for p in sorted(due, key=lambda p: (p.t, p.n)):
            self._mem[p.dst][p.off : p.off + len(p.data)] = p.data
        if due:
            for c in self.ctxs:
                c.pending = [p for p in c.pending if p.t > self.now]
            self.version += 1

    def _fence(self, ctx: Ctx, name: str) -> Op:
        """``__threadfence_system()``: stall until everything posted is visible."""
        if name in self.fences:
            while ctx.pending:
                yield max(p.t for p in ctx.pending)

    def settle(self) -> None:
        """Advance the clock until every posted store is visible."""
        while any(c.pending for c in self.ctxs):
            self.now = min(p.t for c in self.ctxs for p in c.pending)
            self._deliver()

    def write(self, who: str, off: int, data: bytes) -> None:
        """Make ``data`` visible at ``off`` of the memory of ``who`` ("hub", "srv<s>")
        now: a store not posted by a simulated process (the host, or a test)."""
        mem = self._mem[who]
        _need(off >= 0 and off + len(data) <= len(mem), "write outside the buffer")
        mem[off : off + len(data)] = data
        self.version += 1

    def host_poke(self, who: str, value: int = 1) -> None:
        """The host watcher saw a peer die: write ``value`` (non-zero) at
        ``HX_CTRL_HOST_ABORT`` of the OWN control page of ``who``."""
        _need(0 < value <= hx_proto.U64_MAX, "the poke must be a non-zero u64")
        self.write(who, hx_proto.HX_CTRL_HOST_ABORT, value.to_bytes(8, "little"))

    # -- abort and waits -----------------------------------------------------
    def claim(self, ctx: Ctx, phase: int, peer: int, seq: int, ms: int) -> bool:
        """``atomicCAS(abort_word, 0, 1)``: only the first claimer writes the record."""
        if ctx.abort:
            return False
        ctx.abort = 1
        ctx.st[hx_proto.HX_ST_RECORD] = hx_proto.encode_record(phase, peer, seq, ms)
        return True

    def _wait(
        self,
        ctx: Ctx,
        mem: bytearray,
        off: int,
        seq: int,
        budget_ms: int,
        phase: int,
        peer: int,
    ) -> Generator[int, None, int | None]:
        """``hx_wait_flag``: spin until the u64 at ``mem[off]`` has reached ``seq``.
        Returns the ticks waited, or None when the wait ended in an abort."""
        if self.reached(_get_u64(mem, off), seq):
            return 0
        tpm, budget = self.tpm, budget_ms * self.tpm
        stride = 8 * self.poll_ticks  # the host word is read every 8th poll
        t0, next_live = self.now, 1000 * self.tpm
        while True:
            if self.reached(_get_u64(mem, off), seq):
                return self.now - t0
            dt = self.now - t0
            poked = _get_u64(mem, hx_proto.HX_CTRL_HOST_ABORT) != 0
            if poked and ((dt > 0 and dt % stride == 0) or dt > budget):
                self.claim(ctx, hx_proto.HX_PH_HOST, peer, seq, dt // tpm)
                return None
            if dt > budget:
                self.claim(ctx, phase, peer, seq, dt // tpm)
                return None
            if dt >= next_live:
                rec = hx_proto.encode_record(
                    phase, peer, seq, dt // tpm, hx_proto.HX_REC_LIVE
                )
                ctx.st[hx_proto.HX_ST_LIVE] = rec
                next_live = dt + 1000 * tpm
            wake = min(t0 + budget + 1, t0 + next_live)
            if poked:
                wake = min(wake, t0 + (dt // stride + 1) * stride)
            yield wake

    # -- hub -----------------------------------------------------------------
    def begin_step(self, T: int, bucket: int) -> Op:
        """``hx_begin_step``; a no-op once the hub is aborted."""
        h = self.hub
        _need(1 <= T <= self.ibx.rows_cap, f"T={T} outside [1, rows_cap]")
        if h.aborted:
            return
        epoch = (h.epoch + 1) & hx_proto.U64_MAX
        h.st[hx_proto.HX_ST_EPOCH] = epoch
        h.st[hx_proto.HX_ST_STEPS] += 1
        h.cur_T, h.cur_bucket = T, bucket
        h.cur_K = self.ktab(min(T, self.ibx.rows_cap))
        hdr = hx_proto.pack_hdr(
            epoch,
            T,
            bucket,
            h.plan,
            0,
            magic=h.magic,
            opcode=h.opcode,
            version=h.version,
        )
        cut = hx_proto.HX_HDR_OFF_SEQ
        for s in range(self.nsrv):
            self._post(h, f"srv{s}", 0, hdr[:cut], "hdr", -1)
        yield from self._fence(h, "hdr")
        for s in range(self.nsrv):
            self._post(h, f"srv{s}", cut, hdr[cut:], "hdr_seq", -1)

    def dispatch(
        self, layer: int, x: np.ndarray, ids: np.ndarray, w: np.ndarray
    ) -> OpArray:
        """``hx_dispatch``; returns ``hub_ids`` (all -1 once the hub is aborted)."""
        h = self.hub
        ids = np.asarray(ids)
        _need(ids.ndim == 2 and ids.shape[1] == self.topk, "ids must be [T, topk]")
        _need(ids.shape[0] <= self.ibx.rows_cap, "T exceeds rows_cap")
        if h.aborted:
            return np.full(ids.shape, -1, np.int32)
        _need(
            ids.shape[0] == h.cur_T, f"{ids.shape[0]} rows, begin_step said {h.cur_T}"
        )
        _need(x.dtype == np.float16 and w.dtype == np.float32, "x is fp16, w fp32")
        tb, seq, j = self.tables, hx_proto.seq_of(h.epoch, layer), layer % self.nslots
        hub_ids, srv_ids, errors = dispatch_ref(
            x,
            ids,
            w,
            layer,
            tb.slot_map,
            tb.n_cold,
            tb.cold_srv,
            h.cur_K,
            self.nsrv,
            tb.owned,
        )
        h.st[hx_proto.HX_ST_PAIRS_HUB] += int((hub_ids >= 0).sum())
        h.st[hx_proto.HX_ST_PAIRS_REMOTE] += int((srv_ids >= 0).sum())
        if errors:
            self.claim(h, hx_proto.HX_PH_PLAN, 0, seq, 0)
        base = self.ibx.slot_off(j)
        xb = np.ascontiguousarray(x, "<f2").tobytes()
        wb = np.ascontiguousarray(w, "<f4").tobytes()
        for s in range(self.nsrv):
            ib = np.ascontiguousarray(srv_ids[s], "<i2").tobytes()
            self._post(h, f"srv{s}", base, xb, "x", layer)
            self._post(h, f"srv{s}", base + self.ibx.ids_off, ib, "ids", layer)
            self._post(h, f"srv{s}", base + self.ibx.w_off, wb, "w", layer)
        yield from self._fence(h, "dispatch")
        for s in range(self.nsrv):
            flag = self.ibx.req_flag_off(j)
            self._post(h, f"srv{s}", flag, seq.to_bytes(8, "little"), "req_flag", layer)
        h.st[hx_proto.HX_ST_DISPATCHES] += 1
        return hub_ids

    def combine(self, layer: int, y_hub: np.ndarray) -> OpArray:
        """``hx_combine``: wait every server's reply flag in order, sum in fixed
        order; NaN everywhere when the hub is (or becomes) aborted."""
        h, j = self.hub, layer % self.nslots
        seq = hx_proto.seq_of(h.epoch, layer)
        if not h.aborted:
            for s in range(self.nsrv):
                off = self.rpl.flag_off(s, j)
                waited = yield from self._wait(
                    h, self.reply, off, seq, self.wait_ms, hx_proto.HX_PH_COMBINE, s
                )
                if waited is None:
                    break
        if h.aborted:
            return combine_ref(y_hub, [], aborted=True)
        rows, hidden = h.cur_T, self.rpl.hidden
        _need(y_hub.shape == (rows, hidden), f"y_hub must be [{rows}, {hidden}]")
        ys = [
            np.frombuffer(
                self.reply, "<f2", rows * hidden, self.rpl.y_off(s, j)
            ).reshape(rows, hidden)
            for s in range(self.nsrv)
        ]
        h.st[hx_proto.HX_ST_COMBINES] += 1
        return combine_ref(y_hub, ys)

    # -- server --------------------------------------------------------------
    def srv_begin(self, s: int, desc: StepDesc) -> Op:
        """``hx_srv_begin``: wait the next header, check it, adopt its epoch."""
        c, mem = self.srv[s], self.inbox[s]
        if c.aborted:
            return
        expected = (c.last_epoch + 1) & hx_proto.U64_MAX
        waited = yield from self._wait(
            c,
            mem,
            hx_proto.HX_HDR_OFF_SEQ,
            expected,
            self.begin_wait_ms,
            hx_proto.HX_PH_SRV_BEGIN,
            0,
        )
        if waited is None:
            return
        hdr = hx_proto.unpack_hdr(mem)
        bad = self._hdr_mismatch(c, hdr, expected, desc)
        if bad:
            self.claim(c, hx_proto.HX_PH_HDR, bad, expected, 0)
            return
        c.st[hx_proto.HX_ST_EPOCH] = c.last_epoch = hdr.epoch
        c.st[hx_proto.HX_ST_STEPS] += 1
        c.cur_T, c.cur_bucket = hdr.T, hdr.bucket

    @staticmethod
    def _hdr_mismatch(c: Ctx, hdr: hx_proto.Hdr, expected: int, desc: StepDesc) -> int:
        """First mismatching field (``HX_MM_*``) in the order of the kernel, or 0."""
        if hdr.magic != hx_proto.HX_MAGIC:
            return hx_proto.HX_MM_MAGIC
        if hdr.opcode != hx_proto.HX_OPCODE_MOE:
            return hx_proto.HX_MM_OPCODE
        if hdr.version != hx_proto.HX_VERSION:
            return hx_proto.HX_MM_VERSION
        if hdr.plan_hash != c.plan:
            return hx_proto.HX_MM_PLAN
        if hdr.epoch > expected or hdr.hdr_seq != hdr.epoch:
            return hx_proto.HX_MM_SKIPPED
        if hdr.epoch != desc.epoch:
            return hx_proto.HX_MM_EPOCH
        if hdr.T != desc.T:
            return hx_proto.HX_MM_T
        if hdr.bucket != desc.bucket:
            return hx_proto.HX_MM_BUCKET
        return 0

    def srv_wait(self, s: int, layer: int, T: int) -> OpArray:
        """``hx_srv_wait``: wait the request of ``layer``; returns ``ids32 [T, topk]``
        (the ids widened at this moment, -1 outside ``[0, E)`` or once aborted)."""
        c, mem, j = self.srv[s], self.inbox[s], layer % self.nslots
        ids32 = np.full((T, self.topk), -1, np.int32)
        if c.aborted:
            return ids32
        seq = hx_proto.seq_of(c.epoch, layer)
        waited = yield from self._wait(
            c,
            mem,
            self.ibx.req_flag_off(j),
            seq,
            self.wait_ms,
            hx_proto.HX_PH_SRV_WAIT,
            0,
        )
        if waited is None:
            return ids32
        raw = np.frombuffer(
            mem, "<i2", T * self.topk, self.ibx.slot_off(j) + self.ibx.ids_off
        )
        raw = raw.reshape(T, self.topk).astype(np.int32)
        return np.where((raw >= 0) & (raw < self.num_experts), raw, -1)

    def srv_read(self, s: int, layer: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        """``(x, w)`` as they are in the inbox NOW: the MoE chain reads them in place,
        so a request overwritten while it runs corrupts the result."""
        mem, j, hidden = self.inbox[s], layer % self.nslots, self.ibx.hidden
        base = self.ibx.slot_off(j)
        x = np.frombuffer(mem, "<f2", T * hidden, base).reshape(T, hidden)
        w = np.frombuffer(mem, "<f4", T * self.topk, base + self.ibx.w_off)
        return x.copy(), w.reshape(T, self.topk).copy()

    def srv_push(self, s: int, layer: int, T: int, y: np.ndarray) -> Op:
        """``hx_srv_push``: y into the hub's reply slot, fence, reply flag."""
        c, j = self.srv[s], layer % self.nslots
        if c.aborted:
            return
        _need(
            y.dtype == np.float16 and y.shape == (T, self.rpl.hidden), "y: fp16 [T, H]"
        )
        seq = hx_proto.seq_of(c.epoch, layer)
        yb = np.ascontiguousarray(y, "<f2").tobytes()
        self._post(c, "hub", self.rpl.y_off(s, j), yb, "y", layer)
        yield from self._fence(c, "push")
        flag = self.rpl.flag_off(s, j)
        self._post(c, "hub", flag, seq.to_bytes(8, "little"), "reply_flag", layer)

    # -- scheduler -----------------------------------------------------------
    def delay(self, ticks: int) -> Op:
        """Compute time: the calling task resumes ``ticks`` later."""
        until = self.now + ticks
        while self.now < until:
            yield until

    def run(
        self,
        tasks: Mapping[str, Generator[int | None, None, Any]],
        max_turns: int = 20_000_000,
    ) -> dict[str, Any]:
        """Run the task generators to completion; returns each task's return value.

        A task runs until it yields (a wait, a fence or a delay); it resumes when the
        clock reaches the time it yielded or memory changed. The next task is the least
        recently run ready one (ties: dict order), or a seeded random one.
        """
        live = dict(tasks)
        wake = dict.fromkeys(live, self.now)
        seen = dict.fromkeys(live, -1)
        ran = dict.fromkeys(live, -1)
        result: dict[str, Any] = {}
        for turn in range(max_turns):
            if not live:
                return result
            ready = [n for n in live if wake[n] <= self.now or seen[n] != self.version]
            if not ready:
                events = [wake[n] for n in live]
                events += [p.t for c in self.ctxs for p in c.pending]
                self.now = min(events)
                self._deliver()
                continue
            name = (
                self.rng.choice(ready) if self.rng else min(ready, key=ran.__getitem__)
            )
            ran[name] = turn
            try:
                nxt = next(live[name])
            except StopIteration as stop:
                result[name] = stop.value
                del live[name]
                continue
            seen[name] = self.version
            wake[name] = self.now if nxt is None else nxt
        raise RuntimeError(f"no convergence after {max_turns} scheduler turns")
