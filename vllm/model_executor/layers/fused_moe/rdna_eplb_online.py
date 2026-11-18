# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online expert rebalancing for the RDNA2 W4A16 / W4A8 MoE (opt-in).

``VLLM_RDNA_EPLB=1`` moves experts between expert-parallel ranks while the server
runs, so that the experts a workload activates spread evenly over the ranks. A step
is as slow as its slowest rank; a rank whose experts are not activated waits.

Design rule: serving never stalls. Everything below follows from it.

* **Lockstep.** ``step()`` runs once per ``Worker.execute_model`` on every rank, so
  the step counter is the same number on every rank. Every collective is scheduled by
  that counter and by the plan all ranks received, never by rank-local state: a rank
  that hits a local problem keeps the schedule, contributes zeros and raises the
  error word, and all ranks switch the balancer off at the same step.
* **Rank 0 plans, in another process.** The policy (``rdna_eplb_policy``) runs in a
  helper process (``rdna_eplb_planner``) fed through shared memory: one layer's
  counters per step, copied without blocking. No thread: a thread holding the GIL
  would add up to the interpreter's switch interval to a step.
* **Mailbox.** Every ``mailbox_steps`` steps the ranks all-reduce one small int32
  message. Rank 0 writes a plan or nothing, the others write zeros, so every rank
  holds the same plan at the same step and no rank has to compute it.
* **Transfer.** One swap ``(layer, a, b)`` at a time: the two owners pack their
  expert into a send buffer, then each step all-reduces one chunk (``chunk_kb`` per
  direction; the other ranks add zeros, an integer sum, hence exact).
* **Commit.** A vote all-reduce carries both owners' checksums of what they sent and
  what they received. If it passes, the owners overwrite the slot rows and
  ``expert_map`` in place at that step boundary (captured graphs keep the same
  pointers) and every rank updates the owner table of the load counters. If it does
  not, nothing changes and the balancer goes off.

The balancer needs the load counters (``rdna_eplb_stats``, switched on by the same
variable), expert parallelism over the TP group, one pipeline stage, and the linear
placement strategy. The MTP draft model's BF16 experts and the expert-offload cache
are not moved. A swap reorders the ranks' partial sums, so outputs match in value
but not bit for bit (fp16 all-reduce association).

Options (all optional):

* ``VLLM_RDNA_EPLB_MAILBOX_STEPS`` steps between mailbox collectives (256);
* ``VLLM_RDNA_EPLB_CHUNK_KB`` KiB moved per direction and per step (64);
* ``VLLM_RDNA_EPLB_CHUNK_EVERY`` steps between two chunks (1);
* ``VLLM_RDNA_EPLB_DECISION_STEPS`` steps between two decisions (4096);
* ``VLLM_RDNA_EPLB_MAX_SWAPS`` swaps of one plan (32);
* ``VLLM_RDNA_EPLB_OBJECTIVE`` ``touch`` (decode, default) or ``pairs`` (prefill);
* ``VLLM_RDNA_EPLB_CHAOS`` N > 0: no policy, rank 0 posts N random swaps per
  decision (a qualification mode for the transfer and the commit).
"""

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import rdna_eplb_stats as es
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

logger = init_logger(__name__)

MAILBOX_ENV = "VLLM_RDNA_EPLB_MAILBOX_STEPS"
CHUNK_ENV = "VLLM_RDNA_EPLB_CHUNK_KB"
CHUNK_EVERY_ENV = "VLLM_RDNA_EPLB_CHUNK_EVERY"
DECISION_ENV = "VLLM_RDNA_EPLB_DECISION_STEPS"
MAX_SWAPS_ENV = "VLLM_RDNA_EPLB_MAX_SWAPS"
OBJECTIVE_ENV = "VLLM_RDNA_EPLB_OBJECTIVE"
CHAOS_ENV = "VLLM_RDNA_EPLB_CHAOS"
CHAOS_SEED_ENV = "VLLM_RDNA_EPLB_CHAOS_SEED"

# Per-expert tensors of the RDNA2 MoE layouts (leading dimension = local experts).
EXPERT_TENSORS = (
    "w13_weight_packed",
    "w13_weight_scale",
    "w13_qzeros",
    "w2_weight_packed",
    "w2_weight_scale",
    "w2_qzeros",
)

# Mailbox message (int32): header, then SWAP_WORDS per swap.
HEADER = 4
M_SEQ, M_SWAPS, M_ERRORS = 0, 1, 2
# layer, expert a, expert b, rank of a, rank of b, int32 words of one expert
SWAP_WORDS = 6
# Vote message (int32).
VOTE_WORDS = 8
V_READY, V_SENT_A, V_SENT_B, V_GOT_A, V_GOT_B, V_ERRORS = 0, 1, 2, 3, 4, 5
CHECK_MOD = 8191
CHECK_MASK = 0x3FFFFFFF
RING_SLICE = 4096  # ring records copied to the planner per step


class Collective(Protocol):
    """What the balancer needs from the TP group."""

    world_size: int
    rank: int
    device: torch.device

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...


class _TpCollective:
    """The worker's tensor-parallel group."""

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        group = get_tp_group()
        self._group = group
        self.world_size = int(group.world_size)
        self.rank = int(group.rank_in_group)
        self.device = group.device
        self.stages = int(get_pp_group().world_size)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._group._all_reduce_out_place(tensor)


@dataclass(frozen=True)
class OnlineConfig:
    mailbox_steps: int = 256
    chunk_kb: int = 64
    chunk_every: int = 1
    decision_steps: int = 4096
    max_swaps: int = 32
    objective: str = "touch"
    chaos: int = 0
    chaos_seed: int = 0

    def __post_init__(self) -> None:
        if self.mailbox_steps < 1 or self.chunk_every < 1 or self.decision_steps < 1:
            raise ValueError("online EPLB step counts must be >= 1")
        if self.chunk_every % self.mailbox_steps == 0:
            raise ValueError(
                f"{CHUNK_EVERY_ENV} must leave transfer steps outside "
                f"{MAILBOX_ENV}; every transfer step would be a mailbox step"
            )
        if not 1 <= self.chunk_kb <= 4096:
            raise ValueError(f"{CHUNK_ENV} must be 1..4096, got {self.chunk_kb}")
        if not 1 <= self.max_swaps <= 1024:
            raise ValueError(f"{MAX_SWAPS_ENV} must be 1..1024, got {self.max_swaps}")
        if self.objective not in ("touch", "pairs"):
            raise ValueError(f"{OBJECTIVE_ENV} must be touch or pairs")
        if self.chaos < 0:
            raise ValueError(f"{CHAOS_ENV} must be >= 0")

    @classmethod
    def from_env(cls) -> "OnlineConfig":
        env = os.environ
        return cls(
            mailbox_steps=int(env.get(MAILBOX_ENV, 256)),
            chunk_kb=int(env.get(CHUNK_ENV, 64)),
            chunk_every=int(env.get(CHUNK_EVERY_ENV, 1)),
            decision_steps=int(env.get(DECISION_ENV, 4096)),
            max_swaps=int(env.get(MAX_SWAPS_ENV, 32)),
            objective=env.get(OBJECTIVE_ENV, "touch"),
            chaos=int(env.get(CHAOS_ENV, 0)),
            chaos_seed=int(env.get(CHAOS_SEED_ENV, 0)),
        )

    @property
    def chunk_words(self) -> int:
        return self.chunk_kb * 256


@dataclass
class Swap:
    layer: int
    a: int
    b: int
    rank_a: int
    rank_b: int
    words: int
    done: int = 0  # int32 words transferred so far


class _Layer:
    """One MoE layer on this rank: the tensors a swap rewrites."""

    def __init__(self, stats: es.LayerStats, local_rank: int, ranks: int) -> None:
        module = stats.module() if stats.module is not None else None
        if module is None:
            raise RuntimeError(f"MoE layer {stats.name!r} is gone")
        if getattr(module, "_rdna_expert_cache", None) is not None:
            raise RuntimeError("the expert-offload cache does not support EPLB")
        if stats.ranks != ranks or stats.rank != local_rank:
            raise RuntimeError(
                f"{stats.name!r}: expert-parallel rank {stats.rank}/{stats.ranks} is "
                f"not the TP rank {local_rank}/{ranks}"
            )
        strategy = getattr(module, "expert_placement_strategy", "linear") or "linear"
        if strategy != "linear":
            raise RuntimeError(f"placement strategy {strategy!r} is not supported")
        expert_map = module.expert_map
        if expert_map is None or expert_map.dtype != torch.int32:
            raise RuntimeError(f"{stats.name!r} has no int32 expert_map")
        # The resident layout keeps its kernel tensors on a shim.
        holder = getattr(module, "_rdna2_resident", module)
        local = int((expert_map >= 0).sum())
        views: list[torch.Tensor] = []
        for name in EXPERT_TENSORS:
            tensor = getattr(holder, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
                continue
            if tensor.shape[0] != local:
                raise RuntimeError(
                    f"{stats.name!r}.{name} has {tensor.shape[0]} rows, the rank owns"
                    f" {local} experts"
                )
            data = tensor.data
            row_bytes = data[0].numel() * data.element_size()
            if not data.is_contiguous() or row_bytes % 4:
                raise RuntimeError(f"{stats.name!r}.{name} is not int32-addressable")
            flat = data.view(local, -1)
            if flat.element_size() != 4:
                flat = flat.view(torch.int32)
            views.append(flat)
        if len(views) < 4:
            raise RuntimeError(f"{stats.name!r} has no RDNA2 W4A16 expert tensors")
        self.stats = stats
        self.views = views
        self.widths = [int(v.shape[1]) for v in views]
        self.words = sum(self.widths)
        self.expert_map = expert_map
        # Host copies: the local slot of every global expert, and its rank. The
        # owner array is the counters' own host table, updated in place.
        self.slot = expert_map.detach().cpu().numpy().astype(np.int32)
        self.owner = stats.owner_host.numpy()
        mine = self.owner == local_rank
        if not np.array_equal(mine, self.slot >= 0):
            raise RuntimeError(f"{stats.name!r}: owner table and expert_map disagree")


def _copy_async(host: torch.Tensor, device: torch.Tensor) -> Any:
    """Queue a device-to-host copy; the event that reports it (None on CPU)."""
    host.copy_(device, non_blocking=True)
    if device.device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record()
    return event


class OnlineEplb:
    """Per-worker balancer; see the module docstring."""

    def __init__(
        self,
        cfg: OnlineConfig,
        *,
        collective: Collective | None = None,
        stats: es.EplbStats | None = None,
        planner: Any = None,
    ) -> None:
        self.cfg = cfg
        self._collective = collective
        self._stats = stats
        self._planner = planner
        self.t = 0
        # No more collectives. Only ever set from data every rank holds (config,
        # a mailbox or a vote result), so all ranks set it at the same step.
        self.off = False
        self._bound = False
        self._error: str | None = None
        self.rank = 0
        self.world = 1
        self.layers: list[_Layer] = []
        self._queue: list[Swap] = []
        self._xfer: Swap | None = None
        self._ok = False  # this rank takes part in the current swap
        self._slot = -1  # its local slot in that swap
        self._outbox: np.ndarray | None = None
        self._plan_seq = 0
        self._plan_t0 = 0
        self._plan_wall = 0.0
        self._plan_swaps = 0
        self._snap: Iterator[None] | None = None
        self._waiting = False
        self._next_decision = cfg.decision_steps
        self.plans = 0
        self.committed = 0
        self.collectives = 0

    @classmethod
    def from_env(cls) -> "OnlineEplb | None":
        if os.environ.get(es.ONLINE_ENV, "0") != "1":
            return None
        cfg = OnlineConfig.from_env()
        logger.warning("Online EPLB is ON (%s=1): %s", es.ONLINE_ENV, cfg)
        return cls(cfg)

    # ---- the step boundary ---------------------------------------------------

    def step(self) -> None:
        """One step boundary; the same call sequence on every rank."""
        if self.off:
            return
        if not self._bound:
            self._bind()
            if self.off:
                return
        self.t += 1
        if self.t % self.cfg.mailbox_steps == 0:
            self._mailbox()
        elif self._xfer is not None:
            if self.t % self.cfg.chunk_every == 0:
                self._advance()
        elif self._queue:
            self._begin(self._queue.pop(0))
        if self.rank == 0 and not self.off:
            self._feed()

    def close(self) -> None:
        if self._planner is not None:
            self._planner.close()
            self._planner = None

    # ---- setup (first step) --------------------------------------------------

    def _bind(self) -> None:
        self._bound = True
        col = self._collective
        if col is None:
            col = self._collective = _TpCollective()
            if col.stages > 1:
                logger.warning("Online EPLB: off with pipeline parallelism")
                self.off = True
                return
        self.rank, self.world = col.rank, col.world_size
        if self.world < 2:
            logger.warning("Online EPLB: off, one rank has nothing to balance")
            self.off = True
            return
        device = col.device
        cfg = self.cfg
        max_words = 1
        try:
            stats = self._stats if self._stats is not None else es.COLLECTOR
            if stats is None or not stats.layers:
                raise RuntimeError("no RDNA2 MoE layer registered its load counters")
            self._stats = stats
            with gpu_sync_allowed():
                self.layers = [_Layer(s, self.rank, self.world) for s in stats.layers]
            max_words = max(layer.words for layer in self.layers)
        except Exception as exc:  # reported through the mailbox, never raised
            self._fail(f"setup: {exc}")
            self.layers = []
        self._mail_host = torch.zeros(
            HEADER + SWAP_WORDS * cfg.max_swaps, dtype=torch.int32
        )
        self._mail = torch.zeros_like(self._mail_host, device=device)
        self._vote = torch.zeros(VOTE_WORDS, dtype=torch.int32, device=device)
        self._chunk = torch.zeros(2 * cfg.chunk_words, dtype=torch.int32, device=device)
        self._send = torch.zeros(max_words, dtype=torch.int32, device=device)
        self._recv = torch.zeros(max_words, dtype=torch.int32, device=device)
        self._check = (
            torch.arange(max_words, dtype=torch.int64, device=device) % CHECK_MOD + 1
        )
        self._rng = np.random.default_rng(cfg.chaos_seed)
        if self.rank == 0 and self._error is None and not cfg.chaos:
            self._start_planner()
        if self.rank == 0:
            logger.info(
                "Online EPLB: %d layers, %d ranks, %.2f MiB per expert, %d chunks "
                "per swap",
                len(self.layers),
                self.world,
                max_words * 4 / 2**20,
                -(-max_words // cfg.chunk_words),
            )

    def _start_planner(self) -> None:
        if self._planner is not None:
            return
        try:
            from vllm.model_executor.layers.fused_moe.rdna_eplb_planner import (
                PlannerClient,
                PlannerSpec,
            )

            stats = self._stats
            assert stats is not None
            first = self.layers[0].stats
            spec = PlannerSpec(
                layers=len(self.layers),
                experts=first.num_experts,
                ranks=self.world,
                nb=stats.nb,
                edges=tuple(stats.edges_host),
                state_words=first.off.size,
                off_touch=first.off.touch,
                off_calls=first.off.calls,
                off_rows=first.off.rows,
                off_ordinal=first.off.ordinal,
                ring_records=stats.ring_n,
                ring_words=es.WORDS,
                max_swaps=self.cfg.max_swaps,
                objective=self.cfg.objective,
                expert_bytes=4 * max(layer.words for layer in self.layers),
            )
            # The planner compares two windows of ring records. A ring that
            # cannot hold the older one never yields a plan, so say so once.
            window = self.cfg.decision_steps + len(self.layers)
            window += -(-stats.ring_n // RING_SLICE) + 2
            held = stats.ring_n // max(1, len(self.layers))
            if held - window < spec.min_samples:
                logger.warning(
                    "Online EPLB: the counter ring holds %d steps but a decision "
                    "window is about %d; raise %s or lower "
                    "VLLM_RDNA_EPLB_DECISION_STEPS, else the placement is kept",
                    held,
                    window,
                    es.RING_ENV,
                )
            planner = PlannerClient(spec)
            for i, layer in enumerate(self.layers):
                planner.owner[i] = layer.owner
            planner.start()
            self._planner = planner
        except Exception as exc:
            # No planner means no plan: the placement in force stays.
            logger.error("Online EPLB: planner unavailable (%s); placement kept", exc)
            self._planner = None

    # ---- local failures ------------------------------------------------------

    def _fail(self, reason: str) -> None:
        """A rank-local problem: keep the schedule, report it at the next vote."""
        if self._error is None:
            self._error = reason
            logger.error("Online EPLB rank %d: %s", self.rank, reason)
        self._ok = False

    def _disable(self, reason: str) -> None:
        """Switch off; called with data every rank holds, so all do it together."""
        self.off = True
        self._queue.clear()
        self._xfer = None
        self._outbox = None
        logger.error(
            "Online EPLB rank %d: OFF at step %d (%s); the placement in force stays",
            self.rank,
            self.t,
            reason,
        )
        self.close()

    def _idle(self) -> bool:
        return self._xfer is None and not self._queue and self._outbox is None

    # ---- mailbox -------------------------------------------------------------

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._collective is not None
        self.collectives += 1
        return self._collective.all_reduce(tensor)

    def _mailbox(self) -> None:
        message = self._mail_host.numpy()
        message[:] = 0
        if self.rank == 0 and self._xfer is None and not self._queue:
            chaos = self.cfg.chaos and self._outbox is None and self._error is None
            if chaos and self.t >= self._next_decision:
                self._next_decision = self.t + self.cfg.decision_steps
                self._outbox = self._chaos_plan()
            if self._outbox is not None:
                message[: self._outbox.shape[0]] = self._outbox
                self._outbox = None
        if self._error is not None:
            message[M_ERRORS] = 1
        # The scheduled mailbox handshake deliberately synchronizes the host.
        with gpu_sync_allowed():
            self._mail.copy_(self._mail_host)
            out = self._reduce(self._mail)
            result = out.cpu().numpy()
        if result[M_ERRORS] != 0:
            self._disable(f"{int(result[M_ERRORS])} rank(s) reported an error")
            return
        if result[M_SEQ] != 0:
            self._accept(result)

    def _accept(self, message: np.ndarray) -> None:
        count = int(message[M_SWAPS])
        if not 1 <= count <= self.cfg.max_swaps:
            self._disable(f"malformed plan ({count} swaps)")
            return
        body = message[HEADER : HEADER + SWAP_WORDS * count].reshape(count, SWAP_WORDS)
        swaps = [Swap(*(int(x) for x in row)) for row in body]
        for s in swaps:
            ranks_ok = 0 <= s.rank_a < self.world and 0 <= s.rank_b < self.world
            if not ranks_ok or s.rank_a == s.rank_b or s.words < 1 or s.layer < 0:
                self._disable("malformed swap in the plan")
                return
        self._queue = swaps
        self._plan_seq = int(message[M_SEQ])
        self._plan_t0 = self.t
        self._plan_wall = time.monotonic()
        self._plan_swaps = count
        self.plans += 1
        if self.rank == 0:
            logger.info(
                "Online EPLB: plan %d at step %d, %d swap(s) monotonic=%.9f",
                self._plan_seq,
                self.t,
                count,
                self._plan_wall,
            )

    def _encode(self, swaps: list[tuple[int, int, int]]) -> np.ndarray | None:
        """Rank 0: the mailbox message of a plan (None if it is not valid here)."""
        swaps = swaps[: self.cfg.max_swaps]
        out = np.zeros(HEADER + SWAP_WORDS * len(swaps), dtype=np.int32)
        seen: set[tuple[int, int]] = set()
        for i, (pos, a, b) in enumerate(swaps):
            if not 0 <= pos < len(self.layers):
                return None
            layer = self.layers[pos]
            experts = layer.owner.shape[0]
            if not (0 <= a < experts and 0 <= b < experts):
                return None
            if layer.owner[a] == layer.owner[b] or {(pos, a), (pos, b)} & seen:
                return None
            seen.update(((pos, a), (pos, b)))
            out[HEADER + SWAP_WORDS * i : HEADER + SWAP_WORDS * (i + 1)] = (
                pos,
                a,
                b,
                int(layer.owner[a]),
                int(layer.owner[b]),
                layer.words,
            )
        out[M_SEQ] = self._plan_seq + 1
        out[M_SWAPS] = len(swaps)
        return out if swaps else None

    def _chaos_plan(self) -> np.ndarray | None:
        rng = self._rng
        swaps: list[tuple[int, int, int]] = []
        used: set[tuple[int, int]] = set()
        for _ in range(min(self.cfg.chaos, self.cfg.max_swaps)):
            pos = int(rng.integers(len(self.layers)))
            owner = self.layers[pos].owner
            rank_a, rank_b = (int(r) for r in rng.choice(self.world, 2, replace=False))
            a = int(rng.choice(np.flatnonzero(owner == rank_a)))
            b = int(rng.choice(np.flatnonzero(owner == rank_b)))
            if (pos, a) in used or (pos, b) in used:
                continue
            used.update(((pos, a), (pos, b)))
            swaps.append((pos, a, b))
        return self._encode(swaps)

    # ---- transfer ------------------------------------------------------------

    def _begin(self, swap: Swap) -> None:
        """Start a swap: the two owners pack their expert into the send buffer."""
        self._xfer = swap
        self._ok = False
        self._slot = -1
        if not 0 <= swap.layer < len(self.layers):
            self._fail(f"plan names layer {swap.layer}, {len(self.layers)} known")
            return
        layer = self.layers[swap.layer]
        experts = layer.owner.shape[0]
        if (
            not (0 <= swap.a < experts and 0 <= swap.b < experts)
            or layer.owner[swap.a] != swap.rank_a
            or layer.owner[swap.b] != swap.rank_b
            or layer.words != swap.words
        ):
            self._fail(f"plan disagrees with the placement held here: {swap}")
            return
        if self.rank not in (swap.rank_a, swap.rank_b):
            return
        mine = swap.a if self.rank == swap.rank_a else swap.b
        slot = int(layer.slot[mine])
        if slot < 0:
            self._fail(f"expert {mine} of layer {swap.layer} is not local")
            return
        at = 0
        for view, width in zip(layer.views, layer.widths):
            self._send[at : at + width].copy_(view[slot])
            at += width
        self._slot = slot
        self._ok = True

    def _advance(self) -> None:
        swap = self._xfer
        assert swap is not None
        if swap.done >= swap.words:
            self._vote_and_commit(swap)
            return
        n = min(self.cfg.chunk_words, swap.words - swap.done)
        lo, hi = swap.done, swap.done + n
        chunk = self._chunk[: 2 * n]
        chunk.zero_()
        # The owner of ``a`` writes the first half, the owner of ``b`` the second.
        first = self.rank == swap.rank_a
        if self._ok:
            (chunk[:n] if first else chunk[n:]).copy_(self._send[lo:hi])
        out = self._reduce(chunk)
        if self._ok:
            self._recv[lo:hi].copy_(out[n:] if first else out[:n])
        swap.done = hi

    def _checksum(self, buffer: torch.Tensor, words: int) -> torch.Tensor:
        total = (buffer[:words].to(torch.int64) * self._check[:words]).sum()
        return (total & CHECK_MASK).to(torch.int32).view(1)

    def _vote_and_commit(self, swap: Swap) -> None:
        vote = self._vote
        vote.zero_()
        if self._ok:
            first = self.rank == swap.rank_a
            vote[V_READY : V_READY + 1].fill_(1)
            sent = V_SENT_A if first else V_SENT_B
            got = V_GOT_B if first else V_GOT_A
            vote[sent : sent + 1].copy_(self._checksum(self._send, swap.words))
            vote[got : got + 1].copy_(self._checksum(self._recv, swap.words))
        if self._error is not None:
            vote[V_ERRORS : V_ERRORS + 1].fill_(1)
        out = self._reduce(vote)
        with gpu_sync_allowed():
            result = out.cpu().numpy()
            passed = (
                result[V_READY] == 2
                and result[V_SENT_A] == result[V_GOT_A]
                and result[V_SENT_B] == result[V_GOT_B]
                and result[V_ERRORS] == 0
            )
            if not passed:
                self._disable(f"swap vote failed: {result.tolist()}")
                return
            self._commit(swap)
        self._xfer = None
        self.committed += 1
        if not self._queue and self.rank == 0:
            end = time.monotonic()
            logger.info(
                "Online EPLB: plan %d done, %d swap(s) in %d steps (%.1f s) "
                "monotonic=%.9f",
                self._plan_seq,
                self._plan_swaps,
                self.t - self._plan_t0,
                end - self._plan_wall,
                end,
            )

    def _commit(self, swap: Swap) -> None:
        """The vote passed on every rank: rewrite the slots and the maps in place."""
        layer = self.layers[swap.layer]
        device = self._vote.device
        pair = torch.tensor([swap.a, swap.b], dtype=torch.int64, device=device)
        if self.rank in (swap.rank_a, swap.rank_b):
            first = self.rank == swap.rank_a
            leaving, arriving = (swap.a, swap.b) if first else (swap.b, swap.a)
            slot = self._slot
            at = 0
            for view, width in zip(layer.views, layer.widths):
                view[slot].copy_(self._recv[at : at + width])
                at += width
            slots = (-1, slot) if first else (slot, -1)
            layer.expert_map.index_copy_(
                0, pair, torch.tensor(slots, dtype=torch.int32, device=device)
            )
            layer.slot[leaving] = -1
            layer.slot[arriving] = slot
        # Every rank: expert ``a`` now lives on the rank of ``b`` and vice versa.
        layer.owner[swap.a] = swap.rank_b
        layer.owner[swap.b] = swap.rank_a
        layer.stats.owner.index_copy_(
            0,
            pair,
            torch.tensor([swap.rank_b, swap.rank_a], dtype=torch.uint8, device=device),
        )
        if self._planner is not None:
            self._planner.owner[swap.layer, swap.a] = swap.rank_b
            self._planner.owner[swap.layer, swap.b] = swap.rank_a

    # ---- rank 0: feed the planner, one small slice per step ------------------

    def _feed(self) -> None:
        planner = self._planner
        if planner is None:
            return
        if self._snap is not None:
            try:
                next(self._snap)
            except StopIteration:
                self._snap = None
                self._waiting = True
            return
        if self._waiting:
            answer = planner.poll()
            if answer is None:
                return
            self._waiting = False
            self._next_decision = self.t + self.cfg.decision_steps
            if answer:
                self._outbox = self._encode(answer)
                if self._outbox is None:
                    logger.warning("Online EPLB: planner answer rejected: %s", answer)
            return
        if self.t >= self._next_decision and self._idle() and self._error is None:
            self._snap = self._snapshot()

    def _snapshot(self) -> Iterator[None]:
        """Copy the counters into the planner's shared memory, a slice per step."""
        stats, planner = self._stats, self._planner
        assert stats is not None and planner is not None
        for i, layer in enumerate(self.layers):
            counters = layer.stats
            event = _copy_async(counters.host_state, counters.state)
            yield
            while event is not None and not event.query():
                yield
            np.copyto(planner.state[i], counters.host_state.numpy())
        gseq = 0
        if stats.ring is not None and stats.gseq is not None:
            assert stats.host_ring is not None and stats.host_gseq is not None
            records = stats.ring.shape[0]
            for lo in range(0, records, RING_SLICE):
                hi = min(records, lo + RING_SLICE)
                event = _copy_async(stats.host_ring[lo:hi], stats.ring[lo:hi])
                yield
                while event is not None and not event.query():
                    yield
                np.copyto(planner.ring[lo:hi], stats.host_ring[lo:hi].numpy())
            event = _copy_async(stats.host_gseq, stats.gseq)
            yield
            while event is not None and not event.query():
                yield
            gseq = int(stats.host_gseq.numpy()[0])
        planner.submit(self.t, gseq)


# One balancer per process; None (the default) is the whole cost when off.
ONLINE: OnlineEplb | None = OnlineEplb.from_env()
ENABLED = ONLINE is not None
