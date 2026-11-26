# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor parallelism of the Inkling-2 serving model (TP = 1, 2, 4).

What is sharded and what is replicated (``docs/rdna2/inkling2-port-spec.md``,
"tensor parallelism"):

* **Attention** by KV-head group: rank ``r`` owns KV heads ``[r * nkv / tp,
  (r + 1) * nkv / tp)`` and the q heads of those groups (the GQA group is 3 at every
  TP degree: 24 / 8, 12 / 4, 6 / 2 heads).  ``q`` / output-gate / relative-bias /
  sink / ``k`` / ``v`` / dynamic-conv-tap projections are row slices, ``o_proj`` a
  column slice (its output is an fp32 *partial* that is all-reduced), the per-head
  ``q`` / ``rel`` scales and the k / v short-conv weights follow their heads.  The KV
  cache and the k / v conv state hold only the rank's heads.
* **MoE**: the router is replicated (every rank takes the same decision), the
  intermediate dimension of every expert is split (``1152 -> 576 / 288``) for both the
  NVFP4 and the MXFP8 layers; the partial expert outputs are all-reduced.
* **lm_head and embedding** by vocabulary (``VocabParallelEmbedding`` padding: a
  multiple of 64, then an even split); logits are all-gathered (greedy: the per-block
  argmax partials).  The embedding lookup returns zero rows for the ids of other ranks
  and one all-reduce adds the owner's row back (exact), then ``embed_norm`` runs on the
  complete rows.
* **Replicated on every rank**: the fp32 residual stream, the block states, the
  attention-residual queries, the message gates, every norm, the short convolutions of
  the attention / MLP outputs (and their conv state), the vision tower and the DSpark
  draft.  Replicated compute reads identical all-reduced inputs, so it stays
  bit-identical across ranks (the one-shot ``rdna_ar`` sums in rank order, NCCL hands
  every rank the same reduced bits), which is what keeps the router decisions of all
  ranks equal.

Two all-reduces per layer visit (the attention output and the MoE output) are the
whole communication of the layer stack: the second consumer of each partial (the short
convolution, then the next layer's attention-residual) is nonlinear, so nothing can be
deferred.  Every collective goes through :class:`TensorParallel`, which keeps the
model code free of ``vllm.distributed`` (and replaceable in tests).
"""

from __future__ import annotations

import torch

__all__ = [
    "NO_TP",
    "TensorParallel",
    "VllmTensorParallel",
    "check_supported",
    "current_tensor_parallel",
    "lm_head_shard",
]

#: the MoE expert GEMM kernels need ``K % 32 == 0`` (``w2``'s K is the local
#: intermediate size, the NVFP4 / MXFP8 scale blocks are 16 / 32 wide)
_MOE_GRANULE = 32


class TensorParallel:
    """Rank, size and collectives of the tensor-parallel group of one model.

    This base class is the single-rank group (every collective is the identity);
    :class:`VllmTensorParallel` is the group of the engine, and the tests subclass it
    with an in-process group.
    """

    def __init__(self, size: int = 1, rank: int = 0) -> None:
        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"tensor parallel rank {rank} outside [0, {size})")
        self.size = size
        self.rank = rank

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, rank={self.rank})"

    @property
    def enabled(self) -> bool:
        return self.size > 1

    # -- partitioning -----------------------------------------------------------------
    def local(self, total: int, what: str) -> int:
        """Size of one rank's share of ``total`` units (an even split or an error)."""
        if total % self.size:
            raise ValueError(
                f"{what}: {total} is not divisible by tensor parallel size {self.size}"
            )
        return total // self.size

    def span(self, total: int, what: str) -> tuple[int, int]:
        """``[lo, hi)`` of this rank's contiguous share of ``total`` units."""
        per = self.local(total, what)
        return self.rank * per, (self.rank + 1) * per

    def narrow(self, tensor: torch.Tensor, dim: int, what: str) -> torch.Tensor:
        """This rank's contiguous slice of ``tensor`` along ``dim`` (a view)."""
        lo, hi = self.span(tensor.shape[dim], what)
        return tensor.narrow(dim, lo, hi - lo)

    # -- collectives ------------------------------------------------------------------
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of ``x`` over the group, as a new tensor (``x`` is left alone)."""
        if self.size > 1:
            raise NotImplementedError(f"{type(self).__name__} has no all_reduce")
        return x

    def rdna_ar_handle(self, nbytes: int = 0) -> int:
        """Active Wave-7 BAR1 P2P AllReduce handle if usable for ``nbytes``, else -1."""
        return -1

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Concatenation of the ranks' ``x`` along ``dim`` (rank order)."""
        if self.size > 1:
            raise NotImplementedError(f"{type(self).__name__} has no all_gather")
        return x


NO_TP = TensorParallel()


class VllmTensorParallel(TensorParallel):
    """The engine's tensor-parallel group (``vllm.distributed``).

    ``all_reduce`` is ``tensor_model_parallel_all_reduce``: with ``VLLM_RDNA_AR=1`` the
    HIP one-shot push all-reduce of ``rdna_all_reduce.py`` takes every message up to
    its gate (fp16 / fp32 / bf16), RCCL everything larger; both are legal inside a
    captured HIP graph.
    """

    def __init__(self) -> None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        super().__init__(
            get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return x
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(x)

    def rdna_ar_handle(self, nbytes: int = 0) -> int:
        if self.size <= 1:
            return -1
        from vllm.distributed.parallel_state import get_tp_group

        comm = getattr(get_tp_group(), "device_communicator", None)
        rc = getattr(comm, "rdna_ar_comm", None)
        if rc is None or getattr(rc, "disabled", False):
            return -1
        max_b = int(getattr(rc, "max_bytes", 0))
        if nbytes > 0 and max_b > 0 and nbytes > max_b:
            return -1
        oneshot_b = int(getattr(rc, "oneshot_max_bytes", 0))
        if nbytes > 0 and oneshot_b > 0 and nbytes > oneshot_b:
            return -1
        return int(getattr(rc, "handle", -1))

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.size == 1:
            return x
        from vllm.distributed import tensor_model_parallel_all_gather

        return tensor_model_parallel_all_gather(x, dim)


def current_tensor_parallel() -> TensorParallel:
    """The engine's tensor-parallel group, or the single-rank group when distributed
    state is not initialised (CPU tests, offline tools)."""
    from vllm.distributed import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        return NO_TP
    tp = VllmTensorParallel()
    return tp if tp.enabled else NO_TP


def lm_head_shard(
    vocab_size: int, tp_size: int = 1, tp_rank: int = 0
) -> tuple[int, int, int]:
    """``(first vocabulary id, rows, valid rows)`` of rank ``tp_rank``'s ``lm_head``.

    TP=1 keeps exactly ``vocab_size`` rows (fp32 logits ``[M, vocab_size]``, the layout
    the V2 sampler reads). TP>1 follows ``VocabParallelEmbedding``: the vocabulary is
    padded to a multiple of 64 and split evenly, so every rank owns ``padded // tp``
    rows; the rows of the last rank(s) past ``vocab_size`` are padding (zero weights,
    logits written as ``-inf``).
    """
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank {tp_rank} outside [0, {tp_size})")
    if tp_size == 1:
        return 0, vocab_size, vocab_size
    padded = -(-vocab_size // 64) * 64
    if padded % tp_size:
        raise ValueError(f"padded vocab {padded} not divisible by tp {tp_size}")
    per = padded // tp_size
    lo = tp_rank * per
    return lo, per, max(min(lo + per, vocab_size) - lo, 0)


def check_supported(cfg, tp: TensorParallel) -> None:
    """Refuse a tensor-parallel degree the sharding or the kernels cannot serve
    (an error naming every reason, like ``glue.check_kernel_shapes``)."""
    if not tp.enabled:
        return
    problems: list[str] = []
    for name, nh, nkv in (
        ("swa", cfg.swa_num_attention_heads, cfg.swa_num_key_value_heads),
        ("full", cfg.num_attention_heads, cfg.num_key_value_heads),
    ):
        if nkv % tp.size:
            problems.append(
                f"{name} attention has {nkv} KV heads (sharded by KV head group, so "
                "tp must divide it)"
            )
            continue
        if (nh // tp.size) % 2:
            problems.append(
                f"{name} heads {nh}/{nkv} leave {nh // tp.size} q heads per rank "
                "(the attention kernels need an even count)"
            )
        width, rest = divmod(cfg.hidden_size, nkv // tp.size)
        if rest or width % 4:
            problems.append(
                f"hidden_size {cfg.hidden_size} over {nkv // tp.size} local KV heads "
                "is not a multiple of 4 per head (the conv state packs the "
                "attention / MLP streams head-major)"
            )
    inter = cfg.intermediate_size
    if inter % (tp.size * _MOE_GRANULE):
        problems.append(
            f"MoE intermediate size {inter} / {tp.size} is not a multiple of "
            f"{_MOE_GRANULE} (the expert GEMMs need K % 32 == 0 in the down "
            "projection)"
        )
    if problems:
        raise ValueError(
            f"Inkling-2 cannot run with tensor_parallel_size={tp.size}: "
            + "; ".join(problems)
        )
