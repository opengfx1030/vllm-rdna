# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch and host-sync accounting for one RidgeFill prefill forward.

Analytical, not measured. Every term names the operation in ``runtime.py`` or
``model.py`` that issues it, so the before/after comparison is reproducible in
a CPU test and reviewable without a GPU. The compiled A/B records real
counters; this module explains the gap and guards against a regression that
re-adds a per-layer host read.

Terms counted as a *sync* are host reads that drain the device: ``.cpu()``,
``.tolist()`` on a device tensor, ``.item()``, and boolean indexing whose
output size is data dependent. Launches are device kernels the row plan adds;
attention, GDN and MoE kernels are unchanged and not counted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardCost:
    """What the row plan costs one prefill forward."""

    syncs: int
    host_transfers: int
    launches: int

    def __add__(self, other: ForwardCost) -> ForwardCost:
        return ForwardCost(
            self.syncs + other.syncs,
            self.host_transfers + other.host_transfers,
            self.launches + other.launches,
        )


def cost_before(
    *, layers: int, filled_rows: int, seqs: int = 1, prefills_from_steps: bool = True
) -> ForwardCost:
    """The per-layer row-plan cost the draft measured before this rework.

    ``observe_enter`` reads ``seq_lens.cpu()`` and ``query_start_loc.cpu()``
    once and, when the metadata has no ``num_prefills``, one ``.item()``.
    ``capture_split`` reads the overflow flag. Then each of ``layers`` late
    layers rebuilds the mask: one ``positions.cpu()`` sync, one ``torch.zeros``
    launch and one fill launch per filled row. The MoE branch reads
    ``bool(exact.any())`` and boolean-indexes ``block_input[exact]``, whose
    size is data dependent, so each layer syncs twice more.
    """
    syncs = 2 + (1 if prefills_from_steps else 0) + 1
    syncs += layers * 3  # positions.cpu(), any(), masked_select size
    launches = layers * (1 + filled_rows + 2 + 2 + 1)
    transfers = 2 + layers
    del seqs
    return ForwardCost(syncs=syncs, host_transfers=transfers, launches=launches)


def cost_after(
    *, layers: int, filled_rows: int, seqs: int = 1, prefills_from_steps: bool = True
) -> ForwardCost:
    """This rework's per-forward row-plan cost.

    ``begin_forward`` runs once, outside the compiled model: the two metadata
    reads, the prefill count, the positions read and the overflow read are the
    only syncs, and the split's arithmetic runs on host copies. It uploads the
    two index tensors; ``capture_split`` gathers and casts the split state's
    filled rows once. Every later layer reuses them: one ``index_copy_`` for
    the block input, one map GEMM on F rows, and in the MoE branch one
    ``index_select``, one ``index_copy_`` and one ``zeros_like``. No layer
    reads the host.
    """
    syncs = 2 + (1 if prefills_from_steps else 0) + 1 + 1
    transfers = 2 + 1 + 1 + 2  # metadata, prefill count, positions, two uploads
    launches = 2 + 2  # two index uploads; split-state gather and cast
    launches += layers * 5  # replace, GEMM, MoE select, MoE scatter, zeros
    del filled_rows, seqs
    return ForwardCost(syncs=syncs, host_transfers=transfers, launches=launches)


def cost_decode() -> ForwardCost:
    """A decode forward: ``num_prefills`` is a plain int, so nothing syncs.

    The row plan is empty and the model takes the untouched compiled path.
    """
    return ForwardCost(syncs=0, host_transfers=0, launches=0)


def cost_cache_only(
    *, layers: int, filled_rows: int, qsa_layers: int | None = None
) -> ForwardCost:
    """What the cache-write-only path adds, per forward.

    Adds, for each late QSA layer: the K/V projection and its norm+RoPE on F
    rows, the main cache store, the indexer K projection and the raw and
    compressed stores. Removes the q projection, the sparse selection and the
    attention kernel over F rows, plus ``o_proj`` over F rows. GDN layers keep
    their full-batch state update, so nothing is removed there. The work saved
    is the attention scoring and ``o_proj`` over F rows; this module counts
    launches only. ``qsa_layers`` defaults to ``layers`` so the number is an
    upper bound, not a guess at the layer split.
    """
    count = layers if qsa_layers is None else qsa_layers
    del filled_rows
    return ForwardCost(syncs=0, host_transfers=0, launches=count * 6)


def table(
    *,
    layers: int,
    filled_rows: int,
    seqs: int = 1,
    cache_only: bool = False,
    qsa_layers: int | None = None,
) -> dict[str, ForwardCost]:
    """The three columns TESTPLAN.md quotes."""
    before = cost_before(layers=layers, filled_rows=filled_rows, seqs=seqs)
    after = cost_after(layers=layers, filled_rows=filled_rows, seqs=seqs)
    extra = (
        cost_cache_only(layers=layers, filled_rows=filled_rows, qsa_layers=qsa_layers)
        if cache_only
        else ForwardCost(0, 0, 0)
    )
    return {"before": before, "after": after, "cache_only": extra}


__all__ = [
    "ForwardCost",
    "cost_after",
    "cost_before",
    "cost_cache_only",
    "cost_decode",
    "table",
]
