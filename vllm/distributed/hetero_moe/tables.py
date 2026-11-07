# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch tables of one hub and the reference spill rule (spec 6.2, 9.2).

``build_island_tables(plan, hub)`` turns the plan into what the hub needs per
layer (all int32, numpy, no torch):

* ``slot_map[E]``: slot of the expert in the hub's slot pool, -1 when not
  resident. It is also the ``expert_map`` of the hub's fused MoE call.
* ``n_cold[E]`` / ``cold_srv[E, MAX_COLD]``: the island's servers that hold a copy
  of the expert, in the plan's owner order (primary first, replicas ascending).

Dispatch rule (the plan format's runtime rule and the spill rule of spec 6.2):

* a pair is valid iff ``0 <= id < num_experts`` and ``owned[id]``; any other pair
  is dropped (``owned`` is 0 for the experts of the other island in ``partition``
  mode: that island computes them);
* a hub-resident expert stays on the hub iff its slot ordinal among the *touched*
  resident experts of this call (prefix popcount of the touched-slot bitmap in
  slot order) is below ``K``; ``K`` is a static table per capture width;
* every other valid pair goes to server ``cold_srv[e][(row + layer) % n_cold[e]]``;
  an owned expert that is neither kept nor has a copy (``n_cold[e] == 0``) is a
  plan error.

``split_pairs_loop`` is the plain-loop statement of the rule that the vectorised
torch version (``ref_hx.split_pairs``), the CPU reference of the ``hx`` ops and the
HIP dispatch kernel are all tested against.
"""

from dataclasses import dataclass, field

import numpy as np
import regex as re

from .plan import MAX_COLD, HeteroPlan, PlanError

# Default K per capture width W = 3, 6, 12, 24, 48 (HeteroDesign's cost model,
# `perf_model.spill_k`, 4 servers per hub): the number of touched hub-resident
# experts that stay on the hub when the hub and its servers finish together.
# H1/replicate: the hub covers the whole expert set and touches twice the experts
# of a partition hub, so its balance points are higher. Re-tuned in S5 from
# `simulate`'s hub_spilled_per_step and the combine-wait counters. [INFERENCE]
DEFAULT_SPILL_K_FULL: dict[int, int] = {3: 10, 6: 21, 12: 39, 24: 65, 48: 110}
DEFAULT_SPILL_K_PARTITION: dict[int, int] = {3: 7, 6: 13, 12: 23, 24: 39, 48: 63}
_REFERENCE_SERVERS = 4


@dataclass(frozen=True)
class LayerTables:
    layer: int
    kind: str
    hub_only: bool
    resident: np.ndarray  # int32 [n_slots]: expert id of each slot
    slot_map: np.ndarray  # int32 [E]: slot or -1
    n_cold: np.ndarray  # int32 [E]
    cold_srv: np.ndarray  # int32 [E, MAX_COLD]: server index, -1 padded
    owned: np.ndarray  # uint8 [E]: the expert belongs to this hub's island

    @property
    def n_slots(self) -> int:
        return int(self.resident.shape[0])

    @property
    def uses_hx(self) -> bool:
        """The layer talks to the servers: every plan layer except hub-only ones
        (the servers derive the same layer list from the plan)."""
        return not self.hub_only


@dataclass
class StackedTables:
    """The tables of the layers that talk to the servers, stacked for upload
    (``L`` = highest such layer + 1). Hub-only layers (the MTP draft) are not part of
    it: their ids are not table indexes. Layers the plan does not list have
    ``present == 0`` and are never dispatched."""

    present: np.ndarray  # int32 [L]
    hub_only: np.ndarray  # int32 [L]
    n_slots: np.ndarray  # int32 [L]
    slot_map: np.ndarray  # int32 [L, E]
    n_cold: np.ndarray  # int32 [L, E]
    cold_srv: np.ndarray  # int32 [L, E, MAX_COLD]
    owned: np.ndarray  # uint8 [L, E]


@dataclass
class IslandTables:
    plan_id: str
    island: int
    hub_dev: int
    num_servers: int
    num_experts: int
    top_k: int
    layers: dict[int, LayerTables]
    spill_k: dict[int, int]  # capture width -> K
    prefill_k: int  # K for rows above the largest width
    server_ids: list[str] = field(default_factory=list)
    # Why spilling is off (K = max n_slots everywhere): a hub-resident expert
    # without a server copy cannot leave the hub. None when spilling is allowed.
    spill_disabled: str | None = None

    @property
    def widths(self) -> list[int]:
        return sorted(self.spill_k)

    def bucket_for(self, rows: int) -> int:
        """Smallest capture width that holds ``rows``; 0 when none does."""
        for width in self.widths:
            if rows <= width:
                return width
        return 0

    def k_for(self, rows: int) -> int:
        bucket = self.bucket_for(rows)
        return self.spill_k[bucket] if bucket else self.prefill_k

    def stacked(self) -> StackedTables:
        hx = {li: lt for li, lt in self.layers.items() if lt.uses_hx}
        n = max(hx) + 1 if hx else 0
        e = self.num_experts
        out = StackedTables(
            present=np.zeros(n, np.int32),
            hub_only=np.zeros(n, np.int32),
            n_slots=np.zeros(n, np.int32),
            slot_map=np.full((n, e), -1, np.int32),
            n_cold=np.zeros((n, e), np.int32),
            cold_srv=np.full((n, e, MAX_COLD), -1, np.int32),
            owned=np.zeros((n, e), np.uint8),
        )
        for li, lt in hx.items():
            out.present[li] = 1
            out.hub_only[li] = int(lt.hub_only)
            out.n_slots[li] = lt.n_slots
            out.slot_map[li] = lt.slot_map
            out.n_cold[li] = lt.n_cold
            out.cold_srv[li] = lt.cold_srv
            out.owned[li] = lt.owned
        return out


def parse_spill_k(text: str) -> dict[int, int]:
    """``"3:7,6:14,12:26"`` -> ``{3: 7, 6: 14, 12: 26}``."""
    out: dict[int, int] = {}
    for item in re.split(r"[,\s]+", text.strip()):
        if not item:
            continue
        bucket, sep, k = item.partition(":")
        if not sep or not bucket.isdigit() or not k.isdigit() or int(bucket) < 1:
            raise PlanError(f"bad spill K entry {item!r}: expected BUCKET:K")
        out[int(bucket)] = int(k)
    if not out:
        raise PlanError("empty spill K table")
    return out


def default_spill_k(num_servers: int, partition: bool = False) -> dict[int, int]:
    """Cost-model table for the plan shape, scaled linearly to the island's server
    count (the cost model covers 4 servers; other counts are a first guess,
    [INFERENCE]; never below 1)."""
    table = DEFAULT_SPILL_K_PARTITION if partition else DEFAULT_SPILL_K_FULL
    scale = num_servers / _REFERENCE_SERVERS
    return {w: max(1, round(k * scale)) for w, k in table.items()}


def build_layer_tables(plan: HeteroPlan, hub_dev: int, layer: int) -> LayerTables:
    island = plan.island_of(hub_dev)
    resident = plan.hub_resident(hub_dev, layer)
    e_n = plan.num_experts
    slot_map = np.full(e_n, -1, np.int32)
    slot_map[resident] = np.arange(resident.shape[0], dtype=np.int32)
    n_cold = np.zeros(e_n, np.int32)
    cold_srv = np.full((e_n, MAX_COLD), -1, np.int32)
    if not plan.is_hub_only(layer):
        for e, copies in enumerate(plan.cold_copies(layer, island)):
            n_cold[e] = len(copies)
            cold_srv[e, : len(copies)] = copies
    owners = plan.island_owners(layer, island)
    owned = np.fromiter((1 if row else 0 for row in owners), np.uint8, count=e_n)
    return LayerTables(
        layer=layer,
        kind=plan.layer_kind(layer),
        hub_only=plan.is_hub_only(layer),
        resident=resident,
        slot_map=slot_map,
        n_cold=n_cold,
        cold_srv=cold_srv,
        owned=owned,
    )


def build_island_tables(
    plan: HeteroPlan,
    hub_dev: int,
    *,
    spill_k: dict[int, int] | None = None,
    prefill_k: int | None = None,
) -> IslandTables:
    """Tables of the island of ``hub_dev``.

    ``spill_k`` precedence: the argument, ``runtime.spill_k`` of the plan, then
    the default scaled to the island's server count. ``prefill_k`` (rows above
    the largest capture width) defaults to no spill."""
    if plan.role(hub_dev) != "hub":
        raise PlanError(f"device {plan.device_id(hub_dev)} is not a hub")
    island = plan.island_of(hub_dev)
    if len(plan.hubs(island)) != 1:
        raise PlanError(
            f"island {island} has {len(plan.hubs(island))} hubs; the runtime "
            "supports one hub per island (TP across islands is the H2 shape)"
        )
    servers = plan.experts(island)
    layers = {li: build_layer_tables(plan, hub_dev, li) for li in plan.layer_ids}
    k_table = (
        spill_k
        or plan.spill_k()
        or default_spill_k(len(servers), plan.island_mode == "partition")
    )
    max_slots = max((lt.n_slots for lt in layers.values()), default=0)
    if prefill_k is None:
        prefill_k = max_slots
    # A pair that leaves the hub needs a server copy of its expert.
    bare = [
        li
        for li, lt in layers.items()
        if not lt.hub_only and (lt.n_cold[lt.resident] == 0).any()
    ]
    spill_disabled = None
    if bare:
        spill_disabled = (
            f"layers {bare[:4]}{'...' if len(bare) > 4 else ''} have hub-resident "
            "experts without a server copy (plan is not cold_complete)"
        )
        k_table = dict.fromkeys(k_table, max_slots)
        prefill_k = max_slots
    return IslandTables(
        plan_id=plan.plan_id,
        island=island,
        hub_dev=hub_dev,
        num_servers=len(servers),
        num_experts=plan.num_experts,
        top_k=plan.top_k,
        layers=layers,
        spill_k=dict(sorted(k_table.items())),
        prefill_k=int(prefill_k),
        server_ids=[plan.device_id(d) for d in servers],
        spill_disabled=spill_disabled,
    )


def split_pairs_loop(
    topk_ids: np.ndarray, lt: LayerTables, layer: int, k_keep: int
) -> tuple[np.ndarray, np.ndarray]:
    """The rule in plain loops (the test oracle).

    Returns ``(hub_ids, dest)``, both int32 ``[T, top_k]``: ``hub_ids`` is the
    global expert id of a pair that runs on the hub, else -1; ``dest`` is the
    server index of a pair that runs remotely, else -1 (hub pairs and dropped
    pairs: invalid ids and experts of another island). Raises ``PlanError`` for
    an owned expert that is neither kept nor has a server copy."""
    t_n, k_n = topk_ids.shape
    e_n = lt.slot_map.shape[0]

    def valid(e: int) -> bool:
        return 0 <= e < e_n and bool(lt.owned[e])

    touched = set()
    for t in range(t_n):
        for j in range(k_n):
            e = int(topk_ids[t, j])
            if valid(e) and lt.slot_map[e] >= 0:
                touched.add(int(lt.slot_map[e]))
    keep_slots = set(sorted(touched)[:k_keep])
    hub_ids = np.full((t_n, k_n), -1, np.int32)
    dest = np.full((t_n, k_n), -1, np.int32)
    for t in range(t_n):
        for j in range(k_n):
            e = int(topk_ids[t, j])
            if not valid(e):
                continue
            slot = int(lt.slot_map[e])
            if slot >= 0 and slot in keep_slots:
                hub_ids[t, j] = e
            elif lt.n_cold[e] > 0:
                dest[t, j] = lt.cold_srv[e, (t + layer) % int(lt.n_cold[e])]
            else:
                raise PlanError(
                    f"layer {layer} expert {e} is owned by the island but is "
                    "neither on the hub nor on a server"
                )
    return hub_ids, dest
