# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers of the hetero-MoE CPU tests.

``hetero(name)`` loads a module of ``vllm/distributed/hetero_moe`` by path under a
private package name, so the numpy/torch-only parts (plan, tables, reference ops,
runtime orchestration) are testable without importing the vLLM stack. The plan
builders use ExpertStats' solver API with budgets in "expert slots" of 1000
bytes, like ``tests/tools/rdna/test_hetero_moe.py``.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = ROOT / "vllm" / "distributed" / "hetero_moe"
PKG = "_hetero_moe_under_test"
SLOT = 1000


def hetero(name: str):
    """A module of the hetero_moe package, e.g. ``hetero("plan")``."""
    if PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PKG, PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[PKG] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{PKG}.{name}")


def stripe_doc(
    *,
    islands: int = 1,
    servers: int = 4,
    experts: int = 64,
    layers: int = 6,
    top_k: int = 8,
    stripe: int = 4,
    hub_slots: int | None = None,
    island_mode: str = "replicate",
    mtp: bool = False,
) -> dict:
    """Stripe (no statistics) plan document at test scale."""
    from tools.rdna.hetero_moe.placement import (
        LayerPlacement,
        SolverConfig,
        Topology,
        fallback_plan,
        plan_document,
    )

    topo = Topology(islands=islands, n_hubs=1, n_experts=servers)
    per_island = experts // islands if island_mode == "partition" else experts
    slots = hub_slots if hub_slots is not None else per_island // stripe
    cfg = SolverConfig(
        expert_bytes=SLOT,
        hub_budget_bytes=(slots * layers + (experts if mtp else 0)) * SLOT,
        expert_budget_bytes=((per_island // servers + 1) * layers + 4) * SLOT,
        max_hub_per_layer=experts,
        fallback_stripe=stripe,
        island_mode=island_mode,
        mtp_policy="hub" if mtp else "omit",
    )
    placements = fallback_plan(layers, experts, topo, cfg)
    if mtp:
        assert island_mode == "replicate", "MTP helper covers replicate plans"
        placements[layers] = LayerPlacement(
            layers,
            "mtp",
            hub=[list(range(experts))],
            owner=-np.ones(experts, dtype=np.int64),
            replicas={},
            hub_only=True,
        )
    model = {"num_layers": layers, "num_experts": experts, "top_k": top_k}
    return plan_document(placements, topo, cfg, model, {}, "uniform-fallback", [])


def solved_doc(tmp_path, *, experts: int = 64, layers: int = 6, top_k: int = 8):
    """H1 plan solved from synthetic Zipf statistics (one island, 4 servers)."""
    from tools.rdna.hetero_moe import stats_io, synth
    from tools.rdna.hetero_moe.placement import (
        SolverConfig,
        Topology,
        plan_document,
    )
    from tools.rdna.hetero_moe.placement import solve as solve_placement

    data = synth.generate(
        kind="zipf",
        layers=layers,
        experts=experts,
        top_k=top_k,
        alpha=1.1,
        decode_rows=3000,
        prefill_rows=2000,
        seed=1,
    )
    record = stats_io.load_stats([synth.write_raw(data, str(tmp_path / "stats"))])
    topo = Topology(islands=1, n_hubs=1, n_experts=4)
    cfg = SolverConfig(
        expert_bytes=SLOT,
        hub_budget_bytes=(experts // 4) * layers * SLOT,
        expert_budget_bytes=((experts // 4) * layers + 8) * SLOT,
        max_hub_per_layer=experts,
    )
    placements, _, _ = solve_placement(record, topo, cfg)
    model = {"num_layers": layers, "num_experts": experts, "top_k": top_k}
    return plan_document(placements, topo, cfg, model, {}, "stats", [])


# ---- reference island shared by the runtime and method tests ----------------------

E, TOP_K, HIDDEN, INTER, GROUP = 64, 8, 256, 128, 128
EXACT = {"rtol": 0, "atol": 1e-12}  # float64 partials: the sum is exact to rounding


def stripe_plan(**kw):
    return hetero("plan").plan_from_doc(stripe_doc(layers=3, **kw))


def replicated_doc(*, copies: int = 3, **kw) -> dict:
    """Stripe plan document whose experts all have ``copies`` server copies: the
    stripe's primary first, then the next servers (replicas ascending, as the plan
    format wants them). A cold pair then goes to a different server under the
    ``row`` and the ``primary`` decode dispatch rules. The server budgets are raised
    to hold the copies."""
    doc = stripe_doc(layers=3, **kw)
    servers = [d for d, dev in enumerate(doc["devices"]) if dev["role"] == "expert"]
    for dev in doc["devices"]:
        if dev["role"] == "expert":
            dev["expert_budget_bytes"] *= copies
    for layer in doc["layers"]:
        for row in layer["owners"]:
            held = [d for d in row if d in servers]
            if not held:
                continue
            start = servers.index(held[0])
            more = [servers[(start + s) % len(servers)] for s in range(1, len(servers))]
            row[:] = [d for d in row if d not in servers] + [held[0]]
            row += sorted(more[: copies - 1])
    doc["plan_id"] = hetero("plan").compute_plan_id(doc)
    return doc


def replicated_plan(**kw):
    return hetero("plan").plan_from_doc(replicated_doc(**kw))


def make_experts(layers: int = 3) -> dict:
    """Checkpoint experts per layer: {layer: {expert id: Expert}}."""
    ref_moe = hetero("ref_moe")
    return {
        layer: ref_moe.make_experts(E, HIDDEN, INTER, GROUP, seed=10 + layer)
        for layer in range(layers)
    }


def routing(rows, seed, bad=True):
    import torch

    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    ids = torch.stack(
        [torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows)]
    ).to(torch.int32)
    if bad and rows > 1:
        ids[0, 0] = -1  # a padded / dropped pair
        ids[1, 1] = E + 7  # beyond the table: invalid as well
    w = torch.softmax(torch.randn(rows, TOP_K, generator=gen), dim=-1)
    return x, w.float(), ids


def invalid_pairs(ids) -> int:
    return int(((ids < 0) | (ids >= E)).sum())


def full_moe(experts, layer, x, w, ids):
    return hetero("ref_moe").moe_partial(x, w, ids, experts[layer])


class Island:
    """One hub and its servers over the reference experts (one plan island)."""

    def __init__(
        self,
        plan,
        hub_dev,
        experts,
        *,
        spill_k=None,
        exact=True,
        decode_dispatch="row",
    ):
        import torch

        self.plan = plan
        self.experts = experts
        self.tables = hetero("tables").build_island_tables(
            plan, hub_dev, spill_k=spill_k, decode_dispatch=decode_dispatch
        )
        servers = plan.experts(plan.island_of(hub_dev))
        fns = {s: self._server_fn(dev) for s, dev in enumerate(servers)}
        self.hub = hetero("ref_hx").RefHub(self.tables, fns, exact=exact)
        # the layer's slot_map (the hub MoE's expert_map) and its slot pool
        self.slot_map = {
            li: torch.from_numpy(lt.slot_map.copy())
            for li, lt in self.tables.layers.items()
        }
        self.slots = {
            li: [experts[li][int(e)] for e in lt.resident]
            for li, lt in self.tables.layers.items()
        }

    def _server_fn(self, dev):
        ref_moe = hetero("ref_moe")

        def fn(layer, x, ids, w):
            held = {
                int(e): self.experts[layer][int(e)]
                for e in self.plan.server_experts(dev, layer)
            }
            return ref_moe.moe_partial(x, w, ids, held)

        return fn

    def hub_moe(self, layer, x, w):
        ref_moe = hetero("ref_moe")

        def fn(hub_ids):
            return ref_moe.moe_partial(
                x,
                w,
                hub_ids,
                {},
                slot_map=self.slot_map[layer],
                slot_experts=self.slots[layer],
            )

        return fn

    def forward(self, layer, x, w, ids):
        """Dispatch -> hub MoE -> combine, the sequence of ``HubRuntime``."""
        import torch

        ref_hx = hetero("ref_hx")
        hub = self.hub
        ref_hx.begin_step(hub, x.shape[0], 0)
        hub_ids = torch.empty_like(ids)
        ref_hx.dispatch(hub, x, ids, w, hub_ids, layer)
        y = self.hub_moe(layer, x, w)(hub_ids)
        return ref_hx.combine(hub, y, y, layer), hub_ids
