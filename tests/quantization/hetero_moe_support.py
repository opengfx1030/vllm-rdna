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
