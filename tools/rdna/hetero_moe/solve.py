# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement solver CLI: statistics + byte table + budgets -> placement plan.

    python -m tools.rdna.hetero_moe.solve --stats RAW.jsonl[:W] ... --out plan.json
    python -m tools.rdna.hetero_moe.solve --stats converted.json --out plan.json
    python -m tools.rdna.hetero_moe.solve --out plan.json      # no stats: striping

The plan is ``hetero-moe-placement/1`` (see ``placement.py`` for the model and
the objective, ``validate.py`` for the invariants). Defaults reproduce the
par1-cs25 box: two islands, each one W7800 (hub) + four V620 (experts); island
HIP indices 0 | 1-4 on switch c0 and 5 | 6-9 on switch 00.

Budgets (bytes available for *routed experts* per device):

* hub: ``--hub-expert-budget-gib``, or derived as ``--hub-vram-gib`` minus the
  hub's non-expert weights from the byte table minus ``--hub-reserve-gib``
  (KV cache, activations, workspaces; a planning assumption, tune it);
* expert devices: ``--expert-budget-gib`` (default 26: the 29.98 GiB V620 minus
  non-torch memory, workspace/graphs, the optional lm_head shard and a margin,
  per the design spec). ``--hub-vram-gib`` defaults to 44.98 (W7800 48 GB) and
  ``--hub-reserve-gib`` to 12.1 (non-torch, workspace, 8 GiB of KV).

Island modes: ``replicate`` (default) writes island-0 owners, every island a
full copy; ``partition`` solves every island on its own half of each layer
(experts split by LPT on expected touches) and writes global device indices.

The estimate block compares the plan with the no-statistics stripe on the same
statistics (analytic expectation and a sampled simulation).
"""

import argparse
import json
import sys
from pathlib import Path

from . import model_bytes
from .placement import (
    SolverConfig,
    Topology,
    fallback_plan,
    plan_document,
    solve,
    step_shape,
)
from .simulate import (
    PlanArrays,
    add_cost_args,
    analytic_metrics,
    arrays_from_placements,
    cost_kwargs,
    fallback_arrays,
    simulate,
)
from .stats_io import StatsRecord, load_stats

GIB = 1 << 30
DEFAULT_TABLE = str(Path(__file__).parent / "data" / "flashnext_w4a16_bytes.json")
DEFAULT_TOP_K = 10


def parse_rows(text: str) -> dict[int, float]:
    """``3`` or ``1:0.2,3:0.8``."""
    if ":" not in text:
        return {int(text): 1.0}
    return {int(a): float(b) for a, _, b in (x.partition(":") for x in text.split(","))}


def parse_hips(text: str) -> list[list[int]]:
    """``0;5`` or ``1,2,3,4;6,7,8,9``."""
    return [[int(x) for x in group.split(",")] for group in text.split(";")]


def build_topology(args: argparse.Namespace) -> Topology:
    if args.islands == 2 and args.hubs_per_island == 1 and args.experts_per_island == 4:
        base = Topology.brief_default()
        hub_hips, exp_hips, switches = base.hub_hips, base.expert_hips, base.switches
    else:
        hub_hips, exp_hips, switches = [], [], []
    if args.hub_hips:
        hub_hips = parse_hips(args.hub_hips)
    if args.expert_hips:
        exp_hips = parse_hips(args.expert_hips)
    if args.switches:
        switches = args.switches.split(",")
    return Topology(
        islands=args.islands,
        n_hubs=args.hubs_per_island,
        n_experts=args.experts_per_island,
        hub_hips=hub_hips,
        expert_hips=exp_hips,
        switches=switches,
    )


def derive_budgets(args: argparse.Namespace, table: dict) -> tuple[int, int]:
    if args.hub_expert_budget_gib is not None:
        hub = round(args.hub_expert_budget_gib * GIB)
    else:
        active = (
            round(args.hub_active_gib * GIB)
            if args.hub_active_gib is not None
            else table["hub_active_bytes"]
        )
        hub = round(args.hub_vram_gib * GIB - active - args.hub_reserve_gib * GIB)
    if hub < 0:
        raise SystemExit(
            "hub has no room for experts after the active part and reserve"
        )
    return hub, round(args.expert_budget_gib * GIB)


def estimate_block(
    pa: PlanArrays,
    base: PlanArrays | None,
    record: StatsRecord,
    steps: int,
    sim_kwargs: dict,
    shape: dict[int, float],
) -> dict:
    block: dict = {"plan": {"analytic": analytic_metrics(pa, record, shape)}}
    if steps:
        block["plan"]["simulated"] = simulate(
            pa, record, steps=steps, shape=shape, **sim_kwargs
        )
    if base is not None:
        block["stripe_baseline"] = {"analytic": analytic_metrics(base, record, shape)}
        if steps:
            block["stripe_baseline"]["simulated"] = simulate(
                base, record, steps=steps, shape=shape, **sim_kwargs
            )
        a, b = block["plan"]["analytic"], block["stripe_baseline"]["analytic"]
        block["gain_vs_stripe"] = {
            "decode_hub_hit": a["decode_hub_hit"] - b["decode_hub_hit"],
            "cold_touched_critical_ratio": (
                a["cold_touched_critical_per_step"]
                / b["cold_touched_critical_per_step"]
                if b["cold_touched_critical_per_step"]
                else None
            ),
        }
        if steps:  # the analytic figure is max-of-means, a lower bound
            sim_p, sim_b = (
                block["plan"]["simulated"],
                block["stripe_baseline"]["simulated"],
            )
            block["gain_vs_stripe"]["simulated_cold_max_touched_ratio"] = (
                sim_p["cold_max_touched_per_step"] / sim_b["cold_max_touched_per_step"]
                if sim_b["cold_max_touched_per_step"]
                else None
            )
            block["gain_vs_stripe"]["simulated_step_us_ratio"] = (
                sim_p["step_us"] / sim_b["step_us"] if sim_b["step_us"] else None
            )
    block["step_shape"] = {str(k): v for k, v in shape.items()}
    if steps:
        block["dispatch_variants"] = dispatch_variants(
            pa, record, steps, sim_kwargs, shape, block["plan"]
        )
    return block


_VARIANT_KEYS = (
    "step_us",
    "cold_mean_touched_per_step",
    "cold_max_touched_per_step",
    "cold_distinct_experts_per_step",
    "cold_replica_touch_ratio",
    "hub_kept_per_layer_per_hub",
)


def dispatch_variants(
    pa: PlanArrays,
    record: StatsRecord,
    steps: int,
    sim_kwargs: dict,
    shape: dict[int, float],
    plan_block: dict,
) -> dict:
    """Decode cost of the three cold-copy dispatch rules, and the replica benefit.

    Replicas help prefill (rows of one expert spread over its copies) but a decode
    step streams a replicated expert once per copy that receives a row; the
    ``primary`` rule gives the decode-only cost without that duplication. The
    prefill figure is the analytic per-device pair-load imbalance (max / mean) with
    rows spread over the copies against the primary copy only.
    """
    out: dict = {"simulated_decode": {}}
    for rule in ("row", "expert", "primary"):
        kw = {**sim_kwargs, "dispatch": rule}
        res = simulate(pa, record, steps=steps, shape=shape, **kw)
        out["simulated_decode"][rule] = {k: res[k] for k in _VARIANT_KEYS}
    sim = out["simulated_decode"]
    out["decode_step_us_ratio_primary_vs_row"] = (
        sim["primary"]["step_us"] / sim["row"]["step_us"]
    )
    analytic = plan_block["analytic"]
    out["prefill_pair_load_imbalance"] = {
        "replicas_spread": analytic["prefill_pair_load_imbalance"],
        "primary_only": analytic["prefill_pair_load_imbalance_primary_only"],
    }
    return out


SPILL_WIDTHS = (3, 6, 12, 24, 48)


def spill_k_table(
    pa: PlanArrays, record: StatsRecord, sim_kwargs: dict, steps: int = 400
) -> dict:
    """``runtime.spill_k``: experts a hub computes itself (the rest of its touched
    resident experts run on their V620 copies) per hub and layer, for each capture
    width, from the simulator's spill rule at that fixed width.
    """
    by_rule: dict[str, dict[str, int]] = {}
    for rule in ("row", "expert", "primary"):
        kw = {**sim_kwargs, "dispatch": rule}
        by_rule[rule] = {}
        for width in SPILL_WIDTHS:
            res = simulate(pa, record, steps=steps, shape={width: 1.0}, **kw)
            by_rule[rule][str(width)] = int(round(res["hub_kept_per_layer_per_hub"]))
    declared = str(sim_kwargs.get("dispatch", "row"))
    return {
        "spill_k": by_rule[declared],
        "spill_k_dispatch": declared,
        "spill_k_exchange": sim_kwargs.get("exchange", "broadcast"),
        "spill_k_row_model": str(sim_kwargs.get("row_model") or "flashnext-mtp2"),
        "spill_k_by_dispatch": by_rule,
        "spill_k_basis": (
            f"simulate at fixed rows per step, {steps} steps, exchange="
            f"{sim_kwargs.get('exchange', 'broadcast')}, dispatch="
            f"{sim_kwargs.get('dispatch', 'row')}, rows="
            f"{sim_kwargs.get('row_model') or 'flashnext-mtp2'}; kept = touched "
            "hub-resident "
            "experts the hub computes after the spill rule, mean per hub and "
            "target layer"
        ),
    }


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stats", nargs="*", help="raw JSONL[:WEIGHT] files or one JSON")
    p.add_argument("--t0", type=float, help="window start, elapsed_s of raw files")
    p.add_argument("--t1", type=float, help="window end, elapsed_s of raw files")
    p.add_argument("--bytes", dest="table", default=DEFAULT_TABLE)
    p.add_argument("--out", required=True)
    g = p.add_argument_group("topology")
    g.add_argument("--islands", type=int, default=2)
    g.add_argument("--hubs-per-island", type=int, default=1)
    g.add_argument("--experts-per-island", type=int, default=4)
    g.add_argument("--hub-hips", help="e.g. '0;5' (one group per island)")
    g.add_argument("--expert-hips", help="e.g. '1,2,3,4;6,7,8,9'")
    g.add_argument("--switches", help="e.g. 'c0,00'")
    g = p.add_argument_group("budgets")
    g.add_argument("--hub-expert-budget-gib", type=float)
    g.add_argument("--hub-vram-gib", type=float, default=44.98)  # W7800 48 GB
    g.add_argument(
        "--hub-active-gib",
        type=float,
        help="hub non-expert weights (default: the byte table, whole model on one "
        "hub; about 5.8 for a TP2 hub)",
    )
    g.add_argument("--hub-reserve-gib", type=float, default=12.1)  # KV + workspace
    g.add_argument("--expert-budget-gib", type=float, default=26.0)  # 29.98 V620
    g = p.add_argument_group("objective")
    g.add_argument("--w-decode", type=float, default=1.0)
    g.add_argument("--w-prefill", type=float, default=0.05)
    g.add_argument("--rows-per-step", type=parse_rows, help="3 or 1:0.2,3:0.8")
    g.add_argument(
        "--max-hub-per-layer",
        type=int,
        help="default 130 (115 per island with --island-mode partition)",
    )
    g.add_argument("--min-hub-per-layer", type=int, default=0)
    g.add_argument(
        "--hub-mode", choices=("replicate", "shard", "partition"), default="replicate"
    )
    g.add_argument(
        "--island-mode",
        choices=("replicate", "partition"),
        default="replicate",
        help="replicate: every island holds every expert (one engine per switch); "
        "partition: each expert lives in one island (TP2 across the hubs, EP2)",
    )
    g.add_argument("--mtp-policy", choices=("omit", "hub", "solve"), default="hub")
    g.add_argument(
        "--mtp-expert-bytes",
        type=int,
        help="bytes per MTP expert (default: a target expert, W4A16 requantised; "
        "9830400 for the bf16 checkpoint)",
    )
    g.add_argument("--no-cold-complete", action="store_true")
    g.add_argument("--no-replicas", action="store_true")
    g.add_argument("--no-coact", action="store_true", help="ignore the pairs file")
    g.add_argument("--refine-iters", type=int, default=40)
    g = p.add_argument_group("fallback and estimate")
    g.add_argument("--stripe", type=int, default=4, help="fallback hub stripe R")
    g.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="fallback only")
    g.add_argument("--simulate-steps", type=int, default=1000)
    g.add_argument("--seed", type=int, default=0)
    add_cost_args(g)
    args = p.parse_args(argv)

    table = model_bytes.load(args.table)
    topo = build_topology(args)
    hub_budget, expert_budget = derive_budgets(args, table)
    mtp = table.get("mtp") or {}
    max_hub = args.max_hub_per_layer
    if max_hub is None:  # per island half under partition (hub holds <= half)
        max_hub = 115 if args.island_mode == "partition" else 130
    cfg = SolverConfig(
        expert_bytes=table["expert_bytes"],
        hub_budget_bytes=hub_budget,
        expert_budget_bytes=expert_budget,
        # requantised MTP experts have the geometry of a target expert [INFERENCE]
        mtp_expert_bytes=args.mtp_expert_bytes or table["expert_bytes"],
        hub_mode=args.hub_mode,
        island_mode=args.island_mode,
        max_hub_per_layer=max_hub,
        min_hub_per_layer=args.min_hub_per_layer,
        w_decode=args.w_decode,
        w_prefill=args.w_prefill,
        rows_per_step=args.rows_per_step,
        cold_complete=not args.no_cold_complete,
        replicate=not args.no_replicas,
        coact_refine=not args.no_coact,
        refine_iters=args.refine_iters,
        mtp_policy=args.mtp_policy,
        fallback_stripe=args.stripe,
    )
    sim_kwargs = {"seed": args.seed, **cost_kwargs(args)}
    meta = {
        "limits": {
            "max_hub_per_layer": max_hub,
            "min_hub_per_layer": args.min_hub_per_layer,
        },
        "cost_model": cost_kwargs(args),
        "weights": {"decode": args.w_decode, "prefill": args.w_prefill},
        "budgets_gib": {
            "hub_experts": round(hub_budget / GIB, 3),
            "expert_device": round(expert_budget / GIB, 3),
        },
        "mtp_policy": args.mtp_policy,
    }

    if not args.stats:
        n_layers, n_experts = table["num_layers"], table["num_experts"]
        placements = fallback_plan(n_layers, n_experts, topo, cfg)
        model = {
            "name": table.get("model"),
            "num_layers": n_layers,
            "num_experts": n_experts,
            "top_k": args.top_k,
            "mtp_layers": mtp.get("layers", 0),
        }
        pa = arrays_from_placements(placements, topo, n_experts)
        estimate = {
            "plan": {
                "analytic": {
                    "note": "no statistics: striping, no hit-rate estimate",
                    "hub_experts_per_layer": {
                        "total": int(pa.hub.sum()),
                        "per_layer": int(pa.hub[0].sum()),
                    },
                }
            }
        }
        doc = plan_document(
            placements, topo, cfg, model, estimate, "uniform-fallback", [], meta
        )
    else:
        record = load_stats(args.stats, args.t0, args.t1)
        placements, _, _ = solve(record, topo, cfg)
        model = dict(record.model)
        model.setdefault("name", table.get("model"))
        model["num_experts"] = record.num_experts
        model["top_k"] = record.top_k
        if cfg.mtp_policy != "omit" and mtp:
            model["mtp_expert_bytes"] = cfg.mtp_expert_bytes
        pa = arrays_from_placements(placements, topo, record.num_experts)
        base = fallback_arrays(record, topo, cfg)
        shape = step_shape(cfg, record)
        estimate = estimate_block(
            pa, base, record, args.simulate_steps, sim_kwargs, shape
        )
        doc = plan_document(
            placements, topo, cfg, model, estimate, "stats", record.sources, meta
        )
        if args.simulate_steps:
            doc["runtime"] = spill_k_table(pa, record, sim_kwargs)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    print(
        f"wrote {args.out}: plan {doc['plan_id']} source={doc['source']} "
        f"layers={len(doc['layers'])} hub experts={int(pa.hub.sum())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
