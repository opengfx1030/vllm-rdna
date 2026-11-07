# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing simulator and analytic estimate for a placement plan.

For a plan (``hetero-moe-placement/1``, island 0) and routing statistics, report:

* ``hub_hit``: share of (row, expert) selections served by the hub, decode and
  prefill;
* per decode step and layer, the number of *distinct cold experts touched* on
  every V620 (the quantity the solver minimises), its mean over devices, the
  maximum over devices (expected critical path) and the imbalance
  ``E[max] / E[mean]``; summed over layers, per step;
* PCIe traffic: rows sent to each V620 per layer (``hidden_bytes`` each way over
  the intra-switch link: ``latency_us`` + bytes / ``gbps``);
* optional step time ``sum_l max(hub_touched * hub_us, comm + max_touched * exp_us)``
  when the per-expert GEMM costs are given (they come from the RDNA3 port and the
  V620 kernels; none is assumed here).

Dispatch matches the runtime rule: the hub computes a hub-resident expert, else
row ``r`` of layer ``l`` goes to cold copy ``(r + l) % n_cold_copies`` (owners in
listed order). Rows are drawn per step with the step-size distribution of the
statistics; per layer, a row's top-k set is a systematic sample with the exact
per-expert selection probabilities (independent between layers, pairwise
independent within a row) unless a routing trace is supplied (``evaluate_ids``),
which keeps real co-activation.

CLI::

    python -m tools.rdna.hetero_moe.simulate PLAN.json --stats STATS [--steps 2000]
        [--seed 0] [--compare-fallback] [--hub-us X --expert-us Y]
        [--hidden-bytes 5120 --gbps 24.5 --latency-us 6.2]
"""

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from .placement import (
    LayerPlacement,
    SolverConfig,
    Topology,
    fallback_plan,
    owners_lists,
    step_shape,
)
from .sampling import inclusion_probs, systematic_sample
from .stats_io import StatsRecord, load_stats

HIDDEN_BYTES = 5120  # hidden 2560 x fp16 (brief)
PCIE_GBPS = 24.5  # intra-switch per-pair P2P, low end of 24.5-27 GB/s (brief)
PCIE_LATENCY_US = 6.2  # intra-switch (brief)


@dataclass
class PlanArrays:
    layers: list[int]
    num_experts: int
    n_hubs: int
    n_devices: int  # expert devices per island
    hub: np.ndarray  # bool [L, E]
    copies: np.ndarray  # int [L, E, D] local expert-device idx, -1 padded
    ncold: np.ndarray  # int [L, E]
    hub_only: list[bool]
    kinds: list[str]


def parse_plan(doc: dict) -> PlanArrays:
    """Island-0 arrays of a plan document."""
    devices = doc["devices"]
    island = doc["islands"][0]
    roles = {d: devices[d]["role"] for d in island}
    hubs = [d for d in island if roles[d] == "hub"]
    experts = [d for d in island if roles[d] == "expert"]
    local = {d: i for i, d in enumerate(experts)}
    layers = doc["layers"]
    num_experts = doc["model"]["num_experts"]
    n_layers = len(layers)
    hub = np.zeros((n_layers, num_experts), dtype=bool)
    copies = -np.ones((n_layers, num_experts, max(1, len(experts))), dtype=np.int64)
    ncold = np.zeros((n_layers, num_experts), dtype=np.int64)
    for i, layer in enumerate(layers):
        for e, owners in enumerate(layer["owners"]):
            cold = [local[d] for d in owners if d in local]
            if any(d in hubs for d in owners):
                hub[i, e] = True
            copies[i, e, : len(cold)] = cold
            ncold[i, e] = len(cold)
    return PlanArrays(
        [x["layer"] for x in layers],
        num_experts,
        len(hubs),
        len(experts),
        hub,
        copies,
        ncold,
        [bool(x.get("hub_only")) for x in layers],
        [x.get("kind", "target") for x in layers],
    )


def arrays_from_placements(
    placements: dict[int, LayerPlacement], topo: Topology, num_experts: int
) -> PlanArrays:
    """Same arrays straight from solver output (no JSON round trip)."""
    layers = sorted(placements)
    hub = np.zeros((len(layers), num_experts), dtype=bool)
    copies = -np.ones(
        (len(layers), num_experts, max(1, topo.n_experts)), dtype=np.int64
    )
    ncold = np.zeros((len(layers), num_experts), dtype=np.int64)
    for i, li in enumerate(layers):
        lp = placements[li]
        for e, owners in enumerate(owners_lists(lp, topo, num_experts)):
            cold = [d - topo.n_hubs for d in owners if d >= topo.n_hubs]
            hub[i, e] = any(d < topo.n_hubs for d in owners)
            copies[i, e, : len(cold)] = cold
            ncold[i, e] = len(cold)
    return PlanArrays(
        layers,
        num_experts,
        topo.n_hubs,
        topo.n_experts,
        hub,
        copies,
        ncold,
        [placements[li].hub_only for li in layers],
        [placements[li].kind for li in layers],
    )


# ---- analytic estimate --------------------------------------------------------


def _marginals(record: StatsRecord, layer: int) -> tuple[np.ndarray, np.ndarray]:
    ls = record.layers[layer]
    q_dec = ls.decode / ls.decode_tokens if ls.decode_tokens > 0 else 0 * ls.decode
    q_pre = ls.prefill / ls.prefill_tokens if ls.prefill_tokens > 0 else 0 * ls.prefill
    return np.clip(q_dec, 0, 1), np.clip(q_pre, 0, 1)


def analytic_metrics(
    pa: PlanArrays, record: StatsRecord, shape: dict[int, float]
) -> dict:
    """Expected hub hit rates and per-device touched experts (no sampling)."""
    hit_dec = tot_dec = hit_pre = tot_pre = 0.0
    crit = mean_sum = 0.0
    per_layer = []
    n_dev = max(1, pa.n_devices)
    pair_load_ratio = []
    for i, layer in enumerate(pa.layers):
        if pa.hub_only[i] or layer not in record.layers:
            continue
        q_dec, q_pre = _marginals(record, layer)
        hit_dec += float((q_dec * pa.hub[i]).sum())
        tot_dec += float(q_dec.sum())
        hit_pre += float((q_pre * pa.hub[i]).sum())
        tot_pre += float(q_pre.sum())
        touched = np.zeros(n_dev)
        cold = ~pa.hub[i] & (pa.ncold[i] > 0)
        for rows, p in shape.items():
            for e in np.flatnonzero(cold):
                c = int(pa.ncold[i, e])
                for k in range(c):
                    n_k = sum(1 for r in range(rows) if (r + layer) % c == k)
                    if n_k:
                        touched[pa.copies[i, e, k]] += p * (
                            1.0 - (1.0 - q_dec[e]) ** n_k
                        )
        crit += float(touched.max()) if n_dev else 0.0
        mean_sum += float(touched.mean())
        per_layer.append(float(touched.max()))
        # prefill pair load per device (compute balance), uniform over cold copies
        load = np.zeros(n_dev)
        for e in np.flatnonzero(cold):
            c = int(pa.ncold[i, e])
            for k in range(c):
                load[pa.copies[i, e, k]] += q_pre[e] / c
        if load.mean() > 0:
            pair_load_ratio.append(float(load.max() / load.mean()))
    hub_counts = [int(pa.hub[i].sum()) for i in range(len(pa.layers))]
    return {
        "decode_hub_hit": hit_dec / tot_dec if tot_dec else 0.0,
        "prefill_hub_hit": hit_pre / tot_pre if tot_pre else 0.0,
        "cold_touched_critical_per_step": crit,
        "cold_touched_mean_per_step": mean_sum,
        "cold_imbalance": crit / mean_sum if mean_sum else 1.0,
        "prefill_pair_load_imbalance": (
            float(np.mean(pair_load_ratio)) if pair_load_ratio else 1.0
        ),
        "hub_experts_per_layer": {
            "min": min(hub_counts),
            "mean": float(np.mean(hub_counts)),
            "max": max(hub_counts),
            "total": int(sum(hub_counts)),
        },
        "step_rows_mean": sum(k * p for k, p in shape.items()),
    }


# ---- sampled evaluation -------------------------------------------------------


def evaluate_ids(
    pa: PlanArrays,
    ids: dict[int, np.ndarray],
    step_of_row: np.ndarray,
    row_in_step: np.ndarray,
    n_steps: int,
    *,
    hidden_bytes: int = HIDDEN_BYTES,
    gbps: float = PCIE_GBPS,
    latency_us: float = PCIE_LATENCY_US,
    hub_us: float | None = None,
    expert_us: float | None = None,
) -> dict:
    """Metrics of a plan on explicit routing: ``ids[layer]`` is ``[rows, K]``."""
    n_dev = max(1, pa.n_devices)
    e_n = pa.num_experts
    t_max = int(row_in_step.max()) + 1
    max_sum = mean_sum = hub_sel = all_sel = 0.0
    hub_touch_sum = comm_us_sum = step_us = 0.0
    max_touched_steps = np.zeros(n_steps)
    for i, layer in enumerate(pa.layers):
        if pa.hub_only[i] or layer not in ids:
            continue
        sel = ids[layer]
        rows, k = sel.shape
        flat = sel.reshape(-1)
        row_idx = np.repeat(np.arange(rows), k)
        is_hub = pa.hub[i][flat]
        hub_sel += float(is_hub.sum())
        all_sel += float(flat.size)
        # distinct hub experts per step
        h = np.unique(step_of_row[row_idx[is_hub]] * e_n + flat[is_hub])
        hub_touched = np.bincount(h // e_n, minlength=n_steps).astype(np.float64)
        cold = ~is_hub & (pa.ncold[i][flat] > 0)
        cf, cr = flat[cold], row_idx[cold]
        c = pa.ncold[i][cf]
        slot = (row_in_step[cr] + layer) % c
        dev = pa.copies[i][cf, slot]
        step = step_of_row[cr]
        key = (step * n_dev + dev) * e_n + cf
        touched = np.bincount(np.unique(key) // e_n, minlength=n_steps * n_dev)
        touched = touched.reshape(n_steps, n_dev).astype(np.float64)
        rkey = (step * t_max + row_in_step[cr]) * n_dev + dev
        per_step_rows = np.zeros((n_steps, n_dev))
        ukeys = np.unique(rkey)
        np.add.at(per_step_rows, (ukeys // (t_max * n_dev), ukeys % n_dev), 1)
        comm_us = (
            2.0
            * (latency_us + per_step_rows.max(axis=1) * hidden_bytes / (gbps * 1e3))
            * (per_step_rows.max(axis=1) > 0)
        )
        max_touched = touched.max(axis=1)
        max_touched_steps += max_touched
        max_sum += float(max_touched.mean())
        mean_sum += float(touched.mean(axis=1).mean())
        hub_touch_sum += float(hub_touched.mean())
        comm_us_sum += float(comm_us.mean())
        if hub_us is not None and expert_us is not None:
            layer_us = np.maximum(
                hub_touched * hub_us, comm_us + max_touched * expert_us
            )
            step_us += float(layer_us.mean())
    out = {
        "steps": int(n_steps),
        "hub_hit": hub_sel / all_sel if all_sel else 0.0,
        "cold_max_touched_per_step": max_sum,
        "cold_mean_touched_per_step": mean_sum,
        "cold_imbalance": max_sum / mean_sum if mean_sum else 1.0,
        "cold_max_touched_p95_per_step": float(np.quantile(max_touched_steps, 0.95)),
        "hub_touched_per_step": hub_touch_sum,
        "comm_us_per_step": comm_us_sum,
    }
    if hub_us is not None and expert_us is not None:
        out["step_us"] = step_us
    return out


def draw_steps(
    shape: dict[int, float], steps: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """(step index of each row, row index within its step) for ``steps`` steps."""
    sizes = np.array(list(shape))
    probs = np.array(list(shape.values()))
    per_step = rng.choice(sizes, size=steps, p=probs / probs.sum())
    step_of_row = np.repeat(np.arange(steps), per_step)
    starts = np.cumsum(per_step) - per_step
    row_in_step = np.arange(per_step.sum()) - np.repeat(starts, per_step)
    return step_of_row, row_in_step


def sample_ids(
    record: StatsRecord,
    layers: list[int],
    rows: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    out = {}
    for layer in layers:
        if layer not in record.layers:
            continue
        q_dec, q_pre = _marginals(record, layer)
        q = q_dec if q_dec.sum() > 0 else q_pre
        out[layer] = systematic_sample(inclusion_probs(q, record.top_k), rows, rng)
    return out


def simulate(
    pa: PlanArrays,
    record: StatsRecord,
    *,
    steps: int = 2000,
    seed: int = 0,
    shape: dict[int, float] | None = None,
    **kwargs: float | None,
) -> dict:
    rng = np.random.default_rng(seed)
    shape = shape or step_shape(SolverConfig(0, 0, 0), record)
    step_of_row, row_in_step = draw_steps(shape, steps, rng)
    ids = sample_ids(record, pa.layers, int(step_of_row.shape[0]), rng)
    return evaluate_ids(pa, ids, step_of_row, row_in_step, steps, **kwargs)


def fallback_arrays(
    record: StatsRecord, topo: Topology, cfg: SolverConfig
) -> PlanArrays:
    layers = [li for li, s in record.layers.items() if s.kind == "target"]
    placements = fallback_plan(len(layers), record.num_experts, topo, cfg)
    remapped = {li: placements[i] for i, li in enumerate(layers)}
    return arrays_from_placements(remapped, topo, record.num_experts)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("plan")
    p.add_argument("--stats", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hub-us", type=float)
    p.add_argument("--expert-us", type=float)
    p.add_argument("--hidden-bytes", type=int, default=HIDDEN_BYTES)
    p.add_argument("--gbps", type=float, default=PCIE_GBPS)
    p.add_argument("--latency-us", type=float, default=PCIE_LATENCY_US)
    args = p.parse_args(argv)
    with open(args.plan, encoding="utf-8") as f:
        doc = json.load(f)
    record = load_stats(args.stats)
    pa = parse_plan(doc)
    result = simulate(
        pa,
        record,
        steps=args.steps,
        seed=args.seed,
        hub_us=args.hub_us,
        expert_us=args.expert_us,
        hidden_bytes=args.hidden_bytes,
        gbps=args.gbps,
        latency_us=args.latency_us,
    )
    result["analytic"] = analytic_metrics(
        pa, record, step_shape(SolverConfig(0, 0, 0), record)
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
