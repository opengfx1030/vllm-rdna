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
* PCIe traffic per layer: by default the ``hx`` protocol's broadcast (every row of
  the step to every V620 of the hub's group, ``2 x (latency_us + D x T x hidden_bytes
  / gbps)``; ``--exchange rows`` charges only the rows that have a pair on the
  busiest device);
* step time ``sum_l max(hub_fixed + hub_touched * hub_us,
  comm + expert_fixed + max_touched * expert_us)`` (the fixed terms and the
  per-expert GEMM times are HeteroDesign's constants, defaults in this file;
  the expert side is skipped in a layer where no row goes to a V620);

Dispatch matches the runtime rule: the hub computes a hub-resident expert, else
row ``r`` of layer ``l`` goes to cold copy ``(r + l) % n_cold_copies`` (owners in
listed order). Rows are drawn per step with the step-size distribution of the
statistics; per layer, a row's top-k set is a systematic sample with the
per-expert selection probabilities, independent between layers. The rows of a step
are correlated by default (``--row-model``, ``row_model.py``: the rows of an MTP
group and the sequences of a step share experts, fitted on a real id trace, T10
section 6.1); ``--row-model independent`` draws them independently. A routing trace
(``evaluate_ids``) keeps real co-activation.

CLI::

    python -m tools.rdna.hetero_moe.simulate PLAN.json --stats STATS [--steps 2000]
        [--seed 0] [--compare-fallback] [--hub-us X --expert-us Y]
        [--hidden-bytes 5120 --gbps 24.5 --latency-us 6.2]
        [--exchange broadcast|rows] [--decode-dispatch row|expert|primary]
        [--row-model flashnext-mtp2|independent|PATH] [--rows W]
"""

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from . import row_model as row_models
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
# Cost model (HeteroDesign constants, [INFERENCE] until the W7800 microbenchmark):
PCIE_GBPS = 25.0  # intra-switch per-pair P2P
PCIE_LATENCY_US = 13.3  # per direction: one-way link 6.3 + dispatch 3 + wake 3 + ...
HUB_US = 3.7  # per touched expert on a W7800: 2,534,400 B / (0.80 x 864 GB/s)
EXPERT_US = 6.7  # per touched expert on a V620: 2,534,400 B / 380 GB/s


@dataclass
class PlanArrays:
    layers: list[int]
    num_experts: int
    n_hubs: int  # hubs per island
    n_devices: int  # expert devices the arrays index (island 0, or all islands)
    hub: np.ndarray  # bool [L, E]
    copies: np.ndarray  # int [L, E, D] expert-device idx (0..n_devices), -1 padded
    ncold: np.ndarray  # int [L, E]
    hub_only: list[bool]
    kinds: list[str]
    hub_group: np.ndarray | None = None  # int [L, E]: island whose hub holds it
    n_hub_groups: int = 1  # islands in ``partition`` mode, else 1


def _build_arrays(
    rows: list[list[list[int]]],
    layers: list[int],
    num_experts: int,
    groups: list[list[int]],
    is_hub: dict[int, bool],
    partition: bool,
    hub_only: list[bool],
    kinds: list[str],
) -> PlanArrays:
    """Arrays from owner rows (device indices) and the island device groups.

    ``replicate``: only island 0 is modelled (every island is an identical copy).
    ``partition``: all islands; expert devices are numbered over all of them."""
    used = groups if partition else groups[:1]
    local: dict[int, int] = {}
    hub_group: dict[int, int] = {}
    for k, group in enumerate(used):
        for d in group:
            if is_hub[d]:
                hub_group[d] = k
            else:
                local[d] = len(local)
    n_layers = len(layers)
    hub = np.zeros((n_layers, num_experts), dtype=bool)
    group_of = np.zeros((n_layers, num_experts), dtype=np.int64)
    copies = -np.ones((n_layers, num_experts, max(1, len(local))), dtype=np.int64)
    ncold = np.zeros((n_layers, num_experts), dtype=np.int64)
    for i, layer_rows in enumerate(rows):
        for e, owners in enumerate(layer_rows):
            cold = [local[d] for d in owners if d in local]
            held = [hub_group[d] for d in owners if d in hub_group]
            if held:
                hub[i, e] = True
                group_of[i, e] = held[0]
            copies[i, e, : len(cold)] = cold
            ncold[i, e] = len(cold)
    n_hubs = sum(1 for d in used[0] if is_hub[d])
    return PlanArrays(
        layers,
        num_experts,
        n_hubs,
        len(local),
        hub,
        copies,
        ncold,
        hub_only,
        kinds,
        group_of,
        len(used) if partition else 1,
    )


def parse_plan(doc: dict) -> PlanArrays:
    """Arrays of a plan document (island 0, or every island in ``partition``)."""
    devices = doc["devices"]
    layers = doc["layers"]
    return _build_arrays(
        [layer["owners"] for layer in layers],
        [x["layer"] for x in layers],
        doc["model"]["num_experts"],
        doc["islands"],
        {i: d["role"] == "hub" for i, d in enumerate(devices)},
        doc.get("island_mode", "replicate") == "partition",
        [bool(x.get("hub_only")) for x in layers],
        [x.get("kind", "target") for x in layers],
    )


def arrays_from_placements(
    placements: dict[int, LayerPlacement], topo: Topology, num_experts: int
) -> PlanArrays:
    """Same arrays straight from solver output (no JSON round trip)."""
    layers = sorted(placements)
    groups = topo.island_groups()
    is_hub = {d: p < topo.n_hubs for g in groups for p, d in enumerate(g)}
    partition = any(placements[li].island_of is not None for li in layers)
    return _build_arrays(
        [owners_lists(placements[li], topo, num_experts) for li in layers],
        layers,
        num_experts,
        groups,
        is_hub,
        partition,
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
    pair_load_ratio_primary = []
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
        # prefill pair load per device (compute balance): rows spread over the cold
        # copies (the plan format's rule) and, for the replica benefit, primary only
        load = np.zeros(n_dev)
        load_primary = np.zeros(n_dev)
        for e in np.flatnonzero(cold):
            c = int(pa.ncold[i, e])
            load_primary[pa.copies[i, e, 0]] += q_pre[e]
            for k in range(c):
                load[pa.copies[i, e, k]] += q_pre[e] / c
        if load.mean() > 0:
            pair_load_ratio.append(float(load.max() / load.mean()))
            pair_load_ratio_primary.append(
                float(load_primary.max() / load_primary.mean())
            )
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
        "prefill_pair_load_imbalance_primary_only": (
            float(np.mean(pair_load_ratio_primary)) if pair_load_ratio_primary else 1.0
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


def _layer_cost(
    hub_t: np.ndarray,
    cold: np.ndarray,
    comm_us: np.ndarray,
    base_comm: float | np.ndarray,
    costs: tuple[float, float, float, float],
    spill: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step layer time ``max(hub_fixed + h * hub_us, comm + expert_fixed + c *
    expert_us)`` with the runtime's spill rule, and without it.

    ``hub_t`` is ``[steps, hub groups]`` (distinct hub-resident experts touched),
    ``cold`` is ``[steps, hub groups, expert devices per group]``. Spill: a touched
    hub-resident expert may instead run on its V620 copy (``cold_complete``); each
    hub spills ``s`` experts, spread evenly over its V620s, choosing the ``s`` that
    minimise the layer time. Returns ``(layer us with spill, layer us without,
    experts spilled per hub per step ``[steps, hub groups]``); the layer time is the
    busiest hub group's. ``base_comm`` is the exchange a spill needs when no row
    went to a V620 (a scalar or ``[steps]``).
    """
    hub_us, expert_us, hub_fixed, expert_fixed = costs
    nd = cold.shape[2]
    mx, tot = cold.max(axis=2), cold.sum(axis=2)

    bc = np.asarray(base_comm, dtype=np.float64)
    bc = bc[:, None] if bc.ndim else bc

    def cost(s: np.ndarray) -> np.ndarray:
        crit = np.maximum(mx, np.ceil((tot + s) / nd))
        comm = np.where(s > 0, np.maximum(comm_us[:, None], bc), comm_us[:, None])
        return np.maximum(
            hub_fixed + (hub_t - s) * hub_us,
            comm + expert_fixed * (comm > 0) + crit * expert_us,
        )

    base = cost(np.zeros_like(hub_t))
    if not spill:
        return base.max(axis=1), base.max(axis=1), np.zeros(hub_t.shape)
    s_max = int(hub_t.max())
    stack = np.stack([cost(np.minimum(float(s), hub_t)) for s in range(s_max + 1)])
    best = stack.min(axis=0)
    pick = np.minimum(stack.argmin(axis=0), hub_t)  # first minimiser, capped
    return best.max(axis=1), base.max(axis=1), pick


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
    hub_us: float = HUB_US,
    expert_us: float = EXPERT_US,
    hub_fixed_us: float = 0.0,
    expert_fixed_us: float = 0.0,
    spill: bool = True,
    exchange: str = "broadcast",
    dispatch: str = "row",
) -> dict:
    """Metrics of a plan on explicit routing: ``ids[layer]`` is ``[rows, K]``.

    ``exchange``: ``broadcast`` charges the ``hx`` protocol (all T rows of the step
    are pushed to every V620 of a hub group through the hub's link, T rows are
    pulled back: ``2 x (latency + D x T x hidden / gbps)``, the layer is skipped
    when no row has a pair on a V620); ``rows`` charges only the rows that have a
    pair on the busiest device. ``dispatch`` picks the cold copy of a (row, expert):
    ``row`` = ``(row + layer) % n_cold`` (the plan format's rule), ``expert`` =
    ``(expert + layer) % n_cold`` (no expert is streamed on two copies in a step),
    ``primary`` = the first listed cold copy.
    """
    if exchange not in ("broadcast", "rows"):
        raise ValueError(f"exchange must be broadcast or rows, not {exchange!r}")
    if dispatch not in ("row", "expert", "primary"):
        raise ValueError(f"dispatch must be row, expert or primary, not {dispatch!r}")
    n_dev = max(1, pa.n_devices)
    e_n = pa.num_experts
    t_max = int(row_in_step.max()) + 1
    max_sum = mean_sum = hub_sel = all_sel = 0.0
    hub_touch_sum = comm_us_sum = step_us = step_us_no_spill = spilled_sum = 0.0
    kept_sum = distinct_sum = touches_sum = hub_all_sum = 0.0
    n_eval = 0
    base_comm: float | np.ndarray = 2.0 * (latency_us + hidden_bytes / (gbps * 1e3))
    g_total = pa.n_hub_groups
    servers = max(1, n_dev // g_total)
    rows_per_step = np.bincount(step_of_row, minlength=n_steps).astype(np.float64)
    bcast = 2.0 * (latency_us + servers * rows_per_step * hidden_bytes / (gbps * 1e3))
    if exchange == "broadcast":
        base_comm = bcast
    # spill needs a V620 copy of every hub-resident expert (cold_complete plans)
    spill = spill and all(
        bool((pa.ncold[i][pa.hub[i]] > 0).all())
        for i in range(len(pa.layers))
        if not pa.hub_only[i]
    )
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
        # distinct hub experts per step, per hub group; the busiest hub is the cost
        g_n = pa.n_hub_groups
        hg = (
            pa.hub_group[i][flat[is_hub]]
            if pa.hub_group is not None
            else np.zeros(int(is_hub.sum()), dtype=np.int64)
        )
        h = np.unique((step_of_row[row_idx[is_hub]] * g_n + hg) * e_n + flat[is_hub])
        hub_g = (
            np.bincount(h // e_n, minlength=n_steps * g_n)
            .reshape(n_steps, g_n)
            .astype(np.float64)
        )
        cold = ~is_hub & (pa.ncold[i][flat] > 0)
        cf, cr = flat[cold], row_idx[cold]
        c = pa.ncold[i][cf]
        if dispatch == "row":
            slot = (row_in_step[cr] + layer) % c
        elif dispatch == "expert":
            slot = (cf + layer) % c
        else:
            slot = np.zeros_like(c)
        dev = pa.copies[i][cf, slot]
        step = step_of_row[cr]
        key = (step * n_dev + dev) * e_n + cf
        touched = np.bincount(np.unique(key) // e_n, minlength=n_steps * n_dev)
        touched = touched.reshape(n_steps, n_dev).astype(np.float64)
        distinct = np.bincount(
            np.unique(step * e_n + cf) // e_n, minlength=n_steps
        ).astype(np.float64)
        rkey = (step * t_max + row_in_step[cr]) * n_dev + dev
        per_step_rows = np.zeros((n_steps, n_dev))
        ukeys = np.unique(rkey)
        np.add.at(per_step_rows, (ukeys // (t_max * n_dev), ukeys % n_dev), 1)
        rows_busiest = per_step_rows.max(axis=1)
        if exchange == "broadcast":
            comm_us = bcast * (rows_busiest > 0)
        else:
            comm_us = (
                2.0 * (latency_us + rows_busiest * hidden_bytes / (gbps * 1e3))
            ) * (rows_busiest > 0)
        max_touched = touched.max(axis=1)
        max_touched_steps += max_touched
        max_sum += float(max_touched.mean())
        mean_sum += float(touched.mean(axis=1).mean())
        hub_touch_sum += float(hub_g.max(axis=1).mean())
        hub_all_sum += float(hub_g.sum(axis=1).mean())
        comm_us_sum += float(comm_us.mean())
        layer_us, layer_us0, spilled = _layer_cost(
            hub_g,
            touched.reshape(n_steps, g_n, n_dev // g_n),
            comm_us,
            base_comm,
            (hub_us, expert_us, hub_fixed_us, expert_fixed_us),
            spill,
        )
        step_us += float(layer_us.mean())
        step_us_no_spill += float(layer_us0.mean())
        spilled_sum += float(spilled.sum(axis=1).mean())
        kept_sum += float((hub_g - spilled).mean())
        distinct_sum += float(distinct.mean())
        touches_sum += float(touched.sum(axis=1).mean())
        n_eval += 1
    out = {
        "steps": int(n_steps),
        "hub_hit": hub_sel / all_sel if all_sel else 0.0,
        "cold_max_touched_per_step": max_sum,
        "cold_mean_touched_per_step": mean_sum,
        "cold_imbalance": max_sum / mean_sum if mean_sum else 1.0,
        "cold_max_touched_p95_per_step": float(np.quantile(max_touched_steps, 0.95)),
        "hub_touched_per_step": hub_touch_sum,
        "hub_touched_all_per_step": hub_all_sum,
        "n_eval_layers": n_eval,
        "comm_us_per_step": comm_us_sum,
    }
    out["step_us"] = step_us if spill else step_us_no_spill
    out["step_us_no_spill"] = step_us_no_spill
    out["hub_spilled_per_step"] = spilled_sum
    out["hub_kept_per_layer_per_hub"] = kept_sum / n_eval if n_eval else 0.0
    out["cold_distinct_experts_per_step"] = distinct_sum
    out["cold_touches_per_step"] = touches_sum
    out["cold_replica_touch_ratio"] = (
        touches_sum / distinct_sum if distinct_sum else 1.0
    )
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
    model: row_models.RowModel | None = None,
    step_of_row: np.ndarray | None = None,
    row_in_step: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Routing ids of ``rows`` rows per layer from the statistics' decode marginals:
    independent rows, or with ``model`` (and the step structure) the correlated rows
    of ``sampling.correlated_sample``."""
    out = {}
    for layer in layers:
        if layer not in record.layers:
            continue
        q_dec, q_pre = _marginals(record, layer)
        q = inclusion_probs(q_dec if q_dec.sum() > 0 else q_pre, record.top_k)
        if model is not None and model.layers and step_of_row is not None:
            assert row_in_step is not None
            out[layer] = model.sample(layer, q, step_of_row, row_in_step, rng)
        else:
            out[layer] = systematic_sample(q, rows, rng)
    return out


def simulate(
    pa: PlanArrays,
    record: StatsRecord,
    *,
    steps: int = 2000,
    seed: int = 0,
    shape: dict[int, float] | None = None,
    row_model: str | row_models.RowModel | None = None,
    **kwargs: float | str | None,
) -> dict:
    """Simulated decode steps. ``row_model``: the correlated row model (a name or
    path for ``row_model.load``, default the shipped fit of the flashnext-mtp2 trace)
    or ``independent``."""
    model = (
        row_model
        if isinstance(row_model, row_models.RowModel)
        else row_models.load(row_model)
    )
    rng = np.random.default_rng(seed)
    shape = shape or step_shape(SolverConfig(0, 0, 0), record)
    step_of_row, row_in_step = draw_steps(shape, steps, rng)
    ids = sample_ids(
        record,
        pa.layers,
        int(step_of_row.shape[0]),
        rng,
        model,
        step_of_row,
        row_in_step,
    )
    return evaluate_ids(pa, ids, step_of_row, row_in_step, steps, **kwargs)


def fallback_arrays(
    record: StatsRecord, topo: Topology, cfg: SolverConfig
) -> PlanArrays:
    layers = [li for li, s in record.layers.items() if s.kind == "target"]
    placements = fallback_plan(len(layers), record.num_experts, topo, cfg)
    remapped = {li: placements[i] for i, li in enumerate(layers)}
    return arrays_from_placements(remapped, topo, record.num_experts)


def add_cost_args(p: argparse._ActionsContainer) -> None:
    """Cost-model flags shared by ``simulate`` and ``solve`` (microseconds)."""
    p.add_argument("--hub-us", type=float, default=HUB_US, help="per touched expert")
    p.add_argument(
        "--expert-us", type=float, default=EXPERT_US, help="per touched expert"
    )
    p.add_argument("--hub-fixed-us", type=float, default=0.0, help="per layer, hub")
    p.add_argument(
        "--expert-fixed-us", type=float, default=0.0, help="per layer, V620 side"
    )
    p.add_argument("--hidden-bytes", type=int, default=HIDDEN_BYTES)
    p.add_argument("--gbps", type=float, default=PCIE_GBPS)
    p.add_argument(
        "--latency-us",
        type=float,
        default=PCIE_LATENCY_US,
        help="per direction (applied twice per layer)",
    )
    p.add_argument(
        "--exchange",
        choices=("broadcast", "rows"),
        default="broadcast",
        help="broadcast: the hx protocol (every row to every V620 of the hub's "
        "group, 2 x (latency + D x T x hidden / gbps)); rows: only the rows with a "
        "pair on the busiest device",
    )
    p.add_argument(
        "--decode-dispatch",
        choices=("row", "expert", "primary"),
        default="row",
        help="cold copy of a (row, expert): row = (row + layer) %% n_cold (plan "
        "format), expert = (expert + layer) %% n_cold, primary = first cold copy",
    )
    p.add_argument(
        "--row-model",
        default=row_models.DEFAULT_NAME,
        help="rows of a decode step: 'flashnext-mtp2' (default, correlated rows "
        "fitted on the T10 id trace, T10 section 6), 'independent', or the path of a "
        "row-model JSON (tools.rdna.hetero_moe.row_model fit)",
    )


def cost_kwargs(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "hub_us": args.hub_us,
        "expert_us": args.expert_us,
        "hub_fixed_us": args.hub_fixed_us,
        "expert_fixed_us": args.expert_fixed_us,
        "hidden_bytes": args.hidden_bytes,
        "gbps": args.gbps,
        "latency_us": args.latency_us,
        "exchange": args.exchange,
        "dispatch": args.decode_dispatch,
        "row_model": args.row_model,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("plan")
    p.add_argument("--stats", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rows",
        type=int,
        help="fixed rows per step (every step this wide, e.g. a capture width) "
        "instead of the statistics' step-size distribution",
    )
    add_cost_args(p)
    args = p.parse_args(argv)
    with open(args.plan, encoding="utf-8") as f:
        doc = json.load(f)
    record = load_stats(args.stats)
    pa = parse_plan(doc)
    shape = {args.rows: 1.0} if args.rows else step_shape(SolverConfig(0, 0, 0), record)
    result = simulate(
        pa,
        record,
        steps=args.steps,
        seed=args.seed,
        shape=shape,
        **cost_kwargs(args),
    )
    result["analytic"] = analytic_metrics(pa, record, shape)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
