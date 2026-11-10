# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row independence of the routing model, tested on a top-k id trace.

The simulator and the solver treat the rows of a decode step as independent draws
from the layer's per-expert marginal. That fails if the rows of one step share
experts more than chance (the 3 rows of one MTP sequence are consecutive tokens;
rows of different sequences may share a prompt topic). The collector's id ring
(``VLLM_EXPERT_STATS_TRACE_LAYERS``, ``*.trace.npz``) records the real ids of the
newest decode steps of a few layers; this tool compares them with the same number
of rows drawn independently from the same marginal:

    python -m tools.rdna.hetero_moe.trace_check TRACE.npz [--group 3]
        [--plan PLAN.json --stats STATS.json] [--json out.json]

Per layer and rows-per-step band: mean distinct experts per step (real, independent
sample, ratio), and the mean number of shared experts per pair of rows inside a
step, split into rows of one MTP group (``--group`` consecutive rows) and rows of
different groups, against the independent expectation ``sum_e q_e^2``. With
``--plan`` the plan is evaluated on the real ids and on the independent sample of
the same step sizes (hub hit, spilled experts, cold maximum, step time of the
traced layers only).
"""

import argparse
import json
import sys

import numpy as np

from tools.rdna.hetero_moe.sampling import inclusion_probs, systematic_sample


def load_trace(path: str) -> dict:
    data = np.load(path)
    return {
        "names": [str(n) for n in data["names"]],
        "ids": data["ids"],
        "rows": data["rows"],
        "calls": data["calls"],
        "max_rows": int(data["max_rows"]),
    }


def layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def valid_steps(trace: dict, pos: int) -> list[np.ndarray]:
    """The newest decode steps of one layer, oldest first: ``[T_i, K]`` valid ids."""
    ids, rows, calls = trace["ids"][pos], trace["rows"][pos], int(trace["calls"][pos])
    n_slots = ids.shape[0]
    count = min(calls, n_slots)
    steps = []
    for i in range(count):
        slot = (calls - count + i) % n_slots
        block = ids[slot, : int(rows[slot])].astype(np.int64)
        block = block[(block >= 0).all(axis=1)]
        if len(block):
            steps.append(block)
    return steps


def marginal(steps: list[np.ndarray], num_experts: int, top_k: int) -> np.ndarray:
    """Inclusion probabilities (sum == top_k) of the rows in ``steps``."""
    flat = np.concatenate([s.reshape(-1) for s in steps])
    counts = np.bincount(flat, minlength=num_experts).astype(np.float64)
    return inclusion_probs(counts, top_k)


def _shared(block: np.ndarray, num_experts: int, group: int) -> tuple[float, float]:
    """Sum over row pairs of shared experts: (within MTP groups, across groups)."""
    t = len(block)
    onehot = np.zeros((t, num_experts))
    np.put_along_axis(onehot, block, 1.0, axis=1)
    total = (onehot.sum(0) ** 2 - onehot.sum(0)).sum() / 2.0
    within = 0.0
    for start in range(0, t, group):
        g = onehot[start : start + group].sum(0)
        within += (g * g - g).sum() / 2.0
    return within, total - within


def _pair_counts(t: int, group: int) -> tuple[int, int]:
    within = sum(
        (min(group, t - s) * (min(group, t - s) - 1)) // 2 for s in range(0, t, group)
    )
    return within, t * (t - 1) // 2 - within


def analyse_layer(
    steps: list[np.ndarray],
    num_experts: int,
    top_k: int,
    group: int,
    rng: np.random.Generator,
    min_steps: int,
) -> dict:
    q = marginal(steps, num_experts, top_k)
    iid_pair = float((q * q).sum())  # expected shared experts of two independent rows
    bands: dict[int, list[np.ndarray]] = {}
    for s in steps:
        bands.setdefault(len(s), []).append(s)
    out: dict = {"steps": len(steps), "bands": {}, "iid_shared_per_pair": iid_pair}
    for t, blocks in sorted(bands.items()):
        if len(blocks) < min_steps:
            continue
        sample = systematic_sample(q, len(blocks) * t, rng).reshape(len(blocks), t, -1)
        real_d = float(np.mean([len(np.unique(b)) for b in blocks]))
        iid_d = float(np.mean([len(np.unique(b)) for b in sample]))
        n_in, n_out = _pair_counts(t, group)
        real = np.array([_shared(b, num_experts, group) for b in blocks]).mean(axis=0)
        iid = np.array([_shared(b, num_experts, group) for b in sample]).mean(axis=0)
        out["bands"][str(t)] = {
            "steps": len(blocks),
            "distinct_real": real_d,
            "distinct_iid": iid_d,
            "distinct_ratio": real_d / iid_d,
            "shared_per_pair_same_group": float(real[0] / n_in) if n_in else None,
            "shared_per_pair_other_groups": float(real[1] / n_out) if n_out else None,
            "shared_per_pair_iid": float(iid.sum() / (n_in + n_out)),
        }
    return out


PLAN_KEYS = (
    "hub_hit",
    "step_us",
    "hub_touched_per_step",
    "hub_spilled_per_step",
    "hub_kept_per_layer_per_hub",
    "cold_max_touched_per_step",
    "cold_mean_touched_per_step",
    "cold_distinct_experts_per_step",
)


def plan_on_trace(
    doc: dict,
    per_layer: dict[int, list[np.ndarray]],
    record,
    seed: int,
    min_steps: int = 200,
    **kw,
) -> dict:
    """The plan on the real ids of each rows-per-step band against independent rows
    drawn from the same layer's marginal in the trace (``independent``: isolates the
    row correlation) and, with ``record``, from the solver's statistics
    (``independent_stats``: also carries the shift of the workload)."""
    from tools.rdna.hetero_moe.simulate import (
        evaluate_ids,
        parse_plan,
        sample_ids,
    )

    pa = parse_plan(doc)
    layers = [li for li in per_layer if li in pa.layers]
    sizes = [len(s) for s in per_layer[layers[0]]]
    # the padding mask is the same in every layer; keep layers that agree anyway
    layers = [li for li in layers if [len(s) for s in per_layer[li]] == sizes]
    rng = np.random.default_rng(seed)
    top_k = per_layer[layers[0]][0].shape[1]
    qs = {
        li: marginal(per_layer[li], pa.num_experts, top_k)  # the whole trace window
        for li in layers
    }
    out: dict = {"layers": layers, "bands": {}}
    for t in sorted(set(sizes)):
        picks = [i for i, n in enumerate(sizes) if n == t]
        if len(picks) < min_steps:
            continue
        n_steps = len(picks)
        step_of_row = np.repeat(np.arange(n_steps), t)
        row_in_step = np.tile(np.arange(t), n_steps)
        draws = {
            "real": {
                li: np.concatenate([per_layer[li][i] for i in picks]).astype(np.int32)
                for li in layers
            },
            "independent": {
                li: systematic_sample(qs[li], n_steps * t, rng) for li in layers
            },
        }
        if record is not None:
            draws["independent_stats"] = sample_ids(record, layers, n_steps * t, rng)
        band: dict = {"steps": n_steps}
        for label, ids in draws.items():
            m = evaluate_ids(pa, ids, step_of_row, row_in_step, n_steps, **kw)
            band[label] = {k: m[k] for k in PLAN_KEYS}
        band["ratio_real_to_independent"] = {
            k: band["real"][k] / band["independent"][k]
            for k in PLAN_KEYS
            if band["independent"][k]
        }
        out["bands"][str(t)] = band
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--group", type=int, default=3, help="rows per MTP group")
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--min-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", help="evaluate this plan on the real ids")
    ap.add_argument("--stats", help="stats JSON whose marginals draw the reference")
    ap.add_argument("--json", help="write the result here")
    args = ap.parse_args(argv)
    trace = load_trace(args.trace)
    top_k = trace["ids"].shape[-1]
    num_experts = args.experts
    if args.plan:
        with open(args.plan) as f:
            doc = json.load(f)
        num_experts = doc["model"]["num_experts"]
    rng = np.random.default_rng(args.seed)
    result: dict = {"layers": {}, "calls": [int(c) for c in trace["calls"]]}
    per_layer = {}
    for pos, name in enumerate(trace["names"]):
        steps = valid_steps(trace, pos)
        per_layer[layer_index(name)] = steps
        result["layers"][name] = analyse_layer(
            steps, num_experts, top_k, args.group, rng, args.min_steps
        )
    if args.plan:
        from tools.rdna.hetero_moe.stats_io import load_stats

        record = load_stats([args.stats]) if args.stats else None
        result["plan"] = plan_on_trace(
            doc, per_layer, record, args.seed, args.min_steps
        )
    print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    sys.exit(0)


if __name__ == "__main__":
    main()
