# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static expert-to-rank placement from MoE routing histograms.

usage: moe_placement.py HIST.jsonl [--ranks 4] [--bsm 4] [--min-tokens 512]
                       [--folds 4] [--objective max|mean] [--mean-moe-ms 206.2]
                       [--seed 0] [--out placement.json] [--json report.json]
                       [--selftest]

HIST.jsonl is the VLLM_RDNA_MOE_HIST_DIR capture: a meta line then one JSON
object per MoE layer call with >= VLLM_RDNA_MOE_HIST_MIN_TOKENS tokens. The
cost proxy of a layer call is max over ranks of sum_{e in rank} ceil(count_e /
bsm) tiles; the imbalance is that max minus the rank mean. A placement is a
partition of the num_experts experts into ep_size equal ranks. Methods:
linear (rank = e // (E/R), the current placement), lpt (greedy longest
processing time with capacity E/R) and lpt+swap (bounded greedy pairwise swaps
that improve the train objective). Folds are contiguous blocks of the sorted
step list so consecutive steps of one prompt stay together. --out writes the
placement trained on all steps for the best held-out method.
"""

import argparse
import json
import os
import random
import statistics
import sys

try:
    import numpy as np
except ImportError:
    np = None


def load_hist(path, min_tokens):
    """Parse the JSONL capture. Returns (meta, records_by_layer)."""
    meta = None
    records_by_layer = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Tolerate a truncated last line.
            rest = [ln.strip() for ln in lines[i + 1 :] if ln.strip()]
            if not rest:
                break
            print(f"warning: skipping malformed line {i + 1}", file=sys.stderr)
            continue
        if "meta" in obj:
            meta = obj["meta"]
            continue
        if "counts" not in obj:
            print(f"warning: skipping line {i + 1} without counts", file=sys.stderr)
            continue
        if meta is not None:
            expected = meta.get("num_experts")
            if expected is not None and len(obj["counts"]) != expected:
                print(
                    f"warning: line {i + 1} has {len(obj['counts'])} counts, "
                    f"expected {expected}",
                    file=sys.stderr,
                )
                continue
        if obj.get("tokens", 0) < min_tokens:
            continue
        layer = obj.get("layer_idx")
        if layer is None:
            print(f"warning: skipping line {i + 1} without layer_idx", file=sys.stderr)
            continue
        records_by_layer.setdefault(layer, []).append(obj)
    if meta is None:
        raise ValueError(f"{path}: no meta line found")
    return meta, records_by_layer


def tiles(counts, bsm):
    """ceil(count / bsm) per element."""
    if np is not None:
        arr = np.asarray(counts, dtype=np.int64)
        return ((arr + bsm - 1) // bsm).tolist()
    return [(c + bsm - 1) // bsm for c in counts]


def build_T(records, num_experts, bsm):
    """Tiles matrix [len(records), num_experts]."""
    if np is not None:
        T = np.zeros((len(records), num_experts), dtype=np.int64)
        for i, rec in enumerate(records):
            counts = np.asarray(rec["counts"], dtype=np.int64)
            T[i] = (counts + bsm - 1) // bsm
        return T
    T = []
    for rec in records:
        T.append([(c + bsm - 1) // bsm for c in rec["counts"]])
    return T


def linear_placement(num_experts, ranks):
    """The baseline contiguous placement."""
    cap = num_experts // ranks
    return [list(range(r * cap, (r + 1) * cap)) for r in range(ranks)]


def build_lpt(T, num_experts, ranks):
    """Greedy longest-processing-time placement with capacity E/R."""
    if np is not None and isinstance(T, np.ndarray):
        load = T.sum(axis=0).tolist()
    else:
        load = [sum(row[e] for row in T) for e in range(num_experts)]
    order = sorted(range(num_experts), key=lambda e: -load[e])
    capacity = num_experts // ranks
    placement = [[] for _ in range(ranks)]
    rank_load = [0] * ranks
    rank_size = [0] * ranks
    for e in order:
        best_r = -1
        best_load = float("inf")
        for r in range(ranks):
            if rank_size[r] < capacity and rank_load[r] < best_load:
                best_load = rank_load[r]
                best_r = r
        placement[best_r].append(e)
        rank_load[best_r] += load[e]
        rank_size[best_r] += 1
    return placement


def _swap_improve_np(T, placement, objective, max_passes):
    S, _ = T.shape
    R = len(placement)
    loads = np.zeros((S, R), dtype=np.int64)
    for r, experts in enumerate(placement):
        loads[:, r] = T[:, experts].sum(axis=1)
    if objective == "max":
        current_obj = loads.max(axis=1).sum()
    else:
        means = loads.mean(axis=0)
        current_obj = means.max() - means.min()
    for _ in range(max_passes):
        # A swap between two ranks that are neither the bottleneck of any step
        # cannot lower the max objective: only consider pairs touching a
        # bottleneck rank.
        if objective == "max":
            bottleneck = set(int(r) for r in loads.argmax(axis=1))
        else:
            bottleneck = {int(means.argmax()), int(means.argmin())}
        best_swap = None
        best_obj = current_obj
        for r1 in range(R):
            for r2 in range(r1 + 1, R):
                if r1 not in bottleneck and r2 not in bottleneck:
                    continue
                T1 = T[:, placement[r1]]
                T2 = T[:, placement[r2]]
                if objective == "max":
                    other = [r for r in range(R) if r != r1 and r != r2]
                    other_max = (
                        loads[:, other].max(axis=1)
                        if other
                        else np.zeros(S, dtype=np.int64)
                    )
                    L1 = loads[:, r1][:, None] - T1
                    L2 = loads[:, r2][:, None] + T1
                    new_r1 = L1[:, :, None] + T2[:, None, :]
                    new_r2 = L2[:, :, None] - T2[:, None, :]
                    cand = np.maximum(
                        np.maximum(new_r1, new_r2), other_max[:, None, None]
                    )
                    obj = cand.sum(axis=0)
                    idx = np.unravel_index(np.argmin(obj), obj.shape)
                    min_obj = obj[idx]
                    if min_obj < best_obj - 1e-9:
                        best_obj = min_obj
                        best_swap = (r1, r2, idx[0], idx[1])
                else:
                    mean_T1 = T1.mean(axis=0)
                    mean_T2 = T2.mean(axis=0)
                    m1 = means[r1]
                    m2 = means[r2]
                    other = [r for r in range(R) if r != r1 and r != r2]
                    other_max = means[other].max() if other else -np.inf
                    other_min = means[other].min() if other else np.inf
                    new_m1 = m1 - mean_T1[:, None] + mean_T2[None, :]
                    new_m2 = m2 - mean_T2[None, :] + mean_T1[:, None]
                    new_max = np.maximum(np.maximum(new_m1, new_m2), other_max)
                    new_min = np.minimum(np.minimum(new_m1, new_m2), other_min)
                    spread = new_max - new_min
                    idx = np.unravel_index(np.argmin(spread), spread.shape)
                    min_obj = spread[idx]
                    if min_obj < best_obj - 1e-9:
                        best_obj = min_obj
                        best_swap = (r1, r2, idx[0], idx[1])
        if best_swap is None:
            break
        r1, r2, i1, i2 = best_swap
        e1 = placement[r1][i1]
        e2 = placement[r2][i2]
        placement[r1][i1], placement[r2][i2] = e2, e1
        loads[:, r1] += T[:, e2] - T[:, e1]
        loads[:, r2] += T[:, e1] - T[:, e2]
        if objective == "max":
            current_obj = loads.max(axis=1).sum()
        else:
            means = loads.mean(axis=0)
            current_obj = means.max() - means.min()
    return placement


def _swap_improve_py(T, placement, objective, max_passes):
    S = len(T)
    R = len(placement)
    E = len(T[0]) if S > 0 else 0
    loads = [[0] * R for _ in range(S)]
    for s in range(S):
        for r, experts in enumerate(placement):
            loads[s][r] = sum(T[s][e] for e in experts)
    if objective == "max":
        current_obj = sum(max(row) for row in loads)
    else:
        means = [sum(loads[s][r] for s in range(S)) / S for r in range(R)]
        current_obj = max(means) - min(means)
        sum_T = [sum(T[s][e] for s in range(S)) for e in range(E)]
    for _ in range(max_passes):
        improved = False
        if objective == "max":
            for r1 in range(R):
                for r2 in range(r1 + 1, R):
                    other = [r for r in range(R) if r != r1 and r != r2]
                    for i1, e1 in enumerate(placement[r1]):
                        for i2, e2 in enumerate(placement[r2]):
                            new_obj = 0.0
                            for s in range(S):
                                new_r1 = loads[s][r1] - T[s][e1] + T[s][e2]
                                new_r2 = loads[s][r2] - T[s][e2] + T[s][e1]
                                om = max(loads[s][r] for r in other) if other else 0
                                new_obj += max(new_r1, new_r2, om)
                            if new_obj < current_obj - 1e-9:
                                placement[r1][i1], placement[r2][i2] = e2, e1
                                for s in range(S):
                                    loads[s][r1] += T[s][e2] - T[s][e1]
                                    loads[s][r2] += T[s][e1] - T[s][e2]
                                current_obj = new_obj
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        else:
            for r1 in range(R):
                for r2 in range(r1 + 1, R):
                    other = [r for r in range(R) if r != r1 and r != r2]
                    other_max = max(means[r] for r in other) if other else -float("inf")
                    other_min = min(means[r] for r in other) if other else float("inf")
                    for i1, e1 in enumerate(placement[r1]):
                        for i2, e2 in enumerate(placement[r2]):
                            delta = (sum_T[e2] - sum_T[e1]) / S
                            new_m1 = means[r1] + delta
                            new_m2 = means[r2] - delta
                            new_max = max(new_m1, new_m2, other_max)
                            new_min = min(new_m1, new_m2, other_min)
                            new_spread = new_max - new_min
                            if new_spread < current_obj - 1e-9:
                                placement[r1][i1], placement[r2][i2] = e2, e1
                                means[r1] = new_m1
                                means[r2] = new_m2
                                current_obj = new_spread
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return placement


def swap_improve(T, placement, objective, max_passes=3):
    """Greedy pairwise expert swaps that reduce the train objective."""
    if np is not None and isinstance(T, np.ndarray):
        return _swap_improve_np(T, placement, objective, max_passes)
    return _swap_improve_py(T, placement, objective, max_passes)


def evaluate(records_by_layer, placement_by_layer, bsm):
    """Return (fraction, median per-layer max/mean ratio)."""
    total_imbalance = 0.0
    total_mean = 0.0
    layer_medians = []
    for layer, records in records_by_layer.items():
        if not records:
            continue
        placement = placement_by_layer[layer]
        num_experts = len(records[0]["counts"])
        T = build_T(records, num_experts, bsm)
        S = len(records)
        R = len(placement)
        if np is not None and isinstance(T, np.ndarray):
            loads = np.zeros((S, R), dtype=np.int64)
            for r, experts in enumerate(placement):
                loads[:, r] = T[:, experts].sum(axis=1)
            mx = loads.max(axis=1)
            mn = loads.mean(axis=1)
            total_imbalance += float((mx - mn).sum())
            total_mean += float(mn.sum())
            ratios = []
            for s in range(S):
                if mn[s] > 0:
                    ratios.append(float(mx[s] / mn[s]))
            if ratios:
                layer_medians.append(float(statistics.median(ratios)))
        else:
            layer_ratios = []
            for rec in records:
                t = tiles(rec["counts"], bsm)
                loads = [sum(t[e] for e in experts) for experts in placement]
                mx = max(loads)
                mn = sum(loads) / len(loads)
                total_imbalance += mx - mn
                total_mean += mn
                if mn > 0:
                    layer_ratios.append(mx / mn)
            if layer_ratios:
                layer_medians.append(statistics.median(layer_ratios))
    fraction = total_imbalance / total_mean if total_mean > 0 else 0.0
    median_ratio = statistics.median(layer_medians) if layer_medians else 1.0
    return fraction, median_ratio


def validate_placement(placement_by_layer, num_experts, ranks):
    """Assert each expert appears once and every rank has E/R experts."""
    capacity = num_experts // ranks
    for layer, placement in placement_by_layer.items():
        seen = set()
        for rank_experts in placement:
            if len(rank_experts) != capacity:
                raise AssertionError(
                    f"layer {layer}: rank has {len(rank_experts)} experts, "
                    f"expected {capacity}"
                )
            for e in rank_experts:
                if e in seen:
                    raise AssertionError(
                        f"layer {layer}: expert {e} appears more than once"
                    )
                seen.add(e)
        if len(seen) != num_experts:
            raise AssertionError(
                f"layer {layer}: {len(seen)} experts, expected {num_experts}"
            )


def _split_folds(steps, num_folds):
    """Split sorted steps into contiguous blocks."""
    n = len(steps)
    base = n // num_folds
    rem = n % num_folds
    blocks = []
    start = 0
    for i in range(num_folds):
        size = base + (1 if i < rem else 0)
        blocks.append(steps[start : start + size])
        start += size
    return blocks


def analyze(hist_path, ranks, bsm, min_tokens, folds, objective, mean_moe_ms, seed):
    """Cross-validate placements and train the final one on all steps."""
    meta, records_by_layer = load_hist(hist_path, min_tokens)
    num_experts = meta.get("num_experts")
    if num_experts is None:
        raise ValueError("meta line missing num_experts")
    if num_experts % ranks != 0:
        raise ValueError(f"num_experts {num_experts} not divisible by ranks {ranks}")

    all_steps = set()
    for recs in records_by_layer.values():
        for rec in recs:
            all_steps.add(rec["step"])
    all_steps = sorted(all_steps)
    if len(all_steps) < 2:
        raise ValueError("need at least 2 steps for cross-validation")

    num_folds = min(folds, len(all_steps))
    if num_folds < 2:
        raise ValueError("need at least 2 folds for cross-validation")
    fold_blocks = _split_folds(all_steps, num_folds)

    layer_step_records = {}
    for layer, recs in records_by_layer.items():
        d = {}
        for rec in recs:
            d[rec["step"]] = rec
        layer_step_records[layer] = d

    methods = ["linear", "lpt", "lpt+swap"]
    results = {
        m: {
            "folds": [],
            "in_sample": {"fractions": [], "ratios": []},
            "held_out": {"fractions": [], "ratios": []},
        }
        for m in methods
    }

    for fold_idx, test_steps in enumerate(fold_blocks):
        train_steps = [s for s in all_steps if s not in test_steps]
        train_records_by_layer = {}
        test_records_by_layer = {}
        for layer, d in layer_step_records.items():
            train_records_by_layer[layer] = [d[s] for s in train_steps if s in d]
            test_records_by_layer[layer] = [d[s] for s in test_steps if s in d]

        placements = {}
        placements["linear"] = linear_placement(num_experts, ranks)
        placements["lpt"] = {}
        placements["lpt+swap"] = {}
        for layer, recs in train_records_by_layer.items():
            if not recs:
                placements["lpt"][layer] = placements["linear"]
                placements["lpt+swap"][layer] = placements["linear"]
                continue
            T = build_T(recs, num_experts, bsm)
            p_lpt = build_lpt(T, num_experts, ranks)
            placements["lpt"][layer] = p_lpt
            placements["lpt+swap"][layer] = swap_improve(
                T, [list(x) for x in p_lpt], objective
            )

        for method in methods:
            if method == "linear":
                p_by_layer = {
                    layer: placements["linear"] for layer in train_records_by_layer
                }
            else:
                p_by_layer = placements[method]
            in_frac, in_ratio = evaluate(train_records_by_layer, p_by_layer, bsm)
            out_frac, out_ratio = evaluate(test_records_by_layer, p_by_layer, bsm)
            results[method]["in_sample"]["fractions"].append(in_frac)
            results[method]["in_sample"]["ratios"].append(in_ratio)
            results[method]["held_out"]["fractions"].append(out_frac)
            results[method]["held_out"]["ratios"].append(out_ratio)
            results[method]["folds"].append(
                {
                    "fold": fold_idx,
                    "train_steps": train_steps,
                    "test_steps": test_steps,
                    "in_sample_fraction": in_frac,
                    "in_sample_median_ratio": in_ratio,
                    "held_out_fraction": out_frac,
                    "held_out_median_ratio": out_ratio,
                }
            )

    linear_in = results["linear"]["in_sample"]["fractions"]
    linear_out = results["linear"]["held_out"]["fractions"]
    linear_in_mean = sum(linear_in) / len(linear_in) if linear_in else 0.0
    linear_out_mean = sum(linear_out) / len(linear_out) if linear_out else 0.0

    for method in methods:
        in_fracs = results[method]["in_sample"]["fractions"]
        out_fracs = results[method]["held_out"]["fractions"]
        in_ratios = results[method]["in_sample"]["ratios"]
        out_ratios = results[method]["held_out"]["ratios"]
        in_mean = sum(in_fracs) / len(in_fracs) if in_fracs else 0.0
        out_mean = sum(out_fracs) / len(out_fracs) if out_fracs else 0.0
        in_ratio_mean = sum(in_ratios) / len(in_ratios) if in_ratios else 1.0
        out_ratio_mean = sum(out_ratios) / len(out_ratios) if out_ratios else 1.0
        saving_in = (linear_in_mean - in_mean) * mean_moe_ms
        saving_out = (linear_out_mean - out_mean) * mean_moe_ms
        results[method]["in_sample"]["mean_fraction"] = in_mean
        results[method]["in_sample"]["mean_median_ratio"] = in_ratio_mean
        results[method]["in_sample"]["projected_saving_ms"] = saving_in
        results[method]["held_out"]["mean_fraction"] = out_mean
        results[method]["held_out"]["mean_median_ratio"] = out_ratio_mean
        results[method]["held_out"]["projected_saving_ms"] = saving_out

    best_method = min(methods, key=lambda m: results[m]["held_out"]["mean_fraction"])

    final_placement = {}
    if best_method == "linear":
        final_placement = {
            layer: linear_placement(num_experts, ranks) for layer in records_by_layer
        }
    else:
        for layer, recs in records_by_layer.items():
            T = build_T(recs, num_experts, bsm)
            p_lpt = build_lpt(T, num_experts, ranks)
            if best_method == "lpt":
                final_placement[layer] = p_lpt
            else:
                final_placement[layer] = swap_improve(
                    T, [list(x) for x in p_lpt], objective
                )

    report = {
        "meta": {
            "hist": hist_path,
            "num_experts": num_experts,
            "ep_size": ranks,
            "bsm": bsm,
            "min_tokens": min_tokens,
            "folds": num_folds,
            "objective": objective,
            "mean_moe_ms": mean_moe_ms,
            "steps": all_steps,
            "num_layers": len(records_by_layer),
        },
        "methods": results,
        "best_method": best_method,
        "notes": [
            "projected_saving_ms values are estimates derived from --mean-moe-ms"
        ],
    }

    best_in = results[best_method]["in_sample"]["mean_fraction"]
    best_out = results[best_method]["held_out"]["mean_fraction"]
    gain_in = linear_in_mean - best_in
    gain_out = linear_out_mean - best_out
    if gain_out < gain_in - 1e-12:
        drop = gain_in - gain_out
        msg = (
            f"held-out gain {gain_out:.6f} is below in-sample gain {gain_in:.6f} "
            f"(drop {drop:.6f})"
        )
        if gain_out < 0.0 or (gain_in > 0.0 and gain_out < 0.8 * gain_in):
            msg += "; the placement likely overfits the train steps."
        else:
            msg += "; the drop is small."
        report["notes"].append(msg)
    else:
        report["notes"].append(
            f"held-out gain {gain_out:.6f} is at or above in-sample gain {gain_in:.6f}."
        )
    if best_method == "linear":
        report["notes"].append("no method beat the linear placement on held-out folds.")
    return report, final_placement


def _synth_weights(rng, num_experts, alpha):
    """Dirichlet-style popularity vector."""
    if np is not None:
        return rng.dirichlet([alpha] * num_experts)
    weights = [rng.gammavariate(alpha, 1.0) for _ in range(num_experts)]
    total = sum(weights)
    return [w / total for w in weights]


def generate_synthetic(
    path, num_layers, num_steps, num_experts, tokens, top_k, skew, seed, alpha=0.5
):
    """Write a synthetic histogram JSONL capture."""
    meta = {
        "meta": {
            "num_experts": num_experts,
            "ep_size": 4,
            "top_k": top_k,
            "block_size_m": 4,
            "linear_placement": True,
        }
    }
    total_counts = tokens * top_k
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        seq = 0
        if np is not None:
            rng = np.random.default_rng(seed)
            for layer in range(num_layers):
                if skew == "systematic":
                    p = _synth_weights(rng, num_experts, alpha)
                for step in range(num_steps):
                    if skew == "iid":
                        p = _synth_weights(rng, num_experts, alpha)
                    counts = rng.multinomial(total_counts, p).tolist()
                    rec = {
                        "seq": seq,
                        "step": step,
                        "layer": f"model.layers.{layer}.mlp.experts",
                        "layer_idx": layer,
                        "tokens": tokens,
                        "counts": counts,
                    }
                    f.write(json.dumps(rec) + "\n")
                    seq += 1
        else:
            rng = random.Random(seed)
            for layer in range(num_layers):
                if skew == "systematic":
                    p = _synth_weights(rng, num_experts, alpha)
                for step in range(num_steps):
                    if skew == "iid":
                        p = _synth_weights(rng, num_experts, alpha)
                    draws = rng.choices(range(num_experts), weights=p, k=total_counts)
                    counts = [0] * num_experts
                    for d in draws:
                        counts[d] += 1
                    rec = {
                        "seq": seq,
                        "step": step,
                        "layer": f"model.layers.{layer}.mlp.experts",
                        "layer_idx": layer,
                        "tokens": tokens,
                        "counts": counts,
                    }
                    f.write(json.dumps(rec) + "\n")
                    seq += 1


def run_selftest():
    """Synthetic checks: systematic skew, iid popularity, placement validity."""
    import tempfile

    print("selftest: systematic skew")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sys.jsonl")
        generate_synthetic(
            path,
            num_layers=4,
            num_steps=40,
            num_experts=64,
            tokens=512,
            top_k=10,
            skew="systematic",
            seed=0,
            alpha=0.1,
        )
        report, final_placement = analyze(
            path,
            ranks=4,
            bsm=4,
            min_tokens=0,
            folds=4,
            objective="max",
            mean_moe_ms=206.2,
            seed=0,
        )
        linear_out = report["methods"]["linear"]["held_out"]["mean_fraction"]
        lpt_out = report["methods"]["lpt"]["held_out"]["mean_fraction"]
        best_method = report["best_method"]
        best_out = report["methods"][best_method]["held_out"]["mean_fraction"]
        if lpt_out >= linear_out - 1e-9:
            raise AssertionError(
                f"systematic skew: lpt held-out {lpt_out} not better than "
                f"linear {linear_out}"
            )
        validate_placement(final_placement, 64, 4)
        print(
            f"  best method: {best_method}, held-out fraction {best_out:.4f} "
            f"vs linear {linear_out:.4f}, lpt {lpt_out:.4f}"
        )

        print("selftest: iid popularity")
        path2 = os.path.join(td, "iid.jsonl")
        generate_synthetic(
            path2,
            num_layers=4,
            num_steps=40,
            num_experts=64,
            tokens=512,
            top_k=10,
            skew="iid",
            seed=1,
            alpha=0.5,
        )
        report2, final_placement2 = analyze(
            path2,
            ranks=4,
            bsm=4,
            min_tokens=0,
            folds=4,
            objective="max",
            mean_moe_ms=206.2,
            seed=0,
        )
        linear_out2 = report2["methods"]["linear"]["held_out"]["mean_fraction"]
        best_method2 = report2["best_method"]
        best_out2 = report2["methods"][best_method2]["held_out"]["mean_fraction"]
        gain = linear_out2 - best_out2
        if gain > 0.02:
            raise AssertionError(
                f"iid popularity: gain {gain:.4f} is too large; no method should help"
            )
        validate_placement(final_placement2, 64, 4)
        print(f"  best method: {best_method2}, held-out gain {gain:.4f} (expected ~0)")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist", nargs="?", help="JSONL histogram capture")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--bsm", type=int, default=4)
    ap.add_argument("--min-tokens", type=int, default=512)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--objective", choices=["max", "mean"], default="max")
    ap.add_argument("--mean-moe-ms", type=float, default=206.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write final placement JSON")
    ap.add_argument("--json", dest="report_json", help="write report JSON")
    ap.add_argument("--selftest", action="store_true", help="run synthetic tests")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.hist:
        ap.error("HIST.jsonl is required unless --selftest is given")

    report, final_placement = analyze(
        args.hist,
        args.ranks,
        args.bsm,
        args.min_tokens,
        args.folds,
        args.objective,
        args.mean_moe_ms,
        args.seed,
    )

    if args.out:
        meta = report["meta"]
        num_experts = meta["num_experts"]
        ranks = meta["ep_size"]
        validate_placement(final_placement, num_experts, ranks)
        out_obj = {
            "version": 1,
            "num_experts": num_experts,
            "ep_size": ranks,
            "method": report["best_method"],
            "layers": {
                str(layer): placement for layer, placement in final_placement.items()
            },
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)
        print(f"wrote placement to {args.out}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote report to {args.report_json}")

    print(f"best method: {report['best_method']}")
    for method in ["linear", "lpt", "lpt+swap"]:
        m = report["methods"][method]
        in_frac = m["in_sample"]["mean_fraction"]
        out_frac = m["held_out"]["mean_fraction"]
        saving = m["held_out"]["projected_saving_ms"]
        print(
            f"  {method}: in-sample fraction {in_frac:.6f}, "
            f"held-out fraction {out_frac:.6f}, "
            f"held-out saving ~{saving:.2f} ms/step (estimate)"
        )
    for note in report["notes"]:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()
