# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic routing statistics (Zipf-skewed, uniform, topic-clustered).

For tests, the sample plan and what-if studies; real placements come from a
``VLLM_EXPERT_STATS_DIR`` capture. Rows are drawn without replacement with
Plackett-Luce (Gumbel top-k) weights ``(rank + 1) ** -alpha`` over a per-layer
random permutation of the experts (so the hot experts differ per layer), with a
per-layer jitter of ``alpha``. ``--clusters N`` adds a latent topic per row that
boosts the weights of one random group of experts by ``--beta``: rows of one
topic co-select the same experts, which is what co-activation-aware placement
exploits.

CLI::

    python -m tools.rdna.hetero_moe.synth --out DIR [--kind zipf|uniform]
        [--alpha 1.1] [--layers 48] [--experts 512] [--top-k 10]
        [--decode-rows 4000] [--prefill-rows 20000] [--clusters 0] [--pairs]
        [--seed 0]

writes ``DIR/expert_stats-dp0-pid1.jsonl`` (two cumulative snapshots, the
collector's raw format) and, with ``--pairs``, its ``.pairs.npz``.
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from .sampling import gumbel_topk_rows

DEFAULT_ROWS_PER_STEP = {1: 0.25, 3: 0.55, 4: 0.1, 8: 0.1}


@dataclass
class SynthData:
    num_experts: int
    top_k: int
    decode_counts: np.ndarray  # int64 [L, E]
    prefill_counts: np.ndarray
    decode_rows: int
    prefill_rows: int
    step_rows: np.ndarray  # int [S]: rows of each decode step
    trace: np.ndarray | None  # int16 [rows, L, K] decode ids, rows in step order
    pairs: np.ndarray | None  # int32 [L, E, E], a < b
    weights: np.ndarray  # float [L, E] sampling weights (truth)


def _layer_weights(
    layers: int, experts: int, kind: str, alpha: float, jitter: float, rng
) -> np.ndarray:
    weights = np.ones((layers, experts))
    if kind == "uniform":
        return weights
    if kind != "zipf":
        raise ValueError(f"unknown kind {kind!r}")
    ranks = np.arange(experts, dtype=np.float64) + 1.0
    for layer in range(layers):
        a = max(0.0, alpha + jitter * rng.uniform(-1.0, 1.0))
        weights[layer, rng.permutation(experts)] = ranks**-a
    return weights


def generate(
    *,
    kind: str = "zipf",
    layers: int = 48,
    experts: int = 512,
    top_k: int = 10,
    alpha: float = 1.1,
    alpha_jitter: float = 0.3,
    decode_rows: int = 4000,
    prefill_rows: int = 20000,
    clusters: int = 0,
    beta: float = 3.0,
    keep_trace: bool = False,
    pairs: bool = False,
    seed: int = 0,
    rows_per_step: dict[int, float] | None = None,
) -> SynthData:
    rng = np.random.default_rng(seed)
    weights = _layer_weights(layers, experts, kind, alpha, alpha_jitter, rng)
    # Decode steps: how many rows each step carries.
    dist = rows_per_step or DEFAULT_ROWS_PER_STEP
    sizes = np.array(sorted(dist))
    probs = np.array([dist[s] for s in sizes], dtype=np.float64)
    probs /= probs.sum()
    step_rows: list[int] = []
    total = 0
    while total < decode_rows:
        s = int(rng.choice(sizes, p=probs))
        step_rows.append(s)
        total += s
    decode_rows = total
    dec = np.zeros((layers, experts), dtype=np.int64)
    pre = np.zeros((layers, experts), dtype=np.int64)
    trace = (
        np.zeros((decode_rows, layers, top_k), dtype=np.int16) if keep_trace else None
    )
    pair_counts = (
        np.zeros((layers, experts, experts), dtype=np.int32) if pairs else None
    )
    for layer in range(layers):
        logits = np.log(weights[layer])
        groups = None
        if clusters > 0:
            groups = rng.integers(0, clusters, size=experts)
        for phase, rows in (("decode", decode_rows), ("prefill", prefill_rows)):
            done = 0
            while done < rows:
                n = min(4096, rows - done)
                if groups is not None:
                    topic = rng.integers(0, clusters, size=n)
                    boost = beta * (groups[None, :] == topic[:, None])
                    ids = gumbel_topk_rows(logits[None, :] + boost, top_k, rng)
                else:
                    ids = gumbel_topk_rows(
                        np.broadcast_to(logits, (n, experts)), top_k, rng
                    )
                target = dec if phase == "decode" else pre
                target[layer] += np.bincount(ids.ravel(), minlength=experts)
                if phase == "decode":
                    if trace is not None:
                        trace[done : done + n, layer] = ids
                    if pair_counts is not None:
                        a = np.sort(ids, axis=1)
                        for i in range(top_k):
                            for j in range(i + 1, top_k):
                                np.add.at(pair_counts[layer], (a[:, i], a[:, j]), 1)
                done += n
    return SynthData(
        experts,
        top_k,
        dec,
        pre,
        decode_rows,
        prefill_rows,
        np.asarray(step_rows),
        trace,
        pair_counts,
        weights,
    )


def write_raw(data: SynthData, directory: str, snapshots: int = 2) -> str:
    """Write the collector's raw JSONL (+ pairs.npz); returns the JSONL path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "expert_stats-dp0-pid1.jsonl")
    layers, experts = data.decode_counts.shape
    names = [f"model.layers.{i}.mlp.experts" for i in range(layers)]
    hist_len = 65
    hist = np.zeros(hist_len, dtype=np.int64)
    for s in data.step_rows.tolist():
        hist[min(s, hist_len - 1)] += 1
    meta = {
        "type": "meta",
        "schema": "expert-stats/1",
        "model": {
            "name": "synthetic",
            "num_layers": layers,
            "num_experts": experts,
            "top_k": data.top_k,
        },
        "num_experts": experts,
        "top_k": data.top_k,
        "decode_max_tokens": hist_len - 1,
        "pairs": data.pairs is not None,
        "tp_size": 4,
        "dp_rank": 0,
        "pid": 1,
        "time": 1.7e9,
        "layers": [
            {"name": n, "idx": i, "kind": "target"} for i, n in enumerate(names)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        for k in range(1, snapshots + 1):
            frac = k / snapshots
            counts, tokens, steps, hists = {}, {}, {}, {}
            for i, n in enumerate(names):
                dec = np.floor(data.decode_counts[i] * frac).astype(np.int64)
                pre = np.floor(data.prefill_counts[i] * frac).astype(np.int64)
                counts[n] = {"decode": dec.tolist(), "prefill": pre.tolist()}
                tokens[n] = {
                    "decode": int(dec.sum()) // data.top_k,
                    "prefill": int(pre.sum()) // data.top_k,
                }
                steps[n] = {
                    "decode": int(len(data.step_rows) * frac),
                    "prefill": max(1, int(data.prefill_rows * frac // 2048)),
                }
                hists[n] = np.floor(hist * frac).astype(np.int64).tolist()
            snap = {
                "type": "snapshot",
                "seq": k,
                "time": 1.7e9 + 30.0 * k,
                "elapsed_s": 30.0 * k,
                "counts": counts,
                "tokens": tokens,
                "steps": steps,
                "rows_hist": hists,
                "invalid": {n: 0 for n in names},
            }
            f.write(json.dumps(snap, separators=(",", ":")) + "\n")
    if data.pairs is not None:
        np.savez(
            path[: -len(".jsonl")] + ".pairs.npz",
            names=np.array(names),
            pairs=data.pairs,
        )
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--kind", choices=("zipf", "uniform"), default="zipf")
    p.add_argument("--alpha", type=float, default=1.1)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--decode-rows", type=int, default=4000)
    p.add_argument("--prefill-rows", type=int, default=20000)
    p.add_argument("--clusters", type=int, default=0)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--pairs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    data = generate(
        kind=args.kind,
        layers=args.layers,
        experts=args.experts,
        top_k=args.top_k,
        alpha=args.alpha,
        decode_rows=args.decode_rows,
        prefill_rows=args.prefill_rows,
        clusters=args.clusters,
        beta=args.beta,
        pairs=args.pairs,
        seed=args.seed,
    )
    print(write_raw(data, args.out))


if __name__ == "__main__":
    main()
