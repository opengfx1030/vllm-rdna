# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU simulator of EP load balance for Flash-Next on a V620 island.

Routing model. Per layer the decode selection probability of expert ``e`` per row,
``q[e]`` (sum = top-k), comes from measured stats (``hetero-moe-expert-stats/1`` or raw
collector JSONL) or from the parametric Zipf skew model (``zipf_record``). Decode steps
draw ``W`` rows (``3 * batch`` with MTP2, ``batch`` without) as systematic samples with
inclusion probabilities ``q``; rows of one sequence's MTP group and of one step share
experts like the rows of the real T10 trace (``tools.rdna.hetero_moe.row_model``).
Placements are trained on one stats file and scored on steps drawn from another, so the
gain reported is out of sample. ``--train`` and ``--test`` may be the same file: then it
is in sample.

Studies (every one prints a table; ``--json`` writes the raw numbers):

``decode``      per width: max-over-mean of touched experts T, of pairs P and of the
                time model for linear / round robin / pack(pairs) / pack(touch) /
                oracle (pack on the test stats: the best any permutation can do) and
                the predicted MoE time per step;
``redundancy``  extra expert slots per rank (replicate_pack or hot-all-ranks) under
                primary / hash / min_touch routing against the best permutation,
                with the VRAM and KV capacity the slots cost;
``prefill``     persistent rank imbalance of the marginals (linear vs packed, out of
                sample) and, with ``--alpha``, a Dirichlet chunk model of the transient
                part (``--calibrate`` fits alpha to a measured median max/mean);
``ep``          the persistent imbalance at EP 2/4/8;
``replay``      the same placement scoring on real per-step counts (an id trace npz, a
                per-call histogram JSONL or an ``eplb-stats/1`` dump), see ``traces``.

    python -m tools.rdna.eplb.simulate decode --train W1.stats.json --test W2.stats.json
    python -m tools.rdna.eplb.simulate all --train W1.stats.json --test W2.stats.json \\
        --json out.json
"""

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from tools.rdna.eplb import loads, placement
from tools.rdna.eplb.geometry import DEFAULT_COST, FLASHNEXT, CostModel, Geometry
from tools.rdna.hetero_moe import row_model as row_models
from tools.rdna.hetero_moe.sampling import inclusion_probs
from tools.rdna.hetero_moe.stats_io import LayerStats, StatsRecord, load_stats

MEASURED_ITL_MS = {1: 22.38, 8: 43.97}  # TP4 flashnext-mtp2 (tp8-pp2 doc baselines)


# ---- inputs -------------------------------------------------------------------


def record_from_marginals(
    decode: np.ndarray, prefill: np.ndarray, top_k: int, rows: float = 1e6
) -> StatsRecord:
    """A ``StatsRecord`` from per-layer decode / prefill selection probabilities."""
    layers = {}
    for li in range(decode.shape[0]):
        d = np.rint(decode[li] * rows).astype(np.int64)
        p = np.rint(prefill[li] * rows).astype(np.int64)
        layers[li] = LayerStats(
            li, "target", None, d.astype(float), p.astype(float), d, p, rows, rows
        )
    return StatsRecord(
        decode.shape[1], top_k, layers, {"num_experts": decode.shape[1], "top_k": top_k}
    )


def zipf_record(
    layers: int = 48,
    experts: int = 512,
    top_k: int = 10,
    alpha: float = 0.9,
    seed: int = 0,
) -> StatsRecord:
    """Parametric skew model: Plackett-Luce weights ``(rank + 1) ** -alpha`` over a
    random permutation of the experts per layer (hot experts differ per layer)."""
    rng = np.random.default_rng(seed)
    w = (np.arange(experts) + 1.0) ** -alpha
    q = np.stack(
        [inclusion_probs(w[rng.permutation(experts)], top_k) for _ in range(layers)]
    )
    return record_from_marginals(q, q, top_k)


def zipf_alpha_for_top_share(
    top: int, share: float, experts: int = 512, top_k: int = 10
) -> float:
    """Zipf exponent whose top-``top`` experts carry ``share`` of the routed pairs."""
    lo, hi = 0.0, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q = np.sort(inclusion_probs((np.arange(experts) + 1.0) ** -mid, top_k))[::-1]
        if q[:top].sum() / top_k < share:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class LayerData:
    layer: int
    q_train: np.ndarray  # decode inclusion probability per row (sum = top_k)
    q_test: np.ndarray
    pre_train: np.ndarray  # prefill pair share per expert (sum = 1)
    pre_test: np.ndarray


def _norm(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.shape[0])


def layer_data(
    train: StatsRecord, test: StatsRecord, layers: list[int]
) -> list[LayerData]:
    out = []
    for li in layers:
        a, b = train.layers[li], test.layers[li]
        qa = (
            a.decode / a.decode_tokens
            if a.decode_tokens > 0
            else _norm(a.prefill) * train.top_k
        )
        qb = (
            b.decode / b.decode_tokens
            if b.decode_tokens > 0
            else _norm(b.prefill) * test.top_k
        )
        out.append(
            LayerData(
                li,
                inclusion_probs(np.clip(qa, 0, 1), train.top_k),
                inclusion_probs(np.clip(qb, 0, 1), test.top_k),
                _norm(a.prefill if a.prefill.sum() > 0 else a.decode),
                _norm(b.prefill if b.prefill.sum() > 0 else b.decode),
            )
        )
    return out


def layer_ids(spec: str, available: list[int], n_target: int = 48) -> list[int]:
    """``a:b:step`` (range over target layers) or ``i,j,k``; only available ones."""
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        ids = list(range(*parts))
    else:
        ids = [int(x) for x in spec.replace(",", " ").split()]
    return [i for i in ids if i in available and i < n_target]


# ---- decode steps ---------------------------------------------------------------


def expected_touch(q: np.ndarray, width: int) -> np.ndarray:
    """E[touched] of an expert with row probability ``q`` in ``width`` rows."""
    return 1.0 - (1.0 - np.clip(q, 0.0, 1.0)) ** width


def sample_steps(
    ld: LayerData,
    width: int,
    steps: int,
    model: row_models.RowModel,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ids [rows, K], step_of_row, row_in_step)`` of ``steps`` decode steps."""
    step_of_row = np.repeat(np.arange(steps), width)
    row_in_step = np.tile(np.arange(width), steps)
    ids = model.sample(ld.layer, ld.q_test, step_of_row, row_in_step, rng)
    return ids, step_of_row, row_in_step


def step_count(
    width: int, rows_budget: int, min_steps: int = 200, max_steps: int = 4000
) -> int:
    return int(np.clip(rows_budget // max(width, 1), min_steps, max_steps))


def decode_owners(
    ld: LayerData,
    ranks: int,
    width: int,
    train_width: int | None = None,
    static: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """The permutation-only placements compared (``train_width``: step width the touch
    weights are tuned for, default ``width``; ``static``: the width-independent ones
    when already computed)."""
    tw = train_width or width
    if static is None:
        static = static_owners(ld, ranks)
    return {
        **static,
        "pack_touch": placement.pack_owner(expected_touch(ld.q_train, tw), ranks),
        "oracle": placement.pack_owner(expected_touch(ld.q_test, width), ranks),
    }


def static_owners(ld: LayerData, ranks: int) -> dict[str, np.ndarray]:
    """The placements that do not depend on the step width."""
    e = ld.q_train.shape[0]
    return {
        "linear": placement.linear_owner(e, ranks),
        "round_robin": placement.round_robin_owner(e, ranks),
        "pack_pairs": placement.pack_owner(ld.q_train, ranks),
    }


def decode_study(
    data: list[LayerData],
    *,
    ranks: int = 4,
    widths: list[int],
    mtp: int = 2,
    rows_budget: int = 24000,
    seed: int = 0,
    cost: CostModel = DEFAULT_COST,
    geometry: Geometry = FLASHNEXT,
    model: row_models.RowModel | None = None,
    train_width: int | None = None,
) -> dict:
    """Max-over-mean and MoE time per step by width and permutation placement."""
    if model is None:
        model = row_models.load(None) if mtp else row_models.INDEPENDENT
    statics = [static_owners(ld, ranks) for ld in data]
    out: dict[str, dict] = {}
    for width in widths:
        rng = np.random.default_rng(seed + width)
        steps = step_count(width, rows_budget)
        acc: dict[str, list[dict]] = {}
        for ld, static in zip(data, statics):
            ids, sor, _ = sample_steps(ld, width, steps, model, rng)
            counts = loads.step_counts(ids, sor, steps, ld.q_test.shape[0])
            for name, owner in decode_owners(
                ld, ranks, width, train_width, static
            ).items():
                acc.setdefault(name, []).append(
                    loads.summarize(loads.rank_loads(counts, owner, ranks, cost))
                )
        row: dict = {}
        for name, runs in acc.items():
            row[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            row[name]["step_us"] = row[name]["max_us"] * geometry.layers
        row["steps"] = steps
        row["layers"] = len(data)
        out[str(width)] = row
    return {"ranks": ranks, "mtp": mtp, "widths": out}


# ---- redundancy -------------------------------------------------------------------


def redundancy_study(
    data: list[LayerData],
    *,
    ranks: int = 4,
    width: int,
    extras: tuple[int, ...] = (0, 2, 4, 8, 12, 24),
    rows_budget: int = 12000,
    seed: int = 0,
    cost: CostModel = DEFAULT_COST,
    geometry: Geometry = FLASHNEXT,
    model: row_models.RowModel | None = None,
) -> dict:
    """Extra slots per rank against the best permutation (``pack_touch``)."""
    model = model or row_models.load(None)
    rng = np.random.default_rng(seed + 1000 + width)
    steps = step_count(width, rows_budget, 150, 2000)
    runs: dict[str, list[float]] = {}
    for ld in data:
        ids, sor, ris = sample_steps(ld, width, steps, model, rng)
        counts = loads.step_counts(ids, sor, steps, ld.q_test.shape[0])
        w_touch = expected_touch(ld.q_train, width)
        owner = placement.pack_owner(w_touch, ranks)
        base = loads.summarize(loads.rank_loads(counts, owner, ranks, cost))
        runs.setdefault("perm", []).append(base["max_us"])
        runs.setdefault("perm_T", []).append(base["T"])
        for extra in extras:
            if extra == 0:
                continue
            hot = max(1, int(round(extra * ranks / (ranks - 1))))
            kinds = {
                "replicate": placement.replicate_pack(w_touch, ranks, extra),
                "hot": placement.hot_holders(owner, w_touch, ranks, hot),
            }
            for kind, holders in kinds.items():
                for policy in ("primary", "hash", "min_touch"):
                    res = placement.route_loads(
                        ids, sor, ris, steps, holders, policy, cost
                    )
                    s = loads.summarize(res)
                    key = f"{kind}/{policy}/{extra}"
                    runs.setdefault(key, []).append(s["max_us"])
                    runs.setdefault(key + "/T", []).append(s["T"])
                    runs.setdefault(key + "/live", []).append(
                        float(res["T"].sum(axis=1).mean())
                    )
                    runs.setdefault(key + "/slots", []).append(
                        float(holders.extra_slots_per_rank(ld.q_test.shape[0]))
                    )
    mean = {k: float(np.mean(v)) for k, v in runs.items()}
    rows = []
    for extra in extras:
        if extra == 0:
            continue
        for kind in ("replicate", "hot"):
            for policy in ("primary", "hash", "min_touch"):
                key = f"{kind}/{policy}/{extra}"
                nbytes = geometry.redundancy_bytes(1) * mean[key + "/slots"]
                rows.append(
                    {
                        "kind": kind,
                        "policy": policy,
                        "extra_slots_per_rank": mean[key + "/slots"],
                        "T": mean[key + "/T"],
                        "layer_us": mean[key],
                        "gain_vs_perm_pct": 100.0 * (1.0 - mean[key] / mean["perm"]),
                        "live_experts_per_step": mean[key + "/live"],
                        "vram_mib_per_rank": nbytes / (1 << 20),
                        "kv_tokens_lost": geometry.kv_tokens_lost(nbytes),
                        "kv_fraction_lost": geometry.kv_fraction_lost(nbytes),
                    }
                )
    return {
        "ranks": ranks,
        "width": width,
        "perm_T": mean["perm_T"],
        "perm_layer_us": mean["perm"],
        "rows": rows,
    }


# ---- prefill ------------------------------------------------------------------------


def dirichlet_chunks(
    pre: np.ndarray,
    alpha: float,
    chunks: int,
    tokens: int,
    top_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``[chunks, E]`` pair counts of prefill chunks whose expert popularity is
    ``Dirichlet(alpha * E * pre)`` around the marginal ``pre`` (overdispersion = the
    content of the chunk). An expert gets at most one pair per token."""
    p = rng.dirichlet(alpha * pre / pre.sum() * pre.shape[0] + 1e-3, size=chunks)
    return np.minimum(rng.poisson(p * tokens * top_k), tokens).astype(np.int32)


def prefill_study(
    data: list[LayerData],
    *,
    ranks: int = 4,
    alpha: float | None = None,
    chunks: int = 300,
    tokens: int = 2048,
    top_k: int = 10,
    seed: int = 0,
    cost: CostModel = DEFAULT_COST,
) -> dict:
    """Persistent (marginal) imbalance, and with ``alpha`` the chunk-level transient
    (per-chunk max/mean of the rank time and of the pairs)."""
    rng = np.random.default_rng(seed)
    pers: dict[str, list[float]] = {
        "linear": [],
        "round_robin": [],
        "pack": [],
        "oracle": [],
    }
    chunk: dict[str, list[float]] = {}
    for ld in data:
        e = ld.pre_test.shape[0]
        own = {
            "linear": placement.linear_owner(e, ranks),
            "round_robin": placement.round_robin_owner(e, ranks),
            "pack": placement.pack_owner(ld.pre_train, ranks),
            "oracle": placement.pack_owner(ld.pre_test, ranks),
        }
        for name, o in own.items():
            pers[name].append(placement.balance_ratio(ld.pre_test, o, ranks))
        if alpha is not None:
            counts = dirichlet_chunks(ld.pre_test, alpha, chunks, tokens, top_k, rng)
            for name in ("linear", "round_robin", "pack", "oracle"):
                res = loads.rank_loads(counts, own[name], ranks, cost)
                chunk.setdefault(f"{name}/us", []).append(
                    loads.max_over_mean(res["us"])
                )
                chunk.setdefault(f"{name}/pairs", []).append(
                    loads.max_over_mean(res["P"])
                )
    out: dict = {"ranks": ranks, "persistent": {k: _stats(v) for k, v in pers.items()}}
    if alpha is not None:
        out["alpha"] = alpha
        out["chunk"] = {k: _stats(v) for k, v in chunk.items()}
    return out


def _stats(v: list[float]) -> dict[str, float]:
    a = np.asarray(v)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "max": float(a.max()),
    }


def _linear_chunk_ratio(
    data: list[LayerData], alpha: float, *, ranks: int, seed: int, chunks: int
) -> float:
    """Median over layers of the chunk max/mean of the rank time (linear placement)."""
    rng = np.random.default_rng(seed)
    ratios = []
    for ld in data:
        e = ld.pre_test.shape[0]
        counts = dirichlet_chunks(ld.pre_test, alpha, chunks, 2048, 10, rng)
        owner = placement.linear_owner(e, ranks)
        ratios.append(loads.max_over_mean(loads.rank_loads(counts, owner, ranks)["us"]))
    return float(np.median(ratios))


def calibrate_alpha(
    data: list[LayerData],
    target_median: float,
    *,
    ranks: int = 4,
    seed: int = 0,
    chunks: int = 200,
) -> float:
    """Dirichlet concentration at which the linear placement's median per-layer chunk
    max/mean of the rank time equals ``target_median`` (bisection in log alpha)."""
    lo, hi = np.log(1e-3), np.log(2e5)
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        got = _linear_chunk_ratio(
            data, float(np.exp(mid)), ranks=ranks, seed=seed, chunks=chunks
        )
        if got > target_median:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


# ---- EP size --------------------------------------------------------------------


def ep_study(
    data: list[LayerData], sizes: tuple[int, ...] = (2, 4, 8), width: int = 24
) -> dict:
    """Persistent imbalance of the marginals at each EP size (no sampling noise)."""
    out = {}
    for r in sizes:
        buckets: dict[str, list[float]] = {
            "pairs_linear": [],
            "pairs_pack": [],
            "touch_linear": [],
            "touch_pack": [],
        }
        for ld in data:
            e = ld.q_test.shape[0]
            lin = placement.linear_owner(e, r)
            wt = expected_touch(ld.q_test, width)
            pack_p = placement.pack_owner(ld.q_train, r)
            pack_t = placement.pack_owner(expected_touch(ld.q_train, width), r)
            buckets["pairs_linear"].append(placement.balance_ratio(ld.q_test, lin, r))
            buckets["pairs_pack"].append(placement.balance_ratio(ld.q_test, pack_p, r))
            buckets["touch_linear"].append(placement.balance_ratio(wt, lin, r))
            buckets["touch_pack"].append(placement.balance_ratio(wt, pack_t, r))
        out[str(r)] = {k: _stats(v) for k, v in buckets.items()}
    return out


# ---- replay of real per-step counts ---------------------------------------------


def replay_counts(
    counts: np.ndarray,
    train_counts: np.ndarray | None,
    *,
    ranks: int = 4,
    objective: str = "pairs",
    cost: CostModel = DEFAULT_COST,
) -> dict:
    """Score placements on measured ``counts[step, E]``.

    ``train_counts`` (default: the same steps = in sample) is what the packed placement
    is fitted on; ``objective`` picks its weight: ``pairs`` (total pairs) or ``touch``
    (fraction of steps touching the expert).
    """
    e = counts.shape[1]
    src = counts if train_counts is None else train_counts
    w = (
        src.sum(axis=0).astype(float)
        if objective == "pairs"
        else (src > 0).mean(axis=0)
    )
    own = {
        "linear": placement.linear_owner(e, ranks),
        "round_robin": placement.round_robin_owner(e, ranks),
        "pack": placement.pack_owner(w, ranks),
        "oracle": placement.pack_owner(
            counts.sum(axis=0).astype(float)
            if objective == "pairs"
            else (counts > 0).mean(axis=0),
            ranks,
        ),
    }
    return {
        n: loads.summarize(loads.rank_loads(counts, o, ranks, cost))
        for n, o in own.items()
    }


# ---- CLI ------------------------------------------------------------------------


def _fmt_decode(res: dict, itl_ms: dict[int, float]) -> str:
    names = ["linear", "round_robin", "pack_pairs", "pack_touch", "oracle"]
    lines = [
        f"decode, EP{res['ranks']}, MTP{res['mtp']}: E[max]/E[mean] of touched "
        "experts T | pairs P | time, MoE ms per step (all layers)",
        f"{'rows':>5} " + " ".join(f"{n:>26}" for n in names),
    ]
    for width, row in res["widths"].items():
        cells = []
        for n in names:
            r = row[n]
            cells.append(f"T {r['T']:.3f} P {r['P']:.3f} {r['step_us'] / 1000:6.2f}ms")
        lines.append(f"{width:>5} " + " ".join(f"{c:>26}" for c in cells))
    lines.append(
        "gain of pack_touch over linear (MoE ms per step, % of MoE, % of ITL):"
    )
    for width, row in res["widths"].items():
        d = (row["linear"]["step_us"] - row["pack_touch"]["step_us"]) / 1000
        pct = 100 * d / (row["linear"]["step_us"] / 1000)
        batch = int(width) // max(res["mtp"] + 1, 1) if res["mtp"] else int(width)
        itl = itl_ms.get(batch)
        tail = f"  {100 * d / itl:5.2f}% of {itl:.2f} ms ITL" if itl else ""
        lines.append(f"{width:>5}  {d:6.3f} ms  {pct:5.2f}% of MoE{tail}")
    return "\n".join(lines)


def _fmt_red(res: dict) -> str:
    lines = [
        f"redundancy, EP{res['ranks']}, {res['width']} rows: permutation-only T "
        f"ratio {res['perm_T']:.3f}, layer {res['perm_layer_us']:.1f} us",
        f"{'kind':>10} {'routing':>10} {'+slots':>6} {'T':>6} {'live':>7} "
        f"{'gain%':>7} {'VRAM MiB':>9} {'KV lost':>8}",
    ]
    for r in res["rows"]:
        lines.append(
            f"{r['kind']:>10} {r['policy']:>10} "
            f"{r['extra_slots_per_rank']:>6.1f} {r['T']:>6.3f} "
            f"{r['live_experts_per_step']:>7.1f} {r['gain_vs_perm_pct']:>7.2f} "
            f"{r['vram_mib_per_rank']:>9.0f} "
            f"{100 * r['kv_fraction_lost']:>7.1f}%"
        )
    return "\n".join(lines)


def _load_pair(args: argparse.Namespace) -> tuple[StatsRecord, StatsRecord]:
    if args.zipf is not None:
        rec = zipf_record(alpha=args.zipf, seed=args.seed)
        return rec, zipf_record(alpha=args.zipf, seed=args.seed)
    train = load_stats([args.train])
    return train, load_stats([args.test]) if args.test else train


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train", help="stats the placements are fitted on")
    p.add_argument("--test", help="stats the steps are drawn from (default: --train)")
    p.add_argument(
        "--zipf",
        type=float,
        default=None,
        help="parametric skew model: exponent (no stats files)",
    )
    p.add_argument("--ep", type=int, default=4)
    p.add_argument(
        "--mtp",
        type=int,
        default=2,
        help="MTP depth: a request contributes mtp+1 rows per step",
    )
    p.add_argument("--batches", default="1,2,4,8,16,32,64")
    p.add_argument("--layers", default="0:48:4")
    p.add_argument("--rows-budget", type=int, default=24000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write the raw numbers here")


def _replay_main(a: argparse.Namespace) -> int:
    """Score placements on real steps: fit on the other folds, judge on each fold."""
    from tools.rdna.eplb import traces

    sources: dict[str, dict[int, traces.StepCounts]] = {}
    if a.id_trace:
        sources["decode (id trace)"] = traces.id_trace(a.id_trace, a.num_experts)
    if a.hist:
        if a.hist.endswith(".npz"):
            sources["prefill (eplb-stats ring)"] = traces.eplb_stats_hist(a.hist)
        else:
            sources["prefill (call histograms)"] = traces.hist_jsonl(a.hist, 512)[1]
    if not sources:
        print("give --id-trace and/or --hist", file=sys.stderr)
        return 2
    out: dict = {
        "format": "eplb-replay/1",
        "ep": a.ep,
        "objective": a.objective,
        "sources": {},
    }
    for label, per_layer in sources.items():
        print(
            f"\n{label}: EP{a.ep}, objective {a.objective}, {a.folds} contiguous "
            "folds (fit on the others)"
        )
        print(
            f"{'layer':>5} {'steps':>6} {'linear T/P/us':>22} "
            f"{'packed T/P/us':>22} {'oracle T/P/us':>22}"
        )
        rows: dict[str, dict] = {}
        for layer, sc in sorted(per_layer.items()):
            folds = [
                replay_counts(test, train, ranks=a.ep, objective=a.objective)
                for train, test in (
                    traces.split_steps(sc, a.folds, f) for f in range(a.folds)
                )
                if test.shape[0] and train.shape[0]
            ]
            if not folds:
                continue
            mean = {
                name: {
                    k: float(np.mean([f[name][k] for f in folds]))
                    for k in folds[0][name]
                }
                for name in folds[0]
            }
            rows[str(layer)] = {"steps": sc.steps, **mean}
            cells = [
                f"{mean[n]['T']:.3f}/{mean[n]['P']:.3f}/{mean[n]['us']:.3f}"
                for n in ("linear", "pack", "oracle")
            ]
            print(f"{layer:>5} {sc.steps:>6} " + " ".join(f"{c:>22}" for c in cells))
        out["sources"][label] = rows
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decode", "redundancy", "prefill", "ep", "all"):
        _common(sub.add_parser(name))
    sub.choices["prefill"].add_argument("--alpha", type=float, default=None)
    sub.choices["prefill"].add_argument(
        "--calibrate", type=float, default=None, help="measured median max/mean"
    )
    sub.choices["redundancy"].add_argument("--width", type=int, default=24)
    rp = sub.add_parser("replay", help="score placements on a real per-step trace")
    rp.add_argument(
        "--id-trace", help="collector expert_stats-*.trace.npz (decode steps)"
    )
    rp.add_argument(
        "--hist", help="moe_routing_hist.jsonl or an eplb-stats/1 file (prefill calls)"
    )
    rp.add_argument("--num-experts", type=int, default=512)
    rp.add_argument("--objective", choices=("touch", "pairs"), default="touch")
    rp.add_argument("--folds", type=int, default=2, help="contiguous train/test blocks")
    rp.add_argument("--ep", type=int, default=4)
    rp.add_argument("--json", help="write the raw numbers here")
    a = ap.parse_args(argv)
    if a.cmd == "replay":
        return _replay_main(a)
    if a.zipf is None and not a.train:
        ap.error("give --train (and --test) stats, or --zipf")
    train, test = _load_pair(a)
    data = layer_data(
        train, test, layer_ids(a.layers, sorted(set(train.layers) & set(test.layers)))
    )
    mult = a.mtp + 1 if a.mtp else 1
    widths = [int(b) * mult for b in a.batches.split(",")]
    out: dict = {
        "format": "eplb-sim/1",
        "ep": a.ep,
        "mtp": a.mtp,
        "layers": [d.layer for d in data],
    }
    if a.cmd in ("decode", "all"):
        out["decode"] = decode_study(
            data,
            ranks=a.ep,
            widths=widths,
            mtp=a.mtp,
            rows_budget=a.rows_budget,
            seed=a.seed,
        )
        print(_fmt_decode(out["decode"], MEASURED_ITL_MS))
    if a.cmd in ("redundancy", "all"):
        for width in [a.width] if a.cmd == "redundancy" else [mult, 8 * mult]:
            res = redundancy_study(
                data,
                ranks=a.ep,
                width=width,
                rows_budget=a.rows_budget // 2,
                seed=a.seed,
            )
            out.setdefault("redundancy", {})[str(width)] = res
            print("\n" + _fmt_red(res))
    if a.cmd in ("prefill", "all"):
        alpha = getattr(a, "alpha", None)
        if getattr(a, "calibrate", None):
            alpha = calibrate_alpha(data, a.calibrate, ranks=a.ep, seed=a.seed)
        elif a.cmd == "all":
            alpha = calibrate_alpha(data, 1.39, ranks=a.ep, seed=a.seed)
        out["prefill"] = prefill_study(data, ranks=a.ep, alpha=alpha, seed=a.seed)
        print("\nprefill:", json.dumps(out["prefill"], indent=1))
    if a.cmd in ("ep", "all"):
        out["ep_sizes"] = ep_study(data)
        print("\nep sizes:", json.dumps(out["ep_sizes"], indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
