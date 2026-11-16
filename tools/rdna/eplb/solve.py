# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats -> static expert placement (phase 1): a ``VLLM_RDNA_MOE_PLACEMENT`` v1 file.

    python -m tools.rdna.eplb.solve --train W1.stats.json [--test W2.stats.json] \\
        --ep 4 --objective touch --rows 24 --out placement.json --report report.json
    python -m tools.rdna.eplb.solve --counters DIR --rows 24 --out placement.json

Permutation only: every rank keeps ``E / R`` experts per layer, so shapes, kernels,
graphs and the tile8 prefill gate are untouched; only ``expert_map`` changes (the
fork's loader, ``rdna_moe_placement``). Mono K2 needs the linear block and refuses it.

``--objective``
  ``touch``  distinct experts touched per rank in a ``--rows``-row decode step, the
             bandwidth-bound decode cost (default; what a V620 decode kernel pays);
  ``pairs``  routed pairs per rank (compute-bound prefill cost; decode pairs for
             ``--prefill-share 0``);
  ``blend``  ``(1 - s) * decode + s * prefill`` weights packed together with a
             chunk-aware refinement, ``s = --prefill-share``.

A layer whose predicted slowest-rank gain is below ``--min-gain`` stays linear and is
left out of the file (the loader keeps linear for missing layers). With ``--test`` the
gain is measured on a different capture than the fit (out of sample).
"""

import argparse
import json
import sys
from typing import Any

import numpy as np

from tools.rdna.eplb import placement
from tools.rdna.eplb.simulate import expected_touch
from tools.rdna.hetero_moe.sampling import inclusion_probs
from tools.rdna.hetero_moe.stats_io import LayerStats, StatsRecord, load_stats

OBJECTIVES = ("touch", "pairs", "blend")


def _norm(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    return x / s if s > 0 else np.full(x.shape[0], 1.0 / x.shape[0])


def decode_q(ls: LayerStats, top_k: int) -> np.ndarray | None:
    """Per-row selection probability of every expert (sum = top_k), or None."""
    if ls.decode_tokens <= 0 or ls.decode.sum() <= 0:
        return None
    return inclusion_probs(np.clip(ls.decode / ls.decode_tokens, 0.0, 1.0), top_k)


def weights(
    ls: LayerStats,
    top_k: int,
    objective: str,
    rows: int,
    prefill_share: float,
    touch: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(decode weight, prefill weight, blended)`` of one layer, each sums to 1.

    ``touch`` replaces the model ``1 - (1 - q) ** rows`` with a measured touch
    frequency (``summary.touch_weight``).
    """
    q = decode_q(ls, top_k)
    if q is None:
        dec = _norm(ls.prefill.astype(np.float64))
    elif objective == "pairs":
        dec = _norm(q)
    else:
        dec = _norm(touch if touch is not None else expected_touch(q, rows))
    pre = _norm(ls.prefill.astype(np.float64)) if ls.prefill.sum() > 0 else dec
    blend = (1.0 - prefill_share) * dec + prefill_share * pre if prefill_share else None
    return dec, pre, blend


def solve_layer(
    train: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    test: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    ranks: int,
    objective: str,
    prefill_share: float,
    min_gain: float,
) -> tuple[np.ndarray | None, dict]:
    """The placement of one layer (None = keep linear) and its report entry."""
    e = train[0].shape[0]
    lin = placement.linear_owner(e, ranks)
    dec, pre, blend = train
    cand = placement.pack_owner(blend if blend is not None else dec, ranks)
    if objective == "blend" and prefill_share:
        costs = np.stack([(1.0 - prefill_share) * dec, prefill_share * pre])
        cand = placement.pack_empirical(costs, ranks, cand, iters=4000)
    t_dec, t_pre, t_blend = test
    obj = t_blend if t_blend is not None else t_dec
    before = placement.balance_ratio(obj, lin, ranks)
    after = placement.balance_ratio(obj, cand, ranks)
    entry = {
        "decode_ratio_linear": placement.balance_ratio(t_dec, lin, ranks),
        "decode_ratio_packed": placement.balance_ratio(t_dec, cand, ranks),
        "prefill_ratio_linear": placement.balance_ratio(t_pre, lin, ranks),
        "prefill_ratio_packed": placement.balance_ratio(t_pre, cand, ranks),
        "objective_ratio_linear": before,
        "objective_ratio_packed": after,
        "experts_moved": int((cand != lin).sum()),
    }
    if (before - after) / before < min_gain:
        entry["kept_linear"] = True
        return None, entry
    return cand, entry


def solve(
    train: StatsRecord,
    test: StatsRecord | None,
    ranks: int,
    *,
    objective: str = "touch",
    rows: int = 24,
    prefill_share: float = 0.0,
    min_gain: float = 0.01,
    touch: dict[int, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(placement document, report)`` for every target layer in ``train``."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective!r} not in {OBJECTIVES}")
    test = test or train
    layers: dict[str, list[list[int]]] = {}
    report: dict[str, Any] = {}
    for li in train.layer_ids("target"):
        if li not in test.layers:
            continue
        w_train = weights(
            train.layers[li],
            train.top_k,
            objective,
            rows,
            prefill_share,
            (touch or {}).get(li),
        )
        w_test = weights(test.layers[li], test.top_k, objective, rows, prefill_share)
        owner, entry = solve_layer(
            w_train, w_test, ranks, objective, prefill_share, min_gain
        )
        report[str(li)] = entry
        if owner is not None:
            layers[str(li)] = [
                [int(x) for x in np.flatnonzero(owner == r)] for r in range(ranks)
            ]
    doc = {
        "version": 1,
        "num_experts": train.num_experts,
        "ep_size": ranks,
        "method": f"eplb-lean/{objective}",
        "layers": layers,
        "provenance": {
            "objective": objective,
            "rows": rows,
            "prefill_share": prefill_share,
            "min_gain": min_gain,
            "train_sources": train.sources,
            "test_sources": (test.sources if test is not train else "same as train"),
        },
    }
    return doc, report


def validate_doc(doc: dict[str, Any]) -> list[str]:
    """Problems that would make the loader ignore or reject a layer (empty = fine)."""
    problems = []
    e, r = doc.get("num_experts"), doc.get("ep_size")
    if doc.get("version") != 1:
        problems.append("version != 1")
    if not isinstance(e, int) or not isinstance(r, int) or e % r:
        return problems + ["num_experts / ep_size missing or not divisible"]
    for key, entry in doc.get("layers", {}).items():
        if len(entry) != r:
            problems.append(f"layer {key}: {len(entry)} rank lists, ep_size {r}")
            continue
        if any(len(x) != e // r for x in entry):
            problems.append(f"layer {key}: a rank list is not {e // r} long")
        flat = sorted(x for rank in entry for x in rank)
        if flat != list(range(e)):
            problems.append(f"layer {key}: not a permutation of 0..{e - 1}")
    return problems


def _touch_from_counters(
    path: str, rows: int
) -> tuple[StatsRecord, dict[int, np.ndarray]]:
    from tools.rdna.eplb import summary

    stats = summary.load_dir(path)
    rs = stats[0]
    record = summary.to_stats_record(rs)
    touch = {}
    for pos, info in enumerate(rs.layers):
        w, n = summary.touch_weight(rs, pos, rows)
        if n:
            touch[int(info["layer"])] = w
    return record, touch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--train", nargs="+", help="stats the placement is fitted on")
    ap.add_argument("--test", nargs="+", help="stats the gain is measured on")
    ap.add_argument("--counters", help="eplb-stats/1 directory (measured touch bands)")
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--objective", choices=OBJECTIVES, default="touch")
    ap.add_argument(
        "--rows", type=int, default=24, help="decode rows per step (MTP2: 3 x batch)"
    )
    ap.add_argument("--prefill-share", type=float, default=0.0)
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--out", help="placement JSON for VLLM_RDNA_MOE_PLACEMENT")
    ap.add_argument("--report", help="per-layer before/after JSON")
    a = ap.parse_args(argv)
    touch = None
    if a.counters:
        train, touch = _touch_from_counters(a.counters, a.rows)
    elif a.train:
        train = load_stats(a.train)
    else:
        ap.error("give --train or --counters")
    test = load_stats(a.test) if a.test else None
    doc, report = solve(
        train,
        test,
        a.ep,
        objective=a.objective,
        rows=a.rows,
        prefill_share=a.prefill_share,
        min_gain=a.min_gain,
        touch=touch,
    )
    problems = validate_doc(doc)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    changed = len(doc["layers"])
    print(f"{changed} of {len(report)} layers get a non-linear placement")
    dec_before = np.mean([v["decode_ratio_linear"] for v in report.values()])
    dec_after = np.mean(
        [
            v["decode_ratio_packed"]
            if not v.get("kept_linear")
            else v["decode_ratio_linear"]
            for v in report.values()
        ]
    )
    print(
        "mean per-layer weight max/mean (decode weight): "
        f"{dec_before:.4f} -> {dec_after:.4f}"
    )
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
