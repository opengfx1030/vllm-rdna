# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlated row model of a decode step, fitted from a top-k id trace.

The rows of one decode step are not independent draws from the layer marginal: the
verified token of a sequence and its MTP drafts route alike, and the sequences of a
step share some experts (T10 section 6). ``sampling.correlated_sample`` draws steps
with two sharing factors per layer, ``rho`` (row keeps an expert of its group's base
row) and ``tau`` (a group's base keeps an expert of the step's base row);
``rho = tau = 0`` is the independent model. This module fits them per traced layer
(moment matching of the shared experts per row pair, within a group and across
groups, at the most common rows-per-step band of the trace), stores them as JSON and
interpolates over the layer index.

    python -m tools.rdna.hetero_moe.row_model fit TRACE.npz --out row_model.json
    python -m tools.rdna.hetero_moe.row_model show row_model.json
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from tools.rdna.hetero_moe.sampling import correlated_sample

FORMAT = "hetero-moe-row-model/1"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NAME = "flashnext-mtp2"
DEFAULT_PATH = os.path.join(HERE, "row_model_flashnext_mtp2.json")


@dataclass
class RowModel:
    """Sharing factors per layer index; ``group`` consecutive rows are one sequence."""

    group: int = 3
    layers: dict[int, tuple[float, float]] = field(default_factory=dict)
    name: str = "independent"
    meta: dict = field(default_factory=dict)

    def params(self, layer: int) -> tuple[float, float]:
        """(rho, tau) of ``layer``: the fitted layers, linear in between, clamped."""
        if not self.layers:
            return 0.0, 0.0
        xs = sorted(self.layers)
        rho = float(np.interp(layer, xs, [self.layers[x][0] for x in xs]))
        tau = float(np.interp(layer, xs, [self.layers[x][1] for x in xs]))
        return rho, tau

    def sample(
        self,
        layer: int,
        q: np.ndarray,
        step_of_row: np.ndarray,
        row_in_step: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        rho, tau = self.params(layer)
        return correlated_sample(q, step_of_row, row_in_step, self.group, rho, tau, rng)


INDEPENDENT = RowModel()


def load(spec: str | None) -> RowModel:
    """``independent``, the shipped fit (``flashnext-mtp2``, the default), or a path."""
    if spec in ("independent", "none"):
        return INDEPENDENT
    path = DEFAULT_PATH if spec in (None, "", DEFAULT_NAME) else spec
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} document")
    layers = {int(k): (v["rho"], v["tau"]) for k, v in doc["layers"].items()}
    name = DEFAULT_NAME if path == DEFAULT_PATH else os.path.basename(path)
    return RowModel(int(doc["group"]), layers, name, doc.get("meta", {}))


def _pair_shares(blocks: np.ndarray, num_experts: int, group: int) -> np.ndarray:
    """Mean shared experts per row pair: (within a group, across groups) over steps.

    ``blocks`` is ``[steps, T, K]``."""
    n, t, _ = blocks.shape
    onehot = np.zeros((n, t, num_experts), dtype=np.float32)
    np.put_along_axis(onehot, blocks.astype(np.int64), 1.0, axis=2)
    total = onehot.sum(axis=1)  # [n, E] rows selecting each expert
    all_pairs = ((total * total).sum(axis=1) - total.sum(axis=1)) / 2.0
    within = np.zeros(n)
    n_in = 0
    for start in range(0, t, group):
        g = onehot[:, start : start + group].sum(axis=1)
        within += ((g * g).sum(axis=1) - g.sum(axis=1)) / 2.0
        size = min(group, t - start)
        n_in += size * (size - 1) // 2
    n_out = t * (t - 1) // 2 - n_in
    return np.array(
        [within.mean() / max(n_in, 1), (all_pairs - within).mean() / max(n_out, 1)]
    )


def fit_layer(
    blocks: np.ndarray,
    q: np.ndarray,
    group: int,
    seed: int = 0,
    steps: int = 600,
    iters: int = 14,
) -> dict:
    """(rho, tau) matching the shared experts per pair of ``blocks`` (``[n, T, K]``).

    Alternating bisection on the two moments (within-group sharing grows with rho,
    across-group sharing with tau), common random numbers."""
    n, t, k = blocks.shape
    num_experts = q.shape[0]
    target = _pair_shares(blocks, num_experts, group)
    use = min(steps, n)
    step_of_row = np.repeat(np.arange(use), t)
    row_in_step = np.tile(np.arange(t), use)

    def moments(rho: float, tau: float) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ids = correlated_sample(q, step_of_row, row_in_step, group, rho, tau, rng)
        return _pair_shares(ids.reshape(use, t, k), num_experts, group)

    rho, tau = 0.4, 0.2
    for _ in range(3):
        lo, hi = 0.0, 0.98
        for _ in range(iters):
            rho = 0.5 * (lo + hi)
            lo, hi = (rho, hi) if moments(rho, tau)[0] < target[0] else (lo, rho)
        rho = 0.5 * (lo + hi)
        lo, hi = 0.0, 0.98
        for _ in range(iters):
            tau = 0.5 * (lo + hi)
            lo, hi = (tau, hi) if moments(rho, tau)[1] < target[1] else (lo, tau)
        tau = 0.5 * (lo + hi)
    got = moments(rho, tau)
    return {
        "rho": float(rho),
        "tau": float(tau),
        "target_shared_per_pair": target.tolist(),
        "fitted_shared_per_pair": got.tolist(),
        "fit_steps": int(use),
    }


def fit_trace(
    trace_path: str, group: int = 3, band: int | None = None, seed: int = 0
) -> dict:
    """Model document fitted on the most common rows-per-step band of a trace."""
    from tools.rdna.hetero_moe import trace_check as tc

    trace = tc.load_trace(trace_path)
    top_k = trace["ids"].shape[-1]
    layers: dict[str, dict] = {}
    chosen = band
    for pos, name in enumerate(trace["names"]):
        steps = tc.valid_steps(trace, pos)
        sizes = np.array([len(s) for s in steps])
        if chosen is None:
            vals, counts = np.unique(sizes, return_counts=True)
            chosen = int(vals[counts.argmax()])
        picks = [s for s in steps if len(s) == chosen]
        q = tc.marginal(steps, 512, top_k)
        fit = fit_layer(np.stack(picks), q, group, seed)
        fit["band"] = chosen
        fit["band_steps"] = len(picks)
        layers[str(tc.layer_index(name))] = fit
    with open(trace_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "format": FORMAT,
        "group": group,
        "layers": layers,
        "meta": {
            "trace_sha256": sha,
            "trace": os.path.basename(trace_path),
            "fit": "shared experts per row pair, within group and across groups",
            "calls": [int(c) for c in trace["calls"]],
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit")
    a.add_argument("trace")
    a.add_argument("--out", required=True)
    a.add_argument("--group", type=int, default=3)
    a.add_argument("--band", type=int, help="rows per step to fit (default: modal)")
    a.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("show")
    s.add_argument("model", nargs="?", default=DEFAULT_NAME)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        doc = fit_trace(args.trace, args.group, args.band, args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        for layer, fit in doc["layers"].items():
            print(
                f"layer {layer}: rho {fit['rho']:.3f} tau {fit['tau']:.3f} "
                f"shared/pair target {np.round(fit['target_shared_per_pair'], 2)} "
                f"fitted {np.round(fit['fitted_shared_per_pair'], 2)}"
            )
    else:
        model = load(args.model)
        print(model.name, "group", model.group)
        for layer in sorted(model.layers):
            print(layer, model.layers[layer])
    sys.exit(0)


if __name__ == "__main__":
    main()
