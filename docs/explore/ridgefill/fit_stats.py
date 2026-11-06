#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit RidgeFill maps from a statistics capture; score them on held-out rows.

``capture_activations.py --stats`` leaves cumulative fp64 sums after 8, 16,
32 and 64 fit prompts (``stats_pNNN.pt``) and fp16 held-out rows
(``rows_held.pt``). For every split and snapshot this script centers the
sums and solves ``math.fit_ridge``'s estimator for each lambda of ``--lams``
from one fp64 eigendecomposition of the centered Gram ``Gc = V diag(e) V^T``::

    alpha = lam * tr(Gc) / din
    A^T   = V diag(1 / (e + alpha)) V^T Cxy      (= (Gc + alpha I)^-1 Cxy)
    b     = mu_y - A mu_x

with ``Cxy`` the centered cross moment of the split state and the layer's
block input. The snapshot sums are of shifted rows (``capture_probe``); the
centering removes the shift.

Scores, per split, snapshot, layer and lambda (``report.json``):

* ``train``: the fp64 map's rel RMS on the fit rows, from the sums.
* ``held``: ``||y_hat - y||_F / ||y||_F`` on the held-out stride rows (every
  ``raw_stride``-th position), as served: fp16 split state, fp16 ``A`` and
  ``b``, fp32 accumulation, fp16 output.
* ``fill``: the same on the stride rows below ``--fill-limit`` (12288, the
  filled region of a 16K prompt: tail 2048, cut rounded down to 4096).
* ``buckets``: the same per position bucket (0, 1-3, 4-15, 16-63, 64-511,
  512+) over every held-out row, including the dense head rows.

``--old-maps`` scores an existing ``maps.pt`` on the same rows at
``--old-split``. Each bundle takes, per layer, the lambda with the lowest
``fill`` error at the last snapshot (``--lambda-mode global``: one lambda for
every layer, the lowest mean). The selection uses the held-out rows, so its
own score is optimistic by the choice among the lambdas.

Outputs in ``--out`` (default ``<capture>/fit``)::

    report.json        the scores, the chosen lambdas, the learning curve
    bundle_s<split>/   exactly maps.pt, ridgefill.json, pins.json
    DONE               written last, once every bundle passes
                       ``loader.require_bundle`` and its maps load

    python3 fit_stats.py --capture CAPTURE_DIR --old-maps OLD/maps.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "vllm" / "model_executor" / "layers" / "ridgefill"
LAMS = "0.003,0.01,0.03,0.1,0.3"
BUCKETS = ((0, 0), (1, 3), (4, 15), (16, 63), (64, 511), (512, None))
MAP_FORMAT = "ridgefill-w4a16-fp16-v1"
REPORT_FORMAT = "ridgefill-fit-report-v1"
STATS_FORMAT = "ridgefill-stats-v1"
ROWS_FORMAT = "ridgefill-rows-v1"


def _load(name: str):
    """Load one RidgeFill module without importing vllm's package __init__."""
    import importlib.util

    path = PKG / f"{name}.py"
    mod_name = f"ridgefill_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def bucket_name(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


class Moments:
    """Centered fp64 moments of one split from a statistics snapshot.

    The snapshot holds sums of ``u = x - shift_x`` and ``v = y - shift_y``.
    ``Gc = sum(u u^T) - n mu_u mu_u^T`` and ``Cxy = sum(u v^T) - n mu_u
    mu_v^T`` are the centered moments, ``mu_x = shift_x + mu_u``. ``head``
    (``x`` [R, din] and ``y`` {layer: [R, dout]} rows of the fit prompts)
    is subtracted from the sums first, to fit without those rows.
    """

    def __init__(self, stats: dict, split: int, head: dict | None = None) -> None:
        import torch

        if stats.get("format") != STATS_FORMAT:
            raise SystemExit(f"not a statistics snapshot: {stats.get('format')}")
        if split not in stats["x"]:
            raise SystemExit(f"the snapshot has no split {split}")
        self.stats = stats
        self.split = split
        xs = stats["x"][split]
        shift = xs["shift"].double()
        total = xs["sum"].double().clone()
        gram = xs["gram"].double().clone()
        n = float(stats["rows"])
        self.head_u = None
        if head is not None and int(head["x"].shape[0]) > 0:
            u = head["x"].double() - shift
            n -= float(u.shape[0])
            total -= u.sum(dim=0)
            gram -= u.T @ u
            self.head_u = u
            self.head_y = head["y"]
        self.n = n
        self.mu_u = total / n
        gram.addr_(self.mu_u, self.mu_u, alpha=-n)
        self.gc = gram
        self.mu_x = shift + self.mu_u
        self.din = int(gram.shape[0])
        self.trace = float(torch.diagonal(gram).sum())

    def layer(self, layer: int) -> tuple:
        """``(mu_y, Cxy [din, dout], sum |y - mu_y|^2, sum |y|^2)`` in fp64."""
        ys = self.stats["y"][layer]
        shift = ys["shift"].double()
        total = ys["sum"].double().clone()
        sumsq = ys["sumsq"].double().clone()
        cross = self.stats["cross"][self.split][layer].double().clone()
        n = self.n
        if self.head_u is not None:
            v = self.head_y[layer].double() - shift
            total -= v.sum(dim=0)
            sumsq -= (v * v).sum(dim=0)
            cross -= self.head_u.T @ v
        mu_v = total / n
        cross.addr_(self.mu_u, mu_v, alpha=-n)
        centered = float((sumsq - n * mu_v * mu_v).sum())
        raw = float((sumsq + 2.0 * shift * total + n * shift * shift).sum())
        return shift + mu_v, cross, centered, raw


class RidgeSolver:
    """One eigendecomposition of ``Gc``; ridge maps and fit error for any lambda."""

    def __init__(self, moments: Moments) -> None:
        import torch

        start = time.perf_counter()
        evals, vecs = torch.linalg.eigh(moments.gc)
        self.seconds = time.perf_counter() - start
        self.evals = evals.clamp(min=0.0)
        self.vecs = vecs
        self._vecs32 = None
        self.trace = moments.trace
        self.din = moments.din

    def alpha(self, lam: float) -> float:
        return float(lam) * self.trace / float(self.din)

    def project(self, cross):
        """``P = V^T Cxy`` [din, dout], fp64."""
        return self.vecs.T @ cross

    def weights(self, lam: float):
        return 1.0 / (self.evals + self.alpha(lam))

    def train_sse(self, proj, centered: float, lam: float) -> float:
        """``sum |y - A x - b|^2`` on the fit rows, exact in fp64."""
        w = self.weights(lam)
        power = (proj * proj).sum(dim=1)
        return float(
            centered - 2.0 * (w * power).sum() + (self.evals * w * w * power).sum()
        )

    def maps(self, proj, mu_x, mu_y, lam: float, dtype=None) -> tuple:
        """``A`` [dout, din] and ``b`` [dout]; fp32 unless ``dtype`` says."""
        import torch

        dtype = dtype or torch.float32
        if dtype == torch.float64:
            vecs = self.vecs
        else:
            if self._vecs32 is None or self._vecs32.dtype != dtype:
                self._vecs32 = self.vecs.to(dtype)
            vecs = self._vecs32
        scaled = (proj * self.weights(lam).unsqueeze(1)).to(dtype)
        a = (vecs @ scaled).T.contiguous()
        b = mu_y.to(dtype) - a @ mu_x.to(dtype)
        return a, b


class HeldRows:
    """Held-out fp16 rows and their masks, scored as the serve applies a map."""

    def __init__(self, path: Path, fill_limit: int) -> None:
        import torch

        data = _torch_load(path)
        if data.get("format") != ROWS_FORMAT:
            raise SystemExit(f"{path} is not a rows file")
        self.x = data["x"]
        self.y = data["y"]
        self.stride = int(data["stride"])
        pos = data["positions"].long()
        self.rows = int(pos.numel())
        strided = (
            pos % self.stride == 0
            if self.stride > 0
            else torch.ones_like(pos, dtype=torch.bool)
        )
        self.masks = {"held": strided, "fill": strided & (pos < fill_limit)}
        self.counts = {k: int(m.sum()) for k, m in self.masks.items()}
        self.buckets = {}
        for lo, hi in BUCKETS:
            mask = pos >= lo if hi is None else (pos >= lo) & (pos <= hi)
            self.buckets[bucket_name(lo, hi)] = mask
            self.counts[bucket_name(lo, hi)] = int(mask.sum())
        self._split = None
        self._xf = None

    def xf(self, split: int):
        if self._split != split:
            self._xf = None
            self._xf = self.x[split].float()
            self._split = split
        return self._xf

    def score(self, split: int, layer: int, a16, b16) -> dict:
        """Rel RMS of the fp16 map on the held rows, overall and per mask."""
        import torch

        y = self.y[layer].float()
        pred = torch.addmm(b16.float(), self.xf(split), a16.float().T).half().float()
        err = (pred - y).square().sum(dim=1).double()
        ref = y.square().sum(dim=1).double()

        def rel(mask) -> float | None:
            den = float(ref[mask].sum())
            if den <= 0.0:
                return None
            return math.sqrt(float(err[mask].sum()) / den)

        out: dict = {name: rel(mask) for name, mask in self.masks.items()}
        out["buckets"] = {name: rel(mask) for name, mask in self.buckets.items()}
        return out


def head_rows(path: Path, below: int, prompts: int) -> dict:
    """The fit prompts' rows at positions below ``below`` (first ``prompts``)."""
    data = _torch_load(path)
    keep = (data["positions"].long() < below) & (data["prompts"].long() < prompts)
    idx = keep.nonzero().flatten()
    return {
        "x": {s: v.index_select(0, idx) for s, v in data["x"].items()},
        "y": {layer: v.index_select(0, idx) for layer, v in data["y"].items()},
        "rows": int(idx.numel()),
    }


def _mean(values: list) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _rank(value: float | None) -> float:
    """Sort key for an error that may be missing (no rows in its mask)."""
    return math.inf if value is None else value


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(
    out: Path,
    split: int,
    maps: dict,
    layer_info: dict,
    pins: dict,
    head: dict,
) -> Path:
    """``maps.pt``, ``ridgefill.json`` and ``pins.json``, written then renamed."""
    import torch

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save({int(k): v for k, v in sorted(maps.items())}, tmp / "maps.pt")
    first = next(iter(maps.values()))
    manifest = {
        "format": MAP_FORMAT,
        "model": {
            "vocab_sha256": pins["vocab_sha256"],
            "encodings": pins["encodings"],
        },
        "split": int(split),
        **head,
        "din": int(first[0].shape[1]),
        "dout": int(first[0].shape[0]),
        "map_dtype": "fp16",
        "map_sha256": _sha256(tmp / "maps.pt"),
        "layers": {str(k): layer_info[k] for k in sorted(maps)},
    }
    (tmp / "ridgefill.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    pin_file = {"vocab_sha256": pins["vocab_sha256"], "encodings": pins["encodings"]}
    (tmp / "pins.json").write_text(
        json.dumps(pin_file, indent=2) + "\n", encoding="utf-8"
    )
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def check_bundle(out: Path, split: int, layers: list, pins_path: str) -> dict:
    """``loader.require_bundle`` on the bundle, then its maps against the manifest."""
    import torch

    loader = _load("loader")
    loader.require_bundle(str(out), str(out / "pins.json"))
    if pins_path:
        loader.require_bundle(str(out), pins_path)
    manifest = json.loads((out / "ridgefill.json").read_text(encoding="utf-8"))
    if sorted(p.name for p in out.iterdir()) != [
        "maps.pt",
        "pins.json",
        "ridgefill.json",
    ]:
        raise SystemExit(f"{out} holds more than maps.pt, ridgefill.json, pins.json")
    if manifest["map_sha256"] != _sha256(out / "maps.pt"):
        raise SystemExit(f"{out}: maps.pt does not match its manifest sha")
    if manifest["split"] != split:
        raise SystemExit(f"{out}: manifest split {manifest['split']} != {split}")
    maps = torch.load(out / "maps.pt", map_location="cpu", weights_only=True)
    if sorted(maps) != list(layers) or sorted(
        int(k) for k in manifest["layers"]
    ) != list(layers):
        raise SystemExit(f"{out}: layers {sorted(maps)} != {list(layers)}")
    for layer, (a, b) in maps.items():
        want = (manifest["dout"], manifest["din"])
        if a.dtype != torch.float16 or tuple(a.shape) != want:
            raise SystemExit(f"{out}: layer {layer} map {a.dtype} {tuple(a.shape)}")
        if b.dtype != torch.float16 or tuple(b.shape) != (manifest["dout"],):
            raise SystemExit(f"{out}: layer {layer} bias {b.dtype} {tuple(b.shape)}")
        if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
            raise SystemExit(f"{out}: layer {layer} map is not finite")
    return {"path": str(out), "map_sha256": manifest["map_sha256"], "layers": layers}


def fit_split(
    args,
    split: int,
    end: int,
    snapshots: list,
    cap: Path,
    held: HeldRows,
    lams: list,
    report: dict,
    report_path: Path,
) -> tuple[dict, dict, int]:
    """Score every snapshot; return the last snapshot's maps per layer and lambda."""
    layers = list(range(split, end))
    entry = report["splits"].setdefault(str(split), {"snapshots": {}, "curve": {}})
    final: dict = {}
    rows = 0
    for snap in snapshots:
        prompts = int(snap["prompts"])
        last = snap is snapshots[-1]
        start = time.perf_counter()
        stats = _torch_load(cap / snap["file"])
        head = None
        if args.exclude_head > 0:
            head_all = head_rows(cap / "rows_head.pt", args.exclude_head, prompts)
            head = {"x": head_all["x"][split], "y": head_all["y"]}
        moments = Moments(stats, split, head)
        solver = RidgeSolver(moments)
        evals = solver.evals
        out: dict = {
            "prompts": prompts,
            "rows": moments.n,
            "trace": moments.trace,
            "eig_max": float(evals[-1]),
            "eig_min": float(evals[0]),
            "eigh_seconds": round(solver.seconds, 2),
            "alpha": {str(lam): solver.alpha(lam) for lam in lams},
            "layers": {},
        }
        for layer in layers:
            mu_y, cross, centered, raw = moments.layer(layer)
            proj = solver.project(cross)
            scores: dict = {}
            for lam in lams:
                sse = solver.train_sse(proj, centered, lam)
                a, b = solver.maps(proj, moments.mu_x, mu_y, lam)
                a16, b16 = a.half(), b.half()
                score = held.score(split, layer, a16, b16)
                score["train"] = math.sqrt(max(sse, 0.0) / raw) if raw > 0 else None
                scores[str(lam)] = score
                if last:
                    final.setdefault(layer, {})[lam] = (a16, b16)
            out["layers"][str(layer)] = scores
            print(
                f"split {split} p{prompts:03d} layer {layer:2d}: "
                + " ".join(
                    f"{lam:g}:{scores[str(lam)]['fill']:.4f}"
                    for lam in lams
                    if scores[str(lam)]["fill"] is not None
                ),
                flush=True,
            )
        out["mean"] = {
            str(lam): {
                key: _mean([out["layers"][str(L)][str(lam)][key] for L in layers])
                for key in ("train", "held", "fill")
            }
            for lam in lams
        }
        best = min(lams, key=lambda lam: _rank(out["mean"][str(lam)]["fill"]))
        entry["curve"][str(prompts)] = {"lambda": best, **out["mean"][str(best)]}
        out["seconds"] = round(time.perf_counter() - start, 1)
        entry["snapshots"][str(prompts)] = out
        rows = int(moments.n)
        _write_json(report_path, report)
        print(
            f"split {split} p{prompts:03d}: best lambda {best:g}, mean fill "
            f"{out['mean'][str(best)]['fill']:.4f}, {out['seconds']} s",
            flush=True,
        )
        del stats, moments, solver
    return final, entry, rows


def score_old_maps(args, held: HeldRows, layers: list) -> dict:
    """The existing maps' held-out error on the same rows."""
    old = _torch_load(Path(args.old_maps))
    out: dict = {"path": str(args.old_maps), "split": args.old_split, "layers": {}}
    for layer in layers:
        if layer not in old:
            continue
        a16, b16 = old[layer]
        out["layers"][str(layer)] = held.score(
            args.old_split, layer, a16.half(), b16.half()
        )
    out["mean"] = {
        key: _mean([v[key] for v in out["layers"].values()]) for key in ("held", "fill")
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--capture", required=True, help="capture_activations --stats dir")
    p.add_argument("--out", default="", help="default: <capture>/fit")
    p.add_argument("--lams", default=LAMS)
    p.add_argument("--splits", default="", help="default: every captured split")
    p.add_argument(
        "--lambda-mode", choices=("per-layer", "global"), default="per-layer"
    )
    p.add_argument("--fill-limit", type=int, default=12288)
    p.add_argument(
        "--exclude-head",
        type=int,
        default=0,
        help="fit without the fit prompts' positions below this (rows_head.pt)",
    )
    p.add_argument("--old-maps", default="", help="maps.pt to score on the same rows")
    p.add_argument("--old-split", type=int, default=24)
    p.add_argument("--pins", default="", help="the serve's pins file; must match")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--force", action="store_true", help="refit over a DONE fit")
    args = p.parse_args(argv)

    import torch

    torch.set_num_threads(max(1, args.threads))
    started = time.perf_counter()
    cap = Path(args.capture)
    out = Path(args.out) if args.out else cap / "fit"
    meta = json.loads((cap / "capture_meta.json").read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise SystemExit(f"{cap} is not a complete capture")
    done = out / "DONE"
    if done.exists():
        if not args.force:
            raise SystemExit(f"{out} is already fitted (DONE); pass --force")
        done.unlink()
    out.mkdir(parents=True, exist_ok=True)
    pins = json.loads((cap / "pins.json").read_text(encoding="utf-8"))
    if args.pins:
        serve_pins = json.loads(Path(args.pins).read_text(encoding="utf-8"))
        for key in ("vocab_sha256", "encodings"):
            if serve_pins.get(key) != pins.get(key):
                raise SystemExit(f"capture pins {key} differs from {args.pins}")
    lams = sorted(float(v) for v in args.lams.split(",") if v.strip())
    splits = [int(s) for s in args.splits.split(",") if s.strip()] or list(
        meta["splits"]
    )
    end = int(meta["end"])
    snapshots = sorted(meta["snapshots"], key=lambda s: int(s["prompts"]))
    if not snapshots:
        raise SystemExit(f"{cap} has no snapshots")
    held = HeldRows(cap / meta["rows"]["held"]["file"], args.fill_limit)
    report_path = out / "report.json"
    report: dict = {
        "format": REPORT_FORMAT,
        "capture": str(cap.resolve()),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lams": lams,
        "lambda_mode": args.lambda_mode,
        "exclude_head": args.exclude_head,
        "estimator": "math.fit_ridge: centered, alpha = lam * tr(Gc) / din",
        "eval": {
            "metric": "||y_hat - y||_F / ||y||_F",
            "as_served": "fp16 x, fp16 A and b, fp32 accumulate, fp16 output",
            "held_rows": held.rows,
            "stride": held.stride,
            "fill_limit": args.fill_limit,
            "rows_per_mask": held.counts,
            "train": "fp64 map on the fit rows, from the sums",
        },
        "held": meta.get("held", {}).get("meta", {}),
        "splits": {},
    }
    bundles = []
    for split in splits:
        layers = list(range(split, end))
        if args.old_maps and split == args.old_split:
            entry = report["splits"].setdefault(
                str(split), {"snapshots": {}, "curve": {}}
            )
            entry["old_maps"] = score_old_maps(args, held, layers)
            _write_json(report_path, report)
            print(f"old maps at split {split}: mean {entry['old_maps']['mean']}")
        final, entry, rows = fit_split(
            args, split, end, snapshots, cap, held, lams, report, report_path
        )
        last = entry["snapshots"][str(snapshots[-1]["prompts"])]
        scores = {
            layer: {lam: last["layers"][str(layer)][str(lam)] for lam in lams}
            for layer in layers
        }
        glob = min(lams, key=lambda lam: _rank(last["mean"][str(lam)]["fill"]))
        choice = {}
        for layer in layers:
            if args.lambda_mode == "global":
                choice[layer] = glob
            else:
                choice[layer] = min(
                    lams, key=lambda lam: _rank(scores[layer][lam]["fill"])
                )
        maps = {layer: final[layer][choice[layer]] for layer in layers}
        info = {}
        for layer in layers:
            score = scores[layer][choice[layer]]
            info[layer] = {
                "shape": [int(maps[layer][0].shape[0]), int(maps[layer][0].shape[1])],
                "lambda": choice[layer],
                "fit": {
                    "rel_rms": score["train"],
                    "rows": rows,
                    "held_rel_rms_fp16": score["held"],
                    "fill_rel_rms_fp16": score["fill"],
                },
            }
        head = {
            "tail": int(meta.get("tail", 2048)),
            "lambda": glob,
            "lambda_mode": args.lambda_mode,
            "fit": {
                "capture": str(cap.resolve()),
                "snapshot": snapshots[-1]["file"],
                "prompts": int(snapshots[-1]["prompts"]),
                "rows": rows,
                "exclude_head": args.exclude_head,
                "estimator": report["estimator"],
                "selection": "lowest held-out fill rel RMS at this snapshot",
            },
        }
        path = write_bundle(out / f"bundle_s{split}", split, maps, info, pins, head)
        entry["bundle"] = {
            "path": str(path),
            "lambda_mode": args.lambda_mode,
            "global_lambda": glob,
            "layer_lambda": {str(k): v for k, v in choice.items()},
            "mean": {
                key: _mean([scores[layer][choice[layer]][key] for layer in layers])
                for key in ("train", "held", "fill")
            },
        }
        bundles.append((path, split, layers))
        _write_json(report_path, report)
        print(f"bundle {path}: {entry['bundle']['mean']}", flush=True)
        del final, maps

    checked = [check_bundle(path, s, layers, args.pins) for path, s, layers in bundles]
    report["checked"] = checked
    report["seconds"] = round(time.perf_counter() - started, 1)
    _write_json(report_path, report)
    done.write_text(
        json.dumps({"bundles": checked, "report": str(report_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {done} after {report['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
