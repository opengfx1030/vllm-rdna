# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read, window, merge and convert routing statistics.

Two on-disk forms:

* raw ``expert-stats/1`` JSONL written by ``vllm/.../rdna_expert_stats.py``
  (cumulative counters, one snapshot per interval, optional ``.pairs.npz``);
* converted ``hetero-moe-expert-stats/1`` JSON (+ optional ``.coact.npz``), the
  input of the solver and the simulator.

CLI::

    python -m tools.rdna.hetero_moe.stats_io convert RAW.jsonl[:WEIGHT] ... \\
        --out stats.json [--t0 S] [--t1 S] [--engine-commit SHA] [--workload W]
    python -m tools.rdna.hetero_moe.stats_io show stats.json|RAW.jsonl [--top 8,32,64]

Merging is an elementwise sum; a ``:WEIGHT`` suffix scales one file's counts
(the result keeps the raw integer sum in ``counts`` and the weighted floats in
``weight``). ``--t0/--t1`` select a window by ``elapsed_s`` of the raw
snapshots (difference of two cumulative snapshots); the pairs file is
cumulative at its last write and cannot be windowed.

``counts[e]`` is the number of token rows whose top-k set contains ``e``; for a
layer the decode counts sum to ``top_k * tokens``. The MTP layer (kind ``mtp``)
gets output layer index ``num_target_layers + idx``.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FORMAT = "hetero-moe-expert-stats/1"
RAW_SCHEMA = "expert-stats/1"
PHASES = ("decode", "prefill")


@dataclass
class LayerStats:
    """Weighted per-expert row counts of one layer (output layer index)."""

    layer: int
    kind: str
    name: str | None
    decode: np.ndarray  # float64 [E], weighted counts
    prefill: np.ndarray
    decode_raw: np.ndarray  # int64 [E], unweighted sum
    prefill_raw: np.ndarray
    decode_tokens: float = 0.0  # weighted rows
    prefill_tokens: float = 0.0


@dataclass
class StatsRecord:
    num_experts: int
    top_k: int
    layers: dict[int, LayerStats]
    model: dict[str, Any] = field(default_factory=dict)
    window: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, float] = field(default_factory=lambda: {"decode": 0, "prefill": 0})
    rows_hist: dict[int, float] = field(default_factory=dict)  # decode rows/step
    coact: np.ndarray | None = None  # float32 [n_layers, E, E], decode, a < b
    coact_layers: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # sha256 of every input

    def layer_ids(self, kind: str | None = None) -> list[int]:
        return [i for i, s in sorted(self.layers.items()) if kind in (None, s.kind)]


# ---- raw JSONL ---------------------------------------------------------------


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_raw(path: str) -> tuple[dict, list[dict]]:
    """(merged meta, snapshots) of a raw JSONL file; tolerates a cut last line."""
    meta: dict[str, Any] = {}
    layers: dict[str, dict] = {}
    snaps: list[dict] = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}:{i + 1}: malformed JSON") from None
        if obj.get("type") == "meta":
            if obj.get("schema") != RAW_SCHEMA:
                raise ValueError(f"{path}: unsupported schema {obj.get('schema')!r}")
            meta.update({k: v for k, v in obj.items() if k != "layers"})
            for layer in obj["layers"]:
                layers[layer["name"]] = layer
        elif obj.get("type") == "snapshot":
            snaps.append(obj)
    if not meta:
        raise ValueError(f"{path}: no meta line")
    if not snaps:
        raise ValueError(f"{path}: no snapshot (the server never reached a dump)")
    meta["layers"] = list(layers.values())
    return meta, snaps


def _pick(snaps: list[dict], t: float | None, default_last: bool) -> dict | None:
    """Last snapshot with ``elapsed_s <= t``; the final one for ``t is None``."""
    if t is None:
        return snaps[-1] if default_last else None
    chosen = None
    for snap in snaps:
        if snap["elapsed_s"] <= t:
            chosen = snap
    return chosen


def _layer_out_index(meta: dict) -> dict[str, int]:
    targets = [x for x in meta["layers"] if x["kind"] == "target"]
    n_target = len(targets)
    out = {}
    for layer in meta["layers"]:
        idx = layer["idx"]
        out[layer["name"]] = idx if layer["kind"] == "target" else n_target + idx
    return out


@dataclass
class RawWindow:
    meta: dict
    counts: dict[str, dict[str, np.ndarray]]  # layer name -> phase -> int64 [E]
    tokens: dict[str, dict[str, int]]
    steps: dict[str, dict[str, int]]
    rows_hist: dict[str, np.ndarray]
    t0: float
    t1: float
    wall_t0: float
    wall_t1: float
    pairs: dict[str, np.ndarray] | None


def load_raw(path: str, t0: float | None = None, t1: float | None = None) -> RawWindow:
    meta, snaps = parse_raw(path)
    end = _pick(snaps, t1, True)
    if end is None:
        raise ValueError(f"{path}: no snapshot at or before t1={t1}")
    start = _pick(snaps, t0, False)
    names = list(end["counts"])

    def diff(field_name: str, name: str, phase: str | None = None):
        def get(s: dict | None):
            if s is None or name not in s[field_name]:
                return None
            v = s[field_name][name]
            return v[phase] if phase else v

        e, b = get(end), get(start)
        if b is None:
            return e
        if isinstance(e, list):
            return [x - y for x, y in zip(e, b, strict=True)]
        return e - b

    counts, tokens, steps, hist = {}, {}, {}, {}
    for name in names:
        counts[name] = {
            p: np.asarray(diff("counts", name, p), dtype=np.int64) for p in PHASES
        }
        tokens[name] = {p: int(diff("tokens", name, p)) for p in PHASES}
        steps[name] = {p: int(diff("steps", name, p)) for p in PHASES}
        hist[name] = np.asarray(diff("rows_hist", name), dtype=np.int64)
        for p in PHASES:
            if (counts[name][p] < 0).any():
                raise ValueError(f"{path}: counters decreased in {name} {p}")
    pairs = None
    npz = path[: -len(".jsonl")] + ".pairs.npz" if path.endswith(".jsonl") else None
    if npz and os.path.exists(npz):
        data = np.load(npz)
        pairs = {n: data["pairs"][i] for i, n in enumerate(data["names"].tolist())}
    return RawWindow(
        meta,
        counts,
        tokens,
        steps,
        hist,
        start["elapsed_s"] if start else 0.0,
        end["elapsed_s"],
        start["time"] if start else meta["time"],
        end["time"],
        pairs,
    )


def _reference_layer(window: RawWindow, out_idx: dict[str, int]) -> str:
    """The lowest-index target layer: its calls are the step count."""
    kinds = {x["name"]: x["kind"] for x in window.meta["layers"]}
    targets = [n for n in window.counts if kinds.get(n) == "target"]
    return min(targets or list(window.counts), key=lambda n: out_idx[n])


def merge_raw(
    windows: list[tuple[RawWindow, float]], sources: list[str] | None = None
) -> StatsRecord:
    """Elementwise sum of (window, weight) pairs into one record."""
    if not windows:
        raise ValueError("no statistics to merge")
    first = windows[0][0].meta
    num_experts, top_k = first["num_experts"], first["top_k"]
    layers: dict[int, LayerStats] = {}
    rows_hist: dict[int, float] = {}
    steps = {"decode": 0.0, "prefill": 0.0}
    coact_acc: dict[int, np.ndarray] = {}
    coact_w = 0.0
    names: dict[int, tuple[str, str]] = {}
    model = dict(first.get("model") or {})
    for win, weight in windows:
        if win.meta["num_experts"] != num_experts or win.meta["top_k"] != top_k:
            raise ValueError("cannot merge statistics of different geometry")
        out_idx = _layer_out_index(win.meta)
        kinds = {x["name"]: x["kind"] for x in win.meta["layers"]}
        for name, per_phase in win.counts.items():
            li = out_idx[name]
            names[li] = (name, kinds[name])
            ls = layers.get(li)
            if ls is None:
                zero = np.zeros(num_experts)
                zero_i = np.zeros(num_experts, dtype=np.int64)
                ls = layers[li] = LayerStats(
                    li, kinds[name], name, zero, zero.copy(), zero_i, zero_i.copy()
                )
            ls.decode = ls.decode + weight * per_phase["decode"]
            ls.prefill = ls.prefill + weight * per_phase["prefill"]
            ls.decode_raw = ls.decode_raw + per_phase["decode"]
            ls.prefill_raw = ls.prefill_raw + per_phase["prefill"]
            ls.decode_tokens += weight * win.tokens[name]["decode"]
            ls.prefill_tokens += weight * win.tokens[name]["prefill"]
        ref = _reference_layer(win, out_idx)
        for p in PHASES:
            steps[p] += weight * win.steps[ref][p]
        hist = win.rows_hist[ref]
        for rows, n in enumerate(hist.tolist()):
            if n and rows > 0:
                rows_hist[rows] = rows_hist.get(rows, 0.0) + weight * n
        if win.pairs is not None:
            coact_w += weight
            for name, mat in win.pairs.items():
                li = out_idx[name]
                coact_acc[li] = coact_acc.get(li, 0.0) + weight * mat.astype(np.float64)
    n_target = sum(1 for s in layers.values() if s.kind == "target")
    model.update({"num_layers": n_target, "num_experts": num_experts, "top_k": top_k})
    model.setdefault("mtp_layers", sum(1 for s in layers.values() if s.kind == "mtp"))
    model["mtp_layers"] = sum(1 for s in layers.values() if s.kind == "mtp")
    record = StatsRecord(
        num_experts,
        top_k,
        dict(sorted(layers.items())),
        model,
        {
            "t0": min(w.wall_t0 for w, _ in windows),
            "t1": max(w.wall_t1 for w, _ in windows),
        },
        steps,
        rows_hist,
        sources=sources or [],
    )
    if coact_acc:
        order = sorted(coact_acc)
        record.coact_layers = order
        record.coact = np.stack([coact_acc[i] for i in order]).astype(np.float32)
    return record


# ---- converted JSON ----------------------------------------------------------


def _weights_differ(ls: LayerStats) -> bool:
    return not (
        np.array_equal(ls.decode, ls.decode_raw)
        and np.array_equal(ls.prefill, ls.prefill_raw)
    )


def to_json(record: StatsRecord, coact_file: str | None = None) -> dict:
    layers = []
    for li, ls in record.layers.items():
        entry: dict[str, Any] = {"layer": li, "kind": ls.kind}
        for p, raw, weighted, toks in (
            ("decode", ls.decode_raw, ls.decode, ls.decode_tokens),
            ("prefill", ls.prefill_raw, ls.prefill, ls.prefill_tokens),
        ):
            phase: dict[str, Any] = {"counts": raw.tolist(), "tokens": toks}
            if _weights_differ(ls):
                phase["weight"] = [round(float(x), 6) for x in weighted]
            entry[p] = phase
        layers.append(entry)
    ref = next(iter(record.layers.values()))
    out: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "model": record.model,
        "window": record.window,
        "phases": {
            "decode": {
                "steps": record.steps["decode"],
                "tokens": ref.decode_tokens,
            },
            "prefill": {
                "steps": record.steps["prefill"],
                "tokens": ref.prefill_tokens,
            },
        },
        "layers": layers,
        "step_hist": {
            "tokens_per_step": {str(k): v for k, v in sorted(record.rows_hist.items())}
        },
        "sources": record.sources,
    }
    if record.coact is not None and coact_file:
        out["coact"] = {
            "file": os.path.basename(coact_file),
            "dtype": "uint32",
            "shape": list(record.coact.shape),
            "layers": record.coact_layers,
            "phase": "decode",
        }
    return out


def write_json(record: StatsRecord, path: str) -> None:
    coact_file = None
    if record.coact is not None:
        coact_file = (
            path[: -len(".json")] + ".coact.npz"
            if path.endswith(".json")
            else path + ".coact.npz"
        )
        np.savez_compressed(
            coact_file,
            coact=np.rint(record.coact).astype(np.uint32),
            layers=np.asarray(record.coact_layers),
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(record, coact_file), f, separators=(",", ":"))
        f.write("\n")


def read_json(path: str) -> StatsRecord:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    model = doc["model"]
    num_experts, top_k = model["num_experts"], model["top_k"]
    layers: dict[int, LayerStats] = {}
    for entry in doc["layers"]:
        parts: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for p in PHASES:
            sec = entry.get(p, {})
            raw = np.asarray(sec.get("counts", [0] * num_experts), dtype=np.int64)
            weighted = np.asarray(sec.get("weight", raw), dtype=np.float64)
            parts[p] = (raw, weighted, float(sec.get("tokens", 0.0)))
        layers[entry["layer"]] = LayerStats(
            entry["layer"],
            entry.get("kind", "target"),
            None,
            parts["decode"][1],
            parts["prefill"][1],
            parts["decode"][0],
            parts["prefill"][0],
            parts["decode"][2],
            parts["prefill"][2],
        )
    phases = doc.get("phases", {})
    record = StatsRecord(
        num_experts,
        top_k,
        dict(sorted(layers.items())),
        model,
        doc.get("window", {}),
        {p: float(phases.get(p, {}).get("steps", 0)) for p in PHASES},
        {
            int(k): float(v)
            for k, v in doc.get("step_hist", {}).get("tokens_per_step", {}).items()
        },
        sources=doc.get("sources", []),
    )
    spec = doc.get("coact")
    if spec:
        npz = os.path.join(os.path.dirname(path) or ".", spec["file"])
        data = np.load(npz)
        record.coact = data["coact"].astype(np.float32)
        record.coact_layers = [int(x) for x in data["layers"]]
    return record


def load_stats(
    specs: list[str], t0: float | None = None, t1: float | None = None
) -> StatsRecord:
    """Raw JSONL (``PATH[:WEIGHT]``) files merged, or one converted JSON."""
    raws: list[tuple[RawWindow, float]] = []
    sources = []
    for spec in specs:
        path, weight = spec, 1.0
        if ":" in spec and not os.path.exists(spec):
            path, _, w = spec.rpartition(":")
            weight = float(w)
        if path.endswith(".json"):
            if len(specs) != 1:
                raise ValueError("a converted stats JSON cannot be merged")
            return read_json(path)
        raws.append((load_raw(path, t0, t1), weight))
        sources.append(_sha256(path))
    return merge_raw(raws, sources)


# ---- skew summary ------------------------------------------------------------


def skew_summary(
    record: StatsRecord, tops: tuple[int, ...] = (8, 16, 32, 64, 128)
) -> dict[str, Any]:
    """Cumulative selection share of the top-N experts per layer (decode)."""
    out: dict[str, Any] = {"tops": list(tops), "layers": {}, "mean": {}}
    acc = {n: [] for n in tops}
    for li, ls in record.layers.items():
        counts = np.sort(ls.decode if ls.decode.sum() > 0 else ls.prefill)[::-1]
        total = counts.sum()
        if total <= 0:
            continue
        share = {n: float(counts[:n].sum() / total) for n in tops}
        out["layers"][str(li)] = share
        for n in tops:
            acc[n].append(share[n])
    for n in tops:
        if acc[n]:
            out["mean"][str(n)] = float(np.mean(acc[n]))
    return out


def _cmd_convert(args: argparse.Namespace) -> None:
    record = load_stats(args.inputs, args.t0, args.t1)
    record.window.update(
        {"engine_commit": args.engine_commit, "workload": args.workload}
    )
    write_json(record, args.out)
    print(
        f"wrote {args.out}: {len(record.layers)} layers, "
        f"{record.steps['decode']:.0f} decode steps, "
        f"{record.steps['prefill']:.0f} prefill steps"
    )


def _cmd_show(args: argparse.Namespace) -> None:
    record = load_stats([args.input])
    tops = tuple(int(x) for x in args.top.split(","))
    print(json.dumps(skew_summary(record, tops), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    conv = sub.add_parser("convert", help="raw JSONL -> hetero-moe-expert-stats/1")
    conv.add_argument("inputs", nargs="+", help="PATH[:WEIGHT] raw files")
    conv.add_argument("--out", required=True)
    conv.add_argument("--t0", type=float, help="window start (elapsed_s)")
    conv.add_argument("--t1", type=float, help="window end (elapsed_s)")
    conv.add_argument("--engine-commit", default=None)
    conv.add_argument("--workload", default=None)
    conv.set_defaults(func=_cmd_convert)
    show = sub.add_parser("show", help="cumulative top-N selection share per layer")
    show.add_argument("input")
    show.add_argument("--top", default="8,16,32,64,128")
    show.set_defaults(func=_cmd_show)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
