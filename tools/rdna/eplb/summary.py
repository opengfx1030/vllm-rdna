# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read, check and summarise the ``eplb-stats/1`` dumps of the lean counters.

    python -m tools.rdna.eplb.summary DIR [--window 1000] [--json out.json]
                                          [--export-stats stats.json]

``DIR`` holds one ``eplb_stats-rank<R>-pid<P>.npz`` per rank (written by
``vllm/model_executor/layers/fused_moe/rdna_eplb_stats.py``). One file has:

``meta``     JSON: format, rank, ep_size, num_experts, top_k, decode_max_rows,
             band_edges, ring, layers [{name, kind, layer}], owner [L][E] (the placement
             in force: rank of each global expert), pid, start/snapshot unix time;
``pairs``    int64 [L, 2, E]  routed pairs per global expert, phase 0 = decode (valid
             rows <= decode_max_rows) / 1 = prefill;
``touch``    int64 [L, NB, E] decode steps that touched the expert, per band of valid
             rows (band ``b`` = rows in ``(edge[b-1], edge[b]]``);
``calls``    int64 [L, NB + 1] steps per band, last column = prefill calls;
``rows``     int64 [L, 2]     valid rows summed per phase;
``ring``     int32 [N, 32]    the newest records of ALL layers in call order (record
             ``k`` sits in slot ``k % N``; ``ring_count`` = records ever written),
             see the ``W_*`` word indices; ``T``/``P`` words are per rank of the
             placement in force, so ONE rank's ring holds every rank's load;
``hist*``    optional prefill histograms ring ``hist`` int16 [L, H, E] (clipped at
             32767), ``hist_calls`` [L], ``hist_rows`` [L, H]
             (``traces.eplb_stats_hist``).

Every rank computes its counters from the same all-gathered top-k ids, so the id hash
word must agree across ranks for every call; ``consensus`` checks it (a mismatch means
the ranks routed differently, which per-step replica routing could not survive).
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from tools.rdna.eplb import loads
from tools.rdna.hetero_moe.stats_io import LayerStats, StatsRecord

FORMAT = "eplb-stats/1"
WORDS = 32
(W_LAYER, W_ROWS, W_TOTAL, W_HASH, W_TS_LO, W_TS_HI, W_PHASE, W_CALL) = range(8)
W_T0, W_P0, MAX_RANKS = 8, 16, 8
DEFAULT_BAND_EDGES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
DEFAULT_RATIO_TARGET = 1.09  # DeepInfer's slowest/mean target


@dataclass
class RankStats:
    path: str
    meta: dict
    pairs: np.ndarray
    touch: np.ndarray
    calls: np.ndarray
    rows: np.ndarray
    ring: np.ndarray  # [n, 32] int32, oldest first
    ring_total: int  # records ever written

    @property
    def rank(self) -> int:
        return int(self.meta["rank"])

    @property
    def ranks(self) -> int:
        return int(self.meta["ep_size"])

    @property
    def edges(self) -> tuple[int, ...]:
        return tuple(self.meta["band_edges"])

    @property
    def layers(self) -> list[dict]:
        return self.meta["layers"]

    @property
    def first_record(self) -> int:
        return self.ring_total - self.ring.shape[0]


def band_of(rows: np.ndarray | int, edges: tuple[int, ...]) -> np.ndarray | int:
    """Band index of a valid-row count (first edge >= rows); ``len(edges)`` if above."""
    return np.searchsorted(np.asarray(edges), rows, side="left")


def load_rank(path: str) -> RankStats:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not an {FORMAT} file ({meta.get('format')!r})")
        ring_all, total = z["ring"], int(z["ring_count"])
        n = ring_all.shape[0]
        keep = min(total, n)
        order = [(total - keep + i) % n for i in range(keep)]
        return RankStats(
            path,
            meta,
            z["pairs"],
            z["touch"],
            z["calls"],
            z["rows"],
            ring_all[order],
            total,
        )


def load_dir(path: str) -> list[RankStats]:
    files = sorted(glob.glob(os.path.join(path, "eplb_stats-rank*-pid*.npz")))
    stats = sorted((load_rank(f) for f in files), key=lambda s: s.rank)
    if not stats:
        raise FileNotFoundError(f"no eplb_stats-rank*-pid*.npz in {path}")
    return stats


def ring_table(rs: RankStats) -> dict[str, np.ndarray]:
    """The ring as columns: ``layer, rows, total, phase, call, hash, ts, T, P``."""
    r, ranks = rs.ring, rs.ranks
    ts = (r[:, W_TS_HI].astype(np.uint64) << 32) | r[:, W_TS_LO].astype(np.uint32)
    return {
        "k": rs.first_record + np.arange(r.shape[0]),
        "layer": r[:, W_LAYER],
        "rows": r[:, W_ROWS],
        "total": r[:, W_TOTAL],
        "phase": r[:, W_PHASE],
        "call": r[:, W_CALL],
        "hash": r[:, W_HASH].astype(np.uint32),
        "ts": ts,
        "T": r[:, W_T0 : W_T0 + ranks].astype(np.float64),
        "P": r[:, W_P0 : W_P0 + ranks].astype(np.float64),
    }


def consensus(stats: list[RankStats]) -> dict:
    """Do all ranks agree on the id hash of every call both rings still hold?

    A disagreement adds ``detail`` (``_consensus_detail``): which rank is off the
    majority, in which phase, at which row counts, and whether the loads differ too.
    """
    if len(stats) < 2:
        return {"ranks": len(stats), "compared": 0, "mismatches": 0, "ok": True}
    lo = max(s.first_record for s in stats)
    hi = min(s.ring_total for s in stats)
    if hi <= lo:
        return {"ranks": len(stats), "compared": 0, "mismatches": 0, "ok": None}
    cols = [
        ring_table(s)["hash"][lo - s.first_record : hi - s.first_record] for s in stats
    ]
    bad = np.zeros(hi - lo, dtype=bool)
    for c in cols[1:]:
        bad |= c != cols[0]
    idx = np.flatnonzero(bad)
    out: dict = {
        "ranks": len(stats),
        "compared": int(hi - lo),
        "mismatches": int(idx.size),
        "first_mismatch_record": int(lo + idx[0]) if idx.size else None,
        "ok": not idx.size,
    }
    if idx.size:
        out["detail"] = _consensus_detail(stats, cols, lo, idx)
    return out


def _consensus_detail(
    stats: list[RankStats], cols: list[np.ndarray], lo: int, idx: np.ndarray
) -> dict:
    """Who deviates and how, over the records ``lo + idx`` whose hashes differ.

    ``odd_rank``: records where a rank is off the majority hash (None below 3 ranks,
    where no majority exists); ``loads_differ``: the ``T``/``P`` words differ too (an
    expert changed owner); ``hash_only``: same loads (positions swapped in a row, or
    an expert replaced by one of the same owner); ``passes``: groups of records
    more than one layer sweep apart.
    """
    h = np.stack([c[idx] for c in cols])  # [ranks, mismatching records]
    agree = (h[:, None, :] == h[None, :, :]).sum(axis=1)
    major = h[np.argmax(agree, axis=0), np.arange(idx.size)]
    tabs = [ring_table(s) for s in stats]
    at = [lo - s.first_record + idx for s in stats]

    def words(name: str) -> np.ndarray:
        return np.stack([t[name][a] for t, a in zip(tabs, at)])

    t_all, p_all = words("T"), words("P")
    rows, phase = words("rows")[0], words("phase")[0]
    differ = (p_all != p_all[0]).any(axis=(0, 2)) | (t_all != t_all[0]).any(axis=(0, 2))
    values, counts = np.unique(rows, return_counts=True)
    top = np.argsort(-counts, kind="stable")[:8]
    return {
        "odd_rank": (
            {
                str(s.rank): int((h[i] != major).sum())
                for i, s in enumerate(stats)
                if (h[i] != major).any()
            }
            if len(stats) > 2
            else None
        ),
        "phase": {
            name: int((phase == v).sum())
            for v, name in ((0, "decode"), (1, "prefill"))
            if (phase == v).any()
        },
        "rows": {str(int(values[j])): int(counts[j]) for j in top},
        "loads_differ": int(differ.sum()),
        "hash_only": int((~differ).sum()),
        "passes": 1 + int((np.diff(idx) > len(stats[0].layers)).sum()),
    }


def _ratio(x: np.ndarray) -> float | None:
    return loads.max_over_mean(x) if x.shape[0] else None


def layer_summary(rs: RankStats, pos: int, window: int) -> dict:
    """Per-layer max/mean of the rank loads from the ring (this placement)."""
    t = ring_table(rs)
    sel = t["layer"] == pos
    edges = rs.edges
    out: dict = {"layer": rs.layers[pos]["layer"], "name": rs.layers[pos]["name"]}
    dec = sel & (t["phase"] == 0)
    pre = sel & (t["phase"] == 1)
    bands: dict[str, dict] = {}
    for b in np.unique(band_of(t["rows"][dec], edges)):
        m = dec & (band_of(t["rows"], edges) == b)
        label = f"<={edges[b]}" if b < len(edges) else f">{edges[-1]}"
        bands[label] = {
            "steps": int(m.sum()),
            "rows_mean": float(t["rows"][m].mean()),
            "T_max_over_mean": _ratio(t["T"][m]),
            "P_max_over_mean": _ratio(t["P"][m]),
            "T_mean_per_rank": float(t["T"][m].mean()),
        }
    out["decode_bands"] = bands
    out["prefill"] = {
        "calls": int(pre.sum()),
        "P_max_over_mean": _ratio(t["P"][pre]),
        "P_max_over_mean_median_call": (
            float(
                np.median(
                    t["P"][pre].max(axis=1) / np.maximum(t["P"][pre].mean(axis=1), 1e-9)
                )
            )
            if pre.any()
            else None
        ),
    }
    out["windows"] = {
        "decode_T": _windows(t["T"][dec], window),
        "prefill_P": _windows(t["P"][pre], max(window // 10, 1)),
    }
    return out


def _windows(x: np.ndarray, window: int) -> dict:
    """DeepInfer's check: slowest/mean over consecutive windows of ``window`` steps."""
    n = x.shape[0] // window
    if n == 0:
        return {"windows": 0}
    r = [loads.max_over_mean(x[i * window : (i + 1) * window]) for i in range(n)]
    return {
        "windows": n,
        "window_steps": window,
        "median": float(np.median(r)),
        "max": float(np.max(r)),
        "within_target": float(np.mean(np.asarray(r) <= DEFAULT_RATIO_TARGET)),
    }


def topn_share(
    counts: np.ndarray, tops: tuple[int, ...] = (8, 16, 32, 64, 128)
) -> dict:
    c = np.sort(counts)[::-1].astype(np.float64)
    total = c.sum()
    return {str(n): float(c[:n].sum() / total) if total else 0.0 for n in tops}


def summarize(stats: list[RankStats], window: int = 1000) -> dict:
    """One JSON-able report: consensus, per-layer ratios, skew, counter sanity."""
    rs = stats[0]
    layers = [layer_summary(rs, pos, window) for pos in range(len(rs.layers))]
    for pos, entry in enumerate(layers):
        entry["decode_top_share"] = topn_share(rs.pairs[pos, 0])
        entry["decode_steps"] = int(rs.calls[pos, :-1].sum())
        entry["prefill_calls"] = int(rs.calls[pos, -1])
    return {
        "format": "eplb-summary/1",
        "ep_size": rs.ranks,
        "num_experts": int(rs.meta["num_experts"]),
        "band_edges": list(rs.edges),
        "ring_records": int(rs.ring.shape[0]),
        "consensus": consensus(stats),
        "layers": layers,
    }


def touch_weight(rs: RankStats, pos: int, rows: int) -> tuple[np.ndarray, int]:
    """Measured P(expert touched in a decode step of ``rows`` rows) and its step count.

    This is the decode objective straight from the counters (no row-independence
    assumption): the touch frequency of the band that holds ``rows``.
    """
    b = int(band_of(rows, rs.edges))
    n = int(rs.calls[pos, b]) if b < rs.touch.shape[1] else 0
    if n == 0:
        return np.zeros(rs.touch.shape[2]), 0
    return rs.touch[pos, b].astype(np.float64) / n, n


def to_stats_record(rs: RankStats, top_k: int | None = None) -> StatsRecord:
    """The marginals as ``hetero-moe-expert-stats/1`` (solver/simulator input)."""
    layers: dict[int, LayerStats] = {}
    for pos, info in enumerate(rs.layers):
        dec = rs.pairs[pos, 0].astype(np.int64)
        pre = rs.pairs[pos, 1].astype(np.int64)
        layers[int(info["layer"])] = LayerStats(
            int(info["layer"]),
            info["kind"],
            info["name"],
            dec.astype(np.float64),
            pre.astype(np.float64),
            dec,
            pre,
            float(rs.rows[pos, 0]),
            float(rs.rows[pos, 1]),
        )
    return StatsRecord(
        int(rs.meta["num_experts"]),
        top_k or int(rs.meta["top_k"]),
        layers,
        {"num_experts": int(rs.meta["num_experts"]), "top_k": int(rs.meta["top_k"])},
        {"source": rs.path},
        {
            "decode": float(rs.calls[:, :-1].sum(axis=1).max()),
            "prefill": float(rs.calls[:, -1].max()),
        },
    )


def format_table(report: dict, target: float = DEFAULT_RATIO_TARGET) -> str:
    lines = [
        f"EP{report['ep_size']}  consensus: {report['consensus']}",
        f"{'layer':>5} {'rows':>7} {'steps':>7} {'T max/mean':>11} {'P max/mean':>11} "
        f"{'windows<=' + str(target):>13}",
    ]
    for ent in report["layers"]:
        for label, b in ent["decode_bands"].items():
            w = ent["windows"]["decode_T"]
            ok = f"{100 * w['within_target']:.0f}%" if w.get("windows") else "-"
            lines.append(
                f"{ent['layer']:>5} {label:>7} {b['steps']:>7} "
                f"{b['T_max_over_mean']:>11.3f} {b['P_max_over_mean']:>11.3f} {ok:>13}"
            )
        p = ent["prefill"]
        if p["calls"]:
            lines.append(
                f"{ent['layer']:>5} {'prefill':>7} {p['calls']:>7} {'':>11} "
                f"{p['P_max_over_mean']:>11.3f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--export-stats", help="write hetero-moe-expert-stats/1 (rank 0)")
    a = ap.parse_args(argv)
    stats = load_dir(a.dir)
    report = summarize(stats, a.window)
    print(format_table(report))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    if a.export_stats:
        from tools.rdna.hetero_moe.stats_io import write_json

        write_json(to_stats_record(stats[0]), a.export_stats)
    return 0 if report["consensus"]["ok"] is not False else 2


if __name__ == "__main__":
    sys.exit(main())
