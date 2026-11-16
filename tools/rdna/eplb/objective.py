# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which load does MoE kernel time follow: distinct experts (T) or pairs (P)?

    python -m tools.rdna.eplb.objective --stats DIR --trace ROCPROF_DIR [--rows 24]

A decode MoE kernel reads the weights of every touched expert once, so its time should
follow ``T_r`` (touched experts of rank ``r``), not ``P_r`` (pairs); prefill follows
``P_r``. Which one a placement must balance decides what the solver packs, so phase 0
measures it instead of assuming it: the ``eplb-stats/1`` ring gives ``T_r``/``P_r`` of
every MoE call, a rocprofv3 kernel trace (``--kernel-trace``, a server booted with
``CG_MODE=NONE`` so every kernel is a separate dispatch) gives the time of the same
call.

The join is by call order and refuses to guess: the process must have been traced from
its first MoE call, the ring must not have wrapped (``ring_count <= ring size``), and
the number of ``moe_*_gemm_rdna2`` dispatches must be exactly ``kernels_per_call`` times
the number of ring records. Anything else raises :class:`JoinError` with the counts.

Per call the kernel time ``y`` (sum of the w1 and w2 GEMMs) is regressed on the rank's
own load: ``y = a_layer + b * T + c * P`` (layer means removed), also ``T`` alone and
``P`` alone. ``b`` should come out near the per-expert weight-read time (about 6.7 us
for the 2.5 MB W4A16 expert at HBM speed) and ``c`` near zero in decode. ``T`` and ``P``
are strongly correlated at small widths, so the report carries ``corr(T, P)``: above
0.95 the two-variable split is ill conditioned and the single-variable R^2 comparison is
the evidence.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import regex as re

from tools.rdna.eplb import summary

KERNEL_RE = re.compile(r"moe_(?:gptq|w4a8)_gemm_rdna2")


class JoinError(RuntimeError):
    """The kernel trace and the ring cannot be matched call for call."""


def read_kernels(path: str, pattern=KERNEL_RE) -> np.ndarray:
    """``[n, 2]`` int64 (start_ns, end_ns) of the matching dispatches, by start time."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if pattern.search(r["Kernel_Name"]):
                rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
    return np.array(sorted(rows), dtype=np.int64).reshape(-1, 2)


def call_times_us(kernels: np.ndarray, per_call: int) -> np.ndarray:
    """Busy time (us) of consecutive groups of ``per_call`` dispatches."""
    if kernels.shape[0] % per_call:
        raise JoinError(
            f"{kernels.shape[0]} dispatches is not a multiple of {per_call} per call"
        )
    dur = (kernels[:, 1] - kernels[:, 0]).astype(np.float64) / 1e3
    return dur.reshape(-1, per_call).sum(axis=1)


def join(rs: summary.RankStats, times_us: np.ndarray) -> dict[str, np.ndarray]:
    """Per-call arrays ``layer, rows, phase, T, P, y`` of this rank (strict join)."""
    if rs.first_record != 0:
        raise JoinError(
            f"the ring wrapped ({rs.ring_total} records, {rs.ring.shape[0]} kept): "
            "trace a shorter window or enlarge VLLM_RDNA_EPLB_STATS_RING"
        )
    if times_us.shape[0] != rs.ring_total:
        raise JoinError(
            f"{times_us.shape[0]} traced MoE calls vs {rs.ring_total} ring records "
            "(trace the process from its first MoE call; check --kernels-per-call)"
        )
    t = summary.ring_table(rs)
    r = rs.rank
    return {
        "layer": t["layer"],
        "rows": t["rows"],
        "phase": t["phase"],
        "T": t["T"][:, r],
        "P": t["P"][:, r],
        "Tmax": t["T"].max(axis=1),
        "Pmax": t["P"].max(axis=1),
        "y": times_us,
    }


def _r2(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    ss = float(((y - y.mean()) ** 2).sum())
    return coef, (1.0 - float(((y - x @ coef) ** 2).sum()) / ss) if ss > 0 else 0.0


def fit(
    data: dict[str, np.ndarray], *, rows: int | None = None, decode: bool = True
) -> dict:
    """Regress kernel time on the rank's own T and P (layer means removed)."""
    sel = data["phase"] == (0 if decode else 1)
    if rows is not None:
        sel &= data["rows"] == rows
    if sel.sum() < 30:
        return {"calls": int(sel.sum()), "error": "too few calls"}
    layer = data["layer"][sel]
    cols = {k: data[k][sel].astype(np.float64) for k in ("y", "T", "P", "Tmax", "Pmax")}
    for k in cols:  # remove the per-layer mean: layers differ in weights and routing
        for pos in np.unique(layer):
            m = layer == pos
            cols[k][m] -= cols[k][m].mean()
    y, t, p = cols["y"], cols["T"], cols["P"]
    both, r2_both = _r2(y, np.stack([t, p], axis=1))
    (bt,), r2_t = _r2(y, t[:, None])
    (cp,), r2_p = _r2(y, p[:, None])
    corr = float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else 1.0
    return {
        "calls": int(sel.sum()),
        "us_per_touched_expert_T_only": float(bt),
        "us_per_pair_P_only": float(cp),
        "us_per_touched_expert_joint": float(both[0]),
        "us_per_pair_joint": float(both[1]),
        "r2_T": r2_t,
        "r2_P": r2_p,
        "r2_T_and_P": r2_both,
        "corr_T_P": corr,
        "follows": "T"
        if r2_t > r2_p + 0.02
        else "P"
        if r2_p > r2_t + 0.02
        else "unclear",
        "ill_conditioned": corr > 0.95,
        "y_std_us": float(y.std()),
    }


def _pid(path: str) -> int | None:
    m = re.match(r"(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def run(stats_dir: str, trace_dir: str, per_call: int, rows: int | None) -> dict:
    stats = summary.load_dir(stats_dir)
    csvs = {
        _pid(p): p
        for p in glob.glob(
            os.path.join(trace_dir, "**", "*kernel_trace.csv"), recursive=True
        )
    }
    out: dict = {"per_call": per_call, "ranks": {}}
    for rs in stats:
        pid = int(rs.meta["pid"])
        if pid not in csvs:
            raise JoinError(
                f"rank {rs.rank}: no kernel trace for pid {pid} in {trace_dir}"
            )
        times = call_times_us(read_kernels(csvs[pid]), per_call)
        data = join(rs, times)
        out["ranks"][str(rs.rank)] = {
            "decode": fit(data, rows=rows, decode=True),
            "prefill": fit(data, decode=False),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", required=True, help="eplb-stats/1 directory")
    ap.add_argument("--trace", required=True, help="rocprofv3 output directory")
    ap.add_argument("--kernels-per-call", type=int, default=2, help="w1 and w2 GEMM")
    ap.add_argument(
        "--rows", type=int, default=None, help="only decode calls with this many rows"
    )
    a = ap.parse_args(argv)
    try:
        report = run(a.stats, a.trace, a.kernels_per_call, a.rows)
    except JoinError as e:
        print(f"join refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
