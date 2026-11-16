# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Do the shipped per-rank TunableOp tables round the replicated router GEMM alike?

A TP worker on device ordinal ``r`` reads ``tunableop_results<r>.csv``. The router
GEMM of the Flash-Next MoE layers is replicated (every rank multiplies the same
``[rows, 2560]`` hidden state by the same ``[512, 2560]`` weight), so every rank must
get the same logits bit for bit. If two rank files name different rocBLAS solutions
for the key (``tn_512_<rows>_2560_ld_2560_2560_512``) the accumulation order differs
and a near-tie top-k decision can flip on one rank.

PyTorch's TunableOp keeps ONE results file per process, chosen by the device that is
current when the first GEMM runs, so every ordinal needs its own process (as in the
server, one worker per rank):

    for r in 0 1 2 3; do
      PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \\
        PYTORCH_TUNABLEOP_FILENAME=<rows dir>/tunableop_results.csv \\
        HIP_VISIBLE_DEVICES=2,3,4,5 python -m tools.rdna.eplb.tunable_bits run \\
          --ordinal $r --out a-$r.npz
    done
    python -m tools.rdna.eplb.tunable_bits compare a-*.npz [--vs b-*.npz]

``run`` multiplies the same seeded fp16 inputs and stores the logits of the first
repetition, the top-``k`` sets of all repetitions, the file TunableOp opened and the
solver it holds for each key. ``compare`` (CPU, numpy only) lists, per row count and
ordinal pair, the logits that differ bitwise and the rows whose top-``k`` set flips;
with ``--vs`` (a run with the visible devices permuted) it also checks that the same
ordinal gives the same bits on a different physical card, i.e. that the table and
not the card makes the difference.
"""

import argparse
import itertools
import sys
from typing import Any

import numpy as np

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
DEFAULT_ROWS = (3, 24, 512, 1022, 1023, 1024, 1025, 1036, 1045, 1197, 2047, 2048)


def _topk_sets(logits: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Sorted top-``k`` expert ids per row; ties go to the lower id (stable sort)."""
    order = np.argsort(-logits.astype(np.float32), axis=1, kind="stable")[:, :k]
    return np.sort(order, axis=1).astype(np.int16)


def _inputs(rows: int, rep: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded hidden state and router weight, identical in every process."""
    rng = np.random.default_rng([seed, rows, rep])
    x = rng.standard_normal((rows, HIDDEN), dtype=np.float32)
    w = np.random.default_rng([seed, 7]).standard_normal(
        (EXPERTS, HIDDEN), dtype=np.float32
    )
    return x.astype(np.float16), (w / np.sqrt(HIDDEN)).astype(np.float16)


def _bus(torch: Any, ordinal: int) -> int:
    bus = torch.cuda.get_device_properties(ordinal).pci_bus_id
    return int(bus.split(":")[1], 16) if isinstance(bus, str) else int(bus)


def run(args: argparse.Namespace) -> None:
    import torch

    o = args.ordinal
    # before the first GEMM: this picks the results file
    torch.accelerator.set_device_index(o)
    rows_list = [int(r) for r in args.rows.split(",")]
    out: dict[str, Any] = {
        "ordinals": np.array([o]),
        "rows": np.array(rows_list),
        "reps": np.array(args.reps),
        "buses": np.array([_bus(torch, o)]),
    }
    for rows in rows_list:
        sets = []
        for rep in range(args.reps):
            x, w = _inputs(rows, rep, args.seed)
            y = torch.nn.functional.linear(
                torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
            )
            torch.accelerator.synchronize()
            y_cpu = y.cpu().numpy()
            if rep == 0:
                out[f"logits_{o}_{rows}"] = y_cpu
            sets.append(_topk_sets(y_cpu))
        out[f"topk_{o}_{rows}"] = np.stack(sets)
    tun = torch.cuda.tunable
    out[f"tunable_file_{o}"] = np.array(tun.get_filename())
    out["tunable_enabled"] = np.array(
        [tun.is_enabled(), tun.tuning_is_enabled()], dtype=bool
    )
    held = {(r[1]): r[2] for r in tun.get_results()}
    for rows in rows_list:
        key = f"tn_{EXPERTS}_{rows}_{HIDDEN}_ld_{HIDDEN}_{HIDDEN}_{EXPERTS}"
        out[f"solver_{o}_{rows}"] = np.array(held.get(key, "(default heuristic)"))
    np.savez_compressed(args.out, **out)
    print(
        f"ordinal {o} (bus {out['buses'][0]:02x}): TunableOp file "
        f"{tun.get_filename()}, enabled/tuning {out['tunable_enabled'].tolist()}, "
        f"{len(held)} rows loaded; router solvers "
        + ", ".join(f"{r}: {str(out[f'solver_{o}_{r}'])[-6:]}" for r in rows_list)
    )


def _load_many(paths: list[str]) -> dict[str, np.ndarray]:
    """Merge single-ordinal run files into one dict (keys carry the ordinal)."""
    merged: dict[str, np.ndarray] = {}
    ordinals: list[int] = []
    buses: list[int] = []
    for path in paths:
        with np.load(path) as z:
            ordinals += [int(v) for v in z["ordinals"]]
            buses += [int(v) for v in z["buses"]]
            for name in z.files:
                if name not in ("ordinals", "buses"):
                    merged[name] = z[name]
    merged["ordinals"] = np.array(ordinals)
    merged["buses"] = np.array(buses)
    return merged


def _pair_stats(a: dict, b: dict, oa: int, ob: int, rows: int) -> dict:
    ya, yb = a[f"logits_{oa}_{rows}"], b[f"logits_{ob}_{rows}"]
    diff = ya.view(np.int16) != yb.view(np.int16)
    ta, tb = a[f"topk_{oa}_{rows}"], b[f"topk_{ob}_{rows}"]
    flips = (ta != tb).any(axis=2)
    return {
        "elems": int(diff.sum()),
        "of": int(diff.size),
        "max_abs": float(
            np.abs(ya.astype(np.float32) - yb.astype(np.float32)).max(initial=0.0)
        ),
        "flip_rows": int(flips.sum()),
        "rows_total": int(flips.size),
    }


def compare(args: argparse.Namespace) -> dict:
    a = _load_many(args.a)
    ordinals = [int(o) for o in a["ordinals"]]
    rows_list = [int(r) for r in a["rows"]]
    report: dict[str, Any] = {"pairs": {}, "cards": {}}
    print(f"buses {[f'{int(b):02x}' for b in a['buses']]} ordinals {ordinals}")
    for o in ordinals:
        print(f"  ordinal {o}: {a[f'tunable_file_{o}']!s}")
    print("  router solver per ordinal (the rows that differ):")
    for rows in rows_list:
        held = [str(a[f"solver_{o}_{rows}"]).rsplit("_", 1)[-1] for o in ordinals]
        if len(set(held)) > 1:
            print(
                f"    rows {rows}: "
                + ", ".join(f"{o}: {s}" for o, s in zip(ordinals, held))
            )
    print("  rows  ordinal pair  differing logits         max|d|  top-k set flips")
    for rows in rows_list:
        for oa, ob in itertools.combinations(ordinals, 2):
            s = _pair_stats(a, a, oa, ob, rows)
            report["pairs"][f"{rows}:{oa}-{ob}"] = s
            print(
                f"  {rows:5d}  {oa}-{ob}           {s['elems']:7d} of {s['of']:8d}"
                f"  {s['max_abs']:9.5f}  {s['flip_rows']:5d} of {s['rows_total']}"
            )
    if args.vs:
        b = _load_many(args.vs)
        cards = [f"{int(x):02x}" for x in b["buses"]]
        print(f"visiting the same ordinals on buses {cards}:")
        for rows in rows_list:
            for o in ordinals:
                if f"logits_{o}_{rows}" not in b:
                    continue
                s = _pair_stats(a, b, o, o, rows)
                report["cards"][f"{rows}:{o}"] = s
                if s["elems"]:
                    print(f"  rows {rows} ordinal {o}: {s['elems']} logits differ")
        bad = [k for k, s in report["cards"].items() if s["elems"]]
        print(
            "  same table, other card: "
            + ("IDENTICAL bits everywhere" if not bad else f"DIFFERS at {bad}")
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: logits and top-k sets of one ordinal")
    r.add_argument("--ordinal", type=int, required=True)
    r.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)))
    r.add_argument("--reps", type=int, default=16)
    r.add_argument("--seed", type=int, default=20261116)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="CPU: bitwise comparison of run outputs")
    c.add_argument("a", nargs="+")
    c.add_argument("--vs", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
