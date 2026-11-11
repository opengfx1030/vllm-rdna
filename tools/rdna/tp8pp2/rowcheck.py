#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numerical gate for TunableOp rows: tuned vs the rocBLAS default vs an fp64 reference.

    HIP_VISIBLE_DEVICES=<n> python rowcheck.py run default CENSUS [ONLY] --out d.json
    HIP_VISIBLE_DEVICES=<n> PYTORCH_TUNABLEOP_FILENAME=<rows>/tunableop_results.csv \\
        python rowcheck.py run tuned CENSUS [ONLY] --out t.json
    python rowcheck.py compare d.json t.json [--ratio R] [--reject-out FILE] [--verbose]

Two processes, because torch's `tunable.enable(False)` does NOT bypass rows that are
already loaded: with 3330 rows loaded 392 of 556 new `TN` keys gave other bits than a
process that never enabled TunableOp, and after the toggle the bits were the rows'
again. The first version of this gate compared the tuned path with itself and passed
everything with "error ratio 1.00". The default run leaves TunableOp off (the heuristic
solution); the tuned run loads the rows the way a worker does (torch inserts the device
ordinal into the file name: `tunableop_results.csv` -> `tunableop_results0.csv` under
HIP_VISIBLE_DEVICES=<n>) and skips keys without a loaded row. Inputs are seeded by the
key, so both processes see the same fp16 data.

CENSUS lines are `GemmTunableOp_Half_TN,tn_<m>_<n>_<k>_ld_<k>_<k>_<m>` (weight [m,k],
activations [n,k]); ONLY (optional) restricts to the keys listed in it. Every key runs
on two activation sets (gaussian; gaussian with four massive channels, as in LLM
residual streams), eager and replayed from a HIP graph (how the server runs decode).
`compare` rejects a key when, against the fp64 matmul of the same fp16 inputs,

  error   the tuned max error (over max |ref|) or relative L2 error exceeds R x the
          default's (R = 1.5, no absolute slack: the fp16 rounding floor is 2.4e-4 and
          both share it),
  graph   the tuned graph result differs bitwise from the tuned eager one,
  repeat  three tuned eager runs are not bit-identical.

Keys whose tuned bits equal the default's are counted, not rejected. `--reject-out`
writes the rejected `op,key` lines for `tunableop_missing.py harvest --reject`. The
check cannot see an interaction between rows or a schedule effect: that is what the
deterministic concurrency probe is for, see docs/rdna2/tp8-pp2-flashnext-2026-11-06.md.
Run it with the venv python (regex).
"""

import json
import os
import sys
import zlib
from pathlib import Path

import regex as re


def digest(y) -> list[int]:
    """Two exact integer checksums of the fp16 bits: equal checksums = equal bits."""
    import torch

    v = y.contiguous().view(torch.int16).reshape(-1).to(torch.int64)
    idx = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 65521
    return [int(v.sum()), int((v * idx).sum())]


def take_option(argv: list[str], name: str, default=None):
    if name not in argv:
        return default
    i = argv.index(name)
    value = argv[i + 1]
    del argv[i : i + 2]
    return value


def run(argv: list[str]) -> int:
    out = Path(take_option(argv, "--out"))
    mode, census = argv[0], argv[1]
    only = None
    if len(argv) > 2:
        only = {
            tuple(line.strip().split(",")[:2])
            for line in Path(argv[2]).read_text().splitlines()
            if line.startswith("Gemm")
        }
    os.environ.update(
        PYTORCH_TUNABLEOP_ENABLED="0" if mode == "default" else "1",
        PYTORCH_TUNABLEOP_TUNING="0",
        PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED="0",
        TORCH_BLAS_PREFER_HIPBLASLT="0",
    )
    os.environ.pop("PYTORCH_TUNABLEOP_RECORD_UNTUNED", None)
    if mode == "tuned" and "PYTORCH_TUNABLEOP_FILENAME" not in os.environ:
        sys.exit("tuned: set PYTORCH_TUNABLEOP_FILENAME=<rows>/tunableop_results.csv")

    import torch
    import torch.cuda.tunable as tunable

    dev = torch.device("cuda:0")
    pat = re.compile(r"tn_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)$")
    loaded: set[tuple[str, str]] = set()
    if mode == "tuned":
        torch.nn.functional.linear(
            torch.zeros(8, 64, device=dev, dtype=torch.half),
            torch.zeros(16, 64, device=dev, dtype=torch.half),
        )
        torch.accelerator.synchronize()  # the first GEMM loads the rows file
        loaded = {(r[0], r[1]) for r in tunable.get_results()}
        print(f"rows loaded from {tunable.get_filename()}: {len(loaded)}")
        if not loaded:
            sys.exit("no TunableOp rows loaded: the tuned path would equal the default")

    def linear(x: torch.Tensor, w: torch.Tensor, graph: bool) -> torch.Tensor:
        if not graph:
            return torch.nn.functional.linear(x, w)
        xs = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                torch.nn.functional.linear(xs, w)
        torch.cuda.current_stream().wait_stream(side)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            ys = torch.nn.functional.linear(xs, w)
        captured.replay()
        torch.accelerator.synchronize()
        return ys.clone()

    keys: dict[str, dict] = {}
    skipped = 0
    for line in Path(census).read_text().splitlines():
        if not line.startswith("GemmTunableOp_Half_TN"):
            continue
        op, _, key = line.strip().partition(",")
        m = pat.match(key)
        if (only is not None and (op, key) not in only) or not m or key in keys:
            continue
        if "_16_16_16_" in key:
            continue
        M, N, K, lda, ldb, ldc = (int(v) for v in m.groups())
        if lda != K or ldb != K or ldc != M:
            continue
        if mode == "tuned" and (op, key) not in loaded:
            skipped += 1  # no row: the tuned path would be the default
            continue
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(key.encode()))
        w = (torch.randn(M, K, device=dev, generator=g) * 0.05).half()
        rec: dict[str, dict] = {}
        for flavour in ("gaussian", "outliers"):
            x = torch.randn(N, K, device=dev, generator=g)
            if flavour == "outliers":
                x[:, torch.randperm(K, device=dev, generator=g)[:4]] *= 60.0
            x = x.half()
            ref = x.double() @ w.double().t()
            scale = ref.abs().max().item() or 1.0
            y = linear(x, w, False)
            yg = linear(x, w, True)
            err = (y.double() - ref).abs()
            rec[flavour] = {
                "max": err.max().item() / scale,
                "l2": (err.norm() / ref.norm()).item(),
                "eager": digest(y),
                "graph": digest(yg),
                "repeat": all(
                    digest(linear(x, w, False)) == digest(y) for _ in range(2)
                ),
            }
        keys[key] = rec
    Path(out).write_text(
        json.dumps({"mode": mode, "loaded": len(loaded), "keys": keys})
    )
    print(f"{mode}: {len(keys)} keys ({skipped} without a loaded row) -> {out}")
    return 0 if keys else 1


def compare(argv: list[str]) -> int:
    ratio = float(take_option(argv, "--ratio", 1.5))
    reject_out = take_option(argv, "--reject-out")
    verbose = "--verbose" in argv
    argv = [a for a in argv if a != "--verbose"]
    default = json.loads(Path(argv[0]).read_text())["keys"]
    tuned = json.loads(Path(argv[1]).read_text())["keys"]
    bad: list[tuple[str, list[str]]] = []
    equal = 0
    worst = 0.0
    for key in sorted(tuned):
        if key not in default:
            continue
        reasons: set[str] = set()
        same = True
        key_worst = 0.0
        for flavour, t in tuned[key].items():
            d = default[key][flavour]
            for stat in ("max", "l2"):
                r = t[stat] / d[stat] if d[stat] else 1.0
                key_worst = max(key_worst, r)
                if t[stat] > ratio * d[stat]:
                    reasons.add("error")
            same = same and t["eager"] == d["eager"]
            if t["graph"] != t["eager"]:
                reasons.add("graph")
            if not t["repeat"]:
                reasons.add("repeat")
        equal += same
        worst = max(worst, key_worst)
        if reasons:
            bad.append((key, sorted(reasons)))
        print(
            f"{'BAD ' if reasons else 'ok  '}{key} err_ratio={key_worst:.2f} "
            f"bits={'equal' if same else 'differ'}"
            + (f" {sorted(reasons)}" if reasons else "")
        )
        if verbose:
            for flavour, t in tuned[key].items():
                d = default[key][flavour]
                print(
                    f"      {flavour:9s} max d={d['max']:.3e} t={t['max']:.3e}  "
                    f"l2 d={d['l2']:.3e} t={t['l2']:.3e}"
                )
    checked = len([k for k in tuned if k in default])
    print(
        f"checked {checked} keys: tuned bits equal the default's for {equal}, differ "
        f"for {checked - equal}; worst error ratio {worst:.2f}; {len(bad)} rejected "
        f"(error ratio > {ratio})"
    )
    if reject_out is not None:
        Path(reject_out).write_text(
            "".join(f"GemmTunableOp_Half_TN,{k}\n" for k, _ in bad)
        )
    return 1 if bad or not checked else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("run", "compare"):
        sys.exit(__doc__)
    sys.exit(run(args[1:]) if args[0] == "run" else compare(args[1:]))
