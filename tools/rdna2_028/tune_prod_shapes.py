# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tune / measure the fork's production FP16 GEMM shapes on gfx1030.

The frozen shared rows (`tunableop/rocblas-<hash>/`) hold the FP16 dense
projection shapes that Flash-Next actually runs. They are harvested, not
re-derived, so several production shapes were never independently timed and the
small decode batches (M <= 8, the cudagraph capture sizes) were missing
entirely -- the c=1 regression the storage-policy work identified.

This harness closes both gaps. It reads a shape list (a TunableOp results file),
optionally extends it with the missing small-M decode shapes of every (N, K)
pair already seen at M <= 32, then either:

  tune     construct the tensors for each shape with tuning enabled and record
           rocBLAS's chosen solver (>= 10 iterations, 25 ms budget, numerical
           check) into a scratch results file; or
  measure  time every shape with a fixed condition and emit JSON:
             --rows <file>   lookup from that results file
             --heuristic     TunableOp disabled (rocBLAS default)

Three separate processes (heuristic / current rows / scratch rows) give an
apples-to-apples per-shape comparison, because TunableOp caches the chosen
solver per shape for the life of the process.

The op is built from the key alone -- every production key uses contiguous
strides (verified), so `linear(x[M,K], w[N,K])` reproduces `tn_<N>_<M>_<K>` and
`a[M,K] @ b[K,N]` reproduces `nn_<N>_<M>_<K>` exactly. This is also why the
harness exists instead of `torch.cuda.tunable.tune_gemm_in_file`: the offline
path refuses m == 1 TN shapes, which are precisely the c=1 decode shapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

KEY_RE = re.compile(r"^(tn|nn)_(\d+)_(\d+)_(\d+)_ld_\d+_\d+_\d+$")


def parse_gemm_key(key: str):
    """Return (layout, n, m, k) for a regular GemmTunableOp key."""
    match = KEY_RE.match(key)
    if not match:
        return None
    layout, n, m, k = match.group(1), *(int(g) for g in match.groups()[1:])
    return layout, n, m, k


def load_shapes(rows_file: Path):
    """All GemmTunableOp_Half shapes in a results file, as (layout, n, m, k)."""
    shapes = {}
    for row in csv.reader(rows_file.open()):
        if len(row) < 2 or not row[0].startswith("GemmTunableOp"):
            continue
        if not row[0].endswith(("_TN", "_NN")):
            continue  # fp32 strided-batched rows are not fp16 serving shapes
        parsed = parse_gemm_key(row[1])
        if parsed is None:
            continue
        shapes[row[1]] = parsed
    return shapes


def extend_with_small_m(shapes: dict):
    """Add M in 1..8 for every (layout, N, K) that already occurs at M <= 32.

    Those are the decode / MTP-verify projections; the cudagraph capture sizes
    (MTP-0 [1,2,4,8], MTP-2 [3,6,12,24]) run them at small M and the harvested
    table starts at M=9.
    """
    decode_pairs = set()
    for layout, n, m, k in shapes.values():
        if m <= 32:
            decode_pairs.add((layout, n, k))
    added = 0
    for layout, n, k in sorted(decode_pairs):
        for m in range(1, 9):
            key = f"{layout}_{n}_{m}_{k}_ld_"
            existing = [k2 for k2 in shapes if k2.startswith(key)]
            if existing:
                continue
            # strides: TN -> (ldb=K, lda=K, ldc=N); NN -> (ldb=N, lda=K, ldc=N)
            ld = (k, k, n) if layout == "tn" else (n, k, n)
            full = f"{layout}_{n}_{m}_{k}_ld_{ld[0]}_{ld[1]}_{ld[2]}"
            shapes[full] = (layout, n, m, k)
            added += 1
    return added


def make_op(layout: str, n: int, m: int, k: int, device: str):
    """Return (call, tensors) running the exact GEMM the key names."""
    import torch

    if layout == "tn":
        # C[M,N] = linear(x[M,K], w[N,K])  ->  key tn_<N>_<M>_<K>
        x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.1
        w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1

        def call():
            return torch.nn.functional.linear(x, w)

        return call, (x, w)
    # C[M,N] = a[M,K] @ b[K,N]  ->  key nn_<N>_<M>_<K>
    a = torch.randn(m, k, device=device, dtype=torch.float16) * 0.1
    b = torch.randn(k, n, device=device, dtype=torch.float16) * 0.1

    def call():
        return a @ b

    return call, (a, b)


def time_shape(call, device: str, warmup: int, reps: int) -> float:
    import torch

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-source", required=True, type=Path,
                        help="results file whose keys define the shape list")
    parser.add_argument("--mode", choices=["tune", "measure"], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=Path, default=None,
                        help="measure: results file to look up")
    parser.add_argument("--heuristic", action="store_true",
                        help="measure: disable TunableOp entirely")
    parser.add_argument("--scratch", type=Path, default=None,
                        help="tune: directory for the scratch results file")
    parser.add_argument("--rank", type=int, default=None,
                        help="tune: rank ordinal for the scratch filename")
    parser.add_argument("--no-small-m", action="store_true",
                        help="do not add the M in 1..8 decode shapes")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-ms", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # TunableOp reads its env once; a stale value would silently disable the API.
    for name in ("PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING",
                 "PYTORCH_TUNABLEOP_FILENAME", "PYTORCH_TUNABLEOP_RECORD_UNTUNED"):
        os.environ.pop(name, None)
    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

    shapes = load_shapes(args.shape_source)
    if not args.no_small_m:
        added = extend_with_small_m(shapes)
        print(f"[shapes] {len(shapes)} total ({added} small-M decode shapes added)",
              file=sys.stderr, flush=True)
    items = sorted(shapes.items(), key=lambda kv: (kv[1][1], kv[1][2], kv[1][3]))
    if args.limit:
        items = items[: args.limit]

    import torch
    import torch.cuda.tunable as tunable

    device = f"cuda:{args.device}"
    torch.accelerator.set_device_index(args.device)

    if args.mode == "tune":
        assert args.scratch and args.rank is not None, "--scratch and --rank required"
        args.scratch.mkdir(parents=True, exist_ok=True)
        out = args.scratch / f"tunableop_results{args.rank}.csv"
        if out.exists():
            out.unlink()
        tunable.set_filename(str(out), insert_device_ordinal=False)
        tunable.set_max_tuning_duration(args.max_ms)
        tunable.set_max_tuning_iterations(args.iterations)
        tunable.set_numerical_check_tolerances(True, 0.01, 0.01)
        tunable.enable(True)
        tunable.tuning_enable(True)
        # Warm the box once before measuring anything.
        warm_call, warm_t = make_op("tn", 640, 8, 2560, device)
        for _ in range(3):
            warm_call()
        del warm_call, warm_t
        torch.cuda.synchronize()

        def save():
            rows = tunable.get_results()
            with out.open("w") as fh:
                writer = csv.writer(fh, lineterminator="\n")
                writer.writerows(("Validator", *v) for v in tunable.get_validators())
                writer.writerows(rows)

        results = {}
        t0 = time.time()
        for i, (key, (layout, n, m, k)) in enumerate(items):
            call, tensors = make_op(layout, n, m, k, device)
            call()  # tuning fires here
            hit = [r for r in tunable.get_results() if r[1] == key]
            results[key] = (
                {"solver": hit[0][2], "ms": hit[0][3]} if hit else
                {"solver": None, "ms": None}
            )
            del call, tensors
            if i % 25 == 0:
                save()
                print(f"[tune] {i+1}/{len(items)} {key} -> {results[key]} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        save()
        tunable.tuning_enable(False)
        tunable.enable(False)
        print(f"[tune] done {len(items)} shapes in {time.time()-t0:.0f}s", file=sys.stderr)
        payload = {"mode": "tune", "device": args.device, "rows": str(out),
                   "results": results}
    else:
        if args.heuristic:
            tunable.enable(False)
        else:
            assert args.rows, "--rows required for lookup measurement"
            tunable.set_filename(str(args.rows), insert_device_ordinal=False)
            tunable.enable(True)
            tunable.tuning_enable(False)
        results = {}
        t0 = time.time()
        for i, (key, (layout, n, m, k)) in enumerate(items):
            call, tensors = make_op(layout, n, m, k, device)
            ms = time_shape(call, device, args.warmup, args.reps)
            results[key] = ms
            del call, tensors
            if i % 100 == 0:
                print(f"[measure] {i+1}/{len(items)} {key} {ms:.4f}ms "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        payload = {"mode": "measure", "device": args.device,
                   "condition": "heuristic" if args.heuristic else str(args.rows),
                   "results": results}

    text = json.dumps(payload)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
