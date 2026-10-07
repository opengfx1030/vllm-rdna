# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove the frozen rows are actually consumed: lookup-hit evidence per shape.

Enables TunableOp in lookup-only mode against a rows file, runs every shape in
it, and reports the solver TunableOp actually recorded for each key. A hit means
the in-model solver equals the row we shipped, not a heuristic fallback. With
--heur it also emits the lookup-vs-heuristic delta per shape.

  verify_tunableop_lookup.py --rows <file> --out hits.json [--heur heur.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path

KEY_RE = re.compile(r"^(tn|nn)_(\d+)_(\d+)_(\d+)_ld_\d+_\d+_\d+$")


def parse(key):
    m = KEY_RE.match(key)
    return None if not m else (m.group(1), *(int(g) for g in m.groups()[1:]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--heur", type=Path, default=None,
                   help="optional heuristic timing JSON for a lookup-vs-heur delta")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    for name in ("PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING",
                 "PYTORCH_TUNABLEOP_FILENAME", "PYTORCH_TUNABLEOP_RECORD_UNTUNED"):
        os.environ.pop(name, None)
    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

    expected = {}
    for row in csv.reader(args.rows.open()):
        if len(row) >= 4 and parse(row[1]):
            expected[row[1]] = row[2]

    import torch
    import torch.cuda.tunable as tunable

    device = f"cuda:{args.device}"
    torch.accelerator.set_device_index(args.device)
    tunable.set_filename(str(args.rows), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(False)

    items = sorted(expected.items(), key=lambda kv: parse(kv[0])[1:])
    if args.limit:
        items = items[: args.limit]

    def timed(call):
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        vals = []
        for _ in range(args.reps):
            s.record()
            call()
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e))
        return statistics.median(vals)

    hits, misses, lookup_ms = {}, [], {}
    for key, solver in items:
        layout, n, m, k = parse(key)
        if layout == "tn":
            x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.1
            w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
            call = (lambda x=x, w=w: torch.nn.functional.linear(x, w))
        else:
            a = torch.randn(m, k, device=device, dtype=torch.float16) * 0.1
            b = torch.randn(k, n, device=device, dtype=torch.float16) * 0.1
            call = (lambda a=a, b=b: a @ b)
        lookup_ms[key] = timed(call)
        got = [r for r in tunable.get_results() if r[1] == key]
        chosen = got[0][2] if got else None
        hits[key] = chosen
        if chosen != solver:
            misses.append({"key": key, "expected": solver, "got": chosen})
    tunable.enable(False)

    payload = {
        "filename": str(args.rows),
        "shapes": len(items),
        "hit_count": sum(1 for k, v in hits.items() if v == expected[k]),
        "misses": misses,
        "solvers": hits,
        "lookup_ms": lookup_ms,
    }
    if args.heur and args.heur.exists():
        heur = json.loads(args.heur.read_text())["results"]
        payload["heur_ms"] = {k: heur[k] for k in hits if k in heur}
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"lookup hits {payload['hit_count']}/{payload['shapes']} via {args.rows}")
    for miss in misses[:20]:
        print(f"  MISS {miss}")
    return 0 if not misses else 1


if __name__ == "__main__":
    raise SystemExit(main())
