# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probe donor TunableOp FP16 TN/NN rows on the selected GPU.

Run offline with lookup enabled and tuning disabled. The CSV must match the
loaded rocBLAS build; solution IDs are not portable between builds.
"""

import sys

import regex as re
import torch

with open(sys.argv[1]) as csv_file:
    rows = [
        line.strip().split(",")
        for line in csv_file
        if line.startswith("GemmTunableOp_Half_")
    ]
bad, ok, skipped = [], 0, 0
for op, sig, sol, t in rows:
    m = re.match(r"(t|n)(t|n)_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)$", sig)
    if not m:
        skipped += 1
        continue
    ta, tb, M, N, K, lda, ldb, ldc = m.group(1), m.group(2), *map(int, m.groups()[2:])
    try:
        if ta == "t" and tb == "n":  # torch: X[N,K] @ W[M,K].T  -> C[N,M]
            X = torch.randn(N, K, dtype=torch.half, device="cuda")
            W = torch.randn(M, K, dtype=torch.half, device="cuda")
            C = X @ W.t()
        elif ta == "n" and tb == "n":  # torch: X[N,K] @ W[K,M]    -> C[N,M]
            X = torch.randn(N, K, dtype=torch.half, device="cuda")
            W = torch.randn(K, M, dtype=torch.half, device="cuda")
            C = X @ W
        else:
            skipped += 1
            continue
        torch.accelerator.synchronize()
        ok += 1
    except Exception as e:
        bad.append((sig, sol, str(e).splitlines()[0][:80]))
print(f"{sys.argv[1].split('/')[-1]}: ok {ok}, failed {len(bad)}, skipped {skipped}")
for sig, sol, err in bad[:6]:
    print(f"   FAIL {sig} -> {sol}: {err}")

raise SystemExit(1 if bad else 0)
