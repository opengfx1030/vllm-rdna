#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numerical check of TunableOp rows: tuned vs rocBLAS default vs fp32 reference.

    PYTORCH_TUNABLEOP_FILENAME=<rows>/tunableop_results.csv HIP_VISIBLE_DEVICES=<n> \\
        python rowcheck.py census.csv [only_rows.csv]

For every census key `GemmTunableOp_Half_TN,tn_<m>_<n>_<k>_ld_<k>_<k>_<m>` (weight
[m,k], activations [n,k]) the GEMM runs three times on the same inputs: TunableOp
lookup on, TunableOp off (heuristic), fp32 reference. Prints the relative max error of
both against the reference and flags keys where the tuned result is worse than the
default by more than 3x (+1e-4 slack). Run it with the venv python (regex).
"""

import sys
from pathlib import Path

import regex as re
import torch
import torch.cuda.tunable as tunable

census = sys.argv[1]
only = None
if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
    only = {
        tuple(line.strip().split(",")[:2])
        for line in Path(sys.argv[2]).read_text().splitlines()
        if line.startswith("Gemm")
    }
dev = torch.device("cuda:0")
pat = re.compile(r"tn_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)$")
g = torch.Generator(device=dev).manual_seed(1234)

keys = []
for line in Path(census).read_text().splitlines():
    if not line.startswith("GemmTunableOp_Half_TN"):
        continue
    op, _, key = line.strip().partition(",")
    if only is not None and (op, key) not in only:
        continue
    m = pat.match(key)
    if not m:
        continue
    keys.append((op, key, *[int(v) for v in m.groups()]))

tunable.enable(True)
tunable.tuning_enable(False)
bad = []
for _op, key, m, n, k, lda, ldb, ldc in keys:
    if "_16_16_16_" in key or lda != k or ldb != k or ldc != m:
        continue
    w = (torch.randn(m, k, device=dev, generator=g) * 0.05).half()
    x = torch.randn(n, k, device=dev, generator=g).half()
    ref = x.float() @ w.float().t()
    tunable.enable(True)
    y_t = torch.nn.functional.linear(x, w).float()
    tunable.enable(False)
    y_d = torch.nn.functional.linear(x, w).float()
    scale = ref.abs().max().item() or 1.0
    e_t = (y_t - ref).abs().max().item() / scale
    e_d = (y_d - ref).abs().max().item() / scale
    td = (y_t - y_d).abs().max().item() / scale
    flag = e_t > 3 * e_d + 1e-4
    if flag:
        bad.append((key, e_t, e_d, td))
    print(
        f"{'BAD ' if flag else 'ok  '}{key} err_tuned={e_t:.2e} "
        f"err_default={e_d:.2e} tuned_vs_default={td:.2e}"
    )
print(f"checked {len(keys)} keys, {len(bad)} worse than 3x default")
sys.exit(1 if bad else 0)
