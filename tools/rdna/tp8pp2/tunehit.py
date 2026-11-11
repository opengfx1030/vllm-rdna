#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lookup-hit proof on device ordinal D: does every census key have a loaded row?

Reads the rows exactly the way a worker does (PYTORCH_TUNABLEOP_FILENAME + the
device ordinal, lookup only, RECORD_UNTUNED off), then compares the loaded
(op, params) signatures with the census. 16x16x16 warm-up keys and fp32
strided-batched keys are not tuned by the campaign and are skipped.

    PYTORCH_TUNABLEOP_FILENAME=<rows dir>/tunableop_results.csv \\
        python tunehit.py D census.csv
"""

import sys

import torch
import torch.cuda.tunable as t

d = int(sys.argv[1])
census = sys.argv[2]
torch.accelerator.set_device_index(d)
x = torch.randn(64, 2560, device=f"cuda:{d}", dtype=torch.float16)
w = torch.randn(512, 2560, device=f"cuda:{d}", dtype=torch.float16)
torch.nn.functional.linear(x, w)  # first GEMM loads the rows file
torch.accelerator.synchronize()
rows = list(t.get_results())
have = {(r[0], r[1]) for r in rows}
keys = []
with open(census) as fh:
    for line in fh:
        if not line.startswith("Gemm"):
            continue
        op, _, key = line.strip().partition(",")
        if "_16_16_16_" in key or op.startswith("GemmStridedBatched"):
            continue
        keys.append((op, key))
miss = [k for k in keys if k not in have]
print(
    f"ordinal {d}: file={t.get_filename()} rows={len(rows)} census_keys={len(keys)} "
    f"hit={len(keys) - len(miss)} missing={len(miss)}"
)
for op, key in miss[:10]:
    print(f"  MISSING {op},{key}")
sys.exit(1 if miss else 0)
