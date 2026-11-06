#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TunableOp on device ordinal D: which file is read, how many rows load, hit or miss.

One process per ordinal (the way a vLLM worker runs): select D, issue three fp16
GEMMs (two whose keys are in the shipped table, one TP8-only shape that is not), then
print the resolved filename, the number of loaded rows and whether each key is in the
loaded set. Driven by tunetest.sh.
"""

import os
import sys

import torch
import torch.cuda.tunable as t

d = int(sys.argv[1])
torch.accelerator.set_device_index(d)
dev = f"cuda:{d}"


def gemm(m: int, n: int, k: int) -> None:
    # F.linear(x[n,k], w[m,k]) is the census key tn_<m>_<n>_<k>_ld_<k>_<k>_<m>.
    x = torch.randn(n, k, device=dev, dtype=torch.float16)
    w = torch.randn(m, k, device=dev, dtype=torch.float16)
    torch.nn.functional.linear(x, w)


SHAPES = {
    "in_table": (512, 2048, 2560),
    "in_table_b": (640, 261, 2560),
    "tp8_only": (12, 51, 2560),
}
for m, n, k in SHAPES.values():
    gemm(m, n, k)
torch.accelerator.synchronize()
# one (op_signature, params_signature, result, time) per row
rows = list(t.get_results())
loaded = {row[1] for row in rows}
keys = {name: f"tn_{m}_{n}_{k}_ld_{k}_{k}_{m}" for name, (m, n, k) in SHAPES.items()}
hit = {name: key in loaded for name, key in keys.items()}
print(
    f"ordinal {d}: file={t.get_filename()} rows_loaded={len(rows)} "
    f"enabled={t.is_enabled()} tuning={t.tuning_is_enabled()} "
    f"record_untuned={t.record_untuned_is_enabled()} hit={hit} "
    f"env_file={os.environ.get('PYTORCH_TUNABLEOP_FILENAME')}"
)
