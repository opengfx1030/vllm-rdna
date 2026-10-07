# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove PYTORCH_TUNABLEOP_RECORD_UNTUNED capture semantics on this build.

Runs a couple of FP16 GEMMs with lookup rows active and record-untuned pointed
at a scratch dir, then reports which files TunableOp actually wrote (and under
what names). Confirms:
  * the env var is PYTORCH_TUNABLEOP_UNTUNED_FILENAME (NOT _RECORD_UNTUNED_*),
  * whether the device ordinal is appended to the untuned filename,
  * that a shape already in the rows is NOT re-recorded, and a novel shape is.

No /tmp: caller passes the scratch dir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

scratch = Path(sys.argv[1])
rows = Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)

# Clear any inherited TunableOp env so we control it explicitly.
for name in (
    "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_TUNABLEOP_TUNING",
    "PYTORCH_TUNABLEOP_FILENAME",
    "PYTORCH_TUNABLEOP_RECORD_UNTUNED",
    "PYTORCH_TUNABLEOP_UNTUNED_FILENAME",
):
    os.environ.pop(name, None)

os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "1"
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(rows / "tunableop_results.csv")
os.environ["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = str(scratch / "untuned.csv")

import torch  # noqa: E402
import torch.cuda.tunable as tunable  # noqa: E402

before = sorted(p.name for p in scratch.iterdir())
print("scratch before:", before)
print("record_untuned_is_enabled:", tunable.record_untuned_is_enabled())

torch.accelerator.set_device_index(0)
dev = "cuda:0"

# 1) A production shape that IS in the frozen rows (lm_head decode).
x = torch.randn(1, 2560, device=dev, dtype=torch.float16) * 0.1
w = torch.randn(62080, 2560, device=dev, dtype=torch.float16) * 0.1
torch.nn.functional.linear(x, w)
torch.cuda.synchronize()

# 2) A shape that is certainly NOT in the rows.
x2 = torch.randn(45, 67, device=dev, dtype=torch.float16) * 0.1
w2 = torch.randn(123, 67, device=dev, dtype=torch.float16) * 0.1
torch.nn.functional.linear(x2, w2)
torch.cuda.synchronize()

after = sorted(p.name for p in scratch.iterdir())
print("scratch after :", after)
for name in after:
    if name in before:
        continue
    p = scratch / name
    lines = p.read_text().splitlines()
    print(f"--- {name}: {len(lines)} lines ---")
    for line in lines[:10]:
        print("   ", line)
