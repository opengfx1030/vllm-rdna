#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time causal_conv1d_fn at the Flash-Next decode geometry (HIP vs Triton).

Env VLLM_CAUSAL_CONV1D_RDNA2_FWD/UPDATE=1 -> HIP kernel; =0 -> Triton.
Run twice under rocprofv3 (single process) and compare kernel durations.

  VLLM_CAUSAL_CONV1D_RDNA2_FWD=1 rocprofv3 --kernel-trace -f csv -o <out> -- \
    python tools/rdna2_028/probe_conv1d.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, "/home/chenco_adm/vllm-rdna-0.28.0")
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn  # noqa: E402

DEV = "cuda"
DIM = 2560
WIDTH = 4
BATCH = 4  # 4 concurrent requests, 1 token each (decode)
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 300

torch.manual_seed(0)
x = torch.randn(DIM, BATCH, device=DEV, dtype=torch.float16)  # [dim, cu_seq_len]
weight = torch.randn(DIM, WIDTH, device=DEV, dtype=torch.float16)
bias = torch.randn(DIM, device=DEV, dtype=torch.float16)
conv_states = torch.randn(BATCH, DIM, WIDTH - 1, device=DEV, dtype=torch.float16)
query_start_loc = torch.arange(BATCH + 1, device=DEV, dtype=torch.int64)
cache_indices = torch.arange(BATCH, device=DEV, dtype=torch.int64)

out = torch.empty_like(x)
# warmup / JIT
for _ in range(3):
    causal_conv1d_fn(
        x, weight, bias, conv_states, query_start_loc, cache_indices,
        activation="silu",
    )
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(ITERS):
    causal_conv1d_fn(
        x, weight, bias, conv_states, query_start_loc, cache_indices,
        activation="silu",
    )
torch.cuda.synchronize()
t1 = time.perf_counter()
mode = "HIP" if os.environ.get("VLLM_CAUSAL_CONV1D_RDNA2_FWD", "1") == "1" else "TRITON"
print(f"conv1d[{mode}]: {ITERS} iters {(t1 - t0) / ITERS * 1e6:.1f} us/call (wall)")
