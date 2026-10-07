#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the MTP sampling kernels at the real Flash-Next geometry.

Calls rejection_sample() (vllm ... spec_decode.rejection_sampler_utils) with
vocab=248320, num_speculative_steps=2, num_reqs=1 in a loop, so rocprofv3
(which cannot profile the multi-process vLLM engine) can be wrapped around
this single-process probe to read the kernel durations.

Run:
  rocprofv3 --kernel-trace -f csv -o <out> -- \
    python tools/rdna2_028/probe_mtp_sampling.py
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/home/chenco_adm/vllm-rdna-0.28.0")
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402

V = 248320
NUM_SPEC = 2
NUM_REQS = 1
NUM_LOGITS = NUM_REQS * (NUM_SPEC + 1)  # 3
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DEV = "cuda"

torch.manual_seed(0)
target_logits = torch.randn(NUM_LOGITS, V, device=DEV, dtype=torch.float32) * 4
draft_logits = torch.randn(NUM_REQS, NUM_SPEC, V, device=DEV, dtype=torch.float32) * 4
draft_sampled = torch.randint(0, V, (NUM_LOGITS,), device=DEV, dtype=torch.int64)
cu_num_logits = torch.tensor([0, NUM_LOGITS], device=DEV, dtype=torch.int64)
pos = torch.arange(NUM_LOGITS, device=DEV, dtype=torch.int64)
idx_mapping = torch.tensor([0], device=DEV, dtype=torch.int64)
expanded_idx_mapping = torch.zeros(NUM_LOGITS, device=DEV, dtype=torch.int64)
expanded_local_pos = torch.tensor([0, 1, 2], device=DEV, dtype=torch.int64)
temperature = torch.tensor([0.0], device=DEV, dtype=torch.float32)
seed = torch.tensor([12345], device=DEV, dtype=torch.int64)

# warmup / JIT compile
for _ in range(3):
    rejection_sample(
        target_logits, draft_logits, draft_sampled, cu_num_logits, pos,
        idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature,
        seed, NUM_SPEC,
    )
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(ITERS):
    rejection_sample(
        target_logits, draft_logits, draft_sampled, cu_num_logits, pos,
        idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature,
        seed, NUM_SPEC,
    )
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"MTP trio+insert: {ITERS} iters in {t1 - t0:.3f}s -> {(t1 - t0) / ITERS * 1e6:.1f} us/call")
