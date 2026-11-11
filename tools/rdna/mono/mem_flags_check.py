#!/usr/bin/env python3
"""HSA global flags (2 fine, 4 coarse, 8 extended fine) of a caching-allocator
tensor vs the kernel's explicit workspace allocation, under the current
HSA_FORCE_FINE_GRAIN_PCIE.

  PYTHONPATH=<tree> python3 mem_flags_check.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_bench as B  # noqa: E402

ext = B.build_ext()
t = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
w = ext.workspace_alloc(1 << 20, 0)
print("HSA_FORCE_FINE_GRAIN_PCIE", os.environ.get("HSA_FORCE_FINE_GRAIN_PCIE"),
      "| caching allocator flags", ext.mem_flags(t), "| explicit workspace flags", ext.mem_flags(w))
