#!/usr/bin/env python3
"""Calibrate after_bench.graph_time: the per-call cost of a trivial kernel."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402

x = torch.zeros(1024, device="cuda")
print(f"zero_ 1 kernel: {A.graph_time(lambda: x.zero_(), 20):.1f} us")
print(f"zero_ x4 /4:    {A.graph_time(lambda: [x.zero_() for _ in range(4)], 20) / 4:.1f} us")
