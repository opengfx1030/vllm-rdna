#!/usr/bin/env python
"""GDN prefill delta_h: HIP kernel vs the Triton/FLA reference.

Reuses the parity test's input builder and call wrappers, so both arms run
identical tensors and the same call shape the tests verify. Timing is CUDA
events around a batched loop (host-side setup and output allocation included
in both arms).

Geometry is the TP4 production shape: linear_num_value_heads=48 and
linear_num_key_heads=16 divided by TP4 -> H=12, Hg=4.

    HIP_VISIBLE_DEVICES=1 .venv/bin/python tools/rdna2/bench_gdn_prefill_rdna2.py
"""

import json
import sys

import torch

sys.path.insert(0, "tests/kernels/quantization")

import test_gdn_prefill_delta_h_rdna2 as T  # noqa: E402

WARMUP, ITERS = 10, 50


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS * 1000.0


def one(seq_lens, H=12, Hg=4, seed=7):
    k, w, u, g, cu, h0 = T._make_inputs(seq_lens, H, Hg, seed, True)
    ref = timeit(lambda: T._run_ref(k, w, u, g, cu, h0, True))
    hip = timeit(lambda: T._run_hip(k, w, u, g, cu, h0, True))
    return ref, hip


def main():
    out = {"device": torch.cuda.get_device_name(0)}
    for label, lens in (
        ("small_[64,192,128]", [64, 192, 128]),
        ("prefill_[4096]", [4096]),
        ("prefill_[2048,2048]", [2048, 2048]),
        ("prefill_[8192]", [8192]),
        ("prefill_[8192x4]", [8192, 8192, 8192, 8192]),
    ):
        try:
            r, h = one(lens)
            out[label] = {"triton_us": r, "hip_us": h, "delta_pct": (h - r) / r * 100}
        except Exception as e:  # noqa: BLE001
            out[label] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
