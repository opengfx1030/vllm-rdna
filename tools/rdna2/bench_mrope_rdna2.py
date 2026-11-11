#!/usr/bin/env python
"""Isolated harness: mrope_forward_rdna2 (HIP) vs _triton_mrope_forward (Triton).

Times the two arms with preallocated buffers at the production geometry
(rd=64, interleaved, TP4) and at the test-suite geometry, plus the decode shapes.

    HIP_VISIBLE_DEVICES=3 .venv/bin/python tools/rdna2/bench_mrope_rdna2.py
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib  # noqa: E402
from benchlib import ITERS, WARMUP, timeit_best, warm_clocks  # noqa: E402

from vllm.model_executor.layers.rotary_embedding.mrope import (  # noqa: E402
    _triton_mrope_forward,
    mrope_forward_rdna2,
    triton_mrope,
)

# (label, head_size, rotary_dim, n_qh, n_kh, sections, interleaved, neox)
GEOMS = (
    ("prod_tp4", 256, 64, 6, 1, [11, 11, 10], True, True),
    ("prod_tp1", 256, 64, 24, 2, [11, 11, 10], True, True),
    ("prod_tp4_conc", 256, 64, 6, 1, [11, 11, 10], False, True),
    ("test_rd128", 128, 128, 4, 1, [16, 24, 24], False, True),
    ("test_rd256", 256, 256, 24, 2, [32, 48, 48], False, True),
    ("t256_int_neox", 256, 256, 24, 2, [32, 48, 48], True, True),
    ("t256_conc_gptj", 256, 256, 24, 2, [32, 48, 48], False, False),
    ("t256_int_gptj", 256, 256, 24, 2, [32, 48, 48], True, False),
    ("t128_int_neox", 128, 128, 4, 1, [16, 24, 24], True, True),
    ("t128_conc_gptj", 128, 128, 4, 1, [16, 24, 24], False, False),
)
TOKENS = (1, 3, 7, 48, 512, 2048)


def make_inputs(T, hd, rd, n_qh, n_kh):
    """Norm-preserving cos/sin (real cos/sin of random angles), unlike the parity test."""
    g = torch.Generator(device="cuda").manual_seed(T)
    ang = torch.rand(6, T, rd // 2, device="cuda", generator=g) * (2 * math.pi)
    planes = torch.cat([torch.cos(ang)[:3], torch.sin(ang)[3:]], 0).to(torch.float16)
    q = torch.randn(T, n_qh * hd, dtype=torch.float16, device="cuda", generator=g)
    k = torch.randn(T, n_kh * hd, dtype=torch.float16, device="cuda", generator=g)
    return q, k, planes[:3].contiguous(), planes[3:].contiguous()


def arms(geom, T):
    label, hd, rd, n_qh, n_kh, sec, inter, neox = geom
    q, k, cos, sin = make_inputs(T, hd, rd, n_qh, n_kh)
    pool = [q.clone() for _ in range(2)], [k.clone() for _ in range(2)]
    qq, kk = pool[0], pool[1]

    triton = lambda: triton_mrope(qq[0], kk[0], cos, sin, sec, hd, rd, inter, neox)
    hip = lambda: mrope_forward_rdna2(qq[1], kk[1], cos, sin, hd, rd, sec, inter, neox)
    hip_op = lambda: torch.ops._rocm_C.mrope_forward_rdna2(
        qq[1], kk[1], cos, sin, T, n_qh, n_kh, hd, rd, *sec, inter, neox
    )
    return triton, hip, hip_op, (qq[0], kk[0], cos, sin, qq[1], kk[1])


def _spin(ms=6.0):
    """Hold the clocks up: the V620 boosts only under sustained load."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    import time

    t0 = time.time()
    while time.time() - t0 < ms / 1000.0:
        a @ b
    torch.cuda.synchronize()


def timeit_gpu(fn, reps=20):
    """Single-call event pair, taken right after a 6 ms matmul so the GPU is
    boosted. Excludes the python/launch cost that dominates the batched timing
    of the short (decode) shapes."""
    _spin(30)
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        _spin()
        fn()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000.0)
    return best


def profile_one(fn):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in prof.key_averages()) / 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--tokens", type=int, nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    benchlib.ITERS = args.iters
    timeit = lambda fn: benchlib.timeit_best(fn, reps=args.reps)
    toks = args.tokens if args.tokens else TOKENS
    warm_clocks()
    out = {"device": torch.cuda.get_device_name(0)}
    for geom in GEOMS:
        for T in toks:
            triton, hip, hip_op, _ = arms(geom, T)
            if args.gpu:
                gr = timeit_gpu(triton)
                gh = timeit_gpu(hip)
                print(f"{geom[0]:14s} T={T:5d}  GPU-only triton {gr:8.2f} us  hip "
                      f"{gh:8.2f} us ({(gh - gr) / gr * 100:+6.1f}%)", flush=True)
                out[f"{geom[0]}_T{T}_gpu"] = {"triton_us": gr, "hip_us": gh,
                                              "delta_pct": (gh - gr) / gr * 100}
                continue
            r = timeit(triton)
            h = timeit(hip)
            ho = timeit(hip_op)
            row = {"triton_us": r, "hip_us": h, "hip_op_us": ho,
                   "delta_pct": (h - r) / r * 100,
                   "delta_op_pct": (ho - r) / r * 100}
            if args.profile:
                row["kernel_triton_us"] = profile_one(triton)
                row["kernel_hip_us"] = profile_one(hip)
            out[f"{geom[0]}_T{T}"] = row
            print(f"{geom[0]:14s} T={T:5d}  triton {r:8.2f}  hip {h:8.2f} "
                  f"({row['delta_pct']:+6.1f}%)  hip_op {ho:8.2f} "
                  f"({row['delta_op_pct']:+6.1f}%)", flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
