"""Robust timing helpers shared by the isolated kernel harnesses.

The V620 idles at 500 MHz and only boosts after ~10 ms of sustained load, and a
short kernel (prep is ~35-300 us) never ramps the clocks on its own. A single
batched measurement therefore lands anywhere between the 500 MHz and the boosted
number depending on what ran just before it. Two defences:
  * warm_clocks(): 2 s of sustained fp16 matmul before any timing;
  * timeit_best(): repeat the batched measurement and keep the MINIMUM, which is
    the cleanest estimator when the only variation is clock ramp / contention.
"""
import time

import torch

WARMUP, ITERS, REPS = 20, 60, 5


def warm_clocks(seconds=2.0):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def _one(fn):
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


def timeit_best(fn, reps=REPS):
    return min(_one(fn) for _ in range(reps))
