#!/usr/bin/env python
"""Clock/overhead probe for the QSA indexer ops.

The official bench profiles 50 back-to-back iterations of a 5-30 us kernel
after only 10 warmup calls, so the GPU sits at its 500 MHz idle clock for the
whole profile window.  This probe reports the same op three ways:

  cold : exactly the official bench conditions (10 warmup, then 120 iters)
  warm : 2 s of fp16 matmul before the profile
  hold : a 5.7 ms matmul interleaved with every call, so the clocks can never
         drop -- the closest thing to a served decode step

Only the op's own kernels are counted (the matmul is filtered out by duration).
"""

import importlib
import os
import sys

import torch

sys.path.insert(0, "tests/kernels/rocm/qwen4_exp")
sys.path.insert(0, "tools/rdna2")

import bench_qsa_rdna2 as B  # noqa: E402

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops  # noqa: E402
from vllm.models.qwen4_exp.amd.ops import qsa_rdna2  # noqa: E402


def set_gate(gate):
    os.environ["VLLM_RDNA_QSA_HIP"] = "1" if gate else "0"
    reloaded = importlib.reload(qsa_rdna2)
    qsa_ops.qsa_rdna2 = reloaded


_HOLD = {}


def warm_clocks():
    a = torch.randn(4096, 4096, dtype=torch.float16, device=B.DEV)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=B.DEV)
    for _ in range(40):
        a @ b
    torch.cuda.synchronize()


def measure(fn, iters=120, warm=False, hold=False):
    """Median GPU duration of the op's own kernels."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    if warm:
        warm_clocks()
    mat = None
    if hold:
        if not _HOLD:
            _HOLD["a"] = torch.randn(4096, 4096, dtype=torch.float16, device=B.DEV)
            _HOLD["b"] = torch.randn(4096, 4096, dtype=torch.float16, device=B.DEV)

        def mat():
            _HOLD["a"] @ _HOLD["b"]

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if mat is not None:
                mat()
            fn()
        torch.cuda.synchronize()
    times = [
        ev.self_device_time_total
        for ev in prof.events()
        if ev.device_type.name == "CUDA" and ev.self_device_time_total < 400.0
    ]
    if not times:
        return float("nan")
    times.sort()
    return times[len(times) // 2]


def main():
    args = sys.argv[1:]
    filt = None
    if args and not args[0].isdigit():
        filt = args[0]
        args = args[1:]
    rows_list = [int(x) for x in (args or [3, 12, 64, 256, 512, 1024, 2048])]
    ops = [
        ("store_cache_rows", lambda r: B.bench_store(rows=r)),
        ("compress_groups", lambda r: B.bench_compress(rows=r)),
        ("expand_indices", lambda r: B.bench_expand(rows=r)),
    ]
    if filt:
        ops = [o for o in ops if filt in o[0]]
    print(f"{'op':<20}{'rows':>6}{'cold':>16}{'warm':>16}{'hold':>16}")
    for name, mk in ops:
        for rows in rows_list:
            row = {}
            for gate in (False, True):
                set_gate(gate)
                fn = mk(rows)
                fn()
                torch.cuda.synchronize()
                row[gate] = (
                    measure(fn, warm=False, hold=False),
                    measure(fn, warm=True, hold=False),
                    measure(fn, iters=40, hold=True),
                )
            cell = lambda i: " ".join(  # noqa: E731
                f"{row[False][i]:6.1f}->{row[True][i]:6.1f}" for i in (i,)
            )
            print(
                f"{name:<20}{rows:>6}"
                f"{row[False][0]:7.1f}->{row[True][0]:6.1f}"
                f"{row[False][1]:7.1f}->{row[True][1]:6.1f}"
                f"{row[False][2]:7.1f}->{row[True][2]:6.1f}"
            )


if __name__ == "__main__":
    main()
