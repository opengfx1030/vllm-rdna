# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GemmaRMSNorm(x, residual): functional path vs VLLM_RDNA_GEMMA_RESIDUAL_DIRECT=1.

gfx10x, fp16. For each shape: bit-exact check of the output and the residual,
CUDA-graph timed per-call cost (a graph of --calls chained norm calls, min over
--reps replays), and kernel launches per call from torch.profiler, eager and
inside the replayed graph. --compile also counts launches of an inductor
compiled chain, where the functional path's clones may already be elided.

    HIP_VISIBLE_DEVICES=<one V620> python \
        benchmarks/kernels/benchmark_gemma_residual_direct.py
"""

import argparse
import json
import os

import torch

ENV = "VLLM_RDNA_GEMMA_RESIDUAL_DIRECT"
HIDDEN = 5120
EPS = 1e-6


def _weight(gen: torch.Generator) -> torch.Tensor:
    w = torch.randn(HIDDEN, generator=gen) * 0.3
    w[torch.randperm(HIDDEN, generator=gen)[:16]] = torch.randn(16, generator=gen) * 3
    return w.to(torch.float16)


def _act(gen: torch.Generator, rows: int) -> torch.Tensor:
    a = torch.randn(rows, HIDDEN, generator=gen)
    a[:, :4] *= 60.0
    return a.to(torch.float16)


def _make(direct: bool, w: torch.Tensor):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    os.environ[ENV] = "1" if direct else "0"
    with torch.device("cuda"):
        layer = GemmaRMSNorm(HIDDEN, eps=EPS).to(torch.float16)
    loader = getattr(layer.weight, "weight_loader", None)
    if loader is None:
        layer.weight.data.copy_(w)
    else:
        loader(layer.weight, w)
    return layer


def _chain(layer, x, r, calls):
    for _ in range(calls):
        x, r = layer(x, r)
    return x, r


def _graph_time_ms(layer, x0, r0, calls: int, reps: int) -> float:
    x, r = x0.clone(), r0.clone()
    for _ in range(3):  # warm up eager, outside capture
        _chain(layer, x, r, 2)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _chain(layer, x, r, calls)
    for _ in range(5):
        g.replay()
    torch.accelerator.synchronize()
    best = float("inf")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
    for _ in range(reps):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    return best / calls


def _kernel_count(fn, n: int) -> float:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.accelerator.synchronize()
    return (
        sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        / n
    )


def _graph_kernel_count(layer, x0, r0, calls: int) -> float:
    x, r = x0.clone(), r0.clone()
    _chain(layer, x, r, 2)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _chain(layer, x, r, calls)
    return _kernel_count(g.replay, 1) / calls


def _eager_launches(layer, x0, r0) -> float:
    x, r = x0.clone(), r0.clone()  # the in-place path evolves these
    return _kernel_count(lambda: _chain(layer, x, r, 1), 10)


def _inductor_launches(layer, x0, r0) -> float:
    torch._dynamo.reset()  # one compile per (layer, shape)
    fn = torch.compile(_chain, backend="inductor", dynamic=False)
    x, r = x0.clone(), r0.clone()
    fn(layer, x, r, 8)
    torch.accelerator.synchronize()
    return _kernel_count(lambda: fn(layer, x, r, 8), 5) / 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, nargs="+", default=[1, 2, 8, 24, 2048])
    ap.add_argument("--calls", type=int, default=128, help="norm calls per graph")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import vllm.kernels  # noqa: F401  registers the vllm_c IR impls
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config

    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} bus {p.pci_bus_id:02x} {torch.__version__}")
    gen = torch.Generator().manual_seed(0)
    w = _weight(gen)
    rows_out = []
    with (
        set_current_vllm_config(VllmConfig()),
        # gfx10x serves the IR norm ops through vllm_c (platforms/rocm.py).
        ir.ops.rms_norm.set_priority(["vllm_c", "native"]),
        ir.ops.fused_add_rms_norm.set_priority(["vllm_c", "native"]),
        torch.inference_mode(),
    ):
        old = _make(False, w)
        new = _make(True, w)
        assert new.rdna_residual_direct and new._rdna_scale_ready
        for rows in args.shapes:
            x0, r0 = _act(gen, rows).cuda(), _act(gen, rows).cuda()
            ref_o, ref_r = old(x0.clone(), r0.clone())
            xi, ri = x0.clone(), r0.clone()
            o, r = new(xi, ri)
            exact = bool(torch.equal(o, ref_o) and torch.equal(r, ref_r))
            row = {"rows": rows, "bit_exact_out_and_residual": exact}
            for name, layer in (("old", old), ("new", new)):
                row[f"{name}_ms_per_call"] = _graph_time_ms(
                    layer, x0, r0, args.calls, args.reps
                )
                row[f"{name}_launches_eager"] = _eager_launches(layer, x0, r0)
                row[f"{name}_launches_graph"] = _graph_kernel_count(
                    layer, x0, r0, args.calls
                )
            if args.compile:
                for name, layer in (("old", old), ("new", new)):
                    row[f"{name}_launches_inductor"] = _inductor_launches(layer, x0, r0)
            rows_out.append(row)
            print(json.dumps(row), flush=True)

    print()
    hdr = (
        "| rows | bit-exact | old ms/call | new ms/call | saved ms/call "
        "| x128 ms/step | launches/call old -> new (eager / graph) |"
    )
    print(hdr)
    print("|" + "---|" * 7)
    for r in rows_out:
        d = r["old_ms_per_call"] - r["new_ms_per_call"]
        print(
            f"| {r['rows']} | {r['bit_exact_out_and_residual']} "
            f"| {r['old_ms_per_call']:.4f} | {r['new_ms_per_call']:.4f} "
            f"| {d:.4f} | {d * 128:.3f} "
            f"| {r['old_launches_eager']:.1f} -> {r['new_launches_eager']:.1f} / "
            f"{r['old_launches_graph']:.1f} -> {r['new_launches_graph']:.1f} |"
        )
    if args.compile:
        print()
        for r in rows_out:
            print(
                f"inductor launches/call rows={r['rows']}: "
                f"{r['old_launches_inductor']:.2f} -> {r['new_launches_inductor']:.2f}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows_out, f, indent=1)


if __name__ == "__main__":
    main()
