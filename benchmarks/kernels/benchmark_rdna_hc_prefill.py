# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023  (each lambda is timed inside its own loop iteration)
"""Hyper-connection prefill mix on gfx1030: torch path vs fused HIP kernels.

Times one HC mix at Flash-Next shapes (hidden 2560, hc_count 4, lora 320,
merged down+inject N = 336) for a range of token counts, single GPU (the HC
module is replicated on every TP rank, so TP does not change the shape).

  torch    current prefill path of ``rdna_hc_mix``: rocBLAS down GEMM (rows
           padded to 64), hc_silu, rocBLAS up GEMM writing the [M, 4H] gate,
           hc_gate_mix
  up       rocBLAS down GEMM + ``rdna_hc_up_gate_mix_prefill`` (silu, up GEMM,
           sigmoid and mix in one kernel; the gate is never written)
  full     ``rdna_hc_mix_prefill``: split-K HIP down GEMM + the fused kernel
  oldhip   torch path with the ``VLLM_RDNA_HC_PREFILL_HIP=1`` glue kernels
           (silu, gate mix) instead of Triton; needs a rebuilt _rocm_C

``--pieces`` also times each torch op and the fused kernels alone. Run with
the TunableOp rows the server uses (``tools/rdna2_028/tunableop_env.sh``),
otherwise rocBLAS falls back to its heuristics.

  python benchmarks/kernels/benchmark_rdna_hc_prefill.py --m 64,512,1030
  python benchmarks/kernels/benchmark_rdna_hc_prefill.py --jit   # no rebuild
"""

import argparse
import json
import os
import statistics
import sys

import torch


def resolve_device(device_str):
    """Resolve the torch device index and guard physical HIP devices 0 and 1.

    The torch index is mapped through ``HIP_VISIBLE_DEVICES`` when it is set
    (comma list; unset = identity). Only the physical index is refused.
    """
    if device_str.isdigit():
        torch_idx = int(device_str)
    else:
        d = torch.device(device_str)
        torch_idx = d.index if d.index is not None else 0
    hip = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip:
        vis = [int(x.strip()) for x in hip.split(",") if x.strip()]
        if torch_idx >= len(vis):
            print(
                f"torch index {torch_idx} is out of range for HIP_VISIBLE_DEVICES={hip}"
            )
            sys.exit(1)
        phys = vis[torch_idx]
    else:
        phys = torch_idx
    if phys in (0, 1):
        print(
            f"Refusing physical device {phys} (torch index {torch_idx}): "
            "devices 0 and 1 are reserved."
        )
        sys.exit(1)
    name = torch.cuda.get_device_name(torch_idx)
    print(f"Running on torch device {torch_idx} (physical HIP {phys}): {name}")
    torch.accelerator.set_device_index(torch_idx)
    return torch_idx


H, HC, R, NDOWN = 2560, 4, 320, 336


class Ops:
    def __init__(self, jit: bool):
        if not jit:
            import vllm._custom_ops  # noqa: F401

            self.up = torch.ops._rocm_C.rdna_hc_up_gate_mix_prefill
            self.full = torch.ops._rocm_C.rdna_hc_mix_prefill
            self.old = torch.ops._rocm_C
            return
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(here, "../../csrc/rocm/hc_prefill_rdna2.cu")
        jdir = os.environ.get("HC_JIT_DIR", os.path.expanduser("~/.cache/hc_jit"))
        os.makedirs(jdir, exist_ok=True)
        bind = os.path.join(jdir, "bind.cpp")
        with open(bind, "w") as f:
            f.write(
                "#include <torch/extension.h>\n"
                "at::Tensor rdna_hc_up_gate_mix_prefill(const at::Tensor&, "
                "const at::Tensor&, const at::Tensor&, int64_t, int64_t);\n"
                "std::tuple<at::Tensor, at::Tensor> rdna_hc_mix_prefill("
                "const at::Tensor&, const at::Tensor&, const at::Tensor&, "
                "int64_t, int64_t);\n"
                "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {"
                ' m.def("up", &rdna_hc_up_gate_mix_prefill);'
                ' m.def("full", &rdna_hc_mix_prefill); }\n'
            )
        mod = load(
            name="hc_prefill_jit",
            sources=[os.path.abspath(src), bind],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx1030"]
            + os.environ.get("HC_JIT_CFLAGS", "").split(),
            build_directory=jdir,
        )
        self.up, self.full, self.old = mod.up, mod.full, None


def torch_path(xn, w_down, w_up, silu, mix):
    from vllm.model_executor.layers.rdna_ops import _linear_padded_m

    dai = _linear_padded_m(xn, w_down)
    lora = silu(dai[:, :R].contiguous(), HC)
    gate = _linear_padded_m(lora, w_up)
    return mix(xn, gate, HC), dai


def bench(fn, iters):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    g.replay()
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.accelerator.synchronize()
    return s.elapsed_time(e) / iters


def bench_simple(fn, iters):
    """Per-call CUDA-event timing without CUDA graphs.

    The split-K count changes the kernel launch configuration and the C++
    reads ``VLLM_RDNA_HC_DOWN_SPLITS`` per call, so a captured graph would
    freeze the wrong split count. Time the raw launches instead.
    """
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times), min(times)


def sweep_hc_splits(args, w_down, w_up, ops):
    """Sweep VLLM_RDNA_HC_DOWN_SPLITS for the fused down GEMM at large M."""
    split_values = [0, 1, 2, 3, 4, 6, 8]
    m_values = [256, 512, 1024, 1536, 2048]
    print("HC down-GEMM split-K sweep (VLLM_RDNA_HC_DOWN_SPLITS)")
    print("Results differ only by fp32 summation order when the split count changes.")
    header = f"{'M':>6} " + " ".join(f"{s:>7}" for s in split_values)
    header += "  Best   MaxDiff"
    print(header)
    for m in m_values:
        xn = torch.randn(m, HC * H, device="cuda").half()
        os.environ["VLLM_RDNA_HC_DOWN_SPLITS"] = "0"
        ref_out = ops.full(xn, w_down, w_up, R, HC)[0]
        times = []
        diffs = []
        for s in split_values:
            os.environ["VLLM_RDNA_HC_DOWN_SPLITS"] = str(s)
            out = ops.full(xn, w_down, w_up, R, HC)[0]
            diff = (out.float() - ref_out.float()).abs().max().item()
            t_med, _ = bench_simple(
                lambda: ops.full(xn, w_down, w_up, R, HC), args.iters
            )
            times.append(t_med)
            diffs.append(diff)
        best_idx = times.index(min(times))
        best_split = split_values[best_idx]
        diff_str = "0 (bit-exact)" if diffs[best_idx] == 0 else f"{diffs[best_idx]:.2e}"
        print(
            f"{m:>6} "
            + " ".join(f"{t:>7.2f}" for t in times)
            + f" {best_split:>4}   {diff_str}"
        )
    os.environ.pop("VLLM_RDNA_HC_DOWN_SPLITS", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="16,64,100,128,256,384,512,768,1024,1030,1536,2048")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--jit", action="store_true")
    ap.add_argument("--oldhip", action="store_true")
    ap.add_argument("--pieces", action="store_true")
    ap.add_argument("--sweep-splits", action="store_true")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    dev_id = resolve_device(args.device)
    torch.accelerator.set_device_index(dev_id)

    from vllm.model_executor.layers.rdna_ops import _linear_padded_m
    from vllm.models.qwen4_exp.amd.ops import hc as hc_ops

    silu = torch.ops.vllm.qwen4_exp_hc_silu
    mix = torch.ops.vllm.qwen4_exp_hc_gate_mix
    ops = Ops(args.jit)
    torch.manual_seed(0)
    w_down = (torch.randn(NDOWN, HC * H, device="cuda") * 0.02).half()
    w_up = (torch.randn(HC * H, R, device="cuda") * 0.05).half()

    if args.sweep_splits:
        sweep_hc_splits(args, w_down, w_up, ops)
        return

    hdr = f"{'M':>6} {'torch':>8} {'up':>8} {'full':>8} {'x up':>6} {'x full':>6}"
    hdr += f" {'err up':>8} {'err full':>8}"
    if args.oldhip:
        hdr += f" {'oldhip':>8}"
    if args.pieces:
        hdr += f" {'down':>7} {'silu':>7} {'upgemm':>7} {'mix':>7}"
        hdr += f" {'upkern':>7} {'TF/s':>5}"
    print(hdr)
    rows = []
    for m in [int(v) for v in args.m.split(",")]:
        xn = torch.randn(m, HC * H, device="cuda").half()
        ref, ref_dai = torch_path(xn, w_down, w_up, silu, mix)

        def run_up(xn=xn):
            dai = _linear_padded_m(xn, w_down)
            return ops.up(dai, w_up, xn, R, HC), dai

        def run_full(xn=xn):
            return ops.full(xn, w_down, w_up, R, HC)

        e_up = (run_up()[0].float() - ref.float()).abs().max().item()
        out_f, dai_f = run_full()
        e_full = (out_f.float() - ref.float()).abs().max().item()
        e_dai = (dai_f.float() - ref_dai.float()).abs().max().item()
        r = dict(m=m, err_up=e_up, err_full=e_full, err_dai=e_dai)
        r["torch_ms"] = bench(
            lambda: torch_path(xn, w_down, w_up, silu, mix), args.iters
        )
        r["up_ms"] = bench(run_up, args.iters)
        r["full_ms"] = bench(run_full, args.iters)
        line = (
            f"{m:>6} {r['torch_ms']:>8.4f} {r['up_ms']:>8.4f} {r['full_ms']:>8.4f}"
            f" {r['torch_ms'] / r['up_ms']:>6.3f} {r['torch_ms'] / r['full_ms']:>6.3f}"
            f" {e_up:>8.1e} {e_full:>8.1e}"
        )
        if args.oldhip:
            old_silu, old_mix = hc_ops.hc_silu, hc_ops.hc_gate_mix
            r["oldhip_ms"] = bench(
                lambda: torch_path(xn, w_down, w_up, old_silu, old_mix), args.iters
            )
            line += f" {r['oldhip_ms']:>8.4f}"
        if args.pieces:
            dai = _linear_padded_m(xn, w_down)
            lora = silu(dai[:, :R].contiguous(), HC)
            gate = _linear_padded_m(lora, w_up)
            r["down_ms"] = bench(lambda: _linear_padded_m(xn, w_down), args.iters)
            r["silu_ms"] = bench(lambda: silu(dai[:, :R].contiguous(), HC), args.iters)
            r["upgemm_ms"] = bench(lambda: _linear_padded_m(lora, w_up), args.iters)
            r["mix_ms"] = bench(lambda: mix(xn, gate, HC), args.iters)
            r["upkern_ms"] = bench(lambda: ops.up(dai, w_up, xn, R, HC), args.iters)
            tf = 2 * m * R * HC * H / (r["upkern_ms"] * 1e-3) / 1e12
            line += (
                f" {r['down_ms']:>7.4f} {r['silu_ms']:>7.4f} {r['upgemm_ms']:>7.4f}"
                f" {r['mix_ms']:>7.4f} {r['upkern_ms']:>7.4f} {tf:>5.1f}"
            )
        print(line, flush=True)
        rows.append(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
