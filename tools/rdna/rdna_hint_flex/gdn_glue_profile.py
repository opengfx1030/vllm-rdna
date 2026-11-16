#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer GDN decode launch audit and timing (HI2, gfx1030).

Runs the real ``QwenGatedDeltaNetAttention`` decode methods (see
``tests/kernels/mamba/gdn_glue_harness.py``) at HI2 TP2 per-rank dimensions and
reports, for the arena path (flag off) and the in-place path
(``VLLM_RDNA_GDN_DECODE_INPLACE=1``):

  stacks   one eager layer step under torch.profiler(with_stack=True): every GPU
           launch (kernel / memcpy / memset) in launch order, mapped to the
           Python source line that issued it. The profiler's Python stack only
           carries function definition lines, so the step also runs under a
           ``sys.settrace`` hook that brackets each executed line of the GDN
           decode files with a profiler annotation; a launch belongs to the
           innermost annotation that contains its host-side launch call.
  timing   a FULL-graph replay of a stack of layers: per-layer time with the
           inter-node gaps. ``--thrash-mib N`` reads N MiB after every layer
           (the real model streams weights between two GDN layers, which evicts
           the state pages from L2 and the Infinity Cache).
  kernel   ``gdn_decode_rdna2`` alone on an arena-style contiguous state versus
           the paged cache, warm and cold (``--cold-mib``): the kernel's own
           duration from the profiler.

  HIP_VISIBLE_DEVICES=4 flock ~/w4a8_runs/hi2-gpu4.lock \
      python tools/rdna/rdna_hint_flex/gdn_glue_profile.py stacks --c 1 2 \
      --mode arena inplace --out ~/w4a8_runs/hi2-gdn-glue

The projections in the harness are plain fp16 matmuls: launches issued from
their lambdas are tagged ``standin`` and are not part of the layer's glue.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.autograd.profiler import record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.kernels.mamba import gdn_glue_harness as hz  # noqa: E402
from tests.kernels.mamba.gdn_glue_harness import GdnDecodeRig  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
# Files whose executed lines get an annotation.
TRACED = (
    "mamba/gdn/qwen_gdn_linear_attn.py",
    "backends/gdn_attn.py",
    "model_executor/layers/layernorm.py",
    "mamba/ops/causal_conv1d.py",
    "tests/kernels/mamba/gdn_glue_harness.py",
)
MARK = "L|"


def _check_device() -> None:
    props = torch.cuda.get_device_properties(0)
    bus = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}"
    print(
        f"device: {props.name} {props.gcnArchName} pci {bus} "
        f"{props.total_memory / 2**30:.2f} GiB (HIP_VISIBLE_DEVICES="
        f"{os.environ.get('HIP_VISIBLE_DEVICES')})",
        flush=True,
    )
    if "V620" not in props.name or "gfx1030" not in props.gcnArchName:
        raise SystemExit("not a V620/gfx1030: refusing to run")


class LineMarkers:
    """Bracket every executed line of ``TRACED`` files with a profiler range."""

    def __init__(self) -> None:
        self._open: dict[object, record_function] = {}

    def _close(self, frame) -> None:
        rf = self._open.pop(frame, None)
        if rf is not None:
            rf.__exit__(None, None, None)

    def _local(self, frame, event, arg):
        if event == "line":
            self._close(frame)
            code = frame.f_code
            rf = record_function(
                f"{MARK}{code.co_filename}|{frame.f_lineno}|{code.co_name}"
            )
            rf.__enter__()
            self._open[frame] = rf
        elif event == "return":
            self._close(frame)
        return self._local

    def _global(self, frame, event, arg):
        if frame.f_code.co_filename.endswith(TRACED):
            return self._local
        return None

    def __enter__(self) -> LineMarkers:
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc) -> None:
        sys.settrace(None)
        for frame in list(self._open):
            self._close(frame)


def _rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return path


def _src(path: str, line: int) -> str:
    try:
        return Path(path).read_text().splitlines()[line - 1].strip()
    except (OSError, IndexError):
        return ""


def trace_launches(trace_path: Path) -> list[dict]:
    """GPU launches of a chrome trace in launch order; each with the innermost
    annotated source line that was executing when the host issued it."""
    events = json.loads(trace_path.read_text())["traceEvents"]
    marks, runtime, gpu = [], {}, []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat == "user_annotation" and e["name"].startswith(MARK):
            marks.append(e)
        elif cat == "cuda_runtime" and "correlation" in e.get("args", {}):
            runtime[e["args"]["correlation"]] = e
        elif cat in GPU_CATS:
            gpu.append(e)
    marks.sort(key=lambda e: e["ts"])
    rows = []
    for e in gpu:
        rt = runtime.get(e["args"].get("correlation"))
        ts = rt["ts"] if rt else e["ts"]
        enclosing = [
            m
            for m in marks
            if m["ts"] <= ts <= m["ts"] + m["dur"] and m["tid"] == (rt or e)["tid"]
        ]
        enclosing.sort(key=lambda m: m["dur"])  # innermost first
        chain = []
        for m in enclosing:
            _, path, line, func = m["name"].split("|", 3)
            chain.append((_rel(path), int(line), func, _src(path, int(line))))
        rows.append(
            {
                "ts": ts,
                "dur": e["dur"],
                "cat": e["cat"],
                "name": e["name"],
                "api": rt["name"] if rt else None,
                "chain": chain,
            }
        )
    rows.sort(key=lambda r: r["ts"])
    return rows


def _short(name: str) -> str:
    name = name.replace("void ", "").replace("(anonymous namespace)::", "")
    for cut in ("(", "<"):
        if cut in name and name.index(cut) > 8:
            name = name[: name.index(cut)]
    return name[:46]


def is_standin(row: dict) -> bool:
    return bool(row["chain"]) and "gdn_glue_harness" in row["chain"][0][0]


def steady_durations(
    rig: GdnDecodeRig,
    md,
    tokens: int,
    gen: torch.Generator,
    per_step: int,
    steps: int,
    trace: Path,
) -> list[float]:
    """Median duration of every launch position over ``steps`` consecutive
    eager steps. One traced step after an idle gap can run on cold clocks and
    caches (the same kernel took 85 us once and 19 us in steady state), so the
    table reports these medians and keeps the single-step value next to them."""
    from torch.profiler import ProfilerActivity, profile

    hiddens = [rig.random_hidden(tokens, gen) for _ in range(steps)]
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for h in hiddens:
            rig.forward(h, md)
        torch.accelerator.synchronize()
    prof.export_chrome_trace(str(trace))
    rows = trace_launches(trace)
    if len(rows) != per_step * steps:
        raise SystemExit(f"expected {per_step * steps} launches, got {len(rows)}")
    return [
        statistics.median(r["dur"] for r in rows[i::per_step]) for i in range(per_step)
    ]


def run_stacks(args: argparse.Namespace) -> None:
    from torch.profiler import ProfilerActivity, profile

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for mode in args.mode:
        rig = GdnDecodeRig(
            inplace=(mode == "inplace"),
            num_blocks=64,
            mamba_cache_mode=args.cache_mode,
            conv_hip=args.conv_hip,
        )
        print(
            f"\n== mode={mode} (builder in-place: {rig.builder_inplace}, "
            f"cache mode {args.cache_mode}, "
            f"conv {'hip' if args.conv_hip else 'triton'})",
            flush=True,
        )
        gen = torch.Generator("cuda").manual_seed(1)
        for c in args.c:
            slots = [5 + 7 * i for i in range(c)]
            pad = args.pad_to if args.pad_to and args.pad_to >= c else None
            md = rig.build_md(slots, pad_to=pad)
            tokens = pad or c
            for _ in range(args.warm_steps):  # JIT / warm clocks and caches
                rig.forward(rig.random_hidden(tokens, gen), md)
            torch.accelerator.synchronize()
            hidden = rig.random_hidden(tokens, gen)
            if args.cold_mib:
                flush_caches(args.cold_mib)
            trace = out_dir / f"trace-{mode}-c{c}.json"
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
            ) as prof:
                with LineMarkers():
                    rig.forward(hidden, md)
                torch.accelerator.synchronize()
            prof.export_chrome_trace(str(trace))
            rows = trace_launches(trace)
            layer_rows = [r for r in rows if not is_standin(r)]
            cats = Counter(
                "memcpy"
                if r["cat"] == "gpu_memcpy"
                else "memset"
                if r["cat"] == "gpu_memset"
                else "kernel"
                for r in layer_rows
            )
            steady = steady_durations(
                rig,
                md,
                tokens,
                gen,
                len(rows),
                args.steady_steps,
                out_dir / "trace-steady.json",
            )
            busy = sum(r["dur"] for r in layer_rows)
            busy_steady = sum(s for r, s in zip(rows, steady) if not is_standin(r))
            print(
                f"\n-- c={c} (tokens {tokens}): {len(layer_rows)} layer launches "
                f"({dict(cats)}), kernel busy {busy_steady:.1f} us steady "
                f"(median of {args.steady_steps} steps; {busy:.1f} us in the one "
                f"traced step); stand-in projection launches "
                f"{len(rows) - len(layer_rows)}"
            )
            table = []
            for i, (r, us) in enumerate(zip(rows, steady)):
                tag = "standin" if is_standin(r) else "layer"
                site = r["chain"][0] if r["chain"] else ("", 0, "", "")
                via = [f"{p}:{ln}" for p, ln, _f, _s in r["chain"][1:3]]
                table.append(
                    {
                        "i": i,
                        "tag": tag,
                        "cat": r["cat"],
                        "kernel": r["name"],
                        "us": round(us, 2),
                        "traced_step_us": round(r["dur"], 2),
                        "api": r["api"],
                        "site": f"{site[0]}:{site[1]} ({site[2]})",
                        "src": site[3],
                        "via": via,
                    }
                )
                print(
                    f"{i:3d} {tag:7s} {us:7.2f}us {_short(r['name']):46s} "
                    f"{site[0].split('/')[-1]}:{site[1]:<5d} {site[3][:74]}"
                )
            report[f"{mode}-c{c}"] = {
                "launches": len(layer_rows),
                "by_cat": dict(cats),
                "busy_us": round(busy_steady, 2),
                "busy_traced_step_us": round(busy, 2),
                "table": table,
            }
        del rig
        torch.accelerator.empty_cache()
    path = out_dir / "stacks.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {path}")


def flush_caches(mib: int) -> None:
    """Read ``mib`` MiB so L2 and the Infinity Cache lose the state pages."""
    buf = torch.ones(mib << 20, dtype=torch.uint8, device="cuda")
    buf.sum()
    torch.accelerator.synchronize()
    del buf


def _replay_us(graph: torch.cuda.CUDAGraph, iters: int, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        torch.accelerator.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / iters)
    return out


def run_timing(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict] = {}
    for mode in args.mode:
        rig = GdnDecodeRig(
            inplace=(mode == "inplace"),
            num_blocks=args.num_blocks,
            num_layers=args.layers,
            mamba_cache_mode=args.cache_mode,
            conv_hip=args.conv_hip,
        )
        rig.thrash = (
            torch.ones(args.thrash_mib << 18, dtype=torch.float32, device="cuda")
            if args.thrash_mib
            else None
        )
        print(
            f"\n== mode={mode} (builder in-place: {rig.builder_inplace}), "
            f"{args.layers} layers, {args.num_blocks} blocks/layer, "
            f"{args.thrash_mib} MiB cache thrash read after every layer",
            flush=True,
        )
        gen = torch.Generator("cuda").manual_seed(2)
        for c in args.c:
            slots = [1 + 2 * i for i in range(c)]  # distinct, inside the pool
            assert slots[-1] < args.num_blocks, "raise --num-blocks"
            md = rig.build_md(slots)
            hidden = rig.random_hidden(c, gen)
            snap = rig.snapshot()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    rig.forward(hidden, md)
            torch.cuda.current_stream().wait_stream(side)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                rig.forward(hidden, md)
            rig.restore(snap)
            torch.accelerator.synchronize()
            for _ in range(args.warmup):
                graph.replay()
            us = _replay_us(graph, args.iters, args.repeats)
            med = statistics.median(us)
            per_layer = med / args.layers
            print(
                f"c={c}: graph replay {med:.1f} us / {args.layers} layers = "
                f"{per_layer:.2f} us/layer (repeats: "
                f"{', '.join(f'{u / args.layers:.2f}' for u in us)})",
                flush=True,
            )
            result[f"{mode}-c{c}"] = {
                "layers": args.layers,
                "median_us": round(med, 2),
                "per_layer_us": round(per_layer, 3),
                "repeats_per_layer_us": [round(u / args.layers, 3) for u in us],
            }
            del graph
        del rig
        torch.accelerator.empty_cache()
    path = out_dir / f"timing-{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {path}")


def _kernel_durations(path: Path, needle: str) -> list[float]:
    events = json.loads(path.read_text())["traceEvents"]
    return [
        e["dur"] for e in events if e.get("cat") == "kernel" and needle in e["name"]
    ]


def _launch_decode(
    mixed: torch.Tensor,
    ba: torch.Tensor,
    weights: dict[str, torch.Tensor],
    out: torch.Tensor,
    state: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    torch.ops._rocm_C.gdn_decode_rdna2(
        mixed,
        ba[:, hz.HV :],
        ba[:, : hz.HV],
        weights["A_log"],
        weights["dt_bias"],
        out,
        state,
        idx,
        hz.K**-0.5,
        True,
    )


def run_kernel(args: argparse.Namespace) -> None:
    """``gdn_decode_rdna2`` alone: a contiguous arena-style state versus the
    paged cache (2 MiB page stride), warm and cold, the kernel's own duration
    from the profiler."""
    from torch.profiler import ProfilerActivity, profile

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    gen = torch.Generator(dev).manual_seed(3)
    weights = hz.make_weights(args.hidden, dev, 3)
    result: dict[str, float] = {}
    max_c = max(args.c)
    arena = torch.randn(max_c + 1, hz.HV, hz.V, hz.K, device=dev, generator=gen) * 0.05
    _raw, _conv, paged = hz.make_paged_state(64, dev)
    paged.normal_(0.0, 0.05, generator=gen)
    # name -> (state tensor, slots for c requests)
    layouts = {
        "arena rows (1.5 MiB stride), slots 1..c": (
            arena,
            lambda c: list(range(1, c + 1)),
        ),
        "paged (2 MiB stride), slots 1..c": (paged, lambda c: list(range(1, c + 1))),
        "paged (2 MiB stride), slots 5+7i": (
            paged,
            lambda c: [5 + 7 * i for i in range(c)],
        ),
    }
    for c in args.c:
        mixed = torch.randn(c, hz.CONV_DIM, device=dev, generator=gen).to(torch.float16)
        ba = torch.randn(c, 2 * hz.HV, device=dev, generator=gen).to(torch.float16)
        out = torch.empty(c, 1, hz.HV, hz.V, device=dev, dtype=torch.float16)
        for name, (state, slots_fn) in layouts.items():
            idx = torch.tensor(slots_fn(c), device=dev, dtype=torch.int32)
            for cold in (False, True):
                call = functools.partial(
                    _launch_decode, mixed, ba, weights, out, state, idx
                )
                for _ in range(3):
                    call()
                torch.accelerator.synchronize()
                trace = out_dir / "trace-kernel.json"
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.reps):
                        if cold:
                            flush_caches(args.cold_mib or 512)
                        call()
                    torch.accelerator.synchronize()
                prof.export_chrome_trace(str(trace))
                durs = _kernel_durations(trace, "gdn_decode")
                med = statistics.median(durs)
                key = f"c={c} {name} {'cold' if cold else 'warm'}"
                result[key] = round(med, 2)
                print(f"{key:62s} median {med:7.2f} us (n={len(durs)})", flush=True)
    path = out_dir / f"kernel-{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["stacks", "timing", "kernel"])
    parser.add_argument("--c", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--mode", nargs="+", choices=["arena", "inplace"], default=["arena", "inplace"]
    )
    parser.add_argument("--cache-mode", choices=["none", "align"], default="align")
    parser.add_argument("--conv-hip", action="store_true")
    parser.add_argument("--pad-to", type=int, default=0)
    parser.add_argument("--out", default="~/w4a8_runs/hi2-gdn-glue")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warm-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--steady-steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cold-mib", type=int, default=0)
    parser.add_argument("--thrash-mib", type=int, default=0)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()
    _check_device()
    if args.cmd == "stacks":
        run_stacks(args)
    elif args.cmd == "kernel":
        run_kernel(args)
    else:
        run_timing(args)


if __name__ == "__main__":
    main()
