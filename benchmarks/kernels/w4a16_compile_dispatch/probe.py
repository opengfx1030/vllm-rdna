# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which RDNA2 W4A16 op does each GEMM run under vLLM compile? (V620 box)

``RDNA2W4A16LinearKernel.apply_weights`` picks decode / prefill / exllama from
``x.size(0)`` in Python. vLLM compile traces the model once and drops Dynamo's
guards, so that choice may be frozen at the trace-time M. One ``run`` per arm
records what the compiled graph calls, greedy outputs and rough throughput;
``compare`` puts the arms side by side; ``kernels`` summarizes a rocprofv3
kernel-stats CSV. See docs/explore/w4a16-compile-dispatch/README.md.

    M=benchmarks.kernels.w4a16_compile_dispatch.probe
    python -m $M run --model /models/Qwen3.8-27B-AWQ-INT4 --tp 2 --json python.json
    VLLM_RDNA2_W4A16_RUNTIME_DISPATCH=1 python -m $M run ... --json op.json
    python -m $M run ... --enforce-eager --json eager.json
    python -m $M compare python.json op.json eager.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import regex as re

from ._layers import _collect_w4a16_layers

W4A16_OP = re.compile(
    r"(?:torch\.ops\.(?:_rocm_C|_C|vllm)\.|vllm__custom_ops_)"
    r"(gptq_gemm_rdna2_prefill|gptq_gemm_rdna2|gptq_gemm|awq_gemm_rdna2_prefill"
    r"|rdna2_w4a16_gemm|w4a8_gemm_rdna2|moe_w4a8_gemm_rdna2)\b"
)
# Kernel families in rocprofv3 names -> the op that launches them.
KERNEL_FAMILIES = {
    "gemm_q4_kernel_rdna2": "rdna2_decode",
    "gemm_dynamic_kernel": "prefill",
    "gemm_static_kernel": "prefill",
    "gemm_half_q_half": "exllama",
    "w4a8_gemm_kernel": "w4a8_prefill",
}
M_PROBES = (1, 2, 4, 8, 16, 32, 64, 256, 512, 2048)
GREEDY_PROMPTS = [
    "The capital of France is",
    "Write the first five prime numbers:",
    "def fibonacci(n):",
    "Translate to German: good morning",
]


def count_graph_ops(root: Path) -> dict[str, dict[str, int]]:
    """W4A16 op calls per computation_graph.py under a compile cache root."""
    found = {}
    for path in sorted(root.rglob("computation_graph.py")):
        ops = collections.Counter(W4A16_OP.findall(path.read_text()))
        found[str(path.relative_to(root))] = dict(ops)
    return found


def summarize_kernels(path: Path) -> dict[str, dict[str, float]]:
    """Calls and total ms per W4A16 op from a rocprofv3 CSV: kernel stats
    (one row per kernel) or a kernel trace (one row per dispatch, streamed)."""
    out: dict[str, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        def column(*keys: str) -> str | None:
            return next((c for c in fields if all(k in c.lower() for k in keys)), None)

        name = column("name")
        calls, total = column("calls"), column("total")
        start, end = column("start"), column("end")
        if name is None or not ((calls and total) or (start and end)):
            raise KeyError(f"{path} is neither kernel stats nor a kernel trace")
        for r in reader:
            family = next((f for f in KERNEL_FAMILIES if f in r[name]), None)
            if family is None:
                continue
            op = out.setdefault(KERNEL_FAMILIES[family], {"calls": 0, "total_ms": 0.0})
            if calls and total:
                op["calls"] += int(float(r[calls]))
                op["total_ms"] += float(r[total]) / 1e6
            else:
                op["calls"] += 1
                op["total_ms"] += (float(r[end]) - float(r[start])) / 1e6
    return out


def _timed(llm, prompts, params, repeat: int) -> tuple[float, int]:
    times, tokens = [], 0
    for _ in range(repeat):
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        times.append(time.perf_counter() - start)
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return statistics.median(times), tokens


def cmd_run(args) -> dict:
    cache_root = Path(args.cache_root or tempfile.mkdtemp(prefix="vllm-cache-"))
    # A fresh cache per arm: the compile cache key does not include the
    # dispatch env var, so a shared cache would hand one arm the other's graph.
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams
    from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
        _rdna2_w4a16_select_kernel,
    )

    kwargs = json.loads(args.llm_kwargs)
    if not args.enforce_eager:
        kwargs["compilation_config"] = json.loads(args.compilation_config)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        **kwargs,
    )
    layers = llm.apply_model(_collect_w4a16_layers)[0]
    if not layers:
        raise SystemExit("no RDNA2W4A16LinearKernel layers: wrong model or kernel")
    trace_m = llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
    expected = {
        m: dict(
            collections.Counter(
                _rdna2_w4a16_select_kernel(
                    m, k, n, is_awq=awq, w4a8=w4a8, group_size=group
                )
                for _, k, n, awq, group, w4a8 in layers
            )
        )
        for m in sorted({*M_PROBES, trace_m})
    }

    greedy = SamplingParams(temperature=0.0, max_tokens=32)
    texts = [o.outputs[0].text for o in llm.generate(GREEDY_PROMPTS, greedy)]
    decode = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
    prefill = SamplingParams(temperature=0.0, max_tokens=1)
    long_prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 180)
    timing = {}
    for label, prompts, params in (
        ("c1_decode", GREEDY_PROMPTS[:1], decode),
        ("c8_decode", (GREEDY_PROMPTS * 2)[:8], decode),
        ("prefill_2k", [long_prompt], prefill),
    ):
        seconds, tokens = _timed(llm, prompts, params, args.repeat)
        timing[label] = {"s": seconds, "tokens": tokens, "tok_s": tokens / seconds}

    record = {
        "model": args.model,
        "tp": args.tp,
        "enforce_eager": args.enforce_eager,
        "runtime_dispatch": os.environ.get("VLLM_RDNA2_W4A16_RUNTIME_DISPATCH", "0"),
        "w4a8": os.environ.get("VLLM_RDNA2_W4A8_SDOT4", "0"),
        "compilation_config": None if args.enforce_eager else args.compilation_config,
        "trace_m": trace_m,
        "w4a16_layers_per_rank": len(layers),
        "expected_ops_by_m": expected,
        "graph_ops": count_graph_ops(cache_root / "torch_compile_cache")
        if (cache_root / "torch_compile_cache").exists()
        else {},
        "greedy_texts": texts,
        "timing": timing,
        "cache_root": str(cache_root),
    }
    print(json.dumps({k: record[k] for k in ("trace_m", "graph_ops")}, indent=1))
    print(f"expected at trace M={trace_m}: {expected[trace_m]}; at M=1: {expected[1]}")
    return record


def cmd_compare(args) -> None:
    arms = [(p, json.loads(Path(p).read_text())) for p in args.records]
    ref_texts = arms[0][1]["greedy_texts"]
    print(
        "| arm | eager | runtime dispatch | w4a8 | ops in compiled graph | greedy == "
        "first | c=1 tok/s | c=8 tok/s | 2k prefill tok/s |"
    )
    print("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for path, r in arms:
        ops: collections.Counter = collections.Counter()
        for per_file in r["graph_ops"].values():
            ops.update(per_file)
        t = r["timing"]
        print(
            f"| {Path(path).stem} | {r['enforce_eager']} | {r['runtime_dispatch']} "
            f"| {r.get('w4a8', '-')} "
            f"| {dict(ops) or '-'} | {r['greedy_texts'] == ref_texts} "
            f"| {t['c1_decode']['tok_s']:.1f} | {t['c8_decode']['tok_s']:.1f} "
            f"| {t['prefill_2k']['tok_s']:.0f} |"
        )


def cmd_kernels(args) -> None:
    print("| stats file | op | calls | total ms |")
    print("| --- | --- | ---: | ---: |")
    for path in args.stats:
        for op, v in sorted(summarize_kernels(Path(path)).items()):
            print(f"| {Path(path).name} | {op} | {v['calls']} | {v['total_ms']:.1f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="one arm: engine, graph ops, greedy, timing")
    run.add_argument("--model", required=True)
    run.add_argument("--tp", type=int, default=1)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument(
        "--compilation-config",
        default='{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[]}',
        help="as tools/rdna/serve_gfx1030_full.sh passes it",
    )
    run.add_argument("--enforce-eager", action="store_true")
    run.add_argument("--llm-kwargs", default="{}", help="extra LLM(...) JSON")
    run.add_argument("--cache-root", help="VLLM_CACHE_ROOT (default: fresh tmp)")
    run.add_argument("--repeat", type=int, default=3)
    run.add_argument("--json", type=Path)
    cmp_ = sub.add_parser("compare", help="side-by-side table of run records")
    cmp_.add_argument("records", nargs="+")
    kern = sub.add_parser("kernels", help="W4A16 ops in rocprofv3 kernel stats")
    kern.add_argument("stats", nargs="+")
    args = parser.parse_args()

    if args.command == "run":
        record = cmd_run(args)
        if args.json:
            args.json.write_text(json.dumps(record, indent=1))
    elif args.command == "compare":
        cmd_compare(args)
    else:
        cmd_kernels(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
