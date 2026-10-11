# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V620: do the RDNA2 W4A16 ops dequantize baked or exact, at what cost?

Runs every op the RDNA2 dispatcher can pick (``rdna2_decode``, ``prefill``,
``exllama``) directly on random weights and compares each output with three
references built from the same weights and one fp32 GEMM: exact dequant, the
default build's baked fp16 weights and the exact build's fp16 weights
(``reference.py``, bit for bit). Also times each op. Run once per build:

    M=benchmarks.kernels.w4a16_exact_dequant.check_ops
    python -m $M run --json baked.json
    # rebuild with -DVLLM_RDNA2_W4A16_EXACT_DEQUANT=1 (README), then
    python -m $M run --json exact.json
    python -m $M compare baked.json exact.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from benchmarks.kernels.w4a16_exact_dequant import reference as ref

SHAPES_NK = ((2560, 8704), (6144, 2560), (8704, 2560))
DECODE_M = (1, 4, 8, 16, 32)
PREFILL_M = (624, 2048)
FORMATS = (("uint4", 32), ("uint4b8", 128))  # 27B AWQ; GPTQ / AutoRound
OPS = ("rdna2_decode", "prefill", "exllama")


def _ops():
    from vllm import _custom_ops as ops

    return ops


def device_inputs(p: ref.Problem, fmt: str, device: str = "cuda") -> dict:
    """The tensors the ops receive after RDNA2W4A16LinearKernel's
    process_weights_after_loading: shuffled K-packed nibbles, N-packed
    stored zeros ([G, N/8]), fp16 scales, an empty g_idx."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    wtype = scalar_types.uint4 if fmt == "uint4" else scalar_types.uint4b8
    offset = 0 if fmt == "uint4" else 1  # GPTQv1 stores zero - 1
    q = torch.from_numpy(p.q.astype(np.int32)).to(device)
    stored = torch.from_numpy((p.zeros - offset).astype(np.int32)).to(device)
    w_q = pack_quantized_values_into_int32(q, wtype, packed_dim=0).contiguous()
    g_idx = torch.empty(0, dtype=torch.int, device=device)
    _ops().gptq_shuffle(w_q, g_idx, 4)
    return {
        "x": torch.from_numpy(p.x).to(device),
        "w_q": w_q,
        "qzeros": pack_quantized_values_into_int32(stored, wtype, packed_dim=1),
        "scales": torch.from_numpy(p.scales).to(device),
        "g_idx": g_idx,
        "use_v2_format": fmt == "uint4",
    }


def check_shuffle() -> None:
    """The dequant model assumes gptq_shuffle's slot order; stop early if the
    op on this build disagrees."""
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 2**32, size=(64, 128), dtype=np.uint64)
    packed = packed.astype(np.uint32)
    w = torch.from_numpy(packed.view(np.int32).copy()).cuda()
    _ops().gptq_shuffle(w, torch.empty(0, dtype=torch.int, device="cuda"), 4)
    got = w.cpu().numpy().view(np.uint32)
    if not np.array_equal(got, ref.exllama_shuffle(packed)):
        raise SystemExit("gptq_shuffle does not match reference.exllama_shuffle")


def run_op(name: str, d: dict) -> torch.Tensor:
    ops = _ops()
    args = (d["x"], d["w_q"], d["qzeros"], d["scales"], d["g_idx"])
    if name == "rdna2_decode":
        out = ops.gptq_gemm_rdna2(*args, d["use_v2_format"])
    elif name == "prefill":
        out = ops.gptq_gemm_rdna2_prefill(*args, d["use_v2_format"])
    else:
        out = ops.gptq_gemm(*args, True, d["use_v2_format"], 4)
    return out.clone()  # the RDNA2 ops return persistent buffers


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def verdict(r: dict) -> str:
    exact = min(r["vs_exact"], r["vs_exact_fp16"])
    if exact * 5 < r["vs_baked"]:
        return "exact"
    if r["vs_baked"] * 5 < exact:
        return "baked"
    return "unclear"


def cmd_run(args) -> list[dict]:
    check_shuffle()
    records = []
    cells = [(m, n, k) for m in DECODE_M + PREFILL_M for n, k in SHAPES_NK]
    for fmt, g in FORMATS:
        zeros = "random" if fmt == "uint4" else "symmetric"
        for m, n, k in cells:
            p = ref.make_problem(m, n, k, g, zeros, seed=args.seed)
            d = device_inputs(p, fmt)
            wargs = (p.q, p.zeros, p.scales, g)
            x32 = d["x"].float()
            refs = {
                name: x32 @ torch.from_numpy(w).cuda().float()
                for name, w in (
                    ("exact", ref.dequant_exact(*wargs)),
                    ("baked", ref.dequant_baked(*wargs)),
                    ("exact_fp16", ref.dequant_exact_fp16(*wargs)),
                )
            }
            for op in OPS:
                if op == "rdna2_decode" and m > 64:
                    continue
                out = run_op(op, d).float()
                r = {"m": m, "n": n, "k": k, "fmt": fmt, "group_size": g, "op": op}
                for name, c in refs.items():
                    r[f"vs_{name}"] = float((out - c).norm() / c.norm())
                r["verdict"] = verdict(r)
                r["us"] = time_us(
                    lambda op=op, d=d: run_op(op, d), args.warmup, args.iters
                )
                records.append(r)
                print(
                    f"{fmt} G={g} {m}x{n}x{k} {op:12s} vs exact {r['vs_exact']:.2e} "
                    f"baked {r['vs_baked']:.2e} -> {r['verdict']:7s} {r['us']:9.1f} µs"
                )
    return records


def cmd_compare(args) -> None:
    def key(r):
        return (r["fmt"], r["m"], r["n"], r["k"], r["op"])

    base = {key(r): r for r in json.loads(Path(args.baseline).read_text())}
    new = {key(r): r for r in json.loads(Path(args.candidate).read_text())}
    print(
        "| fmt | M×N×K | op | verdict | vs exact | µs | verdict | vs exact | µs "
        "| time ratio |"
    )
    print("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |")
    ratios: dict[str, list[float]] = {}
    for k in sorted(base.keys() & new.keys()):
        a, b = base[k], new[k]
        ratio = b["us"] / a["us"]
        phase = "decode" if a["m"] <= 32 else "prefill"
        ratios.setdefault(f"{a['op']} {phase}", []).append(ratio)
        print(
            f"| {a['fmt']} | {a['m']}×{a['n']}×{a['k']} | {a['op']} "
            f"| {a['verdict']} | {a['vs_exact']:.1e} | {a['us']:.1f} "
            f"| {b['verdict']} | {b['vs_exact']:.1e} | {b['us']:.1f} | {ratio:.3f} |"
        )
    print("\n| op, phase | geomean time ratio (candidate / baseline) | cells |")
    print("| --- | ---: | ---: |")
    for name, rs in sorted(ratios.items()):
        print(f"| {name} | {statistics.geometric_mean(rs):.3f} | {len(rs)} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="accuracy and timing of every op, one build")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--iters", type=int, default=20)
    run.add_argument("--json", type=Path)
    cmp_ = sub.add_parser("compare", help="baseline build vs candidate build")
    cmp_.add_argument("baseline")
    cmp_.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "run":
        records = cmd_run(args)
        if args.json:
            args.json.write_text(json.dumps(records, indent=1))
        return 1 if any(r["verdict"] == "unclear" for r in records) else 0
    cmd_compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
