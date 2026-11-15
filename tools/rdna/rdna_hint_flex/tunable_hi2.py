#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rocBLAS GEMM keys of an rdna_hint_flex Qwen3.5-family checkpoint, per TP rank.

  tunable_hi2.py shapes MODEL --tp 2 --slots gate_up,down --m 1024,2048 > census.csv
  tunable_hi2.py record MODEL --tp 2 --m 49,1024,2048 --dense-m 9,12,32,33,1024
  tunable_hi2.py time   MODEL --tp 2 --m 1024,2048 --dense-m 12,32 --out t.json
  tunable_hi2.py gemm   census.csv --out heuristic.json

``shapes`` (stdlib only) prints ``GemmTunableOp_Half_TN,tn_<R>_<M>_<K>_ld_<K>_<K>_<R>``
lines for the rank's linears: R = output rows of the rank's weight slice, K = its
input columns, M = tokens (the TunableOp key calls the weight rows "m" and the tokens
"n"). Slots:

  gate_up down qkvz ba out qkv o      the hi2 linears (``out`` = GDN out_proj; ``o`` =
                                      attention o_proj: the same shape on this model);
                                      above M=48 they dequantize the slice and run
                                      ``mm_out(x, W.t())`` (rocBLAS)
  lm_head                             dense fp16 head, vocab / tp rows
  fc                                  MTP ``fc`` (column parallel, gathered): hidden /
                                      tp rows, 2 x hidden columns
  d_qkv d_o d_gate_up d_down          the MTP decoder layer: dense fp16 weights of the
                                      same shapes as qkv / o / gate_up / down

``record`` runs the REAL ops on the checkpoint's own per-rank tensors (the hi2 op with
the loader's segments, the dense path through ``rocm_unquantized_gemm_impl`` with the
GEMV caps the launcher sets) under ``PYTORCH_TUNABLEOP_RECORD_UNTUNED=1``: the file
torch writes lists the keys that really reach rocBLAS (a GEMV call writes none). It
fails when a real segment's R or K differs from the formula of ``shapes``.
``time`` times the same real calls (median of CUDA-event windows) and prints TFLOP/s;
run it once without rows and once with ``PYTORCH_TUNABLEOP_FILENAME`` pointing at a
rows file (torch 2.12 ignores rows under RECORD_UNTUNED=1 and ``tunable.enable(False)``
does not unload them: two processes).
``gemm CENSUS --out j.json`` times ``F.linear`` of every key of a census file on cold
rotating weights in a CUDA graph (``--cold-mb``), the same two-process protocol, and
writes the curate_tunableop_rows.py input format.

  HIP_VISIBLE_DEVICES=4 PYTORCH_TUNABLEOP_ENABLED=1 \\
      PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \\
      PYTORCH_TUNABLEOP_UNTUNED_FILENAME=<dir>/untuned.csv \\
      python tunable_hi2.py record ...
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

DECODE_MAX_M = 48  # the hi2 ops run their GEMV up to this many tokens

HI2_SLOTS = ("gate_up", "down", "qkvz", "ba", "out", "qkv", "o")
DENSE_SLOTS = ("lm_head", "fc", "d_qkv", "d_o", "d_gate_up", "d_down")
ALL_SLOTS = HI2_SLOTS + DENSE_SLOTS
# bench_tp2 / tp_layout slot name of each hi2 slot, and the layer kind that has it
CHECKPOINT_SLOT = {
    "gate_up": "gate_up_proj",
    "down": "down_proj",
    "qkvz": "in_proj_qkvz",
    "ba": "in_proj_ba",
    "out": "out_proj",
    "qkv": "qkv_proj",
    "o": "o_proj",
}


def text_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def rank_shapes(cfg: dict, tp: int) -> dict[str, tuple[int, int]]:
    """Slot -> (R, K) of one rank: weight rows and input columns of its slice.

    Mirrors tp_layout.slot_specs (partitions sharded by tp, o_proj / out_proj / down
    row-parallel with the full hidden size as rows); ``record`` checks the formulas
    against the real segments.
    """
    h, inter, hd = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    gate = 2 if cfg.get("attn_output_gate", False) else 1
    kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    nv = cfg["linear_num_value_heads"]
    kv_rows = nkv * hd if nkv < tp else nkv * hd // tp
    qkv = nh * hd * gate // tp + 2 * kv_rows
    shapes = {
        "gate_up": (2 * inter // tp, h),
        "down": (h, inter // tp),
        "qkvz": ((2 * kd + 2 * vd) // tp, h),
        "ba": (2 * nv // tp, h),
        "out": (h, vd // tp),
        "qkv": (qkv, h),
        "o": (h, nh * hd // tp),
        "lm_head": (cfg["vocab_size"] // tp, h),
        "fc": (h // tp, 2 * h),
    }
    shapes["d_qkv"], shapes["d_o"] = shapes["qkv"], shapes["o"]
    shapes["d_gate_up"], shapes["d_down"] = shapes["gate_up"], shapes["down"]
    return shapes


def tn_key(rows: int, tokens: int, cols: int) -> str:
    return f"tn_{rows}_{tokens}_{cols}_ld_{cols}_{cols}_{rows}"


def parse_ints(text: str) -> list[int]:
    """``1,2,5-8`` -> [1, 2, 5, 6, 7, 8]."""
    out: list[int] = []
    for part in filter(None, text.split(",")):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def cmd_shapes(args) -> int:
    shapes = rank_shapes(text_config(args.model), args.tp)
    slots = args.slots.split(",") if args.slots else list(shapes)
    seen: set[str] = set()
    for slot in slots:
        rows, cols = shapes[slot]
        tokens = args.m if slot in HI2_SLOTS else args.dense_m
        for m in parse_ints(tokens):
            key = tn_key(rows, m, cols)
            if key not in seen:  # d_* slots repeat the target keys
                seen.add(key)
                print(f"GemmTunableOp_Half_TN,{key}")
    return 0


def cuda_ms(call, warmup: int, reps: int) -> float:
    """Median CUDA-event time of one call in ms."""
    import torch

    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for _ in range(reps):
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def real_calls(args) -> list[tuple[str, int, int, int, str, object]]:
    """(slot, R, K, M, path, call) for every real call of the requested slots.

    path: ``hi2`` (dequantize + mm_out above 48 tokens); ``dense`` (the dispatch of
    rocm_unquantized_gemm_impl with the default cap of 8 GEMV rows), ``dense32``
    (the same with cap 32, lm_head only: the launcher's VLLM_RDNA_LM_HEAD_GEMV_MAX_M);
    ``gemv`` (gemv_f16_rdna2 alone, up to 32 tokens: what the cap replaces rocBLAS
    with).
    """
    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_tp2 as b2
    from tp_layout import Reader, slot_specs

    cfg = text_config(args.model)
    formula = rank_shapes(cfg, args.tp)
    wanted = args.slots.split(",") if args.slots else list(ALL_SLOTS)
    rd = Reader(args.model)
    calls = []
    for slot in wanted:
        r, k = formula[slot]
        if slot in HI2_SLOTS:
            layer = next(
                i
                for i in range(len(cfg["layer_types"]))
                if any(
                    s["slot"] == CHECKPOINT_SLOT[slot]
                    for s in slot_specs(cfg, args.tp, i)
                )
            )
            spec = next(
                s
                for s in slot_specs(cfg, args.tp, layer)
                if s["slot"] == CHECKPOINT_SLOT[slot]
            )
            segs = b2.build_segments(rd, spec, args.rank)
            if len(segs) != 1:
                raise SystemExit(f"{slot}: {len(segs)} segments (expects 1)")
            d = b2.to_device(segs[0])
            if (d["R"], d["K"]) != (r, k):
                raise SystemExit(
                    f"{slot}: real segment R={d['R']} K={d['K']} != formula {r}x{k}"
                )
            for m in parse_ints(args.m):
                x = torch.randn(m, k, dtype=torch.float16, device="cuda")
                calls.append((slot, r, k, m, "hi2", b2.make_linear(d, x, 0)))
            continue
        w = (torch.randn(r, k, device="cuda") * 0.02).half()
        caps = [("dense", 8)] + ([("dense32", 32)] if slot == "lm_head" else [])
        for m in parse_ints(args.dense_m):
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            for path, cap in caps:
                calls.append(
                    (
                        slot,
                        r,
                        k,
                        m,
                        path,
                        lambda x=x, w=w, cap=cap: rocm_unquantized_gemm_impl(
                            x, w, None, cap
                        ),
                    )
                )
            if m <= 32:
                calls.append(
                    (
                        slot,
                        r,
                        k,
                        m,
                        "gemv",
                        lambda x=x, w=w: ops.gemv_f16_rdna2(x, w, None),
                    )
                )
    return calls


def cmd_record(args) -> int:
    import torch

    calls = real_calls(args)
    for _slot, _r, _k, _m, _path, call in calls:
        call()
    torch.accelerator.synchronize()
    print(f"recorded {len(calls)} real calls", file=sys.stderr)
    return 0


def cmd_time(args) -> int:
    import torch

    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name} {p.gcnArchName} bus {p.pci_bus_id:#x}", file=sys.stderr)
    rows = []
    for slot, r, k, m, path, call in real_calls(args):
        ms = cuda_ms(call, args.warmup, args.reps)
        rows.append(
            {
                "slot": slot,
                "R": r,
                "K": k,
                "M": m,
                "path": path,
                "ms": ms,
                "tflops": 2 * r * k * m / ms / 1e9,
            }
        )
        print(
            f"{slot:9s} {path:8s} R={r:6d} K={k:5d} M={m:5d} {ms:9.4f} ms "
            f"{rows[-1]['tflops']:6.2f} TF/s",
            flush=True,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def cmd_gemm(args) -> int:
    """Per-key time of ``F.linear`` on cold rotating weights, CUDA-graph timed.

    The protocol of bench_tp2.py (and of the prefill GEMM study): each key runs on as
    many distinct random weight tensors as it takes to exceed ``--cold-mb`` (the 128 MB
    Infinity Cache is far below it, as in a forward pass through 64 different layers),
    the calls sit in one CUDA graph and the time per call is graph time / calls
    (median over ``--repeats`` windows, rerun while a KFD eviction hit one). Writes
    ``{"results": {key: ms}}``, the input format of curate_tunableop_rows.py.
    """
    import functools

    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_tp2 as b2
    import regex as re

    key_re = re.compile(r"tn_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)$")

    cold = args.cold_mb << 20
    results: dict[str, float] = {}
    for line in Path(args.census).read_text().splitlines():
        if not line.startswith("GemmTunableOp_Half_TN"):
            continue
        key = line.partition(",")[2].split(",")[0]
        m = key_re.match(key)
        if m is None or key in results:
            continue
        r, tokens, k, lda, ldb, ldc = (int(v) for v in m.groups())
        if (lda, ldb, ldc) != (k, k, r):
            continue
        sets = max(2, -(-cold // (r * k * 2)))
        x = torch.randn(tokens, k, dtype=torch.float16, device="cuda")
        ws = [(torch.randn(r, k, device="cuda") * 0.05).half() for _ in range(sets)]
        calls = [functools.partial(torch.nn.functional.linear, x, w) for w in ws]
        reps = max(1, -(-24 // len(calls)))
        us = statistics.median(b2.timed(calls, reps, args.repeats))
        results[key] = us / 1e3
        print(
            f"{key:44s} {results[key]:9.4f} ms "
            f"{2 * r * k * tokens / results[key] / 1e9:6.2f} TF/s",
            flush=True,
        )
        del calls, ws, x
        torch.accelerator.empty_cache()
    Path(args.out).write_text(json.dumps({"mode": "measure", "results": results}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn in (
        ("shapes", cmd_shapes),
        ("record", cmd_record),
        ("time", cmd_time),
    ):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--tp", type=int, default=2)
        p.add_argument("--rank", type=int, default=0)
        p.add_argument("--slots", default="", help=f"comma list of {ALL_SLOTS}")
        p.add_argument("--m", default="1024,2048", help="tokens of the hi2 slots")
        p.add_argument("--dense-m", default="12", help="tokens of the dense slots")
        if name == "time":
            p.add_argument("--warmup", type=int, default=5)
            p.add_argument("--reps", type=int, default=15)
            p.add_argument("--out", default="")
        p.set_defaults(fn=fn)
    p = sub.add_parser("gemm")
    p.add_argument("census", help="file of GemmTunableOp_Half_TN,<key> lines")
    p.add_argument("--cold-mb", type=int, default=640)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_gemm)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
