# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cost of native INT4 cache writes against the existing FP16 writer."""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_native import check_writer, timed_arms
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--writer-source", type=Path)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 256, 1024, 16384]
    )
    parser.add_argument("--reverse-arm-order", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--check-block-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--module-tag", default="")
    parser.add_argument("--compare-writer-module")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--fixed-source", action="store_true")
    args = parser.parse_args()
    sys.path.insert(
        0, str(Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "pr134_native_fa")
    )
    import pr134_native_fa as module

    source_sha256 = None
    if args.writer_source:
        source_sha256 = hashlib.sha256(args.writer_source.read_bytes()).hexdigest()
        name = f"pr134_writer_t{args.threads}_w{args.width}"
        if args.module_tag:
            name += "_" + args.module_tag
        candidate = load(
            name=name,
            sources=[
                str(args.writer_source),
                str(Path(__file__).with_name("writer_bindings.cpp")),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx1030", "-mno-wavefrontsize64"]
            + (
                []
                if args.fixed_source
                else [
                    f"-DPR134_WRITER_MICRO_THREADS={args.threads}",
                    f"-DPR134_WRITER_MICRO_WIDTH={args.width}",
                ]
            ),
            verbose=True,
        )
        module = SimpleNamespace(
            write_packed=candidate.write_packed, write_fp16=module.write_fp16
        )
    correctness = {
        str(bs): check_writer(module, page_size=bs) for bs in args.check_block_sizes
    }
    if args.check_only:
        result = {
            "correctness": correctness,
            "writer_sha256": source_sha256,
            "compile_geometry": "fixed source"
            if args.fixed_source
            else f"micro override {args.threads}/{args.width}",
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    previous = None
    if args.compare_writer_module:
        sys.path.insert(
            0,
            str(Path(os.environ["TORCH_EXTENSIONS_DIR"]) / args.compare_writer_module),
        )
        previous = importlib.import_module(args.compare_writer_module)
    rows = []
    for dim in (128, 256):
        for tokens in args.tokens:
            heads, bs = 8, args.block_size
            nb = (tokens + bs - 1) // bs
            kv = torch.randn(2, tokens, heads, dim, dtype=torch.float16, device="cuda")
            slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
            inline = torch.empty(
                2, nb, bs, heads, dim // 2 + dim // 16, device="cuda", dtype=torch.uint8
            )
            kc = torch.empty(
                nb, heads, dim // 8, bs, 8, device="cuda", dtype=torch.float16
            )
            vc = torch.empty(nb, heads, dim, bs, device="cuda", dtype=torch.float16)
            fp16 = lambda kv=kv, kc=kc, vc=vc, slots=slots: module.write_fp16(
                kv[0], kv[1], kc, vc, slots
            )
            packed = lambda kv=kv, inline=inline, slots=slots: module.write_packed(
                kv[0], kv[1], inline, slots
            )
            # Both writer variants must capture/replay changed input.
            for fn in (fp16, packed):
                fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                for _ in range(2):
                    kv.mul_(0.9).add_(0.1)
                    graph.replay()
                    expected = [x.clone() for x in (kc, vc, inline)]
                    fn()
                    selected = (0, 1) if fn is fp16 else (2,)
                    for index in selected:
                        actual = (kc, vc, inline)[index].view(torch.uint8)
                        reference = expected[index].view(torch.uint8)
                        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            arms = {"fp16": fp16, "int4_fp16": packed}
            if previous is not None:
                arms["int4_previous"] = lambda kv=kv, inline=inline, slots=slots: (
                    previous.write_packed(kv[0], kv[1], inline, slots)
                )
            if args.reverse_arm_order:
                arms = dict(reversed(list(arms.items())))
            timings = timed_arms(arms, 30)
            rows.append(
                {
                    "tokens": tokens,
                    "heads": heads,
                    "dim": dim,
                    **timings,
                    "allocated_fp16_bytes": 2 * nb * bs * heads * dim * 2,
                    "allocated_int4_bytes": inline.numel(),
                }
            )
    result = {
        "correctness": correctness,
        "measurements": rows,
        "writer_sha256": source_sha256,
        "threads": args.threads,
        "width": args.width,
        "module_tag": args.module_tag,
        "previous_module": args.compare_writer_module,
        "block_size": args.block_size,
        "timing_order": "seven rounds, rotated and reversed arm order",
        "scope": "isolated cache writer; no serving gain claim",
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
