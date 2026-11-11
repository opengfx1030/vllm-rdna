# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native FA-RDNA2 synthetic packed-cache correctness and cost screen."""

import argparse
import hashlib
import importlib.util
import json
import math
import runpy
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def check_writer(module, page_size=16):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import benchmark_int4_fp16_block_writer as reference

    def native_adapter(kv, slots, cache, scales):
        _, capacity, heads, halfdim = cache.shape
        dim = halfdim * 2
        groups = dim // 32
        bs = page_size
        nb = math.ceil(capacity / bs)
        record = halfdim + groups * 2
        # Padded final axis makes every higher cache stride noncompact.
        backing = torch.empty(
            2, nb, bs, heads, record + 8, dtype=torch.uint8, device="cuda"
        )
        inline = backing[..., :record]
        flat = inline.reshape(2, nb * bs, heads, record)
        flat[:, :capacity, :, :halfdim].copy_(cache)
        words = flat[..., halfdim:].view(torch.uint16)
        words[:, :capacity].copy_(scales)
        strided = torch.empty(
            *kv.shape[:-1], dim + 8, dtype=torch.float16, device="cuda"
        )[..., :dim]
        strided.copy_(kv)
        module.write_packed(strided[0], strided[1], inline, slots)
        cache.copy_(flat[:, :capacity, :, :halfdim])
        scales.copy_(words[:, :capacity])

    reference.write_int4_fp16_block = native_adapter
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = reference.check_correctness()
    torch.cuda.current_stream().wait_stream(stream)
    kv = torch.randn(2, 17, 2, 128, dtype=torch.float16, device="cuda")
    slots = torch.arange(17, dtype=torch.int64, device="cuda")
    inline = torch.empty(2, 2, 16, 2, 72, dtype=torch.uint8, device="cuda")
    fn = lambda: module.write_packed(kv[0], kv[1], inline, slots)
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(2):
        kv.mul_(0.9).add_(0.1)
        graph.replay()
        actual = inline.clone()
        fn()
        torch.testing.assert_close(inline, actual, atol=0, rtol=0)
    result["changed_input_graph"] = "PASS"
    codec = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "docs/explore/kv-int4-fp16-scales/reference_codec.py"
        )
    )
    for dim in (128, 256):
        # T64 enters the large-token writer specialization. The first row's
        # tiny negative endpoint detects cancellation in FP32 range subtraction.
        values = torch.randn(2, 64, 1, dim).half()
        values[:, 0, 0] = torch.tensor([15.0, -(2**-24)] * (dim // 2))
        expected_data = torch.empty(2, 64, 1, dim // 2, dtype=torch.uint8)
        expected_words = torch.empty(2, 64, 1, dim // 32, dtype=torch.uint16)
        for side in range(2):
            for token in range(64):
                row = values[side, token, 0].tolist()
                words = codec["quantize_int4_fp16_block_packed"]([row]).scales[0]
                codes = []
                for group, word in enumerate(words):
                    scale = codec["f16_from_bits"](word & ~15)
                    zp = word & 15
                    codes.extend(
                        max(0, min(15, codec["_round_half_away"](v / scale) + zp))
                        for v in row[group * 32 : (group + 1) * 32]
                    )
                expected_data[side, token, 0] = torch.tensor(
                    list(codec["pack_int4"](codes)), dtype=torch.uint8
                )
                expected_words[side, token, 0] = torch.tensor(words, dtype=torch.uint16)
        data, words = (
            torch.empty_like(expected_data, device="cuda"),
            torch.empty_like(expected_words, device="cuda"),
        )
        native_adapter(values.cuda(), torch.arange(64, device="cuda"), data, words)
        torch.testing.assert_close(data.cpu(), expected_data, atol=0, rtol=0)
        torch.testing.assert_close(words.cpu(), expected_words, atol=0, rtol=0)
    result["large_T64_and_range_cancellation_edge"] = "PASS"
    return result


def timed_graph(fn, repeats):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(7):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return {"median_us": sorted(samples)[3], "samples_us": samples}


def timed_arms(functions, repeats):
    """Counterbalance arm position across seven event-timed graph samples."""
    graphs, samples = {}, {name: [] for name in functions}
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    names = list(graphs)
    for iteration in range(7):
        shift = (iteration // 2) % len(names) if len(names) > 2 else 0
        order = names[shift:] + names[:shift]
        if iteration % 2:
            order = order[::-1]
        for name in order:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(repeats):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repeats)
    return {
        name: {"median_us": sorted(values)[3], "samples_us": values}
        for name, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--fp16-only", action="store_true")
    parser.add_argument("--fa-source", type=Path)
    parser.add_argument("--inline-cache", action="store_true")
    parser.add_argument("--kv-splits", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--skip-writer-check", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--compare-native-library", type=Path)
    parser.add_argument("--module-tag", default="")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--misalign-data", action="store_true")
    parser.add_argument("--strided-data", action="store_true")
    parser.add_argument("--stress-scales", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sources = [
        args.fa_source or root / "csrc/rocm/fa_rdna2.cu",
        root / "csrc/rocm/int4_fp16_kv_cache.cu",
        Path(__file__).with_name("native_bindings.cpp"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    name = "pr134_native_fa_old" if args.fp16_only else "pr134_native_fa"
    if args.module_tag:
        name += "_" + args.module_tag
    module = load(
        name=name,
        sources=[str(p) for p in sources],
        extra_include_paths=[str(root / "csrc/rocm")],
        extra_cflags=["-O3"] + (["-DPR134_FP16_ONLY"] if args.fp16_only else []),
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1030", "-mno-wavefrontsize64"],
        verbose=True,
    )
    previous = None
    if args.compare_native_library:
        spec = importlib.util.spec_from_file_location(
            args.compare_native_library.stem, args.compare_native_library
        )
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        assert previous is not module and previous.__file__ != module.__file__
    if args.build_only:
        print(json.dumps({"build": "PASS", "hashes": hashes}, indent=2))
        return
    writer_correctness = (
        "unchanged writer previously qualified"
        if args.skip_writer_check
        else check_writer(module)
    )
    torch.manual_seed(134)
    length, nq, d, hq, hk = (
        args.length,
        args.queries,
        args.dim,
        args.q_heads,
        args.kv_heads,
    )
    bs = args.block_size
    nb = (length + bs - 1) // bs
    shape = (nb, hk, bs, d // 2)
    packed = [
        torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]
    scales = [
        torch.full((nb, hk, bs, d // 32), 0x2C08, dtype=torch.uint16, device="cuda")
        for _ in range(2)
    ]
    if args.inline_cache:
        record = d // 2 + 2 * (d // 32)
        inline = torch.empty(2, nb, bs, hk, record, dtype=torch.uint8, device="cuda")
        payload = [inline[side, ..., : d // 2].permute(0, 2, 1, 3) for side in range(2)]
        metadata = [
            inline[side, ..., d // 2 :].view(torch.uint16).permute(0, 2, 1, 3)
            for side in range(2)
        ]
        for target, source in zip(payload + metadata, packed + scales):
            target.copy_(source)
        packed, scales = payload, metadata
    if args.misalign_data:
        replacements = []
        for data in packed:
            storage = torch.empty(data.numel() + 1, dtype=torch.uint8, device="cuda")
            shifted = storage[1:].reshape(data.shape)
            shifted.copy_(data)
            replacements.append(shifted)
        packed = replacements
    if args.strided_data:
        replacements = []
        for data in packed:
            storage = torch.empty(
                *data.shape[:-1], data.shape[-1] * 2, dtype=torch.uint8, device="cuda"
            )
            strided = storage[..., ::2]
            strided.copy_(data)
            replacements.append(strided)
        packed = replacements
    if args.stress_scales:
        raw = torch.tensor(
            [2**-19, 2**-14, 0.0625, 1.0, 4096.0, 8192.0],
            dtype=torch.float16,
            device="cuda",
        ).view(torch.uint16)
        raw = raw.to(torch.int32) & 0xFFF0
        for side, metadata in enumerate(scales):
            index = torch.arange(metadata.numel(), device="cuda").reshape(
                metadata.shape
            )
            words = raw[index % raw.numel()] | ((index + side) % 16)
            metadata.copy_(words.to(torch.uint16))
    table = torch.randperm(nb, device="cuda", dtype=torch.int32)[None]
    # Each sequence uses the same immutable prefix pages; queries differ.
    table = table.repeat(args.batch, 1)
    seq = torch.full((args.batch,), length, dtype=torch.int32, device="cuda")
    cu = torch.arange(args.batch + 1, dtype=torch.int32, device="cuda") * nq
    q = torch.randn(args.batch * nq, hq, d, dtype=torch.float16, device="cuda")
    kc = torch.empty(nb, hk, d // 8, bs, 8, dtype=torch.float16, device="cuda")
    vc4 = torch.empty(nb, hk, d, bs, dtype=torch.float16, device="cuda")
    vc = vc4.view(nb, hk, d // 8, 8, bs).permute(0, 1, 2, 4, 3)
    slots = torch.arange(nb * bs, device="cuda", dtype=torch.int64)
    outs = {
        name: torch.empty_like(q)
        for name in (
            "packed_decode",
            "packed_prefill",
            "fp16_decode",
            "fp16_prefill",
            "fp16_prefill_gqa",
        )
    }
    if previous is not None:
        outs.update(
            {
                name: torch.empty_like(q)
                for name in ("previous_packed_decode", "previous_packed_prefill")
            }
        )
    if args.fp16_only:
        outs = {name: out for name, out in outs.items() if name.startswith("fp16")}
    if args.decode_only:
        outs = {name: out for name, out in outs.items() if name.endswith("decode")}
    if args.prefill_only:
        outs = {name: out for name, out in outs.items() if not name.endswith("decode")}

    def populate():
        dense = []
        for data, metadata in zip(packed, scales):
            code = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).float()
            bits = metadata.to(torch.int32)
            scale = (bits & 0xFFF0).to(torch.uint16).view(torch.float16).float()
            zp = (bits & 15).float()
            values = (
                (
                    (code - zp.repeat_interleave(32, -1))
                    * scale.repeat_interleave(32, -1)
                )
                .clamp(-65504, 65504)
                .half()
            )
            dense.append(values)
        rows = [
            x.permute(0, 2, 1, 3).reshape(nb * bs, hk, d).contiguous() for x in dense
        ]
        module.write_fp16(*rows, kc, vc4, slots)
        physical = table[0].long()
        logical = [
            x[physical]
            .permute(1, 0, 2, 3)
            .reshape(hk, nb * bs, d)[:, :length]
            .repeat_interleave(hq // hk, 0)
            .float()
            for x in dense
        ]
        logits = (
            torch.einsum(
                "bqhd,htd->bhqt", q.view(args.batch, nq, hq, d).float(), logical[0]
            )
            * d**-0.5
        )
        mask = torch.arange(length, device="cuda")[None] > (
            length - nq + torch.arange(nq, device="cuda")[:, None]
        )
        probs = logits.masked_fill(mask[None, None], float("-inf")).softmax(-1)
        return (
            torch.einsum("bhqt,htd->bqhd", probs, logical[1])
            .reshape(args.batch * nq, hq, d)
            .half()
        )

    def call(name):
        out = outs[name]
        selected = previous if name.startswith("previous_") else module
        core = name.removeprefix("previous_")
        if core.startswith("packed"):
            selected.packed(
                q,
                *packed,
                *scales,
                table,
                cu,
                seq,
                bs,
                args.kv_splits,
                0,
                d**-0.5,
                name.endswith("decode"),
                out,
            )
        elif name == "fp16_decode":
            module.decode(
                q, kc, vc, table, seq, bs, args.kv_splits, 0, d**-0.5, out, cu
            )
        else:
            fn = module.prefill_gqa if name.endswith("gqa") else module.prefill
            fn(q, kc, vc, table, cu, seq, bs, 1, 0, d**-0.5, out)

    parity = {}
    oracle = populate()
    for name, out in outs.items():
        call(name)
        torch.testing.assert_close(out, oracle, atol=0.003, rtol=0.03)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(name)
        errors = []
        for iteration in range(2):
            q.mul_(0.9).add_(0.01)
            packed[0].bitwise_xor_(17)
            if args.stress_scales:
                scales[1].view(torch.int16).bitwise_xor_(1)
            else:
                scales[1].fill_(0x2C08 + iteration)
            oracle = populate()
            graph.replay()
            torch.testing.assert_close(out, oracle, atol=0.003, rtol=0.03)
            errors.append((out.float() - oracle.float()).abs().max().item())
        parity[name] = errors
    # Every arm times the same final Q/K/V after the correctness mutations.
    # Reset also makes FP16-only controls consume exactly the same values.
    torch.manual_seed(134)
    q.normal_()
    for data, metadata in zip(packed, scales):
        data.random_(0, 256)
        metadata.fill_(0x2C08)
    populate()
    measurements = (
        {}
        if args.check_only
        else timed_arms({name: lambda n=name: call(n) for name in outs}, args.repeats)
    )
    result = {
        "args": vars(args)
        | {
            "output": str(args.output),
            "fa_source": str(args.fa_source),
            "compare_native_library": str(args.compare_native_library),
        },
        "source_sha256": hashes,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.cuda.get_device_properties(0)),
        "cache_bytes": {
            "fp16": kc.numel() * kc.element_size() + vc4.numel() * vc4.element_size(),
            "int4_fp16": sum(t.numel() * t.element_size() for t in packed + scales),
        },
        "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
        "previous_library_sha256": (
            hashlib.sha256(args.compare_native_library.read_bytes()).hexdigest()
            if args.compare_native_library
            else None
        ),
        "scope": (
            "synthetic paged GQA, shared prefix across sequences; "
            "no model quality claim"
        ),
        "timing_order": "seven rounds, rotated and reversed arm order",
        "writer_exact_strided_nondefault_stream": writer_correctness,
        "changed_input_graph_maxabs": parity,
        "timings": measurements,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
