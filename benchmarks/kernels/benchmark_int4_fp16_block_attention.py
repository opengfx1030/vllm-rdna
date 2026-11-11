# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated packed INT4 read/dequant + causal QK/PV attention experiment.

Synthetic valid cache words isolate read cost; this does not measure model
quality or validate the quantizer. Layout matches the isolated writer:
cache [2, slots, heads, D/2], uint16 scales [2, slots, heads, D/32].
No vLLM production integration or persistent installation is required.
"""

import argparse
import json
import statistics
from pathlib import Path

import regex as re
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _attention(
    Q,
    Cache,
    Scales,
    Dense,
    Out,
    N: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    h = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q = tl.load(Q + (rows[:, None] * H + h) * D + dims[None, :], rows[:, None] < M, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0.0, tl.float32)
    accumulator = tl.full((BM, D), 0.0, tl.float32)
    for start in range(tl.cdiv(N, BN)):
        tokens = start * BN + tl.arange(0, BN)
        if PACKED:
            off = (tokens[:, None] * H + h) * (D // 2) + dims[None, :] // 2
            group = (tokens[:, None] * H + h) * (D // 32) + dims[None, :] // 32
            kb = tl.load(Cache + off, tokens[:, None] < N, 0).to(tl.int32)
            vb = tl.load(Cache + N * H * (D // 2) + off, tokens[:, None] < N, 0).to(
                tl.int32
            )
            ks = tl.load(Scales + group, tokens[:, None] < N, 0).to(tl.uint16)
            vs = tl.load(Scales + N * H * (D // 32) + group, tokens[:, None] < N, 0).to(
                tl.uint16
            )
            shift = (dims[None, :] % 2) * 4
            kscale = (ks & 0xFFF0).to(tl.uint16).to(tl.float16, bitcast=True)
            vscale = (vs & 0xFFF0).to(tl.uint16).to(tl.float16, bitcast=True)
            kd = (((kb >> shift) & 15) - (ks & 15).to(tl.int32)).to(tl.float32)
            vd = (((vb >> shift) & 15) - (vs & 15).to(tl.int32)).to(tl.float32)
            k = tl.minimum(tl.maximum(kd * kscale.to(tl.float32), -65504), 65504).to(
                tl.float16
            )
            v = tl.minimum(tl.maximum(vd * vscale.to(tl.float32), -65504), 65504).to(
                tl.float16
            )
        else:
            off = (tokens[:, None] * H + h) * D + dims[None, :]
            k = tl.load(Dense + off, tokens[:, None] < N, 0)
            v = tl.load(Dense + N * H * D + off, tokens[:, None] < N, 0)
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * (D**-0.5)
        mask = (tokens[None, :] < N) & (tokens[None, :] <= N - M + rows[:, None])
        scores = tl.where(mask, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp(maximum - new_max)
        probs = tl.exp(scores - new_max[:, None])
        accumulator = accumulator * alpha[:, None]
        accumulator += tl.dot(probs.to(tl.float16), v)
        denominator = denominator * alpha + tl.sum(probs, axis=1)
        maximum = new_max
    result = accumulator / denominator[:, None]
    tl.store(
        Out + (rows[:, None] * H + h) * D + dims[None, :], result, rows[:, None] < M
    )


def decode(cache, scales, d):
    """Exact quantized oracle reconstruction, including finite FP16 clamp."""
    words = scales.to(torch.int32)
    scalar = (words & 0xFFF0).to(torch.int16).view(torch.float16)
    zp = (words & 15).repeat_interleave(32, dim=-1)
    scalar = scalar.repeat_interleave(32, dim=-1)
    data = torch.empty((*cache.shape[:-1], d), device=cache.device, dtype=torch.int32)
    data[..., 0::2] = cache.to(torch.int32) & 15
    data[..., 1::2] = cache.to(torch.int32) >> 4
    return ((data - zp).float() * scalar.float()).clamp(-65504, 65504).half()


def oracle(q, dense):
    m, _, d = q.shape
    n = dense.shape[1]
    scores = torch.einsum("mhd,nhd->hmn", q.float(), dense[0].float()) / d**0.5
    mask = torch.arange(n, device=q.device)[None, :] > (
        n - m + torch.arange(m, device=q.device)[:, None]
    )
    scores.masked_fill_(mask[None, :, :], -torch.inf)
    return torch.einsum("hmn,nhd->mhd", scores.softmax(-1), dense[1].float()).half()


def time_graph(graph, repeats):
    timings = []
    for _ in range(7):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / repeats)
    return {"median_us": statistics.median(timings), "samples_us": timings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--queries", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    n, m, h, d = args.tokens, args.queries, args.heads, args.head_size
    if d not in (32, 64, 128, 256) or not 1 <= m <= n:
        p.error("head-size must be 32/64/128/256; require 1 <= queries <= tokens")
    if (
        not torch.version.hip
        or "gfx10" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        p.error("this experiment requires an externally selected gfx10 HIP device")
    torch.manual_seed(134)
    cache = torch.randint(0, 256, (2, n, h, d // 2), device="cuda", dtype=torch.uint8)
    # Range-stable synthetic scales, with arbitrary zero points in low bits.
    scale = (torch.rand(2, n, h, d // 32, device="cuda") * 0.1 + 0.005).half()
    scales = (
        (scale.view(torch.int16).int() & 0xFFF0)
        | torch.randint(0, 16, scale.shape, device="cuda")
    ).to(torch.uint16)
    dense = decode(cache, scales, d)
    q = torch.randn(m, h, d, device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    results = {
        "args": vars(args) | {"output": str(args.output)},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "scope": (
            "synthetic contiguous single-sequence read microbench; "
            "no model quality claim"
        ),
    }
    handles = {}
    graphs = {}
    for packed in (False, True):
        name = "int4_fp16_block" if packed else "fp16_same_kernel"

        def run(packed=packed):
            return _attention[(h, triton.cdiv(m, 16))](
                q, cache, scales, dense, out, n, h, m, d, packed, 16, 64, num_warps=4
            )

        compiled = run()
        torch.accelerator.synchronize()
        torch.testing.assert_close(out, oracle(q, dense), atol=0.008, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        errors = []
        for seed in (135, 136):
            torch.manual_seed(seed)
            q.copy_(torch.randn_like(q))
            cache.bitwise_xor_(0x55)
            scales.copy_((scales.int() ^ 3).to(torch.uint16))
            dense.copy_(decode(cache, scales, d))
            graph.replay()
            torch.accelerator.synchronize()
            expected = oracle(q, dense)
            torch.testing.assert_close(out, expected, atol=0.008, rtol=0.02)
            errors.append((out.float() - expected.float()).abs().max().item())
        asm = compiled.asm
        isa = asm.get("amdgcn", "")
        results[name] = {
            "eager_and_changed_input_graph_parity": "PASS",
            "changed_input_max_abs": errors,
            "asm_keys": list(asm),
            "dot2_count": len(re.findall(r"\bv_dot2", isa)),
            "fma_count": len(re.findall(r"\bv_fma", isa)),
            "mfma_count": len(re.findall(r"\bv_mfma", isa)),
        }
        if not isa:
            raise RuntimeError(
                "No AMDGCN ISA available; inspect compiler output before timing"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(f".{name}.s").write_text(isa)
        graphs[name] = graph
        handles[name] = run
    # All parity gates for both implementations precede every timing.
    for name in handles:
        results[name]["eager_us"] = triton.testing.do_bench(handles[name]) * 1000
        results[name]["graph"] = time_graph(graphs[name], args.repeats)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
