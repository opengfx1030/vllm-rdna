# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone scratch writer for PR134; no serving-path integration.

Run with the frozen ROCm venv and an explicitly reserved HIP device. Correctness
is checked before timing. Cache slots flatten (page, token-in-page). Metadata is
uint16, not a numerical FP16 tensor: its low nibble holds the zero-point.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _round_away(x):
    return tl.where(x >= 0, tl.floor(x + 0.5), -tl.floor(-x + 0.5)).to(tl.int32)


@triton.jit
def _write(
    X,
    Slots,
    Cache,
    Scales,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    token = tl.program_id(0)
    head_group = tl.program_id(1)
    kv = tl.program_id(2)
    groups: tl.constexpr = DIM // 32
    head = head_group // groups
    group = head_group % groups
    slot = tl.load(Slots + token).to(tl.int64)
    if slot < 0:
        return
    offsets = tl.arange(0, 16)
    src = ((kv * TOKENS + token) * HEADS + head) * DIM + group * 32
    even = tl.load(X + src + 2 * offsets).to(tl.float32)
    odd = tl.load(X + src + 2 * offsets + 1).to(tl.float32)
    lo = tl.minimum(0.0, tl.minimum(tl.min(even, 0), tl.min(odd, 0)))
    hi = tl.maximum(0.0, tl.maximum(tl.max(even, 0), tl.max(odd, 0)))
    ideal = tl.maximum(tl.div_rn(hi - lo, 15.0), 1.0e-6)
    half_bits = ideal.to(tl.float16).to(tl.uint16, bitcast=True)
    scale_bits = half_bits & 0xFFF0
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
    scale_bits += tl.where(scale < ideal, 16, 0).to(tl.uint16)
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
    zp = tl.minimum(15, tl.maximum(0, _round_away(tl.div_rn(-lo, scale))))
    bump = (_round_away(tl.div_rn(lo, scale)) + zp < 0) | (
        _round_away(tl.div_rn(hi, scale)) + zp > 15
    )
    scale_bits += tl.where(bump, 16, 0).to(tl.uint16)
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
    zp = tl.minimum(15, tl.maximum(0, _round_away(tl.div_rn(-lo, scale))))
    qe = tl.minimum(15, tl.maximum(0, _round_away(tl.div_rn(even, scale)) + zp))
    qo = tl.minimum(15, tl.maximum(0, _round_away(tl.div_rn(odd, scale)) + zp))
    dst = ((kv * CAPACITY + slot) * HEADS + head) * (DIM // 2) + group * 16
    tl.store(Cache + dst + offsets, (qe | (qo << 4)).to(tl.uint8))
    metadata = ((kv * CAPACITY + slot) * HEADS + head) * groups + group
    tl.store(Scales + metadata, scale_bits | zp.to(tl.uint16))


def write_int4_fp16_block(kv, slot_mapping, cache, scales):
    """Write FP16 [2,T,H,D] into packed [2,S,H,D/2], scales [2,S,H,D/32].

    Inputs must be finite. Slot IDs must be unique valid physical slot IDs or
    -1 (padding). Group width is 32; no RHT is applied. The reader reconstructs
    (q-zp)*scale in FP32 then clamps to +/-65504 before the FP16 conversion.
    """
    assert kv.ndim == 4 and kv.shape[0] == 2
    _, tokens, heads, dim = kv.shape
    capacity = cache.shape[1]
    assert dim % 32 == 0
    assert kv.dtype == torch.float16 and slot_mapping.dtype == torch.int64
    assert cache.dtype == torch.uint8 and scales.dtype == torch.uint16
    assert cache.shape == (2, capacity, heads, dim // 2)
    assert scales.shape == (2, capacity, heads, dim // 32)
    assert slot_mapping.shape == (tokens,)
    assert all(t.is_contiguous() for t in (kv, slot_mapping, cache, scales))
    if tokens:
        _write[(tokens, heads * (dim // 32), 2)](
            kv,
            slot_mapping,
            cache,
            scales,
            tokens,
            heads,
            dim,
            capacity,
            num_warps=1,
        )


def check_correctness():
    """Compare bytes and metadata with the finite reconstruction CPU codec."""
    codec = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "docs/explore/kv-int4-fp16-scales/reference_codec.py"
        )
    )
    codec["_self_check"]()
    generator = torch.Generator().manual_seed(134)
    checked = 0
    for dim in (32, 64, 128, 256):
        tokens, heads, capacity = 19, 3, 43
        kv = torch.randn(2, tokens, heads, dim, generator=generator).half()
        special = (
            [0.0] * 32,
            [1.0] * 32,
            [-1.0] * 32,
            [65504.0] * 32,
            [-65504.0] * 32,
            [-19936.0, 65504.0] * 16,
            [-65504.0, 65504.0] * 16,
            [2**-24, -(2**-24)] * 16,
        )
        for token, values in enumerate(special):
            kv[:, token, :, :32] = torch.tensor(values).half()
        # Exercise the complete finite FP16 exponent range, including scales
        # and code boundaries absent from unit-normal activation samples.
        bits = torch.randint(0, 0x7C00, kv[:, 10:].shape, generator=generator)
        bits |= torch.randint(0, 2, bits.shape, generator=generator) << 15
        kv[:, 10:] = bits.to(torch.uint16).view(torch.float16)
        slots = torch.tensor(
            [40, 3, 31, 0, 22, 17, 9, 41, -1, 5, 14, 39, 26, -1, 11, 7, 30, 1, 35],
            dtype=torch.int64,
        )
        expected = torch.full((2, capacity, heads, dim // 2), 0xA5, dtype=torch.uint8)
        expected_scales = torch.full(
            (2, capacity, heads, dim // 32), 0xA5A5, dtype=torch.uint16
        )
        for kind in range(2):
            for token, slot in enumerate(slots.tolist()):
                if slot < 0:
                    continue
                for head in range(heads):
                    row = kv[kind, token, head].tolist()
                    result = codec["quantize_int4_fp16_block_packed"]([row])
                    words = result.scales[0]
                    expected_scales[kind, slot, head] = torch.tensor(
                        words, dtype=torch.uint16
                    )
                    codes = []
                    for group, word in enumerate(words):
                        scale = codec["f16_from_bits"](word & ~15)
                        zp = word & 15
                        codes.extend(
                            max(0, min(15, codec["_round_half_away"](v / scale) + zp))
                            for v in row[group * 32 : (group + 1) * 32]
                        )
                    expected[kind, slot, head] = torch.tensor(
                        list(codec["pack_int4"](codes)), dtype=torch.uint8
                    )
                    checked += len(words)
        actual, actual_scales = expected.cuda(), expected_scales.cuda()
        actual.fill_(0xA5)
        actual_scales.fill_(0xA5A5)
        write_int4_fp16_block(kv.cuda(), slots.cuda(), actual, actual_scales)
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales.cpu(), expected_scales, rtol=0, atol=0)
    return {"passed": True, "blocks": checked, "head_dims": [32, 64, 128, 256]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 256, 1024, 16384]
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rep", type=int, default=100, help="Timing milliseconds")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    assert str(getattr(props, "gcnArchName", "")).startswith("gfx10"), props
    result = {
        "device": props.name,
        "arch": props.gcnArchName,
        "correctness": check_correctness(),
        "measurements": [],
    }
    if not args.check_only:
        for tokens in args.tokens:
            kv = torch.randn(
                2, tokens, args.heads, args.head_dim, device="cuda", dtype=torch.float16
            )
            slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
            cache = torch.empty(
                2,
                tokens,
                args.heads,
                args.head_dim // 2,
                device="cuda",
                dtype=torch.uint8,
            )
            scales = torch.empty(
                2,
                tokens,
                args.heads,
                args.head_dim // 32,
                device="cuda",
                dtype=torch.uint16,
            )
            ms = triton.testing.do_bench(
                lambda kv=kv, slots=slots, cache=cache, scales=scales: (
                    write_int4_fp16_block(kv, slots, cache, scales)
                ),
                warmup=25,
                rep=args.rep,
                return_mode="median",
            )
            byte_count = kv.numel() * 2 + cache.numel() + scales.numel() * 2
            assert math.isfinite(ms) and ms > 0
            result["measurements"].append(
                {
                    "tokens": tokens,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "median_ms": ms,
                    "logical_bytes": byte_count,
                    "logical_GB_s": byte_count / (ms * 1e6),
                }
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
