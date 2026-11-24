#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memory fit of an HI2 Inkling-2-Mini directory at a tensor-parallel degree.

Reads the safetensors headers only (``safe_open`` + ``get_slice``: no tensor is
loaded), classifies every tensor and prints the per-class bytes plus this rank's
weight estimate and the KV-cache table of the fork's port spec
(``docs/rdna2/inkling2-port-spec.md``: 48 layer-equivalents, 1.5 MiB per token
at TP=1, split over the KV-head groups).

    python tools/inkling_hi2/memory_estimate.py DIR [--tp 2] [--json]

Dense text tensors on an HI2 directory are named ``model.language_model.*``;
the expert stacks are ``...mlp.experts.{gate_up,down}.hi2_*``.  "Sharded" below
is the fork's TP sharding of the stock model: embed/unembed by vocabulary,
attention by KV-head group; norms, routers, message gates and the vision tower
are replicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_SAFETENSORS_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

#: fp16 full-attention KV per token at TP=1: 8 full layers x 4 KiB
#: (8 KV heads x head_dim 128 x K and V x 2 bytes), port spec "32 KiB per token".
KV_BYTES_PER_TOKEN_TP1 = 8 * 8 * 128 * 2 * 2
#: the five sliding-window KV groups and the conv state are window-bound: their
#: planner size does not grow with the context (port spec, conv branch).
SLIDING_FIXED_GIB_TP1 = 1.42
CONV_GIB_TP1 = 0.25


def _tensor_bytes(dtype: str, shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * _SAFETENSORS_BYTES[dtype]


def census(directory: Path) -> dict[str, int]:
    """``{class: bytes}`` of every tensor in the directory's index (or single file)."""
    from safetensors import safe_open

    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        single = directory / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"{directory}: no index and no model.safetensors")
        with safe_open(single, "pt") as f:
            weight_map = dict.fromkeys(f.keys(), single.name)
    per_shard: dict[str, dict[str, int]] = {}
    for name, shard in weight_map.items():
        per_shard.setdefault(shard, {})[name] = (
            "experts"
            if ".mlp.experts." in name
            else ("vision" if name.startswith("model.visual.") else "dense")
        )
    out = {"experts": 0, "dense": 0, "vision": 0}
    for shard, names in per_shard.items():
        with safe_open(directory / shard, "pt") as f:
            for name, cls in names.items():
                s = f.get_slice(name)
                out[cls] += _tensor_bytes(s.get_dtype(), s.get_shape())
    return out


def estimate(bytes_by_class: dict[str, int], tp: int) -> dict[str, float]:
    """Per-rank GiB.  Lower bound: every dense tensor shardable; upper bound:
    dense replicated (the truth is the fork's sharding: embed/unembed split,
    attention split, routers/norms replicated)."""
    gib = 2**30
    kv = KV_BYTES_PER_TOKEN_TP1 / tp
    out = {
        "experts_gib": bytes_by_class["experts"] / tp / gib,
        "vision_gib": bytes_by_class["vision"] / gib,
        "dense_sharded_gib": bytes_by_class["dense"] / tp / gib,
        "dense_replicated_gib": bytes_by_class["dense"] / gib,
        "kv_mib_per_token": kv / 2**20,
        "kv_fixed_gib": (SLIDING_FIXED_GIB_TP1 + CONV_GIB_TP1) / tp,
    }
    out["weights_low_gib"] = (
        out["experts_gib"] + out["vision_gib"] + out["dense_sharded_gib"]
    )
    out["weights_high_gib"] = (
        out["experts_gib"] + out["vision_gib"] + out["dense_replicated_gib"]
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    c = census(args.directory)
    est = estimate(c, args.tp)
    if args.json:
        print(json.dumps({"bytes": c, "per_rank": est, "tp": args.tp}, indent=1))
        return 0
    gib = 2**30
    print(f"{args.directory}  (TP={args.tp})")
    for cls in ("experts", "dense", "vision"):
        print(f"  {cls:8s} {c[cls] / gib:8.3f} GiB")
    print(f"  {'total':8s} {sum(c.values()) / gib:8.3f} GiB")
    print("per rank:")
    print(f"  experts / TP                  {est['experts_gib']:8.3f} GiB")
    print(f"  vision (replicated)           {est['vision_gib']:8.3f} GiB")
    print(
        f"  dense (sharded..replicated)   {est['dense_sharded_gib']:8.3f} .. "
        f"{est['dense_replicated_gib']:.3f} GiB"
    )
    print(
        f"  weights                       {est['weights_low_gib']:8.3f} .. "
        f"{est['weights_high_gib']:.3f} GiB"
    )
    print(f"  full-attention KV, fp16       {est['kv_mib_per_token']:8.3f} MiB/token")
    print(f"  sliding KV + conv (fixed)     {est['kv_fixed_gib']:8.3f} GiB")
    for ctx in (8192, 32768, 131072):
        kv = est["kv_mib_per_token"] * ctx / 1024 + est["kv_fixed_gib"]
        total = est["weights_high_gib"] + kv
        print(f"  {ctx:>6d} context: weights high + KV {total:8.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
