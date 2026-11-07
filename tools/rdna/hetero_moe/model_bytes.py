# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte table of a Flash-Next checkpoint from its safetensors headers.

Reads only the JSON header of each shard (a local directory or an HTTP(S) base
URL such as ``https://huggingface.co/<repo>/resolve/main``), never the weights.

    python -m tools.rdna.hetero_moe.model_bytes --checkpoint DIR_OR_URL --out table.json

Output ``hetero-moe-model-bytes/1``::

    {
        "format": ...,
        "num_layers": 48,
        "num_experts": 512,
        "expert_bytes": 2534400,  # packed int4 + scales (+ zero points) of one expert
        "mtp": {"layers": 1, "expert_bytes": 9830400, "expert_dtype": "BF16"},
        "active_bytes": {...},  # what the hub holds besides experts, by group
        "hub_active_bytes": N,  # sum of the groups the hub must hold
        "host_bytes": {"ple_ngram": N},  # n-gram tables (host RAM / sidecar)
        "visual_bytes": N,
        "routed_expert_total_bytes": N,
    }

A routed expert is ``layers.L.mlp.experts.E.{gate,up,down}_proj`` with
``weight_packed`` / ``weight_scale`` / ``weight_zero_point`` tensors; the small
``weight_shape`` tensors are metadata and are not counted. MTP experts are the
fused ``mtp.layers.N.mlp.experts.{gate_up,down}_proj`` tensors (bf16 in the
public checkpoint).
"""

import argparse
import json
import os
import struct
import sys
import urllib.request
from collections.abc import Iterable

import regex as re

FORMAT = "hetero-moe-model-bytes/1"
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}  # fmt: skip
# Resident extra of the RDNA2 layout: the loader materializes symmetric zero
# points per expert (w13 [20 groups, 1280/8] + w2 [5 groups, 2560/8], int32).
RDNA2_QZEROS_BYTES = (20 * (1280 // 8) + 5 * (2560 // 8)) * 4

_ROUTED = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_zero_point"
    r"|weight_shape)$"
)
_MTP_EXPERTS = re.compile(
    r"^mtp\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)


def tensor_bytes(dtype: str, shape: Iterable[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * DTYPE_BYTES[dtype]


def _group(name: str) -> str:
    """Account group of a non-expert tensor."""
    if name.startswith("model.visual."):
        return "visual"
    if ".ple.ple_embedding.ngram_embedding." in name:
        return "ple_ngram"
    if name.startswith("mtp."):
        return "mtp_non_expert"
    if name == "lm_head.weight":
        return "lm_head"
    if name.endswith("embed_tokens.weight"):
        return "embed_tokens"
    for key, group in (
        (".self_attn.", "attention"),
        (".linear_attn.", "gdn"),
        ("hyper_connection", "hyper_connection"),
        (".mlp.shared_expert", "shared_expert"),
        (".mlp.gate.", "router_gate"),
        (".ple.", "ple_small"),
    ):
        if key in name:
            return group
    return "other"


def summarize(tensors: Iterable[tuple[str, str, list[int]]], model: str = "") -> dict:
    """Byte table from ``(name, dtype, shape)`` tensor descriptions."""
    per_expert: dict[tuple[int, int], int] = {}
    mtp_total: dict[int, int] = {}
    mtp_experts = 0
    mtp_dtype = None
    groups: dict[str, int] = {}
    for name, dtype, shape in tensors:
        match = _ROUTED.search(name)
        if match and not name.startswith("mtp."):
            if match.group(4) == "weight_shape":
                continue
            key = (int(match.group(1)), int(match.group(2)))
            per_expert[key] = per_expert.get(key, 0) + tensor_bytes(dtype, shape)
            continue
        match = _MTP_EXPERTS.match(name)
        if match:
            layer = int(match.group(1))
            mtp_total[layer] = mtp_total.get(layer, 0) + tensor_bytes(dtype, shape)
            mtp_experts = max(mtp_experts, int(shape[0]))
            mtp_dtype = dtype
            continue
        group = _group(name)
        groups[group] = groups.get(group, 0) + tensor_bytes(dtype, shape)
    if not per_expert:
        raise ValueError("no routed-expert tensors found (wrong checkpoint?)")
    layers = sorted({k[0] for k in per_expert})
    experts_per_layer = {
        layer: sum(1 for k in per_expert if k[0] == layer) for layer in layers
    }
    num_experts = experts_per_layer[layers[0]]
    if set(experts_per_layer.values()) != {num_experts}:
        raise ValueError(f"layers have different expert counts: {experts_per_layer}")
    sizes = set(per_expert.values())
    if len(sizes) != 1:
        raise ValueError(f"routed experts differ in size: {sorted(sizes)[:5]}")
    expert_bytes = sizes.pop()
    table: dict = {
        "format": FORMAT,
        "model": model,
        "num_layers": len(layers),
        "layer_ids": [layers[0], layers[-1]],
        "num_experts": num_experts,
        "expert_bytes": expert_bytes,
        "expert_resident_extra_bytes_rdna2": RDNA2_QZEROS_BYTES,
        "routed_expert_total_bytes": expert_bytes * num_experts * len(layers),
    }
    if mtp_total:
        per_layer = set(mtp_total.values())
        if len(per_layer) != 1:
            raise ValueError("MTP layers differ in size")
        table["mtp"] = {
            "layers": len(mtp_total),
            "num_experts": mtp_experts,
            "expert_bytes": per_layer.pop() // mtp_experts,
            "expert_dtype": mtp_dtype,
        }
    hub_groups = (
        "embed_tokens",
        "lm_head",
        "attention",
        "gdn",
        "hyper_connection",
        "shared_expert",
        "router_gate",
        "ple_small",
        "mtp_non_expert",
        "other",
    )
    table["active_bytes"] = {g: groups.get(g, 0) for g in hub_groups}
    table["hub_active_bytes"] = sum(table["active_bytes"].values())
    table["host_bytes"] = {"ple_ngram": groups.get("ple_ngram", 0)}
    table["visual_bytes"] = groups.get("visual", 0)
    return table


# ---- safetensors headers -----------------------------------------------------


def _local_headers(directory: str) -> Iterable[tuple[str, str, list[int]]]:
    files = sorted(f for f in os.listdir(directory) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"{directory}: no .safetensors files")
    for name in files:
        with open(os.path.join(directory, name), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        yield from _tensors(header)


def _tensors(header: dict) -> Iterable[tuple[str, str, list[int]]]:
    for name, info in header.items():
        if name != "__metadata__":
            yield name, info["dtype"], info["shape"]


def _http(url: str, rng: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return resp.read()


def _remote_headers(base: str) -> Iterable[tuple[str, str, list[int]]]:
    base = base.rstrip("/")
    index = json.loads(_http(f"{base}/model.safetensors.index.json"))
    for shard in sorted(set(index["weight_map"].values())):
        (n,) = struct.unpack("<Q", _http(f"{base}/{shard}", "bytes=0-7"))
        header = json.loads(_http(f"{base}/{shard}", f"bytes=8-{7 + n}"))
        yield from _tensors(header)


def from_checkpoint(location: str) -> dict:
    tensors = (
        _remote_headers(location)
        if location.startswith(("http://", "https://"))
        else _local_headers(location)
    )
    return summarize(tensors, model=location)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    if table.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return table


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="directory or https base URL")
    p.add_argument("--out", help="write the table here (default: stdout)")
    args = p.parse_args(argv)
    table = from_checkpoint(args.checkpoint)
    text = json.dumps(table, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
