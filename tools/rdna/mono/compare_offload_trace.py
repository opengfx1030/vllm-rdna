#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare explicitly selected, single-request GPU-repeat trace steps.

The completion proof must show identical generated token IDs. Sequence lengths
in the trace must also match: a graph bucket alone is not an alignment key.
This is a debug comparison, not a replacement for the KV numerical gate.
"""

import argparse
import json
from pathlib import Path

import torch

_INPUTS = ("residual", "attention", "injection")


def alignment(record):
    if record["rows"].numel() != 1:
        raise ValueError("comparison requires exactly one real decode row")
    result = {}

    def visit(prefix, metadata):
        for name, value in metadata.items():
            key = f"{prefix}.{name}"
            if isinstance(value, dict):
                visit(key, value)
            elif (
                name in ("seq_lens", "positions", "input_ids", "token_ids")
                and isinstance(value, torch.Tensor)
                and value.numel()
            ):
                # Single request: discard padded descriptor rows.
                result[key] = value.reshape(-1)[:1].tolist()

    for layer, metadata in record["metadata"].items():
        visit(layer, metadata)
    if not result:
        raise ValueError("trace has no sequence/token identifiers for alignment")
    return result


def compare_records(left, right, atol=0.0):
    for name in ("version", "rank", "layer"):
        if left[name] != right[name]:
            raise ValueError(f"trace {name} differs")
    if alignment(left) != alignment(right):
        raise ValueError("sequence/token alignment differs")
    if left["tensors"].keys() != right["tensors"].keys():
        raise ValueError("trace tensor keys differ")
    comparisons = {}
    for name, value in left["tensors"].items():
        other = right["tensors"][name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"{name} shape/dtype differs")
        exact = torch.equal(value, other)
        finite = bool(torch.isfinite(value).all() and torch.isfinite(other).all())
        delta = (value.double() - other.double()).abs()
        max_abs = float(delta.max()) if delta.numel() and finite else None
        comparisons[name] = {
            "exact": exact,
            "finite": finite,
            "max_abs": max_abs,
            "different_elements": int((value != other).sum()),
            "within_atol": finite and (exact or max_abs <= atol),
        }
    return {
        "rank": left["rank"],
        "layer": left["layer"],
        "left_call": left["call"],
        "right_call": right["call"],
        "alignment": alignment(left),
        "inputs_identical": all(comparisons[name]["exact"] for name in _INPUTS),
        "tensors": comparisons,
    }


def compare_directory(directory, left_call, right_call, proof, atol=0.0):
    first, repeat = proof["gpu_cached"], proof["gpu_cached_repeat"]
    if first["tokens"] != repeat["tokens"] or first["text"] != repeat["text"]:
        raise ValueError("GPU-repeat proof has different generated tokens/text")
    records = []
    for path in Path(directory).glob(f"rank*-layer*-call{left_call:06d}.pt"):
        left = torch.load(path, map_location="cpu", weights_only=True)
        right_path = path.with_name(
            f"rank{left['rank']}-layer{left['layer']}-call{right_call:06d}.pt"
        )
        if not right_path.exists():
            raise ValueError(f"missing counterpart: {right_path.name}")
        right = torch.load(right_path, map_location="cpu", weights_only=True)
        records.append(compare_records(left, right, atol))
    if not records:
        raise ValueError("no trace records for requested left call")
    records.sort(key=lambda record: (record["layer"], record["rank"]))
    first_divergence = next(
        (
            {"rank": record["rank"], "layer": record["layer"], "tensor": name}
            for record in records
            for name, comparison in record["tensors"].items()
            if not comparison["within_atol"]
        ),
        None,
    )
    return {"atol": atol, "first_divergence": first_divergence, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--left-call", required=True, type=int)
    parser.add_argument("--right-call", required=True, type=int)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if not 0 <= args.atol < float("inf"):
        parser.error("atol must be finite and nonnegative")
    report = compare_directory(
        args.directory,
        args.left_call,
        args.right_call,
        json.loads(Path(args.proof).read_text()),
        args.atol,
    )
    rendered = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
