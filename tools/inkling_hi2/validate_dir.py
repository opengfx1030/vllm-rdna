#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Header-only validation and census of a prepared HI2 Inkling serve directory.

CPU only: it reads config.json, the safetensors index and the 8-byte length +
JSON header of every safetensors file.  No tensor data is read, no ``vllm``
import happens; the format rules are the loader's own
(``vllm/models/inkling2_loop/amd/loader.py::hi2_expert_formats``), so this tool
cannot drift from the runtime.

    $VENV/bin/python tools/inkling_hi2/validate_dir.py \
        --model ~/w4a8_runs/inkling-hi2-runtime/serve-bpw3.5 [--json out.json]

Exit 0: every check passed (warnings are printed but do not fail).  Exit 1: an
error.  Exit 2: bad input.  The offsets inside the row buffers are checked by
the loader when it loads the tensors; this tool checks that every stacked group
of every layer is present, in an indexed shard, with the shapes and dtypes the
kernels read, plus the MTP file and the tokenizer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_AMD = _ROOT / "vllm" / "models" / "inkling2_loop" / "amd"

DTYPE_BYTES = {
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
REQUIRED_GROUP_FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
    "part_off",
)
F_GROUP = ("gate_up", "down")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"validate_dir: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_header(path: Path) -> dict[str, dict]:
    """The safetensors header (tensor name -> dtype/shape/offsets); no data."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: shorter than the 8-byte header length")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 1 << 30:
            raise ValueError(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def numel(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


def nbytes(entry: dict) -> int:
    try:
        lo, hi = entry["data_offsets"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("tensor entry has no data_offsets") from None
    return int(hi) - int(lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the prepared serve directory")
    ap.add_argument(
        "--mtp",
        default=None,
        help="MTP safetensors (default: <model>/mtp.safetensors; pass '' to skip)",
    )
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    if not model.is_dir():
        print(f"validate_dir: {model} is not a directory", file=sys.stderr)
        return 2
    errors: list[str] = []
    warnings: list[str] = []

    cfg_path = model / "config.json"
    idx_path = model / "model.safetensors.index.json"
    if not cfg_path.is_file() or not idx_path.is_file():
        print(
            "validate_dir: config.json / model.safetensors.index.json missing",
            file=sys.stderr,
        )
        return 2
    cfg = json.loads(cfg_path.read_text())
    index = json.loads(idx_path.read_text())
    weight_map = index.get("weight_map") or {}
    if not weight_map:
        print("validate_dir: the index has no weight_map", file=sys.stderr)
        return 2

    loader = _load("_i2_loader_validate", _AMD / "loader.py")
    hi2 = _load("_i2_hi2_validate", _AMD / "hi2.py")

    qcfg = cfg.get("quantization_config") or {}
    text = cfg.get("text_config") or cfg
    layers = int(text.get("num_hidden_layers") or 0)
    hidden = int(text.get("hidden_size") or 0)
    inter = int(text.get("intermediate_size") or 0)
    experts = int(text.get("n_routed_experts") or 0)
    if not (layers and hidden and inter and experts):
        errors.append(
            "config text_config lacks num_hidden_layers/hidden_size/"
            f"intermediate_size/n_routed_experts ({layers}, {hidden}, {inter}, "
            f"{experts})"
        )
    try:
        formats = loader.hi2_expert_formats(qcfg, layers)
        if set(formats) != {"hi2"}:
            errors.append(f"hi2_expert_formats returned {sorted(set(formats))}")
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        errors.append(f"quantization_config is not loadable: {exc}")

    shards: dict[str, dict[str, dict]] = {}
    for rel in sorted(set(weight_map.values())):
        path = model / rel
        if not path.is_file():
            errors.append(f"indexed shard {rel} is missing")
            continue
        try:
            shards[rel] = read_header(path)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    def entry(name: str) -> dict | None:
        rel = weight_map.get(name)
        if rel is None:
            return None
        return (shards.get(rel) or {}).get(name)

    # Every stacked group of every layer, with the shapes/dtypes the kernels read.
    per_field = {f: 0 for f in REQUIRED_GROUP_FIELDS + ("col_scale",)}
    expert_bytes = 0
    groups = 0
    col_scale_groups = 0
    for layer in range(layers):
        for group in F_GROUP:
            base = f"model.language_model.layers.{layer}.mlp.experts.{group}.hi2_"
            k = hidden if group == "gate_up" else inter
            parts = 2 if group == "gate_up" else 1
            rows = experts * parts * (inter if group == "gate_up" else hidden)
            fields = list(REQUIRED_GROUP_FIELDS)
            if entry(base + "col_scale") is not None:
                fields.append("col_scale")
            got = {}
            for field in fields:
                e = entry(base + field)
                if e is None:
                    errors.append(f"missing {base}{field}")
                    continue
                got[field] = e
                per_field[field] += nbytes(e)
                expert_bytes += nbytes(e)
            if "codes" in got:
                if got["codes"]["dtype"] != "I32":
                    errors.append(f"{base}codes dtype {got['codes']['dtype']} != I32")
                if len(got["codes"]["shape"]) != 1:
                    errors.append(
                        f"{base}codes shape {got['codes']['shape']} is not flat"
                    )
            # field -> (dtype, expected shape or None for "any 1-D", label)
            expected = {
                "code_off": ("I32", (rows + 1,), "rows+1"),
                "param_off": ("I32", (rows + 1,), "rows+1"),
                "blk_params": ("U8", None, "flat"),
                "row_consts": ("F16", (rows, 3), "(rows, 3)"),
                "row_nbits": ("U8", (rows,), "(rows,)"),
                "row_block": ("U8", (rows,), "(rows,)"),
                "had": (
                    "U8",
                    (experts, k // hi2.GROUP),
                    f"({experts}, {k // hi2.GROUP})",
                ),
                "part_off": (
                    "I32",
                    (experts * parts + 1,),
                    f"({experts * parts + 1},)",
                ),
            }
            for field, (dt, want, label) in expected.items():
                e = got.get(field)
                if e is None:
                    continue
                if e["dtype"] != dt:
                    errors.append(f"{base}{field} dtype {e['dtype']} != {dt}")
                shape = tuple(e["shape"])
                if (len(shape) != 1) if want is None else (shape != want):
                    errors.append(f"{base}{field} shape {shape} != {label}")
            if "col_scale" in got:
                col_scale_groups += 1
                e = got["col_scale"]
                if e["dtype"] != "F16" or tuple(e["shape"]) != (experts, k):
                    errors.append(
                        f"{base}col_scale {e['dtype']} {tuple(e['shape'])} "
                        f"!= F16 ({experts}, {k})"
                    )
            groups += 1

    # Census: expert stacks vs everything else that the index lists.
    total_bytes = sum(nbytes(e) for h in shards.values() for e in h.values())
    expert_names = {
        f"model.language_model.layers.{L}.mlp.experts.{g}.hi2_{f}"
        for L in range(layers)
        for g in F_GROUP
        for f in REQUIRED_GROUP_FIELDS + ("col_scale",)
    }
    dense_names = [n for n in weight_map if n not in expert_names]
    dense_bytes = sum(nbytes(entry(n)) for n in dense_names if entry(n) is not None)
    vision_bytes = sum(
        nbytes(e)
        for rel, h in shards.items()
        if "vision" in Path(rel).name
        for e in h.values()
    )
    unindexed = sorted(
        p.name
        for p in model.glob("*.safetensors")
        if p.name not in set(weight_map.values())
    )

    if args.mtp == "":
        mtp_path = None
    else:
        mtp_path = (
            Path(args.mtp).expanduser() if args.mtp else model / "mtp.safetensors"
        )
    mtp = {
        "path": str(mtp_path) if mtp_path is not None else "",
        "present": mtp_path is not None and mtp_path.is_file(),
        "bytes": 0,
        "keys": 0,
    }
    if mtp["present"]:
        try:
            assert mtp_path is not None
            h = read_header(mtp_path)
            mtp["bytes"], mtp["keys"] = sum(nbytes(e) for e in h.values()), len(h)
            overlap = set(h) & set(weight_map)
            if overlap:
                errors.append(
                    f"mtp.safetensors shares {len(overlap)} indexed tensors, "
                    f"e.g. {sorted(overlap)[:2]}"
                )
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"mtp.safetensors: {exc}")
    else:
        warnings.append(f"no MTP file at {mtp_path} (DSpark needs one)")

    tokenizer = model / "tokenizer.json"
    if not tokenizer.is_file():
        errors.append("tokenizer.json is missing (the output should carry it)")
    if not (model / "tokenizer_config.json").is_file():
        warnings.append("tokenizer_config.json is missing")
    if experts and col_scale_groups != groups:
        warnings.append(
            f"{groups - col_scale_groups}/{groups} expert groups carry no col_scale "
            "(the loader supports it; the smoothing is then not applied)"
        )

    if args.tp > 1:
        per_rank_experts = expert_bytes / args.tp
        upper = per_rank_experts + dense_bytes + vision_bytes + mtp["bytes"]
        lower = per_rank_experts + dense_bytes / args.tp + vision_bytes + mtp["bytes"]
    else:
        per_rank_experts = expert_bytes
        upper = lower = expert_bytes + dense_bytes + vision_bytes + mtp["bytes"]

    gib = 1 << 30
    report = {
        "model": str(model),
        "quant_method": qcfg.get("quant_method"),
        "version": str(qcfg.get("version")),
        "architectures": cfg.get("architectures"),
        "layers": layers,
        "hidden": hidden,
        "intermediate": inter,
        "experts": experts,
        "groups": groups,
        "col_scale_groups": col_scale_groups,
        "files": sorted(p.name for p in model.iterdir()) if model.is_dir() else [],
        "unindexed_safetensors": unindexed,
        "bytes": {
            "indexed_total": total_bytes,
            "experts": expert_bytes,
            "dense": dense_bytes,
            "vision_shard": vision_bytes,
            "mtp": mtp["bytes"],
            "per_field": per_field,
            "tp": args.tp,
            "per_rank_experts": per_rank_experts,
            "per_rank_upper": upper,
            "per_rank_lower": lower,
        },
        "mtp": mtp,
        "errors": errors,
        "warnings": warnings,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    print(f"model          {model}")
    print(
        f"config         {qcfg.get('quant_method')} {qcfg.get('version')} "
        f"{cfg.get('architectures')}"
    )
    print(
        f"text           {layers} layers, hidden {hidden}, inter {inter}, "
        f"{experts} experts"
    )
    print(f"groups         {groups} ({col_scale_groups} with col_scale)")
    print(f"shards         {sorted(set(weight_map.values()))}")
    if unindexed:
        print(f"unindexed      {unindexed}")
    print(
        "bytes          experts {:.2f} GiB, dense {:.2f} GiB, vision {:.2f} GiB, "
        "MTP {:.2f} GiB, indexed total {:.2f} GiB".format(
            expert_bytes / gib,
            dense_bytes / gib,
            vision_bytes / gib,
            mtp["bytes"] / gib,
            total_bytes / gib,
        )
    )
    for field, b in per_field.items():
        print(f"  {field:<12} {b / gib:8.3f} GiB")
    print(
        f"TP={args.tp}         per rank: experts {per_rank_experts / gib:.2f} GiB, "
        f"{lower / gib:.2f} (dense/TP) .. {upper / gib:.2f} GiB (dense replicated)"
    )
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("VALID" if not errors else f"INVALID ({len(errors)} errors)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
