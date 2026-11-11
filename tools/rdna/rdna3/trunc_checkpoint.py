#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a layer-truncated copy of a Flash-Next safetensors checkpoint.

The truncated model runs the first N decoder layers of the real model with the
real weights, so the same input can be pushed through it on a V620 (gfx1030) and
on a W7800 (gfx1100) and the per-layer hidden states compared
(``tools/rdna/rdna3/trunc_e2e.py``; docs/rdna2/rdna3-flashnext-port-2026-11-07.md
section 12 step G13).

    python tools/rdna/rdna3/trunc_checkpoint.py \
        --src <checkpoint snapshot dir> \
        --dst ~/w4a8_runs/rdna3_port/trunc/ckpt-L4 --layers 4 [--keep-mtp]

What is kept (tensor names of the real checkpoint):

* ``model.language_model.layers.<i>.*`` for ``i < N`` (the PLE module lives in
  layer 1, so every N >= 2 keeps it and its n-gram table);
* everything that is not a decoder layer and not MTP: ``embed_tokens``,
  ``hyper_connection_mixer`` (the final mixer; this model has no separate final
  norm), ``lm_head`` and the vision tower (small; the engine drops it with
  ``language_model_only``);
* ``mtp.*`` only with ``--keep-mtp``. The MTP head is only read when a
  speculative config asks for it.

Weights are streamed tensor by tensor as raw bytes (dtype, shape and the packed
int4 payload are never decoded, so kept tensors are byte-identical to the
source). A source shard whose tensors are all kept (the 100 GB PLE n-gram shard
for any N >= 2) is symlinked, not copied (``--copy-shards`` copies it). Every
non-weight file is copied; ``config.json`` is rewritten (``num_hidden_layers``,
every per-layer list, ``ple_layer_ids``, MTP count), the quantization config is
left untouched. ``model.safetensors.index.json`` is rebuilt and a
``trunc_info.json`` sidecar records the source and the choices.

N must be a multiple of the layer-pattern period (``full_attention_interval``,
4 for Flash-Next) unless ``--allow-ragged``.
"""

import argparse
import difflib
import json
import os
import shutil
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import regex as re

INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
INFO_NAME = "trunc_info.json"
COPY_CHUNK = 64 << 20
# Decoder layers; MTP layers (``mtp.layers.<j>``) are matched by the mtp rule first.
_LAYER_RE = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")
_MTP_RE = re.compile(r"^(?:model\.)?mtp\.")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class TensorInfo:
    """One tensor of a source shard: header entry plus file offsets."""

    __slots__ = ("name", "dtype", "shape", "begin", "end")

    def __init__(
        self, name: str, dtype: str, shape: list[int], begin: int, end: int
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.begin = begin
        self.end = end

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


class Shard:
    """Parsed safetensors header: tensors in file order and the data offset."""

    def __init__(
        self,
        path: Path,
        tensors: list[TensorInfo],
        data_start: int,
        metadata: dict[str, str] | None,
    ) -> None:
        self.path = path
        self.tensors = tensors
        self.data_start = data_start
        self.metadata = metadata


def read_shard(path: Path) -> Shard:
    """Parse a safetensors header (no tensor data is read)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        (hlen,) = struct.unpack("<Q", raw)
        if hlen > 1 << 31:
            raise ValueError(f"{path}: implausible header size {hlen}")
        header = json.loads(f.read(hlen))
    metadata = header.pop("__metadata__", None)
    tensors = [
        TensorInfo(name, e["dtype"], list(e["shape"]), *e["data_offsets"])
        for name, e in header.items()
    ]
    tensors.sort(key=lambda t: t.begin)
    return Shard(path, tensors, 8 + hlen, metadata)


def write_header(
    out: Any, tensors: list[TensorInfo], metadata: dict[str, str] | None
) -> list[TensorInfo]:
    """Write the header of a new shard; returns the tensors with new offsets."""
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    moved: list[TensorInfo] = []
    offset = 0
    for t in tensors:
        moved.append(TensorInfo(t.name, t.dtype, t.shape, offset, offset + t.nbytes))
        header[t.name] = {
            "dtype": t.dtype,
            "shape": t.shape,
            "data_offsets": [offset, offset + t.nbytes],
        }
        offset += t.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    out.write(struct.pack("<Q", len(blob)))
    out.write(blob)
    return moved


def copy_range(src: Any, dst: Any, start: int, nbytes: int) -> None:
    src.seek(start)
    left = nbytes
    while left:
        chunk = src.read(min(left, COPY_CHUNK))
        if not chunk:
            raise OSError("unexpected end of source shard")
        dst.write(chunk)
        left -= len(chunk)


def layer_index(name: str) -> int | None:
    """Decoder layer index of a tensor name, None for non-layer tensors."""
    if _MTP_RE.match(name):
        return None
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def tensor_group(name: str) -> str:
    """Coarse group used for the keep decision and the report."""
    if _MTP_RE.match(name):
        return "mtp"
    if layer_index(name) is not None:
        return "layer"
    if ".visual." in name or name.startswith("visual."):
        return "visual"
    return "other"


def keep_tensor(name: str, layers: int, keep_mtp: bool) -> bool:
    group = tensor_group(name)
    if group == "mtp":
        return keep_mtp
    if group == "layer":
        idx = layer_index(name)
        assert idx is not None
        return idx < layers
    return True


def layer_period(text_cfg: dict[str, Any]) -> int:
    """Period of the layer pattern: full_attention_interval, else from the types."""
    interval = text_cfg.get("full_attention_interval")
    if isinstance(interval, int) and interval > 0:
        return interval
    types = text_cfg.get("layer_types")
    if isinstance(types, list) and types:
        for p in range(1, len(types) + 1):
            if len(types) % p == 0 and types == types[:p] * (len(types) // p):
                return p
    return 1


def truncate_config(
    cfg: dict[str, Any], layers: int, keep_mtp: bool, allow_ragged: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Return (new config, notes). ``cfg`` is not modified."""
    new: dict[str, Any] = json.loads(json.dumps(cfg))
    text = new["text_config"] if isinstance(new.get("text_config"), dict) else new
    total = text.get("num_hidden_layers")
    if not isinstance(total, int):
        raise ValueError("config has no integer num_hidden_layers")
    if not 1 <= layers <= total:
        raise ValueError(f"--layers must be in 1..{total}, got {layers}")
    period = layer_period(text)
    if layers % period and not allow_ragged:
        raise ValueError(
            f"--layers {layers} is not a multiple of the layer-pattern period "
            f"{period}; pass --allow-ragged to cut inside a pattern"
        )
    notes = [f"num_hidden_layers {total} -> {layers}"]
    text["num_hidden_layers"] = layers
    # Any per-layer list (layer_types and friends) shrinks with the model.
    for key, value in text.items():
        if isinstance(value, list) and len(value) == total and key != "ple_layer_ids":
            text[key] = value[:layers]
            notes.append(f"{key}: {total} entries -> {layers}")
    # ple_layer_ids are 1-based layer ids; a PLE layer past the cut is gone.
    ple = text.get("ple_layer_ids")
    if isinstance(ple, list):
        kept = [i for i in ple if i <= layers]
        if kept != ple:
            notes.append(f"ple_layer_ids {ple} -> {kept}")
        text["ple_layer_ids"] = kept
    if not keep_mtp and text.get("mtp_num_hidden_layers"):
        notes.append(f"mtp_num_hidden_layers {text['mtp_num_hidden_layers']} -> 0")
        text["mtp_num_hidden_layers"] = 0
    return new, notes


def source_shards(src: Path) -> list[str]:
    """Shard file names of the source: index order when there is an index."""
    index = src / INDEX_NAME
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = sorted(p.name for p in src.glob("*.safetensors"))
    if not names:
        raise FileNotFoundError(f"no safetensors files under {src}")
    return names


def _human(n: int) -> str:
    gib = n / (1 << 30)
    return f"{gib:.2f} GiB" if gib >= 0.1 else f"{n / (1 << 20):.2f} MiB"


def _non_weight_files(src: Path) -> Iterator[Path]:
    for p in sorted(src.iterdir()):
        if p.name in (INDEX_NAME, CONFIG_NAME) or p.suffix == ".safetensors":
            continue
        if p.is_file():
            yield p


def build(
    src: Path,
    dst: Path,
    layers: int,
    keep_mtp: bool = False,
    allow_ragged: bool = False,
    copy_shards: bool = False,
    force: bool = False,
    dry_run: bool = False,
    log: Any = print,
) -> dict[str, Any]:
    """Create the truncated checkpoint; returns a report dict."""
    src = src.resolve()
    dst = dst.absolute()
    if dst == src or src in dst.parents:
        raise ValueError("--dst must not be inside --src")
    cfg = json.loads((src / CONFIG_NAME).read_text())
    new_cfg, notes = truncate_config(cfg, layers, keep_mtp, allow_ragged)

    # Plan from the real shard headers (the index is cross-checked below).
    plan: dict[str, tuple[Shard, list[TensorInfo], int]] = {}
    groups: dict[str, list[int]] = {}
    seen: set[str] = set()
    for fname in source_shards(src):
        shard = read_shard((src / fname).resolve())
        keep: list[TensorInfo] = []
        for t in shard.tensors:
            seen.add(t.name)
            g = tensor_group(t.name)
            counts = groups.setdefault(g, [0, 0, 0, 0])  # kept n/B, dropped n/B
            if keep_tensor(t.name, layers, keep_mtp):
                keep.append(t)
                counts[0] += 1
                counts[1] += t.nbytes
            else:
                counts[2] += 1
                counts[3] += t.nbytes
        if keep:
            plan[fname] = (shard, keep, len(shard.tensors))
    index_path = src / INDEX_NAME
    if index_path.is_file():
        wmap = json.loads(index_path.read_text())["weight_map"]
        if set(wmap) != seen:
            raise ValueError(
                f"{INDEX_NAME} disagrees with the shard headers: "
                f"{len(set(wmap) - seen)} only in index, "
                f"{len(seen - set(wmap))} only in shards"
            )
    for needed in ("embed_tokens", "lm_head"):
        if not any(needed in n and keep_tensor(n, layers, keep_mtp) for n in seen):
            raise ValueError(f"no {needed} tensor kept: unexpected checkpoint layout")
    for i in range(layers):
        if not any(layer_index(n) == i for n in seen):
            raise ValueError(f"layer {i} has no tensors in the source")

    kept_bytes = sum(sum(t.nbytes for t in keep) for _, keep, _ in plan.values())
    log(f"source {src} -> {dst}  layers={layers} keep_mtp={keep_mtp}")
    for g in sorted(groups):
        kn, kb, dn, db = groups[g]
        log(
            f"  {g:7s} kept {kn:7d} tensors {_human(kb):>11s}   "
            f"dropped {dn:7d} tensors {_human(db):>11s}"
        )
    if dry_run:
        for note in notes:
            log(f"  config: {note}")
        return {"kept_bytes": kept_bytes, "groups": groups, "dry_run": True}

    if dst.exists() and any(dst.iterdir()):
        if not force:
            raise FileExistsError(f"{dst} is not empty (use --force to reuse it)")
        for p in dst.iterdir():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    for p in _non_weight_files(src):
        shutil.copy2(p, dst / p.name)
    (dst / CONFIG_NAME).write_text(json.dumps(new_cfg, indent=2) + "\n")

    weight_map: dict[str, str] = {}
    written = linked = 0
    for fname, (shard, keep, total) in plan.items():
        out_path = dst / fname
        nbytes = sum(t.nbytes for t in keep)
        if len(keep) == total and not copy_shards:
            out_path.symlink_to(shard.path)
            linked += nbytes
            log(f"  link  {fname} -> {shard.path} ({len(keep)} tensors)")
            names = [t.name for t in keep]
        else:
            tmp = out_path.with_name(out_path.name + ".tmp")
            with open(shard.path, "rb") as fin, open(tmp, "wb") as fout:
                moved = write_header(fout, keep, shard.metadata)
                for t in keep:
                    copy_range(fin, fout, shard.data_start + t.begin, t.nbytes)
            os.replace(tmp, out_path)
            written += nbytes
            names = [t.name for t in moved]
            log(f"  write {fname} ({len(keep)}/{total} tensors, {_human(nbytes)})")
        for n in names:
            weight_map[n] = fname
    index = {"metadata": {"total_size": kept_bytes}, "weight_map": weight_map}
    (dst / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    info = {
        "source": str(src),
        "layers": layers,
        "keep_mtp": keep_mtp,
        "allow_ragged": allow_ragged,
        "tensors": len(weight_map),
        "kept_bytes": kept_bytes,
        "linked_bytes": linked,
        "written_bytes": written,
        "config_notes": notes,
        "argv": sys.argv,
    }
    (dst / INFO_NAME).write_text(json.dumps(info, indent=2) + "\n")

    verify(dst, plan)
    for line in difflib.unified_diff(
        json.dumps(cfg, indent=2).splitlines(),
        json.dumps(new_cfg, indent=2).splitlines(),
        "config.json (source)",
        "config.json (truncated)",
        lineterm="",
        n=0,
    ):
        log(line)
    log(
        f"kept {len(weight_map)} tensors, {_human(kept_bytes)} of weights "
        f"({_human(written)} written, {_human(linked)} symlinked); output {dst}"
    )
    return {
        "kept_bytes": kept_bytes,
        "written_bytes": written,
        "linked_bytes": linked,
        "tensors": len(weight_map),
        "groups": groups,
        "config_notes": notes,
        "weight_map": weight_map,
    }


def verify(dst: Path, plan: dict[str, tuple[Shard, list[TensorInfo], int]]) -> None:
    """Re-read the output headers: names, dtypes, shapes and sizes must match."""
    index = json.loads((dst / INDEX_NAME).read_text())["weight_map"]
    got: dict[str, tuple[str, list[int], int]] = {}
    for fname in sorted(set(index.values())):
        for t in read_shard((dst / fname).resolve()).tensors:
            got[t.name] = (t.dtype, t.shape, t.nbytes)
            if index.get(t.name) != fname:
                raise ValueError(f"index maps {t.name} to {index.get(t.name)}")
    want = {
        t.name: (t.dtype, t.shape, t.nbytes)
        for _, keep, _ in plan.values()
        for t in keep
    }
    if got != want:
        raise ValueError("output shards do not match the plan")
    try:
        from safetensors import safe_open
    except ImportError:
        return
    for fname in sorted(set(index.values())):
        with safe_open(str(dst / fname), framework="np") as f:
            keys = set(f.keys())
        expect = {n for n, v in index.items() if v == fname}
        if keys != expect:
            raise ValueError(f"{fname}: safetensors keys differ from the index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:]),
    )
    parser.add_argument("--src", type=Path, required=True, help="checkpoint dir")
    parser.add_argument("--dst", type=Path, required=True, help="output dir")
    parser.add_argument("--layers", type=int, required=True, help="decoder layers")
    parser.add_argument("--keep-mtp", action="store_true", help="keep mtp.* tensors")
    parser.add_argument(
        "--allow-ragged",
        action="store_true",
        help="allow N that is not a multiple of the layer-pattern period",
    )
    parser.add_argument(
        "--copy-shards",
        action="store_true",
        help="copy fully-kept shards instead of symlinking them (100 GB PLE shard)",
    )
    parser.add_argument("--force", action="store_true", help="reuse a non-empty dst")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = parser.parse_args(argv)
    try:
        build(
            args.src,
            args.dst,
            args.layers,
            keep_mtp=args.keep_mtp,
            allow_ragged=args.allow_ragged,
            copy_shards=args.copy_shards,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
