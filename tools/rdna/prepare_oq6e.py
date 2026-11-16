# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish immutable native oQ6e target/MTP metadata views, without repacking.

Only the Python standard library is imported. Admission and renameat2 helpers
follow export_cyankiwi_awq.py; they are local because that module transitively
imports the network acquisition dependency. No exporter globals are modified.
The current server chat template remains an explicit launcher responsibility.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from pathlib import Path

REPO = "mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp"
REVISION = "e171af86f499f1855b0fb71d781105e8dd609610"
ASSET_REPO = "cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4"
ASSET_REVISION = "5d89a6dd0c7d081c99424de02b2dcf0262426a59"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
SEMANTIC_SHA256 = "327de7f97ba22799dee302f7e48263b664816b4e1ef717641491872a0a62bd3b"
INDEX = "model.safetensors.index.json"
MANIFEST = "mlx-affine-manifest.json"
RECEIPT = "prepare-oq6e-receipt.json"
VERSION = 1
CHUNK = 1024 * 1024
MAX_JSON = 64 * CHUNK
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U32": 4,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)
CONTROL_IDS = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
}
PLE_PREFIX = "language_model.model.layers.1.ple.ple_embedding."
PLE_ROWS = 2500012
PLE_SHARDS = 128
EXPECTED_TENSORS = 3748
EXPECTED_MTP_TENSORS = 76
PLE_HASHES = {
    "layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
    "ngram_heads_vocab_sizes": [
        20000003,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000093,
        20000107,
        20000147,
        20000153,
        20000159,
        20000161,
        20000171,
    ],
    "ngram_heads_offsets": [
        0,
        20000003,
        40000026,
        60000059,
        80000106,
        100000165,
        120000228,
        140000297,
        160000374,
        180000455,
        200000548,
        220000655,
        240000802,
        260000955,
        280001114,
        300001275,
    ],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)


def canonical_bytes(value, *, newline=False):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + ("\n" if newline else "")
    ).encode("utf-8")


def _identity(path):
    info = path.lstat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "mode": info.st_mode,
    }


def _unchanged(path, identity):
    require(_identity(path) == identity, f"File changed during publication: {path}")


def _regular(path):
    identity = _identity(path)
    require(stat.S_ISREG(identity["mode"]), f"Not a regular file: {path}")
    return identity


def _open_regular(path):
    before = _regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(
            (info.st_dev, info.st_ino) == (before["device"], before["inode"]),
            f"File replaced: {path}",
        )
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    return stream, before


def _read_json(path):
    stream, before = _open_regular(path)
    with stream:
        require(
            before["size"] <= MAX_JSON, f"JSON exceeds bounded metadata limit: {path}"
        )
        raw = stream.read(MAX_JSON + 1)
    _unchanged(path, before)
    require(len(raw) == before["size"], f"Truncated JSON: {path}")
    document = strict_json(raw)
    require(isinstance(document, dict), f"Expected JSON object: {path}")
    return document, {"identity": before, "sha256": hashlib.sha256(raw).hexdigest()}


def _filename(name):
    # The acquired repository includes .gitattributes, unlike the AWQ exporter.
    require(
        isinstance(name, str)
        and (
            name == ".gitattributes"
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
        )
        and name not in (".", ".."),
        f"Unsafe flat filename: {name!r}",
    )
    return name


def _records(rows, key):
    require(isinstance(rows, list), "Expected acquisition file list")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and key in row, "Invalid acquisition file record")
        name = _filename(row[key])
        require(name not in result, f"Duplicate acquisition file: {name}")
        result[name] = row
    return result


def _directory(path, root=None):
    path = Path(path).absolute()
    require(
        path.resolve(strict=True) == path, f"Symlinked directory is forbidden: {path}"
    )
    require(path.is_dir(), f"Not a directory: {path}")
    if root is not None:
        require(
            path == root or root in path.parents,
            f"Outside protected model root: {path}",
        )
    return path


def _verify_file(path, entry, check=None):
    size = entry.get("size")
    require(type(size) is int and size >= 0, f"Invalid acquisition size: {path.name}")
    lfs = entry.get("lfs")
    if lfs is not None:
        require(
            isinstance(lfs, dict)
            and type(lfs.get("size")) is int
            and lfs["size"] == size,
            f"Invalid LFS size: {path.name}",
        )
        expected, kind = lfs.get("sha256"), "sha256"
        length = 64
    else:
        expected, kind = entry.get("blobId"), "git_blob_sha1"
        length = 40
    require(
        isinstance(expected, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", expected),
        f"Missing acquisition digest: {path.name}",
    )
    if check is not None:
        require(
            check.get("verified") is True
            and type(check.get("bytes")) is int
            and check["bytes"] == size
            and check.get("hash_kind") == kind
            and check.get("digest") == expected,
            f"Acquisition verification disagrees: {path.name}",
        )
    stream, before = _open_regular(path)
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode()) if lfs is None else None
    with stream:
        require(before["size"] == size, f"Source size mismatch: {path.name}")
        while data := stream.read(CHUNK):
            sha.update(data)
            if blob is not None:
                blob.update(data)
    _unchanged(path, before)
    require(
        (sha.hexdigest() if blob is None else blob.hexdigest()) == expected,
        f"Source hash mismatch: {path.name}",
    )
    return {
        "identity": before,
        "sha256": sha.hexdigest(),
        "manifest_digest": expected,
        "hash_kind": kind,
    }


def admit_source(source, acquisition_manifest, acquisition_result):
    manifest, manifest_info = _read_json(acquisition_manifest)
    result, result_info = _read_json(acquisition_result)
    require(
        manifest.get("repo_id") == REPO and manifest.get("revision") == REVISION,
        "Source manifest is not the pinned repository/revision",
    )
    require(
        result.get("ok") is True and result.get("revision") == REVISION,
        "Acquisition did not verify the pinned revision",
    )
    for doc in (manifest, result):
        require(
            isinstance(doc.get("destination"), str)
            and Path(doc["destination"]).absolute() == source,
            "Acquisition destination does not match source",
        )
    files = _records(manifest.get("files"), "rfilename")
    verified = _records(result.get("files"), "file")
    require(set(files) == set(verified), "Manifest/acquisition file sets differ")
    require(
        {INDEX, "config.json", "tokenizer.json"} <= set(files),
        "Missing required source assets",
    )
    require(
        not ({MANIFEST, RECEIPT, "tokenizer-semantics.json"} & files.keys()),
        "Reserved native filename in source",
    )
    identities = {
        name: _verify_file(source / name, entry, verified[name])
        for name, entry in sorted(files.items())
    }
    total = sum(info["identity"]["size"] for info in identities.values())
    require(
        all(
            type(doc.get("total_bytes")) is int and doc["total_bytes"] == total
            for doc in (manifest, result)
        ),
        "Acquisition byte total mismatch",
    )
    return files, {
        "repo_id": REPO,
        "revision": REVISION,
        "source": str(source),
        "acquisition_manifest": {"path": str(acquisition_manifest), **manifest_info},
        "acquisition_result": {"path": str(acquisition_result), **result_info},
        "source_files": identities,
        "acquired_bytes": total,
    }


def read_header(path):
    stream, before = _open_regular(path)
    with stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        require(
            0 < length <= MAX_JSON and length <= before["size"] - 8,
            f"Invalid safetensors header length: {path}",
        )
        raw = stream.read(length)
    _unchanged(path, before)
    require(len(raw) == length, f"Truncated safetensors header: {path}")
    header = strict_json(raw)
    require(isinstance(header, dict), f"Invalid safetensors header: {path}")
    ranges = []
    for name, entry in header.items():
        if name == "__metadata__":
            require(
                isinstance(entry, dict)
                and all(isinstance(v, str) for v in entry.values()),
                "Invalid safetensors metadata",
            )
            continue
        require(
            name
            and isinstance(entry, dict)
            and set(entry) == {"dtype", "shape", "data_offsets"},
            f"Invalid tensor fields: {name}",
        )
        shape, offsets = entry["shape"], entry["data_offsets"]
        require(
            isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape),
            f"Invalid tensor shape: {name}",
        )
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(v) is int and v >= 0 for v in offsets)
            and offsets[0] <= offsets[1] <= before["size"] - 8 - length,
            f"Invalid tensor offsets: {name}",
        )
        require(
            isinstance(entry["dtype"], str) and entry["dtype"] in DTYPE_BYTES,
            f"Unsupported tensor dtype: {name}",
        )
        require(
            offsets[1] - offsets[0] == math.prod(shape) * DTYPE_BYTES[entry["dtype"]],
            f"Tensor shape/length mismatch: {name}",
        )
        ranges.append(tuple(offsets))
    position = 0
    for start, end in sorted(ranges):
        require(start == position, f"Noncontiguous/overlapping tensor payload: {path}")
        position = end
    require(position == before["size"] - 8 - length, f"Unaccounted shard bytes: {path}")
    return header, {
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "header_bytes": length,
        "data_start": length + 8,
        "payload_bytes": position,
    }


def runtime_mapping(name, entry):
    """Return the registered path before native constituent fusion/slicing."""
    if name.startswith("mtp."):
        match = re.match(r"mtp\.layers\.(\d+)\.", name)
        require(
            match is None or match[1] == "0", "MTP has one trained checkpoint layer0"
        )
        runtime = "model." + name.removeprefix("mtp.")
    elif name.startswith("vision_tower."):
        runtime = "visual." + name.removeprefix("vision_tower.")
    else:
        require(
            name.startswith(("language_model.model.", "language_model.lm_head.")),
            f"Undeclared source namespace: {name}",
        )
        runtime = name
    runtime = runtime.replace(".mlp.switch_mlp.", ".mlp.experts.")
    layout = "identity"
    if name.endswith((".linear_attn.conv1d.weight", ".ple.conv1d.weight")):
        require(
            entry["dtype"] == "BF16" and entry["shape"] == [10240, 4, 1],
            f"Invalid declared convolution layout: {name}",
        )
        layout = "gdn_conv_mlx"
    elif name == "vision_tower.patch_embed.proj.weight":
        require(
            entry["dtype"] == "BF16" and entry["shape"] == [1152, 2, 16, 16, 3],
            "Invalid declared vision patch layout",
        )
        layout = "vision_patch_mlx"
    return runtime, layout


def inventory(source, files):
    index, _ = _read_json(source / INDEX)
    weights = index.get("weight_map")
    require(isinstance(weights, dict) and weights, "Missing tensor index")
    shard_names = {name for name in files if name.endswith(".safetensors")}
    require(
        all(isinstance(name, str) and name for name in weights),
        "Invalid indexed tensor name",
    )
    require(
        all(isinstance(name, str) for name in weights.values())
        and set(weights.values()) == shard_names,
        "Index/shard file coverage differs",
    )
    tensors, headers = {}, {}
    for filename in sorted(shard_names):
        header, headers[filename] = read_header(source / _filename(filename))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            require(name not in tensors, f"Duplicate tensor across shards: {name}")
            require(
                weights.get(name) == filename,
                f"Unindexed or misplaced source tensor: {name}",
            )
            runtime, layout = runtime_mapping(name, entry)
            tensors[name] = {
                "file": filename,
                **entry,
                "runtime_name": runtime,
                "layout": layout,
            }
    require(
        set(tensors) == set(weights), "Indexed tensors are missing from source shards"
    )
    require(
        len({t["runtime_name"] for t in tensors.values()}) == len(tensors),
        "Runtime name collision",
    )
    # This pinned MLX index counts shard headers as well as tensor payloads.
    indexed_file_bytes = sum(
        h["data_start"] + h["payload_bytes"] for h in headers.values()
    )
    require(
        index.get("metadata", {}).get("total_size") == indexed_file_bytes,
        "Index total does not match complete shard-file bytes",
    )
    return tensors, headers


def _spec(value, prefix):
    require(
        isinstance(value, dict) and set(value) == {"bits", "group_size", "mode"},
        f"Invalid affine declaration: {prefix}",
    )
    require(
        type(value["bits"]) is int
        and value["bits"] in (6, 8)
        and type(value["group_size"]) is int
        and value["group_size"] in (32, 64, 128)
        and value["mode"] == "affine",
        f"Unsupported affine declaration: {prefix}",
    )
    return dict(value)


def module_specs(config, tensors):
    declarations = [
        config[key] for key in ("quantization", "quantization_config") if key in config
    ]
    require(
        declarations and all(isinstance(d, dict) for d in declarations),
        "Missing MLX quantization map",
    )
    require(
        all(d == declarations[0] for d in declarations),
        "Conflicting duplicate MLX declarations",
    )
    original = declarations[0]
    defaults = _spec(
        {key: original.get(key) for key in ("bits", "group_size", "mode")}, "defaults"
    )
    overrides = {
        key: _spec(value, key) for key, value in original.items() if key not in defaults
    }
    specs = {}
    for name, weight in tensors.items():
        if weight["dtype"] != "U32":
            continue
        require(name.endswith(".weight"), f"Packed non-weight tensor: {name}")
        prefix = name.removesuffix(".weight")
        spec = copy.deepcopy(overrides.get(prefix, defaults))
        scale, bias = tensors.get(prefix + ".scales"), tensors.get(prefix + ".biases")
        require(
            scale is not None and bias is not None, f"Missing affine metadata: {prefix}"
        )
        require(
            scale["dtype"] == bias["dtype"] == "BF16"
            and scale["shape"] == bias["shape"],
            f"Invalid affine metadata dtype/shape: {prefix}",
        )
        shape, metadata = weight["shape"], scale["shape"]
        require(
            len(shape) in (2, 3)
            and all(v > 0 for v in shape)
            and len(metadata) == len(shape)
            and shape[:-1] == metadata[:-1]
            and all(v > 0 for v in metadata),
            f"Invalid affine matrix geometry: {prefix}",
        )
        bits, group = spec["bits"], spec["group_size"]
        require(
            shape[-1] * 32 % bits == 0, f"Incomplete packed code geometry: {prefix}"
        )
        logical_k = shape[-1] * 32 // bits
        require(
            metadata[-1] == (logical_k + group - 1) // group,
            f"Affine metadata group count mismatch: {prefix}",
        )
        require(
            not prefix.startswith("vision_tower."),
            "Vision tensors must remain unquantized",
        )
        specs[prefix] = spec
    require(specs, "No proven packed affine matrices")
    require(
        overrides.keys() <= specs.keys(),
        "Quantization override lacks a packed source matrix",
    )
    for key in ("quantization", "quantization_config"):
        nested = config.get("text_config", {}).get(key)
        require(
            nested is None or nested == original,
            "Conflicting nested MLX quantization declaration",
        )
    for name in tensors:
        if name.endswith((".scales", ".biases")):
            require(name.rsplit(".", 1)[0] in specs, f"Orphan affine metadata: {name}")
    return dict(sorted(specs.items()))


def validate_ple(source, tensors, headers, specs):
    tables = {
        prefix
        for prefix in specs
        if prefix.startswith(PLE_PREFIX + "ngram_embedding.shards.")
    }
    expected = {PLE_PREFIX + f"ngram_embedding.shards.{i}" for i in range(PLE_SHARDS)}
    require(tables == expected, "Incomplete or undeclared PLE shard coverage")
    for prefix in sorted(tables):
        require(
            specs[prefix] == {"bits": 6, "group_size": 32, "mode": "affine"},
            "PLE requires affine6/group32",
        )
        for suffix, shape in (
            ("weight", [PLE_ROWS, 30]),
            ("scales", [PLE_ROWS, 5]),
            ("biases", [PLE_ROWS, 5]),
        ):
            require(
                tensors[prefix + "." + suffix]["shape"] == shape,
                f"Invalid PLE complete-row geometry: {prefix}.{suffix}",
            )
    buffers = {}
    for suffix, expected_values in PLE_HASHES.items():
        record = tensors.get(PLE_PREFIX + suffix)
        require(
            record is not None
            and record["dtype"] == "I64"
            and record["shape"] == [len(expected_values)],
            f"Invalid PLE hash buffer: {suffix}",
        )
        stream, before = _open_regular(source / record["file"])
        with stream:
            stream.seek(
                headers[record["file"]]["data_start"] + record["data_offsets"][0]
            )
            raw = stream.read(len(expected_values) * 8)
        _unchanged(source / record["file"], before)
        require(len(raw) == len(expected_values) * 8, "Truncated PLE hash buffer")
        values = list(struct.unpack("<" + "q" * len(expected_values), raw))
        require(
            values == expected_values, f"Published PLE hash identity mismatch: {suffix}"
        )
        buffers[suffix] = values
    scale = tensors.get(PLE_PREFIX + "ngram_embedding.weight_scale")
    require(
        scale is not None and scale["dtype"] == "BF16" and scale["shape"] == [1],
        "Missing PLE weight_scale buffer",
    )
    return {
        "rows_per_shard": PLE_ROWS,
        "shards": PLE_SHARDS,
        "padded_rows": PLE_ROWS * PLE_SHARDS,
        "hash_buffers": buffers,
    }


def normalize_configs(original, specs):
    text = original.get("text_config", {})
    require(
        original.get("architectures") == ["Qwen4ExpForConditionalGeneration"]
        and original.get("model_type") == "qwen4_exp",
        "Unexpected target architecture",
    )
    require(
        text.get("num_hidden_layers") == 48
        and len(text.get("layer_types", [])) == 48
        and text.get("hc_count") == 4
        and text.get("mtp_num_hidden_layers") == 1
        and text.get("mtp", {}).get("num_hidden_layers") == 1,
        "Unexpected backbone/HC/MTP geometry",
    )
    require(
        original.get("tie_word_embeddings") is False
        and text.get("tie_word_embeddings") is False,
        "Target embedding and head must remain untied",
    )
    require(text.get("eos_token_id") == 248044, "Unexpected text EOS")
    target = copy.deepcopy(original)
    for section in (target, target["text_config"]):
        section.pop("quantization", None)
        section.pop("quantization_config", None)
    target["quantization_config"] = {
        "quant_method": "mlx_affine",
        "format_version": 1,
        "module_specs": copy.deepcopy(specs),
    }
    target["text_config"].update(indexer_budget=4096, indexer_compress_ratio=4)
    target["language_model_only"] = False
    draft = copy.deepcopy(target)
    draft.update(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        hc_mult=4,
        n_predict=1,
        index_share_for_mtp_iteration=False,
    )
    # num_hidden_layers stays48: construction prefixes use mtp.layers.48 while
    # the one registered ModuleList layer and its tensor names remain layer0.
    draft["text_config"]["mtp_use_dedicated_embeddings"] = False
    return {"target": target, "mtp": draft}


def tokenizer_semantics(tokenizer):
    require(
        isinstance(tokenizer.get("model"), dict)
        and tokenizer["model"].get("type") == "BPE",
        "Expected BPE tokenizer",
    )
    model = tokenizer["model"]
    vocab = model.get("vocab")
    require(
        isinstance(vocab, dict)
        and all(
            isinstance(k, str) and type(v) is int and v >= 0 for k, v in vocab.items()
        ),
        "Invalid tokenizer vocabulary",
    )
    require(len(set(vocab.values())) == len(vocab), "Duplicate vocabulary IDs")
    require(isinstance(model.get("merges"), list), "Invalid tokenizer merges")
    merges = []
    for merge in model["merges"]:
        pair = merge.split(" ") if isinstance(merge, str) else merge
        require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(v, str) and v for v in pair),
            "Invalid BPE merge pair",
        )
        merges.append(pair)
    added = tokenizer.get("added_tokens")
    require(
        isinstance(added, list)
        and all(
            isinstance(t, dict)
            and type(t.get("id")) is int
            and t["id"] >= 0
            and isinstance(t.get("content"), str)
            for t in added
        ),
        "Invalid added-token records",
    )
    require(
        len({t["id"] for t in added}) == len(added)
        and len({t["content"] for t in added}) == len(added),
        "Duplicate added tokens",
    )
    for token in added:
        require(
            token["content"] not in vocab or vocab[token["content"]] == token["id"],
            "Conflicting added-token ID",
        )
    return {
        "vocab": vocab,
        "merges": merges,
        "added_tokens": sorted(added, key=lambda t: t["id"]),
    }


def admit_assets(source, assets_source, assets_manifest):
    manifest, manifest_info = _read_json(assets_manifest)
    require(
        manifest.get("repo_id") == ASSET_REPO
        and manifest.get("revision") == ASSET_REVISION,
        "Assets are not the pinned canonical official source",
    )
    require(
        isinstance(manifest.get("destination"), str)
        and Path(manifest["destination"]).absolute() == assets_source,
        "Asset source identity mismatch",
    )
    files = _records(manifest.get("files"), "rfilename")
    require(set(ASSETS) <= files.keys(), "Missing canonical tokenizer/processor assets")
    selected = list(ASSETS) + (
        ["processor_config.json"] if "processor_config.json" in files else []
    )
    identities = {
        name: _verify_file(assets_source / name, files[name]) for name in selected
    }
    require(
        identities["tokenizer.json"]["sha256"] == TOKENIZER_SHA256,
        "Canonical tokenizer identity mismatch",
    )
    raw, _ = _read_json(source / "tokenizer.json")
    official, _ = _read_json(assets_source / "tokenizer.json")
    raw_semantics, semantic = tokenizer_semantics(raw), tokenizer_semantics(official)
    require(
        raw_semantics == semantic,
        "Tokenizer vocabulary/merges/added-token semantics differ",
    )
    semantic_bytes = canonical_bytes(semantic, newline=True)
    semantic_hash = hashlib.sha256(semantic_bytes).hexdigest()
    require(
        semantic_hash == SEMANTIC_SHA256,
        "Tokenizer semantic hash differs from approved identity",
    )
    tokens = semantic["vocab"] | {
        t["content"]: t["id"] for t in semantic["added_tokens"]
    }
    require(
        all(tokens.get(token) == value for token, value in CONTROL_IDS.items()),
        "EOS/control token ID mismatch",
    )
    pipeline_keys = ("normalizer", "pre_tokenizer", "post_processor", "decoder")
    pipelines = {
        "source": {k: raw.get(k) for k in pipeline_keys},
        "canonical": {k: official.get(k) for k in pipeline_keys},
    }
    return (
        selected,
        semantic_bytes,
        {
            "source": str(assets_source),
            "repo_id": ASSET_REPO,
            "revision": ASSET_REVISION,
            "manifest": {"path": str(assets_manifest), **manifest_info},
            "files": identities,
            "semantic_sha256": semantic_hash,
            "source_semantic_sha256": hashlib.sha256(
                canonical_bytes(raw_semantics, newline=True)
            ).hexdigest(),
            "semantic_representation": "tokenizer-semantics.json",
            "semantic_serialization": (
                "UTF-8 sorted compact JSON; vocab, ordered pair-array merges, "
                "full added_tokens sorted by id; trailing LF"
            ),
            "pipelines_equal": pipelines["source"] == pipelines["canonical"],
            "pipelines": pipelines,
            "omitted_source_assets": [
                name
                for name in ("chat_template.jinja", "processor_config.json")
                if name not in selected
            ],
            "control_ids": CONTROL_IDS,
            "template_policy": (
                "No chat_template.jinja published; embedded tokenizer "
                "chat_template removed; retain explicit server template"
            ),
        },
    )


def _delta(before, after, prefix=""):
    changes = {}
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        if key not in before or key not in after:
            changes[path] = {
                "before_present": key in before,
                "after_present": key in after,
                "before": before.get(key),
                "after": after.get(key),
            }
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.update(_delta(before[key], after[key], path))
        elif before[key] != after[key]:
            changes[path] = {"before": before[key], "after": after[key]}
    return changes


def _write_bytes(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _copy_asset(source, destination, expected):
    stream, before = _open_regular(source)
    sha = hashlib.sha256()
    with stream, destination.open("xb") as output:
        while data := stream.read(CHUNK):
            output.write(data)
            sha.update(data)
        output.flush()
        os.fsync(output.fileno())
    _unchanged(source, before)
    require(
        before == expected["identity"] and sha.hexdigest() == expected["sha256"],
        f"Asset changed during copy: {source}",
    )
    return {"sha256": sha.hexdigest(), "bytes": before["size"]}


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(staging, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-overwrite publication requires renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def prepare_checkpoint(
    source,
    destination,
    acquisition_manifest,
    acquisition_result,
    assets_source,
    assets_manifest,
    protected_root,
):
    """Verify every acquired byte and atomically publish new protected metadata views.

    The trusted protected root and its directories must not be writable by an
    adversarial concurrent principal. Inputs are never changed; existing output
    is always refused, including a dangling symlink or an empty directory.
    """
    root = _directory(protected_root)
    source, assets_source = _directory(source, root), _directory(assets_source, root)
    destination = Path(destination).absolute()
    parent = _directory(destination.parent, root)
    _filename(destination.name)
    require(
        destination != root
        and source != destination
        and source not in destination.parents
        and assets_source != destination
        and assets_source not in destination.parents,
        "Destination must be separate from original model sources",
    )
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing existing destination: {destination}")
    acquisition_manifest, acquisition_result, assets_manifest = map(
        lambda p: Path(p).absolute(),
        (acquisition_manifest, acquisition_result, assets_manifest),
    )
    source_id, asset_id, root_id, parent_id = map(
        _identity, (source, assets_source, root, parent)
    )
    files, acquisition = admit_source(source, acquisition_manifest, acquisition_result)
    tensors, headers = inventory(source, files)
    original, original_info = _read_json(source / "config.json")
    specs = module_specs(original, tensors)
    ple = validate_ple(source, tensors, headers, specs)
    require(len(tensors) == EXPECTED_TENSORS, "Pinned source tensor count mismatch")
    digest = hashlib.sha256(canonical_bytes(specs)).hexdigest()
    configs = normalize_configs(original, specs)
    assets, semantics, asset_evidence = admit_assets(
        source, assets_source, assets_manifest
    )
    asset_evidence["asset_deltas"] = {
        name: {
            "source_sha256": acquisition["source_files"].get(name, {}).get("sha256"),
            "canonical_sha256": asset_evidence["files"][name]["sha256"],
            "policy": "canonical official asset",
        }
        for name in assets
    }
    view_tensors = {
        view: {
            name: record
            for name, record in tensors.items()
            if name.startswith("mtp.") == (view == "mtp")
        }
        for view in ("target", "mtp")
    }
    require(
        all(view_tensors.values()), "Both target and native MTP tensors are required"
    )
    require(
        len(view_tensors["mtp"]) == EXPECTED_MTP_TENSORS,
        "Pinned MTP tensor count mismatch",
    )
    require(
        not any(
            name.startswith(("mtp.embed_tokens.", "mtp.lm_head.", "mtp.shared_head."))
            for name in view_tensors["mtp"]
        ),
        "Draft embedding/head must alias target modules",
    )
    publisher = Path(__file__).absolute()
    publisher_info = _regular(publisher)
    publisher_hash = hashlib.sha256(publisher.read_bytes()).hexdigest()
    _unchanged(publisher, publisher_info)
    receipt = {
        "format_version": 1,
        "ok": True,
        "destination": str(destination),
        "protected_root": str(root),
        "publisher": {"version": VERSION, "sha256": publisher_hash},
        "ple": ple,
        "acquisition": acquisition,
        "source_headers": headers,
        "original_config": original,
        "original_config_sha256": original_info["sha256"],
        "original_quantization": {
            key: original[key]
            for key in ("quantization", "quantization_config")
            if key in original
        },
        "module_specs_sha256": digest,
        "tokenizer_processor": asset_evidence,
        "tensor_count": len(tensors),
        "payload_bytes": sum(h["payload_bytes"] for h in headers.values()),
        "indexed_file_bytes": sum(
            h["data_start"] + h["payload_bytes"] for h in headers.values()
        ),
        "transforms": {
            "gdn_conv_mlx": {
                "operation": "transpose(1,2)",
                "source_shape": [10240, 4, 1],
                "destination_shape": [10240, 1, 4],
                "consumers": ["linear_attn", "ple"],
            },
            "vision_patch_mlx": {
                "operation": "permute(0,4,1,2,3)",
                "source_shape": [1152, 2, 16, 16, 3],
                "destination_shape": [1152, 3, 2, 16, 16],
            },
        },
        "mtp": {
            "checkpoint_layer_prefix": "mtp.layers.0",
            "registered_layer_prefix": "model.layers.0",
            "construction_layer_prefix": "mtp.layers.48",
            "trained_layers": 1,
            "speculative_iterations": 2,
            "module_aliases": {
                "model.embed_tokens": "language_model.model.embed_tokens",
                "lm_head": "language_model.lm_head",
            },
        },
        "views": {},
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=parent))
    staging_id = _identity(staging)
    try:
        for view in ("target", "mtp"):
            view_root = staging if view == "target" else staging / "mtp"
            if view == "mtp":
                view_root.mkdir()
            selected = view_tensors[view]
            manifest = {
                "format_version": 1,
                "view": view,
                "source_revision": REVISION,
                "module_specs_sha256": digest,
                "tensors": selected,
            }
            index = {
                "metadata": {
                    "total_size": sum(
                        t["data_offsets"][1] - t["data_offsets"][0]
                        for t in selected.values()
                    )
                },
                "weight_map": {
                    name: record["file"] for name, record in selected.items()
                },
            }
            manifest_bytes = canonical_bytes(manifest, newline=True)
            configs[view]["mlx_affine"] = {
                "manifest": MANIFEST,
                "view": view,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
            output = {MANIFEST: _write_bytes(view_root / MANIFEST, manifest_bytes)}
            output.update(
                {
                    name: _write_bytes(
                        view_root / name, canonical_bytes(value, newline=True)
                    )
                    for name, value in ((INDEX, index), ("config.json", configs[view]))
                }
            )
            links = {}
            for filename in sorted(set(index["weight_map"].values())):
                relative = os.path.relpath(source / filename, view_root)
                (view_root / filename).symlink_to(relative)
                require(
                    (view_root / filename).resolve(strict=True) == source / filename,
                    "Shard link resolved outside verified source",
                )
                links[filename] = relative
            for name in assets:
                if name == "tokenizer_config.json":
                    config, _ = _read_json(assets_source / name)
                    normalized = copy.deepcopy(config)
                    normalized.pop("chat_template", None)
                    output[name] = _write_bytes(
                        view_root / name, canonical_bytes(normalized, newline=True)
                    )
                    asset_evidence["tokenizer_config_delta"] = _delta(
                        config, normalized
                    )
                else:
                    output[name] = _copy_asset(
                        assets_source / name,
                        view_root / name,
                        asset_evidence["files"][name],
                    )
            receipt["views"][view] = {
                "path": str(destination if view == "target" else destination / "mtp"),
                "tensor_count": len(selected),
                "files": output,
                "linked_shards": links,
                "config_delta": _delta(original, configs[view]),
                "mapping": {
                    name: {"runtime_name": t["runtime_name"], "layout": t["layout"]}
                    for name, t in selected.items()
                },
            }
            _fsync_directory(view_root)
        receipt["semantic_representation"] = _write_bytes(
            staging / "tokenizer-semantics.json", semantics
        )
        receipt["views"]["target"]["files"]["tokenizer-semantics.json"] = receipt[
            "semantic_representation"
        ]
        # Recheck all admitted identities immediately before publishing; do not
        # repeat the 140GiB hash pass or hold any tensor payload in memory.
        for name, info in acquisition["source_files"].items():
            _unchanged(source / name, info["identity"])
        for name, info in asset_evidence["files"].items():
            _unchanged(assets_source / name, info["identity"])
        for record in (
            acquisition["acquisition_manifest"],
            acquisition["acquisition_result"],
            asset_evidence["manifest"],
        ):
            _unchanged(Path(record["path"]), record["identity"])
        _unchanged(source, source_id)
        _unchanged(assets_source, asset_id)
        for path, before in ((root, root_id), (parent, parent_id)):
            now = _identity(path)
            require(
                (now["device"], now["inode"], now["mode"])
                == (before["device"], before["inode"], before["mode"])
                and path.resolve(strict=True) == path,
                "Protected publication directory changed",
            )
        for filename in headers:
            stream, before = _open_regular(source / filename)
            with stream:
                os.fsync(stream.fileno())
            _unchanged(source / filename, before)
        _fsync_directory(source)
        _write_bytes(staging / RECEIPT, canonical_bytes(receipt, newline=True))
        _fsync_directory(staging)
        _publish(staging, destination)
        staging = None
        _fsync_directory(parent)
        return receipt
    finally:
        if staging is not None and os.path.lexists(staging):
            current = _identity(staging)
            if (current["device"], current["inode"]) == (
                staging_id["device"],
                staging_id["inode"],
            ):
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "destination",
        "acquisition-manifest",
        "acquisition-result",
        "assets-source",
        "assets-manifest",
        "protected-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_checkpoint(**vars(args))
    print(
        json.dumps(
            {
                "ok": True,
                "destination": receipt["destination"],
                "tensor_count": receipt["tensor_count"],
                "module_specs_sha256": receipt["module_specs_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
