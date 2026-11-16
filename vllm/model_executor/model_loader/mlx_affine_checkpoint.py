# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only admission of an immutable MLX affine checkpoint view."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MANIFEST_NAME = "mlx-affine-manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate affine manifest key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite affine manifest value: {value}")


@dataclass(frozen=True)
class MlxAffineCheckpoint:
    root: Path
    manifest: dict[str, Any]
    tensor_names: frozenset[str]
    shard_paths: tuple[str, ...]
    manifest_sha256: str


def load_mlx_affine_checkpoint(
    model_path: str | Path,
    hf_config: dict[str, Any],
    *,
    protected_model_root: str | Path | None = None,
) -> MlxAffineCheckpoint:
    """Bind tensor selection to config bytes before destination allocation.

    Tensor geometry, components and ownership are validated by
    MlxAffineLoadCoverage, not a second tensor validator here.
    """
    config = hf_config.get("quantization_config")
    pin = hf_config.get("mlx_affine")
    if not isinstance(config, dict) or config.get("quant_method") != "mlx_affine":
        raise ValueError("Expected a normalized mlx_affine quantization config")
    if (
        not isinstance(pin, dict)
        or set(pin) != {"manifest", "manifest_sha256", "view"}
        or pin["manifest"] != _MANIFEST_NAME
        or pin["view"] not in ("target", "mtp")
        or not isinstance(pin["manifest_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", pin["manifest_sha256"]) is None
    ):
        raise ValueError("Missing or invalid affine manifest config pin")

    root = Path(model_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Affine model view is not a directory")
    protected = protected_model_root
    if protected is None:
        protected = root.parent if pin["view"] == "target" else root.parent.parent
    protected = Path(protected).resolve(strict=True)
    if not root.is_relative_to(protected):
        raise ValueError("Affine model view escapes the protected model root")
    path = root / _MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("Affine manifest must be a regular file in the native view")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(_MAX_MANIFEST_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (
        len(raw) > _MAX_MANIFEST_BYTES
        or len(raw) != before.st_size
        or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ValueError("Affine manifest changed or exceeds its bounded size")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin["manifest_sha256"]:
        raise ValueError("Affine manifest does not match its config pin")
    manifest = json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
    if not isinstance(manifest, dict) or manifest.get("view") != pin["view"]:
        raise ValueError("Affine manifest view does not match its config pin")
    specs = config.get("module_specs")
    if not isinstance(specs, dict):
        raise ValueError("Affine checkpoint requires an explicit module spec map")
    specs_digest = hashlib.sha256(
        json.dumps(
            specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    if manifest.get("module_specs_sha256") != specs_digest:
        raise ValueError("Affine manifest and quantization map disagree")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Affine manifest has no selected tensors")
    shard_names = set()
    for name, entry in tensors.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("Invalid affine source record")
        shard = entry.get("file")
        if (
            not isinstance(shard, str)
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.safetensors", shard) is None
        ):
            raise ValueError(f"Unsafe affine shard filename for {name}")
        shard_names.add(shard)
    paths = []
    for name in sorted(shard_names):
        link = root / name
        if not link.is_symlink() or Path(os.readlink(link)).is_absolute():
            raise ValueError(f"Affine shard must be an original relative link: {name}")
        source = link.resolve(strict=True)
        if not source.is_relative_to(protected) or not source.is_file():
            raise ValueError(f"Affine shard escapes protected model storage: {name}")
        paths.append(str(link))
    return MlxAffineCheckpoint(root, manifest, frozenset(tensors), tuple(paths), digest)
