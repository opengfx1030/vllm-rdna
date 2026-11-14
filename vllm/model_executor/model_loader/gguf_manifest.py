# SPDX-License-Identifier: Apache-2.0
"""Bounded GGUF v3 header validation for the supported mixed-Qwen formats.

This validates metadata and payload extents, not payload integrity or execution.
Tensor dimensions remain in GGUF's innermost-first order. Offsets are absolute.
"""

import math
import os
import struct
from dataclasses import dataclass


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    path: str
    offset: int
    nbytes: int
    dims: tuple[int, ...]
    type: int
    name: str


@dataclass(frozen=True)
class Manifest:
    architecture: str
    tensor_count: int
    shard_count: int
    tensor_names: tuple[str, ...]
    tensors: tuple[TensorInfo, ...]
    metadata: tuple
    shard_tensor_counts: tuple[int, ...]


_TYPES = {
    0: (1, 4),
    7: (32, 24),
    8: (32, 34),
    12: (256, 144),
    13: (256, 176),
    20: (32, 18),
    30: (1, 2),
}
_MAX_STRING = 1 << 20
_MAX_ITEMS = 1 << 24
_MAX_HEADER_BYTES = 64 << 20
_SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_SPLIT_TYPES = {"split.no": 2, "split.count": 2, "split.tensors.count": 5}


class _Reader:
    def __init__(self, f):
        self.f = f
        self.size = os.fstat(f.fileno()).st_size

    def _check(self, n):
        if n < 0 or n > self.size - self.f.tell():
            raise ManifestError("truncated GGUF header")
        if self.f.tell() + n > _MAX_HEADER_BYTES:
            raise ManifestError("GGUF header byte budget exceeded")

    def read(self, n):
        self._check(n)
        data = self.f.read(n)
        if len(data) != n:
            raise ManifestError("truncated GGUF header")
        return data

    def skip(self, n):
        self._check(n)
        self.f.seek(n, 1)

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        if n > _MAX_STRING:
            raise ManifestError("oversized GGUF string")
        try:
            return self.read(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError("invalid UTF-8 GGUF string") from exc


def _scalar(r, typ):
    if typ == 8:
        return r.string()
    if typ not in _SCALAR_FORMATS:
        raise ManifestError("unknown metadata value type")
    fmt = "<" + _SCALAR_FORMATS[typ]
    return struct.unpack(fmt, r.read(struct.calcsize(fmt)))[0]


def _value(r, typ):
    if typ != 9:
        return _scalar(r, typ)
    inner, count = r.u32(), r.u64()
    if count > _MAX_ITEMS:
        raise ManifestError("metadata array too large")
    # Token vocabulary/merges and other string arrays are never retained.
    if inner == 8:
        for _ in range(count):
            r.string()
        return None
    if inner not in _SCALAR_FORMATS:
        raise ManifestError("unknown metadata array element type")
    if count > 4096:
        r.skip(count * struct.calcsize("<" + _SCALAR_FORMATS[inner]))
        return None
    return tuple(_scalar(r, inner) for _ in range(count))


def _shard(path, architecture):
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.read(4) != b"GGUF" or r.u32() != 3:
            raise ManifestError("expected GGUF version 3")
        nt, nm = r.u64(), r.u64()
        if nt > _MAX_ITEMS or nm > _MAX_ITEMS:
            raise ManifestError("header item count too large")
        meta = {}
        for _ in range(nm):
            key, typ = r.string(), r.u32()
            if key in meta:
                raise ManifestError("duplicate metadata key")
            meta[key] = (typ, _value(r, typ))
        for key, typ in _SPLIT_TYPES.items():
            if key in meta and meta[key][0] != typ:
                raise ManifestError(f"{key} has invalid GGUF wire type")
        if "general.architecture" in meta and meta["general.architecture"] != (
            8,
            architecture,
        ):
            raise ManifestError("architecture mismatch")
        raw_tensors = []
        names = set()
        for _ in range(nt):
            name = r.string()
            if not name or name in names:
                raise ManifestError("empty or duplicate tensor name")
            names.add(name)
            nd = r.u32()
            if not 1 <= nd <= 4:
                raise ManifestError("invalid tensor dimension count")
            dims = tuple(r.u64() for _ in range(nd))
            if any(d == 0 for d in dims):
                raise ManifestError("tensor dimensions must be positive")
            typ, offset = r.u32(), r.u64()
            if typ not in _TYPES:
                raise ManifestError(f"unsupported GGML tensor type {typ}")
            block, block_bytes = _TYPES[typ]
            if dims[0] % block:
                raise ManifestError("tensor row is not block aligned")
            nbytes = math.prod(dims) // block * block_bytes
            raw_tensors.append((name, offset, nbytes, dims, typ))
        align_type, align = meta.get("general.alignment", (4, 32))
        if (
            align_type != 4
            or not isinstance(align, int)
            or align <= 0
            or align & (align - 1)
        ):
            raise ManifestError("invalid alignment")
        start = (f.tell() + align - 1) // align * align
        ranges = []
        tensors = []
        # No payload/padding requirement for a zero-tensor metadata-only shard.
        for name, off, size, dims, typ in raw_tensors:
            if off % align or start + off + size > r.size:
                raise ManifestError("tensor payload outside shard")
            ranges.append((off, off + size))
            tensors.append(
                TensorInfo(os.fspath(path), start + off, size, dims, typ, name)
            )
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ManifestError("overlapping tensor payloads")
        return meta, tuple(tensors)


def validate_shards(paths, architecture):
    paths = tuple(paths)
    if not paths:
        raise ManifestError("no shards supplied")
    if not isinstance(architecture, str) or not architecture:
        raise ManifestError("expected a nonempty architecture")
    entries = [_shard(path, architecture) for path in paths]
    first = entries[0][0]
    if first.get("general.architecture") != (8, architecture):
        raise ManifestError("first shard missing architecture")
    is_split = any(key in first for key in _SPLIT_TYPES)
    count = first.get("split.count", (None, None))[1] if is_split else 1
    if count != len(paths) or count <= 0:
        raise ManifestError("incomplete shard set")
    total = sum(len(tensors) for _, tensors in entries)
    seen = set()
    tensors = []
    for i, (meta, current) in enumerate(entries):
        if is_split:
            if any(key not in meta for key in _SPLIT_TYPES):
                raise ManifestError("missing split metadata")
            if meta["split.count"][1] != count or meta["split.no"][1] != i:
                raise ManifestError("inconsistent shard numbering")
            if meta["split.tensors.count"][1] != total:
                raise ManifestError("global tensor count mismatch")
        elif any(key in meta for key in _SPLIT_TYPES):
            raise ManifestError("inconsistent split metadata")
        for tensor in current:
            if tensor.name in seen:
                raise ManifestError("duplicate tensor name across shards")
            seen.add(tensor.name)
            tensors.append(tensor)
    return Manifest(
        architecture,
        total,
        count,
        tuple(sorted(seen)),
        tuple(tensors),
        tuple(first.items()),
        tuple(len(ts) for _, ts in entries),
    )
