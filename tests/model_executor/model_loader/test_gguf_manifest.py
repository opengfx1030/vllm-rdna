# SPDX-License-Identifier: Apache-2.0
"""Wire-realistic GGUF fixtures; no model payload expansion."""

import importlib.util
import math
import pathlib
import struct
import sys

import pytest

MODULE = (
    pathlib.Path(__file__).parents[3]
    / "vllm/model_executor/model_loader/gguf_manifest.py"
)
spec = importlib.util.spec_from_file_location("gguf_manifest", MODULE)
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)
GEOMETRY = {
    0: (1, 4),
    7: (32, 24),
    8: (32, 34),
    12: (256, 144),
    13: (256, 176),
    20: (32, 18),
    30: (1, 2),
}


def _string(s):
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture(
    path,
    *,
    tensors=(),
    count=1,
    no=0,
    total=None,
    architecture="qwen4exp",
    metadata=(),
    overrides=None,
    offsets=None,
):
    md = [
        ("split.count", 2, count),
        ("split.no", 2, no),
        ("split.tensors.count", 5, len(tensors) if total is None else total),
    ]
    if architecture is not None:
        md.append(("general.architecture", 8, architecture))
    if overrides:
        md = [(k, *overrides.get(k, (t, v))) for k, t, v in md]
    md += list(metadata)
    b = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(md)))
    for k, t, v in md:
        b += _string(k) + struct.pack("<I", t)
        b += (
            _string(v)
            if t == 8
            else struct.pack("<" + {2: "H", 4: "I", 5: "i", 7: "?"}[t], v)
        )
    payloads = []
    pos = 0
    for i, (name, typ, dims) in enumerate(tensors):
        block, size = GEOMETRY.get(typ, (1, 4))
        length = math.prod(dims) // block * size
        off = (pos + 31) // 32 * 32 if offsets is None else offsets[i]
        payloads.append((off, length))
        pos = max(pos, off + length)
        b += _string(name) + struct.pack("<I", len(dims))
        b += b"".join(struct.pack("<Q", d) for d in dims)
        b += struct.pack("<IQ", typ, off)
    if tensors:
        b += bytes((-len(b)) % 32)
        b += bytes(pos)
    path.write_bytes(b)
    return path


def test_real_split_wire_types_and_global_total_with_empty_first_shard(tmp_path):
    first = fixture(tmp_path / "0.gguf", count=2, no=0, total=2)
    assert first.stat().st_size % 32 != 0
    second = fixture(
        tmp_path / "1.gguf",
        count=2,
        no=1,
        total=2,
        architecture=None,
        tensors=[("a", 0, (4,)), ("b", 20, (160, 2))],
    )
    m = g.validate_shards([first, second], architecture="qwen4exp")
    assert (m.tensor_count, m.shard_count) == (2, 2)
    assert m.tensor_names == ("a", "b")
    by_name = {t.name: t for t in m.tensors}
    assert by_name["b"].dims == (160, 2)
    assert by_name["b"].nbytes == 180
    assert by_name["b"].offset % 32 == 0
    assert by_name["b"].path == str(second)
    assert by_name["b"].offset + by_name["b"].nbytes == second.stat().st_size


def test_consistent_architecture_repeated_on_later_shard_is_valid(tmp_path):
    paths = [
        fixture(
            tmp_path / str(i),
            count=2,
            no=i,
            total=1,
            tensors=[] if i == 0 else [("x", 0, (4,))],
        )
        for i in range(2)
    ]
    assert g.validate_shards(paths, "qwen4exp").tensor_count == 1


@pytest.mark.parametrize("bad", [0, 2, -1])
def test_rejects_inconsistent_later_global_total(tmp_path, bad):
    paths = [
        fixture(tmp_path / "0", count=2, total=1),
        fixture(
            tmp_path / "1",
            count=2,
            no=1,
            total=bad,
            architecture=None,
            tensors=[("x", 0, (4,))],
        ),
    ]
    with pytest.raises(g.ManifestError, match="global tensor count"):
        g.validate_shards(paths, "qwen4exp")


@pytest.mark.parametrize(
    "key,wire,value",
    [
        ("split.count", 7, True),
        ("split.count", 4, 1),
        ("split.no", 4, 0),
        ("split.tensors.count", 4, 0),
    ],
)
def test_rejects_wrong_split_wire_types(tmp_path, key, wire, value):
    p = fixture(tmp_path / "x", overrides={key: (wire, value)})
    with pytest.raises(g.ManifestError, match="wire type"):
        g.validate_shards([p], "qwen4exp")


@pytest.mark.parametrize(
    "typ,block,nbytes",
    [
        (0, 1, 4),
        (7, 32, 24),
        (8, 32, 34),
        (12, 256, 144),
        (13, 256, 176),
        (20, 32, 18),
        (30, 1, 2),
    ],
)
def test_exact_absolute_offsets_and_all_required_geometries(
    tmp_path, typ, block, nbytes
):
    p = fixture(tmp_path / "x", tensors=[("x", typ, (block, 3, 2))])
    (t,) = g.validate_shards([p], "qwen4exp").tensors
    assert (t.name, t.type, t.dims, t.nbytes) == ("x", typ, (block, 3, 2), 6 * nbytes)
    assert t.offset + t.nbytes == p.stat().st_size
    assert t.offset % 32 == 0


def test_rejects_wrong_global_total_even_if_local_shard_count_matches(tmp_path):
    paths = [
        fixture(tmp_path / "0", count=2, total=0),
        fixture(
            tmp_path / "1",
            count=2,
            no=1,
            total=1,
            architecture=None,
            tensors=[("x", 0, (4,))],
        ),
    ]
    with pytest.raises(g.ManifestError, match="global tensor count"):
        g.validate_shards(paths, "qwen4exp")


@pytest.mark.parametrize(
    "case",
    ["duplicate", "unknown", "unaligned_row", "zero_dim", "overlap", "truncated"],
)
def test_rejects_invalid_tensor_headers_and_payload_extents(tmp_path, case):
    tensors = [("x", 0, (4,))]
    offsets = None
    if case == "duplicate":
        tensors *= 2
    if case == "unknown":
        tensors = [("x", 99, (4,))]
    if case == "unaligned_row":
        tensors = [("x", 7, (31,))]
    if case == "zero_dim":
        tensors = [("x", 0, (0,))]
    if case == "overlap":
        tensors = [("x", 0, (4,)), ("y", 0, (4,))]
        offsets = [0, 0]
    p = fixture(tmp_path / "x", tensors=tensors, offsets=offsets)
    if case == "truncated":
        p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(g.ManifestError):
        g.validate_shards([p], "qwen4exp")


@pytest.mark.parametrize(
    "case", ["missing", "reversed", "wrong_arch", "first_arch_missing"]
)
def test_rejects_incomplete_or_inconsistent_shard_sets(tmp_path, case):
    paths = [
        fixture(
            tmp_path / "0",
            count=2,
            total=1,
            architecture=None if case == "first_arch_missing" else "qwen4exp",
        ),
        fixture(
            tmp_path / "1",
            count=2,
            no=1,
            total=1,
            architecture="wrong" if case == "wrong_arch" else "qwen4exp",
            tensors=[("x", 0, (4,))],
        ),
    ]
    if case == "missing":
        paths.pop()
    if case == "reversed":
        paths.reverse()
    with pytest.raises(g.ManifestError):
        g.validate_shards(paths, "qwen4exp")


def test_rejects_duplicate_tensor_across_shards(tmp_path):
    paths = [
        fixture(tmp_path / str(i), count=2, no=i, total=2, tensors=[("x", 0, (4,))])
        for i in range(2)
    ]
    with pytest.raises(g.ManifestError, match="duplicate tensor"):
        g.validate_shards(paths, "qwen4exp")


def test_reader_enforces_header_byte_budget(tmp_path, monkeypatch):
    p = fixture(tmp_path / "x", tensors=[("x", 0, (4,))])
    monkeypatch.setattr(g, "_MAX_HEADER_BYTES", 64)
    with pytest.raises(g.ManifestError, match="header byte budget"):
        g.validate_shards([p], "qwen4exp")


def test_rejects_invalid_utf8_in_metadata_string_array(tmp_path):
    b = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 2))
    b += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen4exp")
    b += _string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 1)
    b += struct.pack("<Q", 2) + b"\xff\xfe"
    p = tmp_path / "bad_arr.gguf"
    p.write_bytes(b)
    with pytest.raises(g.ManifestError, match="invalid UTF-8"):
        g.validate_shards([p], "qwen4exp")


def test_nondefault_alignment_computes_exact_absolute_offset(tmp_path):
    p = tmp_path / "align64.gguf"
    b = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 2))
    b += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen4exp")
    b += _string("general.alignment") + struct.pack("<II", 4, 64)
    b += _string("x") + struct.pack("<IQ", 1, 4) + struct.pack("<IQ", 0, 64)
    b += bytes((-len(b)) % 64)
    start = len(b)
    b += bytes(64 + 16)
    p.write_bytes(b)
    (t,) = g.validate_shards([p], "qwen4exp").tensors
    assert t.offset == start + 64
    assert t.nbytes == 16
