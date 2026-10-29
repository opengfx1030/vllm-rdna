# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata fixtures; executing geometry must not import torch."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest


@pytest.fixture
def geometry():
    path = (
        Path(__file__).resolve().parents[3] / "tools/rdna/mono/expert_cache_worker.py"
    )
    tree = ast.parse(path.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_packed_row_geometry"
    )
    namespace: dict[str, Any] = {"Any": Any}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_packed_row_geometry"]


def spec(padded=64, real=48):
    return NS(page_size_bytes=padded, real_page_size_bytes=real)


def placement(layers, offset=0, stride=64, block_stride=256):
    return NS(
        layers=layers,
        offset=offset,
        layer_stride=stride,
        block_stride=block_stride,
        size=2560,
    )


def config(tensors, groups):
    return NS(kv_cache_tensors=tensors, kv_cache_groups=groups)


def group(names, specs):
    return NS(layer_names=names, kv_cache_spec=NS(kv_cache_specs=specs))


def test_placement_padding_and_owner_aliases(geometry):
    cfg = config(
        [placement(["a", "b"], offset=16), placement(["c"], offset=32)],
        [
            group(["a", "b"], {"a": spec(), "b": spec(64, 64)}),
            group(["c"], {"c": spec(128, 120)}),
        ],
    )
    report = geometry(cfg, 256)
    assert report["available"]
    a, b = report["groups"]
    assert a["defined_content_intervals"] == [[16, 64], [80, 144]]
    assert a["defined_bytes"] == 112
    assert a["candidate_omitted_bytes"] == 144
    assert a["layers"][1] == dict(
        layer="b",
        offset=16,
        layer_index=1,
        layer_stride=64,
        block_stride=256,
        row_offset=80,
        page_size_bytes=64,
        real_page_size_bytes=64,
    )
    # Owners overlay, so their intervals must never be combined.
    assert b["defined_content_intervals"] == [[32, 152]]
    assert b["candidate_omitted_bytes"] == 136


def test_within_owner_alias_union_no_doublecount(geometry):
    cfg = config(
        [placement(["a", "b"], stride=32)],
        [group(["a", "b"], {"a": spec(), "b": spec()})],
    )
    result = geometry(cfg, 256)["groups"][0]
    assert result["defined_content_intervals"] == [[0, 80]]
    assert result["defined_bytes"] == 80
    assert result["candidate_omitted_bytes"] == 176


@pytest.mark.parametrize(
    "change",
    [
        {"block_stride": 64},
        {"layer_stride": 2560},
        {"offset": 240},
        {"offset": -1},
        {"size": 32},
        {"host_resident": True},
    ],
)
def test_unsupported_geometry_never_claims_savings(geometry, change):
    tensor = placement(["a", "b"])
    vars(tensor).update(change)
    result = geometry(
        config([tensor], [group(["a", "b"], {"a": spec(), "b": spec()})]), 256
    )
    assert not result["available"]
    assert "candidate_omitted_bytes" not in result["groups"][0]


@pytest.mark.parametrize("case", ["placement", "spec", "real", "duplicate", "physical"])
def test_missing_metadata_is_unavailable(geometry, case):
    tensor = placement(["a"])
    specs = {"a": spec()}
    tensors = [tensor]
    page: int | None = 256
    if case == "placement":
        tensors = []
    elif case == "spec":
        specs = {}
    elif case == "real":
        del specs["a"].real_page_size_bytes
    elif case == "duplicate":
        tensors.append(placement(["a"]))
    else:
        page = None
    report = geometry(config(tensors, [group(["a"], specs)]), page)
    assert not report["available"]
    assert all("candidate_omitted_bytes" not in g for g in report["groups"])


def test_plain_per_layer_spec_includes_entire_real_payload(geometry):
    # E.g. attention raw-ring position tail belongs to real_page_size_bytes.
    cfg = config(
        [placement(["a"])], [NS(layer_names=["a"], kv_cache_spec=spec(128, 104))]
    )
    result = geometry(cfg, 256)["groups"][0]
    assert result["defined_content_intervals"] == [[0, 104]]


def test_worker_report_exposes_geometry_without_device_work(monkeypatch):
    import sys

    path = (
        Path(__file__).resolve().parents[3] / "tools/rdna/mono/expert_cache_worker.py"
    )
    tree = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace: dict[str, Any] = {"Any": Any, "sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    wrapper = NS(
        worker=NS(supports_physical_page_snapshots=True),
        spec=NS(blocks_per_chunk=1, cpu_page_size_per_worker=256),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_transfer_state",
        NS(_KV_CONNECTOR_AGENT=NS(connector_worker=wrapper)),
    )
    worker = namespace["ExpertCacheWorker"]()
    worker.model_runner = NS(
        kv_cache_config=config([placement(["a"])], [group(["a"], {"a": spec()})])
    )
    report = worker.rdna_kv_snapshot_layout()
    assert report["packed_row_geometry"]["groups"][0]["defined_bytes"] == 48
    assert report["physical_page_bytes"] == 256


@pytest.mark.parametrize(
    "kind",
    [
        "mamba",
        "mamba_empty_state",
        "attention",
        "attention_no_v",
        "packed_attention",
    ],
)
def test_static_spec_never_reads_allocating_byte_properties(geometry, kind):
    class StaticSpec:
        @property
        def page_size_bytes(self):
            raise AssertionError("allocating property accessed")

        @property
        def real_page_size_bytes(self):
            raise AssertionError("allocating property accessed")

    static: Any = StaticSpec()
    static.page_size_padded = 128
    if kind.startswith("mamba"):
        static.shapes = [(2, 3), (0, 9) if kind == "mamba_empty_state" else (4,)]
        static.dtypes = [NS(itemsize=2), NS(itemsize=4)]
        expected = 12 if kind == "mamba_empty_state" else 28
    else:
        static.num_kv_heads = 2
        static.num_head_slots = None
        static.num_states = 3
        static.head_size = 4
        static.head_size_v = 0 if kind == "attention_no_v" else 2
        static.dtype = NS(itemsize=2)
        static.state_content_bytes = 16 if kind == "packed_attention" else None
        expected = {"packed_attention": 96, "attention_no_v": 48}.get(kind, 72)
    cfg = config([placement(["a"])], [NS(layer_names=["a"], kv_cache_spec=static)])
    report = geometry(cfg, 256)["groups"][0]
    assert report["available"]
    assert report["defined_bytes"] == expected
    assert report["layers"][0]["page_size_bytes"] == 128
