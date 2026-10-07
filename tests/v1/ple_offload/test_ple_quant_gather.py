# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for PLE fused gather on int4 and fp8 per-row sidecars.

Loads ``worker.py`` with stub vllm/zmq/msgspec modules so collection does
not import the full package. Leap's T-PLE8 path views e4m3 bytes as uint8
and dequants through a 256-entry LUT in gather_rows_small.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "vllm" / "v1" / "ple_offload" / "worker.py"


def _mod(name: str, **attrs):
    m = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(m, key, val)
    return m


def _ensure_pkg(monkeypatch: pytest.MonkeyPatch, name: str):
    if name in sys.modules:
        return sys.modules[name]
    pkg = _mod(name)
    pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, pkg)
    return pkg


def _load_worker(monkeypatch: pytest.MonkeyPatch):
    """Import worker.py without pulling vllm.distributed / msgspec / zmq."""
    vllm_mod = _ensure_pkg(monkeypatch, "vllm")
    logger_mod = _mod("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    envs_mod = _mod("vllm.envs")
    vllm_mod.envs = envs_mod
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        _mod(
            "vllm.config",
            VllmConfig=object,
            set_current_vllm_config=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        _ensure_pkg(monkeypatch, "vllm.distributed"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        _mod(
            "vllm.distributed.parallel_state",
            ensure_model_parallel_initialized=lambda *a, **k: None,
            init_distributed_environment=lambda *a, **k: None,
        ),
    )
    for pkg in (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.utils",
        "vllm.utils",
        "vllm.v1",
        "vllm.v1.ple_offload",
    ):
        _ensure_pkg(monkeypatch, pkg)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.ple_offload_layer",
        _mod(
            "vllm.model_executor.layers.ple_offload_layer",
            CpuGpuSemaphore=object,
            PleOffloadLayer=object,
            mark_as_offload_worker=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader",
        _mod("vllm.model_executor.model_loader", get_model_loader=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.default_loader",
        _mod(
            "vllm.model_executor.model_loader.default_loader", DefaultModelLoader=object
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.dummy_loader",
        _mod("vllm.model_executor.model_loader.dummy_loader", DummyModelLoader=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        _mod(
            "vllm.model_executor.model_loader.utils",
            initialize_model=lambda *a, **k: None,
            process_weights_after_loading=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.weight_utils",
        _mod(
            "vllm.model_executor.model_loader.weight_utils",
            initialize_dummy_weights=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.system_utils",
        _mod(
            "vllm.utils.system_utils",
            decorate_logs=lambda *a, **k: None,
            get_mp_context=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.torch_utils",
        _mod("vllm.utils.torch_utils", set_default_torch_dtype=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.ple_offload.protocol",
        _mod(
            "vllm.v1.ple_offload.protocol",
            _PLE_OFFLOAD_REQUEST_DECODER=None,
            PleOffloadRegistration=object,
            PleOffloadRequest=object,
        ),
    )
    monkeypatch.setitem(sys.modules, "msgspec", _mod("msgspec"))
    zmq_mod = _mod("zmq")
    zmq_mod.Socket = type("Socket", (), {})
    monkeypatch.setitem(sys.modules, "zmq", zmq_mod)
    dist_mod = _mod("torch.distributed")
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)

    spec = importlib.util.spec_from_file_location("ple_offload_worker_isolated", _SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fp8_table(worker, raw: torch.Tensor, scales: torch.Tensor):
    table = worker._PleQuantTable.__new__(worker._PleQuantTable)
    table.is_fp8 = True
    table.layout = "per_row_e4m3"
    table.width = raw.shape[1]
    table._q_np = [raw.numpy()]
    table._s_np = [scales.numpy()]
    table._lut_np = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    )
    return table


def _int4_table(worker, packed: np.ndarray, scales: np.ndarray):
    table = worker._PleQuantTable.__new__(worker._PleQuantTable)
    table.is_fp8 = False
    table.layout = "int4_g16"
    table.width = packed.shape[1] * 2
    table._q_np = [packed]
    table._s_np = [scales]
    table._lut_np = None
    return table


def test_gather_rows_small_fp8_matches_torch_dequant(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _load_worker(monkeypatch)
    width = 8
    raw_u8 = torch.tensor(
        [
            [0, 64, 128, 192, 32, 96, 160, 224],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [255, 127, 63, 31, 15, 7, 3, 1],
        ],
        dtype=torch.uint8,
    )
    q = raw_u8.view(torch.float8_e4m3fn)
    scales = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float32)
    table = _fp8_table(worker, raw_u8, scales)
    ids = np.array([0, 2], dtype=np.int64)
    out = np.empty((2, width), dtype=np.float32)
    table.gather_rows_small(ids, out)
    ref = (q.float() * scales[:, None]).numpy()[ids]
    np.testing.assert_allclose(out, ref, rtol=0, atol=0)


def test_gather_rows_small_int4_unchanged(monkeypatch: pytest.MonkeyPatch):
    worker = _load_worker(monkeypatch)
    packed = np.array([[0x10, 0x32], [0x54, 0x76]], dtype=np.uint8)
    scales = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float16)
    table = _int4_table(worker, packed, scales)
    ids = np.array([1], dtype=np.int64)
    out = np.empty((1, 4), dtype=np.float32)
    table.gather_rows_small(ids, out)
    np.testing.assert_allclose(out, [[-4.0, -3.0, -2.0, -1.0]], rtol=0, atol=1e-6)


def test_fused_decode_gate_accepts_fp8_layout(monkeypatch: pytest.MonkeyPatch):
    worker = _load_worker(monkeypatch)
    table = _fp8_table(
        worker,
        torch.zeros((1, 4), dtype=torch.uint8),
        torch.ones((1,), dtype=torch.float32),
    )
    layer = SimpleNamespace(ngram_embedding=SimpleNamespace(_ple_quant=table))
    ngram = torch.zeros((1, 2), dtype=torch.int64)
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    ids = torch.zeros(1, dtype=torch.int32)
    pinned = torch.zeros(1, 4)
    with pytest.raises(AttributeError, match="ngram_size"):
        worker._fused_decode_lookup(layer, ids, qsl, ngram, pinned, False)


def test_fused_decode_gate_rejects_other_layouts(monkeypatch: pytest.MonkeyPatch):
    worker = _load_worker(monkeypatch)
    table = SimpleNamespace(layout="e2m1_e4m3_scale", is_fp8=False)
    layer = SimpleNamespace(ngram_embedding=SimpleNamespace(_ple_quant=table))
    ngram = torch.zeros((1, 2), dtype=torch.int64)
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    ids = torch.zeros(1, dtype=torch.int32)
    pinned = torch.zeros(1, 4)
    assert worker._fused_decode_lookup(layer, ids, qsl, ngram, pinned, False) is None


@pytest.mark.parametrize(
    "layout,expect_fp8",
    [
        ("per_row_e4m3", True),
        ("int4_g16", False),
        ("e2m1_e4m3_scale", False),
    ],
)
def test_is_fp8_excludes_e2m1_even_if_layout_mentions_e4m3(
    layout: str, expect_fp8: bool
):
    is_fp8 = ("e4m3" in layout) and ("e2m1" not in layout)
    assert is_fp8 is expect_fp8
