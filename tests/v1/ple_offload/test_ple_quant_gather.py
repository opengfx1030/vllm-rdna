# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for PLE fused gather on int4 and fp8 per-row sidecars.

Legacy tests load ``worker.py`` with stub dependencies. Native INT4 regressions
require the registered ROCm extension and construct real safetensors sidecars
through the actual worker class, without the isolated import harness.
"""

from __future__ import annotations

import gc
import importlib.util
import json
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
    monkeypatch.setattr(vllm_mod, "envs", envs_mod, raising=False)
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
            initialize_single_dummy_weight=lambda *a, **k: None,
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
    table._native_int4 = None
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
    table._native_int4 = None
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


class _RecordingTable:
    """Quant-table stand-in that records the gathered ids (int4 gate)."""

    layout = "int4_g16"
    is_fp8 = False

    def __init__(self) -> None:
        self.ids: list[np.ndarray] = []

    def gather_rows_small(self, ids, out) -> None:
        self.ids.append(np.array(ids))
        out[:] = 0.0


def _ngram_layer(table, ngram_size=3, hpn=2, eos=7):
    heads = (ngram_size - 1) * hpn
    gen = torch.Generator().manual_seed(1)
    return SimpleNamespace(
        ngram_embedding=SimpleNamespace(_ple_quant=table),
        ngram_size=ngram_size,
        heads_per_ngram=hpn,
        eos_token_id=eos,
        layer_multipliers=torch.randint(
            1, 2**62, (ngram_size,), generator=gen, dtype=torch.long
        ),
        ngram_heads_vocab_sizes=torch.tensor([101, 103, 107, 109])[:heads],
        ngram_heads_offsets=torch.tensor([0, 101, 204, 311])[:heads],
        head_dim=2,
        embedding_dim=2 * heads,
    )


def test_fused_spec_verify_matches_single_token_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """MTP verify (k tokens per request) gathers the same ids as k decode steps."""
    worker = _load_worker(monkeypatch)
    num_reqs, k, n = 3, 3, 3
    gen = torch.Generator().manual_seed(0)
    # Small vocabulary so EOS (7) shows up inside contexts and drafts.
    ctx = torch.randint(0, 12, (num_reqs, n - 1), generator=gen, dtype=torch.long)
    tok = torch.randint(0, 12, (num_reqs, k), generator=gen, dtype=torch.long)
    pinned = torch.zeros(num_reqs * k + 2, 8, dtype=torch.float16)

    spec = _RecordingTable()
    qsl = torch.arange(0, (num_reqs + 1) * k, k, dtype=torch.int32)
    out = worker._fused_decode_lookup(
        _ngram_layer(spec), tok.reshape(-1), qsl, ctx, pinned, False
    )
    assert out is not None and out.shape[0] == num_reqs * k
    got = spec.ids[0].reshape(num_reqs, k, -1)

    for j in range(k):
        single = _RecordingTable()
        history = torch.cat([ctx, tok[:, :j]], dim=1)[:, -(n - 1) :]
        worker._fused_decode_lookup(
            _ngram_layer(single),
            tok[:, j].contiguous(),
            torch.arange(num_reqs + 1, dtype=torch.int32),
            history,
            torch.zeros(num_reqs, 8, dtype=torch.float16),
            False,
        )
        np.testing.assert_array_equal(got[:, j], single.ids[0].reshape(num_reqs, -1))


def test_fused_path_rejects_non_uniform_batches(monkeypatch: pytest.MonkeyPatch):
    worker = _load_worker(monkeypatch)
    table = _RecordingTable()
    ids = torch.zeros(5, dtype=torch.long)
    qsl = torch.tensor([0, 3, 5], dtype=torch.int32)  # 3 + 2 tokens: prefill
    ctx = torch.zeros(2, 2, dtype=torch.long)
    pinned = torch.zeros(5, 8, dtype=torch.float16)
    layer = _ngram_layer(table)
    assert worker._fused_decode_lookup(layer, ids, qsl, ctx, pinned, False) is None


@pytest.fixture
def native_int4_class():
    """Require the real ROCm extension, including its custom-class registration."""
    import vllm._C  # noqa: F401

    return torch.classes._C.PleInt4Table


def _int4_shards(rows=3, width=32):
    packed, scales = [], []
    # Signed zeros, subnormals, FP16-overflowing products and non-power-of-two
    # scales distinguish FP32 dequantization from premature FP16 rounding.
    values = torch.tensor(
        [
            0.0,
            -0.0,
            2**-24,
            -(2**-24),
            65504,
            -65504,
            0.333251953125,
            1.0009765625,
            2**-14,
            -(2**-14),
            0.0999755859375,
            -1.0009765625,
        ],
        dtype=torch.float16,
    )
    for shard in range(2):
        index = torch.arange(rows * width // 2).reshape(rows, width // 2)
        packed.append(((index * 37 + shard * 71) % 256).to(torch.uint8))
        index = torch.arange(rows * (width // 16)).reshape(rows, width // 16)
        scales.append(values[(index + shard * rows * (width // 16)) % len(values)])
    return packed, scales


def _int4_numpy_reference(packed, scales):
    """The replaced NumPy dequantization, including its FP32 multiplication."""
    packed = np.concatenate([tensor.numpy() for tensor in packed])
    scales = np.concatenate([tensor.numpy() for tensor in scales])
    lo = (packed & 0xF).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    nib = np.empty((len(packed), packed.shape[1] * 2), dtype=np.float32)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    return torch.from_numpy(
        (nib - 8.0) * np.repeat(scales.astype(np.float32), 16, axis=1)
    )


def _assert_float_bits(actual, expected):
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(
        actual.to(torch.float16).view(torch.int16),
        expected.to(torch.float16).view(torch.int16),
    )


def test_native_int4_bitwise_group_shard_order_and_torch_copy(native_int4_class):
    packed, scales = _int4_shards()
    table = native_int4_class(packed, scales, 3)
    reference = _int4_numpy_reference(packed, scales)
    for ids in (
        torch.tensor([5, 0, 3, 2, 1, 4, 3, 0]),
        torch.tensor([1, 5, 2]),
        torch.empty(0, dtype=torch.int64),
    ):
        out = torch.full((len(ids), 32), 123.0)
        table.gather(ids, out)
        _assert_float_bits(out, reference[ids])
        half = torch.full_like(out, 123, dtype=torch.float16)
        half.copy_(out)
        assert torch.equal(
            half.view(torch.int16), reference[ids].half().view(torch.int16)
        )


@pytest.mark.parametrize("bad_id", [-1, 6, -(2**63), 2**63 - 1])
def test_native_int4_rejects_all_ids_before_writing(native_int4_class, bad_id):
    packed, scales = _int4_shards()
    table = native_int4_class(packed, scales, 3)
    out = torch.full((3, 32), 123.25)
    before = out.clone()
    with pytest.raises(RuntimeError, match="out of bounds"):
        table.gather(torch.tensor([0, 5, bad_id]), out)
    assert torch.equal(out, before)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "counts",
        "zero_rows",
        "wrong_rows",
        "packed_dtype",
        "scale_dtype",
        "packed_rank",
        "scale_rank",
        "scale_groups",
        "group_width",
        "packed_stride",
        "scale_stride",
        "shard_width",
        "meta_device",
    ],
)
def test_native_int4_constructor_rejects_metadata(native_int4_class, case):
    packed, scales = _int4_shards()
    rows = 3
    if case == "empty":
        packed, scales = [], []
    elif case == "counts":
        scales.pop()
    elif case == "zero_rows":
        rows = 0
    elif case == "wrong_rows":
        rows = 4
    elif case == "packed_dtype":
        packed[1] = packed[1].to(torch.int8)
    elif case == "scale_dtype":
        scales[1] = scales[1].float()
    elif case == "packed_rank":
        packed[1] = packed[1].flatten()
    elif case == "scale_rank":
        scales[1] = scales[1].flatten()
    elif case == "scale_groups":
        scales[1] = scales[1][:, :1].contiguous()
    elif case == "group_width":
        packed = [p[:, :15].contiguous() for p in packed]
    elif case == "packed_stride":
        packed[1] = torch.empty((3, 32), dtype=torch.uint8)[:, ::2]
    elif case == "scale_stride":
        scales[1] = torch.empty((3, 4), dtype=torch.float16)[:, ::2]
    elif case == "shard_width":
        packed[1] = torch.empty((3, 8), dtype=torch.uint8)
        scales[1] = torch.ones((3, 1), dtype=torch.float16)
    elif case == "meta_device":
        packed[1] = packed[1].to("meta")
    with pytest.raises(RuntimeError, match="PleInt4Table"):
        native_int4_class(packed, scales, rows)


@pytest.mark.parametrize(
    "case",
    [
        "ids_dtype",
        "ids_rank",
        "ids_stride",
        "ids_device",
        "out_dtype",
        "out_rank",
        "out_rows",
        "out_width",
        "out_stride",
        "out_device",
    ],
)
def test_native_int4_rejects_gather_metadata_before_writing(native_int4_class, case):
    packed, scales = _int4_shards()
    table = native_int4_class(packed, scales, 3)
    ids = torch.tensor([0, 5], dtype=torch.int64)
    out = torch.full((2, 32), 123.25)
    if case == "ids_dtype":
        ids = ids.int()
    elif case == "ids_rank":
        ids = ids.reshape(1, 2)
    elif case == "ids_stride":
        ids = torch.tensor([0, 4, 5, 4])[::2]
    elif case == "ids_device":
        ids = ids.to("meta")
    elif case == "out_dtype":
        out = out.half()
    elif case == "out_rank":
        out = out.flatten()
    elif case == "out_rows":
        out = out[:1]
    elif case == "out_width":
        out = torch.full((2, 16), 123.25)
    elif case == "out_stride":
        out = torch.full((2, 64), 123.25)[:, ::2]
    elif case == "out_device":
        out = out.to("meta")
    before = out.clone()
    with pytest.raises(RuntimeError, match="PleInt4Table"):
        table.gather(ids, out)
    if out.device.type != "meta":
        assert torch.equal(out, before)


def test_native_int4_owns_shard_tensor_lifetime(native_int4_class):
    packed, scales = _int4_shards()
    reference = _int4_numpy_reference(packed, scales)
    table = native_int4_class(packed, scales, 3)
    del packed, scales
    gc.collect()
    ids = torch.tensor([5, 0, 3, 2])
    out = torch.empty((4, 32))
    table.gather(ids, out)
    _assert_float_bits(out, reference[ids])


def test_native_int4_tracks_reallocated_tensor_storage(native_int4_class):
    packed, scales = _int4_shards()
    table = native_int4_class(packed, scales, 3)
    for tensor in packed + scales:
        saved = tensor.clone()
        shape = tensor.shape
        tensor.resize_(tensor.numel() + 4096)
        tensor.resize_(shape).copy_(saved)
        if tensor.dtype == torch.uint8:
            tensor.bitwise_xor_(255)
        else:
            tensor.neg_()
    reference = _int4_numpy_reference(packed, scales)
    out = torch.empty((2, 32))
    ids = torch.tensor([5, 0])
    table.gather(ids, out)
    _assert_float_bits(out, reference[ids])


@pytest.mark.parametrize(
    "layout",
    ["group16_int4_fp16scale_lownibblefirst", "int4_g16", "per_row_e4m3"],
)
def test_quant_sidecar_real_constructor_and_mmap_lifetime(
    monkeypatch, tmp_path, layout
):
    from safetensors.torch import save_file

    from vllm.platforms import current_platform
    from vllm.v1.ple_offload.worker import _PleQuantTable

    packed, scales = _int4_shards()
    if layout == "per_row_e4m3":
        packed = [torch.arange(96, dtype=torch.uint8).reshape(3, 32)] * 2
        scales = [torch.tensor([0.5, 1.25, -2.0])] * 2
        key = "weight_fp8"
        reference = torch.cat(packed).view(torch.float8_e4m3fn).float()
        reference *= torch.cat(scales)[:, None]
    else:
        key = "weight_i4"
        reference = _int4_numpy_reference(packed, scales)
    (tmp_path / "META.json").write_text(
        json.dumps({"layout": layout, "rows": 6, "width": 32, "shards": 2})
    )
    for shard in range(2):
        save_file(
            {key: packed[shard], "weight_scale": scales[shard]},
            str(tmp_path / f"shard_{shard}.safetensors"),
        )
    # Only the shard-size constant changes; import, constructor, mmaps, platform
    # selection and native registration are real, with no patched methods.
    monkeypatch.setattr(_PleQuantTable, "ROWS_PER_SHARD", 3)
    table = _PleQuantTable(str(tmp_path), 6, 32)
    uses_native = (
        layout == "group16_int4_fp16scale_lownibblefirst"
        and current_platform.is_rocm()
    )
    assert (table._native_int4 is not None) == uses_native
    ids = np.array([5, 0, 3, 2, 4, 3], dtype=np.int64)
    out = np.full((len(ids), 32), 123.25, dtype=np.float32)
    table.gather_rows_small(ids, out)
    _assert_float_bits(torch.from_numpy(out), reference[torch.from_numpy(ids)])
    if uses_native:
        invalid_ids = ids.copy()
        invalid_ids[-1] = -1
        before = out.copy()
        with pytest.raises(RuntimeError, match="out of bounds"):
            table.gather_rows_small(invalid_ids, out)
        np.testing.assert_array_equal(out.view(np.uint32), before.view(np.uint32))
        native = table._native_int4
        del table
        gc.collect()
        out.fill(0)
        native.gather(torch.from_numpy(ids), torch.from_numpy(out))
        _assert_float_bits(torch.from_numpy(out), reference[torch.from_numpy(ids)])


@pytest.mark.parametrize("mutation", ["shape", "storage"])
def test_native_int4_revalidates_shards_before_writing(native_int4_class, mutation):
    packed, scales = _int4_shards()
    table = native_int4_class(packed, scales, 3)
    if mutation == "shape":
        packed[1].resize_(1, 48)
    else:
        scales[1].untyped_storage().resize_(0)
    out = torch.full((2, 32), 123.25)
    before = out.clone()
    with pytest.raises(RuntimeError, match="PleInt4Table"):
        table.gather(torch.tensor([0, 5]), out)
    assert torch.equal(out, before)


@pytest.mark.parametrize("alias", ["ids", "packed", "scales"])
def test_native_int4_rejects_output_alias_before_writing(native_int4_class, alias):
    packed, scales = _int4_shards()
    out = torch.full((1, 32), 123.25)
    ids = torch.tensor([0])
    if alias == "ids":
        ids = torch.from_numpy(out.numpy().view(np.int64).reshape(-1)[:1])
        ids[0] = 0
    elif alias == "packed":
        packed[1] = torch.from_numpy(
            out.numpy().view(np.uint8).reshape(-1)[:48].reshape(3, 16)
        )
    else:
        scales[1] = torch.from_numpy(
            out.numpy().view(np.float16).reshape(-1)[:6].reshape(3, 2)
        )
    table = native_int4_class(packed, scales, 3)
    before = out.clone()
    with pytest.raises(RuntimeError, match="overlap"):
        table.gather(ids, out)
    assert torch.equal(out, before)


def test_native_int4_accepts_contiguous_storage_offsets(native_int4_class):
    packed, scales = _int4_shards()
    reference = _int4_numpy_reference(packed, scales)
    for shards in (packed, scales):
        for index, tensor in enumerate(shards):
            backing = torch.cat((tensor.new_zeros(7), tensor.flatten()))
            shards[index] = backing[7:].view(tensor.shape)
    table = native_int4_class(packed, scales, 3)
    ids = torch.tensor([99, 5, 0, 99])[1:3]
    backing = torch.full((4, 32), 123.25)
    out = backing[1:3]
    table.gather(ids, out)
    _assert_float_bits(out, reference[ids])
    assert torch.all(backing[0] == 123.25)
    assert torch.all(backing[3] == 123.25)
