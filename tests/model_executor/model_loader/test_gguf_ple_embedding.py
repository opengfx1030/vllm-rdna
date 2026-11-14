# SPDX-License-Identifier: Apache-2.0
import pytest


def test_factory_rejects_unsupported_cpu_sidecar(tmp_path, monkeypatch):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod

    path = tmp_path / "table.bin"
    path.write_bytes(bytes(180))

    class Config:
        gguf_ple_payload = {
            "path": str(path),
            "data_offset": 0,
            "rows": 2,
            "width": 160,
            "type": 20,
        }

        def get_name(self):
            return "gguf"

    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    with pytest.raises(ValueError, match="VLLM_PLE_CPU_OFFLOAD"):
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(Config(), "ple")


def test_factory_selects_gguf_ple_method(tmp_path):
    from vllm.models.qwen4_exp.common.gguf_ple_embedding import (
        Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod,
    )
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod

    path = tmp_path / "payload.bin"
    path.write_bytes(bytes(4 + 2 * 90))

    class Config:
        def get_name(self):
            return "gguf"

        gguf_ple_payload = {
            "path": str(path),
            "data_offset": 4,
            "rows": 2,
            "width": 160,
            "type": 20,
        }

    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(Config(), "ple")
    assert isinstance(method, Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod)
    assert method.payload["width"] == 160


def test_packed_iq4nl_lookup_matches_reference():
    import torch

    from vllm.models.qwen4_exp.common.gguf_ple_embedding import (
        _dequantize_gguf_iq4nl_rows,
    )

    packed = torch.zeros((2, 90), dtype=torch.uint8)
    blocks = packed.reshape(2, 5, 18)
    blocks[:, :, 1] = 0x3C  # little-endian fp16 scale1.0
    blocks[:, :, 2:] = 0xF0
    got = _dequantize_gguf_iq4nl_rows(packed, 160, torch.float32)
    assert torch.equal(got[0, :16], torch.full((16,), -127.0))
    assert torch.equal(got[0, 16:32], torch.full((16,), 113.0))


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_reference_rounds_only_after_float32_decode(dtype_name):
    import numpy as np
    import torch
    from gguf import quants

    from vllm.models.qwen4_exp.common.gguf_ple_embedding import (
        _dequantize_gguf_iq4nl_rows,
    )

    raw = np.zeros((5, 18), np.uint8)
    raw[:, :2] = (
        np.array([0.01, 0.0333, 2**-24, -0.125, 0.75], dtype="<f2")
        .view(np.uint8)
        .reshape(5, 2)
    )
    raw[:, 2:] = np.arange(16, dtype=np.uint8)[None, :] | (
        (15 - np.arange(16, dtype=np.uint8))[None, :] << 4
    )
    dtype = getattr(torch, dtype_name)
    expected = (
        torch.from_numpy(quants.IQ4_NL.dequantize_blocks(raw).copy())
        .reshape(1, 160)
        .to(dtype)
    )
    got = _dequantize_gguf_iq4nl_rows(torch.from_numpy(raw).reshape(1, 90), 160, dtype)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_rank_local_loader_uses_only_owned_rows_and_tiny_placeholder(tmp_path):
    from types import SimpleNamespace

    import torch

    from vllm.models.qwen4_exp.common.gguf_ple_embedding import (
        Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod,
    )

    raw = (
        torch.arange(8 * 90, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(8, 90)
    )
    path = tmp_path / "table.bin"
    path.write_bytes(b"head" + bytes(raw.reshape(-1).tolist()))
    method = Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(
        {"path": str(path), "data_offset": 4, "rows": 8, "width": 160, "type": 20}
    )

    class Layer(torch.nn.Module):
        def allocate_embedding_weight(self, rows, cols, dtype):
            return torch.empty(rows, cols, dtype=dtype)

    layer = Layer()
    layer.org_vocab_size = 8
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=2, org_vocab_end_index=5
    )
    method.CHUNK_BYTES = 27  # exercise chunks crossing packed row/block boundaries
    method.create_weights(layer, 160, [4], 160, 8, torch.float16)
    assert layer.weight_scale.shape == (1, 1)
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight[:3], raw[2:5])
    assert torch.count_nonzero(layer.weight[3:]) == 0


def test_actual_native_device_embedding_cpu_forward(tmp_path):
    import numpy as np
    import torch
    from gguf import quants

    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.qwen4_exp.common.gguf_ple_embedding import (
        Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod,
    )
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEDeviceEmbedding

    raw = np.zeros((8, 5, 18), dtype=np.uint8)
    raw[:, :, :2] = np.array([0.01], dtype="<f2").view(np.uint8)
    raw[:, :, 2:] = np.arange(16, dtype=np.uint8)[None, None, :] | (
        (15 - np.arange(16, dtype=np.uint8))[None, None, :] << 4
    )
    path = tmp_path / "native.bin"
    path.write_bytes(b"head" + raw.tobytes())
    method = Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(
        {"path": str(path), "data_offset": 4, "rows": 8, "width": 160, "type": 20}
    )
    vc = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(mode=0),
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="file://" + str(tmp_path / "gloo"),
        local_rank=0,
        backend="gloo",
    )
    try:
        with set_current_vllm_config(vc):
            initialize_model_parallel(1, 1)
            layer = Qwen4ExpPLEDeviceEmbedding(
                8,
                160,
                params_dtype=torch.float16,
                padding_size=1,
                prefix="test.ple",
                embedding_method=method,
                max_total_tokens=2,
            )
            method.process_weights_after_loading(layer)
            got = layer(torch.tensor([1, 7]))
            expected = (
                torch.from_numpy(
                    quants.IQ4_NL.dequantize_blocks(raw[[1, 7]].reshape(-1, 18)).copy()
                )
                .reshape(2, 160)
                .half()
            )
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_kernel_source_masks_padded_output_lanes():
    # Supplementary source contract only: not GPU execution/qualification.
    import ast
    import inspect

    import vllm.models.qwen4_exp.common.gguf_ple_embedding as module

    tree = ast.parse(inspect.getsource(module))
    stores = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "store"
    ]
    assert stores
    assert all(any(k.arg == "mask" for k in n.keywords) for n in stores)


def test_rejects_incompatible_explicit_dtype(tmp_path):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod

    path = tmp_path / "payload.bin"
    path.write_bytes(bytes(4 + 2 * 90))

    class Config:
        def get_name(self):
            return "gguf"

        gguf_ple_payload = {
            "path": str(path),
            "data_offset": 0,
            "rows": 2,
            "width": 160,
            "type": 20,
        }

    with pytest.raises(ValueError, match="embedding_dtype"):
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(Config(), "ple", "nvfp4")
