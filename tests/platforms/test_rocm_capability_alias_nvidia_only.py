# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA-only kernel, model and warmup gates must not open on a ROCm gfx1030
(see conftest.py)."""

import importlib
import os
import subprocess
import sys
import types

import pytest

from vllm.platforms import current_platform

SM100_ONLY = [
    pytest.param(False, (10, 3), False, id="gfx1030"),
    pytest.param(True, (10, 0), True, id="sm100"),
]


# -- vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py ----------


@pytest.mark.parametrize(("cuda", "capability", "tma_aligned_scales"), SM100_ONLY)
def test_fused_inv_rope_warmup_tma_scales_are_cuda_only(
    patch_platform, cuda, capability, tma_aligned_scales
):
    patch_platform(cuda, capability)
    module = importlib.import_module(
        "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant"
    )
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                num_attention_heads=64,
                o_groups=8,
                head_dim=512,
                qk_rope_head_dim=64,
            )
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
    )
    keys = module.FusedInvRopeFP8QuantKernel().get_warmup_keys(vllm_config)
    assert keys
    assert {key.tma_aligned_scales for key in keys} == {tma_aligned_scales}


# -- vllm/models/flint/model.py ----------------------------------------------


@pytest.mark.parametrize(("cuda", "capability", "expected"), SM100_ONLY)
def test_flint_kernels_are_sm100_cuda_only(
    patch_platform, monkeypatch, cuda, capability, expected
):
    patch_platform(cuda, capability)
    flint = importlib.import_module("vllm.models.flint.model")
    monkeypatch.setattr(flint, "get_tensor_model_parallel_world_size", lambda: 1)
    assert flint._flint_kernels(None) is expected


# -- vllm/models/deepseek_v41/attention.py -----------------------------------


@pytest.mark.parametrize(("cuda", "capability", "expected"), SM100_ONLY)
def test_deepseek_v41_mxfp8_kv_record_is_sm100_cuda_only(
    patch_platform, cuda, capability, expected
):
    patch_platform(cuda, capability)
    attention = importlib.import_module("vllm.models.deepseek_v41.attention")
    assert attention._use_v41_mxfp8_kv_record() is expected


# -- vllm/model_executor/layers/quantization/online/nvfp4.py -----------------


def test_nvfp4_per_token_online_moe_rejects_gfx1030(gfx1030):
    nvfp4 = importlib.import_module(
        "vllm.model_executor.layers.quantization.online.nvfp4"
    )
    with pytest.raises(ValueError, match="requires a Blackwell"):
        nvfp4.Nvfp4OnlineMoEMethod(moe=object())


def test_nvfp4_per_token_online_moe_capability_check_passes_on_sm100(sm100):
    nvfp4 = importlib.import_module(
        "vllm.model_executor.layers.quantization.online.nvfp4"
    )
    # The capability check is the first statement of __init__: on a B200 the
    # constructor goes on and trips over the stand-in config instead.
    try:
        nvfp4.Nvfp4OnlineMoEMethod(moe=object())
    except Exception as exc:
        assert "requires a Blackwell" not in str(exc)


# -- vllm/model_executor/layers/fused_moe/flashinfer_moe_ep.py ---------------


@pytest.mark.parametrize(
    ("cuda", "capability", "expected"),
    [
        pytest.param(False, (10, 3), False, id="gfx1030"),
        pytest.param(True, (10, 0), True, id="sm100"),
        pytest.param(True, (10, 3), True, id="sm103"),
        pytest.param(True, (12, 0), False, id="sm120"),
    ],
)
def test_flashinfer_moe_ep_supports_current_device_is_cuda_only(
    patch_platform, cuda, capability, expected
):
    patch_platform(cuda, capability)
    moe_ep = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.flashinfer_moe_ep"
    )
    assert moe_ep.supports_current_device() is expected


# -- vllm/kernels/oink_ops.py: no device query on import ---------------------


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_importing_vllm_kernels_needs_no_gpu_on_rocm():
    """OINK_AVAILABLE used to call has_device_capability(100) at import, which
    initialises the device (and raises with every GPU hidden) on ROCm."""
    env = dict(os.environ, HIP_VISIBLE_DEVICES="")
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm.kernels.oink_ops as m; assert m.OINK_AVAILABLE is False",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
