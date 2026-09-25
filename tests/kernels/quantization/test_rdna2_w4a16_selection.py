#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for W4A16 kernel selection logic (ROCm).

Run `pytest tests/kernels/quantization/test_rdna2_w4a16_selection.py`.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x
from vllm.scalar_type import scalar_types


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_choose_mp_linear_kernel_picks_triton_w4a16_for_uint4b8():
    # int4 weights, 16-bit activations (CT W4A16 typical config).
    K, N = 1024, 256
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,  # symmetric int4 (bias=8)
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )

    kernel_type = choose_mp_linear_kernel(config)
    # RDNA2 (gfx1030) has a dedicated W4A16 kernel that is preferred over
    # Hybrid and Triton; CDNA falls back to Triton.
    if on_gfx10x():
        assert kernel_type.__name__ == "RDNA2W4A16LinearKernel"
    else:
        assert kernel_type.__name__ == "TritonW4A16LinearKernel"


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_choose_mp_linear_kernel_picks_triton_w4a16_for_uint4_asymmetric():
    # Asymmetric int4 weights should also be supported (explicit zero points).
    K, N = 512, 512
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,  # asymmetric int4 (bias=8)
        act_type=torch.bfloat16,
        group_size=64,
        zero_points=True,
        has_g_idx=False,
    )

    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == "TritonW4A16LinearKernel"


# ---------------------------------------------------------------------------
# Inner 3-bucket dispatcher: which op (decode / prefill / exllama) the
# RDNA2W4A16LinearKernel hands a given (M, K, N) to. Pure-Python test, no
# GPU required.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,K,N,expected",
    [
        # M <= 32, K < 4096 -> prefill
        (1, 128, 128, "prefill"),
        (16, 512, 256, "prefill"),
        (32, 512, 512, "prefill"),
        # M <= 32, K >= 4096 -> rdna2_decode (K-gated)
        (8, 4096, 512, "rdna2_decode"),
        # 32 < M <= 256, N < 3072 -> rdna2_decode (down-proj / attention)
        (64, 1024, 1024, "rdna2_decode"),
        # 32 < M <= 256, N >= 3072 -> exllama (gate/up-proj)
        (128, 1024, 4096, "exllama"),
        # M > 256 -> exllama
        (300, 512, 2048, "exllama"),
        (1024, 1024, 1024, "exllama"),
    ],
)
def test_rdna2_w4a16_inner_dispatch(M, K, N, expected):
    from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
        _rdna2_w4a16_select_kernel,
    )

    assert _rdna2_w4a16_select_kernel(M, K, N) == expected


def test_rocm_registry_keeps_rdna2_ahead_of_hybrid():
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_KERNELS,
        RDNA2W4A16LinearKernel,
        RDNAHybridW4A16LinearKernel,
    )
    from vllm.platforms import PlatformEnum

    kernels = _POSSIBLE_KERNELS[PlatformEnum.ROCM]
    assert kernels.index(RDNA2W4A16LinearKernel) < kernels.index(
        RDNAHybridW4A16LinearKernel
    )


def test_linear_backend_map_rdna2_and_hybrid():
    from vllm.model_executor.kernels.linear import (
        _LINEAR_BACKEND_KERNEL_MAP,
        RDNA2W4A16LinearKernel,
        RDNAHybridW4A16LinearKernel,
    )

    assert RDNA2W4A16LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["rdna2"]
    assert RDNAHybridW4A16LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["rdna_hybrid"]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_linear_backend_rdna_hybrid_forces_hybrid(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_mod
    from vllm.model_executor.kernels.linear.mixed_precision import (
        rdna_hybrid_w4a16 as hybrid_mod,
    )

    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "rdna_hybrid")
    monkeypatch.setattr(hybrid_mod, "_on_gfx1x", lambda: False)
    monkeypatch.setattr(hybrid_mod, "_on_gfx10x", lambda: True)

    config = MPLinearLayerConfig(
        full_weight_shape=(1024, 256),
        partition_weight_shape=(1024, 256),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == "RDNAHybridW4A16LinearKernel"


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_linear_backend_rdna2_forces_rdna2(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_mod
    from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
        RDNA2W4A16LinearKernel,
    )

    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "rdna2")
    monkeypatch.setattr(
        RDNA2W4A16LinearKernel,
        "can_implement",
        classmethod(lambda cls, c: (True, None)),
    )

    config = MPLinearLayerConfig(
        full_weight_shape=(1024, 256),
        partition_weight_shape=(1024, 256),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == "RDNA2W4A16LinearKernel"
