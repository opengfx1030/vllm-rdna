# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-level SM100 checks (default CUDA graph ceiling, DeepGemm exclusion)
must not take the Blackwell branch on a ROCm gfx1030 (see conftest.py)."""

import types

import pytest

from vllm.platforms import current_platform


def test_gfx1030_is_indistinguishable_from_sm103_by_capability(gfx1030):
    # The premise of every test_rocm_capability_alias_* test. If this ever stops
    # holding the guards are redundant, not wrong.
    assert current_platform.is_device_capability_family(100)
    assert current_platform.has_device_capability(100)
    assert current_platform.is_device_capability(103)
    assert not current_platform.is_device_capability(100)


# -- vllm/config/vllm.py: default CUDA graph capture ceiling -----------------


def _cudagraph_sizes(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int = 8192,
    sizes: list[int] | None = None,
    query_len: int = 1,
):
    """Run VllmConfig._set_cudagraph_sizes on a stand-in with just the fields
    it reads, and return the resulting compilation config."""
    from vllm.config import VllmConfig
    from vllm.config.compilation import CUDAGraphMode

    compilation_config = types.SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        max_cudagraph_capture_size=None,
        cudagraph_capture_sizes=sizes,
        post_init_cudagraph_sizes=lambda: None,
    )
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        compilation_config=compilation_config,
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        speculative_config=None,
        num_speculative_tokens=query_len - 1,
        uniform_decode_query_len=query_len,
        performance_mode="balanced",
    )
    VllmConfig._set_cudagraph_sizes(config)
    return compilation_config


@pytest.mark.parametrize(
    ("cuda", "capability", "ceiling"),
    [
        pytest.param(False, (10, 3), 512, id="gfx1030"),
        pytest.param(True, (10, 0), 1024, id="sm100"),
        pytest.param(True, (9, 0), 512, id="sm90"),
    ],
)
def test_default_capture_ceiling_is_blackwell_only(
    patch_platform, cuda, capability, ceiling
):
    # 1024 is the Blackwell ceiling; it needs more than 256 sequences to show.
    patch_platform(cuda, capability)
    compilation_config = _cudagraph_sizes(max_num_seqs=1024)
    assert compilation_config.max_cudagraph_capture_size == ceiling
    assert max(compilation_config.cudagraph_capture_sizes) == ceiling


@pytest.mark.parametrize(
    ("max_num_seqs", "sizes", "query_len"),
    [
        # flashnext-mtp2: --max-num-seqs 8, MTP 2 (query length 3),
        # cudagraph_capture_sizes [3, 6, 12, 24].
        (8, [3, 6, 12, 24], 3),
        # 27b-awq: --max-num-seqs 8, no MTP, cudagraph_capture_sizes [1, 2, 4, 8].
        (8, [1, 2, 4, 8], 1),
    ],
    ids=["flashnext-mtp2", "27b-awq"],
)
def test_recipe_capture_sizes_do_not_depend_on_the_platform_ceiling(
    gfx1030, max_num_seqs, sizes, query_len
):
    # Both production recipes pass explicit sizes and a max_num_seqs far below
    # 256, so the 512/1024 default never decides their graph set: guarding it
    # changes nothing for them.
    compilation_config = _cudagraph_sizes(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2048,
        sizes=list(sizes),
        query_len=query_len,
    )
    assert compilation_config.cudagraph_capture_sizes == sizes
    assert compilation_config.max_cudagraph_capture_size == max(sizes)


# -- vllm/utils/deep_gemm.py: Blackwell exclusion table ----------------------


@pytest.mark.parametrize(
    ("model_type", "cuda", "capability", "expected"),
    [
        pytest.param("qwen3_5_text", False, (10, 3), False, id="qwen3_5-gfx1030"),
        pytest.param(
            "qwen3_5_moe_text", False, (10, 3), False, id="qwen3_5_moe-gfx1030"
        ),
        pytest.param("qwen3_5_moe_text", True, (10, 0), True, id="qwen3_5_moe-sm100"),
        pytest.param("qwen3_5_text", True, (10, 3), True, id="qwen3_5-sm103"),
        pytest.param("qwen3_5_text", True, (12, 0), True, id="qwen3_5-sm120"),
        pytest.param("kolibri1", True, (12, 0), True, id="kolibri1-sm120"),
        pytest.param("kolibri1", True, (10, 0), False, id="kolibri1-sm100"),
        pytest.param("qwen3_5_text", True, (9, 0), False, id="qwen3_5-sm90"),
    ],
)
def test_deepgemm_blackwell_exclusion_is_cuda_only(
    patch_platform, model_type, cuda, capability, expected
):
    from vllm.utils.deep_gemm import should_auto_disable_deep_gemm

    patch_platform(cuda, capability)
    assert should_auto_disable_deep_gemm(model_type) is expected
