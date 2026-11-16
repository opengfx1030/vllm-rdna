# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils


def _on_gfx11() -> bool:
    """gfx11 check that is safe to evaluate at collection time on any platform."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx11

    return on_gfx11()


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch, m):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(m, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", range(1, 6))
def test_rocm_unquantized_gemm_single_output_real_kernel(dtype, n):
    torch.manual_seed(0)
    x = torch.randn(n, 2048, device="cuda", dtype=dtype)
    weight = torch.randn(1, 2048, device="cuda", dtype=dtype)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def _row_stride_padded(weight: torch.Tensor, pad: int) -> torch.Tensor:
    """A weight whose row stride is offset by ``pad`` elements, as produced by
    ``maybe_pad_weight_avoid_cache_aliasing``."""
    padded = torch.nn.functional.pad(weight, (0, pad))[..., :-pad]
    assert padded.shape == weight.shape
    assert padded.stride() == (weight.shape[1] + pad, 1)
    return padded


def test_rocm_unquantized_gemm_wvsplitk_accepts_row_stride_padded_weight(monkeypatch):
    # wvSplitK reads the weight row stride, so a row-stride-padded weight must
    # still reach it instead of falling back to torch.nn.functional.linear.
    x = torch.randn(4, 1024, dtype=torch.float16)
    weight = _row_stride_padded(torch.randn(128, 1024, dtype=torch.float16), 64)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    assert wvsplitk_mock.call_args.args[0].stride() == weight.stride()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "k,element_size,expected",
    [
        (4096, 2, 64),  # 8192 B row stride, aliases
        (12288, 2, 64),  # 24576 B row stride, aliases
        (1024, 2, 64),  # smallest aliasing bf16/fp16 K
        (2560, 2, 0),  # 5120 B, no aliasing
        (11008, 2, 0),  # 22016 B, no aliasing
        (2048, 1, 128),  # fp8: 2048 B, aliases, twice the pad
        (512, 4, 32),  # fp32: 2048 B, aliases, half the pad
    ],
)
def test_cache_aliasing_pad_elems(k, element_size, expected):
    assert utils.cache_aliasing_pad_elems(k, element_size) == expected


def test_maybe_pad_weight_avoid_cache_aliasing_skips_non_gfx11(monkeypatch):
    # Padding is gfx11-only, even for a K whose row stride would alias there.
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: False)

    weight = torch.randn(128, 4096, dtype=torch.bfloat16)
    assert utils.maybe_pad_weight_avoid_cache_aliasing(weight) is weight


def test_maybe_pad_weight_avoid_cache_aliasing(monkeypatch):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: True)

    weight = torch.randn(128, 4096, dtype=torch.bfloat16)
    padded = utils.maybe_pad_weight_avoid_cache_aliasing(weight)

    assert padded.shape == weight.shape
    assert padded.stride() == (4096 + 64, 1)
    assert torch.equal(padded, weight)
    # Idempotent: the padded stride no longer aliases, so a second call is a no-op
    # and preserves the pointer captured by any graph.
    repadded = utils.maybe_pad_weight_avoid_cache_aliasing(padded)
    assert repadded.data_ptr() == padded.data_ptr()


@pytest.mark.parametrize(
    "weight",
    [
        torch.randn(128, 2560, dtype=torch.bfloat16),  # no aliasing
        torch.randn(2560, 128, dtype=torch.bfloat16).t(),  # not contiguous
        torch.randn(2, 128, 4096, dtype=torch.bfloat16),  # not 2D
    ],
)
def test_maybe_pad_weight_avoid_cache_aliasing_skips(monkeypatch, weight):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: True)

    assert (
        utils.maybe_pad_weight_avoid_cache_aliasing(weight).data_ptr()
        == weight.data_ptr()
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not _on_gfx11(), reason="cache-aliasing padding only applies on gfx11"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_padded_weight_real_kernel(monkeypatch, dtype):
    x = torch.randn(4, 4096, device="cuda", dtype=dtype)
    weight = _row_stride_padded(torch.randn(128, 4096, device="cuda", dtype=dtype), 64)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back(monkeypatch, m):
    # wvSplitK skinny GEMM handles n in [1, 5] (see PR #40687); n > 5 must
    # fall back to torch.nn.functional.linear.
    x = torch.randn(6, 64, dtype=torch.float16)
    weight = torch.randn(m, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


class _LMHead(torch.nn.Module):
    """Stands in for ParallelLMHead: the dispatch matches the class name."""


_LMHead.__name__ = "ParallelLMHead"


@pytest.mark.parametrize(
    ("cap", "is_head", "expected"),
    [(32, True, 32), (32, False, 8), (8, True, 8), (16, True, 16)],
)
def test_gemv_rows_for_layer(monkeypatch, cap, is_head, expected):
    monkeypatch.setattr(utils, "_RDNA_LM_HEAD_GEMV_MAX_M", cap)
    layer = _LMHead() if is_head else torch.nn.Linear(4, 4)
    assert utils._gemv_rows_for_layer(layer) == expected


def _patch_gfx10x_dispatch(monkeypatch):
    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: True)
    for name in ("on_gfx1x", "on_gfx9", "on_gfx11", "on_gfx950", "on_gfx1250"):
        monkeypatch.setattr(f"vllm.platforms.rocm.{name}", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)


@pytest.mark.parametrize("rows", [8, 9, 12, 32, 33])
@pytest.mark.parametrize("cap", [8, 32])
def test_rocm_unquantized_gemm_gfx10x_gemv_row_cap(monkeypatch, rows, cap):
    """gemv_f16_rdna2 serves up to the row cap the caller passes (8 by default)."""
    _patch_gfx10x_dispatch(monkeypatch)
    gemv = MagicMock(side_effect=lambda x, w, b: x @ w.t())
    monkeypatch.setattr(utils.ops, "gemv_f16_rdna2", gemv, raising=False)
    x = torch.randn(rows, 64, dtype=torch.float16)
    weight = torch.randn(96, 64, dtype=torch.float16)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None, cap)

    assert gemv.call_count == (1 if rows <= cap else 0)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, weight), atol=2e-2, rtol=2e-2
    )


def test_rocm_unquantized_gemm_gfx10x_default_cap_is_eight(monkeypatch):
    """Callers that do not pass a cap (every layer but the LM head) keep 8 rows."""
    _patch_gfx10x_dispatch(monkeypatch)
    gemv = MagicMock(side_effect=lambda x, w, b: x @ w.t())
    monkeypatch.setattr(utils.ops, "gemv_f16_rdna2", gemv, raising=False)
    weight = torch.randn(96, 64, dtype=torch.float16)
    utils.rocm_unquantized_gemm_impl(torch.randn(12, 64, dtype=torch.float16), weight)
    assert gemv.call_count == 0
    utils.rocm_unquantized_gemm_impl(torch.randn(8, 64, dtype=torch.float16), weight)
    assert gemv.call_count == 1


def _on_gfx10x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


@pytest.mark.skipif(not _on_gfx10x(), reason="gfx10x-only kernel test")
@pytest.mark.parametrize("rows", [9, 12, 16, 24, 32])
def test_rocm_unquantized_gemm_gfx10x_lm_head_gemv_real_kernel(rows):
    """The dense GEMV matches linear at the verify row counts (9..32) it now serves."""
    torch.manual_seed(0)
    x = torch.randn(rows, 512, dtype=torch.float16, device="cuda")
    weight = torch.randn(62080, 512, dtype=torch.float16, device="cuda") * 0.05
    out = utils.rocm_unquantized_gemm_impl(x, weight, None, 32)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(x, weight), atol=2e-2, rtol=2e-2
    )
