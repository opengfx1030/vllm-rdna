# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused HC prefill mix on gfx1030 (csrc/rocm/hc_prefill_rdna2.cu) against the
torch prefill path of ``rdna_hc_mix`` (rocBLAS + Triton silu/gate mix)."""

import pytest
import torch

from vllm.platforms import current_platform


def _gfx10x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


pytestmark = pytest.mark.skipif(not _gfx10x(), reason="gfx1030 HIP kernels")

HC = 4
HIDDEN = 2560
LORA = 320
DOWN_N = LORA + HC + 12  # merged down+inject, padded to 16 rows
M_VALUES = [9, 17, 63, 64, 65, 100, 128, 200, 255, 256, 384, 513, 1030, 2048]


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    w_down = (torch.randn(DOWN_N, HC * HIDDEN, device="cuda") * 0.02).half()
    w_up = (torch.randn(HC * HIDDEN, LORA, device="cuda") * 0.05).half()
    return w_down, w_up


def _mix(xn, w_down, w_up, mode, monkeypatch):
    from vllm.model_executor.layers import rdna_ops

    monkeypatch.setattr(rdna_ops, "_HC_PREFILL_FUSED", mode)
    return rdna_ops._rdna_hc_mix(
        xn, w_down, None, None, w_up, None, None, LORA, HC
    )


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("m", M_VALUES)
def test_fused_prefill_matches_torch(weights, mode, m, monkeypatch):
    w_down, w_up = weights
    xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
    ref_out, ref_dai = _mix(xn, w_down, w_up, 0, monkeypatch)
    out, dai = _mix(xn, w_down, w_up, mode, monkeypatch)
    assert out.shape == ref_out.shape and dai.shape == ref_dai.shape
    assert dai.dtype == torch.float16 and out.dtype == torch.float16
    # mode 1 shares the rocBLAS down GEMM; mode 2 accumulates it in fp32
    # (split-K) and rounds once, so dai may differ by an fp16 ulp.
    torch.testing.assert_close(dai, ref_dai, atol=4e-3, rtol=2e-3)
    torch.testing.assert_close(out, ref_out, atol=3e-3, rtol=1e-2)


def test_decode_unchanged(weights, monkeypatch):
    w_down, w_up = weights
    xn = torch.randn(4, HC * HIDDEN, device="cuda").half()
    ref = _mix(xn, w_down, w_up, 0, monkeypatch)
    for mode in (1, 2):
        got = _mix(xn, w_down, w_up, mode, monkeypatch)
        torch.testing.assert_close(got[0], ref[0], atol=0, rtol=0)
        torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0)


def test_non_contiguous_falls_back(weights, monkeypatch):
    w_down, w_up = weights
    big = torch.randn(100, 2 * HC * HIDDEN, device="cuda").half()
    xn = big[:, : HC * HIDDEN]
    assert not xn.is_contiguous()
    ref = _mix(xn.contiguous(), w_down, w_up, 0, monkeypatch)
    got = _mix(xn, w_down, w_up, 2, monkeypatch)
    torch.testing.assert_close(got[0], ref[0], atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("mode", [1, 2])
def test_cuda_graph_replay(weights, mode, monkeypatch):
    """Fresh outputs on the current stream: capture once, replay on new data."""
    w_down, w_up = weights
    m = 300
    xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
    _mix(xn, w_down, w_up, mode, monkeypatch)  # warm up outside capture
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, dai = _mix(xn, w_down, w_up, mode, monkeypatch)
    for _ in range(3):
        xn.copy_(torch.randn_like(xn))
        g.replay()
        ref_out, ref_dai = _mix(xn.clone(), w_down, w_up, 0, monkeypatch)
        torch.testing.assert_close(dai, ref_dai, atol=4e-3, rtol=2e-3)
        torch.testing.assert_close(out, ref_out, atol=3e-3, rtol=1e-2)


def test_op_checks_shapes(weights):
    import vllm._custom_ops as ops

    w_down, w_up = weights
    xn = torch.randn(16, HC * HIDDEN, device="cuda").half()
    with pytest.raises(RuntimeError):
        ops.rdna_hc_mix_prefill(xn, w_down, w_up, LORA, 3)
    with pytest.raises(RuntimeError):
        ops.rdna_hc_up_gate_mix_prefill(
            torch.empty(16, DOWN_N, device="cuda"), w_up, xn, LORA, HC
        )
