# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the RidgeFill map ops (csrc/rocm/ridgefill_map_rdna2.cu).

``ridgefill_map_rdna2`` (fp16, dot2) must match an fp32 GEMM of the same fp16
inputs to fp16 rounding. ``ridgefill_map_i8_rdna2`` (dot4) must match a torch
emulation of its own arithmetic (Hadamard-128 rotation of x, per-row int8,
exact integer accumulation) and stay within int8 error of the fp32 product
on maps with outlier columns, which plain per-row int8 does not survive.
Rows cover the M tail of the 128-row block tile and a full prefill chunk.
"""

import pytest
import torch

import vllm._custom_ops  # noqa: F401  (loads _rocm_C)
from vllm.model_executor.layers.ridgefill import ops as rf_ops
from vllm.model_executor.layers.ridgefill.runtime import hadamard, quantize_maps_i8
from vllm.platforms.rdna import on_rdna2

pytestmark = pytest.mark.skipif(
    not on_rdna2() or not rf_ops.has_map_i8_op(),
    reason="RDNA2 (gfx1030) RidgeFill map ops only",
)

DIN, DOUT, LAYERS = 1024, 256, 3


def _data(rows: int, outliers: bool):
    gen = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, DIN, generator=gen)
    w = torch.randn(LAYERS * DOUT, DIN, generator=gen) * 0.02
    if outliers:
        w[:, ::97] *= 50.0  # the fitted maps' outlier columns
    b = torch.randn(LAYERS * DOUT, generator=gen) * 0.1
    return x.half().cuda(), w.half().cuda(), b.half().cuda()


def _layout(y: torch.Tensor, rows: int) -> torch.Tensor:
    """[rows, L * DOUT] -> the op's [L, rows, DOUT]."""
    return y.view(rows, LAYERS, DOUT).transpose(0, 1)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).square().mean().sqrt() / want.square().mean().sqrt())


@pytest.mark.parametrize("rows", [1, 100, 129, 2048])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fp16_map_matches_fp32_gemm(rows: int, with_bias: bool) -> None:
    x, w, b = _data(rows, outliers=False)
    bias = b if with_bias else None
    out = torch.full(
        (LAYERS, rows, DOUT), float("nan"), dtype=torch.half, device="cuda"
    )
    rf_ops.ridgefill_map(x, w, bias, out, DOUT, LAYERS)
    want = x.float() @ w.float().T
    if with_bias:
        want = want + b.float()
    torch.testing.assert_close(out.float(), _layout(want, rows), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("rows", [1, 129, 2048])
def test_i8_map_matches_its_emulation_and_the_fp32_product(rows: int) -> None:
    x, w, b = _data(rows, outliers=True)
    wq, sw = quantize_maps_i8(
        [w[i * DOUT : (i + 1) * DOUT] for i in range(LAYERS)], x.device
    )
    out = torch.full(
        (LAYERS, rows, DOUT), float("nan"), dtype=torch.half, device="cuda"
    )
    rf_ops.ridgefill_map_i8(x, wq, sw, b, out, DOUT, LAYERS)
    assert bool(torch.isfinite(out).all())

    h = hadamard(128, x.device)
    xr = (x.float().view(rows, DIN // 128, 128) @ h).view(rows, DIN)
    sx = (xr.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    xq = torch.round(xr / sx[:, None]).clamp(-127, 127)
    acc = xq.double() @ wq.double().T  # exact: the op accumulates in int32
    emu = acc.float() * sx[:, None] * sw[None, :] + b.float()
    # The op rotates with fp32 butterflies, torch with a matmul: a rounding
    # tie may move one int8 step, so compare by relative RMS, not bitwise.
    assert _rel(out.float(), _layout(emu, rows)) < 2e-3

    want = x.float() @ w.float().T + b.float()
    assert _rel(out.float(), _layout(want, rows)) < 0.03
