# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP-side PLE dilated short-conv (depthwise) for Qwen4Exp on gfx1030.

Opt-in replacement for the ``F.conv1d`` path used by
``Qwen4ExpPLELayer._short_conv_dilated_decode_batched`` and
``_short_conv_dilated_prefill_batched`` in
``vllm/models/qwen4_exp/amd/ple_layer.py``.

Two host wrappers:
  * ``ple_short_conv_decode`` (decode path: one new token per request)
  * ``ple_short_conv_prefill`` (prefill path: variable-length batches)

Gated by ``VLLM_RDNA_PLE_CONV_HIP=1`` and ``on_gfx10x()``. Default off
(torch ``F.conv1d`` fallback).
"""

import torch

from vllm import _custom_ops as ops
from vllm.platforms.rocm import on_gfx10x

from vllm.envs import VLLM_RDNA_PLE_CONV_HIP


def ple_conv_use_rdna2() -> bool:
    """True iff the HIP PLE dilated short-conv path is enabled."""
    return bool(VLLM_RDNA_PLE_CONV_HIP) and on_gfx10x()


# ---------------------------------------------------------------------------
# HIP-side implementations.
# ---------------------------------------------------------------------------


def ple_short_conv_decode(
    x: torch.Tensor,                  # [B, D] fp16
    conv_state: torch.Tensor,         # [num_lines, D, state_len] fp16 in-place
    weight: torch.Tensor,             # [D, K] fp16
    out: torch.Tensor,                # [B, D] fp16
    dilation: int,
    state_len: int,
    silu: bool = True,
    bias: torch.Tensor | None = None,    # [D] fp16 or None
    state_idx: torch.Tensor | None = None,  # [B] int32
    has_init: torch.Tensor | None = None,   # [B] uint8 or None
    null_block: int = -1,
) -> torch.Tensor:
    """Depthwise dilated short-conv + state-shift for the decode path.

    Mirrors ``_short_conv_dilated_decode_batched`` in
    ``vllm/models/qwen4_exp/amd/ple_layer.py``. Validates inputs and
    forwards to ``ops.ple_short_conv_decode_rdna2``.
    """
    if bias is None:
        bias = torch.empty(0, dtype=x.dtype, device=x.device)
    if state_idx is None:
        # Auto-generate identity state indices (one row per request).
        state_idx = torch.arange(
            x.shape[0], dtype=torch.int32, device=x.device
        )
    if has_init is None:
        has_init = torch.ones(x.shape[0], dtype=torch.uint8, device=x.device)
    ops.ple_short_conv_decode_rdna2(
        x.contiguous(), conv_state,
        weight, bias,
        out,
        state_idx.to(torch.int32),
        has_init.to(torch.uint8),
        int(dilation), int(state_len), bool(silu), int(null_block),
    )
    return out


def ple_short_conv_prefill(
    x_packed: torch.Tensor,           # [B, D, max_len] fp16
    init_state: torch.Tensor,         # [B, D, state_len] fp16
    weight: torch.Tensor,             # [D, K] fp16
    out: torch.Tensor,                # [B, D, max_len] fp16
    lengths: torch.Tensor,            # [B] int32
    dilation: int,
    state_len: int,
    silu: bool = True,
    bias: torch.Tensor | None = None,
    valid_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise dilated short-conv for the prefill path.

    Mirrors ``_short_conv_dilated_prefill_batched``. The padding tokens
    beyond each request's true length are zeroed on output via
    ``lengths`` (no extra validity tensor required for prefill).
    """
    if bias is None:
        bias = torch.empty(0, dtype=x_packed.dtype, device=x_packed.device)
    if valid_state is None:
        valid_state = torch.ones(
            lengths.shape[0], dtype=torch.uint8, device=x_packed.device
        )
    ops.ple_short_conv_prefill_rdna2(
        x_packed.contiguous(), init_state.contiguous(),
        weight, bias, out,
        lengths.to(torch.int32),
        valid_state.to(torch.uint8),
        int(dilation), int(state_len), bool(silu),
    )
    return out


__all__ = [
    "ple_conv_use_rdna2",
    "ple_short_conv_decode",
    "ple_short_conv_prefill",
]
