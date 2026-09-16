# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP-side HyperConnection (HC) prefill kernels for Qwen4Exp / Qwen3.8-Flash-Next on gfx1030.

Opt-in replacement for the Triton kernels in
``vllm/models/qwen4_exp/amd/ops/hc.py``. Each function here mirrors the
Triton ``_grouped_gemma_rmsnorm_kernel`` / ``_hc_silu_kernel`` /
``_hc_gate_mix_kernel`` / ``_hc_combine_kernel`` /
``_hc_combine_norm_kernel`` semantics and is registered as a torch.ops
binding via ``csrc/rocm/torch_bindings.cpp`` under
``hc_*_rdna2``.

Gated by ``VLLM_RDNA_HC_PREFILL_HIP=1`` and ``on_gfx10x()``. Default off
(Triton path stays the source of truth until the HIP path is verified
end-to-end on gfx1030).

The dispatcher in ``ops/hc.py`` imports these functions and routes to
them when the env-var gate is set; otherwise it falls through to the
existing Triton kernels unchanged.
"""

import torch

from vllm import _custom_ops as ops
from vllm.platforms.rocm import on_gfx10x

from vllm.envs import VLLM_RDNA_HC_PREFILL_HIP


def hc_use_rdna2() -> bool:
    """True iff the HIP HC prefill path is enabled for this process."""
    return bool(VLLM_RDNA_HC_PREFILL_HIP) and on_gfx10x()


def _contig(t: torch.Tensor) -> torch.Tensor:
    """The HIP kernels require contiguous fp16, but callers pass strided views
    (e.g. a slice of a split()). The Triton kernels take explicit strides, so
    the port has to normalise here."""
    return t if t.is_contiguous() else t.contiguous()


# ---------------------------------------------------------------------------
# HIP-side implementations. Each function allocates its own output tensors
# (mirroring the Triton helpers' ``new_empty`` behaviour) and forwards to
# the matching ``_rocm_C::hc_*_rdna2`` op.
# ---------------------------------------------------------------------------


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """``y = x * rsqrt(sum(x*x) / GROUP_DIM + eps) * (1 + w)``.

    Faithful port of ``_grouped_gemma_rmsnorm_kernel``. The shared-vs-
    per-stream weight selection (``W_SHARED`` in the Triton kernel) is
    derived from ``weight.numel()`` in the host wrapper.
    """
    y = x.new_zeros(x.shape)
    ops.hc_grouped_gemma_rmsnorm_rdna2(
        _contig(x), _contig(weight), y, num_groups, float(eps)
    )
    return y


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``y = (x / HC) * sigmoid(x / HC)``."""
    y = x.new_zeros(x.shape)
    ops.hc_silu_rdna2(_contig(x), y, hc_count)
    return y


def hc_gate_mix(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    """``out[h] = (1/HC) * sum_c sigmoid(g[c*H+h]) * x[c*H+h]``."""
    N, DIM = x.shape
    HC_DIM = DIM // hc_count
    y = x.new_zeros((N, HC_DIM))
    ops.hc_gate_mix_rdna2(_contig(x), _contig(gate), y, hc_count)
    return y


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """``res[c,h] = res[c,h] + block[h] * 2 * sigmoid(inj[c] / HC)``."""
    out = residual.new_zeros(residual.shape)
    ops.hc_combine_rdna2(
        _contig(residual),
        _contig(block_output),
        _contig(injection_logits),
        out,
        hc_count,
    )
    return out


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused combine + grouped Gemma RMSNorm.

    Returns ``(out, y)`` where ``out`` is the combined residual and ``y``
    is the post-norm tensor. The HIP wrapper matches the Triton
    ``_hc_combine_norm_kernel`` exactly: ``out`` is rounded to fp16 to
    match the unfused combine -> RMSNorm boundary.
    """
    out = residual.new_zeros(residual.shape)
    y = residual.new_zeros(residual.shape)
    ops.hc_combine_norm_rdna2(
        _contig(residual),
        _contig(block_output),
        _contig(injection_logits),
        _contig(norm_weight),
        out,
        y,
        hc_count,
        float(eps),
    )
    return out, y


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
    "hc_use_rdna2",
]
