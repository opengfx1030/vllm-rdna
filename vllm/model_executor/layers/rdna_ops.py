# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T46: opaque custom ops with *runtime* decode/prefill dispatch for gfx1030.

torch.compile traces the model once for a dynamic token range, so a Python
branch like `if 0 < n <= 8` is decided at trace time and the decode kernels
never run inside the compiled graph (boot 6/8 of T45: only the hyper-connection
linears, which sit outside the traced region, took the int8 path). Wrapping the
decision in a custom op makes it a runtime choice on the real batch size.

  rdna_dense_gemm   int8-shadow GEMV for decode, fp16 rocBLAS for prefill
  rdna_hc_mix       hyper-connection mix: 2 fused kernels for decode; prefill
                     runs fused HIP kernels (VLLM_RDNA_HC_PREFILL_FUSED=0: torch)
  rdna_shared_expert shared expert (gate_up+silu*mul, down*sigmoid(gate)):
                     2 kernels / torch
"""

import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_DECODE_MAX = 8


def _ntok(x: torch.Tensor) -> int:
    return x.numel() // x.size(-1)


# ---------------------------------------------------------------- dense int8 shadow
def _rdna_dense_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_i8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    n = _ntok(x)
    if 0 < n <= _DECODE_MAX and x.dtype == torch.float16:
        from vllm import _custom_ops as ops

        x2 = x.reshape(-1, x.size(-1)).contiguous()
        out = ops.gemv_i8_rdna2(x2, weight_i8, scale, bias)
        return out.reshape(*x.shape[:-1], weight_i8.shape[0])
    from vllm.model_executor.layers.rdna_dense_int8 import is_released, linear_released

    if is_released(weight):
        return linear_released(x, weight_i8, scale, bias)
    return F.linear(x, weight, bias)


def _rdna_dense_gemm_fake(x, weight, weight_i8, scale, bias):
    return x.new_empty((*x.shape[:-1], weight_i8.shape[0]))


# ---------------------------------------------------------------- hyper-connection mix
def _rdna_hc_mix(
    xn: torch.Tensor,
    w_down: torch.Tensor,
    w_down_i8: torch.Tensor | None,
    s_down: torch.Tensor | None,
    w_up: torch.Tensor,
    w_up_i8: torch.Tensor | None,
    s_up: torch.Tensor | None,
    lora_rank: int,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (block_input [M, H], down_and_injection [M, N_down])."""
    n = _ntok(xn)
    if 0 < n <= _DECODE_MAX and xn.dtype == torch.float16 and xn.is_contiguous():
        from vllm import _custom_ops as ops

        wd, sd = (w_down_i8, s_down) if w_down_i8 is not None else (w_down, None)
        wu, su = (w_up_i8, s_up) if w_up_i8 is not None else (w_up, None)
        dai = ops.rdna_gemv_act(xn, wd, sd, lora_rank, 1.0 / hc_count)
        lora = dai[:, :lora_rank].contiguous()
        block_input = ops.rdna_hc_up_gate_mix(lora, wu, su, xn, hc_count)
        return block_input, dai
    # prefill / fallback: the original op sequence
    from vllm.model_executor.layers.rdna_dense_int8 import weight_for_gemm
    from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu

    w_down = weight_for_gemm(w_down, w_down_i8, s_down)
    w_up = weight_for_gemm(w_up, w_up_i8, s_up)
    if _HC_PREFILL_FUSED and _hc_fused_prefill_ok(
        xn, w_down, w_up, lora_rank, hc_count
    ):
        from vllm import _custom_ops as ops

        if _HC_PREFILL_FUSED == 2:
            return ops.rdna_hc_mix_prefill(xn, w_down, w_up, lora_rank, hc_count)
        dai = _linear_padded_m(xn, w_down)
        block_input = ops.rdna_hc_up_gate_mix_prefill(
            dai, w_up, xn, lora_rank, hc_count
        )
        return block_input, dai
    dai = _linear_padded_m(xn, w_down)
    lora = hc_silu(dai[:, :lora_rank].contiguous(), hc_count)
    gate = _linear_padded_m(lora, w_up)
    block_input = hc_gate_mix(xn, gate, hc_count)
    return block_input, dai


# Pad the GEMM row count (tokens) of the prefill HC GEMMs to a multiple of
# this. TunableOp rows are keyed by the exact row count, and a mixed
# prefill/decode step has an arbitrary one (chunk + decode tokens). Off a
# tuned row, rocBLAS's heuristic runs the skinny down GEMM (N=336, K=10240) at
# ~4 TF/s (0.89 ms at M=512 vs 0.21 ms tuned) -- ~60 ms per 512-token step
# over 97 HC mixes. Padding maps every row count onto the 64-step rows shipped
# in tunableop/<profile>/. 0 disables.
_HC_PAD_M = int(os.environ.get("VLLM_RDNA_HC_PAD_M", "64") or 0)


# Prefill (M > 8) HC mix kernels (csrc/rocm/hc_prefill_rdna2.cu):
#   0  torch: rocBLAS down, Triton silu, rocBLAS up writing the [M, 4H] gate,
#      Triton gate mix
#   1  rocBLAS down + one fused silu/up-GEMM/sigmoid/mix kernel (no gate tensor)
#   2  split-K HIP down GEMM + the fused kernel (default; the ops exist only in
#      gfx1030 builds, elsewhere this falls back to 0)
_HC_PREFILL_FUSED = int(os.environ.get("VLLM_RDNA_HC_PREFILL_FUSED", "2") or 0)
_HC_FUSED_AVAILABLE: bool | None = None


def _hc_fused_available() -> bool:
    global _HC_FUSED_AVAILABLE
    if _HC_FUSED_AVAILABLE is None:
        import vllm._custom_ops  # noqa: F401  (loads _rocm_C)
        from vllm.platforms.rocm import on_gfx10x

        _HC_FUSED_AVAILABLE = on_gfx10x() and hasattr(
            torch.ops._rocm_C, "rdna_hc_mix_prefill"
        )
        logger.info(
            "rdna_hc_mix prefill: %s",
            f"fused HIP mode {_HC_PREFILL_FUSED}"
            if _HC_FUSED_AVAILABLE and _HC_PREFILL_FUSED
            else "torch",
        )
    return _HC_FUSED_AVAILABLE


def _hc_fused_prefill_ok(
    xn: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    lora_rank: int,
    hc_count: int,
) -> bool:
    return (
        _hc_fused_available()
        and hc_count == 4
        and xn.dtype == torch.float16
        and xn.dim() == 2
        and xn.is_contiguous()
        and w_down.dtype == torch.float16
        and w_up.dtype == torch.float16
        and w_down.is_contiguous()
        and w_up.is_contiguous()
        and lora_rank % 32 == 0
        and w_up.shape[1] == lora_rank
        and w_up.shape[0] == xn.shape[1]
        and xn.shape[1] % (32 * hc_count) == 0
        and w_down.shape[1] == xn.shape[1]
        and w_down.shape[0] % 8 == 0
    )


def _linear_padded_m(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    if _HC_PAD_M <= 0 or x.dim() != 2 or m % _HC_PAD_M == 0:
        return F.linear(x, weight)
    mp = (m + _HC_PAD_M - 1) // _HC_PAD_M * _HC_PAD_M
    xp = x.new_empty((mp, x.shape[1]))
    xp[:m].copy_(x)
    xp[m:].zero_()
    return F.linear(xp, weight)[:m]


def _rdna_hc_mix_fake(
    xn, w_down, w_down_i8, s_down, w_up, w_up_i8, s_up, lora_rank, hc_count
):
    m = xn.shape[0]
    n_down = w_down_i8.shape[0] if w_down_i8 is not None else w_down.shape[0]
    return (
        xn.new_empty((m, xn.shape[1] // hc_count)),
        xn.new_empty((m, n_down)),
    )


# ---------------------------------------------------------------- shared expert
def _rdna_shared_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w1_i8: torch.Tensor | None,
    s1: torch.Tensor | None,
    w2: torch.Tensor,
    w2_i8: torch.Tensor | None,
    s2: torch.Tensor | None,
    w_gate: torch.Tensor,
) -> torch.Tensor:
    """Per-rank partial of sigmoid(w_gate.x) * down(silu(gate)*up); caller reduces."""
    n = _ntok(x)
    if (
        0 < n <= _DECODE_MAX
        and x.dtype == torch.float16
        and x.dim() == 2
        and x.is_contiguous()
    ):
        from vllm import _custom_ops as ops

        a, sa = (w1_i8, s1) if w1_i8 is not None else (w1, None)
        b, sb = (w2_i8, s2) if w2_i8 is not None else (w2, None)
        act = ops.rdna_se_gate_up_silu(x, a, sa)
        return ops.rdna_se_down_gated(act, b, sb, x, w_gate)
    from vllm.model_executor.layers.rdna_dense_int8 import weight_for_gemm

    w1 = weight_for_gemm(w1, w1_i8, s1)
    w2 = weight_for_gemm(w2, w2_i8, s2)
    gu = F.linear(x, w1)
    half = gu.shape[-1] // 2
    act = F.silu(gu[..., :half]) * gu[..., half:]
    out = F.linear(act, w2)
    return torch.sigmoid(F.linear(x, w_gate.reshape(1, -1))) * out


def _rdna_shared_expert_fake(x, w1, w1_i8, s1, w2, w2_i8, s2, w_gate):
    n_out = w2_i8.shape[0] if w2_i8 is not None else w2.shape[0]
    return x.new_empty((*x.shape[:-1], n_out))


direct_register_custom_op(
    op_name="rdna_dense_gemm",
    op_func=_rdna_dense_gemm,
    mutates_args=[],
    fake_impl=_rdna_dense_gemm_fake,
)
direct_register_custom_op(
    op_name="rdna_hc_mix",
    op_func=_rdna_hc_mix,
    mutates_args=[],
    fake_impl=_rdna_hc_mix_fake,
)
direct_register_custom_op(
    op_name="rdna_shared_expert",
    op_func=_rdna_shared_expert,
    mutates_args=[],
    fake_impl=_rdna_shared_expert_fake,
)
