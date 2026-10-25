# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 int8 dense linear for AMD RDNA2 (gfx1030) through the sdot4 HIP kernel.

Weights: compressed-tensors W8A8 int8 ([N, K] int8, symmetric, per-channel or
per-tensor fp32 scale), repacked at load into the K-major dword layout
``w8a8_gemm_rdna2`` reads, with the scale as fp16 [1, N].
Activations: fp16, quantized in-kernel per (token, group), dynamic and
symmetric. Static input scales and asymmetric activations are declined.

Opt-in: ``VLLM_RDNA2_W8A8_SDOT4=1`` on gfx1030 with the op built; otherwise the
registry falls through to the next int8 kernel (Triton on ROCm).
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization import rdna2_w8a8
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class RDNA2Int8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 W8A8 kernel is ROCm-only"
        if not rdna2_w8a8.enabled():
            return False, f"requires gfx1030 and {rdna2_w8a8.ENV_VAR}=1"
        if not rdna2_w8a8.dense_ops_built():
            return False, "requires w8a8_gemm_rdna2 built in"
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric activations only"
        if c.is_static_input_scheme:
            return False, "supports dynamic per-token activations only"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, w_s, _, _, _ = self._get_layer_params(layer)
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        n, k = w_q.shape
        if k % 32 or n % 4:
            raise ValueError(
                f"RDNA2 W8A8 needs K % 32 == 0 and N % 4 == 0, got [N={n}, K={k}]"
            )

        # Per-tensor scales (fused QKV/MLP carry one per logical matrix)
        # become per-channel; the kernel reads one fp16 scale per column.
        if not self.config.is_channelwise:
            w_s = convert_to_channelwise(w_s, layer.logical_widths)
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(rdna2_w8a8.pack_weight(w_q.data), requires_grad=False),
        )
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(
                w_s.data.reshape(1, n).to(torch.float16).contiguous(),
                requires_grad=False,
            ),
        )
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, _, _, _ = self._get_layer_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        in_dtype = x_2d.dtype
        if in_dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        out = ops.w8a8_gemm_rdna2(x_2d, w_q, w_s, rdna2_w8a8.act_group(x_2d.shape[-1]))
        if out.dtype != in_dtype:
            out = out.to(in_dtype)
        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], w_q.shape[1])
