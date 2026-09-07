# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) AWQ W4A16 kernel — fp16, native HIP, sequential layout.

Uses the native HIP kernel in ``csrc/rocm/awq_gemm_rdna2.cu`` exposed via
``torch.ops._rocm_C.awq_gemm_rdna2``.

Weight layout (post-process_weights_after_loading):
  qweight: [K, N//8]  int32  — sequential packing (shifts [0,4,...,28])
  scales:  [G, N]     fp16
  qzeros:  [G, N//8]  int32  — sequential packing
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class RDNA2AWQLinearKernel(MPLinearKernel):
    """RDNA2 native AWQ kernel for uint4 (asymmetric, explicit zeros)."""

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        import vllm.envs as envs

        if not current_platform.is_rocm():
            return False, "RDNA2 AWQ kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            return False, "RDNA2 AWQ kernel requires gfx1030"

        if not envs.VLLM_RDNA2_NATIVE_AWQ:
            return (
                False,
                "VLLM_RDNA2_NATIVE_AWQ=0: no speed benefit over Triton path",
            )

        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "awq_gemm_rdna2")
        ):
            return (
                False,
                "torch.ops._rocm_C.awq_gemm_rdna2 missing — rebuild C++ extension",
            )

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                (
                    f"Quant type ({c.weight_type}) not supported; "
                    f"supported: {cls.SUPPORTED_QUANT_TYPES}"
                ),
            )

        if c.act_type != torch.float16:
            return False, "RDNA2 AWQ kernel only supports fp16 activations"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by RDNA2 AWQ kernel",
            )

        gs = c.group_size
        if gs in (32,):
            # Measured on gfx1030: at group_size=32 the kernel runs at
            # 11-24 GB/s effective (~5x slower than the Triton W4A16 path)
            # because scale/zero traffic dominates. Let the Triton kernel
            # (with the M=1 GEMV fast path) serve these layers.
            return (
                False,
                "group_size=32: scale-traffic bound; use the Triton path",
            )
        if gs not in [-1, 64, 128, 256] and gs != c.full_weight_shape[0]:
            return (
                False,
                (
                    f"Group size {gs} not supported; "
                    "supported: [-1, 64, 128, 256] "
                    f"or full K ({c.full_weight_shape[0]})"
                ),
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack weights from checkpoint layout to kernel layout [K, N//8].

        Done in tiles over N to bound scratch memory: the naive version
        materializes an ``[N, K]`` int32 unpack + transpose (~33x the packed
        weight) which OOMs on large layers during loading.
        """

        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32 (8 consecutive K nibbles per word)

            N_dim, K8 = w.shape
            K_dim = K8 * 8
            N8 = N_dim // 8
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            out = torch.empty((K_dim, N8), dtype=torch.int32, device=w.device)

            # CHUNK must be a multiple of 8 (and N is already a multiple of 8,
            # ensured by can_implement), so every chunk packs to whole words.
            CHUNK = 512
            for n0 in range(0, N_dim, CHUNK):
                n1 = min(n0 + CHUNK, N_dim)
                wc = w[n0:n1, :]  # [c, K//8] view
                uc = ((wc.unsqueeze(-1) >> shifts) & 0xF).reshape(
                    n1 - n0, K_dim
                )  # [c, K] int32 (small: only one tile)
                kc = uc.t().contiguous()  # [K, c] int32
                c8 = (n1 - n0) // 8
                packed = torch.sum(
                    (kc.view(K_dim, c8, 8) & 0xF) << shifts,
                    dim=2,
                    dtype=torch.int32,
                )  # [K, c//8] int32
                out[:, n0 // 8 : n0 // 8 + c8] = packed

            x.data = out
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        assert w_zp is not None, "AWQ requires explicit zero points"

        output = ops.awq_gemm_rdna2(x_2d, w_q, w_zp, w_s)

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
