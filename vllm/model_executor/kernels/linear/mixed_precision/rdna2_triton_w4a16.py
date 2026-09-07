# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) W4A16 kernel for AWQ/GPTQ using Triton.

RDNA2 lacks v_pk_fma_f16, v_dot2_f32_f16, and v_dot2_f32_bf16, so the
Triton kernel's fp32-accumulator path is used. This handles both
uint4 (AWQ, asymmetric with explicit zeros) and uint4b8 (GPTQ,
symmetric with bias=8).

Weight layout (post-process_weights_after_loading):
  qweight: [K, N//8]  int32  — GPTQ sequential packing (shifts [0,4,...,28])
  scales:  [K//G, N]  fp16
  qzeros:  [K//G, N//8]  int32  (None for symmetric uint4b8)
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .triton_w4a16 import triton_w4a16_gemm


def _use_w4_kpack() -> bool:
    if not (current_platform.is_rocm() and envs.VLLM_RDNA2_W4_KPACK):
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x() and hasattr(torch.ops._rocm_C, "gemv_w4_kpack")


def _rdna2_w4a16_linear_impl(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
    kpack: bool,
) -> torch.Tensor:
    """Eager dispatch: M=1 decode GEMV vs the generic tl.dot GEMM.

    Wrapped as an opaque custom op so torch.compile/CUDA-graph capture
    executes this with the real runtime M (dynamic-shape guards are not
    re-evaluated at runtime, so a Python ``if M == 1`` at the call site
    would be baked out of the traced graph).
    """
    if kpack:
        # K-packed [K//8, N] int32 layout (HIP gemv_w4_kpack path)
        if x.shape[0] == 1 and group_size in (32, 64, 128, 256):
            from vllm import _custom_ops as ops

            split_groups = 8 if x.shape[1] >= 8192 else 4
            out = ops.gemv_w4_kpack(
                x, w_q, w_s, w_zp, group_size, split_groups
            )
            if bias is not None:
                out = out + bias
            return out
        if (
            1 < x.shape[0] <= 4
            and group_size in (32, 64, 128, 256)
        ):
            # Spec-decode verify batches (MTP k<=3 -> M=3): dedicated
            # broadcast+sum GEMV on the kpack layout. The generic tl.dot
            # GEMM below is 15-17x slower at these M (fp32 dot -> scalar
            # FMA on gfx1030, BLOCK_M=64 mostly padding).
            from vllm.model_executor.layers.rdna2_gemm import (
                rdna2_w4a16_kpack_gemv_m,
            )

            return rdna2_w4a16_kpack_gemv_m(
                x, w_q, w_s, w_zp, group_size, zp_bias, bias
            )
        # M > 4 on K-packed: unpack via indexed shifts (each int32 re-read
        # 8x from cache) into the Triton kernel's N-packed expectation is
        # not free — instead compute via the transposed GEMM form on the
        # packed rows.
        return _kpack_gemv_batched(
            x, w_q, w_s, w_zp, bias, group_size, zp_bias
        )

    if (
        1 <= x.shape[0] <= 4
        and x.shape[1] % 32 == 0
        and group_size in (32, 64, 128, 256)
    ):
        from vllm.model_executor.layers.rdna2_gemm import (
            rdna2_w4a16_gemv,
            rdna2_w4a16_gemv_m,
        )

        if x.shape[0] == 1:
            return rdna2_w4a16_gemv(
                x, w_q, w_s, w_zp, group_size, zp_bias, bias
            ).unsqueeze(0)
        return rdna2_w4a16_gemv_m(
            x, w_q, w_s, w_zp, group_size, zp_bias, bias
        )

    out = triton_w4a16_gemm(x, w_q, w_s, w_zp, group_size, zp_bias)
    if bias is not None:
        out = out + bias
    return out


def _kpack_gemv_batched(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """M > 1 on the K-packed layout.

    Small M (<= 4, spec-decode verify): the Triton tl.dot GEMM below
    (weights re-read from cache, one pass). Large M (prefill): same GEMM —
    compute-bound, so the K-packed indexing overhead is irrelevant.
    """
    from vllm.model_executor.layers.rdna2_gemm import w4a16_kpack_gemm

    return w4a16_kpack_gemm(
        x, w_q, w_s, w_zp, group_size, float(zp_bias), bias
    )


def _rdna2_w4a16_linear_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
    kpack: bool,
) -> torch.Tensor:
    # scales are [K//G, N] in both packing layouts.
    return x.new_empty((x.shape[0], w_s.shape[1]))


direct_register_custom_op(
    op_name="rdna2_w4a16_linear",
    op_func=_rdna2_w4a16_linear_impl,
    fake_impl=_rdna2_w4a16_linear_fake,
)


class RDNA2TritonW4A16LinearKernel(MPLinearKernel):
    """RDNA2-optimized Triton W4A16 kernel for AWQ and GPTQ."""

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4, scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 Triton W4A16 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            return False, "RDNA2 Triton W4A16 kernel requires gfx1030"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                (
                    f"Quant type ({c.weight_type}) not supported; "
                    f"supported: {cls.SUPPORTED_QUANT_TYPES}"
                ),
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8",
            )

        if c.has_g_idx:
            return (
                False,
                "g_idx is not supported by the RDNA2 W4A16 kernel",
            )

        gs = c.group_size
        if gs not in [-1, 32, 64, 128, 256] and gs != c.full_weight_shape[0]:
            return (
                False,
                (
                    f"Group size {gs} not supported; supported: "
                    f"[-1, 32, 64, 128, 256] or full K "
                    f"({c.full_weight_shape[0]})"
                ),
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack weights from checkpoint layout to kernel layout.

        VLLM_RDNA2_W4_KPACK=1 (gfx10x default): [K//8, N] int32 packed
        along K (HIP gemv_w4_kpack layout) + unpacked uint8 zeros.
        Otherwise: [K, N//8] int32 packed along N (Triton layout).
        """

        def repack_w_q_npack(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32

            N_dim, K8 = w.shape
            K_dim = K8 * 8
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
            w_KN = w_unpacked.t().contiguous()
            N8 = N_dim // 8
            w_repacked = torch.sum(
                (w_KN.view(K_dim, N8, 8) & 0xF) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_q_kpack(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32

            N_dim, K8 = w.shape
            K_dim = K8 * 8
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
            # pack along K: [K//8, N], nibble j of int32 row k//8 holds
            # value for k = (k//8)*8 + j
            w_KN = w_unpacked.t().contiguous()  # [K, N]
            pack_shifts = shifts.view(1, 8, 1)  # broadcast [K//8, 8, N]
            x.data = (
                torch.sum(
                    (w_KN.view(K_dim // 8, 8, N_dim) & 0xF) << pack_shifts,
                    dim=1,
                    dtype=torch.int32,
                )
                .contiguous()
            )
            return x

        use_kpack = _use_w4_kpack()
        # The opaque dispatch op reads this to pick the layout path.
        layer.w4a16_kpack_layout = use_kpack
        repack_w_q = repack_w_q_kpack if use_kpack else repack_w_q_npack

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                if use_kpack:
                    # HIP gemv_w4_kpack wants unpacked [K//G, N] uint8.
                    # Checkpoint zp is [N//8, K//G] int32 packed along N
                    # (dim 0): unpack to [N, K//G] first (nibble j of row
                    # q holds n = q*8 + j), then transpose.
                    shifts = torch.arange(8, device=zp.data.device,
                                          dtype=torch.int32) * 4
                    z_n = (
                        ((zp.data.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF)
                        .reshape(-1, zp.data.shape[1])
                        .to(torch.uint8)
                    )  # [N, K//G]
                    replace_parameter(
                        layer,
                        self.w_zp_name,
                        torch.nn.Parameter(
                            z_n.t().contiguous(), requires_grad=False
                        ),
                    )
                else:
                    replace_parameter(
                        layer,
                        self.w_zp_name,
                        torch.nn.Parameter(
                            zp.data.t().contiguous(), requires_grad=False
                        ),
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

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        # Opaque custom op: the M=1 GEMV dispatch must see the real
        # runtime M at capture, not a traced symbolic shape.
        output = torch.ops.vllm.rdna2_w4a16_linear(
            x_2d, w_q, w_s, w_zp, bias, group_size, zp_bias,
            getattr(layer, "w4a16_kpack_layout", False),
        )
        return output.reshape(out_shape)
