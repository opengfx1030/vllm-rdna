# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GPTQ kernel for AMD RDNA2 (gfx1030) — fp16 only.

Drop-in replacement for ExllamaLinearKernel on RDNA2. Two HIP kernels live in
``csrc/rocm/q_gemm_rdna2.cu`` (decode) and
``csrc/rocm/q_gemm_rdna2_prefill.cu`` (multi-config prefill), exposed via
``torch.ops._rocm_C.gptq_gemm_rdna2`` and
``torch.ops._rocm_C.gptq_gemm_rdna2_prefill``. The dispatcher selects
between them (and a fallthrough to upstream ``gptq_gemm`` Exllama) based on
(M, K, N).

gfx1030 has no ``v_dot2_f32_bf16`` (that landed on RDNA3, gfx1100+),
We restrict to fp16 only. bf16-trained checkpoints should be quantized
to fp16. RDNA3 (gfx1100) has a separate kernel that retains the bf16 path
— see ``q_gemm_rdna3.cu`` in the upstream tree.

Registered ahead of Hybrid and TritonW4A16LinearKernel for gfx1030 auto
select. Force Hybrid with ``--linear-backend rdna_hybrid``. Falls through
to Triton on non-RDNA2 ROCm devices (e.g. CDNA/MI300).
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# W4A8 (int4 weights, int8 activations) prefill GEMM gate. Opt-in: the env
# var must be exactly "1" at construction time (read once in
# process_weights_after_loading) for the dispatcher to try the W4A8 path.
# Default is OFF, so unset (or any value other than "1") preserves the
# existing gptq_gemm_rdna2_prefill behaviour byte-for-byte.
W4A8_ENV_VAR = "VLLM_RDNA2_W4A8_SDOT4"
# W4A8 is prefill-only: decode (M < W4A8_MIN_ROWS) keeps the W4A16 arms. The
# wired config is a8_lds_k32_ag (config_id 8, GROUP 32/64/128, M_TILE 8);
# those details live in the C++ entry, which owns the shape/LDS decisions and
# the internal W4A16 fallback.
W4A8_MIN_ROWS = 33


def _awq_prefill_available() -> bool:
    """Check if the AWQ-native prefill kernel is registered.

    hasattr(torch.ops._rocm_C, ...) is unreliable for torch ops because
    dir() only shows 'name' for the namespace object. Use a direct
    attribute access in a try/except instead.
    """
    try:
        torch.ops._rocm_C.awq_gemm_rdna2_prefill  # noqa: B018 - probe op
        return True
    except AttributeError:
        return False


def _w4a8_lds_fits(k: int, group_size: int) -> bool:
    """Mirror of pick_split_k: some group-aligned split <= 16 fits M_TILE=8
    rows of K plus the per-(token, group) scales in 64 KiB of LDS.

    Decided from (k, group_size) only: dynamo must not see a data-dependent
    branch on a runtime value in the traced forward.
    """
    groups = k // group_size
    for split in range(16, 0, -1):
        if groups % split:
            continue
        kps = k // split
        # Mirror compute_split_k: the kernel's K_STEP-wide loop never clamps
        # the tail, so a split whose k_per_split is not a multiple of 32 would
        # over-read the split. k % 32 == 0 is part of the gate, so split=1 is
        # always aligned.
        if kps % 32:
            continue
        if 8 * kps + 8 * (kps // group_size) * 8 <= 64 * 1024:
            return True
    return False


def _rdna2_w4a16_select_kernel(
    m: int,
    k: int,
    n: int,
    is_awq: bool = False,
    w4a8: bool = False,
    group_size: int = 64,
) -> str:
    # Opt-in W4A8 (int4 x int8 sdot4) prefill fast path. Prefill-only:
    # decode (M < W4A8_MIN_ROWS) keeps the W4A16 arms. Eligibility is decided
    # from ints only so dynamo keeps the branch out of the traced region; the
    # C++ entry owns the final shape/LDS decision and falls back to
    # gptq_gemm_rdna2_prefill internally.
    if (
        w4a8
        and m >= W4A8_MIN_ROWS
        and k % 32 == 0
        and k % group_size == 0
        and _w4a8_lds_fits(k, group_size)
    ):
        return "w4a8_prefill"
    # M > 256: exllama is the clear winner for compute-bound GEMMs.
    # AWQ models route to GPTQ prefill (ConfigA for M > 256): the separate
    # AWQ prefill kernel has BLOCK_M=16 (fails on non-aligned chunked-prefill
    # shapes) and is 2-6x slower than GPTQ prefill per microbench.
    if m > 256:
        if is_awq:
            return "prefill"
        return "exllama"
    # 32 < M <= 256 (small prefill): N-dominant split.
    # AWQ routes to GPTQ prefill for the same BLOCK_M=16 reason as above.
    # High N (>=3072) is the MLP gate/up projection shape where
    # exllama is faster; otherwise decode wins (attention/down).
    if m > 32:
        if is_awq:
            return "prefill"
        if n >= 3072:
            return "exllama"
        return "rdna2_decode"
    # M <= 32 (decode): K-dominant split.
    # K >= 4096 means V_DOT2 (decode) is the right path; otherwise
    # the tile-based prefill kernel is faster.
    if k >= 4096:
        return "rdna2_decode"
    return "prefill"


# vLLM compile traces apply_weights once and drops Dynamo's guards, so a Python
# branch on x.size(0) there would follow the trace-time M for every batch. The
# custom op makes the choice at run time instead (per call eager, per size
# under graph capture). Opt-in until docs/explore/w4a16-compile-dispatch says.
_RUNTIME_DISPATCH = os.environ.get("VLLM_RDNA2_W4A16_RUNTIME_DISPATCH", "0") == "1"


def _rdna2_w4a16_gemm(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    w_g_idx: torch.Tensor,
    n: int,
    is_awq: bool,
    size_bits: int,
    w4a8: bool = False,
    group_size: int = 64,
) -> torch.Tensor:
    m = x_2d.size(0)
    k = x_2d.size(1)
    kernel_name = _rdna2_w4a16_select_kernel(
        m, k, n, is_awq=is_awq, w4a8=w4a8, group_size=group_size
    )

    # AWQ stores literal zeros → kernel must NOT add 1 (use_v2_format=True,
    # q_gemm_rdna2.cu:219 picks zero_offset=0). GPTQv1 stores zero-1 →
    # kernel adds 1 to recover the original zero (use_v2_format=False,
    # zero_offset=1). uint4b8 is GPTQv1; uint4 is AWQ.
    use_v2_format = is_awq

    if kernel_name == "w4a8_prefill":
        # Self-contained: C++ allocates the int8 A + scales + sums, fires
        # the W4A8 sdot4 fast path, or falls back internally to
        # gptq_gemm_rdna2_prefill. No post-call test, no env access, no
        # logging in the traced region.
        output = ops.w4a8_gemm_rdna2(x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format)
    elif kernel_name == "awq_prefill" and hasattr(ops, "awq_gemm_rdna2_prefill"):
        output = ops.awq_gemm_rdna2_prefill(
            x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format
        )
    elif kernel_name == "prefill" and hasattr(ops, "gptq_gemm_rdna2_prefill"):
        output = ops.gptq_gemm_rdna2_prefill(
            x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format
        )
    elif kernel_name == "exllama" and hasattr(ops, "gptq_gemm"):
        output = ops.gptq_gemm(x_2d, w_q, w_zp, w_s, True, use_v2_format, size_bits)
    elif kernel_name == "rdna2_decode" and hasattr(ops, "gptq_gemm_rdna2"):
        output = ops.gptq_gemm_rdna2(x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format)
    else:
        if hasattr(ops, "awq_gemm_rdna2_prefill") and use_v2_format:
            output = ops.awq_gemm_rdna2_prefill(
                x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format
            )
        elif hasattr(ops, "gptq_gemm_rdna2_prefill"):
            output = ops.gptq_gemm_rdna2_prefill(
                x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format
            )
        elif hasattr(ops, "gptq_gemm"):
            output = ops.gptq_gemm(x_2d, w_q, w_zp, w_s, True, use_v2_format, size_bits)
        elif hasattr(ops, "gptq_gemm_rdna2"):
            output = ops.gptq_gemm_rdna2(x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format)
        else:
            raise RuntimeError(
                f"RDNA2 W4A16 dispatcher: kernel_name={kernel_name!r} but "
                "neither gptq_gemm nor gptq_gemm_rdna2 ops are "
                "available; rebuild the C++ extension"
            )
    return output


def _rdna2_w4a16_gemm_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    w_g_idx: torch.Tensor,
    n: int,
    is_awq: bool,
    size_bits: int,
    w4a8: bool = False,
    group_size: int = 64,
) -> torch.Tensor:
    return x_2d.new_empty((x_2d.size(0), n))


direct_register_custom_op(
    op_name="rdna2_w4a16_gemm",
    op_func=_rdna2_w4a16_gemm,
    fake_impl=_rdna2_w4a16_gemm_fake,
)


class RDNA2W4A16LinearKernel(MPLinearKernel):
    # uint4b8 — GPTQv1 (zero-bias: stored as zero-1, kernel applies +1)
    # uint4   — AWQ     (no zero-bias: stored as literal 0, kernel must NOT
    #                   add 1). The kernel selects between the two via
    #                   use_v2_format (= weight_type is uint4) and the
    #                   q_gemm_rdna2.cu:219 ternary.
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates via on_gfx10x() in can_implement.
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 W4A16 kernel is ROCm-only"

        if not on_gfx10x():
            return False, "RDNA2 W4A16 kernel requires gfx1030"

        # The HIP op is registered by the C++ extension; if a user is running
        # against a vLLM build that doesn't include it (e.g. partial rebuild),
        # fall through gracefully to the next kernel in the registry.
        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")
        ):
            return (
                False,
                "torch.ops._rocm_C.gptq_gemm_rdna2 missing — rebuild C++ extension",
            )

        if c.act_type != torch.float16:
            return False, "RDNA2 W4A16 kernel only supports fp16 on gfx1030"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"RDNA2 W4A16 kernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                "RDNA2 W4A16 kernel does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide K "
                f"({c.full_weight_shape[0]})",
            )

        # Output features must be a multiple of the pack factor (8 nibbles per
        # int32) and of 8 so that qzeros (packed 4-bit per col) align cleanly
        # against the BLOCK_KN_SIZE*4 = 512 N-stride and per-thread 4 columns.
        if c.partition_weight_shape[1] % 8 != 0:
            return (
                False,
                "Output features must be a multiple of 8 for the RDNA2 "
                "W4A16 kernel (qzeros packing)",
            )

        return True, None

    # ----- Weight prep (identical layout/shuffle as ExllamaLinearKernel) -----

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        device = getattr(layer, self.w_q_name).device

        # Synthesize zero points if the checkpoint doesn't carry them.
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # GPTQv1 quirk: the kernel adds 1 to the stored zero, so we
                # encode (bias - 1) here. See exllama.py for the link to the
                # documentation of this checkpoint-format wart.
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "RDNA2 W4A16 kernel: zero-bias 4-bit quant requires "
                    "explicit zero points (GPTQv1 +1 quirk)."
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        # The RDNA2 HIP ops still take a g_idx tensor; act-order was removed
        # upstream (#54809), so it is always empty (no input reordering).
        layer.rdna2_empty_g_idx = torch.empty((0,), dtype=torch.int, device=device)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            # Same 4-bit shuffle as exllama. The RDNA2 kernel reads weights in
            # the same shuffled int32 layout and uses the (qa & 0x000F000F)
            # bit-trick on top.
            ops.gptq_shuffle(x_cont, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        # AWQ (uint4) only: the AWQ repack in
        # ``_convert_awq_to_standard_format`` produces qzeros as ``[N//8, G]``
        # packed along dim 0. The kernel reads
        # ``b_qzeros[g * (size_n/8) + qcol]`` (see q_gemm_rdna2_common.cuh:106),
        # i.e. layout ``[G, N//8]`` packed along dim 1.
        #
        # Layout trace (packing order is identical in both):
        #   AWQ repack output    new_qz[i, g] packs nibbles for columns
        #                        [i*8, i*8+8) of group g, nibble j = column
        #                        i*8+j (little-endian by shift order).
        #   Kernel reads         qz_row = b_qzeros + g*(size_n/8), and
        #                        load4_zeros reads qz_row[qcol] with
        #                        nibble (n & 7) at column qcol*8+(n & 7).
        #   Transpose            new_qz.T has shape [G, N//8]; element
        #                        [g, i] is the same int32 that was at
        #                        new_qz[i, g]. .contiguous() makes it a
        #                        packed-int32 row per group, matching what
        #                        the kernel reads.
        #
        # GPTQ (uint4b8) takes the synthesized-zeros path above, which is
        # already ``[G, N//8]`` packed along dim 1, so no transform needed.
        if c.weight_type == scalar_types.uint4:

            def transform_w_zp(x):
                assert isinstance(x, BasevLLMParameter)
                return x.data.T.contiguous()

            self._transform_param(layer, self.w_zp_name, transform_w_zp)

        # Resolve the W4A8 opt-in flag ONCE here; never read the env in the
        # forward. The C++ entry owns the shape/LDS eligibility and its own
        # internal fallback, so this flag is a pure opt-in (env + ops built).
        try:
            torch.ops._rocm_C.w4a8_gemm_rdna2  # noqa: B018 - probe op
            w4a8_ops_built = True
        except AttributeError:
            w4a8_ops_built = False
        self._w4a8 = os.environ.get(W4A8_ENV_VAR) == "1" and w4a8_ops_built

    # ----- Forward --------------------------------------------------------

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp = self._get_weight_params(layer)
        w_g_idx = layer.rdna2_empty_g_idx

        assert w_zp is not None, "Zero points are required by RDNA2 W4A16"

        n = c.partition_weight_shape[1]
        is_awq = c.weight_type == scalar_types.uint4
        gemm = (
            torch.ops.vllm.rdna2_w4a16_gemm if _RUNTIME_DISPATCH else _rdna2_w4a16_gemm
        )
        output = gemm(
            x_2d,
            w_q,
            w_zp,
            w_s,
            w_g_idx,
            n,
            is_awq,
            c.weight_type.size_bits,
            self._w4a8,
            c.group_size,
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
