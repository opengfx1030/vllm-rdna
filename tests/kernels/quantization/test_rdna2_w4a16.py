#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 W4A16 GPTQ kernel (gfx1030).

Exercises ``RDNA2W4A16LinearKernel`` end-to-end: it builds a layer with
GPTQ-format checkpoint parameters, runs ``process_weights_after_loading``
(weight shuffle + zero-point synthesis), then ``apply_weights``. The
internal 3-bucket dispatcher routes between the decode kernel, the
multi-config prefill kernel, and an Exllama fallthrough based on (M, K, N).

The kernels are exposed via ``torch.ops._rocm_C.gptq_gemm_rdna2`` /
``gptq_gemm_rdna2_prefill`` and are only built/registered for gfx1030;
tests are skipped elsewhere.

Run `pytest tests/kernels/quantization/test_rdna2_w4a16.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (  # noqa: E402
    RDNA2W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms.rocm import on_gfx10x  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

device = "cuda"

WEIGHT_TYPE = scalar_types.uint4b8  # symmetric int4, bias = 8
PACK_FACTOR = 8  # 8 x 4-bit nibbles per int32

# Skip everything unless we are on the only architecture the kernel is built for.
gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")
    ),
    reason="requires gfx1030 with the _rocm_C.gptq_gemm_rdna2 op built in",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 reference for the RDNA2 W4A16 op.

    x_mk:       [M, K] fp16 activations.
    q_int4_kn:  [K, N] int32 raw stored nibbles in [0, 15].
    scales_gn:  [K//G, N] per-group scales (act dtype).
    zeros_gn:   [K//G, N] int32 raw stored zero points in [0, 15], or None
                for the symmetric path (kernel synthesizes stored zero = 7).
    group_size: G.

    The kernel applies the GPTQv1 "+1" zero-point quirk, so the effective
    zero is ``stored_zero + 1`` (symmetric path: 7 + 1 == bias == 8).
    """
    K, N = q_int4_kn.shape
    s_full = scales_gn.repeat_interleave(group_size, dim=0).to(torch.float32)
    if zeros_gn is None:
        z_full = torch.full(
            (K, N), float(WEIGHT_TYPE.bias), device=x_mk.device, dtype=torch.float32
        )
    else:
        z_full = (zeros_gn + 1).repeat_interleave(group_size, dim=0).to(torch.float32)
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full
    out = x_mk.to(torch.float32) @ w_fp
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


# ---------------------------------------------------------------------------
# Layer construction
# ---------------------------------------------------------------------------


def _build_layer(
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
) -> torch.nn.Module:
    """Build a dummy layer carrying GPTQ-format params, as the loader would."""
    no_loader = lambda *args, **kwargs: None  # noqa: E731

    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=qweight,
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=scales_gn.to(torch.float16),
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
        ),
    )
    if zeros_gn is not None:
        qzeros = pack_quantized_values_into_int32(zeros_gn, WEIGHT_TYPE, packed_dim=1)
        layer.register_parameter(
            "qzeros",
            PackedvLLMParameter(
                data=qzeros,
                weight_loader=no_loader,
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=PACK_FACTOR,
            ),
        )
    return layer


def _run_kernel(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    K, N = q_int4_kn.shape
    has_zp = zeros_gn is not None

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=has_zp,
        has_g_idx=False,
    )
    ok, reason = RDNA2W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected a supported config: {reason}"

    layer = _build_layer(q_int4_kn, scales_gn, zeros_gn)
    kernel = RDNA2W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros" if has_zp else None,
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)
    return kernel.apply_weights(layer, x_mk, bias=bias)


# fp16 uses the exllamav2 bit-trick; allow ~3% relative noise.
_REL_L2_TOL = 5e-2


def _assert_close(out: torch.Tensor, ref: torch.Tensor):
    rel_l2 = (out.to(torch.float32) - ref.to(torch.float32)).norm() / ref.to(
        torch.float32
    ).norm()
    assert rel_l2 < _REL_L2_TOL, f"relative L2 error {rel_l2:.4f} exceeds {_REL_L2_TOL}"


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

# Coverage: each shape exercises one path of the 3-bucket dispatcher.
# (M, K, N, G) and the bucket it routes to per _rdna2_w4a16_select_kernel:
#   M <= 32, K < 4096                   -> prefill
#   M <= 32, K >= 4096                  -> rdna2_decode (K-gated)
#   32 < M <= 256, N < 3072             -> rdna2_decode (down-proj)
#   32 < M <= 256, N >= 3072            -> exllama (gate/up-proj)
#   M > 256                             -> exllama
MKNG_SHAPES = [
    (1, 128, 128, 128),     # prefill (M <= 32, K < 4096)
    (2, 256, 256, 128),     # prefill
    (8, 256, 512, 64),      # prefill
    (8, 4096, 512, 128),    # rdna2_decode (K-gated)
    (16, 512, 256, 128),    # prefill
    (32, 512, 512, 64),     # prefill
    (64, 1024, 1024, 128),  # rdna2_decode (32 < M <= 256, N < 3072)
    (300, 512, 2048, 128),  # exllama (M > 256)
]


@gfx1030_only
@pytest.mark.parametrize("has_zp", [False, True], ids=["no_zp", "with_zp"])
@pytest.mark.parametrize(
    "M,K,N,G", MKNG_SHAPES, ids=[f"m{m}_k{k}_n{n}_g{g}" for m, k, n, g in MKNG_SHAPES]
)
def test_rdna2_w4a16_matches_reference(has_zp, M, K, N, G, dist_init):
    set_random_seed(0)
    assert K % G == 0 and K % PACK_FACTOR == 0 and N % PACK_FACTOR == 0

    groups = K // G
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(torch.float16)
    zeros_gn = (
        torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)
        if has_zp
        else None
    )

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None)
    ref = _reference(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None)

    assert out.shape == (M, N) and out.dtype == torch.float16
    _assert_close(out, ref)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 16], ids=["decode", "prefill"])
def test_rdna2_w4a16_bias(M, dist_init):
    """Bias is added on both the decode (M=1) and prefill (M=16) paths."""
    set_random_seed(0)
    K, N, G = 512, 256, 128
    groups = K // G

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(torch.float16)
    bias = (0.1 * torch.randn(N, device=device, dtype=torch.float32)).to(torch.float16)

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, None, G, bias)
    ref = _reference(x_mk, q_int4_kn, scales_gn, None, G, bias)

    _assert_close(out, ref)
