#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the W4A8 (int4 weights, int8 activations) ops
wired into ``torch.ops._rocm_C`` (gfx1030).

Mirrors the G2 checks in
``benchmarks/kernels/w4a8_sdot4_explore/test_reference.py``:

* ``act_quant``: CUDA kernel vs a CPU NumPy reference (per-token-group
  quant + the tiled ``[T][K/8][MT][8]`` layout).
* ``gemm``: CUDA kernel vs ``gptq_gemm_rdna2_prefill`` on the same packed
  weight buffer (rel-L2 < 0.1). Both ops read the exact shuffled layout
  ``RDNA2W4A16LinearKernel`` leaves behind, so this is the production
  apples-to-apples comparison.
* ``invalid group-size`` returns a non-zero status.
* Clean skip when the ops are absent (no GPU, partial build, non-gfx1030).

The CPU reference is a small focused NumPy model (not the full explore
``reference.py``); it produces bit-exact int8 / fp32 / int32 outputs that
match what ``w4a8_act_quant_kernel`` writes, including the per-group scale
variant used by the ``a8_lds_k32_ag`` config.
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("W4A8 ops are ROCm-only", allow_module_level=True)

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

MTILE = 8  # mirrors W4A8_DEFAULT_M_TILE in rdna2_w4a16.py
A_PERM = (0, 4, 1, 5, 2, 6, 3, 7)  # from csrc/rocm/explore/w4a8_sdot4.cuh
SHUFFLE_SLOTS = (0, 2, 4, 6, 1, 3, 5, 7)  # from explore/reference.py


def _has_w4a8_ops() -> bool:
    """Skip cleanly when the ops are not built (no GPU, partial build,
    different arch). ``_jit_get_all_schemas`` is the only reliable probe;
    ``dir(torch.ops._rocm_C)`` only returns ``['name']`` for the namespace
    object — see test_rdna2_w4a16.py:53-59 for the same workaround.
    """
    if not hasattr(torch.ops, "_rocm_C"):
        return False
    try:
        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    names = [str(s) for s in schemas]
    return any("w4a8_gemm_rdna2" in n for n in names) and any(
        "w4a8_act_quant_rdna2" in n for n in names
    )


gfx1030_w4a8 = pytest.mark.skipif(
    not (on_gfx10x() and _has_w4a8_ops()),
    reason="requires gfx1030 with w4a8_act_quant_rdna2 / w4a8_gemm_rdna2 built in",
)

def _ensure_tp_group() -> None:
    """Make sure a single-rank model-parallel group exists.

    Must run inside a ``set_current_vllm_config`` context (the
    ``default_vllm_config`` fixture provides it), because
    ``ensure_model_parallel_initialized`` reads the current config, and the
    packed parameter classes read the TP rank. pytest's global cleanup tears the
    group down between tests, so probe instead of caching. Off-ROCm runs are
    already skipped at module import; same init as test_triton_w4a16.
    """
    from vllm.distributed.parallel_state import get_tp_group

    try:
        get_tp_group()
        return
    except AssertionError:
        pass
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:0",
        local_rank=0,
    )
    ensure_model_parallel_initialized(1, 1)


# ---------------------------------------------------------------------------
# CPU reference (NumPy) — the per-(token, group) quant variant the
# a8_lds_k32_ag config uses, written in the tiled layout the GEMM reads.
# Bit-exact: the kernel rounds half-to-even via f32 rintf and saturates
# the same way NumPy does.
# ---------------------------------------------------------------------------


def _quant_act_ref(x_mk: np.ndarray, group_size: int, mt: int = MTILE):
    m, k = x_mk.shape
    assert k % group_size == 0 and k % 8 == 0
    num_tiles = (m + mt - 1) // mt
    g = k // group_size
    a_i8 = np.zeros((num_tiles, k // 8, mt, 8), dtype=np.int8)
    a_scale = np.zeros((num_tiles, g, mt), dtype=np.float32)
    a_asum = np.zeros((num_tiles, g, mt), dtype=np.int32)
    for t in range(num_tiles):
        for r in range(mt):
            row = t * mt + r
            for gi in range(g):
                if row < m:
                    chunk = x_mk[row, gi * group_size : (gi + 1) * group_size]
                    gmax = float(np.abs(chunk).max())
                    inv = 0.0 if gmax == 0.0 else 127.0 / gmax
                    a_scale[t, gi, r] = gmax / 127.0
                    q = np.clip(
                        np.rint(chunk.astype(np.float32) * inv), -128, 127
                    ).astype(np.int8)
                    a_asum[t, gi, r] = int(q.astype(np.int32).sum())
                    for c in range(group_size // 8):
                        for lane in range(8):
                            a_i8[
                                t,
                                gi * (group_size // 8) + c,
                                r,
                                lane,
                            ] = q[c * 8 + A_PERM[lane]]
    return a_i8, a_scale, a_asum


# ---------------------------------------------------------------------------
# Weight packing (identical to RDNA2W4A16LinearKernel + the explore reference)
# ---------------------------------------------------------------------------


def _pack_k_major(q_kn: np.ndarray) -> np.ndarray:
    k, n = q_kn.shape
    q = q_kn.astype(np.uint32).reshape(k // 8, 8, n)
    out = np.zeros((k // 8, n), dtype=np.uint32)
    for i in range(8):
        out |= q[:, i, :] << np.uint32(4 * i)
    return out


def _exllama_shuffle(packed: np.ndarray) -> np.ndarray:
    packed = packed.astype(np.uint32)
    out = np.zeros_like(packed)
    for slot, k_off in enumerate(SHUFFLE_SLOTS):
        nibble = (packed >> np.uint32(4 * k_off)) & np.uint32(0xF)
        out |= nibble << np.uint32(4 * slot)
    return out


def _pack_zeros_n_major(z_gn: np.ndarray) -> np.ndarray:
    g, n = z_gn.shape
    z = z_gn.astype(np.uint32).reshape(g, n // 8, 8)
    out = np.zeros((g, n // 8), dtype=np.uint32)
    for j in range(8):
        out |= z[:, :, j] << np.uint32(4 * j)
    return out


# ---------------------------------------------------------------------------
# Layer construction (same shape as test_rdna2_w4a16.py)
# ---------------------------------------------------------------------------


WEIGHT_TYPE = scalar_types.uint4b8  # GPTQv1: stored zero = q - 1, kernel adds 1
PACK_FACTOR = 8


def _build_layer(
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
) -> torch.nn.Module:
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


def _w4a16_reference(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    K, N = q_int4_kn.shape
    s_full = scales_gn.repeat_interleave(group_size, dim=0).to(torch.float32)
    if zeros_gn is None:
        z_full = torch.full(
            (K, N), float(WEIGHT_TYPE.bias), device=x_mk.device, dtype=torch.float32
        )
    else:
        z_full = (zeros_gn + 1).repeat_interleave(group_size, dim=0).to(torch.float32)
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full
    return (x_mk.to(torch.float32) @ w_fp).to(x_mk.dtype)


# ---------------------------------------------------------------------------
# act_quant
# ---------------------------------------------------------------------------


@gfx1030_w4a8
@pytest.mark.parametrize("group_size", [32, 64, 128], ids=["g32", "g64", "g128"])
@pytest.mark.parametrize("M", [1, 8, 32, 200], ids=["m1", "m8", "m32", "m200"])
def test_w4a8_act_quant_matches_numpy(M, group_size):
    set_random_seed(0)
    K = 512
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    G = K // group_size
    num_tiles = (M + MTILE - 1) // MTILE

    a_i8 = torch.empty((num_tiles, K // 8, MTILE, 8), dtype=torch.int8, device=device)
    a_scale = torch.empty((num_tiles, G, MTILE), dtype=torch.float32, device=device)
    a_asum = torch.empty((num_tiles, G, MTILE), dtype=torch.int32, device=device)

    aq_ret = torch.ops._rocm_C.w4a8_act_quant_rdna2(
        x_mk, group_size, a_i8, a_scale, a_asum
    )
    assert aq_ret.numel() > 0, "act_quant returned an empty (ineligible) tensor"

    want_a_i8, want_a_scale, want_a_asum = _quant_act_ref(
        x_mk.cpu().numpy().astype(np.float16), group_size, MTILE
    )
    # The kernel leaves MT-row padding rows at zero; compare element-wise.
    np.testing.assert_array_equal(a_i8.cpu().numpy(), want_a_i8)
    np.testing.assert_allclose(a_scale.cpu().numpy(), want_a_scale, rtol=0, atol=0)
    np.testing.assert_array_equal(a_asum.cpu().numpy(), want_a_asum)


@gfx1030_w4a8
def test_w4a8_act_quant_rejects_bad_group_size():
    set_random_seed(0)
    M, K = 8, 256
    x_mk = torch.zeros((M, K), device=device, dtype=torch.float16)
    num_tiles = (M + MTILE - 1) // MTILE
    a_i8 = torch.empty((num_tiles, K // 8, MTILE, 8), dtype=torch.int8, device=device)
    # group_size=16 is unsupported; the C++ side checks before shape.
    # Buffer shapes use MTILE as a stand-in G.
    a_scale = torch.empty(
        (num_tiles, K // MTILE, MTILE), dtype=torch.float32, device=device
    )
    a_asum = torch.empty(
        (num_tiles, K // MTILE, MTILE), dtype=torch.int32, device=device
    )
    aq_ret = torch.ops._rocm_C.w4a8_act_quant_rdna2(
        x_mk, 16, a_i8, a_scale, a_asum
    )
    # kBadGroup == -6 in w4a8_sdot4_rdna2.cu
    assert aq_ret is None or aq_ret.numel() == 0, "expected no tensor for invalid group_size"


# ---------------------------------------------------------------------------
# gemm vs gptq_gemm_rdna2_prefill
# ---------------------------------------------------------------------------


@gfx1030_w4a8
@pytest.mark.parametrize("group_size", [32, 64, 128], ids=["g32", "g64", "g128"])
def test_w4a8_gemm_matches_w4a16_prefill(group_size, default_vllm_config):
    _ensure_tp_group()
    set_random_seed(0)
    M, K, N = 64, 512, 256
    G = K // group_size
    assert K % group_size == 0 and K % 8 == 0 and N % 8 == 0

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((G, N), device=device, dtype=torch.float32) + 0.01
    ).to(torch.float16)
    zeros_gn = torch.randint(0, 16, (G, N), device=device, dtype=torch.int32)

    layer = _build_layer(q_int4_kn, scales_gn, zeros_gn)
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
    )
    kernel = RDNA2W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros",
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)
    w_q = layer.qweight.data.contiguous()
    w_zp = layer.qzeros.data.contiguous()
    w_s = layer.scales.data.contiguous()

    # Fused entry: self-contained (allocates the int8 A + scales + sums
    # internally), runs act_quant + the W4A8 sdot4 GEMM, and returns a
    # populated [M, N] fp16 tensor. use_v2_format=False -> GPTQv1 zero_offset=1.
    g_idx = torch.empty(0, dtype=torch.int32, device=device)
    out = torch.ops._rocm_C.w4a8_gemm_rdna2(x_mk, w_q, w_zp, w_s, g_idx, False)
    assert out.numel() > 0, "w4a8_gemm returned an empty (ineligible) tensor"

    ref = _w4a16_reference(x_mk, q_int4_kn, scales_gn, zeros_gn, group_size)
    rel_l2 = (out.to(torch.float32) - ref.to(torch.float32)).norm() / ref.to(
        torch.float32
    ).norm()
    assert rel_l2 < 0.1, f"relative L2 error {rel_l2:.4f} >= 0.1"


@gfx1030_w4a8
def test_w4a8_dispatcher_uses_w4a16_when_env_var_unset(
    monkeypatch, default_vllm_config
):
    """With VLLM_RDNA2_W4A8_SDOT4 not set, the dispatcher must take the
    existing W4A16 path; the output of apply_weights equals the output of
    gptq_gemm_rdna2_prefill on the same packed weights.
    """
    monkeypatch.delenv("VLLM_RDNA2_W4A8_SDOT4", raising=False)
    _ensure_tp_group()
    set_random_seed(0)
    M, K, N, group_size = 64, 512, 256, 64
    G = K // group_size

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((G, N), device=device, dtype=torch.float32) + 0.01
    ).to(torch.float16)
    zeros_gn = torch.randint(0, 16, (G, N), device=device, dtype=torch.int32)

    layer = _build_layer(q_int4_kn, scales_gn, zeros_gn)
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
    )
    kernel = RDNA2W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros",
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)
    out = kernel.apply_weights(layer, x_mk)
    # The M=64, K=512, N=256 shape is W4A8-eligible (M >= W4A8_MIN_ROWS), but
    # with the env var unset self._w4a8 is False so the selector still returns
    # "prefill" and the dispatcher takes the W4A16 path.
    expected = torch.ops._rocm_C.gptq_gemm_rdna2_prefill(
        x_mk,
        layer.qweight.data,
        layer.qzeros.data,
        layer.scales.data,
        torch.empty(0, device=device, dtype=torch.int32),
        False,  # use_v2_format=False -> GPTQv1
    )
    # gptq_gemm_rdna2_prefill uses split-K CAS atomics, which are
    # order-dependent — the dispatcher path and a direct call can disagree
    # by a few fp16 ULPs. Compare with rel-L2 < 0.01 (well below the ~0.7%
    # W4A8 quantization error).
    rel_l2 = (out.to(torch.float32) - expected.to(torch.float32)).norm() / expected.to(
        torch.float32
    ).norm()
    assert rel_l2 < 0.01, f"rel-L2 {rel_l2:.4f} >= 0.01"


@gfx1030_w4a8
def test_w4a8_fused_entry_env_on_off(monkeypatch, default_vllm_config):
    """Fused-entry A/B: VLLM_RDNA2_W4A8_SDOT4 on vs off.

    For an eligible shape (M >= W4A8_MIN_ROWS, K % 32 == 0, K % group == 0,
    LDS fits) the env-on dispatch routes to the W4A8 sdot4 fast path and its
    output is close to, but not bit-identical with, the W4A16 prefill
    (rel-L2 < 0.1). For an ineligible shape (M < W4A8_MIN_ROWS, decode) the
    selector returns "prefill" regardless of the flag, so env on == env off
    byte-for-byte.
    """
    _ensure_tp_group()
    set_random_seed(0)
    K, N, group_size = 512, 256, 64
    G = K // group_size
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
    )

    for M, eligible in ((64, True), (16, False)):
        x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
            torch.float16
        )
        q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
        scales_gn = (
            0.05 * torch.rand((G, N), device=device, dtype=torch.float32) + 0.01
        ).to(torch.float16)
        zeros_gn = torch.randint(0, 16, (G, N), device=device, dtype=torch.int32)

        def _run(env_on: bool) -> torch.Tensor:
            if env_on:
                monkeypatch.setenv("VLLM_RDNA2_W4A8_SDOT4", "1")
            else:
                monkeypatch.delenv("VLLM_RDNA2_W4A8_SDOT4", raising=False)
            layer = _build_layer(q_int4_kn, scales_gn, zeros_gn)
            kernel = RDNA2W4A16LinearKernel(
                config,
                w_q_param_name="qweight",
                w_s_param_name="scales",
                w_zp_param_name="qzeros",
                w_gidx_param_name=None,
            )
            kernel.process_weights_after_loading(layer)
            return kernel.apply_weights(layer, x_mk)

        out_off = _run(env_on=False)
        out_on = _run(env_on=True)

        if eligible:
            ref = out_off.to(torch.float32)
            rel_l2 = (out_on.to(torch.float32) - ref).norm() / ref.norm()
            assert rel_l2 < 0.1, f"eligible shape: rel-L2 {rel_l2:.4f} >= 0.1"
            assert rel_l2 > 1e-6, "eligible shape: W4A8 fast path did not fire"
        else:
            assert torch.equal(out_on, out_off), (
                "ineligible shape must be byte-identical with env on/off"
            )
