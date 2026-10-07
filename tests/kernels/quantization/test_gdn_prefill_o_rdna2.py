#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity tests for the ROCm RDNA2 GDN chunked-prefill output kernel.

Compares ``torch.ops._rocm_C.gdn_prefill_o_rdna2``
(``csrc/rocm/gdn_prefill_o_rdna2.cu``, the HIP port of
``chunk_fwd_kernel_o``) against the Triton reference
``vllm.third_party.flash_linear_attention.ops.chunk_o.chunk_fwd_o`` on
synthetic tensors, covering chunk-boundary tails, varlen multi-sequence
batches, and several H/HV (GQA) configs.

The kernel is only built/registered for gfx1030; tests skip elsewhere.

Run ``pytest tests/kernels/quantization/test_gdn_prefill_o_rdna2.py -v``.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 GDN prefill-o kernel is ROCm-only",
                allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 GDN prefill-o kernel is gfx1030-only",
                allow_module_level=True)

import vllm._rocm_C  # noqa: F401,E402  (registers torch.ops._rocm_C.*)

from vllm.third_party.flash_linear_attention.ops.chunk_o import (  # noqa: E402
    chunk_fwd_o,
)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

device = "cuda"

K = 128  # head_k_dim; the HIP kernel requires K == 128
V = 128  # head_v_dim; the HIP kernel requires V == 128
BT = 64  # FLA_CHUNK_SIZE


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """The real pipeline applies use_qk_l2norm_in_kernel upstream;
    normalised q/k keep the gated scores in fp16 range and match the
    dot magnitudes the production kernel actually processes."""
    return F.normalize(x.float(), dim=-1).to(torch.float16)


def _make_g(B: int, T: int, H: int, gen: torch.Generator) -> torch.Tensor:
    """Chunk-local cumulative log-decay: within each 64-token chunk g is
    the inclusive cumsum of small negative increments, restarting per
    chunk. exp(g_j - g_i) stays <= 1 for j >= i, so b_A stays in fp16
    range."""
    pad = (-T) % BT
    incr = torch.rand(B, T + pad, H, device=device, dtype=torch.float32,
                      generator=gen) * 0.1
    g = (-incr).view(B, -1, BT, H).cumsum(dim=2).view(B, -1, H)
    return g[:, :T].contiguous()


def _seed_for(*parts) -> int:
    """Stable per-parametrize seed so test cases don't overlap."""
    h = 0x9E3779B1
    for p in parts:
        h = (h ^ (hash(p) & 0xFFFFFFFF) * 2654435761) & 0xFFFFFFFF
    return h


def _run_ref(q, k, v, h, g, scale, cu_seqlens, chunk_indices) -> torch.Tensor:
    """Reference: Triton ``chunk_fwd_o`` (parity baseline)."""
    return chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale,
                       cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)


def _run_hip(q, k, v, h, g, scale, cu_seqlens, chunk_offsets) -> torch.Tensor:
    """HIP: ``torch.ops._rocm_C.gdn_prefill_o_rdna2``."""
    o = torch.empty_like(v)
    torch.ops._rocm_C.gdn_prefill_o_rdna2(q, k, v, h, g, o, scale,
                                          cu_seqlens, chunk_offsets)
    return o


# --- Uniform-length (non-varlen) cases -------------------------------------

@pytest.mark.parametrize("Hg,H", [(4, 12), (2, 4), (1, 2)])
@pytest.mark.parametrize(
    "B,T",
    [
        (1, 64),     # exactly 1 chunk, no tail
        (1, 65),     # 1 full + 1 tail chunk of length 1
        (2, 100),    # multi-batch with non-aligned tail
        (2, 200),    # multi-batch with aligned
        (3, 130),    # multi-batch with aligned+tail
        (1, 4097),   # ~65 chunks per seq to catch accumulation drift
    ],
)
def test_gdn_prefill_o_rdna2_parity_uniform(Hg, H, B, T):
    """Uniform-length (non-varlen) parity across chunk-boundary tails
    and GQA configs. The T=4097 case exercises ~65 chunks per seq to
    catch accumulation-order drift across many workgroups of the same
    (i_v, i_bh) coordinate."""
    if H % Hg != 0:
        pytest.skip(f"skipping invalid GQA ratio H={H} Hg={Hg}")
    NT = (T + BT - 1) // BT
    gen = torch.Generator(device=device).manual_seed(
        _seed_for("u", Hg, H, B, T))
    q = _l2norm(torch.randn(B, T, Hg, K, device=device, dtype=torch.float32,
                             generator=gen))
    k = _l2norm(torch.randn(B, T, Hg, K, device=device, dtype=torch.float32,
                             generator=gen))
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16,
                    generator=gen)
    g_state = _make_g(B, T, H, gen)
    # h layout: 5D [B, NT, H, V, K] contiguous in K (non-varlen).
    h = torch.randn(B, NT, H, V, K, device=device, dtype=torch.float16,
                    generator=gen) * 0.1
    scale = K**-0.5

    o_ref = _run_ref(q, k, v, h, g_state, scale, None, None)
    o_hip = _run_hip(q, k, v, h, g_state, scale, None, None)
    torch.testing.assert_close(o_hip, o_ref, rtol=1e-3, atol=2e-3)


# --- Varlen cases ---------------------------------------------------------

@pytest.mark.parametrize("Hg,H", [(4, 12), (2, 4), (1, 2)])
@pytest.mark.parametrize(
    "lens",
    [
        [64, 64],          # two clean-aligned sequences
        [65, 100, 99],     # mixed-length with non-aligned tails
        [1],               # single-token sequence
        [1, 63, 1],        # odd-tail sequences
        [0, 65, 0, 130],   # zero-length seqs to verify empty-seq handling
        [130, 200, 264, 7],  # mixed batch
    ],
)
def test_gdn_prefill_o_rdna2_parity_varlen(Hg, H, lens):
    """Varlen (cu_seqlens-flattened, B==1) parity with mixed chunk-
    boundary tails and odd-length sequences. Includes a zero-length
    sequence to verify empty-seq handling and a single-token sequence
    to probe the tail-oddity fdot2 path."""
    if H % Hg != 0:
        pytest.skip(f"skipping invalid GQA ratio H={H} Hg={Hg}")
    cu_list = [0]
    for L in lens:
        cu_list.append(cu_list[-1] + L)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu, BT)
    chunk_offsets = prepare_chunk_offsets(cu, BT)
    T = cu_list[-1]
    gen = torch.Generator(device=device).manual_seed(
        _seed_for("v", Hg, H, T, tuple(lens)))

    q = _l2norm(torch.randn(1, T, Hg, K, device=device, dtype=torch.float32,
                             generator=gen))
    k = _l2norm(torch.randn(1, T, Hg, K, device=device, dtype=torch.float32,
                             generator=gen))
    v = torch.randn(1, T, H, V, device=device, dtype=torch.float16,
                    generator=gen)
    g_state = _make_g(1, T, H, gen)
    NC = chunk_indices.shape[0]
    h = torch.randn(1, NC, H, V, K, device=device, dtype=torch.float16,
                    generator=gen) * 0.1
    scale = K**-0.5

    o_ref = _run_ref(q, k, v, h, g_state, scale, cu, chunk_indices)
    o_hip = _run_hip(q, k, v, h, g_state, scale, cu, chunk_offsets)
    torch.testing.assert_close(o_hip, o_ref, rtol=1e-3, atol=2e-3)


# --- Custom-scale sweep ---------------------------------------------------

@pytest.mark.parametrize("Hg,H", [(4, 12), (2, 4)])
@pytest.mark.parametrize("scale", [K**-0.5, 0.3, 0.05])
def test_gdn_prefill_o_rdna2_parity_uniform_custom_scale(Hg, H, scale):
    """The scale is applied twice (`b_o*scale + (A @ v)*scale`); sweep
    values to make sure both terms scale correctly together."""
    if H % Hg != 0:
        pytest.skip(f"skipping invalid GQA ratio H={H} Hg={Hg}")
    B, T = 2, 130
    NT = (T + BT - 1) // BT
    gen = torch.Generator(device=device).manual_seed(
        _seed_for("s", Hg, H, int(scale * 1000)))
    q = _l2norm(torch.randn(B, T, Hg, K, device=device, dtype=torch.float32,
                             generator=gen))
    k = _l2norm(torch.randn(B, T, Hg, K, device=device, dtype=torch.float32,
                             generator=gen))
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16,
                    generator=gen)
    g_state = _make_g(B, T, H, gen)
    h = torch.randn(B, NT, H, V, K, device=device, dtype=torch.float16,
                    generator=gen) * 0.1

    o_ref = _run_ref(q, k, v, h, g_state, scale, None, None)
    o_hip = _run_hip(q, k, v, h, g_state, scale, None, None)
    torch.testing.assert_close(o_hip, o_ref, rtol=1e-3, atol=2e-3)
