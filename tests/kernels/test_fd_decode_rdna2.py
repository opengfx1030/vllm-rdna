# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness test for the RDNA2 int8 flash-decode kernel (fd_decode).

Builds a synthetic per-token-head int8 KV cache in the exact layout the
triton_attn backend uses (packed int8 rows with inline float32 scale bytes,
split K/V views, strided scale caches) and compares the kernel output
against a pure-torch reference (dequantize + standard softmax attention).
"""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.fixture(scope="module", autouse=True)
def skip_if_not_gfx10x():
    if not current_platform.is_rocm():
        pytest.skip("ROCm only")
    try:
        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            pytest.skip("RDNA2 (gfx10x) only")
    except ImportError:
        pytest.skip("on_gfx10x not available")


def _build_cache(n, kvh, hs, bs, n_blocks, device):
    """Quantized K/V + packed cache + split views + scale caches."""
    padded_hs = hs + 4
    k_ref = torch.randn(n, kvh, hs, dtype=torch.float16, device=device) * 0.5
    v_ref = torch.randn(n, kvh, hs, dtype=torch.float16, device=device) * 0.5
    k_scale = (k_ref.abs().amax(dim=-1) / 127.0).float() + 1e-6
    v_scale = (v_ref.abs().amax(dim=-1) / 127.0).float() + 1e-6
    k_i8 = (k_ref / k_scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    v_i8 = (v_ref / v_scale[..., None]).round().clamp(-127, 127).to(torch.int8)

    packed = torch.zeros(n_blocks, kvh, bs, 2 * padded_hs, dtype=torch.int8,
                         device=device)
    kv_view = packed.transpose(1, 2)
    k_view, v_view = kv_view.split(padded_hs, dim=-1)
    n_used = (n + bs - 1) // bs
    for i in range(n_used):
        lo, hi = i * bs, min((i + 1) * bs, n)
        k_view[i, : hi - lo, :, :hs] = k_i8[lo:hi]
        v_view[i, : hi - lo, :, :hs] = v_i8[lo:hi]

    raw = packed.untyped_storage()
    base_f32 = torch.tensor([], dtype=torch.float32, device=device).set_(raw)

    def f32_units(elements: int) -> int:
        return elements * packed.element_size() // 4

    strides = (f32_units(packed.stride(0)), f32_units(packed.stride(2)),
               f32_units(packed.stride(1)))
    k_scale_cache = torch.as_strided(
        base_f32, size=(n_blocks, bs, kvh), stride=strides,
        storage_offset=f32_units(hs))
    v_scale_cache = torch.as_strided(
        base_f32, size=(n_blocks, bs, kvh), stride=strides,
        storage_offset=f32_units(padded_hs + hs))
    for i in range(n_used):
        lo, hi = i * bs, min((i + 1) * bs, n)
        k_scale_cache[i, : hi - lo, :] = k_scale[lo:hi]
        v_scale_cache[i, : hi - lo, :] = v_scale[lo:hi]

    k_deq = k_i8.float() * k_scale[..., None]
    v_deq = v_i8.float() * v_scale[..., None]
    return k_view, v_view, k_scale_cache, v_scale_cache, k_deq, v_deq


@pytest.mark.parametrize("n", [1, 33, 500, 1024])
def test_fd_decode_vs_reference(n):
    from vllm.model_executor.layers.attention.rdna2_fd_decode import fd_decode

    dev = torch.device("cuda:0")
    nq, kvh, hs, bs = 12, 2, 256, 16
    gqa = nq // kvh
    n_blocks = (n + bs - 1) // bs
    max_bt = n_blocks + 4
    scale = hs ** -0.5

    q = torch.randn(1, nq, hs, dtype=torch.float16, device=dev)
    k_view, v_view, ksc, vsc, k_deq, v_deq = _build_cache(
        n, kvh, hs, bs, n_blocks, dev)

    kv_of_head = torch.arange(nq, device=dev) // gqa
    k_sel = k_deq.permute(1, 0, 2)[kv_of_head]
    v_sel = v_deq.permute(1, 0, 2)[kv_of_head]
    s_ref = torch.einsum("qd,qtd->qt", q[0].float(), k_sel) * scale
    p = torch.softmax(s_ref, dim=-1)
    o_ref = torch.einsum("qt,qtd->qd", p, v_sel).to(torch.float16)

    block_table = torch.arange(max_bt, dtype=torch.int32,
                               device=dev).view(1, -1)
    seqused_k = torch.tensor([n], dtype=torch.int32, device=dev)

    out = fd_decode(q, k_view, v_view, ksc, vsc, block_table, seqused_k,
                    scale, BS=bs)

    diff = (out.float() - o_ref.float()).abs()
    assert diff.max().item() < 0.05, (
        f"n={n}: max abs diff {diff.max().item():.4f}")
