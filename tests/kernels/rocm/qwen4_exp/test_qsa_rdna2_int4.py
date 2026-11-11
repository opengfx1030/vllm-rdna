#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed INT4 + FP16 block-scale KV cache: the fused HIP QSA decode.

The packed ``int4_fp16_block`` record has a different layout from an fp16
cache, so the fused HIP QSA decode reads it natively: the payload and the
scale tail are unpacked into the same fp16 LDS tile the fp16 path fills.

The oracle here is the *materialized* cache: the packed bytes are decoded
back to fp16 in PyTorch and the fp16 fused decode is run over that. The two
must agree, because reading a packed record as the wrong format does not
fault -- it produces finite, plausible, silently wrong numbers.

Run `pytest tests/kernels/rocm/qwen4_exp/test_qsa_rdna2_int4.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 QSA kernels are ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 QSA kernels are gfx1030-only", allow_module_level=True)

# The Qwen4Exp AMD modules import each other: `qsa` pulls in `model` and
# `model` pulls in `qsa`. The model registry loads `model` first, so the
# test does the same before reaching for the layout helpers in `qsa`.
import vllm.models.qwen4_exp.amd.model  # noqa: E402, F401
from vllm.models.qwen4_exp.amd.ops import qsa_rdna2  # noqa: E402
from vllm.v1.attention.ops import fa_rdna2_backend  # noqa: E402

device = "cuda"
ATOL = 2e-2
RTOL = 2e-2


def _int4_record_bytes(head_dim: int) -> int:
    return head_dim // 2 + 2 * (head_dim // 32)


def _pack_kv(key, value, num_pages, page_size, kv_heads, head_dim, slots):
    """Write fp16 K/V into a packed cache with the native writer.

    The cache is ``[2, pages, slots, heads, record_bytes]`` of uint8; the
    writer quantizes each 32-channel group and packs the payload and the
    scale words into one record.
    """
    record_bytes = _int4_record_bytes(head_dim)
    cache = torch.zeros(
        2,
        num_pages,
        page_size,
        kv_heads,
        record_bytes,
        dtype=torch.uint8,
        device=key.device,
    )
    fa_rdna2_backend.reshape_and_cache_int4_fp16_rdna2(key, value, cache, slots)
    return cache


def _int4_read_views(cache, head_dim):
    """``[pages, heads, slots, payload/scale]`` views of the packed cache."""
    data_bytes = head_dim // 2
    sides = cache.unbind(0)
    payloads = tuple(side[..., :data_bytes].permute(0, 2, 1, 3) for side in sides)
    scales = tuple(
        side[..., data_bytes:].view(torch.uint16).permute(0, 2, 1, 3) for side in sides
    )
    return payloads[0], payloads[1], scales[0], scales[1]


def _scale_words_from_bytes(tail):
    """Decode the scale words straight out of the uint8 bytes.

    The oracle deliberately does not use the ``uint16`` view the kernel path
    uses: if that view addressed the wrong bytes, an oracle built on it would
    agree with the kernel and hide the defect.
    """
    lo = tail[..., 0::2].to(torch.int32)
    hi = tail[..., 1::2].to(torch.int32)
    return lo | (hi << 8)


def _materialize(cache, head_dim):
    """Decode the packed records back to fp16 -- the oracle for the kernel.

    A record holds ``head_dim // 2`` payload bytes (two INT4 codes per byte,
    low nibble first) followed by ``head_dim // 32`` little-endian uint16
    scale words. A scale word is ``(fp16_bits(scale) & 0xfff0) | zp``: the
    high 12 bits are an fp16 scale, the low 4 bits a zero-point. One scale
    covers 32 contiguous channels, and the value is ``(code - zp) * scale``.
    """
    data_bytes = head_dim // 2
    out = []
    for side in cache.unbind(0):
        payload = side[..., :data_bytes]
        words = _scale_words_from_bytes(side[..., data_bytes:])
        codes = torch.stack([payload & 0xF, payload >> 4], dim=-1).reshape(
            *payload.shape[:-1], head_dim
        )
        # The scale word is raw bits, so it is masked in int32: ROCm has no
        # bitwise kernels for uint16.
        word_bits = words
        scales = (word_bits & 0xFFF0).to(torch.uint16).view(torch.float16).float()
        zps = word_bits & 0xF
        values = codes.to(torch.int32).float() - zps.repeat_interleave(32, dim=-1)
        values = values * scales.repeat_interleave(32, dim=-1)
        out.append(values.half())
    return out


def _sparse_inputs(
    rows, heads, kv_heads, head_dim, topk, page_size, num_pages, num_requests, seed
):
    """A paged K/V plus an index selection, in the fp16 layout."""
    gen = torch.Generator(device=device).manual_seed(seed)
    table_width = max(1, num_pages // num_requests)
    q = (torch.randn(rows, heads, head_dim, device=device, generator=gen) * 0.5).half()
    k_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim, device=device, generator=gen
    ).half()
    v_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim, device=device, generator=gen
    ).half()
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        num_requests, table_width
    )
    token_to_req = torch.randint(
        0, num_requests, (rows,), dtype=torch.int32, device=device, generator=gen
    )
    logical = torch.randint(
        0,
        table_width * page_size,
        (rows, topk),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    if topk > 1:
        logical[:, -1] = -1
    return q, k_cache, v_cache, logical, block_table, token_to_req


@pytest.mark.parametrize("heads,kv_heads", [(4, 1), (2, 2)])
@pytest.mark.parametrize("topk", [8, 64, 129])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_int4_sparse_paged_attention_matches_materialized(
    heads, kv_heads, topk, head_dim
):
    """The packed decode must match the fp16 decode over the same bytes.

    The fp16 arm runs over the *materialized* cache, i.e. the packed record
    decoded back to fp16. Comparing against the original fp16 input would
    only measure the quantizer; comparing against the materialized values
    measures the reader, which is where a wrong-format read would show up.
    """
    rows, page_size, num_pages, num_requests = 24, 16, 64, 4
    q, k_cache, v_cache, logical, block_table, token_to_req = _sparse_inputs(
        rows,
        heads,
        kv_heads,
        head_dim,
        topk,
        page_size,
        num_pages,
        num_requests,
        seed=rows * 7 + topk + head_dim,
    )

    # The writer takes [tokens, heads, dim] and a slot per token. Every
    # (page, slot, head) of the cache is written so the oracle sees the
    # whole cache, not just the selected tokens.
    flat_k = k_cache.reshape(-1, kv_heads, head_dim)
    flat_v = v_cache.reshape(-1, kv_heads, head_dim)
    slots = torch.arange(flat_k.shape[0], dtype=torch.int64, device=device)
    packed = _pack_kv(flat_k, flat_v, num_pages, page_size, kv_heads, head_dim, slots)

    mat_k, mat_v = _materialize(packed, head_dim)
    mat_k = mat_k.reshape(num_pages, page_size, kv_heads, head_dim)
    mat_v = mat_v.reshape(num_pages, page_size, kv_heads, head_dim)

    k_payload, v_payload, k_scale, v_scale = _int4_read_views(packed, head_dim)
    got = qsa_rdna2.qsa_sparse_paged_attention_int4(
        q,
        k_payload,
        v_payload,
        k_scale,
        v_scale,
        logical,
        block_table,
        token_to_req,
        None,
    )
    want = qsa_rdna2.qsa_sparse_paged_attention(
        q, mat_k, mat_v, logical, block_table, token_to_req, None
    )
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_int4_sparse_paged_attention_op_is_registered():
    assert any(
        str(s).startswith("_rocm_C::qsa_sparse_paged_attention_int4_rdna2")
        for s in torch._C._jit_get_all_schemas()
    )


def test_int4_kv_writer_op_is_reachable():
    """The QSA backend's own write path must have a Python entry point.

    The backend writes the packed cache through ``_custom_ops`` rather than
    the FA backend's helper, so a missing export here would surface as an
    ``AttributeError`` in the middle of a serving run.
    """
    from vllm import _custom_ops

    assert callable(_custom_ops.reshape_and_cache_int4_fp16_rdna2)


@pytest.mark.parametrize("backend", ["rdna_attn", "qsa"])
def test_int4_layout_views_round_trip_the_packed_cache(backend):
    """Each backend's view of a packed cache must decode to the same values.

    The framework hands both backends the same logical per-layer view,
    ``[blocks, 2, slots, heads*record_bytes]``, so the shared helper serves
    both. A view that pointed at the wrong bytes would still be a valid
    tensor, so it is checked against the oracle instead of against its own
    shape.
    """
    from vllm.v1.attention.ops.int4_fp16_kv_layout import (
        int4_fp16_cache_view,
        int4_fp16_read_views,
    )

    head_dim, kv_heads, pages, slots = 256, 2, 4, 8
    record_bytes = _int4_record_bytes(head_dim)
    kv_cache = torch.zeros(
        pages, 2, slots, kv_heads * record_bytes, dtype=torch.uint8, device=device
    )
    cache = int4_fp16_cache_view(kv_cache, kv_heads, head_dim)
    assert cache.shape == (2, pages, slots, kv_heads, record_bytes)

    gen = torch.Generator(device=device).manual_seed(11)
    key = (
        torch.randn(pages * slots, kv_heads, head_dim, device=device, generator=gen)
    ).half()
    value = (
        torch.randn(pages * slots, kv_heads, head_dim, device=device, generator=gen)
    ).half()
    slot_mapping = torch.arange(pages * slots, dtype=torch.int64, device=device)
    from vllm import _custom_ops

    _custom_ops.reshape_and_cache_int4_fp16_rdna2(key, value, cache, slot_mapping)

    # The oracle reads the same physical bytes.
    mat_k, mat_v = _materialize(cache, head_dim)
    mat_k = mat_k.reshape(pages, slots, kv_heads, head_dim)
    mat_v = mat_v.reshape(pages, slots, kv_heads, head_dim)

    k_payload, v_payload, k_scale, v_scale = int4_fp16_read_views(cache, head_dim)
    assert k_payload.shape == (pages, kv_heads, slots, head_dim // 2)
    assert k_scale.shape == (pages, kv_heads, slots, head_dim // 32)

    def decode(payload, scale):
        codes = torch.stack([payload & 0xF, payload >> 4], dim=-1).reshape(
            *payload.shape[:-1], head_dim
        )
        word_bits = scale.to(torch.int32)
        scales = (word_bits & 0xFFF0).to(torch.uint16).view(torch.float16).float()
        zps = word_bits & 0xF
        values = codes.to(torch.int32).float() - zps.repeat_interleave(32, dim=-1)
        return (values * scales.repeat_interleave(32, dim=-1)).half()

    torch.testing.assert_close(
        decode(k_payload, k_scale), mat_k.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        decode(v_payload, v_scale), mat_v.permute(0, 2, 1, 3), atol=1e-3, rtol=1e-3
    )
    assert backend in ("rdna_attn", "qsa")


def test_qsa_backend_publishes_the_packed_cache_shape():
    """The QSA backend must size its INT4 page for both K and V.

    The framework derives the per-layer view from the spec, so a spec that
    forgets the K/V head slot would allocate half the bytes the packed
    records need.
    """
    from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.v1.attention.backends.rdna_attn import RdnaAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    head_dim, kv_heads, block_size = 256, 1, 16
    record_bytes = _int4_record_bytes(head_dim)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_size=head_dim,
        head_size_v=head_dim,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT4_FP16_BLOCK,
    )
    for backend in (RdnaAttentionBackend, Qwen4ExpQSAFlashAttentionBackend):
        customized = backend.customize_spec(spec)
        assert customized.num_head_slots == 2, backend.get_name()
        assert customized.state_content_bytes == kv_heads * record_bytes
        # A page holds both sides for every token of the block: two head
        # slots of one side's records.
        assert customized.page_size_bytes == (block_size * 2 * kv_heads * record_bytes)


def test_resolve_main_kv_layout_identifies_the_packed_cache():
    """The QSA backend must identify the packed layout, not guess it."""
    from vllm.models.qwen4_exp.amd.qsa import resolve_main_kv_layout

    head_dim, kv_heads, pages, slots = 256, 1, 4, 8
    record_bytes = _int4_record_bytes(head_dim)
    kv_cache = torch.zeros(
        pages, 2, slots, kv_heads * record_bytes, dtype=torch.uint8, device=device
    )
    layout = resolve_main_kv_layout(kv_cache, "int4_fp16_block", kv_heads, head_dim)
    assert layout.is_int4
    assert layout.kind == "int4_fp16_block"
    assert layout.k_scale is not None and layout.v_scale is not None
    assert layout.key_cache.shape == (pages, kv_heads, slots, head_dim // 2)
    assert layout.value_cache.shape == (pages, kv_heads, slots, head_dim // 2)
    assert layout.k_scale.shape == (pages, kv_heads, slots, head_dim // 32)
    assert layout.v_scale.shape == (pages, kv_heads, slots, head_dim // 32)


def test_int4_sparse_paged_attention_requires_the_native_kernel(monkeypatch):
    """A packed cache has no Triton reader: the path must refuse, not fall back."""
    from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops

    monkeypatch.setattr(qsa_rdna2, "qsa_use_rdna2_int4", lambda: False)
    with pytest.raises(NotImplementedError):
        qsa_ops.qsa_sparse_paged_attention(
            torch.zeros(1, 4, 128, device=device, dtype=torch.float16),
            torch.zeros(1, 1, 16, 64, device=device, dtype=torch.uint8),
            torch.zeros(1, 1, 16, 64, device=device, dtype=torch.uint8),
            torch.zeros(1, 8, device=device, dtype=torch.int32),
            torch.zeros(1, 1, device=device, dtype=torch.int32),
            torch.zeros(1, device=device, dtype=torch.int32),
            None,
            int4_scales=(
                torch.zeros(1, 1, 16, 4, device=device, dtype=torch.uint16),
                torch.zeros(1, 1, 16, 4, device=device, dtype=torch.uint16),
            ),
        )


def test_int4_sparse_paged_attention_rejects_fp16_cache():
    """The fp16 entry point must keep refusing a non-fp16 cache."""
    with pytest.raises(RuntimeError):
        qsa_rdna2.qsa_sparse_paged_attention(
            torch.zeros(1, 4, 128, device=device, dtype=torch.float16),
            torch.zeros(1, 1, 16, 64, device=device, dtype=torch.uint8),
            torch.zeros(1, 1, 16, 64, device=device, dtype=torch.uint8),
            torch.zeros(1, 8, device=device, dtype=torch.int32),
            torch.zeros(1, 1, device=device, dtype=torch.int32),
            torch.zeros(1, device=device, dtype=torch.int32),
            None,
        )
