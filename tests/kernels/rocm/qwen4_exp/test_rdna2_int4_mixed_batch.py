#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed prefill + decode batches on the INT4 RDNA_ATTN path.

A batch that carries both prefill and decode requests is split into the
leading decode rows and the trailing prefill rows, and each role runs its
own kernel. The split hands each role its own query range, block table and
KV lengths; getting one of those wrong makes the two roles read each
other's KV, which is silent corruption rather than a crash.

This pins the split by running the same work as two single-role batches and
requiring the mixed result to match row for row.

Run `pytest tests/kernels/rocm/qwen4_exp/test_rdna2_int4_mixed_batch.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA_ATTN is ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA_ATTN is gfx1030-only", allow_module_level=True)

from vllm.v1.attention.backends.rdna_attn import (  # noqa: E402
    RdnaAttentionImpl,
    RdnaAttentionMetadata,
)
from vllm.v1.attention.ops import fa_rdna2_backend  # noqa: E402

device = "cuda"
ATOL = 2e-2
RTOL = 2e-2


def _impl(num_heads: int, head_size: int, num_kv_heads: int) -> RdnaAttentionImpl:
    return RdnaAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int4_fp16_block",
    )


def _packed_cache(kv_cache: torch.Tensor, num_kv_heads: int, head_size: int):
    """The writer's view of the RDNA_ATTN packed cache."""
    record_bytes = head_size // 2 + 2 * (head_size // 32)
    return kv_cache.transpose(0, 1).unflatten(-1, (num_kv_heads, record_bytes))


def _metadata(
    *,
    num_actual_tokens: int,
    max_query_len: int,
    query_start_loc: list[int],
    seq_lens: list[int],
    block_table: torch.Tensor,
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_start_loc: list[int] | None,
    prefill_query_start_loc: list[int],
    max_prefill_seq_len: int,
) -> RdnaAttentionMetadata:
    def _i32(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device)

    return RdnaAttentionMetadata(
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        query_start_loc=_i32(query_start_loc),
        max_seq_len=max(seq_lens),
        seq_lens=_i32(seq_lens),
        block_table=block_table,
        slot_mapping=_i32([0] * num_actual_tokens),
        prefill_query_start_loc=_i32(prefill_query_start_loc),
        max_prefill_seq_len=max_prefill_seq_len,
        causal=True,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        decode_query_start_loc=(
            None if decode_query_start_loc is None else _i32(decode_query_start_loc)
        ),
    )


@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 1), (2, 2)])
def test_int4_mixed_batch_matches_single_role_batches(
    head_size, num_heads, num_kv_heads
):
    """A mixed batch must give each role exactly its own KV range.

    Sequence 0 decodes against 33 cached tokens; sequence 1 prefills 16
    new tokens against a different page range. If the split handed a role
    the other role's block table or KV lengths, the mixed result would
    differ from the two single-role runs.
    """
    page_size, num_pages = 16, 8
    dec_ctx, pre_tokens = 33, 16
    impl = _impl(num_heads, head_size, num_kv_heads)
    gen = torch.Generator(device=device).manual_seed(1234)

    record_bytes = head_size // 2 + 2 * (head_size // 32)
    kv_cache = torch.zeros(
        num_pages,
        2,
        page_size,
        num_kv_heads * record_bytes,
        dtype=torch.uint8,
        device=device,
    )

    # Sequence 0 occupies slots 0..32 (pages 0..2), sequence 1 slots 48..63
    # (page 3). They are far apart on purpose: a role mix-up would show.
    total = dec_ctx + pre_tokens
    key = (
        torch.randn(total, num_kv_heads, head_size, device=device, generator=gen) * 0.5
    ).half()
    value = (
        torch.randn(total, num_kv_heads, head_size, device=device, generator=gen) * 0.5
    ).half()
    slots = torch.cat(
        [
            torch.arange(dec_ctx, dtype=torch.int64, device=device),
            48 + torch.arange(pre_tokens, dtype=torch.int64, device=device),
        ]
    )
    fa_rdna2_backend.reshape_and_cache_int4_fp16_rdna2(
        key, value, _packed_cache(kv_cache, num_kv_heads, head_size), slots
    )

    block_table = torch.tensor(
        [[0, 1, 2, 7], [3, 7, 7, 7]], dtype=torch.int32, device=device
    )

    q_dec = (
        torch.randn(1, num_heads, head_size, device=device, generator=gen) * 0.5
    ).half()
    q_pre = (
        torch.randn(pre_tokens, num_heads, head_size, device=device, generator=gen)
        * 0.5
    ).half()
    q_mixed = torch.cat([q_dec, q_pre], dim=0)

    mixed_meta = _metadata(
        num_actual_tokens=1 + pre_tokens,
        max_query_len=pre_tokens,
        query_start_loc=[0, 1, 1 + pre_tokens],
        seq_lens=[dec_ctx, pre_tokens],
        block_table=block_table,
        num_decodes=1,
        num_decode_tokens=1,
        decode_query_start_loc=[0, 1],
        prefill_query_start_loc=[0, pre_tokens],
        max_prefill_seq_len=pre_tokens,
    )
    out_mixed = torch.empty_like(q_mixed)
    impl._forward_int4_fp16(q_mixed, kv_cache, mixed_meta, out_mixed)

    dec_meta = _metadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=[0, 1],
        seq_lens=[dec_ctx],
        block_table=block_table[:1],
        num_decodes=1,
        num_decode_tokens=1,
        decode_query_start_loc=[0, 1],
        prefill_query_start_loc=[0, 0],
        max_prefill_seq_len=0,
    )
    out_dec = torch.empty_like(q_dec)
    impl._forward_int4_fp16(q_dec, kv_cache, dec_meta, out_dec)

    pre_meta = _metadata(
        num_actual_tokens=pre_tokens,
        max_query_len=pre_tokens,
        query_start_loc=[0, pre_tokens],
        seq_lens=[pre_tokens],
        block_table=block_table[1:],
        num_decodes=0,
        num_decode_tokens=0,
        decode_query_start_loc=None,
        prefill_query_start_loc=[0, pre_tokens],
        max_prefill_seq_len=pre_tokens,
    )
    out_pre = torch.empty_like(q_pre)
    impl._forward_int4_fp16(q_pre, kv_cache, pre_meta, out_pre)

    torch.testing.assert_close(out_mixed[:1], out_dec, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(out_mixed[1:], out_pre, atol=ATOL, rtol=RTOL)


def test_int4_mixed_batch_decode_rows_do_not_read_the_prefill_page():
    """The decode rows must not see the prefill sequence's KV.

    This is the failure the split can cause: the roles swap their block
    table or their KV lengths, and every row is then wrong by a finite
    amount, with no error anywhere.
    """
    page_size, num_pages = 16, 8
    head_size, num_heads, num_kv_heads = 128, 4, 1
    impl = _impl(num_heads, head_size, num_kv_heads)
    gen = torch.Generator(device=device).manual_seed(99)

    record_bytes = head_size // 2 + 2 * (head_size // 32)
    kv_cache = torch.zeros(
        num_pages,
        2,
        page_size,
        num_kv_heads * record_bytes,
        dtype=torch.uint8,
        device=device,
    )
    key = (
        torch.randn(32, num_kv_heads, head_size, device=device, generator=gen) * 0.5
    ).half()
    value = (
        torch.randn(32, num_kv_heads, head_size, device=device, generator=gen) * 0.5
    ).half()
    slots = torch.cat(
        [
            torch.arange(16, dtype=torch.int64, device=device),
            48 + torch.arange(16, dtype=torch.int64, device=device),
        ]
    )
    fa_rdna2_backend.reshape_and_cache_int4_fp16_rdna2(
        key, value, _packed_cache(kv_cache, num_kv_heads, head_size), slots
    )

    block_table = torch.tensor(
        [[0, 1, 2, 7], [3, 7, 7, 7]], dtype=torch.int32, device=device
    )
    q_dec = (
        torch.randn(1, num_heads, head_size, device=device, generator=gen) * 0.5
    ).half()
    q_pre = (
        torch.randn(16, num_heads, head_size, device=device, generator=gen) * 0.5
    ).half()

    def _run(block_table_arg, seq_lens):
        meta = _metadata(
            num_actual_tokens=17,
            max_query_len=16,
            query_start_loc=[0, 1, 17],
            seq_lens=seq_lens,
            block_table=block_table_arg,
            num_decodes=1,
            num_decode_tokens=1,
            decode_query_start_loc=[0, 1],
            prefill_query_start_loc=[0, 16],
            max_prefill_seq_len=16,
        )
        out = torch.empty(17, num_heads, head_size, dtype=torch.float16, device=device)
        impl._forward_int4_fp16(torch.cat([q_dec, q_pre], dim=0), kv_cache, meta, out)
        return out

    got = _run(block_table, [16, 16])
    # The same decode row against a different sequence's page range must
    # give a different answer; if it did not, the row would not be reading
    # its own KV and the check above would be vacuous.
    swapped = block_table.clone()
    swapped[0] = torch.tensor([3, 7, 7, 7], dtype=torch.int32, device=device)
    other = _run(swapped, [16, 16])
    assert not torch.allclose(got[:1], other[:1], atol=1e-3, rtol=1e-3)
