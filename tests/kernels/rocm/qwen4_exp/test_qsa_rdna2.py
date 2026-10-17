#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S1/S2/S4 tests for the opt-in RDNA2 HIP QSA decode kernels (gfx1030).

Compares the HIP ops in ``csrc/rocm/qsa_rdna2.cu`` against the Triton
reference kernels in ``vllm/models/qwen4_exp/amd/ops/qsa.py``:

- ``qsa_store_cache_rows_rdna2``   vs ``_store_qsa_rows_kernel``
- ``qsa_compress_groups_rdna2``    vs ``_compress_qsa_groups_kernel``
- ``qsa_mqa_paged_rdna2``          vs ``_qsa_mqa_paged_kernel``
- ``qsa_expand_indices_rdna2``      vs ``_expand_qsa_indices_kernel``
- ``qsa_sparse_paged_attention_rdna2`` vs ``_qsa_sparse_paged_gqa_splitk_kernel``
  + ``_qsa_merge_splitk_kernel``

The HIP path is opt-in (``VLLM_RDNA_QSA_HIP=1``); these tests call the HIP
wrappers directly and pin the Triton side off with a monkeypatched gate, so
they pass in either ambient configuration. The production default stays
Triton.

Run `pytest tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 QSA kernels are ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx10x  # noqa: E402

if not on_gfx10x():
    pytest.skip("RDNA2 QSA kernels are gfx1030-only", allow_module_level=True)

from vllm.triton_utils import triton  # noqa: E402
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops  # noqa: E402
from vllm.models.qwen4_exp.amd.ops import qsa_rdna2  # noqa: E402
from vllm.models.qwen4_exp.amd.ops.qsa import (  # noqa: E402
    _store_qsa_rows_kernel,
)

device = "cuda"
ATOL = 1e-3
RTOL = 1e-3


def _op(name: str) -> bool:
    return any(str(s).startswith(name) for s in torch._C._jit_get_all_schemas())


HIP_QSA_OPS = [
    "_rocm_C::qsa_store_cache_rows_rdna2",
    "_rocm_C::qsa_compress_groups_rdna2",
    "_rocm_C::qsa_mqa_paged_rdna2",
    "_rocm_C::qsa_expand_indices_rdna2",
    "_rocm_C::qsa_sparse_paged_attention_rdna2",
]


@pytest.fixture
def triton_gate(monkeypatch):
    """Force the dispatcher onto the Triton reference path."""
    monkeypatch.setattr(qsa_rdna2, "qsa_use_rdna2", lambda dtype=None: False)


@pytest.mark.parametrize("op", HIP_QSA_OPS)
def test_qsa_op_registered(op):
    assert _op(op), f"{op} not registered (rebuild _rocm_C.abi3.so)"

def test_qsa_env_gate_defaults_on():
    from vllm.envs import VLLM_RDNA_QSA_HIP

    assert VLLM_RDNA_QSA_HIP is True, "HIP QSA path is on by default (S6)"

def test_qsa_dispatcher_env_gate(monkeypatch):
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", False)
    assert qsa_rdna2.qsa_use_rdna2() is False
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    assert qsa_rdna2.qsa_use_rdna2() is True
    assert qsa_rdna2.qsa_use_rdna2(torch.float16) is True
    # The HIP kernels are fp16-only, so a bf16 model keeps Triton.
    assert qsa_rdna2.qsa_use_rdna2(torch.bfloat16) is False
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: False)
    assert qsa_rdna2.qsa_use_rdna2() is False


def test_mqa_paged_routes_to_hip_only_for_fp16(monkeypatch):
    """The dispatcher sends fp16 to the HIP op and bf16 to Triton."""
    seen: list[torch.dtype] = []

    def spy(q, k_cache, page_table, token_to_req, query_positions,
            sequence_lengths, compress_ratio, num_columns, score_scale):
        seen.append(q.dtype)
        return (
            torch.empty(q.shape[0], num_columns, dtype=torch.float32, device=device),
            torch.empty(q.shape[0], dtype=torch.int32, device=device),
        )

    monkeypatch.setattr(qsa_rdna2, "qsa_mqa_paged", spy)
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)

    for dt in (torch.float16, torch.bfloat16):
        q = torch.randn(4, 4, 128, dtype=dt, device=device)
        cache = torch.randn(8, 64, 1, 128, dtype=dt, device=device)
        table = torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
        req = torch.zeros(4, dtype=torch.int32, device=device)
        pos = torch.arange(4, dtype=torch.int64, device=device)
        seq = torch.full((1,), 512, dtype=torch.int32, device=device)
        qsa_ops.qsa_mqa_paged(q, cache, table, req, pos, seq, 4, 512)

    assert seen == [torch.float16], seen


# ---------------------------------------------------------------------------
# S1 parity — qsa_store_cache_rows_rdna2
# ---------------------------------------------------------------------------


def _store_rows_reference(rows, slots, cache, page_size, width):
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slots,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=page_size,
        WIDTH=width,
        BLOCK_D=triton.next_power_of_2(width),
        num_warps=4,
    )
    return cache


@pytest.mark.parametrize("num_rows", [1, 8, 64, 512])
@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("page_size", [16, 256])
@pytest.mark.parametrize("num_blocks", [1, 4, 64])
def test_store_cache_rows_parity(num_rows, width, page_size, num_blocks):
    torch.manual_seed(num_rows + width + page_size + num_blocks)
    rows = torch.randn(num_rows, width, dtype=torch.float16, device=device)
    # Cover every (block, token) decode with a page-aligned random permutation.
    # Unique slots only: with duplicate slots the write order is undefined and
    # the two kernels may resolve the race differently. Rows beyond the cache
    # capacity become NULL_BLOCK_ID rows, as do every fifth slot.
    capacity = num_blocks * page_size
    num_live = min(num_rows, capacity)
    slots = torch.full((num_rows,), -1, dtype=torch.int64, device=device)
    slots[:num_live] = (
        torch.randperm(capacity, device=device)[:num_live].to(torch.int64).contiguous()
    )
    slots[::5] = -1

    ref = torch.zeros(
        num_blocks, page_size, 1, width, dtype=torch.float16, device=device
    )
    hip = torch.zeros_like(ref)
    _store_rows_reference(rows, slots, ref, page_size, width)
    qsa_rdna2.qsa_store_cache_rows(rows, slots, hip, page_size, width)
    torch.cuda.synchronize()

    maxdiff = (ref - hip).abs().max().item()
    print(f"store_cache_rows atol=0 maxdiff={maxdiff} rows={num_rows}")
    assert torch.equal(ref, hip), f"maxdiff={maxdiff}"


def test_store_cache_rows_matches_scatter_semantics():
    """Independent of the Triton kernel: rows land at slots, NULL rows no-op."""
    num_rows, width, page_size, num_blocks = 37, 64, 8, 5
    torch.manual_seed(7)
    rows = torch.randn(num_rows, width, dtype=torch.float16, device=device)
    slots = (
        torch.randperm(num_blocks * page_size, device=device)[:num_rows]
        .to(torch.int64)
        .contiguous()
    )
    slots[::4] = -1
    expected = torch.full(
        (num_blocks, page_size, 1, width), -1.0, dtype=torch.float16, device=device
    )
    for row in range(num_rows):
        slot = int(slots[row])
        if slot >= 0:
            expected[slot // page_size, slot % page_size, 0] = rows[row]
    hip = torch.full_like(expected, -1.0)
    qsa_rdna2.qsa_store_cache_rows(rows, slots, hip, page_size, width)
    torch.cuda.synchronize()
    assert torch.equal(expected, hip)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_store_cache_rows_all_element_sizes_and_strides(dtype, strided):
    """Cover the 2/4/8-byte paths and non-unit width strides.

    The server stores fp16 keys, fp32 values and int64 RoPE positions through
    this helper, and the position rows arrive as a transposed view whose last
    dim stride is not 1.
    """
    num_rows, width, page_size, num_blocks = 24, 16, 4, 6
    torch.manual_seed(11)
    if dtype == torch.int64:
        rows = torch.randint(0, 4096, (num_rows, width), device=device)
        rows = rows.reshape(width, num_rows).t() if strided else rows
        fill = -7
    else:
        base = torch.randn(width, num_rows, dtype=dtype, device=device)
        rows = base.t() if strided else base.t().contiguous()
        fill = -7.0

    capacity = num_blocks * page_size
    slots = torch.full((num_rows,), -1, dtype=torch.int64, device=device)
    slots[:num_rows] = torch.randperm(capacity, device=device)[:num_rows]
    slots[::4] = -1

    # Interleaved width stride exercises the stride-aware scalar fallback.
    interleave = 2 if strided else 1
    backing = torch.full(
        (num_blocks, page_size, 1, width * interleave),
        fill,
        dtype=dtype,
        device=device,
    )
    cache = backing[:, :, :, ::interleave]
    expected = cache.clone()
    for row in range(num_rows):
        slot = int(slots[row])
        if slot >= 0:
            expected[slot // page_size, slot % page_size, 0] = rows[row]

    qsa_rdna2.qsa_store_cache_rows(rows, slots, cache, page_size, width)
    torch.cuda.synchronize()
    assert torch.equal(expected, cache), f"rows stride {rows.stride()}"
    if interleave > 1:
        assert bool((backing[..., 1::2] == fill).all()), "clobbered interleaved columns"


# ---------------------------------------------------------------------------
# S1 parity — qsa_compress_groups_rdna2
# ---------------------------------------------------------------------------


def _compress_inputs(rows, head_dim, ratio, num_requests, num_blocks, load_rope):
    """Build a compressor ring with half the state slots populated."""
    state_size = max(ratio * 2, 2)
    torch.manual_seed(rows + head_dim + ratio + num_requests + int(load_rope))
    raw_keys = torch.randn(rows, 1, head_dim, dtype=torch.float16, device=device)
    raw_positions = torch.randint(
        0, 64, (rows, 1, 3), dtype=torch.int64, device=device
    )
    state_cache = torch.zeros(
        num_blocks, state_size, 1, head_dim, dtype=torch.float16, device=device
    )
    state_cache.normal_(0, 0.5)
    rope_cache = torch.randint(
        0, 512, (num_blocks, state_size, 1, 3), dtype=torch.int64, device=device
    )
    # Half the requests fall back to the raw rows (NULL_BLOCK_ID).
    state_table = torch.full(
        (num_requests, 1), -1, dtype=torch.int32, device=device
    )
    state_table[::2, 0] = torch.arange(
        (num_requests + 1) // 2, dtype=torch.int32, device=device
    ) % num_blocks

    per_request = max(rows // num_requests, 1)
    token_to_req = (torch.arange(rows, device=device) // per_request).clamp_max(
        num_requests - 1
    ).to(torch.int32)
    query_start_loc = torch.empty(num_requests + 1, dtype=torch.int32, device=device)
    query_start_loc[0] = 0
    query_start_loc[1:] = torch.arange(
        1, num_requests + 1, device=device
    ) * per_request
    query_start_loc[-1] = rows
    # Positions advance within each request so chunk_start_position moves too.
    logical_positions = (
        torch.arange(rows, device=device) - query_start_loc[token_to_req.long()]
        + ratio
    ).to(torch.int64)
    logical_positions[::9] = -1  # unmapped rows
    compressed_slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    compressed_slots[::3] = torch.arange(
        (rows + 2) // 3, dtype=torch.int64, device=device
    )
    return (
        raw_keys,
        raw_positions,
        state_cache,
        rope_cache if load_rope else None,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    )


@pytest.mark.parametrize("rows", [1, 8, 64])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
@pytest.mark.parametrize("num_requests", [1, 4])
@pytest.mark.parametrize("load_rope", [False, True])
def test_compress_groups_parity(
    triton_gate, rows, head_dim, ratio, num_requests, load_rope
):
    num_blocks = max(num_requests, 2)
    args = _compress_inputs(
        rows, head_dim, ratio, num_requests, num_blocks, load_rope
    )
    (
        raw_keys,
        raw_positions,
        state_cache,
        rope_cache,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    ) = args

    ref_pooled, ref_first = qsa_ops.qsa_compress_groups_with_ratio(
        raw_keys,
        raw_positions,
        state_cache,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        ratio,
        rope_cache,
    )
    hip_pooled, hip_first = qsa_rdna2.qsa_compress_groups_with_ratio_compat(
        raw_keys,
        raw_positions,
        state_cache,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        ratio,
        rope_cache,
    )
    torch.cuda.synchronize()

    assert hip_pooled.shape == ref_pooled.shape
    maxdiff = (ref_pooled - hip_pooled).abs().max().item()
    print(
        f"compress_groups atol={ATOL} maxdiff={maxdiff} rows={rows} "
        f"head_dim={head_dim} ratio={ratio} load_rope={load_rope}"
    )
    torch.testing.assert_close(hip_pooled, ref_pooled, atol=ATOL, rtol=RTOL)
    assert torch.equal(hip_first, ref_first), (
        f"first_positions mismatch: {(hip_first != ref_first).sum().item()} rows"
    )


def test_compress_groups_all_null_state_table(triton_gate):
    """Every request falls back to raw rows (the state_ok == False branch)."""
    rows, head_dim, ratio, num_requests = 16, 64, 4, 2
    (
        raw_keys,
        raw_positions,
        state_cache,
        rope_cache,
        _,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    ) = _compress_inputs(rows, head_dim, ratio, num_requests, 2, True)
    state_table = torch.full((num_requests, 1), -1, dtype=torch.int32, device=device)

    ref_pooled, ref_first = qsa_ops.qsa_compress_groups_with_ratio(
        raw_keys, raw_positions, state_cache, state_table, token_to_req,
        query_start_loc, logical_positions, compressed_slots, ratio, rope_cache,
    )
    hip_pooled, hip_first = qsa_rdna2.qsa_compress_groups_with_ratio_compat(
        raw_keys, raw_positions, state_cache, state_table, token_to_req,
        query_start_loc, logical_positions, compressed_slots, ratio, rope_cache,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(hip_pooled, ref_pooled, atol=ATOL, rtol=RTOL)
    assert torch.equal(hip_first, ref_first)


# ---------------------------------------------------------------------------
# S1 parity — qsa_mqa_paged_rdna2
# ---------------------------------------------------------------------------


def _mqa_inputs(
    rows, heads, head_dim, num_pages, page_size, table_width, num_requests, ratio
):
    torch.manual_seed(rows + heads + head_dim + num_pages + page_size + ratio)
    q = torch.randn(rows, heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.float16, device=device
    )
    table = torch.randperm(num_pages, device=device)[: num_requests * table_width]
    table = table.to(torch.int32).reshape(num_requests, table_width)
    token_to_req = (
        torch.arange(rows, device=device) % num_requests
    ).to(torch.int32)
    token_to_req[::11] = -1  # padding rows
    query_positions = torch.arange(rows, dtype=torch.int64, device=device) + ratio
    query_positions[::13] = -1
    sequence_lengths = torch.tensor(
        [table_width * page_size * ratio, page_size * ratio * 2][:num_requests],
        dtype=torch.int32,
        device=device,
    )
    return q, k_cache, table, token_to_req, query_positions, sequence_lengths


@pytest.mark.parametrize(
    "rows,heads,head_dim,num_pages,page_size,table_width,num_requests,ratio",
    [
        # Model geometry: 4 indexer heads x 128, ratio 4.
        (64, 4, 128, 48, 64, 24, 2, 4),
        # Arbitrary head counts must work (the DeepSeek indexer kernel only
        # supported H == 64, D == 128 which is exactly what this replaces).
        (8, 1, 128, 16, 32, 8, 1, 1),
        (16, 2, 64, 16, 32, 6, 2, 2),
        (1, 4, 128, 8, 16, 4, 1, 8),
    ],
)
def test_mqa_paged_parity(triton_gate, rows, heads, head_dim, num_pages, page_size,
                          table_width, num_requests, ratio):
    q, k_cache, table, token_to_req, positions, seq_lens = _mqa_inputs(
        rows, heads, head_dim, num_pages, page_size, table_width, num_requests, ratio
    )
    columns = table_width * page_size
    scale = float(head_dim) ** 0.5
    ref_logits, ref_visible = qsa_ops.qsa_mqa_paged(
        q, k_cache, table, token_to_req, positions, seq_lens, ratio, columns
    )
    hip_logits, hip_visible = qsa_rdna2.qsa_mqa_paged(
        q, k_cache, table, token_to_req, positions, seq_lens, ratio, columns, scale
    )
    torch.cuda.synchronize()

    assert hip_logits.shape == ref_logits.shape == (rows, columns)
    assert hip_visible.shape == ref_visible.shape == (rows,)
    assert torch.equal(hip_visible, ref_visible), (
        f"visible_blocks mismatch: {(hip_visible != ref_visible).sum().item()} rows"
    )

    ref_masked = torch.isneginf(ref_logits)
    hip_masked = torch.isneginf(hip_logits)
    assert torch.equal(ref_masked, hip_masked), (
        f"mask mismatch: {(ref_masked != hip_masked).sum().item()} entries"
    )
    finite = ~ref_masked
    maxdiff = (
        (ref_logits[finite] - hip_logits[finite]).abs().max().item()
        if finite.any()
        else 0.0
    )
    print(
        f"mqa_paged atol={ATOL} maxdiff={maxdiff} rows={rows} heads={heads} "
        f"head_dim={head_dim} ratio={ratio} masked={int(ref_masked.sum())}"
    )
    torch.testing.assert_close(
        hip_logits[finite], ref_logits[finite], atol=ATOL, rtol=RTOL
    )


def test_mqa_paged_smaller_num_columns_matches_capacity(triton_gate):
    """A reduced score width must equal the prefix of the capacity-width run."""
    q, k_cache, table, token_to_req, positions, seq_lens = _mqa_inputs(
        32, 4, 128, 32, 64, 16, 2, 4
    )
    columns = table.shape[1] * k_cache.shape[1]
    wide_logits, _ = qsa_ops.qsa_mqa_paged(
        q, k_cache, table, token_to_req, positions, seq_lens, 4, columns
    )
    narrow = 8
    hip_logits, _ = qsa_rdna2.qsa_mqa_paged(
        q, k_cache, table, token_to_req, positions, seq_lens, 4, narrow,
        128**0.5,
    )
    torch.cuda.synchronize()
    ref = wide_logits[:, :narrow]
    finite = ~torch.isneginf(ref)
    assert torch.equal(torch.isneginf(ref), torch.isneginf(hip_logits))
    torch.testing.assert_close(
        hip_logits[finite], ref[finite], atol=ATOL, rtol=RTOL
    )


def test_mqa_paged_rejects_bf16():
    rows, heads, head_dim = 4, 4, 128
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(8, 32, 1, head_dim, dtype=torch.bfloat16, device=device)
    table = torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    seq_lens = torch.full((1,), 64, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="fp16"):
        qsa_rdna2.qsa_mqa_paged(
            q, k_cache, table, token_to_req, positions, seq_lens, 4, 256, 128**0.5
        )


def test_compress_rejects_head_dim_above_block_d():
    """BLOCK_D is 128; wider rows used to be silently truncated."""
    rows, ratio = 4, 4
    raw_keys = torch.randn(rows, 1, 256, dtype=torch.float16, device=device)
    with pytest.raises(RuntimeError, match="head_dim"):
        qsa_rdna2.qsa_compress_groups(
            raw_keys,
            torch.zeros(rows, 1, 3, dtype=torch.int64, device=device),
            torch.zeros(2, 8, 1, 256, dtype=torch.float16, device=device),
            torch.zeros(2, 8, 1, 3, dtype=torch.int64, device=device),
            torch.full((2, 1), -1, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
            torch.arange(0, rows + 1, rows // 2, dtype=torch.int32, device=device),
            torch.full((rows,), ratio, dtype=torch.int64, device=device),
            torch.zeros(rows, dtype=torch.int64, device=device),
            torch.zeros(rows, 1, 256, dtype=torch.float16, device=device),
            torch.zeros(rows, 3, dtype=torch.int64, device=device),
            ratio,
            8,
            256,
            False,
        )


# ---------------------------------------------------------------------------
# S2/S3 — the gate must not change the indexer's answer
# ---------------------------------------------------------------------------


def test_dispatched_mqa_paged_matches_gate_off(monkeypatch):
    q, k_cache, table, token_to_req, positions, seq_lens = _mqa_inputs(
        32, 4, 128, 32, 64, 16, 2, 4
    )
    columns = table.shape[1] * k_cache.shape[1]

    monkeypatch.setattr(qsa_rdna2, "qsa_use_rdna2", lambda dtype=None: False)
    ref_logits, ref_visible = qsa_ops.qsa_mqa_paged(
        q, k_cache, table, token_to_req, positions, seq_lens, 4, columns
    )
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)
    hip_logits, hip_visible = qsa_ops.qsa_mqa_paged(
        q, k_cache, table, token_to_req, positions, seq_lens, 4, columns
    )
    torch.cuda.synchronize()

    assert torch.equal(hip_visible, ref_visible)
    finite = ~torch.isneginf(ref_logits)
    assert torch.equal(finite, ~torch.isneginf(hip_logits))
    torch.testing.assert_close(
        hip_logits[finite], ref_logits[finite], atol=ATOL, rtol=RTOL
    )


def test_dispatched_compress_and_store_match_gate_off(monkeypatch):
    rows, head_dim, ratio, num_requests = 32, 128, 4, 2
    (
        raw_keys,
        raw_positions,
        state_cache,
        rope_cache,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    ) = _compress_inputs(rows, head_dim, ratio, num_requests, 2, True)
    store_rows = torch.randn(rows, head_dim, dtype=torch.float16, device=device)
    store_slots = torch.randperm(rows, device=device).to(torch.int64)

    def run():
        pooled, first = qsa_ops.qsa_compress_groups_with_ratio(
            raw_keys, raw_positions, state_cache, state_table, token_to_req,
            query_start_loc, logical_positions, compressed_slots, ratio, rope_cache,
        )
        cache = torch.zeros(2, 64, 1, head_dim, dtype=torch.float16, device=device)
        qsa_ops.qsa_store_cache_rows(cache, store_slots, store_rows)
        return pooled, first, cache

    monkeypatch.setattr(qsa_rdna2, "qsa_use_rdna2", lambda dtype=None: False)
    ref = run()
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)
    hip = run()
    torch.cuda.synchronize()

    torch.testing.assert_close(hip[0], ref[0], atol=ATOL, rtol=RTOL)
    assert torch.equal(hip[1], ref[1])
    assert torch.equal(hip[2], ref[2])


# ---------------------------------------------------------------------------
# S4 — the HIP ops must survive cudagraph capture/replay (the 2026-09-14
# SIGABRT was a cudagraph-capture fault in the QSA forward, not a kernel).
# ---------------------------------------------------------------------------


def test_dispatched_indexer_selection_matches_gate_off(monkeypatch):
    """S3: end-to-end indexer selection (score -> top-k -> expand) must agree."""
    rows, heads, head_dim, ratio, token_topk = 6, 4, 128, 4, 64
    num_pages, page_size, table_width, num_requests = 32, 64, 16, 2
    torch.manual_seed(5)
    q = torch.randn(rows, heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.float16, device=device
    )
    table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .reshape(1, -1)
        .repeat(num_requests, 1)
    )
    token_to_req = torch.tensor(
        [0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )
    positions = torch.tensor(
        [8, 9, 10, 40, 41, 42], dtype=torch.int64, device=device
    )
    seq_lens = torch.tensor([100, 200], dtype=torch.int32, device=device)

    monkeypatch.setattr(qsa_rdna2, "qsa_use_rdna2", lambda dtype=None: False)
    ref = qsa_ops.qsa_select_paged_tokens(
        q, k_cache, table, token_to_req, positions, seq_lens, token_topk, ratio
    )
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)
    hip = qsa_ops.qsa_select_paged_tokens(
        q, k_cache, table, token_to_req, positions, seq_lens, token_topk, ratio
    )
    torch.cuda.synchronize()

    for row in range(rows):
        ref_set = set(int(v) for v in ref[row] if v >= 0)
        hip_set = set(int(v) for v in hip[row] if v >= 0)
        assert hip_set == ref_set, f"row {row}: {sorted(hip_set)} != {sorted(ref_set)}"
        assert len(hip_set) == len(ref_set)


def test_qsa_hip_ops_are_cudagraph_safe(triton_gate):
    rows, heads, head_dim = 8, 4, 128
    q, k_cache, table, token_to_req, positions, seq_lens = _mqa_inputs(
        rows, heads, head_dim, 16, 32, 8, 2, 4
    )
    columns = table.shape[1] * k_cache.shape[1]
    (
        raw_keys,
        raw_positions,
        state_cache,
        rope_cache,
        state_table,
        ctoken_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    ) = _compress_inputs(rows, head_dim, 4, 2, 2, True)
    store_rows = torch.randn(rows, head_dim, dtype=torch.float16, device=device)
    store_slots = torch.arange(rows, dtype=torch.int64, device=device)
    store_cache = torch.zeros(4, 256, 1, head_dim, dtype=torch.float16, device=device)

    def run():
        qsa_rdna2.qsa_store_cache_rows(
            store_rows, store_slots, store_cache, store_cache.shape[1], head_dim
        )
        qsa_rdna2.qsa_compress_groups_with_ratio_compat(
            raw_keys, raw_positions, state_cache, state_table, ctoken_to_req,
            query_start_loc, logical_positions, compressed_slots, 4, rope_cache,
        )
        qsa_rdna2.qsa_mqa_paged(
            q, k_cache, table, token_to_req, positions, seq_lens, 4, columns,
            head_dim**0.5,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    store_cache.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert bool((store_cache != 0).any()), "store_cache_rows did not replay"


# --- expand_qsa_block_indices_cuda vs qsa_expand_indices_rdna2 ---------------


def _expand_inputs(rows, block_topk, ratio, num_requests, seed=0, strided=False):
    """Build inputs for the index-expansion op.

    Block ids are drawn from a plausible page range; query positions and
    sequence lengths are chosen so that rows cover the three regimes the kernel
    distinguishes: fully expanded, open group with a partial tail, and
    out-of-range (invalid request / token past the sequence end).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    blocks = torch.randint(
        0, 4096, (rows, block_topk), device=device, dtype=torch.int32, generator=gen
    )
    if strided:
        # Force a non-unit row stride: the kernel takes both strides, so this
        # exercises the stride path rather than assuming contiguity.
        wide = torch.full(
            (rows, block_topk * 2), -7, device=device, dtype=torch.int32
        )
        wide[:, ::2] = blocks
        blocks = wide[:, ::2]
    token_to_req = torch.randint(
        0, num_requests, (rows,), device=device, dtype=torch.int32, generator=gen
    )
    if rows:
        token_to_req[0] = -1  # invalid request -> not valid_request
    # Positions spread across the compressed range so some rows are fully
    # expanded, some have an open tail, and some exceed the sequence length.
    query_positions = torch.arange(rows, device=device, dtype=torch.int64) * ratio + 1
    sequence_lengths = torch.randint(
        ratio, ratio * (rows + 2) + 1, (num_requests,), device=device,
        dtype=torch.int32, generator=gen,
    )
    return blocks, query_positions, sequence_lengths, token_to_req


@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
@pytest.mark.parametrize("block_topk", [0, 1, 3, 8])
@pytest.mark.parametrize("rows", [1, 5, 64])
def test_expand_indices_parity(ratio, block_topk, rows):
    """HIP expansion must match the Triton reference exactly (integer op)."""
    num_requests = 4
    blocks, positions, seq_lens, token_to_req = _expand_inputs(
        rows, block_topk, ratio, num_requests, seed=rows * 31 + block_topk
    )
    token_topk = block_topk * ratio
    ref = qsa_ops.expand_qsa_block_indices_cuda(
        blocks, positions, seq_lens, token_to_req, ratio, token_topk
    )
    hip = qsa_rdna2.qsa_expand_indices(
        blocks, positions, seq_lens, token_to_req, ratio, token_topk
    )
    assert hip.shape == ref.shape == (rows, token_topk + ratio - 1)
    assert torch.equal(hip, ref), (
        f"rows={rows} block_topk={block_topk} ratio={ratio}\n"
        f"ref={ref.tolist()}\nhip={hip.tolist()}"
    )


@pytest.mark.parametrize("rows", [0, 3])
def test_expand_indices_empty_and_out_param(rows):
    """Empty input returns the empty output; a supplied `out` is filled in place."""
    ratio, block_topk, num_requests = 4, 3, 2
    blocks, positions, seq_lens, token_to_req = _expand_inputs(
        rows, block_topk, ratio, num_requests, seed=7
    )
    token_topk = block_topk * ratio
    width = token_topk + ratio - 1
    buf = torch.full((rows, width), -9, device=device, dtype=torch.int32)
    ref = qsa_ops.expand_qsa_block_indices_cuda(
        blocks, positions, seq_lens, token_to_req, ratio, token_topk, out=buf
    )
    buf_hip = torch.full((rows, width), -9, device=device, dtype=torch.int32)
    hip = qsa_rdna2.qsa_expand_indices(
        blocks, positions, seq_lens, token_to_req, ratio, token_topk, out=buf_hip
    )
    assert hip.data_ptr() == buf_hip.data_ptr(), "out= must be written in place"
    assert torch.equal(hip, ref)


def test_expand_indices_strided_blocks_parity():
    """A non-contiguous block_indices view must follow the same strides."""
    ratio, block_topk, rows, num_requests = 4, 4, 16, 4
    blocks, positions, seq_lens, token_to_req = _expand_inputs(
        rows, block_topk, ratio, num_requests, seed=11, strided=True
    )
    assert blocks.stride(1) == 2, "expected a non-unit column stride"
    token_topk = block_topk * ratio
    ref = qsa_ops.expand_qsa_block_indices_cuda(
        blocks, positions, seq_lens, token_to_req, ratio, token_topk
    )
    hip = qsa_rdna2.qsa_expand_indices(
        blocks, positions, seq_lens, token_to_req, ratio, token_topk
    )
    assert torch.equal(hip, ref)


def test_expand_indices_routes_to_hip_behind_the_gate(monkeypatch):
    """The dispatcher routes to the HIP op behind the gate.

    Unlike the other QSA ops this one is integer-only, so the fp16 dtype
    restriction does not apply -- but the env/platform gate still must.
    """
    seen: list[torch.dtype] = []
    monkeypatch.setattr(
        qsa_rdna2,
        "qsa_expand_indices",
        lambda *a, **kw: seen.append(a[0].dtype) or torch.empty(0),
    )
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)
    blocks, positions, seq_lens, token_to_req = _expand_inputs(4, 2, 4, 2, seed=3)
    qsa_ops.expand_qsa_block_indices_cuda(
        blocks, positions, seq_lens, token_to_req, 4, 8
    )
    assert seen == [torch.int32]

    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", False)
    seen.clear()
    ref = qsa_ops.expand_qsa_block_indices_cuda(
        blocks, positions, seq_lens, token_to_req, 4, 8
    )
    assert seen == [], "the gate must keep int32 on Triton"
    assert ref.dtype == torch.int32


# --- sparse paged attention vs the splitk + merge Triton pair ----------------


def _sparse_inputs(
    rows, heads, kv_heads, head_dim, topk, page_size, num_pages, num_requests,
    seed=0,
):
    """Paged K/V plus an index selection for the sparse attention.

    Logical token ids are constrained so that their logical page is inside the
    block table's width (otherwise everything would be masked), and a sentinel
    -1 is planted to exercise the invalid-index path.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    table_width = max(1, num_pages // num_requests)
    q = (torch.randn(rows, heads, head_dim, device=device, generator=gen) * 0.5).half()
    k_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim, device=device, generator=gen
    ).half()
    v_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim, device=device, generator=gen
    ).half()
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(num_requests, table_width)
    token_to_req = torch.randint(
        0, num_requests, (rows,), dtype=torch.int32, device=device, generator=gen
    )
    if rows:
        token_to_req[0] = -1  # invalid request -> fully masked row
    logical = torch.randint(
        0, table_width * page_size, (rows, topk), dtype=torch.int32,
        device=device, generator=gen,
    )
    if topk > 1:
        logical[:, -1] = -1
    return q, k_cache, v_cache, logical, block_table, token_to_req


# Heads 4 / kv 1 and heads 2 / kv 2 are the GQA shapes the model uses; topk
# 8 forces num_tiles == 1 (the single-split path that writes straight to fp16)
# while 64 forces multi-split partials through the merge kernel.
@pytest.mark.parametrize("heads,kv_heads", [(4, 1), (2, 2)])
@pytest.mark.parametrize("topk", [8, 64, 129])
@pytest.mark.parametrize("rows", [1, 7, 40])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_sparse_paged_attention_parity(
    triton_gate, heads, kv_heads, topk, rows, head_dim
):
    """HIP sparse attention must match the Triton splitk + merge pair."""
    page_size, num_pages, num_requests = 16, 64, 4
    q, k_cache, v_cache, logical, block_table, token_to_req = _sparse_inputs(
        rows, heads, kv_heads, head_dim, topk, page_size, num_pages, num_requests,
        seed=rows * 7 + topk,
    )
    ref = qsa_ops.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical, block_table, token_to_req
    )
    hip = qsa_rdna2.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical, block_table, token_to_req
    )
    assert hip.shape == ref.shape == q.shape
    assert hip.dtype == ref.dtype == torch.float16
    # The single-split path is bit-exact against Triton; with splits, the only
    # difference is the merge summing the partials in a different order, which
    # lands within 1-2 fp16 ulps (measured: <=6e-5 abs, ~1e-3 rel at |ref|~0.5).
    # Bound it tightly so a real error cannot hide.
    torch.testing.assert_close(hip.float(), ref.float(), rtol=5e-3, atol=1e-4)


def test_sparse_paged_attention_fully_masked_rows_are_zero():
    """A row whose request is invalid must come out as zeros, not NaN."""
    rows, heads, kv_heads, head_dim, topk = 3, 4, 1, 128, 32
    q, k_cache, v_cache, logical, block_table, token_to_req = _sparse_inputs(
        rows, heads, kv_heads, head_dim, topk, 16, 64, 4, seed=5
    )
    token_to_req[:] = -1
    hip = qsa_rdna2.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical, block_table, token_to_req
    )
    assert torch.equal(hip, torch.zeros_like(hip)), "masked rows must be 0"
    assert not torch.isnan(hip.float()).any()


def test_sparse_paged_attention_out_param_is_in_place():
    rows, heads, kv_heads, head_dim, topk = 4, 4, 1, 128, 48
    q, k_cache, v_cache, logical, block_table, token_to_req = _sparse_inputs(
        rows, heads, kv_heads, head_dim, topk, 16, 64, 4, seed=9
    )
    buf = torch.zeros_like(q)
    got = qsa_rdna2.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical, block_table, token_to_req, out=buf
    )
    assert got.data_ptr() == buf.data_ptr()


def test_sparse_paged_attention_routes_to_hip_only_for_fp16(monkeypatch):
    """fp16 goes to the HIP op; bf16 must keep Triton (the kernels are fp16)."""
    seen: list[torch.dtype] = []
    monkeypatch.setattr(
        qsa_rdna2,
        "qsa_sparse_paged_attention",
        lambda *a, **kw: seen.append(a[0].dtype) or torch.empty(0),
    )
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "on_gfx10x", lambda: True)
    q, k_cache, v_cache, logical, block_table, token_to_req = _sparse_inputs(
        4, 4, 1, 128, 32, 16, 64, 4, seed=13
    )
    qsa_ops.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical, block_table, token_to_req
    )
    assert seen == [torch.float16]


# Production Qwen4Exp text geometry: head_dim 256, partial_rotary_factor 0.25,
# mrope_section [11, 11, 10] (sums to rotary_dim // 2).
_MROPE_HD, _MROPE_RD = 256, 64
_MROPE_SECTION = [11, 11, 10]


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("interleaved", [True, False])
def test_mrope_forward_rdna2_matches_triton(is_neox_style, interleaved):
    """The HIP M-RoPE kernel must match ``triton_mrope`` for every layout."""
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        mrope_forward_rdna2,
        triton_mrope,
    )

    torch.manual_seed(0)
    tokens, n_qh, n_kh = 7, 6, 1
    q = torch.randn(tokens, n_qh * _MROPE_HD, dtype=torch.float16, device=device)
    k = torch.randn(tokens, n_kh * _MROPE_HD, dtype=torch.float16, device=device)
    cos = torch.randn(3, tokens, _MROPE_RD // 2, dtype=torch.float16, device=device)
    sin = torch.randn(3, tokens, _MROPE_RD // 2, dtype=torch.float16, device=device)

    ref_q, ref_k = triton_mrope(
        q.clone(),
        k.clone(),
        cos,
        sin,
        _MROPE_SECTION,
        _MROPE_HD,
        _MROPE_RD,
        interleaved,
        is_neox_style,
    )
    assert mrope_forward_rdna2(
        q,
        k,
        cos,
        sin,
        _MROPE_HD,
        _MROPE_RD,
        _MROPE_SECTION,
        interleaved,
        is_neox_style,
    )
    torch.testing.assert_close(q, ref_q, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(k, ref_k, atol=1e-3, rtol=1e-2)
