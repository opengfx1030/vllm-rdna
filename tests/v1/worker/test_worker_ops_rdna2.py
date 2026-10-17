# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-for-bit parity between the gfx1030 HIP ports and the Triton kernels for
the v1 worker's per-step index/arithmetic ops:

  _gather_block_tables_kernel      (gpu/block_table.py)
  _compute_slot_mappings_kernel    (gpu/block_table.py)
  _apply_write_kernel              (gpu/buffer_utils.py, MULTI_GROUP False/True)
  _zero_kv_blocks_kernel           (utils.py)

These run inside FULL cudagraph capture, so the only acceptable difference
between the two paths is none. Every assertion is `torch.equal` (bit-for-bit),
not a tolerance.
"""

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel,
    _gather_block_tables_kernel,
)
from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel
from vllm.v1.worker.utils import _zero_kv_blocks_kernel

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_rdna2()),
    reason="gfx10x (RDNA2) ROCm required",
)

DEVICE = torch.device("cuda")


def hipptrs(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.uint64, device=DEVICE
    )


def triton_gather(
    idx_mapping, src_ptrs, dst_ptrs, strides, num_blocks, num_reqs, num_reqs_padded
):
    _gather_block_tables_kernel[(src_ptrs.numel(), num_reqs_padded)](
        idx_mapping,
        src_ptrs,
        dst_ptrs,
        strides,
        num_blocks,
        num_blocks.numel() // num_blocks.shape[0],
        num_reqs,
        BLOCK_SIZE=1024,
    )


def hip_gather(
    idx_mapping, src_ptrs, dst_ptrs, strides, num_blocks, num_reqs, num_reqs_padded
):
    torch.ops._rocm_C.gather_block_tables_rdna2(
        idx_mapping,
        src_ptrs,
        dst_ptrs,
        strides,
        num_blocks,
        num_blocks.stride(0),
        num_reqs,
        num_reqs_padded,
    )


@pytest.mark.parametrize(
    "num_groups,num_reqs,num_reqs_padded,width",
    [
        # real serving shape: 6 kv-cache groups, 8 seqs, widest row 32768
        (6, 8, 8, 32768),
        # non-multiple-of-1024 widths, and both sides of the 1024 tile
        (3, 5, 7, 1023),
        (2, 4, 4, 1024),
        (1, 2, 3, 1025),
        # single element / single group / single req
        (1, 1, 1, 1),
        (1, 1, 4, 3),
    ],
)
def test_gather_block_tables_parity(num_groups, num_reqs, num_reqs_padded, width):
    torch.manual_seed(num_groups * 1000 + width)
    strides = torch.full((num_groups,), width, dtype=torch.int64, device=DEVICE)
    # Some rows only partially filled; padded rows (>= num_reqs) must be zeroed.
    num_blocks = torch.zeros(
        (num_groups, num_reqs), dtype=torch.int32, device=DEVICE
    )
    for g in range(num_groups):
        num_blocks[g] = torch.randint(
            0, width + 1, (num_reqs,), dtype=torch.int32, device=DEVICE
        )

    src = [
        torch.randint(
            1, 1 << 20, (num_reqs, width), dtype=torch.int32, device=DEVICE
        )
        for _ in range(num_groups)
    ]
    dst_triton = [
        torch.full((num_reqs_padded, width), 7, dtype=torch.int32, device=DEVICE)
        for _ in range(num_groups)
    ]
    dst_hip = [t.clone() for t in dst_triton]

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)

    triton_gather(
        idx_mapping,
        hipptrs(src),
        hipptrs(dst_triton),
        strides,
        num_blocks,
        num_reqs,
        num_reqs_padded,
    )
    hip_gather(
        idx_mapping,
        hipptrs(src),
        hipptrs(dst_hip),
        strides,
        num_blocks,
        num_reqs,
        num_reqs_padded,
    )
    torch.accelerator.synchronize()

    for g in range(num_groups):
        assert torch.equal(dst_triton[g], dst_hip[g]), f"group {g} differs"
        # Padded rows really were zeroed, and copied rows carry the block ids.
        assert (dst_hip[g][num_reqs:] == 0).all()
        for r in range(num_reqs):
            nb = int(num_blocks[g, r])
            if nb:
                assert torch.equal(dst_hip[g][r, :nb], src[g][r, :nb])
            if nb < width:
                # The rest of a partially filled row keeps its sentinel.
                assert (dst_hip[g][r, nb:] == 7).all()


def test_gather_block_tables_zero_reqs_padded_is_inert():
    """num_reqs_padded == 0 must not launch (the wrapper returns early)."""
    width = 8
    src = [torch.ones((2, width), dtype=torch.int32, device=DEVICE)]
    dst = [torch.full((1, width), 5, dtype=torch.int32, device=DEVICE)]
    hip_gather(
        torch.arange(2, dtype=torch.int32, device=DEVICE),
        hipptrs(src),
        hipptrs(dst),
        torch.full((1,), width, dtype=torch.int64, device=DEVICE),
        torch.full((1, 2), width, dtype=torch.int32, device=DEVICE),
        2,
        0,
    )
    torch.accelerator.synchronize()
    assert (dst[0] == 5).all()


def build_slot_mapping_inputs(
    num_groups,
    num_reqs,
    num_tokens,
    max_num_tokens,
    block_width,
    kernel_block_sizes,
    block_sizes,
    mapping_enabled,
    dcp_sharded,
    cp_size=1,
    cp_rank=0,
    cp_interleave=1,
    seed=0,
):
    torch.manual_seed(seed)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    # One -1 request (dummy / cudagraph padding) to exercise is_real_req.
    if num_reqs >= 2:
        idx_mapping[-1] = -1
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    per_req = torch.full((num_reqs,), num_tokens // num_reqs, dtype=torch.int32)
    for i in range(num_tokens % num_reqs):
        per_req[i] += 1
    query_start_loc[1:] = torch.cumsum(per_req, 0)
    positions = torch.randint(
        0, 300000, (max_num_tokens,), dtype=torch.int64, device=DEVICE
    )
    # Requests start mid-block so block_indices > 0 is exercised too.
    positions[:num_tokens] = positions[:num_tokens] % (kernel_block_sizes[0] * 40)
    # Keep the block tables alive: only their raw addresses travel in the
    # pointer tensor the kernel reads.
    block_tables = [
        torch.randint(
            1, 4096, (num_reqs, block_width), dtype=torch.int32, device=DEVICE
        )
        for _ in range(num_groups)
    ]
    return dict(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        positions=positions,
        block_tables=block_tables,
        block_table_ptrs=hipptrs(block_tables),
        block_table_strides=torch.full(
            (num_groups,), block_width, dtype=torch.int64, device=DEVICE
        ),
        block_sizes=torch.tensor(block_sizes, dtype=torch.int32, device=DEVICE),
        kernel_block_sizes=torch.tensor(
            kernel_block_sizes, dtype=torch.int32, device=DEVICE
        ),
        slot_mapping_enabled=torch.tensor(
            mapping_enabled, dtype=torch.bool, device=DEVICE
        ),
        dcp_sharded=torch.tensor(dcp_sharded, dtype=torch.bool, device=DEVICE),
        max_num_tokens=max_num_tokens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
    )


def triton_slot(args, out, num_groups, num_reqs):
    _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
        args["max_num_tokens"],
        args["idx_mapping"],
        args["query_start_loc"],
        args["positions"],
        args["block_table_ptrs"],
        args["block_table_strides"],
        args["block_sizes"],
        args["kernel_block_sizes"],
        args["slot_mapping_enabled"],
        args["dcp_sharded"],
        out,
        out.stride(0),
        args["cp_rank"],
        CP_SIZE=args["cp_size"],
        CP_INTERLEAVE=args["cp_interleave"],
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
    )


def hip_slot(args, out, num_groups, num_reqs):
    torch.ops._rocm_C.compute_slot_mappings_rdna2(
        args["max_num_tokens"],
        args["idx_mapping"],
        args["query_start_loc"],
        args["positions"],
        args["block_table_ptrs"],
        args["block_table_strides"],
        args["block_sizes"],
        args["kernel_block_sizes"],
        args["slot_mapping_enabled"],
        args["dcp_sharded"],
        out,
        out.stride(0),
        args["cp_rank"],
        args["cp_size"],
        args["cp_interleave"],
        PAD_SLOT_ID,
    )


@pytest.mark.parametrize(
    "num_groups,num_reqs,num_tokens,max_num_tokens,kernel_block_sizes,block_sizes,"
    "mapping_enabled,dcp_sharded,cp_size,cp_rank,cp_interleave",
    [
        # real serving shape: 6 groups (one with kernel block size 8), 8 reqs,
        # max_num_batched_tokens 2048, a 24-token decode step.
        (
            6,
            8,
            24,
            2048,
            [1024, 1024, 1024, 1024, 8, 1024],
            [1024] * 6,
            [True] * 6,
            [True] * 6,
            1,
            0,
            1,
        ),
        # zero tokens (cudagraph pad row only)
        (2, 4, 0, 8, [1024, 1024], [1024, 1024], [True] * 2, [True] * 2, 1, 0, 1),
        # single token / single group / single req
        (1, 1, 1, 1, [8], [8], [True], [True], 1, 0, 1),
        # tokens just under / over the 1024 Triton tile
        (1, 3, 1023, 1023, [16], [16], [True], [True], 1, 0, 1),
        (1, 3, 1025, 1026, [16], [16], [True], [True], 1, 0, 1),
        # slot mapping disabled for one group, dummy request present
        (2, 4, 9, 16, [16, 8], [32, 8], [True, False], [True, True], 1, 0, 1),
        # virtual block splitting: kernel block size < scheduler block size
        (2, 4, 40, 64, [256, 8], [1024, 1024], [True] * 2, [True] * 2, 1, 0, 1),
        # context parallelism, sharded
        (2, 4, 33, 64, [16, 16], [16, 16], [True] * 2, [True, True], 4, 1, 1),
        # context parallelism, rank 0, non-sharded (replicated) group
        (2, 4, 33, 64, [16, 16], [16, 16], [True] * 2, [False, True], 4, 0, 2),
    ],
)
def test_compute_slot_mappings_parity(
    num_groups,
    num_reqs,
    num_tokens,
    max_num_tokens,
    kernel_block_sizes,
    block_sizes,
    mapping_enabled,
    dcp_sharded,
    cp_size,
    cp_rank,
    cp_interleave,
):
    block_width = max(kernel_block_sizes) * 40
    args = build_slot_mapping_inputs(
        num_groups,
        num_reqs,
        num_tokens,
        max_num_tokens,
        block_width,
        kernel_block_sizes,
        block_sizes,
        mapping_enabled,
        dcp_sharded,
        cp_size,
        cp_rank,
        cp_interleave,
    )
    out_triton = torch.full(
        (num_groups, max_num_tokens), 99, dtype=torch.int64, device=DEVICE
    )
    out_hip = out_triton.clone()

    triton_slot(args, out_triton, num_groups, num_reqs)
    hip_slot(args, out_hip, num_groups, num_reqs)
    torch.accelerator.synchronize()

    assert torch.equal(out_triton, out_hip)
    # Beyond the requested token count the kernel must have written PAD.
    if num_tokens < max_num_tokens:
        for g in range(num_groups):
            assert (out_hip[g][num_tokens:] == PAD_SLOT_ID).all()


def _apply_write_case(
    dtype, multi_group, num_writes, row_width, content_len, seed=0, group_ids=None
):
    torch.manual_seed(seed)
    num_groups = 3 if multi_group else 1
    out = [
        torch.full((4, row_width), 3, dtype=dtype, device=DEVICE)
        for _ in range(num_groups)
    ]
    # Distinct (row) / (group, row) per write: overlapping writes would race
    # between the two kernels and are never produced by the wrappers.
    if multi_group:
        indices = (
            torch.arange(num_writes, dtype=torch.int32, device=DEVICE) // num_groups
        )
        if group_ids is None:
            group_ids = torch.arange(
                num_writes, dtype=torch.int32, device=DEVICE
            ) % num_groups
    else:
        indices = torch.arange(num_writes, dtype=torch.int32, device=DEVICE)
    starts = torch.randint(
        0, max(row_width - content_len, 0) + 1, (num_writes,), dtype=torch.int32
    ).to(DEVICE)
    contents = torch.randint(
        0, 1000, (num_writes * content_len,), dtype=torch.int32
    ).to(DEVICE)
    if dtype == torch.float32:
        contents = contents.float()
    cu_lens = torch.arange(
        1, num_writes + 1, dtype=torch.int32, device=DEVICE
    ) * content_len
    if group_ids is None:
        group_ids = torch.arange(num_writes, dtype=torch.int32, device=DEVICE) % (
            num_groups
        )

    ref = [t.clone() for t in out]
    if multi_group:
        _apply_write_kernel[(num_writes,)](
            hipptrs(ref),
            torch.full((num_groups,), row_width, dtype=torch.int64, device=DEVICE),
            indices,
            starts,
            contents,
            cu_lens,
            group_ids,
            BLOCK_SIZE=1024,
            MULTI_GROUP=True,
        )
        torch.ops._rocm_C.fused_apply_write_rdna2(
            hipptrs(out),
            torch.full((num_groups,), row_width, dtype=torch.int64, device=DEVICE),
            indices,
            starts,
            contents,
            cu_lens,
            group_ids,
        )
    else:
        _apply_write_kernel[(num_writes,)](
            ref[0],
            row_width,
            indices,
            starts,
            contents,
            cu_lens,
            None,
            BLOCK_SIZE=1024,
            MULTI_GROUP=False,
        )
        torch.ops._rocm_C.apply_write_rdna2(
            out[0], row_width, indices, starts, contents, cu_lens
        )
    torch.accelerator.synchronize()
    return ref, out


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize(
    "content_len", [1, 7, 1023, 1024, 1025, 4096]
)
def test_apply_write_single_group_parity(dtype, content_len):
    ref, out = _apply_write_case(
        dtype, False, 3, content_len + 5, content_len, seed=content_len
    )
    assert torch.equal(ref[0], out[0])


@pytest.mark.parametrize(
    "num_writes,content_len", [(1, 1), (2, 3), (8, 1025), (8, 1024)]
)
def test_apply_write_multi_group_parity(num_writes, content_len):
    ref, out = _apply_write_case(
        torch.int32, True, num_writes, content_len + 7, content_len, seed=num_writes
    )
    for r, o in zip(ref, out):
        assert torch.equal(r, o)


def test_apply_write_zero_writes_is_inert():
    row = torch.full((2, 4), 3, dtype=torch.int32, device=DEVICE)
    empty32 = torch.empty(0, dtype=torch.int32, device=DEVICE)
    torch.ops._rocm_C.apply_write_rdna2(row, 4, empty32, empty32, empty32, empty32)
    torch.ops._rocm_C.fused_apply_write_rdna2(
        hipptrs([row]), torch.ones(1, dtype=torch.int64, device=DEVICE),
        empty32, empty32, empty32, empty32, empty32,
    )
    torch.accelerator.synchronize()
    assert (row == 3).all()


def build_zeroer(shapes, num_blocks, seed=0):
    """`shapes` is [(rows, cols), ...] one per segment, all in one allocation."""
    torch.manual_seed(seed)
    buffers = []
    seg_addrs = []
    seg_block_strides = []
    seg_page_sizes = []
    for rows, cols in shapes:
        buf = torch.zeros((rows, cols), dtype=torch.int32, device=DEVICE)
        buffers.append(buf)
        seg_addrs.append(buf.data_ptr())
        seg_block_strides.append(cols)
        seg_page_sizes.append(cols)
    return (
        buffers,
        torch.tensor(seg_addrs, dtype=torch.uint64, device=DEVICE),
        torch.tensor(seg_block_strides, dtype=torch.int64, device=DEVICE),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=DEVICE),
        num_blocks,
    )


@pytest.mark.parametrize(
    "shapes,num_blocks,chunk_size",
    [
        # one segment, one block: smallest thing the wrapper can see
        ([(4, 4)], 1, 4),
        # page exactly one chunk, then a page spanning 257 chunks (real
        # 1024-token fp16 KV page = 2*256*1024 elements = 262144 int32 words)
        ([(4, 1024)], 2, 1024),
        ([(4, 256 * 1024)], 3, 1024),
        # 12 segments (one per full-attention layer), 8 newly allocated blocks
        ([(64, 4096)] * 12, 8, 1024),
        # segments with different block strides and page sizes
        ([(8, 8), (8, 100), (8, 1024), (8, 1025)], 4, 512),
    ],
)
def test_zero_kv_blocks_parity(shapes, num_blocks, chunk_size):
    buffers_t, seg_addrs, strides, pages, n_blocks = build_zeroer(shapes, num_blocks)
    # Non-zero start so we prove the kernel actually zeroes.
    for b in buffers_t:
        b.fill_(5)
    buffers_h = [b.clone() for b in buffers_t]
    for b in buffers_h:
        b.fill_(5)

    block_ids = torch.arange(n_blocks, dtype=torch.int64, device=DEVICE)
    if shapes[0][0] < n_blocks:
        block_ids = block_ids % shapes[0][0]

    max_page = max(pages).item()
    max_chunks = (max_page + chunk_size - 1) // chunk_size

    # Triton reads the segment addresses from the same allocation, so patch the
    # HIP-side address table to the cloned buffers.
    seg_addrs_h = torch.tensor(
        [b.data_ptr() for b in buffers_h], dtype=torch.uint64, device=DEVICE
    )
    seg_addrs_t = torch.tensor(
        [b.data_ptr() for b in buffers_t], dtype=torch.uint64, device=DEVICE
    )

    _zero_kv_blocks_kernel[(n_blocks, len(shapes), max_chunks)](
        seg_addrs_t, strides, pages, block_ids, BLOCK_SIZE=chunk_size
    )
    torch.ops._rocm_C.zero_kv_blocks_rdna2(
        seg_addrs_h, strides, pages, block_ids, chunk_size, max_chunks
    )
    torch.accelerator.synchronize()

    for b_t, b_h in zip(buffers_t, buffers_h):
        assert torch.equal(b_t, b_h)


def test_zero_kv_blocks_zero_blocks_is_inert():
    buf = torch.full((4, 4), 5, dtype=torch.int32, device=DEVICE)
    torch.ops._rocm_C.zero_kv_blocks_rdna2(
        torch.tensor([buf.data_ptr()], dtype=torch.uint64, device=DEVICE),
        torch.tensor([4], dtype=torch.int64, device=DEVICE),
        torch.tensor([4], dtype=torch.int64, device=DEVICE),
        torch.empty(0, dtype=torch.int64, device=DEVICE),
        4,
        1,
    )
    torch.accelerator.synchronize()
    assert (buf == 5).all()
