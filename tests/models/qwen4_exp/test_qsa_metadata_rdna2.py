# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the gfx1030 HIP QSA metadata builder.

``build_qsa_metadata_rdna2`` is compared bit for bit against
``build_qsa_metadata_triton`` for all three slot-mapping rules (circular ring,
compressed groups, none) and with and without the pre-indexer work metadata.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.platforms.rdna import on_rdna2

pytestmark = pytest.mark.skipif(
    not on_rdna2(), reason="RDNA2 HIP QSA metadata builder requires gfx10x"
)

DEVICE = torch.device("cuda")


def _common(
    query_lengths: list[int],
    seq_lens: list[int],
    num_padding: int,
    num_block_columns: int = 8,
    negative_blocks: bool = False,
):
    num_reqs = len(query_lengths)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(query_lengths, dtype=torch.int32, device=DEVICE)
    query_start_loc = query_start_loc.cumsum(0).to(torch.int32)
    num_mapped_tokens = int(query_start_loc[-1])
    num_tokens = num_mapped_tokens + num_padding

    block_table = (
        torch.arange(num_reqs * num_block_columns, dtype=torch.int32, device=DEVICE)
        .reshape(num_reqs, num_block_columns)
        .add(1)
    )
    if negative_blocks:
        block_table[0, 3:] = -1

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    slot_mapping[1::3] = -1

    return SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        slot_mapping=slot_mapping,
        block_table_tensor=block_table,
    )


def _buffers(num_tokens: int):
    return (
        torch.empty(num_tokens, dtype=torch.int32, device=DEVICE),
        torch.empty(num_tokens, dtype=torch.int64, device=DEVICE),
        torch.empty(num_tokens, dtype=torch.int32, device=DEVICE),
        torch.empty(num_tokens, dtype=torch.int64, device=DEVICE),
    )


def _k_work(num_tokens: int, num_reqs: int, compress_ratio: int):
    if compress_ratio == 1:
        return None
    max_num_work = (num_tokens + (compress_ratio - 1) * num_reqs) // compress_ratio
    return torch.empty(max_num_work, 2, dtype=torch.int32, device=DEVICE)


@pytest.mark.parametrize(
    "query_lengths,seq_lens,num_padding",
    [
        ([1], [1], 0),
        ([5, 3, 4], [17, 9, 33], 4),
        ([128, 129, 1], [1000, 2, 130], 7),
        ([4, 0, 6], [10, 0, 12], 3),
    ],
)
@pytest.mark.parametrize("negative_blocks", [False, True])
@pytest.mark.parametrize(
    "compress_ratio,circular_buffer_size",
    [(4, 0), (1, 8), (1, 0), (8, 0)],
)
@pytest.mark.parametrize("with_k_work", [False, True])
def test_build_qsa_metadata_parity(
    query_lengths: list[int],
    seq_lens: list[int],
    num_padding: int,
    negative_blocks: bool,
    compress_ratio: int,
    circular_buffer_size: int,
    with_k_work: bool,
) -> None:
    common = _common(query_lengths, seq_lens, num_padding, negative_blocks=negative_blocks)
    num_tokens = common.num_actual_tokens
    num_reqs = len(query_lengths)

    k_work = _k_work(num_tokens, num_reqs, compress_ratio) if with_k_work else None
    hip_k_work = None if k_work is None else torch.empty_like(k_work)
    tri_k_work = None if k_work is None else torch.empty_like(k_work)

    hip = qsa_cache.build_qsa_metadata_rdna2(
        common,
        *_buffers(num_tokens),
        storage_block_size=2,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
        k_work_metadata_buffer=hip_k_work,
        request_capacity=num_reqs,
    )
    tri = qsa_cache.build_qsa_metadata_triton(
        common,
        *_buffers(num_tokens),
        storage_block_size=2,
        compress_ratio=compress_ratio,
        circular_buffer_size=circular_buffer_size,
        k_work_metadata_buffer=tri_k_work,
        request_capacity=num_reqs,
    )

    for name, got, want in zip(
        ("token_to_req", "logical_positions", "visible_blocks", "slot_mapping"), hip, tri
    ):
        assert torch.equal(got, want), f"{name} differs from Triton"
    if hip_k_work is not None:
        assert torch.equal(hip_k_work, tri_k_work), "k_work_metadata differs from Triton"


def test_build_qsa_metadata_zero_tokens() -> None:
    common = _common([0], [0], 0)
    num_tokens = common.num_actual_tokens
    assert num_tokens == 0
    hip = qsa_cache.build_qsa_metadata_rdna2(
        common, *_buffers(0), storage_block_size=2, compress_ratio=4
    )
    assert all(t.numel() == 0 for t in hip)


def test_build_qsa_metadata_dispatch_is_not_triton(monkeypatch) -> None:
    """On RDNA2 the dispatcher must resolve to the HIP builder, never Triton."""
    monkeypatch.setattr(qsa_cache, "_qsa_metadata_impl", None)
    assert qsa_cache._resolve_qsa_metadata_builder() is (
        qsa_cache.build_qsa_metadata_rdna2
    )
    assert qsa_cache._qsa_metadata_impl is None
