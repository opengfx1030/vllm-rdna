# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1030 QSA metadata builder must never map a token into the null block.

Block 0 is vLLM's null block (``BlockPool.null_block``). The rows that hold it
belong to no request: dummy runs zero their block tables
(``BlockTables.get_dummy_block_tables``) and the gather zeroes the padded rows
of a real batch. The Triton builder drops such tokens from the circular ring
(``physical_block > 0``); ``build_qsa_metadata_rdna2`` must too, or a dummy
step writes its raw keys into block 0.

Run ``pytest tests/kernels/rocm/qwen4_exp/test_qsa_metadata_null_block_rdna2.py``.
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
RING = 8
COMPRESS_RATIO = 4
STORAGE_BLOCK_SIZE = 256
CIRCULAR = {"compress_ratio": 1, "circular_buffer_size": RING}
COMPRESSED = {"compress_ratio": COMPRESS_RATIO, "circular_buffer_size": 0}


def _metadata(
    query_lens: list[int],
    seq_lens: list[int],
    block_table: torch.Tensor,
    common_slot_mapping: torch.Tensor,
) -> SimpleNamespace:
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    return SimpleNamespace(
        num_actual_tokens=int(common_slot_mapping.numel()),
        query_start_loc=query_start_loc.to(DEVICE),
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        slot_mapping=common_slot_mapping.to(DEVICE),
        block_table_tensor=block_table.to(device=DEVICE, dtype=torch.int32),
    )


def _slots(builder, metadata: SimpleNamespace, rule: dict[str, int]) -> torch.Tensor:
    n = metadata.num_actual_tokens
    _, _, _, slots = builder(
        metadata,
        torch.empty(n, dtype=torch.int32, device=DEVICE),
        torch.empty(n, dtype=torch.int64, device=DEVICE),
        torch.empty(n, dtype=torch.int32, device=DEVICE),
        torch.empty(n, dtype=torch.int64, device=DEVICE),
        storage_block_size=STORAGE_BLOCK_SIZE,
        **rule,
    )
    return slots.cpu()


def _expected_ring_slots(
    query_lens: list[int], seq_lens: list[int], ring_blocks: list[int], n: int
) -> torch.Tensor:
    """Ring rule: the last RING rows of each request, never block 0."""
    expected = torch.full((n,), -1, dtype=torch.int64)
    start = 0
    for q, seq_len, block in zip(query_lens, seq_lens, ring_blocks):
        end = start + q
        for token in range(start, end):
            position = seq_len - q + token - start
            if position >= 0 and token + RING >= end and block > 0:
                expected[token] = block * RING + position % RING
        start = end
    return expected


@pytest.mark.parametrize("null_row", [0, 1, 2])
def test_circular_ring_skips_null_block(null_row: int) -> None:
    """A request row on block 0 gets PAD slots; the other rings stay exact."""
    query_lens = [5, 12, 3]
    seq_lens = [40, 12, 17]
    ring_blocks = [7, 3, 11]
    ring_blocks[null_row] = 0
    n = sum(query_lens) + 4  # four padded tokens past the last request
    metadata = _metadata(
        query_lens,
        seq_lens,
        torch.tensor(ring_blocks).view(-1, 1),
        torch.arange(n, dtype=torch.int64),
    )

    hip = _slots(qsa_cache.build_qsa_metadata_rdna2, metadata, CIRCULAR)

    start = sum(query_lens[:null_row])
    null_tokens = hip[start : start + query_lens[null_row]]
    assert torch.all(null_tokens == -1), (
        f"tokens of the null-block row were mapped into block 0: {null_tokens}"
    )
    assert torch.equal(hip, _expected_ring_slots(query_lens, seq_lens, ring_blocks, n))
    triton_slots = _slots(qsa_cache.build_qsa_metadata_triton, metadata, CIRCULAR)
    assert torch.equal(hip, triton_slots), "HIP ring slots differ from Triton"


@pytest.mark.parametrize("rule", [CIRCULAR, COMPRESSED], ids=["circular", "compressed"])
def test_dummy_batch_maps_no_token(rule: dict[str, int]) -> None:
    """A capture/profile dummy batch (InputBatch.make_dummy, zeroed block
    tables, all-PAD common slot mapping) stores nothing in either cache."""
    num_reqs, per_req = 4, 8
    n = num_reqs * per_req
    metadata = _metadata(
        [per_req] * num_reqs,
        [per_req] * num_reqs,
        torch.zeros(num_reqs, 4, dtype=torch.int32),
        torch.full((n,), -1, dtype=torch.int64),
    )

    hip = _slots(qsa_cache.build_qsa_metadata_rdna2, metadata, rule)

    assert torch.all(hip == -1), f"dummy tokens mapped to slots {hip.tolist()}"
    triton_slots = _slots(qsa_cache.build_qsa_metadata_triton, metadata, rule)
    assert torch.equal(hip, triton_slots)
