# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-for-bit parity between the gfx1030 HIP ports and the Triton kernels for
the v1 worker input-batch ops:

  _prepare_prefill_inputs_kernel            (gpu/input_batch.py)
  _prepare_pos_seq_lens_kernel              (gpu/input_batch.py)
  _combine_sampled_and_draft_tokens_kernel  (gpu/input_batch.py)
  _get_num_sampled_and_rejected_kernel      (gpu/input_batch.py)
  _post_update_kernel                       (gpu/input_batch.py)
  _expand_idx_mapping_kernel                (gpu/input_batch.py)

These run once per decode step inside a FULL cudagraph capture, so the only
acceptable difference between the two paths is none. Every assertion is
`torch.equal` (bit-for-bit), not a tolerance.
"""

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.input_batch import (
    _combine_sampled_and_draft_tokens_kernel,
    _expand_idx_mapping_kernel,
    _get_num_sampled_and_rejected_kernel,
    _post_update_kernel,
    _prepare_pos_seq_lens_kernel,
    _prepare_prefill_inputs_kernel,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_rdna2()),
    reason="gfx10x (RDNA2) ROCm required",
)

DEVICE = torch.device("cuda")
OP = torch.ops._rocm_C


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# _prepare_prefill_inputs_kernel
# ---------------------------------------------------------------------------
def triton_prepare_prefill_inputs(
    input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
    all_token_ids, prefill_len, num_computed_tokens,
):
    num_reqs = idx_mapping.shape[0]
    num_lookahead = next_prefill_tokens.shape[0]
    if num_reqs == 0:
        return
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        num_lookahead,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
        LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
    )


def hip_prepare_prefill_inputs(
    input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
    all_token_ids, prefill_len, num_computed_tokens,
):
    OP.prepare_prefill_inputs_rdna2(
        input_ids,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        next_prefill_tokens.shape[0],
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
    )


def build_prefill_case(num_reqs, max_num_reqs, max_model_len, query_lens,
                       prefill_lens, num_computed, num_lookahead, seed=0):
    torch.manual_seed(seed)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=DEVICE
    ).cumsum(0)
    all_token_ids = torch.randint(
        1, 1 << 20, (max_num_reqs, max_model_len), dtype=torch.int32,
        device=DEVICE,
    )
    return dict(
        input_ids=torch.full(
            (int(query_start_loc[-1]),), 77, dtype=torch.int32, device=DEVICE
        ),
        next_prefill_tokens=torch.full(
            (num_lookahead, max_num_reqs), 88, dtype=torch.int32, device=DEVICE
        ),
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        all_token_ids=all_token_ids,
        prefill_len=torch.tensor(prefill_lens, dtype=torch.int32, device=DEVICE),
        num_computed_tokens=torch.tensor(
            num_computed, dtype=torch.int32, device=DEVICE
        ),
    )


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,max_model_len,query_lens,prefill_lens,num_computed,"
    "num_lookahead",
    [
        # real shape: 8 seqs, 2048-token prefill chunk, one lookahead token
        (8, 8, 4096, [256] * 8, [2048] * 8, [0] * 8, 1),
        # single request, single token
        (1, 1, 64, [1], [1], [0], 1),
        # not-prefill rows (num_computed >= prefill_len) must be inert
        (4, 4, 512, [16] * 4, [8, 8, 64, 64], [8, 8, 0, 0], 1),
        # query_len exactly / just over the 1024 Triton tile
        (2, 2, 4096, [1024, 1025], [4096, 4096], [0, 0], 2),
        # zero scheduled tokens on one row, lookahead still staged
        (3, 3, 256, [0, 7, 0], [64, 64, 64], [0, 0, 64], 4),
        # lookahead running past prefill_len -> zeros
        (2, 2, 128, [4, 4], [6, 5], [0, 0], 4),
        # zero requests: both paths are inert
        (0, 4, 64, [], [], [], 1),
    ],
)
def test_prepare_prefill_inputs_parity(
    num_reqs, max_num_reqs, max_model_len, query_lens, prefill_lens,
    num_computed, num_lookahead,
):
    args = build_prefill_case(
        num_reqs, max_num_reqs, max_model_len, query_lens, prefill_lens,
        num_computed, num_lookahead,
    )
    ref = {k: v.clone() for k, v in args.items() if torch.is_tensor(v)}
    got = {k: v.clone() for k, v in args.items() if torch.is_tensor(v)}

    triton_prepare_prefill_inputs(
        ref["input_ids"], ref["next_prefill_tokens"], ref["idx_mapping"],
        ref["query_start_loc"], ref["all_token_ids"], ref["prefill_len"],
        ref["num_computed_tokens"],
    )
    hip_prepare_prefill_inputs(
        got["input_ids"], got["next_prefill_tokens"], got["idx_mapping"],
        got["query_start_loc"], got["all_token_ids"], got["prefill_len"],
        got["num_computed_tokens"],
    )
    sync()

    assert torch.equal(ref["input_ids"], got["input_ids"])
    assert torch.equal(ref["next_prefill_tokens"], got["next_prefill_tokens"])
    # The copy really happened for the prefilling rows.
    for r in range(num_reqs):
        start = int(args["query_start_loc"][r])
        qlen = int(args["query_start_loc"][r + 1]) - start
        st = int(args["idx_mapping"][r])
        if int(args["num_computed_tokens"][st]) < int(args["prefill_len"][st]):
            nc = int(args["num_computed_tokens"][st])
            assert torch.equal(
                got["input_ids"][start : start + qlen],
                args["all_token_ids"][st, nc : nc + qlen],
            )


@pytest.mark.skipif(not is_uva_available(), reason="UVA not available")
def test_prepare_prefill_inputs_uva_all_token_ids():
    """all_token_ids is a UVA (pinned-host) tensor in the real runner."""
    rows, cols = 4, 512
    uva = StagedWriteTensor(
        (rows, cols), dtype=torch.int32, device=DEVICE, uva_instead_of_gpu=True
    )
    torch.manual_seed(3)
    host = torch.randint(1, 1 << 20, (rows, cols), dtype=torch.int32)
    uva.gpu.copy_(host.to(DEVICE))
    uva.apply_write()
    sync()

    args = build_prefill_case(2, rows, cols, [8, 8], [64, 64], [0, 0], 1, seed=5)
    args["all_token_ids"] = uva.gpu
    # all_token_ids is read-only here, so the same tensor feeds both paths.
    ref = {
        k: v.clone()
        for k, v in args.items()
        if torch.is_tensor(v) and k != "all_token_ids"
    }
    got = {k: v.clone() for k, v in ref.items()}

    triton_prepare_prefill_inputs(
        ref["input_ids"], ref["next_prefill_tokens"], ref["idx_mapping"],
        ref["query_start_loc"], args["all_token_ids"], ref["prefill_len"],
        ref["num_computed_tokens"],
    )
    hip_prepare_prefill_inputs(
        got["input_ids"], got["next_prefill_tokens"], got["idx_mapping"],
        got["query_start_loc"], args["all_token_ids"], got["prefill_len"],
        got["num_computed_tokens"],
    )
    sync()
    assert torch.equal(ref["input_ids"], got["input_ids"])
    assert torch.equal(ref["next_prefill_tokens"], got["next_prefill_tokens"])
    assert torch.equal(got["input_ids"][:8], uva.gpu[0, :8])

# ---------------------------------------------------------------------------
# _prepare_pos_seq_lens_kernel
# ---------------------------------------------------------------------------
def triton_prepare_pos_seq_lens(
    idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens
):
    num_reqs = idx_mapping.shape[0]
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


def hip_prepare_pos_seq_lens(
    idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens
):
    OP.prepare_pos_seq_lens_rdna2(
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
    )


def build_pos_case(num_reqs, max_num_reqs, query_lens, num_computed, seed=0):
    torch.manual_seed(seed)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=DEVICE
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    return dict(
        idx_mapping=idx_mapping,
        query_start_loc=query_start_loc,
        num_computed_tokens=torch.tensor(
            num_computed, dtype=torch.int32, device=DEVICE
        ),
        pos=torch.full((num_tokens,), -3, dtype=torch.int64, device=DEVICE),
        seq_lens=torch.full((max_num_reqs,), -5, dtype=torch.int32, device=DEVICE),
    )


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,query_lens,num_computed",
    [
        # real shape: 8 seqs x 3 decode tokens, no padding rows
        (8, 8, [3] * 8, [1024, 0, 7, 4096, 3, 100, 262140, 12]),
        # padding rows beyond num_reqs must be zeroed
        (5, 12, [3] * 5, [0] * 5),
        (1, 1, [1], [0]),
        # query_len exactly / just over the 1024 Triton tile
        (2, 2, [1024, 1025], [11, 22]),
        # a zero-length request
        (3, 3, [0, 5, 0], [4, 5, 6]),
        # zero requests: only the padding block runs
        (0, 6, [], []),
        (0, 1, [], []),
    ],
)
def test_prepare_pos_seq_lens_parity(
    num_reqs, max_num_reqs, query_lens, num_computed
):
    args = build_pos_case(num_reqs, max_num_reqs, query_lens, num_computed)
    ref = {k: v.clone() for k, v in args.items()}
    got = {k: v.clone() for k, v in args.items()}

    triton_prepare_pos_seq_lens(
        ref["idx_mapping"], ref["query_start_loc"], ref["num_computed_tokens"],
        ref["pos"], ref["seq_lens"],
    )
    hip_prepare_pos_seq_lens(
        got["idx_mapping"], got["query_start_loc"], got["num_computed_tokens"],
        got["pos"], got["seq_lens"],
    )
    sync()

    assert torch.equal(ref["pos"], got["pos"])
    assert torch.equal(ref["seq_lens"], got["seq_lens"])
    # Padding rows really were zeroed.
    if max_num_reqs > num_reqs:
        assert (got["seq_lens"][num_reqs:] == 0).all()
    for r in range(num_reqs):
        start = int(args["query_start_loc"][r])
        qlen = int(args["query_start_loc"][r + 1]) - start
        nc = int(args["num_computed_tokens"][r])
        assert int(got["seq_lens"][r]) == nc + qlen
        if qlen:
            assert torch.equal(
                got["pos"][start : start + qlen],
                torch.arange(nc, nc + qlen, dtype=torch.int64, device=DEVICE),
            )


# ---------------------------------------------------------------------------
# _combine_sampled_and_draft_tokens_kernel
# ---------------------------------------------------------------------------
def triton_combine(
    input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
    prefill_len, draft_tokens, cu_num_logits, logits_indices,
    num_new_sampled_tokens,
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        BLOCK_SIZE=triton.next_power_of_2(
            draft_tokens.shape[-1] + num_new_sampled_tokens
        ),
    )


def hip_combine(
    input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
    prefill_len, draft_tokens, cu_num_logits, logits_indices,
    num_new_sampled_tokens,
):
    OP.combine_sampled_and_draft_tokens_rdna2(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        num_new_sampled_tokens,
    )


def build_combine_case(
    num_reqs, max_num_reqs, steps, num_new_sampled_tokens, query_lens,
    seq_lens, prefill_lens, per_req_logits=None, seed=0,
):
    torch.manual_seed(seed)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=DEVICE
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    if per_req_logits is None:
        per_req_logits = [
            q - (q - steps - num_new_sampled_tokens) for q in query_lens
        ]
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(per_req_logits, dtype=torch.int32, device=DEVICE).cumsum(0)
    total_num_logits = int(cu[-1])
    return dict(
        input_ids=torch.randint(
            1, 1 << 20, (num_tokens,), dtype=torch.int32, device=DEVICE
        ),
        idx_mapping=idx_mapping,
        last_sampled_tokens=torch.randint(
            1, 1 << 20, (max_num_reqs, 1), dtype=torch.int64, device=DEVICE
        ),
        query_start_loc=query_start_loc,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        prefill_len=torch.tensor(prefill_lens, dtype=torch.int32, device=DEVICE),
        draft_tokens=torch.randint(
            1, 1 << 20, (max_num_reqs, steps), dtype=torch.int64, device=DEVICE
        ),
        cu_num_logits=cu,
        logits_indices=torch.full(
            (total_num_logits,), -12345, dtype=torch.int64, device=DEVICE
        ),
    )


@pytest.mark.parametrize(
    "num_reqs,max_num_reqs,steps,num_new_sampled_tokens,query_lens,seq_lens,"
    "prefill_lens",
    [
        # real serving shape: 8 seqs x (2 draft + 1 bonus) MTP2 decode step
        (8, 8, 2, 1, [3] * 8, [1027] * 8, [1024] * 8),
        # single request, one step, no bonus token
        (1, 1, 1, 0, [2], [5], [0]),
        # pure prefill rows: only logits_indices is written
        (4, 4, 2, 1, [3] * 4, [10, 10, 10, 10], [10] * 4),
        # first_logit_seq_pos < prefill_len -> last-sampled store skipped
        (2, 2, 2, 1, [3, 3], [12, 12], [10, 10]),
        # exactly one sampled token, zero draft tokens
        (2, 2, 1, 1, [1, 1], [9, 9], [8, 8]),
        # logits span exactly / just over the 1024 Triton tile
        (1, 1, 1023, 1, [1025], [2048], [0]),
        (1, 1, 1024, 1, [1026], [2048], [0]),
        # zero requests: both paths are inert
        (0, 2, 2, 1, [], [], []),
    ],
)
def test_combine_sampled_and_draft_tokens_parity(
    num_reqs, max_num_reqs, steps, num_new_sampled_tokens, query_lens, seq_lens,
    prefill_lens,
):
    args = build_combine_case(
        num_reqs, max_num_reqs, steps, num_new_sampled_tokens, query_lens,
        seq_lens, prefill_lens,
    )
    ref = {k: v.clone() for k, v in args.items()}
    got = {k: v.clone() for k, v in args.items()}

    triton_combine(
        ref["input_ids"], ref["idx_mapping"], ref["last_sampled_tokens"],
        ref["query_start_loc"], ref["seq_lens"], ref["prefill_len"],
        ref["draft_tokens"], ref["cu_num_logits"], ref["logits_indices"],
        num_new_sampled_tokens,
    )
    hip_combine(
        got["input_ids"], got["idx_mapping"], got["last_sampled_tokens"],
        got["query_start_loc"], got["seq_lens"], got["prefill_len"],
        got["draft_tokens"], got["cu_num_logits"], got["logits_indices"],
        num_new_sampled_tokens,
    )
    sync()

    assert torch.equal(ref["input_ids"], got["input_ids"])
    assert torch.equal(ref["logits_indices"], got["logits_indices"])
    assert (got["logits_indices"] != -12345).all()
    for r in range(num_reqs):
        start = int(args["cu_num_logits"][r])
        end = int(args["cu_num_logits"][r + 1])
        qend = int(args["query_start_loc"][r + 1])
        assert torch.equal(
            got["logits_indices"][start:end],
            torch.arange(qend - (end - start), qend, dtype=torch.int64,
                         device=DEVICE),
        )


# ---------------------------------------------------------------------------
# _get_num_sampled_and_rejected_kernel
# ---------------------------------------------------------------------------
def triton_get_num_sampled(
    num_sampled, num_rejected, seq_lens, cu_num_logits, idx_mapping, prefill_len
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled, num_rejected, seq_lens, cu_num_logits, idx_mapping,
        prefill_len,
    )


def hip_get_num_sampled(
    num_sampled, num_rejected, seq_lens, cu_num_logits, idx_mapping, prefill_len
):
    OP.get_num_sampled_and_rejected_rdna2(
        num_sampled, num_rejected, seq_lens, cu_num_logits, idx_mapping,
        prefill_len,
    )


@pytest.mark.parametrize(
    "seq_lens,prefill_lens,num_sampled,per_req_logits",
    [
        # real shape: 8 decode rows, 3 logits each, 1 sampled each
        ([1027] * 8, [1024] * 8, [1] * 8, [3] * 8),
        # chunked prefill rows (seq_len < prefill_len) force both to 0
        ([10, 10, 512, 512], [1024, 1024, 1024, 1024], [1, 1, 1, 1], [1, 1, 1, 1]),
        # zero sampled tokens on one row
        ([9, 9], [0, 0], [0, 3], [1, 4]),
        # single request
        ([4], [4], [1], [1]),
        # zero requests
        ([], [], [], []),
    ],
)
def test_get_num_sampled_and_rejected_parity(
    seq_lens, prefill_lens, num_sampled, per_req_logits
):
    num_reqs = len(seq_lens)
    torch.manual_seed(num_reqs)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(per_req_logits, dtype=torch.int32, device=DEVICE).cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    prefill_len_t = torch.tensor(prefill_lens, dtype=torch.int32, device=DEVICE)
    base = torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE)

    ns_ref, nr_ref = base.clone(), torch.full_like(base, -1)
    ns_got, nr_got = base.clone(), torch.full_like(base, -1)
    triton_get_num_sampled(ns_ref, nr_ref, seq_lens_t, cu, idx_mapping, prefill_len_t)
    hip_get_num_sampled(ns_got, nr_got, seq_lens_t, cu, idx_mapping, prefill_len_t)
    sync()

    assert torch.equal(ns_ref, ns_got)
    assert torch.equal(nr_ref, nr_got)
    for r in range(num_reqs):
        chunked = seq_lens[r] < prefill_lens[r]
        if chunked:
            assert int(ns_got[r]) == 0
            assert int(nr_got[r]) == 0
        else:
            assert int(ns_got[r]) == num_sampled[r]
            assert int(nr_got[r]) == per_req_logits[r] - num_sampled[r]


# ---------------------------------------------------------------------------
# _post_update_kernel
# ---------------------------------------------------------------------------
def triton_post_update(
    idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
    sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids,
    total_len,
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        num_warps=1,
    )


def hip_post_update(
    idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts,
    sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids,
    total_len,
):
    OP.post_update_rdna2(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
    )


def build_post_update_case(
    idx_mapping, sampled_tokens, num_sampled, num_rejected, seq_lens,
    prefill_lens, max_num_reqs=8, max_model_len=64, vocab_size=None,
    with_bin_counts=True, with_query_start_loc=True, total_len=None,
    seed=0,
):
    torch.manual_seed(seed)
    num_reqs = len(idx_mapping)
    real = [i for i in idx_mapping if i >= 0]
    if total_len is None:
        total_len = [0] * max_num_reqs
        for i in real:
            total_len[i] = 5
    if with_query_start_loc:
        query_start_loc = torch.zeros(
            num_reqs + 1, dtype=torch.int32, device=DEVICE
        )
        query_start_loc[1:] = torch.tensor(
            seq_lens, dtype=torch.int32, device=DEVICE
        ).cumsum(0)
    else:
        query_start_loc = None
    return dict(
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int32, device=DEVICE),
        num_computed_tokens=torch.full(
            (max_num_reqs,), 100, dtype=torch.int32, device=DEVICE
        ),
        last_sampled_tokens=torch.full(
            (max_num_reqs, 1), -1, dtype=torch.int64, device=DEVICE
        ),
        output_bin_counts=(
            torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32, device=DEVICE)
            if with_bin_counts
            else None
        ),
        sampled_tokens=torch.tensor(
            sampled_tokens, dtype=torch.int32, device=DEVICE
        ),
        num_sampled=torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE),
        num_rejected=torch.tensor(num_rejected, dtype=torch.int32, device=DEVICE),
        query_start_loc=query_start_loc,
        all_token_ids=torch.full(
            (max_num_reqs, max_model_len), -9, dtype=torch.int32, device=DEVICE
        ),
        total_len=torch.tensor(total_len, dtype=torch.int32, device=DEVICE),
    )


@pytest.mark.parametrize(
    "idx_mapping,sampled_tokens,num_sampled,num_rejected,seq_lens,"
    "with_bin_counts,with_query_start_loc",
    [
        # real shape: 8 rows, 3 sampled slots each, 2 rejected (MTP2)
        ([0, 1, 2, 3, 4, 5, 6, 7], [[1, 2, 3]] * 8, [3] * 8, [0] * 8, [3] * 8,
         True, True),
        # rows with -1 (cudagraph padding) are skipped
        ([0, 1, 2, -1, -1, -1, -1, -1], [[4, 5, 6]] * 3, [3, 0, 2], [0, 3, 1],
         [3, 3, 3, 0, 0, 0, 0, 0], True, True),
        # no bin counts, no query_start_loc (the non-PP decode path)
        ([0, 1], [[7, 7, 7]] * 2, [3, 3], [0, 0], [3, 3], False, False),
        # repeated token ids exercise the serial read-modify-write
        ([0, 1], [[11, 11, 11], [12, 11, 12]], [3, 3], [0, 0], [3, 3], True,
         True),
        # zero sampled tokens everywhere
        ([0, 1, 2], [[0, 0, 0]] * 3, [0, 0, 0], [0, 0, 0], [0, 0, 0], True,
         True),
        # single row, single token
        ([0], [[13]], [1], [0], [1], True, True),
        # computed_delta == 0 must not touch num_computed_tokens
        ([0, 1], [[1, 2, 3], [4, 5, 6]], [3, 3], [3, 3], [3, 3], True, True),
        # zero rows: both paths are inert
        ([], [], [], [], [], True, True),
    ],
)
def test_post_update_parity(
    idx_mapping, sampled_tokens, num_sampled, num_rejected, seq_lens,
    with_bin_counts, with_query_start_loc,
):
    vocab_size = 16
    steps = max((len(s) for s in sampled_tokens), default=0) or 1
    padded = [s + [0] * (steps - len(s)) for s in sampled_tokens]
    args = build_post_update_case(
        idx_mapping, padded, num_sampled, num_rejected, seq_lens, seq_lens,
        max_num_reqs=8, max_model_len=32, vocab_size=vocab_size,
        with_bin_counts=with_bin_counts,
        with_query_start_loc=with_query_start_loc,
    )
    keys = [
        "num_computed_tokens", "last_sampled_tokens", "sampled_tokens",
        "num_sampled", "num_rejected", "all_token_ids", "total_len",
    ]
    ref = {k: args[k].clone() for k in keys}
    got = {k: args[k].clone() for k in keys}
    ref_bin = args["output_bin_counts"].clone() if with_bin_counts else None
    got_bin = args["output_bin_counts"].clone() if with_bin_counts else None

    triton_post_update(
        args["idx_mapping"], ref["num_computed_tokens"],
        ref["last_sampled_tokens"], ref_bin, ref["sampled_tokens"],
        ref["num_sampled"], ref["num_rejected"], args["query_start_loc"],
        ref["all_token_ids"], ref["total_len"],
    )
    hip_post_update(
        args["idx_mapping"], got["num_computed_tokens"],
        got["last_sampled_tokens"], got_bin, got["sampled_tokens"],
        got["num_sampled"], got["num_rejected"], args["query_start_loc"],
        got["all_token_ids"], got["total_len"],
    )
    sync()

    for k in keys:
        assert torch.equal(ref[k], got[k]), k
    if with_bin_counts:
        assert torch.equal(ref_bin, got_bin)

    # All sampled tokens really landed in all_token_ids at total_len.
    for r, st in enumerate(idx_mapping):
        if st < 0:
            continue
        ns = num_sampled[r]
        tl0 = 5
        assert int(got["total_len"][st]) == tl0 + ns
        if ns:
            assert torch.equal(
                got["all_token_ids"][st, tl0 : tl0 + ns],
                torch.tensor(padded[r][:ns], dtype=torch.int32, device=DEVICE),
            )
            assert int(got["last_sampled_tokens"][st, 0]) == padded[r][ns - 1]
        if with_bin_counts:
            for tok in padded[r][:ns]:
                assert int(got_bin[st, tok]) == padded[r][:ns].count(tok)


@pytest.mark.skipif(not is_uva_available(), reason="UVA not available")
def test_post_update_uva_all_token_ids():
    """all_token_ids is a UVA (pinned-host) tensor in the real runner."""
    max_num_reqs, max_model_len = 4, 64
    uva = StagedWriteTensor(
        (max_num_reqs, max_model_len), dtype=torch.int32, device=DEVICE,
        uva_instead_of_gpu=True,
    )
    uva.gpu.fill_(-9)
    uva.apply_write()
    sync()

    args = build_post_update_case(
        [0, 1], [[3, 4, 5], [6, 7, 8]], [3, 3], [0, 0], [3, 3], [3, 3],
        max_num_reqs=max_num_reqs, max_model_len=max_model_len, vocab_size=16,
    )
    args["all_token_ids"] = uva.gpu
    hip_post_update(
        args["idx_mapping"], args["num_computed_tokens"].clone(),
        args["last_sampled_tokens"].clone(),
        args["output_bin_counts"].clone(), args["sampled_tokens"],
        args["num_sampled"], args["num_rejected"], args["query_start_loc"],
        args["all_token_ids"], args["total_len"].clone(),
    )
    sync()
    assert torch.equal(uva.gpu[0, 5:8], torch.tensor([3, 4, 5], device=DEVICE))
    assert torch.equal(uva.gpu[1, 5:8], torch.tensor([6, 7, 8], device=DEVICE))
    # Untouched rows keep their sentinel.
    assert (uva.gpu[2:] == -9).all()


# ---------------------------------------------------------------------------
# _expand_idx_mapping_kernel
# ---------------------------------------------------------------------------
def triton_expand(idx_mapping, cu_num_logits, max_expand_len):
    num_reqs = idx_mapping.shape[0]
    total = int(cu_num_logits[-1]) if num_reqs else 0
    out = idx_mapping.new_full((total,), -12345)
    loc = idx_mapping.new_full((total,), -12345)
    if num_reqs:
        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            out,
            loc,
            cu_num_logits,
            BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
        )
    return out, loc


def hip_expand(idx_mapping, cu_num_logits, max_expand_len):
    total = int(cu_num_logits[-1]) if idx_mapping.shape[0] else 0
    out = idx_mapping.new_full((total,), -12345)
    loc = idx_mapping.new_full((total,), -12345)
    OP.expand_idx_mapping_rdna2(idx_mapping, out, loc, cu_num_logits)
    return out, loc


@pytest.mark.parametrize(
    "idx_mapping,per_req_logits,max_expand_len",
    [
        # real serving shape: 8 rows x (2 draft + 1 bonus)
        ([0, 1, 2, 3, 4, 5, 6, 7], [3] * 8, 3),
        # negative (padding) rows are copied through verbatim
        ([0, -1, 2, -1], [3, 3, 3, 3], 3),
        # uneven per-row counts
        ([0, 1, 2], [0, 1, 5], 5),
        # single row, single token
        ([7], [1], 1),
        # not a multiple of the Triton tile
        ([0, 1], [1023, 1024], 1024),
        ([0, 1], [1024, 1025], 1025),
        # zero rows: both paths are inert
        ([], [], 1),
    ],
)
def test_expand_idx_mapping_parity(idx_mapping, per_req_logits, max_expand_len):
    num_reqs = len(idx_mapping)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(per_req_logits, dtype=torch.int32, device=DEVICE).cumsum(0)
    im = torch.tensor(idx_mapping, dtype=torch.int32, device=DEVICE)

    out_t, loc_t = triton_expand(im, cu, max_expand_len)
    out_h, loc_h = hip_expand(im, cu, max_expand_len)
    sync()

    assert torch.equal(out_t, out_h)
    assert torch.equal(loc_t, loc_h)
    assert (out_h != -12345).all()
    assert (loc_h != -12345).all()
    for r, st in enumerate(idx_mapping):
        start = int(cu[r])
        end = int(cu[r + 1])
        if end == start:
            continue
        assert (out_h[start:end] == st).all()
        assert torch.equal(
            loc_h[start:end],
            torch.arange(end - start, dtype=torch.int32, device=DEVICE),
        )


# ---------------------------------------------------------------------------
# No silent fallback: the public wrappers must take the HIP path on RDNA2.
# ---------------------------------------------------------------------------
class _NoLaunch:
    def __getitem__(self, _key):
        raise AssertionError("Triton kernel launched on the RDNA2 path")


def test_wrappers_do_not_fall_back_to_triton(monkeypatch):
    import vllm.v1.worker.gpu.input_batch as ib

    for name in (
        "_prepare_prefill_inputs_kernel",
        "_prepare_pos_seq_lens_kernel",
        "_combine_sampled_and_draft_tokens_kernel",
        "_get_num_sampled_and_rejected_kernel",
        "_post_update_kernel",
        "_expand_idx_mapping_kernel",
    ):
        monkeypatch.setattr(ib, name, _NoLaunch())

    num_reqs, qlen = 8, 3
    idx = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = torch.full((num_reqs,), qlen, dtype=torch.int32,
                                     device=DEVICE).cumsum(0)
    all_token_ids = torch.zeros((num_reqs, 64), dtype=torch.int32, device=DEVICE)
    prefill_len = torch.full((num_reqs,), 10, dtype=torch.int32, device=DEVICE)
    num_computed = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)

    ib.prepare_prefill_inputs(
        torch.zeros(num_reqs * qlen, dtype=torch.int32, device=DEVICE),
        torch.zeros((1, num_reqs), dtype=torch.int32, device=DEVICE),
        idx, query_start_loc, all_token_ids, prefill_len, num_computed,
    )
    ib.prepare_pos_seq_lens(
        idx, query_start_loc, num_computed,
        torch.zeros(num_reqs * qlen, dtype=torch.int64, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
    )
    cu = torch.arange(0, (num_reqs + 1) * qlen, qlen, dtype=torch.int32,
                      device=DEVICE)
    logits_indices = ib.combine_sampled_and_draft_tokens(
        torch.zeros(num_reqs * qlen, dtype=torch.int32, device=DEVICE), idx,
        torch.zeros((num_reqs, 1), dtype=torch.int64, device=DEVICE),
        query_start_loc,
        torch.full((num_reqs,), 10 + qlen, dtype=torch.int32, device=DEVICE),
        prefill_len,
        torch.zeros((num_reqs, 2), dtype=torch.int64, device=DEVICE),
        cu, num_reqs * qlen, 1,
    )
    num_sampled, num_rejected = ib.get_num_sampled_and_rejected(
        torch.ones(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.full((num_reqs,), 10 + qlen, dtype=torch.int32, device=DEVICE),
        cu, idx, prefill_len,
    )
    ib.post_update(
        idx, num_computed,
        torch.zeros((num_reqs, 1), dtype=torch.int64, device=DEVICE),
        torch.zeros((num_reqs, 32), dtype=torch.int32, device=DEVICE),
        torch.zeros((num_reqs, 3), dtype=torch.int32, device=DEVICE),
        num_sampled, num_rejected, query_start_loc, all_token_ids,
        torch.full((num_reqs,), 5, dtype=torch.int32, device=DEVICE),
    )
    expanded_idx, expanded_pos = ib.expand_idx_mapping(
        idx, num_reqs * qlen, cu, qlen
    )
    sync()

    assert logits_indices.shape == (num_reqs * qlen,)
    assert expanded_idx.shape == expanded_pos.shape == (num_reqs * qlen,)
    assert torch.equal(expanded_pos, torch.arange(qlen, dtype=torch.int32,
                                                  device=DEVICE).repeat(num_reqs))
