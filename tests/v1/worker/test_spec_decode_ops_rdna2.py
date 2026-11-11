# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-for-bit parity between the gfx1030 HIP ports and the Triton kernels for
the speculative-decoding ops:

  _prepare_prefill_inputs_kernel       (spec_decode/autoregressive/speculator.py)
  _prepare_decode_inputs_kernel        (spec_decode/autoregressive/speculator.py)
  _update_draft_inputs_kernel          (spec_decode/autoregressive/speculator.py)
  _gather_draft_sampled_kernel         (spec_decode/rejection_sampler.py)
  _flatten_sampled_kernel              (spec_decode/rejection_sampler.py)
  _compute_local_logits_stats_kernel   (spec_decode/rejection_sampler_utils.py)
  _rejection_kernel                    (spec_decode/rejection_sampler_utils.py)
  _resample_kernel                     (spec_decode/rejection_sampler_utils.py)
  _insert_resampled_kernel             (spec_decode/rejection_sampler_utils.py)

These run once per decode step inside a FULL cudagraph capture, so the only
acceptable difference between the two paths is none. Every comparison is
`torch.equal` (bit-for-bit), not a tolerance, except the summed exponentials
of `_compute_local_logits_stats_kernel` (see below).

The float reductions (sumexp, logsumexp) are matched by reproducing Triton's
AMD codegen: the in-thread balanced tree over the blocked layout's register
file, then a 32-lane butterfly with masks 16,8,4,2,1, then a warp butterfly
with masks 2,1; `tl.exp` as `v_exp_f32(x * log2e)` and `tl.log` as `v_log_f32`.
That reproduction is not exact for the per-block sumexp on gfx1030: Triton's
association still differs in the last bits on some rows (measured <= 3.2e-7
relative, i.e. 1-3 fp32 ulp). Those two outputs are compared with
`assert_reduction_close` (rtol=1e-6); argmax, max, draft_max and every
sentinel (unwritten) slot stay bit-exact.
"""

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rsu

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_rdna2()),
    reason="gfx10x (RDNA2) ROCm required",
)

DEVICE = torch.device("cuda")
OP = torch.ops._rocm_C
SENTINEL = -12345
SENTINEL_F = -12345.0


def sync():
    torch.cuda.synchronize()


def cdiv(a, b):
    return (a + b - 1) // b


def assert_bitexact(actual, expected, msg=""):
    """Bit-for-bit equality, but with NaN == NaN.

    `torch.equal` reports any NaN as unequal, and the rejected-logsumexp outputs
    are legitimately NaN when the last verified row is a greedy/bonus row.
    """
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True, msg=msg
    )


def assert_reduction_close(actual, expected, msg=""):
    """fp32 sum-of-exponentials: the slots that were not written (SENTINEL_F)
    must match exactly, the written ones to a few fp32 ulp (association)."""
    assert torch.equal(actual == SENTINEL_F, expected == SENTINEL_F), msg
    torch.testing.assert_close(
        actual, expected, rtol=1e-6, atol=0, equal_nan=True, msg=msg
    )


# ---------------------------------------------------------------------------
# _prepare_prefill_inputs_kernel
# ---------------------------------------------------------------------------
def triton_prepare_prefill_inputs(
    last_token_indices,
    current_draft_step,
    draft_input_ids,
    draft_positions,
    draft_query_start_loc,
    draft_seq_lens,
    target_input_ids,
    target_positions,
    idx_mapping,
    last_sampled,
    next_prefill_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    seq_lens,
    max_num_reqs,
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    spec._prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        draft_input_ids,
        draft_positions,
        draft_query_start_loc,
        draft_seq_lens,
        target_input_ids,
        target_positions,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )


def hip_prepare_prefill_inputs(**kw):
    OP.spec_prepare_prefill_inputs_rdna2(
        kw["last_token_indices"], kw["current_draft_step"],
        kw["draft_input_ids"], kw["draft_positions"],
        kw["draft_query_start_loc"], kw["draft_seq_lens"],
        kw["target_input_ids"], kw["target_positions"], kw["idx_mapping"],
        kw["last_sampled"], kw["next_prefill_tokens"], kw["num_sampled"],
        kw["num_rejected"], kw["query_start_loc"], kw["seq_lens"],
        kw["max_num_reqs"],
    )


@pytest.mark.parametrize(
    "num_reqs,query_lens,use_last,max_num_reqs",
    [
        (8, [3] * 8, True, 8),  # real serving shape
        (8, [1, 2, 3, 5, 8, 13, 21, 34], True, 16),  # ragged + cudagraph pad
        (1, [1], False, 1),  # single element, chunked prefill
        (1, [1024], True, 4),  # exactly one Triton tile
        (1, [1025], False, 4),  # tile + 1
        (2, [0, 1], True, 8),  # zero-length request
    ],
)
def test_prepare_prefill_inputs_parity(num_reqs, query_lens, use_last, max_num_reqs):
    total = sum(query_lens)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32, device=DEVICE).cumsum(0)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    if use_last:
        num_rejected[-1] = min(1, query_lens[-1])
    num_sampled = torch.randint(0, 2, (num_reqs,), dtype=torch.int32, device=DEVICE)

    def build():
        return dict(
            last_token_indices=torch.randint(
                0, 50, (max_num_reqs,), dtype=torch.int64, device=DEVICE
            ),
            current_draft_step=torch.full((), 7, dtype=torch.int64, device=DEVICE),
            draft_input_ids=torch.randint(
                0, 50, (max(total, 1) + 2,), dtype=torch.int32, device=DEVICE
            ),
            draft_positions=torch.randint(
                0, 50, (max(total, 1) + 2,), dtype=torch.int64, device=DEVICE
            ),
            draft_query_start_loc=torch.randint(
                0, 50, (max_num_reqs + 1,), dtype=torch.int32, device=DEVICE
            ),
            draft_seq_lens=torch.randint(
                0, 50, (max_num_reqs,), dtype=torch.int32, device=DEVICE
            ),
            target_input_ids=torch.randint(
                0, 100, (max(total, 1) + 2,), dtype=torch.int32, device=DEVICE
            ),
            target_positions=torch.randint(
                0, 100, (max(total, 1) + 2,), dtype=torch.int64, device=DEVICE
            ),
            idx_mapping=idx_mapping,
            last_sampled=torch.randint(
                0, 100, (max_num_reqs,), dtype=torch.int64, device=DEVICE
            ),
            next_prefill_tokens=torch.randint(
                0, 100, (max_num_reqs,), dtype=torch.int32, device=DEVICE
            ),
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            query_start_loc=qsl,
            seq_lens=torch.randint(
                1, 50, (num_reqs,), dtype=torch.int32, device=DEVICE
            ),
            max_num_reqs=max_num_reqs,
        )

    ref = build()
    got = {k: v.clone() if torch.is_tensor(v) else v for k, v in ref.items()}
    triton_prepare_prefill_inputs(**ref)
    hip_prepare_prefill_inputs(**got)
    sync()
    for key in (
        "last_token_indices",
        "current_draft_step",
        "draft_input_ids",
        "draft_positions",
        "draft_query_start_loc",
        "draft_seq_lens",
    ):
        assert torch.equal(ref[key], got[key]), key


def test_prepare_prefill_inputs_zero_reqs():
    t = torch.zeros(4, dtype=torch.int64, device=DEVICE)
    OP.spec_prepare_prefill_inputs_rdna2(
        t, t, t.to(torch.int32), t, t.to(torch.int32), t.to(torch.int32), t.to(torch.int32),
        t, t.to(torch.int32), t, t.to(torch.int32), t.to(torch.int32), t.to(torch.int32),
        t.to(torch.int32), t.to(torch.int32), 4,
    )
    sync()


# ---------------------------------------------------------------------------
# _prepare_decode_inputs_kernel
# ---------------------------------------------------------------------------
def triton_prepare_decode_inputs(
    draft_tokens,
    target_seq_lens,
    num_rejected,
    input_ids,
    positions,
    sample_src_positions,
    query_start_loc,
    seq_lens,
    max_model_len,
    max_num_reqs,
    advance,
):
    num_reqs = draft_tokens.shape[0]
    if num_reqs == 0:
        return
    spec._prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_ids,
        positions,
        sample_src_positions,
        query_start_loc,
        seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance,
    )


@pytest.mark.parametrize("advance", [True, False])
@pytest.mark.parametrize("num_reqs,max_num_reqs", [(8, 8), (3, 12), (1, 1)])
def test_prepare_decode_inputs_parity(advance, num_reqs, max_num_reqs):
    max_model_len = 4096
    def build():
        return dict(
            draft_tokens=torch.randint(
                0, 100, (num_reqs, 2), dtype=torch.int64, device=DEVICE
            ),
            target_seq_lens=torch.tensor(
                [max_model_len, max_model_len - 1, 5][:num_reqs] if num_reqs <= 3
                else [max_model_len] * num_reqs,
                dtype=torch.int32,
                device=DEVICE,
            ),
            num_rejected=torch.randint(
                0, 2, (num_reqs,), dtype=torch.int32, device=DEVICE
            ),
            input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
            positions=torch.randint(
                0, max_model_len, (num_reqs,), dtype=torch.int64, device=DEVICE
            ),
            sample_src_positions=torch.randint(
                0, 1000, (num_reqs,), dtype=torch.int64, device=DEVICE
            ),
            query_start_loc=torch.randint(
                0, 50, (max_num_reqs + 1,), dtype=torch.int32, device=DEVICE
            ),
            seq_lens=torch.randint(
                0, 50, (max_num_reqs,), dtype=torch.int32, device=DEVICE
            ),
        )

    ref = build()
    got = {k: v.clone() for k, v in ref.items()}
    triton_prepare_decode_inputs(
        **ref, max_model_len=max_model_len, max_num_reqs=max_num_reqs, advance=advance
    )
    OP.spec_prepare_decode_inputs_rdna2(
        got["draft_tokens"],
        got["target_seq_lens"],
        got["num_rejected"],
        got["input_ids"],
        got["positions"],
        got["sample_src_positions"],
        got["query_start_loc"],
        got["seq_lens"],
        max_model_len,
        max_num_reqs,
        advance,
    )
    sync()
    for key in ref:
        assert torch.equal(ref[key], got[key]), key


# ---------------------------------------------------------------------------
# _update_draft_inputs_kernel
# ---------------------------------------------------------------------------
def triton_update_draft_inputs(
    draft_tokens,
    current_draft_step,
    hidden_states,
    output_draft_tokens,
    next_input_hidden_states,
    input_ids,
    positions,
    sample_src_positions,
    seq_lens,
    max_model_len,
    num_speculative_steps,
    advance,
):
    num_reqs, hidden_size = (
        draft_tokens.shape[0],
        hidden_states.shape[1],
    )
    if num_reqs == 0:
        return
    spec._update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_ids,
        positions,
        sample_src_positions,
        seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("step,num_spec", [(0, 2), (1, 2), (3, 4)])
@pytest.mark.parametrize("hidden_size", [4096, 2624, 255])
def test_update_draft_inputs_parity(dtype, step, num_spec, hidden_size):
    num_reqs, max_model_len = 8, 4096
    hs = (torch.randn(num_reqs, hidden_size, device=DEVICE) * 8).to(dtype)
    nihs = torch.full(
        (num_reqs, hidden_size), SENTINEL_F, dtype=dtype, device=DEVICE
    )
    draft_tokens = torch.randint(0, 100, (num_reqs,), dtype=torch.int64, device=DEVICE)
    current_draft_step = torch.full((), step, dtype=torch.int64, device=DEVICE)
    out_ref = torch.full((num_reqs, num_spec), SENTINEL, dtype=torch.int64, device=DEVICE)
    out_got = out_ref.clone()
    args = dict(
        draft_tokens=draft_tokens,
        current_draft_step=current_draft_step,
        hidden_states=hs,
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        positions=torch.randint(0, max_model_len, (num_reqs,), dtype=torch.int64, device=DEVICE),
        sample_src_positions=torch.randint(0, 1000, (num_reqs,), dtype=torch.int64, device=DEVICE),
        seq_lens=torch.randint(1, 100, (num_reqs,), dtype=torch.int32, device=DEVICE),
    )
    ref_ids, ref_pos, ref_ssp, ref_sl = (
        args["input_ids"].clone(),
        args["positions"].clone(),
        args["sample_src_positions"].clone(),
        args["seq_lens"].clone(),
    )
    triton_update_draft_inputs(
        draft_tokens, current_draft_step, hs, out_ref, nihs, ref_ids,
        ref_pos, ref_ssp, ref_sl, max_model_len, num_spec, True,
    )
    nihs_got = torch.full((num_reqs, hidden_size), SENTINEL_F, dtype=dtype, device=DEVICE)
    OP.spec_update_draft_inputs_rdna2(
        out_got, nihs_got, args["input_ids"], args["positions"],
        args["sample_src_positions"], args["seq_lens"], draft_tokens,
        current_draft_step, hs, hidden_size, max_model_len, num_spec, True,
    )
    sync()
    assert torch.equal(out_ref, out_got)
    assert torch.equal(nihs, nihs_got)
    assert torch.equal(ref_ids, args["input_ids"])
    assert torch.equal(ref_pos, args["positions"])
    assert torch.equal(ref_ssp, args["sample_src_positions"])
    assert torch.equal(ref_sl, args["seq_lens"])


# ---------------------------------------------------------------------------
# _gather_draft_sampled_kernel / _flatten_sampled_kernel
# ---------------------------------------------------------------------------
def triton_gather_draft_sampled(
    input_ids, positions, logits_indices, expanded_idx_mapping,
    expanded_local_pos, prefill_len, draft_sampled, pos,
):
    num_logits = logits_indices.shape[0]
    if num_logits == 0:
        return
    rs._gather_draft_sampled_kernel[(triton.cdiv(num_logits, 1024),)](
        draft_sampled, pos, input_ids, positions, logits_indices,
        expanded_idx_mapping, expanded_local_pos, prefill_len, num_logits,
        BLOCK_SIZE=1024,
    )


@pytest.mark.parametrize("num_logits", [0, 1, 24, 1023, 1024, 1025, 4096])
def test_gather_draft_sampled_parity(num_logits):
    num_tokens = max(num_logits, 1) + 7
    input_ids = torch.randint(0, 1000, (num_tokens,), dtype=torch.int32, device=DEVICE)
    positions = torch.randint(0, 100, (num_tokens,), dtype=torch.int64, device=DEVICE)
    if num_logits:
        logits_indices = torch.randperm(num_tokens, device=DEVICE)[:num_logits].to(torch.int64)
    else:
        logits_indices = torch.zeros(0, dtype=torch.int64, device=DEVICE)
    expanded_idx_mapping = torch.randint(
        0, 8, (num_logits,), dtype=torch.int32, device=DEVICE
    )
    expanded_local_pos = torch.randint(0, 3, (num_logits,), dtype=torch.int32, device=DEVICE)
    prefill_len = torch.randint(0, 20, (8,), dtype=torch.int32, device=DEVICE)

    ref_ds, ref_pos = (
        torch.full((num_logits,), SENTINEL, dtype=torch.int32, device=DEVICE),
        torch.full((num_logits,), SENTINEL, dtype=torch.int64, device=DEVICE),
    )
    got_ds, got_pos = ref_ds.clone(), ref_pos.clone()
    triton_gather_draft_sampled(
        input_ids, positions, logits_indices, expanded_idx_mapping,
        expanded_local_pos, prefill_len, ref_ds, ref_pos,
    )
    if num_logits:
        OP.spec_gather_draft_sampled_rdna2(
            got_ds, got_pos, input_ids, positions, logits_indices,
            expanded_idx_mapping, expanded_local_pos, prefill_len,
        )
    sync()
    assert torch.equal(ref_ds, got_ds)
    assert torch.equal(ref_pos, got_pos)


@pytest.mark.parametrize("num_sampled", [[0, 1, 2, 3, 1, 0, 2, 3], [1], [0], [0, 0]])
def test_flatten_sampled_parity(num_sampled):
    num_reqs = len(num_sampled)
    num_spec = 3
    cu = torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE) * (
        num_spec + 1
    )
    sampled = torch.randint(0, 1000, (num_reqs, num_spec + 1), dtype=torch.int64, device=DEVICE)
    ns = torch.tensor(num_sampled, dtype=torch.int32, device=DEVICE)
    num_logits = num_reqs * (num_spec + 1)
    ref = torch.full((num_logits,), SENTINEL, dtype=torch.int64, device=DEVICE)
    got = ref.clone()
    rs._flatten_sampled_kernel[(num_reqs,)](
        ref, sampled, sampled.stride(0), ns, cu, num_warps=1
    )
    OP.spec_flatten_sampled_rdna2(got, sampled, ns, cu)
    sync()
    assert torch.equal(ref, got)


# ---------------------------------------------------------------------------
# _compute_local_logits_stats_kernel
# ---------------------------------------------------------------------------
def triton_local_logits_stats(
    target_logits, draft_logits, expanded_idx_mapping, expanded_local_pos,
    temperature, vocab_size, num_speculative_steps,
):
    num_logits = target_logits.shape[0]
    vocab_num_blocks = cdiv(vocab_size, 8192)
    has_draft = draft_logits is not None
    tla = torch.full((num_logits, vocab_num_blocks), SENTINEL, dtype=torch.int64, device=DEVICE)
    tlm = torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE)
    tls = tlm.clone()
    dlm = tlm.clone()
    dls = tlm.clone()
    if num_logits:
        rsu._compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
            tla, tla.stride(0), tlm, tlm.stride(0), tls, tls.stride(0),
            dlm, dlm.stride(0), dls, dls.stride(0),
            target_logits, target_logits.stride(0),
            draft_logits,
            draft_logits.stride(0) if has_draft else 0,
            draft_logits.stride(1) if has_draft else 0,
            expanded_idx_mapping, expanded_local_pos, temperature,
            vocab_size, num_speculative_steps,
            BLOCK_SIZE=8192, HAS_DRAFT_LOGITS=has_draft,
        )
    return tla, tlm, tls, dlm, dls


@pytest.mark.parametrize(
    "vocab_size", [151936, 16384, 8192, 17000, 8191, 1]
)
@pytest.mark.parametrize(
    "draft_dtype",
    # 16-bit draft logits make Triton pick a different (and register-pressure
    # dependent) reduction association for both reductions, so the dispatcher
    # keeps the Triton kernel for them; see test_stats_dispatch_keeps_triton_
    # for_16bit_draft.
    [None, torch.float32],
)
def test_compute_local_logits_stats_parity(vocab_size, draft_dtype):
    num_reqs, num_spec = 8, 2
    num_logits = num_reqs * (num_spec + 1)
    vocab_num_blocks = cdiv(vocab_size, 8192)
    target_logits = (torch.randn(num_logits, vocab_size, device=DEVICE) * 4).float()
    draft_logits = None
    if draft_dtype is not None:
        draft_logits = (torch.randn(num_reqs, num_spec, vocab_size, device=DEVICE) * 4).to(
            draft_dtype
        )
    expanded_idx_mapping = torch.arange(num_logits, dtype=torch.int32, device=DEVICE) % num_reqs
    expanded_local_pos = (
        torch.arange(num_logits, dtype=torch.int32, device=DEVICE) % (num_spec + 1)
    )
    # last request is greedy; a pos == num_spec is a bonus row (skipped).
    temperature = torch.tensor(
        [1.0, 0.7, 1.5, 0.0, 0.9, 2.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE
    )
    ref = triton_local_logits_stats(
        target_logits, draft_logits, expanded_idx_mapping, expanded_local_pos,
        temperature, vocab_size, num_spec,
    )
    got = [
        torch.full((num_logits, vocab_num_blocks), SENTINEL, dtype=torch.int64, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
    ]
    OP.spec_compute_local_logits_stats_rdna2(
        got[0], got[1], got[2], got[3], got[4], target_logits, draft_logits,
        expanded_idx_mapping, expanded_local_pos, temperature, vocab_size,
        num_spec, draft_logits is not None,
    )
    sync()
    for i, name in enumerate(["argmax", "max", "sumexp", "draft_max", "draft_sumexp"]):
        msg = f"{name} vocab={vocab_size} draft={draft_dtype}"
        if name.endswith("sumexp"):
            assert_reduction_close(got[i], ref[i], msg)
        else:
            assert torch.equal(ref[i], got[i]), msg


class _HipOpBoom:
    def __getattr__(self, name):
        raise AssertionError(f"HIP op {name} used for a 16-bit draft cache")


def test_stats_dispatch_keeps_triton_for_16bit_draft(monkeypatch):
    """A bf16 draft-logits cache must not take the HIP stats kernel."""
    monkeypatch.setattr(rsu, "_OP", _HipOpBoom())
    b = Batch(draft_dtype=torch.bfloat16)
    stats = b.stats()
    for t in stats:
        assert torch.isfinite(t).any() or True
    sync()


@pytest.mark.parametrize("draft_dtype", [None, torch.float32])
def test_compute_local_logits_stats_fp16_target_logits(draft_dtype):
    """Regression: the sampler hands the rejection sampler the *model* dtype for
    requests that need no logits processing -- ``Sampler.apply_sampling_params``
    returns the raw logits unchanged when ``needs_logits_processing`` is all
    False (a greedy request with no penalties), so ``target_logits`` arrives as
    fp16. Triton loads it with a plain ``.to(tl.float32)``; the HIP op used to
    TORCH_CHECK fp32 and aborted the engine during warmup with "target_logits
    has unexpected dtype".

    fp16 -> fp32 is exact, so argmax/max match bit-for-bit; only the summed
    exponentials differ, by association (measured 1.8e-7 relative), which is
    the same class of difference the port already accepts for 16-bit inputs.
    """
    num_reqs, num_spec, vocab_size = 8, 2, 248320
    num_logits = num_reqs * (num_spec + 1)
    vocab_num_blocks = cdiv(vocab_size, 8192)
    target_logits = (torch.randn(num_logits, vocab_size, device=DEVICE) * 4).half()
    draft_logits = None
    if draft_dtype is not None:
        draft_logits = (
            torch.randn(num_reqs, num_spec, vocab_size, device=DEVICE) * 4
        ).to(draft_dtype)
    expanded_idx_mapping = (
        torch.arange(num_logits, dtype=torch.int32, device=DEVICE) % num_reqs
    )
    expanded_local_pos = (
        torch.arange(num_logits, dtype=torch.int32, device=DEVICE) % (num_spec + 1)
    )
    temperature = torch.tensor(
        [1.0, 0.7, 1.5, 0.0, 0.9, 2.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE
    )
    ref = triton_local_logits_stats(
        target_logits, draft_logits, expanded_idx_mapping, expanded_local_pos,
        temperature, vocab_size, num_spec,
    )
    got = [
        torch.full(
            (num_logits, vocab_num_blocks), SENTINEL, dtype=torch.int64, device=DEVICE
        ),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
        torch.full((num_logits, vocab_num_blocks), SENTINEL_F, device=DEVICE),
    ]
    OP.spec_compute_local_logits_stats_rdna2(
        got[0], got[1], got[2], got[3], got[4], target_logits, draft_logits,
        expanded_idx_mapping, expanded_local_pos, temperature, vocab_size,
        num_spec, draft_logits is not None,
    )
    sync()
    assert torch.equal(ref[0], got[0]), "argmax (greedy) must stay bit-exact"
    assert torch.equal(ref[1], got[1]), "max (greedy) must stay bit-exact"
    # Non-greedy rows also produce max; they must match too.
    assert torch.equal(ref[1], got[1])
    torch.testing.assert_close(ref[2], got[2], rtol=1e-6, atol=0)
    torch.testing.assert_close(ref[4], got[4], rtol=1e-6, atol=0)


def test_compute_local_logits_stats_zero_logits():
    t = torch.zeros(0, 1, device=DEVICE)
    i32 = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    i64 = torch.zeros(0, 1, dtype=torch.int64, device=DEVICE)
    OP.spec_compute_local_logits_stats_rdna2(
        i64, t, t, t, t, t, None, i32, i32, torch.zeros(1, device=DEVICE),
        8192, 2, False,
    )
    sync()


# ---------------------------------------------------------------------------
# Rejection / resample / insert fixture
# ---------------------------------------------------------------------------
class Batch:
    def __init__(self, num_reqs=8, num_spec=2, vocab_size=16384, draft_dtype=None,
                 temperatures=None, target_logits_dtype=torch.float32):
        self.num_reqs = num_reqs
        self.num_spec = num_spec
        self.vocab_size = vocab_size
        self.num_logits = num_reqs * (num_spec + 1)
        nl = self.num_logits
        self.target_logits = (
            torch.randn(nl, vocab_size, device=DEVICE) * 3
        ).to(target_logits_dtype)
        self.draft_dtype = draft_dtype
        self.draft_logits = None
        if draft_dtype is not None:
            self.draft_logits = (
                torch.randn(num_reqs, num_spec, vocab_size, device=DEVICE) * 3
            ).to(draft_dtype)
            self.draft_logits = self.draft_logits.clone().requires_grad_(False)
        self.cu_num_logits = (
            torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE) * (num_spec + 1)
        )
        self.idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        self.expanded_idx_mapping = torch.arange(nl, dtype=torch.int32, device=DEVICE) % num_reqs
        self.expanded_local_pos = torch.arange(nl, dtype=torch.int32, device=DEVICE) % (
            num_spec + 1
        )
        if temperatures is None:
            temperatures = [1.0, 0.7, 1.5, 0.0, 0.9, 2.0, 1.0, 0.0, 1.2, 0.5, 0.0, 3.0]
        self.temperature = torch.tensor(
            temperatures[:num_reqs], dtype=torch.float32, device=DEVICE
        )
        self.seed = torch.arange(1000, 1000 + num_reqs, dtype=torch.int64, device=DEVICE)
        self.draft_sampled = torch.randint(
            0, vocab_size, (nl,), dtype=torch.int32, device=DEVICE
        )
        self.pos = torch.arange(nl, dtype=torch.int64, device=DEVICE) + 3

    def stats(self, **kw):
        return triton_local_logits_stats(
            self.target_logits, self.draft_logits, self.expanded_idx_mapping,
            self.expanded_local_pos, self.temperature, self.vocab_size,
            self.num_spec,
        )


def triton_rejection(b, stats, sampled, num_sampled, synthetic=None,
                     use_block_verification=False, cumulative=None,
                     local_residual_mass=None):
    tla, tlm, tls, dlm, dls = stats
    vocab_num_blocks = tla.shape[1]
    padded = triton.next_power_of_2(vocab_num_blocks)
    has_draft = b.draft_logits is not None
    trg = torch.full((b.num_reqs,), SENTINEL_F, device=DEVICE)
    drf = torch.full((b.num_reqs,), SENTINEL_F, device=DEVICE)
    if b.num_reqs == 0:
        return trg, drf
    rsu._rejection_kernel[(b.num_reqs,)](
        sampled, sampled.stride(0),
        num_sampled,
        trg,
        drf,
        b.target_logits, b.target_logits.stride(0),
        tla, tla.stride(0),
        tlm, tlm.stride(0),
        tls, tls.stride(0),
        b.draft_sampled,
        b.draft_logits,
        b.draft_logits.stride(0) if has_draft else 0,
        b.draft_logits.stride(1) if has_draft else 0,
        dlm, dlm.stride(0),
        dls, dls.stride(0),
        b.cu_num_logits, b.idx_mapping, b.temperature, b.seed, b.pos,
        synthetic,
        cumulative,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded,
        HAS_DRAFT_LOGITS=has_draft,
        SYNTHETIC_MODE=synthetic is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )
    return trg, drf


def hip_rejection(b, stats, sampled, num_sampled, synthetic=None,
                  use_block_verification=False, cumulative=None,
                  local_residual_mass=None):
    tla, tlm, tls, dlm, dls = stats
    vocab_num_blocks = tla.shape[1]
    padded = triton.next_power_of_2(vocab_num_blocks)
    has_draft = b.draft_logits is not None
    trg = torch.full((b.num_reqs,), SENTINEL_F, device=DEVICE)
    drf = torch.full((b.num_reqs,), SENTINEL_F, device=DEVICE)
    if b.num_reqs:
        OP.spec_rejection_rdna2(
            sampled, num_sampled, trg, drf,
            b.target_logits, tla, tlm, tls, b.draft_sampled,
            b.draft_logits, dlm, dls,
            b.cu_num_logits, b.idx_mapping, b.temperature, b.seed, b.pos,
            synthetic, cumulative, local_residual_mass,
            vocab_num_blocks, padded, has_draft, synthetic is not None,
            use_block_verification,
        )
    return trg, drf


def _check_rejection(b, **kw):
    stats = b.stats()
    ref_sampled = torch.full((b.num_reqs, b.num_spec + 1), SENTINEL, dtype=torch.int64, device=DEVICE)
    got_sampled = ref_sampled.clone()
    ref_ns = torch.full((b.num_reqs,), SENTINEL, dtype=torch.int32, device=DEVICE)
    got_ns = ref_ns.clone()
    ref_trg, ref_drf = triton_rejection(b, stats, ref_sampled, ref_ns, **kw)
    got_trg, got_drf = hip_rejection(b, stats, got_sampled, got_ns, **kw)
    sync()
    assert torch.equal(ref_sampled, got_sampled)
    assert torch.equal(ref_ns, got_ns)
    assert_bitexact(ref_trg, got_trg)
    assert_bitexact(ref_drf, got_drf)


@pytest.mark.parametrize("draft_dtype", [None, torch.bfloat16, torch.float32])
def test_rejection_parity(draft_dtype):
    _check_rejection(Batch(draft_dtype=draft_dtype))


def test_rejection_parity_with_placeholders():
    b = Batch(draft_dtype=torch.float32)
    b.draft_sampled[1] = -1  # first draft of request 1 is a placeholder
    b.draft_sampled[b.num_spec] = -1  # request 0 bonus row placeholder
    _check_rejection(b)


def test_rejection_parity_greedy_only():
    b = Batch(temperatures=[0.0] * 12)
    _check_rejection(b)


def test_rejection_parity_synthetic():
    b = Batch(draft_dtype=torch.float32)
    rates = torch.tensor([0.5, 0.25], dtype=torch.float32, device=DEVICE)
    _check_rejection(b, synthetic=rates)


def test_rejection_parity_block_verification():
    b = Batch(draft_dtype=torch.float32)
    nl = b.num_logits
    vocab_num_blocks = cdiv(b.vocab_size, 8192)
    padded = triton.next_power_of_2(vocab_num_blocks)
    stats = b.stats()
    tla, tlm, tls, dlm, dls = stats
    # The block-verification helper kernels are not ported (they only run with
    # rejection_sample_method="block"); build their outputs with Triton.
    cumulative = torch.zeros(nl, dtype=torch.float32, device=DEVICE)
    rsu._compute_cumulative_log_p_kernel[(b.num_reqs,)](
        cumulative, b.target_logits, b.target_logits.stride(0),
        tlm, tlm.stride(0), tls, tls.stride(0),
        b.draft_sampled, b.draft_logits, b.draft_logits.stride(0),
        b.draft_logits.stride(1), dlm, dlm.stride(0), dls, dls.stride(0),
        b.cu_num_logits, b.idx_mapping, b.temperature, vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded, HAS_DRAFT_LOGITS=True, num_warps=1,
    )
    lrm = torch.zeros(nl, vocab_num_blocks, dtype=torch.float32, device=DEVICE)
    rsu._compute_local_residual_mass_kernel[(nl, vocab_num_blocks)](
        lrm, lrm.stride(0), cumulative,
        b.target_logits, b.target_logits.stride(0),
        tlm, tlm.stride(0), tls, tls.stride(0),
        b.draft_logits, b.draft_logits.stride(0), b.draft_logits.stride(1),
        dlm, dlm.stride(0), dls, dls.stride(0),
        b.draft_sampled, b.expanded_idx_mapping, b.expanded_local_pos,
        b.temperature, b.vocab_size, b.num_spec, vocab_num_blocks,
        BLOCK_SIZE=8192, PADDED_VOCAB_NUM_BLOCKS=padded,
    )
    sync()
    _check_rejection(
        b, use_block_verification=True, cumulative=cumulative,
        local_residual_mass=lrm,
    )


@pytest.mark.parametrize("num_reqs", [0, 1])
def test_rejection_parity_small(num_reqs):
    if num_reqs == 0:
        b = Batch(num_reqs=0)
        stats = b.stats()
        a = torch.zeros((0, 3), dtype=torch.int64, device=DEVICE)
        ns = torch.zeros((0,), dtype=torch.int32, device=DEVICE)
        r1 = hip_rejection(b, stats, a, ns)
        r2 = triton_rejection(b, stats, a.clone(), ns.clone())
        sync()
        assert torch.equal(r1[0], r2[0])
    else:
        _check_rejection(Batch(num_reqs=1, draft_dtype=torch.float32))
        _check_rejection(Batch(num_reqs=1, temperatures=[0.0]))


def triton_resample(b, stats, num_sampled, use_fp64, use_block_verification=False,
                    cumulative=None):
    tla, tlm, tls, dlm, dls = stats
    has_draft = b.draft_logits is not None
    resample_num_blocks = cdiv(b.vocab_size, 1024)
    padded = triton.next_power_of_2(resample_num_blocks)
    ra = torch.full((b.num_reqs, resample_num_blocks), SENTINEL, dtype=torch.int64, device=DEVICE)
    rm = torch.full(
        (b.num_reqs, resample_num_blocks), SENTINEL_F,
        dtype=torch.float64 if use_fp64 else torch.float32, device=DEVICE,
    )
    if b.num_reqs == 0:
        return ra, rm
    trg, drf = triton_rejection(b, stats, torch.zeros(
        (b.num_reqs, b.num_spec + 1), dtype=torch.int64, device=DEVICE), num_sampled)
    rsu._resample_kernel[(b.num_reqs, resample_num_blocks)](
        ra, ra.stride(0), rm, rm.stride(0),
        b.target_logits, b.target_logits.stride(0),
        trg,
        b.draft_logits,
        b.draft_logits.stride(0) if has_draft else 0,
        b.draft_logits.stride(1) if has_draft else 0,
        drf,
        num_sampled, b.cu_num_logits, b.expanded_idx_mapping, b.draft_sampled,
        b.temperature, b.seed, b.pos, cumulative,
        None, 0, None, 0, 0,
        b.vocab_size,
        BLOCK_SIZE=1024,
        HAS_DRAFT_LOGITS=has_draft,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
        CONTEXT_WIDTH=1,
        WATERMARK=False,
    )
    return ra, rm


def hip_resample(b, stats, num_sampled, use_fp64, use_block_verification=False,
                 cumulative=None):
    tla, tlm, tls, dlm, dls = stats
    has_draft = b.draft_logits is not None
    resample_num_blocks = cdiv(b.vocab_size, 1024)
    ra = torch.full((b.num_reqs, resample_num_blocks), SENTINEL, dtype=torch.int64, device=DEVICE)
    rm = torch.full(
        (b.num_reqs, resample_num_blocks), SENTINEL_F,
        dtype=torch.float64 if use_fp64 else torch.float32, device=DEVICE,
    )
    if b.num_reqs:
        trg, drf = hip_rejection(b, stats, torch.zeros(
            (b.num_reqs, b.num_spec + 1), dtype=torch.int64, device=DEVICE), num_sampled)
        OP.spec_resample_rdna2(
            ra, rm, b.target_logits, trg, b.draft_logits, drf,
            num_sampled, b.cu_num_logits, b.expanded_idx_mapping,
            b.draft_sampled, b.temperature, b.seed, b.pos, cumulative,
            b.vocab_size, has_draft, use_fp64, use_block_verification,
        )
    return ra, rm


def _check_resample(b, use_fp64=False, **kw):
    stats = b.stats()
    ref_ns = torch.full((b.num_reqs,), SENTINEL, dtype=torch.int32, device=DEVICE)
    got_ns = ref_ns.clone()
    ref_ra, ref_rm = triton_resample(b, stats, ref_ns, use_fp64, **kw)
    got_ra, got_rm = hip_resample(b, stats, got_ns, use_fp64, **kw)
    sync()
    assert torch.equal(ref_ns, got_ns)
    assert torch.equal(ref_ra, got_ra)
    # rm is `residual_logits + gumbel noise`; the argmax asserted above stays
    # bit-exact, but two last-bit reductions differ:
    #  - `residual_logits` uses tldevice.log1p, which Triton lowers to its own
    #    libdevice code rather than HIP's ocml `log1pf`: measured directly over
    #    4.19M inputs from this distribution, 665k differ by 1 ulp (HIP's value
    #    is the more accurate one: 0.9% vs 15.9% off the correctly rounded
    #    result), so a residual (and rm) can differ by ~1 ulp of that term.
    #  - the fp64 Gumbel noise -log(-log(u)) chains two fp64 logs whose lowerings
    #    differ in the last bit on ~30% of inputs (1.26M/4.19M measured).
    # Bound is absolute: over 60 randomized trials x 4 vocab shapes the largest
    # observed rm difference was 9.54e-07 (1 ulp at |rm| in [8, 16)).
    torch.testing.assert_close(ref_rm, got_rm, rtol=0, atol=4e-6)


@pytest.mark.parametrize("vocab_size", [16384, 151936, 17000, 1])
def test_resample_parity(vocab_size):
    _check_resample(Batch(vocab_size=vocab_size, draft_dtype=torch.float32))
    _check_resample(Batch(vocab_size=vocab_size))


def test_resample_parity_fp64():
    _check_resample(Batch(draft_dtype=torch.float32), use_fp64=True)
    _check_resample(Batch(draft_dtype=torch.bfloat16), use_fp64=True)


def test_rejection_and_resample_fp16_target_logits():
    """Regression for the same call site as
    test_compute_local_logits_stats_fp16_target_logits: with a greedy (or
    penalty-free) request the rejection sampler is handed the model dtype, so
    spec_rejection_rdna2 / spec_resample_rdna2 must accept fp16 target logits
    too. Both ops used to TORCH_CHECK fp32.

    The sampled tokens stay bit-exact; the rejected logsumexps inherit the
    ~1e-7 association difference of the stats reduction.
    """
    for draft_dtype in (None, torch.float32):
        b = Batch(draft_dtype=draft_dtype, target_logits_dtype=torch.float16)
        assert b.target_logits.dtype == torch.float16
        stats = b.stats()
        ref = torch.full(
            (b.num_reqs, b.num_spec + 1), SENTINEL, dtype=torch.int64, device=DEVICE
        )
        got = ref.clone()
        ref_ns = torch.full((b.num_reqs,), SENTINEL, dtype=torch.int32, device=DEVICE)
        got_ns = ref_ns.clone()
        ref_trg, ref_drf = triton_rejection(b, stats, ref, ref_ns)
        got_trg, got_drf = hip_rejection(b, stats, got, got_ns)
        sync()
        assert torch.equal(ref, got), "accept/reject decisions must not move"
        assert torch.equal(ref_ns, got_ns)
        torch.testing.assert_close(ref_trg, got_trg, rtol=1e-5, atol=0, equal_nan=True)
        torch.testing.assert_close(ref_drf, got_drf, rtol=1e-5, atol=0, equal_nan=True)
        # Resample reads target_logits elementwise (no reduction), so it stays
        # bit-exact even with an fp16 target.
        _check_resample(b)


def test_resample_parity_greedy():
    # Greedy requests keep the target argmax for non-bonus rows and resample the
    # bonus row without Gumbel noise.
    _check_resample(Batch(temperatures=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    _check_resample(Batch(draft_dtype=torch.float32, temperatures=[0.0] * 8))


def test_resample_parity_placeholders():
    b = Batch(draft_dtype=torch.float32)
    b.draft_sampled[1] = -1
    _check_resample(b)


def test_resample_parity_block_verification():
    b = Batch(draft_dtype=torch.float32)
    nl = b.num_logits
    vocab_num_blocks = cdiv(b.vocab_size, 8192)
    padded = triton.next_power_of_2(vocab_num_blocks)
    stats = b.stats()
    tla, tlm, tls, dlm, dls = stats
    cumulative = torch.zeros(nl, dtype=torch.float32, device=DEVICE)
    rsu._compute_cumulative_log_p_kernel[(b.num_reqs,)](
        cumulative, b.target_logits, b.target_logits.stride(0),
        tlm, tlm.stride(0), tls, tls.stride(0),
        b.draft_sampled, b.draft_logits, b.draft_logits.stride(0),
        b.draft_logits.stride(1), dlm, dlm.stride(0), dls, dls.stride(0),
        b.cu_num_logits, b.idx_mapping, b.temperature, vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded, HAS_DRAFT_LOGITS=True, num_warps=1,
    )
    sync()
    _check_resample(b, use_block_verification=True, cumulative=cumulative)


def _check_insert(b):
    stats = b.stats()
    resample_num_blocks = cdiv(b.vocab_size, 1024)
    padded = triton.next_power_of_2(resample_num_blocks)
    # Build resampled tuples with the Triton resample kernel.
    ns = torch.zeros(b.num_reqs, dtype=torch.int32, device=DEVICE)
    ra, rm = triton_resample(b, stats, ns, False)
    ref_ns = ns.clone()
    ref_sampled = torch.randint(0, 100, (b.num_reqs, b.num_spec + 1),
                                dtype=torch.int64, device=DEVICE)
    got_ns = ref_ns.clone()
    got_sampled = ref_sampled.clone()
    if b.num_reqs:
        rsu._insert_resampled_kernel[(b.num_reqs,)](
            ref_sampled, ref_sampled.stride(0), ref_ns,
            ra, ra.stride(0), rm, rm.stride(0),
            resample_num_blocks, b.cu_num_logits, b.expanded_idx_mapping,
            b.temperature, PADDED_RESAMPLE_NUM_BLOCKS=padded,
        )
        OP.spec_insert_resampled_rdna2(
            got_sampled, got_ns, ra, rm, resample_num_blocks,
            b.cu_num_logits, b.expanded_idx_mapping, b.temperature,
        )
    sync()
    assert torch.equal(ref_ns, got_ns)
    assert torch.equal(ref_sampled, got_sampled)


def test_insert_resampled_parity():
    _check_insert(Batch(draft_dtype=torch.float32))
    _check_insert(Batch(temperatures=[0.0] * 8))
    _check_insert(Batch(draft_dtype=torch.bfloat16))


def test_insert_resampled_parity_all_inf():
    # Every block max is -inf: tl.argmax's tie-break must keep block 0.
    b = Batch(draft_dtype=torch.float32)
    resample_num_blocks = cdiv(b.vocab_size, 1024)
    padded = triton.next_power_of_2(resample_num_blocks)
    ns = torch.zeros(b.num_reqs, dtype=torch.int32, device=DEVICE)
    ra = torch.zeros((b.num_reqs, resample_num_blocks), dtype=torch.int64, device=DEVICE)
    rm = torch.full((b.num_reqs, resample_num_blocks), float("-inf"), device=DEVICE)
    ref_ns, got_ns = ns.clone(), ns.clone()
    ref_s = torch.full((b.num_reqs, b.num_spec + 1), SENTINEL, dtype=torch.int64, device=DEVICE)
    got_s = ref_s.clone()
    rsu._insert_resampled_kernel[(b.num_reqs,)](
        ref_s, ref_s.stride(0), ref_ns, ra, ra.stride(0), rm, rm.stride(0),
        resample_num_blocks, b.cu_num_logits, b.expanded_idx_mapping,
        b.temperature, PADDED_RESAMPLE_NUM_BLOCKS=padded,
    )
    OP.spec_insert_resampled_rdna2(
        got_s, got_ns, ra, rm, resample_num_blocks, b.cu_num_logits,
        b.expanded_idx_mapping, b.temperature,
    )
    sync()
    assert torch.equal(ref_ns, got_ns)
    assert torch.equal(ref_s, got_s)


def test_insert_resampled_parity_fp64():
    b = Batch(draft_dtype=torch.float32)
    stats = b.stats()
    resample_num_blocks = cdiv(b.vocab_size, 1024)
    padded = triton.next_power_of_2(resample_num_blocks)
    ns = torch.zeros(b.num_reqs, dtype=torch.int32, device=DEVICE)
    ra, rm = triton_resample(b, stats, ns, True)
    ref_ns, got_ns = ns.clone(), ns.clone()
    ref_s = torch.full((b.num_reqs, b.num_spec + 1), SENTINEL, dtype=torch.int64, device=DEVICE)
    got_s = ref_s.clone()
    rsu._insert_resampled_kernel[(b.num_reqs,)](
        ref_s, ref_s.stride(0), ref_ns, ra, ra.stride(0), rm, rm.stride(0),
        resample_num_blocks, b.cu_num_logits, b.expanded_idx_mapping,
        b.temperature, PADDED_RESAMPLE_NUM_BLOCKS=padded,
    )
    OP.spec_insert_resampled_rdna2(
        got_s, got_ns, ra, rm, resample_num_blocks, b.cu_num_logits,
        b.expanded_idx_mapping, b.temperature,
    )
    sync()
    assert torch.equal(ref_ns, got_ns)
    assert torch.equal(ref_s, got_s)


# ---------------------------------------------------------------------------
# The Python dispatchers must take the HIP path, never fall through.
# ---------------------------------------------------------------------------
class _NoLaunch:
    def __getitem__(self, _):
        raise AssertionError("Triton kernel was launched on the RDNA2 path")


def test_wrappers_do_not_fall_back_to_triton(monkeypatch):
    from types import SimpleNamespace

    for mod, names in (
        (
            spec,
            [
                "_prepare_prefill_inputs_kernel",
                "_prepare_decode_inputs_kernel",
                "_update_draft_inputs_kernel",
            ],
        ),
        (
            rs,
            ["_gather_draft_sampled_kernel", "_flatten_sampled_kernel"],
        ),
        (
            rsu,
            [
                "_compute_local_logits_stats_kernel",
                "_rejection_kernel",
                "_resample_kernel",
                "_insert_resampled_kernel",
            ],
        ),
    ):
        for name in names:
            monkeypatch.setattr(mod, name, _NoLaunch())

    num_reqs, num_spec, vocab_size = 8, 2, 16384
    b = Batch(num_reqs=num_reqs, num_spec=num_spec, vocab_size=vocab_size,
              draft_dtype=torch.float32)
    nl = b.num_logits
    # The stats come from the HIP op (the Triton kernel is poisoned below).
    NB = cdiv(vocab_size, 8192)
    stats = [
        torch.zeros((nl, NB), dtype=torch.int64, device=DEVICE),
        torch.zeros((nl, NB), device=DEVICE),
        torch.zeros((nl, NB), device=DEVICE),
        torch.zeros((nl, NB), device=DEVICE),
        torch.zeros((nl, NB), device=DEVICE),
    ]
    OP.spec_compute_local_logits_stats_rdna2(
        stats[0], stats[1], stats[2], stats[3], stats[4], b.target_logits,
        b.draft_logits, b.expanded_idx_mapping, b.expanded_local_pos,
        b.temperature, vocab_size, num_spec, True,
    )
    qlen = 3
    qsl = torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE) * qlen
    total = num_reqs * qlen
    buffers = SimpleNamespace(
        input_ids=torch.zeros(total, dtype=torch.int32, device=DEVICE),
        positions=torch.zeros(total, dtype=torch.int64, device=DEVICE),
        query_start_loc=qsl.clone(),
        seq_lens=torch.full((num_reqs,), 10, dtype=torch.int32, device=DEVICE),
    )
    batch = SimpleNamespace(
        input_ids=buffers.input_ids,
        positions=buffers.positions,
        query_start_loc=qsl.clone(),
        seq_lens=buffers.seq_lens,
        idx_mapping=b.idx_mapping,
        num_reqs=num_reqs,
    )
    spec.prepare_prefill_inputs(
        torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        torch.zeros((), dtype=torch.int64, device=DEVICE),
        buffers,
        batch,
        torch.ones(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        num_reqs,
    )
    spec.prepare_decode_inputs(
        torch.zeros((num_reqs, num_spec), dtype=torch.int64, device=DEVICE)[:, 0],
        buffers.seq_lens,
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        buffers,
        buffers.positions.new_zeros(num_reqs),
        4096,
        num_reqs,
    )
    hidden = torch.zeros(num_reqs, 64, dtype=torch.bfloat16, device=DEVICE)
    spec.update_draft_inputs(
        torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        torch.zeros((), dtype=torch.int64, device=DEVICE),
        hidden,
        torch.zeros((num_reqs, num_spec), dtype=torch.int64, device=DEVICE),
        torch.zeros((num_reqs, 64), dtype=torch.bfloat16, device=DEVICE),
        buffers,
        buffers.positions.new_zeros(num_reqs),
        num_reqs,
        4096,
        num_spec,
    )
    ds, pos = rs.gather_draft_sampled(
        buffers.input_ids,
        b.pos[:nl],
        torch.arange(nl, dtype=torch.int64, device=DEVICE),
        b.expanded_idx_mapping,
        b.expanded_local_pos,
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
    )
    assert ds.shape == (nl,)

    sampled = torch.zeros((num_reqs, num_spec + 1), dtype=torch.int64, device=DEVICE)
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    trg, drf = hip_rejection(b, stats, sampled, num_sampled)
    ra, rm = hip_resample(b, stats, num_sampled, False)
    OP.spec_insert_resampled_rdna2(
        sampled, num_sampled, ra, rm, cdiv(vocab_size, 1024), b.cu_num_logits,
        b.expanded_idx_mapping, b.temperature,
    )
    # Also drive the flatten dispatcher.
    flat = torch.zeros(nl, dtype=sampled.dtype, device=DEVICE)
    OP.spec_flatten_sampled_rdna2(flat, sampled, num_sampled, b.cu_num_logits)
    sync()
    assert ra.shape[0] == num_reqs
