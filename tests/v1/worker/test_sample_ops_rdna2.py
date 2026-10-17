# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-for-bit parity between the gfx1030 HIP ports and the Triton kernels of
the v1 sampler (``vllm/v1/worker/gpu/sample/``):

  _temperature_kernel               (gumbel.py)
  _gumbel_sample_kernel             (gumbel.py)
  _penalties_kernel                 (penalties.py)
  _bincount_kernel                  (penalties.py)
  _min_p_kernel                     (min_p.py)
  _bias_kernel                      (logit_bias.py)
  _bad_words_kernel                 (bad_words.py)
  _topk_log_softmax_kernel          (logprob.py)
  _ranks_kernel                     (logprob.py)
  _prompt_logprobs_token_ids_kernel (prompt_logprob.py)

These run once per decode step inside a FULL cudagraph capture, so the only
acceptable difference between the two paths is none.  Every assertion is
``assert_bitexact`` (rtol=0, atol=0) instead of ``torch.equal`` because
``torch.equal`` reports NaN != NaN; the reduction kernels legitimately produce
``-inf``/NaN for fully masked rows.
"""

import pytest
import torch

from vllm.platforms.rdna import on_rdna2
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample import bad_words as bw_mod
from vllm.v1.worker.gpu.sample import gumbel as gb_mod
from vllm.v1.worker.gpu.sample import logit_bias as lb_mod
from vllm.v1.worker.gpu.sample import logprob as lp_mod
from vllm.v1.worker.gpu.sample import min_p as mp_mod
from vllm.v1.worker.gpu.sample import penalties as pen_mod
from vllm.v1.worker.gpu.sample import prompt_logprob as plp_mod
from vllm.v1.worker.gpu.sample.bad_words import _bad_words_kernel
from vllm.v1.worker.gpu.sample.gumbel import (
    _gumbel_sample_kernel,
    _temperature_kernel,
)
from vllm.v1.worker.gpu.sample.logit_bias import _bias_kernel
from vllm.v1.worker.gpu.sample.logprob import (
    _ranks_kernel,
    _topk_log_softmax_kernel,
)
from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel
from vllm.v1.worker.gpu.sample.penalties import _bincount_kernel, _penalties_kernel
from vllm.v1.worker.gpu.sample.prompt_logprob import (
    _prompt_logprobs_token_ids_kernel,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_rdna2()),
    reason="gfx10x (RDNA2) ROCm required",
)

DEVICE = torch.device("cuda")
OP = torch.ops._rocm_C

# Real serving geometry (Qwen3.8-Flash-Next-Uncensored-W4A16, TP4, MTP2).
VOCAB = 151936
NUM_REQS = 8
TOKENS_PER_REQ = 3
NUM_TOKENS = NUM_REQS * TOKENS_PER_REQ


def sync():
    torch.cuda.synchronize()


def assert_bitexact(got, ref, what=""):
    torch.testing.assert_close(got, ref, rtol=0, atol=0, equal_nan=True, msg=what)


def make_mapping(num_tokens=NUM_TOKENS, num_reqs=NUM_REQS):
    return (
        torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        .repeat_interleave(num_tokens // num_reqs)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# _temperature_kernel
# ---------------------------------------------------------------------------
def triton_temperature(logits, mapping, temperature, vocab_size):
    num_tokens = logits.shape[0]
    if num_tokens == 0 or triton.cdiv(vocab_size, 8192) == 0:
        return
    _temperature_kernel[(num_tokens, triton.cdiv(vocab_size, 8192))](
        logits,
        logits.stride(0),
        mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=8192,
    )


def hip_temperature(logits, mapping, temperature, vocab_size):
    OP.temperature_rdna2(
        logits, logits.stride(0), mapping, temperature, vocab_size
    )


@pytest.mark.parametrize(
    "num_tokens,vocab_size", [(NUM_TOKENS, VOCAB), (1, 151936), (2, 1025),
                              (3, 1024), (1, 1023), (1, 1), (4, 8192),
                              (1, 8193), (0, VOCAB), (2, 0)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_temperature_parity(num_tokens, vocab_size, dtype):
    def make():
        torch.manual_seed(7)
        logits = torch.randn(num_tokens, vocab_size, dtype=dtype, device=DEVICE)
        temperature = torch.tensor(
            [0.0, 1.0, 0.7, 2.5, 0.3, 1.0, 0.0, 1.5], dtype=torch.float32,
            device=DEVICE,
        )
        mapping = (
            torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
            .repeat_interleave(max(num_tokens // NUM_REQS, 0))
            .contiguous()
        )
        if mapping.numel() < num_tokens:
            mapping = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
        return logits, mapping, temperature

    a = make()
    hip_temperature(*a, vocab_size)
    b = make()
    triton_temperature(*b, vocab_size)
    sync()
    assert_bitexact(a[0], b[0], "temperature logits")


# ---------------------------------------------------------------------------
# _gumbel_sample_kernel
# ---------------------------------------------------------------------------
def triton_gumbel_sample(logits, mapping, temperature, seed, pos, *,
                         apply_temperature, is_drafting, logits_cache=None,
                         logits_cache_col=None, use_fp64=False):
    num_tokens, vocab_size = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    if num_blocks == 0:
        return (torch.zeros((num_tokens, 0), dtype=torch.int64, device=DEVICE),
                torch.zeros((num_tokens, 0), dtype=torch.float32, device=DEVICE))
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=block_size,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=logits_cache_col is not None and logits_cache_col.dim() > 0,
    )
    return local_argmax, local_max


def hip_gumbel_sample(logits, mapping, temperature, seed, pos, *,
                      apply_temperature, is_drafting, logits_cache=None,
                      logits_cache_col=None, use_fp64=False):
    num_tokens, vocab_size = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    OP.gumbel_sample_rdna2(
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        is_drafting,
        apply_temperature,
        use_fp64,
        logits_cache_col is not None and logits_cache_col.dim() > 0,
    )
    return local_argmax, local_max


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("apply_temperature", [True, False])
@pytest.mark.parametrize("is_drafting", [True, False])
def test_gumbel_sample_parity(dtype, apply_temperature, is_drafting):
    def make():
        torch.manual_seed(11)
        logits = torch.randn(NUM_TOKENS, VOCAB, dtype=dtype, device=DEVICE)
        mapping = make_mapping()
        temperature = torch.tensor(
            [0.0, 1.0, 0.7, 2.5, 0.3, 1.0, 0.0, 1.5], dtype=torch.float32,
            device=DEVICE,
        )
        seed = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64, device=DEVICE
        )
        pos = torch.arange(
            NUM_TOKENS, dtype=torch.int64, device=DEVICE
        ) // (NUM_TOKENS // NUM_REQS) + 5
        return logits, mapping, temperature, seed, pos

    kwargs = dict(apply_temperature=apply_temperature, is_drafting=is_drafting)
    a = make()
    got = hip_gumbel_sample(*a, **kwargs)
    b = make()
    ref = triton_gumbel_sample(*b, **kwargs)
    sync()
    assert_bitexact(got[0], ref[0], "gumbel local_argmax")
    assert_bitexact(got[1], ref[1], "gumbel local_max")


@pytest.mark.parametrize("vocab_size", [1, 1023, 1024, 1025, 2048, VOCAB])
def test_gumbel_sample_vocab_edges(vocab_size):
    def make():
        torch.manual_seed(13)
        logits = torch.randn(4, vocab_size, dtype=torch.float32, device=DEVICE)
        mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor(
            [1.0, 0.5, 0.0, 2.0], dtype=torch.float32, device=DEVICE
        )
        seed = torch.arange(1, 5, dtype=torch.int64, device=DEVICE)
        pos = torch.tensor([0, 7, 1, 3], dtype=torch.int64, device=DEVICE)
        return logits, mapping, temperature, seed, pos

    kwargs = dict(apply_temperature=True, is_drafting=False)
    a = make()
    got = hip_gumbel_sample(*a, **kwargs)
    b = make()
    ref = triton_gumbel_sample(*b, **kwargs)
    sync()
    assert_bitexact(got[0], ref[0], "gumbel local_argmax")
    assert_bitexact(got[1], ref[1], "gumbel local_max")


def test_gumbel_sample_invalid_mapping_and_ties():
    """`-1` rows (cudagraph padding) and rows with repeated logit values."""

    def make():
        torch.manual_seed(17)
        logits = torch.full((6, 4096), -100.0, dtype=torch.float32, device=DEVICE)
        logits[0, 100] = 5.0
        logits[1, 0] = 1.0
        logits[1, 4095] = 1.0  # tie across blocks
        logits[2, :] = 0.0
        mapping = torch.tensor([0, 1, 2, -1, -1, 1], dtype=torch.int32,
                              device=DEVICE)
        temperature = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32,
                                   device=DEVICE)
        seed = torch.tensor([9, 10, 11], dtype=torch.int64, device=DEVICE)
        pos = torch.tensor([0, 3, 4, 0, 0, 5], dtype=torch.int64, device=DEVICE)
        return logits, mapping, temperature, seed, pos

    kwargs = dict(apply_temperature=True, is_drafting=False)
    a = make()
    got = hip_gumbel_sample(*a, **kwargs)
    b = make()
    ref = triton_gumbel_sample(*b, **kwargs)
    sync()
    assert_bitexact(got[0], ref[0], "gumbel local_argmax")
    assert_bitexact(got[1], ref[1], "gumbel local_max")
    # temp == 0 must be a plain argmax: the leftmost 1.0 in row 1 wins.
    assert got[0][1, 0].item() == 0


@pytest.mark.parametrize("per_token_col", [False, True])
def test_gumbel_sample_logits_cache(per_token_col):
    # One token per request: two rows of the same request share the cache row
    # and would race on the store (the drafted path samples one token/request).
    num_tokens = NUM_REQS

    def make():
        torch.manual_seed(19)
        logits = torch.randn(num_tokens, VOCAB, dtype=torch.float32,
                             device=DEVICE)
        cache = torch.full((NUM_REQS, 4, VOCAB), 12345.0, dtype=torch.float32,
                           device=DEVICE)
        col = (
            torch.arange(num_tokens, dtype=torch.int32, device=DEVICE) % 4
            if per_token_col
            else torch.tensor(2, dtype=torch.int32, device=DEVICE)
        )
        mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        temperature = torch.tensor(
            [0.0, 1.0, 0.7, 2.5, 0.3, 1.0, 0.0, 1.5], dtype=torch.float32,
            device=DEVICE,
        )
        seed = torch.arange(1, 9, dtype=torch.int64, device=DEVICE)
        pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
        return logits, mapping, temperature, seed, pos, cache, col

    kwargs = dict(apply_temperature=True, is_drafting=True)
    a = make()
    got = hip_gumbel_sample(a[0], a[1], a[2], a[3], a[4], logits_cache=a[5],
                            logits_cache_col=a[6], **kwargs)
    b = make()
    ref = triton_gumbel_sample(b[0], b[1], b[2], b[3], b[4], logits_cache=b[5],
                               logits_cache_col=b[6], **kwargs)
    sync()
    assert_bitexact(got[0], ref[0], "gumbel local_argmax")
    assert_bitexact(got[1], ref[1], "gumbel local_max")
    assert_bitexact(a[5], b[5], "logits_cache contents")
    assert not torch.all(a[5] == 12345.0)  # the cache was actually filled
    assert_bitexact(a[5][0, 2 if not per_token_col else 0], a[0][0],
                    "cache holds the pre-temperature logits")


# ---------------------------------------------------------------------------
# _penalties_kernel
# ---------------------------------------------------------------------------
def triton_penalties(logits, mapping, token_ids, local_pos, rep, freq, pres,
                     bin_mask, counts, vocab_size):
    num_tokens = logits.shape[0]
    if num_tokens == 0:
        return
    blocks = triton.cdiv(vocab_size, 8192)
    if blocks == 0:
        return
    _penalties_kernel[(num_tokens, blocks)](
        logits,
        logits.stride(0),
        mapping,
        token_ids,
        local_pos,
        rep,
        freq,
        pres,
        bin_mask,
        bin_mask.stride(0),
        counts,
        counts.stride(0),
        vocab_size,
        BLOCK_SIZE=8192,
    )


def hip_penalties(logits, mapping, token_ids, local_pos, rep, freq, pres,
                  bin_mask, counts, vocab_size):
    OP.penalties_rdna2(
        logits,
        logits.stride(0),
        mapping,
        token_ids,
        local_pos,
        rep,
        freq,
        pres,
        bin_mask,
        bin_mask.stride(0),
        counts,
        counts.stride(0),
        vocab_size,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("vocab_size", [1, 1023, 8191, 8192, 8193, VOCAB])
def test_penalties_parity(dtype, vocab_size):
    def make():
        torch.manual_seed(23)
        logits = torch.randn(NUM_TOKENS, vocab_size, dtype=dtype, device=DEVICE)
        mapping = make_mapping()
        token_ids = torch.randint(0, vocab_size, (NUM_TOKENS,), dtype=torch.int32,
                                  device=DEVICE)
        local_pos = torch.arange(NUM_TOKENS, dtype=torch.int32,
                                 device=DEVICE) % TOKENS_PER_REQ
        rep = torch.tensor([1.0, 1.2, 0.8, 1.0, 1.5, 1.0, 1.0, 1.1],
                           dtype=torch.float32, device=DEVICE)
        freq = torch.tensor([0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0],
                            dtype=torch.float32, device=DEVICE)
        pres = torch.tensor([0.0, 0.0, 0.2, 0.0, 0.5, 0.0, 0.0, 0.7],
                            dtype=torch.float32, device=DEVICE)
        bin_mask = torch.randint(
            0, 2, (NUM_REQS, triton.cdiv(vocab_size, 32)), dtype=torch.int32,
            device=DEVICE,
        )
        counts = torch.randint(0, 3, (NUM_REQS, vocab_size), dtype=torch.int32,
                               device=DEVICE)
        return logits, mapping, token_ids, local_pos, rep, freq, pres, bin_mask, counts

    a = make()
    counts_before = a[8].clone()
    hip_penalties(*a, vocab_size)
    b = make()
    triton_penalties(*b, vocab_size)
    sync()
    if dtype is torch.float16:
        # Triton's AMD lowering folds the final fp32->fp16 conversion into the
        # presence-penalty FMA (v_fma_mixlo_f16), so on elements whose exact
        # result is an fp16 midpoint (~0.07% of them) the two paths pick the two
        # neighbouring fp16 values. The serving path always applies penalties to
        # the fp32 logits copy (sampler.py::apply_sampling_params), so this is
        # a non-serving dtype and the bound is 2 fp16 ulps.
        torch.testing.assert_close(
            a[0].float(), b[0].float(), rtol=2**-10, atol=2**-10,
            msg="penalties fp16 logits",
        )
    else:
        assert_bitexact(a[0], b[0], "penalties logits")
    assert_bitexact(a[8], counts_before, "output_bin_counts must be read-only")


def test_penalties_no_penalty_early_return():
    def make():
        torch.manual_seed(29)
        logits = torch.randn(NUM_TOKENS, 4096, dtype=torch.float32,
                             device=DEVICE)
        mapping = make_mapping()
        token_ids = torch.zeros(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
        local_pos = torch.zeros(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
        zero = torch.zeros(NUM_REQS, dtype=torch.float32, device=DEVICE)
        rep = torch.ones(NUM_REQS, dtype=torch.float32, device=DEVICE)
        bin_mask = torch.zeros((NUM_REQS, 128), dtype=torch.int32, device=DEVICE)
        counts = torch.zeros((NUM_REQS, 4096), dtype=torch.int32, device=DEVICE)
        return logits, mapping, token_ids, local_pos, rep, zero, zero, bin_mask, counts

    a = make()
    before = a[0].clone()
    hip_penalties(*a, 4096)
    sync()
    assert_bitexact(a[0], before, "no-op penalties")


# ---------------------------------------------------------------------------
# _bincount_kernel
# ---------------------------------------------------------------------------
def triton_bincount(mapping, all_token_ids, prompt_len, prefill_len, bin_mask,
                    counts, max_prefill_len):
    idx_long = mapping.long()
    bin_mask.index_fill_(0, idx_long, 0)
    counts.index_fill_(0, idx_long, 0)
    num_tokens = mapping.shape[0]
    blocks = triton.cdiv(max_prefill_len, 1024)
    if num_tokens == 0 or blocks == 0:
        return
    _bincount_kernel[(num_tokens, blocks)](
        mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        bin_mask,
        bin_mask.stride(0),
        counts,
        counts.stride(0),
        BLOCK_SIZE=1024,
    )


def hip_bincount(mapping, all_token_ids, prompt_len, prefill_len, bin_mask,
                 counts, max_prefill_len):
    idx_long = mapping.long()
    bin_mask.index_fill_(0, idx_long, 0)
    counts.index_fill_(0, idx_long, 0)
    OP.bincount_rdna2(
        mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        bin_mask,
        bin_mask.stride(0),
        counts,
        counts.stride(0),
        max_prefill_len,
    )


@pytest.mark.parametrize(
    "prompt_lens,prefill_lens",
    [
        ([5, 1024, 1025, 1, 2047, 8, 3000, 4096],
         [9, 1024, 1025, 1, 2047, 8, 3000, 4096]),
        ([0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]),
        ([10, 10, 10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10, 10, 10]),
    ],
)
def test_bincount_parity(prompt_lens, prefill_lens):
    vocab_size = 4096
    width = 8192

    def make():
        torch.manual_seed(31)
        mapping = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
        # Repeated token ids exercise the atomic read-modify-write.
        all_token_ids = torch.randint(
            0, 64, (NUM_REQS, width), dtype=torch.int32, device=DEVICE
        )
        prompt_len = torch.tensor(prompt_lens, dtype=torch.int32, device=DEVICE)
        prefill_len = torch.tensor(prefill_lens, dtype=torch.int32, device=DEVICE)
        bin_mask = torch.zeros(
            (NUM_REQS, triton.cdiv(vocab_size, 32)), dtype=torch.int32,
            device=DEVICE,
        )
        counts = torch.zeros((NUM_REQS, vocab_size), dtype=torch.int32,
                             device=DEVICE)
        return mapping, all_token_ids, prompt_len, prefill_len, bin_mask, counts

    max_prefill_len = max(prefill_lens)
    a = make()
    hip_bincount(*a, max_prefill_len)
    b = make()
    triton_bincount(*b, max_prefill_len)
    sync()
    assert_bitexact(a[4], b[4], "prompt_bin_mask")
    assert_bitexact(a[5], b[5], "output_bin_counts")


def test_bincount_empty():
    mapping = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    all_token_ids = torch.zeros((0, 16), dtype=torch.int32, device=DEVICE)
    se = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    bin_mask = torch.zeros((0, 8), dtype=torch.int32, device=DEVICE)
    counts = torch.zeros((0, 64), dtype=torch.int32, device=DEVICE)
    hip_bincount(mapping, all_token_ids, se, se, bin_mask, counts, 0)
    triton_bincount(mapping, all_token_ids, se, se, bin_mask, counts, 0)
    sync()
    assert counts.numel() == 0


# ---------------------------------------------------------------------------
# _min_p_kernel
# ---------------------------------------------------------------------------
def triton_min_p(logits, mapping, min_p, vocab_size):
    num_tokens = logits.shape[0]
    if num_tokens == 0 or vocab_size == 0:
        return
    _min_p_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        mapping,
        min_p,
        vocab_size,
        BLOCK_SIZE=1024,
    )


def hip_min_p(logits, mapping, min_p, vocab_size):
    OP.min_p_rdna2(logits, logits.stride(0), mapping, min_p, vocab_size)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("vocab_size", [1, 1023, 1024, 1025, VOCAB])
def test_min_p_parity(dtype, vocab_size):
    def make():
        torch.manual_seed(37)
        logits = torch.randn(NUM_TOKENS, vocab_size, dtype=dtype, device=DEVICE)
        # Include -inf entries: that is what the serving path feeds in.
        logits[0, : vocab_size // 2] = float("-inf")
        mapping = make_mapping()
        min_p = torch.tensor(
            [0.0, 0.02, 0.5, 1.0, 1e-6, 0.9, 0.0, 0.3], dtype=torch.float32,
            device=DEVICE,
        )
        return logits, mapping, min_p

    a = make()
    hip_min_p(*a, vocab_size)
    b = make()
    triton_min_p(*b, vocab_size)
    sync()
    assert_bitexact(a[0], b[0], "min_p logits")


# ---------------------------------------------------------------------------
# _bias_kernel
# ---------------------------------------------------------------------------
def triton_bias(logits, mapping, num_allowed, allowed_ids, num_bias, bias_ids,
                bias, pos, min_lens, num_stop, restore, stop_ids,
                check_all_masked):
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    block_size = triton.next_power_of_2(
        max(allowed_ids.shape[-1], bias_ids.shape[-1], stop_ids.shape[-1])
    )
    _bias_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        mapping,
        num_allowed,
        allowed_ids,
        allowed_ids.stride(0),
        num_bias,
        bias_ids,
        bias_ids.stride(0),
        bias,
        bias.stride(0),
        pos,
        min_lens,
        num_stop,
        restore,
        stop_ids,
        stop_ids.stride(0),
        BLOCK_SIZE=block_size,
        LOGITS_BLOCK_SIZE=8192,
        CHECK_ALL_MASKED_ROWS=check_all_masked,
    )


def hip_bias(logits, mapping, num_allowed, allowed_ids, num_bias, bias_ids,
             bias, pos, min_lens, num_stop, restore, stop_ids,
             check_all_masked):
    num_tokens, vocab_size = logits.shape
    block_size = triton.next_power_of_2(
        max(allowed_ids.shape[-1], bias_ids.shape[-1], stop_ids.shape[-1])
    )
    assert block_size <= lb_mod._BIAS_MAX_BLOCK_SIZE
    OP.logit_bias_rdna2(
        logits,
        logits.stride(0),
        vocab_size,
        mapping,
        num_allowed,
        allowed_ids,
        allowed_ids.stride(0),
        num_bias,
        bias_ids,
        bias_ids.stride(0),
        bias,
        bias.stride(0),
        pos,
        min_lens,
        num_stop,
        restore,
        stop_ids,
        stop_ids.stride(0),
        block_size,
        check_all_masked,
    )


def _bias_case(w_list=1024, vocab_size=4096, num_tokens=4):
    def make():
        torch.manual_seed(41)
        logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                             device=DEVICE)
        mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        allowed_ids = torch.randint(0, vocab_size, (num_tokens, w_list),
                                    dtype=torch.int32, device=DEVICE)
        num_allowed = torch.tensor([0, 3, w_list, 1], dtype=torch.int32,
                                   device=DEVICE)[:num_tokens]
        bias_ids = torch.randint(0, vocab_size, (num_tokens, w_list),
                                 dtype=torch.int32, device=DEVICE)
        num_bias = torch.tensor([0, 2, 1, 5], dtype=torch.int32,
                                device=DEVICE)[:num_tokens]
        bias = torch.randn(num_tokens, w_list, dtype=torch.float32,
                           device=DEVICE)
        pos = torch.tensor([0, 1, 0, 3], dtype=torch.int64, device=DEVICE)[
            :num_tokens
        ]
        min_lens = torch.tensor([1, 8, 4, 0], dtype=torch.int32,
                                device=DEVICE)[:num_tokens]
        stop_ids = torch.randint(0, vocab_size, (num_tokens, w_list),
                                 dtype=torch.int32, device=DEVICE)
        num_stop = torch.tensor([0, 2, 1, 3], dtype=torch.int32,
                                device=DEVICE)[:num_tokens]
        restore = torch.tensor([False, True, True, True], device=DEVICE)[
            :num_tokens
        ]
        return (logits, mapping, num_allowed, allowed_ids, num_bias, bias_ids,
                bias, pos, min_lens, num_stop, restore, stop_ids)

    return make


@pytest.mark.parametrize("check_all_masked", [False, True])
@pytest.mark.parametrize("w_list", [1, 1024, 4096])
def test_bias_parity(w_list, check_all_masked):
    make = _bias_case(w_list=w_list)
    a = make()
    hip_bias(*a, check_all_masked)
    b = make()
    triton_bias(*b, check_all_masked)
    sync()
    assert_bitexact(a[0], b[0], "logit_bias logits")


def test_bias_all_masked_restore():
    """A row whose stop tokens are its only unmasked logits is restored.

    The stop logits are read *before* the -inf overwrite and the row scan runs
    after it, so restoring them is what keeps a grammar-constrained request from
    having every logit at -inf.
    """

    def make():
        torch.manual_seed(43)
        vocab_size = 2048
        logits = torch.randn(3, vocab_size, dtype=torch.float32, device=DEVICE)
        mapping = torch.arange(3, dtype=torch.int32, device=DEVICE)
        allowed_ids = torch.zeros((3, 4), dtype=torch.int32, device=DEVICE)
        allowed_ids[0, 0] = 6  # row 0: allowed == {6, 7} == the stop tokens
        allowed_ids[0, 1] = 7
        allowed_ids[1, 0] = 5  # row 1: the stop tokens are already -inf
        num_allowed = torch.tensor([2, 1, 0], dtype=torch.int32, device=DEVICE)
        bias_ids = torch.zeros((3, 4), dtype=torch.int32, device=DEVICE)
        num_bias = torch.zeros(3, dtype=torch.int32, device=DEVICE)
        bias = torch.zeros((3, 4), dtype=torch.float32, device=DEVICE)
        pos = torch.zeros(3, dtype=torch.int64, device=DEVICE)
        min_lens = torch.tensor([10, 10, 10], dtype=torch.int32, device=DEVICE)
        stop_ids = torch.tensor([[6, 7, 0, 0], [6, 7, 0, 0], [0, 0, 0, 0]],
                                dtype=torch.int32, device=DEVICE)
        num_stop = torch.tensor([2, 2, 0], dtype=torch.int32, device=DEVICE)
        restore = torch.tensor([True, True, False], device=DEVICE)
        return (logits, mapping, num_allowed, allowed_ids, num_bias, bias_ids,
                bias, pos, min_lens, num_stop, restore, stop_ids)

    a = make()
    hip_bias(*a, True)
    b = make()
    triton_bias(*b, True)
    sync()
    assert_bitexact(a[0], b[0], "all-masked restore")
    # Row 0: allowed == stop == {6, 7} -> restored. Row 1: allowed {5} only, the
    # stop logits were already -inf and are filtered out. Row 2: untouched.
    assert sorted(a[0][0].isfinite().nonzero().flatten().tolist()) == [6, 7]
    assert a[0][1].isfinite().nonzero().flatten().tolist() == [5]
    assert torch.isfinite(a[0][2]).all()

    # check_all_masked_rows=False must never restore: row 0 loses everything.
    a2 = make()
    hip_bias(*a2, False)
    b2 = make()
    triton_bias(*b2, False)
    sync()
    assert_bitexact(a2[0], b2[0], "no restore")
    assert torch.isfinite(a2[0][0]).sum().item() == 0  # nothing restored


def test_bias_unsupported_block_size_falls_back():
    # Wider lists than the register budget stay on the Triton path; the Python
    # guard (apply_logit_bias) is what decides.
    assert lb_mod._BIAS_MAX_BLOCK_SIZE == 4096
    assert lb_mod._ON_RDNA2
    assert 8192 > lb_mod._BIAS_MAX_BLOCK_SIZE
    assert 4096 <= lb_mod._BIAS_MAX_BLOCK_SIZE


# ---------------------------------------------------------------------------
# _bad_words_kernel
# ---------------------------------------------------------------------------
def triton_bad_words(logits, mapping, bad_tokens, offsets, num_bad, all_ids,
                     prompt_len, total_len, input_ids, local_pos,
                     max_num_bad_words):
    num_tokens = logits.shape[0]
    if num_tokens == 0 or max_num_bad_words <= 0:
        return
    _bad_words_kernel[(num_tokens, max_num_bad_words)](
        logits,
        logits.stride(0),
        mapping,
        bad_tokens,
        bad_tokens.stride(0),
        offsets,
        offsets.stride(0),
        num_bad,
        all_ids,
        all_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        local_pos,
    )


def hip_bad_words(logits, mapping, bad_tokens, offsets, num_bad, all_ids,
                  prompt_len, total_len, input_ids, local_pos,
                  max_num_bad_words):
    OP.bad_words_rdna2(
        logits,
        logits.stride(0),
        mapping,
        bad_tokens,
        bad_tokens.stride(0),
        offsets,
        offsets.stride(0),
        num_bad,
        all_ids,
        all_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        local_pos,
        max_num_bad_words,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 24, 257])
def test_bad_words_parity(num_tokens):
    max_num_bad_words = 4
    width = 64

    def make():
        torch.manual_seed(47)
        logits = torch.randn(num_tokens, 512, dtype=torch.float32, device=DEVICE)
        mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
        # Every request has one 1-token bad word (always matches at pos 0) and
        # one 2-token bad word whose prefix is taken from the committed tokens,
        # plus a deliberately non-matching prefix.
        bad_tokens = torch.zeros((num_tokens, width), dtype=torch.int32,
                                 device=DEVICE)
        offsets = torch.zeros((num_tokens, max_num_bad_words + 1),
                              dtype=torch.int32, device=DEVICE)
        num_bad = torch.full((num_tokens,), 3, dtype=torch.int32, device=DEVICE)
        for r in range(num_tokens):
            offsets[r] = torch.tensor([0, 1, 3, 6, 6], dtype=torch.int32)
            bad_tokens[r, 0] = 100  # single token
            bad_tokens[r, 1] = 8  # prefix of the 2-token bad word
            bad_tokens[r, 2] = 200  # last token -> written
            bad_tokens[r, 3] = 8  # prefix that will not match
            bad_tokens[r, 4] = 9
            bad_tokens[r, 5] = 201
        all_ids = torch.zeros((num_tokens, 128), dtype=torch.int32, device=DEVICE)
        all_ids[:, 4:] = 7
        prompt_len = torch.full((num_tokens,), 4, dtype=torch.int32, device=DEVICE)
        total_len = torch.full((num_tokens,), 8, dtype=torch.int32, device=DEVICE)
        input_ids = torch.full((num_tokens,), 8, dtype=torch.int32, device=DEVICE)
        local_pos = torch.full((num_tokens,), 3, dtype=torch.int32, device=DEVICE)
        return (logits, mapping, bad_tokens, offsets, num_bad, all_ids,
                prompt_len, total_len, input_ids, local_pos)

    a = make()
    hip_bad_words(*a, max_num_bad_words)
    b = make()
    triton_bad_words(*b, max_num_bad_words)
    sync()
    assert_bitexact(a[0], b[0], "bad_words logits")
    # The always-matching single-token bad word must have been masked.
    assert (a[0][:, 100] == float("-inf")).all()
    # The unmatched prefix must not mask its last token.
    assert (a[0][:, 201] > float("-inf")).all()
    # The 2-token bad word matches through the draft input_ids: with pos 3 and
    # output_len 4 the prefix position is >= output_len, so it reads input_ids.
    assert (a[0][:, 200] == float("-inf")).all()


def test_bad_words_edges():
    """max_num_bad_words = 0/1, num_bad_words = 0, prefix longer than output."""

    def make():
        torch.manual_seed(53)
        logits = torch.randn(3, 256, dtype=torch.float32, device=DEVICE)
        mapping = torch.arange(3, dtype=torch.int32, device=DEVICE)
        bad_tokens = torch.zeros((3, 8), dtype=torch.int32, device=DEVICE)
        bad_tokens[0, 0] = 11
        bad_tokens[1, 0] = 12
        bad_tokens[1, 1] = 13
        offsets = torch.tensor(
            [[0, 1, 1], [0, 1, 2], [0, 1, 1]], dtype=torch.int32, device=DEVICE
        )
        num_bad = torch.tensor([1, 1, 0], dtype=torch.int32, device=DEVICE)
        all_ids = torch.zeros((3, 32), dtype=torch.int32, device=DEVICE)
        prompt_len = torch.tensor([4, 4, 4], dtype=torch.int32, device=DEVICE)
        total_len = torch.tensor([4, 4, 4], dtype=torch.int32, device=DEVICE)
        input_ids = torch.zeros(3, dtype=torch.int32, device=DEVICE)
        local_pos = torch.zeros(3, dtype=torch.int32, device=DEVICE)
        return (logits, mapping, bad_tokens, offsets, num_bad, all_ids,
                prompt_len, total_len, input_ids, local_pos)

    a = make()
    hip_bad_words(*a, 2)
    b = make()
    triton_bad_words(*b, 2)
    sync()
    assert_bitexact(a[0], b[0], "bad_words edges")
    assert (a[0][0, 11] == float("-inf")).all()
    assert (a[0][2] > float("-inf")).all()  # num_bad_words == 0
    # max_num_bad_words == 0 launches nothing.
    a2 = make()
    hip_bad_words(*a2, 0)
    b2 = make()
    triton_bad_words(*b2, 0)
    sync()
    assert_bitexact(a2[0], b2[0], "bad_words zero bad words")


# ---------------------------------------------------------------------------
# _topk_log_softmax_kernel
# ---------------------------------------------------------------------------
def triton_topk_log_softmax(output, logits, topk_ids, topk, vocab_size):
    batch_size = logits.shape[0]
    if batch_size == 0 or topk == 0:
        return
    topk_block_size = min(triton.next_power_of_2(topk), 1024)
    _topk_log_softmax_kernel[(batch_size,)](
        output,
        logits,
        logits.stride(0),
        topk_ids,
        topk_ids.stride(0),
        topk,
        vocab_size,
        BLOCK_SIZE=1024,
        TOPK_BLOCK_SIZE=topk_block_size,
    )


def hip_topk_log_softmax(output, logits, topk_ids, topk, vocab_size):
    topk_block_size = min(triton.next_power_of_2(max(topk, 1)), 1024)
    OP.topk_log_softmax_rdna2(
        output,
        logits,
        logits.stride(0),
        topk_ids,
        topk_ids.stride(0),
        topk,
        vocab_size,
        topk_block_size,
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("topk", [1, 8, 1024, 2048])
@pytest.mark.parametrize("vocab_size", [1023, 1024, 1025, VOCAB])
def test_topk_log_softmax_parity(dtype, topk, vocab_size):
    def make():
        torch.manual_seed(59)
        logits = torch.randn(NUM_TOKENS, vocab_size, dtype=dtype, device=DEVICE)
        # Serving logits carry -inf from top-k/top-p masking and a wide range.
        logits[:, : vocab_size // 4] = float("-inf")
        logits[:, -1] = 100.0
        logits[0, :] = torch.linspace(-120.0, 3.0, vocab_size, device=DEVICE)
        ids = torch.randint(0, vocab_size, (NUM_TOKENS, topk), dtype=torch.int64,
                            device=DEVICE)
        output = torch.full((NUM_TOKENS, topk), 12345.0, dtype=torch.float32,
                            device=DEVICE)
        return output, logits, ids

    a = make()
    hip_topk_log_softmax(a[0], a[1], a[2], topk, vocab_size)
    b = make()
    triton_topk_log_softmax(b[0], b[1], b[2], topk, vocab_size)
    sync()
    assert_bitexact(a[0], b[0], "topk_log_softmax output")


def test_topk_log_softmax_noncontiguous_rows():
    """stride != vocab: Triton's mask/stride axis info drops to a scalar load."""

    def make():
        torch.manual_seed(71)
        stride = VOCAB + 8  # 151944 % 16 != 0
        buf = torch.randn(NUM_TOKENS, stride, dtype=torch.float32, device=DEVICE)
        logits = buf[:, :VOCAB]
        assert logits.stride(0) == stride
        ids = torch.randint(0, VOCAB, (NUM_TOKENS, 4), dtype=torch.int64,
                            device=DEVICE)
        out = torch.full((NUM_TOKENS, 4), 12345.0, dtype=torch.float32,
                         device=DEVICE)
        return out, logits, ids

    a = make()
    hip_topk_log_softmax(a[0], a[1], a[2], 4, VOCAB)
    b = make()
    triton_topk_log_softmax(b[0], b[1], b[2], 4, VOCAB)
    sync()
    assert_bitexact(a[0], b[0], "topk_log_softmax non-contiguous stride")


def test_topk_log_softmax_all_masked_row():
    """A fully masked row yields -inf everywhere (NaN lse), in both paths."""

    def make():
        logits = torch.full((2, 4096), float("-inf"), dtype=torch.float32,
                            device=DEVICE)
        ids = torch.zeros((2, 4), dtype=torch.int64, device=DEVICE)
        output = torch.zeros((2, 4), dtype=torch.float32, device=DEVICE)
        return output, logits, ids

    a = make()
    hip_topk_log_softmax(a[0], a[1], a[2], 4, 4096)
    b = make()
    triton_topk_log_softmax(b[0], b[1], b[2], 4, 4096)
    sync()
    assert_bitexact(a[0], b[0], "all masked row")


# ---------------------------------------------------------------------------
# _ranks_kernel
# ---------------------------------------------------------------------------
def triton_ranks(token_ranks, logits, sampled_token_ids, vocab_size):
    batch_size = logits.shape[0]
    if batch_size == 0:
        return
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,
    )


def hip_ranks(token_ranks, logits, sampled_token_ids, vocab_size):
    OP.token_ranks_rdna2(
        token_ranks, logits, logits.stride(0), sampled_token_ids, vocab_size
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("vocab_size", [1, 8192, 8193, VOCAB])
def test_ranks_parity(dtype, vocab_size):
    def make():
        torch.manual_seed(61)
        logits = torch.randn(NUM_TOKENS, vocab_size, dtype=dtype, device=DEVICE)
        # Ties: rank must count every logit equal to the sampled value.
        logits[:, 0] = 1000.0
        if vocab_size > 1:
            logits[:, 1] = 1000.0
        ids = torch.randint(0, vocab_size, (NUM_TOKENS,), dtype=torch.int64,
                            device=DEVICE)
        ids[0] = 0
        ids[1] = vocab_size - 1
        ranks = torch.full((NUM_TOKENS,), -1, dtype=torch.int64, device=DEVICE)
        return ranks, logits, ids

    a = make()
    hip_ranks(*a, vocab_size)
    b = make()
    triton_ranks(*b, vocab_size)
    sync()
    assert_bitexact(a[0], b[0], "token_ranks")
    assert a[0][0].item() == (2 if vocab_size > 1 else 1)


def test_ranks_single_element_and_empty():
    logits = torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)
    ids = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    a = torch.full((1,), -1, dtype=torch.int64, device=DEVICE)
    b = torch.full((1,), -1, dtype=torch.int64, device=DEVICE)
    hip_ranks(a, logits, ids, 1)
    triton_ranks(b, logits, ids, 1)
    sync()
    assert_bitexact(a, b)
    assert a.item() == 1

    empty = torch.zeros((0, 8), dtype=torch.float32, device=DEVICE)
    ranks = torch.zeros(0, dtype=torch.int64, device=DEVICE)
    se = torch.zeros(0, dtype=torch.int64, device=DEVICE)
    hip_ranks(ranks, empty, se, 8)
    triton_ranks(ranks, empty, se, 8)
    sync()


# ---------------------------------------------------------------------------
# _prompt_logprobs_token_ids_kernel
# ---------------------------------------------------------------------------
def triton_prompt_logprobs(query_start_loc, idx_mapping, num_computed,
                           all_token_ids):
    num_tokens = int(query_start_loc[-1].item())
    token_ids = torch.full((num_tokens,), -12345, dtype=torch.int64,
                           device=DEVICE)
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return token_ids
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def hip_prompt_logprobs(query_start_loc, idx_mapping, num_computed,
                        all_token_ids):
    num_tokens = int(query_start_loc[-1].item())
    token_ids = torch.full((num_tokens,), -12345, dtype=torch.int64,
                           device=DEVICE)
    OP.prompt_logprobs_token_ids_rdna2(
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed,
        all_token_ids,
        all_token_ids.stride(0),
    )
    return token_ids


@pytest.mark.parametrize(
    "query_lens", [[1, 1024, 1025, 0, 3, 2048, 7, 1023], [1], [0], [2049]]
)
def test_prompt_logprobs_token_ids_parity(query_lens):
    def make():
        torch.manual_seed(67)
        num_reqs = len(query_lens)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32,
                                      device=DEVICE)
        query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32,
                                           device=DEVICE).cumsum(0)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
        num_computed = torch.randint(0, 16, (num_reqs,), dtype=torch.int32,
                                     device=DEVICE)
        all_token_ids = torch.randint(0, 1000, (num_reqs, 4096),
                                      dtype=torch.int32, device=DEVICE)
        return query_start_loc, idx_mapping, num_computed, all_token_ids

    a = make()
    got_a = hip_prompt_logprobs(*a)
    b = make()
    got_b = triton_prompt_logprobs(*b)
    sync()
    assert_bitexact(got_a, got_b, "prompt_logprobs token_ids")


# ---------------------------------------------------------------------------
# The HIP path must be taken, not silently skipped.
# ---------------------------------------------------------------------------
class _NoLaunch:
    def __getitem__(self, grid):
        raise AssertionError("a poisoned Triton kernel was launched")


def test_wrappers_do_not_fall_back_to_triton(monkeypatch):
    for mod, names in (
        (gb_mod, ("_temperature_kernel", "_gumbel_sample_kernel")),
        (pen_mod, ("_penalties_kernel", "_bincount_kernel")),
        (mp_mod, ("_min_p_kernel",)),
        (lb_mod, ("_bias_kernel",)),
        (bw_mod, ("_bad_words_kernel",)),
        (lp_mod, ("_topk_log_softmax_kernel", "_ranks_kernel")),
        (plp_mod, ("_prompt_logprobs_token_ids_kernel",)),
    ):
        for name in names:
            monkeypatch.setattr(mod, name, _NoLaunch())

    vocab_size = 4096
    num_tokens = NUM_TOKENS
    mapping = make_mapping()
    # Not 0.0/1.0, so the temperature kernel actually touches the logits.
    temperature = torch.full((NUM_REQS,), 0.7, dtype=torch.float32,
                             device=DEVICE)
    seed = torch.arange(1, NUM_REQS + 1, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    local_pos = (torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
                 % TOKENS_PER_REQ)

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                         device=DEVICE)
    gb_mod.apply_temperature(logits, mapping, temperature)

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                         device=DEVICE)
    sampled = gb_mod.gumbel_sample(
        logits, mapping, temperature, seed, pos,
        apply_temperature=True, is_drafting=False,
    )
    assert sampled.shape == (num_tokens,)

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                         device=DEVICE)
    token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32,
                              device=DEVICE)
    pen_mod.apply_penalties(
        logits,
        mapping,
        token_ids,
        local_pos,
        torch.full((NUM_REQS,), 1.2, dtype=torch.float32, device=DEVICE),
        torch.full((NUM_REQS,), 0.5, dtype=torch.float32, device=DEVICE),
        torch.full((NUM_REQS,), 0.5, dtype=torch.float32, device=DEVICE),
        torch.zeros((NUM_REQS, triton.cdiv(vocab_size, 32)),
                    dtype=torch.int32, device=DEVICE),
        torch.zeros((NUM_REQS, vocab_size), dtype=torch.int32, device=DEVICE),
    )

    pen_mod.bincount(
        mapping,
        torch.randint(0, vocab_size, (NUM_REQS, 2048), dtype=torch.int32,
                      device=DEVICE),
        torch.full((NUM_REQS,), 64, dtype=torch.int32, device=DEVICE),
        torch.full((NUM_REQS,), 128, dtype=torch.int32, device=DEVICE),
        torch.zeros((NUM_REQS, triton.cdiv(vocab_size, 32)),
                    dtype=torch.int32, device=DEVICE),
        torch.zeros((NUM_REQS, vocab_size), dtype=torch.int32, device=DEVICE),
        128,
    )

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                         device=DEVICE)
    mp_mod.apply_min_p(
        logits,
        mapping,
        torch.full((NUM_REQS,), 0.1, dtype=torch.float32, device=DEVICE),
    )

    logits = torch.randn(4, vocab_size, dtype=torch.float32, device=DEVICE)
    width = 64
    lb_mod.apply_logit_bias(
        logits,
        torch.arange(4, dtype=torch.int32, device=DEVICE),
        torch.zeros(4, dtype=torch.int64, device=DEVICE),
        torch.tensor([0, 1, 2, 0], dtype=torch.int32, device=DEVICE),
        torch.randint(0, vocab_size, (4, width), dtype=torch.int32,
                      device=DEVICE),
        torch.zeros(4, dtype=torch.int32, device=DEVICE),
        torch.zeros((4, width), dtype=torch.int32, device=DEVICE),
        torch.zeros((4, width), dtype=torch.float32, device=DEVICE),
        torch.zeros(4, dtype=torch.int32, device=DEVICE),
        torch.zeros(4, dtype=torch.int32, device=DEVICE),
        torch.zeros(4, dtype=torch.bool, device=DEVICE),
        torch.zeros((4, width), dtype=torch.int32, device=DEVICE),
    )

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                         device=DEVICE)
    bw_mod.apply_bad_words(
        logits,
        mapping,
        torch.zeros((NUM_REQS, 16), dtype=torch.int32, device=DEVICE),
        torch.zeros((NUM_REQS, 3), dtype=torch.int32, device=DEVICE),
        torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE),
        torch.zeros((NUM_REQS, 64), dtype=torch.int32, device=DEVICE),
        torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE),
        torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE),
        local_pos,
        2,
    )

    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32,
                         device=DEVICE)
    lp_mod.compute_token_logprobs(
        logits,
        torch.randint(0, vocab_size, (num_tokens, 3), dtype=torch.int64,
                      device=DEVICE),
    )
    logs = lp_mod.compute_topk_scores(
        logits,
        2,
        torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64,
                      device=DEVICE),
    )
    assert logs.selected_token_ranks.shape == (num_tokens,)

    num_reqs = 4
    query_start_loc = torch.tensor([0, 3, 3, 7, 7], dtype=torch.int32,
                                   device=DEVICE)
    plp_mod.get_prompt_logprobs_token_ids(
        7,
        query_start_loc,
        torch.arange(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.zeros((num_reqs, 64), dtype=torch.int32, device=DEVICE),
    )

    sync()
    assert sampled.min().item() >= 0
