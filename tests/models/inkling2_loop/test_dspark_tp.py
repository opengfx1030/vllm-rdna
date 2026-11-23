# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling-2 DSpark draft under tensor parallelism (TP = 2, 4).

The draft is replicated on every rank (weights, KV cache, taps) except for what the
target shards by vocabulary: the embedding table and the LM head.  Under TP the draft

* embeds the anchor token of each block through the sharded table (every rank looks up
  the rows it owns, one all-reduce adds the owner's row back, exactly) and
* takes the argmax of every Markov step over the rank's vocabulary shard only: the
  ``(value, global id)`` partials are all-gathered and reduced by the kernel the
  target's ``greedy_token_ids`` uses (ties to the lowest id).

CPU: the shard geometry and what the draft refuses.  gfx1030 (``_rocm_C`` with the
``inkling2_dspark`` ops): the draft of N in-process ranks (threads that meet at every
collective, the sum taken in rank order like ``rdna_ar``) proposes bit for bit the
tokens and block hidden states of the single-rank draft.

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE:$TFM_MAIN pytest tests/models/inkling2_loop/test_dspark_tp.py
"""

from __future__ import annotations

import types

import pytest
import torch
from test_dspark_model import (
    PADDED,
    SYN_PADDED,
    TINY,
    VOCAB,
    _contexts,
    _gpu_ready,
    _synthetic,
    oracle,
    requires_gpu,
)

from vllm.models.inkling2_loop.amd import dspark as mod
from vllm.models.inkling2_loop.amd import parallel
from vllm.models.inkling2_loop.amd import state as state_mod


@pytest.fixture(scope="module")
def draft():
    return oracle.round_to_bf16(oracle.random_draft(TINY, PADDED, seed=3))


@pytest.mark.parametrize("size", [1, 2, 4])
def test_the_draft_keeps_the_vocabulary_shard_of_the_target_head(draft, size):
    covered = 0
    for rank in range(size):
        tp = parallel.TensorParallel(size, rank)
        model = oracle.build_model(draft, VOCAB, PADDED, 4, 4096, "cpu", tp=tp)
        lo, rows, take = parallel.lm_head_shard(VOCAB, size, rank)
        assert (model._head_first, model._head_rows, model._head_take) == (
            lo,
            rows,
            take,
        )
        assert model._embed_first == parallel.lm_head_shard(PADDED, size, rank)[0]
        s = model._scratch  # the base / logits rows hold the shard, 8-aligned
        assert s.base.shape[1] == s.logits.shape[1] == -(-rows // 8) * 8
        covered += take
    assert covered == VOCAB


def test_a_sampled_draft_is_refused_under_tensor_parallelism(draft, monkeypatch):
    hf = oracle.draft_hf_config(draft, VOCAB, PADDED)
    fake = oracle.fake_vllm_config(hf, TINY.num_steps, 4, 4096)
    fake.speculative_config.draft_sample_method = "probabilistic"
    monkeypatch.setattr(state_mod, "get_current_vllm_config", lambda: fake)
    monkeypatch.setattr(
        mod, "current_tensor_parallel", lambda: parallel.TensorParallel(2, 0)
    )
    with torch.device("cpu"), pytest.raises(NotImplementedError, match="greedy"):
        mod.Inkling2DSparkModel(vllm_config=fake)
    # one rank has no such limit
    monkeypatch.setattr(mod, "current_tensor_parallel", lambda: parallel.NO_TP)
    with torch.device("cpu"):
        mod.Inkling2DSparkModel(vllm_config=fake)


def test_attach_refuses_a_head_of_another_shard(draft):
    model = oracle.build_model(
        draft, VOCAB, PADDED, 4, 4096, "cpu", tp=parallel.TensorParallel(2, 1)
    )
    wrong = torch.zeros(VOCAB, TINY.hidden_size, dtype=torch.float16)  # all rows
    model.lm_head = types.SimpleNamespace(weight=wrong)
    model.model.embed_tokens = types.SimpleNamespace(weight=wrong)
    with pytest.raises(ValueError, match="vocabulary shard"):
        model.attach_target(None)


@pytest.fixture(scope="module")
def synthetic():
    if not _gpu_ready():
        pytest.skip("needs a gfx1030 GPU with the inkling2_dspark ops")
    return _synthetic()


@requires_gpu
@pytest.mark.parametrize("size", [2, 4])
def test_the_ranks_of_a_group_propose_the_single_rank_draft(synthetic, size):
    """Same contexts, same anchors: the sharded embedding and the sharded head give
    the single-rank draft's block hidden states and draft tokens exactly (the
    elementwise math of every vocabulary row is the same, only the argmax is split)."""
    from test_tensor_parallel import ThreadGroup, ThreadTP

    d, lm, emb, norm, anchors = synthetic
    kwargs = {"padded_vocab": SYN_PADDED, "max_reqs": 40, "page_size": 16}
    one = oracle.HipDraft(d, lm, emb, norm, **kwargs)
    # anchors in every shard of the embedding table, a context past the window
    ctx = _contexts([37, 1100, 2100, 90, 300], anchors, seed=5)
    want = one.propose(ctx, capture_logits=False)
    assert (want.tokens != 0).any()
    group = ThreadGroup(size)
    ranks = [
        oracle.HipDraft(d, lm, emb, norm, tp=ThreadTP(group, r), **kwargs)
        for r in range(size)
    ]
    got = group.run(lambda r: ranks[r].propose(ctx, capture_logits=False))
    for r, res in enumerate(got):
        assert torch.equal(res.tokens, want.tokens), (r, res.tokens, want.tokens)
        gap = (res.hidden - want.hidden).norm() / want.hidden.norm()
        assert gap <= 3e-3, (r, float(gap))
        assert torch.equal(res.f_rnn, want.f_rnn)
