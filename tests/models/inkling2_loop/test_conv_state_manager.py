# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the Inkling-2 conv-state KV cache spec and its block manager.

The conv ops read the history of a chunk before they write its state, so a prefix-cache
hit must not read a block that another request completes in the step being scheduled
(the scheduler registers the blocks of the tokens it schedules before they are
computed): ``Inkling2ConvStateManager`` defers such a request to the next step.  These
tests drive the real ``KVCacheManager`` / coordinator with a full-attention group, a
sliding-window group and the conv group, the way the engine groups the layers.

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE pytest tests/models/inkling2_loop/test_conv_state_manager.py -q
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm.models.inkling2_loop.amd.conv_state")

from vllm.models.inkling2_loop.amd.conv_state import (  # noqa: E402
    Inkling2ConvSpec,
    Inkling2ConvStateManager,
    conv_ring_rows,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

BS = 16
HEADS, WIDTH, RING = 8, 832, 3  # the real model at TP=1: 8 kv heads, k|v|attn|mlp


def conv_spec(block_size: int = BS, rows: int = RING, width: int = WIDTH):
    return Inkling2ConvSpec(
        block_size=block_size,
        num_kv_heads=HEADS,
        head_size=width,
        head_size_v=0,
        dtype=torch.float16,
        sliding_window=4,
        tokens_per_state=block_size,
        state_content_bytes=rows * width * 2,
    )


def make_manager(conv, num_blocks: int = 40) -> KVCacheManager:
    full = FullAttentionSpec(
        block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    swa = SlidingWindowSpec(
        block_size=BS,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=2 * BS,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["swa"], swa),
            KVCacheGroupSpec(["conv"], conv),
        ],
    )
    init_none_hash(sha256)
    return KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=BS,
        hash_block_size=BS,
        enable_caching=True,
    )


def make_request(name: str, token_ids: list[int]) -> Request:
    params = SamplingParams(max_tokens=17)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=name,
        prompt_token_ids=token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(BS, sha256),
        session_id=None,
    )


PROMPT = [i for i in range(4) for _ in range(BS)] + [9] * 5  # 4 full blocks + 5
HIT_BLOCKS = 3


def sibling() -> list[int]:
    """Shares the first ``HIT_BLOCKS`` blocks with PROMPT."""
    return PROMPT[: HIT_BLOCKS * BS] + [77] * 5


def schedule(mgr: KVCacheManager, name: str, tokens: list[int]):
    """The scheduler's order: look up the hit, then allocate the rest."""
    req = make_request(name, tokens)
    computed, hit, _ = mgr.get_computed_blocks(req)
    blocks = mgr.allocate_slots(req, len(tokens) - hit, hit, computed)
    return req, hit, blocks


def test_the_registry_hands_the_conv_spec_its_manager():
    assert (
        KVCacheSpecRegistry.get_manager_class(conv_spec()) is Inkling2ConvStateManager
    )
    plain = SlidingWindowSpec(
        block_size=BS,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )
    assert KVCacheSpecRegistry.get_manager_class(plain) is SlidingWindowManager


def test_a_hit_on_blocks_completed_in_the_same_step_is_deferred():
    mgr = make_manager(conv_spec())
    req0, hit0, blocks0 = schedule(mgr, "0", PROMPT)
    assert hit0 == 0 and blocks0 is not None
    free = mgr.block_pool.get_num_free_blocks()
    req1, hit1, blocks1 = schedule(mgr, "1", sibling())
    assert hit1 == HIT_BLOCKS * BS  # every group would hit ...
    assert blocks1 is None  # ... but the conv state of those blocks is not written yet
    assert free > 3 * 4  # not for lack of memory
    mgr.new_step_starts()
    req1, hit1, blocks1 = schedule(mgr, "1", sibling())
    assert hit1 == HIT_BLOCKS * BS and blocks1 is not None
    ids0, ids1 = mgr.get_block_ids("0"), mgr.get_block_ids("1")
    for group in range(3):  # the hit shares the blocks the owner computed
        assert ids1[group][HIT_BLOCKS - 1] == ids0[group][HIT_BLOCKS - 1]
    mgr.free(req0)
    mgr.free(req1)


def test_without_the_manager_the_same_step_hit_would_be_scheduled():
    """Control: a plain sliding-window group of the same window does not defer, so the
    deferral above is the conv manager's doing."""
    plain = SlidingWindowSpec(
        block_size=BS,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )
    mgr = make_manager(plain)
    _, _, blocks0 = schedule(mgr, "0", PROMPT)
    _, hit1, blocks1 = schedule(mgr, "1", sibling())
    assert blocks0 is not None and hit1 == HIT_BLOCKS * BS and blocks1 is not None


def test_a_hit_is_deferred_only_while_the_last_hit_block_is_new_in_the_step():
    mgr = make_manager(conv_spec())
    _, _, blocks0 = schedule(mgr, "0", PROMPT)
    assert blocks0 is not None
    mgr.new_step_starts()
    # step 2: both hit blocks cached a step ago; request 2 completes one more block
    _, hit1, blocks1 = schedule(mgr, "1", sibling())
    assert hit1 == HIT_BLOCKS * BS and blocks1 is not None
    extended = PROMPT[: HIT_BLOCKS * BS] + [5] * BS + [6] * 3
    _, hit2, blocks2 = schedule(mgr, "2", extended)
    assert hit2 == HIT_BLOCKS * BS and blocks2 is not None
    # request 3 hits the block that request 2 completes in this very step: deferred
    follower = extended[: (HIT_BLOCKS + 1) * BS] + [8] * 3
    req3, hit3, blocks3 = schedule(mgr, "3", follower)
    assert hit3 == (HIT_BLOCKS + 1) * BS and blocks3 is None
    mgr.new_step_starts()
    _, hit3, blocks3 = schedule(mgr, "3", follower)
    assert hit3 == (HIT_BLOCKS + 1) * BS and blocks3 is not None


def test_the_page_of_the_real_model_and_its_grouping():
    spec = conv_spec(64)
    assert spec.page_size_bytes == HEADS * RING * WIDTH * 2 == 39936
    assert spec.max_admission_blocks_per_request(8192, 4096) == 65
    grouped = UniformTypeKVCacheSpecs.from_specs({f"l{i}": spec for i in range(48)})
    assert grouped is not None and grouped.page_size_bytes == 48 * 39936
    sliding = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=256,
        dtype=torch.float16,
        sliding_window=4,
    )
    # a conv layer does not join sliding-window layers (the real model's windows are
    # 1024 tokens, so the other order cannot happen)
    assert UniformTypeKVCacheSpecs.from_specs({"c": spec, "s": sliding}) is None


def test_conv_ring_rows_tracks_scheduled_draft_tokens():
    from types import SimpleNamespace

    # No speculative config -> k = 4 ring rows (concurrent in-op single-token safety)
    cfg_none = SimpleNamespace(speculative_config=None, scheduler_config=None)
    assert conv_ring_rows(cfg_none, 4) == 4

    # Static K = 7 drafts (no dynamic batch schedule) -> 4 + 7 = 11 rows
    cfg_k7 = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            num_speculative_tokens_per_batch_size=None,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
    )
    assert conv_ring_rows(cfg_k7, 4) == 11

    # Dynamic schedule [[1, 2, 2], [3, 16, 0]] with mtp block 7 -> 6 rows (-40% vs 10)
    cfg_k2 = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            num_speculative_tokens_per_batch_size=[[1, 2, 2], [3, 16, 0]],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
    )
    assert conv_ring_rows(cfg_k2, 4) == 6

    # All-zero schedule -> 4 + 0 = 4 rows
    cfg_k0 = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            num_speculative_tokens_per_batch_size=[[1, 8, 0]],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    assert conv_ring_rows(cfg_k0, 4) == 4
