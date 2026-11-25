# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache spec and block manager of the Inkling-2 conv state.

Every layer visit has four depthwise short convolutions (k = 4) whose inputs
need the raw inputs of the previous ``k - 1`` tokens.  The state keeps them
position addressed in a small ring per cache block instead of one slot per token:

* block ``b`` of a request covers the tokens ``[b * B, (b + 1) * B)``; its page is
  ``[H, S, W]`` (``S`` ring rows, ``W`` channels of the four streams per head) in
  the model dtype and position ``p`` lives in row ``p % S``;
* a step writes only the last ``S`` positions of every (block, chunk)
  intersection, so a chunk of ``N`` tokens needs ``N / B`` pages (the former
  one-slot-per-token layout needed ``N / 4``) and every completed block ends with
  the raw inputs of its last ``k - 1`` tokens, which is what a prefix-cache hit on
  that block resumes from;
* ``S = k + max_scheduled_drafts`` (``k`` without speculative decoding): a verify
  step of up to ``1 + K`` tokens reads ``k - 1`` history rows from the cache while
  writing ``1 + K`` disjoint rows concurrently across CTAs (``(k - 1) + (1 + K) =
  k + K``), and the rows of any accepted prefix stay intact when later drafts are
  rejected because rows are addressed by position.

The conv ops (``csrc/rocm/inkling2_attn_rdna2.cu``) read the history of a chunk
before they write the state, so a hit must not read a block that another request
completes in the same step: :class:`Inkling2ConvStateManager` defers such a request
to the next step, the way ``MambaManager`` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.request import Request


def conv_ring_rows(vllm_config: VllmConfig, kernel_size: int) -> int:
    """Ring rows per cache block: ``k`` without speculative decoding, or
    ``k + max_scheduled_drafts`` when speculative decoding is active.

    Reading ``k - 1`` prior positions from the cache while up to ``1 + K`` tokens
    of a verify chunk write their own ``pos % ring`` concurrently across CTAs
    requires ``(k - 1) + (1 + K) = k + K`` distinct ring rows, which also keeps
    every accepted history tap intact when up to ``K`` drafts are rejected.
    """
    spec = vllm_config.speculative_config
    drafted = (getattr(spec, "num_speculative_tokens", None) or 0) if spec else 0
    if drafted <= 0:
        return kernel_size
    schedule = getattr(spec, "num_speculative_tokens_per_batch_size", None)
    if schedule:
        sched_cfg = getattr(vllm_config, "scheduler_config", None)
        max_seqs = getattr(sched_cfg, "max_num_seqs", None) or max(
            int(entry[1]) for entry in schedule
        )
        lookup = build_dynamic_sd_schedule_lookup(
            schedule,
            vllm_max_batch_size=max_seqs,
            vllm_num_speculative_tokens=drafted,
        )
        drafted = max(lookup[1:], default=0)
    return kernel_size + drafted


@dataclass(frozen=True, kw_only=True)
class Inkling2ConvSpec(SlidingWindowSpec):
    """Conv state of one layer visit.  The page is ``H * state_content_bytes`` with
    one state per block (``tokens_per_state == block_size``), and ``sliding_window``
    is the kernel size, so the manager keeps the blocks that hold the last ``k - 1``
    positions of every request."""

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, Inkling2ConvSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


class Inkling2ConvStateManager(SlidingWindowManager):
    """Sliding-window manager that does not hand out the conv state of a block which
    another request completes in the step being scheduled: the state is written
    after the history of the step has been read, so the hit would read stale rows.
    The request waits for the next step (``num_gpu_blocks + 1`` blocks needed
    makes the scheduler skip it, as in ``MambaManager``)."""

    def __init__(self, kv_cache_spec: Inkling2ConvSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            return self.block_pool.num_gpu_blocks + 1
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
            prefill_end,
        )

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(
            request,
            num_tokens,
            retention_interval=retention_interval,
            replay_boundaries=replay_boundaries,
        )
        blocks = self.req_to_blocks[request.request_id]
        for idx in range(before, self.num_cached_block.get(request.request_id, 0)):
            block = blocks[idx]
            if not block.is_null and block.block_hash is not None:
                self.cached_blocks_this_step.add(block.block_hash)

    def new_step_starts(self) -> None:
        self.cached_blocks_this_step.clear()


def _register() -> None:
    # The registry fills itself with the built-in specs on first use; registering
    # first would make it non-empty and skip them.
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        Inkling2ConvSpec,
        Inkling2ConvStateManager,
        uniform_type_base_spec=Inkling2ConvSpec,
    )


_register()
