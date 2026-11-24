# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode, decode_cudagraph_capture_grid
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        if block_tables.cp_size > 1:
            input_batch.dcp_local_seq_lens = prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_batch.seq_lens,
                input_batch.num_reqs,
                block_tables.cp_size,
                block_tables.cp_rank,
                block_tables.cp_interleave,
                num_reqs_padded=input_batch.num_reqs_after_padding,
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_lens: Mapping[int, int],
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: Any = None,
        decode_request_sizes: Sequence[int] | None = None,
    ) -> None:
        # Target compilation_config sizes max_cudagraph_capture_size by target
        # decode shapes (e.g. 8 tokens at SEQS=4 without speculative verify, or
        # 6 at SEQS=2, K=2), whereas each DFlash/DSpark request emits a full
        # block of q = 1 + num_speculative_steps tokens (e.g. 2 * 8 = 16 at b=2).
        # Widen a local copy of compilation_config (within max_num_batched_tokens)
        # and ensure every small active request count 1..min(max_reqs, 8) plus
        # the decode request grid is captured regardless of target capture caps.
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        max_reqs_needed = max(
            (min(r, max_seqs) for r in decode_query_lens.values()), default=0
        )
        max_tokens_needed = min(
            max(
                (q * min(r, max_seqs) for q, r in decode_query_lens.items()),
                default=0,
            ),
            max_batched,
        )
        cc = vllm_config.compilation_config
        if max_tokens_needed > 0 and cc is not None:
            cur_max = cc.max_cudagraph_capture_size or 0
            cur_dec = cc.max_decode_cudagraph_capture_tokens
            if cur_max < max_tokens_needed or (
                cur_dec is not None and cur_dec < max_tokens_needed
            ):
                vllm_config = copy.copy(vllm_config)
                cc = copy.copy(cc)
                cc.max_cudagraph_capture_size = max(cur_max, max_tokens_needed)
                if cur_dec is not None:
                    cc.max_decode_cudagraph_capture_tokens = max(
                        cur_dec, max_tokens_needed
                    )
                vllm_config.compilation_config = cc
        if decode_request_sizes is None and max_reqs_needed > 0:
            existing = (
                cc.decode_cudagraph_capture_request_sizes if cc is not None else None
            ) or ()
            decode_request_sizes = sorted(
                s
                for s in {
                    *existing,
                    *range(1, min(max_reqs_needed, 8) + 1),
                    *decode_cudagraph_capture_grid(max_reqs_needed),
                }
                if 1 <= s <= max_reqs_needed
            )
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_lens=decode_query_lens,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
            decode_request_sizes=decode_request_sizes,
        )

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        prepare_forward: Callable[[int, dict[str, Any] | None], None],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """``prepare_forward(num_reqs, attn_metadata)`` is captured ahead of the
        query forward."""

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state

            def forward(cg_mode: CUDAGraphMode) -> None:
                prepare_forward(num_reqs, attn_metadata)
                forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                )

            return forward

        super().capture(create_forward_fn, progress_bar_desc)
