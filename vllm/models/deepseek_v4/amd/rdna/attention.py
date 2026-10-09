# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 sparse MLA attention for AMD RDNA (fp16 end to end).

The CDNA layer (``DeepseekV4ROCMAiterMLAAttention``) assumes bf16 activations
throughout: the fused q-norm/RoPE/KV-insert op, the gather workspace, the
Triton sparse prefill/decode kernels and the cached ``wo_a`` weight. gfx1030
has no bf16 math and the RDNA MoE/linear kernels take fp16, so this subclass
keeps every stage in the activation dtype and routes the hot paths to the
RDNA HIP kernels in ``csrc/rocm/rdna/dsv4``:

- q-norm + RoPE + fp8_ds_mla KV insert: ``dsv4_qnorm_rope_kv_insert_rdna``
  (same cache row as upstream; RoPE dims stored as bf16).
- sparse MLA decode: ``sparse_mla_decode_rdna2`` (fp16 q/out, reads the
  fp8_ds_mla rows directly).
- sparse MLA prefill: ``sparse_mla_prefill_rdna2`` over the fp16 gather
  workspace (same top-k + SWA index combine and sink handling as the Triton
  ragged prefill).
- inverse RoPE + ``wo_a`` bmm with ``wo_a`` cached in the activation dtype.

Everything launches on the current stream without host syncs, so the decode
path can be captured in FULL cudagraphs.
"""

from typing import cast

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.rdna2_w8a16_fp8_block import (
    W8A16_FP8_MAX_TOKENS,
    w8a16_fp8_mm,
)
from vllm.models.deepseek_v4.amd.rdna import ops as rdna_ops
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseMetadata,
    DeepseekV4ROCMAiterSparseSWAMetadata,
    combine_topk_swa_indices,
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _get_cached_wo_a_bf16,
    build_ragged_indices_from_dense,
    rocm_inverse_rope_rows_,
)
from vllm.v1.attention.ops.rocm_rdna2_mla_sparse import _hip_sparse_attn_decode
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


class DeepseekV4RDNAAttention(DeepseekV4ROCMAiterMLAAttention):
    """DeepSeek-V4 sparse MLA attention for RDNA (fp16 activations)."""

    # ── KV insert ──────────────────────────────────────────────────────────
    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        swa_kv_cache = self.swa_cache_layer.kv_cache
        if (
            not isinstance(attn_metadata, dict)
            or swa_kv_cache.dtype != torch.uint8
            or not rdna_ops.has_qnorm_rope_kv_insert()
        ):
            return super()._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        return rdna_ops.qnorm_rope_kv_insert(
            q.contiguous(),
            kv.contiguous(),
            swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.padded_heads,
            self.eps,
            swa_metadata.block_size,
        )

    # ── Output projection ──────────────────────────────────────────────────
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # forward_mqa already inverse-RoPE'd every row (the HIP decode does
        # not fuse it), so only the wo_a bmm and wo_b remain.
        num_tokens = o.shape[0]
        groups = self.n_local_groups
        o_ref = o.reshape(num_tokens, groups, -1)
        wo_a = self._wo_a_w8a16()
        if wo_a is None:
            w = _get_cached_wo_a_bf16(
                self.wo_a, groups, self.o_lora_rank, o_ref.shape[-1], dtype=o.dtype
            )
            z = torch.einsum("tgd,grd->tgr", o_ref, w)
            return self.wo_b(z.flatten(1))
        # fp8 wo_a straight from the checkpoint on the RDNA W8A16 kernel, one
        # GEMM per local group: no dequantized fp16 copy (~0.7 GB at TP=4).
        weights, scales, group_size = wo_a
        o_g = o_ref.transpose(0, 1).contiguous()  # [G, T, D]
        # W8A16 kernel for decode-sized M (atomic epilogue: zeroed output),
        # per-call dequant + rocBLAS for prefill; no persistent fp16 copy.
        small = num_tokens <= W8A16_FP8_MAX_TOKENS
        z = (torch.zeros if small else torch.empty)(
            groups, num_tokens, self.o_lora_rank, dtype=o.dtype, device=o.device
        )
        for g in range(groups):
            w8a16_fp8_mm(
                o_g[g], weights[g], scales[g], group_size, out=z[g], out_zeroed=small
            )
        return self.wo_b(z.transpose(0, 1).reshape(num_tokens, -1))

    def _wo_a_w8a16(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int] | None:
        """Per-group [K, N] fp8 bytes + [K/128, N] fp16 scales for wo_a.

        Built once (the first call is in the profile run, before capture);
        None when wo_a is not block-fp8 or the kernel is missing, which keeps
        the dequantized einsum path.
        """
        cached = getattr(self, "_wo_a_w8a16_cache", False)
        if cached is not False:
            return cached
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            get_fp8_block_weight_scale,
        )

        weight = self.wo_a.weight
        scale = get_fp8_block_weight_scale(self.wo_a)
        result = None
        if (
            hasattr(torch.ops._rocm_C, "gemm_w8a16_fp8_dense")
            and scale is not None
            and weight.element_size() == 1
            and weight.dtype == torch.float8_e4m3fn
            and weight.dim() == 2
            and scale.dim() == 2
        ):
            groups, rank = self.n_local_groups, self.o_lora_rank
            hidden = weight.shape[1]
            rows_per_block = rank * groups // scale.shape[0]
            k_block = hidden // scale.shape[1]
            w = weight.reshape(groups, rank, hidden)
            s = scale.float().reshape(groups, rank // rows_per_block, -1)
            weights = [w[g].t().contiguous().view(torch.uint8) for g in range(groups)]
            scales = [
                s[g].t().repeat_interleave(rows_per_block, dim=1).half().contiguous()
                for g in range(groups)
            ]
            result = (weights, scales, k_block)
        self._wo_a_w8a16_cache = result
        return result

    # ── Attention ──────────────────────────────────────────────────────────
    def _gather_workspace_shape(self, q: torch.Tensor) -> tuple[int, int, int]:
        swa_only = self.compress_ratio <= 1
        n = (
            0
            if swa_only
            else (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        )
        m = n + self.window_size + self.max_num_batched_tokens
        return (self.PREFILL_CHUNK_SIZE, m, q.shape[-1])

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape and output.dtype == q.dtype
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # Profile run: reserve the gather workspace prefill uses, in the
            # activation dtype.
            current_workspace_manager().get_simultaneous(
                (self._gather_workspace_shape(q), q.dtype),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        mla_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        compressed_cache = self.kv_cache if not swa_only else None
        num_decode_tokens = swa_metadata.num_decode_tokens

        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=compressed_cache,
                swa_k_cache=self.swa_cache_layer.kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=mla_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode_rdna(
                q[:num_decode_tokens],
                compressed_cache,
                swa_metadata,
                mla_metadata,
                swa_only,
                output[:num_decode_tokens],
            )
        rocm_inverse_rope_rows_(
            output[:, : self.n_local_heads, :],
            positions,
            self.rotary_emb.cos_sin_cache,
            self.rope_head_dim,
        )

    def _forward_decode_rdna(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                assert swa_metadata.is_valid_token is not None
                topk_ragged_indices, topk_ragged_indptr, _ = (
                    compute_global_topk_ragged_indices_and_indptr(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[: swa_metadata.num_decodes],
                        attn_metadata.block_size // self.compress_ratio,
                        swa_metadata.is_valid_token[:num_decode_tokens],
                    )
                )
            else:
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr
        assert swa_metadata.decode_swa_ragged_indices is not None
        assert swa_metadata.decode_swa_ragged_indptr is not None
        _hip_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            output=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None
        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]
        left_visible = swa_metadata.prefill_left_visible
        right_visible = swa_metadata.prefill_right_visible
        if left_visible is not None:
            assert right_visible is not None
            left_visible = left_visible[num_decode_tokens:]
            right_visible = right_visible[num_decode_tokens:]

        assert self.topk_indices_buffer is not None
        if swa_only:
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            n_compressed = 0
        else:
            if self.compress_ratio == 4:
                topk_indices = self.topk_indices_buffer[num_decode_tokens:][
                    :num_prefill_tokens
                ]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            n_compressed = (
                self.max_model_len + self.compress_ratio - 1
            ) // self.compress_ratio

        workspace_shape = self._gather_workspace_shape(q)
        m_rows = workspace_shape[1]
        kv = current_workspace_manager().get_simultaneous(
            (workspace_shape, q.dtype),
        )[0]
        kv_rows = kv.view(-1, q.shape[-1])
        sink = self.attn_sink[: q.shape[1]]

        for chunk_start in range(0, num_prefills, self.PREFILL_CHUNK_SIZE):
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None and compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=n_compressed,
                use_fnuz=False,
            )

            query_start = int(
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = int(
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                m_rows,
                n_compressed,
                max_image_tokens=self.max_image_tokens,
                left_visible=(
                    left_visible[query_start:query_end]
                    if left_visible is not None
                    else None
                ),
                right_visible=(
                    right_visible[query_start:query_end]
                    if right_visible is not None
                    else None
                ),
            )
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                combined_indices, combined_lens, num_rows=kv_rows.shape[0]
            )
            rdna_ops.sparse_mla_prefill(
                q[query_start:query_end],
                kv_rows,
                ragged_indices,
                ragged_indptr,
                kv_rows.shape[0],
                self.scale,
                sink,
                output[query_start:query_end],
            )
