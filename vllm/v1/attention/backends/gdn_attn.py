# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import os
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec

logger = init_logger(__name__)


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # zeros, not empty: RDNA2 hipMalloc leaves pages uncommitted, and
        # FULL-graph replay must never see capture-time garbage indices.
        # Copies below are blocking — non_blocking=True raced graph replay
        # (16k c=8: some slots duct, then the process dies).
        from vllm.utils.rocm_graph_keepalive import immortal_zeros

        bs = self.decode_cudagraph_max_bs
        self.spec_state_indices_tensor: torch.Tensor = immortal_zeros(
            (bs, self.num_spec + 1), torch.int32, device
        )
        self.non_spec_state_indices_tensor: torch.Tensor = immortal_zeros(
            (bs,), torch.int32, device
        )
        self.spec_sequence_masks: torch.Tensor = immortal_zeros(
            (bs,), torch.bool, device
        )
        self.spec_token_indx: torch.Tensor = immortal_zeros(
            (bs * (self.num_spec + 1),), torch.int32, device
        )
        self.non_spec_token_indx: torch.Tensor = immortal_zeros(
            (bs * (self.num_spec + 1),), torch.int32, device
        )
        self.spec_query_start_loc: torch.Tensor = immortal_zeros(
            (bs + 1,), torch.int32, device
        )
        self.non_spec_query_start_loc: torch.Tensor = immortal_zeros(
            (bs + 1,), torch.int32, device
        )
        self.num_accepted_tokens: torch.Tensor = immortal_zeros(
            (bs,), torch.int32, device
        )
        # Grow-only GPU metadata. Mixed 16k prefill used to torch.gather /
        # async_tensor_h2d a new tensor every step; those frees recycle
        # FULL-graph pages on gfx1030 (16k c=1 PASS, c=4 then ducts).
        self._meta_scratch: dict[str, torch.Tensor] = {}
        self._meta_graveyard: list[torch.Tensor] = []
        self._align_offsets: torch.Tensor | None = None
        self.device = device

    def _grow_meta(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buf = self._meta_scratch.get(name)
        need = list(shape)
        grow = (
            buf is None
            or buf.dtype != dtype
            or buf.device != device
            or buf.dim() != len(need)
            or any(buf.size(i) < need[i] for i in range(len(need)))
        )
        if grow:
            grown = need
            if (
                buf is not None
                and buf.dim() == len(need)
                and buf.dtype == dtype
                and buf.device == device
            ):
                grown = [max(int(buf.size(i)), int(need[i])) for i in range(len(need))]
                self._meta_graveyard.append(buf)
            buf = torch.zeros(*grown, dtype=dtype, device=device)
            self._meta_scratch[name] = buf
        view = buf
        for i, s in enumerate(shape):
            if int(view.size(i)) != int(s):
                view = view.narrow(i, 0, s)
        return view

    def _h2d_meta(
        self, name: str, src: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        t = src.detach().contiguous()
        dst = self._grow_meta(name, tuple(t.shape), t.dtype, device)
        dst.copy_(t)
        return dst

    def _align_block_table(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = self.vllm_config.cache_config.mamba_cache_mode
        if mode in ("all", "none"):
            return block_table
        n_off = 1 + int(self.kv_cache_spec.num_speculative_blocks)
        n_req = int(seq_lens.shape[0])
        bs = int(self.kv_cache_spec.block_size)
        if (
            self._align_offsets is None
            or self._align_offsets.device != block_table.device
            or self._align_offsets.numel() != n_off
        ):
            self._align_offsets = torch.arange(
                n_off, device=block_table.device, dtype=torch.int32
            )
        start32 = self._grow_meta(
            "align_start32", (n_req,), torch.int32, block_table.device
        )
        # Mixed 16k: GPU seq_lens is padded/capture-shaped and can be 0/1
        # on a 16k decode row. (seq-1)//784 then gathers block 0 — the
        # NULL pad in align mode — so n_dec>=2 seqs share one page and
        # duct. CPU computed-token lengths are the last GDN page.
        if seq_lens_cpu is not None and int(seq_lens_cpu.shape[0]) >= n_req:
            # Clone: sl.sub(1) must not mutate engine seq_lens in place.
            sl = seq_lens_cpu[:n_req].to(dtype=torch.int32, copy=True)
            starts = sl.sub_(1).clamp_(min=0).floor_divide_(bs)
            start32.copy_(starts)
        else:
            src = seq_lens[:n_req]
            if src.dtype != torch.int32:
                tmp = self._grow_meta(
                    "align_seq", (n_req,), src.dtype, src.device
                )
                tmp.copy_(src)
                src = tmp
            start32.copy_(src)
            start32.sub_(1)
            start32.clamp_(min=0)
            start32.floor_divide_(bs)
        bt = block_table[:n_req]
        w = int(bt.size(1))
        if (
            getattr(self, "_align_col", None) is None
            or self._align_col.device != bt.device
            or int(self._align_col.numel()) != w
        ):
            self._align_col = torch.arange(w, device=bt.device, dtype=torch.int32)
        # Last non-null column in 0..start. Align pads NULL at col 0 for
        # any sequence longer than one 784-token page; gathering only
        # start can still hit 0 if seq_lens is 1/padded. n_dec>=2 then
        # shares the null page and poisons FULL replay.
        scores = self._grow_meta(
            "align_scores", (n_req, w), torch.int32, bt.device
        )
        scores.copy_(self._align_col.view(1, w).expand(n_req, w))
        scores.add_(1)
        scores.masked_fill_(self._align_col.view(1, w) > start32.unsqueeze(1), 0)
        scores.masked_fill_(bt == 0, 0)
        last = self._grow_meta("align_last", (n_req,), torch.int32, bt.device)
        torch.amax(scores, dim=1, out=last)
        last.sub_(1)
        last.clamp_(min=0)
        idx32 = self._grow_meta(
            "align_idx32", (n_req, n_off), torch.int32, block_table.device
        )
        idx32.copy_(self._align_offsets.view(1, n_off).expand(n_req, n_off))
        idx32.add_(last.unsqueeze(1))
        idx32.clamp_(max=w - 1)
        idx64 = self._grow_meta(
            "align_idx64", (n_req, n_off), torch.int64, block_table.device
        )
        idx64.copy_(idx32)
        out = self._grow_meta(
            "align_bt", (n_req, n_off), block_table.dtype, block_table.device
        )
        torch.gather(bt, 1, idx64, out=out)
        if os.environ.get("VLLM_GDN_ALIGN_DEBUG", "0") == "1":
            null_rows = (out == 0).any(dim=1).nonzero().flatten().tolist()
            if null_rows:
                true_lens = (
                    seq_lens_cpu[:n_req].tolist()
                    if seq_lens_cpu is not None
                    else seq_lens[:n_req].tolist()
                )
                logger.warning(
                    "[gdn-align] null-page gather: path=%s n_req=%s "
                    "null_rows=%s true_lens=%s starts=%s last=%s",
                    "cpu" if (seq_lens_cpu is not None) else "gpu",
                    n_req,
                    null_rows[:8],
                    true_lens[:8],
                    start32.tolist()[:8],
                    last.tolist()[:8],
                )
        return out

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        # Blocking H2D into grow-only GPU scratch. async_tensor_h2d allocated
        # a fresh dest every mixed 16k step and recycled FULL-graph pages.
        return (
            self._h2d_meta(
                "chunk_indices",
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device,
            ),
            self._h2d_meta(
                "chunk_offsets",
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        sl_cpu = m._seq_lens_cpu
        if sl_cpu is None:
            sl_cpu = m.seq_lens_cpu_upper_bound
        block_table_tensor = self._align_block_table(
            m.block_table_tensor,
            m.seq_lens,
            seq_lens_cpu=sl_cpu,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if (
                num_spec_decodes == 0
                or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self._h2d_meta(
                    "spec_seq_masks",
                    spec_sequence_masks_cpu,
                    query_start_loc.device,
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            if (
                os.environ.get("VLLM_ROCM_MIXED_LOG", "0") == "1"
                and num_decodes > 0
                and num_prefills > 0
            ):
                _ids = non_spec_state_indices_tensor.detach().flatten().cpu().tolist()
                logger.info(
                    "gdn_align_mixed n_dec=%s n_pre=%s state_ids=%s",
                    num_decodes,
                    num_prefills,
                    _ids,
                )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                _qsl = non_spec_query_start_loc[
                    num_decodes : num_decodes + num_prefills + 1
                ]
                prefill_query_start_loc = self._grow_meta(
                    "pref_qsl",
                    tuple(_qsl.shape),
                    _qsl.dtype,
                    _qsl.device,
                )
                prefill_query_start_loc.copy_(_qsl)
                prefill_query_start_loc.sub_(num_decode_tokens)
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[
                    num_decodes : num_decodes + num_prefills
                ].contiguous()
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            # Prefill-only cu_seqlens: mixed batches peel 1-token decode
            # seqs onto causal_conv1d_update, so the varlen kernel's
            # nums_dict/batch_ptr must not include those length-1 seqs.
            assert prefill_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    prefill_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes]
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor
            )
            self.non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)
            # Pad to the FULL-graph token count so gdn_decode_rdna2's
            # ssm_state_indices[B] matches mixed_qkv.size(0).
            n_idx = min(
                max(batch_size, num_decode_tokens, m.num_actual_tokens),
                self.non_spec_state_indices_tensor.shape[0],
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :n_idx
            ]

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            n_q = min(n_idx + 1, self.non_spec_query_start_loc.shape[0])
            non_spec_query_start_loc = self.non_spec_query_start_loc[:n_q]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
