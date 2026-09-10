# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA_ATTN: standalone RDNA2 attention backend for the gfx1030 family.

Independent of RocmAttentionImpl/rocm_attn.py so upstream changes to the
ROCM dispatcher cannot touch FA-RDNA2. Kernels live in
``csrc/rocm/fa_rdna2.cu`` and are loaded by
``vllm/v1/attention/ops/fa_rdna2_backend.py`` via ``load_inline``.

Selected on gfx1030 when VLLM_USE_RDNA2_FA=1 (see platforms/rocm.py).
Coverage: head_size {128, 256}, fp16, non-quantized KV cache; anything
else is rejected by validate_configuration and the selector falls back
to ROCM_ATTN/TRITON_ATTN.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    has_native_kv_cache_layout,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVQuantMode,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

_SUPPORTED_HEAD_SIZES: tuple[int, ...] = (128, 256)
_SUPPORTED_ARCH_PREFIX: str = "gfx103"


def _on_gfx10x() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        return _SUPPORTED_ARCH_PREFIX in getattr(props, "gcnArchName", "")
    except Exception:
        return False


def is_available() -> bool:
    return os.environ.get("VLLM_USE_RDNA2_FA", "1") == "1" and _on_gfx10x()


_fa_rdna2_module = None


def _get_fa_rdna2_module():
    global _fa_rdna2_module
    if _fa_rdna2_module is None:
        from vllm.v1.attention.ops import fa_rdna2_backend as _m
        _fa_rdna2_module = _m
    return _fa_rdna2_module


def _reinterpret_v_to_5d(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Reinterpret 4D V ([nb, h, D, bs]) as 5D [nb, h, D/x, bs, x].

    reshape_and_cache writes K packed (x-innermost per (d/x, slot)) but V
    unpacked (slot-innermost per d). The 5D V view must carry the UNPACKED
    strides (..., x*bs, 1, bs), not the packed (..., x*bs, x, 1) a plain
    .view() would produce. Split D into (D/x, x) while slot is still
    innermost, then permute slot back to dim 3.
    """
    if (value_cache.dim() == 4 and key_cache.dim() == 5
            and head_size in _SUPPORTED_HEAD_SIZES):
        num_blocks, h_kv, head_size_d, block_sz = value_cache.shape
        x_dim = key_cache.shape[4]
        if head_size_d % x_dim == 0:
            value_cache = value_cache.view(
                num_blocks, h_kv, head_size_d // x_dim, x_dim, block_sz
            ).permute(0, 1, 2, 4, 3)
    return value_cache


@dataclass
class RdnaAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_query_start_loc: torch.Tensor | None = None


class RdnaAttentionMetadataBuilder(
        AttentionMetadataBuilder[RdnaAttentionMetadata]):
    # fa_rdna2_decode_paged is single-token-only; MTP-verify batches (ql>1)
    # must stay on the piecewise path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Match GDN: decode-first batch so mixed peel can pair q[s:e]
        # with block_table[i]. Without this, FA reports None and the
        # hybrid min() still gets 1 from GDN — but if GDN is skipped we
        # would not reorder and n_dec>=2 would pair the wrong pages.
        self._init_reorder_batch_threshold(1)
        self.use_full_cuda_graph = (
            vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        max_bs = vllm_config.scheduler_config.max_num_seqs
        max_tokens = vllm_config.compilation_config.max_cudagraph_capture_size or max_bs
        self.decode_cudagraph_max_bs = min(max_bs, max_tokens)
        block_size = max(int(kv_cache_spec.block_size), 1)
        max_model_len = vllm_config.model_config.max_model_len
        max_blocks = (max_model_len + block_size - 1) // block_size
        # Persistent decode buffers so FA-RDNA2 HIP kernels in a FULL
        # graph read runtime KV indices, not capture-time dummy tables.
        # Immortal hipMalloc: must not sit in the caching allocator after KV.
        from vllm.utils.rocm_graph_keepalive import immortal_zeros

        self._cg_seq_lens = immortal_zeros(
            (self.decode_cudagraph_max_bs,), torch.int32, device
        )
        self._cg_query_start_loc = immortal_zeros(
            (self.decode_cudagraph_max_bs + 1,), torch.int32, device
        )
        self._cg_slot_mapping = immortal_zeros((max_tokens,), torch.int32, device)
        self._cg_slot_mapping.fill_(-1)
        self._cg_block_table = immortal_zeros(
            (self.decode_cudagraph_max_bs, max_blocks), torch.int32, device
        )
        # Mixed 16k (max_query_len>1) must NOT write the decode _cg_*
        # buffers. Pre-size eager copies to max_num_seqs so the first
        # mixed 16k step does not torch.zeros-grow next to FULL graphs.
        self._eager_max_blocks = max_blocks
        max_batched = int(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._eager_seq_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=device
        )
        self._eager_query_start_loc = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self._eager_slot_mapping = torch.zeros(
            max_batched, dtype=torch.int32, device=device
        )
        self._eager_block_table = torch.zeros(
            max_bs, max_blocks, dtype=torch.int32, device=device
        )
        self._eager_pref_qsl = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self._eager_graveyard: list[torch.Tensor] = []

    def _eager_1d(self, attr: str, src: torch.Tensor) -> torch.Tensor:
        buf: torch.Tensor | None = getattr(self, attr, None)
        n = int(src.numel())
        if (
            buf is None
            or buf.dtype != src.dtype
            or buf.device != src.device
            or buf.numel() < n
        ):
            grown = n if buf is None else max(int(buf.numel()), n)
            if buf is not None:
                self._eager_graveyard.append(buf)
            buf = torch.zeros(grown, dtype=src.dtype, device=src.device)
            setattr(self, attr, buf)
        view = buf[:n]
        view.copy_(src.reshape(-1))
        return view

    def _eager_bt(self, src: torch.Tensor) -> torch.Tensor:
        r, c = int(src.shape[0]), int(src.shape[1])
        buf = self._eager_block_table
        if (
            buf is None
            or buf.dtype != src.dtype
            or buf.device != src.device
            or buf.shape[0] < r
            or buf.shape[1] < c
        ):
            gr = r if buf is None else max(int(buf.shape[0]), r)
            gc = c if buf is None else max(int(buf.shape[1]), c)
            if buf is not None:
                self._eager_graveyard.append(buf)
            buf = torch.zeros(gr, gc, dtype=src.dtype, device=src.device)
            self._eager_block_table = buf
        view = buf[:r, :c]
        view.copy_(src)
        return view

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RdnaAttentionMetadata:
        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.all())
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        if (
            self.use_full_cuda_graph
            and common_attn_metadata.max_query_len <= 1
            and num_reqs <= self.decode_cudagraph_max_bs
            and num_tokens <= self._cg_slot_mapping.shape[0]
        ):
            # FULL graphs pad to the captured token count (3 reqs -> graph of
            # 4). Decode kernels index seq_lens/block_table by token_idx in
            # [0, Q.size(0)). Slicing to num_reqs only OOBs the pad rows
            # (c=4/c=8 first-token-ok then garbage).
            n_pad = min(
                max(num_reqs, num_tokens),
                self._cg_seq_lens.shape[0],
                self._cg_block_table.shape[0],
            )
            self._cg_seq_lens[:num_reqs].copy_(seq_lens[:num_reqs])
            self._cg_seq_lens[num_reqs:].zero_()
            seq_lens = self._cg_seq_lens[:n_pad]
            self._cg_query_start_loc[: num_reqs + 1].copy_(
                query_start_loc[: num_reqs + 1]
            )
            query_start_loc = self._cg_query_start_loc[: num_reqs + 1]
            n_slots = min(num_tokens, slot_mapping.shape[0])
            self._cg_slot_mapping[:n_slots].copy_(slot_mapping[:n_slots])
            self._cg_slot_mapping[n_slots:].fill_(-1)
            slot_mapping = self._cg_slot_mapping[: max(n_slots, n_pad)]
            bt = min(num_reqs, block_table.shape[0])
            bk = min(block_table.shape[1], self._cg_block_table.shape[1])
            self._cg_block_table[:bt, :bk].copy_(block_table[:bt, :bk])
            self._cg_block_table[:bt, bk:].zero_()
            self._cg_block_table[bt:].zero_()
            block_table = self._cg_block_table[:n_pad]
        elif self.use_full_cuda_graph:
            seq_lens = self._eager_1d("_eager_seq_lens", seq_lens[:num_reqs])
            query_start_loc = self._eager_1d(
                "_eager_query_start_loc", query_start_loc[: num_reqs + 1]
            )
            n_slots = min(num_tokens, slot_mapping.shape[0])
            slot_mapping = self._eager_1d(
                "_eager_slot_mapping", slot_mapping[:n_slots]
            )
            bt = min(num_reqs, block_table.shape[0])
            bk = min(block_table.shape[1], self._eager_max_blocks)
            block_table = self._eager_bt(block_table[:bt, :bk])
        num_decodes, num_prefills, num_decode_tokens, _num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=1
            )
        )
        prefill_query_start_loc = None
        if num_prefills > 0 and num_decodes > 0:
            qsl = query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            prefill_query_start_loc = self._eager_1d("_eager_pref_qsl", qsl)
            prefill_query_start_loc.sub_(num_decode_tokens)
        # GPU seq_lens is capture-padded 0/1 (decode) or chunk length
        # (prefill). After 16k skip-compiled, FULL decode then attends
        # 1 or 704 tokens of a 16k KV and ducts; first token was Paris
        # from the last chunk. Copy exact CPU lengths on every step
        # (prefill, mixed, and decode). Never seq_lens_cpu_upper_bound.
        sl_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
        if sl_cpu is not None and int(sl_cpu.shape[0]) >= num_reqs:
            n = min(num_reqs, int(seq_lens.shape[0]), int(sl_cpu.shape[0]))
            seq_lens[:n].copy_(
                sl_cpu[:n].to(dtype=seq_lens.dtype, device=seq_lens.device)
            )
        return RdnaAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            causal=causal,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RdnaAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        common_attn_metadata.query_start_loc.zero_()
        return attn_metadata


class RdnaAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if head_size not in _SUPPORTED_HEAD_SIZES:
            raise NotImplementedError(
                f"RDNA_ATTN: head_size={head_size} not in "
                f"{_SUPPORTED_HEAD_SIZES}")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks
        self._alibi = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None else None
        )
        # Grow-only contig copies. Mixed 16k ``.contiguous()`` allocated
        # default-pool temps that recycled FULL graph-private pages.
        self._fa_contig: dict[str, torch.Tensor] = {}
        self._fa_grave: list[torch.Tensor] = []
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

    def pretouch_eager_contig(self, device) -> None:
        """Pin FA q copy to max chunk so 1k→16k never grows contig temps."""
        from vllm.utils.rocm_graph_keepalive import alloc_eager_or_capture

        pin = 2048 * 16 * 256
        buf = self._fa_contig.get("q")
        if (
            buf is None
            or buf.device != device
            or buf.dtype != torch.float16
            or buf.numel() < pin
        ):
            if buf is not None:
                self._fa_grave.append(buf)
            self._fa_contig["q"] = alloc_eager_or_capture(
                (pin,), torch.float16, device
            )

    def _fa_as_contig(self, name: str, t: torch.Tensor) -> torch.Tensor:
        if t.is_contiguous():
            return t
        from vllm.utils.rocm_graph_keepalive import alloc_eager_or_capture

        buf = self._fa_contig.get(name)
        n = int(t.numel())
        need = (
            buf is None
            or buf.dtype != t.dtype
            or buf.device != t.device
            or buf.numel() < n
        )
        if need:
            grown = n if buf is None else max(int(buf.numel()), n)
            # 1k mixed q is ~1024*H*D; first 16k chunk is 2048*H*D.
            if n >= 4096:
                grown = max(grown, 2048 * 16 * 256)
            if buf is not None:
                self._fa_grave.append(buf)
            buf = alloc_eager_or_capture((grown,), t.dtype, t.device)
            self._fa_contig[name] = buf
        view = buf[:n].view(t.shape)
        view.copy_(t.reshape(-1).view(t.shape))
        return view

    def _fa_pack2d(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """Always compact-copy a 2D table into a private [N, M] buffer.

        Returning a contiguous slice of a wider mixed table is not enough:
        decode_paged uses token_idx*stride(0) when size(0)>1. A slice of
        the mixed page table can have stride(0) != size(1) (eager_bt
        padded width) so seq>=1 reads the wrong 784-token pages.
        """
        n, m = int(t.size(0)), int(t.size(1))
        from vllm.utils.rocm_graph_keepalive import alloc_eager_or_capture

        buf = self._fa_contig.get(name)
        need = n * m
        if (
            buf is None
            or buf.dtype != t.dtype
            or buf.device != t.device
            or buf.numel() < need
        ):
            grown = need if buf is None else max(int(buf.numel()), need)
            if need >= 4096:
                grown = max(grown, 2048 * 16 * 256)
            if buf is not None:
                self._fa_grave.append(buf)
            buf = alloc_eager_or_capture((grown,), t.dtype, t.device)
            self._fa_contig[name] = buf
        packed = buf[:need].view(n, m)
        packed.zero_()
        packed.copy_(t)
        return packed

    def _can_run_fa_rdna2(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> bool:
        if not is_available():
            return False
        if query.dtype != torch.float16:
            return False
        if key_cache.dim() != 5 or value_cache.dim() != 5:
            return False
        if get_kv_quant_mode(self.kv_cache_dtype) != KVQuantMode.NONE:
            return False
        num_q_heads = query.shape[1] if query.dim() >= 2 else 0
        num_kv_heads = key_cache.shape[1] if key_cache.dim() >= 2 else 0
        if num_q_heads > 0 and num_kv_heads > 0 and \
                num_q_heads % num_kv_heads != 0:
            return False
        return True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RdnaAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )
        value_cache = _reinterpret_v_to_5d(key_cache, value_cache,
                                           self.head_size)

        if not self._can_run_fa_rdna2(query, key_cache, value_cache):
            raise NotImplementedError(
                "RDNA_ATTN: outside FA-RDNA2 coverage (dtype/quant/layout); "
                "selector should have routed this layer to ROCM_ATTN.")

        num_actual_tokens = attn_metadata.num_actual_tokens
        max_seqlen_q = attn_metadata.max_query_len
        from vllm.utils import rocm_graph_keepalive as _rgk

        if not _rgk.capturing_full:
            self.pretouch_eager_contig(query.device)
        seqused_k = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        max_seqlen_k = attn_metadata.max_seq_len
        cu_seqlens_q = attn_metadata.query_start_loc

        # FA-RDNA2 + MTP-verify has known online-softmax split-K drift
        # versus the Triton fallback; the prior gate used a `max_seqlen_q
        # == 3` heuristic, but that fired on every chunked-prefill step
        # with three tokens-per-sequence (Qwen3.5 hybrid) producing the
        # 'RDNA_ATTN: MTP verify pass routed to fallback for numerics.'
        # crash on non-MTP probes. We have not measured MTP on this path
        # so the gate is opt-in via VLLM_FARDNA2_ENABLE_SPEC_GATE=1.
        _spec_gate_enabled = os.environ.get(
            "VLLM_FARDNA2_ENABLE_SPEC_GATE", "0") == "1"
        if _spec_gate_enabled:
            _spec_q = int(os.environ.get(
                "VLLM_FARDNA2_SPEC_VERIFY_Q_LEN", "3"))
            if (max_seqlen_q == _spec_q
                    and num_actual_tokens <= 16 * seqused_k.size(0)):
                raise NotImplementedError(
                    "RDNA_ATTN: MTP verify pass routed to fallback "
                    "for numerics.")

        fa = _get_fa_rdna2_module()
        sliding_window = (self.sliding_window[0] + 1
                          if self.sliding_window[0] >= 0 else 0)
        paged_block_size = key_cache.shape[3]
        # Kernels index block_table as row*size(1). A non-contiguous view
        # (common with paged/prefix-cache batches) makes seq>=1 read the
        # wrong pages — first-token duct at c=8. Persistent FULL-graph
        # buffers are already contiguous; this is a no-op then.
        block_table = self._fa_as_contig("block_table", block_table)
        seqused_k = self._fa_as_contig("seqused_k", seqused_k)
        q = self._fa_as_contig("q", query[:num_actual_tokens])

        if max_seqlen_q <= 1:
            # kv_splits=16: sweep 2026-09-04 showed s16 >= s8 at every
            # (ctx, batch) cell for both D=256 geometries (Ornith
            # H_q16/H_kv4, Qwen3.8-27B-rank H_q6/H_kv1); decode CTAs are
            # few (B*H_q) so more splits = more occupancy, and the
            # combine stage costs <10 us.
            out_paged = fa.fa_rdna2_decode_paged(
                q,
                key_cache,
                value_cache,
                block_table,
                seqused_k,
                paged_block_size,
                kv_splits=16,
                sliding_window=sliding_window,
            )
        else:
            _num_seqs = seqused_k.size(0)
            _kv_splits = min(8, (max_seqlen_k + 1023) // 1024)
            # Mixed decode+prefill: tokens packed densely but per-seq query
            # lengths differ. Split-K reduce indexes global q_block as if
            # every sequence were max_seqlen_q long — skip it for mixed.
            _mixed = (
                max_seqlen_q > 1
                and num_actual_tokens < _num_seqs * max_seqlen_q
            )
            if not attn_metadata.causal:
                raise NotImplementedError(
                    "RDNA_ATTN: non-causal prefill not supported")
            n_dec = attn_metadata.num_decode_tokens
            n_dec_seqs = attn_metadata.num_decodes
            n_pref = attn_metadata.num_prefills
            if (
                _mixed
                and n_dec > 0
                and n_pref > 0
                and attn_metadata.prefill_query_start_loc is not None
            ):
                # Pair each decode seq with its query_start_loc span, not
                # q[i] (token i != request i if a prefill just flipped to
                # decode and the batch was reordered). Compact-copy the
                # decode page table so stride(0)==size(1).
                out_view = output[:num_actual_tokens].view(
                    num_actual_tokens, self.num_heads, self.head_size
                )
                qsl = attn_metadata.query_start_loc[: n_dec_seqs + 1]
                qsl_list = qsl.tolist()
                # Token offset of the first prefill token. num_decode_tokens
                # can disagree with qsl[n_dec] if a just-flipped seq has
                # query_len!=1; slicing q[n_dec:] then writing varlen at
                # rebased cu=0 would OOB g_pref_O into weights (seq-after
                # eager still emits duct after mixed 16k).
                pref_tok0 = int(qsl_list[n_dec_seqs])
                bt_dec = self._fa_pack2d(
                    "bt_dec_pack", block_table[:n_dec_seqs]
                )
                sk_dec = self._fa_as_contig(
                    "sk_dec_pack", seqused_k[:n_dec_seqs]
                )
                # One packed launch: per-seq peel reused process-wide
                # g_dec persist (zero_/write/copy) across n_dec>=2 at 16k
                # and the second seq's first token was already duct.
                # Packed [n_dec, pages] is contiguous so max_blocks=stride(0).
                if os.environ.get("VLLM_ROCM_MIXED_LOG", "0") == "1":
                    logger.info(
                        "fa_mixed n_dec=%s n_pref=%s qsl=%s sk_dec=%s "
                        "sk_pre=%s bt_pages=%s capturing_hip=%s",
                        n_dec_seqs,
                        n_pref,
                        qsl_list,
                        sk_dec.detach().cpu().tolist(),
                        seqused_k[n_dec_seqs:].detach().cpu().tolist(),
                        int(bt_dec.size(1)),
                        torch.cuda.is_current_stream_capturing(),
                    )
                q_dec = q[:pref_tok0]
                if not q_dec.is_contiguous():
                    q_dec = self._fa_as_contig("q_dec_pack", q_dec)
                out_dec = fa.fa_rdna2_decode_paged(
                    q_dec,
                    key_cache,
                    value_cache,
                    bt_dec,
                    sk_dec,
                    paged_block_size,
                    kv_splits=16,
                    sliding_window=sliding_window,
                )
                out_view[:pref_tok0].copy_(out_dec)
                torch.cuda.current_stream().synchronize()
                cu_pre = self._fa_as_contig(
                    "cu_pre", attn_metadata.prefill_query_start_loc
                )
                bt_pre = self._fa_pack2d(
                    "bt_pre_pack", block_table[n_dec_seqs:]
                )
                sk_pre = seqused_k[n_dec_seqs:]
                if not sk_pre.is_contiguous():
                    sk_pre = self._fa_as_contig("sk_pre", sk_pre)
                q_pre = self._fa_as_contig("q_pre", q[pref_tok0:])
                cu_last = int(cu_pre[-1].item()) if cu_pre.numel() else 0
                if int(cu_pre[0].item()) != 0 or cu_last != int(q_pre.shape[0]):
                    raise RuntimeError(
                        "mixed FA cu_pre/q_pre mismatch: "
                        f"cu0={int(cu_pre[0].item())} cu_last={cu_last} "
                        f"q_pre={int(q_pre.shape[0])} pref_tok0={pref_tok0} "
                        f"n_dec={n_dec} n_dec_seqs={n_dec_seqs} "
                        f"qsl={qsl_list}"
                    )
                # Per-seq varlen: the batched call with 16k seqused_k and
                # n_dec>=2 left g_pref_O / page-table stride wrong so the
                # still-prefilling seq's first token was already duct.
                # One-seq is the 16k c=1 path that is known-good.
                cu_list = [int(x) for x in cu_pre.tolist()]
                n_pref_seqs = len(cu_list) - 1
                cu1 = self._fa_contig.get("cu_pre_1seq")
                if (
                    cu1 is None
                    or cu1.dtype != cu_pre.dtype
                    or cu1.device != cu_pre.device
                    or cu1.numel() < 2
                ):
                    from vllm.utils.rocm_graph_keepalive import alloc_eager_or_capture

                    cu1 = alloc_eager_or_capture((2,), cu_pre.dtype, cu_pre.device)
                    self._fa_contig["cu_pre_1seq"] = cu1
                for j in range(n_pref_seqs):
                    qs, qe = cu_list[j], cu_list[j + 1]
                    cu1[0] = 0
                    cu1[1] = qe - qs
                    q_j = q_pre[qs:qe]
                    bt_j = bt_pre[j : j + 1]
                    sk_j = sk_pre[j : j + 1]
                    if max_seqlen_k < 4096 and self.head_size == 128:
                        out_j = fa.fa_rdna2_prefill_paged_varlen_short(
                            q_j,
                            key_cache,
                            value_cache,
                            bt_j,
                            cu1[:2],
                            sk_j,
                            paged_block_size,
                            causal=True,
                            sliding_window=sliding_window,
                        )
                    else:
                        out_j = fa.fa_rdna2_prefill_paged_varlen(
                            q_j,
                            key_cache,
                            value_cache,
                            bt_j,
                            cu1[:2],
                            sk_j,
                            paged_block_size,
                            causal=True,
                            sliding_window=sliding_window,
                        )
                    out_view[pref_tok0 + qs : pref_tok0 + qe].copy_(out_j)
                    torch.cuda.current_stream().synchronize()
                return output
            cu_seqlens_q = self._fa_as_contig("cu_seqlens_q", cu_seqlens_q)
            if max_seqlen_k < 4096 and self.head_size == 128:
                out_paged = fa.fa_rdna2_prefill_paged_varlen_short(
                    q,
                    key_cache,
                    value_cache,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    paged_block_size,
                    causal=True,
                    sliding_window=sliding_window,
                )
            elif (
                not _mixed
                and _kv_splits >= 2
                and _num_seqs <= 4
                and self.num_heads * _kv_splits >= 64
                # O_partial is fp32 [N, H, splits, D]. After hybrid KV pin
                # (7e9) + GDN immortal capture + FULL keepalives, gfx1030
                # has 0 B free (24.23 GiB ATen + 1.18 GiB reserved). A
                # 16k/2048-chunk split-K try-alloc of 130-192 MiB OOMs.
                # skip_attn profile_run never counted this linear-attn
                # neighbor (FA). Varlen prefill only needs O (fp16).
                and (
                    int(q.shape[0])
                    * int(self.num_heads)
                    * int(_kv_splits)
                    * int(self.head_size)
                    * 4
                    <= 32 * 1024 * 1024
                )
            ):
                out_paged = fa.fa_rdna2_prefill_paged_varlen_splitk(
                    q,
                    key_cache,
                    value_cache,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    paged_block_size,
                    causal=True,
                    kv_splits=_kv_splits,
                    sliding_window=sliding_window,
                )
            else:
                out_paged = fa.fa_rdna2_prefill_paged_varlen(
                    q,
                    key_cache,
                    value_cache,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    paged_block_size,
                    causal=True,
                    sliding_window=sliding_window,
                )
        output[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size
        ).copy_(out_paged)
        return output

    forward_includes_kv_cache_update: bool = False

    @eager_break_during_capture
    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY,
                              AttentionType.ENCODER):
            return
        # FULL-graph replay must read the persistent slot buffer that
        # RdnaAttentionMetadataBuilder copies into each step — not the
        # capture-time dummy from forward_context.slot_mapping.
        from vllm.forward_context import get_forward_context

        raw = get_forward_context().attn_metadata
        if isinstance(raw, dict):
            for md in raw.values():
                if type(md).__name__ == "RdnaAttentionMetadata":
                    sm = getattr(md, "slot_mapping", None)
                    if isinstance(sm, torch.Tensor):
                        slot_mapping = sm
                        break
        elif raw is not None and type(raw).__name__ == "RdnaAttentionMetadata":
            sm = getattr(raw, "slot_mapping", None)
            if isinstance(sm, torch.Tensor):
                slot_mapping = sm
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )
        # 5D K last-but-one dim is the kernel block; hybrid GDN pages are
        # 784 and must not go through the packed 16/32 native writer.
        block_size = int(key_cache.shape[3]) if key_cache.dim() == 5 else int(
            value_cache.shape[-1]
        )
        if block_size in (16, 32) and has_native_kv_cache_layout(
                key_cache, value_cache):
            PagedAttention.write_to_paged_cache(
                key, value, key_cache, value_cache,
                slot_mapping, self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )
        elif (
            os.environ.get("VLLM_RDNA2_KV_WRITER", "1") != "0"
            and key.dtype == torch.float16
            and key_cache.dim() == 5
            and value_cache.dim() == 4
            and hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "reshape_and_cache_flash_rdna2")
        ):
            # HIP stride-aware writer: hybrid GDN pages (block_size=784)
            # pad stride(0). Same addressing as Triton, no JIT scratch.
            sm = slot_mapping.flatten()
            n = int(key.size(0))
            if sm.numel() > n:
                sm = sm[:n]
            torch.ops._rocm_C.reshape_and_cache_flash_rdna2(
                key, value, key_cache, value_cache, sm,
            )
        else:
            # The native writer assumes densely packed blocks and corrupts
            # stride-padded hybrid layouts (Qwen3.5 GDN block sizes).
            triton_reshape_and_cache_flash(
                key, value, key_cache, value_cache,
                slot_mapping, self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )


class RdnaAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16"]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "RDNA_ATTN"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return RdnaAttentionImpl

    @staticmethod
    def get_builder_cls():
        return RdnaAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The FA-RDNA2 kernels take block_size as a runtime argument;
        # vectorized loads prefer % 8 == 0 but any positive size works.
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return list(_SUPPORTED_HEAD_SIZES)

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return _on_gfx10x()

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # K/V as two head groups so split_kv_cache's x-packed views are
        # view-expressible (mirrors RocmAttentionBackend; without this the
        # framework allocates interleaved K/V and the views fail).
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LHBNC, KVCacheLayout.LBHNC)


if is_available():
    try:
        _get_fa_rdna2_module()
    except Exception as exc:
        logger.debug("RDNA_ATTN pre-warm skipped: %s", exc)
