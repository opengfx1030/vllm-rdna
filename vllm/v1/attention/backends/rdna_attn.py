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
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    has_native_kv_cache_layout,
)
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


class RdnaAttentionMetadataBuilder(
        AttentionMetadataBuilder[RdnaAttentionMetadata]):
    # fa_rdna2_decode_paged is single-token-only; MTP-verify batches (ql>1)
    # must stay on the piecewise path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RdnaAttentionMetadata:
        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.all())
        return RdnaAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal,
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
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

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
        seqused_k = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        max_seqlen_k = attn_metadata.max_seq_len
        cu_seqlens_q = attn_metadata.query_start_loc

        # Gate the FA-RDNA2 fast path off during MTP verify passes: its
        # online-softmax split-K numerics differ enough from the fallback
        # path that spec-accept-rate collapses. Verify passes have
        # max_seqlen_q == num_spec+1 with few tokens per sequence.
        _spec_q = int(os.environ.get("VLLM_FARDNA2_SPEC_VERIFY_Q_LEN", "3"))
        if (max_seqlen_q == _spec_q
                and num_actual_tokens <= 16 * seqused_k.size(0)):
            raise NotImplementedError(
                "RDNA_ATTN: MTP verify pass routed to fallback for numerics.")

        fa = _get_fa_rdna2_module()
        sliding_window = (self.sliding_window[0] + 1
                          if self.sliding_window[0] >= 0 else 0)
        paged_block_size = key_cache.shape[3]

        if max_seqlen_q <= 1:
            # kv_splits=16: sweep 2026-09-04 showed s16 >= s8 at every
            # (ctx, batch) cell for both D=256 geometries (Ornith
            # H_q16/H_kv4, Qwen3.8-27B-rank H_q6/H_kv1); decode CTAs are
            # few (B*H_q) so more splits = more occupancy, and the
            # combine stage costs <10 us.
            out_paged = fa.fa_rdna2_decode_paged(
                query[:num_actual_tokens],
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
            if not attn_metadata.causal:
                raise NotImplementedError(
                    "RDNA_ATTN: non-causal prefill not supported")
            if max_seqlen_k < 4096 and self.head_size == 128:
                out_paged = fa.fa_rdna2_prefill_paged_varlen_short(
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    paged_block_size,
                    causal=True,
                    sliding_window=sliding_window,
                )
            elif (_kv_splits >= 2 and _num_seqs <= 4
                    and self.num_heads * _kv_splits >= 64):
                out_paged = fa.fa_rdna2_prefill_paged_varlen_splitk(
                    query[:num_actual_tokens],
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
                    query[:num_actual_tokens],
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
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )
        block_size = value_cache.shape[3]
        if block_size in (16, 32) and has_native_kv_cache_layout(
                key_cache, value_cache):
            PagedAttention.write_to_paged_cache(
                key, value, key_cache, value_cache,
                slot_mapping, self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
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
