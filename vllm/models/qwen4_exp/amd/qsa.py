# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD ROCm QSA owner with Triton kernels."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, NamedTuple, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops import rocm_octave
from vllm.v1.attention.ops.int4_fp16_kv_layout import (
    INT4_FP16_BLOCK,
    int4_fp16_cache_view,
    int4_fp16_read_views,
    int4_fp16_record_bytes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    KVQuantMode,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer


class QsaMainKvLayout(NamedTuple):
    """How this deployment's main KV cache is stored.

    The layout is identified positively: ``kind`` states which record layout
    the cache holds, so a kernel is never handed bytes it cannot read. A
    packed cache read as fp16 produces finite, plausible, silently wrong
    numbers rather than a fault.
    """

    kind: str
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None

    @property
    def is_int4(self) -> bool:
        return self.kind == INT4_FP16_BLOCK


def resolve_main_kv_layout(
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    num_kv_heads: int,
    head_size: int,
) -> QsaMainKvLayout:
    """Views of the main KV cache for the layout it actually holds.

    The packed ``int4_fp16_block`` cache is ``[blocks, 2, slots, heads,
    record_bytes]``; the shared helper returns the payload/scale views as
    ``[pages, heads, slots, ...]``.
    """
    if kv_cache_dtype == INT4_FP16_BLOCK:
        # The packed cache is [blocks, 2, slots, heads, record_bytes]; the
        # shared helper wants the payload/scale views as
        # [pages, heads, slots, ...].
        cache = int4_fp16_cache_view(kv_cache, num_kv_heads, head_size)
        key, value, k_scale, v_scale = int4_fp16_read_views(cache, head_size)
        return QsaMainKvLayout(INT4_FP16_BLOCK, key, value, k_scale, v_scale)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    return QsaMainKvLayout(
        "fp16",
        canonicalize_singleton_dim_strides(key_cache),
        canonicalize_singleton_dim_strides(value_cache),
    )


def _octave_mid_o_capacity(vllm_config: VllmConfig) -> int:
    """Query rows times splits the Octave split-KV partials hold: every split
    up to the cudagraph capture size, and one split per row of a full chunk."""
    capture = vllm_config.compilation_config.max_cudagraph_capture_size or 4
    return max(
        capture * vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph,
        2 * vllm_config.scheduler_config.max_num_batched_tokens,
    )


# Exact window of the Octave QSA cache: fp16 shadow tables (entries, powers of
# two) of every stored token and of each sequence's leading positions; the
# kernels keep the last 64 and the first 4 selected positions exact.
_OCTAVE_WINDOW_ENTRIES = 16384
_OCTAVE_SINK_ENTRIES = 2048


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Layers run one after another, so they share one set of Octave
        # split-KV partials; reserve it before the workspace is locked.
        if kv_cache_spec.kv_quant_mode.is_octave and is_workspace_manager_initialized():
            num_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            numel = (
                _octave_mid_o_capacity(vllm_config)
                * num_heads
                * (kv_cache_spec.head_size + 2)
            )
            current_workspace_manager().get_simultaneous(((numel,), torch.float32))


_OCTAVE_CACHE_DTYPES: tuple[CacheDType, ...] = (
    "octave_k3v4",
    "octave_k3v3",
    "octave_k3v3_compact",
)
_QSA_CACHE_DTYPES: tuple[CacheDType, ...] = ("auto", "bfloat16", *_OCTAVE_CACHE_DTYPES)


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner.

    With an Octave ``kv_cache_dtype`` the main K/V cache holds one packed
    Octave slot per (token, KV head) and sparse attention runs the Octave
    sparse decode kernel over the selected positions.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        *_QSA_CACHE_DTYPES,
        "float16",
        INT4_FP16_BLOCK,
    ]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.kv_quant_mode == KVQuantMode.INT4_FP16_BLOCK:
            if (
                spec.dtype != torch.uint8
                or spec.head_size not in (128, 256)
                or spec.head_size_v != spec.head_size
            ):
                raise ValueError(
                    "int4_fp16_block requires equal K/V head sizes 128 or 256"
                )
            content = spec.num_kv_heads * int4_fp16_record_bytes(spec.head_size)
            if spec.state_content_bytes is not None:
                if spec.num_head_slots != 2 or spec.state_content_bytes != content:
                    raise ValueError("incompatible int4_fp16_block cache specification")
                return spec
            # K/V as two head slots, so the packed records of one side are
            # contiguous and the framework's per-layer view is
            # [blocks, 2, slots, heads*record_bytes] -- the same convention
            # the RDNA_ATTN backend publishes.
            return replace(spec, num_head_slots=2, state_content_bytes=content)
        from vllm.v1.attention.backends.rocm_octave_attn import (
            RocmOctaveAttentionBackend,
        )

        return RocmOctaveAttentionBackend.customize_spec(spec)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # QSA computes entirely in its own Triton kernels (forward_qsa ->
        # qsa_sparse_paged_attention); flash_attn_varlen_func is never called
        # from this impl. The FlashAttention* base classes are reused only for
        # metadata/plumbing. Requiring a working FA build therefore excludes
        # platforms that can run QSA perfectly well -- gfx1030 (RDNA2) has no
        # flash_attn or AITER build but runs the Triton path correctly.
        if not is_flash_attn_varlen_func_available() and not current_platform.is_rocm():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if (
            self.kv_cache_dtype not in _QSA_CACHE_DTYPES
            and self.kv_cache_dtype != INT4_FP16_BLOCK
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16/FP16, Octave, or packed "
                "int4_fp16_block main KV cache"
            )
        self.supports_quant_query_input = False
        self.octave_fmt: rocm_octave.OctaveFormat | None = None
        if self.kv_cache_dtype in _OCTAVE_CACHE_DTYPES:
            self.octave_fmt = rocm_octave.OctaveFormat.from_cache_dtype(
                self.kv_cache_dtype,
                self.head_size,
                rocm_octave.registered_rope_dim(self.head_size),
            )
            self._octave_signs: torch.Tensor | None = None
            self._octave_mid_o: torch.Tensor | None = None
            vllm_config = get_current_vllm_config()
            self._octave_splits = (
                vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
            )
            self._octave_capacity = _octave_mid_o_capacity(vllm_config)
            self._octave_positions: torch.Tensor | None = None
            device = torch.device("cuda", torch.accelerator.current_device_index())
            dtype = vllm_config.model_config.dtype
            row = (self.num_kv_heads, 2 * self.head_size)

            def table(entries: int) -> tuple[torch.Tensor, ...]:
                return (
                    torch.zeros((entries, *row), dtype=dtype, device=device),
                    torch.full((entries,), -1, dtype=torch.int32, device=device),
                    torch.zeros(entries, dtype=torch.int32, device=device),
                )

            self._wtab, self._wtags, self._wlocks = table(_OCTAVE_WINDOW_ENTRIES)
            self._stab, self._stags, self._slocks = table(_OCTAVE_SINK_ENTRIES)

    def _octave_buffers(
        self, num_q: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Sign bits and split-KV partials, from the workspace every layer
        shares (without one, a buffer allocated once, since captured graphs
        hold its address). Larger batches use fewer splits so they fit."""
        if self._octave_signs is None or self._octave_signs.device != device:
            self._octave_signs = rocm_octave.sign_bits(self.head_size).to(device)
        capacity = self._octave_capacity
        splits = max(2, min(self._octave_splits, capacity // max(num_q, 1)))
        row = self.head_size + 2
        numel = capacity * self.num_heads * row
        if is_workspace_manager_initialized():
            (buf,) = current_workspace_manager().get_simultaneous(
                ((numel,), torch.float32)
            )
        else:
            if self._octave_mid_o is None:
                self._octave_mid_o = torch.empty(
                    numel, dtype=torch.float32, device=device
                )
            buf = self._octave_mid_o
        mid_o = buf[: num_q * self.num_heads * splits * row]
        return (
            self._octave_signs,
            mid_o.view(num_q, self.num_heads, splits, row),
            splits,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if self.kv_cache_dtype == INT4_FP16_BLOCK:
            # The packed writer takes [2, pages, slots, heads, record_bytes];
            # it quantizes and packs each record.
            cache = int4_fp16_cache_view(kv_cache, self.num_kv_heads, self.head_size)
            ops.reshape_and_cache_int4_fp16_rdna2(key, value, cache, slot_mapping)
            return
        if self.octave_fmt is not None:
            n = slot_mapping.shape[0]
            if n == 0:
                return
            signs, _, _ = self._octave_buffers(1, key.device)
            torch.ops._C.octave_cache_store(
                key[:n],
                value[:n],
                kv_cache,
                slot_mapping,
                signs,
                signs,
                self.octave_fmt.kernel_code,
            )
            assert self._octave_positions is not None
            torch.ops._C.octave_window_store(
                key[:n],
                value[:n],
                slot_mapping,
                self._octave_positions[:n],
                self._wtab,
                self._wtags,
                self._wlocks,
                self._stab,
                self._stags,
                self._slocks,
            )
            return
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        from .ops import qsa as _qsa_ops

        if _qsa_ops.MONO_QSA:
            # MonoKernel QSA: the sparse kernel writes every live row; only
            # padding needs zeros
            if output.shape[0] > num_tokens:
                output[num_tokens:].zero_()
        else:
            output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        if self.octave_fmt is not None:
            assert self._octave_positions is not None
            signs, mid_o, splits = self._octave_buffers(num_tokens, query.device)
            torch.ops._C.octave_decode_sparse(
                output[:num_tokens],
                query[:num_tokens],
                kv_cache,
                attn_metadata.block_table,
                token_to_req,
                logical_indices,
                mid_o,
                signs,
                signs,
                self.scale,
                splits,
                self.octave_fmt.kernel_code,
                self._octave_positions[:num_tokens],
                self._wtab,
                self._wtags,
                self._stab,
                self._stags,
            )
            return output
        layout = resolve_main_kv_layout(
            kv_cache, self.kv_cache_dtype, self.num_kv_heads, self.head_size
        )
        key_cache, value_cache = layout.key_cache, layout.value_cache
        if layout.is_int4:
            # The packed payload is uint8; the query stays fp16. The layout
            # was identified above, so the kernel is handed bytes it can read.
            if query.dtype != torch.float16:
                raise NotImplementedError(
                    "Qwen4Exp QSA requires an FP16 query for a packed INT4 KV cache"
                )
        elif key_cache.dtype != query.dtype or query.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise NotImplementedError("Qwen4Exp QSA requires matching BF16/FP16 Q/K/V")
        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            int4_scales=((layout.k_scale, layout.v_scale) if layout.is_int4 else None),
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Qwen4Exp QSA requires BF16 or FP16")
        if cache_config.cache_dtype not in (
            *_QSA_CACHE_DTYPES,
            "float16",
            INT4_FP16_BLOCK,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16/FP16, Octave, or packed "
                "int4_fp16_block main KV cache"
            )
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None and cache_config.cache_dtype != INT4_FP16_BLOCK:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        # The fused HIP QSA decode reads the main KV cache, so which kernel can
        # serve this deployment follows from the cache layout. A packed cache
        # has no Triton reader, so it needs the native packed-int4 kernel;
        # the mode and the cache dtype have to agree.
        from .ops.qsa_rdna2 import qsa_decode_mode, qsa_use_rdna2_int4

        if qsa_decode_mode() == "int4" and cache_config.cache_dtype != INT4_FP16_BLOCK:
            raise NotImplementedError(
                "Qwen4Exp QSA: VLLM_RDNA_QSA_DECODE_MODE=int4 needs an "
                "int4_fp16_block KV cache"
            )
        if cache_config.cache_dtype == INT4_FP16_BLOCK and not qsa_use_rdna2_int4():
            raise NotImplementedError(
                "Qwen4Exp QSA: an int4_fp16_block KV cache needs the native "
                "packed-int4 HIP QSA decode; the Triton path cannot read the "
                "packed record layout"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        # T46: the fused Triton qk-norm+rope+gate kernel has no CUDA-only
        # pieces; on gfx1030 it replaces ~20 eager kernels per QSA layer.
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and text_only
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if (
            self.kv_cache_dtype not in _OCTAVE_CACHE_DTYPES
            and self.kv_cache_dtype != INT4_FP16_BLOCK
            and self.kv_cache_torch_dtype not in (torch.bfloat16, torch.float16)
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 or FP16 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # The fused prepare writes raw BF16 K/V; an Octave cache is packed by
        # do_kv_cache_update instead.
        self.use_fused_qsa_prepare = (
            self.attn_output_gate
            and self.indexer.use_fused_pre_indexer
            and self.head_dim & (self.head_dim - 1) == 0
            and self.kv_cache_dtype not in _OCTAVE_CACHE_DTYPES
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        qkv: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> None:
        # query/key/value are None when the fused prepare runs inside the
        # indexer launch, which then writes the gate into ``gate``.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            if gate is not None:
                gate.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected, main_outputs = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
            gate_out=gate,
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            if impl.octave_fmt is not None:
                flat = positions[0] if positions.dim() == 2 else positions
                impl._octave_positions = flat.to(torch.int32)
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, _ = main_outputs
            # The gate is applied to every row outside this op; padded rows
            # must stay finite so they keep the zero output forward_qsa gives.
            assert gate is not None
            gate[num_tokens:].zero_()
        assert query is not None
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if self.use_fused_qsa_prepare:
            # Norm/RoPE/gate and the K/V cache write happen inside the op.
            attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
            gate = torch.empty_like(attn_output)
            if current_platform.opaque_attention_op():
                torch.ops.vllm.qwen4_exp_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    attn_output,
                    gate,
                    encoded_layer_name,
                )
            else:
                qwen4_exp_qsa_prepare_with_output(
                    hidden_states,
                    positions,
                    qkv,
                    attn_output,
                    gate,
                    encoded_layer_name,
                )
            flat_output = attn_output.view(num_tokens, -1)
            flat_output = flat_output * torch.sigmoid(gate.view(num_tokens, -1))
            output, _ = self.o_proj(flat_output)
            return output

        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen4_exp_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


@eager_break_during_capture
def _qsa_with_output_eager(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen4_exp_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Registered op; dispatches to the decorated break point during capture."""
    _qsa_with_output_eager(
        hidden_states, positions, query, key, value, output, layer_name
    )


direct_register_custom_op(
    op_name="qwen4_exp_qsa_with_output",
    op_func=qwen4_exp_qsa_with_output,
    mutates_args=["output"],
)


def qwen4_exp_qsa_prepare_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    output: torch.Tensor,
    gate: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the QSA transaction with the main QK-norm/RoPE/gate fused in."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen4ExpQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen4Exp QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        None,
        None,
        None,
        output,
        qkv=qkv,
        gate=gate,
    )


direct_register_custom_op(
    op_name="qwen4_exp_qsa_prepare_with_output",
    op_func=qwen4_exp_qsa_prepare_with_output,
    mutates_args=["output", "gate"],
)


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
    "qwen4_exp_qsa_prepare_with_output",
    "qwen4_exp_qsa_with_output",
]
