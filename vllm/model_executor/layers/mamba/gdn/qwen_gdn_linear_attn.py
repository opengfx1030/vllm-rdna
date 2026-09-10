# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x
from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
)
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Optional ROCm AITER Triton kernels for the GDN decode path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = (
    rocm_aiter_ops.are_gdn_triton_kernels_available()
    or rocm_aiter_ops.is_rdna_gdn_triton_kernels_available()
)

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)

MAX_FUSED_GDN_MTP_TOKENS = 8
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)


# Capture vs eager MUST be distinct tables. Mixed 16k decode peel
# (`decode_out`, max(bsz,16)) otherwise narrow-writes the tensor FULL
# capture baked into the graph.
_GDN_PREFILL_SCRATCH_CAPTURE: dict[tuple, torch.Tensor] = {}
_GDN_PREFILL_SCRATCH_EAGER: dict[tuple, torch.Tensor] = {}
_GDN_PREFILL_GRAVEYARD: list[torch.Tensor] = []


def _gdn_alloc_scratch(
    shape: list[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Allocate GDN temps. Capture: caching-allocator zeros (legal in graph).
    Eager 16k: immortal hipMalloc so pages never recycle FULL-graph storage."""
    from vllm.utils import rocm_graph_keepalive as _rgk

    if _rgk.capturing_full:
        t = torch.zeros(*shape, dtype=dtype, device=device)
        return _rgk.keepalive_if_capturing(t)
    # Mixed 16k runs under eager_alloc_isolation (ATen pool (0,3)).
    # immortal hipMalloc bypasses that pool and starves decode GEMM.
    return torch.zeros(*shape, dtype=dtype, device=device)


def _gdn_prefill_scratch(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    zero: bool,
) -> torch.Tensor:
    """Grow-only GDN temps. Mixed 16k varies NT/T; a new torch.zeros of a
    different shape must not hipFree a buffer whose data_ptr is in a FULL
    graph. Old storages stay mapped in the graveyard."""
    from vllm.utils import rocm_graph_keepalive as _rgk

    table = (
        _GDN_PREFILL_SCRATCH_CAPTURE
        if _rgk.capturing_full
        else _GDN_PREFILL_SCRATCH_EAGER
    )
    key = (str(device), name, dtype, len(shape))
    buf = table.get(key)
    need = (
        buf is None
        or buf.dtype != dtype
        or buf.device != device
        or buf.dim() != len(shape)
    )
    if not need:
        need = any(buf.size(i) < shape[i] for i in range(len(shape)))
    if need:
        grown = list(shape)
        if buf is not None and buf.dim() == len(shape):
            grown = [max(int(buf.size(i)), int(shape[i])) for i in range(len(shape))]
            _GDN_PREFILL_GRAVEYARD.append(buf)
        # Pin T (dim 1 of A/w/o) to 2048 and NT (dim 1 of 5D h) to 64 on
        # the first eager alloc. 1k mixed uses T=784/NT=13; 16k c=1 uses
        # T=1568/NT=25 — that grow recycled FULL-graph pages.
        # Mixed 16k also grows decode_ssm_save dim 0 (n_dec 1→2→3); pin
        # to 16 so n_dec>=2 does not torch.zeros next to FULL graphs.
        if not _rgk.capturing_full:
            if name in ("decode_ssm_save", "decode_conv_save") and grown[0] <= 16:
                grown[0] = 16
            if name == "decode_out_acc" and len(grown) > 1 and grown[1] <= 16:
                grown[1] = 16
            # 1D packed qkv (rearrange_fused): 1k uses ~1024*qkv, 16k chunk
            # uses 2048*qkv. That grow recycled size-1 FULL (Parisduct).
            if name == "rearrange_fused":
                grown[0] = max(grown[0], 2048 * 8192)
            if len(grown) >= 2:
                if 256 < grown[1] <= 2048:
                    grown[1] = 2048
                elif len(grown) == 5 and 8 < grown[1] <= 64:
                    grown[1] = 64
        buf = _gdn_alloc_scratch(grown, dtype, device)
        table[key] = buf
    view = buf
    for i, s in enumerate(shape):
        if int(view.size(i)) != int(s):
            view = view.narrow(i, 0, s)
    if not view.is_contiguous():
        # Clone would allocate; grow so the storage is already contiguous
        # for the requested shape on the next call.
        grown = [max(int(buf.size(i)), int(shape[i])) for i in range(len(shape))]
        _GDN_PREFILL_GRAVEYARD.append(buf)
        buf = _gdn_alloc_scratch(grown, dtype, device)
        table[key] = buf
        view = buf[tuple(slice(0, s) for s in shape)]
    if zero:
        view.zero_()
    return view


def _gdn_prefill_chain_rdna2(
    q: torch.Tensor,                # [1, L, Hg, K] fp16 (from prep, B=1)
    k: torch.Tensor,                # [1, L, Hg, K] fp16
    v: torch.Tensor,                # [1, L, H, V]  fp16 (H=HV in Qwen3.5/3.6)
    g_cumsum: torch.Tensor,         # [1, L, H]      fp32 (cumsum'd, from prep)
    beta: torch.Tensor,             # [1, L, H]      fp32 (from prep)
    initial_state: torch.Tensor,    # [N, H, V, K]   fp32 (from ssm_state, may be zeros)
    scale: float,
    cu_seqlens: torch.Tensor,       # [N+1] int32 (always varlen for prefill)
    chunk_indices: torch.Tensor,    # [NT, 2] int32
    chunk_offsets: torch.Tensor,    # [N+1] int32
    chunk_size: int = FLA_CHUNK_SIZE,
):
    """Native HIP prefill chain for gfx1030: replicates chunk.py:23-86 using
    torch.ops._rocm_C.gdn_prefill_* ops. Returns (o, final_state) matching
    the Triton `chunk_gated_delta_rule` convention (final_state fp32)."""
    # The HIP kernels require int32 index tensors. The metadata builder may
    # hand us int64 (prepare_chunk_offsets' cumsum / async_tensor_h2d with
    # dtype=None follows the source dtype), so coerce defensively. `.to` is a
    # no-op when already int32.
    cu_seqlens = cu_seqlens.to(torch.int32)
    chunk_indices = chunk_indices.to(torch.int32)
    chunk_offsets = chunk_offsets.to(torch.int32)

    B, T, Hg, K = q.shape
    H = g_cumsum.shape[-1]
    V = v.shape[-1]
    BT = chunk_size

    NT = chunk_indices.shape[0]
    # Persistent scratch (stable data_ptr) so 16k chunked prefill does not
    # hipMalloc after FULL graphs are captured — that churn overwrites the
    # HIP graph pool on gfx1030 and poisons decode replay (Paris then duct).
    A = _gdn_prefill_scratch(
        "A", (B, T, H, BT), torch.float32, q.device, zero=True
    )
    A_inv = _gdn_prefill_scratch(
        "A_inv", (B, T, H, BT), q.dtype, q.device, zero=True
    )
    w = _gdn_prefill_scratch("w", (B, T, H, K), q.dtype, q.device, zero=True)
    u = _gdn_prefill_scratch("u", tuple(v.shape), v.dtype, v.device, zero=True)
    # h matches the reference chunk_gated_delta_rule_fwd_h layout: 5D
    # [B, NT, H, V, K] (B=1 for the varlen prefill path).
    h = _gdn_prefill_scratch(
        "h", (B, NT, H, V, K), q.dtype, q.device, zero=True
    )
    v_new = _gdn_prefill_scratch(
        "v_new", tuple(v.shape), v.dtype, v.device, zero=True
    )
    final_state = _gdn_prefill_scratch(
        "final_state",
        tuple(initial_state.shape),
        initial_state.dtype,
        initial_state.device,
        zero=True,
    )
    o = _gdn_prefill_scratch("o", tuple(v.shape), v.dtype, v.device, zero=True)

    ops = torch.ops._rocm_C
    ops.gdn_prefill_kkt_rdna2(k, beta, g_cumsum, A, cu_seqlens, chunk_indices)
    ops.gdn_prefill_solve_wy_rdna2(A, k, v, beta, g_cumsum, A_inv, w, u,
                                   cu_seqlens, chunk_indices)
    ops.gdn_prefill_delta_h_rdna2(k, u, w, g_cumsum, h, v_new, initial_state,
                                  final_state, cu_seqlens, chunk_offsets,
                                  chunk_size)
    ops.gdn_prefill_o_rdna2(q, k, v_new, h, g_cumsum, o, scale, cu_seqlens,
                            chunk_offsets)
    return o, final_state


def _gdn_prefill_dispatch_available() -> bool:
    """True iff all 5 GDN prefill HIP ops are registered for this build."""
    # Opt-out: the chain's chunk-local indexing bug (fixed in
    # gdn_prefill_delta_h_rdna2.cu, k/w/u/v_new rows are now rebased per
    # chunk) is covered by a stage-level differential test; keep the env
    # as a fast rollback valve.
    if os.environ.get("VLLM_GDN_HIP_PREFILL") == "0":
        return False
    return (current_platform.is_rocm() and on_gfx10x() and hasattr(
        torch.ops._rocm_C, "gdn_prefill_prep_rdna2"))


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda
        # Stable GDN output so a later GEMM can keep a fixed data_ptr.
        # Size to the decode capture max; prefill (n larger) uses empty_like.
        cap = vllm_config.compilation_config.max_cudagraph_capture_size
        # Size to chunked-prefill max so mixed 16k never reallocates a
        # buffer whose first rows are baked into FULL decode graphs.
        prefill_n = vllm_config.scheduler_config.max_num_batched_tokens
        self._capture_n = cap if cap else 16
        self._packed_out_n = max(self._capture_n, prefill_n)
        # Capture vs eager MUST be distinct storages. Mixed 16k used to
        # write the same `_packed_out` / `_core_attn_buf` the FULL graph
        # captured (first 8 rows of a 2048 buffer), poisoning replay.
        self._packed_out_capture: torch.Tensor | None = None
        self._packed_out_eager: torch.Tensor | None = None
        self._core_attn_buf_capture: torch.Tensor | None = None
        self._core_attn_buf_eager: torch.Tensor | None = None
        # Back-compat aliases; prefer the split helpers below.
        self._packed_out: torch.Tensor | None = None
        self._core_attn_buf: torch.Tensor | None = None
        # Immortal hipMalloc for FULL-capture slots: never torch.zeros
        # during capture (that sits in the default pool after KV).
        self._ensure_immortal_capture_bufs(
            current_platform.current_device(),
            vllm_config.model_config.dtype,
        )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )
        self.gdn_decode_kernel = envs.VLLM_GDN_DECODE_KERNEL.strip().lower()
        if self.gdn_decode_kernel == "cuda":
            reason = self._fused_gdn_decode_unsupported_reason(vllm_config)
            if reason is not None:
                if "VLLM_GDN_DECODE_KERNEL" in os.environ:
                    raise ValueError(
                        f"VLLM_GDN_DECODE_KERNEL=cuda is not supported: {reason}"
                    )
                logger.info_once(
                    "Falling back to the Triton GDN decode path: %s", reason
                )
                self.gdn_decode_kernel = "triton"
        self.enable_fused_gdn_decode = self.gdn_decode_kernel == "cuda"
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
        # One-shot guard for the RDNA2 ssm_state page-commit scan below.
        self._rdna2_ssm_sanitized = False

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _fused_gdn_decode_unsupported_reason(
        self, vllm_config: VllmConfig
    ) -> str | None:
        conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()
        if (
            self.gqa_interleaved_layout
            or self.head_k_dim != 128
            or self.head_v_dim != 128
            or self.norm.activation != "silu"
            or vllm_config.model_config.dtype != torch.bfloat16
            or conv_state_dtype != torch.bfloat16
            or recurrent_state_dtype not in FUSED_GDN_STATE_DTYPES
            or not current_platform.has_device_capability(80)
        ):
            return (
                "the fused CUDA kernel requires a BF16 GDN model with "
                "K=V=128, SiLU gating, non-interleaved GQA layout, BF16 "
                "convolution cache, BF16 or FP32 recurrent state, and a "
                "GPU with compute capability 8.0+"
            )
        if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
            return "torch.ops._C.fused_gdn_decode_post_conv_mtp is not built"
        return None

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        nq, nk, nv = query.numel(), key.numel(), value.numel()
        fused = _gdn_prefill_scratch(
            "rearrange_fused",
            (nq + nk + nv,),
            query.dtype,
            query.device,
            zero=False,
        )
        fused[:nq].copy_(query.reshape(-1))
        fused[nq : nq + nk].copy_(key.reshape(-1))
        fused[nq + nk :].copy_(value.reshape(-1))

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def _ensure_immortal_capture_bufs(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """hipMalloc FULL-capture GDN outputs before graph capture.

        Must not run under Dynamo (no fake impl) and must not run while
        the stream is capturing (hipMalloc is illegal then).
        """
        if not current_platform.is_rocm():
            return
        if device is None:
            try:
                device = current_platform.current_device()
            except Exception:
                return
        if isinstance(device, torch.device) and device.type != "cuda":
            return
        if dtype is None:
            dtype = self.model_config.dtype
        from vllm.utils.rocm_graph_keepalive import immortal_zeros

        n = int(self._capture_n)
        n_eag = int(self._packed_out_n)
        hv = self.num_v_heads // self.tp_size
        if self._packed_out_capture is None:
            self._packed_out_capture = immortal_zeros(
                (n, self.hidden_size), dtype, device
            )
        if self._core_attn_buf_capture is None:
            self._core_attn_buf_capture = immortal_zeros(
                (n, hv, self.head_v_dim), dtype, device
            )
        # Mixed 16k first torch.zeros of these on the default/eager pool
        # recycles FULL-graph pages (seq-after Parisduct). Pin them.
        if self._packed_out_eager is None:
            self._packed_out_eager = immortal_zeros(
                (n_eag, self.hidden_size), dtype, device
            )
        if self._core_attn_buf_eager is None:
            self._core_attn_buf_eager = immortal_zeros(
                (n_eag, hv, self.head_v_dim), dtype, device
            )

    def pretouch_eager_prefill_scratch(self, device, dtype) -> None:
        """Allocate 16k-chunk GDN temps before the first 16k prefill.

        1k mixed pins T/NT but never touches every named buffer at the
        2048-token shape. The first 16k chunk then torch.zeros next to
        FULL-graph pages (Parisduct + size-1 dead).
        """
        from vllm.utils import rocm_graph_keepalive as _rgk

        if _rgk.capturing_full:
            return
        hv = self.num_v_heads // self.tp_size
        hk = self.num_k_heads // self.tp_size
        k = self.head_k_dim
        v = self.head_v_dim
        bt = 64
        t = 2048
        nt = 64
        _gdn_prefill_scratch(
            "A", (1, t, hv, bt), torch.float32, device, zero=False
        )
        _gdn_prefill_scratch(
            "A_inv", (1, t, hv, bt), dtype, device, zero=False
        )
        _gdn_prefill_scratch("w", (1, t, hv, k), dtype, device, zero=False)
        _gdn_prefill_scratch("u", (1, t, hv, v), dtype, device, zero=False)
        _gdn_prefill_scratch(
            "h", (1, nt, hv, v, k), dtype, device, zero=False
        )
        _gdn_prefill_scratch(
            "v_new", (1, t, hv, v), dtype, device, zero=False
        )
        _gdn_prefill_scratch("o", (1, t, hv, v), dtype, device, zero=False)
        _gdn_prefill_scratch("prep_q", (t, hk, k), dtype, device, zero=False)
        _gdn_prefill_scratch("prep_k", (t, hk, k), dtype, device, zero=False)
        _gdn_prefill_scratch("prep_v", (t, hv, v), dtype, device, zero=False)
        _gdn_prefill_scratch(
            "prep_g", (t, hv), torch.float32, device, zero=False
        )
        _gdn_prefill_scratch(
            "prep_beta", (t, hv), torch.float32, device, zero=False
        )
        _gdn_prefill_scratch(
            "rearrange_fused", (t * 8192,), dtype, device, zero=False
        )
        _gdn_prefill_scratch(
            "init_state", (1, hv, v, k), torch.float32, device, zero=False
        )
        _gdn_prefill_scratch(
            "final_state", (1, hv, v, k), torch.float32, device, zero=False
        )
        _gdn_prefill_scratch(
            "decode_ssm_save", (16, hv, v, k), torch.float32, device, zero=False
        )

    def _gdn_split_buf(
        self,
        capture_attr: str,
        eager_attr: str,
        n: int,
        rest: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        from vllm.utils import rocm_graph_keepalive as _rgk

        capturing = bool(_rgk.capturing_full)
        attr = capture_attr if capturing else eager_attr
        buf: torch.Tensor | None = getattr(self, attr)
        if capturing:
            ok = (
                buf is not None
                and buf.dtype == dtype
                and buf.device == device
                and buf.shape[1:] == rest
                and buf.shape[0] >= n
            )
            if not ok:
                # hipMalloc is illegal during capture; caching-allocator
                # zeros + keepalive is the only legal fallback.
                cap_n = max(n, int(self._capture_n))
                buf = torch.zeros((cap_n,) + rest, dtype=dtype, device=device)
                setattr(self, attr, buf)
                _rgk.keepalive_if_capturing(buf)
            return buf[:n]
        cap_n = max(n, self._packed_out_n)
        need = (
            buf is None
            or buf.dtype != dtype
            or buf.device != device
            or buf.shape[1:] != rest
            or buf.shape[0] < n
        )
        if need:
            grown = cap_n if buf is None else max(cap_n, int(buf.shape[0]), n)
            if buf is not None:
                _GDN_PREFILL_GRAVEYARD.append(buf)
            buf = torch.zeros((grown,) + rest, dtype=dtype, device=device)
            setattr(self, attr, buf)
        return buf[:n]

    @eager_break_during_capture
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Opaque full-layer custom op (OLMo pattern). Needed so dynamo
        # does not trace into GDN RMSNorm / conv1d (device_index skip).
        # Packed output keeps a stable data_ptr for breakable FULL replay.
        n = hidden_states.shape[0]
        output = self._gdn_split_buf(
            "_packed_out_capture",
            "_packed_out_eager",
            n,
            (hidden_states.shape[-1],),
            hidden_states.dtype,
            hidden_states.device,
        )
        return torch.ops.vllm.qwen_gdn_full_forward(
            hidden_states,
            output,
            _encode_layer_name(self.prefix),
        )

    def _full_forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        result = self._forward_method(hidden_states)
        num_tokens = result.shape[0]
        output[:num_tokens].copy_(result)
        logger.info_once(
            "Qwen GDN full forward running as vllm::qwen_gdn_full_forward "
            "(opaque to inductor; projections stay eager)"
        )

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output, _ = self.out_proj(core_attn_out)
        return output

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
                use_aiter=True,
            )

            return self._output_projection(core_attn_out, z)
        else:
            return self.forward_cuda(hidden_states)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        if use_fused_gdn_decode:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            output, _ = self.out_proj(core_attn_out.flatten(-2))
            return output

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/PR/28182
        hv = self.num_v_heads // self.tp_size
        core_attn_out = self._gdn_split_buf(
            "_core_attn_buf_capture",
            "_core_attn_buf_eager",
            num_tokens,
            (hv, self.head_v_dim),
            hidden_states.dtype,
            hidden_states.device,
        )
        core_attn_out.zero_()

        if not b.is_contiguous():
            _b = _gdn_prefill_scratch(
                "ba_b", tuple(b.shape), b.dtype, b.device, zero=False
            )
            _b.copy_(b)
            b = _b
        if not a.is_contiguous():
            _a = _gdn_prefill_scratch(
                "ba_a", tuple(a.shape), a.dtype, a.device, zero=False
            )
            _a.copy_(a)
            a = _a
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._output_projection(core_attn_out, z)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True
        if _gdn_prefill_dispatch_available():
            return

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del (
                dummy_mixed_qkv,
                q,
                k,
                v,
                dummy_a,
                dummy_b,
                g,
                beta,
                state,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
            )

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_aiter``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_aiter(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        if os.environ.get("VLLM_LOG_GDN_PTRS") == "1":
            try:
                with open("/tmp/gdn_ptrs.log", "a") as _f:
                    _f.write(
                        f"capturing={torch.cuda.is_current_stream_capturing()} "
                        f"nd={attn_metadata.num_decodes} "
                        f"nat={num_actual_tokens} "
                        f"conv={conv_state.data_ptr()} "
                        f"ssm={ssm_state.data_ptr()} "
                        f"nsi={(non_spec_state_indices_tensor.data_ptr() if non_spec_state_indices_tensor is not None else None)} "
                        f"mq={mixed_qkv.data_ptr()} "
                        f"out={core_attn_out.data_ptr()}\n"
                    )
            except Exception:
                pass
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # Split mixed non-spec-decode+prefill independently. Computed before
        # conv so 1-token decode seqs never enter the varlen prefill kernel
        # (BLOCK_M=8 on query_start_loc [0,1,...,N,N+chunk] OOBs conv_state
        # into adjacent FULL-graph pages on gfx1030).
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Prefer query_start_loc[num_decodes] over num_decode_tokens so a
        # just-flipped seq with query_len!=1 does not leave prefill kernels
        # indexing past the sliced activation (OOB into weights / KV).
        if (
            split_non_spec
            and attn_metadata.non_spec_query_start_loc is not None
        ):
            _nd0 = int(attn_metadata.num_decodes)
            _qsl0 = attn_metadata.non_spec_query_start_loc[: _nd0 + 1]
            num_decode_tokens = int(_qsl0[-1].item())

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            if split_non_spec:
                decode_qkv = mixed_qkv_non_spec[:num_decode_tokens]
                _dec_conv_idx = non_spec_state_indices_tensor[
                    : attn_metadata.num_decodes
                ]
                if _dec_conv_idx.dim() > 1:
                    _dec_conv_idx = _dec_conv_idx.reshape(_dec_conv_idx.shape[0], -1)[
                        :, 0
                    ]
                _dec_conv_idx = _dec_conv_idx.contiguous()
                decode_conv = causal_conv1d_update(
                    decode_qkv,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=_dec_conv_idx,
                    validate_data=True,
                )
                if decode_conv.data_ptr() != decode_qkv.data_ptr():
                    decode_qkv.copy_(decode_conv)
                # Prefill conv/GDN share the hybrid page table with decode.
                # If cache_indices overlap (pad 0 / wrong tail slice), prefill
                # overwrites the live decode conv_state and later tokens duct.
                # Snapshot decode rows, restore after prefill conv.
                _nd_save = int(attn_metadata.num_decodes)
                _dec_idx = non_spec_state_indices_tensor[:_nd_save]
                if _dec_idx.dtype != torch.int64:
                    _dec_idx = _dec_idx.to(torch.int64)
                _csave = _gdn_prefill_scratch(
                    "decode_conv_save",
                    (_nd_save,) + tuple(conv_state.shape[1:]),
                    conv_state.dtype,
                    conv_state.device,
                    zero=False,
                )
                torch.index_select(conv_state, 0, _dec_idx, out=_csave)
                prefill_qkv = mixed_qkv_non_spec[num_decode_tokens:]
                prefill_conv = causal_conv1d_fn(
                    prefill_qkv.transpose(0, 1),
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=attn_metadata.prefill_has_initial_state,
                    cache_indices=attn_metadata.prefill_state_indices,
                    query_start_loc=attn_metadata.prefill_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
                prefill_qkv.copy_(prefill_conv)
                conv_state.index_copy_(0, _dec_idx, _csave)
            else:
                mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                # - "cache_indices" updates the conv_state cache in positions
                #   pointed to by "state_indices_tensor"
                mixed_qkv_non_spec = causal_conv1d_fn(
                    mixed_qkv_non_spec_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            use_hip_prefill = _gdn_prefill_dispatch_available()
            if use_hip_prefill:
                _L = conv_output_prefill.shape[0]
                _HV = self.num_v_heads // self.tp_size
                _H = self.num_k_heads // self.tp_size
                _K = self.head_k_dim
                _V = self.head_v_dim
                _dev = conv_output_prefill.device
                _dtype = conv_output_prefill.dtype
                _L_alloc = max(_L, 2048)
                query_non_spec = _gdn_prefill_scratch(
                    "prep_q", (_L_alloc, _H, _K), _dtype, _dev, zero=True
                )[:_L]
                key_non_spec = _gdn_prefill_scratch(
                    "prep_k", (_L_alloc, _H, _K), _dtype, _dev, zero=True
                )[:_L]
                value_non_spec = _gdn_prefill_scratch(
                    "prep_v", (_L_alloc, _HV, _V), _dtype, _dev, zero=True
                )[:_L]
                g_non_spec = _gdn_prefill_scratch(
                    "prep_g", (_L_alloc, _HV), torch.float32, _dev, zero=True
                )[:_L]
                beta_non_spec = _gdn_prefill_scratch(
                    "prep_beta", (_L_alloc, _HV), torch.float32, _dev, zero=True
                )[:_L]
                torch.ops._rocm_C.gdn_prefill_prep_rdna2(
                    conv_output_prefill, a_prefill, b_prefill,
                    self.A_log, self.dt_bias,
                    query_non_spec, key_non_spec, value_non_spec,
                    g_non_spec, beta_non_spec,
                    attn_metadata.prefill_query_start_loc,
                    attn_metadata.chunk_indices,
                )
            else:
                (
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                ) = fused_post_conv_prep(
                    conv_output=conv_output_prefill,
                    a=a_prefill,
                    b=b_prefill,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    apply_l2norm=True,
                    output_g_exp=False,
                )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a_spec,
                    b=b_spec,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            decode_qkv = mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
            # Slice decode tokens by non_spec query_start_loc, not by
            # token==request index (a just-flipped prefill can sit at
            # qsl[i] != i if the prefix packing is uneven).
            _nd = int(attn_metadata.num_decodes)
            # HIP gdn_decode_rdna2 indexes ssm_state_indices[i * stride].
            # block_table[:, 0] is a strided gather; a 2-row mixed slice
            # with stride != 1 made seq>=1 read the wrong 784-token page.
            _idx = non_spec_state_indices_tensor[:_nd].contiguous()
            if _idx.dtype != torch.int32:
                _idx = _idx.to(torch.int32)
            _qsl = attn_metadata.non_spec_query_start_loc[: _nd + 1]
            _qsl_list = _qsl.tolist()
            _a_all = a
            _b_all = b
            _hv = self.num_v_heads // self.tp_size
            _acc = _gdn_prefill_scratch(
                "decode_out_acc",
                (1, _nd, _hv, self.head_v_dim),
                torch.float16,
                decode_qkv.device,
                zero=True,
            )
            _any = False
            _none = False
            for _i in range(_nd):
                _s = int(_qsl_list[_i])
                _e = int(_qsl_list[_i + 1])
                _part = self._hip_gdn_decode_bt(
                    mixed_qkv_non_spec[_s:_e],
                    _a_all[_s:_e],
                    _b_all[_s:_e],
                    ssm_state,
                    _idx[_i : _i + 1],
                )
                if _part is None:
                    _none = True
                    break
                _acc[:, _i : _i + 1].copy_(_part)
                torch.cuda.current_stream().synchronize()
                _any = True
            if _none or not _any:
                core_attn_out_decode = None
            else:
                core_attn_out_decode = _acc
            if core_attn_out_decode is None:
                query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                    decode_qkv
                )
                core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a[:num_decode_tokens],
                    b=b[:num_decode_tokens],
                    dt_bias=self.dt_bias,
                    q=query_decode,
                    k=key_decode,
                    v=value_decode,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            # Advanced indexing would allocate a new tensor on the default
            # pool and recycle FULL-graph pages. Copy into grow-only scratch.
            _idx = prefill_state_indices
            _init_shape = (_idx.numel(),) + tuple(ssm_state.shape[1:])
            initial_state = _gdn_prefill_scratch(
                "init_state",
                _init_shape,
                ssm_state.dtype,
                ssm_state.device,
                zero=False,
            )
            torch.index_select(ssm_state, 0, _idx, out=initial_state)
            # In-place mask; `initial_state[~mask] = 0` advanced-index writes
            # allocate a temp that recycled FULL-graph pages on mixed 16k.
            if prefill_has_initial_state is not None:
                _m = prefill_has_initial_state.to(dtype=initial_state.dtype)
                _m = _m.view([-1] + [1] * (initial_state.dim() - 1))
                initial_state.mul_(_m)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = (
                _gdn_prefill_chain_rdna2(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g_cumsum=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    scale=self.head_k_dim ** -0.5,
                    cu_seqlens=attn_metadata.prefill_query_start_loc,
                    chunk_indices=attn_metadata.chunk_indices,
                    chunk_offsets=attn_metadata.chunk_offsets,
                )
                if use_hip_prefill
                else self.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=attn_metadata.prefill_query_start_loc,
                    chunk_indices=attn_metadata.chunk_indices,
                    chunk_offsets=attn_metadata.chunk_offsets,
                    use_qk_l2norm_in_kernel=False,
                )
            )
            # Init cache. Advanced indexing assigns a new temp; index_copy_
            # writes in place so mixed 16k does not recycle FULL-graph pages.
            _write = last_recurrent_state
            if _write.dtype != ssm_state.dtype:
                _write = _write.to(ssm_state.dtype)
            _idx = prefill_state_indices
            if _idx.dtype != torch.int64:
                _idx = _idx.to(torch.int64)
            if split_non_spec:
                _nd_save = int(attn_metadata.num_decodes)
                _dec_idx = non_spec_state_indices_tensor[:_nd_save]
                if _dec_idx.dtype != torch.int64:
                    _dec_idx = _dec_idx.to(torch.int64)
                _ssave = _gdn_prefill_scratch(
                    "decode_ssm_save",
                    (_nd_save,) + tuple(ssm_state.shape[1:]),
                    ssm_state.dtype,
                    ssm_state.device,
                    zero=False,
                )
                torch.index_select(ssm_state, 0, _dec_idx, out=_ssave)
            ssm_state.index_copy_(0, _idx, _write)
            if split_non_spec:
                ssm_state.index_copy_(0, _dec_idx, _ssave)

            if split_non_spec:
                # Stitch decode-first without torch.cat (new alloc each mixed
                # 16k step recycles FULL-graph pages).
                _dec = core_attn_out_decode
                _pre = core_attn_out_non_spec
                _n = _dec.shape[1] + _pre.shape[1]
                _stitched = _gdn_prefill_scratch(
                    "mixed_stitch",
                    (1, max(_n, 2048), _pre.shape[2], _pre.shape[3]),
                    _pre.dtype,
                    _pre.device,
                    zero=False,
                )[:, :_n]
                _stitched[:, : _dec.shape[1]].copy_(_dec)
                _stitched[:, _dec.shape[1] :].copy_(_pre)
                torch.cuda.current_stream().synchronize()
                core_attn_out_non_spec = _stitched
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_aiter(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _can_gdn_decode_rdna2(
        self,
        mixed_qkv: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor,
    ) -> bool:
        """HIP packed decode: fp16 activations, fp32 SSM, K=128, gfx10x."""
        if os.environ.get("VLLM_GDN_DECODE_RDNA2", "1") == "0":
            return False
        if not current_platform.is_rocm() or not on_gfx10x():
            return False
        if self.head_k_dim != 128:
            return False
        if mixed_qkv.dtype != torch.float16 or out.dtype != torch.float16:
            return False
        if ssm_state.dtype != torch.float32:
            return False
        return hasattr(torch.ops, "_rocm_C") and hasattr(
            torch.ops._rocm_C, "gdn_decode_rdna2"
        )

    def _run_gdn_decode_rdna2(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """AOT HIP decode. ``out`` is ``[B, 1, HV, V]`` fp16, contiguous."""
        if (
            not torch.cuda.is_current_stream_capturing()
            and not self._rdna2_ssm_sanitized
        ):
            ssm_state_has_nan = torch.isnan(ssm_state.float()).any().item()
            ssm_state_all_zero = not torch.any(ssm_state.float() != 0).item()
            if ssm_state_has_nan or ssm_state_all_zero:
                ssm_state.zero_()
            self._rdna2_ssm_sanitized = True
        logger.info_once(
            "GDN decode using HIP gdn_decode_rdna2 (cudagraph-safe)"
        )
        torch.ops._rocm_C.gdn_decode_rdna2(
            mixed_qkv,
            a,
            b,
            self.A_log,
            self.dt_bias,
            out,
            ssm_state,
            state_indices,
            self.head_k_dim**-0.5,
            True,
        )

    def _hip_gdn_decode_bt(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Packed HIP decode → ``[1, B, HV, V]`` (FLA fused_sigmoid layout).

        Used for mixed prefill+decode steps so the decode slice does not JIT
        ``fused_sigmoid_gating_delta_rule_update`` after FULL graph capture.
        """
        bsz = mixed_qkv.shape[0]
        if bsz == 0:
            hv = self.num_v_heads // self.tp_size
            return mixed_qkv.new_zeros((1, 0, hv, self.head_v_dim))
        hv = self.num_v_heads // self.tp_size
        from vllm.utils import rocm_graph_keepalive as _rgk

        # FULL capture pads to 16 for a stable data_ptr. Mixed eager
        # must be exact B so a 16-row buffer cannot alias capture.
        b_alloc = max(bsz, 16) if _rgk.capturing_full else bsz
        out = _gdn_prefill_scratch(
            "decode_out",
            (b_alloc, 1, hv, self.head_v_dim),
            torch.float16,
            mixed_qkv.device,
            zero=True,
        )[:bsz]
        if not self._can_gdn_decode_rdna2(mixed_qkv, ssm_state, out):
            return None
        self._run_gdn_decode_rdna2(
            mixed_qkv,
            a,
            b,
            ssm_state,
            state_indices[:bsz],
            out,
        )
        perm = out.permute(1, 0, 2, 3)
        if perm.is_contiguous():
            return perm
        contig = _gdn_prefill_scratch(
            "decode_out_bt",
            (1, b_alloc, hv, self.head_v_dim),
            torch.float16,
            mixed_qkv.device,
            zero=False,
        )[:, :bsz]
        contig.copy_(perm)
        return contig

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        if self._can_gdn_decode_rdna2(mixed_qkv_non_spec, ssm_state, out_buf):
            self._run_gdn_decode_rdna2(
                mixed_qkv_non_spec,
                a,
                b,
                ssm_state,
                non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
                out_buf,
            )
            return
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return

    def _forward_core_decode_spec_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        num_actual_tokens = attn_metadata.num_actual_tokens
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv = causal_conv1d_update(
            mixed_qkv[:num_actual_tokens],
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices[:num_requests, 0],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            query_start_loc=cu_seqlens[: num_requests + 1],
            max_query_len=state_indices.size(1),
            validate_data=False,
        )
        self._forward_core_decode_spec_post_conv_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b[:num_actual_tokens],
            a=a[:num_actual_tokens],
            output_gate=output_gate[:num_actual_tokens],
            core_attn_out=core_attn_out[:num_actual_tokens],
            attn_metadata=attn_metadata,
        )

    def _forward_core_decode_spec_post_conv_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        state_indices = attn_metadata.spec_state_indices_tensor
        cu_seqlens = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert state_indices is not None
        assert cu_seqlens is not None
        assert num_accepted_tokens is not None

        num_requests = attn_metadata.num_spec_decodes
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=state_indices[:num_requests],
            cu_seqlens=cu_seqlens[: num_requests + 1],
            num_accepted_tokens=num_accepted_tokens[:num_requests],
            state=self.kv_cache[1],
            output_gate=output_gate,
            norm_weight=self.norm.weight,
            out=core_attn_out,
            scale=self.head_k_dim**-0.5,
            norm_eps=self.layer_norm_epsilon,
        )

    def _forward_core_fused_norm_packed(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkvz[:, :qkv_size], 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        mixed_qkv, output_gate_flat = mixed_qkvz.split(
            [qkv_size, self.value_dim // self.tp_size], dim=-1
        )
        output_gate = output_gate_flat.reshape(
            output_gate_flat.size(0), -1, self.head_v_dim
        )
        b, a = self.split_ba(ba)
        self._forward_core_fused_norm(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            output_gate=output_gate,
            core_attn_out=core_attn_out,
        )

    def _can_use_fused_gdn_mtp_decode(
        self, attn_metadata: GDNAttentionMetadata
    ) -> bool:
        state_indices = attn_metadata.spec_state_indices_tensor
        return (
            attn_metadata.spec_sequence_masks is not None
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_spec_decodes > 0
            and self.kv_cache[1].dtype in FUSED_GDN_STATE_DTYPES
            and self.gdn_decode_kernel == "cuda"
            and self.num_v_heads % self.num_k_heads == 0
            and self.num_v_heads // self.num_k_heads in (1, 2, 3, 4, 8)
            and state_indices is not None
            and state_indices.size(1) <= MAX_FUSED_GDN_MTP_TOKENS
            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
        )

    def _rms_norm_gated_cuda(
        self,
        x: torch.Tensor,
        output_gate: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        x_shape = x.shape
        assert output_gate.shape == x_shape
        assert out.shape == x_shape
        x_2d = x.reshape(-1, x_shape[-1])
        output_gate_2d = output_gate.reshape(-1, x_shape[-1])
        out_2d = out.reshape(-1, x_shape[-1])
        assert x_2d.stride(-1) == 1
        assert output_gate_2d.stride(-1) == 1
        assert out_2d.stride(-1) == 1
        layer_norm_fwd(
            x_2d,
            self.norm.weight.contiguous(),
            self.norm.bias,
            self.norm.eps,
            z=output_gate_2d,
            out=out_2d,
            group_size=(
                x_shape[-1] if self.norm.group_size is None else self.norm.group_size
            ),
            norm_before_gate=self.norm.norm_before_gate,
            is_rms_norm=True,
            activation=self.norm.activation,
        )

    def _forward_core_fused_norm(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        output_gate: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if (
            self._can_use_fused_gdn_mtp_decode(attn_metadata)
            and attn_metadata.num_prefills == 0
        ):
            self._forward_core_decode_spec_fused_norm(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                output_gate=output_gate,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )
            return
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b.contiguous(),
            a=a.contiguous(),
            core_attn_out=core_attn_out,
        )
        num_actual_tokens = attn_metadata.num_actual_tokens
        self._rms_norm_gated_cuda(
            core_attn_out[:num_actual_tokens],
            output_gate[:num_actual_tokens],
            core_attn_out[:num_actual_tokens],
        )


def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``use_aiter=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``use_aiter=True`` (AITER Triton path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if use_aiter:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


def qwen_gdn_attention_core_fused_norm_packed(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core_fused_norm_packed(
        mixed_qkvz=mixed_qkvz,
        ba=ba,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fused_norm_packed_fake(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core_fused_norm_packed",
    op_func=qwen_gdn_attention_core_fused_norm_packed,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fused_norm_packed_fake,
)


def qwen_gdn_full_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Full Qwen GDN forward wrapped as a custom op.

    Prevents inductor from compiling the projections around the GDN
    recurrent core. Tiny fp16 differences in fused matmuls compound
    through the recurrent state and diverge logprobs under cudagraph
    replay. See OlmoHybridGatedDeltaNetAttention for the same rationale.

    Returns ``output`` so inductor cannot constant-fold the pre-op
    allocation through the split.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._full_forward(hidden_states=hidden_states, output=output)
    return output


def qwen_gdn_full_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Fake implementation for torch.compile."""
    return output


_gdn_full_forward_tags = ()
if hasattr(torch, "_C") and hasattr(torch._C, "Tag") and hasattr(
    torch._C.Tag, "cudagraph_unsafe"
):
    _gdn_full_forward_tags = (torch._C.Tag.cudagraph_unsafe,)

direct_register_custom_op(
    op_name="qwen_gdn_full_forward",
    op_func=qwen_gdn_full_forward,
    mutates_args=["output"],
    fake_impl=qwen_gdn_full_forward_fake,
    tags=_gdn_full_forward_tags,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
