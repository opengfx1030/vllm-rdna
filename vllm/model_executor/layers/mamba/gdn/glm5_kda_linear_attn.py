# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA (Kimi Delta Attention) linear-attention layer.

This module implements the KDA layer used by the 34 linear-attention layers
of GLM-5.3-Flash (``Glm5Next``, layers 0-44 except the 11 DSA layers
3, 7, ..., 43). It is the UNC-35 Python half: a complete, cudagraph-safe
**torch fallback (the DEFAULT path)** plus one isolated, env-gated hook that
dispatches to the UNC-35/36 HIP kernels when they are present.

Math source of truth
--------------------
``modeling_glm5_next.py`` (HF transformers reference):

* ``Glm5NextTextForgetGate.forward`` (lines 304-334) -- the log-decay gate
* ``Glm5NextTextRMSNormGated.forward`` (lines 338-358) -- gated output norm
* ``causal_conv1d_update`` / ``causal_conv1d_fn`` (lines 373-413) -- conv
* ``l2norm`` (lines 416-424)
* ``recurrent_kimi_delta_attention`` (lines 427-478) -- decode recurrence
* ``chunk_kimi_delta_attention`` (lines 481-578) -- chunked prefill
* ``Glm5NextTextLinearAttention.forward`` (lines 584-733) -- layer glue

Policy: torch/generic paths are the DEFAULT; the HIP path is OFF unless
``envs.VLLM_GLM5_KDA_HIP`` is set AND ``torch.ops._rocm_C`` exposes the
``glm5_kda_decode_rdna2`` op with the exact signature of DESIGN.md section
"K (KDA) ops" (decode op; the HIP prefill chain is UNC-36 and is NOT
dispatched here yet). If anything is missing or unsupported the layer falls
back to torch silently -- the hook is one method (``_maybe_run_hip_decode``)
and is trivially bypassed.

Plumbing follows ``vllm/models/kimi_k3/amd/kda.py`` /
``vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py``:
``MambaBase`` via ``GatedDeltaNetAttention``, states carried in the standard
mamba cache, per-step scheduling info read from ``GDNAttentionMetadata``.

State layout (MambaSpec, per layer, per sequence slot)
------------------------------------------------------
* conv state: ``[3 * qkv_dim / tp, conv_kernel - 1]`` model dtype -- ONE
  contiguous buffer holding three independent depthwise conv states in
  ``q block | k block | v block`` order (channel order of ``mixed_qkv``).
* ssm state: ``[num_heads / tp, head_dim, head_dim]`` fp32, ``state[k, v]``
  ordering per the HF recurrence.

Weight-loading contract (checkpoint name -> parameter in this module)
----------------------------------------------------------------------
* ``self_attn.{q,k,v}_proj.weight``      -> ``{q,k,v}_proj`` ColumnParallel
* ``self_attn.q_conv1d.weight`` [8192,1,4] -> ``conv1d`` fused param, shard 0
* ``self_attn.k_conv1d.weight`` [8192,1,4] -> ``conv1d`` fused param, shard 1
* ``self_attn.v_conv1d.weight`` [8192,1,4] -> ``conv1d`` fused param, shard 2
  (the loader also accepts shard ids ``"q"/"k"/"v"``)
* ``self_attn.A_log`` [64]               -> ``A_log`` (fp32, head-sharded)
* ``self_attn.dt_bias`` [8192]           -> ``dt_bias`` (fp32, dim-sharded)
* ``self_attn.b_proj.weight``            -> ``b_proj`` ColumnParallel
* ``self_attn.f_a_proj.weight``          -> ``f_a_proj`` Replicated
* ``self_attn.f_b_proj.weight``          -> ``f_b_proj`` ColumnParallel
* ``self_attn.g_a_proj.weight``          -> ``g_a_proj`` Replicated
* ``self_attn.g_b_proj.weight``          -> ``g_b_proj`` ColumnParallel
* ``self_attn.o_norm.weight`` [128]      -> ``o_norm.weight`` (fp32 param)
* ``self_attn.o_proj.weight``            -> ``o_proj`` RowParallel

Cudagraph notes
---------------
The decode path (the only path captured into FULL cudagraphs -- the GDN
builder only captures decode-only batches) is fully capture-safe: no
``.item()``, no device->host syncs, no data-dependent control flow, fixed
shapes (padded tokens map to the reserved null state slot, whose contents
are never read by real sequences). The prefill/mixed path is eager-only and
needs host-side segment boundaries; it performs the syncs under
``gpu_sync_allowed()`` (same pattern as ``mamba_mixer2.py``) and refuses to
run while a stream capture is active.
"""

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

# Generic KDA helpers shared with Kimi-Linear; neither ROCm- nor
# K3-specific, so imported rather than duplicated (same rationale as
# vllm/models/kimi_k3/amd/kda.py).
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KDA_GATE_LOGBOUND_MIN,
    a_log_weight_loader,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.selector import get_mamba_attn_backend

logger = init_logger(__name__)

# Fixed GLM-5.3-Flash KDA geometry (per DESIGN.md / config.json).
_GLM5_KDA_NUM_HEADS = 64
_GLM5_KDA_HEAD_DIM = 128
_GLM5_KDA_CONV_KERNEL = 4
_GLM5_KDA_CHUNK_SIZE = 64


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style l2norm: ``x / sqrt(sum(x^2) + eps)`` (NOT max(..., eps)).

    Exact port of ``l2norm`` in modeling_glm5_next.py (lines 416-424); must
    run on fp32 inputs, matching ``use_qk_l2norm_in_kernel=True`` semantics
    (the reference applies it after the fp32 cast).
    """
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def _glm5_kda_cfg(
    config: PretrainedConfig,
    attr: str,
    dict_key: str,
    default,
):
    """Read a KDA geometry value from the GLM text config.

    Prefers the flat attribute (``config.linear_*``); falls back to the raw
    ``linear_attn_config`` dict carried by config.json, then to the fixed
    GLM-5.3-Flash default.
    """
    value = getattr(config, attr, None)
    if value is not None:
        return value
    linear_attn_config = getattr(config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict):
        value = linear_attn_config.get(dict_key)
        if value is not None:
            return value
    return default


def _make_glm5_kda_conv1d_weight_loader(
    local_qkv_dim: int,
    tp_rank: int,
) -> Callable[..., None]:
    """Load GLM-5.3's three separate depthwise conv weights into one fused
    parameter.

    The checkpoint stores ``{q,k,v}_conv1d.weight`` each as
    ``[qkv_dim, 1, kernel]`` (separate convs, NOT fused). The fused
    parameter is laid out as ``q block | k block | v block`` along dim 0,
    matching the channel order of ``mixed_qkv = cat([q, k, v])`` and the
    conv-state buffer layout; each block is TP-sharded along the channel
    dim consistently with the q/k/v ColumnParallel projections.
    """
    shard_ids = {
        0: 0,
        1: 1,
        2: 2,
        "q": 0,
        "k": 1,
        "v": 2,
        "q_conv1d": 0,
        "k_conv1d": 1,
        "v_conv1d": 2,
    }

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ) -> None:
        if loaded_shard_id is None:
            raise ValueError(
                "GLM-5.3 KDA conv1d weight loading requires a shard id: "
                "0/'q' (q_conv1d), 1/'k' (k_conv1d) or 2/'v' (v_conv1d)."
            )
        shard = shard_ids[loaded_shard_id]
        if loaded_weight.dim() == 2:
            # [qkv_dim, kernel] -> [qkv_dim, 1, kernel]
            loaded_weight = loaded_weight.unsqueeze(1)
        loaded_shard = loaded_weight.narrow(
            0, tp_rank * local_qkv_dim, local_qkv_dim
        )
        target_start = shard * local_qkv_dim
        param.data[target_start : target_start + local_qkv_dim].copy_(
            loaded_shard
        )

    return weight_loader


def _chunk_kimi_delta_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = _GLM5_KDA_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact torch port of ``chunk_kimi_delta_attention``
    (modeling_glm5_next.py lines 481-578) with q/k l2norm applied in fp32
    (``use_qk_l2norm_in_kernel=True``).

    Structured as an explicit loop over chunks so the fp32 decay mask
    ``[B, H, chunk, chunk, Dk]`` is allocated per chunk instead of over the
    whole padded sequence (the reference materialization is
    ``O(H * S * chunk * Dk)`` and does not fit on RDNA2-class memory for
    long segments). Every formula and operation order is identical to the
    reference; chunks are independent for the intra-chunk math and the
    inter-chunk state pass is the reference's sequential scan verbatim.

    Args:
        query/key/value: ``[B, S, H, D]`` (any float dtype; math in fp32).
        g: ``[B, S, H, Dk]`` fp32 log-decay per (head, k-dim) -- the
            ForgetGate output, NOT yet exponentiated.
        beta: ``[B, S, H]`` fp32, sigmoid ALREADY applied.
        initial_state: ``[B, H, Dk, Dv]`` fp32 or None.

    Returns:
        ``(core_attn_out [B, S, H, Dv] in query's dtype,
        final_state [B, H, Dk, Dv] fp32)``.
    """
    initial_dtype = query.dtype
    device = query.device
    batch_size, sequence_length, num_heads, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)

    # Reference lines 497-499: transpose to [B, H, S, .] and cast fp32.
    query = query.transpose(1, 2).contiguous().to(torch.float32)
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    # Reference lines 501-504: FLA l2norm computed in fp32, after casting.
    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)

    state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=torch.float32,
            device=device,
        )
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    core_attn_out = torch.empty(
        batch_size,
        num_heads,
        sequence_length,
        v_head_dim,
        dtype=torch.float32,
        device=device,
    )

    mask = torch.triu(
        torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=device
        ),
        diagonal=0,
    )
    causal_mask = torch.triu(
        torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=device
        ),
        diagonal=1,
    )
    eye = torch.eye(chunk_size, dtype=torch.float32, device=device)

    num_chunks = (sequence_length + chunk_size - 1) // chunk_size
    for c in range(num_chunks):
        s0 = c * chunk_size
        s1 = min(s0 + chunk_size, sequence_length)
        pad_size = chunk_size - (s1 - s0)

        # Reference lines 514-518: pad the (last) partial chunk with zeros.
        q_i = F.pad(query[:, :, s0:s1], (0, 0, 0, pad_size)) * scale
        k_i = F.pad(key[:, :, s0:s1], (0, 0, 0, pad_size))
        v_i = F.pad(value[:, :, s0:s1], (0, 0, 0, pad_size))
        g_i = F.pad(g[:, :, s0:s1], (0, 0, 0, pad_size))
        b_i = F.pad(beta[:, :, s0:s1], (0, pad_size))

        # Reference lines 519-520.
        v_beta = v_i * b_i.unsqueeze(-1)
        k_beta = k_i * b_i.unsqueeze(-1)

        # Reference line 530: within-chunk cumsum of the log-decay.
        g_i = g_i.cumsum(dim=-2)

        # Reference lines 531-533: strictly-lower-triangular intra-chunk A.
        decay_mask = (g_i.unsqueeze(-2) - g_i.unsqueeze(-3)).exp().float()
        attn = -(
            k_beta.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask
        ).sum(dim=-1).masked_fill(mask, 0)

        # Reference lines 534-537: forward substitution for (I - A)^-1.
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

        # Reference lines 539-541.
        attn = attn + eye
        u = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g_i.exp())

        # Reference lines 551-569, single-chunk iteration.
        # Inter chunk:
        attn_inter = (q_i * g_i.exp()) @ state
        # Intra chunk:
        attn_intra = (
            q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask
        ).sum(dim=-1).masked_fill(causal_mask, 0)
        # Delta-rule update:
        v_prime = k_cumdecay @ state
        v_new = u - v_prime

        core_attn_out[:, :, s0:s1] = (
            attn_inter + attn_intra @ v_new
        )[:, :, : s1 - s0]
        state = state * g_i[:, :, -1].exp().unsqueeze(-1) + (
            k_i * (g_i[:, :, -1:] - g_i).exp()
        ).transpose(-1, -2) @ v_new

    # Reference lines 574-576: back to [B, S, H, Dv], trim padding.
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(
        initial_dtype
    )
    return core_attn_out, state


class Glm5NextRMSNormGated(nn.Module):
    """Strict-fp32 gated RMSNorm (``Glm5NextTextRMSNormGated``).

    Exact port of modeling_glm5_next.py lines 338-358:
    ``y = (w * RMSNorm_fp32(x)) * sigmoid(gate_fp32)``, cast back to the
    input dtype. The weight is stored fp32 ("do not downcast on the
    weights") which also matches the ``norm_w [D] fp32`` argument of the
    HIP decode op.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float32)
        )
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights).
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight * hidden_states

        # Apply gating (sigmoid activation per the GLM reference).
        hidden_states = hidden_states * torch.sigmoid(
            gate.to(torch.float32)
        )

        return hidden_states.to(input_dtype)


class Glm5NextKDAAttention(GatedDeltaNetAttention):
    """GLM-5.3-Flash KDA (Kimi Delta Attention) layer.

    Mirrors ``KimiK3DeltaAttention`` (vllm/models/kimi_k3/amd/kda.py) with
    the GLM parameterization: separate q/k/v projections, three separate
    depthwise causal convs carried in ONE contiguous conv-state buffer
    (q | k | v blocks), the GLM forget gate
    ``g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``
    per (head, k-dim), per-head ``beta = sigmoid(b_proj(x))``, and the
    gated output norm ``RMSNormGated(o, sigmoid(g_b(g_a(x))), o_norm)``.
    """

    def get_attn_backend(self) -> type[AttentionBackend]:
        # The stock GDN backend provides GDNAttentionMetadata (request
        # classification, state indices, has_initial_state), which is all
        # this layer consumes. Unlike kimi_k3/amd we do not need a custom
        # metadata variant: the torch fallback computes the chunked math
        # per sequence segment and never reads FLA chunk_indices.
        return get_mamba_attn_backend(self.mamba_type)

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv: (conv_kernel - 1, 3 * qkv_dim / tp) -- or dim-first under
        # VLLM_SSM_CONV_STATE_LAYOUT=DS; ssm: (num_heads / tp, 128, 128).
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_kernel,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        # ---- Fixed GLM-5.3-Flash KDA geometry -------------------------
        self.num_heads = int(
            _glm5_kda_cfg(
                config, "linear_num_heads", "num_heads", _GLM5_KDA_NUM_HEADS
            )
        )
        self.head_dim = int(
            _glm5_kda_cfg(
                config, "linear_head_dim", "head_dim", _GLM5_KDA_HEAD_DIM
            )
        )
        self.conv_kernel = int(
            _glm5_kda_cfg(
                config,
                "linear_conv_kernel_dim",
                "short_conv_kernel_size",
                _GLM5_KDA_CONV_KERNEL,
            )
        )
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        self.qkv_dim = self.num_heads * self.head_dim
        self.local_qkv_dim = divide(self.qkv_dim, self.tp_size)
        # Width of the fused q|k|v conv channel space (conv-state buffer).
        self.local_conv_dim = 3 * self.local_qkv_dim
        self.conv_state_len = self.conv_kernel - 1
        self.q_scale = self.head_dim**-0.5

        # The conv-state window and the spec-decode state slots are not
        # supported by the torch fallback (nor by the UNC-35 HIP decode
        # contract). Fail loudly at construction instead of faking it.
        if self.num_spec > 0:
            raise NotImplementedError(
                "GLM-5.3 KDA (torch fallback) does not support speculative "
                "decoding; run without --speculative-config."
            )

        # ---- Forget gate parameterization ------------------------------
        # GLM-5.3-Flash always sets linear_lower_bound = -5.0; the
        # softplus branch is kept for config compatibility only.
        self.gate_lower_bound = _glm5_kda_cfg(
            config, "linear_lower_bound", "gate_lower_bound", None
        )
        if self.gate_lower_bound is not None:
            assert (
                _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0
            ), (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )

        # ---- Projections ------------------------------------------------
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.qkv_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.qkv_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.qkv_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        # One fused depthwise causal conv parameter, layout [q | k | v]
        # blocks of local_qkv_dim channels each, matching mixed_qkv channel
        # order and the conv-state buffer. Stored fp32; the conv runs in
        # weight dtype and results are cast back (fp32-safe on RDNA2).
        self.conv1d = nn.Parameter(
            torch.empty(
                self.local_conv_dim,
                1,
                self.conv_kernel,
                dtype=torch.float32,
            )
        )
        set_weight_attrs(
            self.conv1d,
            {
                "weight_loader": _make_glm5_kda_conv1d_weight_loader(
                    self.local_qkv_dim, self.tp_rank
                )
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.dt_bias = nn.Parameter(
            torch.empty(self.local_qkv_dim, dtype=torch.float32)
        )
        set_weight_attrs(
            self.dt_bias, {"weight_loader": sharded_weight_loader(0)}
        )

        # beta = sigmoid(b_proj(x)), one value per head.
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )

        # Forget gate low-rank projections. f_a/g_a outputs feed f_b/g_b
        # whose input must be the full head_dim on every rank, so f_a/g_a
        # are replicated (same pattern as g_a_proj in the shared
        # KimiGatedDeltaNetAttention low-rank gate path).
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )

        self.o_norm = Glm5NextRMSNormGated(
            self.head_dim, eps=self.layer_norm_epsilon
        )
        set_weight_attrs(
            self.o_norm.weight, {"weight_loader": default_weight_loader}
        )
        self.o_proj = RowParallelLinear(
            self.qkv_dim,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Staging buffer for the HIP decode op's fp16 conv weight
        # (contract: conv_w [3*H*D, 4] fp16). Filled per decode call so it
        # stays consistent with the fp32 parameter under capture/replay.
        self.register_buffer(
            "_conv_w_model_dtype",
            torch.empty(
                self.local_conv_dim,
                self.conv_kernel,
                dtype=torch.float16,
            ),
            persistent=False,
        )

        if envs.VLLM_GLM5_KDA_HIP:
            logger.info_once(
                "GLM-5.3 KDA HIP decode hook enabled "
                "(VLLM_GLM5_KDA_HIP=1); falls back to torch when the "
                "glm5_kda_decode_rdna2 op is unavailable."
            )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # ------------------------------------------------------------------
    # Forget gate (Glm5NextTextForgetGate, modeling lines 304-334)
    # ------------------------------------------------------------------

    def _log_decay(self, f_raw: torch.Tensor) -> torch.Tensor:
        """Log-decay ``g`` of shape ``[N, H, D]`` (fp32), per (head, k-dim).

        ``f_raw`` is the forget pre-activation ``f_b(f_a(x))`` (before
        dt_bias); ``g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))``
        when the lower bound is set, else the softplus fallback branch.
        """
        num_tokens = f_raw.size(0)
        g = f_raw.float() + self.dt_bias.view(1, -1)
        g = g.view(num_tokens, self.local_num_heads, self.head_dim)
        decay_rate = torch.exp(self.A_log.float()).view(
            1, self.local_num_heads, 1
        )

        if self.gate_lower_bound is not None:
            # Safe lower-bound decay (always the GLM-5.3 case, -5.0).
            return self.gate_lower_bound * torch.sigmoid(decay_rate * g)

        # Softplus "log(1 + exp(x))" with upper-bound restraint to avoid
        # overflows (kept for config compatibility).
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
        return -decay_rate * g_softplus

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)

        # q/k/v and the gate/beta pre-activations. beta_raw and
        # out_gate_raw are kept PRE-sigmoid: the torch path applies the
        # sigmoid where the reference does, and the HIP decode op expects
        # the raw values (contract: beta/out_gate pre-sigmoid, f pre-bias).
        mixed_qkv = torch.cat(
            (
                self.q_proj(hidden_states)[0],
                self.k_proj(hidden_states)[0],
                self.v_proj(hidden_states)[0],
            ),
            dim=-1,
        )
        f_raw = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        beta_raw = self.b_proj(hidden_states)[0]
        out_gate_raw = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        # The core writes the gated-normed output here; o_proj runs in the
        # compiled region around the (possibly eager-broken) core.
        normed_out = torch.empty(
            num_tokens,
            self.local_qkv_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            f_raw=f_raw,
            beta_raw=beta_raw,
            out_gate_raw=out_gate_raw,
            normed_out=normed_out,
        )
        output[:] = self.o_proj(normed_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        f_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        out_gate_raw: torch.Tensor,
        normed_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        m = attn_metadata_raw[self.prefix]
        assert isinstance(m, GDNAttentionMetadata)

        if m.spec_sequence_masks is not None:
            raise NotImplementedError(
                "GLM-5.3 KDA does not support multi-query speculative "
                "decode batches; disable speculative decoding."
            )

        num_actual_tokens = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        f_raw = f_raw[:num_actual_tokens]
        beta_raw = beta_raw[:num_actual_tokens]
        # out_gate_raw keeps full width: the gated norm runs over all
        # num_tokens rows so FULL-graph padded batches keep static shapes.

        conv_state, recurrent_state = self.kv_cache
        # Work in dim-first [slots, dim, state_len]: the conv kernels and
        # the HIP op contract both want that layout. SD layout stores
        # (state_len, dim), so a transpose view is needed.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        # Optional HIP fused decode (env-gated, isolated, default OFF).
        if self._maybe_run_hip_decode(
            mixed_qkv,
            f_raw,
            beta_raw,
            out_gate_raw,
            normed_out,
            m,
            conv_state,
            recurrent_state,
        ):
            return

        # ---- Torch fallback (DEFAULT path) ------------------------------
        num_tokens = normed_out.size(0)
        core_attn_out = torch.zeros(
            num_tokens,
            self.local_num_heads,
            self.head_dim,
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )

        if m.num_prefills > 0:
            self._prefill_batch_torch(
                mixed_qkv,
                f_raw,
                beta_raw,
                core_attn_out,
                m,
                conv_state,
                recurrent_state,
            )
        else:
            # Pure decode: one token per sequence, per-token state index
            # (padded FULL-graph tokens carry the null slot id, which is a
            # reserved scratch slot no real sequence ever occupies).
            assert m.non_spec_state_indices_tensor is not None
            state_indices = m.non_spec_state_indices_tensor[
                : mixed_qkv.size(0)
            ]
            self._decode_batch_torch(
                mixed_qkv,
                f_raw,
                beta_raw,
                core_attn_out,
                state_indices,
                conv_state,
                recurrent_state,
            )

        # Gated output norm (RMSNormGated with sigmoid gate), then flatten
        # [N, H, D] -> [N, H*D] for o_proj in the caller.
        gate = out_gate_raw.view(
            num_tokens, self.local_num_heads, self.head_dim
        )
        normed_out[:num_tokens] = self.o_norm(
            core_attn_out, gate
        ).view(num_tokens, -1)

    # ------------------------------------------------------------------
    # HIP hook (UNC-35/36 decode op; single isolated method)
    # ------------------------------------------------------------------

    def _maybe_run_hip_decode(
        self,
        mixed_qkv: torch.Tensor,
        f_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        out_gate_raw: torch.Tensor,
        normed_out: torch.Tensor,
        m: GDNAttentionMetadata,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> bool:
        """Dispatch the fused HIP decode kernel when available.

        Calls ``torch.ops._rocm_C.glm5_kda_decode_rdna2`` with the exact
        DESIGN.md "K (KDA) ops" signature (conv shift+silu, l2norm,
        lower-bound gate, delta recurrence and gated RMSNorm in one
        kernel). Returns True when the kernel handled the batch; False
        sends the caller to the torch fallback. Default OFF.
        """
        if not envs.VLLM_GLM5_KDA_HIP:
            return False
        # The kernel contract implements only the lower-bound sigmoid gate.
        if self.gate_lower_bound is None:
            return False
        # Decode-only batches; prefill uses the torch chunked math (the
        # HIP prefill chain is UNC-36 and not wired here yet).
        if m.num_prefills > 0 or m.num_decodes == 0:
            return False
        # Contract dtypes: fp16 activations/conv state, fp32 A_log/dt_bias
        # /norm weight/ssm state.
        if mixed_qkv.dtype != torch.float16:
            return False
        if conv_state.dtype != torch.float16:
            return False
        # The kernel reads conv_state as contiguous [slots, dim, 3].
        if not is_conv_state_dim_first():
            return False
        if m.non_spec_state_indices_tensor is None:
            return False
        if not hasattr(torch.ops._rocm_C, "glm5_kda_decode_rdna2"):
            return False

        num_tokens = mixed_qkv.size(0)
        state_indices = m.non_spec_state_indices_tensor[:num_tokens]
        # fp32 parameter -> fp16 staging weight (D2D copy, capture-safe).
        self._conv_w_model_dtype.copy_(self.conv1d.squeeze(1))

        torch.ops._rocm_C.glm5_kda_decode_rdna2(
            mixed_qkv,  # [B, 3*H*D] fp16 pre-conv q|k|v
            self._conv_w_model_dtype,  # [3*H*D, 4] fp16 (q|k|v order)
            conv_state,  # [slots, 3*H*D, 3] fp16, updated in place
            self.A_log,  # [H] fp32
            self.dt_bias,  # [H*D] fp32
            f_raw,  # [B, H*D] fp16 forget pre-activation (pre-bias)
            beta_raw,  # [B, H] fp16 pre-sigmoid b_proj(x)
            out_gate_raw[:num_tokens],  # [B, H*D] fp16 pre-sigmoid
            self.o_norm.weight,  # [D] fp32 o_norm weight
            state_indices,  # [B] int32
            recurrent_state,  # [slots, H, D, D] fp32, updated in place
            normed_out[:num_tokens],  # [B, H*D] fp16 out
            float(self.gate_lower_bound),
            float(self.layer_norm_epsilon),
        )
        return True

    # ------------------------------------------------------------------
    # Torch fallback kernels
    # ------------------------------------------------------------------

    def _decode_batch_torch(
        self,
        mixed_qkv: torch.Tensor,
        f_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        core_attn_out: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        """Recurrent decode step: one token per sequence.

        Exact port of ``causal_conv1d_update`` + one step of
        ``recurrent_kimi_delta_attention`` (with
        ``use_qk_l2norm_in_kernel=True``), vectorized across the batch.
        Fully cudagraph-capture-safe (no syncs, fixed shapes; padded
        tokens map to the reserved null state slot).
        """
        num_tokens = mixed_qkv.size(0)
        if num_tokens == 0:
            return

        # ---- Causal depthwise conv, shift-register update --------------
        # conv_state is dim-first: [slots, dim, state_len].
        conv_slice = conv_state[state_indices]  # [B, dim, K-1]
        x = torch.cat(
            [conv_slice.to(mixed_qkv.dtype), mixed_qkv.unsqueeze(-1)],
            dim=-1,
        )  # [B, dim, K]
        conv_state[state_indices] = x[..., -self.conv_state_len :]
        conv_w = self.conv1d  # fp32 [dim, 1, K]
        conv_out = F.conv1d(
            x.to(conv_w.dtype), conv_w, padding=0, groups=self.local_conv_dim
        )  # [B, dim, 1]
        qkv = F.silu(conv_out).to(mixed_qkv.dtype).squeeze(-1)

        query, key, value = qkv.split(self.local_qkv_dim, dim=-1)
        query = query.view(num_tokens, self.local_num_heads, self.head_dim)
        key = key.view(num_tokens, self.local_num_heads, self.head_dim)
        value = value.view(num_tokens, self.local_num_heads, self.head_dim)

        # ---- Gates ------------------------------------------------------
        g = self._log_decay(f_raw)  # [B, H, D] fp32 log-decay
        beta = torch.sigmoid(beta_raw.float())  # [B, H]

        # ---- Recurrence (fp32; reference lines 464-476) -----------------
        query = _l2norm(query.float(), dim=-1, eps=1e-6) * self.q_scale
        key = _l2norm(key.float(), dim=-1, eps=1e-6)
        value = value.float()

        state = recurrent_state[state_indices]  # [B, H, Dk, Dv] fp32
        state = state * g.exp().unsqueeze(-1)
        kv_mem = (state * key.unsqueeze(-1)).sum(dim=-2)
        delta = (value - kv_mem) * beta.unsqueeze(-1)
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn = (state * query.unsqueeze(-1)).sum(dim=-2)
        recurrent_state[state_indices] = state

        core_attn_out[:num_tokens] = core_attn.to(core_attn_out.dtype)

    def _prefill_batch_torch(
        self,
        mixed_qkv: torch.Tensor,
        f_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        core_attn_out: torch.Tensor,
        m: GDNAttentionMetadata,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        """Prefill / mixed batch: per-sequence segments via the attention
        metadata, chunked delta-rule math exactly matching
        ``chunk_kimi_delta_attention``.

        Mixed non-spec batches are decode-first (see
        ``GDNAttentionMetadataBuilder``): the length-1 decode front is
        peeled to the recurrent update and the prefill tail runs through
        the chunked kernel with the rebased ``prefill_*`` metadata.

        Eager-only: needs host-side segment boundaries (one batched
        device->host transfer under ``gpu_sync_allowed()``, same pattern
        as mamba_mixer2). Refuses to run under an active stream capture.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GLM-5.3 KDA torch-fallback prefill cannot run inside a "
                "CUDA graph capture (it needs host-side segment bounds). "
                "Only decode batches are capturable."
            )

        # ---- Peel the decode front of a mixed batch ---------------------
        if m.num_decodes > 0:
            num_decode_tokens = m.num_decode_tokens
            assert m.non_spec_state_indices_tensor is not None
            decode_indices = m.non_spec_state_indices_tensor[: m.num_decodes]
            self._decode_batch_torch(
                mixed_qkv[:num_decode_tokens],
                f_raw[:num_decode_tokens],
                beta_raw[:num_decode_tokens],
                core_attn_out,
                decode_indices,
                conv_state,
                recurrent_state,
            )
            mixed_qkv = mixed_qkv[num_decode_tokens:]
            f_raw = f_raw[num_decode_tokens:]
            beta_raw = beta_raw[num_decode_tokens:]
            prefill_out = core_attn_out[num_decode_tokens:]
            query_start_loc = m.prefill_query_start_loc
            state_indices = m.prefill_state_indices
            has_initial_state = m.prefill_has_initial_state
        else:
            prefill_out = core_attn_out
            query_start_loc = m.non_spec_query_start_loc
            state_indices = m.non_spec_state_indices_tensor
            has_initial_state = m.has_initial_state

        assert query_start_loc is not None
        assert state_indices is not None
        assert has_initial_state is not None

        # ---- Host-side segment boundaries (one sync, allowed) ----------
        with gpu_sync_allowed():
            query_start_loc_cpu = query_start_loc.tolist()
            state_indices_cpu = state_indices.tolist()
            has_initial_state_cpu = has_initial_state.tolist()

        zero_prefix = torch.zeros(
            self.local_conv_dim,
            self.conv_state_len,
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        conv_w = self.conv1d  # fp32 [dim, 1, K]

        for i in range(m.num_prefills):
            start = query_start_loc_cpu[i]
            end = query_start_loc_cpu[i + 1]
            seg_len = end - start
            if seg_len == 0:
                continue
            slot = state_indices_cpu[i]
            has_state = has_initial_state_cpu[i]

            seg_qkv = mixed_qkv[start:end]  # [L, dim]
            seg_f = f_raw[start:end]
            seg_beta = beta_raw[start:end]

            # ---- Causal conv over [prev state | segment] ----------------
            # Equivalent to HF update_conv_state + causal_conv1d_fn + tail
            # trim: prepending the last (K-1) raw values makes the
            # no-padding conv over the concatenation produce exactly the
            # causal outputs for the segment.
            prefix = (
                conv_state[slot].to(mixed_qkv.dtype)
                if has_state
                else zero_prefix
            )
            xs = torch.cat(
                [prefix, seg_qkv.transpose(0, 1)], dim=-1
            ).unsqueeze(0)  # [1, dim, K-1+L]
            conv_state[slot] = xs[:, :, -self.conv_state_len :]
            conv_out = F.conv1d(
                xs.to(conv_w.dtype),
                conv_w,
                padding=0,
                groups=self.local_conv_dim,
            )  # [1, dim, L]
            qkv = F.silu(conv_out).to(mixed_qkv.dtype).squeeze(0)
            qkv = qkv.transpose(0, 1)  # [L, dim]

            query, key, value = qkv.split(self.local_qkv_dim, dim=-1)
            query = query.view(1, seg_len, self.local_num_heads, self.head_dim)
            key = key.view(1, seg_len, self.local_num_heads, self.head_dim)
            value = value.view(1, seg_len, self.local_num_heads, self.head_dim)

            g = self._log_decay(seg_f)  # [L, H, D] fp32 log-decay
            beta = torch.sigmoid(seg_beta.float())  # [L, H]

            initial_state = (
                recurrent_state[slot].unsqueeze(0) if has_state else None
            )
            core_attn, final_state = _chunk_kimi_delta_attention_torch(
                query,
                key,
                value,
                g.unsqueeze(0),
                beta.unsqueeze(0),
                initial_state,
                chunk_size=_GLM5_KDA_CHUNK_SIZE,
            )
            recurrent_state[slot] = final_state.squeeze(0)
            prefill_out[start:end] = core_attn.squeeze(0).to(
                core_attn_out.dtype
            )
