# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel GDN (gfx1030): the GDN decode core in one launch.

Conv update + gated delta rule + gated RMSNorm of a Qwen4Exp linear-attention
layer, between in_proj and out_proj, as one ``_rocm_C::mono_gdn_rdna2`` launch
(``csrc/rocm/mono_gdn_rdna2.cu``) for pure decode and pure MTP-verify batches
of at most ``gdn_max_rows`` rows; today's split / core / gated-norm path for
anything else, inside the same op. ``vllm::qwen4_exp_mono_gdn_rdna2`` is a
splitting op like the GDN core: an eager break in FULL graphs, or recorded in
the graph when ``gdn_graph`` is on and the batch is capturable.

Installed into the shared GDN layer through ``qwen_gdn_linear_attn.MONO_GDN_HOOK``.
"""

import functools
import os
from typing import cast

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from . import OPTS, offload_trace

logger = init_logger(__name__)

_LOGGED: set = set()


def unsupported(layer) -> str | None:
    """Why this GDN layer cannot take the fused core (None: it can)."""
    if layer.gqa_interleaved_layout or layer.disable_tp_for_ba_proj:
        return "interleaved qkvz / replicated ba layout"
    if layer.head_k_dim != 128 or layer.head_v_dim != 128:
        return "head dims != 128"
    if layer.conv_kernel_size != 4 or layer.activation not in ("silu", "swish"):
        return "conv width / activation"
    if layer.norm.activation not in ("silu", "swish", "sigmoid"):
        return "norm gate activation"
    return None


def install_hook(layer, vllm_config):
    """``MONO_GDN_HOOK``: (op, MonoGDN) for a supported layer, else None."""
    why = unsupported(layer)
    if why is not None:
        logger.info_once("mono gdn_rdna2: layer unsupported (%s), today's path", why)
        return None
    logger.info_once("mono gdn_rdna2: fused GDN decode core enabled")
    return torch.ops.vllm.qwen4_exp_mono_gdn_rdna2, MonoGDN(layer)


class MonoGDN:
    """Per-layer state of the fused GDN decode core."""

    def __init__(self, layer) -> None:
        self.layer = layer
        self.cache: tuple | None = None

    def setup(self, device: torch.device) -> tuple:
        """fp32 A_log / dt_bias, 2-D conv weight and a zeroed scratch, built on
        the first (eager, profile-run) call so nothing allocates under capture."""
        if self.cache is None:
            layer = self.layer
            H = layer.num_k_heads // layer.tp_size
            HV = layer.num_v_heads // layer.tp_size
            nbytes = int(torch.ops._rocm_C.mono_gdn_rdna2_scratch_bytes(64, HV, H))
            conv_w = layer.conv1d.weight.view(
                layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
            )
            self.cache = (
                layer.A_log.detach().float().contiguous(),
                layer.dt_bias.detach().float().contiguous(),
                conv_w.contiguous() if not conv_w.is_contiguous() else conv_w,
                layer.conv1d.bias
                if layer.conv1d.bias is not None
                else torch.empty(0, dtype=torch.float16, device=device),
                torch.zeros(nbytes, dtype=torch.uint8, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
                H,
                1 if layer.norm.activation == "sigmoid" else 0,
            )
            if os.environ.get("VLLM_RDNA_GDN_DIAGNOSTICS") == "1":
                logger.info(
                    "RDNA_GDN_DIAG layer=%s scratch=%s bytes=%d "
                    "kinds=1:bad_nacc,2:prior_conv,3:input,4:prior_ssm,"
                    "5:gating_input,6:conv_output,7:recurrent_fp32,"
                    "8:fp16_overflow,9:norm_output",
                    layer.prefix,
                    hex(self.cache[4].data_ptr()),
                    nbytes,
                )
        return self.cache

    def eligible(self, md: GDNAttentionMetadata, qkvz: torch.Tensor) -> bool:
        layer = self.layer
        if md.num_prefills > 0 or md.num_actual_tokens > OPTS.gdn_max_rows:
            return False
        if (
            qkvz.dtype != torch.float16
            or layer.kv_cache[0].dtype != torch.float16
            or layer.kv_cache[1].dtype != torch.float32
            or layer.norm.weight.dtype != torch.float16
            or layer.conv1d.weight.dtype != torch.float16
        ):
            return False
        if md.spec_sequence_masks is None:  # pure non-spec decode
            idx = md.non_spec_state_indices_tensor
            return md.num_decodes > 0 and idx is not None and idx.dtype == torch.int32
        # pure MTP verify
        idx = md.spec_state_indices_tensor
        return (
            md.num_decodes == 0
            and md.num_spec_decodes > 0
            and idx is not None
            and idx.dtype == torch.int32
            and idx.size(1) <= 8
            and md.num_accepted_tokens is not None
            and md.spec_query_start_loc is not None
        )

    def capturable(self, md: GDNAttentionMetadata, qkvz: torch.Tensor) -> bool:
        """The fused path may be recorded into a FULL graph: every replay of it
        is a same-shape pure decode / verify batch read from the builder's
        static buffers (padding rows carry NULL_BLOCK_ID and are skipped)."""
        if not self.eligible(md, qkvz):
            return False
        if md.spec_sequence_masks is None and md.use_state_arenas:
            # the arena gather / scatter is recorded with this step's count
            return md.num_decodes == qkvz.shape[0]
        return True

    def forward_core(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        y: torch.Tensor,
        graph: bool = False,
    ) -> None:
        """Y = gated-RMSNorm(core(qkvz, ba)) for out_proj: one mono_gdn_rdna2
        launch for pure decode / pure verify batches, today's path otherwise."""
        layer = self.layer
        cache = self.setup(mixed_qkvz.device)
        raw = get_forward_context().attn_metadata
        md = cast(
            GDNAttentionMetadata | None,
            raw.get(layer.prefix) if isinstance(raw, dict) else None,
        )
        use = md is not None and self.eligible(md, mixed_qkvz)
        if md is not None:
            # one line per (kind, width) for decode-sized batches, one for all
            # prefill / mixed batches
            rows = md.num_actual_tokens if md.num_prefills == 0 else -1
            key = (md.spec_sequence_masks is not None, rows, use)
            if key not in _LOGGED:
                _LOGGED.add(key)
                logger.info(
                    "mono gdn_rdna2: %s %s -> %s",
                    "verify" if key[0] else "decode",
                    f"{rows} rows" if rows >= 0 else "with prefill",
                    "fused kernel" if use else "today's path",
                )
        if use:
            assert md is not None
            alog, dtb, conv_w, conv_b, scratch, empty, H, sig = cache
            # in a FULL graph the padded batch replays: size by the buffers
            M = mixed_qkvz.shape[0] if graph else md.num_actual_tokens
            conv_state = (
                layer.kv_cache[0]
                if is_conv_state_dim_first()
                else layer.kv_cache[0].transpose(-1, -2)
            )
            ssm_state = layer.kv_cache[1]
            paged = (conv_state, ssm_state)
            conv_state, ssm_state = layer._bind_decode_state_arenas(
                conv_state, ssm_state, md, md.num_decodes
            )
            if md.spec_sequence_masks is None:
                assert md.non_spec_state_indices_tensor is not None
                sidx, nacc, qsl = md.non_spec_state_indices_tensor[:M], empty, empty
            else:
                assert md.spec_state_indices_tensor is not None
                assert md.num_accepted_tokens is not None
                assert md.spec_query_start_loc is not None
                n = (
                    md.spec_state_indices_tensor.shape[0]
                    if graph
                    else md.num_spec_decodes
                )
                sidx = md.spec_state_indices_tensor[:n]
                nacc = md.num_accepted_tokens[:n]
                qsl = md.spec_query_start_loc[: n + 1]
            trace = None
            if offload_trace.ENABLED:
                from vllm.compilation.breakable_cudagraph import (
                    BreakableCUDAGraphCapture,
                )

                trace = offload_trace.begin_gdn(
                    layer,
                    md,
                    mixed_qkvz[:M],
                    ba[:M],
                    conv_state,
                    ssm_state,
                    sidx,
                    BreakableCUDAGraphCapture.current(),
                )
            torch.ops._rocm_C.mono_gdn_rdna2(
                mixed_qkvz[:M],
                ba[:M],
                conv_w,
                conv_b,
                conv_state,
                ssm_state,
                alog,
                dtb,
                layer.norm.weight,
                sidx,
                nacc,
                qsl,
                y[:M],
                scratch,
                H,
                layer.head_k_dim**-0.5,
                layer.norm.eps,
                sig,
            )
            if trace is not None:
                offload_trace.finish_gdn(trace, y[:M], conv_state, ssm_state)
            layer._unbind_decode_state_arenas(*paged, md, md.num_decodes)
            if y.size(0) > M:
                y[M:].zero_()
            return
        # today's path: split, core, gated norm
        qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
        mixed_qkv, z = mixed_qkvz.split(
            [qkv_size, layer.value_dim // layer.tp_size], dim=-1
        )
        z = z.reshape(z.size(0), -1, layer.head_v_dim)
        b, a = layer.split_ba(ba)
        core = torch.zeros(
            (mixed_qkvz.size(0), layer.num_v_heads // layer.tp_size, layer.head_v_dim),
            dtype=mixed_qkvz.dtype,
            device=mixed_qkvz.device,
        )
        layer._forward_core(
            mixed_qkv=mixed_qkv, b=b.contiguous(), a=a.contiguous(), core_attn_out=core
        )
        y.copy_(layer.norm(core, z).reshape(y.shape))


@eager_break_during_capture
def _core_eager_break(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    y: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Conv + recurrent attention + gated norm into ``y`` (eager in graphs)."""
    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._mono_gdn.forward_core(mixed_qkvz, ba, y)


def _core(
    mixed_qkvz: torch.Tensor,
    ba: torch.Tensor,
    y: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """``vllm::qwen4_exp_mono_gdn_rdna2``. Default: an eager break inside FULL
    captures. With ``gdn_graph``, a capturable batch records the fused kernel
    into the FULL graph instead, anything else breaks as usual."""
    if not OPTS.gdn_graph:
        _core_eager_break(mixed_qkvz, ba, y, layer_name)
        return
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    mono = layer._mono_gdn
    capture = BreakableCUDAGraphCapture.current()
    if capture is not None and capture._capturing:
        raw = get_forward_context().attn_metadata
        md = cast(
            GDNAttentionMetadata | None,
            raw.get(layer.prefix) if isinstance(raw, dict) else None,
        )
        if md is not None and mono.capturable(md, mixed_qkvz):
            key = ("graph", md.spec_sequence_masks is not None, mixed_qkvz.shape[0])
            if key not in _LOGGED:
                _LOGGED.add(key)
                logger.info(
                    "mono gdn_rdna2: %s %d rows captured in the FULL graph",
                    "verify" if key[1] else "decode",
                    key[2],
                )
            mono.forward_core(mixed_qkvz, ba, y, graph=True)
            return
        capture.add_eager(functools.partial(mono.forward_core, mixed_qkvz, ba, y))
        return
    mono.forward_core(mixed_qkvz, ba, y)


direct_register_custom_op(
    op_name="qwen4_exp_mono_gdn_rdna2",
    op_func=_core,
    mutates_args=["mixed_qkvz", "y"],
)
