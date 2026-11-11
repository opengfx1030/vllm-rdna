# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel K2 (gfx1030): one persistent launch for a decoder layer's FFN.

It covers what runs between the attention all-reduce and the FFN all-reduce:
the MLP hyper-connection combine + mix, router (fp32 logits and top-k), shared
expert and the routed symmetric W4A16 experts (``_rocm_C::mono_k2_rdna2``,
``csrc/rocm/mono_k2_rdna2.cu``). The all-reduce stays outside unless
``k2_ar`` puts it inside on the rdna_ar buffers.

The FFN block is per token, so the kernel is valid for any batch of at most
``k2_max_rows`` rows (<= 24): FULL and PIECEWISE CUDA graph captures and eager
steps alike (``k2_modes``, default all three). Larger batches (prefill chunks)
and unsupported layers run today's modules. The MTP drafter can opt in
separately and uses its own row cap and workspace.
The decision is made at run time inside one opaque custom op
(``vllm::qwen4_exp_mono_k2_rdna2``), so torch.compile sees a single op and the
branch follows the actual batch, not the trace-time size.

Fail-stop: a timed-out grid barrier leaves a sticky record; later launches
exit at once and fill the FFN output with NaN instead of returning stale data.
"""

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from . import OPTS, debug
from .layout import (
    LOGGED,
    MAX_ROWS,
    MODES,
    MonoLayer,
    ar_fused,
    mode,
)

logger = init_logger(__name__)


def run(mono: MonoLayer, hidden_states, attn_out, injection):
    """K2 on the persistent kernel; (residual, partial FFN output, injection)."""
    t = mono.t
    m = hidden_states.shape[0]
    res_out = torch.empty_like(hidden_states)
    inj_out = torch.empty(
        (m, 4), dtype=hidden_states.dtype, device=hidden_states.device
    )
    out = torch.empty_like(attn_out)
    if mono.dense_mtp and not getattr(mono, "_dense_call_logged", False):
        mono._dense_call_logged = True
        tensors = {
            "res": hidden_states,
            "blk": attn_out,
            "inj": injection,
            "res_out": res_out,
            "inj_out": inj_out,
            "out": out,
            "workspace": mono.ws,
            **t,
        }
        logger.info(
            "dense MTP K2 call M=%d expert_base=%d tensors=%s",
            m,
            mono.base,
            {
                name: {
                    "ptr": hex(x.data_ptr()),
                    "bytes": x.numel() * x.element_size(),
                    "shape": tuple(x.shape),
                    "stride": tuple(x.stride()),
                    "contiguous": x.is_contiguous(),
                    "dtype": str(x.dtype),
                    "device": str(x.device),
                }
                for name, x in tensors.items()
            },
        )
    torch.ops._rocm_C.mono_k2_rdna2(
        hidden_states,
        attn_out,
        injection.contiguous(),
        t["norm_w"],
        t["wd"],
        t["wu"],
        t["wr"],
        t["se_w1"],
        t["se_w2"],
        t["se_wg"],
        t["w13"],
        t["s13"],
        t["z13"],
        t["w2"],
        t["s2"],
        t["z2"],
        res_out,
        inj_out,
        out,
        mono.ws,
        mono.base,
        1,
        OPTS.k2_num_wg,
        mono.eps,
        2000.0,
        mono.is_mtp,
        mono.dense_mtp,
        getattr(mono, "preserve_fault", False),
    )
    return res_out, out, inj_out


def _k2(
    hidden_states: torch.Tensor,
    attn_out: torch.Tensor,
    injection: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.forward_context import get_forward_context

    mono = get_forward_context().no_compile_layers[layer_name]
    layer = mono.layer
    m = hidden_states.shape[0]
    if not torch.cuda.is_current_stream_capturing():
        # Resolve weights / allocate the workspace on an eager step (profile
        # and warmup runs precede capture); nothing here may run in a capture.
        mono.setup()
    md = mode()
    fits = 0 < m <= MAX_ROWS and hidden_states.dtype == torch.float16 and mono.ready
    use = (
        fits and OPTS.k2 and not OPTS.modules and m <= mono.k2_max_rows and md in MODES
    )
    key = (m, md, use)
    if key not in LOGGED:
        LOGGED.add(key)
        logger.info(
            "mono k2_rdna2: %d rows (%s) -> %s",
            m,
            md,
            "mono kernel" if use else "modules",
        )
    timed = OPTS.timing and fits and md != "eager" and not mono.is_mtp
    st = debug.STATS.get(hidden_states.device.index) if timed else None
    if OPTS.timing and not mono.is_mtp and not torch.cuda.is_current_stream_capturing():
        debug.log_stats()
    if st is not None:
        mark = torch.ops._rocm_C.mono_k2_rdna2_mark
        qsa = getattr(layer, "layer_type", "") != "linear_attention"
        mark(st, 5, m | (32 if qsa else 0))  # attention done
        mark(st, 0 if mono.first else 2, m)
    if use:
        from vllm.distributed import tensor_model_parallel_all_reduce

        res_out, partial, inj_out = run(mono, hidden_states, attn_out, injection)
        if ar_fused(partial):  # the kernel already all-reduced it
            result = res_out, partial, inj_out
        else:
            result = res_out, tensor_model_parallel_all_reduce(partial), inj_out
    else:
        hs, block_input, inj = layer.mlp_hyper_connection.combine_and_mix(
            hidden_states, attn_out, injection
        )
        result = hs, layer.mlp(block_input), inj
    if st is not None:
        torch.ops._rocm_C.mono_k2_rdna2_mark(st, 1, m)
    return result


def _k2_fake(hidden_states, attn_out, injection, layer_name):
    m = hidden_states.shape[0]
    return (
        torch.empty_like(hidden_states),
        torch.empty_like(attn_out),
        hidden_states.new_empty((m, 4)),
    )


direct_register_custom_op(
    op_name="qwen4_exp_mono_k2_rdna2",
    op_func=_k2,
    fake_impl=_k2_fake,
)
