# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel K1 (gfx1030): the attention-side hyper-connection preamble.

Today's ``attn_hyper_connection.combine_and_mix`` / ``mix`` (combine the
previous block's output, grouped RMSNorm, HC down+inject, HC up-mix) as one
launch of the persistent kernel (``_rocm_C::mono_k1_rdna2``, in
``csrc/rocm/mono_k2_rdna2.cu``) for batches of at most ``k1_max_rows`` rows;
today's modules otherwise, inside the same opaque op
``vllm::qwen4_exp_mono_k1_rdna2``.
"""

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from . import OPTS, debug
from .layout import LOGGED, MODES, ZEROS, MonoLayer, mode

logger = init_logger(__name__)


def run(mono: MonoLayer, hidden_states, prev_out, prev_inj):
    """K1 on the persistent kernel; (residual, block input, injection)."""
    t = mono.th
    hidden_states = hidden_states.contiguous()
    m, hch = hidden_states.shape
    h = hch // 4
    if prev_out is None or prev_inj is None:  # mix(): combine with a zero output
        zo, zi = ZEROS[hidden_states.device.index]
        prev_out, prev_inj = zo[:m], zi[:m]
    res_out = hidden_states.new_empty(hidden_states.shape)
    inj_out = torch.empty(
        (m, 4), dtype=hidden_states.dtype, device=hidden_states.device
    )
    out = torch.empty((m, h), dtype=hidden_states.dtype, device=hidden_states.device)
    torch.ops._rocm_C.mono_k1_rdna2(
        hidden_states,
        prev_out.contiguous(),
        prev_inj.contiguous(),
        t["norm_w"],
        t["wd"],
        t["wu"],
        res_out,
        inj_out,
        out,
        mono.ws,
        OPTS.k2_num_wg,
        mono.eps_a,
        2000.0,
        mono.is_mtp,
        getattr(mono, "preserve_fault", False),
    )
    return res_out, out, inj_out


def _k1(
    hidden_states: torch.Tensor,
    prev_out: torch.Tensor | None,
    prev_inj: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention-side HC preamble: K1 when the batch fits, else today's
    combine_and_mix / mix; timing marks 3 and 4 around it."""
    from vllm.forward_context import get_forward_context

    mono = get_forward_context().no_compile_layers[layer_name]
    hc = mono.layer.attn_hyper_connection
    m = hidden_states.shape[0]
    if not torch.cuda.is_current_stream_capturing():
        if mono.is_mtp:
            mono.setup_k1()
        else:
            mono.setup()
    md = mode()
    layout_ready = mono.k1_ready if mono.is_mtp else mono.ready
    use = (
        OPTS.k1
        and layout_ready
        and mono.hc_ok
        and 0 < m <= mono.k1_max_rows
        and hidden_states.dtype == torch.float16
        and md in MODES
    )
    key = ("mtp-hc" if mono.is_mtp else "hc", m, md, use)
    if key not in LOGGED:
        LOGGED.add(key)
        logger.info(
            "mono %sk1_rdna2: %d rows (%s) -> %s",
            "MTP " if mono.is_mtp else "",
            m,
            md,
            "K1 kernel" if use else "modules",
        )
    if not mono.is_mtp:
        debug.mark(3, hidden_states)
    if use:
        out = run(mono, hidden_states, prev_out, prev_inj)
    elif prev_out is not None and prev_inj is not None:
        out = hc.combine_and_mix(hidden_states, prev_out, prev_inj)
    else:
        out = hc.mix(hidden_states)
    if not mono.is_mtp:
        debug.mark(4, hidden_states)
    # a custom op may not return its inputs (mix() hands back hidden_states)
    ins = [t for t in (hidden_states, prev_out, prev_inj) if t is not None]
    return tuple(
        o.clone() if any(o.data_ptr() == t.data_ptr() for t in ins) else o for o in out
    )


def _k1_fake(hidden_states, prev_out, prev_inj, layer_name):
    m = hidden_states.shape[0]
    return (
        hidden_states.new_empty(hidden_states.shape),
        hidden_states.new_empty((m, hidden_states.shape[1] // 4)),
        hidden_states.new_empty((m, 4)),
    )


direct_register_custom_op(
    op_name="qwen4_exp_mono_k1_rdna2",
    op_func=_k1,
    fake_impl=_k1_fake,
)
