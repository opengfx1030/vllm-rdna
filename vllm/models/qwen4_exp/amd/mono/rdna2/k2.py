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

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from . import OPTS, debug, offload_trace
from .layout import (
    LOGGED,
    MAX_ROWS,
    MODES,
    MonoLayer,
    ar_fused,
    mode,
)

logger = init_logger(__name__)


@eager_break_during_capture
def run_offloaded(mono, hidden_states, attn_out, injection):
    """Keep host miss handling between captured computation segments."""
    from vllm.model_executor.layers.fused_moe.rdna_expert_cache import (
        is_capturing_expert_work,
    )

    if is_capturing_expert_work():
        raise RuntimeError("expert offload requires segmented graph capture")
    capture = BreakableCUDAGraphCapture.current()
    trace = (
        offload_trace.begin(mono, hidden_states, attn_out, injection, capture)
        if offload_trace.ENABLED
        else None
    )
    check_finite = getattr(OPTS, "timing", False) and (
        capture is None or capture.replaying
    )
    valid_rows = None
    if check_finite:
        from vllm.forward_context import get_forward_context

        padding = get_forward_context().is_padding
        if padding is not None:
            valid_rows = ~padding[: hidden_states.shape[0]]
            check_finite = bool(valid_rows.any().item())
    if check_finite:
        for name, value in (
            ("residual", hidden_states),
            ("attention", attn_out),
            ("injection", injection),
        ):
            checked = value if valid_rows is None else value[valid_rows]
            if not torch.isfinite(checked).all().item():
                fault = torch.ops._rocm_C.mono_k2_rdna2_error(mono.ws).tolist()
                raise RuntimeError(
                    f"expert cache {mono.layer.layer_idx}: nonfinite {name}; "
                    f"K1 fault={fault}"
                )
    cache = mono.expert_cache
    m = hidden_states.shape[0]
    key = (m, hidden_states.dtype, hidden_states.device)
    buffers = getattr(mono, "_offload_buffers", None)
    if buffers is None:
        mono._offload_buffers = buffers = {}
    if key not in buffers:
        buffers[key] = (
            torch.empty(
                hidden_states.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ),
            torch.empty(attn_out.shape, dtype=attn_out.dtype, device=attn_out.device),
            torch.empty((m, 4), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.empty((m, 10), dtype=torch.int32, device=hidden_states.device),
            torch.empty((m, 10), dtype=torch.float32, device=hidden_states.device),
            torch.empty((m, 4), dtype=hidden_states.dtype, device=hidden_states.device),
        )
    res_out, out, inj_out, route_ids, route_weights, inj_input = buffers[key]
    inj_input.copy_(injection)
    # Native activations use flat row-major pointers. Keep exceptional input
    # layouts in stable storage so stage graphs do not capture temporary copies.
    staged = getattr(mono, "_offload_inputs", None)
    for name, value in (("residual", hidden_states), ("attention", attn_out)):
        if value.is_contiguous():
            continue
        if staged is None:
            mono._offload_inputs = staged = {}
        stage_key = (key, name)
        if stage_key not in staged:
            staged[stage_key] = torch.empty(
                value.shape, dtype=value.dtype, device=value.device
            )
        staged[stage_key].copy_(value)
        if name == "residual":
            hidden_states = staged[stage_key]
        else:
            attn_out = staged[stage_key]
    t = mono.t
    args = (
        hidden_states,
        attn_out,
        inj_input,
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
        mono.offload_ws,
        route_ids,
        route_weights,
        cache.slot_map,
    )
    tail = (mono.base, 1, OPTS.k2_num_wg, mono.eps, 2000.0)
    op = torch.ops._rocm_C.mono_k2_offload_rdna2

    def launch(stage, *, execute=True):
        if BreakableCUDAGraphCapture.current() is None:
            if execute:
                op(*args, stage, *tail)
            return
        graphs = getattr(mono, "_offload_graphs", None)
        if graphs is None:
            mono._offload_graphs = graphs = {}
        # Only the prefix reads the model's captured input addresses. Expert
        # and finish stages consume the fixed workspace and slot tables.
        graph_key = (stage, m)
        if stage == 1:
            graph_key += tuple(tensor.data_ptr() for tensor in args[:3])
        graph = graphs.get(graph_key)
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                op(*args, stage, *tail)
            graphs[graph_key] = graph
        if execute:
            graph.replay()

    launch(1)
    local = route_ids.cpu().reshape(-1).tolist()
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if is_forward_context_available():
        padding = get_forward_context().is_padding
        if padding is not None:
            padded = padding[:m].cpu().tolist()
            local = [
                expert
                for index, expert in enumerate(local)
                if not padded[index // route_ids.shape[1]]
            ]
    # A capture-time batch may have no local experts. Record its pass graph
    # now so later real demand never has to capture on the replay stream.
    # The prefix has completed before recording another workspace reset.
    launch(2, execute=False)
    local = [expert - mono.base for expert in local]
    cache.policy.observe(local)
    for experts in cache.policy.plan_passes(local):
        lease = cache.ensure(experts)
        try:
            launch(2)
        finally:
            cache.release(lease)
    launch(3)
    if check_finite:
        for name, value in (
            ("output", out),
            ("residual output", res_out),
            ("injection output", inj_out),
        ):
            checked = value if valid_rows is None else value[valid_rows]
            if not torch.isfinite(checked).all().item():
                fault = torch.ops._rocm_C.mono_k2_rdna2_error(mono.offload_ws).tolist()
                raise RuntimeError(
                    f"expert cache {mono.layer.layer_idx}: nonfinite {name}; "
                    f"K2 fault={fault}"
                )
    if trace is not None:
        offload_trace.finish(trace, route_ids, route_weights, res_out, out, inj_out)
    return res_out, out, inj_out


def run(mono: MonoLayer, hidden_states, attn_out, injection):
    """K2 on the persistent kernel; (residual, partial FFN output, injection)."""
    t = mono.t
    if getattr(mono, "expert_cache", None) is not None:
        return run_offloaded(mono, hidden_states, attn_out, injection)
    hidden_states = hidden_states.contiguous()
    attn_out = attn_out.contiguous()
    m = hidden_states.shape[0]
    res_out = hidden_states.new_empty(hidden_states.shape)
    inj_out = torch.empty(
        (m, 4), dtype=hidden_states.dtype, device=hidden_states.device
    )
    out = attn_out.new_empty(attn_out.shape)
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
        hidden_states.new_empty(hidden_states.shape),
        attn_out.new_empty(attn_out.shape),
        hidden_states.new_empty((m, 4)),
    )


direct_register_custom_op(
    op_name="qwen4_exp_mono_k2_rdna2",
    op_func=_k2,
    fake_impl=_k2_fake,
)
