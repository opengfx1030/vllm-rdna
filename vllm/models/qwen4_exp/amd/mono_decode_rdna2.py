# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MonoKernel on gfx1030 / RDNA2 (``--monokernel``).

``--monokernel`` (or ``VLLM_ROCM_MONO_DECODE=1``, turned into
``kernel_config.monokernel`` in the API server by the ROCm platform) runs a
Qwen4Exp decode step's eligible parts on the RDNA2 MonoKernel
(``mono/rdna2/``), all on with the validated defaults:

* ``k1``: the attention-side HC preamble on the persistent kernel (<= 24 rows);
* ``k2``: the decode FFN block in one persistent launch (<= 8 rows);
* ``gdn``: the GDN decode core in one launch (<= 64 rows);
* ``qsa``: the QSA decode path (bounded indexer, HIP sparse decode from 3 rows,
  fused norm + RoPE, one-launch indexer stores).

Per-part tuning / debug knobs: ``--additional-config '{"rocm_mono_decode":
{"k2_max_rows": 8, "timing": false, ...}}'`` (``mono.rdna2.Options``), read
only with ``--monokernel``. A deployment the kernels do not cover is refused at
startup with ``ValueError("--monokernel <reason>.")``. Without the switch
nothing here runs.
"""

import os

import torch

from vllm.logger import init_logger

from .mono.rdna2 import OPTS, Options

logger = init_logger(__name__)

# what the parts launch (csrc/rocm/mono_{k2,gdn,qsa_decode}_rdna2.cu)
REQUIRED_OPS = (
    "mono_k1_rdna2",
    "mono_k2_rdna2",
    "mono_k2_rdna2_workspace_bytes",
    "mono_k2_rdna2_workspace_alloc",
    "mono_k2_rdna2_mem_flags",
    "mono_k2_rdna2_mark",
    "mono_k2_rdna2_set_ar",
    "mono_gdn_rdna2",
    "mono_gdn_rdna2_scratch_bytes",
    "mono_qsa_decode_rdna2",
    "mono_qsa_decode_rdna2_splits",
)
RDNA2_MOE_METHOD = "CompressedTensorsWNA16RDNA2MoEMethod"
# The QSA owner reads the main KV cache as FP16/BF16 words. A packed or
# otherwise quantized cache (int4_fp16_block, fp8, ...) has a different record
# layout that these kernels do not read, so the switch refuses it here rather
# than letting a later layer fail on it.
FP16_KV_CACHE_DTYPES = ("auto", "float16", "bfloat16")
_MAX_ROWS = 24  # K1 / K2 build width (mono/rdna2/layout.py MAX_ROWS)
_GDN_MAX_ROWS = 64


def enabled(vllm_config) -> bool:
    """``--monokernel`` / ``VLLM_ROCM_MONO_DECODE=1``."""
    kernel_config = getattr(vllm_config, "kernel_config", None)
    return bool(getattr(kernel_config, "monokernel", False))


def options(vllm_config) -> Options | None:
    """The parts and knobs this config asks for; None without the switch."""
    if not enabled(vllm_config):
        return None
    extra = getattr(vllm_config, "additional_config", None) or {}
    knobs = extra.get("rocm_mono_decode", {}) if isinstance(extra, dict) else {}
    if not isinstance(knobs, dict):
        raise ValueError(
            "--monokernel: additional_config rocm_mono_decode must be a dict"
        )
    opts = Options.from_dict(knobs)
    opts.k1_max_rows = min(_MAX_ROWS, opts.k1_max_rows)
    opts.k2_max_rows = min(_MAX_ROWS, opts.k2_max_rows)
    opts.mtp_k1_max_rows = max(0, min(_MAX_ROWS, opts.mtp_k1_max_rows))
    opts.mtp_k2_max_rows = max(0, min(_MAX_ROWS, opts.mtp_k2_max_rows))
    opts.gdn_max_rows = min(_GDN_MAX_ROWS, opts.gdn_max_rows)
    return opts


def is_qwen4exp(vllm_config) -> bool:
    """Qwen4Exp, or its MTP drafter (the speculative config builds its own
    VllmConfig, which carries the flag too; the drafter keeps its modules)."""
    model_config = getattr(vllm_config, "model_config", None)
    archs = getattr(model_config, "architectures", None) or ()
    return any(a.startswith("Qwen4Exp") for a in archs)


def _on_gfx10x() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


def config_refusal(vllm_config) -> str | None:
    """Why this deployment cannot run the RDNA2 MonoKernel (None: it can)."""
    if not _on_gfx10x():
        return "needs a gfx10x (RDNA2) GPU"
    if not is_qwen4exp(vllm_config):
        archs = getattr(vllm_config.model_config, "architectures", None)
        return f"on gfx10x supports Qwen4Exp only, not {archs}"
    if not vllm_config.use_v2_model_runner:
        return "needs Model Runner V2 (VLLM_USE_V2_MODEL_RUNNER=1)"
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        return "does not support pipeline parallelism (PP > 1)"
    cache_dtype = getattr(
        getattr(vllm_config, "cache_config", None), "cache_dtype", None
    )
    if cache_dtype is not None and cache_dtype not in FP16_KV_CACHE_DTYPES:
        return f"needs an FP16 or BF16 KV cache, got {cache_dtype}"
    return None


def ops_refusal() -> str | None:
    import vllm._custom_ops  # noqa: F401  (loads _rocm_C)

    rocm_c = getattr(torch.ops, "_rocm_C", None)
    missing = [n for n in REQUIRED_OPS if rocm_c is None or not hasattr(rocm_c, n)]
    if missing:
        return f"needs the _rocm_C ops {missing} (rebuild the extension)"
    return None


def moe_refusal(layer) -> str | None:
    """The FFN kernel serves symmetric W4A16 MoE on the RDNA2 method only."""
    mlp = layer.mlp
    experts = getattr(mlp.experts, "routed_experts", mlp.experts)
    qm = getattr(experts, "quant_method", None) or getattr(
        mlp.experts, "_quant_method", None
    )
    name = type(qm).__name__
    if name != RDNA2_MOE_METHOD:
        return f"needs the MoE on symmetric W4A16 ({RDNA2_MOE_METHOD}), got {name}"
    if not getattr(qm, "symmetric", True):
        return f"needs the MoE on symmetric W4A16 ({RDNA2_MOE_METHOD}), got asymmetric"
    from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
        resolve_w4a8_moe,
    )

    if resolve_w4a8_moe():
        return f"needs the MoE on symmetric W4A16 ({RDNA2_MOE_METHOD}), got W4A8"
    return None


def check_config(vllm_config) -> None:
    """API-server check (ROCm platform): refuse early what the workers would."""
    if not enabled(vllm_config) or getattr(vllm_config, "model_config", None) is None:
        return
    if not (_on_gfx10x() or is_qwen4exp(vllm_config)):
        return  # another model's MonoKernel on another GPU: not ours to judge
    opts = options(vllm_config)  # unknown knobs
    why = config_refusal(vllm_config)
    if why is not None:
        raise ValueError(f"--monokernel {why}.")
    if opts is not None and opts.expert_cache_slots:
        tp = vllm_config.parallel_config.tensor_parallel_size
        if tp not in (2, 4):
            raise ValueError("--monokernel expert cache requires TP=2 or TP=4")
        if opts.expert_cache_slots > 512 // tp:
            raise ValueError("--monokernel expert cache exceeds rank-local experts")
        if not opts.k2 or opts.modules:
            raise ValueError("--monokernel expert cache requires K2 enabled")
        compilation = vllm_config.compilation_config
        graph_mode = getattr(compilation, "cudagraph_mode", None)
        if graph_mode is not None and graph_mode.has_piecewise_cudagraphs():
            from vllm.compilation.breakable_cudagraph import (
                is_breakable_cudagraph_enabled,
            )

            if not is_breakable_cudagraph_enabled():
                raise ValueError(
                    "--monokernel expert cache requires "
                    "VLLM_USE_BREAKABLE_CUDAGRAPH=1 for piecewise graphs"
                )
        splitting = compilation.splitting_ops
        if splitting is None:
            splitting = list(compilation._attention_ops)
        compilation.splitting_ops = list(
            dict.fromkeys(
                [
                    *splitting,
                    "vllm::qwen4_exp_mono_k2_rdna2",
                    "vllm::moe_forward",
                    "vllm::moe_forward_shared",
                ]
            )
        )
    if opts is not None and (opts.mtp_drafter or opts.k1 or opts.k2):
        # Configure before env caching and worker spawn. Preserve explicit
        # opt-outs so they fail early instead of being silently overridden.
        raw = os.environ.get("VLLM_RAISE_ON_LOGIT_NANS")
        if raw is None:
            os.environ["VLLM_RAISE_ON_LOGIT_NANS"] = "1"
            logger.info(
                "--monokernel: enabling VLLM_RAISE_ON_LOGIT_NANS=1 for "
                "timeout-poisoned output fail-stop"
            )
        else:
            try:
                enabled_nan_failstop = bool(int(raw))
            except ValueError as exc:
                raise ValueError(
                    "--monokernel VLLM_RAISE_ON_LOGIT_NANS must be 0 or 1."
                ) from exc
            if not enabled_nan_failstop:
                raise ValueError(
                    "--monokernel requires VLLM_RAISE_ON_LOGIT_NANS=1; "
                    "explicitly setting it to 0 disables fail-stop for "
                    "timeout-poisoned NaN outputs."
                )


def install(vllm_config) -> Options | None:
    """Qwen4ExpModel, before its layers are built (every rank): check the
    deployment, set the options and install the GDN / QSA hooks."""
    opts = options(vllm_config)
    if opts is None:
        return None
    if opts.mtp_drafter or opts.k1 or opts.k2:
        from vllm import envs

        if not envs.VLLM_RAISE_ON_LOGIT_NANS:
            raise ValueError(
                "--monokernel target or MTP drafter requires "
                "VLLM_RAISE_ON_LOGIT_NANS=1 so timeout-poisoned NaN outputs "
                "fail-stop before generated tokens are published"
            )
    why = config_refusal(vllm_config) or ops_refusal()
    if why is not None:
        raise ValueError(f"--monokernel {why}.")
    if opts.expert_cache_slots:
        missing = [
            name
            for name in ("mono_k2_offload_rdna2", "moe_fixedpoint_to_half_rdna2")
            if not hasattr(torch.ops._rocm_C, name)
        ]
        if missing:
            raise ValueError(f"--monokernel expert cache needs {missing}; rebuild")
    from .mono.rdna2 import debug, k1, k2, layout  # noqa: F401  (register ops)

    OPTS.assign(opts)
    layout.set_modes()
    if opts.timing:
        debug.wrap_linear_all_reduce()
    if opts.gdn:
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer

        from .mono.rdna2 import gdn

        gdn_layer.MONO_GDN_HOOK = gdn.install_hook
    if opts.qsa:
        from .mono.rdna2 import qsa
        from .ops import qsa as qsa_ops

        spec = getattr(vllm_config, "speculative_config", None)
        width = 1 + (getattr(spec, "num_speculative_tokens", 0) or 0)
        qsa.DECODE_WIDTH = width
        qsa_ops.__dict__["MONO_QSA"] = qsa
    _log_parts(opts)
    return opts


def _log_parts(o: Options) -> None:
    from vllm.distributed import get_tensor_model_parallel_rank

    try:
        rank = get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001  (no TP group, e.g. a unit test)
        rank = 0
    parts = [
        f"k1 (<= {o.k1_max_rows} rows)" if o.k1 else "",
        f"k2 (<= {o.k2_max_rows} rows, {o.k2_modes})" if o.k2 else "",
        f"gdn (<= {o.gdn_max_rows} rows{', in FULL graphs' if o.gdn_graph else ''})"
        if o.gdn
        else "",
        "qsa (HIP sparse decode >= 3 rows)" if o.qsa else "",
        "k2 in-kernel all-reduce" if o.k2_ar else "",
        "timing" if o.timing else "",
        "modules reference" if o.modules else "",
        (
            f"MTP drafter K1/K2 (<= {o.mtp_k1_max_rows}/{o.mtp_k2_max_rows} rows)"
            if o.mtp_drafter
            else ""
        ),
    ]
    logger.info(
        "mono rdna2: --monokernel on TP rank %d: %s",
        rank,
        ", ".join(p for p in parts if p) or "no parts",
    )


def layer_hook(
    layer, vllm_config, prefix: str, is_moe_layer: bool
) -> tuple[str | None, str | None]:
    """Qwen4ExpDecoderLayer: the forward-context names of its K1 and K2 ops
    (None: today's modules). MTP is a separate explicit opt-in; the regular
    target-model route remains governed by the existing options."""
    if not enabled(vllm_config) or not is_moe_layer:
        return None, None
    is_mtp = ".mtp." in f".{prefix}."
    if is_mtp and not OPTS.mtp_drafter:
        return None, None
    k1_on, k2_on = OPTS.k1 or OPTS.timing, OPTS.k2 or OPTS.timing
    if not (k1_on or k2_on):
        return None, None
    why = moe_refusal(layer)
    if why is not None and not is_mtp:
        raise ValueError(f"--monokernel {why}.")
    from .mono.rdna2.layout import MonoLayer

    name = f"{prefix}.mono_rdna2"
    mono = MonoLayer(
        layer,
        is_mtp=is_mtp,
        k1_max_rows=OPTS.mtp_k1_max_rows if is_mtp else OPTS.k1_max_rows,
        k2_max_rows=OPTS.mtp_k2_max_rows if is_mtp else OPTS.k2_max_rows,
    )
    vllm_config.compilation_config.static_forward_context[name] = mono
    if why is not None:
        logger.warning_once(
            "mono MTP drafter K2 stays on modules: %s; K1 is checked independently",
            why,
        )
    return (name if k1_on else None), (name if k2_on else None)


def qsa_part(vllm_config):
    """QSAIndexer: the QSA part module (mono/rdna2/qsa.py) or None."""
    opts = options(vllm_config)
    if opts is None or not opts.qsa:
        return None
    from .mono.rdna2 import qsa

    return qsa


def set_worker_hw_queue_budget(vllm_config) -> None:
    """Engine-core process, before the workers are spawned.

    With a persistent kernel in the decode graph, a 2-HSA-queue cap
    (GPU_MAX_HW_QUEUES=2) makes CLR share HSA queues between streams ("lowest
    load" at stream creation, so per process and per run); whatever rides the
    second queue then overlaps the attention block's small kernels instead of
    the FFN block, and under HSA_FORCE_FINE_GRAIN_PCIE=1 one to three TP4 ranks
    ran their attention-side kernels ~10 % slower per layer while the others
    waited at the all-reduce (2026-10-15, 4x V620). Four HW queues (one
    process per GPU, 8 HQDs) removed the skew; the Flash-Next recipes set 4.
    This raises a lower budget to ``hw_queues`` (0 leaves the launcher's
    value alone)."""
    opts = options(vllm_config)
    if opts is None or opts.hw_queues <= 0 or not _on_gfx10x():
        return
    want = opts.hw_queues
    cur = os.environ.get("GPU_MAX_HW_QUEUES")
    if cur is not None and cur.isdigit() and int(cur) >= want:
        return
    os.environ["GPU_MAX_HW_QUEUES"] = str(want)
    logger.info(
        "mono rdna2: GPU_MAX_HW_QUEUES %s -> %d for the workers (persistent "
        "kernel in the decode graph)",
        cur,
        want,
    )
