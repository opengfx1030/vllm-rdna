# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 / K2 layout: the per-layer weight views the persistent kernel reads, the
per-device workspace, the rdna_ar attach for the in-kernel all-reduce and the
graph-mode helper shared by the parts."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from . import OPTS

logger = init_logger(__name__)

MAX_ROWS = 24  # widest batch K1 / K2 are built for
_WORKSPACES: dict[tuple[int, str], torch.Tensor] = {}
_AR_MAX: dict[int, int] = {}  # per device: rdna_ar slot bytes once attached
ZEROS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
LOGGED: set = set()


def mode() -> str:
    """The graph mode of the current call: full, piecewise or eager.

    Model Runner V2 captures FULL graphs by running the model with
    cudagraph_runtime_mode NONE inside torch.cuda.graph; PIECEWISE capture runs
    with mode PIECEWISE. Graph replays do not re-enter Python, so the
    capture-time choice is what replays."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not torch.cuda.is_current_stream_capturing():
        return "eager"
    if (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    ):
        return "piecewise"
    return "full"


# graph modes K1 / K2 run in (``k2_modes``), filled by set_modes() at install
MODES: set[str] = set()


def set_modes() -> None:
    MODES.clear()
    MODES.update(x.strip() for x in str(OPTS.k2_modes).split(",") if x.strip())


def workspace(device: torch.device, scope: str = "model") -> torch.Tensor:
    """One zero-initialized workspace per device and execution scope."""
    from . import debug

    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    key = (idx, scope)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ops = torch.ops._rocm_C
        shared_width = 320 if scope.endswith("-offload") else 160
        nbytes = int(
            ops.mono_k2_rdna2_workspace_bytes(2560, 320, 336, 512, shared_width, 640)
        )
        # explicit coarse-grained allocation, not the caching allocator (whose
        # blocks can be fine-grained under HSA_FORCE_FINE_GRAIN_PCIE=1)
        ws = ops.mono_k2_rdna2_workspace_alloc(nbytes + 8192, idx)
        _WORKSPACES[key] = ws
        probe = torch.empty(1, dtype=torch.uint8, device=device)
        logger.info(
            "mono k2_rdna2: workspace %d B on cuda:%d, HSA global flags %d "
            "(caching allocator: %d; 2 fine, 4 coarse, 8 extended fine)",
            nbytes + 8192,
            idx,
            ops.mono_k2_rdna2_mem_flags(ws),
            ops.mono_k2_rdna2_mem_flags(probe),
        )
        # stats / tuning block right after the kernel's workspace
        st = ws[nbytes : nbytes + debug.STAT_WORDS * 8].view(torch.int64)
        sc = debug.STAT_ROWS * debug.STAT_COLS
        if OPTS.k2_ar:
            _attach_ar(idx)
        st[sc + 0] = -1  # entry_min
        st[sc + 5] = OPTS.k2_sleep
        st[sc + 6] = OPTS.k2_flags
        if OPTS.timing and scope == "model":
            st[sc + 3] = 1  # enabled
            debug.STATS[idx] = st
    return ws


def _attach_ar(idx: int) -> None:
    """Point the K2 launches of this device at the TP group's rdna_ar one-shot
    buffers; the kernel then returns the all-reduced output."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        comm = getattr(get_tp_group().device_communicator, "rdna_ar_comm", None)
        handle = comm.handle if comm is not None and not comm.disabled else -1
        _AR_MAX[idx] = int(torch.ops._rocm_C.mono_k2_rdna2_set_ar(handle, idx))
    except Exception as e:  # noqa: BLE001
        _AR_MAX[idx] = 0
        logger.warning("mono k2_rdna2 ar: not attached (%s)", e)
    logger.info(
        "mono k2_rdna2 ar: cuda:%d %s",
        idx,
        f"fused all-reduce up to {_AR_MAX[idx]} B" if _AR_MAX[idx] else "off",
    )


def ar_fused(x: torch.Tensor) -> bool:
    """Mirrors the kernel's rule for doing the all-reduce itself."""
    if not OPTS.k2_ar:
        return False
    lim = _AR_MAX.get(x.device.index if x.device.index is not None else 0, 0)
    return lim > 0 and x.numel() * x.element_size() <= lim


def check_expert_cache_workspaces(device: torch.device) -> None:
    """Refuse startup if graph capture left a persistent decoder fault."""
    idx = device.index
    if idx is None:
        idx = torch.accelerator.current_device_index()
    for scope in ("model", "model-offload"):
        ws = _WORKSPACES.get((idx, scope))
        if ws is not None:
            fault = torch.ops._rocm_C.mono_k2_rdna2_error(ws).tolist()
            if fault[0]:
                raise RuntimeError(f"expert cache {scope} capture fault: {fault}")


class MonoLayer:
    """Per-decoder-layer view of the weights K1 / K2 read (resolved on the
    first eager call, after process_weights_after_loading). ``ready`` stays
    False, with a warning, for a geometry the kernel does not cover."""

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        is_mtp: bool = False,
        k1_max_rows: int | None = None,
        k2_max_rows: int | None = None,
    ):
        self.layer = layer
        self.is_mtp = is_mtp
        self.preserve_fault = is_mtp  # production MTP keeps K1/K2 errors sticky
        self.k1_max_rows = OPTS.k1_max_rows if k1_max_rows is None else k1_max_rows
        self.k2_max_rows = OPTS.k2_max_rows if k2_max_rows is None else k2_max_rows
        self.first = getattr(layer, "layer_idx", None) == 0
        self.ready = False
        self.k1_ready = False
        self.k1_reason = ""
        self.hc_ok = False
        self.dense_mtp = False
        self._dense_layout_logged = False
        self._dense_call_logged = False
        self.reason = ""

    def setup(self) -> bool:
        if self.ready or self.reason:
            return self.ready
        try:
            self._setup()
            self.ready = True
        except Exception as e:  # unsupported geometry/quant: stay on modules
            self.reason = str(e)
            logger.warning_once(
                "mono k2_rdna2 disabled for %s: %s", type(self.layer).__name__, e
            )
        return self.ready

    def setup_k1(self) -> bool:
        """Resolve only the HC tensors needed by K1.

        MTP's dense FFN can be unsupported by K2 while its attention
        HyperConnection still has the exact K1 layout. Keep these readiness
        checks independent so that K2 can fall back without disabling K1.
        """
        if self.k1_ready or self.k1_reason:
            return self.k1_ready
        try:
            self._setup_k1()
            self.k1_ready = True
        except Exception as e:  # unsupported HC geometry: stay on modules
            self.k1_reason = str(e)
            logger.warning_once(
                "mono k1_rdna2 disabled for %s: %s",
                type(self.layer).__name__,
                e,
            )
        return self.k1_ready

    def _setup_k1(self) -> None:
        layer = self.layer
        mlp_hc = layer.mlp_hyper_connection
        attn_hc = layer.attn_hyper_connection
        if mlp_hc.hc_count != 4 or not mlp_hc.use_combine:
            raise ValueError("MLP hyper-connection must be hc_count 4 with combine")
        if attn_hc.hc_count != 4 or not attn_hc.use_combine:
            raise ValueError(
                "attention hyper-connection must be hc_count 4 with combine"
            )

        mlp_down = mlp_hc.input_mix_weight_down_block_inject
        mlp_up = mlp_hc.input_mix_weight_up
        down = attn_hc.input_mix_weight_down_block_inject
        up = attn_hc.input_mix_weight_up
        for name, weight in (
            ("MLP down", mlp_down),
            ("MLP up", mlp_up),
            ("attention down", down),
            ("attention up", up),
        ):
            if getattr(weight, "weight_i8", None) is not None:
                raise ValueError(f"{name} HC int8 shadow is not supported")
        if (
            down.weight.shape != mlp_down.weight.shape
            or up.weight.shape != mlp_up.weight.shape
        ):
            raise ValueError("attention and MLP HC weight shapes differ")
        norm_w = attn_hc.hc_norm.weight
        if any(
            weight.dtype != torch.float16 for weight in (norm_w, down.weight, up.weight)
        ):
            raise ValueError("attention HC weights must be fp16")
        if any(weight.device != down.weight.device for weight in (norm_w, up.weight)):
            raise ValueError("attention HC weights must share a device")

        self.th = dict(norm_w=norm_w, wd=down.weight, wu=up.weight)
        self.eps_a = float(attn_hc.config.rms_norm_eps)
        self.hc_ok = True
        idx = down.weight.device.index
        if idx is None:
            idx = torch.accelerator.current_device_index()
        if idx not in ZEROS:
            ZEROS[idx] = (
                torch.zeros(
                    MAX_ROWS,
                    2560,
                    dtype=torch.float16,
                    device=down.weight.device,
                ),
                torch.zeros(
                    MAX_ROWS, 4, dtype=torch.float16, device=down.weight.device
                ),
            )
        self.ws = workspace(down.weight.device, "mtp" if self.is_mtp else "model")

    def _setup(self):
        layer = self.layer
        hc = layer.mlp_hyper_connection
        mlp = layer.mlp
        for name in ("gate", "experts"):
            if not hasattr(mlp, name):
                raise ValueError(f"mlp has no {name}")
        if hc.hc_count != 4 or not hc.use_combine:
            raise ValueError("hyper-connection must be hc_count 4 with combine")
        down = hc.input_mix_weight_down_block_inject
        up = hc.input_mix_weight_up
        if (
            getattr(down, "weight_i8", None) is not None
            or getattr(up, "weight_i8", None) is not None
        ):
            raise ValueError("int8 HC shadows are not supported")
        experts = getattr(mlp.experts, "routed_experts", mlp.experts)
        qm = getattr(experts, "quant_method", None) or getattr(
            mlp.experts, "_quant_method", None
        )
        dense_mtp = self.is_mtp and type(qm).__name__ == "UnquantizedFusedMoEMethod"
        if (
            type(qm).__name__ != "CompressedTensorsWNA16RDNA2MoEMethod"
            and not dense_mtp
        ):
            raise ValueError(f"routed experts use {type(qm).__name__}")
        if dense_mtp:
            self._setup_dense_mtp(layer, hc, experts, mlp)
            return
        if getattr(qm, "_zero_v2", False) or getattr(qm, "_w4a8", False):
            raise ValueError("asymmetric (v2 zeros) or W4A8 experts")
        w13 = experts.w13_weight_packed
        cache = getattr(experts, "_rdna_expert_cache", None)
        if OPTS.expert_cache_slots and cache is None and not self.is_mtp:
            raise ValueError(
                "expert cache was requested but routed weights stayed resident"
            )
        self.expert_cache = cache
        local_experts = cache.logical_num_experts if cache is not None else 128
        physical_experts = cache.slot_count if cache is not None else 128
        if (
            local_experts not in (128, 256)
            or w13.shape[0] != physical_experts
            or w13.shape[2] != 1280
            or w13.shape[1] * 8 != 2560
        ):
            raise ValueError(f"unexpected routed expert shape {tuple(w13.shape)}")
        se = getattr(mlp, "shared_expert", None)
        if se is None or getattr(se, "expert_gate", None) is None:
            raise ValueError("shared expert with expert gate required")
        gu = se.gate_up_proj.weight
        dn = se.down_proj.weight
        wg = se.expert_gate.weight
        max_shared = 320 if cache is not None else 160
        if (
            gu.dtype != torch.float16
            or dn.dtype != torch.float16
            or gu.shape[0] > 2 * max_shared
        ):
            raise ValueError(
                f"shared expert must be fp16 with <= {max_shared} local rows"
            )
        router = mlp.gate.weight
        if router.dtype != torch.float16 or router.shape != (512, 2560):
            raise ValueError("router must be fp16 [512, 2560]")
        emap = getattr(experts, "expert_map", None)
        if emap is None:
            raise ValueError("expert parallelism (expert_map) required")
        loc = (emap >= 0).nonzero().flatten()
        base = int(loc[0]) if loc.numel() else -1
        if loc.numel() != local_experts or not torch.equal(
            emap[base : base + local_experts].cpu(),
            torch.arange(local_experts, dtype=emap.dtype),
        ):
            raise ValueError(
                f"local experts must be one contiguous block of {local_experts}"
            )
        if getattr(experts, "top_k", 10) != 10 or not getattr(
            experts, "renormalize", True
        ):
            raise ValueError("top-10 with renormalization required")
        self.t = dict(
            norm_w=hc.hc_norm.weight,
            wd=down.weight,
            wu=up.weight,
            wr=router,
            se_w1=gu,
            se_w2=dn,
            se_wg=wg.reshape(-1).contiguous(),
            w13=w13,
            s13=experts.w13_weight_scale,
            z13=experts.w13_qzeros,
            w2=experts.w2_weight_packed,
            s2=experts.w2_weight_scale,
            z2=experts.w2_qzeros,
        )
        self.base = base
        self.eps = float(hc.config.rms_norm_eps)
        self.ws = workspace(router.device, "mtp" if self.is_mtp else "model")
        if cache is not None:
            self.offload_ws = workspace(router.device, "model-offload")
        # K1: the attention-side HC module, same geometry as the MLP one
        ahc = layer.attn_hyper_connection
        adown = ahc.input_mix_weight_down_block_inject
        aup = ahc.input_mix_weight_up
        self.hc_ok = (
            ahc.hc_count == 4
            and ahc.use_combine
            and getattr(adown, "weight_i8", None) is None
            and getattr(aup, "weight_i8", None) is None
            and adown.weight.shape == down.weight.shape
            and aup.weight.shape == up.weight.shape
        )
        if self.hc_ok:
            self.th = dict(norm_w=ahc.hc_norm.weight, wd=adown.weight, wu=aup.weight)
            self.eps_a = float(ahc.config.rms_norm_eps)
            self.k1_ready = True
            idx = router.device.index
            if (
                idx not in ZEROS
            ):  # (output, injection) for layers with no pending combine
                ZEROS[idx] = (
                    torch.zeros(
                        MAX_ROWS, 2560, dtype=torch.float16, device=router.device
                    ),
                    torch.zeros(MAX_ROWS, 4, dtype=torch.float16, device=router.device),
                )

    def _setup_dense_mtp(self, layer, hc, experts, mlp) -> None:
        """Resolve the MTP-only unquantized FP16 expert layout for K2."""
        if not self.is_mtp:
            raise ValueError("dense FFN is supported only for the MTP drafter")
        hc_weights = (
            hc.hc_norm.weight,
            hc.input_mix_weight_down_block_inject.weight,
            hc.input_mix_weight_up.weight,
        )
        if any(weight.dtype != torch.float16 for weight in hc_weights):
            raise ValueError("dense MTP hyper-connection weights must be fp16")
        if any(weight.device != experts.w13_weight.device for weight in hc_weights):
            raise ValueError("dense MTP hyper-connection weights must share a device")
        if hc_weights[0].shape != (10240,):
            raise ValueError("dense MTP HC norm must have one weight per HC element")
        if hc_weights[1].shape != (336, 10240) or hc_weights[2].shape != (
            10240,
            320,
        ):
            raise ValueError("dense MTP hyper-connection geometry is unsupported")
        if (
            getattr(experts, "w13_bias", None) is not None
            or getattr(experts, "w2_bias", None) is not None
        ):
            raise ValueError("dense MTP expert biases are not supported")
        w13 = experts.w13_weight
        w2 = experts.w2_weight
        if (
            w13.dtype != torch.float16
            or w2.dtype != torch.float16
            or not w13.is_contiguous()
            or not w2.is_contiguous()
            or w13.ndim != 3
            or w2.ndim != 3
        ):
            raise ValueError("dense MTP experts must be contiguous rank-3 fp16 weights")
        local_e, n2, h = w13.shape
        if n2 % 2:
            raise ValueError("dense MTP w13 second dimension must be even")
        i = n2 // 2
        if (local_e, h, i) != (128, 2560, 640) or tuple(w2.shape) != (
            local_e,
            h,
            i,
        ):
            raise ValueError(
                f"dense MTP expert geometry must be w13[128,1280,2560]/"
                f"w2[128,2560,640], got {tuple(w13.shape)}/{tuple(w2.shape)}"
            )
        if w13.device != w2.device or w13.device != mlp.gate.weight.device:
            raise ValueError("dense MTP expert and router weights must share a device")
        if (
            mlp.gate.weight.dtype != torch.float16
            or tuple(mlp.gate.weight.shape) != (512, 2560)
            or mlp.gate.weight.device != w13.device
            or getattr(mlp.gate, "bias", None) is not None
        ):
            raise ValueError("dense MTP router must be fp16 [512, 2560]")
        if getattr(experts, "top_k", 10) != 10 or not getattr(
            experts, "renormalize", True
        ):
            raise ValueError("dense MTP requires normalized top-10 routing")
        if (
            getattr(experts, "activation", None) != MoEActivation.SILU
            or getattr(experts, "scoring_func", "softmax") != "softmax"
            or getattr(experts, "use_grouped_topk", False)
            or getattr(experts, "custom_routing_function", None) is not None
            or getattr(experts, "e_score_correction_bias", None) is not None
            or getattr(experts, "routed_scaling_factor", 1.0) != 1.0
        ):
            raise ValueError("dense MTP requires plain SiLU and softmax routing")
        if getattr(experts, "apply_router_weight_on_input", False):
            raise ValueError("dense MTP does not support router weights on input")
        emap = getattr(experts, "expert_map", None)
        if emap is None:
            raise ValueError("dense MTP requires expert parallelism (expert_map)")
        loc = (emap >= 0).nonzero().flatten()
        base = int(loc[0]) if loc.numel() else -1
        if loc.numel() != local_e or not torch.equal(
            emap[base : base + local_e].cpu(),
            torch.arange(local_e, dtype=emap.dtype),
        ):
            raise ValueError("dense MTP local experts must be one contiguous block")

        se = getattr(mlp, "shared_expert", None)
        if se is None or getattr(se, "expert_gate", None) is None:
            raise ValueError("dense MTP requires the gated shared expert")
        gu = se.gate_up_proj.weight
        dn = se.down_proj.weight
        wg = se.expert_gate.weight
        if (
            gu.dtype != torch.float16
            or dn.dtype != torch.float16
            or wg.dtype != torch.float16
            or not gu.is_contiguous()
            or not dn.is_contiguous()
            or not wg.is_contiguous()
        ):
            raise ValueError("dense MTP shared expert must be contiguous fp16")
        # TP=4 shards the checkpoint's global shared intermediate width (640)
        # to 160 columns per rank for this kernel.
        if (
            getattr(layer.config, "shared_expert_intermediate_size", None) != 640
            or dn.ndim != 2
            or gu.shape != (2 * dn.shape[1], 2560)
            or dn.shape[0] != 2560
            or dn.shape[1] != 160
            or wg.numel() != 2560
            or getattr(se.gate_up_proj, "bias", None) is not None
            or getattr(se.down_proj, "bias", None) is not None
            or getattr(se.expert_gate, "bias", None) is not None
        ):
            raise ValueError(
                "dense MTP shared expert layout is unsupported: "
                "expected TP4 local width 160 from global width 640; "
                f"gate_up={tuple(gu.shape)}, down={tuple(dn.shape)}, "
                f"gate={tuple(wg.shape)}"
            )
        if any(t.device != w13.device for t in (gu, dn, wg)):
            raise ValueError("dense MTP shared expert must share the FFN device")

        self.t = dict(
            norm_w=hc.hc_norm.weight,
            wd=hc.input_mix_weight_down_block_inject.weight,
            wu=hc.input_mix_weight_up.weight,
            wr=mlp.gate.weight,
            se_w1=gu,
            se_w2=dn,
            se_wg=wg.reshape(-1).contiguous(),
            w13=w13,
            s13=torch.empty(0, dtype=torch.float16, device=w13.device),
            z13=torch.empty(0, dtype=torch.int32, device=w13.device),
            w2=w2,
            s2=torch.empty(0, dtype=torch.float16, device=w13.device),
            z2=torch.empty(0, dtype=torch.int32, device=w13.device),
        )
        if not getattr(self, "_dense_layout_logged", False):
            self._dense_layout_logged = True
            tensors = {**self.t, "shared_gate": wg}
            logger.info(
                "dense MTP K2 layout rank-device=%s expert_base=%d tensors=%s",
                w13.device,
                base,
                {
                    name: {
                        "ptr": hex(t.data_ptr()),
                        "shape": tuple(t.shape),
                        "stride": tuple(t.stride()),
                        "contiguous": t.is_contiguous(),
                        "dtype": str(t.dtype),
                    }
                    for name, t in tensors.items()
                },
            )
        self.base = base
        self.eps = float(hc.config.rms_norm_eps)
        self.dense_mtp = True
        self.ws = workspace(w13.device, "mtp")
