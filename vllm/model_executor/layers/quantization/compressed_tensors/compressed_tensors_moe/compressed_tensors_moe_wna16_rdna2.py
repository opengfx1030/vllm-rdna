# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W4A16 using the fused RDNA2 (gfx1030) HIP kernel.

Uses ``moe_gptq_gemm_rdna2`` — a single HIP kernel launch per GEMM that
handles expert routing + W4A16 dequant + dot product with atomic output.

Weight format (per expert, same as dense RDNA2 W4A16):
  - Packed int32 ``[E, K/8, N]`` with exllama shuffle
  - Scales ``[E, groups, N]`` in activation dtype
  - Zero points ``[E, groups, N/8]`` packed int32: synthesized for symmetric
    checkpoints, the checkpoint's own for asymmetric ones (raw, v2 format)
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
    W4A8_MOE_MIN_TOKENS,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.rdna_routing_hist import (
    RoutingHistogram,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.utils import replace_parameter
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

# Decode-sized batches (<= 1024 routed entries, fp16) route and zero both
# accumulators in one launch; VLLM_RDNA_MOE_ROUTE_HIP=0 keeps align +
# count_and_sort + two fills.
_ROUTE_ZERO_MAX_ENTRIES = 1024
_ROUTE_ZERO_MAX_EXPERTS = 1024
_ROUTE_ZERO = os.environ.get("VLLM_RDNA_MOE_ROUTE_HIP", "1") != "0" and hasattr(
    torch.ops._rocm_C, "rdna_moe_route_zero"
)

# Token count from which the qualified Qwen4-Exp geometry (hidden 2560,
# intermediate 640, 128/512 experts, top-10) uses the 8-row MoE tile instead of
# the 4-row one. 4096 is the shipped cutoff, which a 2048-token chunk never
# reaches. The 2026-10-23 HIP 2-5 serving A/B (bench_results/2026-10-22-mtp-
# mono-efficiency/micro-wna16/tile-threshold-hip25-2026-10-23) measured 1024 at
# +4 % c1 prefill / -4 % TTFT; the cutoff stays 4096 until a paired proof lands.
_TILE8_MIN_TOKENS = int(os.environ.get("VLLM_RDNA2_MOE_TILE8_MIN_TOKENS", "4096"))

# Rows per MoE tile for the W4A8 (int8-activation) prefill GEMMs, any geometry.
# 0 keeps the choice made in _rdna2_fused_moe (4, or 8 from the tile8 cutoff).
# 4/8/16 force it for token counts that take the W4A8 kernel; 16 needs the v2
# kernel (selected automatically, K block from VLLM_RDNA_W4A8_KBLOCK).
_W4A8_BLOCK_M = int(os.environ.get("VLLM_RDNA2_W4A8_BLOCK_M", "0"))
if _W4A8_BLOCK_M not in (0, 4, 8, 16):
    raise ValueError(
        f"VLLM_RDNA2_W4A8_BLOCK_M must be 0, 4, 8 or 16, got {_W4A8_BLOCK_M}"
    )

# VLLM_RDNA2_MOE_DETERMINISTIC=1 makes the routed experts run-to-run
# bit-identical. The default epilogues add the K-slice and top-k partials with
# packed-fp16 CAS (fp32 atomic adds under VLLM_RDNA2_MOE_FP32_ACCUM=1); both
# round in arrival order, which changes between runs, and the 48-layer model
# amplifies that last-ulp noise into logprob changes. This mode sums in a
# fixed order instead:
#   * W4A8 steps whose GEMMs each run as one K block (v2 kernel,
#     VLLM_RDNA_W4A8_KBLOCK=-1): w1 already stores its rows; w2 stores one row
#     per (token, expert) and the fp32 moe_sum adds the top-k rows in order.
#   * every other step runs W4A16 into the Q32 fixed-point accumulators of the
#     expert cache (integer adds are associative).
# Resolved once at import, like VLLM_RDNA2_MOE_FP32_ACCUM.
_DETERMINISTIC = envs.VLLM_RDNA2_MOE_DETERMINISTIC


def _w4a8_single_k_block(block_size_m: int, *size_ks: int) -> bool:
    """Whether the W4A8 kernel runs GEMMs of these K sizes as one K block.

    Mirrors w4a8_kblock_request/resolve_k_block in moe_w4a8_rdna2.cu: -1 asks
    the v2 kernel for the whole K range, capped so the int8 A tile fits in
    32 KiB of LDS. One K block means no split-K partial sums.
    """
    if os.environ.get("VLLM_RDNA_W4A8_KBLOCK", "") != "-1":
        return False
    return block_size_m in (4, 8, 16) and all(
        k <= 32768 // block_size_m for k in size_ks
    )


# Opt-in routing-histogram capture (VLLM_RDNA_MOE_HIST_DIR); None when off.
_ROUTING_HIST = RoutingHistogram.from_env()

# Rows of the per-layer decode scratch (gate_up output, activation): at least
# 16 tokens at top-k 8, and every batch a captured graph replays (largest
# capture size x top-k) up to the fused router's entry limit, so captured
# graphs take no scratch from the allocator. Bigger eager steps allocate.
_MIN_SCRATCH_ROWS = 16 * 8


def _decode_scratch_rows(moe_config: object | None) -> int:
    top_k = getattr(moe_config, "experts_per_token", 0) or 8
    graph_tokens = getattr(moe_config, "max_capture_size", 0) or 0
    return max(_MIN_SCRATCH_ROWS, min(graph_tokens * top_k, _ROUTE_ZERO_MAX_ENTRIES))


def _synthesize_qzeros(
    groups: int, out_features: int, device: torch.device
) -> torch.Tensor:
    """Create packed zero-point tensor for symmetric quant.

    GPTQv1 +1 quirk: kernel adds 1 to stored zeros, so encode
    (bias - 1) = 7 for uint4b8 (bias=8).
    """
    zeros = torch.full(
        (groups, out_features),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    return pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)


class CompressedTensorsWNA16RDNA2MoEMethod(CompressedTensorsWNA16MoEMethod):
    """W4A16 MoE using the fused RDNA2 HIP kernel (moe_gptq_gemm_rdna2).

    Weights are in RDNA2 format (shuffled int32 [E, K/8, N]),
    NOT Triton format (transposed uint8). apply() dispatches through
    the fused HIP kernel directly.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_rdna_expert_cache_slots", 0):
            self._prepare_expert_cache(layer)
            return
        # The base class allocates CT WNA16 MoE buffers N-first (#52798):
        # packed [E, N, K/8], scales [E, N, groups]. The RDNA2 kernel reads
        # K-first ([E, K/8, N], [E, groups, N]); int4 packing runs along K,
        # so a transpose restores the old is_transposed=True layout.
        names = [
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
        ]
        if not self.symmetric:
            names += ["w13_weight_zero_point", "w2_weight_zero_point"]
        for name in names:
            replace_parameter(
                layer, name, getattr(layer, name).data.transpose(1, 2).contiguous()
            )
        self.process_k_first_weights(layer)

    def _prepare_expert_cache(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe.rdna_expert_cache import (
            PackedExpertCache,
            prepare_host_payload,
        )

        device = layer._rdna_expert_cache_device
        host = prepare_host_payload(layer, self.group_size, device, ops.gptq_shuffle)
        cache = PackedExpertCache(
            host,
            layer._rdna_expert_cache_slots,
            device,
            policy=layer._rdna_expert_cache_policy,
            decay_interval=layer._rdna_expert_cache_decay,
            copy_mode=layer._rdna_expert_cache_copy,
        )
        for name, tensor in cache.payload.items():
            if hasattr(layer, name):
                replace_parameter(layer, name, tensor)
            else:
                layer.register_parameter(
                    name, torch.nn.Parameter(tensor, requires_grad=False)
                )
        layer._rdna_expert_cache = cache
        logger.info(
            "RDNA2 expert cache %s: %d logical experts, %d GPU slots; "
            "device %.3f GiB, host %.3f GiB",
            getattr(layer, "layer_name", ""),
            cache.logical_num_experts,
            cache.slot_count,
            cache.device_bytes / 2**30,
            cache.host_bytes / 2**30,
        )
        self._zero_v2 = False
        self._w4a8 = False
        act_dtype = host["w13_weight_scale"].dtype
        width = host["w13_weight_packed"].shape[2]
        layer.rdna2_w1_buf = torch.zeros(160, width, dtype=act_dtype, device=device)
        layer.rdna2_act_buf = torch.empty(
            160, width // 2, dtype=act_dtype, device=device
        )
        layer.rdna2_empty_tw = torch.empty(0, device=device)

    def process_k_first_weights(self, layer: torch.nn.Module) -> None:
        """Shuffle, set up zeros and allocate buffers for K-first weights.

        Also the entry point for callers that already hold K-first weights
        (the resident layout in rdna2_moe_resident.py).
        """
        device = layer.w13_weight_packed.device
        num_experts = layer.w13_weight_packed.shape[0]

        # Shuffle weights in-place per expert (exllama nibble interleave)
        for e in range(num_experts):
            w13_e = layer.w13_weight_packed.data[e].contiguous()
            ops.gptq_shuffle(w13_e, 4)
            layer.w13_weight_packed.data[e] = w13_e
            w2_e = layer.w2_weight_packed.data[e].contiguous()
            ops.gptq_shuffle(w2_e, 4)
            layer.w2_weight_packed.data[e] = w2_e

        # Keep scales as [E, groups, N] in activation dtype
        act_dtype = layer.w13_weight_scale.dtype
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.to(dtype=act_dtype).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.to(dtype=act_dtype).contiguous(),
            requires_grad=False,
        )

        w13_groups = (layer.w13_weight_packed.shape[1] * 8) // self.group_size
        w13_N = layer.w13_weight_packed.shape[2]
        w2_groups = (layer.w2_weight_packed.shape[1] * 8) // self.group_size
        w2_N = layer.w2_weight_packed.shape[2]

        # Packed zero points [E, groups, N/8] int32. Asymmetric checkpoints keep
        # their own (raw, read with zero_offset 0); the resident layouts build
        # this method without __init__ and are symmetric.
        self._zero_v2 = not getattr(self, "symmetric", True)
        if self._zero_v2:
            layer.w13_qzeros = torch.nn.Parameter(
                layer.w13_weight_zero_point.data, requires_grad=False
            )
            layer.w2_qzeros = torch.nn.Parameter(
                layer.w2_weight_zero_point.data, requires_grad=False
            )
            del layer.w13_weight_zero_point, layer.w2_weight_zero_point
        else:
            w13_qz = _synthesize_qzeros(w13_groups, w13_N, device)
            w2_qz = _synthesize_qzeros(w2_groups, w2_N, device)
            layer.w13_qzeros = torch.nn.Parameter(
                w13_qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
                requires_grad=False,
            )
            layer.w2_qzeros = torch.nn.Parameter(
                w2_qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
                requires_grad=False,
            )

        # Pre-allocate reusable buffers for decode (the resident layouts build
        # this method without __init__ and have no MoE config: minimum rows)
        N_gate_up = w13_N
        intermediate = N_gate_up // 2  # gated activation
        buf_size = _decode_scratch_rows(getattr(self, "moe", None))
        layer.rdna2_w1_buf = torch.zeros(
            buf_size, N_gate_up, dtype=act_dtype, device=device
        )
        layer.rdna2_act_buf = torch.empty(
            buf_size, intermediate, dtype=act_dtype, device=device
        )
        layer.rdna2_empty_tw = torch.empty(0, device=device)

        # Resolve the W4A8 MoE opt-in once; the traced forward only reads the
        # resulting Python bool. The resident layouts are hard-off inside
        # resolve_w4a8_moe (same weight buffers, different packing).
        from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
            resolve_w4a8_moe,
        )

        self._w4a8 = resolve_w4a8_moe()

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        if _ROUTING_HIST is not None:
            _ROUTING_HIST.record(layer, topk_ids)
        cache = getattr(layer, "_rdna_expert_cache", None)
        if cache is not None:
            x = x.contiguous()
            result = x.new_empty(x.shape)
            _apply_cached_moe(result, layer, x, topk_weights, topk_ids, activation)
            return result
        return _rdna2_fused_moe(
            x,
            topk_weights,
            topk_ids,
            layer=layer,
            activation=activation,
            apply_router_weight_on_input=(layer.apply_router_weight_on_input),
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            w4a8=getattr(self, "_w4a8", False),
            zero_v2=getattr(self, "_zero_v2", False),
        )


@eager_break_during_capture
def _apply_cached_moe(result, layer, x, topk_weights, topk_ids, activation):
    from vllm.model_executor.layers.fused_moe.rdna_expert_cache import (
        is_capturing_expert_work,
    )

    cache = layer._rdna_expert_cache
    if x.device.type == "cuda" and is_capturing_expert_work():
        raise RuntimeError("expert offload requires a breakable graph boundary")
    local = cache.local_ids(topk_ids, layer.expert_map)
    ids = local.flatten().tolist()
    cache.policy.observe(ids)
    # W2 keeps one fixed-point sum across passes and rounds only once. At
    # 4096/top10/H2560, W1 and W2 fixed-point scratch total about 480 MiB.
    routed_accumulator = torch.zeros(
        x.shape[0], x.shape[1] + 1, dtype=torch.int64, device=x.device
    )
    for experts in cache.policy.plan_passes(ids):
        lease = cache.ensure(experts)
        try:
            _rdna2_fused_moe(
                x,
                topk_weights,
                cache.remap(local, lease),
                layer=layer,
                activation=activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=cache.slot_count,
                expert_map=None,
                ignore_invalid_experts=True,
                routed_accumulator=routed_accumulator,
            )
        finally:
            cache.release(lease)
    ops.moe_fixedpoint_to_half_rdna2(routed_accumulator, result)


def _rdna2_fused_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: RoutedExperts,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w4a8: bool = False,
    zero_v2: bool = False,
    ignore_invalid_experts: bool = False,
    routed_accumulator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused MoE forward using the RDNA2 W4A16 HIP kernel.

    Optimizations vs naive dispatch:
      - BLOCK_SIZE_M=1 for decode (no padding waste, bf16 fast path)
      - Pre-allocated buffers (no torch.zeros per call)
      - moe_sum fused into output accumulation
    """
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    total_tokens = num_tokens * top_k
    w4a8 = w4a8 and num_tokens >= W4A8_MOE_MIN_TOKENS
    N_gate_up = layer.w13_weight_packed.shape[2]
    hidden_size = layer.w2_weight_packed.shape[2]
    dtype = hidden_states.dtype
    device = hidden_states.device
    if routed_accumulator is not None:
        if w4a8:
            raise ValueError("cached fixed-point output requires the W4A16 kernel")
        if (
            routed_accumulator.shape != (num_tokens, hidden_size + 1)
            or routed_accumulator.dtype != torch.int64
            or routed_accumulator.device != device
        ):
            raise ValueError("routed accumulator must be int64 [tokens, hidden+status]")

    intermediate_size = N_gate_up // 2 if activation.is_gated else N_gate_up

    if global_num_experts <= 0:
        global_num_experts = layer.w13_weight_packed.shape[0]

    use_large_prefill_tile8 = (
        num_tokens >= _TILE8_MIN_TOKENS
        and dtype == torch.float16
        and hidden_size == 2560
        and intermediate_size == 640
        and layer.w13_weight_packed.shape[0] == 128
        and top_k == 10
        and global_num_experts == 512
        and layer.w13_weight_scale.shape[1] == 20
        and layer.w2_weight_scale.shape[1] == 5
    )
    # Tile 8 is measured for this Qwen4-Exp geometry; retain existing choices elsewhere.
    block_size_m = 8 if use_large_prefill_tile8 else (1 if num_tokens <= 4 else 4)
    if w4a8 and _W4A8_BLOCK_M:
        block_size_m = _W4A8_BLOCK_M

    if use_large_prefill_tile8:
        logger.info_once(
            "RDNA2 resident MoE: tile8 prefill active "
            "(hidden=2560, intermediate=640, local/global experts=128/512, topk=10)"
        )

    if _DETERMINISTIC and routed_accumulator is None:
        if w4a8 and not _w4a8_single_k_block(
            block_size_m, hidden_states.shape[1], intermediate_size
        ):
            w4a8 = False
        if not w4a8:
            # W4A16 into Q32 fixed point: exact integer sums, one rounding.
            accumulator = torch.zeros(
                num_tokens, hidden_size + 1, dtype=torch.int64, device=device
            )
            _rdna2_fused_moe(
                hidden_states,
                topk_weights,
                topk_ids,
                layer,
                activation,
                apply_router_weight_on_input,
                global_num_experts,
                expert_map,
                zero_v2=zero_v2,
                ignore_invalid_experts=ignore_invalid_experts,
                routed_accumulator=accumulator,
            )
            result = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
            ops.moe_fixedpoint_to_half_rdna2(accumulator, result)
            return result

    # --- Token routing and scratch buffers ---
    out: torch.Tensor | None = None
    if (
        _ROUTE_ZERO
        and routed_accumulator is None
        and total_tokens <= _ROUTE_ZERO_MAX_ENTRIES
        and global_num_experts <= _ROUTE_ZERO_MAX_EXPERTS
        and dtype == torch.float16
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
    ):
        # One launch: align + sort + zero fill of w1_out and out, which both
        # take atomic adds. It was four launches, 13 us per layer in graph
        # replay at 1-32 tokens; the op takes 4-7 us.
        if total_tokens <= layer.rdna2_w1_buf.shape[0]:
            w1_out = layer.rdna2_w1_buf[:total_tokens]
            act_out = layer.rdna2_act_buf[:total_tokens]
        else:  # an eager step above the sized scratch
            w1_out = torch.empty(total_tokens, N_gate_up, dtype=dtype, device=device)
            act_out = torch.empty(
                total_tokens, intermediate_size, dtype=dtype, device=device
            )
        out = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
        sorted_token_ids, expert_ids, num_tokens_post_padded = ops.rdna_moe_route_zero(
            topk_ids, block_size_m, global_num_experts, expert_map, w1_out, out
        )
    else:
        # With expert parallelism the align kernel itself drops the experts
        # that live on other ranks and emits local expert ids. Without it the
        # wrapper would remap expert_ids afterwards with expert_map[expert_ids]:
        # an int32 -> int64 cast plus a gather, two extra launches per layer,
        # and the blocks of the other ranks' experts would still be scheduled.
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=ignore_invalid_experts or expert_map is not None,
        )

        # --- Reuse pre-allocated buffers when possible ---
        if total_tokens <= layer.rdna2_w1_buf.shape[0]:
            w1_out = layer.rdna2_w1_buf[:total_tokens]
            w1_out.zero_()
            act_out = layer.rdna2_act_buf[:total_tokens]
        else:
            w1_out = torch.zeros(
                total_tokens,
                N_gate_up,
                dtype=dtype,
                device=device,
            )
            act_out = torch.empty(
                total_tokens,
                intermediate_size,
                dtype=dtype,
                device=device,
            )

    # --- topk weights (pre-cast to float32 for kernel) ---
    topk_w_float = topk_weights.view(-1).float()
    empty_tw = layer.rdna2_empty_tw
    w1_accumulator = (
        torch.zeros(total_tokens, N_gate_up + 1, dtype=torch.int64, device=device)
        if routed_accumulator is not None
        else w1_out
    )

    # --- w1 GEMM: [M, K] -> [M*top_k, N_gate_up] ---
    if w4a8:
        ops.moe_w4a8_gemm_rdna2(
            hidden_states,
            w1_out,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            layer.w13_qzeros,
            topk_w_float if apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
            0,
            zero_v2,
        )
    else:
        ops.moe_gptq_gemm_rdna2(
            hidden_states,
            w1_accumulator,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            layer.w13_qzeros,
            topk_w_float if apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
            use_v2_format=zero_v2,
        )
    if routed_accumulator is not None:
        ops.moe_fixedpoint_to_half_rdna2(w1_accumulator, w1_out)

    # --- Activation (silu_and_mul etc.) ---
    apply_moe_activation(activation, act_out, w1_out)

    if w4a8 and _DETERMINISTIC:
        # output_topk=0 stores row token * top_k + k (the single K block needs
        # no atomics); moe_sum then adds a token's local experts in k order.
        # Rows of experts on other ranks are neither written nor read.
        expert_rows = torch.empty(total_tokens, hidden_size, dtype=dtype, device=device)
        ops.moe_w4a8_gemm_rdna2(
            act_out,
            expert_rows,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            layer.w2_qzeros,
            topk_w_float if not apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            output_topk=0,
            use_v2_format=zero_v2,
            fp32_accum=False,
        )
        out = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
        ops.moe_sum(
            expert_rows.view(num_tokens, top_k, hidden_size), out, topk_ids, expert_map
        )
        return out

    # --- w2 GEMM: [M*top_k, intermediate] -> [M, hidden] (fused reduce) ---
    # output_topk=top_k: kernel writes to out[token_id / top_k] directly,
    # fusing moe_sum into the atomic accumulation — saves one kernel launch
    # and the w2_out intermediate buffer.
    if out is None:
        out = (
            routed_accumulator
            if routed_accumulator is not None
            else torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
        )
    if w4a8:
        ops.moe_w4a8_gemm_rdna2(
            act_out,
            out,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            layer.w2_qzeros,
            topk_w_float if not apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            output_topk=top_k,
            use_v2_format=zero_v2,
        )
    else:
        ops.moe_gptq_gemm_rdna2(
            act_out,
            out,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            layer.w2_qzeros,
            topk_w_float if not apply_router_weight_on_input else empty_tw,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            output_topk=top_k,
            use_v2_format=zero_v2,
        )
    return out
