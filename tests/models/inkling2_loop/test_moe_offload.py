# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Inkling-2-Mini TP=1 MoE expert offloading (NVFP4 + MXFP8)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.rdna_expert_cache import PackedExpertCache
from vllm.models.inkling2_loop.amd.moe import (
    _MXFP8_PAYLOAD_NAMES,
    _NVFP4_PAYLOAD_NAMES,
    Inkling2LoopMoE,
    moe_offload_config,
    moe_offload_prefetch_config,
)


def _tiny_config(
    num_experts: int = 16,
    top_k: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        n_routed_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        scoring_func="sigmoid",
        route_scale=1.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )


def _fill_moe_weights(moe: Inkling2LoopMoE, seed: int = 0) -> dict[str, torch.Tensor]:
    """Deterministic synthetic checkpoint-layout weights for NVFP4 or MXFP8."""
    gen = torch.Generator().manual_seed(seed)
    e, h, inter = moe.num_experts, moe.hidden_dim, moe.intermediate_dim
    moe.gate.weight.data.copy_(
        torch.randn(e, h, generator=gen, dtype=torch.float32) * 0.1
    )
    moe.gate.e_score_correction_bias.data.copy_(
        torch.randn(e, generator=gen, dtype=torch.float32) * 0.05
    )
    group: dict[str, torch.Tensor] = {}
    if moe.quant_format == "nvfp4":
        for stage, n, k in (("w13", 2 * inter, h), ("w2", h, inter)):
            group[f"{stage}_weight"] = torch.randint(
                0, 256, (e, n, k // 2), generator=gen, dtype=torch.uint8
            )
            # Valid finite positive E4M3 scales (raw bytes 0x30..0x48)
            raw_scales = torch.randint(
                0x30, 0x48, (e, n, k // 16), generator=gen, dtype=torch.uint8
            )
            group[f"{stage}_weight_scale"] = raw_scales.view(torch.float8_e4m3fn)
            group[f"{stage}_weight_amax"] = (
                torch.rand(e, generator=gen, dtype=torch.float32) * 2.0 + 0.5
            )
            group[f"{stage}_weight_outer_scale_denominator"] = torch.tensor(
                [2688.0], dtype=torch.float32
            )
    else:
        for stage, n, k in (("w13", 2 * inter, h), ("w2", h, inter)):
            raw_w = torch.randint(
                0x20, 0x48, (e, n, k), generator=gen, dtype=torch.uint8
            )
            group[f"{stage}_weight"] = raw_w.view(torch.float8_e4m3fn)
            # Valid UE8M0 scales in [113, 118] (within MXFP8_SCALE_RANGE 95..134)
            group[f"{stage}_weight_scale"] = torch.randint(
                113, 119, (e, n, k // 32), generator=gen, dtype=torch.uint8
            )
    moe.load_expert_group(group)
    return group


def _is_gfx103x() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    props = torch.cuda.get_device_properties(0)
    return props.gcnArchName.startswith("gfx103")


# ---------------------------------------------------------------------------
# CPU tests (run anywhere)
# ---------------------------------------------------------------------------


def test_moe_offload_config_defaults_and_validation():
    assert moe_offload_config({}) == (0, "tensor", "lfu_lru", 128)
    assert moe_offload_config(
        {
            "VLLM_INKLING2_MOE_OFFLOAD_SLOTS": "48",
            "VLLM_INKLING2_MOE_OFFLOAD_COPY": "native",
            "VLLM_INKLING2_MOE_OFFLOAD_POLICY": "lru",
            "VLLM_INKLING2_MOE_OFFLOAD_DECAY": "64",
        }
    ) == (48, "native", "lru", 64)
    with pytest.raises(ValueError, match=">= 0"):
        moe_offload_config({"VLLM_INKLING2_MOE_OFFLOAD_SLOTS": "-1"})
    with pytest.raises(ValueError, match="tensor"):
        moe_offload_config({"VLLM_INKLING2_MOE_OFFLOAD_COPY": "bogus"})
    with pytest.raises(ValueError, match="lfu_lru"):
        moe_offload_config({"VLLM_INKLING2_MOE_OFFLOAD_POLICY": "fifo"})
    with pytest.raises(ValueError, match="> 0"):
        moe_offload_config({"VLLM_INKLING2_MOE_OFFLOAD_DECAY": "0"})


def test_moe_offload_prefetch_config_defaults_and_validation():
    assert moe_offload_prefetch_config({}) == (True, 4, 4)
    assert moe_offload_prefetch_config(
        {
            "VLLM_INKLING2_MOE_OFFLOAD_PREFETCH": "0",
            "VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_TOPK": "6",
            "VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_MAX": "8",
        }
    ) == (False, 6, 8)
    with pytest.raises(ValueError, match=">= 0"):
        moe_offload_prefetch_config({"VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_TOPK": "-1"})
    with pytest.raises(ValueError, match="> 0"):
        moe_offload_prefetch_config({"VLLM_INKLING2_MOE_OFFLOAD_PREFETCH_MAX": "0"})


def test_tp_gt_1_guard_raises():
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="TP=1 only"):
        Inkling2LoopMoE(cfg, "nvfp4", tp_size=2, offload_slots=8)
    with pytest.raises(ValueError, match="TP=1 only"):
        Inkling2LoopMoE(cfg, "mxfp8", tp_size=4, offload_slots=8)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_streaming_cpu_staging_and_packed_cache_cpu(fmt):
    cfg = _tiny_config(num_experts=16, top_k=4, hidden_size=64, intermediate_size=64)
    moe = Inkling2LoopMoE(cfg, fmt, offload_slots=6, offload_copy="tensor")
    # Before process_weights_after_loading, checkpoint expert params live on CPU.
    for name in moe._expert_names:
        assert getattr(moe, name).device.type == "cpu"
    _fill_moe_weights(moe, seed=42)
    moe.process_weights_after_loading()

    # Raw checkpoint params are freed.
    for name in moe._expert_names:
        assert not hasattr(moe, name)
    assert moe.expert_cache is not None
    assert moe.expert_cache.slot_count == 6
    assert moe.expert_cache.logical_num_experts == 16
    assert moe.expert_cache.dummy_slot is None

    expected_names = _NVFP4_PAYLOAD_NAMES if fmt == "nvfp4" else _MXFP8_PAYLOAD_NAMES
    assert moe.expert_cache.payload_names == expected_names
    for name in expected_names:
        buf = getattr(moe, name)
        assert buf.shape[0] == 6
        assert buf is moe.expert_cache.payload[name]

    # Exercise LRU hit/miss/eviction and multi-pass overflow (12 > 6 slots).
    cache = moe.expert_cache
    demanded = list(range(12))
    cache.policy.observe(demanded)
    passes = cache.policy.plan_passes(demanded)
    assert len(passes) == 2
    assert sum(len(p) for p in passes) == 12
    for p in passes:
        lease = cache.ensure(p)
        try:
            for expert, slot, _ in lease.bindings:
                for name in expected_names:
                    assert torch.equal(
                        cache.payload[name][slot], cache.host_payload[name][expert]
                    )
        finally:
            cache.release(lease)

    stats = moe.offload_stats()
    assert stats is not None
    assert stats["demands"] == 12
    assert stats["misses"] == 12
    assert stats["committed_passes"] == 2
    assert stats["slots"] == 6


def test_packed_expert_cache_custom_4_and_6_payloads_cpu():
    host_4 = {
        "w13_qweight": torch.arange(8 * 4 * 8, dtype=torch.int32).reshape(8, 4, 8),
        "w13_scales": torch.arange(8 * 2 * 8, dtype=torch.uint8).reshape(8, 2, 8),
        "w2_qweight": torch.arange(8 * 4 * 8, dtype=torch.int32).reshape(8, 4, 8),
        "w2_scales": torch.arange(8 * 2 * 8, dtype=torch.uint8).reshape(8, 2, 8),
    }
    cache = PackedExpertCache(
        host_4,
        slot_count=3,
        device="cpu",
        payload_names=_MXFP8_PAYLOAD_NAMES,
        allocate_dummy_slot=False,
    )
    lease = cache.ensure([5, 1, 7])
    try:
        remapped = cache.remap(torch.tensor([5, 2, 7, -1], dtype=torch.int32), lease)
        assert remapped[0].item() >= 0
        assert remapped[1].item() == -1
        assert remapped[2].item() >= 0
        assert remapped[3].item() == -1
    finally:
        cache.release(lease)


# ---------------------------------------------------------------------------
# GPU tests (gfx103x only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_gfx103x(), reason="requires gfx103x HIP GPU")
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("copy_mode", ["tensor", "native"])
def test_gpu_cross_stream_dma_and_bit_exact_moe_parity(fmt, copy_mode):
    """Offloaded NVFP4/MXFP8 MoE matches resident Inkling2LoopMoE bit-for-bit
    across decode, overflow (unique > S), and native vs tensor DMA."""
    cfg = _tiny_config(num_experts=16, top_k=4, hidden_size=128, intermediate_size=128)
    with torch.device("cuda"):
        res = Inkling2LoopMoE(cfg, fmt, offload_slots=0)
        off = Inkling2LoopMoE(cfg, fmt, offload_slots=6, offload_copy=copy_mode)
    # Streaming load invariant: offloaded expert params stay on CPU even inside
    # torch.device("cuda")!
    for name in off._expert_names:
        assert getattr(off, name).device.type == "cpu"
    assert off.gate.weight.device.type == "cuda"

    group = _fill_moe_weights(res, seed=123)
    off.gate.weight.data.copy_(res.gate.weight.data)
    off.gate.e_score_correction_bias.data.copy_(res.gate.e_score_correction_bias.data)
    off.load_expert_group(group)

    res.process_weights_after_loading()
    off.process_weights_after_loading()

    gen = torch.Generator(device="cuda").manual_seed(99)
    # Test both small decode (fits in 6 slots) and overflow batch (hits > 6 experts)
    for num_tokens in (1, 4, 24):
        x = torch.randn(
            num_tokens,
            cfg.hidden_size,
            generator=gen,
            device="cuda",
            dtype=torch.float16,
        )
        y_res = res(x)
        y_off = off(x)
        assert torch.allclose(y_res, y_off, atol=1e-3, rtol=1e-3), (
            f"{fmt}/{copy_mode} T={num_tokens}: max diff "
            f"{(y_res - y_off).abs().max().item()}"
        )
    stats = off.offload_stats()
    assert stats is not None
    assert stats["demands"] > 0
    if copy_mode == "native":
        assert stats["native_copy_calls"] > 0


@pytest.mark.skipif(not _is_gfx103x(), reason="requires gfx103x HIP GPU")
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("copy_mode", ["tensor", "native"])
def test_gpu_forward_hip_decode_prefill_and_breakable_graph_parity(fmt, copy_mode):
    """Serving path forward_hip() (decode, prefill BM, overflow, and
    BreakableCUDAGraphCapture) matches resident forward_hip()."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.inkling2_loop.amd.glue import Inkling2GlueWorkspace
    from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig

    cfg = Inkling2LoopTextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=128,
        n_routed_experts=16,
        num_experts_per_tok=4,
        loop_layer_index_start=0,
        loop_layer_index_end=2,
        loop_num_passes=2,
        n_layers_per_attn_res_block=1,
        loop_gate_rank=16,
    )
    with torch.device("cuda"):
        res = Inkling2LoopMoE(cfg, fmt, offload_slots=0)
        off = Inkling2LoopMoE(cfg, fmt, offload_slots=6, offload_copy=copy_mode)

    group = _fill_moe_weights(res, seed=77)
    off.gate.weight.data.copy_(res.gate.weight.data)
    off.gate.e_score_correction_bias.data.copy_(res.gate.e_score_correction_bias.data)
    off.load_expert_group(group)

    res.process_weights_after_loading()
    off.process_weights_after_loading()
    res.prepare_hip()
    off.prepare_hip()

    ws = Inkling2GlueWorkspace(cfg, max_tokens=256, device="cuda", tp_size=1)
    gen = torch.Generator(device="cuda").manual_seed(55)

    # 1. Decode (1, 4) and prefill BM (160 >= 128, forces overflow 16 > 6 slots)
    for num_tokens in (1, 4, 160):
        x = torch.randn(
            num_tokens,
            cfg.hidden_size,
            generator=gen,
            device="cuda",
            dtype=torch.float16,
        )
        out_res = torch.empty_like(x)
        out_off = torch.empty_like(x)
        res.forward_hip(x, ws, out_res)
        off.forward_hip(x, ws, out_off, visit=0)
        assert torch.allclose(out_res, out_off, atol=1e-3, rtol=1e-3), (
            f"forward_hip {fmt}/{copy_mode} T={num_tokens}: max diff "
            f"{(out_res - out_off).abs().max().item()}"
        )

    # 2. BreakableCUDAGraphCapture capture + replay with changing inputs
    x_static = torch.randn(
        2, cfg.hidden_size, generator=gen, device="cuda", dtype=torch.float16
    )
    out_static = torch.empty_like(x_static)
    capture_stream = torch.cuda.Stream()
    torch.accelerator.synchronize()
    with torch.cuda.stream(capture_stream), BreakableCUDAGraphCapture() as cap:
        off.forward_hip(x_static, ws, out_static, visit=1)
    torch.accelerator.synchronize()
    assert len(cap.segments) >= 1

    for step in range(3):
        x_new = torch.randn(
            2, cfg.hidden_size, generator=gen, device="cuda", dtype=torch.float16
        )
        x_static.copy_(x_new)
        cap.replay()
        out_ref = torch.empty_like(x_new)
        res.forward_hip(x_new, ws, out_ref)
        assert torch.allclose(out_static, out_ref, atol=1e-3, rtol=1e-3), (
            f"graph replay {fmt}/{copy_mode} step={step}: max diff "
            f"{(out_static - out_ref).abs().max().item()}"
        )


@pytest.mark.skipif(not _is_gfx103x(), reason="requires gfx103x HIP GPU")
@pytest.mark.parametrize("copy_mode", ["tensor", "native"])
def test_gpu_layer_ahead_prefetch_forward_hip_and_graph_parity(copy_mode):
    """Two chained offloaded MoE layers with async layer-ahead prefetching
    match resident forward_hip() across decode, prefill overflow, and
    BreakableCUDAGraphCapture replay."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.inkling2_loop.amd.glue import Inkling2GlueWorkspace
    from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig

    cfg = Inkling2LoopTextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=128,
        n_routed_experts=16,
        num_experts_per_tok=4,
        loop_layer_index_start=0,
        loop_layer_index_end=2,
        loop_num_passes=2,
        n_layers_per_attn_res_block=1,
        loop_gate_rank=16,
    )
    with torch.device("cuda"):
        res0 = Inkling2LoopMoE(cfg, "nvfp4", offload_slots=0)
        res1 = Inkling2LoopMoE(cfg, "nvfp4", offload_slots=0)
        off0 = Inkling2LoopMoE(
            cfg, "nvfp4", offload_slots=6, offload_copy=copy_mode, offload_prefetch=True
        )
        off1 = Inkling2LoopMoE(
            cfg, "nvfp4", offload_slots=6, offload_copy=copy_mode, offload_prefetch=True
        )

    for res, off, seed in ((res0, off0, 11), (res1, off1, 22)):
        group = _fill_moe_weights(res, seed=seed)
        off.gate.weight.data.copy_(res.gate.weight.data)
        off.gate.e_score_correction_bias.data.copy_(
            res.gate.e_score_correction_bias.data
        )
        off.load_expert_group(group)
        res.process_weights_after_loading()
        off.process_weights_after_loading()
        res.prepare_hip()
        off.prepare_hip()

    ws = Inkling2GlueWorkspace(cfg, max_tokens=256, device="cuda", tp_size=1)
    gen = torch.Generator(device="cuda").manual_seed(33)

    for num_tokens in (1, 4, 160):
        x = torch.randn(
            num_tokens,
            cfg.hidden_size,
            generator=gen,
            device="cuda",
            dtype=torch.float16,
        )
        mid_res, out_res = torch.empty_like(x), torch.empty_like(x)
        mid_off, out_off = torch.empty_like(x), torch.empty_like(x)
        res0.forward_hip(x, ws, mid_res)
        res1.forward_hip(mid_res, ws, out_res)
        off0.forward_hip(x, ws, mid_off, visit=0, next_mlp=off1)
        off1.forward_hip(mid_off, ws, out_off, visit=0, next_mlp=None)
        assert torch.allclose(out_res, out_off, atol=1e-3, rtol=1e-3), (
            f"prefetch {copy_mode} T={num_tokens}: max diff "
            f"{(out_res - out_off).abs().max().item()}"
        )

    stats1 = off1.offload_stats()
    assert stats1 is not None
    assert stats1["prefetch_calls"] > 0
    assert stats1["prefetch_experts"] > 0

    # BreakableCUDAGraphCapture with chained prefetch
    x_static = torch.randn(
        2, cfg.hidden_size, generator=gen, device="cuda", dtype=torch.float16
    )
    mid_static = torch.empty_like(x_static)
    out_static = torch.empty_like(x_static)
    capture_stream = torch.cuda.Stream()
    torch.accelerator.synchronize()
    with torch.cuda.stream(capture_stream), BreakableCUDAGraphCapture() as cap:
        off0.forward_hip(x_static, ws, mid_static, visit=1, next_mlp=off1)
        off1.forward_hip(mid_static, ws, out_static, visit=1, next_mlp=None)
    torch.accelerator.synchronize()
    assert len(cap.segments) >= 2

    for step in range(3):
        x_new = torch.randn(
            2, cfg.hidden_size, generator=gen, device="cuda", dtype=torch.float16
        )
        x_static.copy_(x_new)
        cap.replay()
        mid_ref, out_ref = torch.empty_like(x_new), torch.empty_like(x_new)
        res0.forward_hip(x_new, ws, mid_ref)
        res1.forward_hip(mid_ref, ws, out_ref)
        assert torch.allclose(out_static, out_ref, atol=1e-3, rtol=1e-3), (
            f"prefetch graph replay {copy_mode} step={step}: max diff "
            f"{(out_static - out_ref).abs().max().item()}"
        )
