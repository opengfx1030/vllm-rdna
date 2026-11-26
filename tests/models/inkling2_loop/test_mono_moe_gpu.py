# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU reference verification for ``inkling2_mono_moe_k2`` (NVFP4 + MXFP8)."""

from __future__ import annotations

import pytest
import torch

from vllm.models.inkling2_loop.amd import moe as moe_mod
from vllm.models.inkling2_loop.amd.glue import Inkling2GlueWorkspace
from vllm.models.inkling2_loop.amd.moe import Inkling2LoopMoE
from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig


def _is_gfx103x() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx103")


def _fill_weights(layer: Inkling2LoopMoE, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    e, h, inter = layer.num_experts, layer.hidden_dim, layer.intermediate_dim
    layer.gate.weight.data.copy_(
        torch.randn(e, h, generator=gen, dtype=torch.float32) * 0.1
    )
    layer.gate.e_score_correction_bias.data.copy_(
        torch.randn(e, generator=gen, dtype=torch.float32) * 0.05
    )
    group: dict[str, torch.Tensor] = {}
    if layer.quant_format == "nvfp4":
        for stage, n, k in (("w13", 2 * inter, h), ("w2", h, inter)):
            group[f"{stage}_weight"] = torch.randint(
                0, 256, (e, n, k // 2), generator=gen, dtype=torch.uint8
            )
            raw_s = torch.randint(
                0x30, 0x48, (e, n, k // 16), generator=gen, dtype=torch.uint8
            )
            group[f"{stage}_weight_scale"] = raw_s.view(torch.float8_e4m3fn)
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
            group[f"{stage}_weight_scale"] = torch.randint(
                113, 119, (e, n, k // 32), generator=gen, dtype=torch.uint8
            )
    layer.load_expert_group(group)


@pytest.mark.skipif(not _is_gfx103x(), reason="gfx1030 RDNA2 required")
@pytest.mark.parametrize("quant_format", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("num_tokens", [1, 2])
def test_mono_moe_k2_matches_reference(
    quant_format: str,
    tp_size: int,
    num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(torch.ops._rocm_C, "inkling2_mono_moe_k2"):
        pytest.skip("inkling2_mono_moe_k2 not built into _rocm_C")

    cfg = Inkling2LoopTextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=128,
        n_routed_experts=32,
        num_experts_per_tok=8,
        loop_layer_index_start=0,
        loop_layer_index_end=2,
        loop_num_passes=2,
        n_layers_per_attn_res_block=1,
        loop_gate_rank=16,
    )
    with torch.device("cuda"):
        layer = Inkling2LoopMoE(cfg, quant_format, tp_size=tp_size)
    _fill_weights(layer, seed=13 if quant_format == "nvfp4" else 29)
    layer.process_weights_after_loading()
    layer.prepare_hip()

    monkeypatch.setenv("VLLM_INKLING2_MONO_MAX_ROWS", "4")
    ws = Inkling2GlueWorkspace(cfg, max_tokens=16, device="cuda", tp_size=tp_size)
    gen = torch.Generator(device="cuda").manual_seed(300 + num_tokens)
    x = torch.randn(
        num_tokens,
        cfg.hidden_size,
        generator=gen,
        device="cuda",
        dtype=torch.float16,
    )
    out_ref = torch.empty_like(x)
    out_mono = torch.empty_like(x)

    monkeypatch.setattr(moe_mod, "_MONO_ON", False)
    layer.forward_hip(x, ws, out_ref, max_query_len=1)

    monkeypatch.setattr(moe_mod, "_MONO_ON", True)
    monkeypatch.setattr(moe_mod, "_MONO_MAX_ROWS", 4)
    assert layer.can_use_mono(num_tokens, max_query_len=1)
    layer.forward_hip(x, ws, out_mono, max_query_len=1)
    torch.accelerator.synchronize()

    ref64 = out_ref.double()
    mono64 = out_mono.double()
    max_abs = (mono64 - ref64).abs().max().item()
    denom = ref64.abs().clamp(min=1e-6)
    max_rel = ((mono64 - ref64).abs() / denom).max().item()
    print(
        f"[mono_moe {quant_format} tp={tp_size} T={num_tokens}] "
        f"max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}"
    )
    assert max_abs <= 5e-3, (
        f"mono_moe {quant_format} tp={tp_size} T={num_tokens}: max_abs={max_abs}"
    )
