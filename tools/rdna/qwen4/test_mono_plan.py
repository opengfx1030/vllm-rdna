# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Qwen4 monokernel planner (``vllm/models/qwen4_prep``).

No torch, no GPU and no vLLM import: the planner is loaded from its file so
the tests run on any box. The anchor assertions are that the published
Flash-Next config reproduces today's monokernel plan exactly, and that the
``Limits`` the planner uses still equal the constants in the kernel sources.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "vllm/models/qwen4_prep/mono_plan.py"
INIT_PATH = ROOT / "vllm/models/qwen4_prep/__init__.py"
LAYOUT = ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/layout.py"
K2 = ROOT / "csrc/rocm/mono_k2_rdna2.cu"
GDN = ROOT / "csrc/rocm/mono_gdn_rdna2.cu"
QSA = ROOT / "csrc/rocm/mono_qsa_decode_rdna2.cu"
MONO_QSA = ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/qsa.py"
SWITCH = ROOT / "vllm/models/qwen4_exp/amd/mono_decode_rdna2.py"
OPTIONS = ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/__init__.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the class
    # body executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = _load("qwen4_mono_plan", PLAN_PATH)
gate_mod = _load("qwen4_prep_init", INIT_PATH)


def _number(path: Path, name: str) -> int:
    """Read the integer assigned to ``name`` in C++ or Python source.

    Handles ``constexpr int a = 1, b = 2;`` by splitting the declarator list,
    and ignores the same name in another namespace by taking the first match.
    """
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", "//")) or name not in stripped:
            continue
        for part in stripped.replace(";", ",").split(","):
            if name not in part or "=" not in part:
                continue
            digits = ""
            for ch in part.split("=", 1)[1].strip():
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
                elif ch in "+ ":
                    continue
                else:
                    break
            if digits:
                return int(digits)
    raise AssertionError(f"{name} not found in {path}")


def test_limits_match_the_kernel_sources() -> None:
    lim = plan_mod.Limits()
    assert lim.threads == _number(K2, "kThreads")
    assert lim.waves == lim.threads // 32  # kWaves = kThreads / 32
    assert lim.max_rows_k2 == _number(K2, "kMaxM")
    assert lim.max_topk == _number(K2, "kTopK")
    assert lim.max_local_experts == _number(K2, "kLocalE")
    assert lim.hc_count == _number(K2, "kHC")
    assert lim.max_rows_k1 == _number(LAYOUT, "MAX_ROWS")
    assert lim.max_rows_gdn == _number(GDN, "kCapRows")
    assert lim.qsa_head_dim == _number(QSA, "kHD")
    assert lim.max_group_qsa == _number(QSA, "kMaxG")
    assert lim.max_splits_qsa == _number(QSA, "kMaxS")
    assert lim.min_rows_qsa == _number(MONO_QSA, "MIN_ROWS")
    assert lim.max_rows_qsa == 64  # _hip_ok upper bound in mono/rdna2/qsa.py
    assert lim.k2_gate_rows == _number(OPTIONS, "k2_max_rows")
    assert lim.max_rows_k1 == _number(OPTIONS, "k1_max_rows")
    assert lim.max_rows_gdn == _number(OPTIONS, "gdn_max_rows")
    assert lim.hw_queues == _number(OPTIONS, "hw_queues")
    assert lim.max_rows_k1 == _number(SWITCH, "_MAX_ROWS")
    assert lim.max_rows_gdn == _number(SWITCH, "_GDN_MAX_ROWS")


def test_residency_cap_reproduces_the_measured_grid() -> None:
    lim = plan_mod.Limits()
    # K2 build: 128 VGPR, 29 KiB LDS -> two WGs per CU, 144 on a V620.
    assert lim.residency_cap(128, 29 * 1024) == 144
    # One workgroup per CU when LDS alone allows only one.
    assert lim.residency_cap(128, 64 * 1024) == 72
    # Wider code with 256 VGPR: still two WGs per CU (8 waves each).
    assert lim.residency_cap(256, 29 * 1024) == 72


def test_flash_next_plan_is_todays_plan() -> None:
    plan = plan_mod.flash_next_plan(tp=4, spec_tokens=2)
    assert plan.runs
    assert plan.geometry.num_layers == 48
    assert len(plan.layers) == 48
    assert sum(1 for x in plan.layers if x.layer_type == "linear_attention") == 36
    assert sum(1 for x in plan.layers if x.layer_type == "full_attention") == 12
    # One PLE layer: ple_layer_ids [2] is 1-based.
    assert [x.index for x in plan.layers if "ple" in x.parts] == [1]
    gdn = ("k1", "gdn", "attn_ar", "k2", "ffn_ar")
    assert plan.layers[0].parts == gdn
    assert plan.layers[1].parts == ("ple",) + gdn
    qsa = (
        "k1",
        "qsa_prepare",
        "qsa_indexer",
        "qsa_mqa",
        "qsa_select",
        "qsa_sparse",
        "attn_ar",
        "k2",
        "ffn_ar",
    )
    assert plan.layers[3].parts == qsa
    # Today's gates: K1 24 rows, K2 8, GDN 64, QSA from 3; widths 3/6/12/24.
    rows = {s.part: s.rows for s in plan.layers[0].stages}
    assert rows["k1"] == 24
    assert rows["gdn"] == 24
    assert rows["k2"] == 8
    assert plan.widths == (3, 6, 12, 24)
    assert plan.row_cap == 24
    assert plan.grid_wg == 144
    assert plan.limits.hw_queues == 4
    assert not plan.refusals
    assert not plan.warnings
    # One drafter layer, one attention block per layer.
    assert len(plan.drafter) == 1
    assert plan.drafter[0].parts[0] == "k1"


def test_flash_next_geometry_and_sharding() -> None:
    tp4 = plan_mod.flash_next_geometry(4)
    assert (tp4.hidden, tp4.hc_width, tp4.hc_down_rows) == (2560, 10240, 336)
    assert tp4.local_experts == 128
    assert (tp4.q_heads_local, tp4.kv_heads_local, tp4.qsa_group) == (6, 2, 3)
    assert (tp4.lin_k_heads_local, tp4.lin_v_heads_local) == (4, 12)
    tp2 = plan_mod.flash_next_geometry(2)
    assert tp2.local_experts == 256
    # 2 KV heads over TP=2 is one head per rank: group 12, above the HIP
    # sparse-decode build (<= 8), so that part falls back to Triton.
    assert (tp2.q_heads_local, tp2.kv_heads_local, tp2.qsa_group) == (12, 1, 12)


def test_flash_next_tp2_needs_the_offload_slots() -> None:
    geo = plan_mod.flash_next_geometry(2)
    plain = plan_mod.plan_for(geo, spec_tokens=2)
    assert not plain.runs
    assert any("local experts" in r for r in plain.refusals)
    offload = plan_mod.plan_for(geo, spec_tokens=2, offload=True)
    assert offload.runs
    # TP2 keeps the K2 width gate; the sparse-decode group does fall back.
    assert {s.part: s.rows for s in offload.layers[0].stages}["k2"] == 8
    assert any("query heads per KV head" in w for w in offload.warnings)


def test_tp1_is_refused() -> None:
    geo = plan_mod.flash_next_geometry(1)
    plan = plan_mod.plan_for(geo, spec_tokens=0)
    assert not plan.runs
    assert any("local experts" in r for r in plan.refusals)


def test_widths_ladder_and_mtp_depth() -> None:
    assert plan_mod.widths_for(2) == (3, 6, 12, 24)
    assert plan_mod.widths_for(4) == (5, 10, 20, 40)
    geo = plan_mod.flash_next_geometry(4)
    four = plan_mod.plan_for(geo, spec_tokens=4)
    assert four.row_cap == 40
    assert any("k1 rows 40" in w for w in four.warnings)
    # A raised K2 build width is a planner input, not a silent change.
    raised = plan_mod.plan_for(geo, spec_tokens=4, k2_rows=40, gdn_rows=40)
    rows = {s.part: s.rows for s in raised.layers[0].stages}
    assert rows["k2"] == 24  # clamped by the live kernel's build width
    assert any("k2 rows 40" in w for w in raised.warnings)


def test_dense_qwen4_plan_has_no_router_or_experts() -> None:
    geo = plan_mod.Geometry.from_hf(plan_mod.QWEN4_27B_LEAK, tp=4)
    plan = plan_mod.plan_for(geo, spec_tokens=4)
    assert plan.runs
    assert geo.dense
    assert geo.local_experts == 0
    assert (geo.qsa_group, geo.hc_width) == (6, 20480)
    stage_parts = set(plan.layer_parts("linear_attention"))
    assert "k2" in stage_parts and "gdn" in stage_parts
    traffic = plan.traffic
    assert all(t.expert_bytes == 0 for t in traffic.values())
    dense = {t.dense_bytes for t in traffic.values()}
    assert len(dense) == 1  # the weight sweep does not depend on the batch


def test_qsa_fallbacks_are_per_part() -> None:
    cfg = dict(plan_mod.FLASH_NEXT_CONFIG)
    cfg["head_dim"] = 128
    geo = plan_mod.Geometry.from_hf(cfg, tp=4)
    plan = plan_mod.plan_for(geo, spec_tokens=2)
    assert plan.runs
    assert any("head_dim 128" in w for w in plan.warnings)
    sparse = [s for s in plan.layers[3].stages if s.part == "qsa_sparse"]
    assert sparse[0].engine == "triton"

    wide = plan_mod.flash_next_geometry(1)
    plan = plan_mod.plan_for(wide, spec_tokens=2)
    assert any("query heads per KV head" in w for w in plan.warnings)


def test_traffic_model_is_monotone_and_tp_aware() -> None:
    geo4 = plan_mod.flash_next_geometry(4)
    plan4 = plan_mod.plan_for(geo4, spec_tokens=2, context=16384)
    small = plan4.traffic[3]
    large = plan4.traffic[24]
    assert small.expert_bytes < large.expert_bytes
    assert small.kv_bytes < large.kv_bytes
    assert 5.0 < large.floor_ms(plan4.limits.bandwidth_gbs) < 60.0
    # Experts dominate a width-24 MoE step; the dense parts do not scale.
    assert large.expert_bytes > large.dense_bytes
    assert small.dense_bytes == large.dense_bytes

    geo2 = plan_mod.flash_next_geometry(2)
    plan2 = plan_mod.plan_for(geo2, spec_tokens=2, offload=True, context=16384)
    assert plan2.traffic[24].total_bytes > plan4.traffic[24].total_bytes


def test_architecture_gate_leaves_flash_next_alone() -> None:
    assert gate_mod.is_qwen4_architecture("Qwen4ForCausalLM")
    assert gate_mod.is_qwen4_architecture("Qwen4ForConditionalGeneration")
    assert not gate_mod.is_qwen4_architecture("Qwen4ExpForCausalLM")
    assert not gate_mod.is_qwen4_architecture("Qwen4ExpForConditionalGeneration")
    assert not gate_mod.is_qwen4_architecture("Qwen3ForCausalLM")
    arch = plan_mod.FLASH_NEXT_CONFIG["architectures"][0]
    assert arch.startswith("Qwen4Exp")
    assert not gate_mod.is_qwen4_architecture(arch)


def test_synthetic_configs() -> None:
    cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "full_attention_interval": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 24,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 512,
        "hc_count": 4,
        "hc_lowrank": 320,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
    geo = plan_mod.Geometry.from_hf(cfg, tp=4)
    assert len(geo.layer_types) == 60
    assert geo.layer_types[3] == "full_attention"
    assert geo.local_experts == 64
    plan = plan_mod.plan_for(geo, spec_tokens=0)
    assert not plan.runs  # top-8 is a refusal, not a fallback
    assert any("top-8" in r for r in plan.refusals)

    cfg["num_experts_per_tok"] = 10
    geo = plan_mod.Geometry.from_hf(cfg, tp=4)
    plan = plan_mod.plan_for(geo, spec_tokens=0)
    assert plan.runs
    assert any("head_dim 128" in w for w in plan.warnings)
    assert plan.traffic[8].floor_ms(plan.limits.bandwidth_gbs) > 0.0


def test_plan_is_deterministic() -> None:
    a = plan_mod.flash_next_plan(tp=4)
    b = plan_mod.flash_next_plan(tp=4)
    assert a.as_dict() == b.as_dict()


def test_bad_layer_types_raise() -> None:
    cfg = dict(plan_mod.FLASH_NEXT_CONFIG)
    cfg["layer_types"] = ["linear_attention"] * 47
    with pytest.raises(ValueError, match="layer_types"):
        plan_mod.Geometry.from_hf(cfg, tp=4)
    cfg = dict(plan_mod.FLASH_NEXT_CONFIG)
    cfg.pop("layer_types")
    with pytest.raises(ValueError, match="full_attention_interval"):
        plan_mod.Geometry.from_hf(cfg, tp=4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
