# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HI2 (rdna_hint_flex) Inkling checkpoint: format declaration, name map, slices.

CPU only.  The container module and the loader are loaded by file path (they do
not import ``vllm``); the MoE wiring test imports the package, which needs a
built ``vllm`` (the box tree has the compiled extensions).

Run on the box::

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_hi2_loader.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hi2 = _load("_i2_hi2_under_test", "vllm/models/inkling2_loop/amd/hi2.py")
loader = _load("_i2_loader_hi2_under_test", "vllm/models/inkling2_loop/amd/loader.py")


# ---------------------------------------------------------------------------
# name mapping
# ---------------------------------------------------------------------------


def test_hf_name_map_inverts_the_reference_map():
    ref = _load("_i2_ref_checkpoint", "tools/rdna/inkling2/ref_checkpoint.py")
    pairs = ref.build_name_map(6)
    assert len(pairs) > 100
    for serving, hf_name in pairs.items():
        assert loader.hf_to_serving(hf_name) == serving, (hf_name, serving)
    # the exporter composes the four message queries into one HF tensor
    assert (
        loader.hf_to_serving("model.language_model.message_input.query")
        == "model.llm.message_input.query"
    )
    # names identical in both layouts are left alone
    assert loader.hf_to_serving("lm_head.weight") is None
    assert (
        loader.hf_to_serving("model.visual.backbone.embeddings.class_embedding") is None
    )
    assert (
        loader.hf_to_serving(
            "model.language_model.layers.0.mlp.experts.gate_up.hi2_codes"
        )
        is None
    )


def test_parse_tensor_name():
    got = hi2.parse_tensor_name(
        "model.language_model.layers.7.mlp.experts.gate_up.hi2_col_scale"
    )
    assert got == (7, "gate_up", "col_scale")
    assert (
        hi2.parse_tensor_name("model.language_model.layers.7.mlp.gate.weight") is None
    )
    assert (
        hi2.parse_tensor_name("model.llm.layers.7.mlp.experts.gate_up.hi2_codes")
        is None
    )


# ---------------------------------------------------------------------------
# format declaration
# ---------------------------------------------------------------------------

_REAL_QUANT_CONFIG = {
    "quant_method": "rdna_hint_flex",
    "version": "3.2",
    "mode": "flex",
    "target_bpw": 3.5,
    "moe_layout": "stacked",
    "moe_stacks": {"gate_up": ["gate_proj", "up_proj"], "down": ["down_proj"]},
}


def test_quantizer_config_declares_hi2_experts(tmp_path):
    pytest.importorskip("vllm.transformers_utils.repo_utils")
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": _REAL_QUANT_CONFIG})
    )
    assert loader.read_expert_formats(str(tmp_path), None, 24) == ["hi2"] * 24
    assert loader.hi2_expert_formats(_REAL_QUANT_CONFIG, 6) == ["hi2"] * 6


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"moe_layout": "per-expert"}, "stacked"),
        ({"version": "3.1"}, "version"),
        ({"mode": "other"}, "mode"),
        (
            {
                "moe_stacks": {
                    "gate_up": ["up_proj", "gate_proj"],
                    "down": ["down_proj"],
                }
            },
            "gate_up",
        ),
        ({"moe_stacks": {"gate_up": ["gate_proj", "up_proj"]}}, "down"),
    ],
)
def test_a_directory_that_is_not_the_stacked_layout_is_refused(patch, message):
    cfg = {**_REAL_QUANT_CONFIG, **patch}
    with pytest.raises(ValueError, match=message):
        loader.hi2_expert_formats(cfg, 4)


def test_nvfp4_still_uses_hf_quant_config(tmp_path):
    pytest.importorskip("vllm.transformers_utils.repo_utils")
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "modelopt_mixed"}})
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "parameter_groups": [
                        {
                            "compute_dtype": "nvfp4",
                            "tensors": [
                                f"model.llm.layers.{i}.mlp.experts.w13_weight"
                                for i in range(2)
                            ],
                        }
                    ]
                }
            }
        )
    )
    assert loader.read_expert_formats(str(tmp_path), None, 2) == ["nvfp4"] * 2


# ---------------------------------------------------------------------------
# synthetic stacks
# ---------------------------------------------------------------------------


def make_stack(
    experts: int,
    rows_per_part: int,
    k: int,
    parts: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """A valid stack with mixed INT2/INT4 rows and mixed block sizes."""
    g = torch.Generator().manual_seed(seed)
    rows = experts * parts * rows_per_part
    two_bit = torch.randint(0, 2, (rows,), generator=g).bool()
    row_nbits = torch.where(two_bit, torch.tensor(2), torch.tensor(4)).to(torch.uint8)
    blocks = torch.tensor([16, 32, 64])[torch.randint(0, 3, (rows,), generator=g)]
    blocks = torch.where((k % blocks) == 0, blocks, torch.full_like(blocks, 16))
    row_block = blocks.to(torch.uint8)
    words = k * row_nbits.to(torch.int64) // 32
    code_off = torch.zeros(rows + 1, dtype=torch.int64)
    code_off[1:] = torch.cumsum(words, 0)
    nblocks = k // row_block.to(torch.int64)
    param_off = torch.zeros(rows + 1, dtype=torch.int64)
    param_off[1:] = torch.cumsum(nblocks, 0)
    codes = torch.randint(0, 2**31, (int(code_off[-1]),), generator=g).to(
        torch.int32
    ) | (torch.randint(0, 2, (int(code_off[-1]),), generator=g).to(torch.int32) << 31)
    blk_params = torch.randint(
        0, 256, (2 * int(param_off[-1]),), generator=g, dtype=torch.uint8
    )
    # small constants: the packed words are arbitrary (full int32 range), so the
    # step must be tiny for the decoded weights to stay in fp16 range
    row_consts = torch.stack(
        [
            (torch.rand(rows, generator=g) - 0.5) * 0.01,
            torch.rand(rows, generator=g) * 0.01,
            torch.rand(rows, generator=g) / 2**28,
        ],
        dim=1,
    ).to(torch.float16)
    return {
        "codes": codes,
        "code_off": code_off.to(torch.int32),
        "blk_params": blk_params,
        "param_off": param_off.to(torch.int32),
        "row_consts": row_consts,
        "row_nbits": row_nbits,
        "row_block": row_block,
        "had": torch.randint(0, 2, (experts, k // 16), generator=g).to(torch.uint8),
        "col_scale": (torch.rand(experts, k, generator=g) + 0.5).to(torch.float16),
        "part_off": (
            torch.arange(0, experts * parts + 1, dtype=torch.int64) * rows_per_part
        ).to(torch.int32),
    }


def test_a_valid_stack_is_accepted_and_a_bad_offset_is_refused():
    t = make_stack(4, 32, 128, 2, seed=3)
    hi2.validate_stack(t, 2, 128, 4)
    broken = dict(t)
    broken["code_off"] = t["code_off"].clone()
    broken["code_off"][1] += 1
    with pytest.raises(ValueError, match="code_off"):
        hi2.validate_stack(broken, 2, 128, 4)
    with pytest.raises(ValueError, match="experts"):
        hi2.validate_stack(t, 2, 128, 8)


def _same_rows(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a, b)


def test_tp_slice_of_the_fused_gate_up_keeps_the_units_rows():
    experts, inter, hidden = 4, 64, 128
    full = make_stack(experts, inter, hidden, 2, seed=11)
    hi2.validate_stack(full, 2, hidden, experts)
    rank = hi2.slice_stack(
        full,
        "w13",
        parts=2,
        rows_per_part=inter,
        k=hidden,
        lo=32,
        hi=64,
        experts=experts,
    )
    hi2.validate_stack(rank, 2, hidden, experts)
    assert rank["part_off"].tolist() == [0] + [(e + 1) * 32 for e in range(2 * experts)]
    assert rank["had"].shape == full["had"].shape
    assert rank["col_scale"].shape == full["col_scale"].shape
    for e in range(experts):
        for p in range(2):
            want = hi2.decode_part(full, e, p, 2)[32:64]
            got = hi2.decode_part(rank, e, p, 2)
            assert _same_rows(got, want), (e, p)


def test_tp_slice_of_the_down_projection_takes_64_aligned_columns():
    experts, hidden, inter = 4, 128, 128
    full = make_stack(experts, hidden, inter, 1, seed=17)
    hi2.validate_stack(full, 1, inter, experts)
    rank = hi2.slice_stack(
        full,
        "w2",
        parts=1,
        rows_per_part=hidden,
        k=inter,
        lo=64,
        hi=128,
        experts=experts,
    )
    hi2.validate_stack(rank, 1, 64, experts)
    assert rank["had"].shape == (experts, 4)
    assert rank["col_scale"].shape == (experts, 64)
    for e in range(experts):
        want = hi2.decode_part(full, e, 0, 1)[:, 64:128]
        got = hi2.decode_part(rank, e, 0, 1)
        assert _same_rows(got, want), e


def test_a_column_slice_must_be_64_aligned():
    t = make_stack(2, 32, 64, 1, seed=5)
    with pytest.raises(ValueError, match="64-aligned"):
        hi2.slice_stack(
            t,
            "w2",
            parts=1,
            rows_per_part=32,
            k=64,
            lo=16,
            hi=32,
            experts=2,
        )


# ---------------------------------------------------------------------------
# module wiring
# ---------------------------------------------------------------------------


def _init_gate(layer, seed: int = 0) -> None:
    """The router weight is ``torch.empty`` until a checkpoint fills it; give the
    two forward tests a small deterministic one so uninitialized memory cannot
    turn the logits into NaN."""
    g = torch.Generator().manual_seed(seed)
    layer.gate.weight.data.copy_(
        torch.randn(layer.gate.weight.shape, generator=g) * 0.02
    )


def _config(hidden: int, inter: int, experts: int, top_k: int):
    pytest.importorskip("vllm.models.inkling2_loop.amd.moe")
    from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig

    return Inkling2LoopTextConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=experts,
        num_experts_per_tok=top_k,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )


def test_the_moe_collects_both_groups_and_runs_the_reference():
    from vllm.models.inkling2_loop.amd.moe import Inkling2LoopMoE

    experts, hidden, inter, top_k = 4, 128, 64, 2
    cfg = _config(hidden, inter, experts, top_k)
    layer = Inkling2LoopMoE(cfg, "hi2", prefix="test.mlp").eval()
    _init_gate(layer)

    w13 = make_stack(experts, inter, hidden, 2, seed=21)
    w2 = make_stack(experts, hidden, inter, 1, seed=22)
    for group, stack in (("gate_up", w13), ("down", w2)):
        for field in hi2.STACK_FIELDS:
            layer.load_hi2_field(group, field, stack[field])
    layer.process_weights_after_loading()
    assert layer.w13_had.shape == (experts, hidden // 16)
    assert layer.w2_had.shape == (experts, inter // 16)
    assert layer.w2_part_off.tolist() == [e * hidden for e in range(experts + 1)]

    torch.manual_seed(0)
    x = torch.randn(3, hidden, dtype=torch.float16)
    out = layer.forward_reference(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()
    # the eager reference is deterministic
    assert torch.equal(out, layer.forward_reference(x))


def test_an_incomplete_group_is_refused():
    from vllm.models.inkling2_loop.amd.moe import Inkling2LoopMoE

    cfg = _config(128, 64, 4, 2)
    layer = Inkling2LoopMoE(cfg, "hi2", prefix="test.mlp")
    stack = make_stack(4, 64, 128, 2, seed=31)
    layer.load_hi2_field("gate_up", "codes", stack["codes"])
    with pytest.raises(ValueError, match="incomplete"):
        layer.process_weights_after_loading()
    with pytest.raises(ValueError, match="duplicate"):
        layer.load_hi2_field("gate_up", "codes", stack["codes"])


def test_an_unsmoothed_group_finalizes_at_the_end_of_loading():
    """No ``hi2_col_scale`` in the checkpoint: the group completes in
    ``process_weights_after_loading`` and the kernels see an empty smoothing vector."""
    from vllm.models.inkling2_loop.amd.moe import Inkling2LoopMoE

    experts, hidden, inter = 4, 128, 64
    cfg = _config(hidden, inter, experts, 2)
    layer = Inkling2LoopMoE(cfg, "hi2", prefix="test.mlp").eval()
    _init_gate(layer, seed=1)
    for group, stack in (
        ("gate_up", make_stack(experts, inter, hidden, 2, seed=41)),
        ("down", make_stack(experts, hidden, inter, 1, seed=42)),
    ):
        stack.pop("col_scale")
        for field in (*hi2.FIELDS, "part_off"):
            layer.load_hi2_field(group, field, stack[field])
    layer.process_weights_after_loading()
    for stage in ("w13", "w2"):
        weights = layer._hi2_weights(stage)
        assert weights["col_scale"].numel() == 0
        assert layer._hi2_transform[stage] is True
    x = torch.randn(2, hidden, dtype=torch.float16)
    assert torch.isfinite(layer.forward_reference(x).float()).all()


def test_the_hi2_serving_workspace_reuses_the_gate_up_scratch():
    pytest.importorskip("vllm.models.inkling2_loop.amd.glue")
    from vllm.models.inkling2_loop.amd import glue

    cfg = _config(128, 64, 4, 2)
    ws = glue.Inkling2GlueWorkspace(cfg, 8, "cpu", hi2=True)
    assert ws.moe_x_view(8, cfg.hidden_size).shape == (16, cfg.hidden_size)
    assert ws.moe_x_view(8, cfg.intermediate_size).shape == (16, 64)
    sorted_ids, expert_ids, ntpp = ws.moe64_views(8)
    assert sorted_ids.numel() == 16 + 4 * 63 and expert_ids.numel() == 5
    assert ntpp.numel() == 1
    with pytest.raises(ValueError, match="2 \\* intermediate"):
        glue.Inkling2GlueWorkspace(_config(192, 64, 4, 2), 8, "cpu", hi2=True)
