# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HI2 expert decode and routed MoE forward vs the quantizer on its tiny output.

CPU only, no GPU and no lease.  The module builds the fixture with
``tools/inkling_hi2/make_tiny_hi2.py``: the fork's tiny Inkling-2 checkpoint is
quantized to a stacked HI2 directory by the quantizer's own pipeline, then

1. every decoded expert weight must equal the quantizer's ``hi2.format.dequantize``
   of the same stack (float64, bit for bit),
2. a TP slice decoded from this rank's stack must equal the columns of the full
   decode (the smoothed / rotated basis too),
3. the routed MoE forward of ``amd.hi2.reference_moe`` and of ``ExpertsReference``
   must match the transformers reference onto which the quantizer unfuses its
   dequantized experts (fp32 tolerance).

The quantizer and transformers' ``inkling2_loop`` are only present on the box:

    HI2_SRC=$HOME/w4a8_runs/inkling-hi2/tree \\
    PYTHONPATH=$HI2_SRC:$HOME/w4a8_runs/pydeps/tfm-main:$TREE \\
    python -m pytest tests/models/inkling2_loop/test_hi2_reference.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm.models.inkling2_loop.amd import hi2
from vllm.models.inkling2_loop.amd.moe import (
    ExpertsReference,
    Inkling2LoopTopKRouter,
)
from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_TOOLS = REPO_ROOT / "tools" / "rdna" / "inkling2"
GENERATOR = REPO_ROOT / "tools" / "inkling_hi2" / "make_tiny_hi2.py"

_LAYER = 0
_RAW_FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
    "col_scale",
    "part_off",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """``(stock dir, HI2 dir)`` of a freshly quantized tiny Inkling-2 checkpoint."""
    if not (TINY_TOOLS / "make_tiny_checkpoint.py").exists():
        pytest.skip("fork tiny-checkpoint tool missing")
    pytest.importorskip("hi2", reason="quantizer not importable (set PYTHONPATH)")
    pytest.importorskip(
        "transformers.models.inkling2_loop.modeling_inkling2_loop",
        reason="transformers' inkling2_loop is not installed",
    )
    cached = os.environ.get("INKLING2_TINY_OUT")
    if cached and (Path(cached) / "hi2" / "model.safetensors.index.json").exists():
        return Path(cached) / "stock", Path(cached) / "hi2"
    out = tmp_path_factory.mktemp("tiny_hi2")
    generator = _load_module("inkling2_make_tiny_hi2", GENERATOR)
    hi2_dir = generator.make(out, TINY_TOOLS)
    return out / "stock", hi2_dir


def _read_stack(directory: Path, base: str) -> dict[str, torch.Tensor]:
    """The raw stacked fields of ``base`` (canonical names, one safetensors copy)."""
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shard = directory / index["weight_map"][f"{base}.hi2_codes"]
    raw = {}
    with safe_open(str(shard), "pt") as f:
        for field in _RAW_FIELDS:
            key = f"{base}.hi2_{field}"
            if key in index["weight_map"]:
                raw[field] = f.get_tensor(key)
    return hi2.canonicalize_stack(raw)


def _quantizer_stack(directory: Path, base: str):
    from hi2.format import dequantize
    from hi2.stack import Hi2Checkpoint

    return [
        [dequantize(part) for part in parts]
        for parts in Hi2Checkpoint(directory).members(base)
    ]


def _assert_same(
    got: torch.Tensor, want: torch.Tensor, what: str, atol: float = 0.0
) -> None:
    if torch.equal(got, want):
        return
    diff = (got.double() - want.double()).abs().max().item()
    assert diff <= atol, f"{what}: max |diff| {diff:.3e} > {atol:.3e}"


def _stacks(tiny, layer: int):
    stock, directory = tiny
    text = f"model.language_model.layers.{layer}.mlp.experts"
    config = json.loads((stock / "config.json").read_text())["text_config"]
    hidden, inter, experts = (
        config["hidden_size"],
        config["intermediate_size"],
        config["n_routed_experts"],
    )
    w13 = _read_stack(directory, f"{text}.gate_up")
    w2 = _read_stack(directory, f"{text}.down")
    hi2.validate_stack(w13, 2, hidden, experts)
    hi2.validate_stack(w2, 1, inter, experts)
    return hidden, inter, experts, w13, w2


@pytest.mark.parametrize("layer", [0, 5])
def test_decoded_experts_match_the_quantizer(tiny, layer):
    """Our float64 decode is the quantizer's ``dequantize`` of the same stack."""
    stock, directory = tiny
    hidden, inter, experts, w13, w2 = _stacks(tiny, layer)
    text = f"model.language_model.layers.{layer}.mlp.experts"
    want13 = _quantizer_stack(directory, f"{text}.gate_up")
    want2 = _quantizer_stack(directory, f"{text}.down")
    for e in range(experts):
        gate, up, down = hi2.expert_weights(w13, w2, e)
        _assert_same(gate, want13[e][0], f"layer {layer} expert {e} gate")
        _assert_same(up, want13[e][1], f"layer {layer} expert {e} up")
        _assert_same(down, want2[e][0], f"layer {layer} expert {e} down")


def test_tp_slice_of_the_stack_is_the_quantizer_columns(tiny):
    """This rank's TP slice decodes to the quantizer's columns, smoothing and flags
    included (``slice_stack`` is what the loader stores per rank)."""
    stock, directory = tiny
    hidden, inter, experts, w13, w2 = _stacks(tiny, _LAYER)
    text = f"model.language_model.layers.{_LAYER}.mlp.experts"
    want13 = _quantizer_stack(directory, f"{text}.gate_up")
    want2 = _quantizer_stack(directory, f"{text}.down")
    lo, hi = inter // 2, inter
    rank13 = hi2.slice_stack(
        w13,
        "w13",
        parts=2,
        rows_per_part=inter,
        k=hidden,
        lo=lo,
        hi=hi,
        experts=experts,
    )
    rank2 = hi2.slice_stack(
        w2, "w2", parts=1, rows_per_part=hidden, k=inter, lo=lo, hi=hi, experts=experts
    )
    hi2.validate_stack(rank13, 2, hidden, experts)
    hi2.validate_stack(rank2, 1, inter // 2, experts)
    for e in range(experts):
        gate, up, down = hi2.expert_weights(rank13, rank2, e)
        _assert_same(gate, want13[e][0][lo:hi], f"expert {e} gate slice")
        _assert_same(up, want13[e][1][lo:hi], f"expert {e} up slice")
        _assert_same(down, want2[e][0][:, lo:hi], f"expert {e} down slice")


def test_routed_moe_matches_the_transformers_eval(tiny):
    """Soft cap, two-argument activation, route_scale and the fp32 top-k sum match
    the reference model with the quantizer's dequantized experts installed."""
    stock, directory = tiny
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopSparseMoeBlock,
    )

    hidden, inter, experts, w13, w2 = _stacks(tiny, _LAYER)
    text = f"model.language_model.layers.{_LAYER}.mlp.experts"
    want13 = _quantizer_stack(directory, f"{text}.gate_up")
    want2 = _quantizer_stack(directory, f"{text}.down")
    gate_w = torch.stack(
        [torch.stack([want13[e][0], want13[e][1]]) for e in range(experts)]
    )
    gate_w = gate_w.reshape(experts, 2 * inter, hidden).float()
    down_w = (
        torch.stack([want2[e][0] for e in range(experts)])
        .reshape(experts, hidden, inter)
        .float()
    )

    top = AutoConfig.from_pretrained(stock)
    ref = Inkling2LoopSparseMoeBlock(top.text_config).float().eval()
    raw = load_file(str(stock / "model-00001-of-00001.safetensors"))
    prefix = f"model.llm.layers.{_LAYER}.mlp.gate"
    with torch.no_grad():
        ref.experts.gate_up_proj.copy_(gate_w)
        ref.experts.down_proj.copy_(down_w)
        ref.gate.weight.copy_(raw[f"{prefix}.weight"].float())
        ref.gate.e_score_correction_bias.copy_(raw[f"{prefix}.bias"].float())

    cfg = Inkling2LoopTextConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=experts,
        num_experts_per_tok=int(top.text_config.num_experts_per_tok),
        route_scale=float(top.text_config.route_scale),
        mlp_gate_softcap=float(top.text_config.mlp_gate_softcap),
        mlp_up_softcap=float(top.text_config.mlp_up_softcap),
    )
    router = Inkling2LoopTopKRouter(cfg)
    with torch.no_grad():
        router.weight.copy_(raw[f"{prefix}.weight"].float())
        router.e_score_correction_bias.copy_(raw[f"{prefix}.bias"].float())

    gen = torch.Generator().manual_seed(3)
    hidden_states = torch.randn(7, hidden, generator=gen) * 0.4

    logits_ref, weights_ref, ids_ref = ref.gate(hidden_states)
    logits, weights, ids = router(hidden_states)
    _assert_same(logits, logits_ref, "router logits", atol=1e-6)
    _assert_same(weights, weights_ref, "router weights", atol=1e-6)
    # ``sorted=False`` top-k needs the same selected set, not the same column order
    _assert_same(
        ids.double().gather(1, ids.argsort(dim=1)),
        ids_ref.double().gather(1, ids_ref.argsort(dim=1)),
        "router experts",
    )

    with torch.no_grad():
        want = ref.experts(hidden_states, ids_ref, weights_ref)

    dense = ExpertsReference(cfg, dtype=torch.float32)
    with torch.no_grad():
        dense.gate_up_proj.copy_(gate_w)
        dense.down_proj.copy_(down_w)
        got_dense = dense(hidden_states, ids_ref, weights_ref)
    _assert_same(got_dense, want, "dense reference experts", atol=3e-5)

    got_hi2 = hi2.reference_moe(
        hidden_states,
        ids_ref,
        weights_ref,
        w13,
        w2,
        cfg.mlp_gate_softcap,
        cfg.mlp_up_softcap,
    )
    _assert_same(got_hi2, want, "HI2 reference MoE", atol=3e-5)

    # the block with our router end to end (fp32, the model dtype here)
    with torch.no_grad():
        want_block = ref(hidden_states.unsqueeze(0)).reshape(-1, hidden)
    got_block = hi2.reference_moe(
        hidden_states,
        ids,
        weights,
        w13,
        w2,
        cfg.mlp_gate_softcap,
        cfg.mlp_up_softcap,
        experts=experts,
    )
    _assert_same(got_block, want_block, "block with the fork router", atol=3e-5)
