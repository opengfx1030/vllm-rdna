# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate for the tiny Inkling-2 end-to-end harness.

``tools/rdna/inkling2/make_tiny_checkpoint.py`` writes a random-weight checkpoint
in the real Inkling-2-Mini serving format and ``tools/rdna/inkling2/
tiny_reference.py`` loads it into the transformers reference in fp32.  These
tests pin the harness itself, so the numerical gate it provides to the vLLM port
cannot silently be vacuous:

* the checkpoint has exactly the real key families / dtypes / shapes;
* the reference loads it strictly (no orphan tensor, no unfilled parameter);
* the expert dequantisation equals ``ref_checkpoint.py`` bit for bit;
* logits are finite and not degenerate; **every** reference parameter moves the
  logits when perturbed (a dead or misrouted weight would be caught);
* the 70-token prompt crosses the sliding window (16) and the relative-position
  extent (32), shown by patching both and watching only the long prompts move;
* prefill logits equal step-by-step cached decode logits, incl. the looped layers'
  extra cache copies and conv states.

Run on a CPU box with the transformers build that ships ``inkling2_loop``::

    pytest tests/models/test_inkling2_tiny_harness.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import regex as re
import torch

pytest.importorskip("safetensors")
pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "rdna" / "inkling2"
PROMPT_LENS = (5, 40, 70)
CACHED_TOL = 1e-3  # fp32 absolute logit tolerance (observed ~5e-5)
BF16, F32 = "bfloat16", "float32"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


make = _load_tool("make_tiny_checkpoint")
ref = _load_tool("tiny_reference")


def _prompts(vocab: int = 500) -> list[list[int]]:
    gen = torch.Generator().manual_seed(1234)
    return [torch.randint(0, vocab, (n,), generator=gen).tolist() for n in PROMPT_LENS]


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


@pytest.fixture(scope="module", autouse=True)
def _threads():
    ref.configure_threads(8)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny1")
    tensors = make.write_checkpoint(path, make.TinySpec(), seed=0)
    return path, tensors


@pytest.fixture(scope="module")
def model(tiny):
    return ref.build_reference(tiny[0])


# --------------------------------------------------------------------------- #
# checkpoint format
# --------------------------------------------------------------------------- #


def _expected_layer(spec, layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Real-checkpoint ``layers.N.*`` suffixes -> (dtype, shape), written out
    independently of the generator (from the real checkpoint header)."""
    h, inter, e = spec.hidden_size, spec.moe_intermediate, spec.n_routed_experts
    full = layer in spec.full_layer_ids
    kk, nh, nkv, hd, dr = (
        spec.conv_kernel,
        spec.num_heads,
        spec.num_kv_heads,
        spec.head_dim,
        spec.d_rel,
    )
    x: dict[str, tuple[str, tuple[int, ...]]] = {
        "attn.wq_du.weight": (BF16, (nh * hd, h)),
        "attn.wk_dv.weight": (BF16, (nkv * hd, h)),
        "attn.wv_dv.weight": (BF16, (nkv * hd, h)),
        "attn.wo_ud.weight": (BF16, (h, nh * hd)),
        "attn.gate_attn.weight": (BF16, (nh * hd, h)),
        "attn.wr_du.weight": (BF16, (nh * dr, h)),
        "attn.attn_sink_proj.weight": (BF16, (nh, h)),
        "attn.q_l2_clip.scale": (BF16, (nh, hd)),
        "attn.rel_l2_clip.scale": (BF16, (nh, dr)),
        "attn.rel_logits_proj.proj": (
            BF16,
            (dr, spec.rel_extent if full else spec.sliding_window),
        ),
        "mlp_norm.weight": (BF16, (h,)),
        "attn_sconv.weight": (BF16, (h, 1, kk)),
        "mlp_sconv.weight": (BF16, (h, 1, kk)),
        "mlp.gate.weight": (BF16, (e, h)),
        "mlp.gate.bias": (F32, (e,)),
    }
    for side in ("k", "v"):
        if full:
            x[f"attn.{side}_sconv.static_w"] = (BF16, (kk, nkv * hd))
            x[f"attn.{side}_sconv.weight_proj"] = (BF16, (nkv * kk, h))
            x[f"{side}_input_norm.weight"] = (BF16, (h,))
        else:
            x[f"attn.{side}_sconv.weight"] = (BF16, (nkv * hd, 1, kk))
    if full:
        x["q_input_norm.weight"] = (BF16, (h,))
    else:
        x["attn_norm.weight"] = (BF16, (h,))
    if layer in spec.loop_layer_ids:
        r = spec.loop_gate_rank
        for ctl in ("attn", "k", "v", "mlp"):
            x[f"controllers.{ctl}.down_weight"] = (BF16, (r, 2 * h))
            x[f"controllers.{ctl}.forget_weight"] = (BF16, (h, r))
            x[f"controllers.{ctl}.gain_weight"] = (BF16, (h, r))
    for nm, (n, k) in (("w13", (2 * inter, h)), ("w2", (h, inter))):
        pre = f"mlp.experts.{nm}_weight"
        if spec.is_nvfp4_layer(layer):
            x[pre] = ("uint8", (e, n, k // 2))
            x[pre + "_scale"] = ("float8_e4m3fn", (e, n, k // 16))
            x[pre + "_amax"] = (F32, (e,))
            x[pre + "_outer_scale_denominator"] = (F32, (1,))
        else:
            x[pre] = ("float8_e4m3fn", (e, n, k))
            x[pre + "_scale"] = ("uint8", (e, n, k // 32))
    return x


def _expected_checkpoint(spec) -> dict[str, tuple[str, tuple[int, ...]]]:
    h, n = spec.hidden_size, spec.num_hidden_layers
    n_loop = spec.loop_end - spec.loop_start
    p = "model.llm."
    x = {
        p + "embed.weight": (BF16, (spec.vocab_size, h)),
        p + "embed_norm.weight": (BF16, (h,)),
        p + "norm.weight": (BF16, (h,)),
        p + "unembed.weight": (BF16, (spec.vocab_size, h)),
        p + "attn_res.output_query": (F32, (h,)),
        p + "middle_cycle_cross_pass_queries": (F32, (2 * n_loop, h)),
        p + "middle_cycle_cross_pass_kv_queries": (F32, (2 * n_loop, h)),
    }
    for i in range(n):
        x[f"{p}attn_res.mlp_queries.{i}"] = (F32, (h,))
        if i + 1 < n:
            shape = (3, h) if (i + 1) in spec.full_layer_ids else (h,)
            x[f"{p}attn_res.attn_queries.{i}"] = (F32, shape)
        for suffix, v in _expected_layer(spec, i).items():
            x[f"{p}layers.{i}.{suffix}"] = v
    return x


def test_checkpoint_matches_real_serving_format(tiny):
    _, tensors = tiny
    expected = _expected_checkpoint(make.TinySpec())
    assert sorted(tensors) == sorted(expected)
    for key, (dtype, shape) in expected.items():
        assert (_dtype(tensors[key]), tuple(tensors[key].shape)) == (dtype, shape), key


def test_files_index_and_quant_config(tiny):
    path, tensors = tiny
    spec = make.TinySpec()
    cfg = json.loads((path / "config.json").read_text())
    assert cfg["architectures"] == ["Inkling2LoopForConditionalGeneration"]
    assert cfg["model_type"] == "inkling2_loop"
    assert cfg["quantization_config"] == {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
    }
    text = cfg["text_config"]
    assert text["hidden_size"] == 256 and text["num_hidden_layers"] == 6
    assert text["loop_config"] == {
        "core_start": 2,
        "core_end": 4,
        "num_passes": 2,
        "gate_rank": 16,
        "is_varloop": False,
    }
    assert text["n_layers_per_attn_res_block"] == 2
    assert (text["sliding_window_size"], text["rel_extent"]) == (16, 32)
    assert (text["vocab_size"], text["unpadded_vocab_size"]) == (512, 500)
    # full attention at 1, 3, 5: both a loop (3) and non-loop (1, 5) full layer
    assert [i for i, t in enumerate(text["layer_types"]) if t == "hybrid"] == [1, 3, 5]
    assert text["local_layer_ids"] == [0, 2, 4]
    assert "vision_config" in cfg and not any("visual" in k for k in tensors)

    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert sorted(index["weight_map"]) == sorted(tensors)
    assert set(index["weight_map"].values()) == {make.SHARD}
    assert (path / make.SHARD).is_file()

    quant = json.loads((path / "hf_quant_config.json").read_text())
    assert quant["quant_method"] == "inkling2"
    groups = {
        g["compute_dtype"] + g["output_dtype"]: g
        for g in quant["model"]["parameter_groups"]
    }
    nvfp4, mxfp8 = (
        groups["nvfp4uint8"]["tensors"],
        groups["mxfp8float8_e4m3fn"]["tensors"],
    )
    assert (
        len(nvfp4) == 2 * 5 and len(mxfp8) == 2 and all("layers.5." in k for k in mxfp8)
    )
    assert all(k.endswith(("w13_weight", "w2_weight")) for k in nvfp4 + mxfp8)
    listed = [k for g in quant["model"]["parameter_groups"] for k in g["tensors"]]
    assert len(listed) == len(set(listed))
    aux = ("_scale", "_amax", "_outer_scale_denominator")
    assert sorted(listed) == sorted(k for k in tensors if not k.endswith(aux))
    assert spec.is_nvfp4_layer(4) and not spec.is_nvfp4_layer(5)


def test_checkpoint_generation_is_deterministic(tmp_path, tiny):
    spec = make.TinySpec()
    again = make.build_tensors(spec, seed=0)
    other = make.build_tensors(spec, seed=1)
    for k, v in tiny[1].items():
        assert torch.equal(v.view(torch.uint8), again[k].view(torch.uint8)), k
    assert not torch.equal(
        tiny[1]["model.llm.layers.0.attn.wq_du.weight"],
        other["model.llm.layers.0.attn.wq_du.weight"],
    )


def test_quantisers_round_trip_faithfully():
    gen = torch.Generator().manual_seed(7)
    w = torch.randn(3, 64, 128, generator=gen) * 0.05
    packed, scale, amax = make.quantise_nvfp4(w)
    assert torch.equal(amax, w.abs().amax(dim=(1, 2)))
    got = ref.dequant_nvfp4(packed, scale, amax, torch.tensor([make.NVFP4_DENOM]))
    assert (got - w).square().mean().sqrt() / w.square().mean().sqrt() < 0.15
    q, s = make.quantise_mxfp8(w)
    got = ref.dequant_mxfp8(q, s)
    assert (got - w).square().mean().sqrt() / w.square().mean().sqrt() < 0.05
    # NVFP4 uses the full e2m1 range: the largest block value maps to +-6
    assert ((packed & 0x0F) == 7).any() or ((packed >> 4) == 7).any()


def test_expert_dequant_equals_ref_checkpoint_bit_for_bit(tiny):
    rc = ref.load_ref_checkpoint()
    _, tensors = tiny
    for layer in (0, 3, 5):  # NVFP4 sliding/full-loop and the MXFP8 last layer
        prefix = f"model.llm.layers.{layer}.mlp.experts"
        up_ref, down_ref = rc.expert_pair(tensors, layer, prefix, None)
        up, down = ref.dequant_experts(tensors, prefix)
        assert up_ref.dtype == torch.bfloat16
        assert torch.equal(up.to(torch.bfloat16), up_ref), layer
        assert torch.equal(down.to(torch.bfloat16), down_ref), layer
        assert up.isfinite().all() and up.abs().max() > 0


def test_w13_rows_are_interleaved_in_the_checkpoint(tiny):
    """Like the real checkpoint, ``w13`` (and its block scales) store gate/up rows
    interleaved; the reference oracle de-interleaves them (transformers'
    ``Interleave(dim=1)``: ``gate_up[:, j * I + i] = raw[:, 2 * i + j]``)."""
    rc = ref.load_ref_checkpoint()
    _, tensors = tiny
    x = torch.randn(2, 8, 5)
    assert torch.equal(make.interleave_gate_up(x)[:, 0::2], x[:, :4])
    assert torch.equal(make.interleave_gate_up(x)[:, 1::2], x[:, 4:])
    assert torch.equal(rc.deinterleave_gate_up(make.interleave_gate_up(x)), x)
    for layer in (0, 5):  # NVFP4 and MXFP8
        prefix = f"model.llm.layers.{layer}.mlp.experts"
        w, s = tensors[f"{prefix}.w13_weight"], tensors[f"{prefix}.w13_weight_scale"]
        if w.dtype == torch.uint8:
            raw = ref.dequant_nvfp4(
                w,
                s,
                tensors[f"{prefix}.w13_weight_amax"],
                tensors[f"{prefix}.w13_weight_outer_scale_denominator"],
            )
        else:
            raw = ref.dequant_mxfp8(w, s)
        up, _ = ref.dequant_experts(tensors, prefix)
        half = raw.shape[1] // 2
        assert torch.equal(up[:, :half], raw[:, 0::2]), layer
        assert torch.equal(up[:, half:], raw[:, 1::2]), layer
        assert not torch.equal(up, raw)


def test_name_map_adapter_reproduces_real_checkpoint_layout():
    rc = ref.load_ref_checkpoint()
    base = rc.build_name_map(24)
    assert ref.adapt_name_map(base, 24, {5, 11, 17, 23}) == base
    tiny_map = ref.adapt_name_map(rc.build_name_map(6), 6, {1, 3, 5})
    dst = "model.language_model.layers.{}.self_attn.k_proj.sconv"
    assert (
        tiny_map["model.llm.layers.1.attn.k_sconv.static_w"]
        == dst.format(1) + ".static_weight"
    )
    assert (
        tiny_map["model.llm.layers.2.attn.k_sconv.weight"]
        == dst.format(2) + ".conv1d.weight"
    )
    assert "model.llm.layers.5.attn.k_sconv.weight" not in tiny_map


def test_reference_loader_rejects_orphans_and_unfilled(tiny, tmp_path):
    from safetensors.torch import save_file

    path, tensors = tiny
    bad = tmp_path / "orphan"
    bad.mkdir()
    for name in ("config.json",):
        (bad / name).write_text((path / name).read_text())
    extra = dict(tensors)
    extra["model.llm.layers.0.attn.not_a_real_tensor"] = torch.zeros(1)
    save_file(extra, str(bad / make.SHARD))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.llm.layers.0.attn.not_a_real_tensor"] = make.SHARD
    (bad / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="without a reference home"):
        ref.build_reference(bad)

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "config.json").write_text((path / "config.json").read_text())
    short = {
        k: v for k, v in tensors.items() if k != "model.llm.layers.2.mlp_norm.weight"
    }
    save_file(short, str(missing / make.SHARD))
    index["weight_map"] = {k: make.SHARD for k in short}
    (missing / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="never filled"):
        ref.build_reference(missing)


# --------------------------------------------------------------------------- #
# reference behaviour
# --------------------------------------------------------------------------- #


def test_logits_finite_and_not_degenerate(model):
    for ids in _prompts():
        logits = ref.full_logits(model, ids)
        assert logits.shape == (len(ids), 500)
        assert torch.isfinite(logits).all()
        assert logits[-1].std() > 0.1
        # positions differ and the argmax is not a constant
        assert logits.std(dim=0).mean() > 0.1
    long = ref.full_logits(model, _prompts()[2])
    assert len(set(long.argmax(-1).tolist())) > 10


def test_logits_depend_on_early_tokens(model):
    ids = _prompts()[2]
    base = ref.full_logits(model, ids)[-1]
    changed = [(ids[0] + 1) % 500, *ids[1:]]
    assert (ref.full_logits(model, changed)[-1] - base).abs().max() > 1e-3
    tail = [*ids[:-1], (ids[-1] + 1) % 500]
    assert (ref.full_logits(model, tail)[-1] - base).abs().max() > 1e-3


def test_reference_forward_is_deterministic(model):
    ids = _prompts()[1]
    assert torch.equal(ref.full_logits(model, ids), ref.full_logits(model, ids))


def _dead_parameters(model, ids) -> list[str]:
    """Names of parameters / buffers whose perturbation leaves the logits alone."""
    base = ref.full_logits(model, ids)
    gen = torch.Generator().manual_seed(99)
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    dead = []
    with torch.no_grad():
        for name, t in tensors.items():
            if "vision_tower" in name:
                continue
            saved = t.clone()
            noise = torch.randn(t.shape, generator=gen)
            amp = (
                2.0
                if "e_score_correction_bias" in name
                else 0.5 * float(t.abs().mean())
            )
            t.add_(noise * max(amp, 1e-3))
            delta = float((ref.full_logits(model, ids) - base).abs().max())
            t.copy_(saved)
            if delta <= 1e-5:
                dead.append(name)
    assert torch.equal(ref.full_logits(model, ids), base)  # fully restored
    return dead


def test_every_reference_parameter_moves_the_logits(model):
    """Perturb each parameter / buffer in turn: a weight the loader mapped wrongly
    or the forward never reads would leave the logits untouched.

    One parameter is dead *by construction* in the default model: with a single
    loop block there is no earlier block, so ``message_input`` attends over one key
    and softmax is 1 whatever the query is.  The deeper variant below covers it
    (and the message-slot order) with two keys."""
    dead = _dead_parameters(model, _prompts()[1][:24])
    assert dead == ["model.language_model.message_input.query"]


def _logit_shift(model, ids, **patch) -> float:
    base = ref.full_logits(model, ids)[-1]
    text = model.model.language_model
    restore = []
    try:
        if "sliding_window" in patch:
            restore.append(
                (text.config, "sliding_window_size", text.config.sliding_window_size)
            )
            text.config.sliding_window_size = patch["sliding_window"]
        if "rel_extent" in patch:
            for layer in text.layers:
                bias = layer.self_attn.relative_position_bias
                if not layer.self_attn.is_sliding:
                    restore.append((bias, "rel_extent", bias.rel_extent))
                    bias.rel_extent = patch["rel_extent"]
        shifted = ref.full_logits(model, ids)[-1]
    finally:
        for obj, attr, value in restore:
            setattr(obj, attr, value)
    assert torch.equal(ref.full_logits(model, ids)[-1], base)
    return float((shifted - base).abs().max())


def test_sliding_window_masking_is_exercised(model):
    short, mid, long = _prompts()
    assert make.TinySpec.sliding_window == 16 and len(short) < 16 < len(long)
    # widening the window leaves a 5-token prompt untouched, moves the long ones
    assert _logit_shift(model, short, sliding_window=1000) == 0.0
    assert _logit_shift(model, mid, sliding_window=1000) > 1e-3
    assert _logit_shift(model, long, sliding_window=1000) > 1e-3


def test_relative_extent_clipping_is_exercised(model):
    short, _, long = _prompts()
    assert make.TinySpec.rel_extent == 32 and len(short) < 24 < 32 < len(long)
    # narrowing the clip (24 < 32 <= distance) only matters for long prompts
    assert _logit_shift(model, short, rel_extent=24) == 0.0
    assert _logit_shift(model, long, rel_extent=24) > 1e-3


@pytest.mark.parametrize("n_tokens", PROMPT_LENS)
def test_prefill_equals_stepwise_cached_decode(model, n_tokens):
    ids = _prompts()[PROMPT_LENS.index(n_tokens)]
    full = ref.full_logits(model, ids)
    stepped = ref.teacher_forced_decode(model, ids, 3)
    # prefill of N tokens vs prefill of N-3 then 3 single-token cached steps
    assert (stepped - full[-3:]).abs().max() < CACHED_TOL
    assert stepped.argmax(-1).tolist() == full[-3:].argmax(-1).tolist()


def test_greedy_continuation_cached_equals_recompute(model):
    ids = _prompts()[2]
    cached_ids, cached_rows = ref.greedy_cached(model, ids, 6)
    full_ids, full_rows = ref.greedy_recompute(model, ids, 6)
    assert cached_ids == full_ids
    assert (cached_rows - full_rows).abs().max() < CACHED_TOL


@pytest.fixture(scope="module")
def deep_model(tmp_path_factory):
    """8 layers, loop 2:6 x3: two loop blocks, so the message queries attend over
    two keys (the 2-layer loop of the default model leaves a single one)."""
    spec = make.TinySpec(
        num_hidden_layers=8, loop_start=2, loop_end=6, loop_num_passes=3
    )
    path = tmp_path_factory.mktemp("tiny_deep")
    make.write_checkpoint(path, spec, seed=3)
    return ref.build_reference(path)


def test_deeper_loop_variant_is_self_consistent(deep_model):
    ids = _prompts()[2]
    full = ref.full_logits(deep_model, ids)
    assert torch.isfinite(full).all() and full[-1].std() > 0.1
    stepped = ref.teacher_forced_decode(deep_model, ids, 3)
    assert (stepped - full[-3:]).abs().max() < CACHED_TOL


def test_deeper_loop_variant_every_parameter_and_message_slot_is_live(deep_model):
    ids = _prompts()[1][:24]
    assert _dead_parameters(deep_model, ids) == []
    # each of the four message slots (q, k, v, mlp) is consumed by the loop layers
    query = deep_model.model.language_model.message_input.query
    base = ref.full_logits(deep_model, ids)
    gen = torch.Generator().manual_seed(5)
    for slot in range(4):
        saved = query.detach().clone()
        with torch.no_grad():
            query[:, slot] += torch.randn(query[:, slot].shape, generator=gen)
        shift = float((ref.full_logits(deep_model, ids) - base).abs().max())
        with torch.no_grad():
            query.copy_(saved)
        assert shift > 1e-5, f"message slot {slot} does not move the logits"


def test_cli_writes_json_and_npy(tmp_path, tiny, model, monkeypatch):
    prompts = _prompts()
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    out = tmp_path / "logits.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiny_reference.py",
            "--ckpt",
            str(tiny[0]),
            "--prompts-json",
            str(tmp_path / "prompts.json"),
            "--out",
            str(out),
            "--decode-steps",
            "3",
        ],
    )
    assert ref.main() == 0
    report = json.loads(out.read_text())
    last = torch.from_numpy(np.load(out.with_suffix(".npy")))
    decode = torch.from_numpy(np.load(tmp_path / "logits_decode.npy"))
    assert last.shape == (3, 500) and decode.shape == (3, 3, 500)
    for i, (ids, entry) in enumerate(zip(prompts, report["prompts"], strict=True)):
        want = ref.full_logits(model, ids)[-1]
        assert (last[i] - want).abs().max() < 1e-4
        assert (
            torch.equal(decode[i, 0], last[i])
            or (decode[i, 0] - last[i]).abs().max() < 1e-4
        )
        assert entry["n_tokens"] == len(ids) and len(entry["greedy_ids"]) == 3
        assert entry["greedy_ids"][0] == int(want.argmax())
        assert entry["last_top8"]["ids"] == torch.topk(last[i], 8).indices.tolist()
        assert entry["cache_vs_recompute"]["greedy_ids_equal"]
        assert entry["cache_vs_recompute"]["max_abs_logit_diff"] < CACHED_TOL
        assert entry["logits_std"] > 0.1


def test_key_family_names_are_the_real_ones(tiny):
    """Guard against renaming a family in the generator: these regexes are the
    real checkpoint's key families (verified against its safetensors header)."""
    families = {
        r"model\.llm\.(embed|embed_norm|norm|unembed)\.weight",
        r"model\.llm\.attn_res\.(attn_queries|mlp_queries)\.\d+",
        r"model\.llm\.attn_res\.output_query",
        r"model\.llm\.middle_cycle_cross_pass_(kv_)?queries",
        r"model\.llm\.layers\.\d+\.attn\.(wq_du|wk_dv|wv_dv|wo_ud|gate_attn|wr_du|attn_sink_proj)\.weight",
        r"model\.llm\.layers\.\d+\.attn\.(q_l2_clip|rel_l2_clip)\.scale",
        r"model\.llm\.layers\.\d+\.attn\.rel_logits_proj\.proj",
        r"model\.llm\.layers\.\d+\.attn\.[kv]_sconv\.(weight|static_w|weight_proj)",
        r"model\.llm\.layers\.\d+\.(attn_norm|mlp_norm|q_input_norm|k_input_norm|v_input_norm|attn_sconv|mlp_sconv)\.weight",
        r"model\.llm\.layers\.\d+\.controllers\.(attn|k|v|mlp)\.(down|forget|gain)_weight",
        r"model\.llm\.layers\.\d+\.mlp\.gate\.(weight|bias)",
        r"model\.llm\.layers\.\d+\.mlp\.experts\.w(13|2)_weight(_scale|_amax|_outer_scale_denominator)?",
    }
    for key in tiny[1]:
        assert any(re.fullmatch(f, key) for f in families), key
