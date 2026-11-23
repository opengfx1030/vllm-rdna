# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2 vision encoder: batch geometry, chunking and weight loading (CPU only).

The kernels are exercised in ``test_vision_encoder.py`` and
``tests/kernels/core/test_inkling2_vit_rdna2.py``; this file needs no GPU.  The real
checkpoint test reads only the safetensors header of ``~/hfcache/inkling2-mini`` and
loads meta tensors of the stored dtype and shape.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest
import torch

from vllm.models.inkling2_loop.amd.vision import (
    DEFAULT_PIXEL_NORM,
    MAX_CHUNK_TOKENS,
    MAX_IMAGE_TOKENS,
    PATCH_K,
    PATCH_K_PAD,
    PIXEL_NORMS,
    Inkling2VisionEncoder,
    build_plan,
    parse_grids,
    split_chunks,
)

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
sys.path.insert(0, str(TOOLS))
import vision_check as tool  # noqa: E402

CKPT = Path(os.environ.get("INKLING2_CKPT", "~/hfcache/inkling2-mini")).expanduser()
SHARD = CKPT / "model-00001-of-00001.safetensors"

CFG = tool.TINY_CONFIG
D = CFG["hidden_size"]


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def test_pixel_norm_option():
    assert DEFAULT_PIXEL_NORM == "clip_bf16" == tool.DEFAULT_PIXEL_NORM
    assert tuple(PIXEL_NORMS) == tool.PIXEL_NORMS  # the tool's oracle modes
    assert Inkling2VisionEncoder(CFG).pixel_norm == "clip_bf16"
    # the vision config carries the choice, the constructor argument wins
    exact = {**CFG, "pixel_norm": "exact"}
    assert Inkling2VisionEncoder(exact).pixel_norm == "exact"
    override = Inkling2VisionEncoder(exact, pixel_norm="clip_bf16")
    assert override.pixel_norm == "clip_bf16"
    with pytest.raises(ValueError, match="pixel_norm"):
        Inkling2VisionEncoder(CFG, pixel_norm="srgb")
    with pytest.raises(ValueError, match="pixel_norm"):
        Inkling2VisionEncoder({**CFG, "pixel_norm": "bf16"})


def test_plan_geometry():
    plan = build_plan([(3, 5), (7, 2), (1, 1)])
    # CLS + 4 patches per cell: 61, 57 and 5 tokens
    assert plan.meta.tolist() == [[3, 5, 0, 0], [7, 2, 61, 60], [1, 1, 118, 116]]
    assert plan.meta.dtype.name == "int32"
    assert plan.num_tokens == 123
    assert plan.num_patches == 120
    assert plan.num_cells == 30
    assert plan.blk_prefix.tolist() == [0, 1, 2, 3]
    assert plan.blk_prefix.dtype.name == "int32"


def test_plan_counts_query_blocks_of_64_per_image():
    plan = build_plan([(16, 16), (4, 4), (5, 3)])
    # 1025 tokens -> 17 blocks, 65 -> 2, 61 -> 1
    assert plan.blk_prefix.tolist() == [0, 17, 19, 20]
    assert plan.num_tokens == 1025 + 65 + 61


@pytest.mark.parametrize("grid", [(0, 3), (3, 0), (-1, 2)])
def test_plan_rejects_empty_images(grid):
    with pytest.raises(ValueError, match="positive"):
        build_plan([grid])


def test_parse_grids():
    assert parse_grids(torch.tensor([[1, 2, 3], [1, 4, 5]])) == [(2, 3), (4, 5)]
    assert parse_grids([[1, 1, 1]]) == [(1, 1)]
    with pytest.raises(ValueError, match="still images"):
        parse_grids(torch.tensor([[2, 2, 3]]))


def test_chunks_keep_images_whole_and_in_order():
    grids = [(3, 5), (7, 2), (1, 1)]  # 61, 57, 5 tokens
    assert split_chunks(grids, 1000) == [grids]
    assert split_chunks(grids, 100) == [[(3, 5)], [(7, 2), (1, 1)]]
    assert split_chunks(grids, 61) == [[(3, 5)], [(7, 2)], [(1, 1)]]
    # an image above the cap is encoded alone, never split
    assert split_chunks(grids, 10) == [[(3, 5)], [(7, 2)], [(1, 1)]]
    assert split_chunks([], 10) == []
    assert MAX_CHUNK_TOKENS == 1 << 16


def _encoder() -> Inkling2VisionEncoder:
    return Inkling2VisionEncoder(CFG, CFG["text_hidden_size"])


def _expected_param_names() -> set[str]:
    names = set()
    for name in tool.state_names(CFG):
        for proj in ("q_proj", "k_proj", "v_proj"):
            if f"self_attn.{proj}." in name:
                name = name.replace(f"{proj}.", "qkv_proj.")
        names.add(name)
    return names


def test_parameters_are_the_checkpoint_names_with_fused_qkv():
    assert {n for n, _ in _encoder().named_parameters()} == _expected_param_names()


def test_load_weights_layout_dtypes_and_cls_row():
    enc = _encoder()
    state = tool.random_state(CFG, seed=3)
    loaded = enc.load_weights(
        (name, t.bfloat16()) for name, t in state.items()
    )  # the checkpoint holds bf16
    params = dict(enc.named_parameters())
    assert loaded == set(params)

    spec = 2.0**-25  # half a fp16 subnormal step: the largest bf16 -> fp16 rounding
    for i in range(CFG["num_hidden_layers"]):
        pre = f"backbone.layers.{i}."
        qkv = params[pre + "self_attn.qkv_proj.weight"]
        bias = params[pre + "self_attn.qkv_proj.bias"]
        assert qkv.dtype == torch.float16 and bias.dtype == torch.float32
        for j, proj in enumerate(("q_proj", "k_proj", "v_proj")):
            src = state[pre + f"self_attn.{proj}.weight"]
            assert (qkv[j * D : (j + 1) * D].float() - src).abs().max() <= spec
            assert torch.equal(
                bias[j * D : (j + 1) * D], state[pre + f"self_attn.{proj}.bias"]
            )
        for name in ("self_attn.out_proj", "mlp.c_fc", "mlp.c_proj"):
            assert params[pre + name + ".weight"].dtype == torch.float16
            assert params[pre + name + ".bias"].dtype == torch.float32
        for name in ("ln_1", "ln_2"):
            assert params[pre + name + ".weight"].dtype == torch.float32
        for name in ("ls_1", "ls_2"):
            assert torch.equal(
                params[pre + name + ".gamma"], state[pre + name + ".gamma"]
            )

    patch = params["backbone.embeddings.patch_embedding.weight"]
    assert patch.shape == (D, PATCH_K_PAD) and PATCH_K_PAD % 32 == 0
    src = state["backbone.embeddings.patch_embedding.weight"]
    assert (patch[:, :PATCH_K].float() - src).abs().max() <= spec
    assert not patch[:, PATCH_K:].any()  # the K pad is zero
    emb = "backbone.embeddings."
    assert params[emb + "position_embedding"].dtype == torch.float32
    assert torch.equal(
        enc.cls, state[emb + "class_embedding"] + state[emb + "cls_position_embedding"]
    )
    for name in ("proj_in.weight", "proj_out.weight"):
        assert params[name].dtype == torch.float16
    assert torch.equal(
        enc.inv_freq,
        (10000.0 ** (-torch.arange(16, dtype=torch.float64) / 16)).float(),
    )


def test_load_weights_reports_only_complete_parameters():
    enc = _encoder()
    state = tool.random_state(CFG, seed=0)
    missing = "backbone.layers.1.self_attn.v_proj.weight"
    loaded = enc.load_weights((n, t) for n, t in state.items() if n != missing)
    assert "backbone.layers.1.self_attn.qkv_proj.weight" not in loaded
    assert "backbone.layers.1.self_attn.qkv_proj.bias" in loaded
    assert loaded == _expected_param_names() - {
        "backbone.layers.1.self_attn.qkv_proj.weight"
    }


def test_load_weights_rejects_unknown_names_and_bad_shapes():
    enc = _encoder()
    with pytest.raises(KeyError, match="unexpected vision tensor"):
        enc.load_weights(
            [("backbone.layers.0.self_attn.rotary.inv_freq", torch.zeros(4))]
        )
    with pytest.raises(ValueError, match="checkpoint"):
        enc.load_weights([("proj_in.weight", torch.zeros(3, 3))])
    with pytest.raises(ValueError, match="checkpoint"):
        enc.load_weights(
            [
                (
                    "backbone.embeddings.patch_embedding.weight",
                    torch.zeros(D, PATCH_K_PAD),
                )
            ]
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"native_patch_size": 16}, "20x20"),
        ({"adapter_merge_size": 1}, "2x2"),
        ({"num_attention_heads": 4}, "head_dim"),
        ({"projector_bias": True}, "projector bias"),
        ({"hidden_act": "relu"}, "hidden_act"),
    ],
)
def test_unsupported_configs_are_refused(change, match):
    with pytest.raises(NotImplementedError, match=match):
        Inkling2VisionEncoder({**CFG, **change}, CFG["text_hidden_size"])


def test_forward_checks_pixel_shapes():
    enc = _encoder()
    with pytest.raises(ValueError, match="do not match"):
        enc(torch.zeros(3, 40, 40, 3, dtype=torch.float16), [[1, 1, 2]])
    with pytest.raises(ValueError, match="do not match"):
        enc(torch.zeros(2, 40, 39, 3, dtype=torch.float16), [[1, 1, 2]])


def test_forward_refuses_an_image_larger_than_one_launch():
    from vllm.models.inkling2_loop.amd.processing import MAX_IMAGE_CELLS

    assert 4 * MAX_IMAGE_CELLS + 1 <= MAX_IMAGE_TOKENS
    assert 4 * (MAX_IMAGE_CELLS + 1) + 1 > MAX_IMAGE_TOKENS  # the tightest cap
    cells = MAX_IMAGE_CELLS + 1
    with pytest.raises(ValueError, match="ViT tokens"):
        _encoder()(torch.zeros(cells, 40, 40, 3, dtype=torch.float16), [[1, 1, cells]])


@pytest.mark.skipif(not SHARD.exists(), reason=f"no checkpoint at {CKPT}")
def test_real_checkpoint_tensors_all_load_into_the_real_config():
    cfg = json.loads((CKPT / "config.json").read_text())["vision_config"]
    header = _header(SHARD)
    visual = {
        k[len(tool.VISUAL_PREFIX) :]: v
        for k, v in header.items()
        if k.startswith(tool.VISUAL_PREFIX)
    }
    assert len(visual) == 422
    assert {v["dtype"] for v in visual.values()} == {"BF16"}
    with torch.device("meta"):
        enc = Inkling2VisionEncoder(cfg, 2304)
    loaded = enc.load_weights(
        (name, torch.empty(meta["shape"], dtype=torch.bfloat16, device="meta"))
        for name, meta in visual.items()
    )
    assert loaded == {n for n, _ in enc.named_parameters()}
    assert enc.dim == 1024 and enc.num_layers == 23 and enc.out_dim == 2304
