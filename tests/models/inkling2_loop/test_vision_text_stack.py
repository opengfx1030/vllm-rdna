# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``Inkling2MultiModalMixin`` on the real ``Inkling2LoopForConditionalGeneration``.

The served model class is the mixin in front of the text stack and
``SupportsMultiModal`` (see ``model.py`` / ``multimodal.py``); this builds that class
and the text class it wraps on the tiny checkpoint of
``tools/rdna/inkling2/make_tiny_checkpoint.py`` (CPU, torch reference path) and checks
that the mixin leaves the text path untouched: same names loaded from the same
tensors, the image embeddings land after ``embed_norm`` and never change a text row,
and ``model.visual.*`` tensors never reach the text stack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("vllm.models.inkling2_loop.amd.model")

from paged_harness import fake_vllm_config, patched  # noqa: E402

from vllm.config.multimodal import MultiModalConfig  # noqa: E402
from vllm.model_executor.models.interfaces import (  # noqa: E402
    SupportsMultiModal,
    supports_multimodal,
)
from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import vision as vision_mod  # noqa: E402
from vllm.models.inkling2_loop.amd.multimodal import (  # noqa: E402
    VISUAL_PREFIX,
    Inkling2MultiModalMixin,
)
from vllm.models.inkling2_loop.configs import Inkling2LoopConfig  # noqa: E402

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"


def _load_tool(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


make = _load_tool("make_tiny_checkpoint")
vision_tool = _load_tool("vision_check")


def test_served_class_is_the_mixin_over_the_text_stack():
    cls = model_mod.Inkling2LoopForConditionalGeneration
    assert cls.__mro__[1:4] == (
        Inkling2MultiModalMixin,
        model_mod.Inkling2LoopTextForCausalLM,
        torch.nn.Module,
    )
    assert SupportsMultiModal in cls.__mro__
    assert supports_multimodal(cls)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny1")
    tensors = make.write_checkpoint(path, make.TinySpec(), seed=0)
    hf_config = Inkling2LoopConfig.from_pretrained(str(path))
    hf_config.vision_config = dict(vision_tool.TINY_CONFIG)
    return path, tensors, hf_config


def _build(cls, path, hf_config):
    fake = fake_vllm_config(path, hf_config)
    mm = MultiModalConfig()
    fake.model_config.get_multimodal_config = lambda: mm
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)  # the engine builds the model under fp16
    try:
        with (
            patched(state_mod, "get_current_vllm_config", lambda: fake),
            patched(
                sys.modules["vllm.distributed"],
                "get_pp_group",
                lambda: SimpleNamespace(is_first_rank=True),
            ),
        ):
            return cls(vllm_config=fake)
    finally:
        torch.set_default_dtype(old)


@torch.no_grad()
def test_text_path_is_untouched_by_the_mixin(tiny):
    path, tensors, hf_config = tiny
    text_only = _build(model_mod.Inkling2LoopTextForCausalLM, path, hf_config)
    combined = _build(model_mod.Inkling2LoopForConditionalGeneration, path, hf_config)
    assert combined.get_language_model() is combined
    visual = combined.visual
    assert isinstance(visual, vision_mod.Inkling2VisionEncoder)
    # the tower projects into the text stack's hidden size, whatever the vision
    # config says
    assert visual.out_dim == hf_config.text_config.hidden_size == 256

    base = text_only.load_weights(iter(tensors.items()))
    vstate = vision_tool.random_state(
        {**vision_tool.TINY_CONFIG, "text_hidden_size": 256}, seed=2
    )
    stream = list(tensors.items())
    stream.insert(
        3, (VISUAL_PREFIX + "backbone.ln_pre.bias", vstate["backbone.ln_pre.bias"])
    )
    stream += [
        (VISUAL_PREFIX + n, t.bfloat16())
        for n, t in vstate.items()
        if n != "backbone.ln_pre.bias"
    ]
    loaded = combined.load_weights(iter(stream))

    names = {f"visual.{n}" for n, _ in visual.named_parameters()}
    assert loaded == base | names  # exactly the text names plus the tower's
    a = {**dict(text_only.named_parameters()), **dict(text_only.named_buffers())}
    b = {**dict(combined.named_parameters()), **dict(combined.named_buffers())}
    compared = [n for n in base if n in a and n in b]
    assert "lm_head.weight" in compared and "model.embed_tokens.weight" in compared
    for name in compared:
        assert torch.equal(a[name], b[name]), name  # same checkpoint, same weights


@torch.no_grad()
def test_image_embeddings_replace_pads_after_embed_norm(tiny):
    path, tensors, hf_config = tiny
    combined = _build(model_mod.Inkling2LoopForConditionalGeneration, path, hf_config)
    combined.load_weights(iter(tensors.items()))
    # the tiny vocabulary has 500 ids: stand-ins for the marker / pad tokens
    marker, pad = 3, 4
    ids = torch.tensor([11, marker, pad, pad, 13, marker, pad])
    is_mm = ids == pad
    image_embeds = [torch.randn(2, 256).half(), torch.randn(1, 256).half()]
    merged = combined.embed_input_ids(ids, image_embeds, is_multimodal=is_mm)
    plain = combined.model.embed_input_ids(ids)  # the text stack's own (normed) rows
    assert merged.dtype == plain.dtype and merged.shape == (7, 256)
    assert torch.equal(merged[~is_mm], plain[~is_mm])  # markers are ordinary text rows
    assert torch.equal(merged[2:4], image_embeds[0].to(merged.dtype))
    assert torch.equal(merged[6:], image_embeds[1].to(merged.dtype))
    assert torch.equal(combined.embed_input_ids(ids), plain)


@torch.no_grad()
def test_embed_multimodal_has_the_text_hidden_size(tiny, monkeypatch):
    path, tensors, hf_config = tiny
    combined = _build(model_mod.Inkling2LoopForConditionalGeneration, path, hf_config)

    def forward(self, pixels, grid_thw):
        total = sum(r * c for r, c in vision_mod.parse_grids(grid_thw))
        return torch.zeros(total, self.out_dim, dtype=torch.float16)

    monkeypatch.setattr(vision_mod.Inkling2VisionEncoder, "forward", forward)
    grids = [(2, 3), (1, 1)]
    out = combined.embed_multimodal(
        vision_pixels=torch.zeros(7, 40, 40, 3, dtype=torch.float16),
        vision_grid_thw=torch.tensor([[1, r, c] for r, c in grids]),
    )
    assert [tuple(t.shape) for t in out] == [(6, 256), (1, 256)]
