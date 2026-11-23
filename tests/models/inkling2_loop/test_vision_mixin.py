# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``Inkling2MultiModalMixin``: the ``embed_multimodal`` contract, the embedding merge,
the ``model.visual.*`` weight routing and the tower switch (CPU only).

The text stack is a small stand-in with the interface of
``Inkling2LoopForConditionalGeneration`` (``__init__(vllm_config, prefix)``,
``embed_input_ids(input_ids)``, ``load_weights``); the vision tower is the real module
with a CPU stand-in for its GPU ``forward``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.models.inkling2_loop.amd import vision as vision_mod
from vllm.models.inkling2_loop.amd.multimodal import (
    VISUAL_PREFIX,
    Inkling2MultiModalMixin,
    register_inkling2_processor,
)
from vllm.models.inkling2_loop.amd.processing import (
    IMAGE_MARKER_ID,
    IMAGE_PAD_ID,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from vllm.multimodal.processing import InputProcessingContext

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
sys.path.insert(0, str(TOOLS))
import vision_check as tool  # noqa: E402

CFG = tool.TINY_CONFIG
HIDDEN = CFG["text_hidden_size"]  # 192
VOCAB = 200058


class FakeText(nn.Module):
    """The slice of the text stack the mixin touches."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        nn.init.normal_(
            self.model.embed_tokens.weight, generator=torch.Generator().manual_seed(1)
        )
        self.text_seen: list[str] = []

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids) * 2.0  # stands in for ``embed_norm``

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for name, _ in weights:
            self.text_seen.append(name)
        return set(self.text_seen)


class FakeModel(Inkling2MultiModalMixin, FakeText, SupportsMultiModal):
    pass


def _vllm_config(limit_image: int | None = None):
    mm_config = MultiModalConfig(
        limit_per_prompt={} if limit_image is None else {"image": limit_image}
    )
    hf_config = SimpleNamespace(
        vision_config=CFG, text_config=SimpleNamespace(hidden_size=HIDDEN)
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="inkling2",
            hf_config=hf_config,
            multimodal_config=mm_config,
            get_multimodal_config=lambda: mm_config,
        )
    )


@pytest.fixture(autouse=True)
def _first_pp_rank(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
    )


@pytest.fixture
def cell_index_visual(monkeypatch):
    """CPU stand-in for the kernels: row ``i`` of the output is ``i`` everywhere."""

    def forward(self, pixels, grid_thw):
        grids = vision_mod.parse_grids(grid_thw)
        total = sum(r * c for r, c in grids)
        assert pixels.shape == (total, 40, 40, 3)
        rows = torch.arange(total, dtype=torch.float16)[:, None]
        return rows.expand(total, self.out_dim).contiguous()

    monkeypatch.setattr(vision_mod.Inkling2VisionEncoder, "forward", forward)


def _batch(grids):
    cells = sum(r * c for r, c in grids)
    return {
        "vision_pixels": torch.zeros(cells, 40, 40, 3, dtype=torch.float16),
        "vision_grid_thw": torch.tensor([[1, r, c] for r, c in grids]),
    }


def test_class_contract():
    assert FakeModel.supports_multimodal
    assert FakeModel.supports_encoder_tp_data is False
    assert FakeModel.get_placeholder_str("image", 0) == "<|content_image|>"
    assert FakeModel.get_placeholder_str("image", 3) == "<|content_image|>"
    for modality in ("audio", "video"):
        with pytest.raises(ValueError, match="image input only"):
            FakeModel.get_placeholder_str(modality, 0)
    model = FakeModel(vllm_config=_vllm_config())
    assert isinstance(model, SupportsMultiModal)
    assert model.get_language_model() is model
    assert isinstance(model.visual, vision_mod.Inkling2VisionEncoder)
    assert model._tower_model_names == ["visual"]


def test_embed_multimodal_returns_one_tensor_per_image(cell_index_visual):
    model = FakeModel(vllm_config=_vllm_config())
    grids = [(2, 3), (1, 1), (3, 2)]
    out = model.embed_multimodal(**_batch(grids))
    assert [tuple(t.shape) for t in out] == [(6, HIDDEN), (1, HIDDEN), (6, HIDDEN)]
    assert sum(t.shape[0] for t in out) == sum(r * c for r, c in grids)
    assert all(t.dtype == torch.float16 for t in out)
    # images in prompt order, rows in cell order
    assert [t[:, 0].tolist() for t in out] == [
        [0, 1, 2, 3, 4, 5],
        [6],
        [7, 8, 9, 10, 11, 12],
    ]


def test_embed_multimodal_accepts_per_image_lists(cell_index_visual):
    model = FakeModel(vllm_config=_vllm_config())
    grids = [(1, 2), (2, 1)]
    batch = _batch(grids)
    out = model.embed_multimodal(
        vision_pixels=list(batch["vision_pixels"].split([2, 2])),
        vision_grid_thw=list(batch["vision_grid_thw"]),
    )
    assert [tuple(t.shape) for t in out] == [(2, HIDDEN), (2, HIDDEN)]
    single = model.embed_multimodal(
        vision_pixels=[batch["vision_pixels"][:2]],
        vision_grid_thw=[batch["vision_grid_thw"][0]],
    )
    assert [tuple(t.shape) for t in single] == [(2, HIDDEN)]


def test_embed_multimodal_without_images_is_empty():
    model = FakeModel(vllm_config=_vllm_config())
    assert model.embed_multimodal() == ()
    assert model.embed_multimodal(vision_pixels=None, vision_grid_thw=None) == ()


def test_embeddings_replace_the_pad_positions_after_the_text_embedding():
    model = FakeModel(vllm_config=_vllm_config())
    ids = torch.tensor(
        [5, IMAGE_MARKER_ID, *[IMAGE_PAD_ID] * 3, 7, IMAGE_MARKER_ID, IMAGE_PAD_ID]
    )
    is_mm = ids == IMAGE_PAD_ID
    images = [torch.full((3, HIDDEN), 3.0), torch.full((1, HIDDEN), -2.0)]
    merged = model.embed_input_ids(ids, images, is_multimodal=is_mm)
    text = model.embed_input_ids(ids)
    assert torch.equal(
        merged[~is_mm], text[~is_mm]
    )  # text rows (marker included) untouched
    assert torch.equal(merged[2:5], images[0].to(merged.dtype))  # copied, never scaled
    assert torch.equal(merged[7:], images[1].to(merged.dtype))
    # without image embeddings it is the plain (normed) text embedding
    assert torch.equal(model.embed_input_ids(ids), text)
    assert torch.equal(model.embed_input_ids(ids, None), text)
    assert torch.equal(model.embed_input_ids(ids, ()), text)


def test_merge_requires_exactly_as_many_rows_as_pad_positions():
    model = FakeModel(vllm_config=_vllm_config())
    ids = torch.tensor([IMAGE_MARKER_ID, IMAGE_PAD_ID, IMAGE_PAD_ID])
    with pytest.raises(ValueError, match="multimodal"):
        model.embed_input_ids(
            ids, [torch.zeros(3, HIDDEN)], is_multimodal=ids == IMAGE_PAD_ID
        )


def test_visual_weights_go_to_the_tower_and_the_rest_to_the_text_stack():
    model = FakeModel(vllm_config=_vllm_config())
    state = tool.random_state(CFG, seed=5)
    text_names = ["model.llm.embed.weight", "model.llm.layers.0.x", "mtp.fc.weight"]
    stream = [(n, torch.zeros(1)) for n in text_names[:2]]
    stream += [(VISUAL_PREFIX + n, t.bfloat16()) for n, t in state.items()]
    stream += [(text_names[2], torch.zeros(1))]
    loaded = model.load_weights(iter(stream))
    assert model.text_seen == text_names  # nothing visual leaks into the text stack
    visual_module = model.visual
    assert visual_module is not None
    visual = {f"visual.{n}" for n, _ in visual_module.named_parameters()}
    assert loaded == set(text_names) | visual
    qkv = visual_module.backbone.layers[0].self_attn.qkv_proj.weight
    q = state["backbone.layers.0.self_attn.q_proj.weight"]
    assert torch.equal(qkv[: q.shape[0]].float(), q.half().float())


def test_unknown_visual_tensor_fails_loudly():
    model = FakeModel(vllm_config=_vllm_config())
    with pytest.raises(KeyError, match="unexpected vision tensor"):
        model.load_weights(iter([(VISUAL_PREFIX + "backbone.nope", torch.zeros(1))]))


def test_tower_is_absent_when_images_are_disabled():
    model = FakeModel(vllm_config=_vllm_config(limit_image=0))
    assert isinstance(model.visual, StageMissingLayer)
    # the (meta) tower costs no memory and its checkpoint tensors are skipped
    state = tool.random_state(CFG, seed=5)
    stream = [("model.llm.embed.weight", torch.zeros(1))]
    stream += [(VISUAL_PREFIX + n, t.bfloat16()) for n, t in state.items()]
    assert model.load_weights(iter(stream)) == {"model.llm.embed.weight"}


def test_mm_mapping_names_tower_connector_and_language_model():
    keys = FakeModel(vllm_config=_vllm_config()).get_mm_mapping()
    assert keys.language_model == ["model"]
    assert keys.connector == ["visual.proj_in.", "visual.proj_out."]
    assert keys.tower_model == ["visual.backbone."]


def test_processor_registration_builds_the_inkling2_processor():
    class Registered(FakeModel):
        pass

    assert register_inkling2_processor(Registered) is Registered
    mm_config = MultiModalConfig()
    model_config = SimpleNamespace(
        model="inkling2",
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    factory = Registered._processor_factory
    assert factory.info is Inkling2ProcessingInfo
    assert factory.processor is Inkling2MultiModalProcessor
    assert factory.dummy_inputs is Inkling2DummyInputsBuilder
    built = factory.build_processor(InputProcessingContext(model_config, None))
    assert isinstance(built, Inkling2MultiModalProcessor)
