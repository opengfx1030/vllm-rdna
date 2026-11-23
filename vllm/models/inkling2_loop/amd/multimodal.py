# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image input of ``Inkling2LoopForConditionalGeneration``.

``Inkling2MultiModalMixin`` goes before the text stack in the bases (it needs the stack
built before it adds the vision tower, and it wraps ``embed_input_ids`` and
``load_weights`` of the stack):

    class Inkling2LoopForConditionalGeneration(
        Inkling2MultiModalMixin, Inkling2LoopTextForCausalLM, SupportsMultiModal
    ): ...

    register_inkling2_processor(Inkling2LoopForConditionalGeneration)

Image embeddings replace the ``<|unused_200054|>`` positions of the prompt *after* the
text embedding (``embed_norm`` included), as the reference model does.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import _merge_multimodal_embeddings, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from .processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from .vision import Inkling2VisionEncoder

VISUAL_PREFIX = "model.visual."


def register_inkling2_processor(cls: type) -> type:
    """Register the Inkling-2 multimodal processor for the model class."""
    return MULTIMODAL_REGISTRY.register_processor(
        Inkling2MultiModalProcessor,
        info=Inkling2ProcessingInfo,
        dummy_inputs=Inkling2DummyInputsBuilder,
    )(cls)


def _flat_inputs(
    pixels: torch.Tensor | list[torch.Tensor],
    grid_thw: torch.Tensor | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The batched fields as one ``[cells, 40, 40, 3]`` / ``[images, 3]`` pair."""
    if isinstance(pixels, list):
        pixels = torch.cat(pixels) if len(pixels) > 1 else pixels[0]
    if isinstance(grid_thw, list):
        grid_thw = torch.stack([g.reshape(3) for g in grid_thw])
    return pixels, grid_thw.reshape(-1, 3)


class Inkling2MultiModalMixin:
    """The vision tower, its embedding merge and its weight routing.

    Not a ``Protocol``: it precedes the text stack in the bases so ``__init__`` runs
    after the stack is built.
    """

    supports_encoder_tp_data = False
    visual: nn.Module | None

    if TYPE_CHECKING:

        def _mark_tower_model(
            self,
            vllm_config: VllmConfig,
            modalities: set[str] | str,
            *,
            targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
        ): ...

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Inkling-2 serves image input only, got {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)  # type: ignore[call-arg]
        config = vllm_config.model_config.hf_config
        self.visual = None
        if config.vision_config is not None:
            # skipped (no weights, no memory) when --limit-mm-per-prompt image=0
            with self._mark_tower_model(vllm_config, "image"):
                self.visual = Inkling2VisionEncoder(
                    config.vision_config,
                    config.text_config.hidden_size,
                    prefix=maybe_prefix(prefix, "visual"),
                )

    def get_language_model(self) -> nn.Module:
        return cast(nn.Module, self)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector=["visual.proj_in.", "visual.proj_out."],
            tower_model=["visual.backbone."],
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """One ``[rows * cols, hidden]`` tensor per image, in prompt order."""
        pixels = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if pixels is None or grid_thw is None:
            return ()
        assert isinstance(self.visual, Inkling2VisionEncoder)
        pixels, grid_thw = _flat_inputs(
            cast("torch.Tensor | list[torch.Tensor]", pixels),
            cast("torch.Tensor | list[torch.Tensor]", grid_thw),
        )
        embeds = self.visual(pixels, grid_thw)
        return embeds.split(grid_thw.prod(-1).tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = super().embed_input_ids(input_ids)  # type: ignore[misc]
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """``model.visual.*`` tensors go to the tower as they stream by (or are
        dropped when the tower is not built), everything else to the text stack."""
        tower = self.visual if isinstance(self.visual, Inkling2VisionEncoder) else None

        def text_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.startswith(VISUAL_PREFIX):
                    if tower is not None:
                        tower.load_weight(name[len(VISUAL_PREFIX) :], tensor)
                    continue
                yield name, tensor

        loaded = super().load_weights(text_weights())  # type: ignore[misc]
        if tower is not None:
            loaded |= {f"visual.{n}" for n in tower.finish_weights()}
        return loaded


__all__ = ["Inkling2MultiModalMixin", "register_inkling2_processor"]
