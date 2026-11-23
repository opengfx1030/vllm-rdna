# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini image preprocessing and multimodal processing.

The checkpoint's chat template renders an image part as
``<role><|content_image|><|unused_200054|><|end_message|>``; the HF processor
turns the single ``<|unused_200054|>`` into ``rows * cols`` copies.  vLLM sees the
tokenised prompt and does the same, keeping ``<|content_image|>`` (id 200005, a
text token) and expanding to ``[200005] + [200054] * (rows * cols)``; only the
200054 positions receive vision embeddings.  A renderer that emits the marker alone
and the HF template (marker plus one pad) both reach the same final sequence: one
pad directly behind a marker is dropped before the update is applied.

Pixels are not resized.  Each image is padded at the bottom/right to whole
40-pixel cells with the raw value -1 (normalised like any other pixel by the
encoder) and cut into ``[rows * cols, 40, 40, 3]`` row-major cells; the
``vision_grid_thw`` of an image is ``[1, rows, cols]``.  Transparency is composited
on an 8x8 checkerboard (levels 204 / 153), exactly as the HF image processor does.
Audio and video are not accepted.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

# <|content_image|>: block marker of the template, stays a text token
IMAGE_MARKER_ID = 200005
IMAGE_PLACEHOLDER = "<|content_image|>"
# <|unused_200054|>: one per decoder image token, overwritten by the embeddings
IMAGE_PAD_ID = 200054

CELL_SIZE = 40
PAD_VALUE = -1
# profiling images are at most this many pixels per side
MAX_DUMMY_IMAGE_SIDE = 2048
# the encoder takes one image per launch, at most 2**16 ViT tokens (a CLS and four
# patches per cell): ``vision.MAX_IMAGE_TOKENS``
MAX_IMAGE_CELLS = ((1 << 16) - 1) // 4

ImageInput: TypeAlias = bytes | Image.Image | MediaWithBytes | np.ndarray | torch.Tensor


def _composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgba = rgba.astype(np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    blended = rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127
    return (blended // 255).astype(np.uint8)


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    if not image.has_transparency_data:
        return np.array(image.convert("RGB"), dtype=np.uint8)
    return _composite_on_checkerboard(np.array(image.convert("RGBA")))


def _array_to_rgb(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.uint8:
        raise TypeError(f"image arrays must be uint8, got {array.dtype}")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)  # CHW, as ``ImageProcessorItems`` reads it
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return _pil_to_rgb(Image.fromarray(array))


def decode_image(image: ImageInput) -> np.ndarray:
    """Decode to HWC uint8 RGB, compositing transparency on the checkerboard.

    Encoded bytes (also ``MediaWithBytes.original_bytes``) are preferred over the
    already decoded media, so no mode conversion or EXIF handling of the vLLM media
    loader reaches the pixels.
    """
    if isinstance(image, MediaWithBytes):
        image = image.original_bytes
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if data.startswith(b"\xff\xd8") and not data.endswith(b"\xff\xd9"):
            data += b"\xff\xd9"
        with Image.open(io.BytesIO(data)) as pil:
            pil.load()
            return _pil_to_rgb(pil)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, np.ndarray):
        return _array_to_rgb(image)
    return _pil_to_rgb(image)


class Inkling2ImageProcessor:
    """Pads each image to a 40-pixel grid and emits one cell per image token.

    Returns ``vision_pixels`` (fp16 ``[sum(rows * cols), 40, 40, 3]`` raw RGB values,
    padding pixels at -1; 0..255 and -1 are exact in fp16, the encoder normalises)
    and ``vision_grid_thw`` (int64 ``[num_images, 3]``).
    """

    @staticmethod
    def get_grid(width: int, height: int) -> tuple[int, int]:
        return math.ceil(height / CELL_SIZE), math.ceil(width / CELL_SIZE)

    def preprocess(self, images: Sequence[ImageInput]) -> dict[str, torch.Tensor]:
        decoded = [torch.from_numpy(decode_image(image)) for image in images]
        grids = [self.get_grid(pixels.shape[1], pixels.shape[0]) for pixels in decoded]
        for rows, cols in grids:
            if rows * cols > MAX_IMAGE_CELLS:
                raise ValueError(
                    f"an image of {cols * CELL_SIZE}x{rows * CELL_SIZE} px has "
                    f"{rows * cols} cells; at most {MAX_IMAGE_CELLS} are supported"
                )
        out = torch.empty(
            (sum(rows * cols for rows, cols in grids), CELL_SIZE, CELL_SIZE, 3),
            dtype=torch.float16,
        )
        start = 0
        for pixels, (rows, cols) in zip(decoded, grids):
            height, width, _ = pixels.shape
            if (height, width) != (rows * CELL_SIZE, cols * CELL_SIZE):
                padded = pixels.new_full(
                    (rows * CELL_SIZE, cols * CELL_SIZE, 3),
                    PAD_VALUE,
                    dtype=torch.float16,
                )
                padded[:height, :width] = pixels
                pixels = padded
            cells = pixels.view(rows, CELL_SIZE, cols, CELL_SIZE, 3)
            out[start : start + rows * cols].view(
                rows, cols, CELL_SIZE, CELL_SIZE, 3
            ).copy_(cells.permute(0, 2, 1, 3, 4))
            start += rows * cols
        return {
            "vision_pixels": out,
            "vision_grid_thw": torch.tensor(
                [[1, rows, cols] for rows, cols in grids], dtype=torch.int64
            ),
        }


class Inkling2MultiModalDataParser(MultiModalDataParser):
    """Keeps every image item as given (PIL mode, encoded bytes) so transparency
    reaches the checkerboard step; the base parser converts PIL images to RGB."""

    def _parse_image_data(self, data: Any) -> Any:
        if data is None or self.is_embeddings(data):
            return super()._parse_image_data(data)
        if isinstance(data, (Image.Image, MediaWithBytes)) or (
            isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 3
        ):
            return ImageProcessorItems([data])
        return ImageProcessorItems(list(data))


class Inkling2ProcessingInfo(BaseProcessingInfo):
    @cached_property
    def image_processor(self) -> Inkling2ImageProcessor:
        return Inkling2ImageProcessor()

    def get_data_parser(self) -> MultiModalDataParser:
        return Inkling2MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # images only: the template rejects audio, there is no video path
        return {"image": None}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {}


class Inkling2DummyInputsBuilder(BaseDummyInputsBuilder[Inkling2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        # largest square image whose tokens still fit in seq_len
        side = min(MAX_DUMMY_IMAGE_SIDE, CELL_SIZE * max(1, math.isqrt(seq_len)))
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class Inkling2MultiModalProcessor(BaseMultiModalProcessor[Inkling2ProcessingInfo]):
    # prompts arrive as token ids and the update target is a single token id
    requires_tokenizer = False

    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> ProcessorInputs:
        # marker ids directly: the tokenizer may not parse special-token text
        mm_data = self.dummy_inputs.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        return ProcessorInputs(
            prompt=[IMAGE_MARKER_ID] * mm_counts.get("image", 0),
            mm_data_items=self.info.parse_mm_data(mm_data, validate=False),
        )

    def _postprocess_prompt(self, prompt: list[int]) -> list[int]:
        # The checkpoint template writes the pad slot itself, directly behind the
        # marker; the update below re-creates all pads, so that one must go.
        out: list[int] = []
        for token in prompt:
            if token == IMAGE_PAD_ID and out and out[-1] == IMAGE_MARKER_ID:
                continue
            out.append(token)
        return out

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )
        images = mm_items.get("image")
        processed = None
        if isinstance(images, ImageProcessorItems) and images.get_count():
            # raw items keep MediaWithBytes.original_bytes for exact decoding
            processed = BatchFeature(self.info.image_processor.preprocess(images.data))
        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, processed
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thw = hf_inputs.get("vision_grid_thw", torch.empty((0, 3)))
        return dict(
            vision_pixels=MultiModalFieldConfig.flat_from_sizes(
                "image", grid_thw.prod(-1)
            ),
            vision_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_items = out_mm_kwargs.get("image", [])

        def replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_items[item_idx]
            assert item is not None
            grid_thw = item["vision_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod())
            return PromptUpdateDetails.select_token_id(
                [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * num_tokens, IMAGE_PAD_ID
            )

        return [
            PromptReplacement(
                modality="image", target=[IMAGE_MARKER_ID], replacement=replacement
            )
        ]
