# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2 image preprocessing, placeholder expansion and golden token ids.

Pure CPU.  The checks against the transformers ``Inkling2ImageProcessor`` /
``Inkling2Processor`` and the real tokenizer + chat template run when the transformers
build with ``inkling2`` is importable and the checkpoint is on the machine
(``~/hfcache/inkling2-mini`` or ``$INKLING2_CKPT``); everything else is self-contained.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.models.inkling2_loop.amd.processing import (
    IMAGE_MARKER_ID,
    IMAGE_PAD_ID,
    IMAGE_PLACEHOLDER,
    MAX_DUMMY_IMAGE_SIDE,
    MAX_IMAGE_CELLS,
    Inkling2DummyInputsBuilder,
    Inkling2ImageProcessor,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
    decode_image,
)
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.processing import (
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
CKPT = Path(os.environ.get("INKLING2_CKPT", "~/hfcache/inkling2-mini")).expanduser()
HF_INKLING2 = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("transformers.models.inkling2") is not None
)

needs_hf = pytest.mark.skipif(
    not HF_INKLING2, reason="transformers without the inkling2 model"
)
needs_ckpt = pytest.mark.skipif(
    not (HF_INKLING2 and (CKPT / "tokenizer.json").exists()),
    reason=f"transformers+inkling2 or the checkpoint at {CKPT} is missing",
)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _gradient(height: int, width: int) -> np.ndarray:
    y, x, c = np.meshgrid(
        np.arange(height), np.arange(width), np.arange(3), indexing="ij"
    )
    return ((y * 7 + x * 13 + c * 31) % 256).astype(np.uint8)


def _hf_check():
    sys.path.insert(0, str(TOOLS))
    import vision_check

    return vision_check


@pytest.fixture
def processor() -> Inkling2MultiModalProcessor:
    mm_config = MultiModalConfig()
    model_config = SimpleNamespace(
        model="inkling2",
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    info = Inkling2ProcessingInfo(InputProcessingContext(model_config, None))
    return Inkling2MultiModalProcessor(info, Inkling2DummyInputsBuilder(info))


def _apply(processor, prompt: list[int], images: list):
    inputs = ProcessorInputs(
        prompt=prompt, mm_data_items=processor.info.parse_mm_data({"image": images})
    )
    return processor.apply(inputs, TimingContext(enabled=False))


@pytest.mark.parametrize(
    ("height", "width", "rows", "cols"),
    [
        (1, 1, 1, 1),
        (39, 41, 1, 2),
        (40, 40, 1, 1),
        (41, 40, 2, 1),
        (40, 80, 1, 2),
        (81, 121, 3, 4),
        (2048, 2048, 52, 52),
    ],
)
def test_grid_is_ceil_of_40_pixel_cells(height, width, rows, cols):
    assert Inkling2ImageProcessor.get_grid(width, height) == (rows, cols)


def test_cells_are_row_major_with_bottom_right_padding():
    height, width = 85, 70
    pixels = _gradient(height, width)
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(pixels)])

    assert out["vision_grid_thw"].dtype == torch.int64
    assert out["vision_grid_thw"].tolist() == [[1, 3, 2]]
    cells = out["vision_pixels"]
    assert cells.dtype == torch.float16
    assert cells.shape == (6, 40, 40, 3)
    canvas = np.full((120, 80, 3), -1.0, dtype=np.float32)
    canvas[:height, :width] = pixels
    for r in range(3):
        for c in range(2):
            want = torch.from_numpy(
                canvas[r * 40 : (r + 1) * 40, c * 40 : (c + 1) * 40]
            )
            assert torch.equal(cells[r * 2 + c].float(), want), (r, c)


def test_batch_concatenates_images_in_order():
    a, b = _gradient(40, 80), _gradient(50, 40)[::-1].copy()
    out = Inkling2ImageProcessor().preprocess([Image.fromarray(a), Image.fromarray(b)])
    assert out["vision_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 1]]
    cells = out["vision_pixels"]
    assert cells.shape == (4, 40, 40, 3)
    assert torch.equal(cells[0].float(), torch.from_numpy(a[:, :40].astype(np.float32)))
    assert torch.equal(cells[1].float(), torch.from_numpy(a[:, 40:].astype(np.float32)))
    assert torch.equal(cells[2].float(), torch.from_numpy(b[:40].astype(np.float32)))
    assert (cells[3, 10:] == -1).all()
    assert torch.equal(
        cells[3, :10].float(), torch.from_numpy(b[40:].astype(np.float32))
    )


def test_transparency_composited_on_checkerboard():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[0, 0] = (10, 20, 30, 255)
    rgba[0, 1] = (200, 100, 0, 128)
    image = Image.fromarray(rgba, "RGBA")
    background = np.where(
        (np.arange(16)[None, :] // 8 + np.arange(16)[:, None] // 8) % 2 == 0, 204, 153
    )
    expected = np.repeat(background[:, :, None], 3, axis=2)
    expected[0, 0] = (10, 20, 30)
    # (c * a + bg * (255 - a) + 127) // 255 with a = 128, bg = 204
    expected[0, 1] = [(c * 128 + 204 * 127 + 127) // 255 for c in (200, 100, 0)]

    flattened = MediaWithBytes(image.convert("RGB"), _png(image))
    processor = Inkling2ImageProcessor()
    for image_input in (image, _png(image), flattened, rgba):
        cells = processor.preprocess([image_input])["vision_pixels"]
        assert torch.equal(
            cells[0, :16, :16].float(), torch.from_numpy(expected.astype(np.float32))
        )


def test_truncated_jpeg_without_eoi_decodes_like_the_full_file():
    data = _jpeg(_gradient(48, 40))
    assert data.endswith(b"\xff\xd9")
    assert np.array_equal(decode_image(data[:-2]), decode_image(data))


def test_arrays_and_tensors_hwc_and_chw():
    pixels = _gradient(45, 50)
    want = decode_image(Image.fromarray(pixels))
    assert np.array_equal(want, pixels)
    assert np.array_equal(decode_image(pixels), want)
    assert np.array_equal(decode_image(torch.from_numpy(pixels)), want)
    assert np.array_equal(decode_image(pixels.transpose(2, 0, 1).copy()), want)
    gray = pixels[:, :, 0]
    assert np.array_equal(decode_image(gray[:, :, None]), decode_image(gray))


def test_non_uint8_arrays_are_rejected():
    with pytest.raises(TypeError, match="uint8"):
        decode_image(np.zeros((8, 8, 3), dtype=np.float32))


def test_images_larger_than_one_encoder_launch_are_refused():
    wide = np.zeros((40, 40 * (MAX_IMAGE_CELLS + 1), 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="cells"):
        Inkling2ImageProcessor().preprocess([wide])


@needs_hf
@pytest.mark.parametrize(
    "mode", ["RGB", "L", "RGBA", "LA", "P", "P-transparent", "1", "CMYK"]
)
def test_pixels_match_the_transformers_image_processor(mode):
    tool = _hf_check()
    rng = np.random.default_rng(7)
    h, w = 83, 117
    if mode == "RGB":
        image = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    elif mode == "L":
        image = Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8))
    elif mode == "RGBA":
        rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        rgba[:20, :20, 3] = 0
        rgba[20:40, :20, 3] = 255
        image = Image.fromarray(rgba)
    elif mode == "LA":
        image = Image.fromarray(rng.integers(0, 256, (h, w, 2), dtype=np.uint8), "LA")
    elif mode == "P":
        image = Image.fromarray(
            rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ).convert("P")
    elif mode == "P-transparent":
        image = Image.fromarray(
            rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ).convert("P")
        image.info["transparency"] = 3
    elif mode == "1":
        image = Image.fromarray(
            rng.integers(0, 2, (h, w), dtype=np.uint8) * 255
        ).convert("1")
    else:
        image = Image.fromarray(
            rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ).convert("CMYK")
    out = Inkling2ImageProcessor().preprocess([image])
    pixel_values, grid = tool.hf_pixels([image])
    assert torch.equal(out["vision_grid_thw"], grid)
    exact = tool.normalize_raw(out["vision_pixels"], "exact")  # the HF processor's
    assert torch.equal(exact, pixel_values), mode


def test_placeholder_expansion(processor):
    images = [Image.new("RGB", (100, 50)), Image.new("RGBA", (40, 40))]
    prompt = [200000, IMAGE_MARKER_ID, 200010, 200000, IMAGE_MARKER_ID, 200010]
    out = _apply(processor, prompt, images)

    assert out["prompt_token_ids"] == (
        [200000, IMAGE_MARKER_ID]
        + [IMAGE_PAD_ID] * 6
        + [200010, 200000, IMAGE_MARKER_ID, IMAGE_PAD_ID, 200010]
    )
    ranges = out["mm_placeholders"]["image"]
    assert [(r.offset, r.length) for r in ranges] == [(1, 7), (10, 2)]
    assert [int(r.is_embed.sum()) for r in ranges] == [6, 1]
    data = out["mm_kwargs"].get_data()
    assert data["vision_pixels"].shape == (7, 40, 40, 3)
    assert data["vision_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 1]]
    assert data["vision_grid_thw"].device.type == "cpu"


def test_marker_ids_are_the_template_tokens():
    assert (IMAGE_MARKER_ID, IMAGE_PAD_ID) == (200005, 200054)
    assert IMAGE_PLACEHOLDER == "<|content_image|>"


@pytest.mark.parametrize("tail", [[], [200054], [200054, 200054, 200054]])
def test_template_pad_behind_the_marker_is_not_doubled(processor, tail):
    # the chat template writes ``<|content_image|><|unused_200054|>``; an already
    # expanded prompt must come out identical, not with extra placeholders
    prompt = [3, IMAGE_MARKER_ID, *tail, 200006, 4]
    out = _apply(processor, prompt, [Image.new("RGB", (81, 40))])
    assert out["prompt_token_ids"] == [
        3,
        IMAGE_MARKER_ID,
        *[IMAGE_PAD_ID] * 3,
        200006,
        4,
    ]


def test_embed_positions_are_exactly_the_pads(processor):
    out = _apply(processor, [IMAGE_MARKER_ID], [Image.new("RGB", (80, 120))])
    (placeholder,) = out["mm_placeholders"]["image"]
    assert placeholder.length == 1 + 6
    assert placeholder.is_embed.tolist() == [False] + [True] * 6


def test_images_of_one_message_keep_their_own_grids(processor):
    images = [Image.new("RGB", (41, 85)), Image.new("RGB", (100, 50))]
    out = _apply(processor, [IMAGE_MARKER_ID, 9, IMAGE_MARKER_ID], images)
    assert out["prompt_token_ids"] == (
        [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * 6 + [9]
        + [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * 6
    )  # fmt: skip
    data = out["mm_kwargs"].get_data()
    assert data["vision_grid_thw"].tolist() == [[1, 3, 2], [1, 2, 3]]


def test_audio_and_video_are_rejected(processor):
    info = processor.info
    assert info.get_supported_mm_limits() == {"image": None}
    for modality, item in (
        ("audio", np.zeros(1600, dtype=np.float32)),
        ("video", np.zeros((2, 40, 40, 3), dtype=np.uint8)),
    ):
        with pytest.raises(VLLMValidationError):
            info.parse_mm_data({modality: [item]})


def test_dummy_inputs_are_the_largest_square_that_fits(processor):
    builder = processor.dummy_inputs
    assert builder.get_dummy_text({"image": 2}) == IMAGE_PLACEHOLDER * 2
    for seq_len, side in ((1024, 1280), (4096, 2048), (1 << 20, MAX_DUMMY_IMAGE_SIDE)):
        data = builder.get_dummy_mm_data(seq_len, {"image": 1}, {})
        (image,) = data["image"]
        assert image.size == (side, side)
    inputs = processor.get_dummy_inputs(1024, {"image": 1}, {})
    assert inputs.prompt == [IMAGE_MARKER_ID]
    out = processor.apply(inputs, TimingContext(enabled=False))
    assert out["prompt_token_ids"] == [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * 1024
    assert out["mm_kwargs"].get_data()["vision_pixels"].shape == (1024, 40, 40, 3)


@needs_ckpt
def test_golden_token_ids_match_the_real_template_and_transformers_processor(processor):
    import jinja2
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(str(CKPT))
    hf = transformers.AutoProcessor.from_pretrained(str(CKPT))
    assert tok.convert_tokens_to_ids("<|content_image|>") == IMAGE_MARKER_ID
    assert tok.convert_tokens_to_ids("<|unused_200054|>") == IMAGE_PAD_ID

    images = [Image.new("RGB", (100, 50), (9, 8, 7)), Image.new("RGB", (41, 85))]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Compare these."},
                {"type": "image"},
            ],
        }
    ]
    text = tok.apply_chat_template(conversation, tokenize=False)
    template_ids = tok.encode(text, add_special_tokens=False)
    # the template alone writes one marker and ONE pad per image
    assert template_ids.count(IMAGE_MARKER_ID) == 2
    assert template_ids.count(IMAGE_PAD_ID) == 2
    first = template_ids.index(IMAGE_MARKER_ID)
    assert template_ids[first + 1] == IMAGE_PAD_ID

    ours = _apply(processor, template_ids, images)["prompt_token_ids"]
    theirs = hf(text=text, images=images)["input_ids"]
    theirs = theirs[0].tolist() if hasattr(theirs, "tolist") else list(theirs[0])
    assert ours == theirs
    assert ours.count(IMAGE_PAD_ID) == 6 + 6
    assert ours.count(IMAGE_MARKER_ID) == 2

    # the template itself refuses audio (and any non image / text part)
    for part in ({"type": "audio"}, {"type": "video"}):
        with pytest.raises(jinja2.exceptions.TemplateError):
            tok.apply_chat_template(
                [{"role": "user", "content": [part]}], tokenize=False
            )
