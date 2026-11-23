# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end Inkling-2 vision encoder on gfx1030 (native HIP kernels, fp16).

The encoder output is compared with the offline fp32 oracle of
``tools/rdna/inkling2/vision_check.py`` (itself checked against the transformers
``Inkling2VisionModel`` in ``vision_check.py hf-tiny`` / ``real``) and, where the
transformers build with ``inkling2`` is importable, with that model directly:

* a tiny random-weight config (3 blocks, width 128) with three images,
* the real config's shapes (width 1024, 16 heads, projector 4096 -> 2304) with random
  weights and two blocks, through ``Inkling2MultiModalMixin.embed_multimodal``,
* the real checkpoint weights on a real photo and an RGBA image
  (``~/hfcache/inkling2-mini`` / ``$INKLING2_CKPT``; skipped when absent).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.inkling2_loop.amd import vision as vision_mod
from vllm.models.inkling2_loop.amd.multimodal import Inkling2MultiModalMixin
from vllm.models.inkling2_loop.amd.processing import Inkling2ImageProcessor
from vllm.models.inkling2_loop.amd.vision import Inkling2VisionEncoder
from vllm.platforms.rdna import on_rdna2

pytestmark = pytest.mark.skipif(not on_rdna2(), reason="RDNA2 (gfx10x) HIP path only")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "rdna" / "inkling2"))
import vision_check as tool  # noqa: E402

CKPT = Path(os.environ.get("INKLING2_CKPT", "~/hfcache/inkling2-mini")).expanduser()
SHARD = CKPT / "model-00001-of-00001.safetensors"
PHOTO = ROOT / "tests" / "v1" / "ec_connector" / "integration" / "hato.jpg"
RGBA = ROOT / "tests" / "multimodal" / "assets" / "rgba.png"
HF_INKLING2 = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("transformers.models.inkling2") is not None
)
DEV = torch.device("cuda")


def _encoder(cfg: dict, state: dict) -> Inkling2VisionEncoder:
    enc = Inkling2VisionEncoder(cfg, cfg["text_hidden_size"]).to(DEV)
    loaded = enc.load_weights((n, t.bfloat16().to(DEV)) for n, t in state.items())
    assert loaded == {n for n, _ in enc.named_parameters()}
    return enc


def _random_cells(grids, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    n = sum(r * c for r, c in grids)
    cells = torch.randint(-1, 256, (n, 40, 40, 3), generator=gen)
    return cells.half()


def _grid_thw(grids) -> torch.Tensor:
    return torch.tensor([[1, r, c] for r, c in grids])


def _within(stats: dict) -> None:
    assert stats["cos_min"] >= tool.COS_MIN, stats
    assert stats["rel_row_mean"] <= tool.REL_ROW_MEAN, stats
    assert stats["max_abs_over_rms"] <= tool.MAX_ABS_OVER_RMS, stats


@pytest.fixture(scope="module")
def tiny():
    cfg, grids = tool.TINY_CONFIG, [(3, 5), (7, 2), (1, 1)]
    state = tool.random_state(cfg, seed=0)
    return cfg, grids, state, _encoder(cfg, state), _random_cells(grids)


def test_default_pixel_norm_is_the_nvidia_reference(tiny):
    enc = tiny[3]
    assert enc.pixel_norm == vision_mod.DEFAULT_PIXEL_NORM == "clip_bf16"
    assert tool.DEFAULT_PIXEL_NORM == vision_mod.DEFAULT_PIXEL_NORM
    assert sorted(vision_mod.PIXEL_NORMS) == sorted(tool.PIXEL_NORMS)


@pytest.mark.parametrize("pixel_norm", tool.PIXEL_NORMS)
def test_tiny_matches_the_fp32_oracle(tiny, pixel_norm, monkeypatch):
    cfg, grids, state, enc, cells = tiny
    monkeypatch.setattr(enc, "pixel_norm", pixel_norm)
    out = enc(cells.to(DEV), _grid_thw(grids))
    assert out.dtype == torch.float16
    assert tuple(out.shape) == (sum(r * c for r, c in grids), cfg["text_hidden_size"])
    ref = tool.reference_encode(
        state, cfg, cells, _grid_thw(grids).tolist(), pixel_norm=pixel_norm
    )
    stats = tool.compare(out.float().cpu(), ref)
    assert stats["cos_min"] >= 0.9999, stats
    assert stats["max_abs_over_rms"] <= 5e-3, stats
    assert stats["rel_row_mean"] <= 2e-3, stats


@pytest.mark.skipif(not HF_INKLING2, reason="transformers without inkling2")
@pytest.mark.parametrize("pixel_norm", tool.PIXEL_NORMS)
def test_tiny_matches_transformers_vision_model(tiny, pixel_norm, monkeypatch):
    cfg, grids, state, enc, cells = tiny
    monkeypatch.setattr(enc, "pixel_norm", pixel_norm)
    out = enc(cells.to(DEV), _grid_thw(grids))
    pixels = tool.normalize_raw(cells, pixel_norm)
    hf = tool.hf_encode(state, cfg, pixels, _grid_thw(grids))
    stats = tool.compare(out.float().cpu(), hf)
    assert stats["cos_min"] >= 0.9999, stats
    assert stats["max_abs_over_rms"] <= 5e-3, stats


def test_batching_and_chunking_do_not_change_a_bit(tiny, monkeypatch):
    cfg, grids, state, enc, cells = tiny
    cells_dev = cells.to(DEV)
    batch = enc(cells_dev, _grid_thw(grids))
    start = 0
    for grid in grids:
        n = grid[0] * grid[1]
        alone = enc(cells_dev[start : start + n], _grid_thw([grid]))
        assert torch.equal(alone, batch[start : start + n]), grid
        start += n
    # a tiny token cap forces one launch group per image
    monkeypatch.setattr(vision_mod, "MAX_CHUNK_TOKENS", 10)
    assert vision_mod.split_chunks(grids, vision_mod.MAX_CHUNK_TOKENS) == [
        [g] for g in grids
    ]
    assert torch.equal(enc(cells_dev, _grid_thw(grids)), batch)


def test_ragged_image_padded_with_minus_one_matches_the_oracle(tiny):
    # the bottom-right cell of a 2x2 grid is 20x20 pixels of image and 1200 padding
    # pixels: raw -1 normalises to -1.0078 like any other pixel and is attended to
    cfg, _, state, enc, _ = tiny
    cells = _random_cells([(2, 2)], seed=9)
    cells[3, 20:] = -1
    cells[3, :, 20:] = -1
    out = enc(cells.to(DEV), _grid_thw([(2, 2)]))
    ref = tool.reference_encode(state, cfg, cells, [[1, 2, 2]])
    stats = tool.compare(out.float().cpu(), ref)
    assert stats["cos_min"] >= 0.9999, stats
    unpadded = cells.clone()
    unpadded[3, 20:] = 0
    unpadded[3, :, 20:] = 0
    other = enc(unpadded.to(DEV), _grid_thw([(2, 2)]))
    assert not torch.equal(other, out)  # the padding value is part of the input


def test_inputs_are_validated(tiny):
    _, grids, _, enc, cells = tiny
    with pytest.raises(ValueError, match="do not match"):
        enc(cells.to(DEV)[:-1], _grid_thw(grids))
    bf16 = enc(cells.to(DEV).bfloat16(), _grid_thw(grids))  # exact for 0..255 and -1
    assert torch.equal(bf16, enc(cells.to(DEV), _grid_thw(grids)))


# -------------------------------------------------- embed_multimodal contract


class _TextStandIn(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("text embedding is not used here")

    def load_weights(self, weights):
        return {n for n, _ in weights}


class _Model(Inkling2MultiModalMixin, _TextStandIn, SupportsMultiModal):
    pass


def test_embed_multimodal_contract_at_the_real_shapes(monkeypatch):
    # real widths (1024 / 16 heads / 4096 -> 2304), two blocks, random weights
    cfg = {**tool.REAL_CONFIG, "num_hidden_layers": 2}
    state = tool.random_state(cfg, seed=1)
    monkeypatch.setattr(
        "vllm.distributed.get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
    )
    from vllm.config.multimodal import MultiModalConfig

    mm = MultiModalConfig()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vision_config=cfg, text_config=SimpleNamespace(hidden_size=2304)
            ),
            get_multimodal_config=lambda: mm,
        )
    )
    with torch.device(DEV):
        model = _Model(vllm_config=vllm_config)
    loaded = model.load_weights(
        (tool.VISUAL_PREFIX + n, t.bfloat16()) for n, t in state.items()
    )
    visual = model.visual
    assert visual is not None
    assert loaded == {f"visual.{n}" for n, _ in visual.named_parameters()}
    # the constructor's device context placed everything (incl. buffers) on the GPU
    assert {p.device.type for p in visual.parameters()} == {"cuda"}
    assert visual.inv_freq.device.type == "cuda" and visual.cls.is_cuda

    grids = [(2, 3), (1, 1), (4, 2)]
    cells = _random_cells(grids, seed=3)
    out = model.embed_multimodal(
        vision_pixels=cells.to(DEV), vision_grid_thw=_grid_thw(grids)
    )
    assert [tuple(t.shape) for t in out] == [(6, 2304), (1, 2304), (8, 2304)]
    assert sum(t.shape[0] for t in out) == sum(r * c for r, c in grids)
    assert all(t.dtype == torch.float16 and t.is_cuda for t in out)
    ref = tool.reference_encode(state, cfg, cells, _grid_thw(grids).tolist())
    _within(tool.compare(torch.cat(out).float().cpu(), ref))


# ----------------------------------------------------- real weights, real images


@pytest.mark.skipif(
    not (SHARD.exists() and PHOTO.exists() and RGBA.exists()),
    reason=f"real checkpoint ({CKPT}) or the test images are missing",
)
@pytest.mark.parametrize("pixel_norm", tool.PIXEL_NORMS)
def test_real_checkpoint_on_real_images(pixel_norm):
    from PIL import Image

    cfg = json.loads((CKPT / "config.json").read_text())["vision_config"]
    state = tool.load_real_state(CKPT)
    enc = _encoder(cfg, state)
    enc.pixel_norm = pixel_norm

    photo, rgba = Image.open(PHOTO), Image.open(RGBA)
    images = [photo.crop((1000, 600, 1640, 1240)), rgba.crop((0, 0, 640, 600))]
    out = Inkling2ImageProcessor().preprocess(images)
    cells, grid = out["vision_pixels"], out["vision_grid_thw"]
    assert grid.tolist() == [[1, 16, 16], [1, 15, 16]]

    got = enc(cells.to(DEV), grid)
    assert tuple(got.shape) == (496, 2304) and got.dtype == torch.float16
    assert torch.isfinite(got).all()

    capture: dict = {}
    ref = tool.reference_encode(
        state, cfg, cells, grid.tolist(), capture=capture, pixel_norm=pixel_norm
    )
    stats = tool.compare(got.float().cpu(), ref)
    _within(stats)
    # the fp16 noise floor of the same network, for scale
    emu = tool.reference_encode(
        state, cfg, cells, grid.tolist(), emulate_fp16=True, pixel_norm=pixel_norm
    )
    floor = tool.compare(emu, ref)
    assert not tool.within_floor(stats, floor), (stats, floor)
    # fp16 storage cannot overflow on these images: >= 100x headroom everywhere
    report = tool.fp16_range_report(capture)
    assert min(v["fp16_margin"] for v in report.values()) >= 100, report

    start = 0
    for g in grid.tolist():
        n = g[1] * g[2]
        assert torch.equal(
            got[start : start + n],
            enc(cells[start : start + n].to(DEV), torch.tensor([g])),
        )
        start += n
