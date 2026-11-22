#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini vision encoder: offline fp32 oracle and end-to-end checks.

The serving encoder (``vllm/models/inkling2_loop/amd/vision.py``) runs native HIP
kernels in fp16.  This file holds the offline references it is judged against (PyTorch
fp32, never imported by serving code) and a command line that runs the comparisons:

``oracle``
    An independent per-image fp32 implementation written from the architecture
    description (checkpoint tensor names in, ``[sum(rows * cols), text_hidden]`` out).
    ``emulate_fp16=True`` rounds every GEMM operand to fp16 exactly where the kernels do
    (weights, patches, LayerNorm outputs, q / k / v, attention output, GELU output,
    projector input) while the residual stream and all accumulations stay fp32.

``hf``
    The transformers ``Inkling2VisionModel`` (fp32) fed with the renamed checkpoint
    tensors and the HF image processor's pixels.

Pixels: the encoder normalises raw cells itself.  ``--pixel-norm clip_bf16`` (the
default, the NVIDIA vLLM reference: CLIP normalisation rounded to bf16 and mapped back
to [-1, 1]) or ``exact`` (the HF processor's ``2 v / 255 - 1``); the oracle and the
HF model are fed the same pixels, and bundles record theirs.

Modes (``python vision_check.py <mode> ...``):

``hf-tiny``
    Random bf16-exact weights of a tiny config: oracle vs transformers on CPU.  Proves
    the oracle (and the checkpoint -> transformers name mapping) before anything is
    measured against it.
``real``
    Real checkpoint weights and a real image: HF processor + transformers vs oracle on
    CPU, the processing module's pixels vs the HF processor's, and the per-layer
    activation ranges that decide whether fp16 storage can overflow.
``gpu``
    The native HIP encoder on the GPU (tiny random weights or the real checkpoint) vs
    the fp32 oracle: encoder output rows, max / mean error, cosine.

``HIP_VISIBLE_DEVICES`` must select exactly the GPU you were assigned for ``gpu``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CELL = 40
FP16_MAX = 65504.0
# Acceptance of the native fp16 encoder against the fp32 oracle on the real checkpoint
# (docs/rdna2/inkling2-vision.md).  ``within_floor`` compares the native error with the
# fp16 noise floor of the same image (the oracle's ``emulate_fp16``: every GEMM operand
# rounded to fp16 exactly where the kernels do, fp32 everywhere else), with scale-free
# metrics: mean row error at most FLOOR_FACTOR x the floor's, max |d| / rms at most
# FLOOR_FACTOR x the floor's + MAX_ABS_SLACK, minimum row cosine at most
# FLOOR_COS_SLACK below the floor's and at least COS_MIN.  The minimum cosine falls
# with the number of rows and with smooth images, which is why the gate is relative to
# the floor.  REL_ROW_MEAN / MAX_ABS_OVER_RMS / COS_MIN are the absolute caps the
# random-weight end-to-end test applies where no floor is computed.
FLOOR_FACTOR = 2.0
FLOOR_COS_SLACK = 2e-3
MAX_ABS_SLACK = 0.02
COS_MIN = 0.997
REL_ROW_MEAN = 4e-3
MAX_ABS_OVER_RMS = 0.25
VISUAL_PREFIX = "model.visual."

# pixel normalisations of the raw 0..255 (-1 padding) cells; the serving encoder's
# ``pixel_norm`` (vision.py) names and its default
PIXEL_NORMS = ("exact", "clip_bf16")
DEFAULT_PIXEL_NORM = "clip_bf16"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD_PROCESSOR = (0.26862954, 0.2613026, 0.2757771)  # the processor's spelling
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)  # the encoder's spelling

REAL_CONFIG = {
    "model_type": "inkling2_vision",
    "text_hidden_size": 2304,
    "temporal_patch_size": 2,
    "native_patch_size": 20,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 4096,
    "projector_bias": False,
    "hidden_size": 1024,
    "num_hidden_layers": 23,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 32,
    "layer_norm_eps": 1e-5,
}
TINY_CONFIG = {
    **REAL_CONFIG,
    "text_hidden_size": 192,
    "projection_hidden_dim": 256,
    "hidden_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
}


def derived(cfg: dict) -> dict:
    d = cfg["hidden_size"]
    return {
        "dim": d,
        "heads": cfg["num_attention_heads"],
        "head_dim": d // cfg["num_attention_heads"],
        "mlp": int(d * cfg["mlp_ratio"]),
        "patch": cfg["native_patch_size"],
        "merge": cfg["adapter_merge_size"],
        "layers": cfg["num_hidden_layers"],
        "grid": cfg["posemb_grid_size"],
        "eps": cfg["layer_norm_eps"],
        "proj": cfg["projection_hidden_dim"],
        "out": cfg["text_hidden_size"],
    }


# ---------------------------------------------------------------- weights


def state_names(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Checkpoint tensor names (``model.visual.`` stripped) and shapes."""
    s = derived(cfg)
    d, mlp, k = s["dim"], s["mlp"], 3 * s["patch"] ** 2
    shapes = {
        "backbone.embeddings.class_embedding": (d,),
        "backbone.embeddings.cls_position_embedding": (d,),
        "backbone.embeddings.patch_embedding.weight": (d, k),
        "backbone.embeddings.position_embedding": (s["grid"] ** 2, d),
        "backbone.ln_pre.weight": (d,),
        "backbone.ln_pre.bias": (d,),
        "proj_in.weight": (s["proj"], s["merge"] ** 2 * d),
        "proj_out.weight": (s["out"], s["proj"]),
    }
    for i in range(s["layers"]):
        p = f"backbone.layers.{i}."
        for ln in ("ln_1", "ln_2"):
            shapes[p + ln + ".weight"] = (d,)
            shapes[p + ln + ".bias"] = (d,)
        shapes[p + "ls_1.gamma"] = (d,)
        shapes[p + "ls_2.gamma"] = (d,)
        shapes[p + "mlp.c_fc.weight"] = (mlp, d)
        shapes[p + "mlp.c_fc.bias"] = (mlp,)
        shapes[p + "mlp.c_proj.weight"] = (d, mlp)
        shapes[p + "mlp.c_proj.bias"] = (d,)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[p + f"self_attn.{proj}.weight"] = (d, d)
            shapes[p + f"self_attn.{proj}.bias"] = (d,)
    return shapes


def random_state(cfg: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random fp32 tensors, every value exactly representable in bf16 (like the
    checkpoint), scaled so activations stay O(1) through all layers."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name, shape in state_names(cfg).items():
        if name.endswith(("ln_pre.weight", "ln_1.weight", "ln_2.weight")):
            t = 1.0 + 0.1 * torch.randn(shape, generator=gen)
        elif name.endswith((".bias", "ln_pre.bias")):
            t = 0.05 * torch.randn(shape, generator=gen)
        elif name.endswith(".gamma"):
            t = 0.1 + 0.05 * torch.rand(shape, generator=gen)
        elif name.endswith(("class_embedding", "cls_position_embedding")):
            t = torch.randn(shape, generator=gen)
        elif name.endswith("position_embedding"):
            t = 0.5 * torch.randn(shape, generator=gen)
        else:
            t = torch.randn(shape, generator=gen) * (0.8 / math.sqrt(shape[-1]))
        out[name] = t.bfloat16().float()
    return out


def load_real_state(ckpt: Path) -> dict[str, torch.Tensor]:
    """The ``model.visual.*`` tensors of the real checkpoint as fp32 (stripped)."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    with safe_open(str(ckpt / "model-00001-of-00001.safetensors"), "pt") as f:
        for key in f.keys():  # noqa: SIM118
            if key.startswith(VISUAL_PREFIX):
                out[key[len(VISUAL_PREFIX) :]] = f.get_tensor(key).float()
    return out


def hf_name(name: str) -> str:
    """Checkpoint tensor name -> ``Inkling2VisionModel`` state-dict name.

    Mirrors the ``inkling2`` entries of ``transformers/conversion_mapping.py``.
    """
    import regex as re

    name = name.replace("backbone.embeddings.", "embeddings.")
    if name == "embeddings.position_embedding":
        name = "embeddings.position_embedding.weight"
    name = name.replace("backbone.ln_pre.", "pre_layernorm.")
    name = name.replace("backbone.layers.", "layers.")
    name = name.replace("proj_in.", "merger.linear_1.")
    name = name.replace("proj_out.", "merger.linear_2.")
    name = re.sub(r"(layers\.\d+)\.ln_1\.", r"\1.layernorm_before.", name)
    name = re.sub(r"(layers\.\d+)\.ln_2\.", r"\1.layernorm_after.", name)
    name = re.sub(r"(layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1.", name)
    name = re.sub(r"(layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2.", name)
    name = re.sub(r"(layers\.\d+)\.ls_1\.gamma", r"\1.lambda_1", name)
    name = re.sub(r"(layers\.\d+)\.ls_2\.gamma", r"\1.lambda_2", name)
    return name


# ---------------------------------------------------------------- pixels


def normalize_raw(raw: torch.Tensor, mode: str = DEFAULT_PIXEL_NORM) -> torch.Tensor:
    """Raw 0..255 (or -1 padding) -> the encoder's patch pixels in fp32.

    ``exact``: the transformers processor's ``2 v / 255 - 1``.  ``clip_bf16``: the
    reference of the NVIDIA vLLM encoder, ``2 (bf16((v / 255 - mean) / clip_std) std +
    mean) - 1`` rounded to bf16, every operation an IEEE fp32 one (numpy, like the
    upstream test), written from that description.
    """
    if mode == "exact":
        return (2.0 * raw.float() - 255.0) / 255.0
    if mode != "clip_bf16":
        raise ValueError(f"pixel_norm {mode!r}, expected one of {PIXEL_NORMS}")
    f32 = np.float32
    mean = np.array(CLIP_MEAN, f32)
    clip = (raw.float().cpu().numpy() / f32(255) - mean) / np.array(
        CLIP_STD_PROCESSOR, f32
    )
    x = torch.from_numpy(clip).bfloat16().float().numpy() * np.array(CLIP_STD, f32)
    out = f32(2) * (x + mean) - f32(1)
    return torch.from_numpy(out).bfloat16().float().to(raw.device)


def hf_pixels(images: list, device: str = "cpu"):
    """The transformers image processor on PIL images -> (pixel_values, grid_thw)."""
    from transformers.models.inkling2.image_processing_inkling2 import (
        Inkling2ImageProcessor,
    )

    out = Inkling2ImageProcessor()(images, return_tensors="pt")
    return out["pixel_values"].to(device), out["image_grid_thw"]


# ---------------------------------------------------------------- oracle


def _fp16(x: torch.Tensor, on: bool) -> torch.Tensor:
    return x.half().float() if on else x


def _record(cap: dict | None, key: str, x: torch.Tensor) -> None:
    if cap is not None:
        cap.setdefault(key, []).append(float(x.abs().max()))


def _rope_tables(positions: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, ...]:
    """``positions [T, 2]`` (y, x; CLS 0): cos / sin ``[T, head_dim // 2]``.

    Pair ``p``: ``p < head_dim // 4`` rotates by x, the rest by y, with
    ``inv_freq[j] = 10000 ** (-4 j / head_dim)``.
    """
    quarter = head_dim // 4
    inv = 10000.0 ** (-torch.arange(quarter, dtype=torch.float64) * 4 / head_dim)
    inv = inv.float().to(positions.device)
    ang = torch.cat(
        [positions[:, 1:].float() * inv, positions[:, :1].float() * inv], -1
    )
    return ang.cos(), ang.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x [T, H, hd]`` interleaved pairs ``(x[2p], x[2p + 1])``."""
    even, odd = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)


def reference_image(
    state: dict[str, torch.Tensor],
    cfg: dict,
    cells: torch.Tensor,
    rows: int,
    cols: int,
    *,
    emulate_fp16: bool = False,
    capture: dict | None = None,
    raw_input: bool = True,
    pixel_norm: str = DEFAULT_PIXEL_NORM,
) -> torch.Tensor:
    """One image: ``cells [rows * cols, 40, 40, 3]`` -> ``[rows * cols, out]``.

    The cells are raw 0..255 (-1 padding), normalised by ``pixel_norm``; with
    ``raw_input=False`` they are the patch pixels themselves.

    Tokens are ordered like the transformers model: CLS, then the four native
    patches of every cell (top-left, top-right, bottom-left, bottom-right), cells
    row-major.
    """
    s = derived(cfg)
    dev = cells.device
    d, heads, hd, m, p = s["dim"], s["heads"], s["head_dim"], s["merge"], s["patch"]
    n_cells = rows * cols
    f16 = emulate_fp16
    st = {k: v.to(dev).float() for k, v in state.items()}

    x = normalize_raw(cells, pixel_norm) if raw_input else cells.float()
    patches = x.view(n_cells, m, p, m, p, 3).permute(0, 1, 3, 2, 4, 5)
    patches = _fp16(patches.reshape(n_cells * m * m, p * p * 3), f16)

    emb = patches @ _fp16(st["backbone.embeddings.patch_embedding.weight"], f16).T

    grid = st["backbone.embeddings.position_embedding"]
    grid = grid.view(s["grid"], s["grid"], d).permute(2, 0, 1)[None]
    pos = F.interpolate(grid, size=(rows * m, cols * m), mode="bilinear")[0]
    cell = torch.arange(n_cells, device=dev)
    sub = torch.arange(m * m, device=dev)
    py = ((cell // cols)[:, None] * m + (sub // m)[None]).reshape(-1)
    px = ((cell % cols)[:, None] * m + (sub % m)[None]).reshape(-1)
    emb = emb + pos[:, py, px].T

    cls = (
        st["backbone.embeddings.class_embedding"]
        + (st["backbone.embeddings.cls_position_embedding"])
    )
    h = torch.cat([cls[None], emb])
    eps = s["eps"]
    h = F.layer_norm(
        h,
        (d,),
        st["backbone.ln_pre.weight"],
        st["backbone.ln_pre.bias"],
        eps,
    )
    _record(capture, "embed_ln_pre", h)

    positions = torch.cat(
        [torch.zeros(1, 2, device=dev), torch.stack([py, px], -1) + 1]
    )
    cos, sin = _rope_tables(positions, hd)
    scale = hd**-0.5

    for i in range(s["layers"]):
        pre = f"backbone.layers.{i}."

        def w(name: str, pre: str = pre) -> torch.Tensor:
            return st[pre + name]

        a = _fp16(F.layer_norm(h, (d,), w("ln_1.weight"), w("ln_1.bias"), eps), f16)
        q = a @ _fp16(w("self_attn.q_proj.weight"), f16).T + w("self_attn.q_proj.bias")
        k = a @ _fp16(w("self_attn.k_proj.weight"), f16).T + w("self_attn.k_proj.bias")
        v = a @ _fp16(w("self_attn.v_proj.weight"), f16).T + w("self_attn.v_proj.bias")
        q = _apply_rope(q.view(-1, heads, hd), cos, sin)
        k = _apply_rope(k.view(-1, heads, hd), cos, sin)
        v = v.view(-1, heads, hd)
        q, k, v = (_fp16(t, f16) for t in (q, k, v))
        for tag, t in (("q", q), ("k", k), ("v", v)):
            _record(capture, tag, t)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            scale=scale,
        )[0]
        o = _fp16(o.transpose(0, 1).reshape(-1, d), f16)
        _record(capture, "attn_out", o)
        delta = o @ _fp16(w("self_attn.out_proj.weight"), f16).T
        delta = delta + w("self_attn.out_proj.bias")
        _record(capture, "attn_proj", delta)
        h = h + w("ls_1.gamma") * delta
        _record(capture, "resid_attn", h)

        a = _fp16(F.layer_norm(h, (d,), w("ln_2.weight"), w("ln_2.bias"), eps), f16)
        f1 = a @ _fp16(w("mlp.c_fc.weight"), f16).T + w("mlp.c_fc.bias")
        _record(capture, "fc1", f1)
        g = _fp16(F.gelu(f1), f16)
        _record(capture, "gelu", g)
        delta = g @ _fp16(w("mlp.c_proj.weight"), f16).T + w("mlp.c_proj.bias")
        _record(capture, "fc2", delta)
        h = h + w("ls_2.gamma") * delta
        _record(capture, "resid_mlp", h)

    pooled = _fp16(h[1:].reshape(n_cells, m * m * d), f16)
    z = pooled @ _fp16(st["proj_in.weight"], f16).T
    _record(capture, "proj_in", z)
    z = _fp16(F.gelu(z), f16)
    out = z @ _fp16(st["proj_out.weight"], f16).T
    _record(capture, "proj_out", out)
    return out


def reference_encode(
    state: dict[str, torch.Tensor],
    cfg: dict,
    raw_pixels: torch.Tensor,
    grid_thw: list[list[int]],
    *,
    emulate_fp16: bool = False,
    capture: dict | None = None,
    raw_input: bool = True,
    pixel_norm: str = DEFAULT_PIXEL_NORM,
) -> torch.Tensor:
    """Every image on its own through :func:`reference_image`, concatenated."""
    outs, start = [], 0
    for _, rows, cols in grid_thw:
        n = rows * cols
        outs.append(
            reference_image(
                state,
                cfg,
                raw_pixels[start : start + n],
                rows,
                cols,
                emulate_fp16=emulate_fp16,
                capture=capture,
                raw_input=raw_input,
                pixel_norm=pixel_norm,
            )
        )
        start += n
    return torch.cat(outs)


# ---------------------------------------------------------------- transformers


def hf_model(
    state: dict[str, torch.Tensor], cfg: dict, dtype: torch.dtype = torch.float32
):
    from transformers.models.inkling2.configuration_inkling2 import (
        Inkling2VisionConfig,
    )
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

    hf_cfg = Inkling2VisionConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        posemb_grid_size=cfg["posemb_grid_size"],
        layer_norm_eps=cfg["layer_norm_eps"],
        projection_hidden_dim=cfg["projection_hidden_dim"],
        projector_bias=cfg["projector_bias"],
        text_hidden_size=cfg["text_hidden_size"],
        patch_size=cfg["native_patch_size"],
        spatial_merge_size=cfg["adapter_merge_size"],
    )
    hf_cfg._attn_implementation = "sdpa"
    model = Inkling2VisionModel(hf_cfg).float().eval()
    renamed = {hf_name(k): v.float() for k, v in state.items()}
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    missing = [k for k in missing if "rotary_emb" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"hf load: missing {missing[:5]}, unexpected {unexpected[:5]}"
        )
    return model.to(dtype)


@torch.inference_mode()
def hf_encode(
    state: dict[str, torch.Tensor],
    cfg: dict,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``Inkling2VisionModel`` on HF-normalised ``pixel_values`` (fp32 by default;
    ``bfloat16`` is the checkpoint's own serving dtype, used as a noise baseline)."""
    model = hf_model(state, cfg, dtype)
    out = model(pixel_values=pixel_values.to(dtype), image_grid_thw=grid_thw)
    return out.pooler_output.float()


# ---------------------------------------------------------------- metrics


def compare(got: torch.Tensor, want: torch.Tensor) -> dict:
    """Row-wise comparison of two ``[N, D]`` outputs."""
    got, want = got.double().cpu(), want.double().cpu()
    diff = (got - want).abs()
    rms = want.pow(2).mean().sqrt()
    cos = F.cosine_similarity(got, want, dim=-1)
    rel_row = (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-30)
    worst = int(cos.argmin())
    norms = want.norm(dim=-1)
    return {
        "rows": int(got.shape[0]),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "ref_rms": float(rms),
        "max_abs_over_rms": float(diff.max() / rms),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "rel_row_max": float(rel_row.max()),
        "rel_row_mean": float(rel_row.mean()),
        "worst_row": worst,
        "worst_row_norm_over_median": float(norms[worst] / norms.median()),
    }


def within_floor(stats: dict, floor: dict) -> list[str]:
    """The acceptance criteria that a native-vs-fp32 comparison ``stats`` violates,
    given the fp16 noise floor ``floor`` (``compare`` outputs of the same image)."""
    bad = []
    if stats["cos_min"] < max(COS_MIN, floor["cos_min"] - FLOOR_COS_SLACK):
        bad.append("cos_min")
    if stats["rel_row_mean"] > FLOOR_FACTOR * floor["rel_row_mean"]:
        bad.append("rel_row_mean")
    limit = FLOOR_FACTOR * floor["max_abs_over_rms"] + MAX_ABS_SLACK
    if stats["max_abs_over_rms"] > limit:
        bad.append("max_abs_over_rms")
    return bad


def fp16_range_report(capture: dict) -> dict:
    """Per-tensor max over layers and the margin to the fp16 limit."""
    out = {}
    for key, vals in capture.items():
        mx = max(vals)
        out[key] = {"max_abs": mx, "fp16_margin": FP16_MAX / mx if mx else math.inf}
    return out


# ---------------------------------------------------------------- images


def load_images(paths: list[str], max_side: int | None) -> list:
    """Real images, each optionally center-cropped to at most ``max_side`` pixels."""
    from PIL import Image

    images = []
    for path in paths:
        img = Image.open(Path(path).expanduser())
        img.load()
        if max_side:
            w, h = img.size
            w2, h2 = min(w, max_side), min(h, max_side)
            left, top = (w - w2) // 2, (h - h2) // 2
            img = img.crop((left, top, left + w2, top + h2))
        images.append(img)
    return images


# ---------------------------------------------------------------- modes


def mode_hf_tiny(args: argparse.Namespace) -> int:
    cfg = TINY_CONFIG
    state = random_state(cfg, args.seed)
    grids = [(3, 5), (7, 2), (1, 1)]
    gen = torch.Generator().manual_seed(args.seed)
    cells = torch.randint(
        -1, 256, (sum(r * c for r, c in grids), CELL, CELL, 3), generator=gen
    )
    cells[cells == -1] = -1
    grid_thw = [[1, r, c] for r, c in grids]
    report, ok = {}, True
    for mode in PIXEL_NORMS:
        ref = reference_encode(state, cfg, cells, grid_thw, pixel_norm=mode)
        hf = hf_encode(state, cfg, normalize_raw(cells, mode), torch.tensor(grid_thw))
        stats = report[f"oracle_vs_hf_fp32[{mode}]"] = compare(ref, hf)
        ok &= stats["max_abs_over_rms"] < 1e-4 and stats["cos_min"] > 1 - 1e-7
    print(json.dumps(report, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def mode_real(args: argparse.Namespace) -> int:
    from vllm.models.inkling2_loop.amd.processing import Inkling2ImageProcessor

    ckpt = Path(args.ckpt).expanduser()
    cfg = json.loads((ckpt / "config.json").read_text())["vision_config"]
    t0 = time.time()
    state = load_real_state(ckpt)
    print(f"loaded {len(state)} visual tensors in {time.time() - t0:.1f}s")
    images = load_images(args.image, args.max_side)
    out = Inkling2ImageProcessor().preprocess(images)
    cells, grid = out["vision_pixels"], out["vision_grid_thw"]
    hf_pix, hf_grid = hf_pixels(images)
    pix_err = float((normalize_raw(cells, "exact") - hf_pix).abs().max())
    print(
        f"images {[i.size for i in images]}, grid {grid.tolist()}, "
        f"processing vs HF max|dpix| {pix_err:.3e}"
    )
    assert torch.equal(grid, hf_grid), (grid, hf_grid)

    mode = args.pixel_norm
    pix = normalize_raw(cells, mode)  # what the encoder's patch GEMM is given
    capture: dict = {}
    t0 = time.time()
    ref = reference_encode(
        state, cfg, cells, grid.tolist(), capture=capture, pixel_norm=mode
    )
    print(f"oracle fp32 [{mode}]: {time.time() - t0:.1f}s, out {tuple(ref.shape)}")
    report = {"pixel_norm": mode, "fp16_range": fp16_range_report(capture)}
    t0 = time.time()
    hf = hf_encode(state, cfg, pix, hf_grid)
    print(f"transformers fp32: {time.time() - t0:.1f}s")
    report["oracle_vs_hf_fp32"] = compare(ref, hf)
    bf = None
    if args.bf16:
        bf = hf_encode(state, cfg, pix, hf_grid, torch.bfloat16)
        report["hf_bf16_vs_fp32"] = compare(bf, ref)
    emu = reference_encode(
        state, cfg, cells, grid.tolist(), emulate_fp16=True, pixel_norm=mode
    )
    report["fp16_emulation_vs_fp32"] = compare(emu, ref)
    if args.pixel_effect:
        other = next(m for m in PIXEL_NORMS if m != mode)
        alt = reference_encode(state, cfg, cells, grid.tolist(), pixel_norm=other)
        report[f"fp32_oracle[{other}]_vs_fp32_oracle[{mode}]"] = compare(alt, ref)
    print(json.dumps(report, indent=1))
    if args.save:
        bundle = {
            "cells": cells,
            "grid": grid,
            "ref": ref,
            "emu": emu,
            "bf16": bf,
            "pixel_norm": mode,
        }
        torch.save(bundle, args.save)
    return 0


def _gpu_case(enc, state, cfg, cells, grid, bundle, args) -> tuple[dict, bool]:
    # a bundle written before ``pixel_norm`` existed holds the exact normalisation
    mode = bundle.get("pixel_norm", "exact") if bundle else args.pixel_norm
    enc.pixel_norm = mode
    device = torch.device("cuda")
    cells_dev = cells.to(device)
    grid_t = torch.tensor(grid)
    got = enc(cells_dev, grid_t)  # warm up
    torch.accelerator.synchronize()
    t0 = time.time()
    for _ in range(args.reps):
        got = enc(cells_dev, grid_t)
    torch.accelerator.synchronize()
    dt = (time.time() - t0) / args.reps

    if bundle:
        ref, emu, bf = bundle["ref"], bundle["emu"], bundle.get("bf16")
    else:
        ref = reference_encode(state, cfg, cells, grid, pixel_norm=mode)
        emu = reference_encode(
            state, cfg, cells, grid, emulate_fp16=True, pixel_norm=mode
        )
        bf = None
        if args.bf16:
            pix = normalize_raw(cells, mode)
            bf = hf_encode(state, cfg, pix, torch.tensor(grid), torch.bfloat16)
    report = {"pixel_norm": mode}
    report["native_vs_fp32_oracle"] = compare(got.float().cpu(), ref)
    report["native_vs_fp16_emulating_oracle"] = compare(got.float().cpu(), emu)
    report["fp16_emulating_oracle_vs_fp32"] = compare(emu, ref)
    if bf is not None:
        report["hf_bf16_vs_fp32"] = compare(bf, ref)
        report["native_vs_hf_bf16"] = compare(got.float().cpu(), bf)

    # batching must not change a single bit: every image alone vs inside the batch
    start, same = 0, True
    for g in grid:
        n = g[1] * g[2]
        alone = enc(cells_dev[start : start + n], torch.tensor([g]))
        same &= bool(torch.equal(alone, got[start : start + n]))
        start += n
    report["batch_equals_single_bitwise"] = same

    s = derived(cfg)
    tokens = sum(4 * r * c + 1 for _, r, c in grid)
    lin = 2 * tokens * s["layers"] * (4 * s["dim"] ** 2 + 2 * s["dim"] * s["mlp"])
    att = sum(4 * (4 * r * c + 1) ** 2 * s["dim"] * s["layers"] for _, r, c in grid)
    emb = 2 * 4 * sum(r * c for _, r, c in grid) * 3 * s["patch"] ** 2 * s["dim"]
    cells_n = sum(r * c for _, r, c in grid)
    proj = 2 * cells_n * (4 * s["dim"] * s["proj"] + s["proj"] * s["out"])
    report["perf"] = {
        "tokens": tokens,
        "seconds": dt,
        "tflops_total": (lin + att + emb + proj) / dt / 1e12,
        "gflop_attention": att / 1e9,
        "gflop_linear": (lin + emb + proj) / 1e9,
    }
    bad = within_floor(
        report["native_vs_fp32_oracle"], report["fp16_emulating_oracle_vs_fp32"]
    )
    report["violated"] = bad + ([] if same else ["batch_equals_single_bitwise"])
    print(json.dumps(report, indent=1))
    return report, not report["violated"]


def mode_gpu(args: argparse.Namespace) -> int:
    from vllm.models.inkling2_loop.amd.processing import Inkling2ImageProcessor
    from vllm.models.inkling2_loop.amd.vision import Inkling2VisionEncoder

    cases = []  # (name, cells, grid, oracle bundle or None)
    if args.real:
        ckpt = Path(args.ckpt).expanduser()
        cfg = json.loads((ckpt / "config.json").read_text())["vision_config"]
        state = load_real_state(ckpt)
        if args.ref:
            for path in args.ref:
                b = torch.load(path)
                cases.append((path, b["cells"], b["grid"].tolist(), b))
        else:
            images = load_images(args.image, args.max_side)
            out = Inkling2ImageProcessor().preprocess(images)
            grid = out["vision_grid_thw"].tolist()
            cases.append(("images", out["vision_pixels"], grid, None))
    else:
        cfg = TINY_CONFIG
        state = random_state(cfg, args.seed)
        grids = [(3, 5), (7, 2), (1, 1)]
        gen = torch.Generator().manual_seed(args.seed)
        n = sum(r * c for r, c in grids)
        cells = torch.randint(-1, 256, (n, CELL, CELL, 3), generator=gen).half()
        cases.append(("tiny", cells, [[1, r, c] for r, c in grids], None))

    device = torch.device("cuda")
    enc = Inkling2VisionEncoder(cfg, cfg["text_hidden_size"]).to(device)
    loaded = enc.load_weights((n, t.bfloat16().to(device)) for n, t in state.items())
    assert loaded == {n for n, _ in enc.named_parameters()}, "weights not loaded"
    ok = True
    for name, cells, grid, bundle in cases:
        print(f"== {name}: grid {grid}")
        _, case_ok = _gpu_case(enc, state, cfg, cells, grid, bundle, args)
        print(f"== {name}: {'PASS' if case_ok else 'FAIL'}")
        ok &= case_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("hf-tiny")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=mode_hf_tiny)
    for name, fn in (("real", mode_real), ("gpu", mode_gpu)):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", default="~/hfcache/inkling2-mini")
        p.add_argument(
            "--image", nargs="+", default=["tests/v1/ec_connector/integration/hato.jpg"]
        )
        p.add_argument("--max-side", type=int, default=640)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--save", help="real: write the oracle bundle (cells, refs)")
        p.add_argument(
            "--pixel-norm",
            choices=PIXEL_NORMS,
            default=DEFAULT_PIXEL_NORM,
            help="real / gpu without --ref: pixel normalisation (bundles carry theirs)",
        )
        p.add_argument(
            "--pixel-effect",
            action="store_true",
            help="real: also run the fp32 oracle with the other pixel norm",
        )
        p.add_argument(
            "--ref", nargs="+", help="gpu: oracle bundles to use instead of recomputing"
        )
        p.add_argument("--real", action="store_true")
        p.add_argument("--reps", type=int, default=3)
        p.add_argument(
            "--bf16", action="store_true", help="add the HF bf16 noise baseline"
        )
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
