# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini vision encoder on native HIP kernels (RDNA2 / gfx1030).

``Inkling2VisionEncoder`` is the 23-block ViT (width 1024, 16 heads of 64, 20x20
pixel patches, interleaved axial 2-D RoPE, LayerScale) with the 2x2 patch merger
and the two-layer projector of the checkpoint's ``model.visual.*`` tensors.  Every
operation after the CPU image preprocessing runs in ``csrc/rocm/inkling2_vit_rdna2.cu``
(``ops.inkling2_vit_*``): no Triton, no ATen / rocBLAS arithmetic.

Numerics: GEMM operands are fp16 with fp32 accumulation, the residual stream is fp32,
LayerNorm / softmax statistics / GELU are fp32.  The checkpoint is bf16; weights are
cast to fp16 at load (every bf16 value of the real checkpoint is exactly
representable except 60 487 of 318 256 128 values below 6.1e-5 in magnitude, whose
absolute error is at most 3e-8).  See ``docs/rdna2/inkling2-vision.md``.

Pixels: the raw cells are normalised inside the patch gather kernel.  ``pixel_norm``
``"clip_bf16"`` (default) reproduces the reference of the NVIDIA vLLM implementation
(CLIP normalisation rounded to bf16, mapped back to [-1, 1], rounded to bf16);
``"exact"`` is the transformers image processor's ``2 v / 255 - 1``.  A key
``pixel_norm`` of the vision config selects it when the constructor argument is not
given.

Input: ``pixels`` ``[sum(rows * cols), 40, 40, 3]`` raw RGB cells (fp16 or bf16,
values 0..255, padding -1) and ``grid_thw`` ``[n_images, 3]`` = ``[1, rows, cols]``
(CPU).  Output: ``[sum(rows * cols), text_hidden_size]`` fp16, one row per
decoder image token, images in input order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import regex as re
import torch
from torch import nn

from vllm import _custom_ops as ops

CELL = 40
# patch features of a native 20x20 patch (20 * 20 * 3), zero-padded to a multiple of
# the GEMM's K tile so the patch GEMM has no K tail (the pad columns are zero in both
# operands)
PATCH_K = 1200
PATCH_K_PAD = 1216
# one launch encodes at most this many ViT tokens (whole images only): bounds the
# activation workspace and keeps every GEMM operand inside the kernels' 32-bit offsets
MAX_CHUNK_TOKENS = 1 << 16
# an image is never split across launches, so it must fit one
MAX_IMAGE_TOKENS = MAX_CHUNK_TOKENS
# ``pixel_norm`` -> mode of ``ops.inkling2_vit_gather_patches``
PIXEL_NORMS = {"exact": 0, "clip_bf16": 1}
DEFAULT_PIXEL_NORM = "clip_bf16"
_QKV = re.compile(r"^(backbone\.layers\.\d+\.self_attn\.)([qkv])_proj\.(weight|bias)$")


@dataclass(frozen=True)
class VisionPlan:
    """Per-batch geometry shared by all kernels.

    Attributes:
        meta: int32 ``[n_images, 4]`` = rows, cols, first token, first patch row.
        blk_prefix: int32 ``[n_images + 1]``, prefix sums of 64-row query blocks.
        num_tokens: ViT tokens (a CLS plus ``4 rows cols`` patches per image).
        num_patches: ``4 * cells``.
        num_cells: decoder image tokens.

    """

    meta: np.ndarray
    blk_prefix: np.ndarray
    num_tokens: int
    num_patches: int
    num_cells: int


def build_plan(grids: Sequence[tuple[int, int]]) -> VisionPlan:
    """Geometry tables for images of ``(rows, cols)`` cells."""
    meta = np.empty((len(grids), 4), dtype=np.int32)
    blk = np.zeros(len(grids) + 1, dtype=np.int64)
    tok = patch = 0
    for i, (rows, cols) in enumerate(grids):
        if rows < 1 or cols < 1:
            raise ValueError(f"image grid must be positive, got {(rows, cols)}")
        meta[i] = (rows, cols, tok, patch)
        n_tok = 4 * rows * cols + 1
        blk[i + 1] = blk[i] + (n_tok + 63) // 64
        tok += n_tok
        patch += 4 * rows * cols
    return VisionPlan(
        meta=meta,
        blk_prefix=blk.astype(np.int32),
        num_tokens=tok,
        num_patches=patch,
        num_cells=patch // 4,
    )


def parse_grids(
    grid_thw: torch.Tensor | Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    """``[n, 3]`` (or a list of ``[1, rows, cols]``) -> ``[(rows, cols)]``."""
    rows_cols = []
    for item in grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw:
        t, rows, cols = (int(v) for v in item)
        if t != 1:
            raise ValueError(f"only still images are supported, got grid {item}")
        rows_cols.append((rows, cols))
    return rows_cols


def split_chunks(
    grids: Sequence[tuple[int, int]], max_tokens: int = MAX_CHUNK_TOKENS
) -> list[list[tuple[int, int]]]:
    """Consecutive images grouped so a group has at most ``max_tokens`` ViT tokens
    (an image larger than that is a group of its own)."""
    chunks: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    tokens = 0
    for rows, cols in grids:
        n = 4 * rows * cols + 1
        if current and tokens + n > max_tokens:
            chunks.append(current)
            current, tokens = [], 0
        current.append((rows, cols))
        tokens += n
    if current:
        chunks.append(current)
    return chunks


def _as_dict(config: Any) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    return config.to_dict()


class Inkling2VisionEncoder(nn.Module):
    """ViT + merger + projector; weights are named like the checkpoint without the
    ``model.visual.`` prefix, except that q / k / v are one fused ``qkv_proj``."""

    def __init__(
        self,
        vision_config: Mapping[str, Any] | Any,
        text_hidden_size: int | None = None,
        prefix: str = "",
        pixel_norm: str | None = None,
    ) -> None:
        super().__init__()
        cfg = _as_dict(vision_config)
        self.dim = d = int(cfg["hidden_size"])
        self.heads = int(cfg["num_attention_heads"])
        self.mlp_dim = int(d * float(cfg.get("mlp_ratio", 4.0)))
        self.num_layers = int(cfg["num_hidden_layers"])
        self.grid = int(cfg["posemb_grid_size"])
        self.eps = float(cfg.get("layer_norm_eps", 1e-5))
        self.proj_dim = int(cfg["projection_hidden_dim"])
        self.out_dim = int(text_hidden_size or cfg["text_hidden_size"])
        self.pixel_norm = pixel_norm or cfg.get("pixel_norm", DEFAULT_PIXEL_NORM)
        if self.pixel_norm not in PIXEL_NORMS:
            raise ValueError(
                f"pixel_norm must be one of {sorted(PIXEL_NORMS)}, got "
                f"{self.pixel_norm!r}"
            )
        if int(cfg["native_patch_size"]) != 20 or int(cfg["adapter_merge_size"]) != 2:
            raise NotImplementedError(
                "kernels are written for 20x20 native patches and a 2x2 merge, got "
                f"{cfg['native_patch_size']} / {cfg['adapter_merge_size']}"
            )
        if d // self.heads != 64 or d % self.heads:
            raise NotImplementedError(f"head_dim must be 64, got {d} / {self.heads}")
        if d % 128 or d > 2048:
            raise NotImplementedError(f"hidden_size must be a multiple of 128, got {d}")
        if cfg.get("projector_bias", False):
            raise NotImplementedError("projector bias is not part of the checkpoint")
        if cfg.get("hidden_act", "gelu") not in ("gelu", "gelu_python"):
            raise NotImplementedError(f"hidden_act {cfg.get('hidden_act')}")
        if (4 * d) % 32 or self.proj_dim % 32 or self.mlp_dim % 32:
            raise NotImplementedError("GEMM K dimensions must be multiples of 32")

        f16, f32 = torch.float16, torch.float32

        def p(
            *shape: int, dtype: torch.dtype = f32, zeros: bool = False
        ) -> nn.Parameter:
            make = torch.zeros if zeros else torch.empty
            return nn.Parameter(make(*shape, dtype=dtype), requires_grad=False)

        bb = nn.Module()
        emb = nn.Module()
        emb.patch_embedding = nn.Module()
        emb.patch_embedding.weight = p(d, PATCH_K_PAD, dtype=f16, zeros=True)
        emb.position_embedding = p(self.grid * self.grid, d)
        emb.class_embedding = p(d)
        emb.cls_position_embedding = p(d)
        bb.embeddings = emb
        bb.ln_pre = nn.Module()
        bb.ln_pre.weight = p(d)
        bb.ln_pre.bias = p(d)
        bb.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.Module()
            for ln in ("ln_1", "ln_2"):
                m = nn.Module()
                m.weight = p(d)
                m.bias = p(d)
                setattr(layer, ln, m)
            for ls in ("ls_1", "ls_2"):
                m = nn.Module()
                m.gamma = p(d)
                setattr(layer, ls, m)
            layer.self_attn = nn.Module()
            layer.self_attn.qkv_proj = nn.Module()
            layer.self_attn.qkv_proj.weight = p(3 * d, d, dtype=f16)
            layer.self_attn.qkv_proj.bias = p(3 * d)
            layer.self_attn.out_proj = nn.Module()
            layer.self_attn.out_proj.weight = p(d, d, dtype=f16)
            layer.self_attn.out_proj.bias = p(d)
            layer.mlp = nn.Module()
            layer.mlp.c_fc = nn.Module()
            layer.mlp.c_fc.weight = p(self.mlp_dim, d, dtype=f16)
            layer.mlp.c_fc.bias = p(self.mlp_dim)
            layer.mlp.c_proj = nn.Module()
            layer.mlp.c_proj.weight = p(d, self.mlp_dim, dtype=f16)
            layer.mlp.c_proj.bias = p(d)
            bb.layers.append(layer)
        self.backbone = bb
        self.proj_in = nn.Module()
        self.proj_in.weight = p(self.proj_dim, 4 * d, dtype=f16)
        self.proj_out = nn.Module()
        self.proj_out.weight = p(self.out_dim, self.proj_dim, dtype=f16)

        # head_dim / 4 rotation frequencies; built by a factory function so the
        # constructor's default-device context (the loader's) places it with the weights
        quarter = 16
        inv = 10000.0 ** (-np.arange(quarter, dtype=np.float64) / quarter)
        self.register_buffer("inv_freq", torch.tensor(inv, dtype=f32), persistent=False)
        self.register_buffer("cls", torch.zeros(d, dtype=f32), persistent=False)
        self._seen: dict[str, int] = {}

    # ------------------------------------------------------------- weights

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        """Copy one checkpoint tensor (name without ``model.visual.``) into place.

        Call ``finish_weights`` after the last one.
        """
        m = _QKV.match(name)
        if m:
            dst = f"{m.group(1)}qkv_proj.{m.group(3)}"
            lo = "qkv".index(m.group(2)) * self.dim
            window: tuple[slice, ...] = (slice(lo, lo + self.dim),)
        else:
            dst = name
            patch = name.endswith("patch_embedding.weight")
            window = (slice(None), slice(0, PATCH_K)) if patch else ()
        try:
            param = self.get_parameter(dst)
        except AttributeError:
            raise KeyError(f"unexpected vision tensor {name!r}") from None
        target = param[window] if window else param
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"{name}: checkpoint {tuple(tensor.shape)}, model {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor)
        self._seen[dst] = self._seen.get(dst, 0) + 1

    def finish_weights(self) -> set[str]:
        """Derive the CLS row and report what is loaded.

        Returns:
            The names of the parameters that are completely loaded.

        """
        loaded = set()
        for name, _ in self.named_parameters():
            fused = name.endswith(("qkv_proj.weight", "qkv_proj.bias"))
            if self._seen.get(name, 0) == (3 if fused else 1):
                loaded.add(name)
        self._seen.clear()
        emb = "backbone.embeddings."
        if {emb + "class_embedding", emb + "cls_position_embedding"} <= loaded:
            with torch.no_grad():
                self.cls.copy_(
                    self.get_parameter(emb + "class_embedding")
                    + self.get_parameter(emb + "cls_position_embedding")
                )
        return loaded

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for name, tensor in weights:
            self.load_weight(name, tensor)
        return self.finish_weights()

    # ------------------------------------------------------------- forward

    @torch.inference_mode()
    def forward(
        self, pixels: torch.Tensor, grid_thw: torch.Tensor | Sequence[Sequence[int]]
    ) -> torch.Tensor:
        """Encode images.

        Args:
            pixels: ``[sum(rows * cols), 40, 40, 3]`` raw cells (fp16 / bf16).
            grid_thw: ``[n_images, 3]`` = ``[1, rows, cols]``, on the CPU.

        Returns:
            ``[sum(rows * cols), text_hidden_size]`` fp16.

        """
        grids = parse_grids(grid_thw)
        for rows, cols in grids:
            if 4 * rows * cols + 1 > MAX_IMAGE_TOKENS:
                raise ValueError(
                    f"image grid {(rows, cols)} has {4 * rows * cols + 1} ViT tokens, "
                    f"more than the {MAX_IMAGE_TOKENS} of one launch"
                )
        total = sum(r * c for r, c in grids)
        if pixels.shape[0] != total or tuple(pixels.shape[1:]) != (CELL, CELL, 3):
            raise ValueError(
                f"pixels {tuple(pixels.shape)} do not match grids {grids} "
                f"({total} cells of 40x40x3)"
            )
        out = torch.empty(
            (total, self.out_dim), dtype=torch.float16, device=pixels.device
        )
        pixels = pixels.contiguous()
        start = 0
        for chunk in split_chunks(grids):
            cells = sum(r * c for r, c in chunk)
            self._encode(
                pixels[start : start + cells], chunk, out[start : start + cells]
            )
            start += cells
        return out

    def _encode(
        self, cells: torch.Tensor, grids: list[tuple[int, int]], out: torch.Tensor
    ) -> None:
        dev = cells.device
        plan = build_plan(grids)
        meta = torch.from_numpy(plan.meta).to(dev)
        blk = torch.from_numpy(plan.blk_prefix).to(dev)
        T, P, D = plan.num_tokens, plan.num_patches, self.dim
        f16, f32 = torch.float16, torch.float32
        bb = self.backbone

        patches = torch.empty((P, PATCH_K_PAD), dtype=f16, device=dev)
        ops.inkling2_vit_gather_patches(cells, patches, PIXEL_NORMS[self.pixel_norm])
        emb = torch.empty((P, D), dtype=f32, device=dev)
        ops.inkling2_vit_gemm(
            patches, bb.embeddings.patch_embedding.weight, emb, None, 4
        )
        del patches
        h = torch.empty((T, D), dtype=f32, device=dev)
        ops.inkling2_vit_embed(
            emb,
            bb.embeddings.position_embedding,
            self.cls,
            bb.ln_pre.weight,
            bb.ln_pre.bias,
            meta,
            h,
            self.eps,
        )
        del emb
        cs = torch.empty((T, 32, 2), dtype=f32, device=dev)
        ops.inkling2_vit_rope_tables(meta, self.inv_freq, cs)

        # Lifetimes within a ViT block are disjoint: a -> qkv -> o -> h, then
        # a -> f -> h. Sharing (a, o) and (qkv, f) cuts ViT scratch by 45%.
        a = torch.empty((T, D), dtype=f16, device=dev)
        o = a
        scratch = torch.empty(T * max(self.mlp_dim, 3 * D), dtype=f16, device=dev)
        qkv = scratch[: T * 3 * D].view(T, 3 * D)
        f = scratch[: T * self.mlp_dim].view(T, self.mlp_dim)
        nq = int(plan.blk_prefix[-1])
        for layer in bb.layers:
            ops.inkling2_vit_layernorm(
                h, layer.ln_1.weight, layer.ln_1.bias, a, self.eps
            )
            attn = layer.self_attn
            ops.inkling2_vit_gemm(a, attn.qkv_proj.weight, qkv, attn.qkv_proj.bias, 1)
            ops.inkling2_vit_rope_qk(qkv, cs)
            ops.inkling2_vit_attention(qkv, o, meta, blk, nq)
            ops.inkling2_vit_gemm_residual(
                o, attn.out_proj.weight, attn.out_proj.bias, layer.ls_1.gamma, h
            )
            ops.inkling2_vit_layernorm(
                h, layer.ln_2.weight, layer.ln_2.bias, a, self.eps
            )
            ops.inkling2_vit_gemm(a, layer.mlp.c_fc.weight, f, layer.mlp.c_fc.bias, 2)
            ops.inkling2_vit_gemm_residual(
                f, layer.mlp.c_proj.weight, layer.mlp.c_proj.bias, layer.ls_2.gamma, h
            )
        del a, o, qkv, f, scratch, cs

        pooled = torch.empty((P, D), dtype=f16, device=dev)
        ops.inkling2_vit_pool(h, meta, pooled)
        del h
        merged = pooled.view(plan.num_cells, 4 * D)
        z = torch.empty((plan.num_cells, self.proj_dim), dtype=f16, device=dev)
        ops.inkling2_vit_gemm(merged, self.proj_in.weight, z, None, 3)
        ops.inkling2_vit_gemm(z, self.proj_out.weight, out, None, 0)


__all__ = [
    "DEFAULT_PIXEL_NORM",
    "MAX_CHUNK_TOKENS",
    "PATCH_K",
    "PATCH_K_PAD",
    "PIXEL_NORMS",
    "Inkling2VisionEncoder",
    "VisionPlan",
    "build_plan",
    "parse_grids",
    "split_chunks",
]
