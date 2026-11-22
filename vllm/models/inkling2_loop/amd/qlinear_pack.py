# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline quantizer and tile packer of the weight-only quantized decode linears.

``inkling2_qlinear_*`` (``csrc/rocm/inkling2_qlinear_rdna2.cu``) read int8 / int4
weights with one fp16 scale per group of K from a tile layout built here once, at
weight load, from the fp16 weights of the checkpoint.  Symmetric quantization:
``w ~ q * scale`` with ``q`` in ``[-127, 127]`` (int8) or ``[-8, 7]`` (int4) and
``scale = max|w| / qmax`` of the group (int4: optionally the clipping that
minimises the squared error of the group).

Layout (``R`` rows per tile, ``KQ = 32 / R`` K quarters, a chunk is ``CK = 16``
int8 / ``32`` int4 consecutive K elements = 16 bytes, ``nc = K / (CK * KQ)`` chunks
per lane, ``cpi`` chunks per kernel iteration, ``SV = cpi / (group / CK)`` scales
per iteration):

* weights ``uint8 [tiles, nc, 32, 16]``: lane ``l = kq * R + r`` of chunk ``c`` of
  tile ``t`` holds the chunk ``kq * nc + c`` of row ``t * R + r`` (offset binary:
  ``u = q + 2^(bits-1)``); int4 words hold the elements of an 8-element block in
  the nibble slots ``(0, 4, 1, 5, 2, 6, 3, 7)``;
* scales ``fp16 [tiles, niter, 32, SV]``: group ``it * SV + i`` of the lane's quarter.

Every segment of a pack is padded to whole tiles (zero rows with scale 0).

The quantized packs serve 1 to ``MAX_ROWS`` token rows (decode, DSpark verification);
the fp16 packs stay for larger chunks and the prefill (:class:`HybridLinearPack`,
:class:`HybridLMHead`).  Off by default, per lever:

* ``VLLM_INKLING2_LM_HEAD_QUANT``  the lm_head,
* ``VLLM_INKLING2_ATTN_IN_QUANT``  the attention input pack,
* ``VLLM_INKLING2_O_PROJ_QUANT``   o_proj,
* ``VLLM_INKLING2_ATTN_QUANT``     the default of the two attention levers,

each ``w8`` (int8, group 64), ``w4`` (int4, group 32 with clipping) or an explicit
``w8g64`` / ``w4g32`` / ``w4g64`` (a trailing ``n`` drops the int4 clipping).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import torch

from vllm.logger import init_logger

from .linear_pack import LinearPack, LMHeadPack

logger = init_logger(__name__)

CK = {4: 32, 8: 16}  # K elements per 16-byte chunk
NIBBLE_SLOT = (0, 4, 1, 5, 2, 6, 3, 7)  # nibble slot of element p of an 8-block
ACT_NONE = 0
MAX_ROWS = 4  # the kernels take 1..4 activation rows

# the (bits, group) pairs of the serving build of the kernels
SERVING_FORMATS = {(8, 64), (4, 32), (4, 64)}
# (rows per tile, chunks per iteration, waves per block) of the measured fastest
# instances of the serving build (docs/rdna2/inkling2-qlinear-kernels.md)
ATTN_IN_CONFIG = {8: (8, 8, 8), 4: (8, 4, 8)}
O_PROJ_CONFIG = {8: (8, 8, 8), 4: (4, 4, 8)}
LM_HEAD_CPI = 8


def quant_flag(name: str) -> str:
    """The format spec of the lever ``name`` (``""``: off)."""
    spec = os.environ.get(f"VLLM_INKLING2_{name}_QUANT", "").strip().lower()
    return "" if spec in ("0", "off", "none", "false") else spec


def parse_format(spec: str) -> tuple[int, int, bool]:
    """``w8`` -> (8, 64, False); ``w4`` -> (4, 32, True); ``w8g64``; ``w4g64``;
    a trailing ``n`` (``w4g32n``) disables the clipping search of int4."""
    s = spec.lower()
    if not s or s[0] != "w":
        raise ValueError(f"bad quantization format {spec!r}")
    clip = not s.endswith("n")
    s = s.rstrip("n")
    bits_s, _, group_s = s[1:].partition("g")
    bits = int(bits_s)
    group = int(group_s) if group_s else (64 if bits == 8 else 32)
    if (bits, group) not in SERVING_FORMATS:
        raise ValueError(
            f"unsupported quantization format {spec!r}: w8g64, w4g32, w4g64"
        )
    return bits, group, clip and bits == 4


@torch.no_grad()
def quantize_rows(
    w: torch.Tensor, bits: int, group: int, clip: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """``w [N, K]`` -> (``q`` int8 ``[N, K]``, fp16 scales ``[N, K // group]``).

    The scale is rounded to fp16 *before* the division, so the dequantized weights
    of the kernels are exactly ``q * scale``."""
    N, K = w.shape
    if K % group:
        raise ValueError(f"K = {K} is not a multiple of the group {group}")
    wg = w.float().view(N, K // group, group)
    amax = wg.abs().amax(-1, keepdim=True)
    qmax = 2 ** (bits - 1) - 1
    cands = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65) if clip else (1.0,)
    best_s = best_e = None
    for c in cands:
        s = (amax * c / qmax).clamp_min(2.0**-14).half().float()
        if len(cands) == 1:
            best_s = s
            break
        q = torch.round(wg / s).clamp(-qmax - 1, qmax)
        e = ((q * s - wg) ** 2).sum(-1, keepdim=True)
        if best_s is None:
            best_s, best_e = s, e
        else:
            take = e < best_e
            best_s = torch.where(take, s, best_s)
            best_e = torch.where(take, e, best_e)
    q = torch.round(wg / best_s).clamp(-qmax - 1, qmax)
    return q.view(N, K).to(torch.int8), best_s.view(N, K // group).half()


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """fp32 ``q * scale`` (the exact weights the kernels multiply with)."""
    N, K = q.shape
    return (q.float().view(N, K // group, group) * scale.float().unsqueeze(-1)).view(
        N, K
    )


def _pack_chunk(
    q: torch.Tensor, scale: torch.Tensor, bits: int, group: int, R: int, cpi: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``N`` rows (``N % R == 0``) into whole tiles."""
    N, K = q.shape
    ck, kq = CK[bits], 32 // R
    nc = K // (ck * kq)
    gc = group // ck
    sv = cpi // gc
    niter = -(-nc // cpi)
    ntiles = N // R
    zp = 1 << (bits - 1)
    u = (q.to(torch.int16) + zp).to(torch.uint8)
    # [tiles, R, KQ, nc, CK] -> lane order (kq, r): [tiles, nc, KQ, R, CK]
    u = u.view(ntiles, R, kq, nc, ck).permute(0, 3, 2, 1, 4)
    if bits == 8:
        w = u.reshape(ntiles, nc, 32, 16).contiguous()
    else:
        u = u.reshape(ntiles, nc, 32, 4, 8).to(torch.int32)
        word = torch.zeros(u.shape[:-1], dtype=torch.int32, device=u.device)
        for p, slot in enumerate(NIBBLE_SLOT):
            word |= u[..., p] << (4 * slot)
        w = word.contiguous().view(torch.uint8).view(ntiles, nc, 32, 16)
    ng = nc // gc
    s = scale.view(ntiles, R, kq, ng).permute(0, 2, 1, 3).reshape(ntiles, 32, ng)
    if niter * sv != ng:
        s = torch.nn.functional.pad(s, (0, niter * sv - ng))
    s = s.reshape(ntiles, 32, niter, sv).permute(0, 2, 1, 3).contiguous()
    return w, s


def pack_quantized(
    q: torch.Tensor, scale: torch.Tensor, bits: int, group: int, R: int, cpi: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile-pack integers ``q [N, K]`` and fp16 scales ``[N, K // group]`` (rows padded
    with zeros to a multiple of ``R``)."""
    pad = (-q.shape[0]) % R
    if pad:
        q = torch.cat([q, q.new_zeros(pad, q.shape[1])], 0)
        scale = torch.cat([scale, scale.new_zeros(pad, scale.shape[1])], 0)
    return _pack_chunk(q, scale, bits, group, R, cpi)


@torch.no_grad()
def pack_weight(
    weight: torch.Tensor,
    bits: int,
    group: int,
    R: int,
    cpi: int,
    clip: bool = False,
    rows_chunk: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and tile-pack one ``[N, K]`` weight (any float dtype, on the device
    it lives on); the rows are padded to a multiple of ``R``."""
    N, K = weight.shape
    ck, kq = CK[bits], 32 // R
    if K % (ck * kq):
        raise ValueError(f"K = {K} must be a multiple of {ck * kq}")
    gc = group // ck
    nc = K // (ck * kq)
    if nc % gc or cpi % gc:
        raise ValueError(f"group {group}: chunks per lane {nc}, cpi {cpi}")
    ws, ss = [], []
    step = max(R, rows_chunk - rows_chunk % R)
    for lo in range(0, N, step):
        blk = weight[lo : lo + step]
        q, s = quantize_rows(blk, bits, group, clip)
        w, sc = pack_quantized(q, s, bits, group, R, cpi)
        ws.append(w)
        ss.append(sc)
    return torch.cat(ws, 0), torch.cat(ss, 0)


@dataclasses.dataclass(frozen=True)
class QLinearPack:
    """A packed quantized weight and the segments it serves (like ``LinearPack``)."""

    weight: torch.Tensor  # uint8 [tiles, nc, 32, 16]
    scale: torch.Tensor  # fp16 [tiles, niter, 32, SV]
    seg_n: tuple[int, ...]
    seg_x: tuple[int, ...]
    seg_act: tuple[int, ...]
    seg_p: tuple[float, ...]
    seg_valid: tuple[int, ...]
    bits: int
    group: int
    rows_per_tile: int
    cpi: int
    waves: int

    @property
    def nbytes(self) -> int:
        return self.weight.numel() + 2 * self.scale.numel()

    def __call__(
        self, xs: Sequence[torch.Tensor], outs: Sequence[torch.Tensor]
    ) -> None:
        from vllm import _custom_ops as ops

        ops.inkling2_qlinear_grouped(
            list(xs),
            self.weight,
            self.scale,
            list(outs),
            list(self.seg_x),
            list(self.seg_n),
            list(self.seg_act),
            list(self.seg_p),
            list(self.seg_valid),
            self.bits,
            self.group,
            self.rows_per_tile,
            self.cpi,
            self.waves,
        )


_QUANT_FILE: tuple[str, dict] | None = None


def calibrated_weights(bits: int, group: int) -> dict | None:
    """The integers and scales of ``VLLM_INKLING2_QUANT_FILE`` (written by
    ``tools/rdna/inkling2/gptq_quant.py``): ``{"<layer>.<segment>": {"q", "s"}}``, or
    None when the variable is not set. A file of another format than the lever asks
    for is an error, not a silent fallback."""
    global _QUANT_FILE
    path = os.environ.get("VLLM_INKLING2_QUANT_FILE", "").strip()
    if not path:
        return None
    if _QUANT_FILE is None or _QUANT_FILE[0] != path:
        data = torch.load(path, map_location="cpu", weights_only=True)
        logger.info(
            "Inkling-2 qlinear: %d calibrated w%dg%d weights from %s",
            len(data["weights"]),
            data["bits"],
            data["group"],
            path,
        )
        _QUANT_FILE = (path, data)
    data = _QUANT_FILE[1]
    if (data["bits"], data["group"]) != (bits, group):
        raise ValueError(
            f"VLLM_INKLING2_QUANT_FILE holds w{data['bits']}g{data['group']} weights, "
            f"the lever asks for w{bits}g{group}"
        )
    return data["weights"]


def quantize_linear_pack(
    pack: LinearPack,
    fmt: str,
    config: tuple[int, int, int],
    layer: int | None = None,
) -> QLinearPack:
    """The quantized twin of an fp16 :class:`LinearPack` (every segment quantized on its
    own and padded to whole tiles; same segments, inputs and epilogues). Segments the
    calibrated file has for ``layer`` take its integers, the others round to nearest."""
    if pack.weight.dtype != torch.float16:
        raise ValueError(f"only fp16 packs are quantized, got {pack.weight.dtype}")
    bits, group, clip = parse_format(fmt)
    R, cpi, waves = config
    calibrated = calibrated_weights(bits, group) if layer is not None else None
    ws, ss = [], []
    lo = 0
    for name, n in zip(pack.names, pack.seg_n):
        got = calibrated.get(f"{layer}.{name}") if calibrated else None
        if got is not None:
            q = got["q"].to(pack.weight.device)
            s = got["s"].to(pack.weight.device)
            if q.shape != (n, pack.weight.shape[1]):
                raise ValueError(f"layer {layer} {name}: calibrated {tuple(q.shape)}")
            a, b = pack_quantized(q, s, bits, group, R, cpi)
        else:
            a, b = pack_weight(pack.weight[lo : lo + n], bits, group, R, cpi, clip)
        ws.append(a)
        ss.append(b)
        lo += n
    return QLinearPack(
        weight=torch.cat(ws, 0).contiguous(),
        scale=torch.cat(ss, 0).contiguous(),
        seg_n=pack.seg_n,
        seg_x=pack.seg_x,
        seg_act=pack.seg_act,
        seg_p=pack.seg_p,
        seg_valid=pack.seg_valid,
        bits=bits,
        group=group,
        rows_per_tile=R,
        cpi=cpi,
        waves=waves,
    )


class HybridLinearPack:
    """The quantized pack for up to ``MAX_ROWS`` token rows, the fp16 pack (which stays
    the owner of the segment names, sizes and outputs) for larger chunks and prefill."""

    def __init__(self, fp16: LinearPack, quantized: QLinearPack) -> None:
        self.fp16 = fp16
        self.quantized = quantized

    def __getattr__(self, name: str):
        return getattr(self.fp16, name)

    def __call__(
        self,
        xs: Sequence[torch.Tensor],
        outs: Sequence[torch.Tensor],
        variant: int = -1,
    ) -> None:
        if xs[0].shape[0] <= MAX_ROWS:
            self.quantized(xs, outs)
        else:
            self.fp16(xs, outs, variant)


@dataclasses.dataclass(frozen=True)
class QLMHeadPack:
    """The quantized ``lm_head`` rows of one rank (``rows`` real rows, the padding
    of the vocabulary shard masked to ``-inf`` by ``n_valid``)."""

    weight: torch.Tensor
    scale: torch.Tensor
    rows: int
    n_valid: int
    col_base: int
    bits: int
    group: int
    cpi: int

    @property
    def nbytes(self) -> int:
        return self.weight.numel() + 2 * self.scale.numel()

    def __call__(
        self,
        x: torch.Tensor,
        logits: torch.Tensor | None,
        part_val: torch.Tensor | None = None,
        part_idx: torch.Tensor | None = None,
        softcap: float = 0.0,
    ) -> None:
        from vllm import _custom_ops as ops

        ops.inkling2_qlinear_lm_head(
            x,
            self.weight,
            self.scale,
            logits,
            part_val,
            part_idx,
            softcap,
            self.rows,
            self.n_valid,
            self.col_base,
            self.bits,
            self.group,
            self.cpi,
        )


def quantize_lm_head(head: LMHeadPack, fmt: str) -> QLMHeadPack:
    bits, group, clip = parse_format(fmt)
    w, s = pack_weight(head.weight, bits, group, 32, LM_HEAD_CPI, clip)
    return QLMHeadPack(
        w.contiguous(),
        s.contiguous(),
        head.rows,
        head.n_valid,
        head.col_base,
        bits,
        group,
        LM_HEAD_CPI,
    )


class HybridLMHead:
    """The quantized lm_head for up to ``MAX_ROWS`` rows, the fp16 head otherwise."""

    def __init__(self, fp16: LMHeadPack, quantized: QLMHeadPack) -> None:
        self.fp16 = fp16
        self.quantized = quantized

    def __getattr__(self, name: str):
        return getattr(self.fp16, name)

    def __call__(
        self,
        x: torch.Tensor,
        logits: torch.Tensor | None,
        part_val: torch.Tensor | None = None,
        part_idx: torch.Tensor | None = None,
        softcap: float = 30.0,
        variant: int = -1,
    ) -> None:
        if x.shape[0] <= MAX_ROWS:
            self.quantized(x, logits, part_val, part_idx, softcap)
        else:
            self.fp16(x, logits, part_val, part_idx, softcap, variant)


def _lever(name: str, fallback: str | None = None) -> str:
    return quant_flag(name) or (quant_flag(fallback) if fallback else "")


def maybe_quantize_attn_in(
    pack: LinearPack, layer: int | None = None
) -> LinearPack | HybridLinearPack:
    """The attention input pack of layer ``layer`` with the lever ``ATTN_IN`` applied
    (the fp16 pack itself when it is off)."""
    spec = _lever("ATTN_IN", "ATTN")
    if not spec:
        return pack
    bits = parse_format(spec)[0]
    return HybridLinearPack(
        pack, quantize_linear_pack(pack, spec, ATTN_IN_CONFIG[bits], layer)
    )


def maybe_quantize_o_proj(
    pack: LinearPack, layer: int | None = None
) -> LinearPack | HybridLinearPack:
    """o_proj of layer ``layer`` with the lever ``O_PROJ`` applied."""
    spec = _lever("O_PROJ", "ATTN")
    if not spec:
        return pack
    bits = parse_format(spec)[0]
    return HybridLinearPack(
        pack, quantize_linear_pack(pack, spec, O_PROJ_CONFIG[bits], layer)
    )


def maybe_quantize_lm_head(head: LMHeadPack) -> LMHeadPack | HybridLMHead:
    """The lm_head with the lever ``LM_HEAD`` applied."""
    spec = quant_flag("LM_HEAD")
    if not spec:
        return head
    q = quantize_lm_head(head, spec)
    logger.info(
        "Inkling-2 lm_head: %s quantized decode copy %.0f MB next to the fp16 head "
        "(%.0f MB) that serves more than %d rows",
        spec,
        q.nbytes / 2**20,
        head.weight.numel() * 2 / 2**20,
        MAX_ROWS,
    )
    return HybridLMHead(head, q)
