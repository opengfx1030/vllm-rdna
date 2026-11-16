# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-agnostic numerical GGUF decode, not an optimized serving kernel.

Selected rows are selected in packed storage before arithmetic. ``row_width``
is the logical number of values per model row; it defaults to one quant block.
The budget bounds float32 decoded output, not the caller's source buffer or
Torch's temporary arithmetic storage. Callers must supply bounded payloads.
"""

from __future__ import annotations

import torch

_SPECS = {
    "Q5_1": (32, 24),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q8_0": (32, 34),
    "IQ4_NL": (32, 18),
    "F32": (1, 4),
    "BF16": (1, 2),
}
_IQ4 = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _bytes(data):
    if isinstance(data, torch.Tensor):
        if data.dtype != torch.uint8:
            raise TypeError("packed tensor input must have dtype uint8")
        return data.reshape(-1)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("packed input must be a byte buffer or uint8 tensor")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def _f16(x):
    return x.contiguous().view(torch.float16).to(torch.float32)


def _scale_min(s):
    a = s.reshape(-1, 12).to(torch.int64)
    return torch.cat(
        (a[:, :4] & 63, (a[:, 8:12] & 15) | ((a[:, :4] >> 2) & 48)), 1
    ).float(), torch.cat(
        (a[:, 4:8] & 63, (a[:, 8:12] >> 4) | ((a[:, 4:8] >> 2) & 48)), 1
    ).float()


def dequantize(
    data,
    format: str,
    *,
    rows=None,
    row_width: int | None = None,
    budget_bytes: int | None = None,
) -> torch.Tensor:
    if format not in _SPECS:
        raise ValueError(f"unsupported format: {format}")
    if budget_bytes is not None and (
        not isinstance(budget_bytes, int)
        or isinstance(budget_bytes, bool)
        or budget_bytes < 0
    ):
        raise ValueError("budget_bytes must be a nonnegative integer")
    width, size = _SPECS[format]
    raw = _bytes(data)
    if raw.numel() % size:
        raise ValueError("payload size is not block aligned")
    nb = raw.numel() // size
    row_width = width if row_width is None else row_width
    if (
        not isinstance(row_width, int)
        or isinstance(row_width, bool)
        or row_width <= 0
        or row_width % width
    ):
        raise ValueError("row_width must be a positive block-aligned integer")
    blocks_per_row = row_width // width
    if nb % blocks_per_row:
        raise ValueError("payload does not contain complete model rows")
    nr = nb // blocks_per_row
    if rows is not None:
        idx = torch.as_tensor(rows, device=raw.device)
        if idx.ndim != 1 or (
            idx.numel()
            and idx.dtype
            not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            raise ValueError("row ids must be one-dimensional integers")
        idx = idx.to(torch.long)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= nr):
            raise ValueError("row index out of range")
        need = idx.numel() * row_width * 4
        if budget_bytes is None or budget_bytes < 0 or need > budget_bytes:
            raise ValueError("selected rows exceed per-call byte budget")
        raw = raw.reshape(nr, blocks_per_row * size).index_select(0, idx).reshape(-1)
        nb = idx.numel() * blocks_per_row
    elif budget_bytes is not None and (
        budget_bytes < 0 or nb * width * 4 > budget_bytes
    ):
        raise ValueError("decoded payload exceeds per-call byte budget")
    if format in ("F32", "BF16"):
        x = (
            raw.contiguous()
            .view(torch.float32 if format == "F32" else torch.bfloat16)
            .float()
            .reshape(-1, 1)
        )
    else:
        b = raw.reshape(nb, size)
        if format == "Q5_1":
            d = _f16(b[:, :2])
            m = _f16(b[:, 2:4])
            a = b[:, 8:].long()
            lo = torch.cat((a & 15, a >> 4), 1)
            hi = (
                (b[:, 4:8].long()[:, :, None] >> torch.arange(8, device=b.device)) & 1
            ).reshape(nb, 32)
            x = d * (lo | (hi << 4)).float() + m
        elif format in ("Q4_K", "Q5_K"):
            d = _f16(b[:, :2])
            dm = _f16(b[:, 2:4])
            sc, mn = _scale_min(b[:, 4:16])
            qraw = b[:, 16:] if format == "Q4_K" else b[:, 48:]
            qraw = qraw.reshape(nb, 4, 32)
            lo = torch.stack((qraw & 15, qraw >> 4), 2).reshape(nb, 8, 32).long()
            if format == "Q5_K":
                hi = (
                    (
                        b[:, 16:48].long()[:, None, :]
                        >> torch.arange(8, device=b.device).reshape(1, 8, 1)
                    )
                    & 1
                ).reshape(nb, 256)
                q = (lo.reshape(nb, 256) | (hi << 4)).reshape(nb, 8, 32).float()
            else:
                q = lo.float()
            x = ((d * sc).unsqueeze(-1) * q - (dm * mn).unsqueeze(-1)).reshape(nb, 256)
        elif format == "Q8_0":
            q = b[:, 2:].long()
            q = torch.where(q >= 128, q - 256, q).float()
            x = _f16(b[:, :2]) * q
        else:
            a = b[:, 2:].long()
            idx = torch.cat((a & 15, a >> 4), 1).reshape(nb, 32)
            table = torch.tensor(_IQ4, dtype=torch.float32, device=b.device)
            x = _f16(b[:, :2]) * table[idx]
        x = x.reshape(nb, width)
    return x.reshape(-1).float()
