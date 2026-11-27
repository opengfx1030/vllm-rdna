# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``int4_fp16_block`` KV cache layout, in one place.

A packed INT4 record with FP16 per-32-channel block scales::

    record_bytes = head_size // 2 + 2 * (head_size // 32)
    record = [ head_size // 2 payload bytes , head_size // 32 uint16 words ]

* payload: two INT4 codes per byte, low nibble first.
* scale word: ``(fp16_bits(scale) & 0xfff0) | zp`` -- the high 12 bits are an
  FP16 scale whose low 4 mantissa bits are cleared, the low 4 bits are a
  4-bit zero-point.
* one scale covers 32 contiguous head-dimension channels.
* dequant: ``value = (code - zp) * scale``.

The scale words are **not** a separate allocation: they are a window onto the
tail of the same physical records, so they carry their own strides.

Every consumer of this format must identify the layout positively (by
``kind``, never by guessing from a dtype). A kernel that reads these bytes as
fp16 produces finite, plausible, silently wrong numbers rather than a fault,
so the helpers here exist to keep one definition of the layout shared by the
RDNA_ATTN backend, the QSA decode kernels and the MonoKernel.
"""

from __future__ import annotations

from typing import Literal

import torch

INT4_FP16_BLOCK: Literal["int4_fp16_block"] = "int4_fp16_block"

#: Head-dimension channels that share one scale.
SCALE_GROUP_CHANNELS = 32


def int4_fp16_record_bytes(head_size: int) -> int:
    """Bytes one packed INT4 record occupies for ``head_size`` channels."""
    return head_size // 2 + 2 * (head_size // SCALE_GROUP_CHANNELS)


def int4_fp16_cache_view(
    kv_cache: torch.Tensor, num_kv_heads: int, head_size: int
) -> torch.Tensor:
    """Normalize a packed cache to ``[2, blocks, slots, heads, record_bytes]``.

    The framework hands every backend the same logical per-layer view,
    ``[blocks, heads, slots, content]``. A backend that stores a packed INT4
    cache publishes ``num_head_slots=2``, so the head dimension holds the K
    and V slots and the content dimension holds one side's records:
    ``[blocks, 2, slots, heads*record_bytes]``. The writer wants the two
    sides first, hence the transpose.

    The convention is asserted rather than assumed: a cache that does not
    have this shape is rejected instead of being read as the wrong format.
    """
    record_bytes = int4_fp16_record_bytes(head_size)
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != 2
        or kv_cache.shape[-1] != num_kv_heads * record_bytes
        or kv_cache.stride(-1) != 1
    ):
        raise ValueError(
            "invalid int4_fp16_block cache geometry: "
            f"dtype={kv_cache.dtype} shape={tuple(kv_cache.shape)} "
            f"stride={kv_cache.stride()} num_kv_heads={num_kv_heads} "
            f"record_bytes={record_bytes}"
        )
    view = kv_cache.unflatten(-1, (num_kv_heads, record_bytes)).transpose(0, 1)
    s0, s1, s2, s3, s4 = view.stride()
    page_span = view.shape[2] * s2
    side_outer = s1 >= page_span and s0 >= view.shape[1] * s1
    page_outer = s0 >= page_span and s1 >= 2 * s0
    slot_outer = s0 >= num_kv_heads * s3 and s2 >= 2 * s0 and s1 >= page_span
    if not (
        s4 == 1
        and s3 >= record_bytes
        and s2 >= num_kv_heads * s3
        and (side_outer or page_outer or slot_outer)
    ):
        raise ValueError(
            "invalid int4_fp16_block cache strides: "
            f"shape={tuple(view.shape)} stride={view.stride()} "
            f"record_bytes={record_bytes} page_span={page_span} "
            f"side_outer={side_outer} page_outer={page_outer} slot_outer={slot_outer}"
        )
    return view


def int4_fp16_read_views(
    cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``[pages, heads, slots, payload/scale]`` views, keeping strides.

    ``cache`` is ``[2, pages, slots, heads, record_bytes]``. The scale view
    is a window onto the tail of the same records, so it must not be treated
    as an independent allocation.
    """
    data_bytes = head_size // 2
    sides = cache.unbind(0)
    payloads = tuple(side[..., :data_bytes].permute(0, 2, 1, 3) for side in sides)
    scales = tuple(
        side[..., data_bytes:].view(torch.uint16).permute(0, 2, 1, 3) for side in sides
    )
    return payloads[0], payloads[1], scales[0], scales[1]
