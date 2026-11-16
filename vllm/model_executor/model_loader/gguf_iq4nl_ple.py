# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read selected PLE rows directly from a validated GGUF IQ4_NL payload.

This table implements the CPU PLE sidecar's gather protocol without converting
or rewriting the table. The caller must obtain offset/shape/type from a fully
validated shard manifest; this class also checks the selected byte range.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


class GGUFIQ4NLPLETable:
    """Read-only mmap with bounded, vectorized CPU row gathers.

    IQ4_NL blocks are 32 values in 18 bytes: fp16 scale and 16 bytes
    containing low-16 then high-16 nibbles (not interleaved nibbles).
    No mmap-backed arrays escape this object's lifetime.
    """

    layout = "gguf_iq4_nl_int4"
    is_fp8 = False

    def __init__(
        self,
        path: str | Path,
        *,
        data_offset: int,
        rows: int,
        width: int,
        max_gather_bytes: int = 64 << 20,
    ) -> None:
        for name, value in (
            ("data_offset", data_offset),
            ("rows", rows),
            ("width", width),
            ("max_gather_bytes", max_gather_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if rows <= 0 or width <= 0 or width % 32:
            raise ValueError(
                "IQ4_NL table requires positive rows and width divisible by 32"
            )
        if data_offset < 0 or max_gather_bytes <= 0:
            raise ValueError("offset and gather budget are invalid")
        self.path = Path(path)
        self.rows = rows
        self.width = width
        self.row_bytes = width // 32 * 18
        self.packed_bytes = rows * self.row_bytes
        self.max_gather_bytes = max_gather_bytes
        self._file = self.path.open("rb")
        try:
            if data_offset + self.packed_bytes > os.fstat(self._file.fileno()).st_size:
                raise ValueError("IQ4_NL payload extends past end of file")
            self._mapping = np.memmap(
                self._file,
                mode="r",
                dtype=np.uint8,
                offset=data_offset,
                shape=(rows, self.row_bytes),
            )
        except BaseException:
            self._file.close()
            raise
        self._closed = False

    def gather_rows_small(self, ids: np.ndarray, out: np.ndarray) -> None:
        """Decode only requested rows into caller-owned float32 storage."""
        if self._closed:
            raise ValueError("IQ4_NL table is closed")
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("row ids must be a one-dimensional integer array")
        if ids.size and (np.any(ids < 0) or np.any(ids >= self.rows)):
            raise ValueError("row id outside table")
        if out.shape != (ids.size, self.width) or out.dtype != np.float32:
            raise ValueError("output must have requested-row shape and float32 dtype")
        # Admission bounds output payload. Temporary arrays are proportional to
        # these admitted rows, never the full table.
        if ids.size * self.width * 4 > self.max_gather_bytes:
            raise ValueError("IQ4_NL gather exceeds decoded-byte budget")
        if not ids.size:
            return
        packed = self._mapping[ids].reshape(-1, 18)
        scales = packed[:, :2].copy().view("<f2").astype(np.float32)
        codes = np.concatenate((packed[:, 2:] & 15, packed[:, 2:] >> 4), axis=1)
        values = _IQ4_NL_VALUES[codes] * scales
        out[:] = values.reshape(ids.size, self.width)

    def gather_into(self, ids, out) -> None:
        """CPU Torch sidecar protocol; imports Torch only when it is used."""
        import torch

        if ids.device.type != "cpu" or out.device.type != "cpu":
            raise ValueError("IQ4_NL PLE gathers require CPU tensors")
        if ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("row ids must be integer tensors")
        if ids.ndim != 1 or tuple(out.shape) != (ids.numel(), self.width):
            raise ValueError("output shape must match one-dimensional row ids")
        if out.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("output must have a floating-point activation dtype")
        if ids.numel() * self.width * 4 > self.max_gather_bytes:
            raise ValueError("IQ4_NL gather exceeds decoded-byte budget")
        values = np.empty((ids.numel(), self.width), dtype=np.float32)
        self.gather_rows_small(ids.numpy(), values)
        out.copy_(torch.from_numpy(values))

    def close(self) -> None:
        """Release only this table's mappings, including on repeated closes."""
        if not self._closed:
            self._mapping._mmap.close()
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
