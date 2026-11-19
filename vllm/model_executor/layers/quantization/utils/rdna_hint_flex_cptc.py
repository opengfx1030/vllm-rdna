# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPTC v1 codebooks and block-local reference decoding for HI2 v3.3.

The packed stream uses Algorithms 1/2 of arXiv:2611.22019. The serialized
table scale is canonicalized to fp16 before table entries are rounded to fp16;
reconstruction multiplies that value by the block's fp16 scale and rounds once.
"""

import binascii
import math
from dataclasses import dataclass
from typing import Any

import pybase64 as base64
import torch

NBITS = (2, 4)
FIELDS = (
    "codes",
    "code_off",
    "scale",
    "param_off",
    "row_nbits",
    "row_block",
    "had",
)


@dataclass(frozen=True)
class CptcCodebooks:
    depth: int
    coefficients: torch.Tensor  # int32 [2, 2], rows for 2/4 bits
    tables: torch.Tensor  # fp16 [2, 127], mirror expanded at load

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CptcCodebooks":
        if not isinstance(config, dict):
            raise ValueError("rdna_hint_flex CPTC requires explicit codebook metadata")
        expected = {
            "name": "cptc",
            "revision": 1,
            "tap_q": 6,
            "layout": "mirror_int8",
            "shift": 0,
            "block": "row_block",
        }
        if set(config) != set(expected) | {"depth", "tables"}:
            raise ValueError("rdna_hint_flex CPTC: invalid config keys")
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    f"rdna_hint_flex CPTC: unsupported {key}={config.get(key)!r}"
                )
        depth = config.get("depth")
        if type(depth) is not int or not 8 <= depth <= 16:
            raise ValueError(
                "rdna_hint_flex CPTC: shared 2/4-bit depth must be in [8, 16]"
            )
        raw_tables = config.get("tables")
        if not isinstance(raw_tables, dict) or set(raw_tables) != {"2", "4"}:
            raise ValueError(
                "rdna_hint_flex CPTC: tables for exactly 2 and 4 bits required"
            )
        coefficients, tables = [], []
        states = torch.arange(1 << depth, dtype=torch.int64)
        for nbits in NBITS:
            entry = raw_tables[str(nbits)]
            if not isinstance(entry, dict) or set(entry) != {
                "a0",
                "a1",
                "table_scale",
                "tbl_b64",
            }:
                raise ValueError("rdna_hint_flex CPTC: invalid table metadata keys")
            pair = [entry["a0"], entry["a1"]]
            if any(type(a) is not int or not 0 <= a <= 65535 for a in pair):
                raise ValueError(
                    "rdna_hint_flex CPTC: coefficients must be uint16 integers"
                )
            combined = pair[0] | pair[1]
            if combined == 0 or (combined & -combined).bit_length() - 1 != 16 - depth:
                raise ValueError("rdna_hint_flex CPTC: coefficients do not match depth")
            table_scale = entry.get("table_scale")
            if (
                isinstance(table_scale, bool)
                or not isinstance(table_scale, (int, float))
                or not math.isfinite(table_scale)
                or table_scale <= 0
            ):
                raise ValueError(
                    "rdna_hint_flex CPTC: table_scale must be finite and positive"
                )
            table_scale = torch.tensor(table_scale, dtype=torch.float16).item()
            if not math.isfinite(table_scale) or table_scale <= 0:
                raise ValueError(
                    "rdna_hint_flex CPTC: table_scale must be positive finite fp16"
                )
            try:
                raw = base64.b64decode(entry["tbl_b64"], validate=True)
            except (KeyError, TypeError, ValueError, binascii.Error) as error:
                raise ValueError("rdna_hint_flex CPTC: invalid base64 table") from error
            if len(raw) != 64:
                raise ValueError(
                    "rdna_hint_flex CPTC: mirrored table must contain 64 int8 entries"
                )
            if 128 in raw:
                raise ValueError(
                    "rdna_hint_flex CPTC: mirrored int8 table cannot contain -128"
                )
            half = torch.tensor(
                [v if v < 128 else v - 256 for v in raw], dtype=torch.float64
            )
            half = (half * table_scale).to(torch.float16)
            if not bool(torch.isfinite(half).all()) or half[-1] != 0:
                raise ValueError(
                    "rdna_hint_flex CPTC: table must be finite with a zero center"
                )
            indices = (((states * pair[0]) & 65535) >> 10) + (
                ((states * pair[1]) & 65535) >> 10
            )
            histogram = torch.bincount(indices, minlength=127)
            if not torch.equal(histogram, histogram.flip(0)):
                raise ValueError(
                    "rdna_hint_flex CPTC: index histogram is not mirror symmetric"
                )
            coefficients.append(pair)
            tables.append(torch.cat((half, -half[:-1].flip(0))))
        return cls(
            depth, torch.tensor(coefficients, dtype=torch.int32), torch.stack(tables)
        )

    def decode(
        self, codes: torch.Tensor, scale: torch.Tensor, nbits: int
    ) -> torch.Tensor:
        """Unpacked K-bit words ``[..., B]`` -> block-scaled fp16 weights."""
        if nbits not in NBITS:
            raise ValueError("rdna_hint_flex CPTC supports only 2/4-bit streams")
        slot = NBITS.index(nbits)
        state = torch.zeros_like(codes, dtype=torch.int64)
        words = codes.to(torch.int64)
        for j in range(16 // nbits):
            state |= torch.roll(words, -j, dims=-1) << (j * nbits)
        a0, a1 = self.coefficients[slot].tolist()
        index = (((state * a0) & 65535) >> 10) + (((state * a1) & 65535) >> 10)
        table = self.tables[slot].to(codes.device)
        return (table[index].double() * scale.double().unsqueeze(-1)).to(torch.float16)
