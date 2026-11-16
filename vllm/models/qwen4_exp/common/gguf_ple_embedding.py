# SPDX-License-Identifier: Apache-2.0
"""Packed GGUF IQ4_NL PLE storage for pinned-host/UVA embeddings."""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.triton_utils import tl, triton

from .ngram_embedding import Qwen4ExpPLEEmbeddingMethod


class Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Store exact IQ4_NL bytes and decode requested rows through UVA."""

    lookup_dequantizes = True
    BLOCK_VALUES = 32
    BLOCK_BYTES = 18
    LUT = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
    CHUNK_BYTES = 8 << 20

    def __init__(self, payload: dict):
        required = {"path", "data_offset", "rows", "width", "type"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid GGUF IQ4_NL PLE payload descriptor")
        for key in ("data_offset", "rows", "width", "type"):
            value = payload[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
        if (
            payload["type"] != 20
            or payload["data_offset"] < 0
            or payload["rows"] <= 0
            or payload["width"] <= 0
            or payload["width"] % 32
        ):
            raise ValueError("invalid IQ4_NL payload type, bounds, or dimensions")
        self.payload = dict(payload)
        self.rows, self.width = payload["rows"], payload["width"]
        self.row_bytes = self.width // 32 * 18
        self.payload_bytes = self.rows * self.row_bytes
        if not isinstance(payload["path"], (str, os.PathLike)):
            raise ValueError("payload path must be a filesystem path")
        self.path = os.fspath(payload["path"])
        if self.payload_bytes + payload["data_offset"] > os.path.getsize(self.path):
            raise ValueError("IQ4_NL payload extends past end of file")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size, params_dtype, extra_weight_attrs
        if input_size_per_partition != self.width:
            raise ValueError("GGUF IQ4_NL width does not match PLE embedding width")
        n = sum(output_partition_sizes)
        for name, rows, cols, dtype in (
            ("weight", n, self.row_bytes, torch.uint8),
            ("weight_scale", 1, 1, torch.float16),
        ):
            param = nn.Parameter(
                layer.allocate_embedding_weight(rows, cols, dtype), requires_grad=False
            )
            from vllm.model_executor.utils import set_weight_attrs

            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(name, param)
        layer.weight_scale.data.fill_(1)

    def process_weights_after_loading(self, layer):
        start, end = (
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
        )
        if (
            layer.org_vocab_size != self.rows
            or start < 0
            or end < start
            or end > self.rows
        ):
            raise ValueError("GGUF IQ4_NL rows do not match model vocabulary shard")
        count = end - start
        if layer.weight.shape[0] < count or layer.weight.shape[1] != self.row_bytes:
            raise ValueError("allocated packed PLE storage does not match owned rows")
        layer.weight.data.zero_()
        with open(self.path, "rb", buffering=0) as f:
            pos = self.payload["data_offset"] + start * self.row_bytes
            remaining = count * self.row_bytes
            dst = layer.weight.data
            while remaining:
                size = min(remaining, self.CHUNK_BYTES)
                f.seek(pos)
                data = f.read(size)
                if len(data) != size:
                    raise ValueError("truncated IQ4_NL payload while reading rank rows")
                flat_offset = count * self.row_bytes - remaining
                dst.view(torch.uint8).view(-1)[flat_offset : flat_offset + size].copy_(
                    torch.frombuffer(bytearray(data), dtype=torch.uint8)
                )
                pos += size
                remaining -= size

    def lookup_dtype(self, layer):
        return layer.params_dtype

    def embedding(self, layer, input_):
        return _dequantize_gguf_iq4nl_rows(
            layer.weight[input_], self.width, layer.params_dtype
        )

    def dequantize(self, layer, embeddings, output_dtype):
        del layer, output_dtype
        return embeddings

    def lookup_from_pinned(self, layer, ids, output):
        _lookup_gguf_iq4nl_from_pinned[(ids.numel(),)](
            layer._uva_weight,
            ids,
            output,
            self.width,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(self.width),
        )


def _dequantize_gguf_iq4nl_rows(packed, width, dtype):
    blocks = packed.reshape(*packed.shape[:-1], width // 32, 18)
    scale = blocks[..., :2].contiguous().view(torch.float16)[..., :1].to(torch.float32)
    codes = blocks[..., 2:]
    idx = torch.cat((codes & 15, codes >> 4), dim=-1)
    lut = torch.tensor(
        Qwen4ExpPLEGGUFIQ4NLEmbeddingMethod.LUT,
        device=packed.device,
        dtype=torch.float32,
    )
    return (lut[idx.long()] * scale).reshape(*packed.shape[:-1], width).to(dtype)


@triton.jit
def _lookup_gguf_iq4nl_from_pinned(
    weight,
    ids,
    output,
    WIDTH: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(ids + row)
    d = tl.arange(0, BLOCK_D)
    valid = (token >= START) & (token < END) & (d < WIDTH)
    local = tl.where((token >= START) & (token < END), token - START, 0).to(tl.int64)
    offset = local * (WIDTH // 32 * 18) + (d // 32) * 18
    scale_bits = tl.load(weight + offset.to(tl.int64), valid, other=0).to(tl.uint16) | (
        tl.load(weight + (offset + 1).to(tl.int64), valid, other=0).to(tl.uint16) << 8
    )
    code = tl.load(weight + (offset + 2 + d % 16).to(tl.int64), valid, other=0)
    code = tl.where((d % 32) < 16, code & 15, code >> 4)
    values = tl.where(
        code == 0,
        -127.0,
        tl.where(
            code == 1,
            -104.0,
            tl.where(
                code == 2,
                -83.0,
                tl.where(
                    code == 3,
                    -65.0,
                    tl.where(
                        code == 4,
                        -49.0,
                        tl.where(
                            code == 5,
                            -35.0,
                            tl.where(
                                code == 6,
                                -22.0,
                                tl.where(
                                    code == 7,
                                    -10.0,
                                    tl.where(
                                        code == 8,
                                        1.0,
                                        tl.where(
                                            code == 9,
                                            13.0,
                                            tl.where(
                                                code == 10,
                                                25.0,
                                                tl.where(
                                                    code == 11,
                                                    38.0,
                                                    tl.where(
                                                        code == 12,
                                                        53.0,
                                                        tl.where(
                                                            code == 13,
                                                            69.0,
                                                            tl.where(
                                                                code == 14, 89.0, 113.0
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
    tl.store(
        output + row * WIDTH + d, tl.where(valid, values * scale, 0.0), mask=d < WIDTH
    )
