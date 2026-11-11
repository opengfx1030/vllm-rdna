# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""exllamav3 ``exl3_ngram_trellis`` rows decode like the format's definition."""

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.exl3_ple import (
    Exl3NgramRingTable,
    mul1_codebook,
)

WIDTH = 160
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


class _SmallTable(Exl3NgramRingTable):
    ROWS_PER_SHARD = 64


def _pack(symbols: torch.Tensor, scales: torch.Tensor, k: int) -> torch.Tensor:
    """Rows of K-bit symbols -> packed rings (scale word + LSB-first bitstream)."""
    n, dim = symbols.shape
    bits = (symbols.unsqueeze(-1) >> torch.arange(k)) & 1
    bits = bits.reshape(n, dim * k // 16, 16)
    words = (bits << torch.arange(16)).sum(-1)
    words = torch.where(words >= 0x8000, words - 0x10000, words).to(torch.int16)
    return torch.cat((scales.to(torch.float16).view(torch.int16)[:, None], words), 1)


def _reference_rows(symbols, scales, bias_rows, k):
    codebook = mul1_codebook().float()
    n, dim = symbols.shape
    out = torch.empty(n, dim)
    for r in range(n):
        for i in range(dim):
            state = 0
            for j in range((15 + k) // k):
                state |= int(symbols[r, (i - j) % dim]) << (j * k)
            out[r, i] = codebook[state & 0xFFFF] * float(scales[r]) + bias_rows[r, i]
    return out


@pytest.mark.parametrize("k", [4, 5, 6])
def test_ring_rows_match_definition(tmp_path, k):
    torch.manual_seed(k)
    shards, heads = 3, 4
    rows = shards * _SmallTable.ROWS_PER_SHARD
    symbols = torch.randint(0, 1 << k, (rows, WIDTH))
    scales = torch.rand(rows) * 0.05 + 0.01
    packed = _pack(symbols, scales, k)
    offsets = torch.tensor([0, 50, 100, 150])
    bias = torch.randn(heads, WIDTH).half()
    tensors = {
        f"{PREFIX}.shard_{n}.trellis": packed[n * 64 : (n + 1) * 64].clone()
        for n in range(shards)
    }
    tensors[f"{PREFIX}.head_offsets"] = offsets
    tensors[f"{PREFIX}.head_vocab_sizes"] = torch.tensor([50, 50, 50, rows - 150])
    tensors[f"{PREFIX}.layer_multipliers"] = torch.tensor([1, 3, 5])
    tensors[f"{PREFIX}.head_bias"] = bias
    path = tmp_path / "ngram_embedding.safetensors"
    save_file(tensors, str(path))

    table = _SmallTable(str(path), rows, WIDTH, head_offsets=offsets)
    assert k == table.K

    ids = torch.tensor([0, 63, 64, 191, 120, 49, 50, 150, 64, 3])
    out = torch.empty(ids.shape[0], WIDTH, dtype=torch.float32)
    table.gather_into(ids, out)

    heads_of = torch.searchsorted(offsets, ids, right=True) - 1
    want = _reference_rows(
        symbols[ids], scales.half().float()[ids], bias.float()[heads_of], k
    )
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5)


def test_hash_layout_mismatch_is_rejected(tmp_path):
    rows = 2 * _SmallTable.ROWS_PER_SHARD
    packed = _pack(torch.zeros(rows, WIDTH, dtype=torch.long), torch.ones(rows), 5)
    tensors = {
        f"{PREFIX}.shard_{n}.trellis": packed[n * 64 : (n + 1) * 64].clone()
        for n in range(2)
    }
    tensors[f"{PREFIX}.head_offsets"] = torch.tensor([0, 64])
    tensors[f"{PREFIX}.head_vocab_sizes"] = torch.tensor([64, 64])
    tensors[f"{PREFIX}.layer_multipliers"] = torch.tensor([1, 3])
    tensors[f"{PREFIX}.head_bias"] = torch.zeros(2, WIDTH).half()
    path = tmp_path / "ngram_embedding.safetensors"
    save_file(tensors, str(path))

    with pytest.raises(ValueError, match="layer_multipliers"):
        _SmallTable(str(path), rows, WIDTH, layer_multipliers=torch.tensor([1, 7]))


def test_mul1_codebook_is_unit_scale():
    cb = mul1_codebook().float()
    assert cb.shape == (65536,)
    assert torch.isfinite(cb).all()
    assert abs(cb.mean().item()) < 0.05
    assert 0.8 < cb.std().item() < 1.2
