# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal logits at the cut of ``top_k_per_row_{decode,prefill}`` (sampler.cu).

Among equal logits the kernels keep the lowest columns, so every call selects
the same set: the first k columns of a stable descending sort. The kept ties
used to be whichever threads won a shared-memory atomic (when the select runs
down to the last 10 bits) or came first in the order the threshold bin was
filled (when that bin is sorted).

Run ``pytest tests/kernels/test_top_k_per_row_ties.py``.
"""

import pytest
import torch

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a GPU"
)

device = "cuda"
TOP_K = 512
REPEATS = 50
# Decode rows at least this wide are split over several blocks whose
# candidates a second kernel merges (kSplitWorkThreshold is 200,000).
SPLIT_WIDTH = 204_800
KINDS = ["sorted_bin", "quantized", "zeros", "neg_inf"]


def _row(kind: str, length: int, gen: torch.Generator) -> torch.Tensor:
    """`length` logits whose top-k cut falls inside a run of equal values."""
    cols = torch.randperm(length, generator=gen)
    if kind == "sorted_bin":
        # 400 larger values, then 300 copies of 1.0; the rest are below 0.5.
        # The bin holding the cut fits the final sort.
        row = torch.rand(length, generator=gen) * 0.5
        row[cols[:400]] = 2.0 + torch.arange(400, dtype=torch.float32)
        row[cols[400:700]] = 1.0
    elif kind == "quantized":
        # Scores on a 1/16 grid: every value repeats length / 64 times.
        row = torch.randint(0, 64, (length,), generator=gen).float() / 16
    elif kind == "zeros":
        # 300 positives over exact zeros (QSA's relu scores): more ties than
        # the final sort holds, so the select runs to the last 10 bits.
        row = torch.zeros(length)
        row[cols[:300]] = 1.0 + torch.rand(300, generator=gen)
    else:
        # A masked row: 100 finite scores, the rest -inf.
        row = torch.full((length,), float("-inf"))
        row[cols[:100]] = torch.randn(100, generator=gen)
    return row


def _lowest_tied(row: torch.Tensor) -> torch.Tensor:
    """The first TOP_K columns of a stable descending sort, ascending."""
    order = torch.sort(row, descending=True, stable=True).indices[:TOP_K]
    return order.sort().values.to(torch.int32)


def _assert_same_lowest_set(run, expected: torch.Tensor) -> None:
    results = torch.stack([run().sort(dim=-1).values for _ in range(REPEATS)])
    results = results.cpu()
    varying = sum(not torch.equal(r, results[0]) for r in results[1:])
    wrong = [
        i
        for i in range(expected.shape[0])
        if not torch.equal(results[0, i], expected[i])
    ]
    assert varying == 0 and not wrong, (
        f"{varying} of {REPEATS - 1} repeats differ from the first; rows "
        f"{wrong} do not keep the lowest tied columns"
    )


def _decode(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = torch.empty(logits.shape[0], TOP_K, dtype=torch.int32, device=device)
    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        lengths,
        out,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        TOP_K,
    )
    return out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "width,layout",
    [(4096, "rows"), (16384, "rows"), (16384, "columns"), (SPLIT_WIDTH, "rows")],
)
@torch.inference_mode()
def test_decode_keeps_lowest_tied_columns(kind: str, width: int, layout: str):
    """Single-block rows (vectorized and strided loads) and split rows."""
    gen = torch.Generator().manual_seed(width)
    lengths = [width - 37 * r for r in range(4)]
    if layout == "rows":
        logits = torch.empty(len(lengths), width)
    else:
        logits = torch.empty(width, len(lengths)).t()
    # Columns past a row's length must never be selected.
    logits.fill_(1e20)
    for r, n in enumerate(lengths):
        logits[r, :n] = _row(kind, n, gen)
    expected = torch.stack([_lowest_tied(logits[r, :n]) for r, n in enumerate(lengths)])
    logits = logits.to(device)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    _assert_same_lowest_set(lambda: _decode(logits, seq_lens), expected)


@pytest.mark.parametrize("kind", KINDS)
@torch.inference_mode()
def test_prefill_keeps_lowest_tied_columns(kind: str):
    """Row windows at nonzero offsets; the tied bin goes through the
    insertion sort or the 10-bit select."""
    gen = torch.Generator().manual_seed(7)
    width = 20_000
    starts = [0, 100, 1500, 3001]
    ends = [16_000, 14_100, 20_000, 19_001]
    logits = torch.full((len(starts), width), 1e20)
    for r, (s, e) in enumerate(zip(starts, ends)):
        logits[r, s:e] = _row(kind, e - s, gen)
    expected = torch.stack(
        [_lowest_tied(logits[r, s:e]) for r, (s, e) in enumerate(zip(starts, ends))]
    )
    logits = logits.to(device)
    row_starts = torch.tensor(starts, dtype=torch.int32, device=device)
    row_ends = torch.tensor(ends, dtype=torch.int32, device=device)

    def run():
        out = torch.empty(len(starts), TOP_K, dtype=torch.int32, device=device)
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            out,
            len(starts),
            logits.stride(0),
            logits.stride(1),
            TOP_K,
        )
        return out

    _assert_same_lowest_set(run, expected)


@pytest.mark.parametrize("width", [4096, 65536, SPLIT_WIDTH])
@torch.inference_mode()
def test_decode_selects_top_values_on_random_rows(width: int):
    """Distinct random scores, including rows no longer than k (-1 padding)."""
    gen = torch.Generator(device=device).manual_seed(width)
    logits = torch.randn(8, width, device=device, generator=gen)
    lengths = [width, width - 1, width // 2, width // 3 + 1, 600, 513, 512, 100]
    out = _decode(logits, torch.tensor(lengths, dtype=torch.int32, device=device))
    logits, out = logits.cpu(), out.cpu()
    for r, n in enumerate(lengths):
        k = min(n, TOP_K)
        valid = (out[r] >= 0) & (out[r] < n)
        assert torch.all(valid | (out[r] == -1)), f"row {r}: index out of range"
        picked = out[r][valid].long()
        assert picked.numel() == k and picked.unique().numel() == k, f"row {r}"
        got = logits[r, picked].sort(descending=True).values
        assert torch.equal(got, logits[r, :n].topk(k).values), f"row {r}"
