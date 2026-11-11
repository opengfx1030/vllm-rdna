# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``top_k_per_row_decode`` on column-major logits wide enough to be split.

From 200,000 columns (kSplitWorkThreshold in sampler.cu) a decode row is split
over ten blocks that each keep their k best columns, and a second kernel merges
the candidates. Column-major logits (``stride1 != 1``) take the same path
through the kernels' strided loops, which had two defects there:

* They passed the absolute column to the code that stores ``rowStart + column``,
  so a block after the first counted its start twice: the selected set was
  wrong and hundreds of the k indices lay past the row.
* A block no longer than k returns its columns without selecting them. It read
  the logit of each one ignoring ``stride1``, so the merge ranked garbage.

Every case runs on column-major logits (``columns``, the layout under test) and
on a row-major copy of the same rows (``rows``, the control, which always
worked). A selection is right when its indices are inside the row, repeat none
and its logits are the k largest of the row, which is what ``torch.topk``
returns. On rows without equal logits that also fixes the set of columns.
``VLLM_RDNA2_MOE_DETERMINISTIC=1`` makes the kernels keep the lowest columns
among logits tied at the cut, so there the selection of a tied row must be the
first k columns of a stable descending sort.

The launcher reads the knob once per process, so each mode runs in a child
process (this file with ``--worker``), as in test_top_k_per_row_ties.py. Run
``pytest tests/kernels/test_top_k_per_row_strided.py``.
"""

import argparse
import functools
import os
import subprocess
import sys
import tempfile
from typing import NamedTuple

import pytest
import torch

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a GPU"
)

device = "cuda"
KNOB = "VLLM_RDNA2_MOE_DETERMINISTIC"
MODES = ["default", "deterministic"]
LAYOUTS = ["rows", "columns"]
REPEATS = 3
# The widths of the logits: the first is just above the split threshold
# (200,000 columns), the second is past 2^21, where the tie keys (stored
# columns) use their top digit.
SPLIT_WIDTH = 204_800
WIDE_WIDTH = 2_200_000
# Rows without two equal logits: the selected set is unique.
TIE_FREE = ("distinct", "clustered")


class Spec(NamedTuple):
    kind: str
    lengths: tuple[int, ...]  # live columns of each row
    width: int  # columns of the logits
    k: int


# Row lengths of the tied cases: a few columns apart, so the blocks differ.
TIED_LENGTHS = tuple(SPLIT_WIDTH - 37 * r for r in range(4))
WIDE_TIED_LENGTHS = tuple(WIDE_WIDTH - 37 * r for r in range(2))

CASES = {
    # Where the defect was found: hundreds of the indices were past the row.
    "distinct": Spec("distinct", (SPLIT_WIDTH,) * 3, SPLIT_WIDTH, 512),
    # Distinct logits 2^-23 apart: one bin of the first radix step holds
    # thousands of them, so the cut is resolved in the later steps.
    "clustered": Spec("clustered", (SPLIT_WIDTH,) * 3, SPLIT_WIDTH, 512),
    # Blocks of different lengths; the last one takes the remainder.
    "ragged": Spec(
        "distinct", (SPLIT_WIDTH, SPLIT_WIDTH - 1, 150_001, 60_000), SPLIT_WIDTH, 512
    ),
    "k2048": Spec("distinct", (SPLIT_WIDTH,) * 2, SPLIT_WIDTH, 2048),
    "wide": Spec("distinct", (WIDE_WIDTH, WIDE_WIDTH - 1), WIDE_WIDTH, 512),
    # Rows far shorter than the logits are wide: still split, in smaller blocks.
    "short_live": Spec("distinct", (100_000, 50_001, 20_000), SPLIT_WIDTH, 512),
    # Rows under ten times k: blocks of at most k columns return them as they
    # are (the last block takes the remainder, so it may still select).
    "blocks_at_k": Spec("distinct", (513, 3_000, 5_129, 5_300), SPLIT_WIDTH, 512),
    "blocks_at_k2048": Spec("distinct", (12_000, 20_481, 25_000), SPLIT_WIDTH, 2048),
    # The cut falls inside a run of equal logits (the ties test's rows).
    "sorted_bin": Spec("sorted_bin", TIED_LENGTHS, SPLIT_WIDTH, 512),
    "quantized": Spec("quantized", TIED_LENGTHS, SPLIT_WIDTH, 512),
    "zeros": Spec("zeros", TIED_LENGTHS, SPLIT_WIDTH, 512),
    "wide_zeros": Spec("zeros", WIDE_TIED_LENGTHS, WIDE_WIDTH, 512),
}


def _live_row(kind: str, length: int, gen: torch.Generator) -> torch.Tensor:
    """`length` logits of the given kind (see CASES)."""
    cols = torch.randperm(length, generator=gen)
    if kind == "distinct":
        return (cols.float() - length / 2) * 1e-3
    if kind == "clustered":
        return 1.0 + cols.float() * 2.0**-23
    if kind == "sorted_bin":
        # 400 larger values, then 300 copies of 1.0; the rest are below 0.5.
        # The bin holding the cut fits the final sort.
        row = torch.rand(length, generator=gen) * 0.5
        row[cols[:400]] = 2.0 + torch.arange(400, dtype=torch.float32)
        row[cols[400:700]] = 1.0
        return row
    if kind == "quantized":
        # Scores on a 1/16 grid: every value repeats length / 64 times.
        return torch.randint(0, 64, (length,), generator=gen).float() / 16
    # 300 positives over exact zeros: more ties than the final sort holds, so
    # the select runs to the last 10 bits.
    row = torch.zeros(length)
    row[cols[:300]] = 1.0 + torch.rand(300, generator=gen)
    return row


class Case(NamedTuple):
    spec: Spec
    live: list[torch.Tensor]  # the live logits of each row
    logits: torch.Tensor  # [rows, width], 1e20 past each row's length


@functools.cache
def _build(case: str) -> Case:
    spec = CASES[case]
    gen = torch.Generator().manual_seed(spec.width + spec.k)
    live = [_live_row(spec.kind, n, gen) for n in spec.lengths]
    # Columns past a row's length must never be selected.
    logits = torch.full((len(live), spec.width), 1e20)
    for r, row in enumerate(live):
        logits[r, : len(row)] = row
    return Case(spec, live, logits)


def _to_device(logits: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "rows":
        return logits.to(device)
    gpu = logits.t().contiguous().to(device).t()
    assert gpu.stride() == (1, logits.shape[0])
    return gpu


def _worker(path: str) -> None:
    """Runs in a child process: every case on both layouts, REPEATS times."""
    saved = {}
    for case in CASES:
        spec, _, logits = _build(case)
        num_rows = len(spec.lengths)
        seq_lens = torch.tensor(spec.lengths, dtype=torch.int32, device=device)
        for layout in LAYOUTS:
            gpu = _to_device(logits, layout)
            selected = []
            for _ in range(REPEATS):
                # A slot the kernel does not write stays -2, outside any row.
                out = torch.full(
                    (num_rows, spec.k), -2, dtype=torch.int32, device=device
                )
                torch.ops._C.top_k_per_row_decode(
                    gpu,
                    1,
                    seq_lens,
                    out,
                    num_rows,
                    gpu.stride(0),
                    gpu.stride(1),
                    spec.k,
                )
                selected.append(out.sort(dim=-1).values)
            saved[f"{case}-{layout}"] = torch.stack(selected).cpu()
    torch.save(saved, path)


_RUNS: dict[str, dict[str, torch.Tensor]] = {}
_FAILED: dict[str, str] = {}


def _selections(mode: str) -> dict[str, torch.Tensor]:
    """Selections of every case in a child process running in `mode`."""
    if mode not in _RUNS and mode not in _FAILED:
        env = dict(os.environ)
        if mode == "deterministic":
            env[KNOB] = "1"
        else:
            env.pop(KNOB, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "selections.pt")
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", path],
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if proc.returncode == 0:
                _RUNS[mode] = torch.load(path)
            else:
                _FAILED[mode] = (
                    f"{mode} worker exited with {proc.returncode}\n"
                    f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
                )
    if mode in _FAILED:
        pytest.fail(_FAILED[mode], pytrace=False)
    return _RUNS[mode]


def _lowest_tied(row: torch.Tensor, k: int) -> torch.Tensor:
    """The first k columns of a stable descending sort, ascending."""
    order = torch.sort(row, descending=True, stable=True).indices[:k]
    return order.sort().values.to(torch.int32)


def _assert_selects_top_k(
    got: torch.Tensor, row: torch.Tensor, spec: Spec, mode: str
) -> None:
    """`got`: the k selected columns of `row`, ascending."""
    n = len(row)
    outside = int(((got < 0) | (got >= n)).sum())
    assert outside == 0, f"{outside} of {spec.k} indices are not columns of the row"
    assert bool((got[1:] > got[:-1]).all()), "a column is selected twice"
    want = row.topk(spec.k)
    picked = row[got.long()].sort(descending=True).values
    assert torch.equal(picked, want.values), "the logits are not the k largest"
    if spec.kind in TIE_FREE:
        assert torch.equal(got, want.indices.sort().values.to(torch.int32)), (
            "not the columns torch.topk selects"
        )
    elif mode == "deterministic":
        assert torch.equal(got, _lowest_tied(row, spec.k)), (
            "not the lowest columns among the tied logits"
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", CASES)
def test_decode_selects_top_k(mode: str, layout: str, case: str):
    spec, live, _ = _build(case)
    selected = _selections(mode)[f"{case}-{layout}"]
    for repeat, per_row in enumerate(selected):
        for r, row in enumerate(live):
            try:
                _assert_selects_top_k(per_row[r], row, spec, mode)
            except AssertionError as e:
                raise AssertionError(
                    f"call {repeat}, row {r} ({len(row)} columns): {e}"
                ) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, help="Write the selections here")
    with torch.inference_mode():
        _worker(parser.parse_args().worker)
