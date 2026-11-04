# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal logits at the cut of ``top_k_per_row_{decode,prefill}`` (sampler.cu).

When several logits tie exactly at the k-th value, which of the tied columns
make the cut depends on the order threads reach shared-memory atomics (the last
10-bit step of the radix select, the gather that feeds the sorted bin) and on
the slot order of the insertion sort.

* Default: the kernels return *a* valid top-k. The selected logits are the k
  largest, whichever tied columns are kept.
* ``VLLM_RDNA2_MOE_DETERMINISTIC=1``: among tied logits the lowest columns win,
  so the selected set is the first k columns of a stable descending sort and
  every call returns it.

The launcher reads the knob once per process (a CUDA graph must not see it
change), so each mode runs in a child process (this file with ``--worker``)
that saves the sorted selections of REPEATS calls per case; the tests below
check them. Run ``pytest tests/kernels/test_top_k_per_row_ties.py``.
"""

import argparse
import functools
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
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
TOP_K = 512
REPEATS = 50
# Decode rows at least this wide are split over several blocks whose
# candidates a second kernel merges (kSplitWorkThreshold is 200,000).
SPLIT_WIDTH = 204_800
# Past 2^21 columns the tie keys (stored columns) use their top digit and the
# last split blocks cannot skip it.
WIDE_WIDTH = 2_200_000
# Prefill rows before this one run the insertion-sort kernel, the rows from it
# on the radix-sort kernel (kSortingAlgorithmThreshold).
PREFILL_RADIX_FIRST_ROW = 12_288
KINDS = ["sorted_bin", "quantized", "zeros", "neg_inf"]
DECODE_SHAPES = [
    (4096, "rows"),
    (16384, "rows"),
    (16384, "columns"),
    (SPLIT_WIDTH, "rows"),
    (WIDE_WIDTH, "rows"),
]
# Cases that are also captured in a CUDA graph and replayed: the select that
# runs to the last 10 bits and the sorted bin, on a single-block row, a split
# row and both prefill kernels.
GRAPH_KINDS = ["sorted_bin", "zeros"]
GRAPH_IDS = [
    *(f"decode-{k}-16384-rows" for k in GRAPH_KINDS),
    *(f"decode-{k}-{SPLIT_WIDTH}-rows" for k in GRAPH_KINDS),
    *(f"prefill-{k}" for k in GRAPH_KINDS),
    *(f"prefill_radix-{k}" for k in GRAPH_KINDS),
]


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


def _prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    out = torch.empty(num_rows, TOP_K, dtype=torch.int32, device=device)
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        out,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        TOP_K,
    )
    return out


class Case(NamedTuple):
    # The live logits of each checked row, on the CPU.
    live: list[torch.Tensor]
    # Moves the inputs to the GPU; returns run(), one call of the op that
    # gives the TOP_K selected columns of each checked row (int32).
    launch: Callable[[], Callable[[], torch.Tensor]]


def _decode_case(kind: str, width: int, layout: str) -> Case:
    """Single-block rows (vectorized and strided loads) and split rows."""
    gen = torch.Generator().manual_seed(width)
    lengths = [width - 37 * r for r in range(4)]
    if layout == "rows":
        logits = torch.empty(len(lengths), width)
    else:
        logits = torch.empty(width, len(lengths)).t()
    # Columns past a row's length must never be selected.
    logits.fill_(1e20)
    live = [_row(kind, n, gen) for n in lengths]
    for r, row in enumerate(live):
        logits[r, : len(row)] = row

    def launch():
        gpu_logits = logits.to(device)
        seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        return lambda: _decode(gpu_logits, seq_lens)

    return Case(live, launch)


def _prefill_case(kind: str) -> Case:
    """Row windows at nonzero offsets; the tied bin goes through the
    insertion sort or the 10-bit select."""
    gen = torch.Generator().manual_seed(7)
    width = 20_000
    starts = [0, 100, 1500, 3001]
    ends = [16_000, 14_100, 20_000, 19_001]
    logits = torch.full((len(starts), width), 1e20)
    live = []
    for r, (s, e) in enumerate(zip(starts, ends)):
        live.append(_row(kind, e - s, gen))
        logits[r, s:e] = live[-1]

    def launch():
        gpu_logits = logits.to(device)
        row_starts = torch.tensor(starts, dtype=torch.int32, device=device)
        row_ends = torch.tensor(ends, dtype=torch.int32, device=device)
        return lambda: _prefill(gpu_logits, row_starts, row_ends, len(starts))

    return Case(live, launch)


def _prefill_radix_case(kind: str) -> Case:
    """The rows after PREFILL_RADIX_FIRST_ROW use the radix-sort kernel. Every
    row aliases one logits row (stride 0); the last four are checked."""
    gen = torch.Generator().manual_seed(11)
    width, start, end = 20_000, 100, 14_100
    num_rows = PREFILL_RADIX_FIRST_ROW + 4
    row = torch.full((width,), 1e20)
    live = _row(kind, end - start, gen)
    row[start:end] = live

    def launch():
        gpu_logits = row.to(device)[None].expand(num_rows, width)
        row_starts = torch.full((num_rows,), start, dtype=torch.int32, device=device)
        row_ends = torch.full((num_rows,), end, dtype=torch.int32, device=device)
        return lambda: _prefill(gpu_logits, row_starts, row_ends, num_rows)[
            PREFILL_RADIX_FIRST_ROW:
        ]

    return Case([live] * 4, launch)


@functools.cache
def _build(case_id: str) -> Case:
    path, kind, *shape = case_id.split("-")
    if path == "decode":
        return _decode_case(kind, int(shape[0]), shape[1])
    if path == "prefill":
        return _prefill_case(kind)
    return _prefill_radix_case(kind)


def _call_repeatedly(run: Callable[[], torch.Tensor]) -> torch.Tensor:
    return torch.stack([run().sort(dim=-1).values for _ in range(REPEATS)]).cpu()


def _replay_repeatedly(run: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Capture one call in a CUDA graph and replay it. The launcher picks the
    kernel variant while it is captured."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    selected = []
    for _ in range(REPEATS):
        out.fill_(-2)  # a replay that wrote nothing would show
        graph.replay()
        selected.append(out.sort(dim=-1).values)
    return torch.stack(selected).cpu()


def _worker(path: str) -> None:
    """Runs in a child process: every case, REPEATS times, eager and (for
    GRAPH_IDS) replayed from a CUDA graph."""
    saved = {}
    ids = [
        *(f"decode-{k}-{w}-{lay}" for k in KINDS for w, lay in DECODE_SHAPES),
        *(f"prefill-{k}" for k in KINDS),
        *(f"prefill_radix-{k}" for k in KINDS),
    ]
    for case_id in ids:
        run = _build(case_id).launch()
        saved[case_id] = _call_repeatedly(run)
        if case_id in GRAPH_IDS:
            saved[f"{case_id}-graph"] = _replay_repeatedly(run)
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


def _assert_valid_top_k(selected: torch.Tensor, live: list[torch.Tensor]) -> None:
    """Every repeat selects TOP_K distinct in-range columns whose logits are
    the TOP_K largest of the row."""
    for i, row in enumerate(live):
        picked = selected[:, i]
        assert picked.min() >= 0 and picked.max() < len(row), f"row {i}: out of range"
        assert bool((picked[:, 1:] > picked[:, :-1]).all()), (
            f"row {i}: repeats a column"
        )
        got = row[picked.long()].sort(dim=-1, descending=True).values
        want = row.topk(TOP_K).values.expand_as(got)
        assert torch.equal(got, want), f"row {i}: not the {TOP_K} largest logits"


def _assert_lowest_tied(selected: torch.Tensor, live: list[torch.Tensor]) -> None:
    """Every repeat selects the first TOP_K columns of a stable descending
    sort."""
    varying = sum(not torch.equal(r, selected[0]) for r in selected[1:])
    wrong = [
        i
        for i, row in enumerate(live)
        if not torch.equal(selected[0, i], _lowest_tied(row))
    ]
    assert varying == 0 and not wrong, (
        f"{varying} of {REPEATS - 1} repeats differ from the first; rows "
        f"{wrong} do not keep the lowest tied columns"
    )


def _check(mode: str, key: str) -> None:
    selected = _selections(mode)[key]
    live = _build(key.removesuffix("-graph")).live
    _assert_valid_top_k(selected, live)
    if mode == "deterministic":
        _assert_lowest_tied(selected, live)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width,layout", DECODE_SHAPES)
def test_decode_ties(mode: str, kind: str, width: int, layout: str):
    _check(mode, f"decode-{kind}-{width}-{layout}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("path", ["prefill", "prefill_radix"])
def test_prefill_ties(mode: str, kind: str, path: str):
    _check(mode, f"{path}-{kind}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case_id", GRAPH_IDS)
def test_ties_in_a_replayed_cuda_graph(mode: str, case_id: str):
    _check(mode, f"{case_id}-graph")


@pytest.mark.parametrize("width", [4096, 65536, SPLIT_WIDTH])
@torch.inference_mode()
def test_decode_selects_top_values_on_random_rows(width: int):
    """Distinct random scores, including rows no longer than k (-1 padding).
    Holds in either mode, so it runs in this process's."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, help="Write the selections here")
    with torch.inference_mode():
        _worker(parser.parse_args().worker)
