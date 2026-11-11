# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracles for the P2P-less gfx1030 gather replacements.

On hosts where ``all_gather_into_tensor`` returns a corrupt tensor (wrong-rank
shard at offset 0, zero holes) the TP gathers are rebuilt from collectives that
are exact there. These tests prove the replacements are bit-equal to the
reference ``all_gather`` layout and to a reference global argmax, for worlds 2
and 4.

The ranks run as threads over an in-process fabric with exact all-reduce
semantics, so the tests exercise the shipping helpers (extracted from source,
not copied) without a process group, a network rendezvous or a GPU.
"""

import ast
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
COMM_SRC = REPO / "vllm/distributed/communication_op.py"
LOGITS_SRC = REPO / "vllm/model_executor/layers/logits_processor.py"
VOCAB_SHARD = 1024
NO_INDEX = 10**9


def _shipping_function(path: Path, name: str, namespace: dict[str, Any]) -> Any:
    """Compile the top-level function ``name`` from ``path`` into ``namespace``."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(path), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in {path.name}")


class _Fabric:
    """All-reduce across ``world`` threads: every rank sees the exact result."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.rank = threading.local()
        self._barrier = threading.Barrier(world, timeout=30)
        self._slots: list[Any] = [None] * world

    def group(self) -> SimpleNamespace:
        return SimpleNamespace(
            world_size=self.world, rank_in_group=self.rank.value, device_group=None
        )

    def all_reduce(self, tensor: Any, op: Any = None, group: Any = None) -> Any:
        self._slots[self.rank.value] = tensor.clone()
        self._barrier.wait()
        stacked = torch.stack(self._slots)
        if op is None or op == dist.ReduceOp.SUM:
            result = stacked.sum(0)
        elif op == dist.ReduceOp.MAX:
            result = stacked.max(0).values
        elif op == dist.ReduceOp.MIN:
            result = stacked.min(0).values
        else:
            raise AssertionError(f"unsupported reduce op {op}")
        self._barrier.wait()  # every rank has read the slots before reuse
        tensor.copy_(result)
        return tensor


def _run_ranks(
    world: int,
    per_rank: Callable[[int, _Fabric], Any],
    fabric: _Fabric | None = None,
) -> list[Any]:
    fabric = fabric or _Fabric(world)
    results: list[Any] = [None] * world
    errors: list[BaseException] = []

    def worker(rank: int) -> None:
        fabric.rank.value = rank
        try:
            results[rank] = per_rank(rank, fabric)
        except BaseException as exc:  # noqa: BLE001 - surfaced after join
            errors.append(exc)
            fabric._barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    if errors:
        raise errors[0]
    return results


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 64), torch.float16),
        ((3, 97), torch.float16),
        ((8, 15072), torch.float16),  # head logits family: [tokens, vocab shard]
        ((5, 128), torch.float32),
    ],
)
def test_zero_pad_all_reduce_gather_matches_all_gather(
    world: int, shape: tuple[int, int], dtype: Any
) -> None:
    torch.manual_seed(1234)
    shards = [torch.randn(*shape, dtype=dtype) for _ in range(world)]
    expected = torch.cat(shards, dim=-1)  # the all_gather layout

    def per_rank(rank: int, fabric: _Fabric) -> Any:
        gather = _shipping_function(
            COMM_SRC,
            "tensor_model_parallel_all_gather_via_all_reduce",
            {
                "torch": torch,
                "get_tp_group": fabric.group,
                "tensor_model_parallel_all_reduce": lambda t: fabric.all_reduce(t),
            },
        )
        return gather(shards[rank])

    for rank, got in enumerate(_run_ranks(world, per_rank)):
        assert got.shape == expected.shape
        assert torch.equal(got, expected), f"rank {rank} layout mismatch"


@pytest.mark.parametrize("world", [2, 4])
def test_top_tokens_all_reduce_argmax_matches_reference(
    world: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 6
    torch.manual_seed(31337)
    vals = torch.rand(rows, world) * 0.5
    idxs = torch.arange(world)[None, :] * VOCAB_SHARD + torch.randint(
        1, VOCAB_SHARD, (rows, world)
    )
    # Last row: an exact value tie on every rank must resolve to the lowest index.
    vals[-1] = 0.9
    idxs[-1] = torch.arange(world) * VOCAB_SHARD + 7

    def per_rank(rank: int, fabric: _Fabric) -> Any:
        argmax = _shipping_function(
            LOGITS_SRC,
            "_top_tokens_allreduce_argmax",
            {"torch": torch, "get_tp_group": fabric.group},
        )
        return argmax(vals[:, rank].clone(), idxs[:, rank].clone())

    # The helper calls dist.all_reduce directly; route it onto the fabric.
    fabric = _Fabric(world)
    monkeypatch.setattr(dist, "all_reduce", fabric.all_reduce)
    results = _run_ranks(world, per_rank, fabric)

    winners = vals.argmax(dim=-1)
    for row in range(rows - 1):  # unique maxima: plain argmax is the oracle
        assert int(results[0][row]) == int(idxs[row, winners[row]])
    assert int(results[0][-1]) == int(idxs[-1].min())  # lowest-index tie-break
    for rank in range(1, world):
        assert torch.equal(results[rank], results[0]), "ranks disagree"
    assert results[0].dtype == torch.int64
