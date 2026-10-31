# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU (gloo) oracle tests for the gfx1030 gather workarounds.

Why: on P2P-less gfx1030 hosts (fogbox V620 class) the plain
``all_gather_into_tensor`` of the full logits row was reproduced as
corrupted (maintainer PR5 campaign: logits_processor.passing-910acae8.py,
the george-tree linear.py gather_output patch, RUNBOOK-pr5-fast-correct).
The fix replaces the two affected collectives with an all-reduce over a
zero-padded own-rank slice (identical result layout), gated so the default
path is byte-identical when the environment flags are unset.

To stay runnable in the minimal CPU torch venv (no compiled vllm, no ROCm,
no transformers stack), the shipped helper functions are extracted as
SOURCE TEXT from the real files via ``ast`` and executed against real
torch.distributed gloo process groups.  This tests the exact shipping
code, not a copy.

Contracts pinned here:
  * layout/equivalence vs a reference ``dist.all_gather`` (world 2 and 4),
  * env-gate behavior (unset => untouched default dispatch, mutual
    exclusion, all-gather requirement),
  * ColumnParallelLinear wiring: env set => workaround branch; unset =>
    the plain upstream call, no unconditional host copy.

These do NOT claim GPU reproduction — the collective-kernel corruption
itself requires the faulted hardware class.
"""

import ast
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
LP_SRC = REPO / "vllm/model_executor/layers/logits_processor.py"
LIN_SRC = REPO / "vllm/model_executor/layers/linear.py"


# --------------------------------------------------------------------------
# source extraction harness
# --------------------------------------------------------------------------

def _extract_func(src_path: Path, name: str) -> str:
    """Return the exact source of top-level or method ``name`` in the file."""
    text = src_path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                seg = ast.get_source_segment(text, node)
                assert seg is not None, f"no source segment for {name}"
                return seg
    raise AssertionError(f"{name} not found in {src_path.name}")


def _exec_helper(src_path: Path, name: str, ns: dict):
    """Execute the extracted function source in namespace ``ns``; return it."""
    src = _extract_func(src_path, name)
    # de-indent method bodies so they exec at module level
    lines = src.splitlines()
    if lines and lines[0].startswith((" ", "\t")):
        import textwrap

        src = textwrap.dedent(src)
    exec(compile(src, str(src_path), "exec"), ns)
    return ns[name]


class _FakeGroup:
    def __init__(self, world_size=2, rank_in_group=0, device_group=None,
                 cpu_group=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.device_group = device_group
        self.cpu_group = cpu_group


def _lp_namespace(tp_group):
    """Namespace providing what _gather_logits / _head_allreduce_gather
    reference, with the *real* torch and stubbed vllm seams."""
    import types

    vllm = types.ModuleType("vllm")
    vd = types.ModuleType("vllm.distributed")
    vd.get_tp_group = lambda: tp_group
    sys.modules.setdefault("vllm", vllm)
    sys.modules["vllm.distributed"] = vd
    ns = {
        "os": os,
        "torch": torch,
        "get_tp_group": lambda: tp_group,
        "tensor_model_parallel_all_gather": lambda x, dim=-1: x,
        "tensor_model_parallel_gather": lambda x, dim=-1: x,
    }
    return ns


# --------------------------------------------------------------------------
# distributed oracle: zero-pad + all-reduce == all_gather layout
# --------------------------------------------------------------------------

def _rank_main(rank, world, fn_name, init_file):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank,
        world_size=world)
    try:
        fn = {
            "head_layout": _check_head_layout,
            "linear_layout": _check_linear_layout,
            "head_dispatch": _check_head_dispatch,
            "top_tokens": _check_top_tokens,
        }[fn_name]
        fn(rank, world)
    finally:
        dist.destroy_process_group()


def _run_spawned(fn_name, world):
    import tempfile

    init_file = os.path.join(
        tempfile.mkdtemp(prefix="fn_gather_test_"), "init")
    open(init_file, "a").close()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_rank_main, args=(r, world, fn_name, init_file))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"rank exited with {p.exitcode}"


def _check_head_layout(rank, world):
    """_head_allreduce_gather (shipping source) must equal reference
    all_gather. Mirrors the failing shape family: [n_tokens, vocab_shard].
    """
    from unittest.mock import MagicMock

    tp = MagicMock()
    tp.world_size = world
    tp.rank_in_group = rank
    tp.device_group = None  # default group
    ns = _lp_namespace(tp)
    helper = _exec_helper(LP_SRC, "_head_allreduce_gather", ns)

    torch.manual_seed(1234 + rank)
    for shape in ((1, 64), (3, 97), (8, 15072)):
        shard = torch.randn(*shape, dtype=torch.float16)
        got = helper(shard, tp)
        assert got.shape == (*shape[:-1], shape[-1] * world)
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard.contiguous())
        ref = torch.cat(parts, dim=-1)
        assert torch.equal(got, ref), f"layout mismatch at shape {shape}"
        # rank r must own the contiguous block [r*shard, (r+1)*shard)
        sl = slice(rank * shape[-1], (rank + 1) * shape[-1])
        assert torch.equal(got[..., sl], shard)


def _check_linear_layout(rank, world):
    """_gather_output_allreduce (shipping source) must equal reference."""
    from unittest.mock import MagicMock

    tp = MagicMock()
    tp.world_size = world
    tp.rank_in_group = rank
    tp.device_group = None
    ns = _lp_namespace(tp)
    helper = _exec_helper(LIN_SRC, "_gather_output_allreduce", ns)

    torch.manual_seed(7777 + rank)
    for shape in ((2, 33), (5, 128)):
        shard = torch.randn(*shape, dtype=torch.float32)
        got = helper(shard)
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard.contiguous())
        assert torch.equal(got, torch.cat(parts, dim=-1))


@pytest.mark.parametrize("world", [2, 4])
def test_head_allreduce_gather_matches_reference(world):
    _run_spawned("head_layout", world)


@pytest.mark.parametrize("world", [2, 4])
def test_linear_gather_allreduce_matches_reference(world):
    _run_spawned("linear_layout", world)


# --------------------------------------------------------------------------
# env-gate contracts on the shipping _gather_logits source (single proc)
# --------------------------------------------------------------------------

def _gather_logits_fn(ns, tp):
    ns["_head_allreduce_gather"] = _exec_helper(
        LP_SRC, "_head_allreduce_gather", ns)
    gather = _exec_helper(LP_SRC, "_gather_logits", ns)

    class P:
        use_all_gather = True

    p = P()

    def call(x):
        return gather(p, x)

    return call


def test_head_env_flags_unset_keeps_default_dispatch(monkeypatch):
    """Unset env must dispatch to the plain collective — no fallback."""
    for var in ("FLASHNEXT_HEAD_ALLREDUCE", "FLASHNEXT_HEAD_GLOO"):
        monkeypatch.delenv(var, raising=False)
    tp = _FakeGroup()
    ns = _lp_namespace(tp)
    called = {}

    def fake_all_gather(x, dim=-1):
        called["plain"] = True
        return x

    ns["tensor_model_parallel_all_gather"] = fake_all_gather
    call = _gather_logits_fn(ns, tp)
    x = torch.zeros(2, 3)
    out = call(x)
    assert called.get("plain")
    assert out is x


def test_head_env_flags_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("FLASHNEXT_HEAD_ALLREDUCE", "1")
    monkeypatch.setenv("FLASHNEXT_HEAD_GLOO", "1")
    tp = _FakeGroup()
    ns = _lp_namespace(tp)
    call = _gather_logits_fn(ns, tp)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        call(torch.zeros(1, 4))


def test_head_allreduce_requires_all_gather(monkeypatch):
    monkeypatch.setenv("FLASHNEXT_HEAD_ALLREDUCE", "1")
    monkeypatch.delenv("FLASHNEXT_HEAD_GLOO", raising=False)
    tp = _FakeGroup()
    ns = _lp_namespace(tp)
    call = _gather_logits_fn(ns, tp)
    # force use_all_gather False through the namespace class
    import types

    src_ns = ns

    class P:
        use_all_gather = False

    gather_src = _extract_func(LP_SRC, "_gather_logits")
    import textwrap

    g = {}
    exec(compile(textwrap.dedent(gather_src), str(LP_SRC), "exec"),
         src_ns, g)
    p = P()
    g["_head_allreduce_gather"] = ns["_head_allreduce_gather"]
    with pytest.raises(RuntimeError, match="requires all-gather"):
        g["_gather_logits"](p, torch.zeros(1, 4))


def _check_head_dispatch(rank, world):
    """With FLASHNEXT_HEAD_ALLREDUCE=1 the shipping _gather_logits must
    route through the all-reduce helper and reproduce the gather layout
    (world=1 identity here; the TP2/TP4 equivalence tests cover the rest).
    """
    import textwrap

    os.environ["FLASHNEXT_HEAD_ALLREDUCE"] = "1"
    os.environ.pop("FLASHNEXT_HEAD_GLOO", None)
    try:
        from unittest.mock import MagicMock

        tp = MagicMock()
        tp.world_size = 1
        tp.rank_in_group = 0
        tp.device_group = None
        ns = _lp_namespace(tp)
        ns["_head_allreduce_gather"] = _exec_helper(
            LP_SRC, "_head_allreduce_gather", ns)
        g = {}
        exec(compile(textwrap.dedent(_extract_func(LP_SRC, "_gather_logits")),
                     str(LP_SRC), "exec"), ns, g)

        class P:
            use_all_gather = True

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = g["_gather_logits"](P(), x)
        assert out.shape == (2, 3)
        assert torch.equal(out, x)  # world-1 identity: same layout contract
    finally:
        os.environ.pop("FLASHNEXT_HEAD_ALLREDUCE", None)


def test_head_allreduce_dispatches_when_env_set():
    _run_spawned("head_dispatch", 1)


def test_linear_env_gate_wiring():
    """Source-level: ColumnParallelLinear.forward must keep the plain call
    as the default branch, gate the workaround behind the env constant, and
    contain no unconditional host copy."""
    text = LIN_SRC.read_text()
    tree = ast.parse(text)
    fwd = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            seg = ast.get_source_segment(text, node)
            if "gather_output" in seg:
                fwd = seg
                break
    assert fwd is not None, "ColumnParallelLinear.forward not found"
    assert "tensor_model_parallel_all_gather(output_parallel)" in fwd
    assert "FLASHNEXT_GATHER_ALLREDUCE_ENV" in fwd
    assert "_gather_output_allreduce(output_parallel)" in fwd
    # default path must not touch CPU
    assert ".cpu()" not in fwd
    # the env gate is a runtime check, not a hardcoded enable
    assert 'os.environ.get(FLASHNEXT_GATHER_ALLREDUCE_ENV) == "1"' in fwd


# --------------------------------------------------------------------------
# site 3: get_top_tokens (value, index) pair gather -> all-reduce argmax.
# The lane's live-patches logits file (4425834a) also closes the tiny
# (value, index) all_gather in LogitsProcessor.get_top_tokens, because it
# runs the same corrupted all_gather_into_tensor collective.
# --------------------------------------------------------------------------

def _check_top_tokens(rank, world):
    """_top_tokens_allreduce_argmax (shipping source) == reference global
    argmax over per-rank (value, index) pairs, tie-broken by the lowest
    global vocab index; deterministic and identical on every rank."""
    from unittest.mock import MagicMock

    tp = MagicMock()
    tp.world_size = world
    tp.rank_in_group = rank
    tp.device_group = None
    ns = _lp_namespace(tp)
    helper = _exec_helper(LP_SRC, "_top_tokens_allreduce_argmax", ns)

    VSHARD = 1024  # vocab columns owned per rank (indices stay < 1e9)
    torch.manual_seed(31337)  # identical design on every rank
    b = 5
    vals = torch.rand(b, world) * 0.5
    # global vocab index = rank * VSHARD + local offset (unique across ranks)
    idxs = (
        torch.arange(world)[None, :].expand(b, world) * VSHARD
        + torch.randint(1, VSHARD, (b, world))
    )
    # force an exact value tie across all ranks in an extra row
    vals = torch.cat([vals, torch.full((1, world), 0.9)])
    tie_idxs = (torch.arange(world) * VSHARD + 7)[None, :]
    idxs = torch.cat([idxs, tie_idxs])

    got = helper(vals[:, rank].clone().float(), idxs[:, rank].clone().long())
    assert got.dtype == torch.int64

    # reference: every rank sees the full (vals, idxs) matrix
    allv = [torch.empty(b + 1) for _ in range(world)]
    alli = [torch.empty(b + 1, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(allv, vals[:, rank].clone().float())
    dist.all_gather(alli, idxs[:, rank].clone().long())
    V = torch.stack(allv, dim=-1)  # [rows, world]
    I = torch.stack(alli, dim=-1)
    vmax = V.max(dim=-1, keepdim=True).values
    ref = I.masked_fill(V < vmax, 10**9).min(dim=-1).values

    assert torch.equal(got, ref), f"rank {rank}: {got.tolist()} != {ref.tolist()}"
    # tie row must resolve to the lowest global index
    assert int(got[-1]) == int(idxs[-1].min())


@pytest.mark.parametrize("world", [2, 4])
def test_top_tokens_allreduce_matches_reference(world):
    _run_spawned("top_tokens", world)


def test_top_tokens_env_gate_wiring():
    """Source-level: get_top_tokens must keep the plain pair-gather as the
    default branch, gate the all-reduce argmax behind
    FLASHNEXT_HEAD_ALLREDUCE (same flag as the head gather), and contain no
    host copy."""
    text = LP_SRC.read_text()
    tree = ast.parse(text)
    fwd = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_top_tokens":
            fwd = ast.get_source_segment(text, node)
            break
    assert fwd is not None, "get_top_tokens not found"
    assert 'os.environ.get("FLASHNEXT_HEAD_ALLREDUCE") == "1"' in fwd
    assert "_top_tokens_allreduce_argmax(local_max_vals, global_indices)" in fwd
    # default pair-gather branch preserved
    assert "tensor_model_parallel_all_gather(local_pair, dim=-1)" in fwd
    assert ".cpu()" not in fwd
    # helper exists at module level and uses MAX then MIN all-reduces
    helper_src = _extract_func(LP_SRC, "_top_tokens_allreduce_argmax")
    assert "dist.ReduceOp.MAX" in helper_src
    assert "dist.ReduceOp.MIN" in helper_src
