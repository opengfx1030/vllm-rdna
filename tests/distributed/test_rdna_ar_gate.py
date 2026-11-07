# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the gfx1100 gate of rdna_ar and the group-wide agreement.

The gate in ``cuda_communicator.py`` is ``on_gfx10x() or on_rdna3_hip("ar")``:
gfx10x (V620) as before, gfx1100 (W7800) only with ``VLLM_RDNA3_HIP_OPS`` naming
``ar``. The one-shot protocol is symmetric, so the constructor of
``RdnaOneShotAllReduce`` first shares every rank's GPU architecture, opt-in and
wedge marker over the CPU group, and every rank leaves the same way. The gate
expression is read from ``cuda_communicator.py`` and evaluated against the real
platform predicates with their flags monkeypatched; the group is faked. Nothing
here queries or launches on a GPU.

Importing ``vllm.platforms.rocm`` runs its GCN-arch probe (amdsmi first). On a
machine with idle GPUs run pytest with ``sys.modules["amdsmi"] = None`` so that
nothing is woken.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_DEVCOMM = _ROOT / "vllm" / "distributed" / "device_communicators"
_COMM = _DEVCOMM / "cuda_communicator.py"
_SRC = _DEVCOMM / "rdna_all_reduce.py"


# --- the gate expression, from cuda_communicator.py --------------------------


def _gate_test() -> ast.expr:
    """The condition of the ``if`` that builds RdnaOneShotAllReduce."""
    for node in ast.walk(ast.parse(_COMM.read_text(encoding="utf-8"))):
        if isinstance(node, ast.If) and "on_gfx10x" in ast.unparse(node.test):
            return node.test
    raise AssertionError("the rdna_ar gate is not in cuda_communicator.py")


def _truth(node: ast.expr, fns: dict) -> bool:
    """Evaluate the gate's boolean expression lazily (``or`` short-circuits)."""
    if isinstance(node, ast.BoolOp):
        is_or = isinstance(node.op, ast.Or)
        for value in node.values:
            if _truth(value, fns) is is_or:
                return is_or
        return not is_or
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        return bool(fns[node.func.id](*args))
    raise AssertionError(f"unexpected gate shape: {ast.unparse(node)}")


@pytest.fixture
def set_arch(monkeypatch: pytest.MonkeyPatch):
    """Make the platform flags say which GPU this process runs on."""
    from vllm import envs
    from vllm.platforms import rdna, rocm

    envs.disable_envs_cache()  # VLLM_RDNA3_HIP_OPS is read through vllm.envs

    def _set(arch: str, ops: str = "") -> None:
        monkeypatch.setattr(rocm, "_ON_GFX10X", arch == "gfx1030")
        monkeypatch.setattr(rocm, "_ON_GFX1100", arch == "gfx1100")
        monkeypatch.setattr(rdna, "_ON_RDNA3", arch == "gfx1100")
        monkeypatch.setattr(rdna, "_RDNA3_HIP_FAMILIES", None)  # re-read the env
        monkeypatch.setenv("VLLM_RDNA3_HIP_OPS", ops)

    return _set


@pytest.mark.parametrize(
    "arch, ops, expect",
    [
        ("gfx1030", "", True),
        ("gfx1030", "ar", True),
        # gfx10x short-circuits: the family list is not even parsed there.
        ("gfx1030", "no-such-family", True),
        ("gfx1100", "", False),
        ("gfx1100", "none", False),
        ("gfx1100", "hc,gdn,qsa,se,dense,v1", False),
        ("gfx1100", "ar", True),
        ("gfx1100", " AR , hc ", True),
        ("gfx1100", "all", True),
        ("gfx942", "", False),
        ("gfx942", "ar", False),
    ],
)
def test_gate_truth_table(set_arch, arch: str, ops: str, expect: bool):
    from vllm.platforms.rdna import on_rdna3_hip
    from vllm.platforms.rocm import on_gfx10x

    set_arch(arch, ops)
    fns = {"on_gfx10x": on_gfx10x, "on_rdna3_hip": on_rdna3_hip}
    assert _truth(_gate_test(), fns) is expect


def test_gate_is_widened_not_replaced():
    src = _COMM.read_text(encoding="utf-8")
    assert 'os.getenv("VLLM_RDNA_AR", "0") == "1"' in src  # still opt-in
    assert "from vllm.platforms.rdna import on_rdna3_hip" in src
    want = ast.parse('on_gfx10x() or on_rdna3_hip("ar")', mode="eval").body
    assert ast.unparse(_gate_test()) == ast.unparse(want)


# --- rdna_all_reduce.py by path, with stub vllm modules ----------------------


def _load_rdna_ar(monkeypatch: pytest.MonkeyPatch, cache_root: Path):
    """Import rdna_all_reduce.py with stub vllm.logger / platforms / envs."""
    vllm_mod = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: logging.getLogger(name)  # type: ignore[attr-defined]
    platforms_mod = types.ModuleType("vllm.platforms")
    platforms_mod.current_platform = types.SimpleNamespace()  # type: ignore[attr-defined]
    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_CACHE_ROOT = str(cache_root)  # type: ignore[attr-defined]
    vllm_mod.envs = envs_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_mod)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_mod)

    spec = importlib.util.spec_from_file_location("rdna_all_reduce", _SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[cls.__module__]
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


class _Stop(Exception):
    """Raised by the fake group at the exchange after the one under test."""


class _FakeGroup:
    """The slice of torch.distributed that the start of the constructor uses.

    ``votes`` is what every rank's first exchange yields (the test places this
    rank's own vote in its slot); an exchange past ``allow`` raises _Stop.
    """

    def __init__(self, rank: int, votes: list, allow: int = 1) -> None:
        self.rank = rank
        self.votes = votes
        self.allow = allow
        self.sent: list = []
        self.barriers = 0

    def get_rank(self, group=None) -> int:
        return self.rank

    def get_world_size(self, group=None) -> int:
        return len(self.votes)

    def all_gather_object(self, object_list, obj, group=None) -> None:
        self.sent.append(obj)
        if len(self.sent) > self.allow:
            raise _Stop
        object_list[:] = self.votes

    def barrier(self, group=None) -> None:
        self.barriers += 1


def _agree(
    rdna_ar,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int,
    votes: list,
    arch: str | None,
    allowed=True,
):
    """Run this rank's _agree_arch against a fake group of ``votes``."""
    fake = _FakeGroup(rank, votes)
    monkeypatch.setattr(rdna_ar, "dist", fake)
    monkeypatch.setattr(rdna_ar, "device_arch_class", lambda idx: arch)
    monkeypatch.setattr(
        rdna_ar, "arch_allowed", allowed if callable(allowed) else (lambda a: allowed)
    )
    inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
    inst.rank, inst.world_size, inst.disabled = rank, len(votes), True
    return inst, inst._agree_arch(votes[rank][0], None), fake


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize(
    "name, want",
    [
        ("gfx1030", "gfx10x"),
        ("gfx1030:xnack-", "gfx10x"),
        ("gfx1036", "gfx10x"),
        ("gfx1100", "gfx1100"),
        ("gfx1100:sramecc-:xnack-", "gfx1100"),
        ("GFX1100", "gfx1100"),
        ("gfx1101", None),
        ("gfx1151", None),
        ("gfx942:sramecc+:xnack-", None),
        ("", None),
        (None, None),
    ],
)
def test_arch_class_of(monkeypatch, tmp_path, name, want):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar.arch_class_of(name) == want


def test_device_arch_class_reads_the_ranks_own_device(monkeypatch, tmp_path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    asked: list[int] = []

    def props(idx):
        asked.append(idx)
        return types.SimpleNamespace(gcnArchName={0: "gfx1100", 1: "gfx1030"}[idx])

    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    assert rdna_ar.device_arch_class(0) == "gfx1100"
    assert rdna_ar.device_arch_class(1) == "gfx10x"
    assert asked == [0, 1]


def test_device_arch_class_falls_back_to_the_process_flags(monkeypatch, tmp_path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    def broken(idx):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", broken)
    rocm = types.ModuleType("vllm.platforms.rocm")
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm)
    for gfx10x, gfx1100, want in (
        (True, False, "gfx10x"),
        (False, True, "gfx1100"),
        (False, False, None),
    ):
        rocm.on_gfx10x = lambda v=gfx10x: v  # type: ignore[attr-defined]
        rocm.on_gfx1100 = lambda v=gfx1100: v  # type: ignore[attr-defined]
        assert rdna_ar.device_arch_class(0) == want


def test_arch_allowed(monkeypatch, tmp_path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    asked: list[str] = []
    opted = [False]
    rdna = types.ModuleType("vllm.platforms.rdna")

    def on_rdna3_hip(family):
        asked.append(family)
        return opted[0]

    rdna.on_rdna3_hip = on_rdna3_hip  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.platforms.rdna", rdna)

    assert rdna_ar.arch_allowed("gfx10x") is True  # no opt-in involved
    assert asked == []
    assert rdna_ar.arch_allowed("gfx1100") is False  # default: RCCL
    opted[0] = True
    assert rdna_ar.arch_allowed("gfx1100") is True
    assert asked == ["ar", "ar"]
    assert rdna_ar.arch_allowed(None) is False
    assert rdna_ar.arch_allowed("gfx942") is False


@pytest.mark.parametrize(
    "votes, why",
    [
        ([("gfx10x", True), ("gfx10x", True)], None),
        ([("gfx1100", True), ("gfx1100", True)], None),
        ([("gfx1100", True)] * 8, None),
        ([("gfx1100", True), ("gfx10x", True)], "not all one architecture"),
        ([("gfx10x", True), ("gfx10x", True), ("gfx1100", True)], "not all one"),
        ([("gfx1100", True), ("gfx1100", False)], "not enabled on ranks [1]"),
        ([("gfx1100", False), ("gfx1100", False)], "not enabled on ranks [0, 1]"),
        ([("gfx1100", True), (None, False)], "not supported by rdna_ar"),
        ([(None, False), (None, False)], "not supported by rdna_ar"),
    ],
)
def test_check_arch_votes(monkeypatch, tmp_path, votes, why):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    got = rdna_ar.check_arch_votes(votes)
    if why is None:
        assert got is None
    else:
        assert got is not None and why in got


def test_marker_path_is_per_arch(monkeypatch, tmp_path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar.marker_path() == str(tmp_path / "rdna_ar_wedged")
    assert rdna_ar.marker_path("gfx10x") == str(tmp_path / "rdna_ar_wedged")
    assert rdna_ar.marker_path("gfx1100") == str(tmp_path / "rdna_ar_wedged_gfx1100")


def test_gfx1100_wedge_writes_its_own_marker(monkeypatch, tmp_path):
    """A wedge on W7800s must not keep V620 boots on RCCL (shared cache root)."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = (1 | 0x4) | (2 << 8) | (1 << 12) | (750 << 16) | (3 << 32)  # lost write

    class _Ops:
        def rdna_ar_timeout_info(self, handle: int) -> int:
            return code

    inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
    inst.disabled, inst.handle, inst.rank, inst.world_size = False, 0, 0, 2
    inst._ops = _Ops()
    assert inst.arch_class == "gfx10x"  # class default: old instances keep the file
    inst.arch_class = "gfx1100"
    with pytest.raises(RuntimeError, match="rdna_ar wedged"):
        inst.check()
    assert (tmp_path / "rdna_ar_wedged_gfx1100").is_file()
    assert not (tmp_path / "rdna_ar_wedged").exists()


# --- the first exchange of the constructor ----------------------------------


def test_agree_all_gfx10x_goes_on(monkeypatch, tmp_path, caplog):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    votes = [(0, "gfx10x", True, None), (1, "gfx10x", True, None)]
    inst, got, fake = _agree(rdna_ar, monkeypatch, rank=1, votes=votes, arch="gfx10x")
    assert got == [0, 1]
    assert inst.arch_class == "gfx10x"
    assert fake.sent == [votes[1]]  # what this rank shows its peers
    assert _warnings(caplog) == []


def test_agree_all_gfx1100_goes_on(monkeypatch, tmp_path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    votes = [(0, "gfx1100", True, None), (5, "gfx1100", True, None)]
    inst, got, fake = _agree(rdna_ar, monkeypatch, rank=0, votes=votes, arch="gfx1100")
    assert got == [0, 5]
    assert inst.arch_class == "gfx1100"
    assert fake.sent == [votes[0]]


def test_agree_mixed_group_never_constructs_and_logs_once(
    monkeypatch, tmp_path, caplog
):
    """W7800 + V620: every rank leaves, only rank 0 says why."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    votes = [(0, "gfx1100", True, None), (1, "gfx10x", True, None)]
    for rank, arch in ((0, "gfx1100"), (1, "gfx10x")):
        caplog.clear()
        inst, got, fake = _agree(
            rdna_ar, monkeypatch, rank=rank, votes=votes, arch=arch
        )
        assert got is None and inst.disabled is True
        assert len(fake.sent) == 1 and fake.barriers == 0
        warned = _warnings(caplog)
        if rank == 0:
            assert len(warned) == 1 and "not all one architecture" in warned[0]
        else:
            assert warned == []


def test_agree_needs_the_opt_in_on_every_rank(monkeypatch, tmp_path, caplog):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    votes = [(0, "gfx1100", True, None), (5, "gfx1100", False, None)]
    inst, got, fake = _agree(rdna_ar, monkeypatch, rank=0, votes=votes, arch="gfx1100")
    assert got is None and inst.disabled is True
    (warned,) = _warnings(caplog)
    assert "not enabled on ranks [1]" in warned and "VLLM_RDNA3_HIP_OPS" in warned


def test_agree_unknown_architecture_is_rejected(monkeypatch, tmp_path, caplog):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    votes = [(0, "gfx1100", True, None), (1, None, False, None)]
    _, got, _ = _agree(rdna_ar, monkeypatch, rank=0, votes=votes, arch="gfx1100")
    assert got is None
    assert "not supported by rdna_ar" in _warnings(caplog)[0]


def test_agree_opt_in_query_failure_votes_no(monkeypatch, tmp_path, caplog):
    """A rank must still join the exchange when its opt-in query raises."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    def boom(arch):
        raise ValueError("VLLM_RDNA3_HIP_OPS: unknown families ['arr']")

    votes = [(0, "gfx1100", False, None), (5, "gfx1100", False, None)]
    _, got, fake = _agree(
        rdna_ar, monkeypatch, rank=0, votes=votes, arch="gfx1100", allowed=boom
    )
    assert got is None
    assert fake.sent == [votes[0]]  # sent, with allowed=False
    assert any("opt-in query failed" in w for w in _warnings(caplog))


def test_agree_marker_of_this_rank_keeps_every_rank_on_rccl(
    monkeypatch, tmp_path, caplog
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    (tmp_path / "rdna_ar_wedged_gfx1100").write_text("2026-11-06 lost write\n")
    votes = [(0, "gfx1100", True, "2026-11-06 lost write"), (5, "gfx1100", True, None)]
    inst, got, fake = _agree(rdna_ar, monkeypatch, rank=0, votes=votes, arch="gfx1100")
    assert got is None and inst.disabled is True
    assert fake.sent == [votes[0]]  # the marker text travels in the vote
    (warned,) = _warnings(caplog)
    assert "previous run wedged" in warned and "lost write" in warned
    assert "rdna_ar_wedged_gfx1100" in warned


def test_agree_marker_of_a_peer_keeps_this_rank_on_rccl_too(
    monkeypatch, tmp_path, caplog
):
    """The peer sees a marker (another host or cache root) and this rank does
    not: both must leave, or this rank would wait for flags nobody writes."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    votes = [(0, "gfx1100", True, "lost write"), (5, "gfx1100", True, None)]
    inst, got, fake = _agree(rdna_ar, monkeypatch, rank=1, votes=votes, arch="gfx1100")
    assert got is None and inst.disabled is True
    assert len(fake.sent) == 1 and fake.barriers == 0
    assert _warnings(caplog) == []  # the rank that has the marker explains it


@pytest.mark.parametrize(
    "marker, arch, votes",
    [
        (
            "rdna_ar_wedged",
            "gfx1100",
            [(0, "gfx1100", True, None), (5, "gfx1100", True, None)],
        ),
        (
            "rdna_ar_wedged_gfx1100",
            "gfx10x",
            [(1, "gfx10x", True, None), (2, "gfx10x", True, None)],
        ),
    ],
)
def test_agree_ignores_the_other_architectures_marker(
    monkeypatch, tmp_path, marker, arch, votes
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    (tmp_path / marker).write_text("wedge\n")
    inst, got, _ = _agree(rdna_ar, monkeypatch, rank=0, votes=votes, arch=arch)
    assert got == [v[0] for v in votes]
    assert inst.arch_class == arch


# --- the constructor ---------------------------------------------------------


def _construct(rdna_ar, monkeypatch: pytest.MonkeyPatch, fake: _FakeGroup):
    ops = types.SimpleNamespace()
    monkeypatch.setattr(sys.modules["vllm"], "_custom_ops", ops, raising=False)
    monkeypatch.setattr(rdna_ar, "dist", fake)
    return rdna_ar.RdnaOneShotAllReduce(group=None, device=torch.device("cuda", 0))


def test_constructor_stops_after_the_vote_when_the_group_disagrees(
    monkeypatch, tmp_path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    monkeypatch.setattr(rdna_ar, "device_arch_class", lambda idx: "gfx1100")
    monkeypatch.setattr(rdna_ar, "arch_allowed", lambda arch: True)
    fake = _FakeGroup(0, [(0, "gfx1100", True, None), (1, "gfx10x", True, None)])
    comm = _construct(rdna_ar, monkeypatch, fake)
    assert comm.disabled is True and comm.handle == -1
    assert len(fake.sent) == 1 and fake.barriers == 0
    assert comm not in rdna_ar._live


def test_constructor_goes_on_after_the_vote_when_the_group_agrees(
    monkeypatch, tmp_path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    monkeypatch.setattr(rdna_ar, "device_arch_class", lambda idx: "gfx1100")
    monkeypatch.setattr(rdna_ar, "arch_allowed", lambda arch: True)
    fake = _FakeGroup(
        0, [(0, "gfx1100", True, None), (5, "gfx1100", True, None)], allow=1
    )
    with pytest.raises(_Stop):  # the PIX exchange that follows the vote
        _construct(rdna_ar, monkeypatch, fake)
    assert fake.sent[0] == (0, "gfx1100", True, None)
    assert len(fake.sent) == 2


@pytest.mark.parametrize("world", [1, 9])
def test_constructor_outside_2_to_8_ranks_runs_no_collective(
    monkeypatch, tmp_path, world
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    fake = _FakeGroup(0, [(r, "gfx10x", True, None) for r in range(world)])
    comm = _construct(rdna_ar, monkeypatch, fake)
    assert comm.disabled is True and fake.sent == []


# --- tools/rdna/rdna3/ar_probe.py (CPU only: pure helpers, a world of one) ---

_PROBE = _ROOT / "tools" / "rdna" / "rdna3" / "ar_probe.py"
# Abort record: a record, phase 2, peer 1, 1504 ms measured, collective #7.
_ABORT = 1 | (2 << 8) | (1 << 12) | ((1504 // 16) << 16) | (7 << 32)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("ar_probe", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ar_probe", mod)
    spec.loader.exec_module(mod)
    return mod


class _SoloDist:
    """torch.distributed of a world of one: every agreement is the local value."""

    class ReduceOp:
        MAX = "max"

    @staticmethod
    def all_reduce(tensor, op=None, group=None):
        return None

    @staticmethod
    def barrier(group=None):
        return None


class _Comm:
    """One-rank RdnaOneShotAllReduce stand-in: all_reduce returns its input."""

    disabled = False
    handle = 0
    arch_class = "gfx1100"

    def __init__(self, mode: str = "ok", record: int = 0):
        self.mode = mode
        self._ops = types.SimpleNamespace(
            rdna_ar_timeout_info=lambda h: record, rdna_ar_slow_info=lambda h: 0
        )

    def should_use(self, x):
        return True

    def all_reduce(self, x):
        out = x.clone()
        if self.mode == "flip":  # one wrong bit
            out.view(torch.int16 if out.element_size() == 2 else torch.int32)[0] ^= 1
        elif self.mode == "plus1":  # every element wrong
            out = out + 1
        return out


def _probe_ctx(probe, monkeypatch, tmp_path, comm, *argv):
    monkeypatch.setattr(probe, "dist", _SoloDist)
    args = probe.parse_args(["--dry-run", "--gate-kb", "64", *argv])
    ctx = probe.Ctx(
        args=args,
        rank=0,
        world=1,
        gpu=probe.Gpu(0, True),
        gloo=None,
        rdna_ar=_load_rdna_ar(monkeypatch, tmp_path),
        state={},
    )
    ctx.comms[64] = comm
    return ctx


def test_probe_percentiles(probe):
    xs = [float(i) for i in range(1, 101)]
    assert [probe.percentile(xs, q) for q in (0, 50, 99, 100)] == [1, 50, 99, 100]
    s = probe.summarize(xs[::-1])
    assert (s["n"], s["min_us"], s["p50_us"], s["p99_us"], s["max_us"]) == (
        100,
        1.0,
        50.0,
        99.0,
        100.0,
    )
    with pytest.raises(ValueError):
        probe.percentile([], 50)


def test_probe_payload_reference_and_bit_compare(probe):
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        a = probe.make_payload("rand", 1, 4096, dtype, 7)
        assert torch.equal(a, probe.make_payload("rand", 1, 4096, dtype, 7))
        assert not torch.equal(a, probe.make_payload("rand", 0, 4096, dtype, 7))
        assert bool((a != 0).all())
        ints = probe.make_payload("int", 3, 4096, dtype, 7)
        assert torch.equal(ints.to(torch.int64).to(dtype), ints)  # integer valued
    half = torch.float16
    one = torch.tensor([1.0], dtype=half)
    # 1 + 2**-11 is a tie (to even: 1.0); 1 + 3 * 2**-12 is above it (rounds up).
    tie = probe.reference_sum([one, torch.tensor([2.0**-11], dtype=half)])
    up = probe.reference_sum([one, torch.tensor([3 * 2.0**-12], dtype=half)])
    assert tie.item() == 1.0 and up.item() == 1.0 + 2.0**-10
    zero, neg = (
        torch.tensor([0.0, 1.0], dtype=half),
        torch.tensor([-0.0, 1.0], dtype=half),
    )
    assert bool((zero == neg).all()) and probe.bit_mismatches(zero, neg) == 1
    nan = torch.tensor([float("nan")])
    assert probe.bit_mismatches(nan, nan.clone()) == 0
    assert probe.bit_mismatches(zero, zero.to(torch.bfloat16)) > 0
    assert probe.bit_mismatches(zero, zero[:1]) > 0


@pytest.mark.parametrize("world", [2, 4, 8])
def test_probe_argmax_ties_go_to_the_lowest_rank(probe, world):
    want = probe.argmax_expected(64, world)
    for row, got in enumerate(want):
        vals = [probe.argmax_case(64, world, r)[0][row] for r in range(world)]
        assert got % 1_000_003 == row and vals[got // 1_000_003] == max(vals)
        if row % 4 == 0:
            assert got // 1_000_003 == 0 and len(set(vals)) == 1
        else:
            assert vals.count(max(vals)) == 1
    for r in range(world):  # exact in every dtype the kernel takes
        vals = probe.argmax_case(64, world, r)[0]
        assert torch.tensor(vals, dtype=torch.bfloat16).tolist() == vals


def test_probe_env_problems(probe):
    ok = {"HIP_VISIBLE_DEVICES": "0,5"}
    check = probe.preflight_env_problems
    assert check(ok, 2, False) == [] and check({}, 2, True) == []
    assert "not set" in check({}, 2, False)[0]
    assert "lists 1 devices" in check({"HIP_VISIBLE_DEVICES": "0"}, 2, False)[0]
    assert check({**ok, "ROCR_VISIBLE_DEVICES": "0,1"}, 2, False)
    assert check({**ok, "CUDA_VISIBLE_DEVICES": "1,2"}, 2, False)
    assert check({**ok, "CUDA_VISIBLE_DEVICES": "0,5"}, 2, False) == []


def test_probe_parse_args_rules(probe):
    a = probe.parse_args([])
    assert a.phases == ["exact", "latency", "soak", "gather", "rccl"]
    assert probe.parse_args(["--timeout-test"]).phases[-1] == "timeout"
    for bad in (
        ["--phases", "bogus"],
        ["--dry-run", "--timeout-test"],
        ["--sizes-kb", "64,256"],  # the p50 gate needs 8 KiB measured
        ["--p50-size-kb", "512"],  # above the gate
    ):
        with pytest.raises(SystemExit):
            probe.parse_args(bad)


def test_probe_verdict_gates(probe):
    def result(p50=12.0, **over):
        r = {
            "phases": ["exact", "latency", "soak", "gather", "timeout"],
            "exact": [{"bad": 0}],
            "gather": [{"bad": 0}],
            "soak": {"bad": 0},
            "latency": [{"size_kb": 8, "dtype": "float16", "graph": {"p50_us": p50}}],
            "timeout": {"ok": True},
            "wedge": None,
            "marker": False,
        }
        return {**r, **over}

    args = types.SimpleNamespace(p50_size_kb=8, p50_limit_us=20.0, dry_run=False)

    def verdict(*ranks, **kw):
        return probe.verdict(list(ranks), types.SimpleNamespace(**{**vars(args), **kw}))

    assert verdict(result(), result(19.9))["pass"]
    v = verdict(result(), result(20.1))  # the worst rank decides
    assert v["gates"]["p50"] is False and not v["pass"]
    assert verdict(result(25.0), result(25.0), dry_run=True)["pass"]  # plumbing only
    assert not verdict(result(exact=[{"bad": 1}]), result())["gates"]["bit_exact"]
    assert not verdict(result(soak={"bad": 2}), result())["gates"]["bit_exact"]
    unfinished = result()
    del unfinished["soak"]  # requested, never completed
    assert not verdict(unfinished, result())["gates"]["bit_exact"]
    assert not verdict(result(wedge="x"), result())["gates"]["no_wedge"]
    assert not verdict(result(), result(marker=True))["gates"]["no_wedge"]
    assert not verdict(result(timeout={"ok": False}), result())["gates"]["timeout"]


def test_probe_exact_passes_clean_and_sees_one_wrong_bit(probe, monkeypatch, tmp_path):
    argv = ("--sizes-kb", "8,64", "--dtypes", "float16", "--exact-reps", "2")
    ctx = _probe_ctx(probe, monkeypatch, tmp_path, _Comm(), *argv)
    rows = probe.phase_exact(ctx)
    assert len(rows) == 4 and all(r["bad"] == 0 and r["eligible"] for r in rows)
    ctx = _probe_ctx(probe, monkeypatch, tmp_path, _Comm("flip"), *argv)
    assert [r["bad"] for r in probe.phase_exact(ctx)] == [2, 2, 2, 2]


def test_probe_soak_sees_a_wrong_sum(probe, monkeypatch, tmp_path):
    argv = ("--soak-s", "0.05", "--soak-batch", "2", "--graph-k", "2")
    ctx = _probe_ctx(probe, monkeypatch, tmp_path, _Comm(), *argv)
    assert probe.phase_soak(ctx)["bad"] == 0
    ctx = _probe_ctx(probe, monkeypatch, tmp_path, _Comm("plus1"), *argv)
    res = probe.phase_soak(ctx)
    assert res["bad"] > 0 and res["collectives"] == 4  # stopped after one batch


def test_probe_stops_on_an_abort_record(probe, monkeypatch, tmp_path):
    ctx = _probe_ctx(probe, monkeypatch, tmp_path, _Comm(record=_ABORT))
    with pytest.raises(probe.Stop, match="wedge after exact 8 KiB"):
        probe.phase_exact(ctx)
    assert "peer rank 1 never reached it" in ctx.wedge
    assert "~1504 ms" in ctx.wedge


def test_probe_dry_run_end_to_end(tmp_path):
    """Two CPU ranks through every phase: control flow, JSON and verdict."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "out.json"
    argv = [
        sys.executable,
        str(_PROBE),
        "--dry-run",
        "--out",
        str(out),
        "--sizes-kb",
        "8,64,120",
        "--gate-kb",
        "64",
        "--dtypes",
        "float16,bfloat16",
        "--iters",
        "200",
        "--big-iters",
        "100",
        "--eager-iters",
        "50",
        "--rccl-iters",
        "0",
        "--soak-s",
        "1",
        "--exact-reps",
        "1",
        "--graph-k",
        "4",
        "--soak-batch",
        "3",
    ]
    env = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "HIP_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    procs = [
        subprocess.Popen(
            argv,
            env={**env, "RANK": str(r), "LOCAL_RANK": str(r)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for r in range(2)
    ]
    logs = [p.communicate(timeout=300)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], logs
    result = json.loads(out.read_text())
    assert result["errors"] == [] and result["verdict"]["pass"]
    assert set(result["verdict"]["gates"]) == {"bit_exact", "p50", "no_wedge"}
    rank0 = result["ranks"][0]
    assert all(r["bad"] == 0 for r in rank0["exact"])
    assert {c["size_kb"] for c in rank0["latency"]} == {8, 64, 120}
    assert rank0["soak"]["collectives"] > 0 and rank0["soak"]["bad"] == 0
