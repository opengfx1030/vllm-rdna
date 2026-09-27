# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for gfx10x oneshot all-reduce helpers.

Loads ``rdna_all_reduce.py`` by path so collection does not import
``vllm.distributed`` (heavy). Does not launch the HIP kernel. Guards the
T44b abort-record decode, the wedge-marker path, and that VLLM_RDNA_AR
stays default-off.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "vllm" / "distributed" / "device_communicators" / "rdna_all_reduce.py"


def _load_rdna_ar(monkeypatch: pytest.MonkeyPatch, cache_root: Path):
    """Import rdna_all_reduce.py with stub vllm.logger / platforms / envs."""
    vllm_mod = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    monkeypatch.setattr(logger_mod, "init_logger", logging.getLogger, raising=False)
    platforms_mod = types.ModuleType("vllm.platforms")
    monkeypatch.setattr(
        platforms_mod, "current_platform", types.SimpleNamespace(), raising=False
    )
    envs_mod = types.ModuleType("vllm.envs")
    monkeypatch.setattr(envs_mod, "VLLM_CACHE_ROOT", str(cache_root), raising=False)
    monkeypatch.setattr(vllm_mod, "envs", envs_mod, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_mod)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_mod)

    spec = importlib.util.spec_from_file_location("rdna_all_reduce", _SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pack(phase: int, peer: int, spins: int, seq: int) -> int:
    """Match rdna_ar_abort in csrc/rocm/rdna_allreduce.cuh."""
    return (
        1
        | ((phase & 0xF) << 8)
        | ((peer & 0xF) << 12)
        | (((spins >> 10) & 0xFFFF) << 16)
        | ((seq & 0xFFFFFFFF) << 32)
    )


def test_describe_abort_grid_barrier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = _pack(phase=1, peer=0, spins=50 << 10, seq=7)
    msg = rdna_ar.describe_abort(code, rank=2)
    assert "rank 2" in msg
    assert "collective #7" in msg
    assert "~50 ms" in msg
    assert "grid barrier" in msg


def test_describe_abort_peer_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = _pack(phase=2, peer=3, spins=12 << 10, seq=9)
    msg = rdna_ar.describe_abort(code, rank=0)
    assert "peer rank 3" in msg
    assert "collective #9" in msg
    assert "posted P2P write from GPU 3" in msg


def test_marker_path_uses_cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar.marker_path() == str(tmp_path / "rdna_ar_wedged")


def test_rdna_ar_defaults_off():
    comm = (
        _ROOT / "vllm/distributed/device_communicators/cuda_communicator.py"
    ).read_text()
    envs = (_ROOT / "vllm/envs.py").read_text()
    assert 'os.getenv("VLLM_RDNA_AR", "0")' in comm
    assert 'os.getenv("VLLM_RDNA_AR", "0")' in envs


def test_rdna_ar_check_noop_without_tp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    rdna_ar._active = False
    rdna_ar.rdna_ar_check()  # must not raise when there is no TP group
    rdna_ar._active = False


def test_check_writes_marker_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def rdna_ar_timeout_info(self, handle: int) -> int:
            return _pack(phase=2, peer=1, spins=10 << 10, seq=3)

    inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
    inst.disabled = False
    inst.handle = 0
    inst.rank = 0
    inst.world_size = 4
    inst._ops = _Ops()
    with pytest.raises(RuntimeError, match="rdna_ar wedged"):
        inst.check()
    marker = tmp_path / "rdna_ar_wedged"
    assert marker.is_file()
    assert "peer rank 1" in marker.read_text(encoding="utf-8")
    assert inst.disabled is True


@pytest.mark.parametrize("missing_rank", [0, 2])
def test_incomplete_native_build_falls_back_before_gpu_init(
    monkeypatch, tmp_path, caplog, missing_rank
):
    """A missing timeout op on any rank must prevent all native launches."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    names = (
        "rdna_ar_init",
        "rdna_ar_connect",
        "rdna_ar_can",
        "rdna_ar_all_reduce",
        "rdna_ar_timed_out",
        "rdna_ar_timeout_info",
    )
    native_ops = types.SimpleNamespace(**{name: object() for name in names})
    if missing_rank == 0:
        del native_ops.rdna_ar_timeout_info
    monkeypatch.setattr(rdna_ar.torch.ops, "_rocm_C", native_ops)
    monkeypatch.setitem(
        sys.modules,
        "vllm._rocm_C",
        types.SimpleNamespace(__file__="/old-runtime/vllm/_rocm_C.abi3.so"),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Native/GPU initialization reached before compatibility check")

    ops = types.SimpleNamespace(rdna_ar_init=unexpected)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", ops)
    monkeypatch.setattr(rdna_ar.torch.accelerator, "current_device_index", unexpected)
    monkeypatch.setattr(rdna_ar.dist, "get_rank", lambda **kw: 0)
    monkeypatch.setattr(rdna_ar.dist, "get_world_size", lambda **kw: 4)
    votes = []

    def gather(output, value, **kwargs):
        votes.append(value)
        if missing_rank == 0:
            assert "rdna_ar_timeout_info" in value
            assert "/old-runtime/" in value
        else:
            assert value is None
        output[:] = [None] * 4
        output[missing_rank] = "missing rdna_ar_timeout_info in /old-runtime/"

    monkeypatch.setattr(rdna_ar.dist, "all_gather_object", gather)
    with caplog.at_level(logging.WARNING):
        instance = rdna_ar.RdnaOneShotAllReduce(object(), rdna_ar.torch.device("cuda"))
    assert instance.disabled
    assert instance.handle == -1
    assert len(votes) == 1
    assert "native extension" in caplog.text
    assert "RCCL" in caplog.text
    assert not (tmp_path / "rdna_ar_wedged").exists()


_NATIVE_OPS = (
    "rdna_ar_init",
    "rdna_ar_connect",
    "rdna_ar_can",
    "rdna_ar_all_reduce",
    "rdna_ar_timed_out",
    "rdna_ar_timeout_info",
)


@pytest.mark.parametrize("missing", _NATIVE_OPS)
def test_native_contract_requires_every_operator(monkeypatch, tmp_path, missing):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    namespace = types.SimpleNamespace(**{name: object() for name in _NATIVE_OPS})
    delattr(namespace, missing)
    monkeypatch.setattr(rdna_ar.torch.ops, "_rocm_C", namespace)
    assert missing in rdna_ar.native_extension_error()


@pytest.mark.parametrize("foreign", [False, True])
def test_strict_preflight_checks_native_origin(monkeypatch, tmp_path, capsys, foreign):
    """Exercise the launcher's CLI, not merely a source-string assertion."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    namespace = types.SimpleNamespace(**{name: object() for name in _NATIVE_OPS})
    monkeypatch.setattr(rdna_ar.torch.ops, "_rocm_C", namespace)
    source = tmp_path / "source"
    monkeypatch.setattr(
        sys.modules["vllm"],
        "__file__",
        str(source / "vllm/__init__.py"),
        raising=False,
    )
    location = (tmp_path / "old-runtime" if foreign else source) / "vllm/_rocm_C.so"
    extension = types.ModuleType("vllm._rocm_C")
    extension.__file__ = str(location)
    monkeypatch.setitem(sys.modules, "vllm._rocm_C", extension)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.device_communicators.rdna_all_reduce", rdna_ar
    )
    spec = importlib.util.spec_from_file_location(
        "check_rdna_ar_native", _ROOT / "tools/rdna2/check_rdna_ar_native.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["check", "--source-root", str(source)])
    if foreign:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
        assert "outside" in capsys.readouterr().err
    else:
        cli.main()
        assert "native contract verified" in capsys.readouterr().out
