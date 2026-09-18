# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for gfx10x oneshot all-reduce helpers.

Does not launch the HIP kernel. Guards the T44b abort-record decode, the
wedge-marker path, and that VLLM_RDNA_AR stays default-off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm.distributed.device_communicators import rdna_all_reduce as rdna_ar


def _pack(phase: int, peer: int, spins: int, seq: int) -> int:
    """Match rdna_ar_abort in csrc/rocm/rdna_allreduce.cuh."""
    return (
        1
        | ((phase & 0xF) << 8)
        | ((peer & 0xF) << 12)
        | (((spins >> 10) & 0xFFFF) << 16)
        | ((seq & 0xFFFFFFFF) << 32)
    )


def test_describe_abort_grid_barrier():
    code = _pack(phase=1, peer=0, spins=50 << 10, seq=7)
    msg = rdna_ar.describe_abort(code, rank=2)
    assert "rank 2" in msg
    assert "collective #7" in msg
    assert "~50 ms" in msg
    assert "grid barrier" in msg


def test_describe_abort_peer_flag():
    code = _pack(phase=2, peer=3, spins=12 << 10, seq=9)
    msg = rdna_ar.describe_abort(code, rank=0)
    assert "peer rank 3" in msg
    assert "collective #9" in msg
    assert "posted P2P write from GPU 3" in msg


def test_marker_path_uses_cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    assert rdna_ar.marker_path() == str(tmp_path / "rdna_ar_wedged")


def test_rdna_ar_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_RDNA_AR", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_RDNA_AR == "0"


def test_rdna_ar_check_noop_without_tp():
    rdna_ar._active = False
    rdna_ar.rdna_ar_check()  # must not raise when there is no TP group
    rdna_ar._active = False


def test_check_writes_marker_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))

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
