# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for gfx10x oneshot all-reduce helpers.

Loads ``rdna_all_reduce.py`` by path so collection does not import
``vllm.distributed`` (heavy). Does not launch the HIP kernel. Guards the
T44b abort-record decode (stalled peer vs lost write), the live-wait watcher,
the wedge-marker path, and that VLLM_RDNA_AR stays default-off.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import regex as re

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "vllm" / "distributed" / "device_communicators" / "rdna_all_reduce.py"


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


_SEEN = 0x4  # RDNA_AR_REC_SEEN: the peer's own flag slot showed the collective
_LIVE = 0x2  # RDNA_AR_REC_LIVE: report[1] wait still in progress


def _pack(phase: int, peer: int, ms: int, seq: int, bits: int = 0) -> int:
    """Match rdna_ar_ms_code in csrc/rocm/rdna_allreduce.cuh."""
    return (
        (1 | bits)
        | ((phase & 0xF) << 8)
        | ((peer & 0xF) << 12)
        | (min(ms // 16, 0xFFFF) << 16)
        | ((seq & 0xFFFFFFFF) << 32)
    )


def test_describe_abort_grid_barrier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = _pack(phase=1, peer=0, ms=120_000, seq=7)
    msg = rdna_ar.describe_abort(code, rank=2)
    assert "rank 2" in msg
    assert "collective #7" in msg
    assert "~120000 ms" in msg
    assert "grid barrier" in msg


def test_describe_abort_long_wait_saturates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """16 ms units: the 600 s default fits; ~17 min and beyond saturate."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    msg = rdna_ar.describe_abort(_pack(phase=2, peer=1, ms=2_000, seq=4), 0)
    assert "~2000 ms" in msg
    msg = rdna_ar.describe_abort(_pack(phase=2, peer=1, ms=600_000, seq=4), 0)
    assert "~600000 ms" in msg
    sat = rdna_ar.describe_abort(_pack(phase=2, peer=1, ms=1 << 40, seq=4), 0)
    assert f"~{0xFFFF * 16} ms" in sat


def test_describe_abort_stalled_peer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No announce seen: the peer never reached the collective -- not a lost
    write (both 2026-11-03/04 wedges were this, reported as a lost write)."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    msg = rdna_ar.describe_abort(_pack(phase=2, peer=3, ms=1_200, seq=9), rank=0)
    assert "peer rank 3 never reached it" in msg
    assert "collective #9" in msg
    assert "lost" not in msg


def test_describe_abort_lost_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    code = _pack(phase=2, peer=3, ms=12_000, seq=9, bits=_SEEN)
    msg = rdna_ar.describe_abort(code, rank=0)
    assert "peer rank 3 had announced it" in msg
    assert "write was lost" in msg
    assert "~12000 ms" in msg


@pytest.mark.parametrize(
    "env, gate, dropped",
    [
        ({}, 64, None),
        ({"VLLM_RDNA_AR_MAX_KB": "32"}, 32, None),
        ({"VLLM_RDNA_AR_ALGO": "oneshot"}, 64, None),
        ({"VLLM_RDNA_AR_MAX_KB": "20480"}, 64, "VLLM_RDNA_AR_MAX_KB=20480"),
        ({"VLLM_RDNA_AR_ALGO": "twoshot"}, 64, "VLLM_RDNA_AR_ALGO=twoshot"),
        ({"VLLM_RDNA_AR_ALGO": "2stage"}, 64, "VLLM_RDNA_AR_ALGO=2stage"),
        (
            {"VLLM_RDNA_AR_MAX_KB": "256", "VLLM_RDNA_AR_ONESHOT_KB": "128"},
            128,
            "VLLM_RDNA_AR_MAX_KB=256",
        ),
    ],
)
def test_twoshot_cannot_be_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env, gate, dropped
):
    """Two-shot was removed: any request for it keeps one-shot up to the gate
    (RCCL above) and says so once."""
    for name in ("VLLM_RDNA_AR_MAX_KB", "VLLM_RDNA_AR_ONESHOT_KB", "VLLM_RDNA_AR_ALGO"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    got_gate, note = rdna_ar.resolve_gate_kb()
    assert got_gate == gate
    if dropped is None:
        assert note is None
    else:
        assert dropped in note
        assert f"one-shot up to {gate} KiB and RCCL above" in note


def test_twoshot_kernel_is_gone():
    src = (_ROOT / "csrc/rocm/rdna_allreduce.cu").read_text()
    hdr = (_ROOT / "csrc/rocm/rdna_allreduce.cuh").read_text()
    assert "twoshot" not in src
    assert "rdna_ar_twoshot" not in hdr
    # The eligibility check caps at the one-shot gate (the hier gate while the
    # hierarchical mode is on), so a larger tensor is never handed to the
    # one-shot kernel either.
    assert "std::min(g.gate_bytes, g.oneshot_max)" in src
    assert "t.numel() * t.element_size() <= rdna_ar_gate(g)" in src


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


def test_rdna_ar_check_noop_without_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar._live == []
    rdna_ar.rdna_ar_check()  # must not raise when no instance is active


def test_rdna_ar_check_covers_every_live_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A wedge on the second instance (e.g. the EP group) fails the step too;
    an already-disabled instance is skipped."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def __init__(self, code: int) -> None:
            self.code = code
            self.calls = 0

        def rdna_ar_timeout_info(self, handle: int) -> int:
            self.calls += 1
            return self.code

    def make(handle: int, code: int, disabled: bool = False):
        inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
        inst.disabled = disabled
        inst.handle = handle
        inst.rank = 2
        inst.world_size = 4
        inst._ops = _Ops(code)
        return inst

    tp, gone, ep = (
        make(0, 0),
        make(1, _pack(2, 3, 120_000, 5), disabled=True),
        make(2, _pack(2, 1, 120_000, 9)),
    )
    rdna_ar._live.extend([tp, gone, ep])
    with pytest.raises(RuntimeError, match="peer rank 1"):
        rdna_ar.rdna_ar_check()
    assert tp._ops.calls == 1 and gone._ops.calls == 0 and ep.disabled
    rdna_ar.rdna_ar_check()  # the wedged instance is disabled now
    assert tp._ops.calls == 2


def _instance(rdna_ar, ops, rank: int = 0, handle: int = 0):
    inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
    inst.disabled = False
    inst.handle = handle
    inst.rank = rank
    inst.world_size = 4
    inst._ops = ops
    return inst


def test_check_lost_write_writes_marker_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def rdna_ar_timeout_info(self, handle: int) -> int:
            return _pack(phase=2, peer=1, ms=12_000, seq=3, bits=_SEEN)

    inst = _instance(rdna_ar, _Ops())
    with pytest.raises(RuntimeError, match="rdna_ar wedged"):
        inst.check()
    marker = tmp_path / "rdna_ar_wedged"
    assert marker.is_file()
    assert "peer rank 1 had announced it" in marker.read_text(encoding="utf-8")
    assert inst.disabled is True


def test_check_stalled_peer_raises_without_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A peer that never arrived within VLLM_RDNA_AR_WAIT_MS still fails the
    step, but the fabric is fine: the next boot must keep the one-shot path."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def rdna_ar_timeout_info(self, handle: int) -> int:
            return _pack(phase=2, peer=1, ms=600_000, seq=3)

    inst = _instance(rdna_ar, _Ops())
    with pytest.raises(RuntimeError, match="peer rank 1 never reached it"):
        inst.check()
    assert not (tmp_path / "rdna_ar_wedged").exists()
    assert inst.disabled is True


def test_check_logs_late_peer_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        slow = _pack(phase=2, peer=3, ms=5_008, seq=11, bits=_LIVE)

        def rdna_ar_slow_info(self, handle: int) -> int:
            return self.slow

        def rdna_ar_timeout_info(self, handle: int) -> int:
            return 0

    ops = _Ops()
    inst = _instance(rdna_ar, ops, rank=1)
    with caplog.at_level(logging.WARNING):
        inst.check()  # a wait still in progress is the watcher's, not a late peer
        assert "late peer" not in caplog.text
        ops.slow = _pack(phase=2, peer=3, ms=5_008, seq=11)
        inst.check()  # a late peer is logged, not fatal
    assert "late peer -- rank 1 waited ~5008 ms" in caplog.text
    assert inst.disabled is False
    assert not (tmp_path / "rdna_ar_wedged").exists()
    msg = rdna_ar.describe_late(_pack(phase=2, peer=3, ms=5_008, seq=11), 1)
    assert msg == "rank 1 waited ~5008 ms for peer rank 3 at collective #11"
    late = rdna_ar.describe_late(_pack(2, 3, 5_008, 11, bits=_SEEN), 1)
    assert late.endswith("(peer rank 3 had announced it; its write landed late)")


def test_watcher_logs_live_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog
):
    """The model thread is blocked while the GPU spins, so a thread logs the
    wait: first at 10 s, then every 30 s, again for a new wait, never for an
    aborted instance."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        slow = 0
        aborted = False

        def rdna_ar_slow_info(self, handle: int) -> int:
            return self.slow

        def rdna_ar_timed_out(self, handle: int) -> bool:
            return self.aborted

    ops = _Ops()
    rdna_ar._live.append(_instance(rdna_ar, ops, rank=2))
    logged: dict = {}

    def lines(ms: int, seq: int = 7, peer: int = 0, bits: int = _LIVE) -> int:
        ops.slow = _pack(phase=2, peer=peer, ms=ms, seq=seq, bits=bits)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            rdna_ar._watch_once(logged)
        return len(caplog.records)

    assert lines(9_008) == 0  # under 10 s
    assert lines(10_000) == 1
    assert "rank 2 has been waiting ~10000 ms for peer rank 0" in caplog.text
    assert "has not reached it yet" in caplog.text
    assert lines(39_008) == 0  # < 30 s since the last line
    assert lines(40_000) == 1
    assert lines(10_000, seq=8) == 1  # a new wait
    assert lines(12_000, seq=8, bits=_LIVE | _SEEN) == 0
    assert lines(40_000, seq=8, bits=_LIVE | _SEEN) == 1
    assert "has announced it but its write has not landed" in caplog.text
    assert lines(60_000, seq=9, bits=0) == 0  # completed: check() logs it
    ops.aborted = True
    assert lines(90_000, seq=10) == 0


def test_watcher_logs_queue_evictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog
):
    """KFD's evicted_ms total grows when evicted queues come back: a step of
    1 s or more is logged with its length, smaller ones and a missing file
    are not."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    rdna_ar._live.append(_instance(rdna_ar, object(), rank=1))
    stats = tmp_path / "evicted_ms"
    state: dict = {"path": str(stats)}

    def lines(total: int) -> int:
        stats.write_text(f"{total}\n")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            rdna_ar._watch_evictions(state)
        return len(caplog.records)

    assert lines(132_548) == 0  # first read: the baseline
    assert lines(132_551) == 0  # a 3 ms eviction
    assert lines(193_551) == 1
    assert "rank 1: the kernel driver had this process's GPU queues" in caplog.text
    assert "evicted for ~61000 ms" in caplog.text
    assert lines(194_550) == 0  # 999 ms
    stats.unlink()  # the process went away: skipped, not raised
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rdna_ar._watch_evictions(state)
        rdna_ar._watch_evictions({"path": None})  # not exposed by this kernel
    assert not caplog.records


# --- hierarchical mode (VLLM_RDNA_AR_HIER) ---------------------------------

_BOX_A = "pci0000:c0/0000:c0:03.1"
_BOX_B = "pci0000:00/0000:00:01.1"


def test_pci_island_key_is_the_root_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    # Resolved sysfs paths of two V620s on par1-cs25 (HIP 1 and HIP 6).
    a = (
        "/sys/devices/pci0000:c0/0000:c0:03.1/0000:c5:00.0/0000:c6:00.0/"
        "0000:c7:00.0/0000:c8:10.0/0000:cc:00.0/0000:cd:00.0/0000:ce:00.0"
    )
    b = (
        "/sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/"
        "0000:03:00.0/0000:04:10.0/0000:08:00.0/0000:09:00.0/0000:0a:00.0"
    )
    assert rdna_ar.pci_island_key(a) == _BOX_A
    assert rdna_ar.pci_island_key(b) == _BOX_B
    assert rdna_ar.pci_island_key("/sys/devices/pci0000:00") is None
    assert rdna_ar.pci_island_key("/dev/null") is None


@pytest.mark.parametrize(
    "keys, want",
    [
        ([_BOX_A] * 4 + [_BOX_B] * 4, [0] * 4 + [1] * 4),
        # island 0 is the one of rank 0, whatever the keys sort like
        ([_BOX_B] * 4 + [_BOX_A] * 4, [0] * 4 + [1] * 4),
        ([_BOX_A, _BOX_B] * 4, [0, 1] * 4),
        ([_BOX_A] * 2 + [_BOX_B] * 2, [0, 0, 1, 1]),
        ([_BOX_A] * 3 + [_BOX_B] * 3, [0, 0, 0, 1, 1, 1]),
        # a single switch, an uneven split, three roots, an unknown key,
        # too few or too many ranks per island: flat
        ([_BOX_A] * 4, None),
        ([_BOX_A] * 3 + [_BOX_B] * 5, None),
        ([_BOX_A] * 3 + [_BOX_B] * 3 + ["x"] * 2, None),
        ([_BOX_A] * 4 + [None] * 4, None),
        ([_BOX_A, _BOX_B], None),
        ([_BOX_A] * 5 + [_BOX_B] * 5, None),
        ([_BOX_A] * 3 + [_BOX_B] * 2, None),
    ],
)
def test_classify_islands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys, want):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar.classify_islands(keys) == want


@pytest.mark.parametrize(
    "spec, world, want",
    [
        ("0,1,2,3|4,5,6,7", 8, [0, 0, 0, 0, 1, 1, 1, 1]),
        ("4,5,6,7|0,1,2,3", 8, [0, 0, 0, 0, 1, 1, 1, 1]),
        ("0,2|1,3", 4, [0, 1, 0, 1]),
        (" 0, 1 | 2, 3 ", 4, [0, 0, 1, 1]),
    ],
)
def test_parse_island_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec, world, want
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    assert rdna_ar.parse_island_override(spec, world) == want


@pytest.mark.parametrize(
    "spec, world",
    [
        ("0,1,2|3,4,5,6,7", 8),  # unequal
        ("0,1,2,3", 4),  # one island
        ("0,1|2,3|4,5", 6),  # three islands
        ("0,1|1,2", 4),  # rank twice
        ("0,1|2,9", 4),  # out of range
        ("0|1", 2),  # islands of one rank
        ("0,1,2,3,4|5,6,7,8,9", 10),  # islands above 4 ranks
        ("a,b|c,d", 4),
        ("0,1|2", 4),  # rank 3 missing
    ],
)
def test_parse_island_override_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec, world
):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="VLLM_RDNA_AR_HIER_ISLANDS"):
        rdna_ar.parse_island_override(spec, world)


_HIER_ENV = (
    "VLLM_RDNA_AR_HIER",
    "VLLM_RDNA_AR_HIER_ISLANDS",
    "VLLM_RDNA_AR_HIER_ALGO",
    "VLLM_RDNA_AR_HIER_MIN_KB",
    "VLLM_RDNA_AR_HIER_MAX_KB",
    "VLLM_RDNA_AR_HIER_X",
    "VLLM_RDNA_AR_HIER_BLOCKS",
)


def _hier_cfg(rdna_ar, monkeypatch: pytest.MonkeyPatch, env: dict):
    for name in _HIER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return rdna_ar.resolve_hier_config()


def test_hier_defaults_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    cfg = _hier_cfg(rdna_ar, monkeypatch, {})
    assert cfg.mode == "auto" and cfg.notes == ()
    assert (cfg.algo, cfg.min_kb, cfg.max_kb, cfg.owners, cfg.blocks) == (
        -1,
        20,
        1024,
        0,
        0,
    )
    assert cfg.islands == ""
    envs = (_ROOT / "vllm/envs.py").read_text()
    assert 'os.getenv("VLLM_RDNA_AR_HIER", "auto")' in envs
    assert 'VLLM_RDNA_AR_HIER: str = "auto"' in envs


@pytest.mark.parametrize(
    "raw, mode",
    [
        ("1", "1"),
        ("AUTO", "auto"),
        (" on ", "1"),
        ("off", "0"),
        ("0", "0"),
        ("", "auto"),
    ],
)
def test_hier_mode_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw, mode):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    cfg = _hier_cfg(rdna_ar, monkeypatch, {"VLLM_RDNA_AR_HIER": raw})
    assert cfg.mode == mode and cfg.notes == ()


def test_hier_knobs_and_bad_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    cfg = _hier_cfg(
        rdna_ar,
        monkeypatch,
        {
            "VLLM_RDNA_AR_HIER": "auto",
            "VLLM_RDNA_AR_HIER_ISLANDS": "0,1|2,3",
            "VLLM_RDNA_AR_HIER_ALGO": "hier",
            "VLLM_RDNA_AR_HIER_MIN_KB": "16",
            "VLLM_RDNA_AR_HIER_MAX_KB": "2048",
            "VLLM_RDNA_AR_HIER_X": "2",
            "VLLM_RDNA_AR_HIER_BLOCKS": "8",
        },
    )
    assert cfg.islands == "0,1|2,3"
    assert (cfg.algo, cfg.min_kb, cfg.max_kb, cfg.owners, cfg.blocks) == (
        2,
        16,
        2048,
        2,
        8,
    )
    assert cfg.notes == ()
    assert _hier_cfg(rdna_ar, monkeypatch, {"VLLM_RDNA_AR_HIER_ALGO": "flat"}).algo == 1
    bad = _hier_cfg(
        rdna_ar,
        monkeypatch,
        {
            "VLLM_RDNA_AR_HIER": "maybe",
            "VLLM_RDNA_AR_HIER_ALGO": "ring",
            "VLLM_RDNA_AR_HIER_MIN_KB": "-3",
            "VLLM_RDNA_AR_HIER_MAX_KB": "big",
            "VLLM_RDNA_AR_HIER_X": "9",
            "VLLM_RDNA_AR_HIER_BLOCKS": "many",
        },
    )
    assert bad.mode == "0"
    assert (bad.algo, bad.min_kb, bad.max_kb, bad.owners, bad.blocks) == (
        -1,
        20,
        1024,
        0,
        0,
    )
    assert len(bad.notes) == 6
    assert all("VLLM_RDNA_AR_HIER" in n for n in bad.notes)


def test_hier_plan_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """What _plan_hier decides before the staging exists: auto by default and only
    for 2 x 4, 0 is off, 1 any two equal islands, an override wins over sysfs."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    keys: dict[str, list] = {}

    def gather(out, obj, group=None):
        out[:] = keys["now"]

    monkeypatch.setattr(rdna_ar.dist, "all_gather_object", gather)
    monkeypatch.setattr(rdna_ar, "device_island_key", lambda idx: "unused")

    def plan(world: int, now: list, env: dict):
        keys["now"] = now
        for name in _HIER_ENV:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        inst = rdna_ar.RdnaOneShotAllReduce.__new__(rdna_ar.RdnaOneShotAllReduce)
        inst.rank, inst.world_size = 1, world
        return inst._plan_hier(0, None)

    two_by_four = ["a"] * 4 + ["b"] * 4
    two_by_two = ["a"] * 2 + ["b"] * 2
    cfg, islands, _ = plan(8, two_by_four, {})
    assert cfg.mode == "auto" and islands == [0, 0, 0, 0, 1, 1, 1, 1]
    cfg, islands, _ = plan(8, two_by_four, {"VLLM_RDNA_AR_HIER": "0"})
    assert cfg.mode == "0" and islands is None
    _, islands, why = plan(4, two_by_two, {})
    assert islands is None and "auto applies to 8 ranks" in why
    _, islands, _ = plan(8, ["a"] * 8, {})
    assert islands is None
    cfg, islands, _ = plan(8, two_by_four, {"VLLM_RDNA_AR_HIER": "auto"})
    assert cfg.mode == "auto" and islands == [0, 0, 0, 0, 1, 1, 1, 1]
    _, islands, why = plan(4, two_by_two, {"VLLM_RDNA_AR_HIER": "auto"})
    assert islands is None and "auto applies to 8 ranks" in why
    _, islands, _ = plan(4, two_by_two, {"VLLM_RDNA_AR_HIER": "1"})
    assert islands == [0, 0, 1, 1]
    _, islands, why = plan(8, ["a"] * 8, {"VLLM_RDNA_AR_HIER": "1"})
    assert islands is None and why
    _, islands, _ = plan(
        4,
        two_by_two,
        {"VLLM_RDNA_AR_HIER": "1", "VLLM_RDNA_AR_HIER_ISLANDS": "0,2|1,3"},
    )
    assert islands == [0, 1, 0, 1]
    _, islands, why = plan(
        4,
        two_by_two,
        {"VLLM_RDNA_AR_HIER": "1", "VLLM_RDNA_AR_HIER_ISLANDS": "0|1,2,3"},
    )
    assert islands is None and "VLLM_RDNA_AR_HIER_ISLANDS" in why


@pytest.mark.parametrize("phase", [3, 4, 5])
def test_describe_hier_phases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase):
    """The hier waits use the abort-record layout with phases 3..5; the
    decode names the step, and keeps the stalled-vs-lost-write split."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    step = {
        3: "island reduce-scatter",
        4: "cross-root exchange",
        5: "island all-gather",
    }[phase]
    stalled = rdna_ar.describe_abort(_pack(phase, 5, 1_200, 9), rank=1)
    assert f"collective #9 ({step})" in stalled
    assert "peer rank 5 never reached it" in stalled and "lost" not in stalled
    lost = rdna_ar.describe_abort(_pack(phase, 5, 12_000, 9, bits=_SEEN), rank=1)
    assert f"({step})" in lost and "write was lost" in lost
    late = rdna_ar.describe_late(_pack(phase, 5, 5_008, 11), 1)
    assert late == (
        f"rank 1 waited ~5008 ms for peer rank 5 at collective #11 ({step})"
    )
    live = rdna_ar.describe_wait(_pack(phase, 5, 10_000, 11, bits=_LIVE), 1)
    assert f"collective #11 ({step})" in live
    # phases 1 and 2 keep their wording
    plain = rdna_ar.describe_abort(_pack(2, 5, 1_200, 9), rank=1)
    assert "collective #9:" in plain


@pytest.mark.parametrize(
    "phase, marker",
    [(1, False), (2, True), (3, True), (4, True), (5, True)],
)
def test_check_marker_for_lost_write_in_any_flag_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase, marker
):
    """A lost write in the reduce-scatter, cross-root or all-gather wait is
    a wedge like one in the flat exchange: the next boot stays on RCCL."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)

    class _Ops:
        def rdna_ar_timeout_info(self, handle: int) -> int:
            return _pack(phase, 1, 12_000, 3, bits=_SEEN)

    inst = _instance(rdna_ar, _Ops())
    with pytest.raises(RuntimeError, match="rdna_ar wedged"):
        inst.check()
    assert (tmp_path / "rdna_ar_wedged").exists() is marker
    assert inst.disabled is True


def test_hier_kernel_sources_are_wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The ops the python side calls exist in the bindings, and the python
    decode uses the kernel's constants."""
    rdna_ar = _load_rdna_ar(monkeypatch, tmp_path)
    hdr = (_ROOT / "csrc/rocm/rdna_allreduce_hier.cuh").read_text()
    src = (_ROOT / "csrc/rocm/rdna_allreduce.cu").read_text()
    bind = (_ROOT / "csrc/rocm/torch_bindings.cpp").read_text()
    ops = (_ROOT / "vllm/_custom_ops.py").read_text()
    for name in ("rdna_ar_hier_init", "rdna_ar_all_reduce_algo"):
        assert name in src and name in bind and name in ops, name

    def define(name: str) -> int:
        found = re.search(rf"#define {name} (\d+)", hdr)
        assert found is not None, name
        return int(found.group(1))

    assert define("RDNA_AR_ALGO_FLAT") == rdna_ar.ALGO_FLAT
    assert define("RDNA_AR_ALGO_FLAT_ISL") == rdna_ar.ALGO_FLAT_ISL
    assert define("RDNA_AR_ALGO_HIER") == rdna_ar.ALGO_HIER
    assert define("RDNA_AR_MAX_G") == rdna_ar._HIER_MAX_ISLAND
    assert sorted(rdna_ar._PH_STEPS) == [
        define("RDNA_AR_PH_RS"),
        define("RDNA_AR_PH_XR"),
        define("RDNA_AR_PH_AG"),
    ]
    assert "twoshot" not in hdr
