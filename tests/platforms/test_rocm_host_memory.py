# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the userptr-vs-GTT choice of a ROCm worker's paged host memory.

Loads ``rocm_host_memory.py`` by path so collection does not import
``vllm.platforms.rocm`` (HIP init).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[2] / "vllm" / "platforms" / "rocm_host_memory.py"
)
_SPEC = importlib.util.spec_from_file_location("rocm_host_memory", _SRC)
assert _SPEC is not None and _SPEC.loader is not None
rhm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rhm)

ENV = rhm.PAGED_HOST_MEMORY_ENV
GIB = 1 << 30
PAGE = os.sysconf("SC_PAGE_SIZE")


def test_gtt_set_before_hip_initializes(tmp_path: Path):
    environ: dict[str, str] = {}
    outcome = rhm.use_gtt_for_paged_host_memory(environ, 4242, str(tmp_path))
    assert outcome == "set"
    assert environ == {ENV: "0"}
    assert rhm.paged_host_memory_is_gtt(environ)


def test_user_value_is_kept(tmp_path: Path):
    environ = {ENV: "1"}
    assert rhm.use_gtt_for_paged_host_memory(environ, 4242, str(tmp_path)) == "kept"
    assert environ == {ENV: "1"}
    assert not rhm.paged_host_memory_is_gtt(environ)


def test_too_late_once_the_process_has_a_kfd_context(tmp_path: Path):
    (tmp_path / "4242").mkdir()
    environ: dict[str, str] = {}
    assert rhm.use_gtt_for_paged_host_memory(environ, 4242, str(tmp_path)) == "late"
    assert environ == {}


def _sysfs(tmp_path: Path, limit_gib: float, used_gib: list[float]) -> tuple[str, str]:
    limit = tmp_path / "pages_limit"
    limit.write_text(f"{int(limit_gib * GIB) // PAGE}\n")
    for i, used in enumerate(used_gib):
        dev = tmp_path / f"dev{i}"
        dev.mkdir()
        (dev / "mem_info_gtt_used").write_text(f"{int(used * GIB)}\n")
        card = tmp_path / f"card{i}"
        card.mkdir()
        (card / "device").symlink_to(dev)
    # A second DRM node of the same device must not be counted twice.
    (tmp_path / "card0-alias").mkdir()
    (tmp_path / "card0-alias" / "device").symlink_to(tmp_path / "dev0")
    return str(limit), str(tmp_path / "card*" / "device" / "mem_info_gtt_used")


def test_headroom_fits(tmp_path: Path):
    limit, used = _sysfs(tmp_path, 251.8, [10.0, 5.0])
    # Two TP4 servers with exact 23.84 GiB tables: 190.7 GiB < 236.8 GiB free.
    assert rhm.gtt_headroom_error(int(23.84 * GIB), 8, "table", limit, used) is None


def test_headroom_error_names_the_numbers_and_the_way_out(tmp_path: Path):
    limit, used = _sysfs(tmp_path, 251.8, [128.0, 2.0])
    error = rhm.gtt_headroom_error(32 * GIB, 4, "The pinned PLE table", limit, used)
    assert error is not None
    assert error.startswith(
        "The pinned PLE table needs 128.0 GiB of GTT (4 ranks x 32.00 GiB)"
    )
    assert "only 121.8 GiB is free" in error
    assert "251.8 GiB for the whole host, 130.0 GiB already in GTT" in error
    assert f"{ENV}=1" in error


def test_headroom_check_skipped_without_sysfs(tmp_path: Path):
    missing = str(tmp_path / "nope")
    assert rhm.gtt_headroom_error(GIB, 4, "x", missing, missing) is None
