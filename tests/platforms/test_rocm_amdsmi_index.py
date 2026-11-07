# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for ROCR_VISIBLE_DEVICES → amdsmi handle remapping.

Loads ``rocm_visible.py`` by path so collection does not import
``vllm.platforms.rocm`` (HIP init). Also checks that ``get_device_name``
uses ``_amdsmi_index``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "vllm" / "platforms" / "rocm_visible.py"
_ROCM = _ROOT / "vllm" / "platforms" / "rocm.py"
_SPEC = importlib.util.spec_from_file_location("rocm_visible", _SRC)
assert _SPEC is not None and _SPEC.loader is not None
rocm_visible = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rocm_visible)
amdsmi_index_from_rocr = rocm_visible.amdsmi_index_from_rocr


def test_amdsmi_index_from_rocr_maps_logical_zero_to_first_visible():
    # 4x V620 at physical 1..4, display card at 0: ROCR=1,2,3,4
    assert amdsmi_index_from_rocr(0, "1,2,3,4") == 1
    assert amdsmi_index_from_rocr(3, "1,2,3,4") == 4


def test_amdsmi_index_from_rocr_identity_without_env():
    assert amdsmi_index_from_rocr(0, None) == 0
    assert amdsmi_index_from_rocr(2, "") == 2
    assert amdsmi_index_from_rocr(2, "   ") == 2


def test_amdsmi_index_from_rocr_out_of_range_keeps_base():
    assert amdsmi_index_from_rocr(9, "1,2") == 9


def test_amdsmi_index_honors_rocr_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2,3,4")
    base = 0
    assert amdsmi_index_from_rocr(base, os.environ.get("ROCR_VISIBLE_DEVICES")) == 1


def test_get_device_name_uses_amdsmi_index():
    src = _ROCM.read_text()
    assert "cls._amdsmi_index(device_id)" in src
    assert "amdsmi_index_from_rocr(base, os.environ.get(" in src
    assert "device_name=AMD_Radeon_RX_6700_XT" in src


_NO_AMDSMI_PROBE = """
import sys

sys.modules["amdsmi"] = None  # `from amdsmi import ...` raises ImportError
import torch

from vllm.platforms import rocm

assert rocm.with_amdsmi_context(lambda: 7)() == 7
assert rocm.amdsmi_get_processor_handles() == []
assert issubclass(rocm.AmdSmiException, Exception)
# The callers' "amdsmi enumerates no GPU" paths.
assert rocm.RocmPlatform.is_fully_connected([0, 1]) is False
assert rocm.RocmPlatform.is_pix_connected([0, 1]) is False
torch.cuda.get_device_name = lambda device_id=None: "AMD Radeon PRO W7800"
assert rocm.RocmPlatform.get_device_name(0) == "AMD Radeon PRO W7800"
print("no-amdsmi ok")
"""


def test_missing_amdsmi_module_acts_like_an_amdsmi_without_gpus():
    """The rocm_sdk venvs ship no amdsmi: the platform must not NameError.

    G4 on the W7800 found ``NameError: name 'amdsmi_init' is not defined`` from
    the first ``get_device_name()`` (engine startup plan, Triton MoE configs).
    Runs in a subprocess so the import happens with ``amdsmi`` blocked.
    """
    pytest.importorskip("torch")
    proc = subprocess.run(
        [sys.executable, "-c", _NO_AMDSMI_PROBE],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "", "PYTHONPATH": str(_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no-amdsmi ok" in proc.stdout
