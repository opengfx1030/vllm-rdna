# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for ROCR_VISIBLE_DEVICES → amdsmi handle remapping.

Leap's fused-MoE JSON is keyed device_name=AMD_Radeon_Pro_V620. amdsmi
enumerates physical GPUs and ignores ROCR, so logical 0 can be a display
card. _amdsmi_index maps through ROCR so get_device_name() hits the V620
file. Does not initialize HIP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm.platforms.rocm import RocmPlatform, amdsmi_index_from_rocr


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


def test_rocm_platform_amdsmi_index_honors_rocr_only(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2,3,4")
    assert RocmPlatform._amdsmi_index(0) == 1
    assert RocmPlatform._amdsmi_index(1) == 2


def test_rocm_platform_amdsmi_index_without_rocr(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    assert RocmPlatform._amdsmi_index(0) == 0


def test_get_device_name_uses_amdsmi_index():
    src = (
        Path(__file__).resolve().parents[2] / "vllm" / "platforms" / "rocm.py"
    ).read_text()
    assert "cls._amdsmi_index(device_id)" in src
    assert "device_name=AMD_Radeon_RX_6700_XT" in src
