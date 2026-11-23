# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm platform only"),
]


@pytest.mark.parametrize(
    ("amdsmi_count", "native_count", "expected"),
    [(0, 4, 4), (-1, 4, 4), (2, 4, 2), (0, 0, 0)],
)
def test_device_count_uses_native_discovery_when_amdsmi_finds_no_gpus(
    monkeypatch, amdsmi_count, native_count, expected
):
    from vllm.platforms import rocm

    monkeypatch.setattr(torch.cuda, "_device_count_amdsmi", lambda: amdsmi_count)
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: native_count)
    rocm._rocm_device_count_stateless.cache_clear()
    try:
        assert rocm.RocmPlatform.device_count() == expected
    finally:
        rocm._rocm_device_count_stateless.cache_clear()
