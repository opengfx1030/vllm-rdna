# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fake platforms for the gfx10xx capability-alias tests.

ROCm derives the device capability from the GCN arch
(``rocm.py::_capability_from_gcn_arch``), so a gfx1030 (RDNA2) card reports
10.3. ``is_device_capability_family(100)``, ``has_device_capability(100)`` and
``is_device_capability(103)`` are therefore True on it although they are meant
for NVIDIA Blackwell, and any such check that is not behind
``current_platform.is_cuda()`` takes the Blackwell branch on gfx1030.

The ``test_rocm_capability_alias_*`` tests patch the process platform (no GPU is
touched): ``gfx1030`` is a ROCm platform reporting (10, 3), ``sm100`` a CUDA
B200 reporting (10, 0). Each site is checked on both, so a guard is shown to
remove the Blackwell branch on gfx1030 and to leave it alone on CUDA.
"""

import pytest

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.fixture
def patch_platform(monkeypatch):
    """Return ``patch(cuda, capability)``, which makes the process platform
    look like a CUDA (or, if ``cuda`` is False, ROCm) one with that capability."""

    def patch(cuda: bool, capability: tuple[int, int]) -> None:
        cap = DeviceCapability(*capability)
        # has_device_capability() & co. are classmethods and read the
        # capability through ``cls``, so it has to be patched on the class.
        monkeypatch.setattr(
            type(current_platform),
            "get_device_capability",
            classmethod(lambda cls, device_id=0: cap),
        )
        monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
        monkeypatch.setattr(current_platform, "is_rocm", lambda: not cuda)
        monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
        # A plain function that unified_attention() imports inside the call.
        monkeypatch.setattr("vllm.platforms.rocm.on_gfx10x", lambda: not cuda)

    return patch


@pytest.fixture
def gfx1030(patch_platform):
    """A ROCm gfx1030 (RDNA2, Radeon Pro V620): capability (10, 3)."""
    patch_platform(False, (10, 3))


@pytest.fixture
def sm100(patch_platform):
    """A CUDA B200: capability (10, 0)."""
    patch_platform(True, (10, 0))
