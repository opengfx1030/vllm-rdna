# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Visible-device helpers that never touch a GPU.

Kept import-light so CPU tests can load it without initializing HIP.
``amdsmi_index_from_rocr`` is ported from leapdragon/vllm-rdna2-qwen (Aron Hsiao).
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

# KFD topology: one directory per node (node 0 is usually the CPU), static files.
KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def amdsmi_index_from_rocr(base: int, rocr: str | None) -> int:
    """Map a torch-visible ordinal through ROCR_VISIBLE_DEVICES.

    amdsmi enumerates every physical GPU and ignores ROCR. Leap measured
    ROCR=1,2,3,4 with a display card at physical 0 looking up the fused-MoE
    JSON as AMD_Radeon_RX_6700_XT instead of AMD_Radeon_Pro_V620.
    """
    if not rocr:
        return base
    ids = [int(x) for x in rocr.split(",") if x.strip()]
    return ids[base] if 0 <= base < len(ids) else base


def gfx_name(target_version: int) -> str | None:
    """The gfx name of a KFD ``gfx_target_version`` (major*10000 + minor*100 + step).

    100300 is gfx1030, 110000 gfx1100, 90010 gfx90a and 90402 gfx942. 0 (a CPU node)
    and values a gfx name cannot spell give None.
    """
    major = target_version // 10000
    minor = target_version // 100 % 100
    step = target_version % 100
    if major == 0 or minor > 15 or step > 15:
        return None
    return f"gfx{major}{minor:x}{step:x}"


def kfd_gpu_archs(nodes: Path = KFD_NODES) -> list[str]:
    """The gfx name of every GPU node of the KFD topology, in node order.

    Node order is the order HIP numbers its devices in. The files are static data of
    the loaded driver: reading them wakes no GPU and creates no HIP context. A host
    without KFD gives an empty list.
    """
    found: list[tuple[int, str]] = []
    try:
        entries = list(nodes.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            lines = (entry / "properties").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            key, _, value = line.partition(" ")
            if key == "gfx_target_version" and value.strip().isdecimal():
                name = gfx_name(int(value))
                if name:
                    found.append((int(entry.name), name))
                break
    return [name for _, name in sorted(found)]


def _device_indices(value: str | None) -> list[int] | None:
    """The indices of a visible-device list, None when no list is set.

    Raises ValueError for anything but indices (a UUID, -1).
    """
    if value is None or not value.strip():
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not all(token.isdecimal() for token in tokens):
        raise ValueError(f"not a list of device indices: {value!r}")
    return [int(token) for token in tokens]


def first_visible_gpu_arch(
    archs: Sequence[str], hip: str | None, rocr: str | None
) -> str | None:
    """The gfx name of the first GPU a process with these device lists can use.

    ``rocr`` (ROCR_VISIBLE_DEVICES) filters the machine's GPUs and ``hip``
    (HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES) indexes what is left. None when a
    list holds anything but indices or points outside the GPUs.
    """
    try:
        pool = list(archs)
        rocr_ids = _device_indices(rocr)
        if rocr_ids is not None:
            pool = [pool[i] for i in rocr_ids]
        hip_ids = _device_indices(hip)
        return pool[hip_ids[0] if hip_ids is not None else 0]
    except (ValueError, IndexError):
        return None


def visible_gcn_arch(
    environ: Mapping[str, str] = os.environ, nodes: Path = KFD_NODES
) -> str | None:
    """The gfx name of the first GPU this process may use, read without touching one.

    The answer of the KFD topology files alone: no amdsmi (it lists every GPU of the
    machine, ignores the visible-device lists and powers up idle cards) and no HIP
    context. None when the files cannot say.
    """
    archs = kfd_gpu_archs(nodes)
    if len(set(archs)) == 1:
        return archs[0]  # every GPU of the machine has this arch, whichever are visible
    hip = environ.get("HIP_VISIBLE_DEVICES") or environ.get("CUDA_VISIBLE_DEVICES")
    return first_visible_gpu_arch(archs, hip, environ.get("ROCR_VISIBLE_DEVICES"))
