# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a ROCm worker's paged host memory lives: KFD userptr or GTT.

ROCr (libhsakmt) allocates paged host memory (hipHostMalloc, torch's
pin_memory=True) on dGPUs as a KFD userptr BO over anonymous pages by default
(HSA_USERPTR_FOR_PAGED_MEM=1). If host memory compaction or reclaim moves one
of those pages, KFD's MMU notifier evicts every GPU queue of the process
until the whole BO has been faulted back in. On par1-cs25, Flash-Next's
pinned PLE table (24 GiB per TP rank) kept a serving instance's queues
evicted for 8-125 s each time a neighbor server booted, and that wedged
rdna_ar (docs/rdna2/rdna-ar-wedge-2026-11-04.md).

With the variable at 0, ROCr allocates paged host memory as GTT BOs instead.
TTM pages are off the LRU, so neither compaction nor reclaim moves them. They
count against TTM's pages_limit (half of RAM by default), which every GPU
process on the host shares.

Kept import-light (stdlib only) so CPU tests can load it without
initializing HIP.
"""

import glob
import os
from collections.abc import MutableMapping

PAGED_HOST_MEMORY_ENV = "HSA_USERPTR_FOR_PAGED_MEM"
KFD_PROC_DIR = "/sys/class/kfd/kfd/proc"
TTM_PAGES_LIMIT = "/sys/module/ttm/parameters/pages_limit"
GTT_USED_GLOB = "/sys/class/drm/card*/device/mem_info_gtt_used"


def use_gtt_for_paged_host_memory(
    environ: MutableMapping[str, str] = os.environ,
    pid: int | None = None,
    kfd_proc_dir: str = KFD_PROC_DIR,
) -> str:
    """Make this process allocate paged host memory as GTT BOs.

    ROCr reads the variable once, when it opens /dev/kfd, so this has to run
    before HIP initializes in the process. Returns "set"; "kept" when the
    environment already has a value (the user's choice wins); or "late" when
    the process already has a KFD context and the change could not apply.
    """
    if PAGED_HOST_MEMORY_ENV in environ:
        return "kept"
    if os.path.exists(os.path.join(kfd_proc_dir, str(pid or os.getpid()))):
        return "late"
    environ[PAGED_HOST_MEMORY_ENV] = "0"
    return "set"


def paged_host_memory_is_gtt(environ: MutableMapping[str, str] = os.environ) -> bool:
    return environ.get(PAGED_HOST_MEMORY_ENV) == "0"


def gtt_headroom_error(
    nbytes: int,
    ranks: int,
    what: str,
    pages_limit_path: str = TTM_PAGES_LIMIT,
    gtt_used_glob: str = GTT_USED_GLOB,
) -> str | None:
    """Return why `ranks` GTT allocations of `nbytes` cannot fit, or None.

    The bound is TTM's pages_limit, which covers the whole host, minus the
    GTT that every amdgpu device already holds. If sysfs cannot be read, the
    check is skipped and None is returned.
    """
    try:
        with open(pages_limit_path) as f:
            limit = int(f.read()) * os.sysconf("SC_PAGE_SIZE")
        used = 0
        for path in {os.path.realpath(p) for p in glob.glob(gtt_used_glob)}:
            with open(path) as f:
                used += int(f.read())
    except (OSError, ValueError):
        return None
    need = nbytes * ranks
    free = limit - used
    if need <= free:
        return None
    gib = 1 << 30
    return (
        f"{what} needs {need / gib:.1f} GiB of GTT ({ranks} ranks x "
        f"{nbytes / gib:.2f} GiB), but only {free / gib:.1f} GiB is free under "
        f"TTM's pages_limit ({limit / gib:.1f} GiB for the whole host, "
        f"{used / gib:.1f} GiB already in GTT). Stop another server, or export "
        f"{PAGED_HOST_MEMORY_ENV}=1 to keep userptr host memory, which host "
        f"memory compaction can evict."
    )
