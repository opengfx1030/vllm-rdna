# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record the sysfs/procfs files that discovery and the collectors read.

Developer verb (``python -m pciescope capture OUTDIR [--tar FILE]``): runs
``discover`` and the passive collectors through ``RecordingSysfs`` so that every
file they read is mirrored under ``OUTDIR/sys`` (and ``OUTDIR/proc`` for the KFD
process names).  The result replays offline: ``discover(Sysfs(OUTDIR/"sys"))``
gives the same topology, which is what the unit tests use
(``tests/tools/rdna/pciescope/fixtures/sysfs_par1_cs25.tar.gz``).

GPU ``unique_id`` values are replaced by a one-way hash of the same width before
anything is written to a tarball.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import tarfile
from pathlib import Path
from typing import Any

from pciescope import collectors, topology
from pciescope.collectors import amdgpu
from pciescope.sysfs import RecordingSysfs


def _hash64(value: int) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16)


def redact_ids(out: str | os.PathLike[str]) -> int:
    """Replace GPU ``unique_id`` values (sysfs file and KFD property); returns count."""
    n = 0
    for dirpath, _dirs, files in os.walk(out):
        for name in files:
            path = Path(dirpath) / name
            if name == "unique_id":
                raw = path.read_text().strip()
                try:
                    new = _hash64(int(raw, 16))
                except ValueError:
                    continue
                path.write_text(f"{new:016x}\n")
                n += 1
            elif name == "properties" and "topology/nodes" in dirpath:
                lines = path.read_text().splitlines()
                changed = False
                for i, line in enumerate(lines):
                    key, _, val = line.partition(" ")
                    if key == "unique_id" and val.strip().isdigit() and int(val):
                        lines[i] = f"unique_id {_hash64(int(val) & (2**64 - 1))}"
                        changed = True
                if changed:
                    path.write_text("\n".join(lines) + "\n")
                    n += 1
    return n


def capture(
    out: str | os.PathLike[str],
    *,
    sys_root: str = "/sys",
    proc_root: str = "/proc",
    host: str | None = None,
    rounds: int = 2,
) -> dict[str, Any]:
    """Record discovery plus ``rounds`` passes of the passive collectors in ``out``."""
    out = Path(out)
    sfs = RecordingSysfs(sys_root, out / "sys")
    procfs = RecordingSysfs(proc_root, out / "proc")
    topo = topology.discover(sfs, host=host)
    res = collectors.build(
        topo,
        [n for n in collectors.DEFAULT_PASSIVE if n != "plx"],  # sysfs readers only
        sysfs=sfs,
        options={"kfd": {"procfs": procfs}},
    )
    for _ in range(rounds):
        for c in res.collectors:
            c.collect()
    for g in topo.gpu_list():  # the hwmon fallback of the amdgpu collector / probe
        amdgpu.read_temps_c(sfs, g)
        if g.hwmon:
            sfs.read_int(f"{g.hwmon}/power1_average")
    redacted = redact_ids(out)
    files = sum(len(f) for _d, _s, f in os.walk(out))
    return {
        "host": topo.host,
        "nodes": len(topo.nodes),
        "gpus": len(topo.gpus),
        "collectors": [c.name for c in res.collectors],
        "skipped": dict(res.skipped),
        "files": files,
        "redacted_ids": redacted,
    }


def write_tarball(src: str | os.PathLike[str], dest: str | os.PathLike[str]) -> int:
    """Deterministic ``.tar.gz`` of ``src`` (sorted, uid/gid 0, mtime 0); its size."""
    src = Path(src)
    with (
        open(dest, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf,
    ):
        for dirpath, dirs, files in os.walk(src):
            dirs.sort()
            for name in sorted(dirs + files):
                full = os.path.join(dirpath, name)
                ti = tf.gettarinfo(full, os.path.relpath(full, src))
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mtime = 0
                if ti.isdir():
                    ti.mode = 0o755
                elif ti.isreg():
                    ti.mode = 0o644
                if ti.isreg():
                    with open(full, "rb") as fh:
                        tf.addfile(ti, fh)
                else:
                    tf.addfile(ti)
    return os.path.getsize(dest)
