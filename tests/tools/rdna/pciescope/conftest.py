# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for the pciescope tests.

``fixtures/sysfs_par1_cs25.tar.gz`` is a read-only capture of the real fabric of
par1-cs25 (``python -m pciescope capture``): ``sys/`` holds every sysfs file the
topology walk and the passive collectors read, ``proc/`` the ``comm`` of the KFD
processes.  It is extracted once per test session.  Run these tests with
``--confcutdir=tests/tools/rdna/pciescope`` (the repository root conftest imports
vLLM, which these CPU-only tests do not need)::

    python -m pytest --confcutdir=tests/tools/rdna/pciescope tests/tools/rdna/pciescope

Fixtures: ``fx_root`` (extracted directory), ``fx_sysfs`` / ``fx_procfs``
(``pciescope.sysfs.Sysfs`` over ``sys/`` and ``proc/``) and ``fx_topo`` (the
``Topology`` that ``discover(fx_sysfs, host="par1-cs25")`` returns).
"""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_RDNA = _HERE.parents[3] / "tools" / "rdna"
if str(_RDNA) not in sys.path:
    sys.path.insert(0, str(_RDNA))

FIXTURE_TAR = _HERE / "fixtures" / "sysfs_par1_cs25.tar.gz"


@pytest.fixture(scope="session")
def fx_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("par1_cs25")
    with tarfile.open(FIXTURE_TAR) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(root, filter="data")
        else:  # pragma: no cover - interpreters older than 3.11.4
            tf.extractall(root)
    return root


@pytest.fixture(scope="session")
def fx_sysfs(fx_root: Path):
    from pciescope.sysfs import Sysfs

    return Sysfs(fx_root / "sys")


@pytest.fixture(scope="session")
def fx_procfs(fx_root: Path):
    from pciescope.sysfs import Sysfs

    return Sysfs(fx_root / "proc")


@pytest.fixture(scope="session")
def fx_topo(fx_sysfs):
    from pciescope.topology import discover

    return discover(fx_sysfs, host="par1-cs25")
