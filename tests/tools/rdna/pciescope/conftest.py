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


class FakeProbe:
    """Stand-in for ``pciescope.probe``: records how the verbs call it.

    ``cli_run`` prints one JSON document and returns 2 for owner ``refuse``.
    """

    def __init__(self) -> None:
        import types

        self.calls: list[tuple[object, object]] = []
        self.module = types.ModuleType("pciescope.probe")
        self.module.add_cli_arguments = self.add_cli_arguments  # type: ignore[attr-defined]
        self.module.cli_run = self.cli_run  # type: ignore[attr-defined]

    @staticmethod
    def add_cli_arguments(ap) -> None:
        ap.add_argument("--suite", default="quick")
        ap.add_argument("--gpus", nargs="+")
        ap.add_argument("--owner", default=None)
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--force-noisy", action="store_true")
        ap.add_argument("--max-seconds", type=float, default=None)
        ap.add_argument("--repeats", type=int, default=None)
        ap.add_argument("--wait-quiet", type=float, default=0.0)
        ap.add_argument("--list-suites", action="store_true")
        ap.add_argument("--quiet", action="store_true")
        ap.add_argument("--in-situ", type=float, default=None)
        ap.add_argument("--interval", type=float, default=1.0)

    def cli_run(self, args, *, topo, session=None, emit=print) -> int:
        import json

        self.calls.append((args, session))
        emit(json.dumps({"status": "x", "gpus": args.gpus, "dry_run": args.dry_run}))
        return 2 if args.owner == "refuse" else 0


@pytest.fixture()
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> FakeProbe:
    """``pciescope.probe`` replaced by ``FakeProbe`` for the CLI and the MCP tool."""
    import pciescope
    from pciescope import cli

    fake = FakeProbe()
    monkeypatch.setitem(sys.modules, "pciescope.probe", fake.module)
    monkeypatch.setattr(pciescope, "probe", fake.module, raising=False)
    monkeypatch.setattr(cli, "_PROBE", [(fake.module, "")])
    return fake


@pytest.fixture()
def no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pciescope.probe`` not importable, whatever is installed."""
    import pciescope
    from pciescope import cli

    monkeypatch.delattr(pciescope, "probe", raising=False)
    monkeypatch.setitem(sys.modules, "pciescope.probe", None)
    monkeypatch.setattr(cli, "_PROBE", [(None, "ImportError: probe not installed")])
