# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Active measurement engine of pciescope: latency, bandwidth, effectiveness.

Wraps the C++ ``p2p_probe`` binary (``tools/rdna/p2p_probe``), which measures on
the GPUs: host to device and back (pinned and pageable), GPU to GPU (kernel
stores, kernel loads, SDMA), latency curves, bidirectional and concurrent flows,
fan-in and all-to-all. The Python side plans a suite, refuses GPUs whose lease
is not held, runs the binary under the box guards (junction temperature, time
limit, noisy box, KFD evictions), parses its output into ``Measurement`` records
and compares them with the theoretical rate of the PCIe path and with the
numbers of ``docs/rdna2/pcie-iommu-audit-2026-11-06.md``.

Entry points: ``run_suite`` (what ``pciescope measure``, the MCP tool and the TUI
call), ``list_suites``, ``check_leases``, ``add_cli_arguments`` and ``cli_run``
(the ``measure`` verb), ``insitu.observe`` (passive link utilisation) and
``build`` (the fat ``gfx1030`` + ``gfx1100`` binary under the build lock).
GPU work happens only on GPUs whose lease the caller holds.
"""

from __future__ import annotations

from . import insitu
from .api import (
    EXIT_ABORTED,
    EXIT_FAILED,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REFUSED,
    MeasureResult,
    add_cli_arguments,
    check_leases,
    cli_run,
    list_suites,
    run_suite,
)
from .binary import BinaryStatus, BuildResult, binary_status, build

__all__ = [
    "EXIT_ABORTED",
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "EXIT_REFUSED",
    "BinaryStatus",
    "BuildResult",
    "MeasureResult",
    "add_cli_arguments",
    "binary_status",
    "build",
    "check_leases",
    "cli_run",
    "insitu",
    "list_suites",
    "run_suite",
]
