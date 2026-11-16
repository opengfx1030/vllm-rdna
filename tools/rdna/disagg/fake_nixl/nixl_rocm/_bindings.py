# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-in for ``nixl_rocm._bindings`` (G0a spike)."""


class nixlXferTelemetry:  # noqa: N801 - mirrors the NIXL name
    """Fields read by ``NixlKVConnectorStats.record_transfer``."""

    xferDuration: float = 0.0
    postDuration: float = 0.0
    totalBytes: int = 0
    descCount: int = 0
