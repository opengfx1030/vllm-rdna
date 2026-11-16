# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pciescope: PCIe fabric monitor and probe for par1-cs25.

Reads the two Broadcom PEX88096 switches and the GPU links (passive sysfs
collectors, optional switch register reader), measures latency / bandwidth with
the active probe, and exposes everything through a CLI (``python -m pciescope``),
JSONL/Perfetto traces, a TUI and an MCP server.  See
``docs/rdna2/pciescope-2026-11-07.md``.
"""

from pciescope.model import TOOL_VERSION as __version__

__all__ = ["__version__"]
