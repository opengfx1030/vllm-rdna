# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bus/entity helpers of the probe modules (pure, no I/O).

A GPU is identified by the 2-hex-digit PCI bus of its function 0 on the box
(``ce``, ``0a``, ...); the shared entity id of the pciescope data contract is
``gpu:<bus>`` and ``host`` names the host side of a host<->device transfer
(``pciescope.model.HOST`` / ``gpu_entity``, re-exported here).
"""

from __future__ import annotations

from pciescope.model import HOST, bdf_bus, full_bdf, gpu_entity

__all__ = ["HOST", "bdf", "bus_of_entity", "gpu_entity", "normalize_bus"]

_HEX = frozenset("0123456789abcdef")


def normalize_bus(token: str) -> str:
    """Return the lower-case 2-digit bus of ``token``.

    Accepts ``0A``, ``gpu:0a``, ``0a:00.0`` and ``0000:0a:00.0``.

    Raises:
        ValueError: when ``token`` is none of those forms.

    """
    s = token.strip().lower()
    if s.startswith("gpu:"):
        s = s[4:]
    if len(s) == 2 and set(s) <= _HEX:
        return s
    try:
        return bdf_bus(s)
    except ValueError:
        raise ValueError(f"not a PCI bus id: {token!r}") from None


def bus_of_entity(entity: str) -> str | None:
    """Return the bus of a ``gpu:<bus>`` entity id, or None for other kinds."""
    if not entity.startswith("gpu:"):
        return None
    return normalize_bus(entity)


def bdf(bus: str) -> str:
    """Return the sysfs address ``0000:<bus>:00.0`` of a GPU function 0."""
    return full_bdf(f"{normalize_bus(bus)}:00.0")
