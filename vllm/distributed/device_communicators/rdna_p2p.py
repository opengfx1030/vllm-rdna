# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCIe P2P / PIX topology helpers for gfx10x all-reduce.

amdsmi on a same-switch V620 mesh reports type PCIE, hops 2 (toolbox
validated 4x mesh). NCCL_P2P_LEVEL names follow RCCL: PIX is one PCI
switch, PXB multiple bridges, PHB through a CPU root, SYS anything.

Two PEX88096 boards without a cascade are two USPs (PHB), not one PIX
domain. XGMI 1-hop always qualifies at PIX or looser.

These helpers are diagnostic. They do not enable rdna_ar; that stays
VLLM_RDNA_AR=1 (opt-in; when enabled, eligible tensors dispatch ahead of
CUSTOM).
"""

from __future__ import annotations

import os

# amdsmi_topo_get_link_type()["type"]
AMDSMI_LINK_TYPE_PCIE = 1
AMDSMI_LINK_TYPE_XGMI = 2

# Max PCIe hops treated as in-level. Numeric env values follow the NCCL
# user-guide mapping (PIX=1, PXB=2, PHB=3, SYS=4), not PATH_* enums.
_P2P_LEVEL_MAX_PCIE_HOPS: dict[str, int | None] = {
    "loc": 0,
    "0": 0,
    "nvl": None,  # XGMI/NVLink only; PCIe never qualifies
    "pix": 2,
    "1": 2,
    "pxb": 3,
    "2": 3,
    "phb": 4,
    "3": 4,
    "sys": 99,
    "4": 99,
}


def p2p_level_from_env(raw: str | None = None) -> str:
    """Return the canonical NCCL_P2P_LEVEL name (default pxb)."""
    if raw is None:
        raw = os.getenv("NCCL_P2P_LEVEL")
    if raw is None or not str(raw).strip():
        return "pxb"
    key = str(raw).strip().lower()
    if key in _P2P_LEVEL_MAX_PCIE_HOPS:
        if key in ("0", "loc"):
            return "loc"
        if key in ("nvl",):
            return "nvl"
        if key in ("1", "pix"):
            return "pix"
        if key in ("2", "pxb"):
            return "pxb"
        if key in ("3", "phb"):
            return "phb"
        if key in ("4", "sys"):
            return "sys"
        return key
    return "pxb"


def max_pcie_hops_for_level(level: str) -> int | None:
    """PCIe hop ceiling for a level. None means PCIe is never in-level."""
    return _P2P_LEVEL_MAX_PCIE_HOPS.get(p2p_level_from_env(level), 2)


def link_within_p2p_level(hops: int, link_type: int, level: str = "pix") -> bool:
    """Whether one amdsmi hop/type pair is inside ``level``."""
    canonical = p2p_level_from_env(level)
    if link_type == AMDSMI_LINK_TYPE_XGMI and hops == 1:
        return canonical != "loc"
    if link_type != AMDSMI_LINK_TYPE_PCIE:
        return False
    ceiling = max_pcie_hops_for_level(canonical)
    if ceiling is None or ceiling <= 0:
        return False
    return 1 <= hops <= ceiling


def mesh_within_p2p_level(
    pair_links: list[tuple[int, int]],
    level: str = "pix",
) -> bool:
    """True iff every unordered GPU pair is within ``level``."""
    if not pair_links:
        return False
    canonical = p2p_level_from_env(level)
    return all(link_within_p2p_level(h, t, canonical) for h, t in pair_links)
