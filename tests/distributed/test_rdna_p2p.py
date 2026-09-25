# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only PIX / P2P-level tests for gfx10x topology helpers.

Loads ``rdna_p2p.py`` by path so collection does not import
``vllm.distributed`` (torch). Hop ceilings and one-board PIX vs
two-board PHB. Does not gate VLLM_RDNA_AR.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SRC = (
    Path(__file__).resolve().parents[2]
    / "vllm"
    / "distributed"
    / "device_communicators"
    / "rdna_p2p.py"
)
_SPEC = importlib.util.spec_from_file_location("rdna_p2p", _SRC)
assert _SPEC is not None and _SPEC.loader is not None
rdna_p2p = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rdna_p2p)

PCIE = rdna_p2p.AMDSMI_LINK_TYPE_PCIE
XGMI = rdna_p2p.AMDSMI_LINK_TYPE_XGMI

# One-board 4x V620 under a PEX88096: amd-smi type PCIE, hops 2.
_PIX_MESH = [(2, PCIE), (2, PCIE), (2, PCIE), (2, PCIE), (2, PCIE), (2, PCIE)]
# Two USPs (4+4): same-board PIX plus cross-board PHB.
_TWO_BOARD = [(2, PCIE), (2, PCIE), (4, PCIE)]


def test_p2p_level_from_env_aliases_and_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)
    assert rdna_p2p.p2p_level_from_env() == "pix"
    assert rdna_p2p.p2p_level_from_env("1") == "pix"
    assert rdna_p2p.p2p_level_from_env("PXB") == "pxb"
    assert rdna_p2p.p2p_level_from_env("3") == "phb"
    assert rdna_p2p.p2p_level_from_env("nope") == "pix"


@pytest.mark.parametrize(
    "hops,link_type,level,expect",
    [
        (2, PCIE, "pix", True),
        (1, PCIE, "pix", True),
        (3, PCIE, "pix", False),
        (4, PCIE, "pix", False),
        (3, PCIE, "pxb", True),
        (4, PCIE, "pxb", False),
        (4, PCIE, "phb", True),
        (5, PCIE, "phb", False),
        (5, PCIE, "sys", True),
        (2, PCIE, "nvl", False),
        (2, PCIE, "loc", False),
        (1, XGMI, "pix", True),
        (1, XGMI, "nvl", True),
        (1, XGMI, "loc", False),
        (2, XGMI, "pix", False),
    ],
)
def test_link_within_p2p_level(hops: int, link_type: int, level: str, expect: bool):
    assert rdna_p2p.link_within_p2p_level(hops, link_type, level) is expect


def test_one_board_pix_mesh_is_connected():
    assert rdna_p2p.mesh_within_p2p_level(_PIX_MESH, "pix") is True


def test_two_board_mesh_is_phb_not_pix():
    assert rdna_p2p.mesh_within_p2p_level(_TWO_BOARD, "pix") is False
    assert rdna_p2p.mesh_within_p2p_level(_TWO_BOARD, "phb") is True
    assert rdna_p2p.mesh_within_p2p_level(_TWO_BOARD, "sys") is True


def test_empty_mesh_is_not_connected():
    assert rdna_p2p.mesh_within_p2p_level([], "pix") is False
