# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for ROCR_VISIBLE_DEVICES → amdsmi handle remapping.

Loads ``rocm_visible.py`` by path so collection does not import
``vllm.platforms.rocm`` (HIP init). Also checks that ``get_device_name``
uses ``_amdsmi_index``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "vllm" / "platforms" / "rocm_visible.py"
_ROCM = _ROOT / "vllm" / "platforms" / "rocm.py"
_SPEC = importlib.util.spec_from_file_location("rocm_visible", _SRC)
assert _SPEC is not None and _SPEC.loader is not None
rocm_visible = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rocm_visible)
amdsmi_index_from_rocr = rocm_visible.amdsmi_index_from_rocr


def test_amdsmi_index_from_rocr_maps_logical_zero_to_first_visible():
    # 4x V620 at physical 1..4, display card at 0: ROCR=1,2,3,4
    assert amdsmi_index_from_rocr(0, "1,2,3,4") == 1
    assert amdsmi_index_from_rocr(3, "1,2,3,4") == 4


def test_amdsmi_index_from_rocr_identity_without_env():
    assert amdsmi_index_from_rocr(0, None) == 0
    assert amdsmi_index_from_rocr(2, "") == 2
    assert amdsmi_index_from_rocr(2, "   ") == 2


def test_amdsmi_index_from_rocr_out_of_range_keeps_base():
    assert amdsmi_index_from_rocr(9, "1,2") == 9


def test_amdsmi_index_honors_rocr_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2,3,4")
    base = 0
    assert amdsmi_index_from_rocr(base, os.environ.get("ROCR_VISIBLE_DEVICES")) == 1


def test_get_device_name_uses_amdsmi_index():
    src = _ROCM.read_text()
    assert "cls._amdsmi_index(device_id)" in src
    assert "amdsmi_index_from_rocr(base, os.environ.get(" in src
    assert "device_name=AMD_Radeon_RX_6700_XT" in src


_NO_AMDSMI_PROBE = """
import sys

sys.modules["amdsmi"] = None  # `from amdsmi import ...` raises ImportError
import torch

from vllm.platforms import rocm

assert rocm.with_amdsmi_context(lambda: 7)() == 7
assert rocm.amdsmi_get_processor_handles() == []
assert issubclass(rocm.AmdSmiException, Exception)
# The callers' "amdsmi enumerates no GPU" paths.
assert rocm.RocmPlatform.is_fully_connected([0, 1]) is False
assert rocm.RocmPlatform.is_pix_connected([0, 1]) is False
torch.cuda.get_device_name = lambda device_id=None: "AMD Radeon PRO W7800"
assert rocm.RocmPlatform.get_device_name(0) == "AMD Radeon PRO W7800"
print("no-amdsmi ok")
"""


def test_missing_amdsmi_module_acts_like_an_amdsmi_without_gpus():
    """The rocm_sdk venvs ship no amdsmi: the platform must not NameError.

    G4 on the W7800 found ``NameError: name 'amdsmi_init' is not defined`` from
    the first ``get_device_name()`` (engine startup plan, Triton MoE configs).
    Runs in a subprocess so the import happens with ``amdsmi`` blocked.
    """
    pytest.importorskip("torch")
    proc = subprocess.run(
        [sys.executable, "-c", _NO_AMDSMI_PROBE],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "", "PYTHONPATH": str(_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no-amdsmi ok" in proc.stdout


# --------------------------------------------------------------------------
# The arch of a pinned process: KFD topology files, no amdsmi, no HIP context
# --------------------------------------------------------------------------

V620, W7800 = "gfx1030", "gfx1100"
# HIP order of the hetero box before the case move: W7800 at HIP 0 and 5.
MIXED_BOX = [W7800, V620, V620, V620, V620, W7800, V620, V620, V620, V620]


def _kfd_tree(root: Path, versions: list[int]) -> Path:
    """A topology with the CPU as node 0, then one node per ``gfx_target_version``."""
    for node, version in enumerate([0, *versions]):
        directory = root / str(node)
        directory.mkdir(parents=True)
        (directory / "properties").write_text(
            f"cpu_cores_count {32 if node == 0 else 0}\n"
            f"gfx_target_version {version}\n"
            f"vendor_id {0 if node == 0 else 4098}\n"
        )
    return root


@pytest.mark.parametrize(
    ("version", "name"),
    [
        (90006, "gfx906"),
        (90008, "gfx908"),
        (90010, "gfx90a"),
        (90402, "gfx942"),
        (90500, "gfx950"),
        (100100, "gfx1010"),
        (100300, "gfx1030"),
        (110000, "gfx1100"),
        (110001, "gfx1101"),
        (110501, "gfx1151"),
        (120001, "gfx1201"),
        (120500, "gfx1250"),
    ],
)
def test_gfx_name_of_a_kfd_target_version(version, name):
    assert rocm_visible.gfx_name(version) == name


@pytest.mark.parametrize("version", [0, 5, 100, 101600, 100016])
def test_gfx_name_rejects_what_a_gfx_name_cannot_spell(version):
    """A CPU node (0), junk, a minor of 16 and a stepping of 16."""
    assert rocm_visible.gfx_name(version) is None


def test_kfd_gpu_archs_lists_the_gpu_nodes_in_numeric_node_order(tmp_path):
    # Node 10 is the last GPU although "10" sorts before "2" as a string.
    nodes = _kfd_tree(tmp_path, [110000] + [100300] * 8 + [120001])
    assert rocm_visible.kfd_gpu_archs(nodes) == [W7800] + [V620] * 8 + ["gfx1201"]


def test_kfd_gpu_archs_skips_what_is_not_a_gpu_node(tmp_path):
    nodes = _kfd_tree(tmp_path, [100300])
    (nodes / "notes").mkdir()  # not a node number
    (nodes / "7").mkdir()  # a node without a properties file
    (nodes / "8").mkdir()
    (nodes / "8" / "properties").write_text("gfx_target_version none\n")
    assert rocm_visible.kfd_gpu_archs(nodes) == [V620]


def test_kfd_gpu_archs_without_kfd_is_empty(tmp_path):
    assert rocm_visible.kfd_gpu_archs(tmp_path / "missing") == []


@pytest.mark.parametrize(
    ("hip", "rocr", "expected"),
    [
        ("0", None, W7800),
        ("5", None, W7800),
        ("1,2,3,4", None, V620),
        ("5,0", None, W7800),  # the first listed device decides
        ("4,0", None, V620),
        (None, None, W7800),  # no list: HIP device 0
        ("0", "1,2,3,4", V620),  # ROCR filters first: HIP device 0 is GPU 1
        ("1", "5,1,2", V620),  # ROCR keeps [W7800, V620, V620]; HIP 1 is a V620
        (None, "5,1", W7800),
    ],
)
def test_first_visible_gpu_arch_follows_the_lists(hip, rocr, expected):
    assert rocm_visible.first_visible_gpu_arch(MIXED_BOX, hip, rocr) == expected


@pytest.mark.parametrize(
    ("hip", "rocr"),
    [
        ("GPU-0d6c1a2b", None),  # a UUID
        ("-1", None),  # hides every GPU
        ("10", None),  # past the last GPU
        ("0", "11"),
        ("1", "0"),  # ROCR keeps one GPU, HIP asks for the second
        (None, "GPU-0d6c1a2b"),
        (",", None),
    ],
)
def test_first_visible_gpu_arch_is_none_when_the_lists_cannot_be_followed(hip, rocr):
    assert rocm_visible.first_visible_gpu_arch(MIXED_BOX, hip, rocr) is None


def test_visible_gcn_arch_of_a_single_arch_box_needs_no_list(tmp_path):
    nodes = _kfd_tree(tmp_path, [100300] * 6)
    for environ in (
        {},
        {"HIP_VISIBLE_DEVICES": "3"},
        {"ROCR_VISIBLE_DEVICES": "GPU-0d6c1a2b"},  # no index needed: all GPUs agree
    ):
        assert rocm_visible.visible_gcn_arch(environ, nodes) == V620


def test_visible_gcn_arch_of_a_mixed_box_follows_the_lists(tmp_path):
    nodes = _kfd_tree(tmp_path, [110000, 100300, 100300, 100300, 100300, 110000])

    def arch(**environ: str) -> str | None:
        return rocm_visible.visible_gcn_arch(environ, nodes)

    assert arch(HIP_VISIBLE_DEVICES="0,5") == W7800
    assert arch(CUDA_VISIBLE_DEVICES="1,2") == V620
    # HIP_VISIBLE_DEVICES is read first (the platform never lets the two differ).
    assert arch(HIP_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="1") == W7800
    assert arch(ROCR_VISIBLE_DEVICES="1,2", HIP_VISIBLE_DEVICES="0") == V620
    assert arch(HIP_VISIBLE_DEVICES="GPU-0d6c1a2b") is None


def test_visible_gcn_arch_without_kfd_is_none(tmp_path):
    assert rocm_visible.visible_gcn_arch({}, tmp_path / "missing") is None
