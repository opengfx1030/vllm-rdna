# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_FA_RDNA2_GQA_DECODE=2 (16-byte LDS traffic) is bit-identical to =1.

``fa_decode_paged_splitk_gqa_vec_kernel_256`` keeps the grid, the split layout
and the arithmetic order of ``fa_decode_paged_splitk_gqa_kernel_256`` and only
changes how the K / V tile moves through shared memory, so every output element
must match the scalar-LDS kernel exactly, not merely stay within a tolerance.
The switch is read once per process (a static in ``fa_rdna2.cu``), so the
production-dispatch checker ``tools/rdna/port_v031/gqa_decode_check.py`` runs
once per setting (pure decode, MTP verify rows, mixed batches; the TP1/TP2/TP4
head counts of the 27B) and the output hashes of the two runs are compared.

GPU test (gfx103x only, skipped elsewhere):

    pytest -s tests/kernels/attention/test_fa_rdna2_gqa_decode_vec.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CHECK = Path(__file__).resolve().parents[3] / "tools/rdna/port_v031/gqa_decode_check.py"


def _on_gfx103() -> bool:
    try:
        return torch.cuda.is_available() and "gfx103" in (
            torch.cuda.get_device_properties(0).gcnArchName
        )
    except Exception:
        return False


def _run(mode: str, out: Path) -> dict:
    env = dict(os.environ, VLLM_FA_RDNA2_GQA_DECODE=mode)
    cmd = [sys.executable, str(CHECK), "--tag", f"m{mode}", "--repeat", "2"]
    cmd += ["--out", str(out)]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    return {r["name"]: r for r in json.loads(out.read_text())["results"]}


@pytest.mark.skipif(not _on_gfx103(), reason="Requires AMD RDNA2 (gfx103x)")
def test_gqa_decode_vec_is_bitwise_equal(tmp_path):
    scalar = _run("1", tmp_path / "m1.json")
    vec = _run("2", tmp_path / "m2.json")
    assert scalar and scalar.keys() == vec.keys()
    # The cases that engage the grouped kernel (rows x KV heads >= 4) are the
    # ones the switch changes; the rest run the per-head kernel in both runs.
    assert any(r["gate"] >= 4 for r in scalar.values())
    differing = [n for n in scalar if scalar[n]["sha"] != vec[n]["sha"]]
    assert not differing, differing
