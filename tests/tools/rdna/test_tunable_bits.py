# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/rdna/eplb/tunable_bits.py ``compare`` (numpy only).

``run`` needs a GPU and is exercised on the box; here synthetic single-ordinal run
files stand in for it: ordinal 2 rounds one logit differently, which flips a top-k set.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np

from tools.rdna.eplb import tunable_bits as tb

ROWS = (3, 1023)


def run_file(path: Path, ordinal: int, bus: int, bump: bool, solver: str) -> str:
    """What ``run`` stores for one ordinal; ``bump`` changes one logit of 1023 rows."""
    out: dict = {
        "ordinals": np.array([ordinal]),
        "rows": np.array(ROWS),
        "reps": np.array(2),
        "buses": np.array([bus]),
        f"tunable_file_{ordinal}": np.array(f"/rows/tunableop_results{ordinal}.csv"),
        "tunable_enabled": np.array([True, False]),
    }
    for rows in ROWS:
        rng = np.random.default_rng([7, rows])
        logits = rng.standard_normal((rows, 32)).astype(np.float16)
        if bump and rows == 1023:
            logits[5, 3] = np.float16(10.0)  # becomes the largest logit of row 5
        out[f"logits_{ordinal}_{rows}"] = logits
        sets = tb._topk_sets(logits, 4)
        out[f"topk_{ordinal}_{rows}"] = np.stack([sets, sets])
        out[f"solver_{ordinal}_{rows}"] = np.array(
            solver if rows == 1023 else "Gemm_Rocblas_9"
        )
    np.savez(path, **out)
    return str(path)


def test_compare_names_the_odd_ordinal_and_the_card_independence(tmp_path, capsys):
    a = [
        run_file(
            tmp_path / f"a{o}.npz",
            o,
            0xD9 + o,
            o == 2,
            "Gemm_Rocblas_44" if o == 2 else "Gemm_Rocblas_46",
        )
        for o in range(4)
    ]
    # the same ordinals on other cards (devices reversed): the same bits
    b = [
        run_file(
            tmp_path / f"b{o}.npz",
            o,
            0xE2 - o,
            o == 2,
            "Gemm_Rocblas_44" if o == 2 else "Gemm_Rocblas_46",
        )
        for o in range(4)
    ]
    rep = tb.compare(Namespace(a=a, vs=b))
    text = capsys.readouterr().out
    # ordinal 2 differs from every other ordinal on the 1023-row GEMM only
    for pair in ("0-2", "1-2", "2-3"):
        s = rep["pairs"][f"1023:{pair}"]
        assert s["elems"] == 1 and s["flip_rows"] >= 1 and s["of"] == 1023 * 32
    for pair in ("0-1", "0-3", "1-3"):
        assert rep["pairs"][f"1023:{pair}"]["elems"] == 0
    assert all(s["elems"] == 0 for k, s in rep["pairs"].items() if k.startswith("3:"))
    # same ordinal on another card: identical
    assert all(s["elems"] == 0 for s in rep["cards"].values())
    assert "IDENTICAL bits everywhere" in text
    assert "rows 1023: 0: 46, 1: 46, 2: 44, 3: 46" in text.replace("Gemm_Rocblas_", "")


def test_a_card_that_changes_the_bits_is_reported(tmp_path, capsys):
    a = [run_file(tmp_path / "a.npz", 0, 0xD9, False, "Gemm_Rocblas_46")]
    b = [run_file(tmp_path / "b.npz", 0, 0xE2, True, "Gemm_Rocblas_46")]
    rep = tb.compare(Namespace(a=a, vs=b))
    assert rep["cards"]["1023:0"]["elems"] == 1
    assert "DIFFERS" in capsys.readouterr().out


def test_topk_sets_are_sorted_and_stable_on_ties():
    logits = np.array([[1.0, 3.0, 3.0, 2.0]], dtype=np.float16)
    assert tb._topk_sets(logits, 2).tolist() == [[1, 2]]
    assert tb._topk_sets(logits, 3).tolist() == [[1, 2, 3]]
