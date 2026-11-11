# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tools/rdna/prof/tunableop_missing.py.

The helper is stdlib-only, so the tests run on a laptop and on the box without
a GPU. They build a throw-away profile directory and check the missing-key
classification, the shard split used to tune on several GPUs in parallel, and
the multi-file harvest into the four per-rank tables.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "tunableop_missing", ROOT / "tools/rdna/prof/tunableop_missing.py"
)
assert _SPEC is not None and _SPEC.loader is not None
TM = importlib.util.module_from_spec(_SPEC)
sys.modules["tunableop_missing"] = TM
_SPEC.loader.exec_module(TM)

PROFILE = "p1"
VALIDATORS = ["Validator,PT_VERSION,2.12.0", "Validator,HIP_VERSION,714"]
HAVE = "GemmTunableOp_Half_TN,tn_640_2048_2560_ld_2560_2560_640,Gemm_Rocblas_1,0.1"
CENSUS = [
    "GemmTunableOp_Half_TN,tn_640_2048_2560_ld_2560_2560_640",  # in the table
    "GemmTunableOp_Half_TN,tn_4096_1024_2560_ld_2560_2560_4096",
    "GemmTunableOp_Half_TN,tn_4096_1026_2560_ld_2560_2560_4096",
    "GemmTunableOp_Half_TN,tn_4096_1023_2560_ld_2560_2560_4096",
    "GemmTunableOp_Half_TN,tn_3584_1026_2560_ld_2560_2560_3584",
    "GemmTunableOp_Half_NN,nn_16_16_16_ld_16_16_16",  # warm-up noise
    "GemmStridedBatchedTunableOp_float_NN,nn_72_6272_6272_B_4_ld_288_6272_72",
]
MISSING_SUPPORTED = 4
RANKS = (0, 1, 2, 3)  # the shipped TP4 rank files; harvest follows the files on disk


@pytest.fixture
def rows_dir(tmp_path):
    (tmp_path / PROFILE).mkdir()
    registry = {"profiles": {PROFILE: {"dir": PROFILE, "rows_per_rank": 1}}}
    (tmp_path / "profiles.json").write_text(json.dumps(registry))
    (tmp_path / PROFILE / "provenance.json").write_text(
        json.dumps({"rows_per_rank": 1})
    )
    for rank in RANKS:
        lines = [*VALIDATORS, HAVE]
        (tmp_path / PROFILE / f"tunableop_results{rank}.csv").write_text(
            "\n".join(lines) + "\n"
        )
    return tmp_path


@pytest.fixture
def census(tmp_path):
    path = tmp_path / "census.csv"
    path.write_text("\n".join(CENSUS) + "\n")
    return path


def _run(*argv):
    return TM.main([str(a) for a in argv])


def test_find_missing_classifies(rows_dir, census):
    supported, noise, strided = TM.find_missing(
        census, TM.table_path(rows_dir, PROFILE, 0)
    )
    assert len(supported) == MISSING_SUPPORTED
    assert [k[1] for k in noise] == ["nn_16_16_16_ld_16_16_16"]
    assert [k[0] for k in strided] == ["GemmStridedBatchedTunableOp_float_NN"]


def _untuned(rows_dir, census, out, *extra):
    rc = _run(
        "untuned",
        "--census",
        census,
        "--out",
        out,
        "--profile",
        PROFILE,
        "--rows-dir",
        rows_dir,
        *extra,
    )
    assert rc == 0
    return [ln for ln in Path(out).read_text().splitlines() if ln]


def test_shards_partition_the_missing_keys(rows_dir, census, tmp_path):
    whole = _untuned(rows_dir, census, tmp_path / "all.csv")
    assert len(whole) == MISSING_SUPPORTED
    shards = [
        _untuned(rows_dir, census, tmp_path / f"s{i}.csv", "--shard", f"{i}/3")
        for i in range(3)
    ]
    flat = [ln for shard in shards for ln in shard]
    assert sorted(flat) == sorted(whole)  # disjoint and complete
    assert all(len(s) >= 1 for s in shards)


@pytest.mark.parametrize("bad", ["3/3", "-1/3", "0/0"])
def test_bad_shard_is_refused(rows_dir, census, tmp_path, bad):
    rc = _run(
        "untuned",
        "--census",
        census,
        "--out",
        tmp_path / "x.csv",
        "--profile",
        PROFILE,
        "--rows-dir",
        rows_dir,
        f"--shard={bad}",
    )
    assert rc == 1


def _tuned_file(path, rows, validators=VALIDATORS):
    path.write_text("\n".join([*validators, *rows]) + "\n")
    return path


def test_harvest_merges_several_files_into_every_rank(rows_dir, tmp_path):
    new_a = (
        "GemmTunableOp_Half_TN,tn_4096_1024_2560_ld_2560_2560_4096,Gemm_Rocblas_7,0.2"
    )
    new_b = (
        "GemmTunableOp_Half_TN,tn_4096_1026_2560_ld_2560_2560_4096,Gemm_Rocblas_8,0.3"
    )
    dup = HAVE  # already in the table: skipped, not duplicated
    f1 = _tuned_file(tmp_path / "t1.csv", [new_a, dup])
    f2 = _tuned_file(tmp_path / "t2.csv", [new_b, new_a])  # new_a repeated
    rc = _run(
        "harvest",
        "--tuned",
        f1,
        "--tuned",
        f2,
        "--profile",
        PROFILE,
        "--rows-dir",
        rows_dir,
    )
    assert rc == 0
    for rank in RANKS:
        text = (rows_dir / PROFILE / f"tunableop_results{rank}.csv").read_text()
        rows = [ln for ln in text.splitlines() if not ln.startswith("Validator")]
        assert rows == [HAVE, new_a, new_b]
        assert text.splitlines()[:2] == VALIDATORS
    prov = json.loads((rows_dir / PROFILE / "provenance.json").read_text())
    assert prov["rows_per_rank"] == 3


def test_harvest_reject_leaves_out_listed_keys(rows_dir, tmp_path):
    keep = (
        "GemmTunableOp_Half_TN,tn_4096_1024_2560_ld_2560_2560_4096,Gemm_Rocblas_7,0.2"
    )
    bad = "GemmTunableOp_Half_TN,tn_2048_12_2560_ld_2560_2560_2048,Gemm_Rocblas_9,0.1"
    tuned = _tuned_file(tmp_path / "t.csv", [keep, bad, bad])
    # rowcheck writes bare `op,key` lines; a full row names the same key
    reject = tmp_path / "reject.txt"
    reject.write_text("GemmTunableOp_Half_TN,tn_2048_12_2560_ld_2560_2560_2048\n")
    rc = _run(
        "harvest",
        "--tuned",
        tuned,
        "--reject",
        reject,
        "--profile",
        PROFILE,
        "--rows-dir",
        rows_dir,
    )
    assert rc == 0
    for rank in RANKS:
        text = (rows_dir / PROFILE / f"tunableop_results{rank}.csv").read_text()
        rows = [ln for ln in text.splitlines() if not ln.startswith("Validator")]
        assert rows == [HAVE, keep]
    prov = json.loads((rows_dir / PROFILE / "provenance.json").read_text())
    assert prov["rows_per_rank"] == 2


def test_harvest_refuses_validator_mismatch(rows_dir, tmp_path):
    new = "GemmTunableOp_Half_TN,tn_4096_1024_2560_ld_2560_2560_4096,Gemm_Rocblas_7,0.2"
    good = _tuned_file(tmp_path / "good.csv", [new])
    other = _tuned_file(
        tmp_path / "other.csv", [new], validators=["Validator,PT_VERSION,9.9.9"]
    )
    for files in ([other], [good, other]):
        args = ["harvest", "--profile", PROFILE, "--rows-dir", rows_dir]
        for f in files:
            args += ["--tuned", f]
        assert _run(*args) == 1
    # Nothing was written by the refused merges.
    text = (rows_dir / PROFILE / "tunableop_results0.csv").read_text()
    assert new not in text
