# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-27B HI2 TP=2 TunableOp profile: registry, rows and recipe wiring.

CPU only. The data tests read the committed ``tunableop/`` tree; the print-mode tests
run ``serve_rdna.sh PRINT=1`` against a stub venv with a fake ``librocblas.so.5`` and a
fake rows root, so no GPU and no real library are involved.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ROWS = REPO / "tunableop"
CANON = "rocm7.14-rocblas5.5"
DERIVED = "rocm7.14-rocblas5.5-hi2-27b-tp2"
RECIPES = REPO / "tools/rdna/recipes"
LAUNCHER = REPO / "tools/rdna/serve_rdna.sh"
RANKS = range(4)
# per-rank (R, K) of the rows the campaign adds:
# gate_up, in_proj_qkvz, qkv, out / o_proj, ba
NEW_SHAPES = {(17408, 5120), (8192, 5120), (7168, 5120), (5120, 3072), (48, 5120)}


def registry():
    return json.loads((ROWS / "profiles.json").read_text())["profiles"]


def split(path):
    validators, rows = [], {}
    for line in path.read_text().splitlines():
        if line.startswith("Validator,"):
            validators.append(line)
        elif line.startswith("Gemm"):
            op, key, solver = line.split(",", 2)[:3]
            rows[(op, key)] = line
    return validators, rows


def test_registry_entry_is_opt_in_and_same_build():
    profiles = registry()
    derived, canon = profiles[DERIVED], profiles[CANON]
    assert derived["auto"] is False
    assert "auto" not in canon or canon["auto"] is not False
    assert derived["lib_sha256"] == canon["lib_sha256"]
    assert derived["dir"] == DERIVED
    assert derived["derived_from"] == CANON
    assert (ROWS / DERIVED / "provenance.json").is_file()
    assert (ROWS / DERIVED / "README.md").is_file()


@pytest.mark.parametrize("rank", RANKS)
def test_rank_file_is_canonical_plus_new_rows(rank):
    name = f"tunableop_results{rank}.csv"
    canon_validators, canon_rows = split(ROWS / CANON / name)
    validators, rows = split(ROWS / DERIVED / name)
    assert validators == canon_validators
    # nothing dropped, no solver changed
    for key, line in canon_rows.items():
        assert rows[key] == line
    new = [key for key in rows if key not in canon_rows]
    assert new, "the derived profile adds rows"
    for _, key in new:
        parts = key.split("_")
        assert parts[0] == "tn" and parts[4] == "ld", key
        r, k, lda, ldb, ldc = (int(parts[i]) for i in (1, 3, 5, 6, 7))
        assert (r, k) in NEW_SHAPES, key
        assert (lda, ldb, ldc) == (k, k, r), key
    assert len(rows) == registry()[DERIVED]["rows_per_rank"]


def test_new_rows_are_identical_on_every_rank():
    """TP=2 workers use ranks 0 and 1; one tuning serves both (same build)."""
    _, canon = split(ROWS / CANON / "tunableop_results0.csv")
    _, first = split(ROWS / DERIVED / "tunableop_results0.csv")
    new0 = {key: line for key, line in first.items() if key not in canon}
    for rank in RANKS:
        _, rows = split(ROWS / DERIVED / f"tunableop_results{rank}.csv")
        _, canon_r = split(ROWS / CANON / f"tunableop_results{rank}.csv")
        assert {k: v for k, v in rows.items() if k not in canon_r} == new0


def test_only_the_27b_hi2_tp2_recipe_prefers_the_profile():
    preferring = {}
    for recipe in sorted(RECIPES.glob("*.env")):
        for line in recipe.read_text().splitlines():
            if line.startswith("TUNABLEOP_PREFER="):
                preferring[recipe.name] = line.partition("=")[2].strip()
    assert preferring == {"27b-hi2-tp2.env": DERIVED}
    assert DERIVED in registry()


@pytest.fixture()
def stub(tmp_path):
    lib = tmp_path / "venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
    lib.mkdir(parents=True)
    (lib / "librocblas.so.5").write_bytes(b"fake rocblas build")
    sha = hashlib.sha256(b"fake rocblas build").hexdigest()
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    root = tmp_path / "rows"
    root.mkdir()
    entry = {"lib_sha256": sha, "rocblas": "x", "rocm": "x", "torch": "x"}
    (root / "profiles.json").write_text(
        json.dumps(
            {
                "profiles": {
                    CANON: {**entry, "dir": CANON, "rows_per_rank": 1},
                    DERIVED: {
                        **entry,
                        "dir": DERIVED,
                        "rows_per_rank": 2,
                        "auto": False,
                    },
                }
            }
        )
    )
    return tmp_path / "venv", root, sha[:12]


def plan(venv, root, *args, **env):
    drop = ("HIP_VISIBLE_DEVICES", "PCI_BUSES", "VENV", "MODEL", "VIRTUAL_ENV")
    full = {k: v for k, v in os.environ.items() if k not in drop}
    full.pop("TUNABLEOP_PREFER", None)
    full.pop("TUNABLEOP_PROFILE", None)
    full.update(
        VENV=str(venv), MODEL="/nonexistent/model", PRINT="1", TUNABLEOP_ROOT=str(root)
    )
    full.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True,
        text=True,
        env=full,
        cwd=REPO,
        timeout=60,
    )


def tunableop_line(result):
    assert result.returncode == 0, result.stderr
    (line,) = [x for x in result.stdout.splitlines() if x.startswith("# tunableop:")]
    return line


def test_recipe_prefers_the_hi2_profile(stub):
    venv, root, sha = stub
    line = tunableop_line(plan(venv, root, "RECIPE=27b-hi2-tp2"))
    assert (
        f"{DERIVED} (preferred via TUNABLEOP_PREFER for librocblas.so.5 {sha})" in line
    )


def test_empty_prefer_is_the_canonical_rows_arm(stub):
    venv, root, sha = stub
    line = tunableop_line(plan(venv, root, "RECIPE=27b-hi2-tp2", "TUNABLEOP_PREFER="))
    assert f"{CANON} (auto-selected for librocblas.so.5 {sha})" in line
    assert DERIVED not in line


def test_explicit_profile_wins_over_the_preference(stub):
    venv, root, _ = stub
    line = tunableop_line(
        plan(venv, root, "RECIPE=27b-hi2-tp2", f"TUNABLEOP_PROFILE={CANON}")
    )
    assert f"{CANON} (requested" in line


def test_other_recipes_keep_the_canonical_rows(stub):
    venv, root, sha = stub
    for recipe in ("27b-awq", "flashnext-tp8-mtp2"):
        line = tunableop_line(plan(venv, root, f"RECIPE={recipe}"))
        assert f"{CANON} (auto-selected for librocblas.so.5 {sha})" in line, recipe
        assert DERIVED not in line
