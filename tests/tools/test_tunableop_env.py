# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""configure_tunableop: soft preference, opt-in profiles and the per-rank rows log.

The helper is bash; each test sources it in a subshell with a fake librocblas (its
sha256 is what a profile's lib_sha256 is matched against) and a throw-away rows
root, so no GPU, torch or real rows are needed.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "tools/rdna2_028/tunableop_env.sh"
VALIDATORS = "Validator,PT_VERSION,2.12.0\nValidator,ROCBLAS_VERSION,5.5.0.test\n"
ROW = "GemmTunableOp_Half_TN,tn_{n}_8_64_ld_64_64_{n},Gemm_Rocblas_{n},0.1\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_rows(folder: Path, counts: list[int]) -> None:
    folder.mkdir(parents=True)
    for rank, count in enumerate(counts):
        rows = "".join(ROW.format(n=n + 1) for n in range(count))
        (folder / f"tunableop_results{rank}.csv").write_text(VALIDATORS + rows)


@pytest.fixture()
def tree(tmp_path):
    """A rows root with the canonical profile `base` (3 rows per rank) and the
    opt-in profile `model` (5 rows per rank), both registered for library build A."""
    lib = tmp_path / "librocblas.so.5"
    lib.write_bytes(b"rocblas build A")
    root = tmp_path / "tunableop"
    write_rows(root / "base", [3, 3, 3, 3])
    write_rows(root / "model", [5, 5, 5, 5])
    registry = {
        "profiles": {
            "base": {"dir": "base", "lib_sha256": sha(lib.read_bytes())},
            "model": {
                "dir": "model",
                "lib_sha256": sha(lib.read_bytes()),
                "auto": False,
            },
        }
    }
    (root / "profiles.json").write_text(json.dumps(registry))
    return tmp_path


def configure(tree: Path, **env):
    script = (
        'source "$1"; configure_tunableop "$2" "$3"; rc=$?; '
        'printf "RC=%s ENABLED=%s FILE=%s\\n" "$rc" "$PYTORCH_TUNABLEOP_ENABLED" '
        '"${PYTORCH_TUNABLEOP_FILENAME-unset}"'
    )
    full = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TUNABLEOP", "PYTORCH_TUNABLEOP"))
    }
    full.update(TUNEOP_PER_USER_DIR=str(tree / "peruser"), **env)
    return subprocess.run(
        [
            "bash",
            "-c",
            script,
            "check",
            str(HELPER),
            str(tree / "librocblas.so.5"),
            str(tree / "tunableop"),
        ],
        capture_output=True,
        text=True,
        env=full,
    )


def test_opt_in_profile_is_never_auto_selected(tree):
    result = configure(tree)
    assert "RC=0 ENABLED=1" in result.stdout
    assert f"FILE={tree}/tunableop/base/tunableop_results.csv" in result.stdout
    assert "TunableOp profile base auto-selected" in result.stderr
    assert "preferred" not in result.stderr


def test_preferred_profile_wins_when_registered_for_the_library(tree):
    result = configure(tree, TUNABLEOP_PREFER="model")
    assert "RC=0 ENABLED=1" in result.stdout
    assert f"FILE={tree}/tunableop/model/tunableop_results.csv" in result.stdout
    assert "TunableOp profile model preferred (TUNABLEOP_PREFER)" in result.stderr
    assert "auto-selected" not in result.stderr


def test_empty_preference_restores_the_canonical_profile(tree):
    result = configure(tree, TUNABLEOP_PREFER="")
    assert f"FILE={tree}/tunableop/base/tunableop_results.csv" in result.stdout


def test_preference_tuned_for_another_build_falls_back_to_auto(tree):
    registry = json.loads((tree / "tunableop/profiles.json").read_text())
    registry["profiles"]["model"]["lib_sha256"] = sha(b"rocblas build B")
    (tree / "tunableop/profiles.json").write_text(json.dumps(registry))
    result = configure(tree, TUNABLEOP_PREFER="model")
    assert "RC=0 ENABLED=1" in result.stdout
    assert f"FILE={tree}/tunableop/base/tunableop_results.csv" in result.stdout
    assert 'preferred TunableOp profile "model"' in result.stderr
    assert "auto-selecting" in result.stderr


def test_unregistered_preference_warns_and_never_fails(tree):
    result = configure(tree, TUNABLEOP_PREFER="nope")
    assert "RC=0 ENABLED=1" in result.stdout
    assert 'preferred TunableOp profile "nope"' in result.stderr
    assert f"FILE={tree}/tunableop/base/tunableop_results.csv" in result.stdout


def test_explicit_profile_overrides_the_preference(tree):
    result = configure(tree, TUNABLEOP_PREFER="model", TUNABLEOP_PROFILE="base")
    assert f"FILE={tree}/tunableop/base/tunableop_results.csv" in result.stdout
    assert "TunableOp profile base selected" in result.stderr
    assert "preferred" not in result.stderr


def test_explicit_profile_for_another_build_still_refuses(tree):
    registry = json.loads((tree / "tunableop/profiles.json").read_text())
    registry["profiles"]["model"]["lib_sha256"] = sha(b"rocblas build B")
    (tree / "tunableop/profiles.json").write_text(json.dumps(registry))
    result = configure(tree, TUNABLEOP_PROFILE="model")
    assert "RC=1" in result.stdout
    assert "refusing to start" in result.stderr


@pytest.mark.parametrize(
    ("ranks", "expected"),
    [
        ("2", " 0=5 1=5"),
        ("1", " 0=5"),
        ("4", " 0=5 1=5 2=5 3=5"),
        (None, " 0=5 1=5 2=5 3=5"),
    ],
)
def test_rows_per_rank_log_follows_the_rank_count(tree, ranks, expected):
    env = {"TUNABLEOP_PREFER": "model"}
    if ranks is not None:
        env["TUNABLEOP_RANKS"] = ranks
    result = configure(tree, **env)
    line = next(
        row for row in result.stderr.splitlines() if row.startswith("TunableOp rows")
    )
    assert line.endswith(":" + expected)


def test_rows_per_rank_log_counts_each_rank_file(tree):
    write_rows(tree / "tunableop/ranks", [2, 7, 4, 4])
    registry = json.loads((tree / "tunableop/profiles.json").read_text())
    registry["profiles"]["ranks"] = {
        "dir": "ranks",
        "lib_sha256": registry["profiles"]["base"]["lib_sha256"],
        "auto": False,
    }
    (tree / "tunableop/profiles.json").write_text(json.dumps(registry))
    result = configure(tree, TUNABLEOP_PREFER="ranks", TUNABLEOP_RANKS="2")
    assert ":" + " 0=2 1=7" in result.stderr


def test_missing_rank_file_of_the_preferred_profile_falls_back_to_per_user(tree):
    (tree / "tunableop/model/tunableop_results1.csv").unlink()
    result = configure(tree, TUNABLEOP_PREFER="model")
    assert "RC=0 ENABLED=1" in result.stdout
    assert f"FILE={tree}/peruser/tunableop_results.csv" in result.stdout
    assert "falling back to per-user" in result.stderr


def test_unknown_library_still_disables_lookup(tree):
    (tree / "librocblas.so.5").unlink()
    result = configure(tree, TUNABLEOP_PREFER="model")
    assert "RC=0 ENABLED=0 FILE=unset" in result.stdout
    assert "WARNING" in result.stderr
