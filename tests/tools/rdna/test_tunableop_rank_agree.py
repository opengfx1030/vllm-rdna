# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shipped TunableOp rank files name one solver per key (tunableop_agree.py).

CPU only, stdlib. A TP worker on device ordinal ``r`` reads
``tunableop_results<r>.csv``;
a replicated GEMM (the MoE router) must run the same rocBLAS solution on every rank or
its logits differ in the last bit and the ranks route a near-tied token differently
(docs/rdna2/tunableop-rank-agreement-2026-11-16.md). The data tests read the
committed ``tunableop/`` tree, the others build throw-away profile directories.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "tunableop_agree", ROOT / "tools/rdna2_028/tunableop_agree.py"
)
assert _SPEC is not None and _SPEC.loader is not None
TA = importlib.util.module_from_spec(_SPEC)
sys.modules["tunableop_agree"] = TA
_SPEC.loader.exec_module(TA)

ROWS = ROOT / "tunableop"
PROFILES = sorted(p.name for p in ROWS.iterdir() if p.is_dir()) if ROWS.is_dir() else []
VALIDATORS = ["Validator,PT_VERSION,2.12.0", "Validator,HIP_VERSION,714"]
ROUTER_1023 = "tn_512_1023_2560_ld_2560_2560_512"


def row(key: str, solver: int, ms: float) -> str:
    return f"GemmTunableOp_Half_TN,{key},Gemm_Rocblas_{solver},{ms}"


def write_profile(path: Path, per_rank: dict[int, list[str]]) -> Path:
    path.mkdir(parents=True)
    for rank, rows in per_rank.items():
        (path / f"tunableop_results{rank}.csv").write_text(
            "\n".join(VALIDATORS + rows) + "\n"
        )
    return path


@pytest.fixture
def split_profile(tmp_path):
    """Rank 2 tuned a different solver for two keys and agrees on the third."""
    same = row("tn_24_3_2560_ld_2560_2560_24", 7, 0.02)
    return write_profile(
        tmp_path / "p",
        {
            0: [
                row(ROUTER_1023, -46, 0.1397),
                row("tn_640_9_2560_ld_2560_2560_640", 5, 0.05),
                same,
            ],
            1: [
                row(ROUTER_1023, -46, 0.1359),
                row("tn_640_9_2560_ld_2560_2560_640", 5, 0.05),
                same,
            ],
            2: [
                row(ROUTER_1023, -44, 0.1442),
                row("tn_640_9_2560_ld_2560_2560_640", 4, 0.05),
                same,
            ],
            3: [
                row(ROUTER_1023, -46, 0.1415),
                row("tn_640_9_2560_ld_2560_2560_640", 5, 0.05),
                same,
            ],
        },
    )


def test_disagreements_name_the_odd_rank(split_profile):
    _, rows = TA.load(split_profile)
    differ, missing = TA.disagreements(rows)
    assert not missing
    assert {k[1] for k in differ} == {ROUTER_1023, "tn_640_9_2560_ld_2560_2560_640"}
    odd = [
        r
        for r, entry in differ[("GemmTunableOp_Half_TN", ROUTER_1023)].items()
        if entry.solver.endswith("-44")
    ]
    assert odd == [2]
    assert TA.check(split_profile) == 1


def test_agree_keeps_the_fastest_solver_and_everything_else(split_profile, tmp_path):
    before = {r: p.read_text() for r, p in TA.rank_files(split_profile).items()}
    out = tmp_path / "agreed"
    done = TA.agree(split_profile, out)
    assert done == {"keys": 2, "lines": 2, "files": 4}
    assert TA.check(out) == 0
    after = {r: p.read_text().splitlines() for r, p in TA.rank_files(out).items()}
    # the router key: -46 at 0.1359 (rank 1) beats -44; rank 2 gets that exact line
    assert row(ROUTER_1023, -46, 0.1359) in after[2]
    assert row(ROUTER_1023, -46, 0.1397) in after[0]  # an agreeing line keeps its time
    # the 640 key: 5 and 4 tie at 0.05, the lower rank wins, so 5 everywhere
    assert all("Gemm_Rocblas_4," not in "\n".join(lines) for lines in after.values())
    # order and validators are untouched, no key added or dropped
    for r in after:
        old = before[r].splitlines()
        assert len(after[r]) == len(old) and after[r][:2] == VALIDATORS
        assert [x.split(",")[1] for x in after[r][2:]] == [
            x.split(",")[1] for x in old[2:]
        ]
    # idempotent: nothing left to rewrite
    assert TA.agree(out, tmp_path / "again") == {"keys": 0, "lines": 0, "files": 4}


def test_a_key_missing_from_one_file_is_reported_and_not_filled(tmp_path):
    prof = write_profile(
        tmp_path / "p",
        {0: [row("tn_1_1_1_ld_1_1_1", 1, 0.1)], 1: []},
    )
    _, rows = TA.load(prof)
    differ, missing = TA.disagreements(rows)
    assert not differ and list(missing.values()) == [[1]]
    assert TA.check(prof) == 1
    with pytest.raises(SystemExit):
        TA.agree(prof)


def test_a_single_rank_file_has_nothing_to_compare(tmp_path):
    prof = write_profile(tmp_path / "p", {0: [row("tn_1_1_1_ld_1_1_1", 1, 0.1)]})
    assert TA.check(prof) == 0


@pytest.mark.parametrize("profile", PROFILES)
def test_shipped_rank_files_agree_per_key(profile):
    """Drift guard: re-tuning one rank must not bring the split back."""
    files = TA.rank_files(ROWS / profile)
    if len(files) < 2:
        pytest.skip(f"{profile}: fewer than two rank files")
    texts, rows = TA.load(ROWS / profile)
    differ, missing = TA.disagreements(rows)
    assert not missing, (
        f"{profile}: keys missing from some rank file: {list(missing)[:5]}"
    )
    assert not differ, (
        f"{profile}: {len(differ)} keys name different solvers per rank, e.g. "
        + ", ".join(k[1] for k in list(differ)[:3])
        + "; run tools/rdna2_028/tunableop_agree.py write"
    )
    assert len({tuple(TA.parse(t)[0]) for t in texts.values()}) == 1


def test_the_router_gemm_at_1023_rows_runs_one_solver_on_every_rank():
    """The key that made rank 2 route differently in pure 1023-row prefill passes."""
    canon = ROWS / "rocm7.14-rocblas5.5"
    if not canon.is_dir():
        pytest.skip("canonical profile not in this tree")
    _, rows = TA.load(canon)
    key = ("GemmTunableOp_Half_TN", ROUTER_1023)
    assert all(key in r for r in rows.values())
    assert len({r[key].solver for r in rows.values()}) == 1
