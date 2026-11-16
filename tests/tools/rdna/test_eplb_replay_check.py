# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/rdna/eplb/replay_check.py: passes, runs, odd rank, recurrence.

Pure numpy; builds synthetic ``eplb-stats/1`` rank files whose rings hold forward passes
of two counted layers each and flips the hash of chosen (pass, layer, rank) cells.
"""

import json
from pathlib import Path

import numpy as np

from tools.rdna.eplb import replay_check, summary

RANKS, E, K, RING, NL = 4, 16, 4, 512, 2


def make_dump(
    path: Path,
    passes: list[tuple[int, int]],
    flips: set[tuple[int, int, int]] = frozenset(),
    loads_flip: set[tuple[int, int, int]] = frozenset(),
) -> None:
    """One rank file per rank; ``passes`` are (valid rows, phase), NL records each.

    ``flips`` holds (pass, layer, rank) cells whose routing hash differs; ``loads_flip``
    cells where that rank also counted a different load.
    """
    path.mkdir(parents=True, exist_ok=True)
    edges = list(summary.DEFAULT_BAND_EDGES)
    nb = len(edges)
    for rank in range(RANKS):
        ring = np.zeros((RING, summary.WORDS), dtype=np.int32)
        k = 0
        for i, (rows, phase) in enumerate(passes):
            for layer in range(NL):
                w = ring[k % RING]
                w[summary.W_LAYER] = layer
                w[summary.W_ROWS] = rows
                w[summary.W_TOTAL] = rows
                w[summary.W_HASH] = 1000 + k
                w[summary.W_PHASE] = phase
                w[summary.W_CALL] = i
                w[summary.W_TS_LO] = 5000 * k
                w[summary.W_T0 : summary.W_T0 + RANKS] = 3
                w[summary.W_P0 : summary.W_P0 + RANKS] = 7
                if (i, layer, rank) in flips:
                    w[summary.W_HASH] ^= 0x55
                if (i, layer, rank) in loads_flip:
                    w[summary.W_T0] += 1
                k += 1
        meta = {
            "format": summary.FORMAT,
            "rank": rank,
            "ep_size": RANKS,
            "num_experts": E,
            "top_k": K,
            "decode_max_rows": edges[-1],
            "band_edges": edges,
            "ring": RING,
            "layers": [
                {
                    "name": f"model.layers.{3 + 2 * j}.mlp.experts",
                    "kind": "target",
                    "layer": j,
                }
                for j in range(NL)
            ],
            "pid": 100 + rank,
        }
        np.savez(
            path / f"eplb_stats-rank{rank}-pid{100 + rank}.npz",
            meta=np.array(json.dumps(meta)),
            pairs=np.zeros((NL, 2, E), dtype=np.int64),
            touch=np.zeros((NL, nb, E), dtype=np.int64),
            calls=np.zeros((NL, nb + 1), dtype=np.int64),
            rows=np.zeros((NL, 2), dtype=np.int64),
            ring=ring,
            ring_count=np.int64(k),
        )


def replay_json(path: Path, groups: list[tuple[str, int, int]]) -> None:
    (path / "replay.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "name": n,
                        "length": length,
                        "count": c,
                        "kind": "rnd",
                        "distinct_outputs": 1,
                        "cached_tokens": 0.0,
                    }
                    for n, length, c in groups
                ]
            }
        )
    )


def build(tmp_path, flips=frozenset(), loads_flip=frozenset()):
    """Boot probes, then 4 requests of 1023 rows (+1 decode row) and 3 of 2048 rows."""
    passes = [(3, 0)] * 3 + [(700, 1), (3, 0)]  # warm-up and a probe
    for _ in range(4):
        passes += [(1023, 1), (1, 0), (3, 0)]
    for _ in range(3):
        passes += [(2048, 1), (2048, 1), (3, 0)]
    rep = tmp_path / "replay"
    make_dump(rep / "dump", passes, flips, loads_flip)
    replay_json(rep, [("g1023", 1024, 4), ("g4096", 4096, 3)])
    return rep


def first_run_pass(n_warm: int = 5) -> int:
    return n_warm  # index of the first 1023-row pass


def test_clean_replay_has_no_mismatch(tmp_path):
    rep = replay_check.analyze(str(build(tmp_path)))
    assert rep["consensus"]["ok"]
    assert rep["prefill_runs"] == 1 + 4 + 3  # the 700-row probe, 4 + 3 requests
    g1, g2 = rep["groups"]
    assert (g1["runs"], g1["signature"], g1["signature_consistent"]) == (
        4,
        [1023],
        True,
    )
    assert (g2["runs"], g2["signature"]) == (3, [2048, 2048])
    assert g1["records_mismatch"] == g2["records_mismatch"] == 0
    assert rep["mismatching_records_outside_replay_runs"] == 0
    assert all(v["mismatching_records"] == 0 for v in rep["by_rows"].values())


def test_the_odd_rank_and_the_recurring_layer(tmp_path):
    base = first_run_pass()
    run = [base + 3 * j for j in range(4)]  # pass index of each 1023-row prefill
    # layer 1 of runs 0, 1 and 3 on rank 2 (the recurring cell), layer 0 of run 3 too
    flips = {(run[0], 1, 2), (run[1], 1, 2), (run[3], 1, 2), (run[3], 0, 2)}
    rep = replay_check.analyze(str(build(tmp_path, flips)))
    g1 = rep["groups"][0]
    assert g1["runs_with_mismatch"] == 3 and g1["records_mismatch"] == 4
    assert g1["per_run_records"] == [1, 1, 0, 2]
    assert g1["odd_rank"] == {"2": 4}
    assert g1["distinct_cell_sets"] == 2  # {(0,1)} and {(0,0),(0,1)}
    assert g1["cells_every_run"] == 0 and g1["cells_some_runs"] == 2
    assert g1["cell_max_freq"] == 3
    assert 0 < g1["jaccard_mean"] < 1
    # the other group and the other shapes stay clean
    assert rep["groups"][1]["records_mismatch"] == 0
    assert rep["by_rows"]["1023"]["mismatching_records"] == 4
    assert rep["by_rows"]["2048"]["mismatching_records"] == 0
    assert rep["consensus"]["mismatches"] == 4 and not rep["consensus"]["ok"]
    text = replay_check.format_report(rep)
    assert "odd rank {'2': 4}" in text and "rows=1023: 4 passes, 4 records" in text


def test_two_odd_ranks_are_not_blamed_on_one(tmp_path):
    base = first_run_pass()
    flips = {(base, 0, 1), (base, 0, 3)}
    rep = replay_check.analyze(str(build(tmp_path, flips)))
    g1 = rep["groups"][0]
    assert g1["odd_rank"] == {"-2": 1}


def test_load_differences_are_told_from_hash_only(tmp_path):
    base = first_run_pass()
    cells = {(base, 1, 0)}
    rep = replay_check.analyze(str(build(tmp_path, flips=cells, loads_flip=cells)))
    g1 = rep["groups"][0]
    assert (g1["loads_differ"], g1["hash_only"]) == (1, 0)


def test_mismatch_outside_the_replay_is_counted_apart(tmp_path):
    # pass 3 is the 700-row boot probe: before the replayed requests
    rep = replay_check.analyze(str(build(tmp_path, {(3, 0, 2)})))
    assert rep["mismatching_records_outside_replay_runs"] == 1
    assert rep["groups"][0]["records_mismatch"] == 0


def test_a_dump_without_replay_json_gets_the_window_verdict(tmp_path):
    rep_dir = build(tmp_path, {(first_run_pass(), 1, 2)})
    (rep_dir / "replay.json").unlink()
    rep = replay_check.analyze(str(rep_dir / "dump"))
    assert rep["consensus"]["mismatches"] == 1 and "groups" not in rep
    assert rep["by_rows"]["1023"]["mismatching_records"] == 1


def test_cli_writes_json(tmp_path, capsys):
    rep_dir = build(tmp_path, {(first_run_pass(), 1, 2)})
    out = tmp_path / "check.json"
    assert replay_check.main([str(rep_dir), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["groups"][0]["records_mismatch"] == 1
    assert "consensus: ok=False" in capsys.readouterr().out
