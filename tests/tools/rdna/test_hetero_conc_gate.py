# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/rdna/hetero_moe/conc_gate.py: the hetero stand-in's concurrency gate.

Arm directories are built by hand: the probe results (concurrency.json,
conc/run*.json), the hub's serve.log and the servers' hetero/<id>.log with the digest
counters lines HxServer's hub and servers log.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "tools/rdna/hetero_moe/conc_gate.py"

COUNTERS = (
    "{ok} ok, {bad} bad at the first read, {healed} healed, {unhealed} unhealed, "
    "0 late digest words, 4 rescans"
)
HUB_OK = "hub digest: " + COUNTERS.format(ok=90, bad=0, healed=0, unhealed=0)
SERVER_OK = "server digest: " + COUNTERS.format(ok=90, bad=0, healed=0, unhealed=0)
SOLO = {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("conc_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def item(ids, lps=None, top=None):
    tokens = [f"token_id:{i}" for i in ids]
    lps = lps if lps is not None else [-0.1 * (k + 1) for k in range(len(ids))]
    top = top if top is not None else [{t: lp} for t, lp in zip(tokens, lps)]
    return {"tokens": tokens, "lps": lps, "top": top, "text": ""}


def run_json(solo=SOLO, solo2=None, together=None, solo_items=None):
    """One probe result: ``solo_items`` replaces whole solo items (own top-5)."""
    d = {
        "solo": {k: item(v) for k, v in solo.items()},
        "solo2": {k: item(v) for k, v in (solo2 or solo).items()},
        "together": {k: item(v) for k, v in (together or solo).items()},
        "churn": {f"(3, {k})": item(v) for k, v in solo.items()},
    }
    d["solo"].update(solo_items or {})
    return d


def make_arm(root, runs, hub=HUB_OK, servers=None, ready=("exp0", "exp1")):
    """runs[0] is the boot probe (concurrency.json), the rest conc/run1.json ..."""
    arm = Path(root) / "arm"
    (arm / "conc").mkdir(parents=True)
    (arm / "hetero").mkdir()
    (arm / "concurrency.json").write_text(json.dumps(runs[0]))
    (arm / "concurrency.txt").write_text("RESULT PASS: 0 failing\n")
    tally = []
    for n, d in enumerate(runs[1:], 1):
        (arm / "conc" / f"run{n}.json").write_text(json.dumps(d))
        tally.append(f"run\t{n}\t0\tRESULT PASS: 0 failing")
    (arm / "conc" / "runs.tsv").write_text("\n".join(tally) + "\n")
    # the counters are cumulative: the last line of a log is the total
    log = ["(EngineCore pid=7) INFO [runtime.py:431] hetero MoE hx status: record 0x0"]
    if hub is not None:
        early = "hub digest: " + COUNTERS.format(ok=3, bad=0, healed=0, unhealed=0)
        for text in (early, hub):
            log.append(f"(EngineCore pid=7) INFO [hx.py:392] hx hub hub0: {text}")
    (arm / "serve.log").write_text("\n".join(log) + "\n")
    servers = servers if servers is not None else {s: SERVER_OK for s in ready}
    for sid in ready:
        (arm / "hetero" / f"{sid}.ready").write_text("")
        text = servers.get(sid)
        line = "no digest here\n" if text is None else f"hx server {sid}: {text}\n"
        (arm / "hetero" / f"{sid}.log").write_text(line)
    return arm


def head(lines):
    return lines[0]


def test_a_clean_arm_passes_and_flips_are_only_information(gate, tmp_path):
    near_tie = item(
        [1, 2, 3, 4], top=[{}, {}, {"token_id:3": -0.5, "token_id:9": -0.9}, {}]
    )
    # together item 0 flips at token 2 (solo margin 0.4), item 1 at token 1 (the other
    # token is outside the solo top-5: confident)
    flipped = {"0": [1, 2, 9, 4], "1": [5, 9, 7, 8]}
    runs = [
        run_json(),
        run_json(together=flipped, solo_items={"0": near_tie}),
        run_json(),
    ]
    verdict, lines = gate.judge(str(make_arm(tmp_path, runs)), True, 1.0, True)
    assert verdict is True, lines
    assert head(lines) == (
        "HETERO-CONC-GATE PASS: solo and solo2 bit-identical in 3 runs, "
        "hx digest clean on the hub and every server"
    )
    text = "\n".join(lines)
    assert "digest hub: 90 ok, 0 bad at the first read" in text
    assert "digest exp1: 90 ok" in text
    assert "solo: 2/2 items bit-identical" in text
    assert "solo2 equal to solo in 6/6 items" in text
    assert (
        "together 6 items, 2 diverge from the T=3 solo, 0 at or after a special "
        "token, 1 confident"
    ) in text
    assert "long_together 0 items" in text
    assert "boot RESULT PASS" in text and "repeats 0 of 2 FAIL" in text
    assert (
        "run1 together 0 token 2: solo token_id:3 got token_id:9, solo margin 0.400"
        in text
    )


def test_a_flip_after_a_special_token_is_counted_not_failed(gate, tmp_path):
    solo = {"0": [248046, 2, 3, 4]}  # <|im_end|> first
    runs = [run_json(solo), run_json(solo, together={"0": [248046, 5, 3, 4]})]
    verdict, lines = gate.judge(str(make_arm(tmp_path, runs)), True, 1.0, False)
    assert verdict is True
    assert "1 diverge from the T=3 solo, 1 at or after a special token" in "\n".join(
        lines
    )


def test_solo_must_be_bit_identical_across_runs(gate, tmp_path):
    other = {"0": [1, 2, 3, 4], "1": [5, 6, 7, 9]}
    runs = [run_json(), run_json(), run_json(other, solo2=other)]
    verdict, lines = gate.judge(str(make_arm(tmp_path, runs)), True, 1.0, False)
    assert verdict is False
    assert "solo differs between runs for items ['1']" in head(lines)


def test_a_logprob_difference_alone_breaks_the_identity(gate, tmp_path):
    lps = [-0.1, -0.2, -0.3, -0.4000001]
    bumped = {"0": item([1, 2, 3, 4], lps=lps)}
    runs = [run_json(), run_json(solo_items=bumped, solo2=None)]
    verdict, lines = gate.judge(str(make_arm(tmp_path, runs)), True, 1.0, False)
    assert verdict is False
    text = head(lines)
    assert "solo differs between runs for items ['0']" in text
    assert (
        "solo2 differs from solo in 1 of 4 items" in text
    )  # run 2's solo2 is the old one


def test_solo2_must_equal_solo(gate, tmp_path):
    runs = [run_json(), run_json(solo2={"0": [1, 2, 3, 4], "1": [5, 6, 0, 8]})]
    verdict, lines = gate.judge(str(make_arm(tmp_path, runs)), True, 1.0, False)
    assert verdict is False
    assert (
        "solo2 differs from solo in 1 of 4 items (first: run1 item 1, first token "
        "difference at 2)"
    ) in head(lines)


@pytest.mark.parametrize(
    ("hub", "servers", "reason"),
    [
        (
            "hub digest: " + COUNTERS.format(ok=90, bad=1, healed=1, unhealed=0),
            None,
            "digest hub: 1 bad, 0 unhealed",
        ),
        (
            HUB_OK,
            {
                "exp0": SERVER_OK,
                "exp1": "server digest: "
                + COUNTERS.format(ok=90, bad=2, healed=0, unhealed=2),
            },
            "digest exp1: 2 bad, 2 unhealed",
        ),
        (None, None, "no digest line of hub"),
        (HUB_OK, {"exp0": SERVER_OK, "exp1": None}, "no digest line of exp1"),
    ],
)
def test_the_digest_decides_the_transport(gate, tmp_path, hub, servers, reason):
    arm = make_arm(tmp_path, [run_json(), run_json()], hub=hub, servers=servers)
    verdict, lines = gate.judge(str(arm), True, 1.0, False)
    assert verdict is False
    assert reason in head(lines)
    # without the digest requirement the same arm passes on the solo checks alone
    verdict, lines = gate.judge(str(arm), False, 1.0, False)
    assert verdict is True
    assert "digest: not required" in "\n".join(lines)


def test_no_server_is_no_pass(gate, tmp_path):
    arm = make_arm(tmp_path, [run_json()], ready=())
    verdict, lines = gate.judge(str(arm), True, 1.0, False)
    assert verdict is False and "no server (hetero/*.ready) found" in head(lines)


def test_the_last_counters_line_wins(gate, tmp_path):
    arm = make_arm(tmp_path, [run_json()])
    counters = gate.last_digest(str(arm / "serve.log"), "hub")
    assert counters == dict(ok=90, bad=0, healed=0, unhealed=0, late=0, rescans=4)
    assert gate.last_digest(str(arm / "serve.log"), "server") is None
    assert gate.last_digest(str(arm / "missing.log"), "hub") is None


def test_no_probe_results(gate, tmp_path):
    (tmp_path / "arm").mkdir()
    verdict, lines = gate.judge(str(tmp_path / "arm"), True, 1.0, False)
    assert verdict is None and head(lines).startswith("HETERO-CONC-GATE NO DATA")


@pytest.mark.parametrize(("hub", "code"), [(HUB_OK, 0), (None, 1)])
def test_exit_codes_and_first_line(tmp_path, hub, code):
    arm = make_arm(tmp_path, [run_json(), run_json()], hub=hub)
    result = subprocess.run(
        [sys.executable, str(GATE), str(arm)], capture_output=True, text=True
    )
    assert result.returncode == code, result.stdout + result.stderr
    first = result.stdout.splitlines()[0]
    assert first.startswith("HETERO-CONC-GATE " + ("PASS" if code == 0 else "FAIL"))
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [sys.executable, str(GATE), str(empty)], capture_output=True, text=True
    )
    assert result.returncode == 2
