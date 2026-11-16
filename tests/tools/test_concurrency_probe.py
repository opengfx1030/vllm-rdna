# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The concurrency probe must pass a correct but noisy server and fail a
corrupt one.

Flash-Next in production mode adds the experts of a token in packed-fp16 atomics
in arrival order, so two identical requests differ in their logprobs and the
model turns that into flips at near-tied tokens: the alternative's logprob moves
by a nat or more between two runs of the same request (docs/rdna2/
concurrency-probe-noise-2026-11-12.md). Corrupt state (a batch row that reads
another slot) makes both runs sure of different tokens.

Two layers, no GPU and no vLLM import:

* ``compare`` / ``flip_margins`` / ``same_input_noise`` / ``stop_index`` on
  hand-built results, and the pure helpers of ``conc_triage.py``;
* the probe's ``main`` against a fake completions server whose "model" is a
  fixed logit table per (prompt, step) with an optional per-request noise, a
  batch-shape shift or a slot mix-up applied when requests overlap.
"""

import importlib.util
import json
import math
import random
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tools/rdna/port_v031/concurrency_probe.py"
_spec = importlib.util.spec_from_file_location("concurrency_probe", PROBE)
assert _spec is not None and _spec.loader is not None
cp = importlib.util.module_from_spec(_spec)
sys.modules["concurrency_probe"] = cp
_spec.loader.exec_module(cp)

ARGS = types.SimpleNamespace(
    ignore_eos=True,
    tie_nats=0.2,
    fail_nats=1.0,
    max_lp_drop=1.5,
    repeat_lp_slack=0.5,
)


def tid(n: int) -> str:
    return f"token_id:{n}"


def run(tokens: list[int], tops: list[dict[int, float]], text: str = "") -> dict:
    """A probe result: ``tops[i]`` maps token id -> logprob at step i."""
    return {
        "tokens": [tid(t) for t in tokens],
        "top": [{tid(k): v for k, v in top.items()} for top in tops],
        "lps": [top[t] for t, top in zip(tokens, tops)],
        "text": text or " ".join(f"w{t}" for t in tokens),
        "finish_reason": "length",
        "stop_reason": None,
    }


def flip(solo_top: dict[int, float], got_top: dict[int, float]) -> tuple[dict, dict]:
    """One-step results whose tokens are the argmax of each top (token 1 vs 2)."""
    ref = run([1], [solo_top])
    got = run([2], [got_top])
    return ref, got


# --------------------------------------------------------------------------- units
def test_flip_margins_read_both_runs_exactly():
    ref, got = flip({1: -0.2, 2: -2.3, 3: -3.0}, {2: -0.5, 1: -0.7, 3: -4.0})
    margins = cp.flip_margins(ref, got, 0)
    assert margins.solo == pytest.approx(2.1)
    assert margins.got == pytest.approx(0.2)
    assert margins.exact


def test_flip_margins_bound_a_token_missing_from_the_top_k():
    # the solo run never listed token 2: only "at least 0.9 nats below" is known
    ref, got = flip({1: -1.6, 3: -1.7, 4: -2.5}, {2: -1.8, 1: -2.1, 3: -2.2})
    margins = cp.flip_margins(ref, got, 0)
    assert margins.solo == pytest.approx(0.9)
    assert margins.got == pytest.approx(0.3)
    assert not margins.solo_exact and margins.got_exact and not margins.exact
    # only the side that is a bound is marked as one
    status, detail = cp.compare(ref, got, ARGS, noisy=True)
    assert status == "soft" and "solo >0.900 got 0.300" in detail


def test_flip_margins_without_logprobs_is_infinitely_sure():
    ref = run([1], [{1: -0.1}])
    ref["top"] = [{}]
    got = run([2], [{2: -0.1}])
    got["top"] = [{}]
    margins = cp.flip_margins(ref, got, 0)
    assert math.isinf(margins.solo) and math.isinf(margins.got)
    assert not margins.exact


def test_same_input_noise_is_zero_for_identical_runs():
    first = run([1, 2], [{1: -0.1, 5: -3.0}, {2: -0.2, 6: -2.0}])
    second = run([1, 2], [{1: -0.1, 5: -3.0}, {2: -0.2, 6: -2.0}])
    assert cp.same_input_noise(first, second) == (0.0, False)


def test_same_input_noise_measures_the_biggest_shift_of_a_shared_token():
    first = run([1, 2], [{1: -0.1, 5: -3.0}, {2: -0.2, 6: -2.0}])
    second = run([1, 2], [{1: -0.1, 5: -3.0}, {2: -0.25, 6: -3.5}])
    noise, diverged = cp.same_input_noise(first, second)
    assert noise == pytest.approx(1.5)
    assert not diverged


def test_same_input_noise_reports_divergence():
    first = run([1, 2], [{1: -0.1, 5: -3.0}, {2: -0.2, 6: -2.0}])
    second = run([1, 6], [{1: -0.1, 5: -3.0}, {6: -0.2, 2: -0.9}])
    noise, diverged = cp.same_input_noise(first, second)
    assert diverged
    assert noise == pytest.approx(1.8)  # token 2 at the diverging step


def test_noisy_server_flip_the_solo_run_is_sure_of_is_a_warning():
    # the haiku at its stop token (docs): solo 2.1 nats sure, the other run on a tie
    ref, got = flip({1: -0.2, 2: -2.3, 3: -5.0}, {2: -0.5, 1: -0.9, 3: -5.0})
    status, detail = cp.compare(ref, got, ARGS, noisy=True)
    assert status == "soft", detail
    assert "solo 2.100 got 0.400" in detail


def test_noisy_server_flip_both_runs_are_sure_of_fails():
    ref, got = flip({1: -0.2, 2: -3.3, 3: -5.0}, {2: -0.2, 1: -2.5, 3: -5.0})
    status, detail = cp.compare(ref, got, ARGS, noisy=True)
    assert status == "FAIL", detail


def test_bit_identical_server_flip_the_solo_run_is_sure_of_fails():
    # deterministic MoE epilogue: nothing explains a 1.2 nat flip but the batch
    ref, got = flip({1: -0.4, 2: -1.6, 3: -5.0}, {2: -0.9, 1: -0.9, 3: -5.0})
    assert cp.compare(ref, got, ARGS, noisy=False)[0] == "FAIL"
    assert cp.compare(ref, got, ARGS, noisy=True)[0] == "tie"


def test_flat_distribution_with_a_token_outside_the_top_k_is_not_a_failure():
    # B1 of 2026-11-12: " of" was absent from the solo top-5 and "." from the
    # other run's; both runs sat within 0.9 nats of the 5th entry
    ref, got = flip(
        {1: -1.62, 3: -1.65, 4: -2.21, 5: -2.29, 6: -2.51},
        {2: -1.82, 4: -2.0, 3: -2.02, 5: -2.07, 6: -2.41},
    )
    status, detail = cp.compare(ref, got, ARGS, noisy=True)
    assert status == "soft", detail
    assert ">0.890" in detail and ">0.590" in detail


def test_token_far_outside_the_top_k_of_a_sure_solo_run_still_fails():
    top = {1: -0.05, **{k: -6.0 - k * 0.1 for k in range(10, 29)}}
    ref, got = flip(top, {2: -0.1, 1: -4.5, 3: -6.0})
    status, detail = cp.compare(ref, got, ARGS, noisy=True)
    assert status == "FAIL", detail
    assert ">" in detail


def test_flip_already_seen_between_two_solo_runs_is_noise():
    ref, got = flip({1: -0.2, 2: -3.3, 3: -5.0}, {2: -0.2, 1: -2.5, 3: -5.0})
    assert cp.compare(ref, got, ARGS, noise_at=0, noisy=True)[0] == "noise"


def test_near_tie_flips_are_classified_by_the_less_sure_run():
    ref, got = flip({1: -0.5, 2: -0.6, 3: -5.0}, {2: -0.4, 1: -0.9, 3: -5.0})
    assert cp.compare(ref, got, ARGS, noisy=True)[0] == "tie"
    ref, got = flip({1: -0.5, 2: -1.1, 3: -5.0}, {2: -0.4, 1: -1.0, 3: -5.0})
    assert cp.compare(ref, got, ARGS, noisy=True)[0] == "soft"


def test_garbage_and_quality_drop_fail_whatever_the_margins():
    toks = list(range(10, 20))
    tops = [{t: -0.1, 99: -4.0} for t in toks]
    ref = run(toks, tops)
    # same first token, then the other run falls apart: low-confidence salad
    got_toks = [11] + [30] * 9
    got_tops = [{11: -0.1, 99: -4.0}] + [{30: -2.9, 99: -3.0}] * 9
    got = run(got_toks, got_tops)
    status, detail = cp.compare(ref, got, ARGS, noisy=True)
    assert status == "FAIL" and ("garbage" in detail or "quality drop" in detail)


def test_stop_index_is_where_the_forced_run_has_the_natural_stop_token():
    ids = [5, 6, 7, 99, 8, 9]
    forced = run(ids, [{t: -0.1} for t in ids])
    natural = run(ids[:4], [{t: -0.1} for t in ids[:4]])
    natural["finish_reason"] = "stop"
    assert cp.stop_index(forced, natural) == 3
    natural["finish_reason"] = "length"  # never stopped: nothing to cut at
    assert cp.stop_index(forced, natural) is None
    natural["finish_reason"] = "stop"
    forced_other = run([5, 6, 7, 8], [{t: -0.1} for t in [5, 6, 7, 8]])
    assert cp.stop_index(forced_other, natural) is None


def test_flip_after_the_solo_stop_token_is_reported_not_judged():
    ref = run([5, 6, 99, 8], [{t: -0.1, 40: -4.0} for t in [5, 6, 99, 8]])
    got = run([5, 6, 99, 9], [{t: -0.1, 40: -4.0} for t in [5, 6, 99, 9]])
    assert cp.compare(ref, got, ARGS, noisy=True)[0] == "FAIL"
    status, detail = cp.compare(ref, got, ARGS, noisy=True, stop_at=2)
    assert status == "soft" and "after the solo stop token at 2" in detail
    # a flip at the stop token itself is the decision to stop: judged
    assert cp.compare(ref, got, ARGS, noisy=True, stop_at=3)[0] == "FAIL"


_triage_spec = importlib.util.spec_from_file_location(
    "conc_triage", ROOT / "tools/rdna/port_v031/conc_triage.py"
)
assert _triage_spec is not None and _triage_spec.loader is not None
ct = importlib.util.module_from_spec(_triage_spec)
_triage_spec.loader.exec_module(ct)


def test_triage_legacy_rule_is_the_margin_rule_the_probe_used_to_apply():
    # solo margin above 1 nat: fail; a near-tie: not
    ref, got = flip({1: -0.2, 2: -2.3, 3: -5.0}, {2: -0.5, 1: -0.9})
    assert ct.legacy_fails(ref, got, 0)
    ref, got = flip({1: -0.2, 2: -0.9, 3: -5.0}, {2: -0.5, 1: -0.9})
    assert not ct.legacy_fails(ref, got, 0)
    # a token that is only 6th in the solo top-k is outside its top-5: fail,
    # where the probe now reads the margin (0.5) from the top-20
    top = {1: -1.0, 3: -1.1, 4: -1.2, 5: -1.3, 6: -1.4, 2: -1.5}
    ref, got = flip(top, {2: -0.5, 1: -0.9})
    assert ct.legacy_fails(ref, got, 0)
    assert cp.compare(ref, got, ARGS, noisy=True)[0] == "soft"


def test_triage_checks_walk_the_probe_phases():
    solo = {"0": run([1, 2], [{1: -0.1}, {2: -0.1}])}
    flipped = run([1, 3], [{1: -0.1}, {3: -0.1}])
    saved = {
        "prompts": ["p0"],
        "solo": solo,
        "solo2": {"0": flipped},
        "together": {"0": flipped},
        "churn": {"[0, 0]": flipped},
        "long_prompts": ["long"],
        "long_solo": {"0": solo["0"]},
        "long_together": {"0": flipped},
    }
    walked = [
        (phase, key, i, noise) for phase, key, i, _, _, noise, _ in ct.checks(saved)
    ]
    assert walked == [
        ("solo2", 0, 0, 1),
        ("together", 0, 0, 1),
        ("churn", (0, 0), 0, 1),
        ("long", 0, 0, None),
    ]


# ------------------------------------------------------------------- fake server
VOCAB = 300
STOP_ID = VOCAB - 1
STOP_STEP = 20  # the first two prompts end their turn here, like the real ones
STOPPING = set(cp.PROMPTS[:2])


class FakeModel:
    """A fixed logit table per (prompt, step): the top token wins by a big
    margin at most steps and by a near-tie at some. Modes:

    exact        identical requests give identical logprobs
    noisy        every request adds its own noise: a heavy-tailed shift of one
                 competitor's logit at 5% of the steps and, past the stop token
                 that --ignore-eos forces through, fresh random logits
    batch_shape  exact, but overlapping requests see the runner-up of one step
                 pushed 1.3 nats past the top token
    slot_mixup   exact, but overlapping requests read another prompt's logits
    noisy_mixup  noisy, with the slot mix-up on overlapping requests
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.lock = threading.Lock()
        self.inflight = 0
        self.served: dict[tuple, int] = {}

    def table(self, prompt: str, step: int) -> tuple[int, float, list[float]]:
        rng = random.Random(f"{prompt}|{step}")
        top, runner = rng.sample(range(VOCAB - 1), 2)
        gap = rng.choice([4.0, 3.0, 2.5, 0.05, 0.3, 1.5, 0.1])
        logits = [rng.gauss(0.0, 1.0) - 2.0 for _ in range(VOCAB)]
        logits[top] = 3.0
        logits[runner] = 3.0 - gap
        return runner, gap, logits

    def answer(
        self, prompt: str, max_tokens: int, top_k: int, concurrent: bool, forced: bool
    ):
        with self.lock:
            nth = self.served.get((prompt, max_tokens), 0)
            self.served[(prompt, max_tokens)] = nth + 1
        mixed = self.mode.endswith("mixup") and concurrent
        source = prompt + "#other" if mixed else prompt
        tokens, lps, tops = [], [], []
        for step in range(max_tokens):
            runner, gap, logits = self.table(source, step)
            noise = random.Random(f"{prompt}|{step}|{nth}")
            if prompt in STOPPING and step == STOP_STEP:
                logits[STOP_ID] = max(logits) + 3.0  # the model ends its turn
            if self.mode.startswith("noisy"):
                if prompt in STOPPING and step > STOP_STEP:
                    if nth and noise.random() < 0.3:  # junk: logits of its own
                        logits = [noise.gauss(0.0, 4.0) for _ in range(VOCAB)]
                elif prompt not in STOPPING:  # these stay exact up to the stop
                    logits = [v + noise.gauss(0.0, 0.01) for v in logits]
                    if noise.random() < 0.05:  # a competitor's logit jumps
                        shift = noise.choice([-1, 1]) * noise.uniform(0.5, 3.5)
                        logits[runner] += shift
            if self.mode == "batch_shape" and concurrent and step == 11:
                logits[runner] += gap + 1.3  # the tuned-GEMM shape effect
            lse = math.log(sum(math.exp(v) for v in logits))
            best = sorted(range(VOCAB), key=lambda t: -logits[t])
            tokens.append(best[0])
            lps.append(logits[best[0]] - lse)
            tops.append({f"token_id:{t}": logits[t] - lse for t in best[:top_k]})
            if best[0] == STOP_ID and not forced:
                return tokens, lps, tops, "stop"
        return tokens, lps, tops, "length"


class Handler(BaseHTTPRequestHandler):
    model: FakeModel

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = self.model
        with model.lock:
            model.inflight += 1
        try:
            time.sleep(0.004 * body["max_tokens"])  # long enough to overlap
            with model.lock:
                concurrent = model.inflight > 1
            tokens, lps, tops, finish = model.answer(
                body["prompt"],
                body["max_tokens"],
                body["logprobs"],
                concurrent,
                body.get("ignore_eos", False),
            )
        finally:
            with model.lock:
                model.inflight -= 1
        choice = {
            "text": " ".join(f"w{t}" for t in tokens),
            "finish_reason": finish,
            "stop_reason": None,
            "logprobs": {
                "tokens": [f"token_id:{t}" for t in tokens],
                "token_logprobs": lps,
                "top_logprobs": tops,
            },
        }
        data = json.dumps({"choices": [choice]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def serve():
    servers = []

    def start(mode: str) -> str:
        handler = type("H", (Handler,), {"model": FakeModel(mode)})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/v1/completions"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def probe(url: str, *extra: str) -> tuple[int, str]:
    done = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--url",
            url,
            "--model",
            "fake",
            "--num-prompts",
            "6",
            "--max-tokens",
            "48",
            "--long-prompts",
            "2",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return done.returncode, done.stdout


def test_probe_passes_a_bit_identical_server(serve):
    code, out = probe(serve("exact"))
    assert code == 0, out
    assert "RESULT PASS: 0 failing" in out
    assert "run-to-run bit-identical" in out


def judged(data: dict, **kwargs) -> list[str]:
    """Status of every ``together`` result against its solo run."""
    return [
        cp.compare(data["solo"][k], got, ARGS, **kwargs)[0]
        for k, got in data["together"].items()
    ]


def test_probe_passes_a_noisy_server_that_the_old_judging_rejects(serve, tmp_path):
    saved = tmp_path / "run.json"
    code, out = probe(serve("noisy"), "--save-results", str(saved))
    assert code == 0, out
    assert "RESULT PASS: 0 failing" in out
    assert "run-to-run noise" in out and "both runs are sure" in out
    assert "after the solo stop token" in out

    data = json.loads(saved.read_text())
    # the old rule, the solo margin alone, fails the noise ...
    assert "FAIL" in judged(data, noisy=False)
    # ... and both margins alone still fail the junk after the stop token
    assert "FAIL" in judged(data, noisy=True)


def test_probe_fails_a_bit_identical_server_with_a_batch_shape_flip(serve):
    code, out = probe(serve("batch_shape"))
    assert code == 1, out
    assert "FAIL together" in out and "run-to-run bit-identical" in out


@pytest.mark.parametrize("mode", ["slot_mixup", "noisy_mixup"])
def test_probe_fails_a_slot_mixup(serve, mode):
    code, out = probe(serve(mode))
    assert code == 1, out
    assert "FAIL together" in out
    assert ("run-to-run noise" in out) == (mode == "noisy_mixup")
