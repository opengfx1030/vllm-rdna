# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tools/rdna/port_v031/topk_kl.py and agentic_bench.py.

topk_kl estimates KL(ref || cand) from vLLM ``prompt_logprobs`` top-k lists. The
tests build synthetic full-vocabulary distributions, render them the way the
server does (top-k entries plus the scored token with its rank), and check the
estimator against the exact full-vocabulary values. CPU only (numpy).
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"tools/rdna/port_v031/{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TK = _load("topk_kl")
AB = _load("agentic_bench")
VOCAB, LENGTH, TOP = 3000, 257, 20


def _logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def _entries(lp: np.ndarray, window: list[int]) -> list:
    """Server-style prompt_logprobs: None, then top-k plus the scored token."""
    out: list = [None]
    for j in range(1, len(window)):
        row = lp[j - 1]
        order = np.argsort(-row, kind="stable")[:TOP]
        d = {
            str(int(t)): {"logprob": float(row[t]), "rank": r + 1}
            for r, t in enumerate(order)
        }
        act = int(window[j])
        d[str(act)] = {
            "logprob": float(row[act]),
            "rank": int((row > row[act]).sum()) + 1,
        }
        out.append(d)
    return out


def _pair(scale: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3.0, size=(LENGTH - 1, VOCAB))
    lp_a = _logsoftmax(logits)
    lp_b = _logsoftmax(logits + rng.normal(0, scale, size=logits.shape))
    # scored tokens sampled from the reference, so some fall outside the top-k
    window = [0] + [int(rng.choice(VOCAB, p=np.exp(r))) for r in lp_a]
    act = np.asarray(window[1:], np.int32)
    ref = TK.parse_window(window, _entries(lp_a, window), TOP)
    cand = TK.parse_window(window, _entries(lp_b, window), TOP)
    ref = {k: v[None] for k, v in ref.items()} | {"act_id": act[None]}
    cand = {k: v[None] for k, v in cand.items()} | {"act_id": act[None]}
    return lp_a, lp_b, act, ref, cand


@pytest.mark.parametrize("scale", [0.02, 0.2, 1.0])
def test_kl_is_a_lower_bound(scale):
    lp_a, lp_b, act, ref, cand = _pair(scale)
    st = TK.window_stats(ref, cand, 0)
    exact = (np.exp(lp_a) * (lp_a - lp_b)).sum(-1)
    assert np.all(st["kl"] >= -1e-12)
    assert np.all(st["kl"] <= exact + 1e-9)
    # the bound keeps most of the divergence when the top-k holds most of the mass
    assert st["kl"].mean() > 0.4 * exact.mean()
    rows = np.arange(LENGTH - 1)
    np.testing.assert_allclose(st["dlp"], lp_b[rows, act] - lp_a[rows, act])
    assert np.array_equal(st["top1"], lp_a.argmax(-1) == lp_b.argmax(-1))
    assert np.all((st["coverage"] > 0) & (st["coverage"] <= 1))


def test_identical_runs_give_zero():
    _, _, _, ref, _ = _pair(0.5)
    st = TK.window_stats(ref, ref, 0)
    assert st["kl"].max() == 0.0
    assert st["top1"].all()
    assert not st["dlp"].any()
    res = TK.domain_pair(ref, ref, reps=200, worst=1)
    assert res["kl"] == 0.0 and res["dnll"] == 0.0 and res["top1"] == 1.0
    assert res["identical_positions"] == 1.0


def test_domain_pair_refuses_different_tokens():
    _, _, _, ref, cand = _pair(0.2)
    cand = dict(cand, act_id=cand["act_id"] + 1)
    with pytest.raises(SystemExit):
        TK.domain_pair(ref, cand, reps=10, worst=1)


def test_parse_window_rejects_short_lists():
    lp_a, _, _, _, _ = _pair(0.0)
    window = [0, 1, 2]
    entries = _entries(lp_a, window)
    key = str(window[1])
    others = [(k, v) for k, v in entries[1].items() if k != key][: TOP - 3]
    entries[1] = {key: entries[1][key], **dict(others)}
    with pytest.raises(ValueError):
        TK.parse_window(window, entries, TOP)


def test_system_prompt_prefix():
    chat = types.SimpleNamespace(
        prompt=lambda item: [{"role": "user", "content": item}],
        fingerprint=lambda: "v1",
    )
    chat_sys = types.SimpleNamespace(
        prompt=lambda item: [
            {"role": "system", "content": "task rules"},
            {"role": "user", "content": item},
        ]
    )
    text = types.SimpleNamespace(prompt=lambda item: f"Q: {item}")
    mods = {"chat": chat, "chat_sys": chat_sys, "text": text}
    registry = types.SimpleNamespace(NAMES=list(mods), get=mods.__getitem__)
    AB.prefix_tasks(registry, "AGENT\n")
    assert chat.prompt("x") == [
        {"role": "system", "content": "AGENT\n"},
        {"role": "user", "content": "x"},
    ]
    assert chat_sys.prompt("x")[0] == {
        "role": "system",
        "content": "AGENT\n\ntask rules",
    }
    assert text.prompt("x") == "Q: x"
    assert chat.fingerprint().startswith("v1+system:")
    assert chat_sys.fingerprint().startswith("+system:")
