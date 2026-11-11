# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the G0a descriptor arithmetic and the G0b layouts/patterns.

python tools/rdna/disagg/test_g0_desc.py   (plain asserts; also collected by pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g0_desc import digest, num_entries, resolve, row_starts  # noqa: E402


def brute(rows: np.ndarray) -> list[tuple[int, int, int]]:
    out = []
    for r, (addr, ln, _dev, stride, count) in enumerate(rows.tolist()):
        for k in range(count):
            out.append((addr + k * stride, ln, r))
    return out


def test_resolve_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 9))
        rows = np.zeros((n, 5), dtype=np.uint64)
        rows[:, 0] = rng.integers(1 << 30, 1 << 40, n)
        rows[:, 1] = rng.integers(1, 1 << 20, n)
        rows[:, 2] = rng.integers(0, 4, n)
        rows[:, 3] = rows[:, 1] + rng.integers(0, 1 << 12, n)
        rows[:, 4] = rng.integers(1, 40, n)
        full = brute(rows)
        assert num_entries(rows) == len(full)
        starts = row_starts(rows)
        ids = rng.permutation(len(full))[: max(1, len(full) // 2)]
        addr, ln, row = resolve(rows, starts, ids)
        for i, e in enumerate(ids.tolist()):
            assert (int(addr[i]), int(ln[i]), int(row[i])) == full[e]


def test_resolve_rejects_out_of_range() -> None:
    rows = np.array([[1000, 10, 0, 10, 3]], dtype=np.uint64)
    starts = row_starts(rows)
    assert resolve(rows, starts, [2])[0].tolist() == [1020]
    for bad in ([3], [-1]):
        try:
            resolve(rows, starts, bad)
        except IndexError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_digest_is_content_hash() -> None:
    a = np.arange(1000, dtype=np.uint8)
    b = a.copy()
    assert digest(a) == digest(b)
    b[500] ^= 1
    assert digest(a) != digest(b)


def test_probe_layouts_and_patterns() -> None:
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "port_v031")
    )
    import g0b_xfer_probe as p

    dev = torch.device("cpu")
    for kind in ("contig", "scatter", "strided"):
        size = p.parse_size("4M")
        lay = p.build_layout(kind, size, 7)
        assert lay.payload == sum(ln for _, ln in lay.windows)
        spans = sorted(lay.windows)
        assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:])), (
            "overlapping windows"
        )
        assert spans[-1][0] + spans[-1][1] <= lay.pool
        assert p.Layout.from_json(lay.to_json()) == lay
        base = p.make_base(lay.payload, dev)
        arena = torch.zeros(lay.pool, dtype=torch.uint8)
        salt = p.salt_of(3, 11)
        p.fill(arena, lay, base, salt)
        ok, bad = p.verify(arena, lay, base, salt)
        assert ok and bad == 0, kind
        arena[spans[len(spans) // 2][0] + 5] ^= 1
        ok, bad = p.verify(arena, lay, base, salt)
        assert not ok and bad >= 1, f"{kind}: a flipped byte went unnoticed"
        arena[spans[len(spans) // 2][0] + 5] ^= 1
        if kind != "contig":
            # a stray write between windows must be caught by the gap check
            gap = spans[0][0] + spans[0][1] + 8
            arena[gap] = 1
            ok, _ = p.verify(arena, lay, base, salt)
            assert not ok, "a write outside the windows went unnoticed"
        base2 = p.make_base(lay.payload, dev)
        assert torch.equal(base, base2), "the base pattern is not deterministic"
    s0, s1 = p.salt_of(0, 1), p.salt_of(1, 1)
    assert s0 != s1 and -(1 << 63) <= s0 < (1 << 63)


def _run(tokens: list[str], lps: list[float], top_gap: float = 1.0) -> dict:
    return {
        "tokens": tokens,
        "token_logprobs": lps,
        "top": [{t: lp, "other": lp - top_gap} for t, lp in zip(tokens, lps)],
    }


def test_compare_runs() -> None:
    import g0a_requests as rq

    n = rq.MAX_TOKENS
    toks = [f"token_id:{i}" for i in range(n)]
    lps = [-0.1 - 0.01 * (i % 7) for i in range(n)]
    ref = _run(toks, lps)
    same = rq.compare_runs(ref, _run(list(toks), list(lps)))
    assert same["exact"] and same["bitwise_logprobs"] and same["finite"]
    assert same["identical_prefix"] == n and same["ppl_ref"] == same["ppl_test"]
    # tiny logprob noise keeps the tokens exact but is reported as not bitwise
    noisy = rq.compare_runs(ref, _run(list(toks), [x + 1e-4 for x in lps]))
    assert noisy["exact"] and not noisy["bitwise_logprobs"]
    assert abs(noisy["ppl_ref"] - noisy["ppl_test"]) < 0.02
    # a divergence at token 5 is exact=False and reports the near-tie gap
    bad_toks = list(toks)
    bad_toks[5] = "token_id:other"
    div = rq.compare_runs(ref, _run(bad_toks, list(lps), top_gap=0.002))
    assert not div["exact"] and div["identical_prefix"] == 5
    assert abs(div["divergence_top2_gap_ref"] - 1.0) < 1e-9
    assert div["ref_token"] == toks[5] and div["test_token"] == "token_id:other"
    # NaN anywhere is flagged
    nan = rq.compare_runs(ref, _run(list(toks), [*lps[:-1], float("nan")]))
    assert not nan["finite"]
    # a short test run can never be exact
    short = rq.compare_runs(ref, _run(toks[:10], lps[:10]))
    assert not short["exact"] and short["identical_prefix"] == 10


def test_compare_cmd_controls() -> None:
    import argparse
    import json
    import tempfile

    import g0a_requests as rq

    n = rq.MAX_TOKENS
    toks = [f"token_id:{i}" for i in range(n)]
    lps = [-0.2] * n
    hit_toks = list(toks)
    hit_toks[7] = "token_id:alt"
    runs = []
    for length in rq.LENGTHS:
        for rep, tk in ((0, toks), (1, toks), ("hit", hit_toks)):
            r = _run(list(tk), list(lps))
            r.update({"length": length, "rep": rep})
            runs.append(r)
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "base.json")
        out = os.path.join(d, "aa.json")
        with open(ref, "w") as f:
            json.dump({"runs": runs}, f)
        a = argparse.Namespace(ref=ref, test=ref, aa=True, out=out)
        assert rq.cmd_compare(a) == 0  # controls never fail the arm
        with open(out) as f:
            rep = json.load(f)
        assert all(x["exact"] for x in rep["A/A'"])
        hit = rep["prefix-hit"]
        assert all(not x["exact"] and x["identical_prefix"] == 7 for x in hit)
        # a decode-style test file (no "rep") against the reference
        dec = [dict(_run(list(toks), list(lps)), length=n_) for n_ in rq.LENGTHS]
        tf = os.path.join(d, "decode.json")
        with open(tf, "w") as f:
            json.dump({"runs": dec}, f)
        a = argparse.Namespace(ref=ref, test=tf, aa=False, out=None)
        assert rq.cmd_compare(a) == 0
        dec[2] = dict(_run(hit_toks, list(lps)), length=rq.LENGTHS[2])
        with open(tf, "w") as f:
            json.dump({"runs": dec}, f)
        assert rq.cmd_compare(a) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
