#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request driver of the G0a spike (docs/rdna2/disagg-g0-2026-11-06.md).

  g0a_requests.py prompts  --model DIR --docs DIR --out prompts.json
  g0a_requests.py base     --url U --served S --prompts P --out base.json [--rpc-reset]
  g0a_requests.py prefill  --url U --served S --prompts P --store DIR --out p.json
  g0a_requests.py decode   --url U --served S --prompts P --pjson p.json \
                           --stats DIR --out d.json
  g0a_requests.py compare  --ref base.json --test d.json [--ref2 base2.json]

Prompts are token-id lists cut from disjoint windows of the repo's own docs, so
no two lengths share a prefix block and every request is a cold prefill. All
generation is greedy, 64 tokens, top-5 logprobs, ``ignore_eos``; tokens are
reported as ``token_id:<n>`` so comparisons are on ids, not detokenised text.

``prefill`` sends each prompt to a ``kv_producer`` instance with
``do_remote_decode``, then (producer still running) captures the side-channel
handshake payloads and dumps, through ``/collective_rpc``, the memory windows a
decode-side read of each request addresses (``g0a_ext.py``). ``decode`` replays
the saved ``kv_transfer_params`` against a ``kv_consumer`` instance.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

LENGTHS = (1024, 1025, 16384, 16385)
OFFSETS = (0, 20000, 40000, 62000)
MAX_TOKENS = 64
HERE = os.path.dirname(os.path.abspath(__file__))


def post(url: str, body: dict[str, Any], timeout: float = 3600) -> dict[str, Any]:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def rpc(base: str, method: str, args: list[str], timeout: float = 1800) -> list[Any]:
    out = post(
        base + "/collective_rpc",
        {"method": method, "args": args, "timeout": timeout},
        timeout=timeout + 60,
    )
    return out["results"]


def completion_body(served: str, ids: list[int], max_tokens: int) -> dict[str, Any]:
    return {
        "model": served,
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": 5,
        "ignore_eos": True,
        "return_tokens_as_token_ids": True,
    }


def summarize(resp: dict[str, Any], seconds: float) -> dict[str, Any]:
    choice = resp["choices"][0]
    lp = choice["logprobs"]
    return {
        "tokens": lp["tokens"],
        "token_logprobs": lp["token_logprobs"],
        "top": lp["top_logprobs"],
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "usage": resp.get("usage"),
        "seconds": round(seconds, 3),
    }


def load_prompts(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)["prompts"]


def cmd_prompts(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(Path(p).read_text() for p in sorted(glob.glob(f"{a.docs}/*.md")))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    need = max(o + n for o, n in zip(OFFSETS, LENGTHS))
    assert len(ids) >= need, f"docs give {len(ids)} tokens, need {need}"
    prompts = [
        {"length": n, "offset": o, "ids": ids[o : o + n]}
        for o, n in zip(OFFSETS, LENGTHS)
    ]
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "docs_tokens": len(ids), "prompts": prompts}, f)
    print(f"{len(ids)} doc tokens; wrote {len(prompts)} prompts to {a.out}")
    return 0


def cmd_base(a: argparse.Namespace) -> int:
    """Greedy generation per length. Reps 0..repeat-1 are cold (the prefix cache is
    reset before every rep after the first): rep 0 is the reference and rep 1 the
    A/A' noise-floor pair. The last rep, ``hit``, is not preceded by a reset: it
    resumes from the cached prompt state and recomputes only the tail, the same
    situation as the decode side after a transfer, without any transfer."""
    runs = []
    for p in load_prompts(a.prompts):
        for rep in [*range(a.repeat), "hit"]:
            if rep not in (0, "hit"):
                reset = post(a.url + "/reset_prefix_cache", {})
                print("reset_prefix_cache:", reset)
            t0 = time.time()
            resp = post(
                a.url + "/v1/completions",
                completion_body(a.served, p["ids"], MAX_TOKENS),
            )
            run = summarize(resp, time.time() - t0)
            run.update({"length": p["length"], "rep": rep})
            runs.append(run)
            print(
                f"L={p['length']} rep={rep}: {run['seconds']} s, "
                f"first tokens {run['tokens'][:4]}"
            )
    with open(a.out, "w") as f:
        json.dump({"kind": "base", "runs": runs}, f)
    return 0


def cmd_prefill(a: argparse.Namespace) -> int:
    base = a.url
    results = []
    regions = rpc(base, "g0a_regions", [])
    with open(os.path.join(a.store, "regions_producer.json"), "w") as f:
        json.dump(regions, f, indent=1)
    for p in load_prompts(a.prompts):
        body = completion_body(a.served, p["ids"], 1)
        body["kv_transfer_params"] = {"do_remote_decode": True}
        t0 = time.time()
        resp = post(base + "/v1/completions", body)
        secs = time.time() - t0
        params = resp.get("kv_transfer_params")
        assert params, f"producer returned no kv_transfer_params: {list(resp)}"
        print(f"L={p['length']}: prefill {secs:.2f} s; params keys {sorted(params)}")
        results.append(
            {
                "length": p["length"],
                "seconds": round(secs, 3),
                "kv_transfer_params": params,
                "usage": resp.get("usage"),
                "first_token": resp["choices"][0]["logprobs"]["tokens"],
            }
        )
    # Producer idle: capture the handshake payloads, then dump every request.
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "g0a_handshake_server.py"),
            "capture",
            "--store",
            a.store,
            "--port",
            str(a.side_channel_port),
            "--tp",
            str(a.tp),
        ],
        check=True,
    )
    for r in results:
        tag = f"L{r['length']}"
        out = rpc(base, "g0a_dump", [a.store, tag, json.dumps(r["kv_transfer_params"])])
        r["dump"] = out
        total = sum(x["bytes"] for x in out)
        print(
            f"{tag}: dumped {total / 2**20:.1f} MiB over {len(out)} ranks "
            f"({[round(x['bytes'] / 2**20, 1) for x in out]} MiB/rank)"
        )
    with open(a.out, "w") as f:
        json.dump({"kind": "prefill", "runs": results, "regions": regions}, f)
    return 0


def cmd_decode(a: argparse.Namespace) -> int:
    with open(a.pjson) as f:
        pruns = {r["length"]: r for r in json.load(f)["runs"]}
    regions = rpc(a.url, "g0a_regions", [])
    with open(os.path.join(a.stats, "regions_consumer.json"), "w") as f:
        json.dump(regions, f, indent=1)
    runs = []
    for p in load_prompts(a.prompts):
        params = dict(pruns[p["length"]]["kv_transfer_params"])
        params["remote_port"] = a.stand_in_port
        params["remote_host"] = "127.0.0.1"
        body = completion_body(a.served, p["ids"], MAX_TOKENS)
        body["kv_transfer_params"] = params
        t0 = time.time()
        try:
            resp = post(a.url + "/v1/completions", body)
        except Exception as e:  # noqa: BLE001 - record and continue with the next length
            print(f"L={p['length']}: request failed: {e!r}")
            runs.append({"length": p["length"], "error": repr(e)})
            continue
        run = summarize(resp, time.time() - t0)
        run["length"] = p["length"]
        runs.append(run)
        print(f"L={p['length']}: {run['seconds']} s, first tokens {run['tokens'][:4]}")
    with open(a.out, "w") as f:
        json.dump({"kind": "decode", "runs": runs, "regions": regions}, f)
    return 0


def cmd_rpc(a: argparse.Namespace) -> int:
    print(json.dumps(rpc(a.url, a.method, a.arg), indent=1))
    return 0


def ppl(run: dict[str, Any]) -> float:
    lps = [x for x in run["token_logprobs"] if x is not None]
    return math.exp(-sum(lps) / len(lps))


def compare_runs(ref: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    n = min(len(ref["tokens"]), len(test["tokens"]))
    div = next((k for k in range(n) if ref["tokens"][k] != test["tokens"][k]), n)
    deltas = [
        abs(ref["token_logprobs"][k] - test["token_logprobs"][k]) for k in range(div)
    ]
    out = {
        "tokens": [len(ref["tokens"]), len(test["tokens"])],
        "identical_prefix": div,
        "exact": div == n == MAX_TOKENS,
        "max_dlogprob_on_prefix": max(deltas) if deltas else 0.0,
        "bitwise_logprobs": all(d == 0.0 for d in deltas),
        "ppl_ref": ppl(ref),
        "ppl_test": ppl(test),
        "finite": all(
            x is not None and math.isfinite(x)
            for x in ref["token_logprobs"] + test["token_logprobs"]
        ),
    }
    if div < n:
        vals = sorted(ref["top"][div].values(), reverse=True)
        out["divergence_top2_gap_ref"] = vals[0] - vals[1] if len(vals) > 1 else None
        out["ref_token"] = ref["tokens"][div]
        out["test_token"] = test["tokens"][div]
    return out


def _compare_maps(
    ref: dict[Any, dict[str, Any]], test: dict[Any, dict[str, Any]], label: str
) -> tuple[bool, list[dict[str, Any]]]:
    ok = True
    report = []
    for key in sorted(test, key=str):
        t = test[key]
        r = ref.get(key) or ref.get((key[0], 0))
        if "error" in t or r is None:
            ok = False
            report.append({"length": key[0], "error": t.get("error", "no reference")})
            print(f"{label} L={key[0]}: ERROR {t.get('error')}")
            continue
        res = compare_runs(r, t)
        res["length"] = key[0]
        report.append(res)
        ok &= res["exact"] and res["finite"]
        print(
            f"{label} L={key[0]}: exact={res['exact']} "
            f"identical_prefix={res['identical_prefix']}/"
            f"{MAX_TOKENS} bitwise_logprobs={res['bitwise_logprobs']} "
            f"max|dlogprob|={res['max_dlogprob_on_prefix']:.5f} "
            f"ppl_ref={res['ppl_ref']:.4f} ppl_test={res['ppl_test']:.4f} "
            f"finite={res['finite']}"
            + (
                f" DIVERGES ref={res['ref_token']} test={res['test_token']} "
                f"top2gap={res['divergence_top2_gap_ref']}"
                if "ref_token" in res
                else ""
            )
        )
    return ok, report


def cmd_compare(a: argparse.Namespace) -> int:
    with open(a.ref) as f:
        ref = {(r["length"], r.get("rep", 0)): r for r in json.load(f)["runs"]}
    if a.aa:
        # controls inside the reference file: rep 1 (cold again after a reset) and
        # ``hit`` (prefix-cache hit, tail recompute) against rep 0
        ref0 = {(n, 0): r for (n, rep), r in ref.items() if rep == 0}
        ok, report = True, {}
        for label, rep in (("A/A'", 1), ("prefix-hit", "hit")):
            test = {(n, 0): r for (n, rp), r in ref.items() if rp == rep}
            if not test:
                continue
            ok_c, report[label] = _compare_maps(ref0, test, label)
            ok &= ok_c
    else:
        with open(a.test) as f:
            test = {(r["length"], r.get("rep", 0)): r for r in json.load(f)["runs"]}
        ok, report = _compare_maps(ref, test, "D vs ref")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    note = "(controls: informational)" if a.aa else ""
    print("COMPARE", "PASS" if ok else "FAIL", note)
    return 0 if ok or a.aa else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts")
    p.add_argument("--model", required=True)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prompts)

    for name, fn in (
        ("base", cmd_base),
        ("prefill", cmd_prefill),
        ("decode", cmd_decode),
    ):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True)
        p.add_argument("--served", default="flash-next")
        p.add_argument("--prompts", required=True)
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
        if name == "base":
            p.add_argument("--repeat", type=int, default=2)
        if name == "prefill":
            p.add_argument("--store", required=True)
            p.add_argument("--tp", type=int, default=4)
            p.add_argument("--side-channel-port", type=int, default=5600)
        if name == "decode":
            p.add_argument("--pjson", required=True)
            p.add_argument("--stats", required=True)
            p.add_argument("--stand-in-port", type=int, default=5601)

    p = sub.add_parser("rpc", help="call a worker-extension method on every rank")
    p.add_argument("--url", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--arg", action="append", default=[])
    p.set_defaults(fn=cmd_rpc)

    p = sub.add_parser("compare")
    p.add_argument("--ref", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out")
    p.add_argument("--aa", action="store_true", help="repeat 0 vs repeat 1 of --ref")
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
