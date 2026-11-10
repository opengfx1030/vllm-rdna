#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request driver of the G0a / DisaggRework spikes (docs/rdna2/disagg-g0-2026-11-06.md).

  g0a_requests.py prompts  --model DIR --docs DIR --out prompts.json [--spec 1025:3,..]
  g0a_requests.py base     --url U --served S --prompts P --out base.json
                           [--trace-dir D]
  g0a_requests.py prefill  --url U --served S --prompts P --store DIR --out p.json
  g0a_requests.py decode   --url U --served S --prompts P --pjson p.json \
                           --stats DIR --out d.json [--trace-dir D]
  g0a_requests.py compare  --ref base.json --test d.json [--ref-rep hit]

Prompts are token-id lists cut from disjoint windows of the repo's own docs, so no two
prompts share a prefix block and every request is a cold prefill; each has an ``id``
(``L<length>_<k>``). All generation is greedy, 64 tokens, top-20 logprobs,
``ignore_eos``; tokens are reported as ``token_id:<n>`` so comparisons are on ids, not
detokenised text. ``compare`` reports, besides the token and logprob agreement, whether
the first token's top-20 distribution is bitwise equal: the first token is sampled from
the one step that runs the handed-over state and involves no drafts.

``prefill`` captures the side-channel handshake payloads of a ``kv_producer`` instance,
then sends each prompt with ``do_remote_decode`` and, right after the response, dumps
through ``/collective_rpc`` the memory windows a decode-side read of that request
addresses (``g0a_ext.py``). ``decode`` replays the saved ``kv_transfer_params`` against
a ``kv_consumer`` instance. With ``--trace-dir`` every request of ``base``/``prefill``/
``decode`` is wrapped in ``g0r_begin``/``g0r_end`` (``g0r_trace.py``).
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

DEFAULT_SPEC = (
    "1025:3,2049:3,3073:3,2048:1,4097:1,8193:1,16384:1,16385:1,32769:1,65537:1"
)
MAX_TOKENS = 64
LOGPROBS = 20
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


def trace_begin(
    a: argparse.Namespace, tag: str, length: int, *, raw: bool, raw_at: int | None
) -> None:
    """Arm the worker-side state tracer for the next request (``--trace-dir``)."""
    if not getattr(a, "trace_dir", None):
        return
    plan = {"raw": raw, "raw_at": raw_at, "length": length}
    try:
        rpc(a.url, "g0r_begin", [a.trace_dir, tag, json.dumps(plan)], timeout=300)
    except Exception as e:  # noqa: BLE001 - tracing is a diagnostic, never a verdict
        print(f"trace_begin {tag} failed: {e!r}")


def trace_end(a: argparse.Namespace, tag: str) -> None:
    if not getattr(a, "trace_dir", None):
        return
    try:
        res = rpc(a.url, "g0r_end", [], timeout=300)
    except Exception as e:  # noqa: BLE001
        print(f"trace_end {tag} failed: {e!r}")
        return
    errs = [r["last_error"] for r in res if r and r.get("errors")]
    print(
        f"trace {tag}: steps/rank {[r['steps'] for r in res]}"
        + (f" ERRORS {errs[0]}" if errs else "")
    )


SPEC_COUNTERS = ("drafts", "draft_tokens", "accepted_tokens")


def spec_metrics(base: str) -> dict[str, float] | None:
    """Sum of the server's speculative-decoding counters (None without /metrics)."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
            text = r.read().decode()
    except OSError:
        return None
    out = dict.fromkeys(SPEC_COUNTERS, 0.0)
    for line in text.splitlines():
        for k in SPEC_COUNTERS:
            if line.startswith(f"vllm:spec_decode_num_{k}_total"):
                out[k] += float(line.rsplit(" ", 1)[1])
    return out


def timed_completion(
    base: str, body: dict[str, Any]
) -> tuple[dict[str, Any], float, dict[str, float] | None]:
    """POST one completion; also return the speculative counters it moved."""
    m0 = spec_metrics(base)
    t0 = time.time()
    resp = post(base + "/v1/completions", body)
    secs = time.time() - t0
    time.sleep(0.5)  # the engine's stat logger runs after the response is sent
    m1 = spec_metrics(base)
    spec = {k: m1[k] - m0[k] for k in SPEC_COUNTERS} if m0 and m1 else None
    return resp, secs, spec


def completion_body(served: str, ids: list[int], max_tokens: int) -> dict[str, Any]:
    return {
        "model": served,
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": LOGPROBS,
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


def load_prompts(path: str, only: str | None = None) -> list[dict[str, Any]]:
    with open(path) as f:
        prompts = json.load(f)["prompts"]
    if only:
        keep = set(only.split(","))
        prompts = [p for p in prompts if p["id"] in keep]
    return prompts


def cmd_prompts(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    text = "\n\n".join(Path(p).read_text() for p in sorted(glob.glob(f"{a.docs}/*.md")))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    prompts = []
    offset = a.start
    for item in a.spec.split(","):
        length, count = (int(x) for x in item.split(":"))
        for k in range(count):
            prompts.append(
                {
                    "id": f"L{length}_{k}",
                    "length": length,
                    "offset": offset,
                    "ids": ids[offset : offset + length],
                }
            )
            offset += length
    assert len(ids) >= offset, f"docs give {len(ids)} tokens, need {offset}"
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "docs_tokens": len(ids), "prompts": prompts}, f)
    print(f"{len(ids)} doc tokens; wrote {len(prompts)} prompts to {a.out}")
    return 0


def is_raw_id(prompt_id: str) -> bool:
    """Raw state pages are saved for the first prompt of every length only."""
    return prompt_id.endswith("_0")


def cmd_base(a: argparse.Namespace) -> int:
    """Greedy generation per prompt. Reps 0..repeat-1 are cold (the prefix cache is
    reset before every rep after the first): rep 0 is the reference and rep 1 the
    A/A' noise-floor pair (prompts longer than ``--repeat-max-len`` run rep 0 only).
    The last rep, ``hit``, is not preceded by a reset: it resumes from the cached
    prompt state and recomputes only the tail, the same situation as the decode side
    after a transfer, without any transfer."""
    runs = []
    for p in load_prompts(a.prompts, a.only):
        n = p["length"]
        repeat = a.repeat if n <= a.repeat_max_len else 1
        for rep in [*range(repeat), "hit"]:
            if rep not in (0, "hit"):
                reset = post(a.url + "/reset_prefix_cache", {})
                print("reset_prefix_cache:", reset)
            tag = f"{p['id']}_{'hit' if rep == 'hit' else f'cold{rep}'}"
            traced = rep in (0, "hit")
            if traced:
                trace_begin(a, tag, n, raw=is_raw_id(p["id"]), raw_at=n - 1)
            resp, secs, spec = timed_completion(
                a.url, completion_body(a.served, p["ids"], MAX_TOKENS)
            )
            if traced:
                trace_end(a, tag)
            run = summarize(resp, secs)
            run.update({"id": p["id"], "length": n, "rep": rep, "spec": spec})
            runs.append(run)
            print(
                f"{p['id']} rep={rep}: {run['seconds']} s, "
                f"first tokens {run['tokens'][:4]}"
            )
    with open(a.out, "w") as f:
        json.dump({"kind": "base", "runs": runs}, f)
    return 0


def cmd_prefill(a: argparse.Namespace) -> int:
    """Send every prompt to the producer (``do_remote_decode``) and dump the memory
    windows a decode-side read addresses right after each response, so the producer's
    blocks may expire (``kv_lease_duration``) before the next prompt needs the pool."""
    base = a.url
    results = []
    regions = rpc(base, "g0a_regions", [])
    with open(os.path.join(a.store, "regions_producer.json"), "w") as f:
        json.dump(regions, f, indent=1)
    # The handshake payload is static after boot: capture it once, before any request.
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
    for p in load_prompts(a.prompts, a.only):
        n = p["length"]
        body = completion_body(a.served, p["ids"], 1)
        body["kv_transfer_params"] = {"do_remote_decode": True}
        trace_begin(a, f"{p['id']}_P", n, raw=is_raw_id(p["id"]), raw_at=n - 1)
        t0 = time.time()
        resp = post(base + "/v1/completions", body)
        secs = time.time() - t0
        trace_end(a, f"{p['id']}_P")
        params = resp.get("kv_transfer_params")
        assert params, f"producer returned no kv_transfer_params: {list(resp)}"
        print(f"{p['id']}: prefill {secs:.2f} s; params keys {sorted(params)}")
        out = rpc(base, "g0a_dump", [a.store, p["id"], json.dumps(params)])
        total = sum(x["bytes"] for x in out)
        print(
            f"{p['id']}: dumped {total / 2**20:.1f} MiB over {len(out)} ranks "
            f"({[round(x['bytes'] / 2**20, 1) for x in out]} MiB/rank)"
        )
        results.append(
            {
                "id": p["id"],
                "length": n,
                "seconds": round(secs, 3),
                "kv_transfer_params": params,
                "usage": resp.get("usage"),
                "first_token": resp["choices"][0]["logprobs"]["tokens"],
                "dump": out,
            }
        )
    with open(a.out, "w") as f:
        json.dump({"kind": "prefill", "runs": results, "regions": regions}, f)
    return 0


def cmd_decode(a: argparse.Namespace) -> int:
    with open(a.pjson) as f:
        pruns = {r["id"]: r for r in json.load(f)["runs"]}
    regions = rpc(a.url, "g0a_regions", [])
    with open(os.path.join(a.stats, "regions_consumer.json"), "w") as f:
        json.dump(regions, f, indent=1)
    runs = []
    for p in load_prompts(a.prompts, a.only):
        n = p["length"]
        params = dict(pruns[p["id"]]["kv_transfer_params"])
        params["remote_port"] = a.stand_in_port
        params["remote_host"] = "127.0.0.1"
        body = completion_body(a.served, p["ids"], MAX_TOKENS)
        body["kv_transfer_params"] = params
        trace_begin(a, f"{p['id']}_D", n, raw=is_raw_id(p["id"]), raw_at=None)
        try:
            resp, secs, spec = timed_completion(a.url, body)
        except Exception as e:  # noqa: BLE001 - record and continue with the next prompt
            print(f"{p['id']}: request failed: {e!r}")
            runs.append({"id": p["id"], "length": n, "error": repr(e)})
            trace_end(a, f"{p['id']}_D")
            continue
        trace_end(a, f"{p['id']}_D")
        run = summarize(resp, secs)
        run.update({"id": p["id"], "length": n, "spec": spec})
        runs.append(run)
        print(f"{p['id']}: {run['seconds']} s, first tokens {run['tokens'][:4]}")
    with open(a.out, "w") as f:
        json.dump({"kind": "decode", "runs": runs, "regions": regions}, f)
    return 0


def cmd_rpc(a: argparse.Namespace) -> int:
    print(json.dumps(rpc(a.url, a.method, a.arg), indent=1))
    return 0


def ppl(run: dict[str, Any]) -> float:
    lps = [x for x in run["token_logprobs"] if x is not None]
    return math.exp(-sum(lps) / len(lps))


def acceptance(run: dict[str, Any]) -> float | None:
    """Fraction of proposed MTP draft tokens the target accepted for this request."""
    s = run.get("spec")
    if not s or not s.get("draft_tokens"):
        return None
    return s["accepted_tokens"] / s["draft_tokens"]


def first_token_stats(ref: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    """Agreement of the first generated token's top-k distribution: bitwise equality,
    the largest |dlogprob| over the shared candidates, and whether the candidate sets
    agree. The first token comes from the one step that runs the handed-over state
    and involves no drafts."""
    a, b = ref["top"][0], test["top"][0]
    common = set(a) & set(b)
    return {
        "first_token_equal": ref["tokens"][0] == test["tokens"][0],
        "first_top_bitwise": a == b,
        "first_top_set_equal": set(a) == set(b),
        "first_top_max_d": max((abs(a[k] - b[k]) for k in common), default=None),
    }


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
        "acceptance": [acceptance(ref), acceptance(test)],
        "finite": all(
            x is not None and math.isfinite(x)
            for x in ref["token_logprobs"] + test["token_logprobs"]
        ),
        **first_token_stats(ref, test),
    }
    if div < n:
        vals = sorted(ref["top"][div].values(), reverse=True)
        out["divergence_top2_gap_ref"] = vals[0] - vals[1] if len(vals) > 1 else None
        out["ref_token"] = ref["tokens"][div]
        out["test_token"] = test["tokens"][div]
    return out


def _fmt_acc(acc: list[float | None]) -> str:
    return "/".join("-" if x is None else f"{x:.3f}" for x in acc)


def _compare_maps(
    ref: dict[Any, dict[str, Any]], test: dict[Any, dict[str, Any]], label: str
) -> tuple[bool, list[dict[str, Any]]]:
    ok = True
    report = []
    for key in sorted(test, key=lambda k: (test[k].get("length", 0), str(k))):
        t = test[key]
        r = ref.get(key) or ref.get((key[0], 0))
        if "error" in t or r is None:
            ok = False
            report.append({"id": key[0], "error": t.get("error", "no reference")})
            print(f"{label} {key[0]}: ERROR {t.get('error')}")
            continue
        res = compare_runs(r, t)
        res["id"] = key[0]
        res["length"] = t.get("length")
        report.append(res)
        # exact = the 64 greedy tokens agree; the first token's top-k must also be
        # bitwise equal (it comes from the one step that runs the handed-over state)
        ok &= res["exact"] and res["finite"] and res["first_top_bitwise"]
        d1 = res["first_top_max_d"]
        print(
            f"{label} {key[0]}: exact={res['exact']} "
            f"identical_prefix={res['identical_prefix']}/"
            f"{MAX_TOKENS} bitwise_logprobs={res['bitwise_logprobs']} "
            f"first_top_bitwise={res['first_top_bitwise']} "
            f"first_top_max_d={'-' if d1 is None else f'{d1:.5f}'} "
            f"max|dlogprob|={res['max_dlogprob_on_prefix']:.5f} "
            f"finite={res['finite']} acc={_fmt_acc(res['acceptance'])}"
            + (
                f" DIVERGES ref={res['ref_token']} test={res['test_token']} "
                f"top2gap={res['divergence_top2_gap_ref']}"
                if "ref_token" in res
                else ""
            )
        )
    rows = [x for x in report if "error" not in x]
    print(
        f"SUMMARY {label}: prompts={len(report)} "
        f"first_top_bitwise={sum(x['first_top_bitwise'] for x in rows)} "
        f"tokens_exact={sum(x['exact'] for x in rows)} "
        f"logprobs_bitwise={sum(x['bitwise_logprobs'] for x in rows)}"
    )
    return ok, report


def _by_id(path: str) -> dict[tuple[str, Any], dict[str, Any]]:
    with open(path) as f:
        return {(r["id"], r.get("rep", 0)): r for r in json.load(f)["runs"]}


def cmd_compare(a: argparse.Namespace) -> int:
    ref = _by_id(a.ref)
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
        test = _by_id(a.test)
        refm = {(n, 0): r for (n, rp), r in ref.items() if str(rp) == a.ref_rep}
        ok, report = _compare_maps(refm, test, f"{a.label} vs ref rep {a.ref_rep}")
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
    p.add_argument(
        "--spec",
        default=DEFAULT_SPEC,
        help="comma list of LENGTH:COUNT, one disjoint window of the docs per prompt",
    )
    p.add_argument("--start", type=int, default=0, help="offset of the first window")
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
        p.add_argument("--only", help="comma list of prompt ids to run")
        p.add_argument("--trace-dir", help="arm g0r_trace around every request")
        p.set_defaults(fn=fn)
        if name == "base":
            p.add_argument("--repeat", type=int, default=2)
            p.add_argument(
                "--repeat-max-len",
                type=int,
                default=8193,
                help="longer prompts run one cold rep (plus the hit)",
            )
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
    p.add_argument("--label", default="D", help="name of --test in the report lines")
    p.add_argument("--out")
    p.add_argument("--aa", action="store_true", help="repeat 0 vs repeat 1 of --ref")
    p.add_argument(
        "--ref-rep",
        default="0",
        help="rep of --ref the --test runs are compared with: 0 (cold) or hit",
    )
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
