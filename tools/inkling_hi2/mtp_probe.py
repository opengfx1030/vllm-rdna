#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark/MTP probe for an HI2 Inkling-2-Mini server (GPU session, no lease logic).

Compares a spec-decoding boot (DSPARK=1) with the plain boot (DSPARK=0) of the
same directory, and reads the per-position acceptance the engine logs:

  * greedy outputs identical with and without DSpark (same model, same prompts:
    speculative decoding may not change the tokens) and deterministic on repeat;
  * mean acceptance length and per-position acceptance rates from the server log
    (the ``SpecDecoding metrics:`` line; optionally also from ``/metrics``);
  * decode speed at concurrency 1 (spec vs plain) and 2 (spec), tokens/s;
  * the batch-size drop: at >= the configured threshold no drafts are made and
    the requests still succeed (``vllm:spec_decode_num_drafts_total`` delta).

    $VENV/bin/python tools/inkling_hi2/mtp_probe.py \
        --spec-url http://127.0.0.1:18230 --plain-url http://127.0.0.1:18231 \
        --log ~/w4a8_runs/inkling-hi2-runtime/logs/hi2-dspark.log --json out.json

Exit codes: 0 all checks passed, 1 a check failed, 2 bad input, 3 a server is
unreachable.  Nothing here runs on a GPU host by itself: it only needs the two
servers up (the runbook boots them).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROMPTS: tuple[dict[str, str], ...] = (
    {
        "name": "code_sum",
        "prompt": "Write a Python function that returns the sum of a list of ints. "
        "Only the code.",
    },
    {
        "name": "code_bug",
        "prompt": "This Python loop never stops: while i < 10: print(i). Why? "
        "Answer in two sentences.",
    },
    {
        "name": "chat_intro",
        "prompt": "Introduce yourself in three short sentences.",
    },
    {
        "name": "chat_plan",
        "prompt": "List three steps to test a language model backend, one per line.",
    },
    {
        "name": "agentic_patch",
        "prompt": "A shell script fails with 'permission denied' when run as "
        "./x.sh. Give the command that fixes it and one sentence why.",
    },
    {
        "name": "agentic_log",
        "prompt": "Given the log line 'CUDA: out of memory' and 'gfx1030', "
        "state the likely cause in one sentence.",
    },
    {
        "name": "math",
        "prompt": "What is 17 * 23 + 5? Give the number only.",
    },
    {
        "name": "refactor",
        "prompt": "Rewrite this to be clearer, then explain in one sentence: "
        "for x in d: if d[x] > 0: d[x] = d[x] - 1",
    },
)

SPEC_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
)


def base_url(raw: str) -> str:
    """The server base without a path: http://host:port, never .../v1."""
    parsed = urllib.parse.urlsplit(raw.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{raw!r}: not an http(s) base URL")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError(
            f"{raw!r}: give the server base without a path (got {parsed.path!r}); "
            "the probe appends /v1/chat/completions itself"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


_SERVED: dict[str, str] = {}


def served_model(base: str, timeout: float = 30.0) -> str:
    """The id /v1/models reports; vLLM answers 404 for any other model name."""
    if base in _SERVED:
        return _SERVED[base]
    url = f"{base}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GET {url}: HTTP {e.code} {e.read()[:200]!r}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RuntimeError(f"GET {url}: {e}") from e
    models = body.get("data") or []
    if not models:
        raise RuntimeError(f"GET {url}: no served model")
    _SERVED[base] = str(models[0]["id"])
    return _SERVED[base]


def _http_json(url: str, payload: dict | None, timeout: float):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{req.get_method()} {url}: HTTP {e.code}: {detail}") from e


def health(base: str, timeout: float = 5.0) -> bool:
    try:
        status, _ = _http_json(f"{base}/health", None, timeout)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False
    return status == 200


def complete(
    base: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    seed: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    payload = {
        "model": served_model(base, timeout),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        # vLLM extension: the generated token ids, so the cross-process
        # comparison is token-level and not a re-detokenisation of the text.
        "return_token_ids": True,
    }
    t0 = time.monotonic()
    try:
        status, body = _http_json(f"{base}/v1/chat/completions", payload, timeout)
    except RuntimeError:
        # Older servers reject the token-id extension as an unknown field; the
        # text is still comparable through /tokenize.
        payload.pop("return_token_ids", None)
        status, body = _http_json(f"{base}/v1/chat/completions", payload, timeout)
    elapsed = time.monotonic() - t0
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"{base}: HTTP {status}")
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    ids = choice.get("token_ids")
    if ids is None:
        ids = tokenize(base, text, timeout) if text else []
    return {
        "text": text,
        "token_ids": [int(i) for i in ids],
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
        "elapsed": elapsed,
    }


def tokenize(base: str, text: str, timeout: float = 60.0) -> list[int]:
    """Token ids from the server's own tokenizer (the fallback for responses
    that do not return them)."""
    url = f"{base}/tokenize"
    payload = {
        "model": served_model(base, timeout),
        "prompt": text,
        "add_special_tokens": False,
    }
    status, body = _http_json(url, payload, timeout)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"{url}: HTTP {status}")
    return [int(t) for t in body.get("tokens") or []]


def diff_tokens(a: list[int], b: list[int]) -> dict:
    """Token-level diff: first divergence index (None if identical) + count."""
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    count = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
    if first is None and count:
        first = n
    return {"first": first, "count": count}


def floor_verdict(spec: list[int], plain: list[int], floor: list[int] | None) -> dict:
    """Identity against the cross-process floor (plain boot #1 vs #2).

    The stack is not bitwise reproducible across processes (TunableOp rows, PLE
    state), so a difference is only a finding when it exceeds what two plain
    boots of the same weights already show on that prompt.
    """
    sd = diff_tokens(spec, plain)
    if floor is None:
        return {"ok": sd["count"] == 0, "spec": sd, "floor": None}
    fd = diff_tokens(plain, floor)
    return {
        "ok": sd["count"] == 0 or sd["count"] <= fd["count"],
        "spec": sd,
        "floor": fd,
    }


def no_draft_verdict(before: float, after: float) -> dict:
    delta = after - before
    return {"delta": delta, "ok": delta == 0}


def settled_drafts(base: str, settle: float = 0.2, tries: int = 3) -> float:
    """The drafts counter after it stopped moving (leftover work drained)."""
    last = metric(base, "vllm:spec_decode_num_drafts_total")
    for _ in range(tries):
        time.sleep(settle)
        now = metric(base, "vllm:spec_decode_num_drafts_total")
        if now == last:
            return now
        last = now
    return last


def metric(base: str, name: str, timeout: float = 10.0) -> float:
    """Sum one Prometheus counter over all its label sets (0 if absent)."""
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=timeout) as resp:  # noqa: S310
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0.0
    total = 0.0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        head = head.split("{", 1)[0].strip()
        if head != name:
            continue
        try:
            total += float(value)
        except ValueError:
            continue
    return total


def spec_metrics(base: str, timeout: float = 10.0) -> dict[str, float]:
    return {name: metric(base, name, timeout) for name in SPEC_METRICS}


def parse_log_metrics(tail: str) -> dict:
    """The last ``SpecDecoding metrics:`` line of a server log tail.

    Plain string parsing (no ``re``: the repo's import hook forbids it here).
    """
    mean = None
    positions = None
    for line in tail.splitlines():
        if "Mean acceptance length:" in line:
            token = line.split("Mean acceptance length:", 1)[1].split(",")[0].strip()
            with contextlib.suppress(ValueError):
                mean = float(token)
        if "Per-position acceptance rate:" in line:
            tail_part = line.split("Per-position acceptance rate:", 1)[1]
            values = []
            for item in tail_part.split("Avg Draft")[0].split(","):
                with contextlib.suppress(ValueError):
                    values.append(float(item.strip()))
            positions = values
    return {"mean_acceptance_length": mean, "per_position": positions}


def _tokens(result: dict) -> int:
    return int((result.get("usage") or {}).get("completion_tokens") or 0)


def _garbage(text: str) -> bool:
    if not text.strip():
        return True
    return "\ufffd" in text


@dataclass
class Report:
    checks: dict[str, bool] = field(default_factory=dict)
    numbers: dict[str, object] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, note: str = "") -> None:
        self.checks[name] = bool(ok)
        status = "PASS" if ok else "FAIL"
        self.log.append(f"{status} {name}{(': ' + note) if note else ''}")


def measure_stock(
    report: Report,
    stock: str,
    log_path: str | None,
    max_tokens: int,
    timeout: float,
) -> None:
    """Stock-model DSPARK=1 acceptance on the same 8 prompts (a reference)."""
    log_pos = 0
    if log_path:
        path = Path(log_path).expanduser()
        log_pos = path.stat().st_size if path.is_file() else 0
    model = served_model(stock, timeout)
    print(f"mtp_probe: stock-spec {stock} serves {model!r}")
    before = spec_metrics(stock)
    results = [complete(stock, p["prompt"], max_tokens, timeout) for p in PROMPTS]
    after = spec_metrics(stock)
    drafts = (
        after["vllm:spec_decode_num_drafts_total"]
        - before["vllm:spec_decode_num_drafts_total"]
    )
    accepted = (
        after["vllm:spec_decode_num_accepted_tokens_total"]
        - before["vllm:spec_decode_num_accepted_tokens_total"]
    )
    drafted = (
        after["vllm:spec_decode_num_draft_tokens_total"]
        - before["vllm:spec_decode_num_draft_tokens_total"]
    )
    report.numbers["stock_drafts"] = drafts
    report.numbers["stock_accepted_tokens"] = accepted
    report.numbers["stock_draft_tokens"] = drafted
    report.numbers["stock_mean_acceptance_length"] = (
        round(1 + accepted / drafts, 3) if drafts else None
    )
    report.numbers["stock_acceptance_rate"] = (
        round(accepted / drafted, 4) if drafted else None
    )
    report.check(
        "stock_dspark_active",
        drafts > 0 and accepted > 0,
        f"{drafts:g} drafts, {accepted:g} accepted",
    )
    report.check(
        "stock_text_clean",
        all(not _garbage(r["text"]) for r in results),
        f"{len(results)} prompts",
    )
    if log_path:
        path = Path(log_path).expanduser()
        if path.is_file():
            with open(path, "rb") as f:
                f.seek(log_pos)
                parsed = parse_log_metrics(f.read().decode("utf-8", "replace"))
            report.numbers["stock_per_position"] = parsed["per_position"]
            report.numbers["stock_mean_acceptance_length_log"] = parsed[
                "mean_acceptance_length"
            ]
            report.check(
                "stock_per_position_logged",
                bool(parsed["per_position"]),
                str(parsed["per_position"]),
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-url", default=None, help="DSpark (DSPARK=1) server")
    ap.add_argument(
        "--plain-url",
        default=None,
        help="the same directory served with DSPARK=0 (greedy identity, speed)",
    )
    ap.add_argument(
        "--log",
        default=None,
        help="the spec server's stdout log (per-position acceptance)",
    )
    ap.add_argument(
        "--floor-json",
        default=None,
        help="the previous probe JSON: its plain-boot outputs are the floor",
    )
    ap.add_argument(
        "--settle-seconds",
        type=float,
        default=10.0,
        help="idle wait before the batch-K=0 window (leftover requests drain)",
    )
    ap.add_argument(
        "--stock-spec-url",
        default=None,
        help="stock model served with DSPARK=1 (acceptance reference)",
    )
    ap.add_argument(
        "--stock-log",
        default=None,
        help="the stock spec server's stdout log (per-position acceptance)",
    )
    ap.add_argument(
        "--record-only",
        action="store_true",
        help="phase 1 of the floor: record the plain outputs, gate nothing",
    )
    ap.add_argument(
        "--stock-only",
        action="store_true",
        help="measure the stock DSPARK=1 reference only and exit",
    )
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--drop-batch",
        type=int,
        default=0,
        help="concurrency at which the schedule sets K=0 (3 for [[1,2,2],[3,8,0]])",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    try:
        spec = base_url(args.spec_url) if args.spec_url else None
        plain = base_url(args.plain_url) if args.plain_url else None
        stock = base_url(args.stock_spec_url) if args.stock_spec_url else None
    except ValueError as e:
        print(f"mtp_probe: {e}", file=sys.stderr)
        return 2
    report = Report()

    if args.stock_only:
        if stock is None:
            print("mtp_probe: --stock-only needs --stock-spec-url", file=sys.stderr)
            return 2
        if not health(stock):
            print(f"mtp_probe: {stock} is not serving (/health)", file=sys.stderr)
            return 3
        try:
            measure_stock(report, stock, args.stock_log, args.max_tokens, args.timeout)
        except RuntimeError as e:
            print(f"mtp_probe: {e}", file=sys.stderr)
            return 3
        for line in report.log:
            print(line)
        for name, value in report.numbers.items():
            print(f"  {name}: {value}")
        passed = all(report.checks.values())
        print("PASS" if passed else "FAIL")
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "checks": report.checks,
                        "numbers": report.numbers,
                        "log": report.log,
                        "outputs": {},
                    },
                    indent=2,
                )
                + "\n"
            )
        return 0 if passed else 1

    if spec is None:
        print("mtp_probe: --spec-url is required", file=sys.stderr)
        return 2
    if not health(spec):
        print(f"mtp_probe: {spec} is not serving (/health)", file=sys.stderr)
        return 3
    if plain and not health(plain):
        print(f"mtp_probe: {plain} is not serving (/health)", file=sys.stderr)
        return 3
    if stock and not health(stock):
        print(f"mtp_probe: stock {stock} is not serving (/health)", file=sys.stderr)
        return 3
    try:
        spec_model = served_model(spec, args.timeout)
        plain_model = served_model(plain, args.timeout) if plain else None
    except RuntimeError as e:
        print(f"mtp_probe: {e}", file=sys.stderr)
        return 3
    print(f"mtp_probe: spec {spec} serves {spec_model!r}")
    print(f"mtp_probe: POST {spec}/v1/chat/completions model={spec_model!r}")
    if plain:
        print(f"mtp_probe: plain {plain} serves {plain_model!r}")
        print(f"mtp_probe: POST {plain}/v1/chat/completions model={plain_model!r}")

    log_pos = 0
    if args.log:
        path = Path(args.log).expanduser()
        log_pos = path.stat().st_size if path.is_file() else 0

    before = spec_metrics(spec)

    # 1. determinism and garbage on the spec server
    first = complete(spec, PROMPTS[0]["prompt"], args.max_tokens, args.timeout)
    again = complete(spec, PROMPTS[0]["prompt"], args.max_tokens, args.timeout)
    report.check(
        "spec_deterministic",
        first["token_ids"] == again["token_ids"],
        f"{_tokens(first)} tokens",
    )
    report.check(
        "spec_text_clean",
        all(
            not _garbage(r["text"])
            for r in (first, again)
            if r["finish_reason"] in {"stop", "length"}
        ),
        f"finish={first['finish_reason']}",
    )

    # 2. token-level identity against the plain boot, gated by the
    #    cross-process floor measured by two plain boots (--floor-json)
    outputs: dict[str, dict] = {"spec": {}, "plain": {}}
    floor_tokens: dict[str, list[int]] = {}
    if args.floor_json:
        prior = Path(args.floor_json).expanduser()
        if not prior.is_file():
            print(f"mtp_probe: no floor JSON at {prior}", file=sys.stderr)
            return 2
        floor_tokens = (json.loads(prior.read_text()).get("outputs") or {}).get(
            "plain"
        ) or {}
        if sorted(floor_tokens) != sorted(p["name"] for p in PROMPTS):
            print(
                f"mtp_probe: floor JSON lacks the {len(PROMPTS)} prompt outputs",
                file=sys.stderr,
            )
            return 2
    if plain:
        for p in PROMPTS:
            so = complete(spec, p["prompt"], args.max_tokens, args.timeout)
            po = complete(plain, p["prompt"], args.max_tokens, args.timeout)
            outputs["spec"][p["name"]] = so["token_ids"]
            outputs["plain"][p["name"]] = po["token_ids"]
            verdict = floor_verdict(
                so["token_ids"], po["token_ids"], floor_tokens.get(p["name"])
            )
            sd, fd = verdict["spec"], verdict["floor"]
            report.numbers[f"spec_diff[{p['name']}]"] = sd["count"]
            report.numbers[f"spec_first_diff[{p['name']}]"] = sd["first"]
            if fd is not None:
                report.numbers[f"floor_diff[{p['name']}]"] = fd["count"]
                if not args.record_only:
                    report.check(
                        f"greedy_within_floor[{p['name']}]",
                        verdict["ok"],
                        f"spec_diff={sd['count']} floor={fd['count']} "
                        f"first={sd['first']}",
                    )
            elif not args.record_only:
                report.check(
                    f"greedy_identical[{p['name']}]",
                    verdict["ok"],
                    f"spec_diff={sd['count']} first={sd['first']}",
                )
        report.numbers["spec_diff_max"] = max(
            report.numbers[f"spec_diff[{p['name']}]"] for p in PROMPTS
        )
        if floor_tokens:
            report.numbers["floor_diff_max"] = max(
                report.numbers[f"floor_diff[{p['name']}]"] for p in PROMPTS
            )

    # 3. per-position acceptance from the log / the Prometheus counters
    if args.log:
        path = Path(args.log).expanduser()
        if path.is_file():
            with open(path, "rb") as f:
                f.seek(log_pos)
                tail = f.read().decode("utf-8", "replace")
            log_metrics = parse_log_metrics(tail)
        else:
            log_metrics = {"mean_acceptance_length": None, "per_position": None}
        report.numbers["per_position"] = log_metrics["per_position"]
        report.check(
            "per_position_logged",
            bool(log_metrics["per_position"]),
            str(log_metrics["per_position"]),
        )

    # 4. throughput record (not a gate)
    def timed(base: str, count: int, max_tokens: int) -> float:
        prompts = [PROMPTS[i % len(PROMPTS)]["prompt"] for i in range(count)]

        def run(prompt: str) -> dict:
            return complete(base, prompt, max_tokens, args.timeout)

        t0 = time.monotonic()
        with futures.ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(run, prompts))
        elapsed = time.monotonic() - t0
        tokens = sum(_tokens(r) for r in results)
        return tokens / elapsed if elapsed > 0 else 0.0

    speed_spec_1 = timed(spec, 1, args.max_tokens)
    speed_spec_2 = timed(spec, 2, args.max_tokens)
    report.numbers["spec_tok_s_c1"] = round(speed_spec_1, 3)
    report.numbers["spec_tok_s_c2"] = round(speed_spec_2, 3)
    if plain:
        speed_plain_1 = timed(plain, 1, args.max_tokens)
        report.numbers["plain_tok_s_c1"] = round(speed_plain_1, 3)
        report.numbers["speedup_c1"] = (
            round(speed_spec_1 / speed_plain_1, 3) if speed_plain_1 else None
        )
        # No pass/fail on speed: the HI2 block-64 GEMM pads a 3-row verify to 64
        # rows, so the ratio is a record, not a gate (README step 3b).
        report.check(
            "spec_c1_speed_measured",
            speed_spec_1 > 0 and speed_plain_1 > 0,
            f"{speed_spec_1:.1f} vs {speed_plain_1:.1f} tok/s",
        )

    # 5. the configured no-draft region: drafts must stop and requests succeed.
    #    The counters are read only after the earlier cells have drained.
    if args.drop_batch > 1 and not args.record_only:
        if args.settle_seconds:
            print(
                f"mtp_probe: settling {args.settle_seconds:g}s before the "
                f"batch-{args.drop_batch} window",
                flush=True,
            )
            time.sleep(args.settle_seconds)
        window_before = settled_drafts(spec, max(args.settle_seconds, 0.2))
        n = args.drop_batch
        prompts = [PROMPTS[i % len(PROMPTS)]["prompt"] for i in range(n)]
        start = threading.Barrier(n)

        def run_long(prompt: str) -> dict:
            start.wait(timeout=args.timeout)
            return complete(spec, prompt, min(args.max_tokens, 48), args.timeout)

        with futures.ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(run_long, prompts))
        window_after = metric(spec, "vllm:spec_decode_num_drafts_total")
        verdict = no_draft_verdict(window_before, window_after)
        report.numbers["drafts_before_batch"] = window_before
        report.numbers["drafts_after_batch"] = window_after
        report.numbers["drafts_delta_at_batch"] = verdict["delta"]
        report.check(
            f"no_drafts_at_batch_{n}",
            verdict["ok"],
            f"{window_before:g}->{window_after:g} delta {verdict['delta']:g}",
        )
        report.check(
            f"batch_{n}_all_ok",
            all(not _garbage(r["text"]) for r in results),
            f"{len(results)} requests",
        )

    after = spec_metrics(spec)
    drafts = (
        after["vllm:spec_decode_num_drafts_total"]
        - before["vllm:spec_decode_num_drafts_total"]
    )
    accepted = (
        after["vllm:spec_decode_num_accepted_tokens_total"]
        - before["vllm:spec_decode_num_accepted_tokens_total"]
    )
    drafted = (
        after["vllm:spec_decode_num_draft_tokens_total"]
        - before["vllm:spec_decode_num_draft_tokens_total"]
    )
    report.numbers["drafts"] = drafts
    report.numbers["accepted_tokens"] = accepted
    report.numbers["draft_tokens"] = drafted
    report.numbers["mean_acceptance_length"] = (
        round(1 + accepted / drafts, 3) if drafts else None
    )
    report.numbers["acceptance_rate"] = (
        round(accepted / drafted, 4) if drafted else None
    )
    report.check(
        "dspark_drafting_active",
        drafts > 0 and accepted > 0,
        f"{drafts:g} drafts, {accepted:g} accepted",
    )

    # 6. the stock model's DSPARK=1 acceptance on the same prompts
    if stock is not None:
        try:
            measure_stock(report, stock, args.stock_log, args.max_tokens, args.timeout)
        except RuntimeError as e:
            print(f"mtp_probe: {e}", file=sys.stderr)
            return 3

    for line in report.log:
        print(line)
    for name, value in report.numbers.items():
        print(f"  {name}: {value}")
    passed = all(report.checks.values())
    print("PASS" if passed else "FAIL")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "checks": report.checks,
                    "numbers": report.numbers,
                    "log": report.log,
                    "outputs": outputs,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
