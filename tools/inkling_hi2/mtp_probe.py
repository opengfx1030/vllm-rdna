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
    }
    t0 = time.monotonic()
    status, body = _http_json(f"{base}/v1/chat/completions", payload, timeout)
    elapsed = time.monotonic() - t0
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"{base}: HTTP {status}")
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    return {
        "text": text,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
        "elapsed": elapsed,
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-url", required=True, help="DSpark (DSPARK=1) server")
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
        spec = base_url(args.spec_url)
        plain = base_url(args.plain_url) if args.plain_url else None
    except ValueError as e:
        print(f"mtp_probe: {e}", file=sys.stderr)
        return 2
    report = Report()

    if not health(spec):
        print(f"mtp_probe: {spec} is not serving (/health)", file=sys.stderr)
        return 3
    if plain and not health(plain):
        print(f"mtp_probe: {plain} is not serving (/health)", file=sys.stderr)
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
        first["text"] == again["text"] and _tokens(first) == _tokens(again),
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

    # 2. greedy identity / plain decode
    spec_out: list[dict] = []
    plain_out: list[dict] = []
    if plain:
        for p in PROMPTS:
            spec_out.append(complete(spec, p["prompt"], args.max_tokens, args.timeout))
            plain_out.append(
                complete(plain, p["prompt"], args.max_tokens, args.timeout)
            )
            report.check(
                f"greedy_identical[{p['name']}]",
                spec_out[-1]["text"] == plain_out[-1]["text"],
                f"{_tokens(spec_out[-1])}/{_tokens(plain_out[-1])} tokens",
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
        report.numbers["log"] = log_metrics
        report.check(
            "per_position_acceptance_logged",
            bool(log_metrics["per_position"]),
            str(log_metrics["per_position"]),
        )

    # 4. speed at c=1 (spec vs plain) and c=2 (spec)
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

    # 5. the configured no-draft region: drafts must stop and requests succeed
    if args.drop_batch > 1:
        drafts_before = metric(spec, "vllm:spec_decode_num_drafts_total")
        prompts = [PROMPTS[i % len(PROMPTS)]["prompt"] for i in range(args.drop_batch)]

        def run_long(prompt: str) -> dict:
            return complete(spec, prompt, min(args.max_tokens, 48), args.timeout)

        with futures.ThreadPoolExecutor(max_workers=args.drop_batch) as pool:
            results = list(pool.map(run_long, prompts))
        drafts_after = metric(spec, "vllm:spec_decode_num_drafts_total")
        report.numbers["drafts_delta_at_batch"] = drafts_after - drafts_before
        report.check(
            f"no_drafts_at_batch_{args.drop_batch}",
            drafts_after - drafts_before == 0,
            f"delta {drafts_after - drafts_before:g}",
        )
        report.check(
            f"batch_{args.drop_batch}_all_ok",
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
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
