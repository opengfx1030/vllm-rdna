# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded repeated/extended conversation probe; no user chat contents captured."""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--output", required=True)
parser.add_argument("--lines", type=int, default=400)
parser.add_argument("--cache-salt")
parser.add_argument("--model", default="active")
parser.add_argument("--skip-identical", action="store_true")
args = parser.parse_args()
args.cache_salt = args.cache_salt or uuid.uuid4().hex
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)


def metrics():
    with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as r:
        return r.read().decode()


def counter(snapshot, name):
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in snapshot.splitlines()
        if line.startswith("vllm:" + name + "{")
    )


before = metrics()
(out / "metrics-before.txt").write_text(before)
for line in before.splitlines():
    if (
        line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
        and float(line.rsplit(" ", 1)[1]) != 0
    ):
        raise SystemExit("User request in progress; postponing bounded probe")

background = "\n".join(
    f"Record {i:04d}: amber birch cedar delta. This line is stable background data."
    for i in range(args.lines)
)
messages = [
    {"role": "system", "content": "Reply with the requested word only."},
    {"role": "user", "content": background + "\nReply exactly: blue"},
]
results = []
previous_metrics = before
cases = (
    ["initial", "followup"]
    if args.skip_identical
    else ["initial", "identical", "followup"]
)
for name in cases:
    if name == "followup":
        messages = messages + [
            {"role": "assistant", "content": results[0]["text"]},
            {"role": "user", "content": "Now reply exactly: green"},
        ]
    body = dict(
        model=args.model,
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=16,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    if args.cache_salt:
        body["cache_salt"] = args.cache_salt
    req = urllib.request.Request(
        args.base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    chunks = []
    usage = None
    with urllib.request.urlopen(req, timeout=90) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                continue
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                token = choice.get("delta", {}).get("content")
                if token:
                    if first is None:
                        first = time.monotonic() - start
                    chunks.append(token)
    after = metrics()
    row = dict(
        case=name,
        ttft_s=first,
        total_s=time.monotonic() - start,
        text="".join(chunks),
        usage=usage,
        prefix_hits_delta=counter(after, "prefix_cache_hits_total")
        - counter(previous_metrics, "prefix_cache_hits_total"),
    )
    previous_metrics = after
    results.append(row)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    (out / f"metrics-after-{name}.txt").write_text(after)

assert [r["text"].strip() for r in results] == (
    ["blue", "green"] if args.skip_identical else ["blue", "blue", "green"]
), results

for row in results[1:]:
    details = row["usage"].get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", row["prefix_hits_delta"])
    prompt = row["usage"]["prompt_tokens"]
    if cached < prompt * 0.7:
        raise SystemExit(
            f"FAIL {row['case']}: cached {cached}/{prompt}; "
            "repeated context was recomputed"
        )
    if row["ttft_s"] >= results[0]["ttft_s"] * 0.5:
        raise SystemExit(
            f"FAIL {row['case']}: TTFT did not fall below half of initial prefill"
        )
print("PASS: identical and extended context reuse cached tokens with lower TTFT")
