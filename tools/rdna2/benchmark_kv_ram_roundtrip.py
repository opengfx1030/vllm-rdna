# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a long-context RAM KV-cache eviction and reload round trip."""

import argparse
import json
import time
import urllib.request
from typing import Any


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _metrics(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        body = response.read().decode()
    needles = ("offload", "prefix_cache", "kv_cache_usage")
    return [
        line
        for line in body.splitlines()
        if not line.startswith("#") and any(needle in line for needle in needles)
    ]


def _stream_chat(base_url: str, model: str, content: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 96,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens", 0)
    decode_seconds = max(ended - (first_token_at or ended), 1e-9)
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "total_seconds": ended - started,
        "decode_tokens_per_second": completion_tokens / decode_seconds,
        "text": "".join(text_parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="active")
    parser.add_argument("--settle-seconds", type=float, default=15)
    args = parser.parse_args()

    _get_json(f"{args.base_url}/v1/models")
    session_a = (
        "Session A sentinel is ALPHA-SESSION-200K. Preserve it."
        + " alpha" * 199_900
        + "\nReply exactly with ALPHA-SESSION-200K, then one short sentence."
    )
    session_b = (
        "Session B sentinel is BETA-SESSION-100K. Preserve it."
        + " beta" * 99_900
        + "\nReply exactly with BETA-SESSION-100K, then one short sentence."
    )

    result: dict[str, Any] = {"metrics_before": _metrics(args.base_url)}
    result["session_a_cold"] = _stream_chat(args.base_url, args.model, session_a)
    time.sleep(args.settle_seconds)
    result["metrics_after_a"] = _metrics(args.base_url)

    result["session_b_switch"] = _stream_chat(args.base_url, args.model, session_b)
    time.sleep(args.settle_seconds)
    result["metrics_after_b"] = _metrics(args.base_url)

    continuation = (
        session_a
        + result["session_a_cold"]["text"]
        + "\nContinue session A. Reply exactly with ALPHA-SESSION-RELOADED, "
        "then one short sentence."
    )
    result["session_a_reload"] = _stream_chat(args.base_url, args.model, continuation)
    time.sleep(args.settle_seconds)
    result["metrics_after_a_reload"] = _metrics(args.base_url)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
