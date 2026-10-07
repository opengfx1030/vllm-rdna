# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise multimodal prompt KV eviction and RAM-cache restoration."""

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import pybase64 as base64

BASE_URL = "http://127.0.0.1:8080"
MODEL = "active"


def _image_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def _metric_values() -> dict[str, float]:
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    wanted = {
        (
            'vllm:kv_offload_total_bytes_total{engine="0",model_name="active",'
            'transfer_type="CPU_to_GPU"}'
        ): "cpu_to_gpu_bytes",
        (
            'vllm:external_prefix_cache_hits_total{engine="0",model_name="active"}'
        ): "external_hit_tokens",
        (
            'vllm:prompt_tokens_by_source_total{engine="0",model_name="active",'
            'source="external_kv_transfer"}'
        ): "external_transfer_tokens",
    }
    result: dict[str, float] = {}
    for line in lines:
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        if name in wanted:
            result[wanted[name]] = float(value)
    return result


def _stream_chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
    data = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
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
                    first_token_at = first_token_at or time.perf_counter()
                    pieces.append(piece)
    ended = time.perf_counter()
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "total_seconds": ended - started,
        "text": "".join(pieces),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    image1 = _image_url(repo / "tests/multimodal/assets/image1.png")
    image2 = _image_url(repo / "tests/multimodal/assets/image2.png")
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image1}},
        {"type": "image_url", "image_url": {"url": image2}},
        {
            "type": "text",
            "text": (
                "Preserve both images and sentinel MULTIMODAL-A."
                + " imagecontext" * 59_500
                + "\nTranscribe both images in order, then say MULTIMODAL-A."
            ),
        },
    ]
    first_messages = [{"role": "user", "content": user_content}]

    result: dict[str, Any] = {"metrics_before": _metric_values()}
    cold = _stream_chat(first_messages)
    result["multimodal_cold"] = cold
    time.sleep(15)
    result["metrics_after_cold"] = _metric_values()

    pressure_messages = [
        {
            "role": "user",
            "content": (
                "Pressure-session sentinel TEXT-B."
                + " pressure" * 219_000
                + "\nReply exactly TEXT-B."
            ),
        }
    ]
    result["pressure_session"] = _stream_chat(pressure_messages)
    time.sleep(15)
    result["metrics_after_pressure"] = _metric_values()

    continued_messages = [
        *first_messages,
        {"role": "assistant", "content": cold["text"]},
        {
            "role": "user",
            "content": (
                "What exact text was in each image? End with MULTIMODAL-RELOADED."
            ),
        },
    ]
    result["multimodal_reload"] = _stream_chat(continued_messages)
    time.sleep(15)
    result["metrics_after_reload"] = _metric_values()
    combined = (cold["text"] + " " + result["multimodal_reload"]["text"]).lower()
    result["content_valid"] = all(
        phrase in combined
        for phrase in ("hello, ai world", "safe is important", "multimodal")
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
