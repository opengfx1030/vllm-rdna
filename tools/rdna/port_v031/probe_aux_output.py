#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the aux-output data plane of a running server (routed experts).

The server must run with ``--enable-return-routed-experts``; the flag is
boot-time only, there is no runtime toggle. Two-lane rides the flag: lane D
owns the capture buffer and the response field, lane P returns no rows.

On a feature-off server (the A/B baseline arm) run with ``--capture-ref``:
the probe asserts the field is absent and saves the greedy texts. On the
feature-on arm run with ``--compare <ref.json> --layers L --topk K``: it
asserts the same greedy texts and that every completion carries
``routed_experts`` (a base64 ``.npy`` int array shaped
``[tokens, layers, topk]``, values in ``[0, experts)``).

Exit 0 only when all checks pass.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request

import numpy as np
import pybase64 as base64

PROMPTS = (
    "The capital of France is",
    "1+1=",
    " ".join(str(i) for i in range(400)),
)


def post(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": None,
        }
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def decode_routed(raw: str) -> np.ndarray:
    buffer = io.BytesIO(base64.b64decode(raw))
    return np.load(buffer, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--capture-ref")
    parser.add_argument("--compare")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--experts", type=int)
    args = parser.parse_args()

    texts = []
    failures = []
    for prompt in PROMPTS:
        data = post(args.url, args.model, prompt, args.max_tokens)
        choice = data["choices"][0]
        text = choice["text"]
        texts.append(text)
        routed_raw = choice.get("routed_experts")
        if args.capture_ref:
            if routed_raw is not None:
                failures.append(
                    f"feature off returned routed_experts for prompt {prompt[:24]!r}"
                )
            continue
        if routed_raw is None:
            failures.append(
                f"feature on returned no routed_experts for prompt {prompt[:24]!r}"
            )
            continue
        routed = decode_routed(routed_raw)
        if routed.ndim != 3:
            failures.append(
                f"routed_experts ndim {routed.ndim} for prompt {prompt[:24]!r}"
            )
            continue
        tokens, layers, topk = routed.shape
        if tokens <= 0:
            failures.append(f"routed_experts has no tokens: {routed.shape}")
        if args.layers is not None and layers != args.layers:
            failures.append(
                f"routed_experts layers {layers} != {args.layers} for prompt "
                f"{prompt[:24]!r}"
            )
        if args.topk is not None and topk != args.topk:
            failures.append(
                f"routed_experts topk {topk} != {args.topk} for prompt {prompt[:24]!r}"
            )
        if routed.min() < 0:
            failures.append("routed_experts has a negative expert id")
        if args.experts is not None and int(routed.max()) >= args.experts:
            failures.append(
                f"routed_experts expert id {int(routed.max())} >= {args.experts} "
                f"for prompt {prompt[:24]!r}"
            )
        if text != "" and not tokens >= 1:
            failures.append("routed_experts token count is inconsistent")
        print(
            f"PASS shape prompt={prompt[:24]!r} routed={routed.shape} "
            f"dtype={routed.dtype}",
            flush=True,
        )

    if args.capture_ref:
        with open(args.capture_ref, "w") as f:
            json.dump({"texts": texts}, f)
        print(f"captured {len(texts)} texts -> {args.capture_ref}")
    elif args.compare:
        with open(args.compare) as f:
            reference = json.load(f)["texts"]
        if texts != reference:
            for index, (got, want) in enumerate(zip(texts, reference)):
                if got != want:
                    failures.append(f"text {index} differs: {got!r} vs {want!r}")
            if len(texts) != len(reference):
                failures.append(f"text count {len(texts)} != {len(reference)}")

    for failure in failures:
        print(f"FAIL {failure}", flush=True)
    print(f"{'PASS' if not failures else 'FAIL'} aux-output probe")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
