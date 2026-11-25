#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision probe for an HI2 Inkling-2-Mini server booted with VISION=1.

Sends deterministic PNG images as OpenAI ``image_url`` data URLs and checks that
the multimodal class answers; the same images go to the stock server so the two
answers can be compared (word overlap, not equality: the experts are quantized).
The bit-exact equality of the 422 bf16 ``model.visual.*`` tensors is a CPU check
(``check_real_dir.py --vision``); this probe is the served path.

    $VENV/bin/python tools/inkling_hi2/vision_probe.py \
        --make-images ~/w4a8_runs/inkling-hi2-runtime/images
    $VENV/bin/python tools/inkling_hi2/vision_probe.py \
        --hi2-url http://127.0.0.1:18230 --stock-url http://127.0.0.1:18222 \
        --images ~/w4a8_runs/inkling-hi2-runtime/images --json out.json

A set from the eval harness can be passed with ``--images`` instead of
``--make-images`` (any *.png/*.jpg); one request per image.

Exit codes: 0 every check passed, 1 a check failed, 2 bad input, 3 a server (or
the image set) is unavailable.
"""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures as futures
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

PROMPT = "Describe the picture in one sentence."
TEXT_PROMPT = "Name the capital of France in one word."
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def write_png(path: Path, width: int, height: int, pixel) -> None:
    """A minimal RGB PNG writer (no Pillow): pixel(x, y) -> (r, g, b)."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel(x, y))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def make_images(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    w = h = 64
    images = []

    def square(x, y):
        return (220, 30, 30) if 16 <= x < 48 and 16 <= y < 48 else (250, 250, 250)

    def disc(x, y):
        inside = (x - 32) ** 2 + (y - 32) ** 2 <= 20**2
        return (30, 60, 220) if inside else (250, 250, 250)

    def checker(x, y):
        return (10, 10, 10) if ((x // 8) + (y // 8)) % 2 else (245, 245, 245)

    def bars(x, y):
        if x < w // 3:
            return (200, 40, 40)
        if x < 2 * w // 3:
            return (40, 180, 60)
        return (40, 60, 200)

    for name, pixel in (
        ("red_square", square),
        ("blue_disc", disc),
        ("checker", checker),
        ("color_bars", bars),
    ):
        path = directory / f"{name}.png"
        write_png(path, w, h, pixel)
        images.append(path)
    return images


def _post(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status, json.loads(resp.read())


def health(base: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def describe(base: str, image: Path, max_tokens: int, timeout: float) -> dict:
    data = binascii.b2a_base64(image.read_bytes(), newline=False).decode()
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{data}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    status, body = _post(f"{base}/v1/chat/completions", payload, timeout)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"{base}: HTTP {status} for {image.name}")
    choice = (body.get("choices") or [{}])[0]
    return {
        "text": (choice.get("message") or {}).get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "tokens": int((body.get("usage") or {}).get("completion_tokens") or 0),
    }


def text_only(base: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    status, body = _post(f"{base}/v1/chat/completions", payload, timeout)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"{base}: HTTP {status} for a text request")
    choice = (body.get("choices") or [{}])[0]
    return {
        "text": (choice.get("message") or {}).get("content") or "",
        "finish_reason": choice.get("finish_reason"),
    }


def words(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {w for w in cleaned.split() if len(w) > 2}


def overlap(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def clean(result: dict) -> bool:
    return (
        bool(result["text"].strip())
        and "\ufffd" not in result["text"]
        and result.get("finish_reason") in {"stop", "length"}
    )


@dataclass
class Report:
    checks: dict[str, bool] = field(default_factory=dict)
    numbers: dict[str, object] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, note: str = "") -> None:
        self.checks[name] = bool(ok)
        self.log.append(
            f"{'PASS' if ok else 'FAIL'} {name}{(': ' + note) if note else ''}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hi2-url", default=None, help="the VISION=1 HI2 server")
    ap.add_argument("--stock-url", default=None, help="the stock server (comparison)")
    ap.add_argument("--images", default=None, help="a directory of PNG/JPG images")
    ap.add_argument("--make-images", default=None, help="write the built-in set here")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--min-overlap", type=float, default=0.3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.make_images:
        made = make_images(Path(args.make_images).expanduser())
        print(f"vision_probe: wrote {len(made)} images to {args.make_images}")
        if not args.hi2_url:
            return 0
    if not args.hi2_url:
        print("vision_probe: give --hi2-url (or --make-images alone)", file=sys.stderr)
        return 2
    if not args.images:
        print("vision_probe: give --images DIR or --make-images DIR", file=sys.stderr)
        return 2
    images = sorted(
        p
        for p in Path(args.images).expanduser().iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"vision_probe: no PNG/JPG images in {args.images}", file=sys.stderr)
        return 3

    hi2 = args.hi2_url.rstrip("/")
    stock = args.stock_url.rstrip("/") if args.stock_url else None
    if not health(hi2):
        print(f"vision_probe: {hi2} is not serving (/health)", file=sys.stderr)
        return 3
    if stock and not health(stock):
        print(f"vision_probe: {stock} is not serving (/health)", file=sys.stderr)
        return 3

    report = Report()
    for image in images:
        got = describe(hi2, image, args.max_tokens, args.timeout)
        report.check(f"answered[{image.stem}]", clean(got), f"{got['tokens']} tokens")
        report.numbers[f"hi2[{image.stem}]"] = got["text"]
        if stock:
            ref = describe(stock, image, args.max_tokens, args.timeout)
            report.numbers[f"stock[{image.stem}]"] = ref["text"]
            score = overlap(got["text"], ref["text"])
            report.numbers[f"overlap[{image.stem}]"] = round(score, 3)
            report.check(
                f"agree_with_stock[{image.stem}]",
                score >= args.min_overlap,
                f"word overlap {score:.2f}",
            )

    # The tower must not break plain text (it shares the decoder) and a mixed
    # batch must survive: images and text in flight together.
    if args.concurrency > 1:
        jobs = []
        for i in range(args.concurrency):
            if i % 2 == 0:
                jobs.append(("image", images[i % len(images)]))
            else:
                jobs.append(("text", None))

        def run(job) -> bool:
            kind, image = job
            try:
                if kind == "image":
                    return clean(describe(hi2, image, args.max_tokens, args.timeout))
                return clean(text_only(hi2, 16, args.timeout))
            except RuntimeError:
                return False

        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(run, jobs))
        report.check(
            f"mixed_batch_{args.concurrency}",
            all(results),
            f"{sum(results)}/{len(results)}",
        )

    for line in report.log:
        print(line)
    for name, value in report.numbers.items():
        if name.startswith(("overlap",)):
            print(f"  {name}: {value}")
    passed = all(report.checks.values())
    print("PASS" if passed else "FAIL")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"checks": report.checks, "numbers": report.numbers, "log": report.log},
                indent=2,
            )
            + "\n"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
