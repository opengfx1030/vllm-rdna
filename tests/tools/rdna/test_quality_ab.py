# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject incomparable quality evidence through the real comparison CLI."""

import copy
import hashlib
import json
import math
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

CLIENT = (
    Path(__file__).resolve().parents[3] / "tools/rdna/port_v031/quality_ab.py"
)


def _bundle(arm):
    passages = []
    response = {"usage": {"prompt_tokens_details": {"cached_tokens": 0}}}
    for index, text in enumerate(("First passage.", "Second passage.")):
        passages.append({
            "passage": text,
            "passage_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_tokens": 3,
            "tokens": ["first", "second", "third"],
            "token_logprobs": [None, -1.0, -2.0],
            "ppl": math.exp(1.5),
            "greedy_tokens": ["answer"],
            "echo_response": copy.deepcopy(response),
            "greedy_response": copy.deepcopy(response),
            "cache_salts": {
                "echo": f"{arm}-{index}-echo",
                "greedy": f"{arm}-{index}-greedy",
            },
        })
    return {"complete": True, "require_cold_cache": True, "passages": passages}


def _compare(tmp_path, a, b):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, value in zip(paths, (a, b)):
        path.write_text(json.dumps(value))
    return subprocess.run(
        [sys.executable, str(CLIENT), "compare", *map(str, paths)],
        capture_output=True, text=True, timeout=20,
    )


def test_comparison_recomputes_ppl_from_raw_scores(tmp_path):
    a, b = _bundle("a"), _bundle("b")
    a["ppl"], b["ppl"] = 1.0, 100.0
    result = _compare(tmp_path, a, b)
    assert result.returncode == 0, result.stdout + result.stderr
    b["passages"][0]["token_logprobs"][1] = -3.0
    result = _compare(tmp_path, a, b)
    assert result.returncode == 1 and "FAIL" in result.stdout


@pytest.mark.parametrize("mutation", [
    "truncated", "incomplete", "missing", "nan", "infinity", "unknown-cache",
    "warm-cache", "same-arm-salt", "cross-arm-salt", "passage", "tokenization",
])
def test_incomparable_bundles_cannot_pass(tmp_path, mutation):
    a, b = _bundle("a"), _bundle("b")
    first = b["passages"][0]
    if mutation == "truncated":
        b["passages"].pop()
    elif mutation == "incomplete":
        b["complete"] = False
    elif mutation in ("missing", "nan", "infinity"):
        first["token_logprobs"][1] = {
            "missing": None, "nan": float("nan"), "infinity": float("-inf")
        }[mutation]
    elif mutation == "unknown-cache":
        first["echo_response"]["usage"] = {}
    elif mutation == "warm-cache":
        first["echo_response"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    elif mutation == "same-arm-salt":
        b["passages"][1]["cache_salts"] = first["cache_salts"]
    elif mutation == "cross-arm-salt":
        first["cache_salts"] = a["passages"][0]["cache_salts"]
    elif mutation == "passage":
        first["passage"] += " changed"
    elif mutation == "tokenization":
        first["tokens"][1] = "different"
    result = _compare(tmp_path, a, b)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "INCOMPARABLE" in result.stdout


def test_collect_cli_keeps_raw_evidence_without_recording_bearer_key(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((body, self.headers.get("Authorization")))
            tokens = ["one", "two", "answer"] if body.get("echo") else ["answer"]
            logprobs = [None, -1.25, -2.5] if body.get("echo") else [-2.5]
            response = json.dumps({
                "usage": {
                    "prompt_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "choices": [{
                    "text": "answer",
                    "logprobs": {"tokens": tokens, "token_logprobs": logprobs},
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    passage = tmp_path / "passage.txt"
    passage.write_text("one two " * 20)
    key = tmp_path / "api.key"
    key.write_text("secret-bearer-value")
    output = tmp_path / "collection.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        result = subprocess.run([
            sys.executable, str(CLIENT), "collect",
            "--url", f"http://127.0.0.1:{server.server_port}", "--model", "test",
            "--out", str(output), "--files", str(passage), "--passages", "2",
            "--passage-chars", "20", "--gen-tokens", "1",
            "--key-file", str(key), "--require-cold-cache",
        ], capture_output=True, text=True, timeout=20)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert result.returncode == 0, result.stdout + result.stderr
    bundle = json.loads(output.read_text())
    assert bundle["complete"] is True and bundle["scored_tokens"] == 2
    assert len(requests) == 4
    assert len({body["cache_salt"] for body, _ in requests}) == 4
    assert all(auth == "Bearer secret-bearer-value" for _, auth in requests)
    assert "secret-bearer-value" not in output.read_text()
    assert all("echo_response" in row and "greedy_response" in row
               for row in bundle["passages"])
