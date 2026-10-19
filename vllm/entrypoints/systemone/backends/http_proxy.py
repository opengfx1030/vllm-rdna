# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proxy ``POST /v1/systemone`` to another System One server."""

import json
import urllib.error
import urllib.request

from vllm.entrypoints.systemone.errors import SystemOneError
from vllm.entrypoints.systemone.protocol import DecisionRequest


class HTTPBackend:
    """Forward each decision to ``url``. Batching stays local to the caller."""

    def __init__(self, url: str, *, timeout_s: float, api_key: str | None) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._api_key = api_key

    def load(self) -> None:
        if not self._url:
            raise SystemOneError(
                "--systemone-url is required for the http backend", 400
            )

    def decide_batch(self, requests: list[DecisionRequest]) -> list[dict]:
        return [self._post(request) for request in requests]

    def _post(self, request: DecisionRequest) -> dict:
        data = json.dumps(request.body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        outgoing = urllib.request.Request(
            self._url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(outgoing, timeout=self._timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            status = exc.code if 400 <= exc.code <= 599 else 502
            raise SystemOneError(
                f"systemone upstream returned {exc.code}: {detail}", status
            ) from exc
        except Exception as exc:
            raise SystemOneError(
                f"systemone upstream request failed: {exc}", 503
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemOneError("systemone upstream returned non-JSON", 502) from exc
        if not isinstance(payload, dict) or "answers" not in payload:
            raise SystemOneError("systemone upstream response is missing answers", 502)
        return payload
