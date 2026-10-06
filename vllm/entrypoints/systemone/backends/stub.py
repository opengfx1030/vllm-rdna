# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic backend for tests. It does not read the state."""

import threading
import time

from vllm.entrypoints.systemone.protocol import (
    DecisionRequest,
    build_local_response,
    stub_probabilities,
)


class StubBackend:
    """Answers every question with a fixed distribution."""

    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.delay_s = delay_s
        self.started = started
        self.release = release
        self.batch_sizes: list[int] = []

    def load(self) -> None:
        return None

    def decide_batch(self, requests: list[DecisionRequest]) -> list[dict]:
        self.batch_sizes.append(len(requests))
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            if not self.release.wait(timeout=5):
                raise TimeoutError("stub backend was not released")
        elif self.delay_s:
            time.sleep(self.delay_s)
        return [
            build_local_response(
                request.model,
                request.tasks,
                {task.name: stub_probabilities(task) for task in request.tasks},
            )
            for request in requests
        ]
