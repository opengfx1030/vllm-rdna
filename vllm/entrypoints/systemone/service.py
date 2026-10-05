# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local decision service. Idle when the backend is off."""

import asyncio
from argparse import Namespace

from vllm.entrypoints.systemone.backends.gliner2 import Gliner2Backend
from vllm.entrypoints.systemone.backends.http_proxy import HTTPBackend
from vllm.entrypoints.systemone.backends.stub import StubBackend
from vllm.entrypoints.systemone.batcher import MicroBatcher
from vllm.entrypoints.systemone.config import (
    SystemOneConfig,
    resolve_systemone_config,
)
from vllm.entrypoints.systemone.errors import SystemOneError
from vllm.entrypoints.systemone.protocol import DecisionRequest
from vllm.logger import init_logger

logger = init_logger(__name__)


class SystemOneService:
    """One decision backend plus the micro-batcher in front of it."""

    def __init__(
        self,
        config: SystemOneConfig,
        *,
        backend: StubBackend | HTTPBackend | Gliner2Backend | None = None,
        engine_args: Namespace | None = None,
        engine_present: bool = True,
    ) -> None:
        self.config = config
        self.model_name = config.model or ""
        self._backend = backend or build_backend(
            config, engine_args=engine_args, engine_present=engine_present
        )
        self._batcher: MicroBatcher | None = None
        self._started = False

    async def start(self) -> None:
        self._batcher = MicroBatcher(
            self._backend.decide_batch,
            max_batch=self.config.max_batch,
            max_wait_ms=self.config.max_wait_ms,
            max_queue=self.config.max_queue,
            timeout_s=self.config.timeout_s,
        )
        try:
            await asyncio.get_running_loop().run_in_executor(
                self._batcher.executor, self._backend.load
            )
        except Exception:
            await self._batcher.shutdown()
            self._batcher = None
            raise
        self._batcher.start()
        self._started = True
        logger.info(
            "System One route enabled backend=%s model=%s device=%s",
            self.config.backend,
            self.config.model,
            self.config.device,
        )

    async def shutdown(self) -> None:
        self._started = False
        batcher = self._batcher
        self._batcher = None
        if batcher is not None:
            await batcher.shutdown()

    async def ask(self, request: DecisionRequest) -> dict:
        if not self._started or self._batcher is None:
            raise SystemOneError("systemone is not ready", 503)
        return await self._batcher.submit(request)


def build_backend(
    config: SystemOneConfig,
    *,
    engine_args: Namespace | None,
    engine_present: bool,
) -> StubBackend | HTTPBackend | Gliner2Backend:
    if config.backend == "stub":
        return StubBackend()
    if config.backend == "http":
        return HTTPBackend(
            config.url or "",
            timeout_s=config.timeout_s,
            api_key=config.api_key,
        )
    if config.backend == "gliner2":
        if not config.model:
            raise SystemOneError(
                "--systemone-model is required for the gliner2 backend", 400
            )
        return Gliner2Backend(
            model=config.model,
            device=config.device,
            dtype=config.dtype,
            vram_reserve_gb=config.vram_reserve_gb,
            engine_args=engine_args,
            check_engine_devices=engine_present,
        )
    raise SystemOneError(f"unknown systemone backend {config.backend!r}", 400)


async def init_systemone_state(
    state,
    args: Namespace,
    *,
    engine_present: bool = True,
) -> None:
    """Start the decision service when the serve-side backend is not off.

    No-op when the feature is off, which is the default. The model load runs
    on the batcher's worker thread, not on the engine workers.
    """
    config = resolve_systemone_config(args)
    if config.backend == "off":
        return
    service = SystemOneService(config, engine_args=args, engine_present=engine_present)
    await service.start()
    state.systemone_service = service
