# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone System One server (Mode B).

GLiNER2 is an encoder classifier, not a vLLM model class. ``vllm serve``
cannot load it. This process serves only ``POST /v1/systemone`` with the
same micro-batcher as Mode A, and it does not start EngineCore.

There is no main-engine GPU set in this process, so a ``cuda:N`` device is
not checked against tensor-parallel ranks. The free-VRAM reserve still applies.
"""

import argparse
import os
from argparse import Namespace
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vllm.entrypoints.systemone.api_router import attach_router
from vllm.entrypoints.systemone.config import (
    add_systemone_cli_args,
    validate_systemone_args,
)
from vllm.entrypoints.systemone.service import init_systemone_state


def build_systemone_app(args: Namespace) -> FastAPI:
    """App that serves only the decision route."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_systemone_state(app.state, args, engine_present=False)
        try:
            yield
        finally:
            service = getattr(app.state, "systemone_service", None)
            if service is not None:
                await service.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.args = args
    attach_router(app)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.entrypoints.systemone.server",
        description=(
            "Serve POST /v1/systemone only. GLiNER2 is not a vLLM model "
            "and cannot be loaded by `vllm serve`."
        ),
    )
    add_systemone_cli_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--model",
        default=None,
        help="Alias of --systemone-model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Alias of --systemone-device.",
    )
    args = parser.parse_args(argv)
    _apply_aliases(parser, args)
    if args.systemone_backend is None and not os.environ.get("VLLM_SYSTEMONE_BACKEND"):
        args.systemone_backend = "gliner2"
    validate_systemone_args(args)
    import uvicorn

    uvicorn.run(
        build_systemone_app(args),
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _apply_aliases(parser: argparse.ArgumentParser, args: Namespace) -> None:
    if args.model is not None:
        if args.systemone_model not in (None, args.model):
            parser.error("--model and --systemone-model disagree")
        args.systemone_model = args.model
    if args.device is not None:
        if args.systemone_device not in (None, args.device):
            parser.error("--device and --systemone-device disagree")
        args.systemone_device = args.device


if __name__ == "__main__":
    main()
