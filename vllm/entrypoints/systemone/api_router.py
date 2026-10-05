# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /v1/systemone``. Registered only when the serve-side backend is on.

vllm-project/vllm#59299 registers the same path from
``vllm.entrypoints.generate.structured_decisions`` when
``--enable-structured-decisions`` is set. If that route is already on the
app, this module does not register a second one. The decision service is
then installed as ``systemone_backend_provider`` and answers only requests
whose ``model`` is the configured decision model.
"""

import json
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.systemone.errors import SystemOneError, error_body
from vllm.entrypoints.systemone.protocol import parse_request
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATH = "/v1/systemone"


class _WireResponse:
    """Object with ``model_dump`` so an upstream route can return our JSON."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return self._payload


def systemone_route_registered(app: FastAPI) -> bool:
    """True when some router already owns POST ``/v1/systemone``."""
    for route in app.routes:
        if getattr(route, "path", None) != _PATH:
            continue
        methods = getattr(route, "methods", None) or set()
        if "POST" in methods:
            return True
    return False


def register_structured_decisions_api_router(app: FastAPI) -> bool:
    """Mount our route unless #59299 (or anything else) already did.

    Returns:
        True when this module registered the route.
    """
    if systemone_route_registered(app):
        app.state.systemone_stepped_aside = True
        logger.info(
            "POST /v1/systemone is already registered; the systemone "
            "backend will not add a second route"
        )
        return False
    app.state.systemone_stepped_aside = False
    app.add_api_route(_PATH, systemone, methods=["POST"])
    return True


def attach_router(app: FastAPI) -> None:
    """Mount the decision route. Does not touch ``/v1/models``.

    Used by the standalone server, which has no upstream route to defer to.
    Still refuses to add a second POST if one is already present.
    """
    register_structured_decisions_api_router(app)


def install_decision_provider(state: Any, service: Any) -> None:
    """Publish the loaded backend, and wrap an upstream handler if we stepped aside.

    The wrapper answers when ``request.model`` is this deployment's decision
    model. Every other model stays on the upstream handler.
    """
    state.systemone_backend_provider = service
    if not getattr(state, "systemone_stepped_aside", False):
        return
    handler = getattr(state, "serving_structured_decisions", None)
    if handler is None or getattr(handler, "_rdna_systemone_wrapped", False):
        if handler is None:
            logger.warning(
                "POST /v1/systemone belongs to another router and no "
                "serving_structured_decisions handler is installed, so the "
                "systemone backend is not reachable on that route"
            )
        return
    original = handler.create_decision
    handler.create_decision = _wrap_create_decision(original, service)
    handler._rdna_systemone_wrapped = True


def _wrap_create_decision(original: Callable, service: Any) -> Callable:
    async def create_decision(request: Any, raw_request: Any = None) -> Any:
        model = getattr(request, "model", None)
        if service.model_name and model == service.model_name:
            payload = _payload_of(request)
            raw = _dump_bytes(payload)
            parsed = parse_request(raw, default_model=service.model_name)
            return _WireResponse(await service.ask(parsed))
        return await original(request, raw_request)

    return create_decision


def _payload_of(request: Any) -> Any:
    dump = getattr(request, "model_dump", None)
    if callable(dump):
        return dump()
    return request


def _dump_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


async def systemone(request: Request) -> JSONResponse:
    service = getattr(request.app.state, "systemone_service", None)
    if service is None:
        return JSONResponse(
            error_body("systemone is not enabled", 503),
            status_code=503,
        )
    raw = await request.body()
    try:
        parsed = parse_request(raw, default_model=service.model_name)
        payload = await service.ask(parsed)
    except SystemOneError as exc:
        return JSONResponse(
            error_body(exc.message, exc.status_code),
            status_code=exc.status_code,
        )
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
