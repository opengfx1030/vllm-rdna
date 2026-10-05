# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /v1/systemone``. Registered only when the serve-side backend is on."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.systemone.errors import SystemOneError
from vllm.entrypoints.systemone.protocol import parse_request


def attach_router(app: FastAPI) -> None:
    """Mount the decision route. Does not touch ``/v1/models``."""

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        service = getattr(request.app.state, "systemone_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="systemone is not enabled")
        raw = await request.body()
        try:
            parsed = parse_request(raw, default_model=service.model_name)
            payload = await service.ask(parsed)
        except SystemOneError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=exc.message
            ) from exc
        return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
