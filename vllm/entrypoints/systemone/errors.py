# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class SystemOneError(Exception):
    """Request failure with the HTTP status the route should return."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SystemOneStartupError(RuntimeError):
    """Fail closed before the server accepts traffic."""


# vLLM ``ErrorResponse`` field names. llama.cpp only requires an ``error``
# key; this object satisfies that and the vLLM error model.
_ERROR_TYPES = {
    400: "BadRequestError",
    429: "RateLimitError",
    500: "InternalServerError",
    501: "NotImplementedError",
    502: "BadGatewayError",
    503: "ServiceUnavailableError",
    504: "TimeoutError",
}


def error_body(message: str, status_code: int) -> dict:
    """JSON body for a System One failure. ``error.code`` is the HTTP status."""
    return {
        "error": {
            "message": message,
            "type": _ERROR_TYPES.get(status_code, "SystemOneError"),
            "param": None,
            "code": status_code,
        }
    }
