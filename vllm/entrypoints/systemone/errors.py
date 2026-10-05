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
