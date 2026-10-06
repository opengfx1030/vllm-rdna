# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refuse hetero work inside HIP/CUDA graph capture."""

import torch


def stream_is_capturing() -> bool:
    """True when the current device stream is capturing a graph."""
    try:
        if not torch.cuda.is_available():
            return False
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def assert_outside_capture(capturing: bool | None = None) -> None:
    """Raise if a graph capture is in progress.

    Args:
        capturing: Test override. None reads the current stream.

    Raises:
        RuntimeError: Capture is active. Transfers, histogram updates,
            and hot-set re-placement stay outside the captured region.
    """
    active = stream_is_capturing() if capturing is None else capturing
    if active:
        raise RuntimeError(
            "hetero MoE transfers and hot-set updates stay outside "
            "HIP/CUDA graph capture. Serve this path eagerly or in "
            "piecewise mode so the side stream is not recorded."
        )
