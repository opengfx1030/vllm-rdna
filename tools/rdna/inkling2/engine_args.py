# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine options the Inkling-2 check tools share: HIP graphs and tensor parallelism.

``real_check.py``, ``long_context_check.py``, ``graph_check.py`` and
``vision_engine_check.py`` boot the same ``LLM``; this module owns the arguments that
select the graph mode, the graph capture grid and the all-reduce backend, so that the
tools cannot disagree about what ``--cudagraph-mode FULL_AND_PIECEWISE`` means.
"""

from __future__ import annotations

import argparse
import os


def add_graph_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--cudagraph-mode",
        default="eager",
        help="eager (enforce_eager) or a CUDAGraphMode name: FULL_AND_PIECEWISE (the "
        "serving recipe), FULL_DECODE_ONLY, PIECEWISE",
    )
    ap.add_argument(
        "--decode-sizes",
        default="",
        help="request counts of the uniform-decode (FULL) graphs, e.g. 1,2,4,8 "
        "(default: the model's dense grid 1 .. 16)",
    )
    ap.add_argument(
        "--prefill-sizes",
        default="",
        help="token counts of the prefill / mixed (PIECEWISE) graphs, e.g. "
        "16,64,256,1024 (default: the engine's grid)",
    )


def graph_kwargs(
    mode: str, prefill_sizes: str = "", decode_sizes: str = ""
) -> dict[str, object]:
    """``LLM(...)`` keyword arguments of a graph mode: ``enforce_eager`` for ``eager``,
    else a ``compilation_config`` with the mode and the capture grids that were
    given."""
    if mode == "eager":
        return {"enforce_eager": True}
    config: dict[str, object] = {"cudagraph_mode": mode}
    if prefill_sizes:
        sizes = [int(s) for s in prefill_sizes.split(",")]
        config["prefill_cudagraph_capture_token_sizes"] = sizes
        config["max_prefill_cudagraph_capture_tokens"] = max(sizes)
    if decode_sizes:
        config["decode_cudagraph_capture_request_sizes"] = [
            int(s) for s in decode_sizes.split(",")
        ]
    return {"compilation_config": config}


def add_tp_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument(
        "--ar-backend",
        choices=["rccl", "rdna"],
        default="rccl",
        help="all-reduce of a tensor-parallel group: RCCL, or the one-shot HIP "
        "all-reduce (VLLM_RDNA_AR=1; RCCL above its size gate)",
    )
    ap.add_argument(
        "--ar-max-kb",
        type=int,
        help="VLLM_RDNA_AR_MAX_KB / ONESHOT_KB (the default gate is 64 KiB)",
    )


def apply_tp_env(ar_backend: str, ar_max_kb: int | None) -> None:
    """Environment of the engine processes (call before the engine is built)."""
    if ar_backend != "rdna":
        return
    os.environ["VLLM_RDNA_AR"] = "1"
    if ar_max_kb:
        os.environ["VLLM_RDNA_AR_MAX_KB"] = str(ar_max_kb)
        os.environ["VLLM_RDNA_AR_ONESHOT_KB"] = str(ar_max_kb)
