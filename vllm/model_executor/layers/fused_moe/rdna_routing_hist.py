# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in capture of per-layer MoE routing histograms (RDNA2 prefill study).

``VLLM_RDNA_MOE_HIST_DIR=<dir>`` makes expert-parallel rank 0 write one JSON
line per MoE layer call with at least ``VLLM_RDNA_MOE_HIST_MIN_TOKENS`` tokens
(default 512, so decode steps are skipped) to ``<dir>/moe_routing_hist.jsonl``
(truncated when the first record of a server process is written).

First line (written once per process)::

    {
        "meta": {
            "num_experts": 512,
            "ep_size": 4,
            "top_k": 10,
            "block_size_m": 4,
            "linear_placement": true,
            "pid": 123,
        }
    }

Then one line per recorded call::

    {"seq": 17, "step": 0, "layer": "model.layers.3.mlp.experts",
     "layer_idx": 3, "tokens": 2048, "counts": [... num_experts ints ...]}

``counts[e]`` is the number of (token, top-k slot) pairs routed to GLOBAL
expert ``e``. ``step`` increments whenever ``layer_idx`` does not increase,
i.e. once per forward pass. Every rank computes the same routing, so one rank
is enough. ``tools/rdna/prof/moe_placement.py`` turns a capture into
per-layer expert-to-rank placements and scores them on held-out steps.

The hook copies the histogram to the host on every recorded call (one device
sync), so it is for capture runs only, never for timed runs. It is skipped
during CUDA-graph capture and while compiling.
"""

import json
import os
import time
from typing import TextIO

import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HIST_DIR_ENV = "VLLM_RDNA_MOE_HIST_DIR"
HIST_MIN_TOKENS_ENV = "VLLM_RDNA_MOE_HIST_MIN_TOKENS"
HIST_FILE_NAME = "moe_routing_hist.jsonl"
DEFAULT_MIN_TOKENS = 512

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_index(layer_name: str) -> int:
    """Decoder layer index parsed from a module name, or -1 when absent."""
    match = _LAYER_INDEX.search(layer_name)
    return int(match.group(1)) if match else -1


class RoutingHistogram:
    """Append-only routing histogram writer (see the module docstring)."""

    def __init__(self, directory: str, min_tokens: int = DEFAULT_MIN_TOKENS):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, HIST_FILE_NAME)
        self.min_tokens = min_tokens
        self._file: TextIO | None = None
        self._seq = 0
        self._step = -1
        self._last_layer_idx = 1 << 30  # the first record opens step 0
        self._warned_unindexed = False

    @classmethod
    def from_env(cls) -> "RoutingHistogram | None":
        directory = os.environ.get(HIST_DIR_ENV, "")
        if not directory:
            return None
        min_tokens = int(os.environ.get(HIST_MIN_TOKENS_ENV, DEFAULT_MIN_TOKENS))
        logger.warning(
            "RDNA MoE routing-histogram capture is ON (%s=%s, min tokens %d): "
            "one device sync per MoE call, for capture runs only",
            HIST_DIR_ENV,
            directory,
            min_tokens,
        )
        return cls(directory, min_tokens)

    @staticmethod
    def _is_recording_rank(layer: torch.nn.Module) -> bool:
        config = layer.moe_config
        rank = config.ep_rank if config.ep_size > 1 else config.tp_rank
        return rank == 0 and config.dp_rank == 0

    def record(self, layer: torch.nn.Module, topk_ids: torch.Tensor) -> None:
        """Record one MoE call; a no-op for small batches and non-zero ranks."""
        num_tokens = topk_ids.shape[0]
        if num_tokens < self.min_tokens:
            return
        if torch.compiler.is_compiling() or (
            topk_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            return
        if not self._is_recording_rank(layer):
            return
        name = getattr(layer, "layer_name", "") or ""
        idx = layer_index(name)
        if idx < 0:
            if not self._warned_unindexed:
                self._warned_unindexed = True
                logger.warning(
                    "RDNA MoE routing histogram: layer %r has no layer index; skipped",
                    name,
                )
            return
        num_experts = int(layer.global_num_experts)
        ids = topk_ids.reshape(-1)
        counts = torch.bincount(ids[ids >= 0], minlength=num_experts)
        row = counts[:num_experts].tolist()  # device sync
        out = self._file if self._file is not None else self._open(layer, num_experts)
        if idx <= self._last_layer_idx:
            self._step += 1
        self._last_layer_idx = idx
        self._seq += 1
        out.write(
            json.dumps(
                {
                    "seq": self._seq,
                    "step": self._step,
                    "layer": name,
                    "layer_idx": idx,
                    "tokens": int(num_tokens),
                    "counts": row,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    def _open(self, layer: torch.nn.Module, num_experts: int) -> TextIO:
        self._file = out = open(self.path, "w", buffering=1)  # noqa: SIM115
        meta = {
            "num_experts": num_experts,
            "ep_size": int(layer.moe_config.ep_size),
            "top_k": int(layer.top_k),
            "block_size_m": 4,
            "linear_placement": getattr(layer, "expert_placement_strategy", "linear")
            == "linear",
            "pid": os.getpid(),
            "time": time.time(),
        }
        out.write(json.dumps({"meta": meta}, separators=(",", ":")) + "\n")
        logger.info("RDNA MoE routing histogram -> %s", self.path)
        return out
