#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache plan of the real Inkling-2-Mini config, without loading weights or a GPU.

Builds the model of one tensor parallel rank on the meta device, takes the cache spec of
every layer, lets the engine's own planner (``vllm.v1.core.kv_cache_planning``) group
them, and prints per ``(max_model_len, tensor parallel size, speculative tokens)``:

* the groups (spec kind, layers, blocks one request of ``max_model_len`` needs),
* the bytes of one pool block and the memory one maximum-length request needs
  (``one-max-request``: what the engine requires before it accepts the length),
* with ``--budget-gib``, the largest ``max_model_len`` that memory holds
  (the planner's binary search, the same one ``--max-model-len -1`` runs).

A tensor parallel rank owns ``num_kv_heads / tp`` heads of every KV and conv cache, so
the per-rank specs are those of the model the rank builds
(``Inkling2LoopTextForCausalLM`` with ``TensorParallel(tp, 0)``): the KV pages shrink
with the heads, but the conv state of a rank holds its own k / v heads next to the
replicated attention / MLP streams spread over fewer heads, so its row is wider (832
channels per head at TP=1, 1408 at TP=2, 2560 at TP=4) and dividing the TP=1 spec would
undercount it.  With ``--drafts`` the conv ring has ``3 + drafts`` rows.  The device
capability and memory that ``create_engine_config`` asks for are patched to a V620
(gfx1030, 32 GiB), so the script runs on any machine that has the repo's torch
(``HIP_VISIBLE_DEVICES=`` keeps it off the GPUs)::

    python tools/rdna/inkling2/kv_plan.py --ckpt ~/hfcache/inkling2-mini \\
        --max-model-len 262144 1048576 --tp 1 2 4 --drafts 0 7 --budget-gib 9.0 17.3
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
from unittest import mock

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.models.inkling2_loop.amd.conv_state import Inkling2ConvSpec  # noqa: E402
from vllm.models.inkling2_loop.amd.model import (  # noqa: E402
    Inkling2LoopTextForCausalLM,
)
from vllm.models.inkling2_loop.amd.parallel import TensorParallel  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402
from vllm.platforms.interface import DeviceCapability  # noqa: E402
from vllm.utils.mem_utils import format_gib  # noqa: E402
from vllm.v1.core import kv_cache_planning as planning  # noqa: E402
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs  # noqa: E402

GIB = 2**30


@contextlib.contextmanager
def v620_platform():
    """``create_engine_config`` asks the platform for the device capability and memory:
    answer as a V620 (gfx1030, 32 GiB) so that no GPU is needed."""
    platform = type(current_platform)
    with (
        mock.patch.object(
            platform,
            "get_device_capability",
            classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
        ),
        mock.patch.object(
            platform,
            "get_device_total_memory",
            classmethod(lambda cls, device_id=0: 32 * 2**30),
        ),
    ):
        yield


def engine_config(args, max_model_len: int):
    engine_args = EngineArgs(
        model=os.path.expanduser(args.ckpt),
        dtype="float16",
        max_model_len=max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        limit_mm_per_prompt={"image": 0},
        enforce_eager=True,
    )
    cfg = engine_args.create_engine_config()
    cfg.cache_config.kv_cache_layout = args.layout
    return cfg


def layer_specs(cfg, tp: int) -> dict:
    """Cache spec of every layer of one TP=``tp`` rank (meta device, no weights): the
    owners of the model that rank builds."""
    context = cfg.compilation_config.static_forward_context
    context.clear()
    with set_current_vllm_config(cfg), torch.device("meta"):
        Inkling2LoopTextForCausalLM(
            vllm_config=cfg, serving=False, tp=TensorParallel(tp, 0)
        )
    return {name: layer.get_kv_cache_spec(cfg) for name, layer in context.items()}


def with_drafts(spec, drafts: int):
    """``spec`` with the conv ring of ``drafts`` speculative tokens (``k - 1 + drafts``
    rows of the rank's own row width); any other spec is unchanged."""
    if not isinstance(spec, Inkling2ConvSpec):
        return spec
    ring_rows = spec.sliding_window - 1 + drafts
    return dataclasses.replace(
        spec, state_content_bytes=ring_rows * spec.head_size * spec.dtype.itemsize
    )


def rank_groups(cfg, specs: dict, drafts: int):
    """KV cache groups of one TP rank (``specs``: :func:`layer_specs`) with ``drafts``
    speculative tokens."""
    groups = planning.get_kv_cache_groups(
        cfg, {name: with_drafts(spec, drafts) for name, spec in specs.items()}
    )
    return planning.get_kv_cache_config_builder(cfg), groups


def group_rows(cfg, groups) -> list[tuple[str, int, int]]:
    rows = []
    for group in groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            blocks = spec.max_memory_usage_pages(cfg)
            first = next(iter(spec.kv_cache_specs.values()))
        else:
            blocks = -(-spec.max_memory_usage_bytes(cfg) // spec.page_size_bytes)
            first = spec
        rows.append((type(first).__name__, len(group.layer_names), blocks))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/hfcache/inkling2-mini")
    ap.add_argument("--max-model-len", type=int, nargs="+", default=[4096])
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--drafts", type=int, nargs="+", default=[0])
    ap.add_argument("--layout", default="BLNHC")
    ap.add_argument(
        "--budget-gib",
        type=float,
        nargs="*",
        default=[],
        help="per-rank KV memory: print the largest max_model_len it holds",
    )
    args = ap.parse_args()

    with v620_platform():
        for max_len in args.max_model_len:
            cfg = engine_config(args, max_len)
            print(
                f"max_model_len {max_len}: max_num_batched_tokens "
                f"{args.max_num_batched_tokens}, max_in_flight_tokens "
                f"{cfg.max_in_flight_tokens}, block size {cfg.cache_config.block_size}",
                flush=True,
            )
            for tp in args.tp:
                specs = layer_specs(cfg, tp)
                for drafts in args.drafts:
                    builder, groups = rank_groups(cfg, specs, drafts)
                    per_block = planning._pool_bytes_per_block_from_builder(
                        builder, cfg, groups
                    )
                    need = planning._max_memory_usage_bytes_from_groups(
                        builder, cfg, groups
                    )
                    rows = group_rows(cfg, groups)
                    blocks = sum(r[2] for r in rows)
                    print(
                        f"L={max_len} tp={tp} drafts={drafts} kv_heads={8 // tp}: "
                        f"groups={len(groups)} pool block {per_block / 2**20:.3f} MiB, "
                        f"one max-length request {blocks} blocks = "
                        f"{blocks * per_block / GIB:.3f} GiB "
                        f"(planner: {format_gib(need)} GiB)",
                        flush=True,
                    )
                    print("   ", rows, flush=True)

        if not args.budget_gib:
            return 0
        cfg = engine_config(args, max(args.max_model_len))
        for tp in args.tp:
            specs = layer_specs(cfg, tp)
            for drafts in args.drafts:
                builder, groups = rank_groups(cfg, specs, drafts)
                for budget in args.budget_gib:
                    fit = planning._estimate_max_model_len_from_groups(
                        builder, cfg, groups, int(budget * GIB)
                    )
                    print(
                        f"tp={tp} drafts={drafts}: {budget} GiB of KV per rank hold "
                        f"max_model_len {fit} "
                        f"(capped at {cfg.model_config.max_model_len})",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
