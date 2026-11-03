#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 driver: the worker-level two-lane checks, without the engine (S3, 7.1).

Boots one ``GPUWorker`` with lane mode on, then feeds synthetic
``SchedulerOutput`` streams through the lane runners and measures:

* handoff equivalence: the same prompt+decode tokens from (a) one baseline
  runner, (b) the P runner's prefill followed by the D runner's handoff
  decode, for a set of prompt lengths;
* E-GIL: D step latency alone vs D step latency while P runs 1,024-token
  chunks on the other lane thread;
* P chunk rate alone vs while D decodes;
* KFD queue census, strict-mode sync raises, lane trace.

Artifacts (JSON + the lane trace) go under ``--out-dir``. The trace file is
written by the lane runtime under ``VLLM_CACHE_ROOT``, so this driver points
that at the run directory.

Example (HIP 9, one V620):

  HIP_VISIBLE_DEVICES=9 python tools/rdna/prof/lane_driver.py \
      --model ~/models/Qwen3.5-0.8B --lengths 1,5,1023,1024,1025 \
      --decode-steps 8 --out-dir ~/w4a8_runs/lanes/async/accept/m1

The attention backend defaults to RDNA_ATTN (the recipes' ``ATTN=fa``); pass
``--attention-backend TRITON_ATTN`` for the Triton path of ``ATTN=triton``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
from typing import Any

# The ten acceptance lengths (M1): block-size edges around 1k, 2k and 4k plus
# a 16k prefill and its tail. Run with --compile they are the compiled-handoff
# regression case: any mismatch fails the run (see ``main``).
DEFAULT_LENGTHS = (
    1,
    5,
    1023,
    1024,
    1025,
    2047,
    2048,
    4097,
    16384,
    16385,
)
DEFAULT_DECODE_STEPS = 8
PREFILL_CHUNK = 1024

LANE_DECODE = 0
LANE_PREFILL = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("MODEL", "facebook/opt-125m"))
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu", default=os.environ.get("HIP_VISIBLE_DEVICES", "9"))
    parser.add_argument("--kv-gib", type=float, default=8.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--attention-backend", default="RDNA_ATTN")
    parser.add_argument("--lane-decode-wgps", type=int, default=20)
    parser.add_argument("--lengths", default=",".join(str(n) for n in DEFAULT_LENGTHS))
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    parser.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-handoff", action="store_true")
    parser.add_argument(
        "--no-spec",
        action="store_true",
        help="disable speculative decoding (k=0) for the handoff bisect",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-gil", action="store_true")
    parser.add_argument("--decode-context", type=int, default=4096)
    parser.add_argument("--gil-context", type=int, default=16384)
    parser.add_argument("--solo-steps", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--soak-s",
        type=float,
        default=0.0,
        help="repeat handoff comparisons for this long (M1 no-hang gate)",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def preimport_env(args: argparse.Namespace) -> None:
    """Environment the lane runtime reads; set before importing vllm."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    # The recipes' queue budget: 3 baseline + 4 lane streams = 7 (hard 8).
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
    # TP1 still rendezvouses through env://; the launcher sets these.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    if args.attention_backend.upper() == "RDNA_ATTN":
        os.environ.setdefault("VLLM_USE_RDNA2_FA", "1")
    else:
        os.environ.setdefault("VLLM_USE_RDNA2_FA", "0")


def percentiles(samples: list[float], quantiles: tuple[float, ...]) -> dict:
    if not samples:
        return {}
    ordered = sorted(samples)
    out = {}
    for q in quantiles:
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        out[f"p{int(q * 100)}_ms"] = 1000.0 * ordered[index]
    return out


def summarize(samples: list[float]) -> dict:
    stats = percentiles(samples, (0.5, 0.9, 0.99))
    stats.update(
        {
            "n": len(samples),
            "mean_ms": 1000.0 * statistics.fmean(samples) if samples else 0.0,
            "max_ms": 1000.0 * max(samples) if samples else 0.0,
        }
    )
    return stats


class Allocator:
    """Per-KV-group block id allocator; ids are never reused."""

    def __init__(self, kv_cache_config: Any) -> None:
        self._groups = list(kv_cache_config.kv_cache_groups)
        self._num_blocks = int(kv_cache_config.num_blocks)
        # vLLM reserves block id 0 as the null/padding block; the real
        # KVCacheManager never hands it out. A single-block request on a
        # mamba/GDN model keeps its state in block 0 and then padding or a
        # dummy row clobbers it, so the driver starts at 1 (M1 L<=16 bug).
        self._next = [1 for _ in self._groups]

    def num_blocks(self) -> int:
        return self._num_blocks

    def reset(self) -> None:
        """Restart allocation: sequential runs may reuse the same blocks."""
        self._next = [1 for _ in self._groups]

    def allocate(self, num_tokens: int) -> tuple[list[int], ...]:
        ids: list[list[int]] = []
        for group, cursor in zip(self._groups, self._next):
            block_size = int(group.kv_cache_spec.block_size)
            need = math.ceil(num_tokens / block_size)
            if cursor + need > self._num_blocks:
                raise RuntimeError(
                    f"lane driver: block pool exhausted "
                    f"({cursor}+{need}>{self._num_blocks})"
                )
            ids.append(list(range(cursor, cursor + need)))
            self._next[self._groups.index(group)] = cursor + need
        return tuple(ids)


def _make_so(
    *,
    new: list[Any] | None = None,
    cached: Any = None,
    num_scheduled: dict[str, int] | None = None,
    finished: set[str] | None = None,
    lane: int = 0,
    seq: int = 0,
    fence: tuple[int, int] = (0, 0),
    partitioned: bool = True,
    spec_tokens: dict[str, list[int]] | None = None,
    zero_blocks: list[int] | None = None,
) -> Any:
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        SchedulerOutput,
    )

    num_scheduled = num_scheduled or {}
    so = SchedulerOutput(
        scheduled_new_reqs=new or [],
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs={},
        finished_req_ids=finished or set(),
        free_encoder_mm_hashes=[],
    )
    so.lane_seq = seq
    so.lane_fence = fence
    so.partitioned = partitioned
    so.new_block_ids_to_zero = zero_blocks
    return so


class LaneDriver:
    def __init__(
        self,
        *,
        worker: Any,
        allocator: Allocator,
        prefill_chunk: int,
        timeout: float,
        strict: bool,
        trace: bool,
        out_dir: str | None,
    ) -> None:
        self.worker = worker
        self.decode_runner = worker.model_runner
        self.prefill_runner = worker.prefill_model_runner
        self.runtime = worker.lane_runtime
        self.allocator = allocator
        self.prefill_chunk = prefill_chunk
        self.timeout = timeout
        self.strict = strict
        self.trace = trace
        self.out_dir = out_dir
        self._lock = threading.Lock()
        self.processed = [0, 0]
        self._lane = LANE_DECODE
        self._stream = None
        self.failures: list[str] = []
        self.strict_raises = 0

    # ------------------------------------------------------------- sequential
    def _context(self, lane: int) -> Any:
        assert self.runtime is not None
        context = self.runtime.context(lane)
        driver = self

        class _Ctx:
            def current_stream(self, _context=context) -> Any:
                stream = _context.current_stream()
                if stream is not None:
                    return stream
                return driver._stream

            def on_batch_done(self, scheduler_output: Any) -> None:
                context.on_batch_done(scheduler_output)

        return _Ctx()

    def _submit_direct(
        self,
        lane: int,
        scheduler_output: Any,
        *,
        sample: bool,
        timeout: float | None = None,
    ) -> Any:
        """One sequential lane step on the calling thread (before lane start)."""
        from vllm.config import set_current_vllm_config
        from vllm.v1.lane import lane_scope
        from vllm.v1.worker.workspace import use_workspace_lane

        runner = (
            self.prefill_runner if lane == int(LANE_PREFILL) else self.decode_runner
        )
        assert runner is not None
        scheduler_output.lane_fence = tuple(self.processed)
        scheduler_output.partitioned = False
        with (
            set_current_vllm_config(self.worker.vllm_config),
            lane_scope(lane),
            use_workspace_lane(int(lane)),
        ):
            stored = runner.lane_ctx
            runner.lane_ctx = self._context(lane)
            try:
                executed = runner.execute_model(scheduler_output)
                if not sample:
                    _resolve_output(executed)
                    return None
                sampled = runner.sample_tokens(None)
                if sampled is None:
                    # A failed execute_model leaves no state behind; resolve
                    # it to raise the original exception (S3, R-S3-8).
                    _resolve_output(executed)
                return sampled
            finally:
                runner.lane_ctx = stored
                if scheduler_output.total_num_scheduled_tokens:
                    self.processed[lane] += 1

    # -------------------------------------------------------------- streams
    def _begin_lane(
        self,
        lane: int,
        stream: Any,
    ) -> None:
        self._lane = lane
        self._stream = stream
        for runner in (self.decode_runner, self.prefill_runner):
            if runner is None:
                continue
            stored = runner.lane_ctx
            runner.lane_ctx = self._context(lane) if stored is None else stored

    def _release(self, lane: int, req_id: str) -> None:
        so = _make_so(
            finished={req_id},
            lane=lane,
            seq=self.processed[lane] + 1,
            fence=tuple(self.processed),
            partitioned=False,
        )
        self._submit_direct(lane, so, sample=False, timeout=self.timeout)

    # -------------------------------------------------------------- requests
    def _prompt_blocks(
        self, prompt: list[int], extra_tokens: int = 0
    ) -> tuple[list[int], ...]:
        """Blocks for the prompt plus `extra_tokens` of decode tail.

        The engine allocates a block when a decode crosses a block boundary
        (attention block or mamba block); the driver allocates the whole run
        up front so those writes land in real blocks instead of the null
        block. The prefill row zeroes all of them.
        """
        return self.allocator.allocate(len(prompt) + extra_tokens)

    def _new_request(self, req_id: str, prompt: list[int], blocks: Any) -> Any:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(prompt),
            mm_features=[],
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=64, ignore_eos=True
            ),
            pooling_params=None,
            block_ids=blocks,
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(prompt),
        )

    def _cached_chunk(
        self,
        req_id: str,
        *,
        num_computed: int,
        num_output: int,
        chunk: int,
        all_ids: list[int],
    ) -> Any:
        from vllm.v1.core.sched.output import CachedRequestData

        return CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={req_id: list(all_ids)},
            new_block_ids=[None],
            num_computed_tokens=[num_computed],
            num_output_tokens=[num_output],
        )

    def _cached_chunks(self, req_id: str, chunks: list[tuple[int, int]]) -> Any:
        from vllm.v1.core.sched.output import CachedRequestData

        return CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in chunks],
            all_token_ids={},
            new_block_ids=[None for _ in chunks],
            num_computed_tokens=[c for c, _ in chunks],
            num_output_tokens=[o for _, o in chunks],
        )

    def zero_kv(self) -> None:
        """Zero the KV pages before a comparison run.

        The allocator hands out pages that are not zeroed; the engine's KV
        initialization zeroes them. A comparison must start from zeroed pages
        on both sides or the first tokens read another run's leftovers.
        """
        for cache in self.decode_runner.kv_caches:
            cache.zero_()

    # ------------------------------------------------------------ workloads
    def run_baseline(
        self, req_id: str, prompt: list[int], decode_steps: int
    ) -> list[int]:
        """One runner: chunked prefill, sample t1, then decode."""
        self.zero_kv()
        self.allocator.reset()
        blocks = self._prompt_blocks(prompt, extra_tokens=64)
        new = self._new_request(req_id, prompt, blocks)
        chunks = [
            (offset, prompt[offset : offset + self.prefill_chunk])
            for offset in range(0, len(prompt), self.prefill_chunk)
        ]
        sampled: list[int] = []
        for offset, chunk in chunks:
            if offset == 0:
                so = _make_so(
                    new=[new],
                    num_scheduled={req_id: len(chunk)},
                    zero_blocks=list(blocks[0]),
                    seq=self.processed[0] + 1,
                )
            else:
                so = _make_so(
                    cached=self._cached_chunk(
                        req_id,
                        num_computed=offset,
                        num_output=0,
                        chunk=len(chunk),
                        all_ids=prompt,
                    ),
                    num_scheduled={req_id: len(chunk)},
                    seq=self.processed[0] + 1,
                )
            output = self._submit_direct(0, so, sample=True)
            sampled = _sampled_tokens(output)
        tokens = list(sampled)
        context = prompt + tokens
        for step in range(decode_steps):
            position = len(prompt) + step
            so = _make_so(
                cached=self._cached_chunk(
                    req_id,
                    num_computed=position,
                    num_output=step + 1,
                    chunk=1,
                    all_ids=context,
                ),
                num_scheduled={req_id: 1},
                seq=self.processed[0] + 1,
            )
            output = self._submit_direct(0, so, sample=True)
            tokens.extend(_sampled_tokens(output))
        return tokens

    def run_handoff(self, req_id: str, prompt: list[int], decode_steps: int) -> dict:
        """P prefills, D receives the handoff and decodes."""
        from vllm.v1.core.sched.output import NewRequestData

        self.zero_kv()
        self.allocator.reset()
        blocks = self._prompt_blocks(prompt, extra_tokens=64)
        new = self._new_request(req_id, prompt, blocks)
        sampled: list[int] = []
        offset = 0
        for chunk in range(0, len(prompt), self.prefill_chunk):
            tokens = prompt[chunk : chunk + self.prefill_chunk]
            if chunk == 0:
                so = _make_so(
                    new=[new],
                    num_scheduled={req_id: len(tokens)},
                    zero_blocks=list(blocks[0]),
                    lane=int(LANE_PREFILL),
                    seq=self.processed[int(LANE_PREFILL)] + 1,
                )
            else:
                so = _make_so(
                    cached=self._cached_chunk(
                        req_id,
                        num_computed=offset,
                        num_output=0,
                        chunk=len(tokens),
                        all_ids=prompt,
                    ),
                    num_scheduled={req_id: len(tokens)},
                    lane=int(LANE_PREFILL),
                    seq=self.processed[int(LANE_PREFILL)] + 1,
                )
            output = self._submit_direct(int(LANE_PREFILL), so, sample=True)
            sampled = _sampled_tokens(output)
            offset += len(tokens)

        handoff_new = NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(prompt),
            mm_features=[],
            sampling_params=self._sampling_params(decode_steps),
            pooling_params=None,
            block_ids=blocks,
            num_computed_tokens=len(prompt),
            lora_request=None,
            prefill_token_ids=prompt + list(sampled),
            # The prompt is computed; t1 is the first generated token, so the
            # row is a decode over the prompt (5.1).
            prefill_len=len(prompt),
        )
        p_sampled = list(sampled)
        # No zeroing: these blocks hold the prefill KV the decode reads.
        so = _make_so(
            new=[handoff_new],
            num_scheduled={req_id: 1},
            lane=int(LANE_DECODE),
            seq=self.processed[int(LANE_DECODE)] + 1,
        )
        output = self._submit_direct(0, so, sample=True)
        tokens = list(sampled)
        tokens.extend(_sampled_tokens(output))
        context = prompt + tokens
        for step in range(max(decode_steps - 1, 0)):
            position = len(prompt) + 1 + step
            so = _make_so(
                cached=self._cached_chunk(
                    req_id,
                    num_computed=position,
                    num_output=step + 2,
                    chunk=1,
                    all_ids=context,
                ),
                num_scheduled={req_id: 1},
                lane=int(LANE_DECODE),
                seq=self.processed[int(LANE_DECODE)] + 1,
            )
            output = self._submit_direct(0, so, sample=True)
            tokens.extend(_sampled_tokens(output))
        # The P slot releases on the next P output (5.1).
        self._release(int(LANE_PREFILL), req_id)
        return {
            "prompt_len": len(prompt),
            "t1": list(sampled),
            "p_sampled": p_sampled,
            "tokens": tokens,
        }

    def run_self_handoff(
        self, req_id: str, prompt: list[int], decode_steps: int
    ) -> dict:
        """D prefills, D drops the request, D re-adds the handoff payload.

        Same payload as run_handoff but no P lane: isolates the runner's
        "add a request that already has computed KV/GDN state" path from
        lane sharing.
        """
        from vllm.v1.core.sched.output import NewRequestData

        self.zero_kv()
        self.allocator.reset()
        blocks = self._prompt_blocks(prompt, extra_tokens=64)
        new = self._new_request(req_id, prompt, blocks)
        sampled: list[int] = []
        offset = 0
        for chunk in range(0, len(prompt), self.prefill_chunk):
            tokens = prompt[chunk : chunk + self.prefill_chunk]
            if chunk == 0:
                so = _make_so(
                    new=[new],
                    num_scheduled={req_id: len(tokens)},
                    zero_blocks=list(blocks[0]),
                    seq=self.processed[0] + 1,
                )
            else:
                so = _make_so(
                    cached=self._cached_chunk(
                        req_id,
                        num_computed=offset,
                        num_output=0,
                        chunk=len(tokens),
                        all_ids=prompt,
                    ),
                    num_scheduled={req_id: len(tokens)},
                    seq=self.processed[0] + 1,
                )
            output = self._submit_direct(0, so, sample=True)
            sampled = _sampled_tokens(output)
            offset += len(tokens)

        # Drop the request from D's input batch, then re-add the handoff row.
        self._release(0, req_id)
        handoff_new = NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(prompt),
            mm_features=[],
            sampling_params=self._sampling_params(decode_steps),
            pooling_params=None,
            block_ids=blocks,
            num_computed_tokens=len(prompt),
            lora_request=None,
            prefill_token_ids=prompt + list(sampled),
            # The prompt is computed; t1 is the first generated token, so the
            # row is a decode over the prompt (5.1).
            prefill_len=len(prompt),
        )
        # No zeroing: these blocks hold the prefill KV the decode reads.
        so = _make_so(
            new=[handoff_new],
            num_scheduled={req_id: 1},
            seq=self.processed[0] + 1,
        )
        output = self._submit_direct(0, so, sample=True)
        tokens = list(sampled)
        tokens.extend(_sampled_tokens(output))
        context = prompt + tokens
        for step in range(max(decode_steps - 1, 0)):
            position = len(prompt) + 1 + step
            so = _make_so(
                cached=self._cached_chunk(
                    req_id,
                    num_computed=position,
                    num_output=step + 2,
                    chunk=1,
                    all_ids=context,
                ),
                num_scheduled={req_id: 1},
                seq=self.processed[0] + 1,
            )
            output = self._submit_direct(0, so, sample=True)
            tokens.extend(_sampled_tokens(output))
        return {
            "prompt_len": len(prompt),
            "t1": list(sampled),
            "tokens": tokens,
        }

    def _sampling_params(self, decode_steps: int) -> Any:
        from vllm.sampling_params import SamplingParams

        return SamplingParams(
            temperature=0.0, max_tokens=decode_steps + 2, ignore_eos=True
        )

    def release_d(self, req_id: str) -> None:
        self._release(int(LANE_DECODE), req_id)

    # ---------------------------------------------------------------- E-GIL
    def run_concurrent(
        self,
        *,
        prompt: list[int],
        decode_steps: int,
        p_prompt: list[int],
        solo_steps: int,
    ) -> dict:
        """Lane runtime: D decode latency alone and with P chunks running."""
        from vllm.v1.core.sched.output import NewRequestData

        runtime = self.runtime
        assert runtime is not None
        # Fences name processed batches of THIS lane runtime: the sequential
        # runs above were direct calls and published no fence events.
        self.processed = [0, 0]
        runtime.start()
        decode_stream = runtime.streams.stream_for(0, True)
        p_stream = runtime.streams.stream_for(1, True)
        self._begin_lane(0, decode_stream)
        self._begin_lane(1, p_stream)

        # D's request first: P prefills its prompt, then the handoff.
        blocks = self._prompt_blocks(prompt, extra_tokens=64)
        new = self._new_request(f"d-{os.getpid()}", prompt, blocks)
        for offset in range(0, len(prompt), self.prefill_chunk):
            chunk = prompt[offset : offset + self.prefill_chunk]
            so = (
                _make_so(
                    new=[new],
                    num_scheduled={new.req_id: len(chunk)},
                    lane=int(LANE_PREFILL),
                    seq=self._next_seq(int(LANE_PREFILL)),
                )
                if offset == 0
                else _make_so(
                    cached=self._cached_chunk(
                        new.req_id,
                        num_computed=offset,
                        num_output=0,
                        chunk=len(chunk),
                        all_ids=prompt,
                    ),
                    num_scheduled={new.req_id: len(chunk)},
                    lane=int(LANE_PREFILL),
                    seq=self._next_seq(int(LANE_PREFILL)),
                )
            )
            sampled = _sampled_tokens(self._lane_submit(int(LANE_PREFILL), so))
        d_req_id = f"d-{os.getpid()}"
        handoff = NewRequestData(
            req_id=d_req_id,
            prompt_token_ids=list(prompt),
            mm_features=[],
            sampling_params=self._sampling_params(decode_steps),
            pooling_params=None,
            block_ids=blocks,
            num_computed_tokens=len(prompt),
            lora_request=None,
            prefill_token_ids=prompt + list(sampled),
            # The prompt is computed; t1 is the first generated token, so the
            # row is a decode over the prompt (5.1).
            prefill_len=len(prompt),
        )
        so, sampled = self._lane_submit_with_output(
            int(LANE_DECODE),
            _make_so(
                new=[handoff],
                num_scheduled={d_req_id: 1},
                lane=int(LANE_DECODE),
                seq=self._next_seq(int(LANE_DECODE)),
            ),
        )
        tokens = list(sampled)
        del so

        def decode_steps_fn(count: int) -> list[float]:
            latencies: list[float] = []
            for step in range(count):
                position = len(prompt) + 1 + step
                so = _make_so(
                    cached=self._cached_chunk(
                        d_req_id,
                        num_computed=position,
                        num_output=step + 2,
                        chunk=1,
                        all_ids=prompt + tokens,
                    ),
                    num_scheduled={d_req_id: 1},
                    lane=int(LANE_DECODE),
                    seq=self._next_seq(int(LANE_DECODE)),
                )
                start = time.monotonic()
                _out, sampled = self._lane_submit_with_output(int(LANE_DECODE), so)
                latencies.append(time.monotonic() - start)
                tokens.extend(sampled)
            return latencies

        warmup = decode_steps_fn(4)
        solo = decode_steps_fn(int(solo_steps))
        solo_p50 = percentiles(solo, (0.5,)).get("p50_ms", 0.0)

        # A fresh P request (no handoff): its chunks are the P workload.
        p_chunks = [
            p_prompt[o : o + self.prefill_chunk]
            for o in range(0, len(p_prompt), self.prefill_chunk)
        ]

        def submit_p_chunks(
            tag: str, ready: threading.Event | None = None
        ) -> tuple[str, list[float]]:
            p_new = self._new_request(
                f"{tag}-{os.getpid()}",
                p_prompt,
                self._prompt_blocks(p_prompt),
            )
            chunk_latencies: list[float] = []
            for index, chunk in enumerate(p_chunks):
                so = (
                    _make_so(
                        new=[p_new],
                        num_scheduled={p_new.req_id: len(chunk)},
                        lane=int(LANE_PREFILL),
                        seq=self._next_seq(int(LANE_PREFILL)),
                    )
                    if index == 0
                    else _make_so(
                        cached=self._cached_chunk(
                            p_new.req_id,
                            num_computed=index * self.prefill_chunk,
                            num_output=0,
                            chunk=len(chunk),
                            all_ids=p_prompt,
                        ),
                        num_scheduled={p_new.req_id: len(chunk)},
                        lane=int(LANE_PREFILL),
                        seq=self._next_seq(int(LANE_PREFILL)),
                    )
                )
                start = time.monotonic()
                self._lane_submit(int(LANE_PREFILL), so)
                chunk_latencies.append(time.monotonic() - start)
                if ready is not None and (index == 1 or index + 1 == len(p_chunks)):
                    ready.set()
            self._release(int(LANE_PREFILL), p_new.req_id)
            return p_new.req_id, chunk_latencies

        # P alone, no D lane work.
        _, solo_p = submit_p_chunks("p-solo")
        solo_p_rate = _chunks_per_second(solo_p)

        # Concurrent: P chunks while D decodes.
        results: dict[str, Any] = {}
        p_ready = threading.Event()

        def p_thread() -> None:
            try:
                results["p"] = submit_p_chunks("p-concurrent", p_ready)
            except Exception as err:  # noqa: BLE001 - reported in the JSON
                results["error"] = repr(err)

        thread = threading.Thread(target=p_thread, name="lane-driver-p")
        thread.start()
        # Measure the steady overlap, not the P-activation transient (D5).
        p_ready.wait(timeout=self.timeout)
        concurrent = decode_steps_fn(int(solo_steps))

        thread.join()
        _, concurrent_p = results.get("p", ("", []))
        concurrent_p_rate = _chunks_per_second(concurrent_p)
        d_p99 = percentiles(concurrent, (0.99,)).get("p99_ms", 0.0)
        ratio = d_p99 / solo_p50 if solo_p50 else float("inf")
        reverse_latency = None
        if self.trace:
            reverse_latency = self._reverse_latency()
        self.release_d(d_req_id)
        runtime.shutdown()
        return {
            "d_context": len(prompt),
            "d_solo_p50_ms": solo_p50,
            "d_solo": summarize(solo),
            "d_concurrent": summarize(concurrent),
            "d_p99_ratio": ratio,
            "p_chunks_solo": summarize(solo_p),
            "p_chunks_concurrent": summarize(concurrent_p),
            "p_chunk_rate_share": (
                concurrent_p_rate / solo_p_rate if solo_p_rate else 0.0
            ),
            "p_chunk_solo_per_s": solo_p_rate,
            "p_chunk_concurrent_per_s": concurrent_p_rate,
            "reverse_ms": reverse_latency,
            "warmup_steps": len(warmup),
            "tokens": tokens,
        }

    def _next_seq(self, lane: int) -> int:
        with self._lock:
            return self.processed[lane] + 1

    def _lane_submit(self, lane: int, scheduler_output: Any) -> Any:
        """Submit one lane batch; sample it so the fence event is published.

        The lane runtime records the end-of-batch fence event in
        ``sample_tokens`` (R-S5-5), so a non-empty batch that is only executed
        never publishes the event the other lane may fence on.
        """
        if scheduler_output.total_num_scheduled_tokens:
            executed, _ = self._lane_submit_with_output(lane, scheduler_output)
            return executed
        runtime = self.runtime
        assert runtime is not None
        scheduler_output.lane_fence = tuple(self.processed)
        scheduler_output.partitioned = True
        future = runtime.submit(int(lane), "execute_model", (scheduler_output,))
        return future.result(timeout=self.timeout)

    def _lane_submit_with_output(
        self, lane: int, scheduler_output: Any
    ) -> tuple[Any, list[int]]:
        runtime = self.runtime
        assert runtime is not None
        scheduler_output.lane_fence = tuple(self.processed)
        scheduler_output.partitioned = True
        future = runtime.submit(int(lane), "execute_model", (scheduler_output,))
        executed = future.result(timeout=self.timeout)
        if not scheduler_output.total_num_scheduled_tokens:
            return executed, []
        sample_future = runtime.submit(int(lane), "sample_tokens", (None,))
        sampled = _sampled_tokens(sample_future.result(timeout=self.timeout))
        # The fence event is published by sample_tokens; only now is this
        # batch "processed" for the other lane's fence.
        with self._lock:
            self.processed[int(lane)] += 1
        return executed, sampled

    def _reverse_latency(self) -> float:
        """Time until a D step scheduled behind a running 16k P prefill runs."""
        runtime = self.runtime
        assert runtime is not None
        self.allocator.reset()
        total = self.prefill_chunk * 16
        prompt = _seeded_prompt(total, 11)
        blocks = self._prompt_blocks(prompt, extra_tokens=64)
        req_id = f"r-{os.getpid()}"
        new = self._new_request(req_id, prompt, blocks)
        started = threading.Event()

        def p_load() -> None:
            for index in range(16):
                so = (
                    _make_so(
                        new=[new],
                        num_scheduled={req_id: self.prefill_chunk},
                        lane=int(LANE_PREFILL),
                        seq=self._next_seq(int(LANE_PREFILL)),
                    )
                    if index == 0
                    else _make_so(
                        cached=self._cached_chunk(
                            req_id,
                            num_computed=index * self.prefill_chunk,
                            num_output=0,
                            chunk=self.prefill_chunk,
                            all_ids=prompt,
                        ),
                        num_scheduled={req_id: self.prefill_chunk},
                        lane=int(LANE_PREFILL),
                        seq=self._next_seq(int(LANE_PREFILL)),
                    )
                )
                self._lane_submit(int(LANE_PREFILL), so)
                started.set()

        thread = threading.Thread(target=p_load, name="lane-driver-load")
        thread.start()
        started.wait(timeout=self.timeout)
        d_blocks = self._prompt_blocks([1], extra_tokens=64)
        d_req = f"x-{os.getpid()}"
        d_new = self._new_request(d_req, [1], d_blocks)
        start = time.monotonic()
        self._lane_submit(
            int(LANE_DECODE),
            _make_so(
                new=[d_new],
                num_scheduled={d_req: 1},
                lane=int(LANE_DECODE),
                seq=self._next_seq(int(LANE_DECODE)),
            ),
        )
        latency = time.monotonic() - start
        thread.join()
        self._release(int(LANE_PREFILL), req_id)
        self._release(int(LANE_DECODE), d_req)
        return latency

    def run_strict_probe(self) -> bool:
        """Whether a short lane workload completes without a strict raise."""
        return not any("strict" in message for message in self.failures)


def _resolve_output(output: Any) -> Any:
    """Resolve an ``AsyncModelRunnerOutput`` (raises its exception)."""
    if hasattr(output, "get_output"):
        return output.get_output()
    return output


def _chunks_per_second(latencies: list[float]) -> float:
    total = sum(latencies)
    return len(latencies) / total if total else 0.0


def _sampled_tokens(output: Any) -> list[int]:
    if output is None:
        return []
    if hasattr(output, "get_output"):
        output = output.get_output()
    if os.environ.get("LANE_DEBUG_OUTPUT"):
        logprobs = getattr(output, "logprobs", None)
        print(
            "SAMPLER_OUTPUT",
            type(output).__name__,
            "logprobs=",
            None if logprobs is None else str(logprobs)[:400],
            flush=True,
        )
    sampled = getattr(output, "sampled_token_ids", None)
    if sampled is None:
        sampled = getattr(output, "sampled_token_ids_cpu", None)
    if sampled is None:
        return []
    values = sampled.tolist() if hasattr(sampled, "tolist") else sampled
    return [int(value) for value in values[0]]


def _build_config(args: argparse.Namespace) -> Any:
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        gpu_memory_utilization=0.35,
        kv_cache_memory_bytes=int(args.kv_gib * (1 << 30)),
        enforce_eager=not args.compile,
        trust_remote_code=True,
        seed=args.seed,
        disable_log_stats=True,
        language_model_only=True,
        skip_mm_profiling=True,
    )
    cfg = engine_args.create_engine_config()
    if args.no_spec:
        cfg.speculative_config = None
    cfg.cache_config.kv_cache_layout = "LBHNC"
    cfg.scheduler_config.two_lane = True
    cfg.scheduler_config.two_lane_decode_wgps = args.lane_decode_wgps
    cfg.scheduler_config.two_lane_trace = args.trace
    cfg.scheduler_config.two_lane_strict = args.strict
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cfg.attention_config.backend = AttentionBackendEnum[args.attention_backend]
    return cfg


def boot(args: argparse.Namespace) -> tuple[Any, Allocator]:
    from threading import Lock

    from vllm.v1.worker.worker_base import WorkerWrapperBase

    cfg = _build_config(args)
    wrapper = WorkerWrapperBase(rpc_rank=0)
    kwargs = {
        "vllm_config": cfg,
        "local_rank": cfg.parallel_config.rank,
        "rank": cfg.parallel_config.rank,
        "distributed_init_method": "env://",
        "is_driver_worker": True,
        "shared_worker_lock": Lock(),
    }
    wrapper.init_worker(all_kwargs=[kwargs])
    worker = wrapper.worker
    from vllm.config import set_current_vllm_config
    from vllm.v1.core.kv_cache_planning import (
        get_global_kv_cache_groups,
        get_kv_cache_configs,
    )

    with set_current_vllm_config(cfg):
        worker.init_device()
        worker.load_model()
        available = worker.determine_available_memory()
        specs = [worker.get_kv_cache_spec()]
        kv_config = get_kv_cache_configs(
            cfg,
            specs,
            [available],
            get_global_kv_cache_groups(cfg, specs),
        )[0]
        kv_config.kv_cache_layout = cfg.cache_config.kv_cache_layout
        worker.initialize_from_config(kv_config)
        diag = []
        for name, layer in worker.compilation_config.static_forward_context.items():
            cache = getattr(layer, "kv_cache", None)
            if cache is not None and hasattr(cache, "shape"):
                diag.append(
                    {
                        "name": name,
                        "shape": list(cache.shape),
                        "stride": list(cache.stride()),
                        "contiguous": cache.is_contiguous(),
                    }
                )
        print("KV_DIAG " + json.dumps(diag[:4]))
        worker.compile_or_warm_up_model()
    return worker, Allocator(kv_config)


def _kfd_queues() -> int:
    path = f"/sys/class/kfd/kfd/proc/{os.getpid()}/queues"
    try:
        return len(os.listdir(path))
    except OSError:
        return -1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preimport_env(args)
    if args.out_dir:
        out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        os.makedirs(out_dir, exist_ok=True)
        os.environ.setdefault("VLLM_CACHE_ROOT", out_dir)
    else:
        out_dir = None
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    report: dict[str, Any] = {
        "argv": sys.argv,
        "model": args.model,
        "gpu": args.gpu,
        "attention_backend": args.attention_backend,
        "lengths": lengths,
        "decode_steps": args.decode_steps,
        "strict": args.strict,
        "trace": args.trace,
        "compile": args.compile,
    }
    started = time.monotonic()
    worker = None
    try:
        worker, allocator = boot(args)
        report["boot_s"] = time.monotonic() - started
        report["kfd_queues_after_boot"] = _kfd_queues()
        driver = LaneDriver(
            worker=worker,
            allocator=allocator,
            prefill_chunk=args.prefill_chunk,
            timeout=args.timeout,
            strict=args.strict,
            trace=args.trace,
            out_dir=out_dir,
        )
        report["lane_streams"] = (
            list(worker.lane_runtime.streams.masks().items())
            if worker.lane_runtime is not None
            else []
        )
        report["kfd_queues_after_streams"] = _kfd_queues()
        if not args.skip_handoff:
            # The first request on each runner reads startup state (profile and
            # warmup wrote the dummy state slots). Run one throwaway request on
            # both paths so the comparison starts from steady state.
            if not args.skip_warmup:
                warm_prompt = _seeded_prompt(8, args.seed + 7)
                driver.run_baseline(f"warm-b-{os.getpid()}", warm_prompt, 1)
                driver.release_d(f"warm-b-{os.getpid()}")
                driver.run_handoff(f"warm-h-{os.getpid()}", warm_prompt, 1)
                driver.release_d(f"warm-h-{os.getpid()}")
            equivalence: list[dict] = []
            for length in lengths:
                prompt = _seeded_prompt(length, args.seed)
                decode_steps = args.decode_steps
                baseline = driver.run_baseline(f"b-{length}", prompt, decode_steps)
                driver.release_d(f"b-{length}")
                handoff = driver.run_handoff(f"h-{length}", prompt, decode_steps)
                driver.release_d(f"h-{length}")
                self_handoff = driver.run_self_handoff(
                    f"s-{length}", prompt, decode_steps
                )
                driver.release_d(f"s-{length}")
                equivalence.append(
                    {
                        "L": length,
                        "equal": handoff["tokens"] == baseline,
                        "baseline_tokens": baseline,
                        "handoff_tokens": handoff["tokens"],
                        "self_equal": self_handoff["tokens"] == baseline,
                        "self_tokens": self_handoff["tokens"],
                        "p_sampled": handoff.get("p_sampled"),
                        "t1": handoff["t1"],
                    }
                )
            report["handoff"] = equivalence
            if args.compile:
                # Regression case: with the D lane compiled (FULL replay), a
                # handoff decode must sample exactly what the eager baseline
                # samples. A mismatch is a failure, not a report line.
                mismatched = [e["L"] for e in equivalence if not e["equal"]]
                if mismatched:
                    report["handoff_regression_failed"] = mismatched
        if not args.skip_gil:
            prompt = _seeded_prompt(args.decode_context, args.seed)
            report["egil"] = driver.run_concurrent(
                prompt=prompt,
                decode_steps=args.decode_steps,
                p_prompt=_seeded_prompt(args.gil_context, args.seed + 1),
                solo_steps=args.solo_steps,
            )
        if args.soak_s > 0:
            soak = {"iterations": 0, "mismatches": 0, "errors": []}
            deadline = time.monotonic() + args.soak_s
            while time.monotonic() < deadline:
                index = soak["iterations"]
                length = lengths[index % len(lengths)]
                prompt = _seeded_prompt(length, args.seed + index)
                try:
                    baseline = driver.run_baseline(
                        f"sb-{index}", prompt, args.decode_steps
                    )
                    driver.release_d(f"sb-{index}")
                    handoff = driver.run_handoff(
                        f"sh-{index}", prompt, args.decode_steps
                    )
                    driver.release_d(f"sh-{index}")
                    if handoff["tokens"] != baseline:
                        soak["mismatches"] += 1
                    soak["iterations"] += 1
                except Exception as err:  # noqa: BLE001 - soak reports it
                    soak["errors"].append(repr(err))
                    break
            soak["d_reqs"] = int(driver.decode_runner.req_states.num_reqs)
            soak["p_reqs"] = int(driver.prefill_runner.req_states.num_reqs)
            report["soak"] = soak
        report["kfd_queues_before_shutdown"] = _kfd_queues()
        worker.shutdown()
        report["shutdown"] = True
    except Exception as err:  # noqa: BLE001 - the driver reports every failure
        report["error"] = repr(err)
        report["traceback"] = traceback.format_exc()
        if worker is not None:
            try:
                worker.shutdown()
            except Exception:  # noqa: BLE001 - already failing
                report["shutdown_error"] = traceback.format_exc()
    report["elapsed_s"] = time.monotonic() - started
    text = json.dumps(report, indent=2, sort_keys=True)
    if out_dir:
        path = os.path.join(out_dir, "lane_driver.json")
        with open(path, "w", encoding="utf-8") as file:
            file.write(text + "\n")
        print(f"M1 report: {path}")
    print(text)
    failed = "error" in report or bool(report.get("handoff_regression_failed"))
    return 1 if failed else 0


def _seeded_prompt(length: int, seed: int) -> list[int]:
    values = []
    state = (seed + 1) * 2654435761
    for index in range(length):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        values.append(1000 + state % 50000 + index % 13)
    return values


if __name__ == "__main__":
    sys.exit(main())
