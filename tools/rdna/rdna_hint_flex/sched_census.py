#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only census of the per-step token count M the production Scheduler produces.

Every linear of the 27B HI2 model (recipe 27b-hi2-tp2: TP=2, block size 1024,
mamba cache mode align, max_num_batched_tokens 2048, max_num_seqs 8, prefix
caching, chunked prefill, optional MTP) runs a GEMM whose N dimension is M, the
total number of tokens the scheduler puts into the step. TunableOp rows are
exact-key, so the set of M values (and the FLOPs they carry) decides which rows
need tuning. This script runs the real ``Scheduler`` (hybrid KV cache groups,
block-aligned prefill split, prefix cache, spec-decode drafts) with a faked
model runner: no model runs and no GPU is touched (``torch.cuda`` is never
called; run it with HIP_VISIBLE_DEVICES set empty).

Workloads (each for MTP off and MTP on with fixed draft acceptance):
  cells           the five scoreboard cells of ``vllm bench serve --dataset-name
                  random``; prompts come from the real ``RandomDataset`` (seed
                  12345) and the checkpoint tokenizer. Per cell: the endpoint
                  ready check's solo test run of request 0, then the timed run
                  (closed loop at the cell's concurrency, all arrive at t=0).
  probe           ``tools/rdna/port_v031/concurrency_probe.py`` phases in order
                  on one server (solo, solo2, together, churned, long solo, long
                  together), real prompts, prefix cache kept across phases.
  warm_repeat     one prompt of L tokens sent twice; second pass hits the cache.
  mixed           64 requests, log-uniform prompts, shared prefixes, c=8.
  prompt_logprobs a request with prompt_logprobs (alone, and next to decodes).

Per step it records M, prefill vs decode+draft tokens, request count, lm_head
rows (decode: 1 + drafts; request finishing its prompt: 1; other prefill: 0),
and for steps with a prefill the (num_computed_before, num_new) pair of every
request. Aggregates: histogram of M > 48 and of M in 9..48, histogram of rows
in 9..48, and the FLOP-weighted share of each M (FLOPs ~ M).

A step is [M, prefill tokens, decode+draft tokens, requests, lm_head rows,
decode requests, prefill requests]. With prompt_logprobs the V2 runner also runs
lm_head over the whole step in 1024-row chunks (hist_plp_lm_head_chunks).

Modelled, not measured: the sync Scheduler (no async-scheduling overlap), a fixed
number of accepted drafts per spec step (--accepted; steady state with 2 = three
tokens per request per step), all requests alive at once (pool sized so nothing
is preempted), and the churned probe phase's 0.15 s arrival spacing as one new
request every --churn-interval steps.

Run from the source tree (it imports ``tests.v1.core.utils``):

  cd $TREE && env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \\
      HIP_VISIBLE_DEVICES= $VENV/bin/python tools/rdna/rdna_hint_flex/sched_census.py \\
      --out-dir ~/w4a8_runs/hi2-tunable/sched

Writes census.json and census.txt into --out-dir.
"""

import argparse
import collections
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler
from vllm.benchmarks.datasets import RandomDataset
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.platforms.interface import DeviceCapability
from vllm.platforms.rocm import RocmPlatform
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request

SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--BlivionIaG--Qwen3.8-27B-HI2-3.0bpw"
    "/snapshots/db2a3aa3cd5efb0f92403cd4864e1f0d99378149"
)
PROBE_PATH = Path(__file__).resolve().parents[1] / "port_v031" / "concurrency_probe.py"

# (input_len, output_len, max_concurrency, num_prompts) of the scoreboard cells.
CELLS = [
    (1024, 512, 1, 16),
    (1024, 512, 8, 16),
    (16384, 1024, 1, 8),
    (16384, 1024, 8, 16),
    (32768, 2048, 1, 4),
]
WARM_LENGTHS = (1500, 3100, 4200, 16000)
GEMV_MAX_M = 48  # hi2 layers run a GEMV up to M=48, dequant + rocBLAS above
LM_HEAD_CHUNK = 1024  # PromptLogprobsWorker CHUNK_SIZE (V2 model runner)
SAMPLED_TOKEN = 100  # never EOS (the cells run with --ignore-eos)
WORKLOADS = ("cells", "probe", "warm_repeat", "mixed", "prompt_logprobs")
M_CLASSES = ("2048", "1024", "mult128", "mult64", "arbitrary")


@dataclass
class Job:
    ids: list[int]
    max_tokens: int
    arrive: int = 0  # first scheduler step at which the request may be added
    prompt_logprobs: int | None = None


class Sim:
    """One production Scheduler (fixed speculative setup) with a faked runner."""

    def __init__(self, args: argparse.Namespace, num_spec: int, accepted: int):
        self.num_spec = num_spec
        self.accepted = accepted
        self.sched = build_scheduler(args, num_spec)
        self.hasher = get_request_block_hasher(self.sched.hash_block_size, sha256)
        self.counter = 0

    def reset(self) -> None:
        assert self.sched.reset_prefix_cache(), "prefix cache reset refused"

    def request(self, job: Job) -> Request:
        params = SamplingParams(
            ignore_eos=True,
            max_tokens=job.max_tokens,
            prompt_logprobs=job.prompt_logprobs,
        )
        params.update_from_generation_config({}, EOS_TOKEN_ID)
        self.counter += 1
        return Request(
            request_id=f"r{self.counter}",
            prompt_token_ids=job.ids,
            sampling_params=params,
            pooling_params=None,
            block_hasher=self.hasher,
        )

    def run(self, jobs: list[Job], concurrency: int | None, label: str) -> dict:
        """Drive the scheduler until every job finished (closed loop)."""
        sched = self.sched
        block = sched.mamba_state_block_size
        pending = collections.deque(sorted(jobs, key=lambda j: j.arrive))
        active: dict[str, Request] = {}
        reqs: dict[str, Request] = {}
        info: dict[str, dict] = {}
        steps: list[list[int]] = []
        prefill_steps: list[dict] = []
        plp_chunks: collections.Counter = collections.Counter()
        step = 0
        while pending or active:
            while (
                pending
                and pending[0].arrive <= step
                and (concurrency is None or len(active) < concurrency)
            ):
                req = self.request(pending.popleft())
                sched.add_request(req)
                active[req.request_id] = reqs[req.request_id] = req
                info[req.request_id] = {
                    "prompt_len": req.num_prompt_tokens,
                    "hit": None,
                    "chunks": [],
                    "first_step": step,
                }
            if not active:
                step = pending[0].arrive
                continue
            out = sched.schedule()
            ids = list(out.num_scheduled_tokens)
            if not ids:
                raise RuntimeError(f"{label}: no progress at step {step}")
            m = pf_tokens = rows = 0
            pf_pairs: list[list[int]] = []
            dec_pairs: list[list[int]] = []
            plp_step = False
            samples = []
            for rid in ids:
                req = reqs[rid]
                n = out.num_scheduled_tokens[rid]
                # schedule() already advanced the computed counter.
                start = req.num_computed_tokens - n
                if info[rid]["hit"] is None:
                    info[rid]["hit"] = start
                pf = max(0, min(n, req.num_prompt_tokens - start))
                m += n
                pf_tokens += pf
                if pf:
                    pf_pairs.append([start, n])
                    info[rid]["chunks"].append(pf)
                    # Aligned split: a prompt chunk ends on a state block boundary,
                    # or at the identical-resend replay stop (prompt - 1) or the
                    # prompt end.
                    assert (
                        start + pf
                    ) % block == 0 or start + pf >= req.num_prompt_tokens - 1, (
                        label,
                        rid,
                        start,
                        pf,
                    )
                    plp_step |= req.sampling_params.prompt_logprobs is not None
                else:
                    dec_pairs.append([start, n])
                if req.num_computed_tokens >= req.num_tokens:
                    spec = out.scheduled_spec_decode_tokens.get(rid, ())
                    rows += 1 + len(spec)
                    # A one-token request (full prefix hit) is padded to 1 + k rows
                    # with -1 placeholders that can never be accepted.
                    accepted = self.accepted if all(t >= 0 for t in spec) else 0
                    samples.append([SAMPLED_TOKEN] * min(accepted + 1, 1 + len(spec)))
                else:
                    samples.append([])
            if plp_step:
                # V2 runner: lm_head over the whole step's hidden states, in chunks.
                for a in range(0, m, LM_HEAD_CHUNK):
                    plp_chunks[min(LM_HEAD_CHUNK, m - a)] += 1
            steps.append(
                [
                    m,
                    pf_tokens,
                    m - pf_tokens,
                    len(ids),
                    rows,
                    len(dec_pairs),
                    len(pf_pairs),
                ]
            )
            if pf_pairs:
                prefill_steps.append(
                    {"step": step, "m": m, "prefill": pf_pairs, "decode": dec_pairs}
                )
            sched.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=ids,
                    req_id_to_index={rid: i for i, rid in enumerate(ids)},
                    sampled_token_ids=samples,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
            # The drafter runs after the forward of every scheduled request.
            drafted = [rid for rid in ids if rid in sched.requests]
            if self.num_spec and drafted:
                drafts = list(range(101, 101 + self.num_spec))
                sched.update_draft_token_ids(
                    DraftTokenIds(drafted, [drafts] * len(drafted))
                )
            for rid in [r for r in active if r not in sched.requests]:
                assert reqs[rid].num_output_tokens == reqs[rid].max_tokens, rid
                i = info[rid]
                assert i["hit"] + sum(i["chunks"]) == i["prompt_len"], (label, rid, i)
                del active[rid]
            step += 1
            if step > 400_000:
                raise RuntimeError(f"{label}: did not drain")
        return {
            "label": label,
            "steps": steps,
            "prefill_steps": prefill_steps,
            "plp_lm_head_chunks": dict(plp_chunks),
            "requests": list(info.values()),
            "preemptions": sum(r.num_preemptions for r in reqs.values()),
        }


def build_scheduler(args: argparse.Namespace, num_spec: int) -> Scheduler:
    block = args.block_size

    class CensusScheduler(Scheduler):
        # 27B per rank at TP2: one full-attention group (4 KV heads / 2, head
        # 256) and three GDN groups (16 layers each), all on 1024-token blocks.
        def __init__(self, vllm_config, kv_cache_config, **kwargs):
            if vllm_config.model_config.multimodal_config is not None:
                vllm_config.model_config.multimodal_config.language_model_only = True
            vllm_config.cache_config.mamba_cache_mode = "align"
            vllm_config.cache_config.mamba_block_size = block
            attn = FullAttentionSpec(
                block_size=block, num_kv_heads=2, head_size=256, dtype=torch.float16
            )
            state = MambaSpec(
                block_size=block,
                shapes=((3, 5120), (24, 128, 128)),
                dtypes=(torch.float16, torch.float32),
                page_size_padded=attn.page_size_bytes,
                mamba_cache_mode="align",
                num_speculative_blocks=num_spec,
            )
            kv_cache_config.kv_cache_groups = [
                KVCacheGroupSpec(["attn"], attn),
                *[KVCacheGroupSpec([f"gdn{i}"], state) for i in range(3)],
            ]
            super().__init__(
                vllm_config=vllm_config, kv_cache_config=kv_cache_config, **kwargs
            )

    spec_config = None
    if num_spec:
        spec_config = SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=num_spec,
            target_model_config=ModelConfig(
                model=str(args.model),
                trust_remote_code=True,
                dtype="float16",
                seed=42,
                skip_tokenizer_init=True,
                max_model_len=args.max_model_len,
            ),
            target_parallel_config=ParallelConfig(),
        )
    sched = create_scheduler(
        model=str(args.model),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_blocks=args.pool_blocks,
        block_size=block,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        scheduler_cls=CensusScheduler,
        speculative_config=spec_config,
    )
    # The platform gate (on_rdna_family) is off on this CPU run, so pin what the
    # recipe has: aligned prefill split on the recurrent-state block grid.
    sched.need_mamba_block_aligned_split = True
    sched.mamba_state_block_size = block
    assert sched.mamba_prefill_checkpoint_alignment is None
    assert sched.num_spec_tokens == num_spec
    assert sched._mixed_step is None and sched.hash_block_size == block
    return sched


def m_class(m: int) -> str:
    if m == 2048 or m == 1024:
        return str(m)
    if m % 128 == 0:
        return "mult128"
    return "mult64" if m % 64 == 0 else "arbitrary"


def aggregate(segments: list[dict]) -> dict:
    steps = [s for seg in segments for s in seg["steps"]]
    hist_gt = collections.Counter(s[0] for s in steps if s[0] > GEMV_MAX_M)
    hist_mid = collections.Counter(s[0] for s in steps if 9 <= s[0] <= GEMV_MAX_M)
    hist_rows = collections.Counter(s[4] for s in steps if 9 <= s[4] <= GEMV_MAX_M)
    plp = collections.Counter()
    for seg in segments:
        plp.update({int(k): v for k, v in seg["plp_lm_head_chunks"].items()})
    flops = sum(m * c for m, c in hist_gt.items())
    classes = dict.fromkeys(M_CLASSES, 0.0)
    for m, c in hist_gt.items():
        classes[m_class(m)] += m * c / flops
    return {
        "steps": len(steps),
        "prefill_steps": sum(1 for s in steps if s[1]),
        "tokens": sum(s[0] for s in steps),
        "prefill_tokens": sum(s[1] for s in steps),
        "max_m": max(s[0] for s in steps),
        "preemptions": sum(seg["preemptions"] for seg in segments),
        "hist_m_gt48": dict(sorted(hist_gt.items())),
        "hist_m_9_48": dict(sorted(hist_mid.items())),
        "hist_rows_9_48": dict(sorted(hist_rows.items())),
        "flop_share_m_gt48": {m: m * c / flops for m, c in sorted(hist_gt.items())}
        if flops
        else {},
        "flop_class_share_m_gt48": classes if flops else {},
        "hist_plp_lm_head_chunks": dict(sorted(plp.items())),
    }


def length_stats(lens: list[int]) -> dict:
    return {
        "n": len(lens),
        "min": min(lens),
        "max": max(lens),
        "mean": sum(lens) / len(lens),
        "median": float(np.median(lens)),
        "lens": lens,
    }


def make_cells(tokenizer, seed: int) -> list[dict]:
    """The prompts ``vllm bench serve`` sends, through the real RandomDataset."""
    cells = []
    for inp, out, conc, num in CELLS:
        samples = RandomDataset(
            random_seed=seed, dataset_path=None, disable_shuffle=False
        ).sample(
            tokenizer=tokenizer,
            num_requests=num,
            prefix_len=0,
            input_len=inp,
            output_len=out,
            range_ratio=0.0,
            request_id_prefix="",
            batchsize=1,
            no_oversample=False,
        )
        # The server tokenizes the prompt text again (completions endpoint).
        ids = [tokenizer.encode(s.prompt, add_special_tokens=True) for s in samples]
        cells.append(
            {
                "name": f"{inp}:{out}:{conc}:{num}",
                "input_len": inp,
                "output_len": out,
                "concurrency": conc,
                "ids": ids,
                "output_lens": [s.expected_output_len for s in samples],
                "dataset_prompt_lens": [s.prompt_len for s in samples],
            }
        )
    return cells


def run_cells(sim: Sim, cells: list[dict]) -> dict:
    result = {}
    for cell in cells:
        sim.reset()
        outs = cell["output_lens"]
        test = sim.run([Job(cell["ids"][0], outs[0])], 1, "test_run")
        main = sim.run(
            [Job(i, o) for i, o in zip(cell["ids"], outs)],
            cell["concurrency"],
            "run",
        )
        result[cell["name"]] = {
            "segments": [test, main],
            "agg": aggregate([test, main]),
        }
        result[cell["name"]]["agg_run_only"] = aggregate([main])
    return result


def load_probe():
    spec = importlib.util.spec_from_file_location("concurrency_probe", PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_prompts(tokenizer, args: argparse.Namespace) -> tuple[list, list]:
    """Token ids of the probe's short PROMPTS and of its long_prompt() prompts."""
    probe = load_probe()
    prompts = probe.PROMPTS[: args.probe_num_prompts]
    longs = [
        probe.long_prompt(k, args.probe_long_sentences)
        for k in range(args.probe_long_prompts)
    ]
    return (
        [tokenizer.encode(t, add_special_tokens=True) for t in prompts],
        [tokenizer.encode(t, add_special_tokens=True) for t in longs],
    )


def run_probe(sim: Sim, args: argparse.Namespace, short: list, longs: list) -> dict:
    """Phases of concurrency_probe.main(), in order, on one server."""
    mt = args.probe_max_tokens
    n = len(short)
    # concurrency_probe: lengths of the churned phase, start 0.15 s * k apart,
    # approximated as one new request every --churn-interval scheduler steps.
    lengths = [16, mt, 24, mt, 32, 40, 48, mt]
    churn = [
        Job(
            short[(k * 3) % n],
            min(lengths[k % len(lengths)], mt),
            arrive=k * args.churn_interval,
        )
        for k in range(2 * n)
    ]
    sim.reset()
    phases = {
        "solo": sim.run([Job(p, mt) for p in short], 1, "solo"),
        "solo2": sim.run([Job(p, mt) for p in short], 1, "solo2"),
        "together": sim.run([Job(p, mt) for p in short], None, "together"),
        "churned": sim.run(churn, None, "churned"),
        "long_solo": sim.run([Job(p, mt) for p in longs], 1, "long_solo"),
        "long_together": sim.run([Job(p, mt) for p in longs], None, "long_together"),
    }
    segs = list(phases.values())
    return {"segments": segs, "agg": aggregate(segs)}


def run_warm_repeat(sim: Sim, args: argparse.Namespace) -> dict:
    result = {}
    rng = np.random.default_rng(args.seed)
    for length in WARM_LENGTHS:
        ids = rng.integers(0, 248000, size=length).tolist()
        sim.reset()
        first = sim.run([Job(ids, args.warm_max_tokens)], 1, "pass1")
        second = sim.run([Job(ids, args.warm_max_tokens)], 1, "pass2")
        result[str(length)] = {
            "segments": [first, second],
            "agg": aggregate([first, second]),
        }
    return result


def mixed_jobs(args: argparse.Namespace) -> tuple[list[Job], list[int]]:
    """64 jobs, log-uniform prompts in [200, 30000], outputs 64..512; half of them
    start with a multiple-of-1024 prefix of an earlier prompt (the second value:
    the reused prefix length per job, 0 = none)."""
    rng = np.random.default_rng(args.seed)
    prompts: list[list[int]] = []
    jobs: list[Job] = []
    reused: list[int] = []
    for _ in range(64):
        length = int(math.exp(rng.uniform(math.log(200), math.log(30000))))
        out = int(rng.integers(64, 513))
        reuse = rng.random() < 0.5
        donors = [p for p in prompts if len(p) > 1024]
        prefix = 0
        if reuse and donors:
            length = max(length, 1025)
            donor = donors[int(rng.integers(0, len(donors)))]
            k = int(
                rng.integers(1, min((len(donor) - 1) // 1024, (length - 1) // 1024) + 1)
            )
            prefix = k * 1024
        ids = rng.integers(0, 248000, size=length).tolist()
        if prefix:
            ids[:prefix] = donor[:prefix]
        prompts.append(ids)
        jobs.append(Job(ids, out))
        reused.append(prefix)
    return jobs, reused


def run_mixed(sim: Sim, args: argparse.Namespace) -> dict:
    sim.reset()
    jobs, reused = mixed_jobs(args)
    seg = sim.run(jobs, 8, "mixed")
    hits = [r["hit"] for r in seg["requests"]]
    return {
        "segments": [seg],
        "agg": aggregate([seg]),
        "reuse": {
            "requests_planned": sum(1 for p in reused if p),
            "planned_prefix_len": reused,
            "requests_with_hit": sum(1 for h in hits if h),
            "hit_tokens": sum(hits),
            "prompt_tokens": sum(len(j.ids) for j in jobs),
        },
    }


def run_prompt_logprobs(sim: Sim, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, 248000, size=5000).tolist()
    sim.reset()
    alone = sim.run([Job(ids, 16, prompt_logprobs=5)], 1, "alone")
    sim.reset()
    decoders = [Job(rng.integers(0, 248000, size=100).tolist(), 256) for _ in range(3)]
    mixed = sim.run(
        [*decoders, Job(ids, 16, arrive=4, prompt_logprobs=5)], None, "beside_decodes"
    )
    return {
        "segments": [alone, mixed],
        "agg": aggregate([alone, mixed]),
    }


def rle(seq: list[int], limit: int = 60) -> str:
    runs = []
    for v in seq:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    text = " ".join(f"{v}x{c}" if c > 1 else str(v) for v, c in runs[:limit])
    return text + (f" ... (+{len(runs) - limit} runs)" if len(runs) > limit else "")


def fmt_hist(hist: dict, top: int = 0) -> str:
    items = sorted(hist.items(), key=lambda kv: -kv[1]) if top else list(hist.items())
    items = items[:top] if top else items
    return ", ".join(f"{k}:{v}" for k, v in items) or "-"


def fmt_result(name: str, variant: str, res: dict, detail: bool) -> list[str]:
    agg = res["agg"]
    lines = [
        (
            f"== {name} [{variant}] steps={agg['steps']} "
            f"prefill_steps={agg['prefill_steps']} "
            f"tokens={agg['tokens']} max_M={agg['max_m']} "
            f"preemptions={agg['preemptions']}"
        )
    ]
    for seg in res["segments"]:
        ms = [s[0] for s in seg["steps"]]
        reqs = seg["requests"]
        hits = [r["hit"] for r in reqs]
        lines.append(
            f"  seg {seg['label']}: {len(ms)} steps, {len(reqs)} reqs, "
            f"prompt_len={[r['prompt_len'] for r in reqs][:12]}, hit={hits[:12]}"
        )
        lines.append(f"    M per step: {rle(ms, 200 if detail else 40)}")
        chunks = [r["chunks"] for r in reqs]
        lines.append(
            f"    prefill chunks: req0 {chunks[0][:40]}, last req {chunks[-1][:40]}"
        )
    lines.append(f"  M>48 count: {fmt_hist(agg['hist_m_gt48'], 30)}")
    shares = agg["flop_share_m_gt48"]
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:15]
    lines.append(
        "  FLOP share M>48 (top): " + ", ".join(f"{m}:{s:.3f}" for m, s in top)
    )
    lines.append(
        "  FLOP share by class: "
        + ", ".join(f"{c}={s:.3f}" for c, s in agg["flop_class_share_m_gt48"].items())
    )
    lines.append(f"  M 9..48: {fmt_hist(agg['hist_m_9_48'])}")
    lines.append(f"  rows 9..48: {fmt_hist(agg['hist_rows_9_48'])}")
    if agg["hist_plp_lm_head_chunks"]:
        lines.append(
            f"  prompt_logprobs lm_head chunk rows: {agg['hist_plp_lm_head_chunks']}"
        )
    if "agg_run_only" in res:
        run = res["agg_run_only"]
        lines.append(
            "  timed run only (no test run): FLOP share by class: "
            + ", ".join(
                f"{c}={s:.3f}" for c, s in run["flop_class_share_m_gt48"].items()
            )
            + f"; M>48 count: {fmt_hist(run['hist_m_gt48'], 30)}"
        )
    if "reuse" in res:
        reuse = {k: v for k, v in res["reuse"].items() if k != "planned_prefix_len"}
        lines.append(f"  prefix reuse: {reuse}")
    return lines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, default=SNAPSHOT, help="checkpoint snapshot")
    ap.add_argument("--out-dir", type=Path, default=Path("sched_census"))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--mtp", type=int, default=2, help="draft tokens of the MTP runs")
    ap.add_argument(
        "--accepted",
        default="2,1",
        help="accepted drafts per spec step, one MTP variant each (2 = steady-state "
        "3 tokens/request/step)",
    )
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--pool-blocks", type=int, default=20000)
    ap.add_argument(
        "--churn-interval",
        type=int,
        default=3,
        help="probe churned phase: scheduler steps between arrivals",
    )
    ap.add_argument("--probe-num-prompts", type=int, default=8)
    ap.add_argument("--probe-max-tokens", type=int, default=64)
    ap.add_argument("--probe-long-prompts", type=int, default=4)
    ap.add_argument("--probe-long-sentences", type=int, default=90)
    ap.add_argument("--warm-max-tokens", type=int, default=64)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    workloads = args.workloads.split(",")
    unknown = set(workloads) - set(WORKLOADS)
    assert not unknown, f"unknown workloads {unknown}"
    variants = [("mtp0", 0, 0)] + [
        (f"mtp{args.mtp}_acc{a}", args.mtp, int(a)) for a in args.accepted.split(",")
    ]
    init_none_hash(sha256)
    tokenizer = get_tokenizer(str(args.model), tokenizer_mode="auto")
    census: dict = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "num_special_tokens_to_add": int(tokenizer.num_special_tokens_to_add()),
        "cells_prompts": {},
        "probe_prompts": {},
        "results": {},
    }
    cells = make_cells(tokenizer, args.seed) if "cells" in workloads else []
    for cell in cells:
        lens = [len(i) for i in cell["ids"]]
        census["cells_prompts"][cell["name"]] = {
            "realised_prompt_tokens": length_stats(lens),
            "dataset_reported_prompt_len": cell["dataset_prompt_lens"],
            "requested_input_len": cell["input_len"],
            "output_lens": cell["output_lens"],
            "server_retokenize_mismatch": sum(
                a != b for a, b in zip(lens, cell["dataset_prompt_lens"])
            ),
        }
    short = longs = []
    if "probe" in workloads:
        short, longs = probe_prompts(tokenizer, args)
        census["probe_prompts"] = {
            "short_lens": [len(p) for p in short],
            "long_lens": [len(p) for p in longs],
        }
    runners = {
        "cells": lambda s: run_cells(s, cells),
        "probe": lambda s: run_probe(s, args, short, longs),
        "warm_repeat": lambda s: run_warm_repeat(s, args),
        "mixed": lambda s: run_mixed(s, args),
        "prompt_logprobs": lambda s: run_prompt_logprobs(s, args),
    }
    # The Rocm platform needs an arch to build the config; no device is opened.
    with patch.object(
        RocmPlatform, "get_device_capability", return_value=DeviceCapability(10, 3)
    ):
        for label, num_spec, accepted in variants:
            sim = Sim(args, num_spec, accepted)
            for name in workloads:
                t0 = time.time()
                census["results"].setdefault(name, {})[label] = runners[name](sim)
                print(f"{name} [{label}] done in {time.time() - t0:.1f}s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "census.json").write_text(json.dumps(census))
    lines = [
        f"config: {json.dumps(census['config'])}",
        (
            "checks asserted on every request of every run: prefix hit + sum(prefill "
            "chunks) == prompt length; every prefill chunk ends on a 1024-token state "
            "block boundary "
            "or at prompt-1 (identical-resend replay stop) / the prompt end"
        ),
        "",
    ]
    for name, per_cell in census["cells_prompts"].items():
        st = per_cell["realised_prompt_tokens"]
        lines.append(
            f"cell {name}: realised prompt tokens min={st['min']} "
            f"median={st['median']:.0f} mean={st['mean']:.1f} max={st['max']} "
            f"(IN={per_cell['requested_input_len']}, "
            f"retokenize mismatches={per_cell['server_retokenize_mismatch']}) "
            f"lens={st['lens']}"
        )
    if census["probe_prompts"]:
        lines.append(f"probe prompt tokens: {census['probe_prompts']}")
    lines.append("")
    for name, per_variant in census["results"].items():
        detail = name != "cells"
        for variant, res in per_variant.items():
            # cells / warm_repeat / prompt_logprobs hold one entry per sub-workload
            subs = res.items() if "agg" not in res else [(None, res)]
            for sub, subres in subs:
                lines += fmt_result(
                    f"{name}/{sub}" if sub else name, variant, subres, detail
                )
            lines.append("")
    (args.out_dir / "census.txt").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
