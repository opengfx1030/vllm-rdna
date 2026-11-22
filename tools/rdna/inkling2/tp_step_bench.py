#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step time of the real Inkling-2-Mini checkpoint at TP = 1, 2, 4 (engine level).

Boots the checkpoint like ``real_check.py`` (offline ``LLM``, eager, fp16,
``--tensor-parallel-size`` GPUs) with the engine core in this process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``: the TP workers stay separate processes of the
executor) and drives ``LLMEngine.step()`` one step at a time, like
``profile_decode.py`` pass A1 does at TP=1 (the numbers are therefore comparable with
its 23.7 ms at batch 1 / 35.8 ms at batch 4):

* ``--reps`` phases per batch size: one prefill step of ``b`` 22-token prompts, then
  ``--decode-steps`` decode steps;
* ``--long-prefill N``: the prefill of one ``N``-token prompt in chunks of
  ``--max-num-batched-tokens`` (time to the first token, tokens/s);
* ``--profile-dir DIR``: a torch profiler trace of ``--profile-steps`` decode steps per
  batch size on every worker; the rank-0 trace is summarised by kernel family (GPU
  time of the all-reduce / all-gather kernels, RCCL or ``rdna_ar``, per step) in the
  ``--json`` output (``comm``: per-call duration statistics of the collectives; the
  minimum is the cost without waiting for the other ranks, the mean includes it);
* ``--no-comm``: replaces the all-reduce of the model by the identity in every worker
  (the outputs are meaningless): the step time without communication, so the
  difference to a normal run is the all-reduce cost per step;
* the planner's view of the booted engine (``kv_capacity``): KV groups, blocks one
  request needs at a given length, and the longest single request the blocks hold.

    HIP_VISIBLE_DEVICES=4,5 python tp_step_bench.py --ckpt ~/hfcache/inkling2-mini \
        --tensor-parallel-size 2 --ar-backend rdna --json tp2.json
"""

from __future__ import annotations

import os

# the engine core must live in this process: set before vllm is imported
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import glob
import gzip
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

PROMPT_LEN = 22


def kernel_family(name: str) -> str:
    n = name.lower()
    if "nccl" in n or "rccl" in n:
        return "rccl"
    if "rdna_ar" in n or "rdna_allreduce" in n or "allreduce" in n:
        return "rdna_ar"
    if "inkling2_moe" in n or "moe_" in n or "nvfp4" in n or "mxfp8" in n:
        return "moe"
    if "inkling2_attn" in n or "attn" in n:
        return "attention"
    if "inkling2_linear" in n or "linear" in n or "gemv" in n:
        return "linear"
    if "inkling2" in n:
        return "glue"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    return "other"


def summarize_trace(path: Path, steps: int) -> dict:
    """GPU kernel time of one profiler trace by family (microseconds per step)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    fam_us: dict[str, float] = defaultdict(float)
    fam_n: dict[str, int] = defaultdict(int)
    for e in kernels:
        fam = kernel_family(e["name"])
        fam_us[fam] += float(e["dur"])
        fam_n[fam] += 1
    total = sum(fam_us.values())
    comm = {}
    for fam in ("rccl", "rdna_ar"):
        durs = sorted(
            float(e["dur"]) for e in kernels if kernel_family(e["name"]) == fam
        )
        if durs:
            comm[fam] = {
                "calls_per_step": len(durs) / steps,
                "min_us": durs[0],
                "p10_us": durs[len(durs) // 10],
                "median_us": statistics.median(durs),
                "mean_us": statistics.fmean(durs),
            }
    return {
        "steps": steps,
        "kernels_per_step": len(kernels) / steps,
        "gpu_busy_us_per_step": total / steps,
        "family_us_per_step": {k: v / steps for k, v in sorted(fam_us.items())},
        "family_kernels_per_step": {k: v / steps for k, v in sorted(fam_n.items())},
        "comm": comm,
    }


class Rig:
    def __init__(self, llm, seed: int = 1234) -> None:
        from vllm import SamplingParams

        self.llm = llm
        self.eng = llm.llm_engine
        self.rng = random.Random(seed)
        self.counter = 0
        self.SamplingParams = SamplingParams

    def _prompt(self, n: int) -> dict:
        return {"prompt_token_ids": [self.rng.randrange(1000, 50000) for _ in range(n)]}

    def run_phase(
        self, nreq: int, ndec: int, prompt_len: int = PROMPT_LEN, before_decode=None
    ) -> dict:
        """``nreq`` fresh requests: all prefill steps, then ``ndec`` decode steps."""
        sp = self.SamplingParams(
            temperature=0.0, max_tokens=ndec + 16, ignore_eos=True, detokenize=False
        )
        ids = []
        for _ in range(nreq):
            self.counter += 1
            rid = f"p{self.counter}"
            self.eng.add_request(rid, self._prompt(prompt_len), sp)
            ids.append(rid)
        prefill_ms, produced = [], 0
        while (
            produced < nreq
        ):  # the step that finishes every prompt emits its first token
            t0 = time.perf_counter()
            outs = self.eng.step()
            prefill_ms.append((time.perf_counter() - t0) * 1e3)
            produced += sum(len(o.outputs[0].token_ids) > 0 for o in outs)
        if before_decode:
            before_decode()
        decode_ms = []
        for _ in range(ndec):
            t0 = time.perf_counter()
            self.eng.step()
            decode_ms.append((time.perf_counter() - t0) * 1e3)
        self.eng.abort_request(ids)
        assert not self.eng.has_unfinished_requests()
        return {"prefill_ms": prefill_ms, "decode_ms": decode_ms}


def kv_capacity(llm) -> dict:
    """The planner's view of the booted engine (in-process engine core).

    ``blocks(L)`` is what ``get_max_concurrency_for_kv_cache_config`` charges one
    request of length ``L``: the sum over the KV cache groups of
    ``cdiv(max_memory_usage_bytes, page_size_bytes)``; the engine holds
    ``num_blocks``.  ``longest_single_request`` is the largest ``L`` with
    ``blocks(L) <= num_blocks`` (the engine refuses a ``max_model_len`` above it).
    """
    from vllm.utils.math_utils import cdiv

    vllm_config = llm.llm_engine.vllm_config
    kv_config = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config
    model_config = vllm_config.model_config
    groups = kv_config.kv_cache_groups
    booted_len = model_config.max_model_len

    def blocks(length: int) -> int:
        model_config.max_model_len = length
        try:
            return sum(
                cdiv(
                    g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                    g.kv_cache_spec.page_size_bytes,
                )
                for g in groups
            )
        finally:
            model_config.max_model_len = booted_len

    lo, hi = 1, 1 << 23
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if blocks(mid) <= kv_config.num_blocks:
            lo = mid
        else:
            hi = mid - 1
    page = groups[0].kv_cache_spec.page_size_bytes
    return {
        "num_blocks": kv_config.num_blocks,
        # a block id owns one page in each of the group_size shared tensors
        "block_id_bytes": max(len(g.layer_names) for g in groups) * page,
        "groups": [
            {
                "spec": type(g.kv_cache_spec).__name__,
                "layers": len(g.layer_names),
                "block_size": g.kv_cache_spec.block_size,
                "page_size_bytes": g.kv_cache_spec.page_size_bytes,
                "blocks_at_booted_len": cdiv(
                    g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                    g.kv_cache_spec.page_size_bytes,
                ),
            }
            for g in groups
        ],
        "blocks_at": {str(n): blocks(n) for n in (8192, 65536, 262144, 1048576)},
        "longest_single_request": lo,
    }


def drop_all_reduce(model) -> str:
    """``--no-comm`` (run in every worker): the model's all-reduce is the identity."""
    cls = type(model.tp)
    cls.all_reduce = lambda self, x: x
    return cls.__name__


def memory_report(model) -> dict:
    """GiB of one worker's device once the engine is up (run in every worker)."""
    import torch

    gib = float(1 << 30)
    free, total = torch.accelerator.get_memory_info()
    reserved = torch.accelerator.memory_reserved()
    return {
        "torch_allocated": torch.accelerator.memory_allocated() / gib,
        "torch_peak_allocated": torch.accelerator.max_memory_allocated() / gib,
        "torch_reserved": reserved / gib,
        "device_used": (total - free) / gib,
        "device_total": total / gib,
        "non_torch_used": (total - free - reserved) / gib,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--ar-backend", choices=["rccl", "rdna"], default="rccl")
    ap.add_argument("--ar-max-kb", type=int, help="VLLM_RDNA_AR_MAX_KB / ONESHOT_KB")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--long-prefill", type=int, default=0)
    ap.add_argument("--profile-dir", default="")
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--no-comm",
        action="store_true",
        help="timing experiment: the all-reduce of the model is the identity (outputs "
        "are meaningless)",
    )
    args = ap.parse_args()
    # the workers receive functions (apply_model: memory report, --no-comm): pickle
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if args.ar_backend == "rdna":
        os.environ["VLLM_RDNA_AR"] = "1"
        if args.ar_max_kb:
            os.environ["VLLM_RDNA_AR_MAX_KB"] = str(args.ar_max_kb)
            os.environ["VLLM_RDNA_AR_ONESHOT_KB"] = str(args.ar_max_kb)

    from vllm import LLM

    extra = {}
    if args.profile_dir:
        profile_dir = Path(args.profile_dir).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        extra["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_activities": ["CPU", "CUDA"],
        }
    t0 = time.time()
    llm = LLM(
        model=args.ckpt,
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(args.batch, 2),
        skip_tokenizer_init=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 0},  # text only
        **extra,
    )
    boot = time.time() - t0
    cache = llm.llm_engine.vllm_config.cache_config
    print(
        f"BOOT OK in {boot:.1f}s tp={args.tensor_parallel_size} ar={args.ar_backend} "
        f"num_gpu_blocks={cache.num_gpu_blocks} block_size={cache.block_size}",
        flush=True,
    )
    if args.no_comm:
        print(
            "no-comm: all-reduce is the identity in", llm.apply_model(drop_all_reduce)
        )
    rig = Rig(llm)
    batches = sorted({1, args.batch})
    for b in batches:  # warm-up: allocator, kernels of each batch shape
        rig.run_phase(b, 3)

    results: dict = {
        "args": vars(args),
        "boot_s": boot,
        "num_gpu_blocks": cache.num_gpu_blocks,
        "block_size": cache.block_size,
        "decode": {},
    }
    try:
        results["kv_capacity"] = kv_capacity(llm)
        cap = results["kv_capacity"]
        print(
            f"KV planner: {cap['num_blocks']} blocks of {cap['block_id_bytes']} B, "
            f"{len(cap['groups'])} groups; blocks one request needs: "
            f"{cap['blocks_at']}; longest single request "
            f"{cap['longest_single_request']} tokens",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001  (diagnostic only: report, do not stop)
        results["kv_capacity"] = {"error": repr(e)}
        print("KV planner report failed:", repr(e), flush=True)
    try:
        results["memory_gib"] = llm.apply_model(memory_report)
        for rank, m in enumerate(results["memory_gib"]):
            print(
                f"memory rank {rank} (GiB): "
                + ", ".join(f"{k} {v:.2f}" for k, v in m.items()),
                flush=True,
            )
    except Exception as e:  # noqa: BLE001  (diagnostic only)
        print("memory report failed:", repr(e), flush=True)
    for b in batches:
        phases = [rig.run_phase(b, args.decode_steps) for _ in range(args.reps)]
        dec = [x for p in phases for x in p["decode_ms"]]
        pre = [x for p in phases for x in p["prefill_ms"]]
        results["decode"][b] = {
            "decode_ms": dec,
            "prefill_ms": pre,
            "decode_median": statistics.median(dec),
            "decode_mean": statistics.fmean(dec),
            "decode_min": min(dec),
            "decode_max": max(dec),
            "prefill_median": statistics.median(pre),
        }
        d = results["decode"][b]
        print(
            f"batch {b}: decode ms median {d['decode_median']:.2f} mean "
            f"{d['decode_mean']:.2f} min {d['decode_min']:.2f} "
            f"max {d['decode_max']:.2f}; prefill step ({PROMPT_LEN} tokens x {b}) "
            f"median {d['prefill_median']:.1f}",
            flush=True,
        )

    if args.long_prefill:
        n = args.long_prefill
        runs = []
        for _ in range(max(args.reps, 3)):
            ph = rig.run_phase(1, 0, prompt_len=n)
            runs.append(sum(ph["prefill_ms"]))
        med = statistics.median(runs)
        results["long_prefill"] = {
            "tokens": n,
            "ms": runs,
            "median_ms": med,
            "tokens_per_s": n / (med / 1e3),
        }
        print(
            f"long prefill {n} tokens (chunks of {args.max_num_batched_tokens}): "
            f"median {med:.0f} ms = {n / (med / 1e3):.0f} tokens/s",
            flush=True,
        )

    if args.profile_dir:
        results["profile"] = {}
        for b in batches:
            before = set(glob.glob(str(profile_dir / "*")))
            prefix = f"tp{args.tensor_parallel_size}_b{b}"

            def start(prefix=prefix):
                llm.start_profile(profile_prefix=prefix)

            rig.run_phase(b, args.profile_steps, before_decode=start)
            llm.stop_profile()
            time.sleep(5)  # the workers write their traces after stop_profile returns
            new = sorted(set(glob.glob(str(profile_dir / "*"))) - before)
            print(f"profile batch {b}: {[Path(p).name for p in new]}", flush=True)
            rank0 = [p for p in new if "_tp0" in Path(p).name]
            traces = [p for p in (rank0 or new) if ".json" in Path(p).name]
            if traces:
                summary = summarize_trace(Path(traces[0]), args.profile_steps)
                results["profile"][b] = summary
                fam = {
                    k: round(v / 1e3, 3)
                    for k, v in summary["family_us_per_step"].items()
                }
                print(
                    f"  rank 0 ({Path(traces[0]).name}): GPU busy "
                    f"{summary['gpu_busy_us_per_step'] / 1e3:.2f} ms/step, "
                    f"{summary['kernels_per_step']:.0f} kernels/step; per family "
                    f"{fam} ms/step",
                    flush=True,
                )
                for name, c in summary["comm"].items():
                    print(
                        f"  {name}: {c['calls_per_step']:.0f} calls/step, per call min "
                        f"{c['min_us']:.1f} p10 {c['p10_us']:.1f} median "
                        f"{c['median_us']:.1f} mean {c['mean_us']:.1f} us (the mean "
                        "includes the wait for the other rank)",
                        flush=True,
                    )
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
