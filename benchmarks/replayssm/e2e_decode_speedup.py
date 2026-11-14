# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end autoregressive decode benchmark: ReplaySSM vs the standard SSM kernel.

Loads a hybrid Mamba2 model, replicates one prompt across the batch, and times a
long greedy decode (CUDA graphs on) once with the standard kernel and once with
ReplaySSM, then reports the per-step / throughput speedup. The two modes run in
separate subprocesses so each gets a clean CUDA context.

The FlashInfer FP4-MoE autotuner is disabled by default (it is unstable under
CUDA-graph capture on the pre-release Blackwell FP4 path); pass
--no-disable-flashinfer-autotune for non-FP4 models.

Examples:
    python e2e_decode_speedup.py --model-id nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
    python e2e_decode_speedup.py --dtype auto --buffer-len 16 \
        --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4   # B300 NVFP4

RDNA2 (gfx1030) LeapQuant arm
-----------------------------
The ``leapquant`` arm runs the same decode with the opt-in LeapQuant INT8 state
storage of Gated DeltaNet models (``--replayssm-state-dtype leapquant``, HIP
kernels, docs/rdna2/leapquant-gdn-2026-11-13.md). The arm passes
``replayssm_state_dtype="leapquant"`` to ``LLM(...)``, the CacheConfig field the
flag sets; no other arm does, so a baseline never runs LeapQuant by accident
(``--llm-kwarg replayssm_state_dtype=...`` is refused). The retired environment
variable VLLM_RDNA_GDN_LEAPQUANT=1 is refused in every arm: the engine ignores
it, so it would mislabel a run. ``--modes standard,leapquant``
with ``--mamba-ssm-cache-dtype float32`` is the FP32-state baseline against
LeapQuant. ``--num-speculative-tokens N`` adds MTP-N to every arm and
``--monokernel`` the MonoKernel, as in tools/rdna/recipes/flashnext-leapquant-mtp2.env.
The engine's start-up validator fails loudly off gfx10x, and the worker checks that
the engine config holds the format.

    python e2e_decode_speedup.py --model-id <Gated DeltaNet model> --dtype float16 \
        --modes standard,leapquant --mamba-ssm-cache-dtype float32 \
        --num-speculative-tokens 2 --monokernel --no-disable-flashinfer-autotune \
        --env VLLM_USE_V2_MODEL_RUNNER=1 --env VLLM_GDN_HIP_PREFILL=1 \
        --batch-size 8 --num-steps 512 --warmup-steps 64
    python e2e_decode_speedup.py --modes standard,leapquant --dry-run   # CPU only

Not covered: a 16-bit per-window baseline (the 16-bit state with a record ring of
the RFC, vllm-project/vllm#64725). It needs Gated DeltaNet ReplaySSM kernels
(vllm-project/vllm#62004), which this tree does not have; ``--modes replayssm``
is Mamba2 only.

Credits: the ReplaySSM benchmark is from vllm-project/vllm#48018 (Johnny-Liou).
The LeapQuant arm, the arm/env handling and --dry-run were added in
opengfx1030/vllm-rdna; the comparison follows item 13 (section 4) of the
LeapQuant RFC, vllm-project/vllm#64725 (lihaoyang-amd).

"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

DEFAULT_PROMPT = "My cat wrote all this CUDA code for a new language model and"

MODE_LABEL = {
    "standard": "standard",
    "replayssm": "ReplaySSM",
    "leapquant": "LeapQuant",
}
DEFAULT_MODES = "standard,replayssm"
# CacheConfig field behind ``--replayssm-state-dtype``: the only switch of the
# LeapQuant state format. ``VLLM_RDNA_GDN_LEAPQUANT`` was removed and is ignored
# by the engine. Read here without importing vllm (--dry-run is CPU only).
STATE_DTYPE_KEY = "replayssm_state_dtype"
RETIRED_ENV = "VLLM_RDNA_GDN_LEAPQUANT"


def parse_modes(text: str) -> list[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODE_LABEL]
    if unknown or not modes or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError(
            f"--modes takes distinct names from {sorted(MODE_LABEL)}, got {text!r}"
        )
    return modes


def parse_key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def parse_llm_kwarg(text: str) -> tuple[str, object]:
    key, value = parse_key_value(text)
    if key == STATE_DTYPE_KEY:
        raise argparse.ArgumentTypeError(
            f"{STATE_DTYPE_KEY} is set by the arm (leapquant: 'leapquant'), "
            "not by --llm-kwarg"
        )
    try:
        return key, json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(
            f"--llm-kwarg {key}: the value must be JSON, got {value!r}"
        ) from e


def parse_env(text: str) -> tuple[str, str]:
    key, value = parse_key_value(text)
    if key == RETIRED_ENV:
        raise argparse.ArgumentTypeError(
            f"{RETIRED_ENV} was replaced by --replayssm-state-dtype leapquant, "
            "which the leapquant arm sets; the engine ignores the variable"
        )
    return key, value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="E2E decode speedup: ReplaySSM or LeapQuant vs the standard "
        "SSM kernel.",
        epilog="See the module docstring for the RDNA2 LeapQuant example.",
    )
    p.add_argument("--model-id", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--buffer-len", type=int, default=16, help="ReplaySSM input-buffer length."
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument(
        "--disable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable the FlashInfer FP4-MoE autotuner (default: on). "
        "It is unstable under CUDA-graph capture on the "
        "pre-release Blackwell FP4 path; pass "
        "--no-disable-flashinfer-autotune for non-FP4 models.",
    )
    p.add_argument(
        "--mamba-ssm-cache-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="SSM state dtype (all modes). 'auto' = config-driven; "
        "'float32' = fp32 state, 'bfloat16' = s16 state.",
    )
    p.add_argument(
        "--baseline-ssm-config",
        default="",
        help="Pin the STANDARD baseline's SSM launch config as "
        "'bsm,nw' via override_ssm_config (forces the in-process "
        "engine so the override reaches the kernel). Empty = off.",
    )
    p.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes(DEFAULT_MODES),
        metavar="MODES",
        help=f"Arms to run, in this order, from {sorted(MODE_LABEL)} (default: "
        f"{DEFAULT_MODES}). The first arm is the baseline of the speedup lines. "
        "'leapquant' = LeapQuant INT8 Gated DeltaNet state on RDNA2 "
        "(--replayssm-state-dtype leapquant, HIP); 'standard' with "
        "--mamba-ssm-cache-dtype float32 is its FP32 baseline.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size of every arm.",
    )
    p.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=0,
        help="MTP draft tokens per step in every arm (0 = no speculative "
        "decoding). Uses use_local_argmax_reduction, as the RDNA launchers do.",
    )
    p.add_argument(
        "--monokernel",
        action="store_true",
        help="Run every arm with the RDNA MonoKernel (--monokernel of the server).",
    )
    p.add_argument(
        "--llm-kwarg",
        action="append",
        type=parse_llm_kwarg,
        default=[],
        metavar="KEY=JSON",
        help="Extra LLM(...) keyword argument for every arm, e.g. "
        "--llm-kwarg quantization='\"awq\"' (repeatable).",
    )
    p.add_argument(
        "--env",
        action="append",
        type=parse_env,
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable of every arm's worker process (repeatable).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each arm's worker command, environment and LLM(...) keyword "
        "arguments, then exit. Imports neither torch nor vllm.",
    )
    p.add_argument(
        "--worker",
        choices=sorted(MODE_LABEL),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = p.parse_args(argv)
    if args.num_speculative_tokens < 0:
        p.error("--num-speculative-tokens must be >= 0")
    args.llm_kwargs = dict(args.llm_kwarg)
    args.extra_env = dict(args.env)
    return args


def resolve_max_model_len(args) -> int:
    if args.max_model_len is not None:
        return args.max_model_len
    return args.num_steps + 256


def build_llm_kwargs(args, mode) -> dict:
    """LLM(...) keyword arguments of one arm. Pure: imports nothing."""
    max_model_len = resolve_max_model_len(args)
    llm_kwargs = dict(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(max_model_len, args.batch_size * 64),
        enforce_eager=False,
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # SSM state dtype (applies to every arm).
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
    )
    if args.disable_flashinfer_autotune:
        # FP4-MoE autotuner is unstable under CUDA-graph capture on Blackwell;
        # re-enable (--no-disable-flashinfer-autotune) only for non-FP4 models.
        llm_kwargs["enable_flashinfer_autotune"] = False
    if mode == "replayssm":
        llm_kwargs.update(use_replayssm=True, replayssm_buffer_len=args.buffer_len)
    if mode == "leapquant":
        llm_kwargs[STATE_DTYPE_KEY] = "leapquant"
    if args.num_speculative_tokens > 0:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
            "use_local_argmax_reduction": True,
        }
    if args.monokernel:
        llm_kwargs["monokernel"] = True
    llm_kwargs.update(args.llm_kwargs)
    return llm_kwargs


def arm_env(args, mode, environ) -> dict:
    """Environment of one arm's worker: the caller's plus ``--env``. The format
    is an ``LLM(...)`` argument (build_llm_kwargs), not an environment variable.
    The retired variable stops every arm, as the launcher does: the engine
    ignores it, so a run under it would be mislabeled."""
    env = dict(environ)
    env.update(args.extra_env)
    if env.get(RETIRED_ENV, "0") == "1":
        raise SystemExit(
            f"{RETIRED_ENV}=1 is set in the environment. It was replaced by "
            "--replayssm-state-dtype leapquant, which the 'leapquant' arm "
            f"passes itself; the engine ignores {RETIRED_ENV} and the "
            f"'{mode}' arm would not run what the variable says. Unset it."
        )
    return env


def run_worker(args):
    # override_ssm_config is a module global; it only reaches the model if the
    # engine runs in-process (default V1 spawns a separate EngineCore). Force it.
    if args.worker == "standard" and args.baseline_ssm_config:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    mode = args.worker
    if mode == "leapquant":
        print(f"[{mode}] LeapQuant INT8 Gated DeltaNet state requested", flush=True)

    import torch

    from vllm import LLM, SamplingParams

    llm_kwargs = build_llm_kwargs(args, mode)

    _ssm_cm = None
    if mode == "standard" and args.baseline_ssm_config:
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import override_ssm_config

        _bsm, _nw = (int(x) for x in args.baseline_ssm_config.split(","))
        _ssm_cm = override_ssm_config((_bsm, _nw))
        _ssm_cm.__enter__()  # active through LLM() graph capture + decode
        print(
            f"[{mode}] override_ssm_config -> (BLOCK_SIZE_M={_bsm}, num_warps={_nw})",
            flush=True,
        )

    llm = LLM(**llm_kwargs)
    if mode == "leapquant":
        # The validator already refused hosts and models that cannot run it.
        state_dtype = llm.llm_engine.vllm_config.cache_config.replayssm_state_dtype
        if state_dtype != "leapquant":
            raise RuntimeError(f"the engine config has {STATE_DTYPE_KEY}={state_dtype}")
    prompts = [args.prompt] * args.batch_size

    def timed_generate(n_tokens):
        sp = SamplingParams(
            n=1,
            temperature=0.0,
            ignore_eos=True,
            min_tokens=n_tokens,
            max_tokens=n_tokens,
        )
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        elapsed = time.perf_counter() - t0
        produced = min(len(o.outputs[0].token_ids) for o in outs)
        assert produced == n_tokens, f"expected {n_tokens} tokens, got {produced}"
        return elapsed

    timed_generate(args.warmup_steps)

    best = None
    for _ in range(args.repeats):
        elapsed = timed_generate(args.num_steps)
        tok_s = args.batch_size * args.num_steps / elapsed
        per_step_ms = elapsed / args.num_steps * 1e3
        print(
            f"[{mode}] {elapsed:.3f}s  {tok_s:,.0f} tok/s  {per_step_ms:.3f} ms/step",
            flush=True,
        )
        if best is None or elapsed < best["elapsed_s"]:
            best = {
                "mode": mode,
                "elapsed_s": elapsed,
                "tok_s": tok_s,
                "per_step_ms": per_step_ms,
            }

    print("RESULT_JSON " + json.dumps(best), flush=True)
    if _ssm_cm is not None:
        _ssm_cm.__exit__(None, None, None)


def worker_command(args, mode) -> list[str]:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        mode,
        "--model-id",
        args.model_id,
        "--prompt",
        args.prompt,
        "--batch-size",
        str(args.batch_size),
        "--num-steps",
        str(args.num_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--repeats",
        str(args.repeats),
        "--buffer-len",
        str(args.buffer_len),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--mamba-ssm-cache-dtype",
        args.mamba_ssm_cache_dtype,
        "--baseline-ssm-config",
        args.baseline_ssm_config,
    ]
    cmd.append(
        "--disable-flashinfer-autotune"
        if args.disable_flashinfer_autotune
        else "--no-disable-flashinfer-autotune"
    )
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.tensor_parallel_size != 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    if args.num_speculative_tokens:
        cmd += ["--num-speculative-tokens", str(args.num_speculative_tokens)]
    if args.monokernel:
        cmd.append("--monokernel")
    for key, value in args.llm_kwargs.items():
        cmd += ["--llm-kwarg", f"{key}={json.dumps(value)}"]
    return cmd


def run_one_mode(args, mode) -> dict:
    cmd = worker_command(args, mode)
    env = arm_env(args, mode, os.environ)

    result = None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON ") :])
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"mode '{mode}' worker exited with {proc.returncode}")
    if result is None:
        raise RuntimeError(f"mode '{mode}' produced no RESULT_JSON line")
    return result


def dry_run(args) -> None:
    for mode in args.modes:
        env = arm_env(args, mode, os.environ)
        delta = {k: v for k, v in env.items() if os.environ.get(k) != v}
        print(f"arm {mode} ({MODE_LABEL[mode]})")
        print(f"  env +{json.dumps(delta, sort_keys=True)}")
        print(f"  cmd {shlex.join(worker_command(args, mode))}")
        print(f"  LLM {json.dumps(build_llm_kwargs(args, mode))}")
        if mode == "leapquant":
            print("  engine validates replayssm_state_dtype=leapquant: gfx10x (HIP)")


def main(argv=None):
    args = parse_args(argv)
    if args.worker is not None:
        run_worker(args)
        return
    if args.dry_run:
        dry_run(args)
        return

    print(
        f"model={args.model_id}  batch_size={args.batch_size}  "
        f"steps={args.num_steps}  buffer_len={args.buffer_len}  dtype={args.dtype}"
    )

    results = [run_one_mode(args, mode) for mode in args.modes]

    print()
    header = f"{'mode':<10}{'ms/step':>12}{'tok/s':>16}{'wall (s)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{MODE_LABEL[r['mode']]:<10}{r['per_step_ms']:>12.3f}"
            f"{r['tok_s']:>16,.0f}{r['elapsed_s']:>12.3f}"
        )
    print("-" * len(header))
    base = results[0]
    for r in results[1:]:
        speedup = base["per_step_ms"] / r["per_step_ms"]
        print(
            f"speedup ({MODE_LABEL[base['mode']]} / {MODE_LABEL[r['mode']]}, "
            f"per step): {speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
