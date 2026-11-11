# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Private TP4 cache-format qualification; explicitly run after integration gates.

Uses the existing launcher and quality probes. Each invocation owns one server
process group and one fresh output directory. No source/build mutation occurs.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_events import classify  # noqa: E402

GPUS = "2,3,4,5"
TREE = Path("/home/chenco_adm/vllm-pr-int4fp16")
RUNTIME = Path("/home/chenco_adm/Apps/vllm/venv-pr134-int4fp16")
MODEL = Path(
    "/home/chenco_adm/.cache/huggingface/hub/models--wtdcode--"
    "Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/"
    "0939125b929543a783ce700c90e36dd1a575c00c"
)
FAULT = re.compile(
    r"Memory Fault|HSA_STATUS_ERROR|page fault|Segmentation fault|"
    r"core dumped|amdgpu.*(?:timeout|reset)",
    re.I,
)
KERNEL_FAULT = re.compile(
    r"amdgpu.*(?:fault|timeout|reset)|UTCL2|AER|Hardware Error", re.I
)
CELLS = (
    (1024, 512, 1, 8),
    (1024, 512, 8, 32),
    (16384, 1024, 1, 8),
    (16384, 1024, 8, 32),
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def manifest():
    paths = [*TREE.glob("vllm/*.so"), *TREE.glob("tunableop/**/*.csv")]
    for folder in (
        "csrc/rocm",
        "vllm/v1/attention",
        "vllm/config",
        "vllm/v1/core",
        "vllm/v1/worker",
        "tools/rdna",
    ):
        paths += [
            p
            for p in (TREE / folder).rglob("*")
            if p.suffix in (".py", ".cu", ".cpp", ".h", ".hpp", ".sh", ".env")
        ]
    paths += [
        TREE / "vllm/_custom_ops.py",
        TREE / "vllm/utils/torch_utils.py",
        TREE / "vllm/v1/kv_cache_interface.py",
        TREE / "vllm/model_executor/layers/quantization/kv_cache.py",
    ]
    return {
        str(p.relative_to(TREE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
    }


def run(command, out, env):
    print("command:", json.dumps([str(x) for x in command]), flush=True)
    with out.open("w") as log:
        subprocess.run(
            [str(x) for x in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def kernel_log():
    result = subprocess.run(
        ["sudo", "-n", "dmesg"], capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0, "kernel fault gate cannot read dmesg"
    return result.stdout


def check_gpu_owners(gpus=GPUS):
    owners = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            env = dict(
                x.split(b"=", 1)
                for x in (proc / "environ").read_bytes().split(b"\0")
                if b"=" in x
            )
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, ValueError):
            continue
        # Parent shells carry our mask too; only GPU workload processes block.
        if not re.search(
            r"python.*(?:api_server|cli.main serve|benchmark_native)|"
            r"VLLM::(?:Worker|EngineCore)|PleOffloadWorker",
            command,
        ):
            continue
        # Spawned vLLM workers may retain CUDA_VISIBLE_DEVICES as the ROCm
        # alias even when the parent was launched with HIP_VISIBLE_DEVICES.
        mask = next(
            (
                env[key].decode()
                for key in (
                    b"HIP_VISIBLE_DEVICES",
                    b"CUDA_VISIBLE_DEVICES",
                    b"ROCR_VISIBLE_DEVICES",
                )
                if env.get(key)
            ),
            "",
        )
        unknown = not mask or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", mask)
        if unknown or set(mask.split(",")) & set(gpus.split(",")):
            owners.append({"pid": int(proc.name), "mask": mask, "command": command})
    if owners:
        raise RuntimeError(f"authorized GPUs already owned: {owners}")


def summarize(result, concurrency):
    assert result["failed"] == 0 and result["completed"] > 0, result.get("errors")
    per_request = []
    for n, o, first, latency, gaps in zip(
        result["input_lens"],
        result["output_lens"],
        result["ttfts"],
        result["latencies"],
        result["itls"],
    ):
        assert first > 0 and latency >= first and o > 0
        decode = (o - 1) / (latency - first) if o > 1 and latency > first else None
        per_request.append(
            {
                "input_tokens": n,
                "output_tokens": o,
                "prefill_tokens_per_s": n / first,
                "decode_tokens_per_s": decode,
                "ttft_s": first,
                "itl_chunk_median_ms": statistics.median(gaps) * 1000 if gaps else None,
                "itl_chunk_mean_ms": statistics.mean(gaps) * 1000 if gaps else None,
            }
        )
    duration = result["duration"]
    all_gaps = [gap * 1000 for gaps in result["itls"] for gap in gaps]
    return {
        "per_request": per_request,
        "mean_request_prefill_tokens_per_s": statistics.mean(
            x["prefill_tokens_per_s"] for x in per_request
        ),
        "mean_request_decode_tokens_per_s": statistics.mean(
            x["decode_tokens_per_s"]
            for x in per_request
            if x["decode_tokens_per_s"] is not None
        ),
        "whole_run_input_tokens_per_s": result["total_input_tokens"] / duration,
        "aggregate_output_tokens_per_s": result["total_output_tokens"] / duration,
        "wave_prefill_proxy_tokens_per_s": min(concurrency, result["completed"])
        * statistics.mean(result["input_lens"])
        / (result["p99_ttft_ms"] / 1000),
        "mean_ttft_s": statistics.mean(result["ttfts"]),
        "itl_chunk_median_ms": statistics.median(all_gaps),
        "itl_chunk_mean_ms": statistics.mean(all_gaps),
        "acceptance_length": result.get("spec_decode_acceptance_length"),
        "acceptance_rate": result.get("spec_decode_acceptance_rate"),
        "note": (
            "Wave prefill is a proxy. Whole-run input rate is separate. "
            "ITL measures stream chunks, not TPOT."
        ),
    }


def qualification(events, raw_status, cleanup, reference_mode):
    """Classify a run's hardware attribution without touching the raw gate.

    The raw status stays exactly as the unmodified gate computed it, so a
    corrected-link PCIe/AER event still fails the raw status. This artifact
    reports the corrected/fatal split separately so a candidate can be run
    explicitly as diagnostic-only against such a reference.
    """
    verdict = events["verdict"] if events else "incomplete"
    fatal = events["fatal_events"] if events else []
    if raw_status == "PASS":
        mode = "strict-pass"
    elif verdict == "corrected-only" and not fatal:
        mode = "diagnostic-only"
    else:
        mode = "fail"
    return {
        "qualification_mode": mode,
        "raw_status": raw_status,
        "raw_cleanup": cleanup,
        "reference_mode": reference_mode,
        "hardware_attribution": (
            "clean"
            if mode == "strict-pass"
            else "contaminated-corrected-link-noise"
            if mode == "diagnostic-only"
            else "unqualified"
        ),
        "kernel_events": {
            "verdict": verdict,
            "corrected_event_groups": events["corrected_event_groups"] if events else 0,
            "corrected_error_count": events["corrected_error_count"] if events else 0,
            "fatal_event_groups": events["fatal_event_groups"] if events else 0,
            "fatal_events": fatal,
            "endpoints_with_nonfatal_err": events["endpoints_with_nonfatal_err"]
            if events
            else 0,
            "endpoints_with_bad_tlp": events["endpoints_with_bad_tlp"] if events else 0,
            "aer_recovery_buffer_overflows": events["aer_recovery_buffer_overflows"]
            if events
            else [],
        },
        "note": (
            "Raw status is the unmodified gate result. Corrected-link PCIe/AER "
            "noise is classified separately and never relaxes the raw gate."
        ),
    }


def normal_eos(base, model, output):
    spec = importlib.util.spec_from_file_location(
        "concurrency_probe", TREE / "tools/rdna/port_v031/concurrency_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def complete(prompt):
        response = request(
            base + "/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0,
                "seed": 12345,
                "ignore_eos": False,
                "logprobs": 5,
                "return_tokens_as_token_ids": True,
            },
        )
        choice = response["choices"][0]
        reason = module.looks_like_garbage(choice["text"])
        lps = [
            x
            for x in (choice.get("logprobs") or {}).get("token_logprobs", [])
            if x is not None
        ]
        issues = []
        if choice["finish_reason"] not in ("stop", "length"):
            issues.append("unexpected finish reason")
        if not choice["text"].strip() or reason:
            issues.append(reason or "empty output")
        if not lps or not all(math.isfinite(x) for x in lps):
            issues.append("missing/nonfinite token logprobs")
        elif statistics.mean(lps) < -2.5:
            issues.append(f"mean token logprob {statistics.mean(lps)} below -2.5")
        return {"prompt": prompt, "response": response, "issues": issues}

    rounds = []
    for concurrency in (1, 8, 8, 1):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rounds.append(
                {
                    "concurrency": concurrency,
                    "requests": list(pool.map(complete, module.PROMPTS)),
                }
            )
        save(output, {"status": "IN_PROGRESS", "rounds": rounds})
        assert not any(item["issues"] for item in rounds[-1]["requests"]), output
    save(output, {"status": "PASS", "requests": 48, "rounds": rounds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dtype", choices=("auto", "int4_fp16_block"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18134)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument(
        "--recipe", choices=("flashnext-mtp2", "27b-awq"), default="flashnext-mtp2"
    )
    parser.add_argument("--layout", choices=("LHBNC", "LBHNC"), default="LBHNC")
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--kv-bytes", type=int, default=4026531840)
    parser.add_argument("--hw-queues", type=int, default=4)
    parser.add_argument(
        "--reference", type=Path, help="Completed FP16 arm for paired PPL check"
    )
    parser.add_argument("--max-ppl-ratio", type=float, default=1.05)
    parser.add_argument(
        "--diagnostic-reference",
        action="store_true",
        help="Allow a reference whose raw status is FAIL but whose only "
        "kernel events are corrected-link noise. The candidate is then "
        "recorded as diagnostic-only with contaminated hardware "
        "attribution; the reference's raw FAIL is preserved.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Attention block size to request. The hybrid alignment may raise "
        "it: a packed INT4 page is smaller, so the attention block grows "
        "until its page covers the mamba page.",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="SSM state dtype for the mamba cache. 'auto' keeps the model's own "
        "(float32) and leaves the server command untouched. 'float16' halves "
        "the mamba state page, which is what the hybrid page alignment uses to "
        "pick the attention block size.",
    )
    parser.add_argument(
        "--w4a8",
        choices=("0", "1"),
        default=None,
        help="Enable the int8-activation prefill GEMM path "
        "(VLLM_RDNA2_W4A8_SDOT4=1). Prefill only: decode stays W4A16. The "
        "default leaves the recipe's own W4A8 setting alone.",
    )
    parser.add_argument(
        "--decode-stall-ms",
        type=int,
        default=None,
        help="Override the decode-stall cap (VLLM_RDNA_DECODE_STALL_MS) that "
        "bounds prefill work in steps that carry decodes. 0 disables the cap. "
        "The default leaves the recipe's value alone.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="max_num_batched_tokens for the server. The default follows the "
        "attention block size the hybrid page alignment will pick: 2048 for "
        "an FP16 KV cache (block 1024), 4096 for a packed KV cache with an "
        "FP16 SSM state (block 2048), 6144 for a packed KV cache with the "
        "model's own FP32 SSM state (block 3072). A prefill chunk ends on a "
        "mamba state block boundary, so the budget must cover two blocks for "
        "two requests to be prefilled per scheduler step; at one block per "
        "step the long-prompt prefill drains at half the rate.",
    )
    parser.add_argument(
        "--gpus", default=GPUS, help="Comma-separated HIP device indices for this arm."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute only after source/build/layout gates pass",
    )
    args = parser.parse_args()
    if args.max_num_batched_tokens is None:
        # Two mamba state blocks per scheduler step, so two requests are
        # prefilled per step. The block size the hybrid page alignment picks
        # follows the KV cache page: a packed KV cache has a smaller
        # attention page, so the block grows until the page covers the mamba
        # state page (2048 with an FP16 SSM state, 3072 with FP32).
        if args.cache_dtype == "auto":
            args.max_num_batched_tokens = 2048
        elif args.mamba_ssm_cache_dtype == "float16":
            args.max_num_batched_tokens = 4096
        else:
            args.max_num_batched_tokens = 6144
    gpus = args.gpus
    assert os.environ.get("HIP_VISIBLE_DEVICES") == gpus, (
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
        f"does not match --gpus {gpus!r}"
    )
    env = os.environ.copy()
    for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(key, None)
    for key in (
        "TUNABLEOP_PROFILE",
        "TUNABLEOP_ALLOW_MISMATCH",
        "TUNABLEOP_ROOT",
        "VLLM_DFLASH2_PROPOSAL_PROBE_PATH",
    ):
        env.pop(key, None)
    env.update(
        VLLM_TREE=str(TREE),
        VENV=str(RUNTIME),
        HIP_VISIBLE_DEVICES=gpus,
        VLLM_SERVER_DEV_MODE="1",
        VLLM_RDNA_PHASE_TIMING="0",
        VLLM_RDNA_OP_TIMING="0",
        VLLM_KV_CACHE_LAYOUT=args.layout,
        PYTORCH_TUNABLEOP_TUNING="0",
        PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED="0",
        TUNEOP_REQUIRE_HASH_ROWS="1",
    )
    # The two serving levers the recipe exposes. Both are environment knobs
    # read before the server starts, so they are pinned here to keep an arm
    # self-contained: the caller's environment cannot leak into it.
    knobs = {
        "w4a8": "recipe" if args.w4a8 is None else args.w4a8,
        "decode_stall_ms": (
            "recipe" if args.decode_stall_ms is None else args.decode_stall_ms
        ),
    }
    if args.w4a8 is None:
        # No override: drop any inherited value so the recipe's own W4A8
        # setting governs the arm.
        env.pop("VLLM_RDNA2_W4A8_SDOT4", None)
    else:
        env["VLLM_RDNA2_W4A8_SDOT4"] = args.w4a8
    if args.decode_stall_ms is None:
        # No override: drop any inherited value so the recipe's own setting
        # (VLLM_RDNA_DECODE_STALL_MS) governs the arm.
        env.pop("VLLM_RDNA_DECODE_STALL_MS", None)
    else:
        env["VLLM_RDNA_DECODE_STALL_MS"] = str(args.decode_stall_ms)
    features = "prefix-caching mamba-align language-model-only skip-mm-profiling"
    if args.recipe == "flashnext-mtp2":
        features += " expert-parallel"
    # Only pass the SSM dtype when it is not the model's own: the control arm
    # must run the same server command as the arm it is compared against.
    extra_args = f"--kv-cache-dtype {args.cache_dtype} --seed 12345"
    if args.mamba_ssm_cache_dtype != "auto":
        extra_args += f" --mamba-ssm-cache-dtype {args.mamba_ssm_cache_dtype}"
    command = [
        "bash",
        TREE / "tools/rdna/serve_rdna.sh",
        f"RECIPE={args.recipe}",
        f"MODEL={args.model}",
        f"VENV={RUNTIME}",
        f"VLLM_TREE={TREE}",
        f"HIP_VISIBLE_DEVICES={gpus}",
        "ROCR_VISIBLE_DEVICES=",
        "HOST=127.0.0.1",
        f"PORT={args.port}",
        "PLE=bf16",
        "TP=4",
        "MTP=2",
        "SEQS=8",
        f"MAXLEN={args.max_model_len}",
        f"MAXBAT={args.max_num_batched_tokens}",
        f"KV={args.kv_bytes}",
        f"BLOCK_SIZE={args.block_size}",
        f"GPU_MAX_HW_QUEUES={args.hw_queues}",
        "OMP_NUM_THREADS=4",
        "TUNABLEOP=1",
        "COMPILE_MODE=3",
        f"FEATURES={features}",
        f"VLLM_CACHE_ROOT={args.output}/cache/vllm",
        f"TRITON_CACHE_DIR={args.output}/cache/triton",
        f"TORCHINDUCTOR_CACHE_DIR={args.output}/cache/inductor",
        f"EXTRA_ARGS={extra_args}",
    ]
    if not args.run:
        print(
            json.dumps(
                {
                    "command": [str(x) for x in command],
                    "cells": CELLS,
                    "status": "PLAN_ONLY; integration/build/layout gates required",
                },
                indent=2,
            )
        )
        return
    reference_mode = None
    if args.cache_dtype == "int4_fp16_block":
        assert args.reference is not None, (
            "packed qualification requires a completed FP16 reference"
        )
        reference_status = json.loads((args.reference / "status.json").read_text())
        reference_qualification = json.loads(
            (args.reference / "qualification.json").read_text()
        )
        assert reference_status["cleanup"] == "PASS", "reference cleanup did not pass"
        assert reference_qualification["kernel_events"]["fatal_events"] == [], (
            "reference has fatal device events; hardware attribution unresolved"
        )
        strict_reference = reference_status["status"] == "PASS"
        assert strict_reference or args.diagnostic_reference, (
            f"reference raw status is {reference_status['status']}, not PASS; "
            f"re-run with --diagnostic-reference to qualify a diagnostic-only "
            f"candidate against corrected-link-only hardware noise"
        )
        reference_mode = reference_qualification["qualification_mode"]
        if reference_mode != "strict-pass":
            print(
                json.dumps(
                    {
                        "reference": str(args.reference),
                        "reference_mode": reference_mode,
                        "reference_hardware_attribution": reference_qualification[
                            "hardware_attribution"
                        ],
                        "note": "Candidate is diagnostic-only: the reference raw "
                        "FAIL is preserved and the pair carries contaminated "
                        "hardware attribution.",
                    },
                    indent=2,
                ),
                flush=True,
            )
        reference_run = json.loads((args.reference / "run.json").read_text())
        reference_args = dict(
            x.split("=", 1) for x in reference_run["command"] if "=" in x
        )
        current_args = dict(str(x).split("=", 1) for x in command if "=" in str(x))
        for key in (
            "RECIPE",
            "MODEL",
            "TP",
            "MTP",
            "SEQS",
            "MAXLEN",
            "KV",
            "BLOCK_SIZE",
            "GPU_MAX_HW_QUEUES",
            "OMP_NUM_THREADS",
            "TUNABLEOP",
            "COMPILE_MODE",
            "FEATURES",
            "PLE",
        ):
            assert reference_args[key] == current_args[key], (
                f"reference configuration differs: {key}"
            )
        # MAXBAT is deliberately not a pairing constraint: the budget is
        # derived from the attention block size, which the hybrid page
        # alignment picks per KV dtype (1024 for FP16 KV, 2048 for a packed
        # KV cache with an FP16 SSM state, 3072 with FP32). A packed
        # candidate and its FP16 reference therefore carry different budgets
        # for the same requested block size. The invariant that matters is
        # per arm: the budget must cover two mamba state blocks, which the
        # server log assertion above enforces on both arms.
        assert reference_run["kv_cache_layout"] == args.layout
        assert reference_run.get("knobs") == knobs, (
            "reference levers differ: "
            f"{reference_run.get('knobs')} != {knobs}; the paired PPL check "
            "would compare two different server configurations"
        )
        assert reference_run["boot_id"] == boot_id()
        assert (
            json.loads((args.reference / "manifest.before.json").read_text())
            == manifest()
        ), "reference source/binary/profile mismatch"
    model_config = json.loads((args.model / "config.json").read_text())
    text_config = model_config.get("text_config", model_config)
    assert text_config.get("indexer_n_heads") is None, (
        "QSA checkpoint bypasses ordinary RDNA_ATTN; unsupported qualification model"
    )
    assert not args.output.exists(), "fresh output required"
    check_gpu_owners(gpus)
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", args.port)) != 0, "port already in use"
    args.output.mkdir(parents=True)
    before = manifest()
    kernel_before = kernel_log()
    (args.output / "kernel.before.log").write_text(kernel_before)
    save(args.output / "manifest.before.json", before)
    boot = boot_id()
    save(
        args.output / "run.json",
        {
            "boot_id": boot,
            "command": [str(x) for x in command],
            "gpus": gpus,
            "cache_dtype": args.cache_dtype,
            "kv_cache_layout": args.layout,
            "recipe": args.recipe,
            "knobs": knobs,
            "model": str(args.model),
            "reference": str(args.reference) if args.reference else None,
            "reference_mode": reference_mode,
            "model_config_sha256": hashlib.sha256(
                (args.model / "config.json").read_bytes()
            ).hexdigest(),
        },
    )
    base = f"http://127.0.0.1:{args.port}"
    server_log = (args.output / "serve.log").open("w")
    server = subprocess.Popen(
        command,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    save(args.output / "server-owner.json", {"pid": server.pid, "pgid": server.pid})
    status = "FAIL"
    events = None
    try:
        deadline = time.monotonic() + 3600
        while True:
            assert server.poll() is None, "server exited during startup"
            try:
                models = request(base + "/v1/models")
                break
            except (urllib.error.URLError, TimeoutError):
                assert time.monotonic() < deadline, "startup timeout"
                time.sleep(5)
        save(args.output / "models.json", models)
        model = models["data"][0]["id"]
        text = (args.output / "serve.log").read_text(errors="replace")
        assert "RDNA_ATTN" in text and "MTP" in text, (
            "wrong attention/spec architecture"
        )
        assert "TunableOp lookup enabled" in text and "tuning off" in text, (
            "lookup-only profile not evidenced"
        )
        assert "falling back to per-user" not in text, "unqualified TunableOp fallback"
        assert re.search(r"[Cc]aptur", text), "graph capture not evidenced"
        if args.cache_dtype != "auto":
            assert args.cache_dtype in text, "packed dtype not evidenced"
        # The mamba block-aligned prefill split ends every prefill chunk on a
        # mamba state block boundary, so a scheduler step carries whole blocks.
        # With a per-step budget of one block only one request is prefilled
        # per step and the long-prompt prefill drains at half the rate
        # (gfx1030, 27B AWQ, 16k x c=8: 42.2 -> 37.4 tok/s, TTFT 50.7 ->
        # 103.3 s at block 2048, per-token prefill cost unchanged). Refuse
        # an arm that cannot cover two blocks: it would look like a kernel
        # regression rather than a scheduling cliff.
        state_block = re.search(r"state_block=(\d+)", text)
        scheduled = re.search(r"max_num_scheduled_tokens is set to (\d+)", text)
        assert state_block and scheduled, "mamba state block size not evidenced"
        state_block = int(state_block.group(1))
        scheduled = int(scheduled.group(1))
        assert scheduled >= 2 * state_block, (
            f"max_num_scheduled_tokens={scheduled} cannot cover two mamba "
            f"state blocks of {state_block} tokens; the block-aligned prefill "
            f"split would prefill one request per step. Raise "
            f"--max-num-batched-tokens to at least {2 * state_block}."
        )
        for name, script, extra in (
            ("greedy", "tools/probe_greedy_correctness.py", ["--max-tokens", "48"]),
            ("ppl", "tools/rdna/port_v031/ppl_probe.py", ["--max-ppl", "6"]),
            ("prefix", "tools/rdna/port_v031/prefix_probe.py", []),
            ("concurrency", "tools/rdna/port_v031/concurrency_probe.py", ["--verbose"]),
        ):
            run(
                [
                    RUNTIME / "bin/python",
                    TREE / script,
                    "--url",
                    base + "/v1/completions",
                    "--model",
                    model,
                    *extra,
                ],
                args.output / f"{name}.log",
                env,
            )
            if name == "ppl" and args.reference is not None:
                pattern = r"^PPL ([0-9.]+) "
                candidate = float(
                    re.search(pattern, (args.output / "ppl.log").read_text(), re.M)[1]
                )
                reference = float(
                    re.search(pattern, (args.reference / "ppl.log").read_text(), re.M)[
                        1
                    ]
                )
                ratio = candidate / reference
                save(
                    args.output / "ppl-paired.json",
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "ratio": ratio,
                        "max_ratio": args.max_ppl_ratio,
                    },
                )
                assert ratio <= args.max_ppl_ratio, (
                    "paired PPL regression beyond configured gate"
                )
        run(
            [
                RUNTIME / "bin/python",
                TREE / "tools/rdna/port_v031/ladder_warmup.py",
                "--url",
                base,
                "--model",
                model,
                "--max-seqs",
                "8",
                "--seed",
                "12345",
            ],
            args.output / "warmup.log",
            env,
        )
        normal_eos(base, model, args.output / "normal-eos.json")
        metrics = {}
        for n, out, concurrency, count in CELLS:
            name = f"bench-{n}x{out}-c{concurrency}"
            reset = request(base + "/reset_prefix_cache", {})
            save(args.output / f"reset-{name}.json", reset)
            assert (
                reset is True
                or isinstance(reset, dict)
                and reset.get("success") is True
            ), reset
            run(
                [
                    RUNTIME / "bin/python",
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "bench",
                    "serve",
                    "--backend",
                    "openai",
                    "--endpoint",
                    "/v1/completions",
                    "--base-url",
                    base,
                    "--model",
                    model,
                    "--tokenizer",
                    args.model,
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(n),
                    "--random-output-len",
                    str(out),
                    "--num-prompts",
                    str(count),
                    "--max-concurrency",
                    str(concurrency),
                    "--temperature",
                    "0",
                    "--ignore-eos",
                    "--request-rate",
                    "inf",
                    "--seed",
                    "12345",
                    "--save-result",
                    "--save-detailed",
                    "--result-dir",
                    args.output,
                    "--result-filename",
                    name + ".json",
                ],
                args.output / f"{name}.log",
                env,
            )
            result = json.loads((args.output / f"{name}.json").read_text())
            assert result["completed"] == count
            metrics[name] = summarize(result, concurrency)
            save(args.output / "metrics.json", metrics)
        run(
            [
                RUNTIME / "bin/python",
                TREE / "tools/probe_greedy_correctness.py",
                "--url",
                base + "/v1/completions",
                "--model",
                model,
                "--max-tokens",
                "48",
            ],
            args.output / "greedy-after.log",
            env,
        )
        normal_eos(base, model, args.output / "normal-eos-after.json")
        assert not FAULT.search(
            (args.output / "serve.log").read_text(errors="replace")
        ), "fault in server log"
        kernel_after = kernel_log()
        (args.output / "kernel.after.log").write_text(kernel_after)
        last_line = kernel_before.splitlines()[-1]
        assert last_line in kernel_after, (
            "kernel log rotated; cannot establish clean delta"
        )
        delta = kernel_after.rsplit(last_line, 1)[1]
        (args.output / "kernel.delta.log").write_text(delta)
        # Classify before the raw gate so the corrected/fatal split is preserved
        # even when the raw gate fails. The raw assertion below is unchanged.
        events = classify(delta)
        save(args.output / "kernel.events.json", events)
        assert not KERNEL_FAULT.search(delta), (
            "new system GPU/PCIe fault; attribution requires investigation"
        )
        assert before == manifest(), "source/binary/rank profile changed while serving"
        assert boot == boot_id(), "boot changed"
        status = "PASS"
    finally:
        # Save provenance before teardown too: retain it even if shutdown fails.
        save(args.output / "manifest.after.json", manifest())
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)
        except ProcessLookupError:
            pass
        # Parent exit need not mean all workers exited. Scope cleanup strictly
        # to the new session this invocation created, including orphan workers.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server_log.close()
        cleanup = "PASS"
        for attempt in range(10):
            try:
                check_gpu_owners(gpus)
                break
            except RuntimeError:
                if attempt == 9:
                    cleanup = "workers remain"
                    break
                time.sleep(1)
        kernel_stopped = kernel_log()
        (args.output / "kernel.after-stop.log").write_text(kernel_stopped)
        shutdown_fault = status == "PASS" and bool(
            KERNEL_FAULT.search(
                kernel_stopped.rsplit(kernel_before.splitlines()[-1], 1)[-1]
            )
        )
        final_status = "FAIL" if shutdown_fault else status
        if cleanup == "workers remain":
            final_status = "FAIL"
        save(args.output / "status.json", {"status": final_status, "cleanup": cleanup})
        save(
            args.output / "qualification.json",
            qualification(events, final_status, cleanup, reference_mode),
        )
        if cleanup == "workers remain":
            raise RuntimeError("GPU workers remain after serving")
        if shutdown_fault:
            raise RuntimeError(
                "GPU/PCIe fault during shutdown; attribution requires investigation"
            )


if __name__ == "__main__":
    main()
