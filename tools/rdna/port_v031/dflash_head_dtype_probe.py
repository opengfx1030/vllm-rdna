# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare current and proposed DFlash2 shared-head projections on one TP rank.

This is an offline diagnostic, not a serving-quality or acceptance test. Inputs
are seeded synthetic activations passed through the draft's final RMSNorm.
The target checkpoint stores a BF16 head, but the FP16 target loader converts
it to FP16. Current DFlash2 casts that shared FP16 head and hidden activations
back to BF16 on each candidate call; the proposed path uses the FP16 runtime
head directly. This probe models that loader and candidate casting sequence.
Run once per physical GPU/rank with its matching TunableOp CSV; the script never
tunes and does not modify the checkpoint or shared profile.

Example (after setting the same ROCm library paths as the recipe):

  HIP_VISIBLE_DEVICES=6 python tools/rdna/port_v031/dflash_head_dtype_probe.py \
    --draft-path /path/to/Qwen3.8-Flash-Next-DFlash2 \
    --target-path /path/to/Qwen3.8-Flash-Next-AWQ-W4A16 --rank 0 \
    --rows tunableop/rocm7.14-rocblas5.5/tunableop_results0.csv \
    --out /path/to/head-rank0.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def _find_tensor(model_path: Path, name: str):
    from safetensors import safe_open

    index = model_path / "model.safetensors.index.json"
    if index.exists():
        manifest = json.loads(index.read_text())
        shard = manifest["weight_map"].get(name)
        if shard is None:
            raise KeyError(f"{name!r} absent from {index}")
        files = [model_path / shard]
    else:
        files = sorted(model_path.glob("*.safetensors"))
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            # safetensors.safe_open exposes keys(), not a mapping __contains__.
            if name in reader.keys():  # noqa: SIM118
                return reader.get_tensor(name), str(path)
    raise KeyError(f"{name!r} absent from {model_path} safetensors")


def _finite_range(tensor):
    import torch

    finite = torch.isfinite(tensor)
    values = tensor[finite]
    return {
        "count": tensor.numel(),
        "nonfinite": int((~finite).sum().item()),
        "min": float(values.min().item()) if values.numel() else None,
        "max": float(values.max().item()) if values.numel() else None,
        "max_abs": float(values.abs().max().item()) if values.numel() else None,
    }


def _time_call(call, warmup: int, reps: int):
    import torch

    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        call()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return {"median_ms": statistics.median(times), "min_ms": min(times)}


def _topk_comparison(logits_a, logits_b, k: int):
    import torch

    ids_a = torch.topk(logits_a.float(), k, dim=-1).indices.cpu()
    ids_b = torch.topk(logits_b.float(), k, dim=-1).indices.cpu()
    same_rows = (ids_a == ids_b).all(dim=-1)
    overlap = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids_a, ids_b)]
    return {
        "exact_order_rows": int(same_rows.sum().item()),
        "row_count": ids_a.shape[0],
        "mean_set_overlap": statistics.mean(overlap),
        "min_set_overlap": min(overlap),
        "top1_matches": int((ids_a[:, 0] == ids_b[:, 0]).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.rank < args.tp_size:
        parser.error("rank must be in [0, tp-size)")
    if not args.rows.is_file():
        parser.error(f"TunableOp rows not found: {args.rows}")
    if args.activation_scale <= 0 or args.reps <= 0 or args.warmup < 0:
        parser.error("activation-scale and reps must be positive; warmup nonnegative")

    # Match the recipe's rocBLAS lookup-only mode before importing torch.
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(args.rows.resolve())

    import torch
    import torch.cuda.tunable as tunable
    import torch.nn.functional as F

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a ROCm GPU")
    torch.accelerator.set_device_index(args.device)
    tunable.set_filename(str(args.rows.resolve()), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(False)

    config = json.loads((args.draft_path / "config.json").read_text())
    if config.get("has_own_lm_head", False):
        raise ValueError("This diagnostic expects DFlash2 to share the target head")
    head_cpu, head_file = _find_tensor(args.target_path, "lm_head.weight")
    norm_cpu, norm_file = _find_tensor(args.draft_path, "norm.weight")
    if head_cpu.dtype != torch.bfloat16:
        raise ValueError(f"Expected BF16 checkpoint head, got {head_cpu.dtype}")
    if head_cpu.ndim != 2 or head_cpu.shape[0] % args.tp_size:
        raise ValueError(f"Unexpected shared-head shape: {tuple(head_cpu.shape)}")
    if norm_cpu.numel() != head_cpu.shape[1]:
        raise ValueError("Final norm width differs from shared-head input width")
    local_n = head_cpu.shape[0] // args.tp_size
    full_shape = list(head_cpu.shape)
    local_cpu = head_cpu.narrow(0, args.rank * local_n, local_n).contiguous()
    checkpoint_weight_range = _finite_range(local_cpu)
    norm_range = _finite_range(norm_cpu)
    roundtrip = local_cpu.to(torch.float16).to(torch.bfloat16)
    weight_changed = int((roundtrip != local_cpu).sum().item())
    weight_overflow = int(
        (local_cpu.abs() > torch.finfo(torch.float16).max).sum().item()
    )
    # The target's model_config.dtype is FP16, so its weight loader casts the
    # checkpoint before `load_dflash_model` aliases this head into DFlash2.
    weight_fp16 = local_cpu.to(device=f"cuda:{args.device}", dtype=torch.float16)
    weight_bf16 = weight_fp16.to(torch.bfloat16)
    runtime_weight_range = _finite_range(weight_fp16)
    norm_weight = norm_cpu.to(device=f"cuda:{args.device}")
    del head_cpu, local_cpu, roundtrip

    top_k = int(config.get("dflash_config", {}).get("selector_top_k", 8))
    eps = float(config.get("rms_norm_eps", 1e-6))
    generator = torch.Generator(device=f"cuda:{args.device}")
    generator.manual_seed(args.seed)
    cells = []
    for m in args.m:
        raw = (
            torch.randn(
                (m, weight_bf16.shape[1]),
                device=weight_bf16.device,
                generator=generator,
            )
            * args.activation_scale
        ).to(torch.bfloat16)
        final_hidden_bf16 = F.rms_norm(raw, (raw.shape[-1],), norm_weight, eps)
        # compute_candidates casts to the actual shared head dtype first;
        # current LogitsProcessor then casts back to its draft BF16 head_dtype.
        hidden_fp16 = final_hidden_bf16.to(torch.float16)
        hidden_bf16 = hidden_fp16.to(torch.bfloat16)
        bf16_logits = F.linear(hidden_bf16, weight_bf16)
        # UnquantizedEmbeddingMethod.apply follows this dispatch on ROCm.
        # On gfx1030 the default skinny-GEMM setting selects gemv_f16_rdna2
        # for M<=8 and rocBLAS for larger M; F.linear alone misses that path.
        fp16_logits = rocm_unquantized_gemm_impl(hidden_fp16, weight_fp16)
        fp16_rocblas_logits = F.linear(hidden_fp16, weight_fp16)
        reference = F.linear(hidden_fp16.float(), weight_fp16.float())
        ref = reference.float()
        bf = bf16_logits.float()
        fp = fp16_logits.float()
        k = min(top_k, local_n)
        cells.append(
            {
                "m": m,
                "final_hidden_bf16": _finite_range(final_hidden_bf16),
                "hidden_bf16": _finite_range(hidden_bf16),
                "bf16_logits": _finite_range(bf16_logits),
                "fp16_logits": _finite_range(fp16_logits),
                "fp16_dispatch_vs_rocblas_max_abs_error": float(
                    (fp - fp16_rocblas_logits.float()).abs().max().item()
                ),
                "bf16_vs_fp16": {
                    "max_abs_error": float((bf - fp).abs().max().item()),
                    "mean_abs_error": float((bf - fp).abs().mean().item()),
                    "topk": _topk_comparison(bf, fp, k),
                },
                "bf16_vs_fp32": {
                    "mean_abs_error": float((bf - ref).abs().mean().item()),
                    "topk": _topk_comparison(bf, ref, k),
                },
                "fp16_vs_fp32": {
                    "mean_abs_error": float((fp - ref).abs().mean().item()),
                    "topk": _topk_comparison(fp, ref, k),
                },
                "current_bf16_recast_timing": _time_call(
                    lambda hidden_fp16=hidden_fp16, weight_fp16=weight_fp16: F.linear(
                        hidden_fp16.to(torch.bfloat16),
                        weight_fp16.to(torch.bfloat16),
                    ),
                    args.warmup,
                    args.reps,
                ),
                "bf16_cached_weight_timing": _time_call(
                    lambda hidden_bf16=hidden_bf16, weight_bf16=weight_bf16: F.linear(
                        hidden_bf16, weight_bf16
                    ),
                    args.warmup,
                    args.reps,
                ),
                "proposed_fp16_dispatch_timing": _time_call(
                    lambda hidden_fp16=hidden_fp16, weight_fp16=weight_fp16: (
                        rocm_unquantized_gemm_impl(hidden_fp16, weight_fp16)
                    ),
                    args.warmup,
                    args.reps,
                ),
                "fp16_rocblas_timing": _time_call(
                    lambda hidden_fp16=hidden_fp16, weight_fp16=weight_fp16: F.linear(
                        hidden_fp16, weight_fp16
                    ),
                    args.warmup,
                    args.reps,
                ),
            }
        )

    result = {
        "kind": "synthetic_normalized_activation_shared_head_runtime_cast_probe",
        "serving_quality_test": False,
        "modelled_runtime": "target BF16 checkpoint -> FP16 loader -> shared head; "
        "draft BF16 final norm -> FP16 compute_candidates cast -> "
        "BF16 processor recast (current) or FP16 direct head (proposed)",
        "target_checkpoint": str(args.target_path),
        "draft_checkpoint": str(args.draft_path),
        "head_file": head_file,
        "norm_file": norm_file,
        "rank": args.rank,
        "tp_size": args.tp_size,
        "gpu": torch.cuda.get_device_name(args.device),
        "rows": str(args.rows.resolve()),
        "head_full_shape": full_shape,
        "head_local_shape": list(weight_bf16.shape),
        "checkpoint_weight_range": checkpoint_weight_range,
        "runtime_fp16_weight_range": runtime_weight_range,
        "norm_range": norm_range,
        "checkpoint_bf16_weight_values_changed_by_runtime_fp16_roundtrip": (
            weight_changed
        ),
        "checkpoint_bf16_weight_values_over_fp16_max": weight_overflow,
        "seed": args.seed,
        "activation_scale": args.activation_scale,
        "top_k": top_k,
        "fp16_dispatch": "rocm_unquantized_gemm_impl (recipe default uses "
        "gemv_f16_rdna2 at M<=8, rocBLAS above)",
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rank": args.rank,
                "head_local_shape": result["head_local_shape"],
                "weight_changed": weight_changed,
                "cells": [
                    {
                        "m": c["m"],
                        "current_bf16_recast_ms": c["current_bf16_recast_timing"][
                            "median_ms"
                        ],
                        "fp16_dispatch_ms": c["proposed_fp16_dispatch_timing"][
                            "median_ms"
                        ],
                        "topk_exact": c["bf16_vs_fp16"]["topk"]["exact_order_rows"],
                    }
                    for c in cells
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
