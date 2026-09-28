# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V620 harness for the W4A8 sdot4 explore: gates G0, G2 and G3.

Needs a gfx1030 GPU, a ROCm torch, hipcc, and a vLLM build with the RDNA2
W4A16 ops (the baseline). The explore kernels are compiled into a
standalone .so on first use (see lib.py); nothing is registered with vLLM.

    python -m benchmarks.kernels.w4a8_sdot4_explore.bench peak
    python -m benchmarks.kernels.w4a8_sdot4_explore.bench check --quick
    python -m benchmarks.kernels.w4a8_sdot4_explore.bench bench --cells prefill
    python -m benchmarks.kernels.w4a8_sdot4_explore.bench all --json w4a8.json

Each subcommand prints markdown rows for docs/explore/w4a8-sdot4/TESTPLAN.md
and, with --json, writes machine-readable records.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.kernels.w4a8_sdot4_explore import reference as ref
from benchmarks.kernels.w4a8_sdot4_explore.lib import (
    PROBE_MACS,
    LdsTooBig,
    W4A8Lib,
    load,
)

DEVICE = "cuda"
# G3 output check: W4A8 vs the W4A16 op on random data is ~3e-2 (mostly the
# W4A16 baked bias); garbage is ~1. A timing without it proves nothing
# (JartX: a corrupted engine benchmarked faster).
SANITY_REL_L2 = 0.1

# (M, N, K, note). N/K are per-rank Qwen3.8-27B-AWQ FFN shapes at TP=2 plus
# the 2026-09-10 microbench cells (docs/profiling/...-prefill-microbench.md).
PREFILL_CELLS = [
    (32, 2560, 8704, "down-proj class"),
    (96, 2560, 8704, "ConfigA >=96 band"),
    (128, 6144, 2560, "microbench M=128"),
    (256, 6144, 2560, "M=256 boundary"),
    (624, 1024, 2560, "small-N"),
    (624, 6144, 2560, "microbench mid-M"),
    (624, 8704, 2560, "TP=2 intermediate"),
    (624, 12288, 2560, "microbench high-N"),
    (1856, 6144, 2560, "large-M profile band"),
    (2048, 2560, 8704, "full-chunk down"),
    (2048, 6144, 2560, "microbench M=2048"),
    (2048, 8704, 2560, "full-chunk intermediate"),
]
K_SWEEP_CELLS = [(624, 6144, k, "K sweep") for k in (1024, 2560, 4096, 5120, 8704)]
DECODE_CELLS = [
    (m, n, k, "decode skinny")
    for m in (1, 2, 4)
    for n, k in ((2560, 8704), (6144, 2560))
]
# Tails: partial M tiles, partial N tiles, N not a multiple of N_TILE.
EDGE_CELLS = [
    (1, 8, 256, "M=1 N=8"),
    (15, 24, 256, "partial tiles"),
    (17, 1032, 512, "N_TILE + 8"),
    (33, 520, 1024, "odd M, N % 256 != 0"),
    (64, 2048, 384, "K = 3 groups of 128"),
]
QUICK_PROD = [PREFILL_CELLS[2], PREFILL_CELLS[5], PREFILL_CELLS[9]]
CELL_SETS = {
    "prefill": PREFILL_CELLS,
    "k-sweep": K_SWEEP_CELLS,
    "decode": DECODE_CELLS,
    "edge": EDGE_CELLS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ops():
    from vllm import _custom_ops as ops

    return ops


def time_us(
    fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 20,
    flush: Callable[[], Any] | None = None,
) -> float:
    """Median device time of ``fn`` in microseconds (events per iteration)."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        if flush is not None:
            flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a64 - b64) / max(np.linalg.norm(b64), 1e-30))


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class DeviceProblem:
    """A reference Problem uploaded in the layouts the kernels read."""

    def __init__(self, p: ref.Problem, ops):
        self.p = p
        self.x = torch.from_numpy(p.x).to(DEVICE)
        w = torch.from_numpy(ref.pack_k_major(p.q_kn).view(np.int32)).to(DEVICE)
        self.g_idx = torch.empty(0, dtype=torch.int, device=DEVICE)
        ops.gptq_shuffle(w, self.g_idx, 4)  # the production RDNA2 layout
        self.w = w
        self.qzeros = torch.from_numpy(p.qzeros.view(np.int32)).to(DEVICE)
        self.scales = torch.from_numpy(p.scales_gn).to(DEVICE)
        # AWQ (uint4) stores literal zeros: use_v2_format=True, zero_offset=0.
        self.use_v2_format = p.zero_offset == 0

    def act_buffers(
        self, m_tile: int, per_group: bool = False
    ) -> tuple[torch.Tensor, ...]:
        m, k = self.p.x.shape
        rows = -(-m // m_tile) * m_tile
        groups = k // self.p.group_size
        return (
            torch.empty(rows * k, dtype=torch.int8, device=DEVICE),
            torch.empty(
                rows * groups if per_group else m, dtype=torch.float32, device=DEVICE
            ),
            torch.empty(rows * groups, dtype=torch.int32, device=DEVICE),
        )

    def w4a16_kernel(self, baseline: str) -> str:
        """Which W4A16 op to compare against; "auto" asks the production
        selector (GPTQ M > 256 goes to exllama, not ConfigA)."""
        if baseline != "auto":
            return baseline
        from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (
            _rdna2_w4a16_select_kernel,
        )

        m, k = self.p.x.shape
        n = self.p.q_kn.shape[1]
        return _rdna2_w4a16_select_kernel(m, k, n, is_awq=self.use_v2_format)

    def w4a16(self, ops, kernel: str = "prefill") -> torch.Tensor:
        args = (self.x, self.w, self.qzeros, self.scales, self.g_idx)
        if kernel == "prefill":
            out = ops.gptq_gemm_rdna2_prefill(*args, self.use_v2_format)
        elif kernel == "exllama":
            out = ops.gptq_gemm(*args, True, self.use_v2_format, 4)
        elif kernel == "rdna2_decode":
            out = ops.gptq_gemm_rdna2(*args, self.use_v2_format)
        else:
            raise ValueError(f"unknown W4A16 kernel {kernel}")
        # The RDNA2 ops return persistent buffers: clone before the next call.
        return out.clone()


def select_configs(lib: W4A8Lib, names: str | None) -> list:
    """Comma list of config names or ids; "ag" adds every per-(token, group)
    config, the only kind that passed G1 so far."""
    if not names:
        return lib.configs
    configs = []
    for n in names.split(","):
        if n == "ag":
            configs += [c for c in lib.configs if c.a_group]
        else:
            configs.append(lib.config(int(n) if n.isdigit() else n))
    return configs


def print_table(title: str, header: list[str], rows: list[list[Any]]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


# ---------------------------------------------------------------------------
# G0: issue-rate probe
# ---------------------------------------------------------------------------


def cmd_peak(lib: W4A8Lib, args) -> list[dict]:
    props = torch.cuda.get_device_properties(0)
    mps = props.multi_processor_count
    blocks = mps * 16
    out = torch.zeros(blocks * 256, dtype=torch.int32, device=DEVICE)
    records = []
    for kind in ("fma_f32", "fdot2", "sdot4"):
        t = time_us(
            lambda kind=kind: lib.probe(kind, blocks, args.probe_iters, out),
            warmup=2,
            iters=args.repeat,
        )
        lane_instr = blocks * 256 * args.probe_iters * lib.probe_chains
        records.append(
            {
                "gate": "G0",
                "kind": kind,
                "us": t,
                "lane_instr_per_s": lane_instr / (t * 1e-6),
                "mac_per_s": lane_instr * PROBE_MACS[kind] / (t * 1e-6),
            }
        )
    by = {r["kind"]: r for r in records}
    ratio = by["sdot4"]["mac_per_s"] / by["fdot2"]["mac_per_s"]
    # multi_processor_count reports WGPs on RDNA (4 SIMD32 = 128 lanes each);
    # cross-check the implied clock against amd-smi before trusting it.
    clock = by["fma_f32"]["lane_instr_per_s"] / (mps * 128) / 1e9
    for r in records:
        r.update(sdot4_over_fdot2=ratio, implied_ghz=clock, device=props.name)
    print_table(
        f"G0 peak probe ({props.name}, {mps} MPs, {args.probe_iters} iters)",
        ["kind", "µs", "T lane-instr/s", "T MAC/s"],
        [
            [
                r["kind"],
                f"{r['us']:.1f}",
                f"{r['lane_instr_per_s'] / 1e12:.2f}",
                f"{r['mac_per_s'] / 1e12:.2f}",
            ]
            for r in records
        ],
    )
    verdict = "PASS" if ratio >= args.g0_min_ratio else "FAIL"
    print(
        f"\nsdot4/fdot2 MAC ratio = {ratio:.2f} (need >= {args.g0_min_ratio}): "
        f"**{verdict}**; implied clock {clock:.2f} GHz (if MP = WGP)"
    )
    return records


# ---------------------------------------------------------------------------
# G2: correctness
# ---------------------------------------------------------------------------


def check_layout_and_split(lib: W4A8Lib, ops) -> list[dict]:
    """gptq_shuffle == reference shuffle; C and Python split-K rules agree."""
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 2**32, size=(64, 256), dtype=np.uint64).astype(np.uint32)
    want = ref.exllama_shuffle(packed)
    w = torch.from_numpy(packed.view(np.int32).copy()).to(DEVICE)
    ops.gptq_shuffle(w, torch.empty(0, dtype=torch.int, device=DEVICE), 4)
    shuffle_ok = bool(np.array_equal(to_np(w).view(np.uint32), want))
    mismatches = []
    for cfg in lib.configs:
        for m, n, k, _ in PREFILL_CELLS + K_SWEEP_CELLS + DECODE_CELLS:
            for g in ref.SUPPORTED_GROUP_SIZES:
                if k % g:
                    continue
                c = lib.pick_split_k(m, n, k, g, cfg.id)
                py = ref.pick_split_k(
                    m, n, k, g, cfg.m_tile, cfg.n_tile, a_group=cfg.a_group
                )
                if c != py:
                    mismatches.append((cfg.name, m, n, k, g, c, py))
    print(f"\ngptq_shuffle matches reference.exllama_shuffle: {shuffle_ok}")
    print(f"split-K C/Python mismatches: {mismatches or 'none'}")
    return [
        {
            "gate": "G2",
            "check": "layout",
            "shuffle_ok": shuffle_ok,
            "split_mismatches": mismatches,
        }
    ]


def check_cell(
    lib: W4A8Lib,
    ops,
    cell,
    weight_type: str,
    group_size: int,
    configs,
    seed: int,
    baseline: str = "prefill",
) -> list[dict]:
    m, n, k, note = cell
    p = ref.make_problem(m, n, k, group_size, weight_type, seed=seed)
    wargs = (p.q_kn, p.zeros_eff_gn, p.scales_gn, group_size)
    acts = {pg: ref.quantize_act(p.x, group_size, pg) for pg in (False, True)}
    orcs = {pg: ref.oracle_w4a8(a, *wargs) for pg, a in acts.items()}
    act, orc = acts[False], orcs[False]
    dp = DeviceProblem(p, ops)
    groups = k // group_size

    w4a16_ref = ref.w4a16_reference(p.x, *wargs)
    # The gfx1030 W4A16 dequant bakes s*(-1024 - z) into fp16 (see
    # PRIOR-ART-RDNA3.md); this emulates it bit for bit.
    baked_ref = p.x.astype(np.float64) @ ref.w4a16_rdna2_weights(*wargs).astype(
        np.float64
    )
    base = {
        "gate": "G2",
        "m": m,
        "n": n,
        "k": k,
        "note": note,
        "weight_type": weight_type,
        "group_size": group_size,
    }
    kernel = dp.w4a16_kernel(baseline)
    w4a16_out = to_np(dp.w4a16(ops, kernel))
    records = [
        {
            **base,
            "check": "w4a16_baseline",
            "w4a16_kernel": kernel,
            "w4a16_rel_l2": rel_l2(w4a16_out, w4a16_ref),
            "w4a16_rel_l2_vs_baked": rel_l2(w4a16_out, baked_ref),
            "baked_model_rel_l2": rel_l2(baked_ref, w4a16_ref),
            "a8_rel_l2_vs_w4a16_ref": rel_l2(orc.c, w4a16_ref),
            "a8_group_rel_l2_vs_w4a16_ref": rel_l2(orcs[True].c, w4a16_ref),
        }
    ]
    try:
        q_v, s_v, _ = ops.scaled_int8_quant(dp.x)
        records[0]["vllm_int8_quant_match"] = bool(
            np.array_equal(to_np(q_v), act.a_i8)
            and np.array_equal(to_np(s_v).ravel(), act.scale)
        )
    except (AttributeError, RuntimeError) as e:  # op not built on this box
        records[0]["vllm_int8_quant_match"] = f"n/a ({type(e).__name__})"

    for cfg in configs:
        act, orc = acts[cfg.a_group], orcs[cfg.a_group]
        a, a_scale, asum = dp.act_buffers(cfg.m_tile, cfg.a_group)
        lib.act_quant(dp.x, a, a_scale, asum, group_size, cfg.m_tile, cfg.a_group)
        a_ok = np.array_equal(to_np(a), ref.tile_a(act.a_perm, cfg.m_tile))
        want_scale = (
            ref.tile_groups(act.scale, cfg.m_tile) if cfg.a_group else act.scale
        )
        s_ok = np.array_equal(to_np(a_scale), want_scale)
        sum_ok = np.array_equal(to_np(asum), ref.tile_groups(act.asum, cfg.m_tile))

        out32 = torch.empty(m, n, dtype=torch.float32, device=DEVICE)
        # f32 output needs split 1, i.e. the whole K range of A in LDS; where
        # that exceeds 64 KiB the config is checked through the fp16 path only.
        f32_ratio, f32_ok = float("nan"), True
        try:
            lib.gemm(
                a,
                dp.w,
                dp.qzeros,
                dp.scales,
                a_scale,
                asum,
                out32,
                k,
                group_size,
                p.zero_offset,
                cfg.id,
                split_k=1,
            )
            err32 = np.abs(to_np(out32) - orc.c)
            bound32 = ref.f32_flush_bound(orc.mag, groups, cfg.a_group)
            f32_ratio = float((err32 / bound32).max())
            f32_ok = bool((err32 <= bound32).all())
        except LdsTooBig:
            pass

        split = lib.pick_split_k(m, n, k, group_size, cfg.id)
        runs = []
        try:
            for _ in range(2):  # pk4 CAS atomics make split_k > 1 order-dependent
                out16 = torch.empty(m, n, dtype=torch.float16, device=DEVICE)
                lib.gemm(
                    a,
                    dp.w,
                    dp.qzeros,
                    dp.scales,
                    a_scale,
                    asum,
                    out16,
                    k,
                    group_size,
                    p.zero_offset,
                    cfg.id,
                    split_k=split,
                )
                runs.append(to_np(out16))
        except LdsTooBig as e:  # no group-aligned split fits: ineligible
            records.append(
                {
                    **base,
                    "check": "w4a8",
                    "config": cfg.name,
                    "split_k": split,
                    "skipped": str(e),
                    "pass": None,
                }
            )
            continue
        c16 = runs[0].astype(np.float64)
        err16 = np.abs(c16 - orc.c)
        bound16 = ref.f16_output_bound(orc.mag, groups, split, cfg.a_group)
        ok = bool(
            a_ok
            and s_ok
            and sum_ok
            and f32_ok
            and (err16 <= bound16).all()
            and np.isfinite(c16).all()
        )
        records.append(
            {
                **base,
                "check": "w4a8",
                "config": cfg.name,
                "split_k": split,
                "act_quant_exact": bool(a_ok and s_ok and sum_ok),
                "f32_checked": not np.isnan(f32_ratio),
                "f32_worst_err_over_bound": f32_ratio,
                "f16_worst_err_over_bound": float((err16 / bound16).max()),
                "f16_rel_l2": rel_l2(c16, orc.c),
                "f16_repeatable": bool(np.array_equal(runs[0], runs[1])),
                "pass": ok,
            }
        )
    return records


def cmd_check(lib: W4A8Lib, args) -> list[dict]:
    ops = _ops()
    records = check_layout_and_split(lib, ops)
    cells = EDGE_CELLS + (QUICK_PROD if args.quick else PREFILL_CELLS + DECODE_CELLS)
    groups = (32, 128) if args.quick else ref.SUPPORTED_GROUP_SIZES
    configs = select_configs(lib, args.configs)
    for cell in cells:
        for wt in ref.ELIGIBLE_WEIGHT_TYPES:
            for g in groups:
                if cell[2] % g:
                    continue
                label = f"{cell[0]}x{cell[1]}x{cell[2]} {wt} G={g}"
                print(f"[check] {label}", flush=True)
                try:
                    records += check_cell(
                        lib, ops, cell, wt, g, configs, args.seed, args.baseline
                    )
                except RuntimeError as e:  # LDS-ineligible configs skip inside
                    print(f"[check] ERROR {label}: {e}", flush=True)
                    records.append(
                        {
                            "check": "w4a8",
                            "m": cell[0],
                            "n": cell[1],
                            "k": cell[2],
                            "weight_type": wt,
                            "group_size": g,
                            "config": "(cell aborted)",
                            "error": str(e),
                            "pass": False,
                        }
                    )

    def row(r: dict) -> list:
        head = [
            f"{r['m']}x{r['n']}x{r['k']}",
            r["weight_type"],
            r["group_size"],
            r["config"],
            r.get("split_k", "-"),
        ]
        if "act_quant_exact" not in r:  # skipped for LDS, or an aborted cell
            verdict = "skip (LDS)" if r["pass"] is None else "**ERROR**"
            return head + ["-"] * 5 + [verdict]
        return head + [
            "yes" if r["act_quant_exact"] else "NO",
            f"{r['f32_worst_err_over_bound']:.3f}" if r["f32_checked"] else "n/a",
            f"{r['f16_worst_err_over_bound']:.3f}",
            f"{r['f16_rel_l2']:.2e}",
            "yes" if r["f16_repeatable"] else "no",
            "PASS" if r["pass"] else "**FAIL**",
        ]

    w4a8 = [r for r in records if r.get("check") == "w4a8"]
    rows = [row(r) for r in w4a8]
    print_table(
        "G2 correctness (err/bound <= 1 passes)",
        [
            "M×N×K",
            "fmt",
            "G",
            "config",
            "split",
            "act-quant exact",
            "f32 err/bound (n/a: split 1 exceeds LDS)",
            "f16 err/bound",
            "f16 rel-L2",
            "same bits ×2",
            "verdict",
        ],
        rows,
    )
    base_rows = [
        [
            f"{r['m']}x{r['n']}x{r['k']}",
            r["weight_type"],
            r["group_size"],
            r["w4a16_kernel"],
            f"{r['w4a16_rel_l2']:.2e}",
            f"{r['baked_model_rel_l2']:.2e}",
            f"{r['w4a16_rel_l2_vs_baked']:.2e}",
            f"{r['a8_rel_l2_vs_w4a16_ref']:.2e}",
            f"{r['a8_group_rel_l2_vs_w4a16_ref']:.2e}",
            r["vllm_int8_quant_match"],
        ]
        for r in records
        if r.get("check") == "w4a16_baseline"
    ]
    print_table(
        "Baseline sanity and A8 error on random data (all rel-L2)",
        [
            "M×N×K",
            "fmt",
            "G",
            "W4A16 op",
            "W4A16 op vs exact",
            "baked emu vs exact",
            "W4A16 op vs baked emu",
            "W4A8 per-token vs exact",
            "W4A8 per-group vs exact",
            "vLLM int8 quant == ref",
        ],
        base_rows,
    )
    passed = sum(r["pass"] is True for r in w4a8)
    skipped = sum(r["pass"] is None for r in w4a8)
    print(
        f"\nG2: {passed}/{len(w4a8) - skipped} pass, {skipped} (config, cell) "
        "pairs skipped because no group-aligned split fits 64 KiB of LDS"
    )
    return records


# ---------------------------------------------------------------------------
# G3: timing vs the production W4A16 path
# ---------------------------------------------------------------------------


def cmd_bench(lib: W4A8Lib, args) -> list[dict]:
    ops = _ops()
    configs = select_configs(lib, args.configs)
    flush = None
    if args.cold:
        scratch = torch.empty(256 * 2**20, dtype=torch.uint8, device=DEVICE)
        flush = scratch.zero_  # evict L2 and the 128 MiB Infinity Cache
    records = []
    cells = [c for name in args.cells.split(",") for c in CELL_SETS[name]]
    for m, n, k, note in cells:
        if k % args.group_size:
            continue
        p = ref.make_problem(m, n, k, args.group_size, args.weight_type, seed=args.seed)
        dp = DeviceProblem(p, ops)
        kernel = dp.w4a16_kernel(args.baseline)
        out16 = dp.w4a16(ops, kernel).float()
        t16 = time_us(
            lambda dp=dp, kernel=kernel: dp.w4a16(ops, kernel),
            args.warmup,
            args.iters,
            flush,
        )
        for cfg in configs:
            a, a_scale, asum = dp.act_buffers(cfg.m_tile, cfg.a_group)
            out = torch.empty(m, n, dtype=torch.float16, device=DEVICE)
            split = args.split_k or lib.pick_split_k(m, n, k, args.group_size, cfg.id)
            skip = {
                "gate": "G3",
                "m": m,
                "n": n,
                "k": k,
                "note": note,
                "group_size": args.group_size,
                "weight_type": args.weight_type,
                "config": cfg.name,
                "split_k": split,
            }
            if (k // args.group_size) % split:
                records.append({**skip, "skipped": "split does not divide K/G"})
                continue

            def quant(dp=dp, a=a, a_scale=a_scale, asum=asum, cfg=cfg):
                lib.act_quant(
                    dp.x, a, a_scale, asum, args.group_size, cfg.m_tile, cfg.a_group
                )

            def gemm(
                dp=dp,
                a=a,
                a_scale=a_scale,
                asum=asum,
                out=out,
                cfg=cfg,
                k=k,
                split=split,
            ):
                lib.gemm(
                    a,
                    dp.w,
                    dp.qzeros,
                    dp.scales,
                    a_scale,
                    asum,
                    out,
                    k,
                    args.group_size,
                    dp.p.zero_offset,
                    cfg.id,
                    split,
                )

            quant()
            try:  # one untimed launch: the split must fit 64 KiB of LDS
                gemm()
            except LdsTooBig as e:
                records.append({**skip, "skipped": str(e)})
                continue
            t_q = time_us(quant, args.warmup, args.iters, flush)
            t_g = time_us(gemm, args.warmup, args.iters, flush)
            sanity = (
                torch.linalg.vector_norm(out.float() - out16)
                / torch.linalg.vector_norm(out16).clamp_min(1e-30)
            ).item()
            flops = 2.0 * m * n * k
            records.append(
                {
                    "gate": "G3",
                    "m": m,
                    "n": n,
                    "k": k,
                    "note": note,
                    "group_size": args.group_size,
                    "weight_type": args.weight_type,
                    "config": cfg.name,
                    "split_k": split,
                    "cold": args.cold,
                    "w4a16_kernel": kernel,
                    "w4a16_us": t16,
                    "act_quant_us": t_q,
                    "w4a8_gemm_us": t_g,
                    "speedup_gemm": t16 / t_g,
                    "speedup_total": t16 / (t_q + t_g),
                    "w4a8_tops": flops / (t_g * 1e-6) / 1e12,
                    "w4a16_tflops": flops / (t16 * 1e-6) / 1e12,
                    "rel_l2_vs_w4a16": sanity,
                    "pass": sanity < SANITY_REL_L2,
                }
            )
    rows = [
        [
            f"{r['m']}x{r['n']}x{r['k']}",
            r["config"],
            r["split_k"],
            r["w4a16_kernel"],
            f"{r['w4a16_us']:.1f}",
            f"{r['act_quant_us']:.1f}",
            f"{r['w4a8_gemm_us']:.1f}",
            f"{r['speedup_gemm']:.2f}",
            f"{r['speedup_total']:.2f}",
            f"{r['w4a16_tflops']:.1f}",
            f"{r['w4a8_tops']:.1f}",
            f"{r['rel_l2_vs_w4a16']:.1e}" + ("" if r["pass"] else " **BAD**"),
        ]
        for r in records
        if "skipped" not in r
    ]
    temp = "cold" if args.cold else "hot"
    print_table(
        f"G3 timing, G={args.group_size} {args.weight_type}, {temp} caches, "
        f"median of {args.iters}",
        [
            "M×N×K",
            "config",
            "split",
            "W4A16 op",
            "W4A16 µs",
            "act-quant µs",
            "W4A8 GEMM µs",
            "× GEMM",
            "× total",
            "W4A16 TFLOP/s",
            "W4A8 TOP/s",
            "rel-L2 vs W4A16",
        ],
        rows,
    )
    for r in records:
        if "skipped" in r:
            print(
                f"skipped {r['config']} {r['m']}x{r['n']}x{r['k']} "
                f"split={r['split_k']}: {r['skipped']}"
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["peak", "check", "bench", "all"])
    parser.add_argument("--json", type=Path, help="write records here")
    parser.add_argument("--hipcc", help="hipcc matching torch's ROCm")
    parser.add_argument(
        "--configs", help="comma list of config names or ids; ag = all *_ag"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="check: fewer cells")
    parser.add_argument("--cells", default="prefill", help=",".join(CELL_SETS))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--weight-type", default="uint4", choices=ref.ELIGIBLE_WEIGHT_TYPES
    )
    parser.add_argument("--cold", action="store_true", help="flush caches per iter")
    parser.add_argument(
        "--split-k",
        type=int,
        default=0,
        help="bench: force this split for every config (0 = pick_split_k)",
    )
    parser.add_argument(
        "--baseline",
        default="auto",
        choices=["auto", "prefill", "exllama", "rdna2_decode"],
        help="W4A16 op to compare with; auto = the production selector",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5, help="peak: timed runs")
    parser.add_argument("--probe-iters", type=int, default=20000)
    parser.add_argument("--g0-min-ratio", type=float, default=1.8)
    parser.add_argument(
        "--w4a16-force-config",
        type=int,
        help="VLLM_RDNA2_PREFILL_FORCE_CONFIG for the baseline (1 = ConfigA)",
    )
    args = parser.parse_args()

    if args.w4a16_force_config is not None:  # read once by the C++ dispatcher
        os.environ["VLLM_RDNA2_PREFILL_FORCE_CONFIG"] = str(args.w4a16_force_config)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"device arch {arch}, torch {torch.__version__}, hip {torch.version.hip}")
    lib = load(hipcc=args.hipcc)

    records: list[dict] = []
    if args.command in ("peak", "all"):
        records += cmd_peak(lib, args)
    if args.command in ("check", "all"):
        args.quick = args.quick or args.command == "all"
        records += cmd_check(lib, args)
    if args.command in ("bench", "all"):
        records += cmd_bench(lib, args)
    if args.json:
        args.json.write_text(json.dumps(records, indent=1, default=str))
    failed = [r for r in records if r.get("pass") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
