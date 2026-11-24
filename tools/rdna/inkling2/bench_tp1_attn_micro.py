#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone JIT parity + latency microbench for Inkling-2-Mini TP=1 attention & conv.

Compiles ``csrc/rocm/inkling2_attn_rdna2.cu`` via ``torch.utils.cpp_extension.load``
(default ``-O3`` on ``gfx1030``) without requiring an editable ``_rocm_C`` rebuild.

Verifies numerical parity (``max_abs``, ``rmse``) and benchmarks CUDA-graph latency:
  1. GQA KV-sharing ``i2_decode_fused_gqa_kernel`` (``grid(nkv, Q, R)``, ``NW=8/16``)
     vs legacy ``nh``-grid ``i2_decode_fused_nh_kernel`` and PyTorch fp64 oracle across
     ``q_len = 1``, ``R = 1, 2, 4, 8``, ``ctx = 512, 1024, 16384`` on SWA (24/8, W=512)
     and Full (12/4, W=0, log_scale), plus ``G in {1, 2, 3, 4}``.
  2. Multi-query speculative verify (``q_len = 2, 3, 5, 8``, ``R = 1, 2, 4``):
     1-launch GQA fused verify ``splits=(0, 0)`` vs 3-launch ``(decode+tile+combine)``
     on SWA, and ``skip_decode`` (``tile_impl | 2``) vs dead decode launch on Full.
  3. Mixed batch ``q_lens = [1, 8, 1, 64]`` and ``[1, 3, 1, 8]`` parity.
  4. Multi-step speculative verify + draft-rejection rollback with ``conv_writes_in_op``
     (``ring = 4 + K_max``: ``6`` for ``K=2``, ``11`` for ``K=7``) vs separate
     ``inkling2_conv_state_write`` and dense PyTorch short-conv oracle, plus 48-visit
     latency comparison (eliminating all 192 ``inkling2_conv_state_write`` launches).

Usage (on par1-cs25 V620, e.g. HIP_VISIBLE_DEVICES=1):
    python tools/rdna/inkling2/bench_tp1_attn_micro.py
    python tools/rdna/inkling2/bench_tp1_attn_micro.py --cold --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
from typing import Any

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc" / "rocm" / "inkling2_attn_rdna2.cu"

HEAD_DIM = 128
D_REL = 16
HIDDEN = 2304
SWA_NH, SWA_NKV, SWA_WINDOW, SWA_REL = 24, 8, 512, 512
FULL_NH, FULL_NKV, FULL_REL = 12, 4, 1024
KV_BLOCK = 64
Q_CLIP = 22.627416997969522
REL_CLIP = 8.0
LOG_ALPHA = 0.1
LOG_N_FLOOR = 8192.0
RMS_EPS = 1e-6

# tile_impl bit flags in inkling2_attn_rdna2.cu
TILE_SHIPPED = 0
TILE_PIPE = 1
TILE_SKIP_DECODE = 2
TILE_FORCE_NW8 = 4
TILE_FORCE_NW16 = 8
TILE_LEGACY_NH = 16

_PYBIND_TAIL = """

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inkling2_attention", &inkling2_attention);
  m.def("inkling2_kv_prep", &inkling2_kv_prep);
  m.def("inkling2_sconv_static", &inkling2_sconv_static);
  m.def("inkling2_sconv_dynamic", &inkling2_sconv_dynamic);
  m.def("inkling2_conv_state_write", &inkling2_conv_state_write);
}
"""


def build_jit(archs: str = "gfx1030", opt: str | None = None, verbose: bool = False):
    """JIT-compile ``csrc/rocm/inkling2_attn_rdna2.cu`` via ``cpp_extension.load``."""
    from torch.utils.cpp_extension import load

    opt_flag = opt or os.environ.get("I2_JIT_OPT", "-O3")
    src_text = SOURCE.read_text() + _PYBIND_TAIL
    digest = hashlib.md5("\0".join([src_text, archs, opt_flag]).encode()).hexdigest()[
        :12
    ]
    name = f"i2_tp1_attn_micro_{digest}"
    work = (
        pathlib.Path(
            os.environ.get("I2_JIT_DIR", "~/.cache/vllm_inkling2_jit")
        ).expanduser()
        / name
    )
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(parents=True, exist_ok=True)
    cu = work / "src" / "inkling2_attn_rdna2.cu"
    if not cu.exists() or cu.read_text() != src_text:
        cu.write_text(src_text)
    os.environ["PYTORCH_ROCM_ARCH"] = archs
    arch_flags = [
        f"--offload-arch={a.strip()}"
        for a in archs.replace(",", ";").split(";")
        if a.strip()
    ]
    return load(
        name=name,
        sources=[str(cu)],
        extra_include_paths=[str(REPO / "csrc" / "rocm")],
        extra_cuda_cflags=[opt_flag, *arch_flags],
        build_directory=str(work / "build"),
        verbose=verbose,
    )


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def split_policy_legacy(
    num_reqs: int,
    num_kv_heads: int,
    window: int,
    max_seq_len: int,
    max_query_len: int,
    fused_decode: bool = True,
) -> tuple[int, int]:
    keys = window + 16 if window > 0 else max_seq_len
    groups = max(num_reqs * num_kv_heads, 1)
    splits_decode = (
        0
        if (fused_decode and window > 0)
        else min(max(cdiv(256, groups), 1), max(cdiv(keys, 16) // 2, 1), 256)
    )
    tiles = cdiv(max_query_len, 16)
    ctas = tiles * groups
    splits_tile = 1
    if max_query_len > 1 and ctas < 144:
        key_tiles = cdiv(keys + 16, 64)
        splits_tile = min(max(cdiv(144, ctas), 1), max(key_tiles // 4, 1), 16)
    return splits_decode, splits_tile


def workspace_numel(
    num_reqs: int, num_tokens: int, nh: int, splits_decode: int, splits_tile: int
) -> int:
    rows = splits_decode * num_reqs + splits_tile * num_tokens
    return max(1, rows * nh * 130)


def engine_kv_cache(pages: int, nkv: int, bs: int, g: torch.Generator) -> torch.Tensor:
    """Engine token-major ``[B, bs, H_kv, 256]`` viewed as ``[B, H_kv, bs, 256]``."""
    raw = torch.randn((pages, bs, nkv, 2 * HEAD_DIM), device="cuda", generator=g).half()
    raw[0].fill_(float("nan"))  # block 0 is null_block
    return raw.permute(0, 2, 1, 3)


def scaled_clip_ref(
    x: torch.Tensor,
    scale: torch.Tensor,
    clip: float,
    positions: torch.Tensor,
    log_scale: bool,
) -> torch.Tensor:
    xf = x.float() * scale.float()
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    factor = torch.clamp(clip / (norm + 1e-6), max=1.0)
    y = (xf * factor).to(torch.float16)
    if log_scale:
        tau = 1.0 + LOG_ALPHA * torch.log(
            torch.clamp((positions.float() + 1.0) / LOG_N_FLOOR, min=1.0)
        )
        y = (y.float() * tau[:, None, None]).to(torch.float16)
    return y


@torch.no_grad()
def attention_reference(case: dict[str, Any]) -> torch.Tensor:
    """Exact fp64 softmax / accumulation reference matching Inkling-2 semantics."""
    q_lens: list[int] = case["q_lens"]
    seq_lens: list[int] = case["seq_lens"]
    nh, nkv = case["nh"], case["nkv"]
    groups = nh // nkv
    window = case["window"]
    rel_extent = case["rel_extent"]
    log_scale = case["log_scale"]
    T = sum(q_lens)

    q_prep = scaled_clip_ref(
        case["q"].view(T, nh, HEAD_DIM),
        case["q_scale"],
        Q_CLIP,
        case["positions"],
        log_scale,
    ).double()
    rel_prep = scaled_clip_ref(
        case["rel"].view(T, nh, D_REL),
        case["rel_scale"],
        REL_CLIP,
        case["positions"],
        log_scale,
    ).double()
    proj = case["rel_proj_t"][:rel_extent].double()  # [R_ext, 16]
    rel_logits = torch.einsum("thd,rd->thr", rel_prep, proj)  # [T, nh, R_ext]
    sinks = case["sink"].double()  # [T, nh]
    out = torch.zeros((T, nh, HEAD_DIM), dtype=torch.float64, device="cuda")

    kv = case["kv"]  # [B, nkv, bs, 256]
    bs = kv.shape[2]
    table = case["block_table"]
    qsl = [0]
    for ql in q_lens:
        qsl.append(qsl[-1] + ql)

    scale = HEAD_DIM**-0.5
    for r, (ql, L) in enumerate(zip(q_lens, seq_lens)):
        if ql <= 0:
            continue
        s, e = qsl[r], qsl[r + 1]
        if L <= 0:
            out[s:e].zero_()
            continue
        p0 = L - ql
        lo = max(0, p0 - window + 1) if window > 0 else 0
        pos_k = torch.arange(lo, L, device="cuda")
        blks = table[r, pos_k // bs].long()
        kv_rows = kv[blks, :, pos_k % bs].double()  # [N_k, nkv, 256]
        kk = kv_rows[..., :HEAD_DIM]
        vv = kv_rows[..., HEAD_DIM:]

        qc = q_prep[s:e].view(ql, nkv, groups, HEAD_DIM)
        sc = torch.einsum("chgd,nhd->chgn", qc, kk).reshape(ql, nh, -1) * scale
        pos_q = torch.arange(p0, L, device="cuda")
        dist = pos_q[:, None] - pos_k[None, :]  # [ql, N_k]
        idx = dist.clamp(0, rel_extent - 1)[:, None, :].expand(ql, nh, -1)
        bias = torch.gather(rel_logits[s:e], 2, idx)
        bias = bias.masked_fill(((dist < 0) | (dist >= rel_extent))[:, None, :], 0.0)
        sc = sc + bias
        vis = dist >= 0
        if window > 0:
            vis = vis & (dist < window)
        sc = sc.masked_fill(~vis[:, None, :], float("-inf"))
        logits = torch.cat([sc, sinks[s:e, :, None]], dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., :-1]
        oo = torch.einsum("chgn,nhd->chgd", probs.view(ql, nkv, groups, -1), vv)
        out[s:e] = oo.reshape(ql, nh, HEAD_DIM)

    res = out.float()
    if case["gate"] is not None:
        gate = torch.sigmoid(case["gate"].float()).view(T, nh, HEAD_DIM)
        res = res * gate
    return res.to(torch.float16).reshape(T, nh * HEAD_DIM)


def make_attn_case(
    q_lens: list[int],
    seq_lens: list[int],
    *,
    nh: int,
    nkv: int,
    window: int,
    rel_extent: int,
    log_scale: bool,
    with_gate: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    R = len(q_lens)
    T = sum(q_lens)
    n_blocks_per_req = [max(1, cdiv(max(L, 1), KV_BLOCK)) for L in seq_lens]
    total_pages = sum(n_blocks_per_req) + 2
    kv = engine_kv_cache(total_pages, nkv, KV_BLOCK, g)
    perm = torch.randperm(total_pages - 1, device="cuda", generator=g) + 1
    table = torch.zeros((R, max(n_blocks_per_req)), dtype=torch.int32, device="cuda")
    cursor = 0
    for r, nb in enumerate(n_blocks_per_req):
        if seq_lens[r] > 0:
            table[r, :nb] = perm[cursor : cursor + nb].int()
        cursor += nb

    qsl_list = [0]
    pos_parts = []
    for ql, L in zip(q_lens, seq_lens):
        qsl_list.append(qsl_list[-1] + ql)
        if ql > 0:
            p0 = max(0, L - ql)
            pos_parts.append(
                torch.arange(p0, p0 + ql, device="cuda", dtype=torch.int64)
            )
    positions = (
        torch.cat(pos_parts)
        if pos_parts
        else torch.zeros((0,), device="cuda", dtype=torch.int64)
    )

    def rnd(*shape, scale: float = 1.0, dtype=torch.float16):
        return (torch.randn(shape, device="cuda", generator=g) * scale).to(dtype)

    return {
        "q_lens": q_lens,
        "seq_lens": seq_lens,
        "R": R,
        "T": T,
        "nh": nh,
        "nkv": nkv,
        "window": window,
        "rel_extent": rel_extent,
        "log_scale": log_scale,
        "max_q": max(q_lens) if q_lens else 1,
        "q": rnd(max(T, 1), nh * HEAD_DIM, scale=1.2)[:T],
        "rel": rnd(max(T, 1), nh * D_REL, scale=2.0)[:T],
        "sink": (rnd(max(T, 1), nh) + 3.0)[:T],
        "gate": rnd(max(T, 1), nh * HEAD_DIM, scale=1.5)[:T] if with_gate else None,
        "q_scale": (0.7 + 0.6 * torch.rand((nh, HEAD_DIM), device="cuda", generator=g)),
        "rel_scale": (0.7 + 0.6 * torch.rand((nh, D_REL), device="cuda", generator=g)),
        "rel_proj_t": rnd(rel_extent, D_REL, scale=0.15),
        "positions": positions,
        "kv": kv,
        "block_table": table,
        "seq_lens_t": torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        "qsl_t": torch.tensor(qsl_list, dtype=torch.int32, device="cuda"),
    }


def run_attn_kernel(
    mod,
    case: dict[str, Any],
    splits: tuple[int, int],
    tile_impl: int = 0,
) -> torch.Tensor:
    T, nh = case["T"], case["nh"]
    out = torch.full(
        (max(T, 1), nh * HEAD_DIM), float("nan"), dtype=torch.float16, device="cuda"
    )[:T]
    ws = torch.empty(
        workspace_numel(case["R"], max(T, 1), nh, splits[0], splits[1]),
        dtype=torch.float32,
        device="cuda",
    )
    mod.inkling2_attention(
        case["q"],
        case["rel"],
        case["sink"],
        case["gate"],
        case["q_scale"],
        case["rel_scale"],
        case["rel_proj_t"],
        case["positions"],
        case["kv"],
        case["block_table"],
        case["seq_lens_t"],
        case["qsl_t"],
        out,
        ws,
        case["window"],
        case["rel_extent"],
        case["log_scale"],
        LOG_ALPHA,
        LOG_N_FLOOR,
        Q_CLIP,
        REL_CLIP,
        case["max_q"],
        splits[0],
        splits[1],
        tile_impl,
    )
    return out


def err_stats(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0
    return max_abs, rmse


# ---------------------------------------------------------------------------
# Parity Suites
# ---------------------------------------------------------------------------
def run_parity_suite(mod) -> None:
    print("=== 1. Parity: Decode (q_len=1), GQA NW=8/16 vs Oracle & Legacy ===")
    tol_abs = 2e-3
    for kind, nh, nkv, window, rel_ext, log_sc in (
        ("SWA-512(24/8)", SWA_NH, SWA_NKV, SWA_WINDOW, SWA_REL, False),
        ("SWA-1024(24/8)", SWA_NH, SWA_NKV, 1024, 1024, False),
        ("Full(12/4)", FULL_NH, FULL_NKV, 0, FULL_REL, True),
    ):
        for ctx in (512, 1024, 16384):
            for R in (1, 2, 4, 8):
                seq_lens = [max(1, ctx - r * 7) for r in range(R)]
                case = make_attn_case(
                    [1] * R,
                    seq_lens,
                    nh=nh,
                    nkv=nkv,
                    window=window,
                    rel_extent=rel_ext,
                    log_scale=log_sc,
                    seed=100 + R + ctx,
                )
                ref = attention_reference(case)
                out_nh = run_attn_kernel(mod, case, (0, 1), TILE_LEGACY_NH)
                out_nw16 = run_attn_kernel(mod, case, (0, 0), TILE_FORCE_NW16)
                out_nw8 = run_attn_kernel(mod, case, (0, 0), TILE_FORCE_NW8)

                nh_vs_16_abs, _ = err_stats(out_nw16, out_nh)
                abs16, rmse16 = err_stats(out_nw16, ref)
                abs8, rmse8 = err_stats(out_nw8, ref)
                assert torch.isfinite(out_nw16).all() and torch.isfinite(out_nw8).all()
                assert nh_vs_16_abs == 0.0, (
                    f"{kind} ctx={ctx} R={R}: NW=16 vs nh max_abs={nh_vs_16_abs}"
                )
                assert abs16 <= tol_abs and abs8 <= tol_abs, (
                    f"{kind} ctx={ctx} R={R}: abs16={abs16} abs8={abs8}"
                )
        mx_a = max(abs16, abs8)
        mx_r = max(rmse16, rmse8)
        print(
            f"  [PASS] {kind:15s}: R=1,2,4,8 x ctx=512,1024,16384 "
            f"(NW=16 == nh; max_abs={mx_a:.2e}, rmse={mx_r:.2e})"
        )

    print(
        "=== 2. Numerical Parity: GQA Group Sizes G in {1, 2, 3, 4} & Padded Rows ==="
    )
    for G in (1, 2, 3, 4):
        nkv = 2
        nh = nkv * G
        case = make_attn_case(
            [1, 3, 1, 5],
            [0, 300, 600, 900],
            nh=nh,
            nkv=nkv,
            window=512,
            rel_extent=512,
            log_scale=False,
            seed=40 + G,
        )
        ref = attention_reference(case)
        for nw_flag, label in ((TILE_FORCE_NW8, "NW=8"), (TILE_FORCE_NW16, "NW=16")):
            got = run_attn_kernel(mod, case, (0, 0), nw_flag)
            assert torch.count_nonzero(got[0].float()) == 0, "padded row must be zero"
            mx, rm = err_stats(got, ref)
            assert mx <= tol_abs, f"G={G} {label}: max_abs={mx}"
        print(
            f"  [PASS] G={G} (nh={nh}, nkv={nkv}) mixed+padded: "
            f"max_abs={mx:.2e}, rmse={rm:.2e}"
        )

    print("=== 3. Parity: Multi-Query Verify (q_len=2,3,5,8) & Mixed Batches ===")
    for q_len in (2, 3, 5, 8):
        for R in (1, 2, 4):
            for ctx in (64, 512, 16384):
                seq_lens = [max(q_len, ctx - 13 * r) for r in range(R)]
                # SWA: 1-launch GQA fused (0, 0) vs old (0, S_B) tile + combine & oracle
                case_swa = make_attn_case(
                    [q_len] * R,
                    seq_lens,
                    nh=SWA_NH,
                    nkv=SWA_NKV,
                    window=SWA_WINDOW,
                    rel_extent=SWA_REL,
                    log_scale=False,
                    seed=700 + q_len * 17 + R + ctx,
                )
                ref_swa = attention_reference(case_swa)
                old_splits = split_policy_legacy(R, SWA_NKV, SWA_WINDOW, ctx, q_len)
                out_old = run_attn_kernel(mod, case_swa, old_splits, TILE_SHIPPED)
                out_gqa8 = run_attn_kernel(mod, case_swa, (0, 0), TILE_FORCE_NW8)
                out_gqa16 = run_attn_kernel(mod, case_swa, (0, 0), TILE_FORCE_NW16)
                abs_old, _ = err_stats(out_old, ref_swa)
                abs_g8, rmse_g8 = err_stats(out_gqa8, ref_swa)
                abs_g16, rmse_g16 = err_stats(out_gqa16, ref_swa)
                assert abs_g8 <= tol_abs and abs_g16 <= tol_abs, (
                    f"SWA verify q={q_len} R={R} ctx={ctx}: "
                    f"g8={abs_g8} g16={abs_g16} (old={abs_old})"
                )

                # Full: skip_decode (TILE_PIPE | TILE_SKIP_DECODE) == TILE_PIPE
                case_full = make_attn_case(
                    [q_len] * R,
                    seq_lens,
                    nh=FULL_NH,
                    nkv=FULL_NKV,
                    window=0,
                    rel_extent=FULL_REL,
                    log_scale=True,
                    seed=900 + q_len * 17 + R + ctx,
                )
                ref_full = attention_reference(case_full)
                f_splits = split_policy_legacy(R, FULL_NKV, 0, ctx, q_len)
                out_f_base = run_attn_kernel(mod, case_full, f_splits, TILE_PIPE)
                out_f_skip = run_attn_kernel(
                    mod, case_full, (1, f_splits[1]), TILE_PIPE | TILE_SKIP_DECODE
                )
                diff_skip, _ = err_stats(out_f_skip, out_f_base)
                abs_full, _ = err_stats(out_f_skip, ref_full)
                assert diff_skip == 0.0 and abs_full <= tol_abs
                if ctx <= 1024:
                    out_f_gqa = run_attn_kernel(mod, case_full, (0, 0), TILE_FORCE_NW16)
                    abs_fgqa, _ = err_stats(out_f_gqa, ref_full)
                    assert abs_fgqa <= tol_abs
        mx_ga = max(abs_g8, abs_g16)
        mx_gr = max(rmse_g8, rmse_g16)
        print(
            f"  [PASS] Verify q_len={q_len}: R=1,2,4 x ctx=64,512,16384 "
            f"(SWA max_abs={mx_ga:.2e} rmse={mx_gr:.2e}; Full skip bit-exact)"
        )

    # Mixed batch [1, 8, 1, 64]
    for kind, nh, nkv, window, rel_ext, log_sc, splits in (
        ("SWA mixed [1,8,1,64]", SWA_NH, SWA_NKV, SWA_WINDOW, SWA_REL, False, (0, 4)),
        ("Full mixed [1,8,1,64]", FULL_NH, FULL_NKV, 0, FULL_REL, True, (8, 4)),
        (
            "SWA mixed [1,3,1,8] 1-launch",
            SWA_NH,
            SWA_NKV,
            SWA_WINDOW,
            SWA_REL,
            False,
            (0, 0),
        ),
    ):
        ql = [1, 3, 1, 8] if "1-launch" in kind else [1, 8, 1, 64]
        sl = [700, 1024, 95, 2048]
        case = make_attn_case(
            ql,
            sl,
            nh=nh,
            nkv=nkv,
            window=window,
            rel_extent=rel_ext,
            log_scale=log_sc,
            seed=2026,
        )
        ref = attention_reference(case)
        got = run_attn_kernel(mod, case, splits, TILE_PIPE)
        mx, rm = err_stats(got, ref)
        assert mx <= tol_abs, f"{kind}: max_abs={mx}"
        print(f"  [PASS] {kind:28s}: max_abs={mx:.2e}, rmse={rm:.2e}")

    print(
        "=== 4. Numerical Parity: Multi-Step Verify + Rollback + conv_writes_in_op ==="
    )
    run_conv_rollback_parity(mod)


@torch.no_grad()
def run_conv_rollback_parity(mod) -> None:
    """Test in-op conv state write vs separate conv_state_write + dense PyTorch oracle
    across multi-step verify + random draft rejections (including 0 drafts accepted)."""
    for K_max in (2, 7):
        ring_rows = 4 + K_max  # 6 for K=2, 11 for K=7
        for conv_bs in (16, 64):
            rng = random.Random(1000 + K_max * 31 + conv_bs)
            g = torch.Generator(device="cuda").manual_seed(2000 + K_max + conv_bs)
            R = 3
            total_len = 96
            nkv = SWA_NKV
            C_kv = nkv * HEAD_DIM
            C_post = HIDDEN
            row_w = 2 * HEAD_DIM + 2 * (C_post // nkv)  # 832

            # Truth token activations per request
            xk_truth = [
                torch.randn(total_len, C_kv, device="cuda", generator=g).half()
                for _ in range(R)
            ]
            xv_truth = [
                torch.randn(total_len, C_kv, device="cuda", generator=g).half()
                for _ in range(R)
            ]
            xp_truth = [
                torch.randn(total_len, C_post, device="cuda", generator=g).half()
                for _ in range(R)
            ]
            dk_truth = [
                torch.randn(total_len, nkv * 4, device="cuda", generator=g).half() * 0.3
                for _ in range(R)
            ]

            kw_static = (
                torch.randn(C_kv, 1, 4, device="cuda", generator=g, dtype=torch.float32)
                * 0.4
            )
            vw_static = (
                torch.randn(C_kv, 1, 4, device="cuda", generator=g, dtype=torch.float32)
                * 0.4
            )
            kw_dyn = (
                torch.randn(4, C_kv, device="cuda", generator=g, dtype=torch.float32)
                * 0.4
            )
            vw_dyn = (
                torch.randn(4, C_kv, device="cuda", generator=g, dtype=torch.float32)
                * 0.4
            )
            pw_static = (
                torch.randn(
                    C_post, 1, 4, device="cuda", generator=g, dtype=torch.float32
                )
                * 0.4
            )

            # Dense PyTorch oracle over the true sequence
            def dense_static(seq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
                x = seq.float()  # [N, C]
                pad = torch.cat([torch.zeros(3, x.shape[1], device="cuda"), x], dim=0)
                w2 = w.squeeze(1)  # [C, 4]
                acc = torch.zeros_like(x)
                for lag in range(4):
                    acc = acc + pad[3 - lag : 3 - lag + x.shape[0]] * w2[:, 3 - lag]
                return (acc + x) / math.sqrt(5.0)

            want_post = [dense_static(xp_truth[r], pw_static) for r in range(R)]

            max_blocks = cdiv(total_len + K_max + 8, conv_bs) + 1
            num_pages = R * max_blocks + 2
            conv_a = torch.full(
                (num_pages, ring_rows, nkv, row_w),
                float("nan"),
                dtype=torch.float16,
                device="cuda",
            ).permute(0, 2, 1, 3)
            conv_b = conv_a.clone()
            kv_pages = R * cdiv(total_len + K_max + 8, KV_BLOCK) + 2
            kv_a = torch.zeros(
                (kv_pages, nkv, KV_BLOCK, 2 * HEAD_DIM),
                dtype=torch.float16,
                device="cuda",
            )
            kv_b = kv_a.clone()

            conv_table = torch.zeros((R, max_blocks), dtype=torch.int32, device="cuda")
            for r in range(R):
                for b in range(max_blocks):
                    conv_table[r, b] = 1 + r * max_blocks + b

            pos_cur = [0] * R
            step_idx = 0
            max_oracle_err = 0.0
            while any(p < total_len - K_max - 1 for p in pos_cur):
                is_prefill = step_idx == 0
                q_lens = [13 if is_prefill else (1 + K_max) for _ in range(R)]
                # Force 0-drafts-accepted (keep=1) frequently to stress rollback!
                keeps = [
                    q_lens[r]
                    if is_prefill
                    else (1 if (step_idx + r) % 2 == 0 else rng.randint(1, 1 + K_max))
                    for r in range(R)
                ]
                xk_parts, xv_parts, xp_parts, dk_parts, slots = [], [], [], [], []
                seq_lens, seq_idx_parts = [], []
                qsl = [0]
                for r in range(R):
                    ql, keep = q_lens[r], keeps[r]
                    p0 = pos_cur[r]
                    junk_n = ql - keep
                    xk_r = xk_truth[r][p0 : p0 + keep]
                    xv_r = xv_truth[r][p0 : p0 + keep]
                    xp_r = xp_truth[r][p0 : p0 + keep]
                    dk_r = dk_truth[r][p0 : p0 + keep]
                    if junk_n > 0:
                        xk_r = torch.cat(
                            [
                                xk_r,
                                8.0 * torch.randn(junk_n, C_kv, device="cuda").half(),
                            ]
                        )
                        xv_r = torch.cat(
                            [
                                xv_r,
                                8.0 * torch.randn(junk_n, C_kv, device="cuda").half(),
                            ]
                        )
                        xp_r = torch.cat(
                            [
                                xp_r,
                                8.0 * torch.randn(junk_n, C_post, device="cuda").half(),
                            ]
                        )
                        dk_r = torch.cat(
                            [
                                dk_r,
                                2.0
                                * torch.randn(junk_n, nkv * 4, device="cuda").half(),
                            ]
                        )
                    xk_parts.append(xk_r)
                    xv_parts.append(xv_r)
                    xp_parts.append(xp_r)
                    dk_parts.append(dk_r)
                    qsl.append(qsl[-1] + ql)
                    seq_lens.append(p0 + ql)
                    seq_idx_parts.append(
                        torch.full((ql,), r, dtype=torch.int32, device="cuda")
                    )
                    for i in range(ql):
                        p = p0 + i
                        slots.append(
                            (
                                1
                                + r * cdiv(total_len + K_max + 8, KV_BLOCK)
                                + p // KV_BLOCK
                            )
                            * KV_BLOCK
                            + (p % KV_BLOCK)
                        )

                xk = torch.cat(xk_parts)
                xv = torch.cat(xv_parts)
                xp = torch.cat(xp_parts)
                dk = torch.cat(dk_parts)
                qsl_t = torch.tensor(qsl, dtype=torch.int32, device="cuda")
                sl_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
                sidx_t = torch.cat(seq_idx_parts)
                slot_t = torch.tensor(slots, dtype=torch.int64, device="cuda")

                in_op = max(q_lens) <= ring_rows - 3  # True on every verify step!
                assert in_op == (not is_prefill)

                out_a = torch.empty(
                    (xp.shape[0], C_post), dtype=torch.float32, device="cuda"
                )
                out_b = torch.empty_like(out_a)
                use_dyn = step_idx % 3 == 1

                # Path A: in-op state write when q_len <= ring_rows - 3 (verify!)
                mod.inkling2_kv_prep(
                    xk,
                    xv,
                    kw_dyn if use_dyn else kw_static,
                    vw_dyn if use_dyn else vw_static,
                    dk if use_dyn else None,
                    dk if use_dyn else None,
                    conv_a,
                    conv_table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    kv_a,
                    slot_t,
                    0,
                    HEAD_DIM,
                    conv_bs,
                    in_op,
                    Q_CLIP,
                    RMS_EPS,
                )
                if not in_op:
                    mod.inkling2_conv_state_write(
                        xk, conv_a, conv_table, sidx_t, qsl_t, sl_t, 0, nkv, conv_bs
                    )
                    mod.inkling2_conv_state_write(
                        xv,
                        conv_a,
                        conv_table,
                        sidx_t,
                        qsl_t,
                        sl_t,
                        HEAD_DIM,
                        nkv,
                        conv_bs,
                    )

                mod.inkling2_sconv_static(
                    xp,
                    pw_static,
                    out_a,
                    conv_a,
                    conv_table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    2 * HEAD_DIM,
                    nkv,
                    conv_bs,
                    in_op,
                )
                if not in_op:
                    mod.inkling2_conv_state_write(
                        xp,
                        conv_a,
                        conv_table,
                        sidx_t,
                        qsl_t,
                        sl_t,
                        2 * HEAD_DIM,
                        nkv,
                        conv_bs,
                    )

                # Path B: always separate conv_state_write
                mod.inkling2_kv_prep(
                    xk,
                    xv,
                    kw_dyn if use_dyn else kw_static,
                    vw_dyn if use_dyn else vw_static,
                    dk if use_dyn else None,
                    dk if use_dyn else None,
                    conv_b,
                    conv_table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    kv_b,
                    slot_t,
                    0,
                    HEAD_DIM,
                    conv_bs,
                    False,
                    Q_CLIP,
                    RMS_EPS,
                )
                mod.inkling2_conv_state_write(
                    xk, conv_b, conv_table, sidx_t, qsl_t, sl_t, 0, nkv, conv_bs
                )
                mod.inkling2_conv_state_write(
                    xv, conv_b, conv_table, sidx_t, qsl_t, sl_t, HEAD_DIM, nkv, conv_bs
                )

                mod.inkling2_sconv_static(
                    xp,
                    pw_static,
                    out_b,
                    conv_b,
                    conv_table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    2 * HEAD_DIM,
                    nkv,
                    conv_bs,
                    False,
                )
                mod.inkling2_conv_state_write(
                    xp,
                    conv_b,
                    conv_table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    2 * HEAD_DIM,
                    nkv,
                    conv_bs,
                )

                # Bit-exact parity between in-op write and separate conv_state_write
                assert torch.equal(out_a, out_b), f"sconv mismatch at step {step_idx}"
                assert torch.equal(kv_a, kv_b), f"kv_prep mismatch at step {step_idx}"
                assert torch.equal(
                    torch.nan_to_num(conv_a, nan=-999.0),
                    torch.nan_to_num(conv_b, nan=-999.0),
                ), f"conv ring mismatch at step {step_idx}"

                # Accepted-prefix parity vs dense PyTorch oracle
                for r in range(R):
                    s = qsl[r]
                    keep = keeps[r]
                    p0 = pos_cur[r]
                    err = float(
                        (out_a[s : s + keep] - want_post[r][p0 : p0 + keep])
                        .abs()
                        .max()
                        .item()
                    )
                    max_oracle_err = max(max_oracle_err, err)
                    assert err < 1e-5, f"rollback oracle error {err} at step {step_idx}"
                    pos_cur[r] += keep
                step_idx += 1

            print(
                f"  [PASS] Rollback K_max={K_max} (ring={ring_rows}) bs={conv_bs:2d}: "
                f"{step_idx} steps bit-exact, max_oracle_err={max_oracle_err:.2e}"
            )


# ---------------------------------------------------------------------------
# Latency Microbenchmarks (CUDA Graph Replay)
# ---------------------------------------------------------------------------
L3_BYTES = 128 << 20  # 128 MiB V620 Infinity Cache


def time_graph(calls: list, reps: int = 20, rounds: int = 5) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            for fn in calls:
                fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(reps):
            for fn in calls:
                fn()
    graph.replay()
    torch.accelerator.synchronize()
    best = float("inf")
    for _ in range(rounds):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        graph.replay()
        ev1.record()
        torch.accelerator.synchronize()
        best = min(best, ev0.elapsed_time(ev1) * 1e3 / (reps * len(calls)))
    return best


def make_attn_call(
    mod,
    q_lens: list[int],
    seq_lens: list[int],
    *,
    nh: int,
    nkv: int,
    window: int,
    rel_extent: int,
    log_scale: bool,
    splits: tuple[int, int],
    tile_impl: int,
    cold: bool,
    reps: int,
) -> float:
    keys_per_req = min(max(seq_lens), window + 15) if window > 0 else max(seq_lens)
    bytes_per_call = len(q_lens) * keys_per_req * nkv * 2 * HEAD_DIM * 2
    n_worlds = (
        min(40, max(1, math.ceil(2 * L3_BYTES / max(bytes_per_call, 1)))) if cold else 1
    )
    fns = []
    for i in range(n_worlds):
        c = make_attn_case(
            q_lens,
            seq_lens,
            nh=nh,
            nkv=nkv,
            window=window,
            rel_extent=rel_extent,
            log_scale=log_scale,
            seed=5000 + i,
        )
        out = torch.zeros((c["T"], nh * HEAD_DIM), dtype=torch.float16, device="cuda")
        ws = torch.empty(
            workspace_numel(c["R"], c["T"], nh, splits[0], splits[1]),
            dtype=torch.float32,
            device="cuda",
        )

        def _fn(c=c, out=out, ws=ws):
            mod.inkling2_attention(
                c["q"], c["rel"], c["sink"], c["gate"], c["q_scale"], c["rel_scale"],
                c["rel_proj_t"], c["positions"], c["kv"], c["block_table"],
                c["seq_lens_t"], c["qsl_t"], out, ws, c["window"], c["rel_extent"],
                c["log_scale"], LOG_ALPHA, LOG_N_FLOOR, Q_CLIP, REL_CLIP,
                c["max_q"], splits[0], splits[1], tile_impl,
            )  # fmt: skip

        fns.append(_fn)
    eff_reps = max(2, reps // n_worlds)
    return time_graph(fns, reps=eff_reps)


def _make_conv_step(
    mod,
    in_op: bool,
    xk: torch.Tensor,
    xv: torch.Tensor,
    xp1: torch.Tensor,
    xp2: torch.Tensor,
    kw: torch.Tensor,
    vw: torch.Tensor,
    pw1: torch.Tensor,
    pw2: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conv_cache: torch.Tensor,
    table: torch.Tensor,
    sidx_t: torch.Tensor,
    qsl_t: torch.Tensor,
    sl_t: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    nkv: int,
):
    c_off2 = 2 * HEAD_DIM + HIDDEN // nkv

    def _run():
        for _ in range(48):
            mod.inkling2_kv_prep(
                xk,
                xv,
                kw,
                vw,
                None,
                None,
                conv_cache,
                table,
                sidx_t,
                qsl_t,
                sl_t,
                kv_cache,
                slots,
                0,
                HEAD_DIM,
                KV_BLOCK,
                in_op,
                Q_CLIP,
                RMS_EPS,
            )
            if not in_op:
                mod.inkling2_conv_state_write(
                    xk, conv_cache, table, sidx_t, qsl_t, sl_t, 0, nkv, KV_BLOCK
                )
                mod.inkling2_conv_state_write(
                    xv, conv_cache, table, sidx_t, qsl_t, sl_t, HEAD_DIM, nkv, KV_BLOCK
                )
            mod.inkling2_sconv_static(
                xp1,
                pw1,
                out1,
                conv_cache,
                table,
                sidx_t,
                qsl_t,
                sl_t,
                2 * HEAD_DIM,
                nkv,
                KV_BLOCK,
                in_op,
            )
            if not in_op:
                mod.inkling2_conv_state_write(
                    xp1,
                    conv_cache,
                    table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    2 * HEAD_DIM,
                    nkv,
                    KV_BLOCK,
                )
            mod.inkling2_sconv_static(
                xp2,
                pw2,
                out2,
                conv_cache,
                table,
                sidx_t,
                qsl_t,
                sl_t,
                c_off2,
                nkv,
                KV_BLOCK,
                in_op,
            )
            if not in_op:
                mod.inkling2_conv_state_write(
                    xp2,
                    conv_cache,
                    table,
                    sidx_t,
                    qsl_t,
                    sl_t,
                    c_off2,
                    nkv,
                    KV_BLOCK,
                )

    return _run


def run_latency_benchmarks(mod, *, cold: bool, reps: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    mode_tag = "cold-L3" if cold else "warm"

    print(
        f"\n=== Bench A: Decode (q=1) Legacy nh-grid vs GQA nkv-grid ({mode_tag}) ==="
    )
    print(
        f"{'Layer':14s} {'Ctx':>6s} {'R':>2s} | {'Old nh(16)':>10s} "
        f"{'GQA NW=8':>10s} {'GQA NW=16':>10s} {'GQA Auto':>10s} | "
        f"{'Speedup':>8s} {'GB/s':>7s}"
    )
    print("-" * 90)
    decode_swa_delta: dict[int, float] = {}
    for label, nh, nkv, window, rel_ext, log_sc, ctx_list in (
        ("SWA (24/8)", SWA_NH, SWA_NKV, SWA_WINDOW, SWA_REL, False, [2048]),
        ("Full (12/4)", FULL_NH, FULL_NKV, 0, FULL_REL, True, [512, 1024]),
    ):
        for ctx in ctx_list:
            for R in (1, 2, 4, 8):
                ql, sl = [1] * R, [ctx] * R
                t_old = make_attn_call(
                    mod,
                    ql,
                    sl,
                    nh=nh,
                    nkv=nkv,
                    window=window,
                    rel_extent=rel_ext,
                    log_scale=log_sc,
                    splits=(0, 1),
                    tile_impl=TILE_LEGACY_NH,
                    cold=cold,
                    reps=reps,
                )
                t_nw8 = make_attn_call(
                    mod,
                    ql,
                    sl,
                    nh=nh,
                    nkv=nkv,
                    window=window,
                    rel_extent=rel_ext,
                    log_scale=log_sc,
                    splits=(0, 0),
                    tile_impl=TILE_FORCE_NW8,
                    cold=cold,
                    reps=reps,
                )
                t_nw16 = make_attn_call(
                    mod,
                    ql,
                    sl,
                    nh=nh,
                    nkv=nkv,
                    window=window,
                    rel_extent=rel_ext,
                    log_scale=log_sc,
                    splits=(0, 0),
                    tile_impl=TILE_FORCE_NW16,
                    cold=cold,
                    reps=reps,
                )
                t_auto = make_attn_call(
                    mod,
                    ql,
                    sl,
                    nh=nh,
                    nkv=nkv,
                    window=window,
                    rel_extent=rel_ext,
                    log_scale=log_sc,
                    splits=(0, 0),
                    tile_impl=0,
                    cold=cold,
                    reps=reps,
                )
                keys = min(ctx, window + 15) if window > 0 else ctx
                kv_bytes = R * keys * nkv * 2 * HEAD_DIM * 2
                gbps = kv_bytes / (t_auto * 1e3)
                sp = t_old / t_auto
                if window > 0:
                    decode_swa_delta[R] = (t_old - t_auto) * 40 / 1e3
                print(
                    f"{label:14s} {ctx:6d} {R:2d} | {t_old:8.2f}us {t_nw8:8.2f}us "
                    f"{t_nw16:8.2f}us {t_auto:8.2f}us | {sp:7.2f}x {gbps:6.0f}"
                )
                results.append(
                    {
                        "bench": "decode_gqa",
                        "layer": label,
                        "ctx": ctx,
                        "R": R,
                        "old_nh_us": t_old,
                        "gqa_nw8_us": t_nw8,
                        "gqa_nw16_us": t_nw16,
                        "gqa_auto_us": t_auto,
                        "speedup": sp,
                        "gbps": gbps,
                    }
                )
    print(
        "  Projected 40-SWA-visit decode savings (ms/step): "
        + ", ".join(f"R={r}: -{d:.3f} ms" for r, d in sorted(decode_swa_delta.items()))
    )

    print(
        f"\n=== Bench B: Verify (q_len=2,3,8) Old 3-Launch vs 1-Launch ({mode_tag}) ==="
    )
    print(
        f"{'q_len':>5s} {'R':>2s} | {'SWA Old':>10s} {'SWA NW=8':>10s} "
        f"{'SWA NW=16':>10s} {'SWA Auto':>10s} | {'Full Old':>10s} "
        f"{'Full Skip':>10s} | {'48-Visit Saved':>15s}"
    )
    print("-" * 102)
    for q_len in (2, 3, 8):
        for R in (1, 2, 4):
            ql = [q_len] * R
            sl_swa = [2048] * R
            old_swa_splits = split_policy_legacy(R, SWA_NKV, SWA_WINDOW, 2048, q_len)
            t_swa_old = make_attn_call(
                mod,
                ql,
                sl_swa,
                nh=SWA_NH,
                nkv=SWA_NKV,
                window=SWA_WINDOW,
                rel_extent=SWA_REL,
                log_scale=False,
                splits=old_swa_splits,
                tile_impl=TILE_LEGACY_NH,
                cold=cold,
                reps=reps,
            )
            t_swa_nw8 = make_attn_call(
                mod,
                ql,
                sl_swa,
                nh=SWA_NH,
                nkv=SWA_NKV,
                window=SWA_WINDOW,
                rel_extent=SWA_REL,
                log_scale=False,
                splits=(0, 0),
                tile_impl=TILE_FORCE_NW8,
                cold=cold,
                reps=reps,
            )
            t_swa_nw16 = make_attn_call(
                mod,
                ql,
                sl_swa,
                nh=SWA_NH,
                nkv=SWA_NKV,
                window=SWA_WINDOW,
                rel_extent=SWA_REL,
                log_scale=False,
                splits=(0, 0),
                tile_impl=TILE_FORCE_NW16,
                cold=cold,
                reps=reps,
            )
            t_swa_auto = make_attn_call(
                mod,
                ql,
                sl_swa,
                nh=SWA_NH,
                nkv=SWA_NKV,
                window=SWA_WINDOW,
                rel_extent=SWA_REL,
                log_scale=False,
                splits=(0, 0),
                tile_impl=0,
                cold=cold,
                reps=reps,
            )

            # Full verify at ctx=4096 (8 visits per step)
            sl_full = [4096] * R
            old_full_splits = split_policy_legacy(R, FULL_NKV, 0, 4096, q_len)
            t_full_old = make_attn_call(
                mod,
                ql,
                sl_full,
                nh=FULL_NH,
                nkv=FULL_NKV,
                window=0,
                rel_extent=FULL_REL,
                log_scale=True,
                splits=old_full_splits,
                tile_impl=TILE_PIPE,
                cold=cold,
                reps=reps,
            )
            t_full_skip = make_attn_call(
                mod,
                ql,
                sl_full,
                nh=FULL_NH,
                nkv=FULL_NKV,
                window=0,
                rel_extent=FULL_REL,
                log_scale=True,
                splits=(1, old_full_splits[1]),
                tile_impl=TILE_PIPE | TILE_SKIP_DECODE,
                cold=cold,
                reps=reps,
            )
            saved_ms = (
                40.0 * (t_swa_old - t_swa_auto) + 8.0 * (t_full_old - t_full_skip)
            ) / 1e3
            print(
                f"{q_len:5d} {R:2d} | {t_swa_old:8.2f}us {t_swa_nw8:8.2f}us "
                f"{t_swa_nw16:8.2f}us {t_swa_auto:8.2f}us | "
                f"{t_full_old:8.2f}us {t_full_skip:8.2f}us | {-saved_ms:+12.3f} ms"
            )
            results.append(
                {
                    "bench": "verify_attn",
                    "q_len": q_len,
                    "R": R,
                    "swa_old_us": t_swa_old,
                    "swa_nw8_us": t_swa_nw8,
                    "swa_nw16_us": t_swa_nw16,
                    "swa_auto_us": t_swa_auto,
                    "full_old_us": t_full_old,
                    "full_skip_us": t_full_skip,
                    "step_saved_ms": saved_ms,
                }
            )

    print(
        "\n=== Bench C: Verify Conv State Write Elimination (48 Visits: 144 vs 336) ==="
    )
    print(
        f"{'q_len':>5s} {'R':>2s} | {'Separate (+192 wr)':>19s} "
        f"{'In-Op (0 extra)':>17s} | {'Saved / Step':>14s}"
    )
    print("-" * 68)
    for q_len in (2, 3, 8):
        ring_rows = 4 + (q_len - 1)
        for R in (1, 2, 4):
            T = R * q_len
            g = torch.Generator(device="cuda").manual_seed(8000 + q_len * 10 + R)
            nkv = SWA_NKV
            C_kv = nkv * HEAD_DIM
            row_w = 2 * HEAD_DIM + 2 * (HIDDEN // nkv)
            conv_cache = torch.zeros(
                (R * 4 + 2, ring_rows, nkv, row_w), dtype=torch.float16, device="cuda"
            ).permute(0, 2, 1, 3)
            kv_cache = torch.zeros(
                (R * 4 + 2, nkv, KV_BLOCK, 2 * HEAD_DIM),
                dtype=torch.float16,
                device="cuda",
            )
            table = torch.arange(1, R * 4 + 1, dtype=torch.int32, device="cuda").view(
                R, 4
            )
            qsl_t = torch.arange(
                0, (R + 1) * q_len, q_len, dtype=torch.int32, device="cuda"
            )
            sl_t = torch.full((R,), 128, dtype=torch.int32, device="cuda")
            sidx_t = torch.repeat_interleave(
                torch.arange(R, dtype=torch.int32, device="cuda"), q_len
            )
            slots = torch.arange(
                KV_BLOCK, KV_BLOCK + T, dtype=torch.int64, device="cuda"
            )
            xk = torch.randn((T, C_kv), device="cuda", generator=g).half()
            xv = torch.randn((T, C_kv), device="cuda", generator=g).half()
            xp1 = torch.randn((T, HIDDEN), device="cuda", generator=g).half()
            xp2 = torch.randn((T, HIDDEN), device="cuda", generator=g).half()
            kw = torch.randn((C_kv, 1, 4), device="cuda", generator=g)
            vw = torch.randn((C_kv, 1, 4), device="cuda", generator=g)
            pw1 = torch.randn((HIDDEN, 1, 4), device="cuda", generator=g)
            pw2 = torch.randn((HIDDEN, 1, 4), device="cuda", generator=g)
            out1 = torch.empty((T, HIDDEN), dtype=torch.float32, device="cuda")
            out2 = torch.empty((T, HIDDEN), dtype=torch.float32, device="cuda")

            fn_sep = _make_conv_step(
                mod,
                False,
                xk,
                xv,
                xp1,
                xp2,
                kw,
                vw,
                pw1,
                pw2,
                out1,
                out2,
                conv_cache,
                table,
                sidx_t,
                qsl_t,
                sl_t,
                kv_cache,
                slots,
                nkv,
            )
            fn_inop = _make_conv_step(
                mod,
                True,
                xk,
                xv,
                xp1,
                xp2,
                kw,
                vw,
                pw1,
                pw2,
                out1,
                out2,
                conv_cache,
                table,
                sidx_t,
                qsl_t,
                sl_t,
                kv_cache,
                slots,
                nkv,
            )
            ms_sep = time_graph([fn_sep], reps=4) / 1e3
            ms_inop = time_graph([fn_inop], reps=4) / 1e3
            saved = ms_sep - ms_inop
            print(
                f"{q_len:5d} {R:2d} | {ms_sep:16.3f} ms "
                f"{ms_inop:14.3f} ms | {-saved:+11.3f} ms"
            )
            results.append(
                {
                    "bench": "verify_conv_in_op",
                    "q_len": q_len,
                    "R": R,
                    "separate_ms": ms_sep,
                    "in_op_ms": ms_inop,
                    "saved_ms": saved,
                }
            )

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="gfx1030")
    ap.add_argument("--opt", default="-O3")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument(
        "--cold", action="store_true", help="cycle caches to flush 128 MiB L3"
    )
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print(f"JIT compiling {SOURCE} ({args.archs}, {args.opt}) ...")
    mod = build_jit(args.archs, args.opt, verbose=args.verbose_build)
    print("JIT build complete.\n")

    if not args.skip_parity:
        run_parity_suite(mod)

    rows: list[dict[str, Any]] = []
    if not args.skip_bench:
        rows = run_latency_benchmarks(mod, cold=args.cold, reps=args.reps)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
