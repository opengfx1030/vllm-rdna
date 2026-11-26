#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async OpenAI SSE benchmark & KLD verification suite for Inkling-2-Mini.

Covers all serving regimes cleanly via continuous async streaming:
  * decode       : c={1,2,4,8} steady-state decode TTFT, ITL (p50/p95/p99/max),
                   TPOT, and aggregate / per-request tok/s.
  * prefill      : uncached 1k/4k/16k prompts measuring cold TTFT and prefill tok/s.
  * mixed_stall  : 4 continuous decode streams while injecting 3 staggered cold
                   4k prefill bursts every 1.5s; reports decode max ITL spike,
                   p99 ITL, prefill TTFT, and server KV preemptions.
  * kld          : captures greedy top-20 logprobs on 16 fixed prompts under both
                   solo (c=1) and concurrent (c=8) batching.
  * all          : runs decode, prefill, mixed_stall, and kld in one pass.
  * compare      : computes KL(P_base || P_cand) and top-1 match between two
                   saved JSON runs (solo vs solo, c8 vs c8).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

FIXED_KLD_PROMPTS = [
    (
        "Solve step by step: if a train travels at 84 km/h for 2.5 hours and "
        "then 60 km/h for 45 minutes, what is the total distance in km?"
    ),
    "Write a Python function `quicksort(arr)` that sorts a list of integers cleanly.",
    (
        "Explain why PCIe BAR1 Peer-to-Peer atomic ordering differs between posted "
        "writes and non-posted reads."
    ),
    (
        "List the five largest European Union countries by population and their "
        "capital cities."
    ),
    "Compute the derivative of f(x) = x^3 * exp(-2x) with respect to x.",
    (
        "In AMD RDNA2 Wave32 execution, how many threads execute in one wavefront, "
        "and what does v_dot2_f32_f16 compute?"
    ),
    (
        "Translate the following sentence to French: 'The asynchronous scheduler "
        "avoided head-of-line blocking during long prefill bursts.'"
    ),
    (
        "What is the time complexity of heapify vs inserting n elements one by one "
        "into a binary heap?"
    ),
    (
        "Write a SQL query returning the top 3 departments by average salary among "
        "active employees."
    ),
    (
        "Summarize the difference between Mixture-of-Experts all-to-all dispatch "
        "and tensor-parallel all-reduce."
    ),
    (
        "Given an array [3, 1, 4, 1, 5, 9, 2, 6], trace the first two merges of "
        "bottom-up merge sort."
    ),
    (
        "Why can bfloat16 matrix accumulation lose accuracy if emulated naively "
        "in float16?"
    ),
    (
        "State Bayes' theorem and apply it to a medical test with 99% sensitivity, "
        "95% specificity, and 1% prevalence."
    ),
    "Write a minimal C++ RAII wrapper around hipEvent_t using move semantics.",
    (
        "Explain the purpose of RoPE (Rotary Position Embeddings) in causal "
        "self-attention."
    ),
    "Calculate 17^4 mod 100 using modular exponentiation.",
]


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize_ms(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "count": len(s),
        "mean": round(statistics.fmean(s), 3),
        "p50": round(_percentile(s, 50.0), 3),
        "p95": round(_percentile(s, 95.0), 3),
        "p99": round(_percentile(s, 99.0), 3),
        "max": round(s[-1], 3),
    }


def _make_prompt(target_tokens: int, nonce: str) -> str:
    # Approximately 12 tokens per line so prompt length closely tracks target_tokens
    # while remaining completely uncached via the leading nonce.
    header = (
        f"[nonce={nonce}] Carefully read the following numbered ledger entries "
        "and reply ONLY with the final line index.\n"
    )
    n_lines = max(1, (target_tokens - 24) // 13)
    lines = [
        (
            f"Entry {i:05d}: key={(i * 131 + 17) % 997} "
            f"bucket={(i * 37) % 64} status=active."
        )
        for i in range(1, n_lines + 1)
    ]
    return header + "\n".join(lines) + "\nFinal line index:"


async def _fetch_metrics(session: aiohttp.ClientSession, base: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        async with session.get(
            f"{base}/metrics", timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status != 200:
                return out
            text = await resp.text()
    except Exception:
        return out
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if (
            "num_preemptions" in line
            or "kv_cache_usage_perc" in line
            or "num_requests_waiting" in line
        ):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].split("{")[0]
                with contextlib.suppress(ValueError):
                    out[key] = out.get(key, 0.0) + float(parts[-1])
    return out


async def _stream_completion(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    ignore_eos: bool = True,
    temperature: float = 0.0,
    logprobs: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs

    t0 = time.perf_counter()
    t_first: float | None = None
    t_prev: float | None = None
    itls_ms: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    text_parts: list[str] = []
    lp_tokens: list[str] = []
    lp_tops: list[dict[str, float]] = []

    async with session.post(f"{base}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        buf = b""
        async for chunk in resp.content.iter_chunked(65536):
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.strip()
                if not line.startswith(b"data: "):
                    continue
                data_b = line[6:]
                if data_b == b"[DONE]":
                    continue
                ev = json.loads(data_b)
                now = time.perf_counter()
                usage = ev.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(
                        usage.get("completion_tokens") or completion_tokens
                    )
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = int(
                        prompt_details.get("cached_tokens") or cached_tokens
                    )
                choices = ev.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                txt = ch0.get("text") or ""
                lp = ch0.get("logprobs")
                if txt or lp:
                    if t_first is None:
                        t_first = now
                    elif t_prev is not None:
                        itls_ms.append((now - t_prev) * 1000.0)
                    t_prev = now
                    if txt:
                        text_parts.append(txt)
                    if lp:
                        toks = lp.get("tokens") or []
                        tops = lp.get("top_logprobs") or []
                        lp_tokens.extend(toks)
                        for d in tops:
                            if isinstance(d, dict):
                                lp_tops.append({str(k): float(v) for k, v in d.items()})

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ttft_ms = (t_first - t0) * 1000.0
    wall_s = t_end - t0
    if completion_tokens == 0:
        completion_tokens = max(len(lp_tokens), len(text_parts))
    decode_s = max(t_end - t_first, 1e-9)
    tpot_ms = (decode_s * 1000.0) / max(completion_tokens - 1, 1)

    return {
        "t0": t0,
        "t_first": t_first,
        "t_end": t_end,
        "wall_s": wall_s,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itls_ms": itls_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "text": "".join(text_parts),
        "lp_tokens": lp_tokens,
        "lp_tops": lp_tops,
    }


async def run_decode(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    concurrencies: list[int],
    decode_tokens: int,
    prompt_tokens: int,
    reps: int,
) -> dict[str, Any]:
    # Warmup
    await _stream_completion(
        session,
        base,
        model,
        _make_prompt(prompt_tokens, f"warm-{uuid.uuid4().hex[:8]}"),
        16,
    )
    results: dict[str, Any] = {}
    for c in concurrencies:
        run_itls: list[float] = []
        run_ttfts: list[float] = []
        run_tpots: list[float] = []
        total_toks = 0
        total_wall = 0.0
        for rep in range(reps):
            prompts = [
                _make_prompt(
                    prompt_tokens, f"dec-c{c}-r{rep}-i{i}-{uuid.uuid4().hex[:6]}"
                )
                for i in range(c)
            ]
            t0 = time.perf_counter()
            outs = await asyncio.gather(
                *[
                    _stream_completion(session, base, model, p, decode_tokens)
                    for p in prompts
                ]
            )
            dt = time.perf_counter() - t0
            total_wall += dt
            for o in outs:
                total_toks += o["completion_tokens"]
                run_ttfts.append(o["ttft_ms"])
                run_tpots.append(o["tpot_ms"])
                run_itls.extend(o["itls_ms"])
        agg_tok_s = total_toks / max(total_wall, 1e-9)
        per_req_tok_s = agg_tok_s / c
        results[f"c{c}"] = {
            "concurrency": c,
            "reps": reps,
            "total_completion_tokens": total_toks,
            "wall_s": round(total_wall, 3),
            "agg_tok_s": round(agg_tok_s, 2),
            "per_req_tok_s": round(per_req_tok_s, 2),
            "ttft_ms": _summarize_ms(run_ttfts),
            "tpot_ms": _summarize_ms(run_tpots),
            "itl_ms": _summarize_ms(run_itls),
        }
        print(
            f"[decode c={c}] agg={agg_tok_s:.2f} tok/s  req={per_req_tok_s:.2f} tok/s  "
            f"ttft_p50={results[f'c{c}']['ttft_ms']['p50']:.1f}ms  "
            f"tpot_p50={results[f'c{c}']['tpot_ms']['p50']:.2f}ms  "
            f"itl_p50={results[f'c{c}']['itl_ms']['p50']:.2f}ms  "
            f"itl_max={results[f'c{c}']['itl_ms']['max']:.1f}ms",
            flush=True,
        )
    return results


async def run_prefill(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    lengths: list[int],
    reps: int,
) -> dict[str, Any]:
    await _stream_completion(
        session, base, model, _make_prompt(512, f"pwarm-{uuid.uuid4().hex[:8]}"), 8
    )
    results: dict[str, Any] = {}
    for n_tok in lengths:
        ttfts: list[float] = []
        actual_prompts: list[int] = []
        cached_list: list[int] = []
        for rep in range(reps):
            p = _make_prompt(n_tok, f"pref-{n_tok}-r{rep}-{uuid.uuid4().hex[:8]}")
            out = await _stream_completion(session, base, model, p, 8)
            ttfts.append(out["ttft_ms"])
            actual_prompts.append(out["prompt_tokens"])
            cached_list.append(out["cached_tokens"])
        p_mean = statistics.fmean(actual_prompts) if actual_prompts else float(n_tok)
        ttft_p50 = _percentile(sorted(ttfts), 50.0)
        tok_s = p_mean / max(ttft_p50 / 1000.0, 1e-9)
        results[f"len_{n_tok}"] = {
            "target_tokens": n_tok,
            "actual_prompt_tokens_mean": round(p_mean, 1),
            "cached_tokens_max": max(cached_list) if cached_list else 0,
            "ttft_ms": _summarize_ms(ttfts),
            "prefill_tok_s": round(tok_s, 1),
        }
        print(
            f"[prefill {int(p_mean)} tok] ttft_p50={ttft_p50:.1f}ms  "
            f"ttft_max={max(ttfts):.1f}ms  prefill={tok_s:.1f} tok/s  "
            f"cached_max={results[f'len_{n_tok}']['cached_tokens_max']}",
            flush=True,
        )
    return results


async def run_mixed_stall(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    n_decode: int,
    decode_tokens: int,
    n_prefills: int,
    prefill_tokens: int,
    interval_s: float,
) -> dict[str, Any]:
    m_before = await _fetch_metrics(session, base)

    async def _decode_worker(idx: int) -> dict[str, Any]:
        p = _make_prompt(128, f"mixdec-{idx}-{uuid.uuid4().hex[:8]}")
        return await _stream_completion(session, base, model, p, decode_tokens)

    async def _prefill_injector() -> list[dict[str, Any]]:
        await asyncio.sleep(0.5)
        pre_outs = []
        for k in range(n_prefills):
            p = _make_prompt(prefill_tokens, f"mixpref-{k}-{uuid.uuid4().hex[:8]}")
            pre_outs.append(
                asyncio.create_task(_stream_completion(session, base, model, p, 8))
            )
            if k + 1 < n_prefills:
                await asyncio.sleep(interval_s)
        return await asyncio.gather(*pre_outs)

    t0 = time.perf_counter()
    dec_tasks = [asyncio.create_task(_decode_worker(i)) for i in range(n_decode)]
    pref_task = asyncio.create_task(_prefill_injector())
    dec_outs = await asyncio.gather(*dec_tasks)
    pref_outs = await pref_task
    wall_s = time.perf_counter() - t0
    m_after = await _fetch_metrics(session, base)

    all_itls: list[float] = []
    dec_toks = 0
    dec_tpots: list[float] = []
    for o in dec_outs:
        all_itls.extend(o["itls_ms"])
        dec_toks += o["completion_tokens"]
        dec_tpots.append(o["tpot_ms"])

    pref_ttfts = [o["ttft_ms"] for o in pref_outs]
    pref_cached = max((o["cached_tokens"] for o in pref_outs), default=0)
    preemptions_delta = max(
        0.0,
        m_after.get(
            "vllm:num_preemptions_total", m_after.get("vllm:num_preemptions", 0.0)
        )
        - m_before.get(
            "vllm:num_preemptions_total", m_before.get("vllm:num_preemptions", 0.0)
        ),
    )
    stalls_100 = sum(1 for x in all_itls if x > 100.0)
    stalls_250 = sum(1 for x in all_itls if x > 250.0)
    stalls_500 = sum(1 for x in all_itls if x > 500.0)

    out = {
        "n_decode_streams": n_decode,
        "decode_tokens_each": decode_tokens,
        "n_prefills": n_prefills,
        "prefill_tokens_each": prefill_tokens,
        "wall_s": round(wall_s, 3),
        "decode_total_tokens": dec_toks,
        "decode_agg_tok_s": round(dec_toks / max(wall_s, 1e-9), 2),
        "decode_tpot_ms": _summarize_ms(dec_tpots),
        "decode_itl_ms": _summarize_ms(all_itls),
        "stalls_gt_100ms": stalls_100,
        "stalls_gt_250ms": stalls_250,
        "stalls_gt_500ms": stalls_500,
        "prefill_ttft_ms": _summarize_ms(pref_ttfts),
        "prefill_cached_tokens_max": pref_cached,
        "preemptions_delta": round(preemptions_delta, 1),
    }
    print(
        f"[mixed_stall] dec_agg={out['decode_agg_tok_s']:.2f} tok/s  "
        f"itl_p50={out['decode_itl_ms']['p50']:.2f}ms  "
        f"itl_p99={out['decode_itl_ms']['p99']:.1f}ms  "
        f"itl_max={out['decode_itl_ms']['max']:.1f}ms  "
        f"stalls(>100/>250/>500)={stalls_100}/{stalls_250}/{stalls_500}  "
        f"pref_ttft_p50={out['prefill_ttft_ms']['p50']:.1f}ms  "
        f"preemptions={out['preemptions_delta']}",
        flush=True,
    )
    return out


async def run_kld(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    async def _run_mode(concurrent: bool) -> list[dict[str, Any]]:
        if not concurrent:
            res = []
            for p in FIXED_KLD_PROMPTS:
                res.append(
                    await _stream_completion(
                        session,
                        base,
                        model,
                        p,
                        max_tokens,
                        ignore_eos=True,
                        temperature=0.0,
                        logprobs=20,
                    )
                )
            return res
        # Run in two c=8 batches across all 16 prompts
        out: list[dict[str, Any]] = []
        for off in (0, 8):
            batch = FIXED_KLD_PROMPTS[off : off + 8]
            out.extend(
                await asyncio.gather(
                    *[
                        _stream_completion(
                            session,
                            base,
                            model,
                            p,
                            max_tokens,
                            ignore_eos=True,
                            temperature=0.0,
                            logprobs=20,
                        )
                        for p in batch
                    ]
                )
            )
        return out

    def _pack(outs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "text": o["text"],
                "tokens": o["lp_tokens"],
                "top_logprobs": o["lp_tops"],
            }
            for o in outs
        ]

    solo = _pack(await _run_mode(False))
    c8 = _pack(await _run_mode(True))
    solo_toks = sum(len(x["tokens"]) for x in solo)
    c8_toks = sum(len(x["tokens"]) for x in c8)
    print(
        f"[kld] captured solo={len(solo)} prompts ({solo_toks} tok) "
        f"c8={len(c8)} prompts ({c8_toks} tok)",
        flush=True,
    )
    return {"max_tokens": max_tokens, "solo": solo, "c8": c8}


def _kl_step(
    p_dict: dict[str, float], q_dict: dict[str, float], eps: float = 1e-9
) -> tuple[float, float, bool]:
    """Return ``(kl_tail_floored, kl_overlap, top1_eq)`` for one top-20 step.

    Tokens present in P's top-20 that fall outside Q's top-20 (rank >= 21) are
    floored at ``q_floor = min(q_lin) / q_sum`` rather than ``1e-9`` so normal
    rank-21 boundary churn does not inject false ~10*p KL spikes.
    """
    if not p_dict or not q_dict:
        return 0.0, 0.0, True
    p_lin = {k: math.exp(v) for k, v in p_dict.items()}
    q_lin = {k: math.exp(v) for k, v in q_dict.items()}
    p_sum = sum(p_lin.values())
    q_sum = sum(q_lin.values())
    if p_sum <= 0 or q_sum <= 0:
        return 0.0, 0.0, True
    p_top1 = max(p_dict.items(), key=lambda kv: kv[1])[0]
    q_top1 = max(q_dict.items(), key=lambda kv: kv[1])[0]
    q_floor = max(min(q_lin.values()) / q_sum, eps)
    kl_floor = 0.0
    kl_ov = 0.0
    ov_mass = 0.0
    for k, pv in p_lin.items():
        pn = pv / p_sum
        if pn <= 0.0:
            continue
        if k in q_lin:
            qn = max(q_lin[k] / q_sum, eps)
            kl_floor += pn * math.log(pn / qn)
            kl_ov += pn * math.log(pn / qn)
            ov_mass += pn
        else:
            kl_floor += pn * math.log(pn / q_floor)
    return (
        max(0.0, kl_floor),
        (max(0.0, kl_ov) if ov_mass > 0 else 0.0),
        (p_top1 == q_top1),
    )


def compare_kld(base_path: Path, cand_path: Path) -> int:
    b = json.loads(base_path.read_text())
    c = json.loads(cand_path.read_text())
    bk = b.get("kld") or {}
    ck = c.get("kld") or {}
    exit_code = 0
    for mode in ("solo", "c8"):
        bm = bk.get(mode) or []
        cm = ck.get(mode) or []
        if not bm or not cm:
            print(f"[compare {mode}] missing KLD data", flush=True)
            continue
        kls: list[float] = []
        kls_ov: list[float] = []
        top1_hits = 0
        tok_hits = 0
        matched_pos = 0
        total_pos = 0
        prompt_flips = 0
        for bp, cp in zip(bm, cm):
            btoks = bp.get("tokens") or []
            ctoks = cp.get("tokens") or []
            btops = bp.get("top_logprobs") or []
            ctops = cp.get("top_logprobs") or []
            n = min(len(btops), len(ctops))
            flipped = False
            for i in range(n):
                kl_f, kl_o, top1_eq = _kl_step(btops[i], ctops[i])
                same_tok = i < len(btoks) and i < len(ctoks) and btoks[i] == ctoks[i]
                total_pos += 1
                if not flipped:
                    # Include up to and including the first divergent greedy token
                    # (same prompt prefix); positions strictly after a token flip
                    # condition on different tokens and are counted in flips only.
                    kls.append(kl_f)
                    kls_ov.append(kl_o)
                    top1_hits += int(top1_eq)
                    tok_hits += int(same_tok)
                    matched_pos += 1
                    if not same_tok:
                        flipped = True
                        prompt_flips += 1
        mean_kl = statistics.fmean(kls) if kls else 0.0
        mean_kl_ov = statistics.fmean(kls_ov) if kls_ov else 0.0
        p95_kl = _percentile(sorted(kls), 95.0)
        max_kl = max(kls) if kls else 0.0
        top1_rate = top1_hits / max(matched_pos, 1)
        tok_rate = tok_hits / max(matched_pos, 1)
        warn = " WARN_TOP1" if top1_rate < 0.99 else ""
        print(
            f"[compare {mode}] matched_pos={matched_pos}/{total_pos} "
            f"flips={prompt_flips}/{len(bm)} mean_kl={mean_kl:.6f} "
            f"kl_overlap={mean_kl_ov:.6f} p95_kl={p95_kl:.6f} max_kl={max_kl:.6f} "
            f"top1_match={top1_rate:.4f} tok_match={tok_rate:.4f}{warn}",
            flush=True,
        )
        if mean_kl >= 1e-3:
            exit_code = 1
    return exit_code


async def async_main(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    model = args.model
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=64)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if not model:
            async with session.get(f"{base}/v1/models") as resp:
                resp.raise_for_status()
                models_data = (await resp.json()).get("data") or []
                model = models_data[0]["id"] if models_data else "inkling2-mini"

        report: dict[str, Any] = {
            "url": base,
            "model": model,
            "suite": args.suite,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if args.suite in ("decode", "all"):
            concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
            report["decode"] = await run_decode(
                session,
                base,
                model,
                concs,
                args.decode_tokens,
                args.decode_prompt_tokens,
                args.reps,
            )
        if args.suite in ("prefill", "all"):
            lens = [int(x) for x in args.prefill_lengths.split(",") if x.strip()]
            report["prefill"] = await run_prefill(
                session, base, model, lens, args.prefill_reps
            )
        if args.suite in ("mixed_stall", "all"):
            report["mixed_stall"] = await run_mixed_stall(
                session,
                base,
                model,
                args.stall_decode_streams,
                args.stall_decode_tokens,
                args.stall_prefills,
                args.stall_prefill_tokens,
                args.stall_interval_s,
            )
        if args.suite in ("kld", "all"):
            report["kld"] = await run_kld(session, base, model, args.kld_tokens)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2))
            print(f"[saved] {out_path}", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        ap = argparse.ArgumentParser(prog="bench_i2_suite.py compare")
        ap.add_argument("baseline", type=Path)
        ap.add_argument("candidate", type=Path)
        ns = ap.parse_args(sys.argv[2:])
        return compare_kld(ns.baseline, ns.candidate)

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18221")
    ap.add_argument("--model", default="inkling2-mini")
    ap.add_argument(
        "--suite",
        choices=("decode", "prefill", "mixed_stall", "kld", "all"),
        default="all",
    )
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--decode-prompt-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prefill-lengths", default="1024,4096,16384")
    ap.add_argument("--prefill-reps", type=int, default=2)
    ap.add_argument("--stall-decode-streams", type=int, default=4)
    ap.add_argument("--stall-decode-tokens", type=int, default=256)
    ap.add_argument("--stall-prefills", type=int, default=3)
    ap.add_argument("--stall-prefill-tokens", type=int, default=4096)
    ap.add_argument("--stall-interval-s", type=float, default=1.5)
    ap.add_argument("--kld-tokens", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
