#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump and compare RidgeFill cache-only vs stock QSA cache writes (PR 155).

``run``
    Offline TP4 engine in the serve shape of ``flashnext-mtp2`` with MTP off
    and prefix caching off (chunked prefill, 2048 batched tokens, block 1024,
    compile mode 3, vision cap, so the stock QSA path is the serve's), plus
    the RidgeFill cache-only fill of the ``rocblas`` A/B arm. The worker
    extension ``dump_probe.RidgeFillDumpProbe`` runs, for the first filled
    forwards of one PG-19 eval window, the cache-only writer (A) and the stock
    writer twice (B, B2) on the same block input and saves the filled rows'
    cache entries per rank. Needs the GPUs; the box script owns the env.

``compare DIR``
    CPU only. Per forward, layer, rank and cache: RMS and max abs of A - B and
    of B - B2, bitwise equality, the K split into RoPE and pass-through dims
    and per KV head, the stored M-RoPE positions against the rows' positions,
    and the attention output the cache-only op zeroes for the filled rows.
    A cache is flagged when RMS(A - B) exceeds ``--factor`` times the floor,
    max(RMS(B2 - B), fp16 unit roundoff * RMS(B)). Writes ``compare.json``.

Usage::

    python3 cache_dump.py run --model M --windows W.json --map DIR \\
        --pins PINS --out DUMP
    python3 cache_dump.py compare DUMP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

EXTENSION = "vllm.model_executor.layers.ridgefill.dump_probe.RidgeFillDumpProbe"
# compilation config of flashnext-mtp2 at MTP=0 (serve_rdna.sh assembles it).
COMPILATION = (
    '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE",'
    '"cudagraph_capture_sizes":[1,2,4,8],"compile_ranges_endpoints":[]}'
)
# fp16 unit roundoff: the smallest floor a bitwise-deterministic B can have.
FP16_EPS = 2.0**-11
FACTOR = 10.0
K_CACHES = ("main_k", "compressed_k")
PER_HEAD = ("main_k", "main_v")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the box venv has torch
        raise SystemExit("torch is required") from exc
    return torch


def ids_sha(ids: list[int]) -> str:
    """The window checksum ``ppl_long.py prepare`` stores."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def _fill_env(args) -> dict[str, str]:
    """The ``rocblas`` arm of the box A/B: cache-only fill, rocBLAS map GEMM."""
    env = {
        "VLLM_RDNA_RIDGEFILL": "1",
        "VLLM_RDNA_RIDGEFILL_CACHE_ONLY": "1",
        "VLLM_RDNA_RIDGEFILL_DEBUG": "1",
        "VLLM_RDNA_RIDGEFILL_MAP": args.map,
        "VLLM_RDNA_RIDGEFILL_PINS": args.pins,
        "VLLM_RDNA_RIDGEFILL_MAP_FUSED": str(args.map_fused),
    }
    if args.tail:
        env["VLLM_RDNA_RIDGEFILL_TAIL"] = str(args.tail)
    return env


def _run(args) -> int:
    out = Path(args.out)
    if out.exists() and any(out.glob("rank*.pt")):
        raise SystemExit(f"{out} already holds a dump, refusing to overwrite")
    spec = json.loads(Path(args.windows).read_text(encoding="utf-8"))
    window = spec["windows"][args.window]
    ids = [int(t) for t in window["ids"]]
    if window.get("sha") and ids_sha(ids) != window["sha"]:
        raise SystemExit(f"window {args.window} ids do not match its sha")
    env = _fill_env(args)
    # The workers inherit the driver's environment.
    os.environ.update(env)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tp,
        enable_expert_parallel=args.tp > 1,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=0,
        prefill_schedule_interval=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        mamba_cache_mode="align",
        # vision-cap: not text-only, so the stock QSA call takes the serve's
        # unfused K norm and RoPE (the fused kernel needs a text-only model).
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": 1605632},
        attention_backend=args.attention_backend,
        # Mode 3 decides the CustomOp dispatch (custom_ops 'none') the eager
        # filled forwards use; enforce_eager would enable every custom op.
        compilation_config=json.loads(args.compilation_config),
        disable_log_stats=True,
        worker_extension_cls=args.extension,
    )
    llm.collective_rpc("ridgefill_dump_arm", args=(str(out), args.forwards, args.skip))
    result = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(max_tokens=1, temperature=0.0),
    )
    ranks = llm.collective_rpc("ridgefill_dump_save")
    meta = {
        "model": args.model,
        "windows": args.windows,
        "window": args.window,
        "book": window.get("book"),
        "sha": window.get("sha"),
        "tokens": len(ids),
        "forwards": args.forwards,
        "skip": args.skip,
        "env": env,
        "extension": args.extension,
        "compilation_config": args.compilation_config,
        "token": list(result[0].outputs[0].token_ids),
        "ranks": ranks,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for rank in ranks:
        print(
            f"rank {rank['rank']}: {rank['records']} records over "
            f"{rank['filled_forwards_seen']} filled forwards, layers {rank['layers']}"
        )
    if not any(rank["records"] for rank in ranks):
        print("no filled forward reached the cache-only op", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _rms(t) -> float:
    if t.numel() == 0:
        return 0.0
    return float(t.double().pow(2).mean().sqrt())


def pair_stats(a, b, b2) -> dict[str, Any]:
    """A - B against the B2 - B floor for one cache tensor."""
    a, b, b2 = a.double(), b.double(), b2.double()
    d_ab = a - b
    d_bb = b2 - b
    ref = _rms(b)
    ab = _rms(d_ab)
    bb = _rms(d_bb)
    floor = max(bb, FP16_EPS * ref)
    return {
        "n": int(b.numel()),
        "rms": ref,
        "ab_rms": ab,
        "ab_rel": ab / ref if ref else 0.0,
        "ab_max": float(d_ab.abs().max()) if d_ab.numel() else 0.0,
        "ab_equal": bool(a.shape == b.shape and bool((a == b).all())),
        "ab_ndiff": int((d_ab != 0).sum()),
        "bb_rms": bb,
        "bb_rel": bb / ref if ref else 0.0,
        "bb_max": float(d_bb.abs().max()) if d_bb.numel() else 0.0,
        "bb_equal": bool((d_bb == 0).all()),
        "ratio": ab / floor if floor else 0.0,
    }


def _parts(name: str, a, b, b2, rotary_dim: int | None):
    """The whole tensor, then the RoPE / pass-through dims and each head."""
    yield name, a, b, b2
    if name in K_CACHES and rotary_dim and 0 < rotary_dim < b.shape[-1]:
        r = rotary_dim
        yield f"{name}.rope", a[..., :r], b[..., :r], b2[..., :r]
        yield f"{name}.pass", a[..., r:], b[..., r:], b2[..., r:]
    if name in PER_HEAD and b.dim() == 3 and b.shape[1] > 1:
        for h in range(b.shape[1]):
            yield f"{name}[h{h}]", a[:, h], b[:, h], b2[:, h]


def expected_ring_positions(positions, rows):
    """The ``[n, 3]`` M-RoPE positions the ring should hold for ``rows``."""
    if positions.dim() == 2:
        return positions.t().index_select(0, rows)
    return positions.index_select(0, rows).unsqueeze(1).expand(-1, 3)


def compare_record(rec: dict, factor: float = FACTOR) -> list[dict[str, Any]]:
    """Every comparison row of one (forward, layer, rank) record."""
    torch = _torch()
    rows: list[dict[str, Any]] = []
    base = {
        "forward": rec["forward"],
        "layer": rec["layer_idx"],
        "consumed": bool(rec["consumed"]),
        "fallback": bool(rec["fallback"]),
    }
    a_snap = rec["A"] if rec["A"] is not None else rec["B"]
    for name in sorted(rec["B"]):
        if name == "raw_pos":
            continue
        a, b, b2 = a_snap[name], rec["B"][name], rec["B2"][name]
        for part, pa, pb, pb2 in _parts(name, a, b, b2, rec.get("rotary_dim")):
            stats = pair_stats(pa, pb, pb2)
            stats["flag"] = stats["ratio"] > factor
            rows.append({**base, "cache": part, **stats})
    if "raw_pos" in rec["B"]:
        got = {k: rec[k]["raw_pos"].reshape(-1, 3) for k in ("B", "B2")}
        got["A"] = a_snap["raw_pos"].reshape(-1, 3)
        want = expected_ring_positions(rec["positions"], rec["slot_rows"]["raw"])
        exact = {
            "ab_equal": bool(torch.equal(got["A"], got["B"])),
            "bb_equal": bool(torch.equal(got["B"], got["B2"])),
            "a_rows_equal": bool(torch.equal(got["A"], want.to(got["A"].dtype))),
            "b_rows_equal": bool(torch.equal(got["B"], want.to(got["B"].dtype))),
            "n": int(got["B"].numel()),
        }
        exact["flag"] = not all(v for k, v in exact.items() if k != "n")
        rows.append({**base, "cache": "raw_pos", **exact})
    out = rec["out"]
    stock = _rms(out["stock_rms"])
    floor = max(_rms(out["floor_rms"]), FP16_EPS * stock)
    # A's filled rows are zero unless the op fell back to the stock call.
    dropped = stock if out["a_zero"] else 0.0
    rows.append(
        {
            **base,
            "cache": "attn_out",
            "n": int(out["stock_rms"].numel()),
            "rms": stock,
            "ab_rms": dropped,
            "ab_rel": dropped / stock if stock else 0.0,
            "bb_rms": _rms(out["floor_rms"]),
            "block_rms": _rms(out["block_rms"]),
            "out_vs_block": stock / _rms(out["block_rms"])
            if out["block_rms"].numel()
            else 0.0,
            "ratio": dropped / floor if floor else 0.0,
            # Only a consumed output matters: a predicted layer's filled rows
            # are re-predicted from the split state by the next layer.
            "flag": bool(rec["consumed"]) and dropped > factor * floor,
        }
    )
    return rows


def compare_dir(path: Path, factor: float = FACTOR) -> dict[str, Any]:
    torch = _torch()
    files = sorted(path.glob("rank*.pt"))
    if not files:
        raise SystemExit(f"{path} has no rank*.pt")
    rows: list[dict[str, Any]] = []
    for file in files:
        data = torch.load(file, map_location="cpu", weights_only=True)
        for rec in data["records"]:
            for row in compare_record(rec, factor):
                rows.append({"rank": int(data["rank"]), **row})
    flagged = [r for r in rows if r.get("flag")]
    return {"factor": factor, "rows": rows, "flagged": flagged}


def _worst(rows: list[dict], key: str) -> float:
    return max((float(r.get(key, 0.0)) for r in rows), default=0.0)


def format_table(result: dict[str, Any]) -> str:
    """One line per (forward, layer, cache), worst over the ranks."""
    groups: dict[tuple, list[dict]] = {}
    for row in result["rows"]:
        groups.setdefault((row["forward"], row["layer"], row["cache"]), []).append(row)
    lines = [
        f"{'fwd':>3} {'layer':>5} {'pre':>3} {'cache':<16} {'rms_B':>10} "
        f"{'rel(A-B)':>10} {'max|A-B|':>10} {'rel(B2-B)':>10} {'max|B2-B|':>10} "
        f"{'ratio':>9} flag"
    ]
    for (forward, layer, cache), group in sorted(groups.items()):
        pre = "y" if group[0]["consumed"] else "n"
        flag = "FLAG" if any(r.get("flag") for r in group) else ""
        if cache == "raw_pos":
            ok = all(
                r["ab_equal"] and r["bb_equal"] and r["a_rows_equal"] for r in group
            )
            state = "equal" if ok else "DIFFER"
            lines.append(
                f"{forward:>3} {layer:>5} {pre:>3} {cache:<16} positions {state} "
                f"(A=B, B=B2, A=rows) {flag}"
            )
            continue
        lines.append(
            f"{forward:>3} {layer:>5} {pre:>3} {cache:<16} "
            f"{_worst(group, 'rms'):>10.4g} {_worst(group, 'ab_rel'):>10.3g} "
            f"{_worst(group, 'ab_max'):>10.3g} {_worst(group, 'bb_rel'):>10.3g} "
            f"{_worst(group, 'bb_max'):>10.3g} {_worst(group, 'ratio'):>9.3g} {flag}"
        )
    lines.append(f"{len(result['flagged'])} flagged rows (factor {result['factor']:g})")
    return "\n".join(lines)


def _compare(args) -> int:
    path = Path(args.dir)
    result = compare_dir(path, args.factor)
    out = Path(args.json) if args.json else path / "compare.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(format_table(result))
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="dump the caches on the GPUs")
    run.add_argument("--model", required=True)
    run.add_argument("--windows", required=True, help="ppl_long.py windows JSON")
    run.add_argument("--window", type=int, default=0)
    run.add_argument("--map", required=True, help="VLLM_RDNA_RIDGEFILL_MAP")
    run.add_argument("--pins", required=True, help="VLLM_RDNA_RIDGEFILL_PINS")
    run.add_argument("--map-fused", type=int, default=0)
    run.add_argument("--tail", type=int, default=0, help="0 keeps the default")
    run.add_argument("--out", required=True)
    run.add_argument("--forwards", type=int, default=2)
    run.add_argument("--skip", type=int, default=0)
    run.add_argument("--tp", type=int, default=4)
    run.add_argument("--max-model-len", type=int, default=262144)
    run.add_argument("--block-size", type=int, default=1024)
    run.add_argument("--max-num-seqs", type=int, default=8)
    run.add_argument("--max-num-batched-tokens", type=int, default=2048)
    run.add_argument("--kv-cache-memory-bytes", type=int, default=4026531840)
    run.add_argument("--attention-backend", default="RDNA_ATTN")
    run.add_argument("--compilation-config", default=COMPILATION)
    run.add_argument("--extension", default=EXTENSION)
    cmp = sub.add_parser("compare", help="compare a dump on the CPU")
    cmp.add_argument("dir")
    cmp.add_argument("--factor", type=float, default=FACTOR)
    cmp.add_argument("--json", default="")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    torch = _torch()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
