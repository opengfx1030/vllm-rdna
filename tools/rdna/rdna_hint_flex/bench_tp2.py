# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-path kernel timing on the real per-rank shapes of an rdna_hint_flex
checkpoint under tensor parallelism (cache-cold, CUDA graph).

Every fused linear of the model (qkv, o, GDN in_proj / out_proj, gate_up, down)
is built from the checkpoint's own tensors (real row nbits / block mix and
transforms), sliced for one rank exactly as the loader does. The timed region
is a CUDA graph that cycles through enough distinct weight sets that the set is
larger than the 128 MB Infinity Cache, so the kernel streams its weights from
DRAM as it does in a decode step. The sets are the layers that have the slot
(both layer kinds for gate_up / down) in a fixed low-discrepancy order: the
share of INT4 rows differs a lot between layers, so a prefix of layers would
not represent the model, and an order that varied between runs would make two
runs incomparable. Time per call is graph time / calls (the mean weight bytes
per call are reported); each cell is repeated and min / median over the
repeats are printed with GB/s and the share of the 512 GB/s HBM floor (the
rate these kernels can actually stream at is lower, see the weights doc).

  HIP_VISIBLE_DEVICES=4 python tools/rdna/rdna_hint_flex/bench_tp2.py MODEL \
      [--tp 2] [--rank 0] [--m 1,2,3,4,8] [--slots gate_up_proj,down_proj] \
      [--variants linear,gemv,transform,linear_i8] [--repeats 5] [--json out]

Variants: ``linear`` (the serving op: transform + GEMV, mode 0), ``gemv`` (GEMV
on a pre-transformed activation, with the kernel RDNA_HINT_FLEX_GEMV selects),
``transform`` (the activation transform alone), ``linear_i8`` (mode 1),
``g2:MT:RPW:WPB`` / ``g2p:MT:RPW:WPB`` (GEMV v2 with an explicit config on a
natural / pair-interleaved activation), ``l2:MT:RPW:WPB`` (what the serving op
runs with v2: the pair-interleaving transform, then GEMV v2), plus any
registered by RDNA_HINT_FLEX_AB (see VARIANTS). ``{mt}`` in a variant stands
for the token tile the v2 instantiations offer at that M (M up to 4, then 6,
then 8). ``--check "MT:RPW:WPB;..."`` compares v2 with v1 and with the exact
float64 decode on the real tensors of the layers in ``--check-layers`` (default:
the first in bench order; RDNA_HINT_FLEX_GEMV must be unset). ``--dense
lm_head`` times the dense fp16 head GEMV of the rank next to the T45 int8 head.
``--stream`` times the dense fp16 GEMV over cold weights of 4 MiB to 1.27 GiB:
what a streaming kernel reaches at the working-set size of each hi2 GEMV. A
timing window is rerun while this process's KFD eviction counter moved during
it (the evicted time is the host's, not the kernel's); the run ends with the
counter and the number of reruns.
"""

import argparse
import functools
import glob
import json
import os
import statistics

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu

try:  # run as a script from the repo root or from this directory
    from tp_layout import (
        Reader,
        cols_of,
        dense_bytes,
        part_rows,
        slot_specs,
        text_config,
    )
except ImportError:  # pragma: no cover
    from tools.rdna.rdna_hint_flex.tp_layout import (
        Reader,
        cols_of,
        dense_bytes,
        part_rows,
        slot_specs,
        text_config,
    )

IC_BYTES = 128 << 20  # V620 Infinity Cache
COLD_BYTES = 640 << 20  # weight working set per cell: well beyond the cache
DECODE_MAX_M = 48


def build_segments(rd: Reader, spec: dict, rank: int) -> list[hu.Hi2]:
    """The loader's segments of one fused linear for this rank (CPU)."""
    k0, k1 = cols_of(spec, rank)
    k_part = k1 - k0
    cache: dict[str, hu.Hi2] = {}
    parts: list[hu.Hi2] = []
    for p in spec["parts"]:
        m = p["module"]
        if m not in cache:
            buf = {f: rd.get(f"{m}.hi2_{f}") for f in hu.FIELDS}
            if rd.has(f"{m}.hi2_col_scale"):
                buf["col_scale"] = rd.get(f"{m}.hi2_col_scale")
            t = hu.canonicalize(buf)
            hu.validate(t)
            cache[m] = t
        r0, n = part_rows(p, rank)
        parts.append(hu.select_rows(cache[m], r0, r0 + n))
    if k_part != spec["k_full"]:
        parts = [hu.select_cols(p, k0, k1) for p in parts]
    groups: list[list[hu.Hi2]] = []
    for p in parts:
        hu.validate(p, k_part)
        if groups and hu.same_transform(groups[-1][0], p):
            groups[-1].append(p)
        else:
            groups.append([p])
    return [hu.concat_rows(g) for g in groups]


def to_device(t: hu.Hi2) -> dict:
    d = {f: t[f].contiguous().cuda() for f in hu.FIELDS}
    d["col_scale"] = (
        t["col_scale"].contiguous().cuda()
        if t.get("col_scale") is not None
        else torch.empty(0, dtype=torch.float16, device="cuda")
    )
    d["transform"] = hu.needs_transform(t)
    d["R"] = hu.num_rows(t)
    d["K"] = hu.num_cols(t)
    d["bytes"] = sum(
        v.numel() * v.element_size() for v in d.values() if isinstance(v, torch.Tensor)
    )
    return d


def clone_set(d: dict) -> dict:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def weights(d: dict) -> tuple:
    """The seven weight tensors the GEMV ops take, in argument order."""
    return (d["codes"], d["code_off"], d["blk_params"], d["param_off"],
            d["row_consts"], d["row_nbits"], d["row_block"])  # fmt: skip


def graph_us(calls: list, reps: int) -> float:
    """GPU time per call of one CUDA graph that runs `calls` `reps` times."""
    for c in calls:
        c()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for c in calls:
                c()
    g.replay()
    torch.accelerator.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * len(calls))


def kfd_evicted_ms() -> int | None:
    """KFD queue-eviction time of this process in ms (None: no KFD stats)."""
    vals = []
    for path in glob.glob(f"/sys/class/kfd/kfd/proc/{os.getpid()}/stats_*/evicted_ms"):
        try:
            with open(path) as f:
                vals.append(int(f.read()))
        except (OSError, ValueError):
            pass
    return max(vals) if vals else None


WINDOWS_RERUN = 0


def timed(calls: list, reps: int, repeats: int) -> list[float]:
    """`repeats` timings of graph_us(calls, reps), rerun while a KFD eviction hit
    the window (such a window measures the host, not the kernel)."""
    global WINDOWS_RERUN
    for _ in range(5):
        e0 = kfd_evicted_ms() or 0
        ts = [graph_us(calls, reps) for _ in range(repeats)]
        e1 = kfd_evicted_ms() or 0
        if e0 == e1:
            return ts
        WINDOWS_RERUN += 1
        print(f"  (KFD eviction in this window: +{e1 - e0} ms: rerun)", flush=True)
    raise SystemExit("KFD evictions keep hitting the timing window")


def make_linear(d: dict, x: torch.Tensor, mode: int):
    return functools.partial(
        ops.rdna_hint_flex_linear_rdna2,
        x, d["col_scale"], d["had"], d["codes"], d["code_off"], d["blk_params"],
        d["param_off"], d["row_consts"], d["row_nbits"], d["row_block"],
        d["transform"], DECODE_MAX_M, mode,
    )  # fmt: skip


def make_gemv(d: dict, x: torch.Tensor):
    out = torch.empty(x.shape[0], d["R"], dtype=torch.float16, device="cuda")
    return functools.partial(ops.rdna_hint_flex_gemv_rdna2, x, *weights(d), out)


def pi_permute(x: torch.Tensor) -> torch.Tensor:
    """Natural order -> pair-interleaved: stored[2i] = x[i], stored[2i+1] = x[i+8]
    inside every 16-column group."""
    g = x.reshape(*x.shape[:-1], -1, 16)
    return torch.stack([g[..., :8], g[..., 8:]], dim=-1).reshape(x.shape)


def check_slot(seg0: hu.Hi2, d: dict, ms: list[int], cfgs: list[str]) -> None:
    """GEMV v2 vs v1 (and vs the exact decode in float64) on the real tensors
    of one layer. The activation is random and counted as pre-transformed."""
    if os.environ.get("RDNA_HINT_FLEX_GEMV", "") in ("v2", "2", "auto"):
        raise SystemExit("--check needs RDNA_HINT_FLEX_GEMV unset (v1 baseline)")
    r, k = d["R"], d["K"]
    nref = min(256, r)
    w_ref = hu.dequantize_stored(hu.select_rows(seg0, 0, nref)).double()
    for m in ms:
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        y1 = torch.empty(m, r, dtype=torch.float16, device="cuda")
        ops.rdna_hint_flex_gemv_rdna2(x, *weights(d), y1)
        ref = x.double().cpu() @ w_ref.T
        e1 = (y1[:, :nref].double().cpu() - ref).abs().max().item()
        scale = y1.abs().max().item()
        for c in cfgs:
            mt, rpw, wpb = (int(v) for v in c.split(":"))
            y2 = torch.empty_like(y1)
            y3 = torch.empty_like(y1)
            for y, xx, pi in ((y2, x, False), (y3, pi_permute(x), True)):
                ops.rdna_hint_flex_gemv2_rdna2(xx, *weights(d), y, mt, rpw, wpb, pi)
            torch.accelerator.synchronize()
            same = torch.equal(y2, y3)
            diff = (y1.double() - y2.double()).abs()
            e2 = (y2[:, :nref].double().cpu() - ref).abs().max().item()
            a, b = y1.view(torch.int16).int(), y2.view(torch.int16).int()
            ulp = torch.where((a >= 0) == (b >= 0), (a - b).abs(), 0).max().item()
            print(
                f"  check M={m:2d} cfg {c:7s} |v2-v1| max {diff.max().item():.3e} "
                f"({diff.max().item() / scale:.2e} of max|y|), "
                f"{int((diff > 0).sum())}/{diff.numel()} differ, max {ulp} ulp; "
                f"vs f64 ref (rows 0-{nref}): v1 {e1:.3e} v2 {e2:.3e}; "
                f"pi==nat {same}"
            )


def make_transform(d: dict, x: torch.Tensor):
    out = torch.empty_like(x)
    return functools.partial(
        ops.rdna_hint_flex_act_transform_rdna2, x, d["col_scale"], d["had"], out
    )


def make_g2(d: dict, x: torch.Tensor, mt=0, rpw=0, wpb=0, pi=False):
    """GEMV v2 on a pre-transformed activation (pi: pair-interleaved order)."""
    out = torch.empty(x.shape[0], d["R"], dtype=torch.float16, device="cuda")
    return functools.partial(
        ops.rdna_hint_flex_gemv2_rdna2, x, *weights(d), out, mt, rpw, wpb, pi
    )


def make_l2(d: dict, x: torch.Tensor, mt=0, rpw=0, wpb=0):
    """What rdna_hint_flex_linear_rdna2 launches with RDNA_HINT_FLEX_GEMV=v2:
    the pair-interleaving transform, then GEMV v2 (explicit config)."""
    xt = torch.empty_like(x)
    out = torch.empty(x.shape[0], d["R"], dtype=torch.float16, device="cuda")
    tr = d["transform"]

    def call():
        src = x
        if tr:
            ops.rdna_hint_flex_act_transform_rdna2(
                x, d["col_scale"], d["had"], xt, True
            )
            src = xt
        ops.rdna_hint_flex_gemv2_rdna2(src, *weights(d), out, mt, rpw, wpb, tr)

    return call


def get_variant(name: str):
    """`linear`, `gemv`, ... or `g2[p]:MT:RPW:WPB` / `l2:MT:RPW:WPB` (v2 with
    an explicit configuration; g2p reads a pair-interleaved activation)."""
    if name in VARIANTS:
        return VARIANTS[name]
    kind, *cfg = name.split(":")
    mt, rpw, wpb = (int(v) for v in cfg) if cfg else (0, 0, 0)
    if kind in ("g2", "g2p"):
        return lambda d, x: make_g2(d, x, mt, rpw, wpb, kind == "g2p")
    if kind == "l2":
        return lambda d, x: make_l2(d, x, mt, rpw, wpb)
    raise SystemExit(f"unknown variant {name}")


# name -> factory(set, x) -> callable. Extra variants for A/B are appended here.
VARIANTS = {
    "linear": lambda d, x: make_linear(d, x, 0),
    "linear_i8": lambda d, x: make_linear(d, x, 1),
    "gemv": make_gemv,
    "transform": make_transform,
}


def bench_cell(sets: list[dict], variant: str, m: int, repeats: int) -> list[float]:
    k = sets[0]["K"]
    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
    calls = [get_variant(variant)(d, x) for d in sets]
    reps = max(1, -(-24 // len(calls)))
    return timed(calls, reps, repeats)


def stream_rates(repeats: int) -> None:
    """Read-only reference: the dense fp16 GEMV (M=1, the lm_head kernel) over
    cold weights of 4 MiB to 1.27 GiB per call. No kernel of the hi2 path: it
    shows what a streaming kernel reaches at that working-set size."""
    k = 5120
    pool = torch.randn((1400 << 20) // (2 * k), k, dtype=torch.float16, device="cuda")
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")
    for mb in (4, 8, 16, 32, 64, 256, 1270):
        r = ((mb << 20) // (2 * k)) // 64 * 64
        slices = [pool[i * r : (i + 1) * r] for i in range(pool.shape[0] // r)]
        need = min(len(slices), max(3, -(-COLD_BYTES // (r * k * 2))))
        calls = [
            functools.partial(ops.gemv_f16_rdna2, x, s, None) for s in slices[:need]
        ]
        ts = timed(calls, max(1, -(-24 // len(calls))), repeats)
        print(
            f"stream R={r:6d} K={k} {r * k * 2 / 1e6:7.1f} MB per call "
            f"({need} sets): min {min(ts):8.1f} us med "
            f"{statistics.median(ts):8.1f} us  {r * k * 2 / min(ts) / 1e3:6.1f} GB/s",
            flush=True,
        )
    del pool
    torch.accelerator.empty_cache()


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--m", default="1,2,3,4,8")
    ap.add_argument("--slots", default="")
    ap.add_argument("--variants", default="linear,gemv,transform")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--layers", type=int, default=0, help="distinct layers (0=auto)")
    ap.add_argument("--dense", default="", help="lm_head: dense fp16 head GEMV")
    ap.add_argument("--stream", action="store_true", help="read-only reference rates")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--check",
        default="",
        help="compare these g2 configs (';'-separated MT:RPW:WPB) with v1 on the "
        "real tensors",
    )
    ap.add_argument(
        "--check-layers",
        default="",
        help="real layer indices for --check (default: the first in bench order)",
    )
    args = ap.parse_args()
    ms = [int(v) for v in args.m.split(",")]
    variants = args.variants.split(",")
    cfg = text_config(args.model)
    rd = Reader(args.model)
    want = set(filter(None, args.slots.split(",")))
    p = torch.cuda.get_device_properties(0)
    print(
        f"device {p.name} {p.gcnArchName} bus {p.pci_bus_id:#x}; TP={args.tp} "
        f"rank={args.rank}; cold set >= {COLD_BYTES >> 20} MiB per cell "
        f"(IC {IC_BYTES >> 20} MiB)"
    )
    rows: list[dict] = []
    if args.stream:
        stream_rates(args.repeats)
        return

    if args.dense:
        d = dense_bytes(rd, cfg, args.tp)
        n, k = cfg["vocab_size"] // args.tp, cfg["hidden_size"]
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        # T45 shadow (VLLM_RDNA_DENSE_INT8): per-row int8 + fp16 scale. Timing
        # does not depend on the values.
        w8 = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        sc = (1e-4 + 1e-3 * torch.rand(n, device="cuda")).to(torch.float16)
        for m in ms:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            f16 = functools.partial(ops.gemv_f16_rdna2, x, w, None)
            cases = [("fp16 gemv_f16_rdna2", f16, n * k * 2)]
            if m <= 8:
                i8 = functools.partial(ops.gemv_i8_rdna2, x, w8, sc, None)
                cases.append(("T45 gemv_i8_rdna2 ", i8, n * k + 2 * n))
            for name, call, nbytes in cases:
                ts = timed([call], 4, args.repeats)
                t, med = min(ts), statistics.median(ts)
                print(
                    f"lm_head R={n} K={k} M={m} {name}: min {t:9.1f} us "
                    f"med {med:9.1f} us {nbytes / t / 1e3:6.1f} GB/s "
                    f"({nbytes / 1e9:.3f} GB; dense fp16 head "
                    f"{d['lm_head'] / 1e9:.3f} GB)"
                )
                rows.append(
                    {
                        "slot": "lm_head",
                        "variant": name.strip(),
                        "M": m,
                        "min_us": t,
                        "med_us": med,
                        "gbs": nbytes / t / 1e3,
                    }
                )
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return

    # Every layer that has a slot (both layer kinds for gate_up / down), taken
    # in a fixed low-discrepancy order until the working set is cold: the share
    # of INT4 rows differs a lot between layers, so a prefix of layers would
    # not represent the model (and a set order that varies between runs would
    # make two runs incomparable).
    layers_of_slot: dict[str, list[int]] = {}
    for i in range(len(cfg["layer_types"])):
        for s in slot_specs(cfg, args.tp, i):
            layers_of_slot.setdefault(s["slot"], []).append(i)
    for slot, layers in layers_of_slot.items():
        if want and slot not in want:
            continue
        order = sorted(layers, key=lambda i: (i * 0.6180339887498949) % 1.0)
        if args.check:
            ids = [int(v) for v in args.check_layers.split(",") if v] or order[:1]
            for layer in ids:
                if layer not in layers:
                    continue  # this slot does not exist in that layer kind
                spec = next(
                    s for s in slot_specs(cfg, args.tp, layer) if s["slot"] == slot
                )
                seg = build_segments(rd, spec, args.rank)[0]
                d = to_device(seg)
                print(f"\n{slot} layer {layer}: R={d['R']} K={d['K']}")
                check_slot(seg, d, ms, args.check.split(";"))
                del d
            torch.accelerator.empty_cache()
            continue
        sets: list[dict] = []
        total = 0
        for layer in order:
            spec = next(s for s in slot_specs(cfg, args.tp, layer) if s["slot"] == slot)
            segs = build_segments(rd, spec, args.rank)
            if len(segs) != 1:
                raise SystemExit(f"{slot}: {len(segs)} segments (bench expects 1)")
            sets.append(to_device(segs[0]))
            total += sets[-1]["bytes"]
            if len(sets) >= args.layers if args.layers else total >= COLD_BYTES:
                break
        base = list(sets)
        need = args.layers or max(3, -(-COLD_BYTES // (total // len(sets))))
        while len(sets) < need:  # fewer layers than the cold set needs: clone
            sets.append(clone_set(base[len(sets) % len(base)]))
        per_layer = sum(d["bytes"] for d in sets) / len(sets)  # mean bytes per call
        r, k = sets[0]["R"], sets[0]["K"]
        print(
            f"\n{slot}: R={r} K={k} {per_layer / 1e6:.2f} MB/call (mean), "
            f"{len(sets)} sets ({len(sets) * per_layer / 1e6:.0f} MB, "
            f"{len(base)} real layers)"
        )
        for m in ms:
            for v0 in variants:
                # {mt}: tokens per block the v2 instantiations offer for this M
                v = v0.replace("{mt}", str(m if m <= 4 else (6 if m <= 6 else 8)))
                try:
                    ts = bench_cell(sets, v, m, args.repeats)
                except RuntimeError as e:
                    if "not instantiated" not in str(e):
                        raise
                    continue
                t, med = min(ts), statistics.median(ts)
                row = {
                    "slot": slot,
                    "R": r,
                    "K": k,
                    "M": m,
                    "variant": v,
                    "calls": len(layers),
                    "min_us": t,
                    "med_us": med,
                    "bytes": per_layer,
                }
                gbs = per_layer / t / 1e3
                row["gbs"] = gbs
                row["pct512"] = 100 * (per_layer / 512e3) / t
                print(
                    f"  M={m:2d} {v:10s} min {t:8.1f} us  med {med:8.1f} us  "
                    f"{gbs:6.1f} GB/s  {row['pct512']:5.1f}% of 512GB/s floor "
                    f"({per_layer / 512e3:6.1f} us)"
                )
                rows.append(row)
        del sets, base
        torch.accelerator.empty_cache()
    nslots = len({r["slot"] for r in rows})
    if nslots > 1:
        # Whole-step estimate: launches per step x min time per call, summed
        # over the slots that were run with the same variant and M.
        tot: dict[tuple[str, int], list[float]] = {}
        for r in rows:
            t = tot.setdefault((r["variant"], r["M"]), [0.0, 0.0, 0])
            t[0] += r["calls"] * r["min_us"]
            t[1] += r["calls"] * r["bytes"]
            t[2] += 1
        print("\nWhole step (sum over slots of launches per step x min us per call):")
        for (v, m), (us, nbytes, n) in sorted(
            tot.items(), key=lambda kv: (kv[0][1], kv[0][0])
        ):
            part = "" if n == nslots else f"  (only {int(n)}/{nslots} slots)"
            print(
                f"  M={m:2d} {v:12s} {us / 1e3:7.2f} ms  "
                f"{nbytes / us / 1e3:6.1f} GB/s{part}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


def main() -> None:
    try:
        run()
    finally:
        print(
            f"kfd evicted_ms (this process, loading included): {kfd_evicted_ms()}; "
            f"timing windows rerun because of an eviction: {WINDOWS_RERUN}"
        )


if __name__ == "__main__":
    main()
