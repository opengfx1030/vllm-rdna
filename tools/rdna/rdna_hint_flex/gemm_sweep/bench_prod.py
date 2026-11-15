# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The shipped dense prefill GEMM op against rocBLAS, one interleaved window per cell.

Contenders of a cell (shape, M), all on cold rotating weights (>= ``--ws-mb``) and
launched through one CUDA graph each, timed in interleaved rounds with the
hwmon clock / power sampler of ``sweep.py`` (the measurement code is imported
from there, so the numbers are comparable with the sweep CSVs):

  rocblas_heur    torch.mm, TunableOp off (what serving runs without rows)
  rocblas_tuned   torch.mm, TunableOp on with the rows of ``--rows`` (a cell
                  without a row is tuned first when ``--tune-missing``, otherwise
                  the contender is skipped)
  hip             ``torch.ops._rocm_C.rdna_hint_flex_dense_gemm_rdna2`` with each
                  design of ``--designs`` forced (kind ``hip``, variant = the sweep name)
  hip_auto        the same op in table mode (what ``RDNA_HINT_FLEX_PREFILL_GEMM=hip``
                  runs; the pick is in the note)

    HIP_VISIBLE_DEVICES=5 python bench_prod.py --shapes all --m 512,1024,2048 \\
        --rows ../../../../tunableop/rocm7.14-rocblas5.5-hi2-27b-tp2/tunableop_results0.csv \\
        --csv prod.csv --label slotA

``gen_dispatch.py`` turns such CSVs into the dispatch table; ``report_prod.py``-style
tables are made by the same columns as ``sweep.py``'s CSVs (``vs_rb`` = best
rocBLAS row / this row).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ctypes
import importlib
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sweep = importlib.import_module("sweep")
gen = importlib.import_module("gen_dispatch")


def merge_rows(paths: list[Path], out: Path) -> int:
    """One TunableOp CSV from several (validators of the first, rows deduplicated by
    key, later files win)."""
    validators: list[list[str]] = []
    rows: dict[str, list[str]] = {}
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.reader(fh):
                if not r:
                    continue
                if r[0] != "Validator":
                    rows[f"{r[0]},{r[1]}"] = r
                elif p == paths[0] and r not in validators:
                    validators.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerows(validators)
        w.writerows(rows.values())
    return len(rows)


def has_row(tunable, n: int, m: int, k: int) -> str | None:
    for r in tunable.get_results():
        if f"_{n}_{m}_{k}_" in r[1]:
            return r[2]
    return None


# The shipped designs (kDesigns in rdna_hint_flex_dense_gemm_rdna2.cu) as the
# -DDEV_* values of dense_dev.hip; "D1+brot=1+rast=4" is D1 with BROT on and RAST 4.
DEV_BASES = {
    "D1": dict(bm=128, bn=128, bk=32, tm=8, tn=16, nbuf=2, pad=0, rast=8, lw=8, swz=1, ku=1),
    "D2": dict(bm=128, bn=128, bk=32, tm=8, tn=16, nbuf=2, pad=0, rast=4, lw=8, swz=1, ku=1),
    "D3": dict(bm=128, bn=64, bk=64, tm=8, tn=8, nbuf=1, pad=8, rast=4, lw=8, swz=0, ku=1),
    "D4": dict(bm=256, bn=128, bk=32, tm=8, tn=16, nbuf=2, pad=8, rast=4, lw=8, swz=0, ku=1),
}  # fmt: skip
DEV_KNOBS = dict(brot=0, skip=0, prio=0, dma=0, buf=0)


def parse_dev(spec: str) -> dict[str, int]:
    base, *kv = spec.split("+")
    d = {**DEV_BASES[base], **DEV_KNOBS}
    for item in kv:
        key, val = item.split("=")
        if key not in d:
            raise SystemExit(f"{spec}: unknown knob {key}")
        d[key] = int(val)
    return d


def compile_dev(spec: str, d: dict[str, int], so: Path) -> tuple[str, int, str]:
    """The library and, next to it, the device ISA (<name>.s) for isa_mix.py."""
    base = [
        "nice", "-n", "19", sweep.HIPCC, "-O3", "-std=c++17", "--offload-arch=gfx1030",
        "-Rpass-analysis=kernel-resource-usage",
        *[f"-DDEV_{k.upper()}={v}" for k, v in d.items()],
        str(HERE / "dense_dev.hip"),
    ]  # fmt: skip
    p = subprocess.run(
        [*base, "-shared", "-fPIC", "-o", str(so)], capture_output=True, text=True
    )
    so.with_suffix(".remarks").write_text(p.stderr)
    if p.returncode == 0:
        subprocess.run(
            [*base, "--offload-device-only", "-S", "-o", str(so.with_suffix(".s"))],
            capture_output=True,
            text=True,
        )
    return spec, p.returncode, p.stderr[-2000:] if p.returncode else ""


class DevLib:
    """A dense_dev.hip build: the design ``spec`` through ctypes."""

    def __init__(self, torch, spec: str, d: dict[str, int], so: Path):
        self.torch, self.spec, self.bk = torch, spec, d["bk"]
        self.lib = ctypes.CDLL(str(so))
        self.lib.variant_launch.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        )
        self.lib.variant_launch.restype = ctypes.c_int
        info = (ctypes.c_int * 8)()
        self.lib.variant_info(info)
        self.vgpr, self.lds, self.scratch, self.threads, self.blocks = list(info)[:5]

    def launch(self, a, w, c) -> None:
        m, k = a.shape
        rc = self.lib.variant_launch(
            a.data_ptr(), w.data_ptr(), c.data_ptr(), m, w.shape[0], k,
            a.stride(0), w.stride(0), c.stride(0),
            self.torch.cuda.current_stream().cuda_stream,
        )  # fmt: skip
        if rc != 0:
            raise RuntimeError(f"{self.spec}: launch rc={rc}")


def build_devs(torch, specs: list[str], force: bool) -> list[DevLib]:
    out = Path(os.environ.get("SWEEP_OUT", str(sweep.OUT))) / "dev"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for spec in specs:
        so = out / (spec.replace("+", "_").replace("=", "") + ".so")
        items.append((spec, parse_dev(spec), so))
    todo = [it for it in items if force or not it[2].exists()]
    if todo:
        print(f"{len(items)} dev designs, {len(todo)} to compile", flush=True)
        with (
            sweep.build_lock(),
            concurrent.futures.ThreadPoolExecutor(max_workers=min(len(todo), 8)) as ex,
        ):
            for spec, rc, msg in ex.map(lambda it: compile_dev(*it), todo):
                print(f"{'ok  ' if rc == 0 else 'FAIL'} {spec} {msg}", flush=True)
    return [DevLib(torch, spec, d, so) for spec, d, so in items if so.exists()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--shapes", default="all", help="comma list of slot names (sweep.SHAPES) or all"
    )
    ap.add_argument("--m", default="512,700,1024,1500,2048")
    ap.add_argument(
        "--designs", default="0,1,2,3", help="forced designs to time ('' = none)"
    )
    ap.add_argument("--no-auto", action="store_true")
    ap.add_argument(
        "--dev", default="", help="comma list of dense_dev.hip designs, e.g. D1+brot=1"
    )
    ap.add_argument("--dev-force", action="store_true", help="recompile the --dev libs")
    ap.add_argument(
        "--build-only", action="store_true", help="compile the --dev libs and exit"
    )
    ap.add_argument(
        "--rows", default="", help="comma list of TunableOp CSVs to merge and read"
    )
    ap.add_argument("--tune-missing", action="store_true")
    ap.add_argument("--tune-ms", type=float, default=300)
    ap.add_argument("--tune-iters", type=int, default=30)
    ap.add_argument("--rot-mb", type=int, default=640)
    ap.add_argument("--ws-mb", type=float, default=640)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--win-ms", type=float, default=60)
    ap.add_argument("--warm-s", type=float, default=2.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--label", default="prod")
    ap.add_argument(
        "--check-only", action="store_true", help="correctness vs fp32, no timing"
    )
    args = ap.parse_args()

    import torch
    import torch.cuda.tunable as tunable

    from vllm import _custom_ops as ops

    print("device:", sweep.device_check(torch), flush=True)
    merged = (
        Path(os.environ.get("SWEEP_OUT", str(sweep.OUT))) / "tunableop_prod_rows.csv"
    )
    paths = [Path(p) for p in args.rows.split(",") if p]
    nrows = merge_rows(paths, merged) if paths else 0
    if not paths:
        merged.parent.mkdir(parents=True, exist_ok=True)
        merged.write_text("")
    tunable.set_filename(str(merged), insert_device_ordinal=False)
    if nrows:
        tunable.read_file(str(merged))
    tunable.enable(False)
    print(
        f"{nrows} TunableOp rows merged from {len(paths)} files -> {merged}", flush=True
    )

    designs = [int(d) for d in args.designs.split(",") if d != ""]
    devs = build_devs(torch, [s for s in args.dev.split(",") if s], args.dev_force)
    if args.build_only:
        return
    wr = sweep.Writer(args.csv)
    shapes = {
        k: s
        for k, s in sweep.SHAPES.items()
        if args.shapes == "all" or k in args.shapes.split(",")
    }
    ms = [int(m) for m in args.m.split(",")]
    for name, (n, k) in shapes.items():
        nsets = sweep.sets_for(n, k, args.ws_mb)
        for m in ms:
            x, wsets = sweep.make_data(torch, m, n, k, nsets)
            flop = 2.0 * m * n * k
            c = torch.empty(m, n, dtype=torch.float16, device="cuda")
            ref32 = torch.empty(m, n, dtype=torch.float32, device="cuda")
            for lo in range(0, m, 512):  # fp32 truth in row chunks
                ref32[lo : lo + 512] = x[lo : lo + 512].float() @ wsets[0].float().t()
            common = dict(shape=name, N=n, K=k, M=m)

            def vs32(out, ref32=ref32):
                return sweep.errors(torch, out, ref32)[1]

            def mm_cold(x=x, c=c, wsets=wsets):
                return [lambda w=w: torch.mm(x, w.t(), out=c) for w in wsets]

            cons = []
            tunable.enable(False)
            torch.mm(x, wsets[0].t(), out=c)
            heur = sweep.Contender(
                "rocblas_heur", "default", mm_cold(), flop, tun=False
            )
            heur.err = (None, None, vs32(c))
            cons.append(heur)
            sol = has_row(tunable, n, m, k)
            if sol is None and args.tune_missing:
                sweep.tune_rocblas(
                    torch,
                    x,
                    wsets[0],
                    tunable,
                    args.tune_ms,
                    args.tune_iters,
                    args.rot_mb,
                )
                sweep.save_rows(tunable, merged)
                sol = has_row(tunable, n, m, k)
            tuned = None
            if sol is not None:
                tunable.enable(True)
                tunable.tuning_enable(False)
                torch.mm(x, wsets[0].t(), out=c)
                tunable.enable(False)
                tuned = sweep.Contender(
                    "rocblas_tuned", "tunableop", mm_cold(), flop, tun=True, note=sol
                )
                tuned.err = (None, None, vs32(c))
                cons.append(tuned)
            for d in designs:
                if k % (64 if d == 2 else 32):
                    continue
                try:
                    ops.rdna_hint_flex_dense_gemm_rdna2(x, wsets[0], c, d)
                    torch.accelerator.synchronize()
                except RuntimeError as ex:
                    wr.add(
                        phase=args.label,
                        kind="hip",
                        variant=gen.DESIGNS[d],
                        note=f"ERR {ex}",
                        **common,
                    )
                    continue
                con = sweep.Contender(
                    "hip", gen.DESIGNS[d],
                    [lambda w=w, d=d, x=x, c=c: ops.rdna_hint_flex_dense_gemm_rdna2(x, w, c, d) for w in wsets],
                    flop, note=f"design {d}",
                )  # fmt: skip
                con.err = (None, None, vs32(c))
                cons.append(con)
            for dv in devs:
                if k % dv.bk:
                    continue
                try:
                    dv.launch(x, wsets[0], c)
                    torch.accelerator.synchronize()
                except RuntimeError as ex:
                    wr.add(
                        phase=args.label,
                        kind="dev",
                        variant=dv.spec,
                        note=f"ERR {ex}",
                        **common,
                    )
                    continue
                con = sweep.Contender(
                    "dev", dv.spec,
                    [lambda w=w, dv=dv, x=x, c=c: dv.launch(x, w, c) for w in wsets],
                    flop, vgpr=dv.vgpr, lds=dv.lds, scratch=dv.scratch, threads=dv.threads,
                    blocks_per_wgp=dv.blocks, note=f"vgpr {dv.vgpr} lds {dv.lds}",
                )  # fmt: skip
                con.err = (None, None, vs32(c))
                cons.append(con)
            if not args.no_auto:
                pick = ops.rdna_hint_flex_dense_gemm_pick_rdna2(m, n, k)
                ops.rdna_hint_flex_dense_gemm_rdna2(x, wsets[0], c, -1)
                con = sweep.Contender(
                    "hip_auto", "auto",
                    [lambda w=w, x=x, c=c: ops.rdna_hint_flex_dense_gemm_rdna2(x, w, c, -1) for w in wsets],
                    flop, tun=False, note=f"pick {pick}",
                )  # fmt: skip
                con.err = (None, None, vs32(c))
                cons.append(con)
            if args.check_only:
                for con in cons:
                    print(
                        f"{con.kind:14s} {con.variant:40s} {name:13s} M={m:<5} rel_fp32 {con.err[2]:.2e}",
                        flush=True,
                    )
            else:
                clk = sweep.measure(
                    torch, tunable, cons, args.rounds, args.warm_s, args.win_ms
                )
                best_us = min(
                    statistics.median(con.times)
                    for con in cons
                    if con.kind in ("rocblas_heur", "rocblas_tuned")
                )
                heur_us = statistics.median(heur.times)
                for con in cons:
                    med = statistics.median(con.times)
                    row = dict(
                        phase=args.label, kind=con.kind, variant=con.variant, us=round(med, 1),
                        us_min=round(min(con.times), 1),
                        tflops=round(con.flop / med / 1e6, 2) if con.flop else None,
                        vs_rb=round(best_us / med, 3) if con.flop else None,
                        gfx_mhz=con.clk[0] or clk[0], power_w=con.clk[1] or clk[1], **common,
                    )  # fmt: skip
                    row["rel_fp32"] = (
                        f"{con.err[2]:.2e}"
                        if con.err and con.err[2] is not None
                        else None
                    )
                    row.update(con.extra)
                    if con.kind in ("hip", "hip_auto", "dev"):
                        row["note"] = (
                            f"{con.extra.get('note', '')} vs_heur {heur_us / med:.3f}"
                        )
                    wr.add(**row)
            x = wsets = c = ref32 = cons = None
            torch.cuda.memory.empty_cache()


if __name__ == "__main__":
    main()
