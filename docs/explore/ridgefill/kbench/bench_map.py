# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the RidgeFill map GEMM variants against rocBLAS on one GPU.

The real shape: every prefill chunk (``MAXBAT=2048``) applies 24 maps
``[F, 10240] x [10240, 2560]`` to the same split state, F <= 2048 filled rows.
Arms per F, interleaved round-robin so they share the clock and thermal state:

* ``rocblas_per_layer``: 24 x ``F.linear`` (the serve path today);
* ``rocblas_stacked``: one ``F.linear`` on the 61440-row stacked map;
* each ``libmapk.so`` variant, per layer (24 launches) and stacked (one).
  int8 variants include the per-token activation quantization, once per
  forward. With ``--had 128`` (the default) that kernel also applies the
  orthonormal Hadamard rotation to every 128-wide K block, and the weights
  carry the same rotation (done once here, at load time in a serve), so
  per-row x per-token int8 survives the fitted maps' outlier columns.

Accuracy is checked on every layer against an fp32 GEMM of the same fp16
inputs (max abs and relative RMS; int8 includes its quantization error).
With ``--capture`` the maps and the split state are the real fitted maps and
captured activations; otherwise random data of the same scale.

    HIP_VISIBLE_DEVICES=3 python3 bench_map.py --lib ./libmapk.so \
        --capture ~/w4a8_runs/lanes/ridgefill/capture --f 2048 1024 512
"""

import argparse
import ctypes
import json
import statistics
from pathlib import Path

import torch

DIN, DOUT, LAYERS = 10240, 2560, 24


def load_lib(path: str):
    lib = ctypes.CDLL(path)
    lib.mapk_count.restype = ctypes.c_int
    lib.mapk_info.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.mapk_run.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.mapk_quant_rows.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    variants = []
    buf = ctypes.create_string_buffer(256)
    for i in range(lib.mapk_count()):
        lib.mapk_info(i, buf, 256)
        name, dot, bm, bn, bkb, threads = buf.value.decode().split()
        variants.append(
            dict(
                id=i,
                name=name,
                int8=int(dot) == 1,
                bm=int(bm),
                bn=int(bn),
                bkb=int(bkb),
                threads=int(threads),
            )
        )
    return lib, variants


def load_data(args, dev):
    if args.capture:
        cap = torch.load(args.capture / "capture.pt", map_location="cpu", mmap=True)
        maps = torch.load(args.capture / "maps.pt", map_location="cpu", mmap=True)
        x = cap["split_state"][args.row0 : args.row0 + max(args.f)].half()
        order = sorted(maps)
        w = torch.cat([maps[i][0] for i in order]).to(dev)
        b = torch.cat([maps[i][1] for i in order]).to(dev)
        return x.to(dev), w, b
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(max(args.f), DIN, generator=g) * 0.1).half().to(dev)
    w = (torch.randn(LAYERS * DOUT, DIN, generator=g) * 0.02).half().to(dev)
    b = (torch.randn(LAYERS * DOUT, generator=g) * 0.1).half().to(dev)
    return x, w, b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib", required=True)
    p.add_argument("--capture", type=Path, default=None)
    p.add_argument("--row0", type=int, default=0)
    p.add_argument("--f", type=int, nargs="+", default=[2048])
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--only", nargs="*", default=None, help="variant names")
    p.add_argument("--no-stacked", action="store_true")
    p.add_argument("--had", type=int, default=128, help="0: no rotation")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    dev = torch.device("cuda")
    lib, variants = load_lib(args.lib)
    if args.only:
        variants = [v for v in variants if v["name"] in args.only]
    x_all, w16, b16 = load_data(args, dev)
    stream = torch.cuda.current_stream().cuda_stream

    # int8 weights: optional block-Hadamard rotation of K, then per output row
    # absmax; done once (load time in a serve).
    wf = w16.float()
    hmat = None
    if args.had:
        hmat = torch.ones(1, 1, device=dev)
        while hmat.shape[0] < args.had:
            hmat = torch.cat(
                [torch.cat([hmat, hmat], 1), torch.cat([hmat, -hmat], 1)], 0
            )
        hmat /= args.had**0.5
        wf = (wf.view(wf.shape[0], DIN // args.had, args.had) @ hmat).view_as(wf)
    sw = (wf.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
    wq = torch.round(wf / sw[:, None]).clamp(-127, 127).to(torch.int8)
    sw = sw.contiguous()
    del wf
    results = {"f": {}, "variants": [v["name"] for v in variants]}

    def measure(f: int) -> dict:
        x = x_all[:f].contiguous()
        macs = f * DIN * DOUT * LAYERS
        out = torch.empty(LAYERS, f, DOUT, dtype=torch.float16, device=dev)
        xq = torch.empty(f, DIN, dtype=torch.int8, device=dev)
        sx = torch.empty(f, dtype=torch.float32, device=dev)

        def quant():
            rc = lib.mapk_quant_rows(
                x.data_ptr(),
                DIN * 2,
                f,
                DIN,
                xq.data_ptr(),
                sx.data_ptr(),
                args.had,
                stream,
            )
            assert rc == 0, rc

        def run(v, stacked):
            if v["int8"]:
                quant()
                xp, ldx, wp, ldw, kb = xq, DIN, wq, DIN, DIN
                sxp, swp = sx.data_ptr(), sw.data_ptr()
            else:
                xp, ldx, wp, ldw, kb = x, DIN * 2, w16, DIN * 2, DIN * 2
                sxp = swp = None
            if stacked:
                rc = lib.mapk_run(
                    v["id"],
                    xp.data_ptr(),
                    ldx,
                    wp.data_ptr(),
                    ldw,
                    f,
                    LAYERS * DOUT,
                    kb,
                    DOUT,
                    sxp,
                    swp,
                    b16.data_ptr(),
                    out.data_ptr(),
                    stream,
                )
                assert rc == 0, (v["name"], rc)
                return
            esz = 1 if v["int8"] else 2
            for layer in range(LAYERS):
                off = layer * DOUT
                rc = lib.mapk_run(
                    v["id"],
                    xp.data_ptr(),
                    ldx,
                    wp.data_ptr() + off * ldw,
                    ldw,
                    f,
                    DOUT,
                    kb,
                    DOUT,
                    sxp,
                    None if swp is None else swp + off * 4,
                    b16.data_ptr() + off * 2,
                    out[layer].data_ptr(),
                    stream,
                )
                assert rc == 0, (v["name"], rc, esz)

        def rocblas_per_layer():
            for layer in range(LAYERS):
                sl = slice(layer * DOUT, (layer + 1) * DOUT)
                torch.nn.functional.linear(x, w16[sl], b16[sl], out=None)

        def rocblas_stacked():
            torch.nn.functional.linear(x, w16, b16)

        arms = {"rocblas_per_layer": rocblas_per_layer}
        if not args.no_stacked:
            arms["rocblas_stacked"] = rocblas_stacked
        for v in variants:
            arms[v["name"] + "/layer"] = lambda v=v: run(v, False)
            if not args.no_stacked:
                arms[v["name"] + "/stacked"] = lambda v=v: run(v, True)
        if any(v["int8"] for v in variants):
            arms["int8_act_quant_only"] = quant

        # Accuracy: every layer against fp32 of the same fp16 inputs.
        err = {}
        refs = []
        for layer in range(LAYERS):
            sl = slice(layer * DOUT, (layer + 1) * DOUT)
            refs.append(x.float() @ w16[sl].float().T + b16[sl].float())
        for name, fn in arms.items():
            if not name.endswith(("/layer", "/stacked")):
                continue
            out.zero_()
            fn()
            torch.cuda.current_stream().synchronize()
            mx = 0.0
            num = 0.0
            den = 0.0
            for layer in range(LAYERS):
                d = out[layer].float() - refs[layer]
                mx = max(mx, float(d.abs().max()))
                num += float(d.square().sum())
                den += float(refs[layer].square().sum())
            err[name] = (mx, (num / den) ** 0.5)
        rb = [
            torch.nn.functional.linear(
                x, w16[i * DOUT : (i + 1) * DOUT], b16[i * DOUT : (i + 1) * DOUT]
            ).float()
            for i in range(LAYERS)
        ]
        mx = max(float((rb[i] - refs[i]).abs().max()) for i in range(LAYERS))
        num = sum(float((rb[i] - refs[i]).square().sum()) for i in range(LAYERS))
        den = sum(float(refs[i].square().sum()) for i in range(LAYERS))
        err["rocblas_per_layer"] = (mx, (num / den) ** 0.5)
        del rb, refs
        if "int8_act_quant_only" in arms:
            # The kernel's rotation and rounding against torch's.
            quant()
            xr = x.float()
            if hmat is not None:
                xr = (xr.view(f, DIN // args.had, args.had) @ hmat).view(f, DIN)
            d = xq.float() * sx[:, None] - xr
            err["int8_act_quant_only"] = (
                float(d.abs().max()),
                float(d.square().mean().sqrt() / xr.square().mean().sqrt()),
            )
            del xr, d

        # Timing: warm up, then round-robin so every arm sees the same drift.
        for fn in arms.values():
            fn()
            fn()
        torch.cuda.current_stream().synchronize()
        times = {k: [] for k in arms}
        for _ in range(args.rounds):
            for name, fn in arms.items():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                e.synchronize()
                times[name].append(s.elapsed_time(e))
        base = statistics.median(times["rocblas_per_layer"])
        print(f"\nF={f}  ({macs / 1e12:.3f} T-MAC per forward)")
        print(
            f"{'arm':38s} {'median ms':>10s} {'min':>8s} {'max':>8s} "
            f"{'T-MAC/s':>8s} {'x rocBLAS':>9s} {'max err':>9s} {'rel rms':>9s}"
        )
        rows = {}
        for name, ts in times.items():
            med = statistics.median(ts)
            tm = macs / (med * 1e-3) / 1e12 if name != "int8_act_quant_only" else 0
            e = err.get(name, (float("nan"), float("nan")))
            rows[name] = dict(
                median_ms=med,
                min_ms=min(ts),
                max_ms=max(ts),
                tmacs=tm,
                speedup=base / med,
                max_err=e[0],
                rel_rms=e[1],
            )
            print(
                f"{name:38s} {med:10.2f} {min(ts):8.2f} {max(ts):8.2f} "
                f"{tm:8.2f} {base / med:9.3f} {e[0]:9.2e} {e[1]:9.2e}"
            )
        return rows

    for f in args.f:
        results["f"][f] = measure(f)
    if args.json:
        sha = Path(__file__).resolve().parents[4] / ".sha"
        results["sha"] = sha.read_text().strip() if sha.exists() else None
        results["device"] = torch.cuda.get_device_name()
        results["had"] = args.had
        results["capture"] = str(args.capture) if args.capture else None
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
