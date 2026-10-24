#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone BF16 DFlash2 body GEMV vs lookup-only tuned rocBLAS.

Builds only the adjacent HIP source as a shared library and launches on the
current PyTorch HIP stream. It does not register an op or alter vLLM dispatch.
Run after setting the same TunableOp filename/env as the serving workers.
"""

import argparse
import ctypes
import hashlib
import itertools
import os
import pathlib
import statistics
import subprocess
import tempfile

import torch
import torch.cuda.tunable as tunable
import torch.nn.functional as F

# Exact BF16 TN families from the spec2 tune36 shape source; N x K per rank.
# Excludes the shared FP16 62,080-row target head, which is optimized separately.
DEFAULT_SHAPES = (
    "2560x12800,3840x2560,2560x1920,2560x2560,1536x2560,2560x1024,640x2560,256x2560"
)


def build_kernel(hipcc: str, output_dir: pathlib.Path) -> pathlib.Path:
    source = pathlib.Path(__file__).with_suffix(".cu")
    path = output_dir / "dflash2_bf16_skinny_probe.so"
    subprocess.run(
        [
            hipcc,
            "-O3",
            "--offload-arch=gfx1030",
            "-mno-wavefrontsize64",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(path),
        ],
        check=True,
    )
    return path


def bench_graph(
    fn, repetitions: int = 25, ops_per_graph: int = 16
) -> tuple[float, float, float]:
    for _ in range(4):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(ops_per_graph):
            fn()
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(3):
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (repetitions * ops_per_graph))
    return statistics.median(samples), min(samples), max(samples)


def check_close(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    # Both results are BF16 with FP32 accumulation, but the reduction order
    # differs; a few BF16 ULPs near cancellation are expected.
    delta = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().clamp_min(0.05)
    max_abs = delta.max().item()
    max_scaled = (delta / scale).max().item()
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)
    return max_abs, max_scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipcc", default="/opt/rocm/core-7.14/bin/hipcc")
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--m", default="3,24")
    parser.add_argument("--row-tiles", default="0")
    parser.add_argument(
        "--shared-x", action="store_true", help="bench both global-X and shared-X"
    )
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--rows", type=pathlib.Path, required=True)
    args = parser.parse_args()
    row_tiles = [int(item) for item in args.row_tiles.split(",")]
    if not row_tiles or any(tile not in (0, 4, 8, 12) for tile in row_tiles):
        parser.error("row-tiles must contain 0 (automatic), 4, 8, or 12")
    if os.environ.get("PYTORCH_TUNABLEOP_ENABLED") != "1":
        raise RuntimeError("Set PYTORCH_TUNABLEOP_ENABLED=1 for the serving baseline")
    if os.environ.get("PYTORCH_TUNABLEOP_TUNING") != "0":
        raise RuntimeError("Set PYTORCH_TUNABLEOP_TUNING=0; this is lookup-only")
    if not os.environ.get("PYTORCH_TUNABLEOP_FILENAME"):
        raise RuntimeError("Set PYTORCH_TUNABLEOP_FILENAME to tuned rank rows")
    if not torch.cuda.is_available():
        raise RuntimeError("HIP device required")
    rows = args.rows.resolve(strict=True)
    tunable.set_filename(str(rows), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(False)
    rows_hash = hashlib.sha256(rows.read_bytes()).hexdigest()
    print(f"# TunableOp rows={rows} sha256={rows_hash}", flush=True)

    build_dir = tempfile.TemporaryDirectory(prefix="dflash2-bf16-gemv-")
    library = ctypes.CDLL(str(build_kernel(args.hipcc, pathlib.Path(build_dir.name))))
    launch = library.bf16_skinny_gemv_launch
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    launchers = {"global": launch}
    if args.shared_x:
        shared_launch = library.bf16_skinny_gemv_shared_x_launch
        shared_launch.argtypes = launch.argtypes
        shared_launch.restype = ctypes.c_int
        launchers["shared_x"] = shared_launch
    torch.manual_seed(20261020)
    print(
        "variant,M,N,K,row_tile,rocblas_us,rocblas_min_us,rocblas_max_us,"
        "probe_us,probe_min_us,probe_max_us,probe_over_rocblas,"
        "max_abs_vs_bf16_blas,max_scaled_vs_bf16_blas,"
        "max_abs_vs_fp32,max_scaled_vs_fp32",
        flush=True,
    )
    for m, row_tile in itertools.product(
        (int(item) for item in args.m.split(",")), row_tiles
    ):
        if not 1 <= m <= 24:
            raise ValueError("Prototype supports M=1..24")
        for shape in args.shapes.split(","):
            n, k = (int(item) for item in shape.split("x"))
            if k % 8:
                raise ValueError("K must be a multiple of eight for uint4 loads")
            # Identical operands across the row-tile sweep.
            torch.manual_seed(20261020 + m * 1000003 + n * 257 + k)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
            expected = F.linear(x, w)
            fp32_reference = F.linear(x.float(), w.float())
            reference_us, reference_min, reference_max = bench_graph(
                lambda x=x, w=w: F.linear(x, w), repetitions=args.repetitions
            )
            for variant, variant_launch in launchers.items():
                y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

                def candidate(
                    x=x,
                    w=w,
                    y=y,
                    m=m,
                    n=n,
                    k=k,
                    tile=row_tile,
                    variant=variant,
                    variant_launch=variant_launch,
                ) -> None:
                    # torch.cuda.graph temporarily changes the current stream.
                    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
                    status = variant_launch(
                        x.data_ptr(), w.data_ptr(), y.data_ptr(), m, n, k, tile, stream
                    )
                    if status != 0:
                        raise RuntimeError(
                            f"HIP {variant} launch failed: {status} for {m}x{n}x{k}"
                        )

                candidate()
                torch.accelerator.synchronize()
                max_abs, max_scaled = check_close(y, expected)
                fp32_delta = (y.float() - fp32_reference).abs()
                fp32_max_abs = fp32_delta.max().item()
                fp32_max_scaled = (
                    (fp32_delta / fp32_reference.abs().clamp_min(0.05)).max().item()
                )
                candidate_us, candidate_min, candidate_max = bench_graph(
                    candidate, repetitions=args.repetitions
                )
                print(
                    f"{variant},{m},{n},{k},{row_tile},{reference_us:.2f},"
                    f"{reference_min:.2f},{reference_max:.2f},{candidate_us:.2f},"
                    f"{candidate_min:.2f},{candidate_max:.2f},"
                    f"{candidate_us / reference_us:.3f},{max_abs:.5g},"
                    f"{max_scaled:.5g},{fp32_max_abs:.5g},{fp32_max_scaled:.5g}",
                    flush=True,
                )

    # This case exceeds FP16's finite range but stays ordinary BF16; it
    # guards against accidental FP16 conversion in a future prototype edit.
    x = torch.full((3, 256), 2.0**20, device="cuda", dtype=torch.bfloat16)
    w = torch.full((64, 256), 0.0625, device="cuda", dtype=torch.bfloat16)
    y = torch.empty((3, 64), device="cuda", dtype=torch.bfloat16)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for variant, variant_launch in launchers.items():
        for row_tile in row_tiles:
            status = variant_launch(
                x.data_ptr(), w.data_ptr(), y.data_ptr(), 3, 64, 256, row_tile, stream
            )
            if status != 0:
                raise RuntimeError(f"HIP {variant} full-range launch failed: {status}")
            torch.accelerator.synchronize()
            assert torch.isfinite(y).all().item()
            check_close(y, F.linear(x, w))
    print("BF16 full-range check: PASS", flush=True)

    # N=641 leaves an inactive wave in the last W2 CTA. All threads must
    # still cross the shared-X barriers; K=1920 ends with a partial K tile.
    x_tail = torch.randn(3, 1920, device="cuda", dtype=torch.bfloat16)
    w_tail = torch.randn(641, 1920, device="cuda", dtype=torch.bfloat16) * 0.02
    y_tail = torch.empty(3, 641, device="cuda", dtype=torch.bfloat16)
    for variant, variant_launch in launchers.items():
        for row_tile in (0, 4):
            status = variant_launch(
                x_tail.data_ptr(),
                w_tail.data_ptr(),
                y_tail.data_ptr(),
                3,
                641,
                1920,
                row_tile,
                stream,
            )
            if status != 0:
                raise RuntimeError(f"HIP {variant} tail launch failed: {status}")
            torch.accelerator.synchronize()
            check_close(y_tail, F.linear(x_tail, w_tail))
    print("Partial output wave/K tile/M row check: PASS", flush=True)
    build_dir.cleanup()


if __name__ == "__main__":
    main()
