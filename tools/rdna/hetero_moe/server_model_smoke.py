# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test of ``ExpertServerModel`` on one V620 (gfx1030). Needs a GPU.

Loads the experts of a few layers from the Flash-Next W4A16 checkpoint, runs the
known-answer selftest (fp32 CPU reference from the dequantized checkpoint against
the production chain), then times ``moe()`` per layer and row count with CUDA
events (eager, and with ``--graph`` replayed from a CUDA graph, checking the
replay against the eager result). Exit status 0 only when everything passed.

    HIP_VISIBLE_DEVICES=<V620 hip index> PYTHONPATH=<tree> \\
      python -m tools.rdna.hetero_moe.server_model_smoke \\
        --model-dir <checkpoint snapshot dir> \\
        --layers 0,23,47 --experts-per-layer 16 --rows 3,24 --graph

``--experts-per-layer N`` holds the first N experts of each loaded layer
(``--expert-stride S`` takes every S-th instead: 0, S, 2S, ...). Layers not in
``--layers`` hold nothing and are not loaded. ``--local-frac F`` makes a pair local
with probability F in the timed batches (1.0: every pair runs on this server, the
worst case; the hetero plan sends about 2 to 3 experts per server and step at 3
rows).

The imported ``vllm`` path is printed first: the venv's editable install points at
another tree, so check it. On the box use ``venv-7.14.0`` (its torch carries the
gfx1030 and the gfx1100 device libraries) with ``PYTHONPATH`` at the exact tree.
Never run this on a W7800 (HIP 0 / 5): the process refuses any device that is not
gfx10.
"""

import argparse
import functools
import statistics
import sys
import time

DEFAULT_LAYERS = [0, 23, 47]


def _int_list(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not an int list: {text}") from err


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Usage and details: see the module docstring.",
    )
    p.add_argument("--model-dir", required=True, help="checkpoint directory")
    p.add_argument(
        "--layers",
        type=_int_list,
        default=DEFAULT_LAYERS,
        help="comma-separated layers to load (default 0,23,47)",
    )
    p.add_argument(
        "--experts-per-layer",
        type=int,
        default=16,
        metavar="N",
        help="experts held per loaded layer (default 16)",
    )
    p.add_argument(
        "--expert-stride",
        type=int,
        default=1,
        metavar="S",
        help="held expert ids are 0, S, 2S, ... (default 1)",
    )
    p.add_argument("--device", default="cuda:0", help="V620 device (default cuda:0)")
    p.add_argument(
        "--rows",
        type=_int_list,
        default=[3, 24],
        help="comma-separated batch row counts to time (default 3,24)",
    )
    p.add_argument("--num-layers", type=int, default=48, help="layers in the model")
    p.add_argument("--num-experts", type=int, default=512, help="experts per layer")
    p.add_argument("--top-k", type=int, default=10, help="experts per token")
    p.add_argument(
        "--max-capture-rows",
        type=int,
        default=48,
        help="graph capture size the scratch buffers are sized for",
    )
    p.add_argument("--group-size", type=int, default=128, help="quantization group")
    p.add_argument(
        "--local-frac",
        type=float,
        default=1.0,
        help="probability that a timed pair is local to this server (default 1)",
    )
    p.add_argument(
        "--selftest-experts",
        type=int,
        default=2,
        help="held experts per layer in the known-answer selftest (default 2)",
    )
    p.add_argument("--iters", type=int, default=20, help="calls per timed burst")
    p.add_argument("--reps", type=int, default=5, help="timed bursts per cell")
    p.add_argument("--warmup", type=int, default=3, help="untimed calls per cell")
    p.add_argument("--graph", action="store_true", help="also time CUDA graph replay")
    p.add_argument("--seed", type=int, default=1234, help="inputs and expert choice")
    args = p.parse_args(argv)
    if not args.layers or not args.rows:
        p.error("--layers and --rows need at least one value")
    if args.experts_per_layer < 1 or args.expert_stride < 1:
        p.error("--experts-per-layer and --expert-stride must be at least 1")
    if args.iters < 1 or args.reps < 1:
        p.error("--iters and --reps must be at least 1")
    if not 0.0 <= args.local_frac <= 1.0:
        p.error("--local-frac must be in [0, 1]")
    return args


def build_expert_sets(
    layers: list[int],
    per_layer: int,
    *,
    num_layers: int,
    num_experts: int,
    stride: int,
) -> list[list[int]]:
    """``expert_sets`` for the model: ids 0, S, 2S, ... in the loaded layers."""
    for layer in layers:
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer {layer} outside [0, {num_layers})")
    ids = [i * stride for i in range(per_layer)]
    if ids[-1] >= num_experts:
        raise ValueError(
            f"expert id {ids[-1]} outside [0, {num_experts}): lower "
            "--experts-per-layer or --expert-stride"
        )
    return [list(ids) if layer in layers else [] for layer in range(num_layers)]


def _median_min(samples: list[float]) -> tuple[float, float]:
    return statistics.median(samples), min(samples)


def make_batch(model, layer: int, rows: int, local_frac: float, seed: int):
    """Seeded ``x, ids, w`` on the model's device, plus pair and expert counts."""
    import torch

    gen = torch.Generator().manual_seed(seed + 7919 * layer + rows)
    held = torch.tensor(model.expert_sets[layer], dtype=torch.int32)
    width = min(model.top_k, len(held))
    ids = torch.full((rows, model.top_k), -1, dtype=torch.int32)
    for r in range(rows):
        ids[r, :width] = held[torch.randperm(len(held), generator=gen)[:width]]
    keep = torch.rand(rows, model.top_k, generator=gen) < local_frac
    ids = torch.where(keep, ids, torch.full_like(ids, -1))
    x = torch.randn(rows, model.hidden, generator=gen).to(torch.float16)
    w = torch.softmax(torch.randn(rows, model.top_k, generator=gen), dim=-1)
    pairs = int((ids >= 0).sum())
    touched = len({e for e in ids.flatten().tolist() if e >= 0})
    return x.to(model.device), ids.to(model.device), w.to(model.device), pairs, touched


def _burst_us(run, stream, iters: int, reps: int) -> list[float]:
    """Per-call microseconds of ``reps`` bursts of ``iters`` back-to-back calls."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record(stream)
        for _ in range(iters):
            run()
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    return samples


def _graph_cell(model, layer, batch, stream, args):
    """Capture one ``moe()`` call, check its replay against eager, time replays."""
    import torch

    from vllm.distributed.hetero_moe.server_model import (
        SELFTEST_ABS_FLOOR,
        SELFTEST_REL_TOL,
    )

    eager = model.moe(layer, *batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model.moe(layer, *batch)
    graph.replay()
    diff = (out.float() - eager.float()).abs().max().item()
    peak = eager.float().abs().max().item()
    ok = peak > 0.0 and diff <= SELFTEST_REL_TOL * peak + SELFTEST_ABS_FLOOR
    return _burst_us(graph.replay, stream, args.iters, args.reps), diff, ok


def _memory_line(device) -> str:
    import torch

    try:
        free, total = torch.accelerator.get_memory_info(device)
    except (AttributeError, RuntimeError):
        return "device memory free/total: n/a"
    return f"device memory free/total: {free / 2**30:.2f} / {total / 2**30:.2f} GiB"


def run(args: argparse.Namespace) -> int:
    import torch

    import vllm
    from vllm.distributed.hetero_moe.server_model import ExpertServerModel, expert_bytes

    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    arch = getattr(props, "gcnArchName", "")
    print(f"vllm: {vllm.__file__}")
    print(f"torch {torch.__version__}, {device}: {props.name}, {arch or 'not ROCm'}")
    if not arch.startswith("gfx10"):
        print(
            f"server_model_smoke: {device} is {arch or 'not a ROCm device'}, not a "
            "gfx10 V620; the RDNA2 kernels would run as silent no-ops. Set "
            "HIP_VISIBLE_DEVICES to a V620 and use --device cuda:0.",
            file=sys.stderr,
        )
        return 2
    sets = build_expert_sets(
        args.layers,
        args.experts_per_layer,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        stride=args.expert_stride,
    )
    print(_memory_line(device))
    model = ExpertServerModel(
        args.model_dir,
        sets,
        device,
        num_experts=args.num_experts,
        top_k=args.top_k,
        max_capture_rows=args.max_capture_rows,
        group_size=args.group_size,
    )
    held = args.experts_per_layer * len(set(args.layers))
    weights, qzeros = expert_bytes(model.hidden, model.inter, model.group_size)
    print(
        f"checkpoint {args.model_dir}\n"
        f"layers {args.layers}, {args.experts_per_layer} experts each "
        f"(stride {args.expert_stride}), hidden {model.hidden}, inter {model.inter}"
    )
    start = time.perf_counter()
    model.load()
    print(
        f"load: {time.perf_counter() - start:.1f} s; model owns "
        f"{model.memory_bytes() / 2**30:.3f} GiB ({held} experts x "
        f"{weights:,} + {qzeros:,} B qzeros, shared scratch, maps)"
    )
    print(_memory_line(device))

    start = time.perf_counter()
    try:
        result = model.known_answer_selftest(
            experts_per_layer=args.selftest_experts, seed=args.seed
        )
    except RuntimeError as err:
        print(f"SELFTEST FAILED\n{err}", file=sys.stderr)
        return 1
    print(
        f"selftest ok in {time.perf_counter() - start:.1f} s: max abs error "
        f"{result['max_abs_err']:.4g} (reference max {result['max_abs_ref']:.4g}, "
        f"tolerance {result['rel_tol']:.0%} + {result['abs_floor']:g})"
    )
    for layer, rec in result["per_layer"].items():
        print(
            f"  layer {layer:2d} experts {rec['experts']}: err "
            f"{rec['max_abs_err']:.4g} / {rec['max_abs_err_3rows']:.4g} (3 rows), "
            f"tol {rec['tolerance']:.4g}, moe {rec['moe_ms']:.1f} ms"
        )

    stream = torch.cuda.current_stream(device)
    ok = True
    header = f"{'rows':>5} {'layer':>5} {'pairs':>5} {'experts':>7}"
    header += f" {'eager p50':>10} {'min':>8}"
    if args.graph:
        header += f" {'graph p50':>10} {'min':>8} {'replay-eager':>13}"
    print(f"\nmoe() per call, microseconds ({args.iters} calls x {args.reps} bursts)")
    print(header)
    for rows in args.rows:
        for layer in args.layers:
            x, ids, w, pairs, touched = make_batch(
                model, layer, rows, args.local_frac, args.seed
            )
            call = functools.partial(model.moe, layer, x, ids, w)
            for _ in range(args.warmup):
                call()
            eager = _burst_us(call, stream, args.iters, args.reps)
            line = f"{rows:5d} {layer:5d} {pairs:5d} {touched:7d}"
            line += " {:10.1f} {:8.1f}".format(*_median_min(eager))
            if args.graph:
                samples, diff, cell_ok = _graph_cell(
                    model, layer, (x, ids, w), stream, args
                )
                ok = ok and cell_ok
                line += " {:10.1f} {:8.1f} {:13.4g}".format(*_median_min(samples), diff)
                line += "" if cell_ok else "  MISMATCH"
            print(line)
    print(_memory_line(device))
    print("OK" if ok else "FAILED: CUDA graph replay differs from eager")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        print(
            "server_model_smoke: needs a CUDA/ROCm device (--device cuda:N on a "
            f"V620, HIP_VISIBLE_DEVICES set); none is available for {args.device!r}",
            file=sys.stderr,
        )
        return 2
    with torch.accelerator.device_index(torch.device(args.device).index):
        return run(args)


if __name__ == "__main__":
    sys.exit(main())
