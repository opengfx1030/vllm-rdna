# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify split K2 graphs with real TP2/TP4 fused all-reduce and eviction.

Run on two or four otherwise idle gfx1030 GPUs (``--tp``) with
HIP_VISIBLE_DEVICES set explicitly. rdna_ar itself supports world 2..8
(RDNA_AR_MAX_WORLD); the topology here mirrors expert-parallel serving: each
rank owns 512/tp experts and a 640/tp-wide shared-expert slice.
The CPU Gloo group exchanges IPC handles and sums independent local references;
it never substitutes for the fused GPU collective being checked.
"""

import argparse
import importlib.util
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GLOBAL_EXPERTS = 512  # model-wide routed-expert count (Qwen4Exp)
TOTAL_SHARED_INTER = 640  # shared-expert intermediate width before TP


def _topology(tp):
    """(local experts, per-rank shared width, AR probe sum) for world size tp.

    Expert-parallel serving splits routed experts 512/tp and the shared
    expert's intermediate width 640/tp across ranks; the AR probe feeds
    rank+1.0 per rank, so a correct sum is tp*(tp+1)/2.
    """
    if tp not in (2, 4):
        raise ValueError("only TP2 and TP4 are qualified here")
    return GLOBAL_EXPERTS // tp, TOTAL_SHARED_INTER // tp, tp * (tp + 1) // 2


def _connect_ar(rank, tp):
    ops = torch.ops._rocm_C
    packed = None
    for turn in range(tp):
        if rank == turn:
            packed = ops.rdna_ar_init(
                rank, tp, torch.tensor(list(range(tp))), 256 * 1024, ""
            )
        dist.barrier()
    raw = bytes(packed.tolist())
    handle = int.from_bytes(raw[:8], "little", signed=True)
    handles = [None] * tp
    dist.all_gather_object(handles, raw[8:])
    buffer = torch.frombuffer(bytearray(b"".join(handles)), dtype=torch.uint8)
    ops.rdna_ar_connect(handle, buffer.view(tp, -1).contiguous())
    # Neither rank may publish a collective before both IPC maps are ready.
    dist.barrier()
    probe = ops.rdna_ar_all_reduce(
        handle, torch.full((256,), rank + 1.0, device="cuda", dtype=torch.half)
    )
    torch.accelerator.synchronize()
    if not torch.equal(probe, torch.full_like(probe, _topology(tp)[2])):
        diagnostic = {
            "rank": rank,
            "event": "ar_probe_failed",
            "minimum": probe.min().item(),
            "maximum": probe.max().item(),
            "unique": probe.unique().cpu().tolist(),
            "timed_out": ops.rdna_ar_timed_out(handle),
            "timeout_info": ops.rdna_ar_timeout_info(handle),
        }
        print(json.dumps(diagnostic), flush=True)
        raise AssertionError(diagnostic)
    return handle


def _sum_cpu(tensor):
    result = tensor.detach().float().cpu()
    dist.all_reduce(result)
    return result.half()


def _worker(rank, args, port):
    tp = args.tp
    local_experts, shared_inter, _ = _topology(tp)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(4)
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", rank=rank, world_size=tp, timeout=timedelta(seconds=180)
    )
    import vllm._custom_ops  # noqa: F401

    assert torch.version.hip
    assert torch.cuda.get_device_properties(rank).gcnArchName.startswith("gfx1030")
    root = Path(__file__).resolve().parents[2]
    tests = root / "tests/models/qwen4_exp"
    helper = _load("tp2_ar_kernel_helpers", tests / "test_amd_mono_rdna2_kernels.py")
    pool = _load("tp2_ar_pool_helpers", tests / "test_amd_mono_rdna2_offload.py")
    print(json.dumps({"rank": rank, "event": "ar_connect_begin"}), flush=True)
    handle = _connect_ar(rank, tp)
    print(json.dumps({"rank": rank, "event": "ar_connect_passed"}), flush=True)
    helper.EL, helper.ISE = local_experts, shared_inter
    gen = torch.Generator(device="cuda").manual_seed(411)
    print(json.dumps({"rank": rank, "event": "layer_build_begin"}), flush=True)
    layer, golden = helper._make_layer(gen, keep_golden=False)
    print(json.dumps({"rank": rank, "event": "layer_build_done"}), flush=True)
    # Different shared contributions per rank; router/HC stay identical.
    layer["se_w2"].mul_(0.5 + rank * 0.5)
    base = rank * local_experts
    ops = torch.ops._rocm_C
    host = {name: layer[name].cpu().pin_memory() for name in pool._PAYLOAD}
    cases = {}
    reports = []

    for rows in args.rows:
        print(
            json.dumps({"rank": rank, "event": "capture_begin", "rows": rows}),
            flush=True,
        )
        inputs = helper._inputs(rows, 1200 + rows)
        buffers = pool._buffers(torch, helper, layer, rows, args.slots, local_experts)
        graphs = {}
        ops.mono_k2_rdna2_set_ar(handle, rank)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # Recording stages never executes them or advances the AR sequence.
        for stage in (1, 2, 3):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                pool._stage(torch, layer, inputs, buffers, stage, base)
            graphs[stage] = graph
        torch.cuda.current_stream().wait_stream(side)
        cases[rows] = inputs, buffers, graphs
        print(
            json.dumps({"rank": rank, "event": "capture_done", "rows": rows}),
            flush=True,
        )

    for iteration in range(args.repeats):
        for rows in args.rows:
            inputs, buffers, graphs = cases[rows]
            for dst, src in zip(
                inputs, helper._inputs(rows, 2000 + 100 * iteration + rows), strict=True
            ):
                dst.copy_(src)

            # A native local oracle isolates AR from expert-pass scheduling.
            # The graph already captured its AR handle; changing the launcher
            # handle here does not alter that graph's device parameters.
            ops.mono_k2_rdna2_set_ar(-1, rank)
            resident = pool._full_reference(
                torch, helper, layer, inputs, local_experts, base
            )
            native_sum = _sum_cpu(resident[1])
            independent, residual, injection, reference_ids, reference_weights = (
                helper._today_ffn(layer, *inputs, return_routes=True, expert_base=base)
            )
            independent_sum = _sum_cpu(independent)
            gaps = helper._golden_gaps(golden, *inputs)
            ops.mono_k2_rdna2_set_ar(handle, rank)
            dist.barrier()

            graphs[1].replay()
            demand = pool._demand(torch, buffers, base)
            passes = 0
            # Deliberately refill every pass from pinned host storage. This
            # maximizes slot churn and tests the same graph with new payloads.
            for start in range(0, len(demand), args.slots):
                active = demand[start : start + args.slots]
                buffers["map"].fill_(-1)
                for slot, expert in enumerate(active):
                    for name in pool._PAYLOAD:
                        buffers["payload"][name][slot].copy_(
                            host[name][expert], non_blocking=True
                        )
                    buffers["map"][expert] = slot
                graphs[2].replay()
                passes += 1
            graphs[3].replay()
            torch.accelerator.synchronize()
            fault = ops.mono_k2_rdna2_error(buffers["ws"]).tolist()
            assert fault[0] == 0, (rank, rows, iteration, fault)
            assert not ops.rdna_ar_timed_out(handle)
            assert bool(torch.isfinite(buffers["out"]).all())
            torch.testing.assert_close(
                buffers["out"].cpu(), native_sum, rtol=0, atol=2e-3
            )
            assert torch.equal(buffers["res_out"], residual)
            torch.testing.assert_close(buffers["inj_out"], injection, rtol=0, atol=5e-3)
            delta = (
                (buffers["out"].float().cpu() - independent_sum.float())
                .abs()
                .amax(dim=1)
            )
            bad = [
                row
                for row, (difference, gap) in enumerate(zip(delta.tolist(), gaps))
                if difference >= 1e-2 and gap >= helper.TIE_GAP
            ]
            actual_ids = buffers["ids"].cpu().tolist()
            actual_weights = buffers["weights"].cpu().tolist()
            reference_ids = reference_ids.cpu().tolist()
            reference_weights = reference_weights.cpu().tolist()
            route_differences = []
            for row in range(rows):
                actual = set(actual_ids[row])
                reference = set(reference_ids[row])
                if actual != reference or row in bad:
                    route_differences.append(
                        {
                            "row": row,
                            "fp32_gap": gaps[row],
                            "native_ids": actual_ids[row],
                            "reference_global_ids": reference_ids[row],
                            "native_only": sorted(actual - reference),
                            "reference_only": sorted(reference - actual),
                            "native_weights": actual_weights[row],
                            "reference_weights": reference_weights[row],
                            "output_max_abs": delta[row].item(),
                        }
                    )
            native_error = (
                (buffers["out"].cpu().float() - native_sum.float()).abs().amax(dim=1)
            )
            if bad:
                diagnostic = {
                    "event": "independent_reference_failed",
                    "rank": rank,
                    "rows": rows,
                    "iteration": iteration,
                    "bad_rows": bad,
                    "native_max_abs_by_row": native_error.tolist(),
                    "independent_max_abs_by_row": delta.tolist(),
                    "routes": route_differences,
                    "native_host_sum_head": native_sum[:, :8].tolist(),
                    "independent_host_sum_head": independent_sum[:, :8].tolist(),
                    "native_host_sum_minmax": [
                        native_sum.min().item(),
                        native_sum.max().item(),
                    ],
                    "independent_host_sum_minmax": [
                        independent_sum.min().item(),
                        independent_sum.max().item(),
                    ],
                }
                print(json.dumps(diagnostic), flush=True)
                raise AssertionError(diagnostic)
            report = {
                "rank": rank,
                "rows": rows,
                "iteration": iteration,
                "passes": passes,
                "local_demand": len(demand),
                "route_differences": route_differences,
                "native_max_abs": native_error.max().item(),
                "independent_max_abs": delta.max().item(),
            }
            rank_reports = [None] * tp
            dist.all_gather_object(rank_reports, report)
            if rank == 0:
                reports.extend(rank_reports)
                print(json.dumps({"cell": rank_reports}), flush=True)

    if rank == 0:
        print(json.dumps({"status": "passed", "cells": reports}), flush=True)
    # Destroy graphs before their IPC-backed AR buffers/processes disappear.
    graph = None
    graphs = None
    cases.clear()
    torch.accelerator.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2, choices=(2, 4))
    parser.add_argument("--rows", default="24,12,6,3,1")
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.rows = [int(value) for value in args.rows.split(",")]
    local_experts, _, _ = _topology(args.tp)
    if not 1 <= args.slots <= local_experts or any(
        not 1 <= rows <= 24 for rows in args.rows
    ):
        parser.error(
            f"slots must be 1..{local_experts} and every row width 1..24"
        )
    # Enumerate AFTER init: torch.cuda.device_count() consults amd_smi before
    # the first CUDA init, and amd_smi reports 0 processors on OEM V620 cards
    # that expose no serial/GUID (post-init it uses the ROCr enumeration and
    # reports the real device list).
    torch.cuda.init()
    if torch.cuda.device_count() != args.tp:
        parser.error(
            f"select exactly {args.tp} idle GPUs with HIP_VISIBLE_DEVICES"
        )
    # rdna_ar_init() hipDeviceEnablePeerAccess()es every peer up front, and on
    # P2P-incapable hosts the runtime fails inside the kernel init with an
    # opaque "invalid device ordinal" (e.g. four V620s behind separate PCIe
    # switches). Refuse with the actual reason and the supported fallback.
    for a in range(args.tp):
        if not all(
            torch.cuda.can_device_access_peer(a, b)
            for b in range(args.tp)
            if b != a
        ):
            parser.error(
                f"rdna_ar requires GPU peer access; device {a} reports none "
                "-- this host must run collectives through the RCCL fallback "
                "(VLLM_RDNA_AR=0), so the custom-allreduce kernel cannot be "
                "exercised here"
            )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(args, port), nprocs=args.tp, join=True)


if __name__ == "__main__":
    main()
