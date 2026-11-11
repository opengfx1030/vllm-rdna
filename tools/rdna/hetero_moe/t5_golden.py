# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T5 golden run of ONE expert server: what its local chain computes. Needs a V620.

For every (width, pattern, layer) of the deterministic input family the server's
share of the pairs (``expected_split`` on the hub's tables) goes through
``ExpertServerModel.moe`` eagerly; the digest of the fp16 result (and the bytes, up
to width 48) is written to ``DIR/golden_<server>.npz``. The hub replay later proves
that the same server process, fed over ``hx`` and replayed from a CUDA graph (or
run eagerly at 2048 rows), returns identical bytes.

    HIP_VISIBLE_DEVICES=<hip of the server> PYTHONPATH=<tree> \\
      python -m tools.rdna.hetero_moe.t5_golden --plan P --model M \\
        --server exp0 --hub hub0 --out DIR --widths 3,24,48,2048 --seed 1

Exit codes: 0 written, 4 setup failure (device, plan, model).
"""

import argparse
import sys
import time
from pathlib import Path

from tools.rdna.hetero_moe import t5_common as C


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--plan", required=True)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--server", required=True, help="plan device id, e.g. exp0")
    p.add_argument("--hub", default="hub0", help="hub whose tables define the split")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--widths",
        type=C.parse_int_list,
        default=list(C.DEFAULT_WIDTHS),
        help="e.g. 3,24",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--patterns", type=int, default=C.DEFAULT_PATTERNS)
    p.add_argument("--spill-k", default=None, help='override the plan K, "3:10,24:20"')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The recipe environment must be in place before the first HIP call.
    from vllm.distributed.hetero_moe import server

    server.apply_env_defaults()
    import torch

    torch.set_num_threads(4)
    from vllm.distributed.hetero_moe.plan import load_plan

    try:
        plan = load_plan(args.plan)
        dev = plan.device_index(args.server)
        if plan.role(dev) != "expert":
            raise ValueError(f"{args.server} is not an expert device of the plan")
        s_idx = plan.server_index(dev)
        tables = C.build_tables(plan, args.hub, args.spill_k)
        layers = list(plan.target_layers)
        if layers != list(range(len(layers))):
            raise ValueError(f"target layers {layers[:4]}... are not 0..N-1")
        if C.hx_layers(tables) != layers:
            raise ValueError("the hub's hx layers differ from the plan's target layers")
        device = torch.empty(1, device="cuda").device
        props = torch.cuda.get_device_properties(device)
        want = plan.arch(dev)
        if want and want not in props.gcnArchName:
            raise ValueError(
                f"{args.server} runs on {props.gcnArchName}, the plan says {want}"
            )
        print(
            f"T5 golden device {device} {props.name} arch {props.gcnArchName} "
            f"bus {props.pci_bus_id:02x} (server {args.server} = index {s_idx})",
            flush=True,
        )
        t0 = time.monotonic()
        model = server.load_expert_model(
            plan, dev, args.model, device, max_capture_rows=48
        )
        print(f"T5 golden model loaded in {time.monotonic() - t0:.0f} s", flush=True)
    except Exception as exc:  # noqa: BLE001 - one setup line, exit 4
        C.emit("golden setup", False, f"{type(exc).__name__}: {exc}")
        return C.EXIT_SETUP

    golden = C.Golden.new(plan.plan_id, args.server, args.seed, layers)
    hidden = 2560
    for width in args.widths:
        k_keep = tables.k_for(width)
        digests = {}
        t0 = time.monotonic()
        for pattern in range(args.patterns):
            inp = C.make_inputs(
                args.seed,
                width,
                pattern,
                nlayers=len(layers),
                num_experts=plan.num_experts,
                top_k=plan.top_k,
                hidden=hidden,
            )
            digests[pattern] = C.inputs_digest(inp)
            x_pool = torch.from_numpy(inp.x).to(device)
            ys = []
            for i, layer in enumerate(layers):
                _, dest = C.expected_split(tables, layer, inp.ids[i], k_keep)
                ids = torch.from_numpy(C.server_ids(inp.ids[i], dest, s_idx)).to(device)
                w = torch.from_numpy(inp.w[i]).to(device)
                y = model.moe(layer, x_pool[layer % x_pool.shape[0]], ids, w)
                ys.append(y.cpu().numpy())
            golden.add_pattern(width, pattern, ys)
        golden.add_width(width, args.patterns, k_keep, digests)
        print(
            f"T5 golden W={width} K={k_keep} patterns={args.patterns} "
            f"layers={len(layers)} {time.monotonic() - t0:.0f} s",
            flush=True,
        )
    path = golden.save(C.golden_path(Path(args.out), args.server))
    return 0 if C.emit("golden", True, f"{args.server} -> {path}") else 1


if __name__ == "__main__":
    sys.exit(main())
