#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the REAL HI2 Inkling directory on CPU (no GPU, no lease).

The CPU proxy for the first boot: it builds the serving text model (serving=False),
streams every text tensor of the quantizer's output through ``load_weights`` and
checks that no parameter is left unloaded.  With ``--stock`` it also checks that
the unembedding is the stock unembed (the top-level ``lm_head.weight`` aliasing)
and that the dense text tensors equal the stock ones.

    $VENV/bin/python tools/inkling_hi2/check_real_dir.py \
        --model ~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5 \
        --stock ~/hfcache/inkling2-mini

Exit codes: 0 loaded, all parameters filled; 1 load or comparison failed;
2 bad input.  Vision tensors are skipped here (the language model only; the
multimodal class loads ``model.visual.*``).  This reads the whole text shard:
about 2.5-4 minutes and 10+ GiB of page cache on the box.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stock", default=None, help="stock HF directory (unembed check)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    from vllm.models.inkling2_loop.amd import hi2
    from vllm.models.inkling2_loop.amd.model import Inkling2LoopTextForCausalLM
    from vllm.models.inkling2_loop.amd.multimodal import VISUAL_PREFIX
    from vllm.models.inkling2_loop.amd.parallel import TensorParallel
    from vllm.models.inkling2_loop.configs import Inkling2LoopConfig

    torch.set_num_threads(args.threads)
    model_dir = Path(args.model).expanduser()
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    cfg = Inkling2LoopConfig.from_pretrained(str(model_dir))

    # the model ctor reads vLLM's global config in ``state.py``; a serving=False
    # build only needs these fields
    import vllm.models.inkling2_loop.amd.state as state

    fake = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16),
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(block_size=16),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=0),
    )
    state.get_current_vllm_config = lambda: fake
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=cfg, dtype=torch.float16, model=str(model_dir), revision=None
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=0),
    )
    model = Inkling2LoopTextForCausalLM(
        vllm_config=vllm_config,
        serving=False,
        tp=TensorParallel(args.tp, 0),
    )

    def stream():
        for shard in sorted(set(weight_map.values())):
            with safe_open(str(model_dir / shard), "pt") as h:
                names = h.keys()
                for name in names:
                    if name.startswith(VISUAL_PREFIX) or name.startswith("mtp."):
                        continue
                    yield name, h.get_tensor(name)

    t0 = time.time()
    try:
        loaded = model.load_weights(stream())
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        print(f"FAIL: load_weights raised {type(exc).__name__}: {exc}")
        return 1
    missing = sorted(n for n, _ in model.named_parameters() if n not in loaded)
    expert_buffers = sum(
        1
        for n, _ in model.named_buffers()
        if ".w13_" in n or ".w2_" in n or ".hi2_" in n
    )
    print(
        f"load_weights: {len(loaded)} tensors in {time.time() - t0:.0f}s; "
        f"missing parameters: {len(missing)}; expert buffers built: {expert_buffers}"
    )
    if missing:
        print("  first:", missing[:5])
        return 1

    if args.stock:
        stock = Path(args.stock).expanduser()
        sidx = json.loads((stock / "model.safetensors.index.json").read_text())
        smap = sidx["weight_map"]
        tensor = model.lm_head.weight
        with safe_open(str(stock / smap["model.llm.unembed.weight"]), "pt") as h:
            want = h.get_tensor("model.llm.unembed.weight").float()
        rows = min(tensor.shape[0], want.shape[0])
        got = tensor[:rows].float()
        same = torch.equal(got, want[:rows])
        print(
            f"lm_head: {tuple(tensor.shape)} vs stock {tuple(want.shape)}; "
            f"first {rows} rows equal: {same}"
        )
        if not same:
            return 1
        # the quantizer writes the dense text tensors under HF names; at least a
        # sample must be bit-equal to the stock file
        from vllm.models.inkling2_loop.amd import loader

        checked = 0
        for name in weight_map:
            if not name.startswith("model.language_model.") or hi2.parse_tensor_name(
                name
            ):
                continue
            served = loader.hf_to_serving(name)
            if served is None or served not in smap:
                continue
            with safe_open(str(model_dir / weight_map[name]), "pt") as h:
                a = h.get_tensor(name)
            with safe_open(str(stock / smap[served]), "pt") as h:
                b = h.get_tensor(served)
            if a.shape != b.shape or not torch.equal(a.float(), b.float()):
                print(f"FAIL: dense tensor {name} differs from stock {served}")
                return 1
            checked += 1
        print(f"dense text tensors bit-equal to stock: {checked}")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
