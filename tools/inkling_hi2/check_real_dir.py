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
        --stock ~/hfcache/inkling2-mini [--vision] [--mtp]

Exit codes: 0 loaded, all parameters filled; 1 load or comparison failed;
2 bad input.  The text check skips ``model.visual.*``; ``--vision`` compares the
422-tensor bf16 vision tower bit-exact against the stock file (the multimodal
loader copies it as loaded), and ``--mtp`` checks the MTP file DSpark loads
(present, names and bf16 tensors equal to the stock mtp.safetensors).  This reads
the whole text shard: about 2.5-4 minutes and 10+ GiB of page cache on the box.
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


def compare_vision(
    model_dir: Path,
    stock_dir: Path,
    weight_map: dict[str, str],
    stock_map: dict[str, str],
    prefix: str = "model.visual.",
) -> tuple[int, int, list[str]]:
    """Bit-exact compare of every ``model.visual.*`` tensor with the stock file.

    Returns ``(compared, bad, messages)``; the tensors are read shard by shard
    (the vision tower is ~0.6 GiB, not the whole text stack).
    """
    import contextlib

    names = sorted(n for n in weight_map if n.startswith(prefix))
    files: dict[Path, object] = {}

    def opened(path: Path, stack):
        if path not in files:
            files[path] = stack.enter_context(safe_open(str(path), "pt"))
        return files[path]

    bad = 0
    messages: list[str] = []
    with contextlib.ExitStack() as stack:
        for name in names:
            rel = stock_map.get(name)
            if rel is None:
                messages.append(f"vision tensor {name} is not in the stock index")
                bad += 1
                continue
            a = opened(model_dir / weight_map[name], stack).get_tensor(name)
            b = opened(stock_dir / rel, stack).get_tensor(name)
            if (
                a.dtype != b.dtype
                or a.shape != b.shape
                or not torch.equal(a.float(), b.float())
            ):
                if bad < 5:
                    messages.append(f"vision tensor {name} differs from stock {rel}")
                bad += 1
    return len(names), bad, messages


def compare_mtp(mtp_path: Path, stock_mtp: Path) -> tuple[int, list[str]]:
    """The MTP file DSpark loads: same names, all bf16, bit-equal to stock."""
    messages: list[str] = []
    if not mtp_path.is_file():
        return 0, ["the serve directory has no mtp.safetensors (DSpark)"]
    if not stock_mtp.is_file():
        return 0, [f"{stock_mtp} is missing; cannot compare the MTP file"]
    with safe_open(str(mtp_path), "pt") as a, safe_open(str(stock_mtp), "pt") as b:
        an, bn = set(a.keys()), set(b.keys())
        if an != bn:
            return len(an), [
                (
                    "mtp.safetensors names differ from stock: "
                    f"missing {sorted(bn - an)[:3]}, extra {sorted(an - bn)[:3]}"
                )
            ]
        non_bf16 = sorted(
            n
            for n in an
            if a.get_slice(n).get_dtype() != "BF16"
            or b.get_slice(n).get_dtype() != "BF16"
        )
        if non_bf16:
            return len(an), [
                (
                    "mtp.safetensors has non-bf16 tensors "
                    f"{non_bf16[:3]} (the draft loader raises on them)"
                )
            ]
        bad = 0
        for name in sorted(an):
            x, y = a.get_tensor(name), b.get_tensor(name)
            if x.dtype != y.dtype or not torch.equal(x.float(), y.float()):
                if bad < 5:
                    messages.append(f"MTP tensor {name} differs from stock")
                bad += 1
        return len(an), messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stock", default=None, help="stock HF directory (unembed check)")
    ap.add_argument(
        "--vision",
        action="store_true",
        help="also compare model.visual.* bit-exact with --stock",
    )
    ap.add_argument(
        "--mtp",
        action="store_true",
        help="also check mtp.safetensors against --stock's (DSpark loads it)",
    )
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if (args.vision or args.mtp) and not args.stock:
        print("FAIL: --vision and --mtp compare against the stock directory")
        return 2

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

        if args.vision:
            compared, bad, messages = compare_vision(
                model_dir, stock, weight_map, smap, VISUAL_PREFIX
            )
            for message in messages:
                print(f"FAIL: {message}")
            print(f"vision tensors bit-equal to stock: {compared - bad}/{compared}")
            if bad:
                return 1

        if args.mtp:
            try:
                compared, messages = compare_mtp(
                    model_dir / "mtp.safetensors", stock / "mtp.safetensors"
                )
            except Exception as exc:  # noqa: BLE001 - the message is the finding
                print(f"FAIL: mtp.safetensors: {exc}")
                return 1
            for message in messages:
                print(f"FAIL: {message}")
            print(
                f"mtp.safetensors: {compared} tensors bit-equal to stock: "
                f"{not messages}"
            )
            if messages:
                return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
