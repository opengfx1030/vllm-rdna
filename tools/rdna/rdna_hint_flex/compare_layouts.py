# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load every routed-expert stage of a per-expert and a stacked rdna_hint_flex
checkpoint through RdnaHintFlexMoEMethod (on the CPU) and check that the stage
tensors the kernels get are equal.

  python compare_layouts.py PER_EXPERT_DIR STACKED_DIR [--tp 1 --rank 0]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.rdna_hint_flex_moe import (
    RdnaHintFlexMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.rdna_hint_flex_utils import (
    STACK_GROUPS,
)

_SHARD = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def _index(d: Path) -> dict[str, Path]:
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    return {k: d / v for k, v in idx["weight_map"].items()}


def _get(idx, name):
    with safe_open(idx[name], "pt") as f:
        return f.get_tensor(name)


def _layer(cfg, tp, rank):
    e = cfg["num_experts"]
    layer = torch.nn.Module()
    m = RdnaHintFlexMoEMethod(None, 0)
    m.create_weights(
        layer, e, cfg["hidden_size"], cfg["intermediate_size"] // tp, torch.float16
    )
    layer.moe_config = SimpleNamespace(tp_size=tp, tp_rank=rank)
    layer.global_num_experts = e
    layer._map_global_expert_id_to_local_expert_id = lambda g: g
    return layer, m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("per_expert", type=Path)
    p.add_argument("stacked", type=Path)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    a = p.parse_args()
    cfg = json.loads((a.per_expert / "config.json").read_text())
    ia, ib = _index(a.per_expert), _index(a.stacked)
    bad = 0
    for li in range(cfg["num_hidden_layers"]):
        pre = f"model.layers.{li}.mlp.experts."
        la, ma = _layer(cfg, a.tp, a.rank)
        lb, mb = _layer(cfg, a.tp, a.rank)
        for name in (n for n in ia if n.startswith(pre)):
            e, proj, field = name[len(pre) :].split(".")
            stage = "w2" if proj == "down_proj" else "w13"
            ma.absorb(
                la, _get(ia, name), f"{stage}_weight.{field}", _SHARD[proj], int(e)
            )
        for name in (n for n in ib if n.startswith(pre)):
            group, field = name[len(pre) :].split(".", 1)
            if group in STACK_GROUPS:
                mb.absorb_stack(lb, _get(ib, name), group, field.removeprefix("hi2_"))
            else:
                e, proj, field = name[len(pre) :].split(".")
                stage = "w2" if proj == "down_proj" else "w13"
                mb.absorb(lb, _get(ib, name), f"{stage}_weight.{field}",
                          _SHARD[proj], int(e))  # fmt: skip
        stacked = sorted(lb._rdna_hint_flex_moe_stacks)
        bad_before = bad
        ma.process_weights_after_loading(la)
        mb.process_weights_after_loading(lb)
        for stage in ("w13", "w2"):
            ta = getattr(la, f"_rdna_hint_flex_{stage}")
            tb = getattr(lb, f"_rdna_hint_flex_{stage}")
            diff = [
                f
                for f in sorted(set(ta) | set(tb))
                if f not in ta
                or f not in tb
                or ta[f].dtype != tb[f].dtype
                or not torch.equal(ta[f], tb[f])
            ]
            fa = getattr(la, f"_rdna_hint_flex_{stage}_tr")
            fb = getattr(lb, f"_rdna_hint_flex_{stage}_tr")
            if diff or fa != fb:
                bad += 1
                print(f"layer {li} {stage}: DIFFER {diff} flag {fa} vs {fb}")
        state = "equal" if bad == bad_before else "DIFFER"
        print(f"layer {li}: stacked stages {stacked}, {state}", flush=True)
    print("RESULT", "PASS" if bad == 0 else f"FAIL ({bad} stages differ)")


if __name__ == "__main__":
    main()
