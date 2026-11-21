#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp32 CPU oracle for a tiny Inkling-2 checkpoint (``make_tiny_checkpoint.py``).

Loads the checkpoint directory into the real transformers reference
(``Inkling2LoopForConditionalGeneration``) with the key mapping of
``ref_checkpoint.py`` (imported, not re-derived) and runs it on raw token ids.

Outputs, for ``--out logits.json``:

* ``logits.json``: per prompt, last-position top-8 ids / values, logits
  statistics, the greedy continuation (``--decode-steps N``, KV/conv-cache path)
  and, when decoding, the cache-vs-recompute self-check;
* ``logits.npy``: float32 ``[P, V]`` last-position logits (V = unpadded vocab);
* ``logits_decode.npy``: float32 ``[P, N, V]`` logits that produced each greedy
  token (row 0 is the prefill's last position), only with ``--decode-steps``.

``--expert-dtype fp32`` (default) dequantises the experts exactly; ``bf16``
reproduces the rounding of ``ref_checkpoint.expert_pair`` bit for bit.

Usage:
    python tiny_reference.py --ckpt DIR --prompts-json FILE --out logits.json \
        [--decode-steps 8]
``FILE`` holds ``[[ids...], ...]`` or ``{"prompts": [[ids...], ...]}``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
FP8_E4M3 = torch.float8_e4m3fn


def configure_threads(n: int = 8) -> None:
    """The tiny ops are latency bound: all cores of a big box thrash (a forward
    that takes 3 s with 8 threads does not finish in minutes with 96)."""
    torch.set_num_threads(max(1, min(n, os.cpu_count() or 1)))


def load_ref_checkpoint():
    """Import the sibling ``ref_checkpoint.py`` (the real-checkpoint oracle)."""
    name = "inkling2_ref_checkpoint"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / "ref_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def adapt_name_map(
    base: dict[str, str], num_layers: int, full_layers: set[int]
) -> dict[str, str]:
    """``ref_checkpoint.build_name_map`` hardcodes the real full-attention layers
    (5, 11, 17, 23) for the k/v short-conv keys; re-key them for ``full_layers``."""
    m = dict(base)
    for i in range(num_layers):
        for side in ("k", "v"):
            src = f"model.llm.layers.{i}.attn.{side}_sconv"
            dst = f"model.language_model.layers.{i}.self_attn.{side}_proj.sconv"
            for leaf in (".static_w", ".weight_proj", ".weight"):
                m.pop(src + leaf, None)
            if i in full_layers:
                m[f"{src}.static_w"] = f"{dst}.static_weight"
                m[f"{src}.weight_proj"] = f"{dst}.weight_proj.weight"
            else:
                m[f"{src}.weight"] = f"{dst}.conv1d.weight"
    return m


def dequant_nvfp4(
    w: torch.Tensor, s: torch.Tensor, amax: torch.Tensor, denom: torch.Tensor
) -> torch.Tensor:
    """U8 ``[E, N, K/2]`` + E4M3 ``[E, N, K/16]`` -> fp32 ``[E, N, K]``.
    Same arithmetic order as ``ref_checkpoint.dequant_nvfp4_expert``."""
    lut = load_ref_checkpoint().FP4_LUT
    w = w.contiguous().view(torch.uint8).to(torch.long)
    e, n, kd2 = w.shape
    vals = torch.stack((lut[w & 0x0F], lut[(w >> 4) & 0x0F]), dim=-1).reshape(
        e, n, kd2 * 2
    )
    glob = (amax.double() / denom.double()[0]).float()[:, None, None]
    scale = s.contiguous().view(torch.uint8).view(FP8_E4M3).float()
    scale = scale.repeat_interleave(16, dim=-1) * glob
    return vals * scale


def dequant_mxfp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 ``[E, N, K]`` + UE8M0 ``[E, N, K/32]`` -> fp32 ``[E, N, K]``."""
    vals = w.contiguous().view(torch.uint8).view(FP8_E4M3).float()
    scale = torch.exp2(s.contiguous().view(torch.uint8).float() - 127.0)
    return vals * scale.repeat_interleave(32, dim=-1)


def dequant_experts(
    tensors: dict[str, torch.Tensor], prefix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gate_up [E, 2I, H], down [E, H, I])`` fp32 for ``<prefix>.mlp.experts``."""
    out = []
    for nm in ("w13", "w2"):
        w, s = tensors[f"{prefix}.{nm}_weight"], tensors[f"{prefix}.{nm}_weight_scale"]
        if w.dtype == torch.uint8:  # NVFP4 packs two elements per byte
            out.append(
                dequant_nvfp4(
                    w,
                    s,
                    tensors[f"{prefix}.{nm}_weight_amax"],
                    tensors[f"{prefix}.{nm}_weight_outer_scale_denominator"],
                )
            )
        else:
            out.append(dequant_mxfp8(w, s))
    return out[0], out[1]


def read_checkpoint(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors.update(load_file(str(ckpt_dir / shard)))
    missing = set(index["weight_map"]) - set(tensors)
    if missing:
        raise RuntimeError(
            f"index lists tensors missing from the shards: {sorted(missing)[:5]}"
        )
    return tensors


def build_reference(
    ckpt_dir: str | Path,
    *,
    attn_impl: str = "eager",
    expert_dtype: torch.dtype = torch.float32,
    msg_layout: str | None = None,
    msg_order: str | None = None,
):
    """Return the fp32 reference model with every text weight loaded from ``ckpt_dir``.

    Strict: every checkpoint tensor must be consumed and every non-vision model
    parameter / buffer must be filled exactly once, otherwise this raises.
    """
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration,
    )

    rc = load_ref_checkpoint()
    ckpt_dir = Path(ckpt_dir)
    cfg = AutoConfig.from_pretrained(ckpt_dir)
    text = cfg.text_config
    n_layers = text.num_hidden_layers
    full_layers = {i for i, t in enumerate(text.layer_types) if t == "hybrid"}
    tensors = read_checkpoint(ckpt_dir)

    model = Inkling2LoopForConditionalGeneration._from_config(
        cfg, attn_implementation=attn_impl, dtype=torch.float32
    )
    model.eval()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    state = {**params, **buffers}

    name_map = adapt_name_map(rc.build_name_map(n_layers), n_layers, full_layers)
    name_map["model.llm.unembed.weight"] = "lm_head.weight"
    consumed: set[str] = set()
    filled: set[str] = set()

    def fill(dst: str, value: torch.Tensor) -> None:
        if dst not in state:
            raise KeyError(f"reference has no parameter/buffer {dst!r}")
        if dst in filled:
            raise KeyError(f"reference tensor {dst!r} filled twice")
        target = state[dst]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"{dst}: checkpoint {tuple(value.shape)} vs model {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(value.to(target.dtype))
        filled.add(dst)

    for src, dst in name_map.items():
        if src not in tensors:
            continue  # name_map spans every layer; loop-only/full-only keys are absent
        value = tensors[src]
        if src.endswith("unembed.weight"):  # padded to vocab_size on disk
            value = value[: text.unpadded_vocab_size]
        fill(dst, value)
        consumed.add(src)

    # cross-pass message queries (same layout/order handling as ref_checkpoint.py)
    order = (msg_order or os.environ.get("INKLING_MSG_ORDER", "q,k,v,m")).split(",")
    layout = msg_layout or os.environ.get("INKLING_MSG_LAYOUT", "layer")
    n_loop = text.loop_layer_index_end - text.loop_layer_index_start
    q_src = tensors["model.llm.middle_cycle_cross_pass_queries"].float()
    kv_src = tensors["model.llm.middle_cycle_cross_pass_kv_queries"].float()
    if layout == "layer":
        q_src, kv_src = q_src.view(n_loop, 2, -1), kv_src.view(n_loop, 2, -1)
    else:
        q_src = q_src.view(2, n_loop, -1).transpose(0, 1)
        kv_src = kv_src.view(2, n_loop, -1).transpose(0, 1)
    slots = {"q": q_src[:, 0], "m": q_src[:, 1], "k": kv_src[:, 0], "v": kv_src[:, 1]}
    fill(
        "model.language_model.message_input.query",
        torch.stack([slots[c] for c in order], dim=1),
    )
    consumed |= {
        "model.llm.middle_cycle_cross_pass_queries",
        "model.llm.middle_cycle_cross_pass_kv_queries",
    }

    for layer in range(n_layers):
        src = f"model.llm.layers.{layer}.mlp.experts"
        dst = f"model.language_model.layers.{layer}.mlp.experts"
        up, down = dequant_experts(tensors, src)
        fill(dst + ".gate_up_proj", up.to(expert_dtype))
        fill(dst + ".down_proj", down.to(expert_dtype))
        consumed |= {k for k in tensors if k.startswith(src + ".")}

    orphans = sorted(set(tensors) - consumed)
    if orphans:
        raise RuntimeError(
            f"checkpoint tensors without a reference home: {orphans[:8]}"
        )
    unfilled = sorted(n for n in state if n not in filled and "vision_tower" not in n)
    if unfilled:
        raise RuntimeError(f"reference tensors never filled: {unfilled[:8]}")
    return model


@torch.no_grad()
def full_logits(model, ids: list[int]) -> torch.Tensor:
    """Logits ``[S, V]`` of every position from one uncached forward."""
    out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
    return out.logits[0].float()


@torch.no_grad()
def teacher_forced_decode(model, ids: list[int], n_decode: int) -> torch.Tensor:
    """Prefill ``ids[:-n_decode]``, then feed the last ``n_decode`` ids one at a
    time through the cache.  Returns logits ``[n_decode, V]`` for those positions."""
    split = len(ids) - n_decode
    out = model(input_ids=torch.tensor([ids[:split]], dtype=torch.long), use_cache=True)
    cache = out.cache
    rows = []
    for tok in ids[split:]:
        out = model(
            input_ids=torch.tensor([[tok]], dtype=torch.long),
            cache=cache,
            use_cache=True,
        )
        cache = out.cache
        rows.append(out.logits[0, -1].float())
    return torch.stack(rows)


@torch.no_grad()
def greedy_cached(model, ids: list[int], steps: int) -> tuple[list[int], torch.Tensor]:
    """Greedy continuation through the HF cache.  Returns the ``steps`` new ids and
    the logits ``[steps, V]`` that produced each of them."""
    out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=True)
    cache = out.cache
    last = out.logits[0, -1].float()
    new_ids: list[int] = []
    rows = []
    for step in range(steps):
        rows.append(last)
        tok = int(last.argmax())
        new_ids.append(tok)
        if step + 1 == steps:
            break
        out = model(
            input_ids=torch.tensor([[tok]], dtype=torch.long),
            cache=cache,
            use_cache=True,
        )
        cache = out.cache
        last = out.logits[0, -1].float()
    return new_ids, torch.stack(rows)


@torch.no_grad()
def greedy_recompute(
    model, ids: list[int], steps: int
) -> tuple[list[int], torch.Tensor]:
    """Same continuation without any cache (full re-forward at every step)."""
    seq = list(ids)
    new_ids: list[int] = []
    rows = []
    for _ in range(steps):
        last = full_logits(model, seq)[-1]
        rows.append(last)
        tok = int(last.argmax())
        new_ids.append(tok)
        seq.append(tok)
    return new_ids, torch.stack(rows)


def top8(logits: torch.Tensor) -> dict[str, list]:
    top = torch.topk(logits, 8)
    return {"ids": top.indices.tolist(), "values": [float(v) for v in top.values]}


def read_prompts(path: str | Path) -> list[list[int]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data["prompts"]
    return [[int(t) for t in p] for p in data]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode-steps", type=int, default=0)
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--expert-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--msg-layout", default=None, choices=[None, "layer", "slot"])
    ap.add_argument("--msg-order", default=None, help="slot order, default q,k,v,m")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    configure_threads(args.threads)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = build_reference(
        Path(args.ckpt).expanduser(),
        attn_impl=args.attn_impl,
        expert_dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.expert_dtype],
        msg_layout=args.msg_layout,
        msg_order=args.msg_order,
    )
    prompts = read_prompts(Path(args.prompts_json).expanduser())
    out_path = Path(args.out).expanduser()
    steps = args.decode_steps

    report = {
        "ckpt": str(args.ckpt),
        "attn_impl": args.attn_impl,
        "expert_dtype": args.expert_dtype,
        "decode_steps": steps,
        "prompts": [],
    }
    last_rows: list[np.ndarray] = []
    decode_rows: list[np.ndarray] = []
    for ids in prompts:
        logits = full_logits(model, ids)
        last = logits[-1]
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite logits for prompt of {len(ids)} tokens")
        entry = {
            "n_tokens": len(ids),
            "logits_mean": float(last.mean()),
            "logits_std": float(last.std()),
            "last_top8": top8(last),
        }
        last_rows.append(last.numpy())
        if steps:
            new_ids, rows = greedy_cached(model, ids, steps)
            ref_ids, ref_rows = greedy_recompute(model, ids, steps)
            entry["greedy_ids"] = new_ids
            entry["decode_top8"] = [top8(r) for r in rows]
            entry["cache_vs_recompute"] = {
                "max_abs_logit_diff": float((rows - ref_rows).abs().max()),
                "greedy_ids_equal": new_ids == ref_ids,
            }
            decode_rows.append(rows.numpy())
        report["prompts"].append(entry)
        print(
            f"prompt of {len(ids):3d} tokens: logits std {entry['logits_std']:.3f}, "
            f"top-1 {entry['last_top8']['ids'][0]}"
            + (f", greedy {entry['greedy_ids']}" if steps else ""),
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    npy = out_path.with_suffix(".npy")
    np.save(npy, np.stack(last_rows).astype(np.float32))
    if steps:
        np.save(
            out_path.with_name(out_path.stem + "_decode.npy"),
            np.stack(decode_rows).astype(np.float32),
        )
    print(f"wrote {out_path} and {npy.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
