#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the Thinking Machines Inkling-2-Mini serving checkpoint into the
transformers reference model, dequantising the MoE experts on the fly.

The published checkpoint uses a *serving* layout whose tensor names differ from
the transformers reference implemented in
``transformers.models.inkling2_loop``.  This module owns the mapping and the
NVFP4 / MXFP8 expert dequant so the reference can be used as a numerical oracle
while the vLLM port is developed.

Layout facts pinned against the checkpoint (single shard + separate MTP file):

* layers 0-19 store NVFP4 experts (``w13_weight`` U8 ``[E, N, K/2]``, E4M3 block
  scales ``[E, N, K/16]``, per-expert ``amax``; global = ``amax / 2688``).
* layers 20-23 store MXFP8 experts (``w13_weight`` F8_E4M3 ``[E, N, K]``, UE8M0
  block scales ``[E, N, K/32]``).
* ``w13`` rows (and their block scales) are interleaved gate/up on disk
  (``[g0, u0, g1, u1, ...]``; verified on the real weights, see
  ``docs/rdna2/inkling2-port-spec.md``).  The reference's ``gate_up_proj`` is
  ``[gate rows; up rows]`` (``chunk(2)``), so ``expert_pair`` de-interleaves, as
  the transformers conversion ``Interleave(dim=1)`` does.  This module fills the
  reference directly and never goes through that conversion.
* ``attn_queries.{i}`` is the attention-residual query of layer ``i + 1``;
  full-attention layers (5, 11, 17, 23) carry three queries.
* ``attn_res.output_query`` is the text model's ``final_input.query``.
* ``middle_cycle_cross_pass_queries`` is ``message_input.query``
  ``[loop_layers, 4, hidden]``.

Usage (CPU oracle, needs the fp16 expert cache to be warm):

    python tools/rdna/inkling2/ref_checkpoint.py --ckpt DIR [--prompt TEXT]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

NON_EXPERT_SUFFIXES = (
    "weight",
    "weight_scale",
    "weight_amax",
    "weight_outer_scale_denominator",
)


def dequant_nvfp4_expert(
    w: torch.Tensor, s: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """U8 ``[N, K/2]`` + E4M3 ``[N, K/16]`` -> bf16 ``[N, K]``."""
    w = w.contiguous().view(torch.uint8).to(torch.long)
    n, kd2 = w.shape
    k = kd2 * 2
    lo = FP4_LUT[w & 0x0F]
    hi = FP4_LUT[(w >> 4) & 0x0F]
    vals = torch.stack((lo, hi), dim=-1).reshape(n, k)
    scale = s.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    scale = scale.repeat_interleave(16, dim=-1) * global_scale
    return (vals * scale).to(torch.bfloat16)


def dequant_mxfp8_expert(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 ``[N, K]`` + UE8M0 ``[N, K/32]`` -> bf16 ``[N, K]``."""
    vals = w.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    n, k = vals.shape
    scale = torch.exp2(s.contiguous().view(torch.uint8).to(torch.float32) - 127.0)
    return (vals * scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def deinterleave_gate_up(x: torch.Tensor) -> torch.Tensor:
    """``[E, 2I, ...]`` rows ``[g0, u0, g1, u1, ...]`` -> ``[gate rows; up rows]``,
    ``out[:, j * I + i] = x[:, 2 * i + j]`` (transformers' ``Interleave(dim=1)`` on
    ``mlp.experts.w13_weight``)."""
    e, n2 = x.shape[:2]
    rest = tuple(x.shape[2:])
    return x.reshape(e, n2 // 2, 2, *rest).transpose(1, 2).reshape(x.shape).contiguous()


def expert_pair(
    ckpt: dict, layer: int, prefix: str, cache_dir: Path | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(gate_up [E, 2I, H], down [E, H, I])`` for one layer.

    ``cache_dir`` holds one ``l<layer>.deint.safetensors`` per layer so a second run
    does not pay the dequant again (~2 min per layer on CPU).  The ``.deint`` suffix
    marks the gate/up row order of ``w13`` (caches written before the checkpoint's
    interleaved layout was known hold the raw order and are never read).
    """
    cached = (
        cache_dir / f"l{layer}.deint.safetensors" if cache_dir is not None else None
    )
    if cached is not None and cached.exists():
        from safetensors.torch import load_file

        blob = load_file(str(cached))
        return blob["w13"], blob["w2"]

    w13, s13 = ckpt[f"{prefix}.w13_weight"], ckpt[f"{prefix}.w13_weight_scale"]
    w2, s2 = ckpt[f"{prefix}.w2_weight"], ckpt[f"{prefix}.w2_weight_scale"]
    nvfp4 = (
        w13.dim() == 3
        and w13.dtype == torch.uint8
        and w13.shape[-1] * 2 == s13.shape[-1] * 16
    )
    n_exp = w13.shape[0]
    # NVFP4 packs two elements per byte, MXFP8 one
    width = 2 if nvfp4 else 1
    up = torch.empty(
        (n_exp,) + (w13.shape[1], w13.shape[2] * width), dtype=torch.bfloat16
    )
    down = torch.empty(
        (n_exp,) + (w2.shape[1], w2.shape[2] * width), dtype=torch.bfloat16
    )
    for e in range(n_exp):
        if nvfp4:
            g13 = float(ckpt[f"{prefix}.w13_weight_amax"][e]) / float(
                ckpt[f"{prefix}.w13_weight_outer_scale_denominator"][0]
            )
            g2 = float(ckpt[f"{prefix}.w2_weight_amax"][e]) / float(
                ckpt[f"{prefix}.w2_weight_outer_scale_denominator"][0]
            )
            up[e] = dequant_nvfp4_expert(w13[e], s13[e], g13)
            down[e] = dequant_nvfp4_expert(w2[e], s2[e], g2)
        else:
            up[e] = dequant_mxfp8_expert(w13[e], s13[e])
            down[e] = dequant_mxfp8_expert(w2[e], s2[e])
    up = deinterleave_gate_up(up)
    if cached is not None:
        from safetensors.torch import save_file

        cached.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w13": up, "w2": down}, str(cached))
    return up, down


def build_name_map(num_layers: int) -> dict[str, str]:
    m = {
        "model.llm.embed.weight": "model.language_model.embed_tokens.weight",
        "model.llm.embed_norm.weight": (
            "model.language_model.embed_tokens.embed_norm.weight"
        ),
        "model.llm.norm.weight": "model.language_model.norm.weight",
        # stored padded to the embedding rows (201024); the reference head
        # keeps the unpadded vocabulary (200058) -- trimmed in load_mapped()
        "model.llm.attn_res.output_query": "model.language_model.final_input.query",
    }
    for i in range(num_layers):
        m[f"model.llm.attn_res.mlp_queries.{i}"] = (
            f"model.language_model.layers.{i}.mlp_input.query"
        )
        if i + 1 < num_layers:
            m[f"model.llm.attn_res.attn_queries.{i}"] = (
                f"model.language_model.layers.{i + 1}.attention_input.query"
            )
        src = f"model.llm.layers.{i}"
        dst = f"model.language_model.layers.{i}"
        m.update(
            {
                f"{src}.attn_norm.weight": f"{dst}.input_layernorm.weight",
                # full-attention layers normalise the three attention inputs
                # separately and have no single input_layernorm
                f"{src}.q_input_norm.weight": f"{dst}.q_input_layernorm.weight",
                f"{src}.k_input_norm.weight": f"{dst}.k_input_layernorm.weight",
                f"{src}.v_input_norm.weight": f"{dst}.v_input_layernorm.weight",
                f"{src}.mlp_norm.weight": f"{dst}.post_attention_layernorm.weight",
                f"{src}.attn_sconv.weight": (
                    f"{dst}.post_attention_convolution.conv1d.weight"
                ),
                f"{src}.mlp_sconv.weight": f"{dst}.post_mlp_convolution.conv1d.weight",
                f"{src}.attn.wq_du.weight": f"{dst}.self_attn.q_proj.linear.weight",
                f"{src}.attn.q_l2_clip.scale": f"{dst}.self_attn.q_proj.norm.scale",
                f"{src}.attn.wk_dv.weight": f"{dst}.self_attn.k_proj.linear.weight",
                f"{src}.attn.wv_dv.weight": f"{dst}.self_attn.v_proj.linear.weight",
                f"{src}.attn.wo_ud.weight": f"{dst}.self_attn.o_proj.linear.weight",
                f"{src}.attn.gate_attn.weight": (
                    f"{dst}.self_attn.o_proj.gate_proj.weight"
                ),
                f"{src}.attn.attn_sink_proj.weight": (
                    f"{dst}.self_attn.sink_proj.weight"
                ),
                f"{src}.attn.wr_du.weight": (
                    f"{dst}.self_attn.relative_position_bias.linear.weight"
                ),
                f"{src}.attn.rel_l2_clip.scale": (
                    f"{dst}.self_attn.relative_position_bias.norm.scale"
                ),
                f"{src}.attn.rel_logits_proj.proj": (
                    f"{dst}.self_attn.relative_position_bias.proj"
                ),
                f"{src}.mlp.gate.weight": f"{dst}.mlp.gate.weight",
                f"{src}.mlp.gate.bias": f"{dst}.mlp.gate.e_score_correction_bias",
            }
        )
        # k/v short convs: static depthwise conv on sliding layers, a static +
        # hidden-state-projected dynamic conv on the four full-attention layers.
        for side in ("k", "v"):
            base_s, base_d = (
                f"{src}.attn.{side}_sconv",
                f"{dst}.self_attn.{side}_proj.sconv",
            )
            if i in (5, 11, 17, 23):
                m[f"{base_s}.static_w"] = f"{base_d}.static_weight"
                m[f"{base_s}.weight_proj"] = f"{base_d}.weight_proj.weight"
            else:
                m[f"{base_s}.weight"] = f"{base_d}.conv1d.weight"

        # Loop layers carry four rank-64 message gates (q/k/v/mlp).
        for gate, ctl in (
            ("q_message_gate", "attn"),
            ("k_message_gate", "k"),
            ("v_message_gate", "v"),
            ("mlp_message_gate", "mlp"),
        ):
            for part in ("down", "forget", "gain"):
                m[f"{src}.controllers.{ctl}.{part}_weight"] = (
                    f"{dst}.{gate}."
                    + {
                        "down": "down_proj",
                        "forget": "forget_gate_proj",
                        "gain": "message_gate_proj",
                    }[part]
                    + ".weight"
                )
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument(
        "--check", action="store_true", help="shape-check only (meta model)"
    )
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    ap.add_argument("--prompts", default="", help="''|''-separated prompt list")
    ap.add_argument(
        "--chat-template-file",
        default="",
        help="Jinja chat template (checkpoint tokenizer_config has none)",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    args = ap.parse_args()

    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForConditionalGeneration,
    )

    ckpt_dir = Path(args.ckpt)
    cfg = AutoConfig.from_pretrained(ckpt_dir)
    n_layers = cfg.text_config.num_hidden_layers
    idx = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
    handles = {}
    for f in sorted(set(idx["weight_map"].values())):
        handles[f] = safe_open(str(ckpt_dir / f), framework="pt")

    def get(name: str) -> torch.Tensor:
        return handles[idx["weight_map"][name]].get_tensor(name)

    print("building reference model on CPU (bf16, meta then empty) ...", flush=True)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = Inkling2LoopForConditionalGeneration(cfg)
    model = model.to_empty(device="cpu")
    ref = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    ref["lm_head.weight"] = ref["lm_head.weight"]  # trimmed below

    name_map = build_name_map(n_layers)
    name_map["model.llm.unembed.weight"] = "lm_head.weight"
    used: set[str] = set()
    missing: list[str] = []
    t0 = time.time()
    for i, (src, dst) in enumerate(sorted(name_map.items())):
        if src not in idx["weight_map"]:
            missing.append(src)
            continue
        if dst not in ref and dst not in buffers:
            print(f"  skip {src}: reference has no {dst}")
            missing.append(src)
            continue
        target = ref[dst] if dst in ref else buffers[dst]
        shape = tuple(target.shape)
        if src.endswith("unembed.weight"):
            shape = tuple(target.shape)  # padding trimmed on load
        if args.check:
            src_shape = tuple(get(src).shape)
            ok = tuple(src_shape) == shape or (
                src.endswith("unembed.weight") and src_shape[1:] == shape[1:]
            )
            if not ok:
                print(f"  SHAPE {src} {src_shape} vs {dst} {shape}")
            continue
        tensor = get(src)
        if src.endswith("unembed.weight"):
            tensor = tensor[: shape[0]]
        with torch.no_grad():
            target.copy_(tensor.to(target.dtype).view_as(target))
        used.add(dst)
        if i % 100 == 0:
            print(
                f"  {i}/{len(name_map)} {src} -> {dst} {tuple(tensor.shape)} "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )
    print(
        f"copied {len(used)} tensors in {time.time() - t0:.0f}s, missing {len(missing)}"
    )
    if args.check:
        print("check mode: not running the model")
        return 0

    # Cross-pass message queries: the checkpoint splits the four per-slot
    # queries into ``middle_cycle_cross_pass_queries`` (query, mlp) and
    # ``middle_cycle_cross_pass_kv_queries`` (key, value).
    orders = [
        o.split(",")
        for o in os.environ.get(
            "INKLING_MSG_ORDERS", os.environ.get("INKLING_MSG_ORDER", "q,k,v,m")
        ).split(";")
    ]
    n_loop = (
        cfg.text_config.loop_layer_index_end - cfg.text_config.loop_layer_index_start
    )
    layout = os.environ.get("INKLING_MSG_LAYOUT", "layer")
    q_src = get("model.llm.middle_cycle_cross_pass_queries").float()
    kv_src = get("model.llm.middle_cycle_cross_pass_kv_queries").float()
    if layout == "layer":  # 24 rows = 12 layers x 2 slots
        q_src = q_src.view(n_loop, 2, -1)
        kv_src = kv_src.view(n_loop, 2, -1)
    else:  # 24 rows = 2 slots x 12 layers
        q_src = q_src.view(2, n_loop, -1).transpose(0, 1)
        kv_src = kv_src.view(2, n_loop, -1).transpose(0, 1)
    slots = {"q": q_src[:, 0], "m": q_src[:, 1], "k": kv_src[:, 0], "v": kv_src[:, 1]}

    def set_order(order: list[str]) -> None:
        msg = torch.stack([slots[c] for c in order], dim=1)
        with torch.no_grad():
            ref["model.language_model.message_input.query"].copy_(msg)
        print(f"message_input.query slots {order} -> {tuple(msg.shape)}")

    # Experts: dequantised per layer (9.9 GB bf16 for all 24 layers is not kept).
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    for layer in range(n_layers):
        dst = f"model.language_model.layers.{layer}.mlp.experts"
        src = f"model.llm.layers.{layer}.mlp.experts"
        up, down = expert_pair(
            {
                k: get(k)
                for k in idx["weight_map"]
                if k.startswith(src + ".w13") or k.startswith(src + ".w2")
            },
            layer,
            src,
            cache_dir,
        )
        with torch.no_grad():
            ref[dst + ".gate_up_proj"].copy_(up)
            ref[dst + ".down_proj"].copy_(down)
        print(f"  layer {layer} experts loaded ({up.dtype})", flush=True)

    # Anything not mapped must be a vision tensor (text-only for now).
    unmapped = [n for n in ref if n not in used and "vision_tower" not in n]
    print(f"reference params not filled: {len(unmapped)}")
    for n in unmapped[:10]:
        print("   ", n, tuple(ref[n].shape))

    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    print("tokenizer:", type(tok).__name__, flush=True)
    ids = None
    if not args.raw:
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception as exc:  # checkpoint template may need processor state
            print(
                f"chat template unavailable ({type(exc).__name__}: {exc}); raw prompt"
            )
    if ids is None:
        ids = tok(args.prompt, return_tensors="pt")["input_ids"]
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = torch.as_tensor(ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    prompts = args.prompts.split("|") if args.prompts else [args.prompt]
    template = (
        Path(args.chat_template_file).read_text() if args.chat_template_file else None
    )
    for order in orders:
        set_order(order)
        for prompt in prompts:
            if args.raw or not hasattr(tok, "apply_chat_template"):
                ids = torch.as_tensor(tok(prompt, return_tensors="pt")["input_ids"])
            else:
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    chat_template=template,
                )
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                ids = torch.as_tensor(ids)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            print(
                f"order {order} prompt {prompt!r} tokens: {tuple(ids.shape)}",
                flush=True,
            )
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=False)
            logits = out.logits[0, -1].float()
            top = torch.topk(logits, 8)
            print(
                "  top-8:",
                [
                    (tok.decode([t]), round(float(v), 2))
                    for v, t in zip(top.values, top.indices)
                ],
            )
            print(
                f"  logits: mean {float(logits.mean()):.3f} "
                f"std {float(logits.std()):.3f} max {float(logits.max()):.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
