#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a random-weight *tiny* Inkling-2 checkpoint in the real serving format.

The directory mimics ``thinkingmachines/Inkling-2-Mini`` key for key (same
``model.llm.*`` names, dtypes and quantised expert layout) at a size a CPU can
run in seconds, so the whole vLLM port can be gated against the transformers
reference end to end (see ``tiny_reference.py``).

Layout facts mirrored from the real checkpoint header (checked on the box):

* dense tensors are bf16; controllers (message gates) are bf16; attention
  residual queries, ``middle_cycle_cross_pass_*queries`` and ``mlp.gate.bias`` are
  fp32; ``mlp.gate.weight`` is bf16;
* ``embed``/``unembed`` have ``vocab_size`` rows; the reference head keeps only
  the first ``unpadded_vocab_size`` of them.  The padding rows are filled with
  random data (not zeros) so a loader that forgets to trim them is caught;
* NVFP4 experts (all layers but the last): ``w13_weight`` U8 ``[E, 2I, H/2]``,
  ``w13_weight_scale`` F8_E4M3 ``[E, 2I, H/16]``, ``w13_weight_amax`` F32
  ``[E]``, ``w13_weight_outer_scale_denominator`` F32 ``[1]`` (= 2688), same for
  ``w2`` (``[E, H, I/2]``, ``[E, H, I/16]``).  value = e2m1 * e4m3 * amax/2688;
* MXFP8 experts (the last layer): ``w13_weight`` F8_E4M3 ``[E, 2I, H]``,
  ``w13_weight_scale`` U8 (UE8M0) ``[E, 2I, H/32]``, ``w2_weight``
  ``[E, H, I]`` / ``[E, H, I/32]``.  value = e4m3 * 2^(s - 127);
* full-attention layers (``layer_types == "hybrid"``) have three attention
  residual queries (``[3, H]``), ``{q,k,v}_input_norm`` instead of
  ``attn_norm`` and dynamic k/v short convs (``static_w`` ``[k, C]`` +
  ``weight_proj`` ``[nkv * k, H]``); sliding layers have ``attn_norm`` and
  static ``k_sconv.weight`` ``[C, 1, k]``;
* loop layers carry ``controllers.{attn,k,v,mlp}.{down,forget,gain}_weight``;
* sliding layers' relative-position ``proj`` is ``[d_rel, sliding_window_size]``
  and full layers' is ``[d_rel, rel_extent]`` (equal in the real checkpoint,
  different here on purpose: 16 vs 32).

There is no vision tower, no MTP file and no tokenizer: the harness works on
raw token ids.  ``config.json`` still carries a (tiny) ``vision_config`` because
the transformers ``Inkling2LoopConfig`` validates it.

Usage: ``python make_tiny_checkpoint.py --out DIR [--seed N] [--list]``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

SHARD = "model-00001-of-00001.safetensors"
NVFP4_DENOM = 2688.0  # 6 (max e2m1) * 448 (max e4m3)
FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FULL = "hybrid"
SLIDING = "hybrid_sliding"


@dataclass(frozen=True)
class TinySpec:
    """Shapes of the tiny model.  Defaults follow the harness contract."""

    hidden_size: int = 256
    num_hidden_layers: int = 6
    n_layers_per_attn_res_block: int = 2
    loop_start: int = 2
    loop_end: int = 4
    loop_num_passes: int = 2
    loop_gate_rank: int = 16
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 32
    d_rel: int = 8
    rel_extent: int = 32
    sliding_window: int = 16
    log_scaling_n_floor: int = 24
    conv_kernel: int = 4
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate: int = 128
    vocab_size: int = 512
    unpadded_vocab_size: int = 500
    full_layers: tuple[int, ...] | None = None  # default: every odd layer

    def __post_init__(self) -> None:
        if self.loop_start % self.n_layers_per_attn_res_block:
            raise ValueError("loop_start must be a multiple of the attn-res block")
        if self.loop_end % self.n_layers_per_attn_res_block:
            raise ValueError("loop_end must be a multiple of the attn-res block")
        for dim in (self.hidden_size, self.moe_intermediate):
            if dim % 32:
                raise ValueError("hidden/intermediate must be multiples of 32")

    @property
    def full_layer_ids(self) -> tuple[int, ...]:
        if self.full_layers is not None:
            return tuple(self.full_layers)
        return tuple(range(1, self.num_hidden_layers, 2))

    @property
    def local_layer_ids(self) -> list[int]:
        full = set(self.full_layer_ids)
        return [i for i in range(self.num_hidden_layers) if i not in full]

    @property
    def layer_types(self) -> list[str]:
        full = set(self.full_layer_ids)
        return [FULL if i in full else SLIDING for i in range(self.num_hidden_layers)]

    @property
    def loop_layer_ids(self) -> range:
        return range(self.loop_start, self.loop_end)

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def is_nvfp4_layer(self, layer: int) -> bool:
        return layer < self.num_hidden_layers - 1

    def text_config(self) -> dict:
        return {
            "model_type": "inkling2_loop_text",
            "model_max_length": 4096,
            "torch_dtype": "bfloat16",
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.unpadded_vocab_size - 1,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "d_rel": self.d_rel,
            "rel_extent": self.rel_extent,
            "q_bias": False,
            "o_bias": False,
            "log_scaling_n_floor": self.log_scaling_n_floor,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06,
            "use_value_norm_no_gamma": True,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": self.local_layer_ids,
            "layer_types": self.layer_types,
            "sconv_kernel_size": self.conv_kernel,
            "unpadded_vocab_size": self.unpadded_vocab_size,
            "final_logit_softcapping": 30.0,
            "swa_head_dim": self.head_dim,
            "swa_num_attention_heads": self.num_heads,
            "swa_num_key_value_heads": self.num_kv_heads,
            "sliding_window_size": self.sliding_window,
            "n_layers_per_attn_res_block": self.n_layers_per_attn_res_block,
            "attn_res_normalize_values": True,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_shared_experts": 0,
            "intermediate_size": self.moe_intermediate,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "use_gate_bias": True,
            "loop_config": {
                "core_start": self.loop_start,
                "core_end": self.loop_end,
                "num_passes": self.loop_num_passes,
                "gate_rank": self.loop_gate_rank,
                "is_varloop": False,
            },
        }

    def config(self) -> dict:
        return {
            "dtype": "bfloat16",
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "model_type": "inkling2_loop",
            "eos_token_id": self.unpadded_vocab_size - 1,
            "text_config": self.text_config(),
            # The transformers config validates ``vision_config`` against the
            # text width; the tiny checkpoint ships no vision weights.
            "vision_config": {
                "model_type": "inkling2_vision",
                "text_hidden_size": self.hidden_size,
                "native_patch_size": 4,
                "adapter_merge_size": 2,
                "projection_hidden_dim": 32,
                "projector_bias": False,
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "mlp_ratio": 2.0,
                "posemb_grid_size": 4,
                "layer_norm_eps": 1e-05,
            },
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        }


class _Rng:
    def __init__(self, seed: int) -> None:
        self._gen = torch.Generator().manual_seed(seed)

    def normal(self, *shape: int, std: float = 1.0, mean: float = 0.0) -> torch.Tensor:
        return torch.randn(*shape, generator=self._gen) * std + mean


def quantise_nvfp4(w: torch.Tensor):
    """fp32 ``[E, N, K]`` -> (packed U8 ``[E, N, K/2]``, e4m3 scales ``[E, N, K/16]``,
    amax ``[E]``).  Per-expert global scale ``amax / 2688``, per-16 block scale."""
    e, n, k = w.shape
    amax = w.abs().amax(dim=(1, 2))
    glob = (amax / NVFP4_DENOM)[:, None, None]
    blocks = w.reshape(e, n, k // 16, 16)
    block_max = blocks.abs().amax(-1)
    scale_q = (block_max / 6.0 / glob).clamp(2.0**-9, 448.0).to(torch.float8_e4m3fn)
    scale = scale_q.float() * glob
    q = blocks / scale[..., None]
    idx = (q.abs()[..., None] - FP4_MAGNITUDES).abs().argmin(-1)
    nib = idx | (((q < 0) & (idx != 0)).long() << 3)
    nib = nib.reshape(e, n, k)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scale_q.contiguous(), amax.contiguous()


def quantise_mxfp8(w: torch.Tensor):
    """fp32 ``[E, N, K]`` -> (e4m3 ``[E, N, K]``, UE8M0 U8 scales ``[E, N, K/32]``)."""
    e, n, k = w.shape
    blocks = w.reshape(e, n, k // 32, 32)
    block_max = blocks.abs().amax(-1).clamp(min=2.0**-100)
    exp = torch.ceil(torch.log2(block_max / 448.0)).clamp(-127, 127)
    scale = (exp + 127).to(torch.uint8)
    q = (blocks / torch.exp2(exp)[..., None]).clamp(-448.0, 448.0)
    return q.to(torch.float8_e4m3fn).reshape(e, n, k).contiguous(), scale.contiguous()


def build_tensors(spec: TinySpec, seed: int) -> dict[str, torch.Tensor]:
    """All checkpoint tensors, keyed like ``model.llm.*`` in the real file."""
    r = _Rng(seed)
    h, vocab, n_layers = spec.hidden_size, spec.vocab_size, spec.num_hidden_layers
    full = set(spec.full_layer_ids)
    bf16 = torch.bfloat16
    t: dict[str, torch.Tensor] = {}
    p = "model.llm."

    def dense(name: str, *shape: int, fan_in: int) -> None:
        t[name] = r.normal(*shape, std=fan_in**-0.5).to(bf16)

    t[p + "embed.weight"] = r.normal(vocab, h).to(bf16)
    t[p + "embed_norm.weight"] = (
        (r.normal(h, std=0.35, mean=0.74)).clamp(min=0.2).to(bf16)
    )
    t[p + "norm.weight"] = r.normal(h, std=0.75, mean=5.6).clamp(3.9, 8.0).to(bf16)
    dense(p + "unembed.weight", vocab, h, fan_in=h)  # padding rows stay random

    for i in range(n_layers - 1):  # query of layer i + 1
        shape = (3, h) if (i + 1) in full else (h,)
        t[f"{p}attn_res.attn_queries.{i}"] = r.normal(*shape, std=0.4)
    for i in range(n_layers):
        t[f"{p}attn_res.mlp_queries.{i}"] = r.normal(h, std=0.4)
    t[p + "attn_res.output_query"] = r.normal(h, std=0.4)
    n_loop = spec.loop_end - spec.loop_start
    # two slots per loop layer in each tensor (see ref_checkpoint.py)
    t[p + "middle_cycle_cross_pass_queries"] = r.normal(2 * n_loop, h, std=0.7)
    t[p + "middle_cycle_cross_pass_kv_queries"] = r.normal(2 * n_loop, h, std=0.7)

    inter, n_exp, kk = spec.moe_intermediate, spec.n_routed_experts, spec.conv_kernel
    for i in range(n_layers):
        lp = f"{p}layers.{i}."
        is_full = i in full
        a = lp + "attn."
        dense(a + "wq_du.weight", spec.q_dim, h, fan_in=h)
        dense(a + "wk_dv.weight", spec.kv_dim, h, fan_in=h)
        dense(a + "wv_dv.weight", spec.kv_dim, h, fan_in=h)
        dense(a + "wo_ud.weight", h, spec.q_dim, fan_in=spec.q_dim)
        dense(a + "gate_attn.weight", spec.q_dim, h, fan_in=h)
        dense(a + "wr_du.weight", spec.num_heads * spec.d_rel, h, fan_in=h)
        dense(a + "attn_sink_proj.weight", spec.num_heads, h, fan_in=h)
        t[a + "q_l2_clip.scale"] = r.normal(
            spec.num_heads, spec.head_dim, std=0.3, mean=1.0
        ).to(bf16)
        t[a + "rel_l2_clip.scale"] = r.normal(
            spec.num_heads, spec.d_rel, std=1.0, mean=1.0
        ).to(bf16)
        extent = spec.rel_extent if is_full else spec.sliding_window
        t[a + "rel_logits_proj.proj"] = r.normal(spec.d_rel, extent, std=0.3).to(bf16)
        for side in ("k", "v"):
            if is_full:
                t[f"{a}{side}_sconv.static_w"] = r.normal(
                    kk, spec.kv_dim, std=1.0, mean=0.3
                ).to(bf16)
                dense(
                    f"{a}{side}_sconv.weight_proj", spec.num_kv_heads * kk, h, fan_in=h
                )
            else:
                t[f"{a}{side}_sconv.weight"] = r.normal(
                    spec.kv_dim, 1, kk, std=0.6, mean=0.25
                ).to(bf16)

        if is_full:
            for nm in ("q", "k", "v"):
                t[f"{lp}{nm}_input_norm.weight"] = (
                    r.normal(h, std=0.3, mean=1.0).clamp(min=0.1).to(bf16)
                )
        else:
            t[lp + "attn_norm.weight"] = (
                r.normal(h, std=0.3, mean=1.0).clamp(min=0.1).to(bf16)
            )
        t[lp + "mlp_norm.weight"] = (
            r.normal(h, std=0.2, mean=1.0).clamp(min=0.1).to(bf16)
        )
        t[lp + "attn_sconv.weight"] = r.normal(h, 1, kk, std=0.6, mean=0.25).to(bf16)
        t[lp + "mlp_sconv.weight"] = r.normal(h, 1, kk, std=0.4, mean=0.1).to(bf16)

        if i in spec.loop_layer_ids:
            for ctl in ("attn", "k", "v", "mlp"):
                c = f"{lp}controllers.{ctl}."
                dense(c + "down_weight", spec.loop_gate_rank, 2 * h, fan_in=2 * h)
                t[c + "forget_weight"] = r.normal(h, spec.loop_gate_rank, std=0.12).to(
                    bf16
                )
                t[c + "gain_weight"] = r.normal(h, spec.loop_gate_rank, std=0.12).to(
                    bf16
                )

        dense(lp + "mlp.gate.weight", n_exp, h, fan_in=h)
        t[lp + "mlp.gate.bias"] = r.normal(n_exp, std=0.05)

        ex = lp + "mlp.experts."
        w13 = r.normal(n_exp, 2 * inter, h, std=1.2 * h**-0.5)
        w2 = r.normal(n_exp, h, inter, std=0.6 * inter**-0.5)
        if spec.is_nvfp4_layer(i):
            for nm, w in (("w13", w13), ("w2", w2)):
                packed, scale, amax = quantise_nvfp4(w)
                t[f"{ex}{nm}_weight"] = packed
                t[f"{ex}{nm}_weight_scale"] = scale
                t[f"{ex}{nm}_weight_amax"] = amax
                t[f"{ex}{nm}_weight_outer_scale_denominator"] = torch.tensor(
                    [NVFP4_DENOM]
                )
        else:
            for nm, w in (("w13", w13), ("w2", w2)):
                q, scale = quantise_mxfp8(w)
                t[f"{ex}{nm}_weight"] = q
                t[f"{ex}{nm}_weight_scale"] = scale
    return t


def _parameter_groups(tensors: dict[str, torch.Tensor], spec: TinySpec) -> list[dict]:
    expert_w = {
        k
        for k in tensors
        if ".mlp.experts." in k and k.endswith(("w13_weight", "w2_weight"))
    }
    aux = ("_scale", "_amax", "_outer_scale_denominator")
    names = [k for k in tensors if ".mlp.experts." not in k or k in expert_w]
    last = f"layers.{spec.num_hidden_layers - 1}."
    nvfp4 = sorted(k for k in expert_w if last not in k)
    mxfp8 = sorted(k for k in expert_w if last in k)
    unembed = "model.llm.unembed.weight"
    fp32 = sorted(k for k in names if tensors[k].dtype == torch.float32)
    bf16 = sorted(
        k for k in names if tensors[k].dtype == torch.bfloat16 and k != unembed
    )
    assert not any(k.endswith(aux) for k in names)
    cast = {"weight_block_shape": [], "weight_transform": "cast"}
    return [
        {
            "compute_dtype": "float32",
            "output_dtype": "float32",
            "required_activation_quantization": None,
            "tensors": fp32,
            **cast,
        },
        {
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "required_activation_quantization": None,
            "tensors": bf16,
            **cast,
        },
        {
            "compute_dtype": "nvfp4",
            "output_dtype": "uint8",
            "required_activation_quantization": {
                "amax_type": "per_token",
                "block_size": 16,
                "format": "nvfp4_e2m1",
                "rounding": "nearest",
                "scale_dtype": "float8_e4m3fn",
                "scale_selection_rule": "fouroversix_mae",
            },
            "tensors": nvfp4,
            "weight_block_shape": [1, 16],
            "weight_scale_format": "amax_denominator",
            "weight_transform": "nvfp4_nearest",
        },
        {
            "compute_dtype": "mxfp8",
            "output_dtype": "float8_e4m3fn",
            "required_activation_quantization": {
                "block_size": 32,
                "format": "mxfp8_e4m3",
                "rounding": "nearest",
                "scale_dtype": "ue8m0",
            },
            "tensors": mxfp8,
            "weight_block_shape": [32, 32],
            "weight_transform": "mxfp8_nearest",
        },
        {
            "compute_dtype": "bfloat16",
            "fwd_output_dtype": "float32",
            "output_dtype": "bfloat16",
            "required_activation_quantization": None,
            "tensors": [unembed],
            **cast,
        },
    ]


def write_checkpoint(
    out: Path, spec: TinySpec | None = None, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Write the checkpoint directory and return the tensors that were saved."""
    spec = spec or TinySpec()
    out.mkdir(parents=True, exist_ok=True)
    tensors = build_tensors(spec, seed)
    save_file(tensors, str(out / SHARD), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in tensors.values())
    index = {
        "metadata": {"total_size": total},
        "weight_map": {k: SHARD for k in sorted(tensors)},
    }
    dump = {
        "config.json": spec.config(),
        "hf_quant_config.json": {
            "model": {"parameter_groups": _parameter_groups(tensors, spec)},
            "quant_method": "inkling2",
        },
        "model.safetensors.index.json": index,
        "generation_config.json": {
            "do_sample": False,
            "eos_token_id": spec.unpadded_vocab_size - 1,
        },
    }
    for name, payload in dump.items():
        (out / name).write_text(json.dumps(payload, indent=2) + "\n")
    return tensors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="checkpoint directory to create")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=TinySpec.num_hidden_layers)
    ap.add_argument("--loop-start", type=int, default=TinySpec.loop_start)
    ap.add_argument("--loop-end", type=int, default=TinySpec.loop_end)
    ap.add_argument("--loop-passes", type=int, default=TinySpec.loop_num_passes)
    ap.add_argument("--list", action="store_true", help="print every key, dtype, shape")
    args = ap.parse_args()
    spec = TinySpec(
        num_hidden_layers=args.num_layers,
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        loop_num_passes=args.loop_passes,
    )
    tensors = write_checkpoint(Path(args.out).expanduser(), spec, args.seed)
    print(
        f"wrote {len(tensors)} tensors to {args.out} (seed {args.seed}, "
        f"full layers {list(spec.full_layer_ids)}, "
        f"loop {spec.loop_start}:{spec.loop_end} x{spec.loop_num_passes})"
    )
    if args.list:
        for k in sorted(tensors):
            v = tensors[k]
            print(f"{k} {str(v.dtype).removeprefix('torch.')} {list(v.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
