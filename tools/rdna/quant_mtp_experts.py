#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize the Flash-Next MTP drafter experts to W4A16 (offline, CPU only).

The AWQ/compressed-tensors Flash-Next checkpoints (e.g.
wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) quantize the main experts but leave the
MTP drafter in BF16, with its experts fused as
``mtp.layers.<i>.mlp.experts.gate_up_proj`` [E, 2*I, H] and ``down_proj``
[E, H, I]. This script RTN-quantizes those experts into the main experts'
compressed-tensors layout (symmetric int4, uint4b8 nibbles packed 8 per int32
along K, per-group scales), so the drafter MoE runs on the same W4A16 MoE
kernel as the target. Every other mtp.* tensor is copied unchanged.

The result is a derived checkpoint directory: one new shard for the MTP
tensors, a rewritten index and config (the ``ignore`` list keeps everything
under ``mtp.`` except the experts), and symlinks to every other source file.

Usage:
    python tools/rdna/quant_mtp_experts.py <src_checkpoint_dir> <dst_dir>

Before writing anything it checks one main expert: repacking its nibbles
gives the same int32 words (nibble order), and some group reaches q = -8
(the absmax / 7.5 scale convention).
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch

MTP_EXPERTS_RE = re.compile(
    r"^mtp\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
MAIN_EXPERT_PACKED_RE = re.compile(r"\.mlp\.experts\.\d+\.gate_proj\.weight_packed$")
# Everything under mtp. stays unquantized except the routed experts.
MTP_IGNORE_EXCEPT_EXPERTS = r"re:^mtp\.(?!layers\.\d+\.mlp\.experts(\.|$)).*"
_MTP_IGNORE_ALL = ("re:.*mtp\\..*", "re:^mtp.*")
NEW_SHARD = "model_mtp_w4a16.safetensors"


def pack_nibbles(nib: torch.Tensor) -> torch.Tensor:
    """[N, K] nibbles (0..15) -> [N, K/8] int32, element 0 in the low bits."""
    n, k = nib.shape
    nib = nib.to(torch.int32).reshape(n, k // 8, 8)
    packed = torch.zeros(n, k // 8, dtype=torch.int32)
    for i in range(8):
        packed |= nib[:, :, i] << (4 * i)
    return packed.contiguous()


def quant_rtn(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] float -> (packed int32 [N, K/8], scale bf16 [N, K/group_size]).

    compressed-tensors symmetric min-max, as the main experts were quantized:
    scale = absmax / 7.5, q = round(w / scale) clamped to [-8, 7] (so the
    negative extreme lands on -8), stored as uint4b8 (nibble = q + 8,
    dequant = (nibble - 8) * scale); nibble 0 in the low bits.
    """
    n, k = w.shape
    if k % group_size or k % 8:
        raise ValueError(f"K={k} must be a multiple of {group_size} and 8")
    wf = w.float().reshape(n, k // group_size, group_size)
    scale = wf.abs().amax(dim=-1).clamp_min(1e-8) / 7.5
    q = torch.round(wf / scale[..., None]).clamp_(-8, 7).reshape(n, k)
    return pack_nibbles(q + 8), scale.to(torch.bfloat16).contiguous()


def unpack(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of :func:`quant_rtn`: [N, K/8] int32 + scales -> [N, K] fp32."""
    n, k8 = packed.shape
    p = packed.to(torch.int64) & 0xFFFFFFFF
    nib = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    q = nib.reshape(n, k8 * 8).float() - 8
    return (q.reshape(n, -1, group_size) * scale.float()[..., None]).reshape(n, -1)


def mtp_ignore_list(ignore: list[str]) -> list[str]:
    """Drop the catch-all mtp ignores and keep mtp.* minus the experts."""
    kept = [p for p in ignore if p not in _MTP_IGNORE_ALL]
    if MTP_IGNORE_EXCEPT_EXPERTS not in kept:
        kept.append(MTP_IGNORE_EXCEPT_EXPERTS)
    return kept


def weight_scheme(quant_config: dict) -> tuple[int, int, bool]:
    """(num_bits, group_size, symmetric) of the checkpoint's weight scheme."""
    for group in quant_config.get("config_groups", {}).values():
        weights = group.get("weights") or {}
        if weights:
            return (
                int(weights["num_bits"]),
                int(weights["group_size"]),
                bool(weights.get("symmetric", True)),
            )
    raise ValueError("no weight scheme in quantization_config.config_groups")


def quantize_mtp_experts(
    tensors: dict[str, torch.Tensor], group_size: int
) -> tuple[dict[str, torch.Tensor], float]:
    """Replace fused MTP expert tensors with per-expert W4A16 tensors.

    Returns the new tensor dict and the mean relative |dw|/|w| of the first
    four experts (a sanity number, ~0.1 for int4 g128).
    """
    out: dict[str, torch.Tensor] = {}
    fused: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in tensors.items():
        m = MTP_EXPERTS_RE.match(name)
        if m:
            fused.setdefault(m.group(1), {})[m.group(2)] = t
        else:
            out[name] = t
    if not fused:
        raise ValueError("no mtp.layers.<i>.mlp.experts.{gate_up,down}_proj tensors")

    err_sum, err_n = 0.0, 0
    for layer, parts in sorted(fused.items()):
        gu, dn = parts["gate_up_proj"], parts["down_proj"]
        num_experts, two_inter, hidden = gu.shape
        inter = two_inter // 2
        if dn.shape != (num_experts, hidden, inter):
            raise ValueError(
                f"down_proj shape {tuple(dn.shape)} != {(num_experts, hidden, inter)}"
            )
        for e in range(num_experts):
            for proj, w in (
                ("gate_proj", gu[e, :inter]),
                ("up_proj", gu[e, inter:]),
                ("down_proj", dn[e]),
            ):
                packed, scale = quant_rtn(w, group_size)
                pre = f"mtp.layers.{layer}.mlp.experts.{e}.{proj}."
                out[pre + "weight_packed"] = packed
                out[pre + "weight_scale"] = scale
                out[pre + "weight_shape"] = torch.tensor(
                    list(w.shape), dtype=torch.int64
                )
                if e < 4:
                    wq = unpack(packed, scale, group_size)
                    wf = w.float()
                    err_sum += ((wq - wf).abs().mean() / wf.abs().mean()).item()
                    err_n += 1
    return out, err_sum / max(err_n, 1)


def _check_packing(src: Path, weight_map: dict[str, str], group_size: int) -> None:
    from safetensors import safe_open

    key = next((k for k in weight_map if MAIN_EXPERT_PACKED_RE.search(k)), None)
    if key is None:
        raise SystemExit(
            "no main-expert gate_proj.weight_packed to check the packing against"
        )
    base = key.removesuffix("weight_packed")
    with safe_open(src / weight_map[key], "pt") as f:
        p0 = f.get_tensor(base + "weight_packed")
    p = p0.to(torch.int64) & 0xFFFFFFFF
    nib = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    nib = nib.reshape(p0.shape[0], -1)
    order_ok = torch.equal(pack_nibbles(nib), p0)
    # absmax / 7.5 puts a group's negative extreme on q = -8; absmax / 7 never
    # reaches it. Some group of a real expert must hit -8.
    hits_minus8 = bool((nib == 0).any())
    print(
        f"packing check on {base}: nibble order matches={order_ok}, "
        f"q=-8 present={hits_minus8} (scale = absmax / 7.5)"
    )
    if not (order_ok and hits_minus8):
        raise SystemExit(
            "main-expert packing or scale convention differs from this tool; "
            "not writing"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="source checkpoint directory")
    ap.add_argument("dst", type=Path, help="derived checkpoint directory (created)")
    ap.add_argument(
        "--skip-check",
        action="store_true",
        help="skip the main-expert packing and scale-convention check",
    )
    args = ap.parse_args(argv)

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = args.src, args.dst
    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg["quantization_config"]
    num_bits, group_size, symmetric = weight_scheme(qcfg)
    if num_bits != 4 or not symmetric:
        raise SystemExit(
            "expected symmetric int4 main experts, "
            f"got bits={num_bits} symmetric={symmetric}"
        )

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    if not args.skip_check:
        _check_packing(src, weight_map, group_size)

    mtp_files = {f for k, f in weight_map.items() if MTP_EXPERTS_RE.match(k)}
    if not mtp_files:
        raise SystemExit("no fused BF16 MTP experts in the index (already quantized?)")
    tensors: dict[str, torch.Tensor] = {}
    for fn in sorted(mtp_files):
        with safe_open(src / fn, "pt") as f:
            for name in f.keys():  # noqa: SIM118
                tensors[name] = f.get_tensor(name)
    out, rel_err = quantize_mtp_experts(tensors, group_size)
    print(f"mean relative |dw|/|w| (first 4 experts): {rel_err:.4f}")

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / NEW_SHARD), metadata={"format": "pt"})
    size_gib = (dst / NEW_SHARD).stat().st_size / 2**30
    print(f"wrote {NEW_SHARD}: {size_gib:.2f} GiB, {len(out)} tensors")

    new_map = {k: v for k, v in weight_map.items() if v not in mtp_files}
    new_map.update({k: NEW_SHARD for k in out})
    index["weight_map"] = new_map
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    qcfg["ignore"] = mtp_ignore_list(qcfg.get("ignore", []))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print("ignore:", qcfg["ignore"])

    skip = {"config.json", "model.safetensors.index.json", *mtp_files}
    for entry in src.iterdir():
        if entry.name in skip or entry.name.startswith("."):
            continue
        link = dst / entry.name
        if not link.exists():
            os.symlink(entry.resolve(), link)
    print("checkpoint ready:", dst)


if __name__ == "__main__":
    main()
