# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank rdna_hint_flex weight layout of a Qwen3.5-family checkpoint.

Reads only the small per-row fields (``hi2_row_nbits``, ``hi2_row_block``,
``hi2_had``, ``hi2_col_scale``) of the checkpoint, applies the tensor-parallel
partitioning the loader uses (``RdnaHintFlexLinearMethod._partitions``) and
reports, for every fused linear of every layer, the rank's R, K, INT2/4/8 row
mix, number of kernel segments (consecutive partitions that share one
activation transform are one segment) and the bytes one decode token reads.
It also lists the dense tensors a decode token reads (lm_head, MTP layer).

  python tools/rdna/rdna_hint_flex/tp_layout.py MODEL_DIR [--tp 2] [--rank 0]
      [--json out.json]

``slot_specs`` and ``part_sources`` are what bench_tp2.py uses to pick the
checkpoint rows that make up each fused layer.
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open

ROW_META_BYTES = 16  # code_off 4 + param_off 4 + row_consts 6 + nbits 1 + block 1


class Reader:
    """Tensor access by name through the safetensors index."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.wm: dict[str, str] = json.load(f)["weight_map"]
        self._open: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.wm

    def _file(self, name: str):
        fn = self.wm[name]
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self._open[fn]

    def get(self, name: str) -> torch.Tensor:
        return self._file(name).get_tensor(name)


def text_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    return c.get("text_config", c)


def slot_specs(cfg: dict, tp: int, layer: int) -> list[dict]:
    """The fused linears of one decoder layer. Each has `k` (this rank's input
    columns), `k_full`, and `parts`: one entry per partition of the fused layer
    with the checkpoint `module`, the partition's first row inside that tensor
    (`base`), its `full` row count and the rank's `rows` of it."""
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    hd = cfg["head_dim"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    gate = 2 if cfg.get("attn_output_gate", False) else 1
    kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    nv = cfg["linear_num_value_heads"]
    kind = cfg["layer_types"][layer]
    pre = f"model.language_model.layers.{layer}"

    def part(module: str, full: int, base: int = 0, replicated: bool = False) -> dict:
        return {
            "module": module,
            "base": base,
            "full": full,
            "rows": full if replicated else full // tp,
            "sharded": not replicated,
        }

    def slot(name: str, k_full: int, k: int, parts: list[dict]) -> dict:
        return {"slot": name, "kind": kind, "k_full": k_full, "k": k, "parts": parts}

    out: list[dict] = []
    if kind == "full_attention":
        sa = f"{pre}.self_attn"
        kv_rep = nkv < tp
        out.append(
            slot(
                "qkv_proj",
                h,
                h,
                [
                    part(f"{sa}.q_proj", nh * hd * gate),
                    part(f"{sa}.k_proj", nkv * hd, replicated=kv_rep),
                    part(f"{sa}.v_proj", nkv * hd, replicated=kv_rep),
                ],
            )
        )
        out.append(
            slot(
                "o_proj",
                nh * hd,
                nh * hd // tp,
                [part(f"{sa}.o_proj", h, replicated=True)],
            )
        )
    else:
        la = f"{pre}.linear_attn"
        out.append(
            slot(
                "in_proj_qkvz",
                h,
                h,
                [
                    part(f"{la}.in_proj_qkv", kd, base=0),
                    part(f"{la}.in_proj_qkv", kd, base=kd),
                    part(f"{la}.in_proj_qkv", vd, base=2 * kd),
                    part(f"{la}.in_proj_z", vd),
                ],
            )
        )
        out.append(
            slot(
                "in_proj_ba",
                h,
                h,
                [part(f"{la}.in_proj_b", nv), part(f"{la}.in_proj_a", nv)],
            )
        )
        out.append(
            slot(
                "out_proj",
                vd,
                vd // tp,
                [part(f"{la}.out_proj", h, replicated=True)],
            )
        )
    out.append(
        slot(
            "gate_up_proj",
            h,
            h,
            [part(f"{pre}.mlp.gate_proj", inter), part(f"{pre}.mlp.up_proj", inter)],
        )
    )
    out.append(
        slot(
            "down_proj",
            inter,
            inter // tp,
            [part(f"{pre}.mlp.down_proj", h, replicated=True)],
        )
    )
    return out


def part_rows(p: dict, rank: int) -> tuple[int, int]:
    """(first checkpoint row, row count) of this rank's share of a partition."""
    return p["base"] + (rank * p["rows"] if p["sharded"] else 0), p["rows"]


def cols_of(spec: dict, rank: int) -> tuple[int, int]:
    """This rank's input-column span of a fused linear."""
    if spec["k"] == spec["k_full"]:
        return 0, spec["k_full"]
    return rank * spec["k"], (rank + 1) * spec["k"]


def _same(a: dict, b: dict) -> bool:
    ca, cb = a["col_scale"], b["col_scale"]
    if (ca is None) != (cb is None):
        return False
    if ca is not None and not torch.equal(ca, cb):
        return False
    return torch.equal(a["had"], b["had"])


def segments_of(rd: Reader, spec: dict, rank: int) -> list[dict]:
    """Group consecutive partitions that share (col_scale, had) into segments;
    each has the per-row `nbits`/`block` arrays, `k` and the transform facts."""
    k0, k1 = cols_of(spec, rank)
    cache: dict[str, dict] = {}
    parts = []
    for p in spec["parts"]:
        m = p["module"]
        if m not in cache:
            cache[m] = {
                "nbits": rd.get(m + ".hi2_row_nbits"),
                "block": rd.get(m + ".hi2_row_block"),
                "had": rd.get(m + ".hi2_had"),
                "col_scale": rd.get(m + ".hi2_col_scale")
                if rd.has(m + ".hi2_col_scale")
                else None,
            }
        c = cache[m]
        r0, n = part_rows(p, rank)
        scale = c["col_scale"]
        parts.append(
            {
                "nbits": c["nbits"][r0 : r0 + n],
                "block": c["block"][r0 : r0 + n],
                "had": c["had"][k0 // 16 : k1 // 16],
                "col_scale": None if scale is None else scale[k0:k1],
            }
        )
    segs: list[dict] = []
    for p in parts:
        if segs and _same(segs[-1], p):
            segs[-1]["nbits"] = torch.cat([segs[-1]["nbits"], p["nbits"]])
            segs[-1]["block"] = torch.cat([segs[-1]["block"], p["block"]])
        else:
            segs.append(dict(p))
    for s in segs:
        s["k"] = k1 - k0
        s["has_col_scale"] = s["col_scale"] is not None
        s["rotated"] = int(s["had"].sum())
    return segs


def seg_bytes(seg: dict) -> dict:
    nb = seg["nbits"].to(torch.int64)
    bl = seg["block"].to(torch.int64)
    k = seg["k"]
    return {
        "codes": int((k * nb // 32).sum()) * 4,
        "params": int(2 * (k // bl).sum()),
        "rowmeta": ROW_META_BYTES * int(nb.numel()),
        "xform": k // 16 + (2 * k if seg["has_col_scale"] else 0),
    }


def dense_bytes(rd: Reader, cfg: dict, tp: int) -> dict:
    """fp16 bytes the rank reads per forward through the dense tensors."""
    h, v, inter = cfg["hidden_size"], cfg["vocab_size"], cfg["intermediate_size"]
    hd = cfg["head_dim"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    gate = 2 if cfg.get("attn_output_gate", False) else 1
    d = {"lm_head": v // tp * h * 2}
    if rd.has("mtp.fc.weight"):
        d["mtp_layer"] = (
            nh * hd * gate // tp * h  # q
            + 2 * max(nkv // tp, 1) * hd * h  # k, v
            + h * nh * hd // tp  # o
            + 2 * (inter // tp) * h  # gate_up
            + h * (inter // tp)  # down
            + h // tp * 2 * h  # fc: column-parallel, gather_output
        ) * 2
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = text_config(args.model)
    rd = Reader(args.model)
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    ks: dict[str, int] = {}
    segs_hist: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))
    for layer in range(cfg["num_hidden_layers"]):
        for spec in slot_specs(cfg, args.tp, layer):
            segs = segments_of(rd, spec, args.rank)
            a = agg[spec["slot"]]
            ks[spec["slot"]] = spec["k"]
            a["layers"] += 1
            a["segments"] += len(segs)
            segs_hist[spec["slot"]][len(segs)] += 1
            for s in segs:
                nb = s["nbits"]
                a["rows"] += nb.numel()
                a["int4"] += int((nb == 4).sum())
                a["int8"] += int((nb == 8).sum())
                for k2, v2 in seg_bytes(s).items():
                    a[k2] += v2
    print(f"TP={args.tp} rank={args.rank} model={args.model}")
    print(
        f"{'slot':13s} {'layers':>6s} {'seg/lyr':>7s} {'R':>6s} {'K':>6s} "
        f"{'%int4':>6s} {'%int8':>6s} {'MB/layer':>9s} {'MB total':>9s}  seg histogram"
    )
    total = 0.0
    summary = {}
    for slot, a in agg.items():
        n = a["layers"]
        by = a["codes"] + a["params"] + a["rowmeta"] + a["xform"]
        total += by
        print(
            f"{slot:13s} {int(n):6d} {a['segments'] / n:7.2f} {a['rows'] / n:6.0f} "
            f"{ks[slot]:6d} {100 * a['int4'] / a['rows']:6.1f} "
            f"{100 * a['int8'] / a['rows']:6.1f} {by / n / 1e6:9.2f} {by / 1e6:9.1f}"
            f"  {dict(segs_hist[slot])}"
        )
        summary[slot] = {
            "layers": int(n),
            "segments_per_layer": a["segments"] / n,
            "R": a["rows"] / n,
            "K": ks[slot],
            "frac_int4": a["int4"] / a["rows"],
            "frac_int8": a["int8"] / a["rows"],
            "bytes_per_layer": by / n,
            "codes": a["codes"] / n,
            "params": a["params"] / n,
            "rowmeta": a["rowmeta"] / n,
            "xform": a["xform"] / n,
        }
    d = dense_bytes(rd, cfg, args.tp)
    print(f"hi2 linears, bytes read per rank per token: {total / 1e9:.3f} GB")
    for k2, v2 in d.items():
        print(f"{k2:13s} {v2 / 1e9:.3f} GB per forward")
    tot = total + d["lm_head"]
    for bw in (512.0, 437.0):
        print(f"floor hi2 + lm_head at {bw:.0f} GB/s: {tot / bw / 1e6:.2f} ms/token")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "dense": d, "hi2_bytes": total}, f, indent=1)


if __name__ == "__main__":
    main()
