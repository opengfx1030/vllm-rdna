#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect the AWQ checkpoint tensor shapes/dtypes for W4A8 real-weight tests."""
import json
import os
import struct
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4"
    "/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
)


def headers(fn):
    with open(fn, "rb") as f:
        ln = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(ln))
    return hdr


def find(name):
    for i in range(1, 20):
        fn = f"{MODEL}/model-{i:05d}-of-00005.safetensors"
        if not os.path.exists(fn):
            continue
        h = headers(fn)
        if name in h:
            return fn, h[name]
    return None, None


def main():
    from safetensors import safe_open

    for layer in (
        "model.language_model.layers.3.mlp.down_proj",
        "model.language_model.layers.3.mlp.gate_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.self_attn.q_proj",
    ):
        print("===", layer)
        fn_p, e_p = find(layer + ".weight_packed")
        for suf in (".weight_packed", ".weight_zero_point", ".weight_scale", ".weight_shape"):
            fn, e = find(layer + suf)
            if e:
                print(f"   {suf:20s} {e['shape']} {e['dtype']}")
        if fn_p:
            with safe_open(fn_p, framework="pt") as f:
                ws = f.get_tensor(layer + ".weight_shape")
                print("   weight_shape value:", ws.tolist())
        fn_z, _ = find(layer + ".weight_zero_point")
        if fn_z:
            with safe_open(fn_z, framework="pt") as f:
                z = f.get_tensor(layer + ".weight_zero_point")
                u = z.flatten().unique()[:20].tolist()
                print("   zero_point dtype:", z.dtype, "min/max:", z.min().item(), z.max().item(), "uniq[:20]:", u)


if __name__ == "__main__":
    main()
