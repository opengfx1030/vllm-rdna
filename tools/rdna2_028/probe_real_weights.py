#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight W4A8 vs W4A16 op test on the AWQ 27B checkpoint.

Loads actual compressed-tensors AWQ weights, converts to the kernel layout
(qweight [K/8,N] shuffled, qzeros [G,N/8], scales [G,N]), and compares
w4a8_gemm_rdna2 vs gptq_gemm_rdna2_prefill vs an fp32 dequant reference.
Decisive test for a *data-dependent* W4A8 bug that random weights miss.
"""
import os

import numpy as np
import torch
from safetensors import safe_open

SHUFFLE_SLOTS = (0, 2, 4, 6, 1, 3, 5, 7)
GROUP = 32
MODEL = (
    "/home/chenco_adm/.cache/huggingface/hub/models--cyankiwi--Qwen3.8-27B-AWQ-INT4"
    "/snapshots/63768c10df38c0395e12ef49edac1bd539eaeeea"
)


def _find(name):
    for i in range(1, 20):
        fn = f"{MODEL}/model-{i:05d}-of-00005.safetensors"
        if os.path.exists(fn):
            with safe_open(fn, framework="pt") as f:
                if name in f.keys():
                    return f.get_tensor(name)
    raise KeyError(name)


def _pack_k_major(q_kn):
    k, n = q_kn.shape
    q = q_kn.astype(np.uint32).reshape(k // 8, 8, n)
    out = np.zeros((k // 8, n), dtype=np.uint32)
    for i in range(8):
        out |= q[:, i, :] << np.uint32(4 * i)
    return out


def _exllama_shuffle(packed):
    packed = packed.astype(np.uint32)
    out = np.zeros_like(packed)
    for slot, k_off in enumerate(SHUFFLE_SLOTS):
        nibble = (packed >> np.uint32(4 * k_off)) & np.uint32(0xF)
        out |= nibble << np.uint32(4 * slot)
    return out


def _pack_zeros_n_major(z_gn):
    g, n = z_gn.shape
    z = z_gn.astype(np.uint32).reshape(g, n // 8, 8)
    out = np.zeros((g, n // 8), dtype=np.uint32)
    for j in range(8):
        out |= z[:, :, j] << np.uint32(4 * j)
    return out


def _unpack_q(wp, k, n):
    """wp [N, K/8] -> qkn [K, N]."""
    qkn = np.zeros((k, n), dtype=np.int64)
    for i in range(8):
        qkn[i::8, :] = (wp >> np.uint32(4 * i)).T & 0xF
    return qkn


def _unpack_z(wz, n, g):
    """wz [N/8, G] packed -> zng [N, G]."""
    zng = np.zeros((n, g), dtype=np.int64)
    for j in range(8):
        zng[j::8, :] = (wz >> np.uint32(4 * j)) & 0xF
    return zng


def test_layer(layer_name, k_shard=None, m=2001, label=""):
    wp = _find(layer_name + ".weight_packed").to(torch.int64).cpu().numpy()  # [N, K/8]
    ws = _find(layer_name + ".weight_scale").to(torch.float32).cpu().numpy()  # [N, G]
    wz = _find(layer_name + ".weight_zero_point").to(torch.int64).cpu().numpy()  # [N/8, G]
    wsh = _find(layer_name + ".weight_shape").cpu().numpy()
    n, k = int(wsh[0]), int(wsh[1])
    assert k % GROUP == 0

    if k_shard is not None:
        sh = k_shard
        wp = wp[:, 0 : sh // 8]
        ws = ws[:, 0 : sh // GROUP]
        wz = wz[:, 0 : sh // GROUP]
        k = sh

    qkn = _unpack_q(wp, k, n)
    zng = _unpack_z(wz, n, k // GROUP)
    s_full = np.repeat(ws, GROUP, axis=1).T  # [K, N]
    z_full = np.repeat(zng, GROUP, axis=1).T  # [K, N]
    x_np = (0.25 * np.random.default_rng(0).standard_normal((m, k))).astype(np.float32)
    x = torch.from_numpy(x_np).to(torch.float16).cuda()

    for tag, zoff in (("lit/off0", 0), ("gptq/off1", 1)):
        w = (qkn.astype(np.float32) - (z_full.astype(np.float32) + zoff)) * s_full.astype(np.float32)
        ref = torch.from_numpy(x_np @ w).cuda()

        wq = torch.from_numpy(
            _exllama_shuffle(_pack_k_major(qkn.astype(np.int32))).astype(np.uint32)
        ).cuda()
        wzp = torch.from_numpy(_pack_zeros_n_major(zng.T.astype(np.int32)).astype(np.uint32)).cuda()
        wsc = torch.from_numpy(ws.T.astype(np.float16).copy()).cuda()  # [G, N]
        gi = torch.empty(0, dtype=torch.int32, device="cuda")

        use_v2 = tag.startswith("lit")
        out8 = torch.ops._rocm_C.w4a8_gemm_rdna2(x, wq, wzp, wsc, gi, use_v2)
        outg = torch.ops._rocm_C.gptq_gemm_rdna2_prefill(x, wq, wzp, wsc, gi, use_v2)

        def rl(a, b):
            d = b.float().norm()
            return float("nan") if d == 0 else ((a.float() - b.float()).norm() / d).item()

        print(
            f"[{label}{layer_name.split('.')[-1]}] k={k} n={n} m={m} zero={tag}: "
            f"w4a8vref={rl(out8, ref):.5f} gptqvref={rl(outg, ref):.5f} "
            f"w4a8vgptq={rl(out8, outg):.5f}"
        )


def main():
    torch.manual_seed(0)
    test_layer("model.language_model.layers.3.self_attn.o_proj", label="FULL ")
    test_layer("model.language_model.layers.3.mlp.down_proj", k_shard=4352, label="SHARD")


if __name__ == "__main__":
    main()
