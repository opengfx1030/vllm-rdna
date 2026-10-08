# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the RDNA2 GPTQ MoE conversion + kernel against an fp32 dequant of real
Qwen3.6-35B GPTQ expert weights (layer 0, a few experts, gate_proj)."""

import json

import torch
from safetensors import safe_open

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _process_weights_rdna2

M = "/home/chenco_adm/models/Qwen3.6-35B-A3B-GPTQ-Int4"
with open(f"{M}/model.safetensors.index.json") as _fh:
    idx = json.load(_fh)["weight_map"]


def get(name):
    with safe_open(f"{M}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


pre = "model.language_model.layers.0.mlp.experts"
E, G = 4, 128
qw, sc, qz = [], [], []
for e in range(E):
    qw.append(get(f"{pre}.{e}.gate_proj.qweight"))  # [K/8, N] int32
    sc.append(get(f"{pre}.{e}.gate_proj.scales"))  # [K/G, N]
    qz.append(get(f"{pre}.{e}.gate_proj.qzeros"))  # [K/G, N/8]
K = qw[0].shape[0] * 8
N = qw[0].shape[1]
print(
    "K",
    K,
    "N",
    N,
    "scales",
    sc[0].dtype,
    tuple(sc[0].shape),
    "qzeros[0,0]",
    hex(int(qz[0][0, 0]) & 0xFFFFFFFF),
)
dev = "cuda"
# MoeWNA16 load transform: qweight.T.contiguous().view(uint8); scales.T
w13 = torch.stack([w.T.contiguous().view(torch.uint8) for w in qw]).to(
    dev
)  # [E, N, K/2]
s13 = torch.stack([s.T for s in sc]).to(dev).half()  # [E, N, G']
w2 = w13.clone()
s2 = s13.clone()
cw13, _, cs13, _, cz13, _, *_ = _process_weights_rdna2(w13, w2, s13, s2, G)


# fp32 reference dequant (GPTQ sym: zero point 8)
def deq(w, s):
    nib = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], 1).reshape(
        K, N
    )  # [K, N]
    return (nib.float() - 8) * s.float().repeat_interleave(G, 0)


torch.manual_seed(0)
x = torch.randn(3, K, dtype=torch.float16, device=dev)
ids = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int32, device=dev)
si, ei, ntp = moe_align_block_size(ids, 1, E)
out = torch.zeros(6, N, dtype=torch.float16, device=dev)
ops.moe_gptq_gemm_rdna2(
    x, out, cw13, cs13, cz13, torch.empty(0, device=dev), si, ei, ntp, 2, 1, False, 0
)
torch.accelerator.synchronize()
err = 0.0
mag = 0.0
for m in range(3):
    for k in range(2):
        e = int(ids[m, k])
        ref = x[m].float() @ deq(qw[e].to(dev), sc[e].to(dev))
        err = max(err, (out[m * 2 + k].float() - ref).abs().max().item())
        mag = max(mag, ref.abs().max().item())
print(f"max|kernel - ref| = {err:.4f}  (ref max {mag:.3f})")
