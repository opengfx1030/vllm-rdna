#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only parametric performance model for the heterogeneous MoE design.

Anchors are measured TP4/TP8 Flash-Next W4A16 + MTP2 numbers on V620 (the
constants below cite their source).  Everything about the W7800 hub and the
V620 expert servers is an assumption held in one named parameter, so a later
GPU measurement replaces a parameter, not the model.  The model ranks
topologies and says which measurement decides each question; it does not
promise tok/s.

    python tools/rdna/hetero_moe/perf_model.py            # all tables
    python tools/rdna/hetero_moe/perf_model.py --sens     # + sensitivity
    python tools/rdna/hetero_moe/perf_model.py --json     # machine readable

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

MiB = 1024**2
GiB = 1024**3

# ---------------------------------------------------------------------------
# Checkpoint bytes: exact, from the safetensors headers (spec section 2)
# ---------------------------------------------------------------------------
EXPERT_B = 2_534_400  # one routed expert, W4A16 g128
EXPERT_MIB = EXPERT_B / MiB
EXPERT_BF16_MIB = 9.375  # MTP experts ship bf16 (production TP4 reads these)
N_EXP, TOPK, N_LAYERS, N_GDN, N_QSA = 512, 10, 48, 36, 12
GDN_W_MIB = 110.547
QSA_W_MIB = 95.0 + 3.125  # attention + indexer
HC_MIB = 25.195  # both sublayers, replicated on every TP rank
SHARED_MIB = 9.38
ROUTER_MIB = 2.5
LM_HEAD_MIB = 1.184 * 1024
GDN_STATE_MIB = 3.06  # per sequence per GDN layer (ssm fp32 + conv)
AVG_W_MIB = (N_GDN * GDN_W_MIB + N_QSA * QSA_W_MIB) / N_LAYERS
ROW_B = 2560 * 2  # one hidden row, fp16
ROW_REQ_B = ROW_B + TOPK * 6  # x + 10 ids/weights ("5.2 KiB")

# GiB table for the memory model (sum of the byte table in the spec)
GIB = dict(
    gdn=3.886, qsa=1.113, indexer=0.037, hc=1.193, router=0.117, shared=0.440,
    embed=1.184, lm_head=1.184, ple_proj=0.061, mtp_dense=0.169,
    mtp_experts_bf16=4.688, mtp_experts_w4=1.20, experts=58.0, vision=0.836,
)  # fmt: skip
EXPERT_QZ_B = 19_200  # V620 repack adds synthesized qzeros per expert


@dataclass
class Gpu:
    name: str
    arch: str
    bw_peak: float  # B/s
    eta: float  # achieved fraction of peak on streaming kernels
    vram: int
    fp16: float  # FLOP/s peak (matrix for W7800)

    @property
    def bw(self) -> float:
        return self.bw_peak * self.eta


# V620: 512 GB/s, 380 GB/s achieved on expert GEMMs at W=3
# (docs/rdna/monokernel/p0-results.md:262); 29.98 GiB (sysfs).
V620 = Gpu("V620", "gfx1030", 512e9, 380 / 512, 32_195_477_504, 40.5e12)
# W7800 48GB: 864 GB/s, 96 MB Infinity Cache, 90.4 TFLOPS FP16 matrix (AMD
# product page); 44.98 GiB visible (sysfs).  eta is an assumption.
W7800 = Gpu("W7800", "gfx1100", 864e9, 0.80, 48_301_604_864, 90.4e12)

# ---------------------------------------------------------------------------
# TP4 measured decode (older eager trace, tree 15e9e7adbd; p0-results.md:149-279)
# per-layer averages in us at W=3 (c1), and the c8 block times at W=24
# ---------------------------------------------------------------------------
OLD_ATTN_BLOCK, OLD_HC = 252.0, 66.0  # 12.33 ms / 49 blocks; HC pre per sublayer
OLD_SHARED, OLD_ROUTER, OLD_EXPERTS, OLD_RESID = 19.0, 39.0, 52.0, 4.0
OLD_AR = 10.6  # 1.04 ms / 98 calls
OLD_ATTN_CORE = OLD_ATTN_BLOCK - OLD_HC
OLD_ATTN_BLOCK_24 = 490.0  # 24.04 ms / 49
TOK_US = (OLD_ATTN_BLOCK_24 - OLD_ATTN_BLOCK) / 21.0  # per extra token, TP4 GPU

# Today's measured numbers (docs/rdna2/tp8-pp2-flashnext-2026-11-06.md; TP8 c8
# 41.56 ms is the flat all-reduce tree, hier gives 37.2)
TODAY = {
    "TP4": dict(c1=22.38, c8=43.97, c8_tps=262.6, pre_tps=2966.0, pre8_tps=2806.0),
    "TP8": dict(c1=19.59, c8=41.56, c8_tps=290.3, pre_tps=3100.0),
}
ACC = 2.01  # tokens per step at 1k c8 (tp8-pp2 doc:431; c1 is 1.84)
# Prefill chunks block the decode steps of the other streams (ITL p99 370 ms at c8).
# One constant, in chunks of prefill time per 1k c8 run, makes TP4 reproduce its
# measured 262.6 tok/s: (4096 / 262.6 - 8192 / 2806 - 512 / ACC * 0.04397) / 0.69.
STALL_CHUNKS = 2.14

# all-reduce kernel time per call, us (c1 rows, c8 rows).  TP4 c1 from the
# trace (10.6); TP4 c8 rdna_ar one-shot 43 us at 120 KiB; TP8 flat one-shot 29 /
# 107 us (tp8-pp2 doc); TP2 across roots about 7+7 us one-way + add [INFERENCE]
AR_US = {2: (18.0, 28.0), 4: (OLD_AR, 45.0), 8: (29.0, 107.0)}

PCIE_BW = 25.0e9  # push, per GPU link, per direction (audit)
PCIE_LAT_US = 6.3  # one-way flag latency intra-switch (6.1-6.3)
# HxServer's torch-free loopback, four V620 (hub HIP 1, three servers, real kernels,
# graph mode, synthetic server compute): hx layer time p50 in us at T rows
# (docs/rdna2/hetero-hx-2026-11-07.md section 8.3, feat/hetero-hx).  Exchange alone.
LOOPBACK_US = {3: 35.9, 24: 64.2, 48: 98.6}
LOOPBACK_D = 3
LOOPBACK_MODE = {3: "c1", 24: "c8"}  # the decode step width each row stands for

EMAX = {1: 0.0, 2: 0.564, 4: 1.029, 8: 1.424}  # E[max of D std normals]


@dataclass
class Assume:
    """Every unmeasured knob.  Defaults are the central case."""

    eta_h: float = 0.80  # W7800 achieved fraction of 864 GB/s on streaming kernels
    kappa: float = 1.00  # W7800 small-kernel chain speed vs V620 (port quality x clock)
    gamma: float = 0.30  # share of small-kernel time that grows with per-GPU heads
    comp_ratio: float = 1.25  # per-token head-parallel kernels, W7800 / V620
    phi: float = 0.60  # per-token work that does not shard with TP (TP8 c8 says >= 0.7)
    rho_hc: float = 0.5  # share of the HC weight-stream time that is truly BW bound
    rho_small: float = 0.5  # same for router / shared expert GEMVs
    hit: float = 0.60  # share of distinct touched experts resident on the hub
    detect_us: float = 3.0  # V620 server: device-gated wake (host driven: ~30)
    eff_hub: float = 0.30  # W7800 all-in prefill FLOP efficiency vs 90.4 TFLOPS
    eff_v620_moe: float = 10e12  # V620 all-in expert-GEMM FLOP/s at prefill
    prefill_hub_share: float = 0.15  # share of prefill expert FLOPs run on a hub
    hub_dense_scale: float = 1.0  # hub dense weights: 0.5 = W8A16, 0.25 = W4A16
    lmhead_remote: bool = False  # vocab-parallel lm_head on the expert servers
    mtp_w4: bool = True  # hub MTP experts requantized to W4A16
    locality: float = 0.90  # distinct experts at W>=24 vs uniform (TP4 delta fit)


@dataclass
class Topo:
    name: str
    gpu: Gpu
    tp: int  # dense TP degree; EP degree when experts are local
    experts: str  # "local" or "hub"
    n_hubs: int = 1  # EP halves (hub topologies)
    servers_per_hub: int = 4
    mtp_expert_mib: float = EXPERT_BF16_MIB
    dense_scale: float = 1.0
    srv_share: float = 1.0  # hubs sharing one expert-server set (2 = H2')
    s_l: dict = field(default_factory=dict)  # calibrated latency scale per mode


def bytes_us(mib: float, gpu: Gpu, rho: float = 1.0) -> float:
    """Stream time of mib MiB; (1-rho) of it does not speed up on faster HBM."""
    return mib * MiB * (rho / gpu.bw + (1.0 - rho) / V620.bw) * 1e6


def n_touched(W: int, a: Assume) -> float:
    q = TOPK / N_EXP
    n = N_EXP * (1.0 - (1.0 - q) ** W)
    loc = 1.0 - (1.0 - a.locality) * min(1.0, (W - 1) / 23.0)
    return n * loc


def n_max(mu: float, D: int) -> float:
    if D <= 1:
        return mu
    return mu + EMAX[D] * math.sqrt(max(mu, 0.0) * (1.0 - 1.0 / D))


def tok_scale(t: Topo, a: Assume) -> float:
    speed = a.comp_ratio if t.gpu.arch == "gfx1100" else 1.0
    return (a.phi + (1.0 - a.phi) * 4.0 / t.tp) / speed


def attn_core_us(t: Topo, W: int, S: int, a: Assume, s: float) -> float:
    g = t.gpu
    w = AVG_W_MIB * t.dense_scale / t.tp
    st = S * 2 * (N_GDN / N_LAYERS) * GDN_STATE_MIB / t.tp
    b = bytes_us(w, g, 0.9) + bytes_us(st, g, 1.0)
    b4 = bytes_us(AVG_W_MIB / 4, V620, 0.9) + bytes_us(
        2 * (N_GDN / N_LAYERS) * GDN_STATE_MIB / 4, V620, 1.0
    )
    l0 = max(OLD_ATTN_CORE - b4, 20.0)
    kap = a.kappa if g.arch == "gfx1100" else 1.0
    grow = 1.0 + a.gamma * (4.0 / t.tp - 1.0)
    lat = l0 * kap * grow * s
    tok = TOK_US * max(W - 3, 0) * tok_scale(t, a) * kap * s
    return b + lat + tok


def hc_us(t: Topo, a: Assume, s: float) -> float:
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    lat = max(2 * OLD_HC - bytes_us(HC_MIB, V620, a.rho_hc), 10.0)
    return bytes_us(HC_MIB, t.gpu, a.rho_hc) + lat * kap * s


def router_us(t: Topo, a: Assume, s: float) -> float:
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    lat = max(OLD_ROUTER - bytes_us(ROUTER_MIB, V620, a.rho_small), 5.0)
    return bytes_us(ROUTER_MIB, t.gpu, a.rho_small) + lat * kap * s


def shared_us(t: Topo, a: Assume, s: float) -> float:
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    mib = SHARED_MIB * t.dense_scale / t.tp
    lat = max(OLD_SHARED - bytes_us(SHARED_MIB / 4, V620, a.rho_small), 3.0)
    return bytes_us(mib, t.gpu, a.rho_small) + lat * kap * s


def expert_us(n: float, gpu: Gpu, a: Assume, mib: float = EXPERT_MIB) -> float:
    """Streaming time of n experts on one device (physical: bytes / achieved BW).
    TP4 at W=3 (7.3 experts, 18 MB) gives 47 us = the measured 52 us minus launches."""
    return n * mib * MiB / gpu.bw * 1e6


def experts_local_us(t: Topo, W: int, a: Assume, s: float, mib: float) -> float:
    """Experts on the dense GPUs (EP = tp).  Mean per-rank count: the cross-rank
    imbalance wait is inside the measured all-reduce time."""
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    return expert_us(n_touched(W, a) / t.tp, t.gpu, a, mib) + 5.0 * kap * s


def transport_us(W: int, D: int, a: Assume, s: float = 1.0) -> float:
    """The hx exchange of one layer with no expert compute: hub dispatch kernel 3 us,
    push, flag detection, server fixed part 5 us, pull."""
    push = PCIE_LAT_US + D * W * ROW_REQ_B / PCIE_BW * 1e6
    pull = PCIE_LAT_US + D * W * ROW_B / PCIE_BW * 1e6
    return 3.0 + push + a.detect_us + 5.0 * s + pull


def remote_us(n_rem: float, W: int, t: Topo, a: Assume, s: float) -> float:
    D = t.servers_per_hub
    return transport_us(W, D, a, s) + expert_us(
        n_max(n_rem / D, D) * t.srv_share, V620, a
    )


def moe_hub_us(t: Topo, W: int, a: Assume, s: float) -> tuple[float, dict]:
    """Hub runs shared expert + resident experts concurrently with the V620
    servers.  Resident experts above the balance point spill to the servers
    (all experts live on the V620s), so the hub never becomes the long pole
    just because it has the memory."""
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    N = n_touched(W, a) / t.n_hubs
    n_res = a.hit * N
    t_sh = shared_us(t, a, s)
    best = None
    steps = 40
    for i in range(steps + 1):
        n_hub = n_res * i / steps
        t_h = t_sh + expert_us(n_hub, t.gpu, a) + 5.0 * kap * s
        t_r = remote_us(N - n_hub, W, t, a, s)
        tt = max(t_h, t_r)
        if best is None or tt < best[0]:
            best = (tt, n_hub, t_h, t_r)
    tt, n_hub, t_h, t_r = best
    return tt + 3.0, dict(hub=t_h, remote=t_r, n_hub=n_hub, n_res=n_res, N=N)


def layer_us(t: Topo, W: int, S: int, a: Assume, s: float) -> tuple[float, dict]:
    ar1, ar8 = AR_US.get(t.tp, (0.0, 0.0))
    ar = ar1 if W <= 3 else ar8 if W >= 24 else ar1 + (ar8 - ar1) * (W - 3) / 21.0
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    c = {
        "hc": hc_us(t, a, s),
        "attn_core": attn_core_us(t, W, S, a, s),
        "router": router_us(t, a, s),
        "resid": OLD_RESID * kap * s,
        "ar": 2 * ar if t.tp > 1 else 0.0,
    }
    info: dict = {}
    if t.experts == "local":
        c["shared"] = shared_us(t, a, s)
        c["experts"] = experts_local_us(t, W, a, s, EXPERT_MIB)
    else:
        c["moe"], info = moe_hub_us(t, W, a, s)
    return sum(c.values()), {**c, **info}


def lm_head_us(t: Topo, a: Assume) -> float:
    if t.experts == "hub" and a.lmhead_remote:
        shard = LM_HEAD_MIB * t.dense_scale / (t.tp * t.servers_per_hub)
        return bytes_us(shard, V620) + 2 * (PCIE_LAT_US + 5.0)
    return bytes_us(LM_HEAD_MIB * t.dense_scale / t.tp, t.gpu)


def mtp_layer_us(t: Topo, W: int, S: int, a: Assume, s: float) -> float:
    kap = a.kappa if t.gpu.arch == "gfx1100" else 1.0
    c = {
        "hc": hc_us(t, a, s),
        "attn_core": attn_core_us(t, W, S, a, s),
        "router": router_us(t, a, s),
        "shared": shared_us(t, a, s),
        "resid": OLD_RESID * kap * s,
    }
    mib = EXPERT_MIB if (t.experts == "hub" and a.mtp_w4) else t.mtp_expert_mib
    c["experts"] = experts_local_us(t, W, a, s, mib)
    ar1, ar8 = AR_US.get(t.tp, (0.0, 0.0))
    c["ar"] = 2 * (ar1 if W <= 3 else ar8) if t.tp > 1 else 0.0
    return sum(c.values())


def step_us(t: Topo, W: int, S: int, a: Assume, mode: str) -> dict:
    s = t.s_l[mode]
    per_layer, comp = layer_us(t, W, S, a, s)
    target = per_layer * N_LAYERS + 300.0
    head = lm_head_us(t, a) + 200.0 * s
    wd = (3, 1) if W <= 3 else (W, S)
    draft = 0.0
    for w in wd:
        draft += mtp_layer_us(t, w, max(S, 1), a, s) + lm_head_us(t, a) + 350.0 * s
    gap = 500.0
    return dict(target=target, head=head, draft=draft, gap=gap, layer=per_layer,
                comp=comp, step=target + head + draft + gap)  # fmt: skip


def calibrate(base: Topo, a: Assume) -> None:
    for mode, W, S, key in (("c1", 3, 1, "c1"), ("c8", 24, 8, "c8")):
        goal = TODAY["TP4"][key] * 1e3
        lo, hi = 0.02, 2.0
        for _ in range(80):
            mid = (lo + hi) / 2
            base.s_l[mode] = mid
            if step_us(base, W, S, a, mode)["step"] > goal:
                hi = mid
            else:
                lo = mid
    base.s_l["c16"] = base.s_l["c8"]


def topos(a: Assume) -> dict[str, Topo]:
    ds = a.hub_dense_scale
    w7800 = replace(W7800, eta=a.eta_h)
    return {
        "TP4": Topo("TP4-V620", V620, 4, "local"),
        "TP8": Topo("TP8-V620", V620, 8, "local"),
        "W2": Topo("W2 (2x W7800, TP2, no offload)", w7800, 2, "local", dense_scale=ds),
        "H1": Topo(
            "H1 (hub TP1 + 4 V620 / switch)", w7800, 1, "hub", 1, 4, dense_scale=ds
        ),
        "H2": Topo("H2 (hubs TP2 + 8 V620)", w7800, 2, "hub", 2, 4, dense_scale=ds),
        "H2p": Topo(
            "H2' (hubs TP2 + 4 V620 shared)",
            w7800,
            2,
            "hub",
            2,
            4,
            dense_scale=ds,
            srv_share=2.0,
        ),
    }


def setup(a: Assume | None = None) -> tuple[Assume, dict[str, Topo]]:
    a = a or Assume()
    ts = topos(a)
    calibrate(ts["TP4"], a)
    for t in ts.values():
        t.s_l = dict(ts["TP4"].s_l)
    return a, ts


# ---------------------------------------------------------------------------
# Prefill, one 2048-token chunk
# ---------------------------------------------------------------------------
G_P, Q_P = 26.2e6 + 15.7e6 + 15.7e6 + 0.25e6, 31.5e6 + 2.6e6 + 15.7e6 + 1.6e6
DENSE_FLOP_TOK = 2 * (N_GDN * G_P + N_QSA * Q_P + N_LAYERS * (13.1e6 + 4.9e6 + 1.3e6))
EXP_FLOP_TOK = 2 * 3 * 2560 * 640 * TOPK * N_LAYERS
CHUNK = 2048
TP4_CHUNK_S = CHUNK / TODAY["TP4"]["pre_tps"]


def prefill_chunk(kind: str, a: Assume) -> dict:
    fd, fe = DENSE_FLOP_TOK * CHUNK, EXP_FLOP_TOK * CHUNK
    if kind == "TP4":
        return dict(chunk=TP4_CHUNK_S, e_v=(fd + fe) / (4 * TP4_CHUNK_S))
    if kind == "TP8":
        return dict(chunk=CHUNK / TODAY["TP8"]["pre_tps"])
    e_h = a.eff_hub * W7800.fp16
    ar_tp2 = 96 * 0.0004  # TP2 AR of 10 MiB, cross-root, ~0.4 ms
    if kind == "W2":
        t_d, t_e = fd / (2 * e_h), fe / (2 * e_h)
        return dict(chunk=t_d + t_e + ar_tp2, dense=t_d, experts=t_e)
    n = 1 if kind == "H1" else 2
    D = 4 * n
    t_d = fd / (n * e_h)
    f_hot = fe * a.prefill_hub_share
    t_hub = t_d + f_hot / (n * e_h) + (ar_tp2 if n == 2 else 0.0)
    t_srv = (fe - f_hot) / (D * a.eff_v620_moe)
    link_star = N_LAYERS * 4 * CHUNK * ROW_B / PCIE_BW
    link_tree = N_LAYERS * 1 * CHUNK * ROW_B / PCIE_BW
    chunk = max(t_hub, t_srv, link_tree) * 1.10
    return dict(chunk=chunk, dense=t_d, hub=t_hub, srv=t_srv, link_star=link_star,
                link_tree=link_tree)  # fmt: skip


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
KV_TOK_TP1_KIB = 24.0 + 2.73  # 12 layers x 2 KV heads x 256 x 2 x 2 B + indexer
NONTORCH_GIB = 1.6  # per GPU, measured 1.4-1.7 (tp8-pp2 doc)
WORKSPACE_GIB = 2.5  # graphs, 2048-token chunk buffers [INFERENCE]


def mem_hub(tp: int, hot_gib: float, a: Assume, vram_gib: float = 44.98) -> dict:
    s = a.hub_dense_scale
    w = (GIB["gdn"] + GIB["qsa"] + GIB["indexer"] + GIB["shared"]) * s / tp
    w += (GIB["embed"] + GIB["lm_head"]) * s / tp
    w += GIB["hc"] + GIB["router"] + GIB["ple_proj"]
    mtp = (
        GIB["mtp_dense"] / tp
        + (GIB["mtp_experts_w4"] if a.mtp_w4 else GIB["mtp_experts_bf16"]) / tp
    )
    kv_free = vram_gib - w - mtp - hot_gib - NONTORCH_GIB - WORKSPACE_GIB
    kv_tok = (min(KV_TOK_TP1_KIB, 12.0 + 2.73) if tp >= 2 else KV_TOK_TP1_KIB) * 1024
    return dict(dense=w, mtp=mtp, hot=hot_gib, fixed=NONTORCH_GIB + WORKSPACE_GIB,
                kv_gib=kv_free, kv_tokens=kv_free * GiB / kv_tok)  # fmt: skip


def mem_v620(experts_frac: float, rep: float, lm_shard_gib: float = 0.0) -> dict:
    """experts_frac of the 58 GiB expert set per device, times replication."""
    e = GIB["experts"] * experts_frac * rep + (
        N_EXP * N_LAYERS * EXPERT_QZ_B / GiB * experts_frac * rep
    )
    free = 29.98 - e - lm_shard_gib - NONTORCH_GIB - 1.0
    return dict(experts=e, lm=lm_shard_gib, fixed=NONTORCH_GIB + 1.0, free=free)


def pcie_traffic(W: int, D: int, h_spill: float = 0.0) -> dict:
    down = D * W * ROW_REQ_B
    up = D * W * ROW_B
    return dict(down_kib=down / 1024, up_kib=up / 1024,
                step_down_mib=down * N_LAYERS / MiB, step_up_mib=up * N_LAYERS / MiB,
                push_us=PCIE_LAT_US + down / PCIE_BW * 1e6,
                pull_us=PCIE_LAT_US + up / PCIE_BW * 1e6)  # fmt: skip


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def composite(step_ms: float, chunk_s: float, users: int) -> float:
    """1k/512 aggregate output tok/s for `users` concurrent streams on one engine,
    anchored on TP4 = 262.6 measured (prefill 2806 tok/s at 1k c8)."""
    pre_rate = TODAY["TP4"]["pre8_tps"] * TP4_CHUNK_S / chunk_s
    t_pre = users * 1024 / pre_rate
    t_dec = 512 / ACC * step_ms / 1e3
    return users * 512 / (t_pre + t_dec + STALL_CHUNKS * chunk_s)


KINDS = {"TP4": "TP4", "TP8": "TP8", "W2": "W2", "H1": "H1", "H2": "H2", "H2p": "H2"}


def results(a: Assume, ts: dict[str, Topo]) -> dict:
    out: dict = {"assume": asdict(a), "phi": a.phi, "s_l": ts["TP4"].s_l, "rows": {}}
    for k, t in ts.items():
        r: dict = {}
        for mode, W, S in (("c1", 3, 1), ("c8", 24, 8), ("c16", 48, 16)):
            st = step_us(t, W, S, a, mode)
            r[mode] = {
                x: (v / 1e3 if isinstance(v, float) else v)
                for x, v in st.items()
                if x != "comp"
            }
            r[mode]["comp"] = st["comp"]
        pc = prefill_chunk(KINDS[k], a)
        r["prefill"] = pc
        r["tps8"] = composite(r["c8"]["step"], pc["chunk"], 8)
        r["tps16"] = composite(r["c16"]["step"], pc["chunk"], 16)
        r["ttft16k"] = 8 * pc["chunk"]
        r["pre_tps"] = CHUNK / pc["chunk"]
        out["rows"][k] = r
    return out


# Measured today, for the machine-level table (tp8-pp2-flashnext-2026-11-06.md:
# 424-441): TP4 x PP2 c1 step 23.08 ms, 1k c8 output 272.7 tok/s, 16k c1 TTFT
# 3.40 s, 16k prefill 4823 tok/s per request.
PP2 = dict(c1=23.08, c8_tps=272.7, ttft=3.40, pre_tps=4823.0)


def deployments(o: dict) -> list[dict]:
    """Machine level, all ten GPUs.  `users` counts concurrent 1k/512 streams;
    every engine runs at its c8 point (8 users) unless the row says otherwise."""
    r = o["rows"]
    d: list[dict] = []

    def add(name, gpus, users, c1, tps, pre, ttft, src):
        d.append(
            dict(name=name, gpus=gpus, users=users, c1=c1, tps=tps, pre=pre,
                 ttft=ttft, src=src)
        )  # fmt: skip

    tp4, h1, h2 = r["TP4"], r["H1"], r["H2"]
    w2, h2p = r["W2"], r["H2p"]
    two_tp4 = 2 * TODAY["TP4"]["c8_tps"]
    add("D0a 2 x TP4 (today)", "8 V620", 16, tp4["c1"]["step"], two_tp4,
        2 * TODAY["TP4"]["pre_tps"], tp4["ttft16k"], "measured")  # fmt: skip
    add("D0b TP8 (today)", "8 V620", 8, TODAY["TP8"]["c1"], TODAY["TP8"]["c8_tps"],
        TODAY["TP8"]["pre_tps"], 5.28, "measured")  # fmt: skip
    add("D0c TP4 x PP2 (today)", "8 V620", 8, PP2["c1"], PP2["c8_tps"],
        PP2["pre_tps"], PP2["ttft"], "measured")  # fmt: skip
    add("D1 2 x H1 (hub TP1 + 4 V620 per switch)", "2 W7800 + 8 V620", 16,
        h1["c1"]["step"], 2 * h1["tps8"], 2 * h1["pre_tps"], h1["ttft16k"],
        "model")  # fmt: skip
    add("D2 H2 (TP2 hubs + 8 V620 experts)", "2 W7800 + 8 V620", 8,
        h2["c1"]["step"], h2["tps8"], h2["pre_tps"], h2["ttft16k"],
        "model")  # fmt: skip
    add("D2b H2, 16 users on the one engine", "2 W7800 + 8 V620", 16,
        h2["c1"]["step"], h2["tps16"], h2["pre_tps"], h2["ttft16k"],
        "model")  # fmt: skip
    add("D3 W2 + 2 x TP4 (engine split, no offload)", "2 W7800 + 8 V620", 24,
        min(w2["c1"]["step"], tp4["c1"]["step"]),
        w2["tps8"] + two_tp4, w2["pre_tps"] + 2 * TODAY["TP4"]["pre_tps"],
        w2["ttft16k"], "measured + model")  # fmt: skip
    add("D4 H2' (TP2 hubs + 4 V620) + 1 x TP4", "2 W7800 + 8 V620", 16,
        h2p["c1"]["step"], h2p["tps8"] + TODAY["TP4"]["c8_tps"],
        h2p["pre_tps"] + TODAY["TP4"]["pre_tps"], h2p["ttft16k"],
        "measured + model")  # fmt: skip
    tp8 = TODAY["TP8"]
    add("D5 TP8 + W2 (engine split, no offload)", "2 W7800 + 8 V620", 16,
        min(w2["c1"]["step"], tp8["c1"]), w2["tps8"] + tp8["c8_tps"],
        w2["pre_tps"] + tp8["pre_tps"], w2["ttft16k"], "measured + model")  # fmt: skip
    return d


def table(header: list[str], rows: list[list[str]]) -> str:
    """GitHub markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def md_tables(o: dict, ts: dict[str, Topo]) -> str:
    rows = o["rows"]
    out = [
        f"Calibration: s_l(c1)={o['s_l']['c1']:.3f}, "
        f"s_l(c8)={o['s_l']['c8']:.3f}, phi={o['phi']:.2f}\n"
    ]
    head = [
        "Engine",
        "c1 step ms",
        "c8 step ms",
        "c16 step ms",
        "1k c8 tok/s",
        "prefill chunk s",
        "16k TTFT s",
    ]
    body = [
        [
            ts[k].name,
            f"{r['c1']['step']:.1f}",
            f"{r['c8']['step']:.1f}",
            f"{r['c16']['step']:.1f}",
            f"{r['tps8']:.0f}",
            f"{r['prefill']['chunk']:.2f}",
            f"{r['ttft16k']:.1f}",
        ]
        for k, r in rows.items()
    ]
    out += [table(head, body), ""]
    keys = ["hc", "attn_core", "router", "shared", "experts", "moe", "ar", "resid"]
    for mode in ("c1", "c8"):
        out.append(f"Per-layer components, {mode} (us):\n")
        body = []
        for k, r in rows.items():
            c = r[mode]["comp"]
            body.append(
                [
                    ts[k].name,
                    *(f"{c.get(x, 0):.0f}" for x in keys),
                    f"{r[mode]['layer'] * 1e3:.0f}",
                ]
            )
        out += [table(["Engine", *keys, "layer"], body), ""]
    out.append("Step composition (ms): target / head / drafters / gap\n")
    parts = ("target", "head", "draft", "gap")
    body = [
        [ts[k].name] + ["/".join(f"{r[m][x]:.1f}" for x in parts) for m in ("c1", "c8")]
        for k, r in rows.items()
    ]
    out.append(table(["Engine", "c1", "c8"], body))
    return "\n".join(out)


def md_deployments(o: dict) -> str:
    head = [
        "Deployment",
        "GPUs",
        "users",
        "best c1 ITL ms",
        "aggregate 1k/512 tok/s",
        "aggregate prefill tok/s",
        "best 16k TTFT s",
        "basis",
    ]
    body = [
        [
            d["name"],
            d["gpus"],
            str(d["users"]),
            f"{d['c1']:.1f}",
            f"{d['tps']:.0f}",
            f"{d['pre']:.0f}",
            f"{d['ttft']:.1f}",
            d["src"],
        ]
        for d in deployments(o)
    ]
    return table(head, body)


def md_memory(a: Assume) -> str:
    """Per-device memory (GiB) and KV capacity."""
    body = []
    for name, tp, hot in (
        ("H1 hub", 1, 12.0),
        ("H2 hub (each)", 2, 11.0),
        ("W2 hub (each)", 2, 29.45),
    ):
        m = mem_hub(tp, hot, a)
        body.append(
            [
                name,
                f"{m['dense']:.1f}",
                f"{m['mtp']:.1f}",
                f"{m['hot']:.1f}",
                f"{m['fixed']:.1f}",
                f"{m['kv_gib']:.1f}",
                f"{m['kv_tokens'] / 1e3:.0f}k",
            ]
        )
    for name, frac, rep, lm in (
        ("H1 V620 (primaries)", 0.25, 1.0, 0.30),
        ("H1 V620 (replicas to the 26 GiB budget)", 0.25, 1.75, 0.30),
        ("H2 V620 (primaries)", 0.125, 1.0, 0.15),
        ("H2 V620 (replicas to the 26 GiB budget)", 0.125, 3.5, 0.15),
    ):
        m = mem_v620(frac, rep, lm)
        body.append(
            [
                name,
                "-",
                "-",
                f"{m['experts']:.1f}",
                f"{m['fixed']:.1f}",
                f"free {m['free']:.1f}",
                "-",
            ]
        )
    head = ["Device", "dense", "MTP", "experts", "fixed", "KV", "KV tokens"]
    return table(head, body)


def md_traffic() -> str:
    head = [
        "rows T",
        "down per layer (4 servers)",
        "up per layer",
        "push us",
        "pull us",
        "per step down+up (48 layers)",
    ]
    body = []
    for w in (3, 6, 12, 24, 48, 2048):
        t = pcie_traffic(w, 4)
        body.append(
            [
                str(w),
                f"{t['down_kib']:.0f} KiB",
                f"{t['up_kib']:.0f} KiB",
                f"{t['push_us']:.1f}",
                f"{t['pull_us']:.1f}",
                f"{t['step_down_mib'] + t['step_up_mib']:.0f} MiB",
            ]
        )
    return table(head, body)


def loopback_rows(a: Assume) -> list[dict]:
    """The model's exchange time against HxServer's loopback (same server count)."""
    out = []
    for w, meas in LOOPBACK_US.items():
        mod = transport_us(w, LOOPBACK_D, a)
        out.append(dict(w=w, model=mod, measured=meas, delta=meas - mod,
                        pct=100.0 * (meas / mod - 1.0)))  # fmt: skip
    return out


def loopback_refit(base: Assume) -> list[dict]:
    """H1 decode steps with `detect_us` raised so that the model's exchange equals the
    loopback at that width (V620-only loopback with three senders: approximate)."""
    ref = rows_for(base)["H1"]
    out = []
    for r in loopback_rows(base):
        mode = LOOPBACK_MODE.get(r["w"])
        if mode is None:
            continue
        det = base.detect_us + r["delta"]
        step = rows_for(base, detect_us=det)["H1"][mode]["step"]
        out.append(dict(mode=mode, w=r["w"], detect_us=det, before=ref[mode]["step"],
                        after=step))  # fmt: skip
    return out


def md_loopback(base: Assume) -> str:
    head = ["rows T", "model us (3 servers)", "loopback us", "delta us", "delta"]
    body = [
        [str(r["w"]), f"{r['model']:.1f}", f"{r['measured']:.1f}", f"{r['delta']:+.1f}",
         f"{r['pct']:+.0f} %"]
        for r in loopback_rows(base)
    ]  # fmt: skip
    head2 = [
        "H1 mode",
        "T",
        "detect_us refit",
        "step ms (model)",
        "step ms (loopback exchange)",
    ]
    body2 = [
        [r["mode"], str(r["w"]), f"{r['detect_us']:.1f}", f"{r['before']:.1f}",
         f"{r['after']:.1f}"]
        for r in loopback_refit(base)
    ]  # fmt: skip
    return table(head, body) + "\n\n" + table(head2, body2)


# Scenario sets shared by the scenario table, the H1 ladder and the stage-2 table.
PESSIMISTIC = {"eta_h": 0.65, "kappa": 1.4, "phi": 0.25, "detect_us": 30.0}
OPTIMISTIC = {"eta_h": 0.95, "kappa": 0.8, "phi": 1.0}
LEVERS = {"lmhead_remote": True, "hub_dense_scale": 0.5}
GATE_GREEN = 1.15  # first H1 boot: within 15 % of the central model
GATE_PARITY = 1.10  # H1 with both levers against measured TP4 slot A


ENGINES = ("W2", "H1", "H2")


def sensitivity(base: Assume) -> str:
    knobs = [
        ("eta_h", [0.65, 0.95]),
        ("kappa", [0.8, 1.4]),
        ("gamma", [0.15, 0.5]),
        ("comp_ratio", [1.0, 1.7]),
        ("rho_hc", [0.0, 1.0]),
        ("phi", [0.25, 1.0]),
        ("hit", [0.35, 0.88]),
        ("detect_us", [30.0, 3.0]),
        ("hub_dense_scale", [0.5, 0.25]),
        ("lmhead_remote", [True, True]),
    ]
    modes = ("c1", "c8")
    head = ["knob (low / high)"] + [f"{e} {m}" for e in ENGINES for m in modes]
    a0, ts0 = setup(replace(base))
    r0 = results(a0, ts0)["rows"]
    body = [["central"] + [f"{r0[e][m]['step']:.1f}" for e in ENGINES for m in modes]]
    for name, vals in knobs:
        pair = []
        for v in vals:
            a1, ts1 = setup(replace(base, **{name: v}))
            pair.append(results(a1, ts1)["rows"])
        lo, hi = pair
        body.append(
            [f"{name} {vals[0]} / {vals[1]}"]
            + [
                f"{lo[e][m]['step']:.1f} / {hi[e][m]['step']:.1f}"
                for e in ENGINES
                for m in modes
            ]
        )
    return table(head, body)


def scenarios(base: Assume) -> str:
    sets = [
        ("central", {}),
        ("+ lm_head on the expert servers", {"lmhead_remote": True}),
        ("+ W8A16 hub dense", {"hub_dense_scale": 0.5}),
        ("+ both", LEVERS),
        (
            "pessimistic (eta 0.65, kappa 1.4, phi 0.25, host-driven servers)",
            PESSIMISTIC,
        ),
        ("optimistic (eta 0.95, kappa 0.8, phi 1.0)", OPTIMISTIC),
    ]
    head = ["scenario"] + [f"{e} c1 / c8" for e in ENGINES] + ["H2 prefill chunk s"]
    body = []
    for name, kw in sets:
        a1, ts1 = setup(replace(base, **kw))
        rr = results(a1, ts1)["rows"]
        body.append(
            [name]
            + [
                f"{rr[e]['c1']['step']:.1f} / {rr[e]['c8']['step']:.1f}"
                for e in ENGINES
            ]
            + [f"{rr['H2']['prefill']['chunk']:.2f}"]
        )
    return table(head, body)


def prefill_table(a: Assume) -> str:
    head = ["eff_hub (all-in share of 90.4 TFLOPS)"]
    head += [f"{e} chunk s" for e in ENGINES]
    body = []
    for e in (0.2, 0.3, 0.4, 0.5):
        b = replace(a, eff_hub=e)
        body.append(
            [f"{e:.1f} ({e * 90.4:.0f} TFLOPS)"]
            + [f"{prefill_chunk(k, b)['chunk']:.2f}" for k in ENGINES]
        )
    body.append(["TP4 measured", f"{TP4_CHUNK_S:.2f}", "", ""])
    return table(head, body)


# ---------------------------------------------------------------------------
# Stage plan: H1 against today's TP4 slot A, and the stage-2 candidates
# ---------------------------------------------------------------------------
H1_LADDER = (
    ("central", {}),
    ("+ lm_head on the servers", {"lmhead_remote": True}),
    ("+ W8A16 hub dense", {"hub_dense_scale": 0.5}),
    ("+ both levers", LEVERS),
    ("pessimistic", PESSIMISTIC),
    ("optimistic", OPTIMISTIC),
)


def rows_for(base: Assume, **kw: Any) -> dict:
    """The `results` rows of one assumption set."""
    a, ts = setup(replace(base, **kw))
    return results(a, ts)["rows"]


def h1_ladder(base: Assume) -> list[dict]:
    """H1 (stage 1) against measured TP4 slot A, one lever at a time."""
    t4 = TODAY["TP4"]
    out = []
    for name, kw in H1_LADDER:
        r = rows_for(base, **kw)["H1"]
        c1, c8 = r["c1"]["step"], r["c8"]["step"]
        out.append(
            dict(name=name, c1=c1, c8=c8, c16=r["c16"]["step"], tps8=r["tps8"],
                 c1_ratio=c1 / t4["c1"], c8_ratio=c8 / t4["c8"])
        )  # fmt: skip
    return out


def h1_gates(base: Assume) -> dict:
    """Thresholds of the stage-1 performance gates (spec section 10)."""
    by = {r["name"]: r for r in h1_ladder(base)}
    c, p, b = by["central"], by["pessimistic"], by["+ both levers"]
    t4 = TODAY["TP4"]
    return dict(
        green_c1=c["c1"] * GATE_GREEN, green_c8=c["c8"] * GATE_GREEN,
        red_c1=p["c1"], red_c8=p["c8"],
        parity_c1=t4["c1"] * GATE_PARITY, parity_c8=t4["c8"] * GATE_PARITY,
        levers_c1=b["c1"], levers_c8=b["c8"],
    )  # fmt: skip


def h1_fit(a: Assume, slots: float | None = None, kv_gib: float | None = None) -> dict:
    """One W7800 at TP1: the slot pool (experts per target layer) against KV.
    Give the slots per layer or the KV reserve; the other follows."""
    per_slot = N_LAYERS * EXPERT_B / GiB  # GiB of one slot in every target layer
    if slots is None:
        assert kv_gib is not None
        slots = (mem_hub(1, 0.0, a)["kv_gib"] - kv_gib) / per_slot
    m = mem_hub(1, slots * per_slot, a)
    return dict(slots=slots, pool_gib=slots * per_slot, dense=m["dense"],
                mtp=m["mtp"], fixed=m["fixed"], kv_gib=m["kv_gib"],
                kv_tokens=m["kv_tokens"])  # fmt: skip


def dense_offload_gain_us(t: Topo, W: int, a: Assume, s: float, f: float) -> float:
    """Upper bound on the saving per layer (us) of moving a fraction f of the weight
    stream of the two dependent attention-projection phases (in, out) from the hub
    to the V620 servers.  Each phase becomes a round trip (push, wake, chain, pull)
    that overlaps the hub's remaining share; zero when it loses."""
    w = AVG_W_MIB * t.dense_scale / t.tp / 2.0  # MiB per phase on the hub
    t_h = bytes_us(w, t.gpu, 0.9)
    t_r = remote_us(0.0, W, t, a, s) + bytes_us(f * w / t.servers_per_hub, V620, 0.9)
    return max(0.0, 2.0 * (t_h - max((1.0 - f) * t_h, t_r)))


def dense_offload(base: Assume) -> list[dict]:
    """Best-case gain of TP across hub + V620 for the byte-bound attention part."""
    out = []
    for name, kw in (("H2", {}), ("H2 + W8A16 hub dense", {"hub_dense_scale": 0.5})):
        a, ts = setup(replace(base, **kw))
        t = ts["H2"]
        for mode, W, S in (("c1", 3, 1), ("c8", 24, 8)):
            s = t.s_l[mode]
            gain, f = max(
                (dense_offload_gain_us(t, W, a, s, i / 20), i / 20)
                for i in range(1, 21)
            )
            step = step_us(t, W, S, a, mode)["step"]
            out.append(
                dict(name=name, mode=mode, f=f if gain > 0 else 0.0, us_layer=gain,
                     ms_step=gain * N_LAYERS / 1e3,
                     pct=100.0 * gain * N_LAYERS / step)
            )  # fmt: skip
    return out


def pd_split_tps(rows: dict) -> float:
    """D3 with every prefill moved to the W2 engine (1k/512, 8 decode users per
    engine): the two TP4 engines only decode (no prefill stalls), W2 prefills for
    all and decodes its own users in the time left.  The KV handoff cost is
    neglected [INFERENCE], and the handoff itself is blocked by the G0a verdict of
    docs/rdna2/disagg-g0-2026-11-06.md."""
    dec_s = 512 / ACC * rows["TP4"]["c8"]["step"] / 1e3  # one request on a TP4
    pair = 2 * 8 * 512 / dec_s
    w2 = rows["W2"]
    pre_s = 1024 * w2["prefill"]["chunk"] / CHUNK  # W2 seconds per request
    busy = pair / 512 * pre_s
    own = 8 * 512 / (512 / ACC * w2["c8"]["step"] / 1e3)
    return pair + own * max(0.0, 1.0 - busy)


def stage2(base: Assume) -> list[dict]:
    """Stage-2 candidates on all ten GPUs.  `c8` is the engine step at 8 users;
    `tps` is the aggregate 1k/512 output rate at `users`."""
    o0 = results(*setup(replace(base)))
    ol = results(*setup(replace(base, **LEVERS)))
    d0 = {d["name"].split()[0]: d for d in deployments(o0)}
    dl = {d["name"].split()[0]: d for d in deployments(ol)}
    r0, rl = o0["rows"], ol["rows"]
    rows: list[dict] = []

    def add(
        name: str,
        gpus: int,
        users: int,
        c1: float,
        c8: float,
        tps: float,
        pre: float,
        ttft: float,
        basis: str,
    ) -> None:
        rows.append(
            dict(name=name, gpus=gpus, users=users, c1=c1, c8=c8, tps=tps,
                 per_gpu=tps / gpus, per_user=tps / users, pre=pre, ttft=ttft,
                 basis=basis)
        )  # fmt: skip

    def dep(name: str, d: dict, c8: float, gpus: int = 10) -> None:
        add(name, gpus, d["users"], d["c1"], c8, d["tps"], d["pre"], d["ttft"],
            d["src"])  # fmt: skip

    dep("D0a 2 x TP4 (today)", d0["D0a"], r0["TP4"]["c8"]["step"], 8)
    dep("D0b TP8 (today)", d0["D0b"], TODAY["TP8"]["c8"], 8)
    dep("D1 2 x H1 (one hub per switch)", d0["D1"], r0["H1"]["c8"]["step"])
    dep("D1 + both levers", dl["D1"], rl["H1"]["c8"]["step"])
    h1 = r0["H1"]
    add("P2 PP2 across the islands (hubs are the stage heads)", 10, 16,
        h1["c1"]["step"] + 0.02, h1["c8"]["step"], 2 * h1["tps8"],
        2 * h1["pre_tps"], 4.5 * h1["prefill"]["chunk"], "model")  # fmt: skip
    dep("H2, 16 users", d0["D2b"], r0["H2"]["c8"]["step"])
    dep("H2 + both levers, 16 users", dl["D2b"], rl["H2"]["c8"]["step"])
    slow = max(r0["W2"]["c8"]["step"], r0["TP4"]["c8"]["step"])
    dep("D3 W2 + 2 x TP4", d0["D3"], slow)
    slow_l = max(rl["W2"]["c8"]["step"], rl["TP4"]["c8"]["step"])
    dep("D3 with W8A16 on W2", dl["D3"], slow_l)
    tp8_c8 = TODAY["TP8"]["c8"]
    dep("D5 TP8 + W2", d0["D5"], max(r0["W2"]["c8"]["step"], tp8_c8))
    dep("D5 with W8A16 on W2", dl["D5"], max(rl["W2"]["c8"]["step"], tp8_c8))
    dep("D4 H2' + 1 x TP4", d0["D4"], r0["H2p"]["c8"]["step"])
    w2 = r0["W2"]
    add("D3-PD (W2 prefills, 2 x TP4 decode)", 10, 24, d0["D3"]["c1"], slow,
        pd_split_tps(r0), w2["pre_tps"], w2["ttft16k"],
        "model, handoff blocked")  # fmt: skip
    return rows


ACC_C1 = 1.84  # tokens per step at c1 (tp8-pp2 doc)


def acc_at(users: int) -> float:
    """MTP2 accepted tokens per step: 1.84 at one user, 2.01 from eight (linear)."""
    return ACC_C1 + (ACC - ACC_C1) * (min(users, 8) - 1) / 7.0


def lat_scale(t: Topo, users: int) -> float:
    """Latency scale s_l at `users` streams, interpolated between the c1 and c8
    calibrations (flat from eight users)."""
    s1, s8 = t.s_l["c1"], t.s_l["c8"]
    return s1 + (s8 - s1) * (min(users, 8) - 1) / 7.0


def step_ms_at(t: Topo, users: int, a: Assume) -> float:
    """Decode step (ms) of one engine with `users` streams (3 verify rows each)."""
    t2 = replace(t, s_l={"u": lat_scale(t, users)})
    return step_us(t2, 3 * users, users, a, "u")["step"] / 1e3


SPILL_WIDTHS = (3, 6, 12, 24, 48)  # the capture widths (rows)


def spill_k(base: Assume, kind: str) -> dict[int, float]:
    """Spill count K per capture width: the number of touched hub-resident experts
    that stay on the hub, i.e. the model's balance point between the hub (shared
    expert + resident experts) and the servers.  `kind` is `H1` or `H2` (per hub)."""
    a, ts = setup(replace(base))
    t = ts[kind]
    return {
        w: moe_hub_us(t, w, a, lat_scale(t, w // 3))[1]["n_hub"] for w in SPILL_WIDTHS
    }


def md_spill_k(base: Assume) -> str:
    head = ["Engine (per hub)"] + [f"W = {w}" for w in SPILL_WIDTHS]
    body = [
        [name] + [f"{k:.1f}" for k in spill_k(base, kind).values()]
        for name, kind in (("H1 (hub + 4 servers)", "H1"), ("H2 (each hub)", "H2"))
    ]
    return table(head, body)


# Engines of each deployment (keys of `topos`); the levers apply to hub engines.
ITL_SETS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("TP8 (today)", ("TP8",), False),
    ("D0a 2 x TP4 (today)", ("TP4", "TP4"), False),
    ("D1 2 x H1 + both levers", ("H1", "H1"), True),
    ("H2 + both levers", ("H2",), True),
    ("D3 W2 + 2 x TP4", ("W2", "TP4", "TP4"), False),
    ("D3 with W8A16 on W2", ("W2", "TP4", "TP4"), True),
    ("D5 TP8 + W2 + W8A16", ("TP8", "W2"), True),
    ("D4 H2' + 1 x TP4", ("H2p", "TP4"), False),
)
ITL_USERS = (1, 2, 4, 8, 16, 24)
MAX_USERS_PER_ENGINE = 16  # the widest capture bucket (48 rows)


def place_users(engines: list[Topo], a: Assume) -> dict:
    """Mean per-user decode ITL (ms per token) at each total concurrency of
    ITL_USERS when a router puts every user on the engine that ends up with the
    lowest ITL.  None = more users than the engines hold."""
    cache: dict[tuple[int, int], float] = {}

    def itl(e: int, n: int) -> float:
        if (e, n) not in cache:
            cache[(e, n)] = step_ms_at(engines[e], n, a) / acc_at(n)
        return cache[(e, n)]

    row: dict = {}
    for u in ITL_USERS:
        if u > MAX_USERS_PER_ENGINE * len(engines):
            row[u] = None
            continue
        load = [0] * len(engines)
        for _ in range(u):
            best = min(
                (e for e in range(len(engines)) if load[e] < MAX_USERS_PER_ENGINE),
                key=lambda e: itl(e, load[e] + 1),
            )
            load[best] += 1
        row[u] = sum(n * itl(e, n) for e, n in enumerate(load) if n) / u
    return row


def itl_curve(base: Assume) -> dict:
    """`place_users` for every deployment of ITL_SETS (prefill stalls excluded)."""
    plain = setup(replace(base))
    lev = setup(replace(base, **LEVERS))
    out: dict = {}
    for name, keys, levers in ITL_SETS:
        a, ts = lev if levers else plain
        out[name] = place_users([ts[k] for k in keys], a)
    return out


def md_itl(base: Assume) -> str:
    cur = itl_curve(base)
    head = ["Deployment (ITL ms per token, mean over users)"] + [
        f"{u} users" for u in ITL_USERS
    ]
    body = [
        [name] + ["-" if v is None else f"{v:.1f}" for v in (row[u] for u in ITL_USERS)]
        for name, row in cur.items()
    ]
    best = ["best"]
    for u in ITL_USERS:
        cand = {n: r[u] for n, r in cur.items() if r[u] is not None}
        best.append(min(cand, key=lambda n: cand[n]).split(" (")[0] if cand else "-")
    return table(head, body + [best])


ITL_TAG = {
    "TP8 (today)": "TP8",
    "D0a 2 x TP4 (today)": "D0a",
    "D1 2 x H1 + both levers": "D1L",
    "H2 + both levers": "H2L",
    "D3 W2 + 2 x TP4": "D3",
    "D3 with W8A16 on W2": "D3L",
    "D5 TP8 + W2 + W8A16": "D5L",
    "D4 H2' + 1 x TP4": "D4",
}
ITL_SCENARIOS = (
    ("central", {}),
    ("optimistic (eta 0.95, kappa 0.8, phi 1.0)", OPTIMISTIC),
    ("pessimistic (eta 0.65, kappa 1.4, phi 0.25, host-driven servers)", PESSIMISTIC),
)


PLAIN_SETS = (  # the deployments that need no hx protocol
    "TP8 (today)",
    "D0a 2 x TP4 (today)",
    "D3 W2 + 2 x TP4",
    "D3 with W8A16 on W2",
    "D5 TP8 + W2 + W8A16",
)
H2_GO_RATIO = 0.92  # H2 + levers must be 8 % below the best deployment without hx


def h2_ratio(cur: dict) -> dict:
    """ITL of H2 + both levers divided by the lowest ITL of any deployment without
    the hx protocol, per concurrency (None where H2 cannot hold the users)."""
    h2 = cur["H2 + both levers"]
    return {
        u: None
        if h2[u] is None
        else h2[u] / min(v for n in PLAIN_SETS if (v := cur[n][u]) is not None)
        for u in ITL_USERS
    }


def md_itl_scenarios(base: Assume) -> str:
    """Which deployment has the lowest ITL at each concurrency, per W7800 scenario,
    and how H2 + levers compares with the best deployment without the hx protocol."""
    head = ["W7800 assumptions"] + [f"{u} users" for u in ITL_USERS]
    best, ratio = [], []
    for name, kw in ITL_SCENARIOS:
        cur = itl_curve(replace(base, **kw))
        rr = h2_ratio(cur)
        b, r = [name], [name]
        for u in ITL_USERS:
            cand = {n: v[u] for n, v in cur.items() if v[u] is not None}
            top = min(cand, key=lambda n: cand[n])
            b.append(f"{ITL_TAG[top]} {cand[top]:.1f}")
            r.append("-" if rr[u] is None else f"{rr[u]:.2f}")
        best.append(b)
        ratio.append(r)
    return (
        table(head, best)
        + "\n\nH2 + both levers ITL divided by the best deployment without hx "
        + f"(H2 builds only below {H2_GO_RATIO}):\n\n"
        + table(head, ratio)
    )


def stage_plan(base: Assume) -> dict:
    """Everything the stage plan prints, machine readable."""
    return dict(
        h1_ladder=h1_ladder(base),
        h1_gates=h1_gates(base),
        h1_fit=[
            dict(name=n, **h1_fit(replace(base, **kw), **sel)) for n, kw, sel in H1_FITS
        ],
        stage2=stage2(base),
        spill_k={k: spill_k(base, k) for k in ("H1", "H2")},
        itl=itl_curve(base),
        h2_ratio={
            name.split(" ")[0]: h2_ratio(itl_curve(replace(base, **kw)))
            for name, kw in ITL_SCENARIOS
        },
        dense_offload=dense_offload(base),
    )


H1_FITS: tuple[tuple[str, dict, dict], ...] = (
    ("130 slots per layer (solver cap), W4A16 MTP", {}, {"slots": 130}),
    ("130 slots per layer, bf16 MTP experts", {"mtp_w4": False}, {"slots": 130}),
    ("106 slots per layer (the 12.0 GiB row of 5.2)", {}, {"slots": 106}),
    ("8 GiB KV reserve (the solver's default), W4A16 MTP", {}, {"kv_gib": 8.0}),
    ("8 GiB KV reserve, bf16 MTP experts", {"mtp_w4": False}, {"kv_gib": 8.0}),
)


def md_h1_ladder(base: Assume) -> str:
    head = ["H1 (stage 1)", "c1 step ms", "c8 step ms", "c16 step ms",
            "1k c8 tok/s", "c1 vs TP4 slot A", "c8 vs TP4 slot A"]  # fmt: skip
    body = [
        [r["name"], f"{r['c1']:.1f}", f"{r['c8']:.1f}", f"{r['c16']:.1f}",
         f"{r['tps8']:.0f}", f"{r['c1_ratio']:.2f}x", f"{r['c8_ratio']:.2f}x"]
        for r in h1_ladder(base)
    ]  # fmt: skip
    g = h1_gates(base)
    foot = (
        f"Gates: green c1 <= {g['green_c1']:.1f} and c8 <= {g['green_c8']:.1f} ms "
        f"(central x {GATE_GREEN}); red above c1 {g['red_c1']:.1f} or c8 "
        f"{g['red_c8']:.1f} ms (pessimistic); parity with both levers c1 <= "
        f"{g['parity_c1']:.1f} and c8 <= {g['parity_c8']:.1f} ms "
        f"(TP4 slot A x {GATE_PARITY})."
    )
    return table(head, body) + "\n\n" + foot


def md_h1_fit(base: Assume) -> str:
    head = ["H1 hub (one W7800, TP1)", "slots per layer", "slot pool GiB", "dense",
            "MTP", "fixed", "KV GiB", "KV tokens"]  # fmt: skip
    body = []
    for name, kw, sel in H1_FITS:
        m = h1_fit(replace(base, **kw), **sel)
        body.append(
            [name, f"{m['slots']:.0f}", f"{m['pool_gib']:.1f}", f"{m['dense']:.1f}",
             f"{m['mtp']:.1f}", f"{m['fixed']:.1f}", f"{m['kv_gib']:.1f}",
             f"{m['kv_tokens'] / 1e3:.0f}k"]
        )  # fmt: skip
    return table(head, body)


def md_stage2(base: Assume) -> str:
    head = ["Candidate", "GPUs", "users", "c1 step ms", "c8 step ms", "tok/s",
            "tok/s per GPU", "tok/s per user", "prefill tok/s", "16k TTFT s",
            "basis"]  # fmt: skip
    body = [
        [r["name"], str(r["gpus"]), str(r["users"]), f"{r['c1']:.1f}",
         f"{r['c8']:.1f}", f"{r['tps']:.0f}", f"{r['per_gpu']:.0f}",
         f"{r['per_user']:.1f}", f"{r['pre']:.0f}", f"{r['ttft']:.1f}", r["basis"]]
        for r in stage2(base)
    ]  # fmt: skip
    return table(head, body)


def md_offload(base: Assume) -> str:
    head = ["H2 hub dense chain", "step", "share f moved", "us per layer",
            "ms per step", "% of the step"]  # fmt: skip
    body = [
        [r["name"], r["mode"], f"{r['f']:.2f}", f"{r['us_layer']:.1f}",
         f"{r['ms_step']:.2f}", f"{r['pct']:.1f}"]
        for r in dense_offload(base)
    ]  # fmt: skip
    return table(head, body)


def solver_constants(a: Assume, ts: dict[str, Topo]) -> str:
    s = ts["H2"].s_l["c1"]
    chain = 5.0 * a.kappa * s
    rows = [
        ("hub per-expert us (--hub-us)", expert_us(1, ts["H2"].gpu, a)),
        ("V620 per-expert us (--expert-us)", expert_us(1, V620, a)),
        ("hub fixed us per layer, H2", shared_us(ts["H2"], a, s) + chain),
        ("hub fixed us per layer, H1", shared_us(ts["H1"], a, s) + chain),
        ("V620 fixed us per layer", 5.0 * s),
    ]
    return "\n".join(f"{n:<36}{v:.1f}" for n, v in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sens", action="store_true")
    args = ap.parse_args()
    a, ts = setup()
    o = results(a, ts)
    if args.json:
        o["stage_plan"] = stage_plan(Assume())
        print(json.dumps(o, indent=1, default=str))
        return
    print(md_tables(o, ts))
    print()
    print(f"TP8 check: model c1 {o['rows']['TP8']['c1']['step']:.1f} ms "
          f"(measured {TODAY['TP8']['c1']}), c8 {o['rows']['TP8']['c8']['step']:.1f} "
          f"(measured {TODAY['TP8']['c8']} flat, 37.2 hier)")  # fmt: skip
    print()
    print(md_deployments(o))
    print()
    print(md_memory(a))
    print()
    print(md_traffic())
    print()
    print(md_loopback(Assume()))
    print()
    print(prefill_table(a))
    print()
    print(scenarios(Assume()))
    print()
    print(md_h1_ladder(Assume()))
    print()
    print(md_h1_fit(a))
    print()
    print(md_spill_k(Assume()))
    print()
    print(md_stage2(Assume()))
    print()
    print(md_itl(Assume()))
    print()
    print(md_itl_scenarios(Assume()))
    print()
    print(md_offload(Assume()))
    print()
    print(solver_constants(a, ts))
    if args.sens:
        print()
        print(sensitivity(Assume()))


if __name__ == "__main__":
    main()
