#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expected per-op decode latencies of the Flash-Next W4A16 step on a Radeon PRO
W7800 (gfx1100), TP2 / EP2, from bytes and the V620 trace. A MODEL, not a measurement.

    python tools/rdna/rdna3/perop_expectations.py            # markdown table
    python tools/rdna/rdna3/perop_expectations.py --json     # every number
    python tools/rdna/rdna3/perop_expectations.py --scenario pessimistic

A dense op is ``t(W) = max(bytes / BW, flops(W) / FLOPS) + kappa * s_l * chain`` per
call and rank (a fp16 GEMV does 2 FLOP per 2-byte weight and row, so ``flops =
bytes x W``). ``W`` is the number of rows of the step (3 per sequence with MTP2:
3 = one request, 24 = c8, 48 = c16). ``BW = eta x 864 GB/s``; ``FLOPS`` is a share
of the 90.4 TFLOPS fp16 peak: the v_dot2 GEMV route (M <= 8) and the WMMA GEMM
route. ``chain`` is the part of the V620 trace anchor
(docs/rdna/monokernel/p0-results.md, TP4, width 3) that is not weight streaming:
the anchor minus the bytes one V620 rank streams at its achieved 380 GB/s, i.e. the
launch chain of the small kernels, taken over with ``kappa`` 1.0 because the same
HIP kernels are assumed. ``s_l`` = 0.611 is the design doc's latency calibration at
c1 (the anchors are an older tree; the current TP4 tree runs at 0.611 of their
chain). The c8 value 0.484 is not used: it absorbs the W-dependent terms that this
model lists as separate rows. The other constants follow
docs/rdna2/hetero-moe-design-2026-11-07.md section 5.3 (eta_h, kappa, eff_hub).

What this does NOT know: nothing here has run on a W7800. The Triton / torch routes
that gfx1100 takes by default (``VLLM_RDNA3_HIP_OPS`` empty) have another chain than
the HIP kernels the anchors were measured with; the GPU plan
(docs/rdna2/rdna3-flashnext-port-2026-11-07.md) replaces each row by a measurement.
"""

import argparse
import json
from dataclasses import dataclass

MIB = 2**20
PEAK_BW = 864e9  # W7800 48 GB SKU, B/s
V620_BW = 380e9  # achieved on expert streaming (design doc 1.3), B/s
WMMA_PEAK = 90.4e12  # fp16 FLOP/s
VALU_EFF = 0.35  # share of WMMA_PEAK a v_dot2 GEMV sustains (M <= 8 route)
SCENARIOS = {
    # eta (share of 864 GB/s), kappa (chain scale), gemm efficiency (WMMA, all-in)
    "pessimistic": (0.65, 1.4, 0.20),
    "central": (0.80, 1.0, 0.30),
    "optimistic": (0.95, 0.8, 0.40),
}
WIDTHS = (3, 24, 48)


# Design doc 5.3 latency calibration s_l at c1, used at every width (see docstring).
LATENCY_SCALE = 0.611


# Model geometry (design doc 2.1 and 2.2).
HIDDEN = 2560
EXPERT_BYTES = 2_534_400  # W4A16 sym g128 routed expert
N_EXPERTS = 512
TOPK = 10
TP = 2
VOCAB = 248_320
GDN_STATE_BYTES = 3.06 * MIB  # per GDN layer and sequence at TP1
QSA_BUDGET = 2048  # selected tokens per query token
QSA_KV_PER_TOKEN = 256 * 2 * 2  # head_dim x (K, V) x fp16, one KV head per rank
# V620 anchors (TP4, width 3): HC pre 66, shared expert 19, router 39, experts 52
# (gate_up 32 + down 15 + act/fill), mean attention block 12.33 ms / 49 less its HC.
V620_ATTENTION_CORE_US = 12.33e3 / 49 - 66.0
MEAN_ATTN_MIB = (36 * 110.55 + 12 * (95.0 + 3.125)) / 48  # TP1, per layer


def distinct_experts(rows: int) -> float:
    """Distinct experts per layer touched by ``rows`` tokens (design doc 2.2)."""
    n = N_EXPERTS * (1 - (1 - TOPK / N_EXPERTS) ** rows)
    return n * (0.9 if rows >= 24 else 1.0)


@dataclass(frozen=True)
class DenseOp:
    name: str
    calls: int  # per decode step (target forward; drafter passes excluded)
    mib: float  # weight bytes per call and rank
    v620_anchor_us: float  # TP4 width-3 trace
    v620_mib: float  # bytes the anchor streamed on one V620 rank

    def chain_us(self) -> float:
        stream = self.v620_mib * MIB / V620_BW * 1e6
        return max(self.v620_anchor_us - stream, 0.0)

    def us(self, rows: int, eta: float, kappa: float, gemm_eff: float) -> float:
        byte_us = self.mib * MIB / (eta * PEAK_BW) * 1e6
        eff = VALU_EFF if rows <= 8 else gemm_eff
        flop_us = self.mib * MIB * rows / (eff * WMMA_PEAK) * 1e6
        return max(byte_us, flop_us) + kappa * LATENCY_SCALE * self.chain_us()


DENSE_OPS = (
    DenseOp("HC preamble (mix, gate, combine; decode)", 96, 12.6, 66.0, 12.6),
    DenseOp("router: gate GEMV + top-k + align/sort glue", 48, 2.5, 39.0, 2.5),
    DenseOp("shared expert (fused gate-up-silu + down)", 48, 9.38 / TP, 19.0, 9.38 / 4),
    DenseOp(
        "attention core, GDN/QSA mix 36:12 (dense GEMVs + conv + recurrent/QSA)",
        48,
        MEAN_ATTN_MIB / TP,
        V620_ATTENTION_CORE_US,
        MEAN_ATTN_MIB / 4,
    ),
)


def experts_us(
    rows: int, eta: float, kappa: float, per_expert_us: float = 0.0
) -> float:
    """Routed W4A16 experts of one layer on one EP2 rank (+ act/fill glue)."""
    per = per_expert_us or EXPERT_BYTES / (eta * PEAK_BW) * 1e6
    return distinct_experts(rows) / TP * per + kappa * LATENCY_SCALE * 6.0


def gdn_state_us(rows: int, eta: float) -> float:
    """Recurrent-state read + write of one GDN layer (bandwidth floor)."""
    seqs = max(rows // 3, 1)
    return seqs * 2 * (GDN_STATE_BYTES / TP) / (eta * PEAK_BW) * 1e6


def qsa_kv_us(rows: int, eta: float) -> float:
    """Sparse KV read of one QSA layer at a context >= 2048 tokens (upper bound)."""
    return rows * QSA_BUDGET * QSA_KV_PER_TOKEN / (eta * PEAK_BW) * 1e6


def lm_head_us(rows: int, eta: float, gemm_eff: float) -> float:
    """fp16 lm_head shard (vocab / TP2 x hidden), one pass."""
    nbytes = VOCAB / TP * HIDDEN * 2
    eff = VALU_EFF if rows <= 8 else gemm_eff
    return max(nbytes / (eta * PEAK_BW), nbytes * rows / (eff * WMMA_PEAK)) * 1e6


def table(scenario: str) -> dict:
    eta, kappa, gemm_eff = SCENARIOS[scenario]
    rows: dict = {}
    for op in DENSE_OPS:
        rows[op.name] = {
            "calls_per_step": op.calls,
            "mib_per_call": round(op.mib, 2),
            "chain_us": round(op.chain_us(), 1),
            "us": {w: round(op.us(w, eta, kappa, gemm_eff), 1) for w in WIDTHS},
        }
    rows["GDN recurrent state read+write (per GDN layer, floor)"] = {
        "calls_per_step": 36,
        "us": {w: round(gdn_state_us(w, eta), 1) for w in WIDTHS},
    }
    rows["QSA sparse KV read (per QSA layer, context >= 2k, upper bound)"] = {
        "calls_per_step": 12,
        "us": {w: round(qsa_kv_us(w, eta), 1) for w in WIDTHS},
    }
    rows["routed W4A16 experts (per layer, one EP2 rank)"] = {
        "calls_per_step": 48,
        "local_experts": {w: round(distinct_experts(w) / TP, 1) for w in WIDTHS},
        "us": {w: round(experts_us(w, eta, kappa), 1) for w in WIDTHS},
        "us_at_5us_per_expert": {
            w: round(experts_us(w, eta, kappa, 5.0), 1) for w in WIDTHS
        },
    }
    rows["lm_head shard + argmax (per pass)"] = {
        "calls_per_step": 3,
        "us": {w: round(lm_head_us(w, eta, gemm_eff), 1) for w in WIDTHS},
    }
    total = {}
    for w in WIDTHS:
        us = sum(op.calls * op.us(w, eta, kappa, gemm_eff) for op in DENSE_OPS)
        us += 36 * gdn_state_us(w, eta) + 12 * qsa_kv_us(w, eta)
        us += 48 * experts_us(w, eta, kappa) + 3 * lm_head_us(w, eta, gemm_eff)
        total[w] = round(us / 1e3, 2)
    return {
        "scenario": scenario,
        "eta": eta,
        "kappa": kappa,
        "rows": rows,
        "target_forward_ms_without_ar_and_gaps": total,
    }


def markdown(t: dict) -> str:
    lines = [
        f"Scenario {t['scenario']} (eta {t['eta']}, kappa {t['kappa']}); "
        "microseconds per call and rank at W = 3 / 24 / 48",
        "",
        "| op | calls/step | W=3 | W=24 | W=48 |",
        "|---|---|---|---|---|",
    ]
    for name, row in t["rows"].items():
        us = row["us"]
        lines.append(
            f"| {name} | {row['calls_per_step']} | {us[3]} | {us[24]} | {us[48]} |"
        )
    tot = t["target_forward_ms_without_ar_and_gaps"]
    lines += [
        "",
        "Sum over a step of the rows above (all-reduce, launch gaps, embedding, "
        f"drafter passes excluded), ms: {tot[3]} / {tot[24]} / {tot[48]}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="central")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    t = table(args.scenario)
    print(json.dumps(t, indent=1) if args.json else markdown(t))


if __name__ == "__main__":
    main()
