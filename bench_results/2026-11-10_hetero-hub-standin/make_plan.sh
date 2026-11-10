#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Plan of the V620-only hetero-MoE stand-in (hub CFG=h1v of tools/rdna/tp8pp2/driver.sh,
# recipe flashnext-hetero-h1-v620): the W1 routing statistics of the T10 collection
# (bench_results/2026-11-07_expert-stats-t10) solved for ONE V620 hub (HIP 1, bus d4) and
# FOUR V620 servers (HIP 2-5, buses d9 dc df e2) of the interim layout (leaf A = hub,
# leaf B = servers, box rules 2026-11-08), hub arch gfx1030.
#
# Why not the T10 plan H1.W1: it gives the W7800 hub 130 slots per layer (23.5 GiB of
# experts); a V620 has 29.7 GiB free, of which the non-expert weights (with the MTP head)
# take ~15.4 GiB (19.03 GiB of weights with 32 slots per layer measured on the first
# stand-in boot, 24.75 GiB reserved peak in eager mode), i.e. ~0.113 GiB per slot and
# layer. 40 slots per layer (5.7 GiB of hub experts with the MTP head's 512) leaves
# ~3 GiB for the compiled-mode peak [INFERENCE].
#
# Cost constants: the V620 hub costs what a V620 server costs (6.7 us per (token, expert)
# pair, 3.1 us fixed per layer), the links are the measured Gen4 x8 of the interim layout
# (13.9 GB/s, 13.3 us latency).
#
# Run from the repo root with PY=<python with numpy + regex>. solve/validate are those of
# feat/expert-stats 58889261df; cli.py retarget only sets the hub arch (the solver
# writes gfx1100 hubs).
set -euo pipefail
: "${PY:?}"
D=${D:-bench_results/2026-11-10_hetero-hub-standin}
STATS=${STATS:-bench_results/2026-11-07_expert-stats-t10/W1.stats.json}
export PYTHONPATH=.

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
"$PY" -m tools.rdna.hetero_moe.solve --stats "$STATS" --islands 1 --hubs-per-island 1 \
    --experts-per-island 4 --hub-hips 1 --expert-hips 2,3,4,5 --switches c6 \
    --island-mode replicate --hub-expert-budget-gib 6.0 --max-hub-per-layer 40 \
    --expert-budget-gib 26 --mtp-policy hub --mtp-expert-bytes 2534400 \
    --hub-us 6.7 --expert-us 6.7 --hub-fixed-us 3.1 --expert-fixed-us 3.1 \
    --latency-us 13.3 --gbps 13.9 --simulate-steps 0 --out "$tmp"
"$PY" vllm/distributed/hetero_moe/cli.py retarget "$tmp" --arch hub0=gfx1030 \
    --out "$D/H1v.W1.plan.json"
"$PY" -m tools.rdna.hetero_moe.validate "$D/H1v.W1.plan.json" --stats "$STATS"
