# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lean expert-parallel load balancing (EPLB) study tools for Flash-Next on RDNA2.

``geometry``   Flash-Next byte/geometry constants and the decode/prefill cost model
``placement``  expert-to-rank policies: linear, round robin, min-max packing,
               replication, replica routing (primary / hash / min-touch)
``loads``      per-step rank loads (touched experts, routed pairs, time model)
``simulate``   CPU simulator: routing-skew model from stats, placement scoring,
               redundancy trade, decode/prefill split, trace replay
``traces``     readers for real routing data (collector JSONL, id traces, per-call
               histograms, the ``eplb-stats/1`` counter dump)
``policy``     EMA windows, hysteresis (gain vs move cost), schedule replay
``summary``    per-layer / per-rank max-over-mean report of a counter dump
``solve``      static placement (phase 1): stats -> ``VLLM_RDNA_MOE_PLACEMENT`` JSON
``objective``  which objective (touched experts or routed pairs) a measured MoE
               kernel time follows

Design and gates: ``docs/rdna2/eplb-lean-design-2026-11-16.md``. Pure Python +
numpy; no torch, no GPU.
"""
