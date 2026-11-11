# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tools for the heterogeneous MoE mode (hub W7800 + expert V620).

``stats_io``    raw expert-stats JSONL -> ``hetero-moe-expert-stats/1`` JSON
``model_bytes`` byte table of a checkpoint (safetensors headers, no weights read)
``solve``       placement solver CLI (``hetero-moe-placement/1`` plan)
``simulate``    routing simulator for a plan (hub hit rate, V620 balance)
``validate``    plan invariants checker
``synth``       synthetic routing statistics (Zipf / uniform / clustered)

Pure Python + numpy, no torch and no GPU.
"""
