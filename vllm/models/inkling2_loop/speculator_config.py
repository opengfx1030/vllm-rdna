# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator config of the Inkling-2 DSpark draft.

Kept apart from ``amd/dspark.py``: config modules are imported in every process and
must not import the speculator, which is resolved by name on the GPU worker only.

``speculator_options`` (both opt-in):

* ``{"adaptive_k": {}}`` makes the scheduler pick the number of drafts every step
  verifies from the running acceptance of the batch
  (``vllm/v1/spec_decode/dynamic/adaptive.py``); the batch-size schedule
  (``num_speculative_tokens_per_batch_size``) bounds it.  The dict may override the
  step cost model (``StepCostModel`` fields) and the estimator (``AdaptiveSpecTokens``
  fields).
* ``{"draft_vocab": "ids.npy"}`` restricts the greedy draft to the token ids listed
  in an offline file (``tools/rdna/inkling2/draft_vocab.py``): its LM-head sweep and
  folded-head rows shrink in proportion, at a small cost in acceptance.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from vllm.v1.spec_decode.dynamic.adaptive import AdaptiveSpecTokens, StepCostModel
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

# Milliseconds per decode step on one V620 (gfx1030, TP=1, native weights,
# FULL_AND_PIECEWISE), measured at batch 1 on real text (docs/rdna2/
# inkling2-speculation-profile.md, section 3.3): the target forward of 1, 3, 5 and
# 8 verified rows is 15.9, 23.0, 27.0 and 32.7 ms (+3.55 ms per row up to 3 rows,
# +1.94 ms per row above: the expert union saturates); LM head 2.12; sampling and
# gaps 0.11; the draft 4.15 + 0.13 per drafted token (Markov chain); host 0.4.
_V620_STEP_COST = StepCostModel(
    forward_ms=(15.90, 19.45, 22.99, 24.93, 26.87, 28.81, 30.75, 32.67),
    forward_row_ms=1.94,
    base_ms=2.23,
    draft_ms=4.15,
    draft_token_ms=0.13,
    draft_request_ms=0.35,
    host_ms=0.4,
)
# Measured on 2x V620 (TP=2, FULL_AND_PIECEWISE, one-shot HIP all-reduce,
# par1-cs25, 2026-11-24): step times for K = 0, 1, 2, 3, 7 are 12.22, 18.47,
# 20.14, 21.47 and 25.78 ms; 32,768-id draft_vocab drops K=2 from 20.14 to
# 19.02 ms (-1.12 ms, matching half the TP=1 LM-head sweep).
_V620_TP2_STEP_COST = StepCostModel(
    forward_ms=(11.07, 13.83, 15.41, 16.65, 17.64, 18.63, 19.61, 20.60),
    forward_row_ms=0.99,
    base_ms=1.15,
    draft_ms=3.00,
    draft_token_ms=0.09,
    draft_request_ms=0.25,
    host_ms=0.4,
)
# Measured on 4x V620 (TP=4, FULL_AND_PIECEWISE, one-shot HIP all-reduce,
# par1-cs25, 2026-11-24): step times for K = 0, 1, 2, 3, 7 are 9.50, 14.87,
# 16.06, 17.02 and 20.11 ms; 32,768-id draft_vocab drops K=2 to 15.61 ms and
# K=7 to 19.50 ms (-0.45 / -0.61 ms, matching a quarter of the TP=1 sweep).
_V620_TP4_STEP_COST = StepCostModel(
    forward_ms=(8.85, 11.35, 12.47, 13.36, 14.06, 14.77, 15.47, 16.17),
    forward_row_ms=0.70,
    base_ms=0.65,
    draft_ms=2.40,
    draft_token_ms=0.07,
    draft_request_ms=0.20,
    host_ms=0.4,
)
_V620_STEP_COSTS = {
    1: _V620_STEP_COST,
    2: _V620_TP2_STEP_COST,
    4: _V620_TP4_STEP_COST,
}
# What the draft vocabulary shrinks (draft_micro.py, graph replay of the whole draft at
# K = 2 / 7: 4.47 / 5.08 ms over the full vocabulary, 2.37 / 2.65 ms over 32768 of its
# 200058 ids, 2.15 ms over 16384): the head sweeps of the vocabulary are 2.35 ms of the
# fixed part and 0.08 of the 0.13 ms per drafted token (the id lookup is in the
# measurement).
_FULL_VOCAB_DRAFT_MS = 2.35
_FULL_VOCAB_TOKEN_MS = 0.08
_ESTIMATOR_OPTIONS = (
    "decay",
    "prior_accept",
    "prior_weight",
    "engine_decay",
    "engine_weight",
    "stop_margin",
    "probe_interval",
    "min_steps",
    "k_tolerance",
)


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2_loop.amd.dspark.Inkling2DSparkSpeculator"
    supported_options = DSparkSpeculatorConfig.supported_options | {
        "speculator_options"
    }

    @classmethod
    def apply_options(cls, spec) -> None:
        cls.validate_options(spec)
        super().apply_options(spec)

    @classmethod
    def validate_options(cls, spec) -> None:
        options = spec.speculator_options or {}
        unknown = set(options) - {"adaptive_k", "draft_vocab"}
        if unknown:
            raise ValueError(
                f"unknown Inkling-2 DSpark speculator_options {sorted(unknown)}; "
                "supported: adaptive_k, draft_vocab"
            )
        if spec.enable_adaptive_verification:
            raise ValueError(
                "the Inkling-2 DSpark draft does not support "
                "enable_adaptive_verification (its Markov head keeps no per-position "
                "draft distribution to estimate acceptance from); use "
                'speculator_options={"adaptive_k": {}} instead'
            )
        if "draft_vocab" in options and not isinstance(options["draft_vocab"], str):
            raise ValueError("draft_vocab must be the path of an .npy of token ids")
        adaptive = options.get("adaptive_k")
        if adaptive is not None and not isinstance(adaptive, dict | bool):
            raise ValueError("adaptive_k must be a bool or a dict of overrides")
        if isinstance(adaptive, dict):
            valid = set(StepCostModel.__dataclass_fields__) | set(_ESTIMATOR_OPTIONS)
            if set(adaptive) - valid:
                raise ValueError(
                    f"unknown adaptive_k options {sorted(set(adaptive) - valid)}; "
                    f"supported: {sorted(valid)}"
                )

    def _adaptive_options(self) -> dict[str, Any] | None:
        adaptive = (self.spec_config.speculator_options or {}).get("adaptive_k")
        if adaptive is None or adaptive is False:
            return None
        return adaptive if isinstance(adaptive, dict) else {}

    @property
    def uses_adaptive_num_tokens(self) -> bool:
        return self._adaptive_options() is not None

    def hash_factors(self) -> list[Any]:
        options = self.spec_config.speculator_options or {}
        return [
            *super().hash_factors(),
            self._adaptive_options() is not None,
            options.get("draft_vocab"),
        ]

    def make_num_tokens_policy(
        self, max_tokens: Sequence[int], tensor_parallel_size: int = 1
    ) -> AdaptiveSpecTokens | None:
        options = self._adaptive_options()
        if options is None:
            return None
        cost = {k: v for k, v in options.items() if k not in _ESTIMATOR_OPTIONS}
        if "forward_ms" in cost:
            cost["forward_ms"] = tuple(cost["forward_ms"])
        estimator = {k: options[k] for k in _ESTIMATOR_OPTIONS if k in options}
        return AdaptiveSpecTokens(
            cost=replace(self._step_cost(tensor_parallel_size), **cost),
            max_tokens=max_tokens,
            **estimator,
        )

    def _step_cost(self, tensor_parallel_size: int = 1) -> StepCostModel:
        """The step cost model for the active TP size, scaled for draft_vocab."""
        tp = max(int(tensor_parallel_size), 1)
        if tp >= 4:
            base = _V620_STEP_COSTS[4]
        elif tp >= 2:
            base = _V620_STEP_COSTS[2]
        else:
            base = _V620_STEP_COSTS[1]
        path = (self.spec_config.speculator_options or {}).get("draft_vocab")
        if path is None:
            return base
        hf_config = self.spec_config.draft_model_config.hf_config
        dropped = 1.0 - np.load(path, mmap_mode="r").shape[0] / hf_config.vocab_size
        # The draft LM head is sharded across TP ranks, so vocab-pruning savings
        # scale inversely with TP (measured: -2.1 ms TP1, -1.12 ms TP2, -0.5 ms TP4).
        return replace(
            base,
            draft_ms=base.draft_ms - (_FULL_VOCAB_DRAFT_MS / tp) * dropped,
            draft_token_ms=base.draft_token_ms - (_FULL_VOCAB_TOKEN_MS / tp) * dropped,
        )
