# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, opt-in DFlash2 proposal diagnostics for greedy eager runs.

The JSONL output contains token IDs and scores, but no prompt text. Target
argmax is taken from raw logits, so use this only for temperature-zero requests
without grammar or logit penalties. This is a correctness diagnostic, not a
throughput benchmark.
"""

import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def compare_proposal(
    proposal: dict[str, Any], target_ids: list[int], logit_positions: list[int]
) -> dict[str, Any]:
    """Match a proposal to its next verification using stable token positions."""
    expected = proposal["sample_positions"][: len(target_ids)]
    actual = [position + 1 for position in logit_positions]
    if expected != actual:
        return {
            "matched": False,
            "reason": "position_mismatch",
            "expected": expected,
            "actual": actual,
        }

    selected = proposal["selected_ids"]
    candidates = proposal["candidate_ids"]
    prefix_valid = True
    positions = []
    for step, target_id in enumerate(target_ids):
        ids = candidates[step]
        positions.append(
            {
                "step": step,
                "prefix_valid": prefix_valid,
                "target_greedy": target_id,
                "top16_hit": target_id in ids,
                "head_top1_hit": target_id == ids[0],
                "selected_hit": target_id == selected[step],
                "selected_index": ids.index(selected[step])
                if selected[step] in ids
                else -1,
            }
        )
        prefix_valid &= target_id == selected[step]
    return {"matched": True, "positions": positions}


class DFlash2ProposalProbe:
    """Store at most 128 eager request-steps and append paired JSONL records."""

    MAX_RECORDS = 128

    def __init__(self, path: str):
        output = Path(path)
        if not output.is_absolute():
            raise ValueError("VLLM_DFLASH2_PROPOSAL_PROBE_PATH must be absolute")
        self.path = output
        self.min_position = int(
            os.environ.get("VLLM_DFLASH2_PROPOSAL_PROBE_MIN_POSITION", "0")
        )
        if self.min_position < 0:
            raise ValueError("VLLM_DFLASH2_PROPOSAL_PROBE_MIN_POSITION must be >= 0")
        self.pending: dict[int, dict[str, Any]] = {}
        self.records = 0

    @property
    def has_pending(self) -> bool:
        return bool(self.pending) and self.records < self.MAX_RECORDS

    def record_proposal(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        scores: torch.Tensor,
        selected_ids: torch.Tensor,
        sample_positions: torch.Tensor,
        state_slots: torch.Tensor,
    ) -> None:
        if self.records >= self.MAX_RECORDS:
            return
        num_reqs, num_steps, _ = candidate_ids.shape
        positions = (
            sample_positions[: num_reqs * num_steps]
            .view(num_reqs, num_steps)
            .detach()
            .cpu()
            .tolist()
        )
        eligible = [
            row
            for row, row_positions in enumerate(positions)
            if row_positions[0] >= self.min_position
        ]
        if not eligible:
            return
        candidates = candidate_ids.detach().cpu().tolist()
        unary = unary_logits.detach().cpu().tolist()
        edge_scores = scores.detach().cpu().tolist()
        selected = selected_ids.detach().cpu().tolist()
        slots = (
            state_slots[: num_reqs * num_steps]
            .view(num_reqs, num_steps)[:, 0]
            .detach()
            .cpu()
            .tolist()
        )
        for row in eligible:
            slot = slots[row]
            if slot < 0:
                continue
            self.pending[slot] = {
                "candidate_ids": candidates[row],
                "head_top1_ids": [ids[0] for ids in candidates[row]],
                "unary_logits": unary[row],
                "selector_scores": edge_scores[row],
                "selected_ids": selected[row],
                "sample_positions": positions[row],
            }

    def record_verification(
        self,
        target_argmax: torch.Tensor,
        logit_positions: torch.Tensor,
        state_slots: list[int],
        cu_num_logits: list[int],
        num_drafts: list[int],
    ) -> None:
        if not self.has_pending:
            return
        target = target_argmax.detach().cpu().tolist()
        positions = logit_positions.detach().cpu().tolist()
        output_records = []
        for req, slot in enumerate(state_slots):
            if self.records >= self.MAX_RECORDS:
                break
            proposal = self.pending.pop(slot, None)
            if proposal is None:
                continue
            count = min(num_drafts[req], len(proposal["selected_ids"]))
            if count == 0:
                continue
            start = cu_num_logits[req]
            end = start + count
            comparison = compare_proposal(
                proposal, target[start:end], positions[start:end]
            )
            output_records.append({"state_slot": slot, **proposal, **comparison})
            self.records += 1
        if output_records:
            os.makedirs(self.path.parent, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                for record in output_records:
                    stream.write(
                        json.dumps(
                            _json_safe(record), separators=(",", ":"), allow_nan=False
                        )
                        + "\n"
                    )
