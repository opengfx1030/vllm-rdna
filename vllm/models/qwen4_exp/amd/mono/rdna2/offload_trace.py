# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, synchronous expert-offload receipts at real host execution boundaries.

Set VLLM_RDNA_EXPERT_TRACE_DIR before worker startup. MAX_ROWS (default 8)
limits rows in each receipt; MAX_CALLS (default 96) limits receipts per layer.
These debug copies intentionally synchronize and can change timing. No tracing
occurs while recording graphs, on metadata-free warmups, or on prefill batches.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import regex as re
import torch

ENABLED = bool(os.environ.get("VLLM_RDNA_EXPERT_TRACE_DIR"))
_COUNTS: dict[tuple[str, int, int], int] = {}
_FIELDS = (
    "num_actual_tokens",
    "num_decodes",
    "num_decode_tokens",
    "num_spec_decodes",
    "num_spec_decode_tokens",
    "num_prefills",
    "num_prefill_tokens",
    "query_start_loc",
    "seq_lens",
    "positions",
    "input_ids",
    "token_ids",
    "spec_query_start_loc",
    "non_spec_query_start_loc",
    "spec_token_indx",
    "non_spec_token_indx",
    "num_accepted_tokens",
    "cache_slot_indices",
    "spec_state_indices_tensor",
    "non_spec_state_indices_tensor",
)


def _metadata(value, limit):
    """Only bounded alignment fields, never whole cache/block-table tensors."""
    result = {}
    for name in _FIELDS:
        item = getattr(value, name, None)
        if isinstance(item, torch.Tensor):
            result[name] = item.detach().flatten()[:limit].cpu().clone()
        elif isinstance(item, (int, float, bool, str)):
            result[name] = item
    # Some attention backends keep decode sequence lengths in nested metadata.
    for name in ("decode", "decode_metadata"):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = _metadata(item, limit)
    return result


def _alignment_metadata(metadata, limit):
    # Builders commonly share one metadata object across many layer names.
    # Snapshot each object once instead of synchronizing for every alias.
    seen = set()
    result = {}
    for name, item in metadata.items():
        if id(item) not in seen:
            seen.add(id(item))
            result[name] = _metadata(item, limit)
    return result


def begin(mono, hidden_states, attn_out, injection, capture, *, kind="k2"):
    """Return a CPU-owned receipt, or None without touching device tensors."""
    if not ENABLED:
        return None
    arm_file = os.environ.get("VLLM_RDNA_EXPERT_TRACE_ARM_FILE")
    if arm_file and not Path(arm_file).exists():
        return None
    if capture is not None and not capture.replaying:
        return None
    # HIP can report capture on a Breakable replay's safe host boundary.
    # Its explicit replay context takes precedence over the raw query.
    if capture is None and torch.cuda.is_current_stream_capturing():
        return None
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = context.attn_metadata
    if not isinstance(metadata, dict) or not metadata or context.is_padding is None:
        return None
    # Reject dummy/profile calls and prefill: ordinals describe real decode
    # invocations, independent of bucket padding and layer count.
    if any(
        getattr(item, "num_prefills", 0) or getattr(item, "num_prefill_tokens", 0)
        for item in metadata.values()
    ):
        return None
    if kind == "gdn":
        target = os.environ.get("VLLM_RDNA_GDN_TRACE_SEQ_LEN")
        if target:
            seq_lens = None
            for item in metadata.values():
                for candidate in (
                    item,
                    getattr(item, "decode", None),
                    getattr(item, "decode_metadata", None),
                ):
                    value = getattr(candidate, "seq_lens", None)
                    if isinstance(value, torch.Tensor) and value.numel():
                        seq_lens = value
                        break
                if seq_lens is not None:
                    break
            if seq_lens is None or int(seq_lens.flatten()[0].item()) != int(target):
                return None
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    layer = int(mono.layer.layer_idx)
    key = (kind, rank, layer)
    ordinal = _COUNTS.get(key, 0)
    max_calls = max(
        0,
        int(
            os.environ.get(
                "VLLM_RDNA_GDN_TRACE_MAX_CALLS"
                if kind == "gdn"
                else "VLLM_RDNA_EXPERT_TRACE_MAX_CALLS",
                "4" if kind == "gdn" else "96",
            )
        ),
    )
    max_rows = max(0, int(os.environ.get("VLLM_RDNA_EXPERT_TRACE_MAX_ROWS", "8")))
    if ordinal >= max_calls or max_rows == 0:
        return None
    rows = (~context.is_padding[: hidden_states.shape[0]]).nonzero().flatten().cpu()
    if rows.numel() == 0:
        return None
    _COUNTS[key] = ordinal + 1
    rows = rows[:max_rows]
    # CPU indices avoid retaining any live model storage in the receipt.
    tensors = {
        name: value.detach()[rows.to(value.device)].cpu().clone()
        for name, value in (
            ("residual", hidden_states),
            ("attention", attn_out),
            ("injection", injection),
        )
    }
    return {
        "version": 1,
        "kind": kind,
        "rank": rank,
        "layer": layer,
        "call": ordinal,
        "padded_rows": hidden_states.shape[0],
        "rows": rows,
        "metadata": _alignment_metadata(metadata, max_rows * 16 + 1),
        "tensors": tensors,
    }


def finish(receipt, route_ids, route_weights, res_out, out, inj_out):
    """Snapshot before stable output storage is reused; atomically publish."""
    if receipt is None:
        return
    rows = receipt["rows"]
    for name, value in (
        ("route_ids", route_ids),
        ("route_weights", route_weights),
        ("residual_out", res_out),
        ("output", out),
        ("injection_out", inj_out),
    ):
        receipt["tensors"][name] = value.detach()[rows.to(value.device)].cpu().clone()
    _publish(receipt)


def _publish(receipt):
    directory = Path(os.environ["VLLM_RDNA_EXPERT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    name = f"rank{receipt['rank']}-layer{receipt['layer']}-call{receipt['call']:06d}.pt"
    if receipt["kind"] != "k2":
        name = f"{receipt['kind']}-{name}"
    path = directory / name
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    torch.save(receipt, temporary)
    temporary.replace(path)


def begin_gdn(layer, md, mixed_qkvz, ba, conv_state, ssm_state, sidx, capture):
    """Layer-one non-spec decode state immediately before the fused update."""
    target_layer = os.environ.get("VLLM_RDNA_GDN_TRACE_LAYER")
    if not ENABLED or target_layer is None or md.spec_sequence_masks is not None:
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer.prefix)
    if match is None or int(match[1]) != int(target_layer):
        return None
    owner = SimpleNamespace(layer=SimpleNamespace(layer_idx=int(match[1])))
    receipt = begin(owner, mixed_qkvz, ba, mixed_qkvz, capture, kind="gdn")
    if receipt is None:
        return None
    tensors = receipt["tensors"]
    tensors["mixed_qkvz"] = tensors.pop("residual")
    tensors["ba"] = tensors.pop("attention")
    del tensors["injection"]
    rows = receipt["rows"]
    indices = sidx.detach()[rows.to(sidx.device)].flatten().cpu().long()
    # Keep NULL_BLOCK_ID too: its unexpected mutation is diagnostic evidence.
    tensors["state_indices"] = indices.clone()
    for name, state in (("conv_before", conv_state), ("ssm_before", ssm_state)):
        tensors[name] = state.detach()[indices.to(state.device)].cpu().clone()
    return receipt


def finish_gdn(receipt, y, conv_state, ssm_state):
    if receipt is None:
        return
    tensors = receipt["tensors"]
    tensors["y"] = y.detach()[receipt["rows"].to(y.device)].cpu().clone()
    indices = tensors["state_indices"]
    for name, state in (("conv_after", conv_state), ("ssm_after", ssm_state)):
        tensors[name] = state.detach()[indices.to(state.device)].cpu().clone()
    _publish(receipt)
