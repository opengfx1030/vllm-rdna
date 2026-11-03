# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque ops for the RidgeFill fill steps.

The model forward is compiled with ``fullgraph``, so the old
``torch._dynamo.disable`` split point is refused: a graph break is not a
subtree. These ops are the PLE pattern instead. Dynamo sees one node per
step and never traces the body, which is where the host reads, the row plan
and the map GEMM live.

Every op is tensor-in/tensor-out so the graph keeps its shapes. The session
is read inside the impl, so the row plan is built once per forward and every
late layer reuses it. The fill decision itself is made *outside* the compiled
region: a forward with no filled rows passes ``None`` and never reaches these
ops, so its graph is the flag-off graph.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Set by ``bind_flashnext``. One model per process, so a module global is
# enough and keeps the op signatures free of non-tensor arguments.
_SESSION: Any = None
# Decoder layers by forward-context name, for the cache-only op. The decoder
# layer is not in ``no_compile_layers``, so the PLE lookup does not apply.
_LAYERS: dict[str, Any] = {}


def set_session(session: Any) -> None:
    global _SESSION
    _SESSION = session


def register_layer(name: str, layer: Any) -> None:
    _LAYERS[name] = layer


def _gather_positions(positions: torch.Tensor | None, index: torch.Tensor):
    """Gather rows from 1-D positions or channel-first M-RoPE positions."""
    if positions is None:
        return None
    if positions.dim() == 1:
        return positions.index_select(0, index)
    return positions.index_select(-1, index)


def ridgefill_capture(hidden_states: torch.Tensor) -> torch.Tensor:
    """Store the split layer's four-stream residual and build the row tensors.

    Runs once per fill forward, at layer ``split``. The row plan itself was
    built outside the compiled region; this only materializes the device
    tensors and the overflow check.
    """
    _SESSION.capture_split(hidden_states)
    return hidden_states


def _capture_fake(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def ridgefill_predict(
    x_s: torch.Tensor,
    filled_index: torch.Tensor,
    layer_idx: int,
    dout: int,
) -> torch.Tensor:
    """Apply layer ``layer_idx``'s map to the filled rows of the split state.

    Returns ``[F, dout]`` in ``x_s``'s dtype, or an empty ``[0, dout]`` when
    the layer has no fitted map. ``x_s`` is an argument only so the graph
    keeps the data dependency; the gathered rows are cached per forward.
    """
    return _SESSION.predict_rows(x_s, filled_index, layer_idx, dout)


def _predict_fake(
    x_s: torch.Tensor,
    filled_index: torch.Tensor,
    layer_idx: int,
    dout: int,
) -> torch.Tensor:
    return x_s.new_empty((filled_index.shape[0], dout))


def ridgefill_replace(
    block_input: torch.Tensor,
    prediction: torch.Tensor,
    filled_index: torch.Tensor,
) -> None:
    """Overwrite the filled rows of ``block_input`` with the prediction."""
    if prediction.shape[0] == 0:
        # A layer without a fitted map keeps its own block input.
        return
    block_input.index_copy_(0, filled_index, prediction.to(block_input.dtype))


def ridgefill_cache_only(
    block_input: torch.Tensor,
    positions: torch.Tensor | None,
    filled_index: torch.Tensor,
    exact_index: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Give filled rows their cache side effects and return the exact output.

    Filled rows write QSA K/V and the indexer keys, then stop: their residual
    is unused because every late layer re-predicts from the split state.
    Exact rows run the layer's attention on the compacted batch. A cache
    layout without a cache-only writer falls back to the full call, which
    rewrites the same slots with the same values.
    """
    layer = _LAYERS[layer_name]
    attn = layer.self_attn
    try:
        # Order matters: the filled K/V must be in the cache before the exact
        # rows attend over it. Filled rows precede the cut, so every exact
        # row's causal window sees them.
        attn.cache_only_forward(
            hidden_states=block_input.index_select(0, filled_index),
            positions=_gather_positions(positions, filled_index),
            row_index=filled_index,
        )
    except NotImplementedError:
        return attn(hidden_states=block_input, positions=positions)
    attn_out = torch.zeros_like(block_input)
    if exact_index.shape[0] == 0:
        # A chunk entirely before the cut has no exact rows to attend.
        return attn_out
    attn_out.index_copy_(
        0,
        exact_index,
        attn(
            hidden_states=block_input.index_select(0, exact_index),
            positions=_gather_positions(positions, exact_index),
            row_index=exact_index,
        ),
    )
    return attn_out


def _cache_only_fake(
    block_input: torch.Tensor,
    positions: torch.Tensor | None,
    filled_index: torch.Tensor,
    exact_index: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(block_input)


direct_register_custom_op(
    op_name="ridgefill_capture",
    op_func=ridgefill_capture,
    mutates_args=[],
    fake_impl=_capture_fake,
)

direct_register_custom_op(
    op_name="ridgefill_predict",
    op_func=ridgefill_predict,
    mutates_args=[],
    fake_impl=_predict_fake,
)

direct_register_custom_op(
    op_name="ridgefill_replace",
    op_func=ridgefill_replace,
    mutates_args=["block_input"],
)

direct_register_custom_op(
    op_name="ridgefill_cache_only",
    op_func=ridgefill_cache_only,
    mutates_args=[],
    fake_impl=_cache_only_fake,
)

__all__ = [
    "register_layer",
    "ridgefill_cache_only",
    "ridgefill_capture",
    "ridgefill_predict",
    "ridgefill_replace",
    "set_session",
]
