# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symbolic transfer sizes. No measured bandwidth is baked in."""


def activation_bytes(hidden_size: int, cold_pairs: int) -> int:
    """fp16 activation bytes for one MoE layer.

    A token row is ``hidden_size * 2`` bytes. The send carries one row
    per cold pair.

    Args:
        hidden_size: Model hidden size.
        cold_pairs: ``(token, expert)`` pairs that miss the hot set.

    Returns:
        ``hidden_size * 2 * cold_pairs``.
    """
    if hidden_size < 0 or cold_pairs < 0:
        raise ValueError("sizes must be non-negative")
    return hidden_size * 2 * cold_pairs


def decode_round_trips(num_moe_layers: int) -> int:
    """Cold round trips in one decode step, for one micro-batch.

    Each MoE layer is one send of rows plus one receive of weighted
    outputs. Layers with an empty cold set skip the send.

    Args:
        num_moe_layers: Routed MoE layers in the stack.

    Returns:
        The layer count. Empty-cold skips are not subtracted here.
    """
    if num_moe_layers < 0:
        raise ValueError("layer count must be non-negative")
    return num_moe_layers


def transfer_seconds(nbytes: int, bandwidth: str, latency: str) -> str:
    """Symbolic transfer time. Bandwidth and latency stay names.

    Args:
        nbytes: Bytes on the wire for this hop.
        bandwidth: Symbol for bytes per second. Not a measurement.
        latency: Symbol for the fixed hop delay. Not a measurement.

    Returns:
        An unevaluated expression.
    """
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    return f"({nbytes}) / ({bandwidth}) + ({latency})"
