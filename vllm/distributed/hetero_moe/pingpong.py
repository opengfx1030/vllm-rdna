# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ping-pong micro-batches across one MoE layer.

MegaScale-Infer splits a batch into micro-batches and overlaps the
expert hop of one with the attention of the next. Here the cold round
trip is that hop. ``begin_moe`` issues the send (and the local hot
experts). ``end_moe`` is the receive. Attention of micro-batch ``i+1``
runs between those two calls.

The GPU model runner is not rewritten. ``split_routed_forward`` still
completes one batch before returning. A runner that has micro-batches
calls :func:`run_ping_pong`.
"""

from collections.abc import Callable


def run_ping_pong(
    num_microbatches: int,
    num_layers: int,
    attn: Callable[[int, int], None],
    begin_moe: Callable[[int, int], None],
    end_moe: Callable[[int, int], None],
    cold_nonempty: Callable[[int, int], bool],
) -> list[str]:
    """Schedule micro-batches so a cold send overlaps the next attention.

    Args:
        num_microbatches: Micro-batches in this step.
        num_layers: Routed MoE layers.
        attn: ``(microbatch, layer)`` attention on the fast tier.
        begin_moe: Issue hot compute and the cold send. Must not wait.
        end_moe: Wait for that send and combine.
        cold_nonempty: False skips the send and the matching receive.

    Returns:
        A log of ``attn``, ``send``, and ``recv`` events.

    Raises:
        ValueError: A non-positive micro-batch or layer count.
    """
    if num_microbatches < 1 or num_layers < 1:
        raise ValueError("ping-pong needs at least one micro-batch and layer")
    log: list[str] = []
    for layer in range(num_layers):
        inflight: tuple[int, int] | None = None
        for mb in range(num_microbatches):
            attn(mb, layer)
            log.append(f"attn:{mb}:{layer}")
            if inflight is not None:
                end_moe(inflight[0], inflight[1])
                log.append(f"recv:{inflight[0]}:{inflight[1]}")
                inflight = None
            if cold_nonempty(mb, layer):
                begin_moe(mb, layer)
                log.append(f"send:{mb}:{layer}")
                inflight = (mb, layer)
        if inflight is not None:
            end_moe(inflight[0], inflight[1])
            log.append(f"recv:{inflight[0]}:{inflight[1]}")
    return log
