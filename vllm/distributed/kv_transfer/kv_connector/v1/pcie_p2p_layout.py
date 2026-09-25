# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block geometry and transfer params for same-host PCIe P2P KV copies.

The copy itself is SDMA (`hipMemcpy` device-to-device on an IPC mapping).
These helpers stay free of HIP so the scheduler contract can be tested
on CPU.
"""

from __future__ import annotations

from typing import Any


def remote_prefill_token_count(prompt_len: int, num_computed_tokens: int) -> int:
    """Tokens the decode engine should load instead of recomputing.

    The last prompt token stays local so decode has a position to attend
    from. A non-positive result means there is nothing to pull.
    """
    if prompt_len <= 0:
        return 0
    return max(0, (prompt_len - 1) - num_computed_tokens)


def block_planes(shape: tuple[int, ...], num_blocks: int) -> tuple[int, int]:
    """Return ``(planes, elements_per_block)`` for a contiguous cache tensor.

    The block dimension is the first axis whose length is ``num_blocks``.
    Axes before it are independent planes (K and V are two planes when the
    tensor is ``[2, num_blocks, ...]``). Axes after it are one contiguous
    block. Plane ``p``, block ``b`` starts at element
    ``(p * num_blocks + b) * elements_per_block``.
    """
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    planes = 1
    for index, size in enumerate(shape):
        if size == num_blocks:
            tail = 1
            for axis in shape[index + 1 :]:
                tail *= axis
            return planes, tail
        planes *= size
    raise ValueError(
        f"shape {shape} has no dimension of length {num_blocks} for the block axis"
    )


def finished_transfer_params(
    *,
    block_ids: tuple[list[int], ...],
    engine_id: str,
    request_id: str,
    handshake_host: str,
    handshake_port: int,
    tp_size: int,
    remote_num_tokens: int,
) -> dict[str, Any]:
    """Params the prefill engine returns so decode can pull over PCIe.

    Field names match the disagg proxy contract used by NixlConnector
    (``do_remote_prefill``, ``remote_block_ids``, host, port, tp size).
    """
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": tuple(list(group) for group in block_ids),
        "remote_engine_id": engine_id,
        "remote_request_id": request_id,
        "remote_host": handshake_host,
        "remote_port": handshake_port,
        "tp_size": tp_size,
        "remote_num_tokens": remote_num_tokens,
        "transfer_mode": "pcie_p2p",
    }
