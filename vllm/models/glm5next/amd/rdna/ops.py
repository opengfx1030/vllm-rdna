# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrappers and fake impls for the Glm5Next gfx1030 HIP ops."""

import contextlib

import torch

with contextlib.suppress(ImportError):
    import vllm._rocm_C  # noqa: F401

try:
    from torch.library import register_fake
except ImportError:  # older torch
    from torch.library import impl_abstract as register_fake


def _has_op(name: str) -> bool:
    # hasattr is unreliable on torch op namespaces (dir() only shows "name").
    try:
        getattr(torch.ops._rocm_C, name)
    except (AttributeError, RuntimeError):
        return False
    return True


def hip_sparse_indexer_available() -> bool:
    """Whether ``SparseAttnIndexerKpool.forward_hip`` has a logits path here.

    CDNA runs the AITER kernels. RDNA has no AITER build, but the ROCm
    logits dispatch carries its own gfx10x paths: ``rocm_fp8_mqa_logits``
    uses the HIP ``mqa_logits_prefill_rdna2`` kernel, and
    ``rocm_fp8_paged_mqa_logits`` falls back to the torch reference when the
    HIP decode kernel does not take the head count. Top-k is the ``_C``
    ``top_k_per_row_*`` op on both.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rdna import on_rdna_family

    return bool(rocm_aiter_ops.is_enabled()) or on_rdna_family()


def glm5_kda_slot_is_empty(slot: int, num_slots: int) -> bool:
    """Return whether a KDA state slot must not be read or written.

    Matches ``glm5_kda_slot_is_empty`` in ``glm5_kda_slot.cuh`` and
    ``fused_recurrent_kda`` (``state_idx <= 0``). ``NULL_BLOCK_ID`` is 0
    and ``PAD_SLOT_ID`` is -1.

    Args:
        slot: State index from the scheduler.
        num_slots: Length of the recurrent-state buffer.

    Returns:
        True when the slot is padding, the null block, or out of range.

    """
    return int(slot) <= 0 or int(slot) >= int(num_slots)


def has_glm5_kda_recurrent() -> bool:
    return _has_op("glm5_kda_recurrent_rdna2")


def glm5_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor,
    lower_bound: float,
) -> None:
    """In-place post-conv KDA recurrence. ``out`` and ``ssm_state`` are written.

    Accepts the decode path's ``[1, T, H, 128]`` / ``[1, T, H]`` views
    (``fused_recurrent_kda``'s layout) as well as ``[T, H, 128]`` /
    ``[T, H]``; the op itself takes the squeezed shapes. Token and head
    strides are free (the kernel indexes with them), the last dimension
    must be contiguous.

    Args:
        q: Post-conv query, fp16 or bf16, ``[1, T, H, 128]`` or ``[T, H, 128]``.
        k: Post-conv key, same shape and dtype as ``q``.
        v: Post-conv value, same shape and dtype as ``q``.
        g: Raw forget pre-activation, same shape and dtype as ``q``.
        beta: Raw beta, ``[1, T, H]`` or ``[T, H]``.
        a_log: Per-head fp32 ``A_log``, ``[H]``.
        dt_bias: fp32 ``[H * 128]``.
        state_indices: int32 slots, length ``T``. Slot 0 (NULL_BLOCK_ID)
            and negative slots (PAD_SLOT_ID is -1) zero their output row and
            leave the state untouched.
        ssm_state: fp32 ``[slots, H, 128, 128]`` in ``[V, K]`` order.
        out: Pre-norm output, same shape and dtype as ``q``.
        lower_bound: Bounded-gate floor.

    Raises:
        ValueError: A tensor has the wrong rank or a non-unit last stride.

    """
    q, k, v, g, out = (
        _squeeze_leading(t, 3, n)
        for t, n in ((q, "q"), (k, "k"), (v, "v"), (g, "g"), (out, "out"))
    )
    beta = _squeeze_leading(beta, 2, "beta")
    torch.ops._rocm_C.glm5_kda_recurrent_rdna2(
        q,
        k,
        v,
        g,
        beta,
        a_log.reshape(-1),
        dt_bias.reshape(-1),
        state_indices,
        ssm_state,
        out,
        lower_bound,
    )


def _squeeze_leading(t: torch.Tensor, rank: int, name: str) -> torch.Tensor:
    if t.dim() == rank + 1 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != rank:
        raise ValueError(
            f"glm5_kda_recurrent: {name} must have {rank} dims "
            f"(or a leading 1), got {tuple(t.shape)}"
        )
    if t.stride(-1) != 1:
        raise ValueError(f"glm5_kda_recurrent: {name} must be contiguous on dim -1")
    return t


if has_glm5_kda_recurrent():

    @register_fake("_rocm_C::glm5_kda_recurrent_rdna2")
    def _glm5_kda_recurrent_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor,
        lower_bound: float,
    ) -> None:
        return None
