# SPDX-License-Identifier: Apache-2.0
"""Hold skip_compiled HIP / ATen CUDA tensors alive for FULL graph capture.

HIP's graph mempool does not retain freed capture-time storages (private
pool stays ~3 MiB). Eager 16k prefill then recycles those pages and FULL
replay emits first-token-ok then ``duct``.

Do **not** call ``torch.cuda.is_current_stream_capturing()`` from compiled
paths — Dynamo traces that as a torch.* op that returns a non-Tensor.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_KEEP: list[Any] = []
capturing_full: bool = False
_orig_is_capturing: Any = None


def install_rdna2_capture_guard() -> None:
    """After FULL capture, ignore HIP's is_current_stream_capturing.

    gfx1030 reports capturing during eager mixed 16k and during graph
    replay (rdna2_graph_keepalive.cuh). Persist already uses an atomic;
    PYNCCL / rdna_ar / GDN still trust the HIP query and then allocate
    into the graph pool — 16k mixed recycles size-1 FULL workspace.
    """
    global _orig_is_capturing
    if _orig_is_capturing is not None:
        return
    _orig_is_capturing = torch.cuda.is_current_stream_capturing

    def _guarded() -> bool:
        return bool(capturing_full)

    torch.cuda.is_current_stream_capturing = _guarded  # type: ignore[method-assign]


def keepalive_if_capturing(t: Any) -> Any:
    if t is not None and capturing_full:
        _KEEP.append(t)
    return t


def alloc_eager_or_capture(
    shape: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Capture: caching-allocator zeros (graph-legal). Eager: immortal hipMalloc.

    Mixed 16k skip_compiled ``empty_like`` recycles FULL graph-private pages.
    """
    dims = tuple(int(s) for s in shape)
    if capturing_full:
        t = torch.zeros(*dims, dtype=dtype, device=device)
        return keepalive_if_capturing(t)
    # Mixed 16k runs under eager_alloc_isolation (ATen pool (0,3)).
    # immortal hipMalloc bypasses that pool and starves SiluAndMul.
    t = torch.zeros(*dims, dtype=dtype, device=device)
    return t


def immortal_zeros(
    shape: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """hipMalloc + from_blob. Never returned to the caching allocator.

    Capture-time ``torch.zeros`` sits in the default pool after the ~7 GiB
    KV allocation; mixed 16k KV OOB then overwrites FULL persist pages.
    """
    dims = [int(s) for s in shape]
    try:
        ref = torch.empty(0, dtype=dtype, device=device)
        t = torch.ops._rocm_C.rdna2_immortal_zeros(ref, dims)
    except Exception:
        t = torch.zeros(*dims, dtype=dtype, device=device)
    _KEEP.append(t)
    return t


def _keep_cuda(obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            _KEEP.append(obj)
        return
    if isinstance(obj, (tuple, list)):
        for x in obj:
            _keep_cuda(x)


class _KeepCudaTensorsMode(TorchDispatchMode):
    """Pin every ATen CUDA tensor created while FULL skip_compiled captures."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if capturing_full:
            _keep_cuda(out)
        return out


@contextmanager
def keep_aten_cuda_tensors() -> Iterator[None]:
    """Context manager: keep all ATen CUDA outputs (not just GEMM wrappers)."""
    with _KeepCudaTensorsMode():
        yield


class _KeepAllCudaTensorsMode(TorchDispatchMode):
    """Keep every ATen CUDA tensor, including eager 16k skip_compiled temps."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        _keep_cuda(out)
        return out


@contextmanager
def keep_all_aten_cuda_tensors() -> Iterator[None]:
    with _KeepAllCudaTensorsMode():
        yield


def hip_stream_is_capturing() -> bool:
    """Raw HIP query, ignoring the Python FULL-capture guard."""
    fn = _orig_is_capturing or torch.cuda.is_current_stream_capturing
    try:
        return bool(fn())
    except Exception:
        return False


def hybrid_linear_attn_workspace_bytes(max_num_tokens: int) -> int:
    """Scratch that ``profile_run(skip_attn=True)`` never allocates.

    Hybrid Qwen3.5/3.8 is 48 GDN + 16 FA layers sharing one 784-token page.
    skip_attn dummy_run skips both, so the KV pin fills the GPU. The first
    mixed 16k step then hipMalloc's this workspace with 0 B free and either
    OOMs or recycles FULL-graph pages (seq-after ``duct``).

    Grow-only GDN tables are *shared across the 48 layers* (not ×48). At
    chunk 2048, TP=2, H_v=24, K=V=128, FLA chunk 64::

        GDN chain A/A_inv/w/u/h/v_new/o/prep/stitch  ~140 MiB
        FA persist O [2048, 12, 256] fp16            ~13 MiB
        AWQ GEMM persist C [2048, hidden]            ~20 MiB
        fragmentation / extra NT from 2 prefills     ~80 MiB

    256 MiB covers that. Kept alive so the KV pin is sized around it.
    """
    n = int(max_num_tokens)
    # Scale from the 2048-token production chunk; never below 64 MiB.
    return max(64 << 20, (256 << 20) * max(n, 1) // 2048)


def reserve_hybrid_linear_attn_workspace(
    device: torch.device | str,
    max_num_tokens: int,
    existing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pin skip_attn-omitted GDN/FA scratch so profiling counts it."""
    need = hybrid_linear_attn_workspace_bytes(max_num_tokens)
    if (
        existing is not None
        and existing.is_cuda
        and existing.numel() >= need
        and existing.device == torch.device(device)
    ):
        return existing
    return torch.zeros(need, dtype=torch.uint8, device=device)
