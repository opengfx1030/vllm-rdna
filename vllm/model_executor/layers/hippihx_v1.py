# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve half of the hippihx V1 ABI (github.com/BlivionIaG/hippihx).

hippihx owns the RDNA tile contracts and the C consume ABI
(``include/hippihx/v1.h``, rev 4). Its artifacts per fatbin slot are
``libhippihx_v1.so`` (host symbols, no device code) and
``hippihx_<arch>.hsaco`` (one raw AMDGPU code object). This module:

- loads both once, on first use, when ``VLLM_HIPPIHX=1``. It opens
  ``VLLM_HIPPIHX_LIB``, then calls ``hippihx_v1_load`` with the current
  device's arch and ``VLLM_HIPPIHX_CODE_OBJECT``;
- plans per op and params outside graph capture, caches the plan, and
  allocates and zeros its scratch once (page-commit);
- runs a ready plan with tensor descriptors built from torch views, on the
  current stream. Nothing here reads device memory.

A plan that is not ready returns ``None`` and the caller keeps its extras
kernel. Every hippihx op is not ready until its body migrates, so with
the flag on, today this only loads and plans. With the flag off, nothing
is imported or loaded.

Needs the torch-free ``hippihx`` Python package (``pip install
git+https://github.com/BlivionIaG/hippihx``) and a hippihx build.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_TORCH_DTYPES: dict[torch.dtype, str] = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.int8: "i8",
}


@dataclass
class HippihxPlan:
    """A ready V1 plan and the zeroed scratch it owns."""

    spec: Any  # hippihx.v1.OpSpec
    plan: Any  # hippihx.v1_ctypes.Plan
    scratch: torch.Tensor | None


class HippihxRuntime:
    """``libhippihx_v1.so`` with one slot's code object loaded."""

    def __init__(self, lib: ctypes.CDLL, arch: str) -> None:
        import hippihx.v1 as v1
        import hippihx.v1_ctypes as cv

        self.lib = lib
        self.arch = arch
        self.v1 = v1
        self.cv = cv
        self._plans: dict[tuple, HippihxPlan | None] = {}

    def plan_raw(self, qualname: str, dtype: torch.dtype | None, **params: Any):
        """``hippihx_v1_plan``: returns (status, ctypes plan). No caching."""

        spec = self.v1.find_spec(qualname)
        dtype_code = 0
        if dtype is not None:
            dtype_code = self.v1.V1_DTYPES.get(_TORCH_DTYPES.get(dtype, ""), -1)
        caps = self.cv.Caps(arch=self.arch.encode(), wave=0, dtype=dtype_code)
        array = self.cv.params_array(spec, params) if spec.params else None
        plan = self.cv.new_plan()
        rc = self.lib.hippihx_v1_plan(
            spec.v1_id, ctypes.byref(caps), array, len(spec.params), ctypes.byref(plan)
        )
        return self.v1.V1Status(rc), plan

    def plan(
        self,
        qualname: str,
        dtype: torch.dtype | None,
        device: torch.device,
        **params: Any,
    ) -> HippihxPlan | None:
        """Cached ready plan. None when not ready, refused, or new under capture."""

        key = (qualname, dtype, tuple(sorted(params.items())))
        if key in self._plans:
            return self._plans[key]
        if _capturing():
            return None  # plan and zero scratch before capture, never inside it
        status, plan = self.plan_raw(qualname, dtype, **params)
        result = None
        if status is self.v1.V1Status.OK and plan.ready:
            scratch = None
            if plan.scratch_nbytes:
                scratch = aligned_zeros(
                    plan.scratch_nbytes, self.v1.SCRATCH_ALIGN, device
                )
            result = HippihxPlan(self.v1.find_spec(qualname), plan, scratch)
        elif status is not self.v1.V1Status.OK:
            logger.warning_once(
                "hippihx: %s plan refused (%s); keeping the extras kernel.",
                qualname,
                status.name,
            )
        self._plans[key] = result
        return result

    def descriptors(self, spec: Any, tensors: dict[str, torch.Tensor | None]):
        """``hippihx_v1_tensor[]`` in the op's slot order. Missing slots stay NULL."""

        slots = spec.tensors
        array = (self.cv.Tensor * max(1, len(slots)))()
        for index, slot in enumerate(slots):
            t = tensors.get(slot.name)
            if t is None:
                continue
            code = self.v1.V1_DTYPES.get(_TORCH_DTYPES.get(t.dtype, ""), -1)
            array[index] = self.cv.tensor(
                t.data_ptr(), code, tuple(t.shape), tuple(t.stride())
            )
        return array

    def run_raw(
        self,
        plan: Any,
        spec: Any,
        tensors: dict[str, torch.Tensor | None],
        scratch: torch.Tensor | None,
    ):
        """``hippihx_v1_run`` on the current stream. Returns the V1 status."""

        array = self.descriptors(spec, tensors)
        scratch_ptr = scratch.data_ptr() if scratch is not None else None
        rc = self.lib.hippihx_v1_run(
            ctypes.byref(plan),
            array,
            len(spec.tensors),
            scratch_ptr,
            plan.scratch_nbytes,
            _stream_ptr(),
        )
        return self.v1.V1Status(rc)

    def run(self, p: HippihxPlan, tensors: dict[str, torch.Tensor | None]) -> bool:
        status = self.run_raw(p.plan, p.spec, tensors, p.scratch)
        if status is not self.v1.V1Status.OK:
            logger.warning_once(
                "hippihx: %s run refused (%s); keeping the extras kernel.",
                p.spec.qualname,
                status.name,
            )
            return False
        return True


_RUNTIME: HippihxRuntime | None = None
_TRIED = False


def aligned_zeros(nbytes: int, align: int, device: torch.device) -> torch.Tensor:
    """Zeroed uint8 scratch whose data_ptr is a multiple of ``align``.

    hippihx_v1_run refuses misaligned scratch. The CUDA/HIP caching
    allocator already aligns to 512, CPU torch only to 64, so pad and slice
    rather than rely on the allocator.
    """

    raw = torch.zeros(nbytes + align, dtype=torch.uint8, device=device)
    offset = (-raw.data_ptr()) % align
    return raw[offset : offset + nbytes]


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _stream_ptr() -> int | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.current_stream().cuda_stream


def _device_arch() -> str:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(":")[0]


def _code_object_path(arch: str) -> str | None:
    path = envs.VLLM_HIPPIHX_CODE_OBJECT
    if path and os.path.isdir(path):
        import hippihx.v1 as v1

        path = os.path.join(path, v1.CODE_OBJECT.format(arch=arch))
    return path


def get_runtime() -> HippihxRuntime | None:
    """Load hippihx once. None when off, missing, or the slot did not load."""

    global _RUNTIME, _TRIED
    if _TRIED:
        return _RUNTIME
    _TRIED = True
    if not envs.VLLM_HIPPIHX:
        return None
    try:
        import hippihx.v1 as v1
        import hippihx.v1_ctypes as cv
    except ImportError:
        logger.warning("VLLM_HIPPIHX=1 but the hippihx package is not installed.")
        return None
    if not envs.VLLM_HIPPIHX_LIB:
        logger.warning("VLLM_HIPPIHX=1 but VLLM_HIPPIHX_LIB is not set.")
        return None
    try:
        lib = cv.load(envs.VLLM_HIPPIHX_LIB)
    except (OSError, RuntimeError) as exc:
        logger.warning("hippihx: cannot load %s: %s", envs.VLLM_HIPPIHX_LIB, exc)
        return None
    arch = _device_arch()
    path = _code_object_path(arch)
    if not path:
        logger.warning("VLLM_HIPPIHX=1 but VLLM_HIPPIHX_CODE_OBJECT is not set.")
        return None
    status = v1.V1Status(lib.hippihx_v1_load(arch.encode(), path.encode()))
    if status is not v1.V1Status.OK:
        # FOREIGN_ISA covers HSA_OVERRIDE_GFX_VERSION and a wrong-arch object.
        logger.warning(
            "hippihx: %s code object %s not loaded (%s).", arch, path, status.name
        )
        return None
    logger.info("hippihx: loaded %s (%s), V1 ABI rev %d.", path, arch, v1.ABI_REVISION)
    _RUNTIME = HippihxRuntime(lib, arch)
    return _RUNTIME


def enabled() -> bool:
    """True once hippihx loaded for this device. Cheap after the first call."""

    return get_runtime() is not None


def _bucket(n: int) -> int:
    """Plan bound for a row count: next power of two, so plans stay few."""

    return 1 << max(0, (n - 1).bit_length())


def fa_fdot2_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int,
    kv_splits: int,
    sliding_window: int,
    scale: float,
) -> bool:
    """``attention.fa_fdot2`` decode into ``out``. False: run the extras kernel.

    Tensors follow the fa_rdna2_decode_paged layouts: query/out
    ``[tokens, H_q, D]``, 5-D paged K/V, per-token block_table/seq_lens.
    """

    rt = get_runtime()
    if rt is None:
        return False
    num_tokens, num_q_heads, head_dim = query.shape
    p = rt.plan(
        "attention.fa_fdot2",
        query.dtype,
        query.device,
        mode=0,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=key_cache.shape[1],
        block_size=block_size,
        kv_splits=kv_splits,
        sliding_window=sliding_window,
        causal=1,
        max_tokens=_bucket(num_tokens),
        scale=scale,
    )
    if p is None:
        return False
    return rt.run(
        p,
        {
            "q": query,
            "k_cache": key_cache,
            "v_cache": value_cache,
            "block_table": block_table,
            "seq_lens": seq_lens,
            "out": out,
        },
    )
