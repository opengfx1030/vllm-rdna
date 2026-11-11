# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-step TP-wide fail-stop for Mono MTP and offloaded expert kernels."""

import torch

from vllm.distributed.parallel_state import get_tp_group

from .layout import _WORKSPACES

_ZERO: dict[int, torch.Tensor] = {}
# Development worker RPC may select wait-only completion to isolate pacing.
_USE_TP_COLLECTIVE = True


def enabled(vllm_config) -> bool:
    # Compute once at runner construction; resident, non-Mono routes stay off.
    from vllm.models.qwen4_exp.amd.mono_decode_rdna2 import options

    opts = options(vllm_config)
    return bool(
        opts is not None
        and (opts.mtp_drafter or getattr(opts, "expert_cache_slots", 0) > 0)
    )


def local_error_view(device: torch.device) -> torch.Tensor:
    idx = device.index
    if idx is None:
        idx = torch.accelerator.current_device_index()
    # Keep the one-arena direct view contract. Combining scopes must not
    # mutate either sticky header, and resident "model" scope is excluded.
    errors = [
        ws[8:12].view(torch.int32)
        for scope in ("mtp", "model-offload")
        if (ws := _WORKSPACES.get((idx, scope))) is not None
    ]
    if len(errors) == 1:
        return errors[0]
    if len(errors) == 2:
        return errors[0] + errors[1]
    # Still participate in the collective when this rank used the fallback
    # path. Allocate once; never mutate this sentinel.
    zero = _ZERO.get(idx)
    if zero is None:
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        _ZERO[idx] = zero
    return zero


def collect_tp_error(device: torch.device) -> torch.Tensor:
    """One post-proposal GPU collective; no host/device synchronization."""
    error = local_error_view(device)
    if not _USE_TP_COLLECTIVE:  # diagnostic only; caller still waits for its copy
        return error
    return get_tp_group().all_reduce(error)


def raise_if_error(error_cpu: torch.Tensor, device: torch.device | None = None) -> None:
    if int(error_cpu.item()) == 0:
        return
    details = ""
    if device is not None:
        idx = device.index
        if idx is None:
            idx = torch.accelerator.current_device_index()
        headers = {
            scope: ws[8:20].view(torch.int32).cpu().tolist()
            for scope in ("mtp", "model-offload")
            if (ws := _WORKSPACES.get((idx, scope))) is not None
        }
        details = f"; local [err_code, err_phase, err_wg]={headers}"
    raise RuntimeError(f"Mono K2 fault on a TP rank; restart the engine{details}")


def clear_mtp_fault_for_recovery(device: torch.device) -> None:
    """Explicit test/offline recovery; caller must stop all inference first."""
    idx = device.index
    if idx is None:
        idx = torch.accelerator.current_device_index()
    torch.accelerator.synchronize()
    ws = _WORKSPACES.get((idx, "mtp"))
    if ws is not None:
        ws.zero_()  # also discard partially written gu/acc/scratch and stats
        torch.accelerator.synchronize()
