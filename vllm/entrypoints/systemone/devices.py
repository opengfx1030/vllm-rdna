# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spare-device and dtype checks for the decision model.

The decision model runs in the API-server process. It must not open a context
on a GPU the main engine's tensor-parallel (or pipeline / data-parallel)
workers use. Those devices own uncached P2P, FA/QSA graph capture, and the
persistent heaps.
"""

import os
from argparse import Namespace

from vllm.entrypoints.systemone.errors import SystemOneStartupError

_GIB = 1024**3


def parse_device(device: str) -> tuple[str, int | None]:
    """Return ``("cpu", None)`` or ``("cuda", index)``.

    ROCm/HIP devices use torch's ``cuda:N`` namespace. ``cuda`` without an
    index is rejected so the spare-device check cannot follow the process
    default device.

    Raises:
        ValueError: The device string is not ``cpu`` or ``cuda:N``.
    """
    text = device.strip().lower()
    if text == "cpu":
        return "cpu", None
    prefix = "cuda:"
    if text.startswith(prefix) and text[len(prefix) :].isdigit():
        return "cuda", int(text[len(prefix) :])
    raise ValueError(
        "--systemone-device must be 'cpu' or 'cuda:N'. "
        "ROCm/HIP uses the torch cuda device namespace."
    )


def resolve_dtype(device_kind: str, requested: str | None) -> str:
    """Pick float16 or float32. bfloat16 is rejected.

    Args:
        device_kind: ``cpu`` or ``cuda``.
        requested: ``float16``, ``float32``, ``auto``, or None.

    Returns:
        ``float16`` or ``float32``. CPU defaults to float32. CUDA defaults
        to float16.

    Raises:
        ValueError: The dtype is missing or is bfloat16.
    """
    if requested is None or requested.strip().lower() in {"", "auto"}:
        return "float32" if device_kind == "cpu" else "float16"
    norm = requested.strip().lower().removeprefix("torch.")
    norm = {
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
    }.get(norm, norm)
    if norm in {"bfloat16", "bf16"}:
        raise ValueError(
            "systemone rejects bfloat16. Use float16 or float32. "
            "The decision model rides stock torch on ROCm."
        )
    if norm not in {"float16", "float32"}:
        raise ValueError(
            f"--systemone-dtype must be float16, float32, or auto. Got {requested!r}."
        )
    return norm


def read_visible_devices() -> list[str] | None:
    """Return the process GPU mask, or None when neither mask is set.

    HIP_VISIBLE_DEVICES wins when it is the only one set. If both are set
    they must be identical; a mismatch fails closed.

    Raises:
        SystemOneStartupError: The two masks disagree or contain an empty slot.
    """
    hip = os.environ.get("HIP_VISIBLE_DEVICES") or ""
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    if hip and cuda and hip != cuda:
        raise SystemOneStartupError(
            "HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES disagree "
            f"({hip!r} vs {cuda!r}). Set one mask, or make them identical, "
            "before choosing --systemone-device."
        )
    raw = hip or cuda
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if any(part == "" for part in parts):
        raise SystemOneStartupError(
            "The GPU visibility mask contains an empty entry: " + raw
        )
    return parts


def engine_logical_devices(args: Namespace) -> set[int]:
    """Logical ``cuda:N`` indices the main engine occupies in this mask.

    The span is ``data_parallel_size * tensor_parallel_size *
    pipeline_parallel_size``, which is the prefix of the visible mask vLLM
    assigns to engine replicas. Explicit ``--device-ids`` are included too,
    so a non-prefix placement is still treated as an engine device.
    """
    tp = _at_least_one(getattr(args, "tensor_parallel_size", 1))
    pp = _at_least_one(getattr(args, "pipeline_parallel_size", 1))
    dp = _at_least_one(getattr(args, "data_parallel_size", 1))
    occupied = set(range(tp * pp * dp))
    for item in getattr(args, "device_ids", None) or []:
        if isinstance(item, int):
            occupied.add(item)
        elif isinstance(item, str) and item.strip().isdigit():
            occupied.add(int(item.strip()))
    return occupied


def assert_spare_device(device: str, args: Namespace) -> tuple[str, int]:
    """Reject a decision device that maps onto an engine GPU.

    Returns:
        The ``cuda`` kind and logical index.

    Raises:
        SystemOneStartupError: The device is not a spare GPU in this process.
        ValueError: The device string is not ``cuda:N``.
    """
    kind, index = parse_device(device)
    if kind != "cuda" or index is None:
        raise SystemOneStartupError(
            "assert_spare_device is only used for cuda:N devices."
        )
    visible = read_visible_devices()
    occupied = engine_logical_devices(args)
    if visible is not None and index >= len(visible):
        raise SystemOneStartupError(
            f"--systemone-device cuda:{index} is outside the visible GPU "
            f"mask ({len(visible)} device(s): {','.join(visible)}). A spare "
            "GPU has to be listed in HIP_VISIBLE_DEVICES or "
            "CUDA_VISIBLE_DEVICES after the engine ranks. On a 4-GPU TP=4 "
            "box there is no spare device: use --systemone-device cpu, or "
            "Mode B (`python -m vllm.entrypoints.systemone.server`) on "
            "another host."
        )
    target = str(index) if visible is None else visible[index]
    occupied_physical: list[str] = []
    for logical in sorted(occupied):
        if visible is None:
            occupied_physical.append(str(logical))
        elif logical < len(visible):
            occupied_physical.append(visible[logical])
    if index in occupied or target in occupied_physical:
        tp = _at_least_one(getattr(args, "tensor_parallel_size", 1))
        pp = _at_least_one(getattr(args, "pipeline_parallel_size", 1))
        dp = _at_least_one(getattr(args, "data_parallel_size", 1))
        mask = "unset" if visible is None else ",".join(visible)
        raise SystemOneStartupError(
            f"--systemone-device cuda:{index} maps to {target!r}, which "
            "the main engine uses "
            f"(tensor_parallel_size={tp}, pipeline_parallel_size={pp}, "
            f"data_parallel_size={dp}; occupied logical devices "
            f"{sorted(occupied)}; visible mask {mask}). The decision model "
            "must use a spare device, never a tensor-parallel engine device. "
            "Those devices own uncached P2P, FA/QSA graph capture, and the "
            "persistent heaps. Lowering --gpu-memory-utilization does not "
            "make a TP rank spare: that setting is applied on every rank. "
            "On a 4-GPU TP=4 box use --systemone-device cpu or run Mode B "
            "on another host. Refusing to start."
        )
    return kind, index


def assert_free_vram(index: int, reserve_gb: float) -> None:
    """Fail closed when the spare device has less free memory than the reserve.

    Called only after :func:`assert_spare_device`, so this never initializes
    a context on an engine GPU.

    Raises:
        SystemOneStartupError: Free memory is below the reserve, or it cannot
            be read.
    """
    import torch

    try:
        free, _total = torch.cuda.mem_get_info(index)
    except Exception as exc:
        raise SystemOneStartupError(
            f"Cannot read free VRAM on cuda:{index} ({exc}). "
            "Refusing to start the decision model."
        ) from exc
    reserve = int(reserve_gb * _GIB)
    if free < reserve:
        raise SystemOneStartupError(
            f"cuda:{index} has {free / _GIB:.2f} GiB free, below "
            f"--systemone-vram-reserve-gb={reserve_gb}. The decision model "
            "only runs on a spare device, and that device must keep this "
            "reserve. Refusing to start."
        )


def open_decision_stream(index: int):
    """Open a non-default torch stream on ``cuda:index``.

    Graph capture records the default stream (and any stream the engine
    explicitly joins). The decision model must not use those.

    Returns:
        A ``torch.cuda.Stream`` that is not the device default stream.

    Raises:
        SystemOneStartupError: The new stream is the default stream.
    """
    import torch

    stream = torch.cuda.Stream(device=index)
    default = torch.cuda.default_stream(index)
    new_id = getattr(stream, "cuda_stream", None)
    default_id = getattr(default, "cuda_stream", None)
    if new_id is not None and new_id == default_id:
        raise SystemOneStartupError(
            "systemone created the default CUDA stream on "
            f"cuda:{index}. The decision model must use a dedicated stream "
            "so it cannot join graph capture. Refusing to start."
        )
    return stream


def _at_least_one(value: object) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return number if number >= 1 else 1
