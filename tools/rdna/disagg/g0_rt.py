# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes HIP helpers for the P/D go/no-go spikes (docs/rdna2/disagg-g0-2026-11-06.md).

Plain hipMemcpy on raw device addresses. The HIP runtime is resolved through
the libraries PyTorch already loaded: a second libamdhip64 would own separate
streams and host registrations.
"""

from __future__ import annotations

import contextlib
import ctypes

import numpy as np

H2D = 1
D2H = 2
D2D = 3

_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    import torch

    candidates: list[ctypes.CDLL] = []
    with contextlib.suppress(OSError):
        candidates.append(ctypes.CDLL(torch._C.__file__))
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        with contextlib.suppress(OSError):
            candidates.append(ctypes.CDLL(name))
    for lib in candidates:
        try:
            lib.hipMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.hipMemcpy.restype = ctypes.c_int
            lib.hipSetDevice.argtypes = [ctypes.c_int]
            lib.hipSetDevice.restype = ctypes.c_int
            lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
            lib.hipGetDevice.restype = ctypes.c_int
            lib.hipDeviceSynchronize.argtypes = []
            lib.hipDeviceSynchronize.restype = ctypes.c_int
        except AttributeError:
            continue
        _lib = lib
        return lib
    raise RuntimeError("no HIP runtime exporting hipMemcpy was found")


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: hipError_t={rc}")


def set_device(device: int) -> None:
    _check(_load().hipSetDevice(int(device)), "hipSetDevice")


def get_device() -> int:
    dev = ctypes.c_int(-1)
    _check(_load().hipGetDevice(ctypes.byref(dev)), "hipGetDevice")
    return dev.value


def sync() -> None:
    _check(_load().hipDeviceSynchronize(), "hipDeviceSynchronize")


def d2h(dst: np.ndarray, src_addr: int, nbytes: int | None = None) -> None:
    """Copy ``nbytes`` from device address ``src_addr`` into the host array."""
    n = dst.nbytes if nbytes is None else int(nbytes)
    assert dst.flags.c_contiguous and dst.flags.writeable and n <= dst.nbytes
    _check(
        _load().hipMemcpy(
            ctypes.c_void_p(dst.ctypes.data), ctypes.c_void_p(int(src_addr)), n, D2H
        ),
        "hipMemcpy D2H",
    )


def h2d(dst_addr: int, src: np.ndarray, nbytes: int | None = None) -> None:
    """Copy the host array into device address ``dst_addr``."""
    n = src.nbytes if nbytes is None else int(nbytes)
    assert src.flags.c_contiguous and n <= src.nbytes
    _check(
        _load().hipMemcpy(
            ctypes.c_void_p(int(dst_addr)), ctypes.c_void_p(src.ctypes.data), n, H2D
        ),
        "hipMemcpy H2D",
    )


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


_ipc_ready = False


def _ipc_lib() -> ctypes.CDLL:
    """The HIP runtime with the IPC / async-copy / stream prototypes declared."""
    global _ipc_ready
    lib = _load()
    if not _ipc_ready:
        lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
        lib.hipIpcGetMemHandle.restype = ctypes.c_int
        lib.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _IpcHandle,
            ctypes.c_uint,
        ]
        lib.hipIpcOpenMemHandle.restype = ctypes.c_int
        lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.hipIpcCloseMemHandle.restype = ctypes.c_int
        lib.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.hipMemcpyAsync.restype = ctypes.c_int
        lib.hipStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        lib.hipStreamCreateWithFlags.restype = ctypes.c_int
        lib.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.hipStreamSynchronize.restype = ctypes.c_int
        lib.hipStreamQuery.argtypes = [ctypes.c_void_p]
        lib.hipStreamQuery.restype = ctypes.c_int
        _ipc_ready = True
    return lib


def ipc_get_handle(ptr: int) -> bytes:
    """64-byte hipIpcMemHandle_t of the allocation that contains ``ptr``."""
    h = _IpcHandle()
    _check(
        _ipc_lib().hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)),
        "hipIpcGetMemHandle",
    )
    return ctypes.string_at(ctypes.addressof(h), 64)


def ipc_open(handle: bytes) -> int:
    """Map another process's allocation (flag 1 = lazy peer access); its address."""
    assert len(handle) == 64
    h = _IpcHandle()
    ctypes.memmove(ctypes.addressof(h), handle, 64)
    out = ctypes.c_void_p()
    _check(
        _ipc_lib().hipIpcOpenMemHandle(ctypes.byref(out), h, 1),
        "hipIpcOpenMemHandle",
    )
    return int(out.value or 0)


def ipc_close(ptr: int) -> None:
    _check(
        _ipc_lib().hipIpcCloseMemHandle(ctypes.c_void_p(ptr)),
        "hipIpcCloseMemHandle",
    )


def stream_create() -> int:
    s = ctypes.c_void_p()
    _check(_ipc_lib().hipStreamCreateWithFlags(ctypes.byref(s), 1), "hipStreamCreate")
    return int(s.value or 0)


def memcpy_async(dst: int, src: int, nbytes: int, stream: int, kind: int = D2D) -> None:
    _check(
        _ipc_lib().hipMemcpyAsync(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            nbytes,
            kind,
            ctypes.c_void_p(stream),
        ),
        "hipMemcpyAsync",
    )


def stream_sync(stream: int) -> None:
    _check(
        _ipc_lib().hipStreamSynchronize(ctypes.c_void_p(stream)),
        "hipStreamSynchronize",
    )


def stream_done(stream: int) -> bool:
    """hipStreamQuery: True when the stream is idle (hipSuccess), False while busy."""
    rc = _ipc_lib().hipStreamQuery(ctypes.c_void_p(stream))
    if rc == 0:
        return True
    if rc == 600:  # hipErrorNotReady
        return False
    raise RuntimeError(f"hipStreamQuery failed: hipError_t={rc}")
