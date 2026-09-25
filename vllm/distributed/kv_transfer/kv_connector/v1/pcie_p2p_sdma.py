# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP IPC and SDMA copies for the PCIe P2P KV connector.

Kernel stores into a peer's cached KV are not coherent on gfx1030. The
copy engine (``hipMemcpy`` device-to-device on an IPC mapping) is the
path that lands in the destination's memory.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_size_t, c_void_p

_HIP_IPC_HANDLE_SIZE = 64
# hipIpcMemLazyEnablePeerAccess
_HIP_IPC_LAZY_ENABLE_PEER = 1
# hipMemcpyDeviceToDevice
_HIP_MEMCPY_DEVICE_TO_DEVICE = 3
# hipErrorPeerAccessAlreadyEnabled
_HIP_PEER_ALREADY = 704

_lib: ctypes.CDLL | None = None


class _IpcHandle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_byte * _HIP_IPC_HANDLE_SIZE)]


def _hip() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libamdhip64.so")
        _lib.hipGetErrorString.restype = ctypes.c_char_p
        _lib.hipGetErrorString.argtypes = [c_int]
        _lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), c_void_p]
        _lib.hipIpcGetMemHandle.restype = c_int
        _lib.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(c_void_p),
            _IpcHandle,
            c_int,
        ]
        _lib.hipIpcOpenMemHandle.restype = c_int
        _lib.hipMemcpyAsync.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            c_int,
            c_void_p,
        ]
        _lib.hipMemcpyAsync.restype = c_int
        _lib.hipDeviceEnablePeerAccess.argtypes = [c_int, c_int]
        _lib.hipDeviceEnablePeerAccess.restype = c_int
        _lib.hipGetLastError.argtypes = []
        _lib.hipGetLastError.restype = c_int
    return _lib


def _check(err: int, what: str) -> None:
    if err == 0:
        return
    text = _hip().hipGetErrorString(err)
    msg = text.decode() if text else str(err)
    raise RuntimeError(f"pcie_p2p HIP {what}: {msg} ({err})")


def export_ipc_handle(ptr: int) -> bytes:
    """IPC handle for a device allocation base pointer."""
    handle = _IpcHandle()
    _check(
        _hip().hipIpcGetMemHandle(ctypes.byref(handle), c_void_p(ptr)),
        "hipIpcGetMemHandle",
    )
    return bytes(handle.data)


def open_ipc_handle(raw: bytes) -> int:
    """Map a peer allocation into this process. Returns the device pointer."""
    if len(raw) != _HIP_IPC_HANDLE_SIZE:
        raise ValueError(
            f"IPC handle is {len(raw)} bytes, expected {_HIP_IPC_HANDLE_SIZE}"
        )
    handle = _IpcHandle()
    handle.data = (ctypes.c_byte * _HIP_IPC_HANDLE_SIZE).from_buffer_copy(raw)
    opened = c_void_p()
    _check(
        _hip().hipIpcOpenMemHandle(
            ctypes.byref(opened), handle, _HIP_IPC_LAZY_ENABLE_PEER
        ),
        "hipIpcOpenMemHandle",
    )
    if not opened.value:
        raise RuntimeError("pcie_p2p HIP hipIpcOpenMemHandle returned a null pointer")
    return int(opened.value)


def enable_peer_access(device: int) -> None:
    """Allow this process to DMA to ``device``. Already-enabled is success."""
    err = _hip().hipDeviceEnablePeerAccess(device, 0)
    if err in (0, _HIP_PEER_ALREADY):
        # "already enabled" sticks in the HIP last-error slot.
        _hip().hipGetLastError()
        return
    _check(err, "hipDeviceEnablePeerAccess")


def memcpy_device_async(
    dst: int, src: int, nbytes: int, stream: int
) -> None:
    """Queue one SDMA device-to-device copy on ``stream``."""
    if nbytes < 0:
        raise ValueError(f"nbytes must be >= 0, got {nbytes}")
    if nbytes == 0:
        return
    _check(
        _hip().hipMemcpyAsync(
            c_void_p(dst),
            c_void_p(src),
            c_size_t(nbytes),
            _HIP_MEMCPY_DEVICE_TO_DEVICE,
            c_void_p(stream),
        ),
        "hipMemcpyAsync",
    )
