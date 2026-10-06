# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned staging and length-prefixed frames for the host-staged link.

TCP and shared memory move these frames. They do not invent a device
bandwidth. hipMemcpyAsync is the non-blocking ``copy_`` on the side
stream when the tensors live on a HIP device; this module does not
synchronize the caller.
"""

import io
import socket
import struct
from multiprocessing import shared_memory

import torch

_HEADER = struct.Struct(">Q")


def pack_tensors(payload: dict[str, torch.Tensor]) -> bytes:
    """Serialize a dict of CPU tensors.

    Args:
        payload: Named tensors. They are moved to CPU first.

    Returns:
        A ``torch.save`` blob.
    """
    cpu = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in payload.items()
    }
    buf = io.BytesIO()
    torch.save(cpu, buf)
    return buf.getvalue()


def unpack_tensors(blob: bytes) -> dict[str, torch.Tensor]:
    """Inverse of :func:`pack_tensors`.

    Args:
        blob: Bytes from :func:`pack_tensors`.

    Returns:
        CPU tensors.
    """
    loaded = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError("wire payload must be a dict of tensors")
    return loaded


def frame(blob: bytes) -> bytes:
    """Length-prefix ``blob`` with a big-endian uint64."""
    return _HEADER.pack(len(blob)) + blob


def unframe(data: bytes) -> bytes:
    """Strip one frame. The buffer must hold exactly that frame.

    Args:
        data: Header plus payload.

    Returns:
        The payload bytes.

    Raises:
        ValueError: The buffer is short or long.
    """
    if len(data) < _HEADER.size:
        raise ValueError("truncated frame header")
    (size,) = _HEADER.unpack(data[: _HEADER.size])
    body = data[_HEADER.size :]
    if len(body) != size:
        raise ValueError("truncated frame body")
    return body


def _read_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        part = conn.recv(remaining)
        if not part:
            raise ConnectionError("socket closed during frame read")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def read_frame(conn: socket.socket) -> bytes:
    """Read one length-prefixed frame from ``conn``."""
    header = _read_exact(conn, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    return _read_exact(conn, size)


def write_frame(conn: socket.socket, blob: bytes) -> None:
    """Write one length-prefixed frame."""
    conn.sendall(frame(blob))


def shm_write_read(blob: bytes) -> bytes:
    """Copy ``blob`` through a shared-memory segment and read it back.

    Args:
        blob: Bytes to stage.

    Returns:
        The bytes read from the segment.
    """
    segment = shared_memory.SharedMemory(create=True, size=len(blob))
    try:
        segment.buf[: len(blob)] = blob
        return bytes(segment.buf[: len(blob)])
    finally:
        segment.close()
        segment.unlink()


def stage_pinned(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Copy ``tensor`` into a pinned CPU buffer when the allocator allows.

    Args:
        tensor: Source, any device.

    Returns:
        The staged CPU tensor and whether it is pinned. A CPU-only
        build may return an unpinned tensor; the flag says which.
    """
    cpu = tensor.detach().to(device="cpu").contiguous()
    try:
        pinned = torch.empty(cpu.shape, dtype=cpu.dtype, pin_memory=True)
    except Exception:
        return cpu, False
    if not pinned.is_pinned():
        return cpu, False
    pinned.copy_(cpu, non_blocking=True)
    return pinned, True


def exchange_tcp(blob: bytes, address: str, timeout: float = 5.0) -> bytes:
    """Send one frame and read one frame.

    Args:
        blob: Request bytes.
        address: ``host:port``.
        timeout: Socket timeout in seconds.

    Returns:
        Response payload, without the length header.

    Raises:
        ValueError: ``address`` is empty or has no port.
    """
    if not address or ":" not in address:
        raise ValueError("host-staged TCP needs host:port in VLLM_HETERO_MOE_HOST_ADDR")
    host, _, port_s = address.rpartition(":")
    conn = socket.create_connection((host, int(port_s)), timeout=timeout)
    try:
        write_frame(conn, blob)
        return read_frame(conn)
    finally:
        conn.close()
