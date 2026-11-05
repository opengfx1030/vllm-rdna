# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve the visible HIP indices to their PCI bus ids before a launch."""

import ctypes
import os
import sys

_LIB = ctypes.CDLL("libamdhip64.so")
_LIB.hipDeviceGetPCIBusId.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
]
_LIB.hipDeviceGetPCIBusId.restype = ctypes.c_int


def bus_of(index: int) -> str:
    buf = ctypes.create_string_buffer(32)
    rc = _LIB.hipDeviceGetPCIBusId(buf, len(buf), index)
    if rc != 0:
        return f"error{rc}"
    return buf.value.decode()


def main() -> int:
    ids = [i.strip() for i in os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")]
    ids = [i for i in ids if i]
    if not ids:
        ids = [str(i) for i in range(32)]
    parts = [f"HIP {ids[i]} bus {bus_of(i)}" for i in range(len(ids))]
    print(", ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
