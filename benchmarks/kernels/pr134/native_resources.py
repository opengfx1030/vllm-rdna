# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Query native decode resources and HIP occupancy without launching kernels."""

import argparse
import ctypes
import json
from pathlib import Path

import regex as re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packed-tile-128", type=int, default=64)
    parser.add_argument("--packed-tile-256", type=int, default=32)
    args = parser.parse_args()
    hip = ctypes.CDLL("libamdhip64.so")
    pointer = ctypes.c_void_p
    int_pointer = ctypes.POINTER(ctypes.c_int)
    signatures = {
        "hipInit": [ctypes.c_uint],
        "hipSetDevice": [ctypes.c_int],
        "hipModuleLoad": [ctypes.POINTER(pointer), ctypes.c_char_p],
        "hipModuleGetFunction": [ctypes.POINTER(pointer), pointer, ctypes.c_char_p],
        "hipFuncGetAttribute": [int_pointer, ctypes.c_int, pointer],
        "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor": [
            int_pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_size_t,
        ],
        "hipModuleUnload": [pointer],
    }
    for name, types in signatures.items():
        getattr(hip, name).argtypes = types
        getattr(hip, name).restype = ctypes.c_int

    def checked(name, *values):
        status = getattr(hip, name)(*values)
        if status:
            raise RuntimeError(f"{name} returned HIP error {status}")

    checked("hipInit", 0)
    checked("hipSetDevice", 0)  # Logical zero is physical HIP2 in our launcher.
    module = pointer()
    checked("hipModuleLoad", ctypes.byref(module), str(args.code_object).encode())
    results = []
    names = re.findall(r"\.name:\s+(\S+)", args.metadata.read_text())
    try:
        for name in names:
            if "fa_decode_paged_splitk_kernel" not in name:
                continue
            packed = "IhLb0ELb0ELb1E" in name
            fp16 = "I6__halfLb0ELb0ELb0E" in name
            if not (packed or fp16):
                continue
            dim = 256 if "kernel_256" in name else 128
            block = 256 if dim == 256 else 128
            tile = 32 if dim == 256 else 64
            if packed:
                tile = args.packed_tile_256 if dim == 256 else args.packed_tile_128
            dynamic = dim * 2 + tile * (dim + 8) * 4 + tile * 4 + (dim // 32 + 1) * 4
            function = pointer()
            checked(
                "hipModuleGetFunction", ctypes.byref(function), module, name.encode()
            )
            row = {
                "name": name,
                "dtype": "packed" if packed else "fp16",
                "head_dim": dim,
                "block_threads": block,
                "kv_tile_tokens": tile,
                "dynamic_lds_bytes_at_launch": dynamic,
            }
            for key, attribute in {
                "max_threads": 0,
                "fixed_lds_bytes": 1,
                "local_bytes_per_thread": 3,
                "registers_per_thread": 4,
            }.items():
                value = ctypes.c_int()
                checked("hipFuncGetAttribute", ctypes.byref(value), attribute, function)
                row[key] = value.value
            for key, shared in (
                ("active_blocks_at_actual_lds", dynamic),
                ("active_blocks_if_no_dynamic_lds", 0),
            ):
                value = ctypes.c_int()
                checked(
                    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                    ctypes.byref(value),
                    function,
                    block,
                    shared,
                )
                row[key] = value.value
            results.append(row)
    finally:
        checked("hipModuleUnload", module)
    result = {
        "scope": "HIP resource query only; no kernel launches",
        "kernels": results,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
