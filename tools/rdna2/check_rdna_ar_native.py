# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject a stale or foreign RDNA all-reduce extension before model loading."""

import argparse
import importlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        import vllm

        source = args.source_root.resolve()
        if Path(vllm.__file__).resolve().parent != source / "vllm":
            raise RuntimeError(f"Python vLLM comes from {vllm.__file__}, not {source}")
        extension = importlib.import_module("vllm._rocm_C")
        from vllm.distributed.device_communicators.rdna_all_reduce import (
            native_extension_error,
        )

        error = native_extension_error(source)
        if error:
            raise RuntimeError(error)
    except (ImportError, OSError, RuntimeError) as exc:
        parser.exit(2, f"RDNA all-reduce preflight failed: {exc}\n")
    print(f"RDNA all-reduce native contract verified: {extension.__file__}")
    print("GPU correctness and performance still require multi-rank qualification.")


if __name__ == "__main__":
    main()
