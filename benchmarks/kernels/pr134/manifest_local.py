# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compute the same manifest as benchmarks/kernels/pr134/qualify_serving.py.

Usage: python manifest_local.py [tree_root] > manifest.json
Kept out of the manifest-covered paths on purpose (benchmarks/ is not covered),
so it can be run on either side of the local/remote mirror split.
"""

import hashlib
import json
import sys
from pathlib import Path


def manifest(tree: Path) -> dict:
    paths = [*tree.glob("vllm/*.so"), *tree.glob("tunableop/**/*.csv")]
    for folder in (
        "csrc/rocm",
        "vllm/v1/attention",
        "vllm/config",
        "vllm/v1/core",
        "vllm/v1/worker",
        "tools/rdna",
    ):
        paths += [
            p
            for p in (tree / folder).rglob("*")
            if p.suffix in (".py", ".cu", ".cpp", ".h", ".hpp", ".sh", ".env")
        ]
    paths += [
        tree / "vllm/_custom_ops.py",
        tree / "vllm/utils/torch_utils.py",
        tree / "vllm/v1/kv_cache_interface.py",
        tree / "vllm/model_executor/layers/quantization/kv_cache.py",
    ]
    return {
        str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if p.exists()
    }


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(json.dumps(manifest(root), indent=2, sort_keys=True))
