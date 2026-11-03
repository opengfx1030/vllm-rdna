# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One toy D-LSA step: KL on indexer projections, frozen attention.

Run from the repo root:

    python3 docs/explore/lsa/example_distill.py

The frozen queries and keys stand in for one attention layer. Only the
index projections and the temperature move. This is not a training recipe
for Qwen or Flash-Next. Those need the plan in ``README.md``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _reference():
    path = Path(__file__).with_name("reference_lsa.py")
    spec = importlib.util.spec_from_file_location("lsa_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ref = _reference()
    history = ref.train_toy_indexer()
    print(f"steps {len(history) - 1}")
    print(f"kl_start {history[0]:.6f}")
    print(f"kl_end {history[-1]:.6f}")
    if not history[-1] < history[0]:
        raise SystemExit("toy KL did not fall")


if __name__ == "__main__":
    main()
