# CPU-only cleanup ordering check

The six recorded tests pass. The harness AST-loads only `run_engine_core`; it does not import torch, initialize ROCm, or exercise real vLLM workers. It uses fake process/handler cases plus short-lived subprocesses delivering real SIGTERM. It checks ROCm cleanup ignores SIGTERM until cleanup completes and restores the default disposition; non-ROCm retains default behavior. The test compares frozen pre-proposal baseline and candidate snapshots.

This supports the signal-ordering hypothesis and tests handler scope with a mock engine. It does **not** establish that real graph-mode workers exit, that live serving is unaffected, or that the cleanup patch fixes the graph qualification failure. A fresh bounded graph-mode fault run remains required.

To reproduce, copy this directory to a temporary location, rename `check_cpu.py.txt` to `check_cpu.py`, `core.py.original.txt` to `core.py.original`, and `core.py.candidate.txt` to `core.py.candidate`; run `/private/tmp/mono-local-validation-venv/bin/python check_cpu.py -v`. The test checks the baseline against the recorded SHA before running.
