# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run startup-plan unit tests without installing torch or a GPU runtime.

Loads the real startup-plan source and existing pytest suite, with stand-ins
only for its import-time dependencies (platform, config types, envs, torch
version strings, logging). This tests persistence and invalidation, not engine
startup, memory profiling, compilation, or kernels. Run in a fresh process:

    .venv/bin/python tools/rdna2/test_startup_plan_cpu.py

Use --source /path/to/startup_plan.py to reproduce failures on an older version.
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=root / "vllm/v1/worker/startup_plan.py"
    )
    args = parser.parse_args()

    def module(name: str, **attrs) -> ModuleType:
        result = ModuleType(name)
        result.__dict__.update(attrs)
        sys.modules[name] = result
        return result

    for name in ("vllm", "vllm.utils", "vllm.v1", "vllm.v1.worker"):
        module(name, __path__=[])
    sys.modules["vllm"].__version__ = "cpu-contract-test"
    module(
        "torch",
        __version__="cpu-contract-test",
        version=SimpleNamespace(cuda=None, hip=None),
    )
    values = {
        "VLLM_ENABLE_STARTUP_PLAN": lambda: (
            os.getenv("VLLM_ENABLE_STARTUP_PLAN") == "1"
        ),
        "VLLM_CACHE_ROOT": lambda: os.environ["VLLM_CACHE_ROOT"],
    }
    envs = module("vllm.envs")

    def env_value(name):
        if name not in values:
            raise AttributeError(name)
        return values[name]()

    envs.__getattr__ = env_value
    module("vllm.config", VllmConfig=SimpleNamespace)
    module("vllm.logger", init_logger=logging.getLogger)
    module("vllm.platforms", current_platform=None)
    module("vllm.utils.mem_constants", GiB_bytes=1 << 30)

    name = "vllm.v1.worker.startup_plan"
    spec = importlib.util.spec_from_file_location(name, args.source)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    sys.modules[name] = source
    spec.loader.exec_module(source)

    return pytest.main(
        ["--noconftest", "-q", str(root / "tests/v1/worker/test_startup_plan.py")]
    )


if __name__ == "__main__":
    raise SystemExit(main())
