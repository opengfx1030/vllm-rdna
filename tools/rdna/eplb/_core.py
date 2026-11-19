# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared pure-numpy policy, loaded by file path.

``vllm/model_executor/layers/fused_moe/rdna_eplb_policy.py`` is the one copy of the
geometry, cost model, placement primitives and swap policy (the online planner runs
it in a helper process). It imports only numpy, but ``import vllm...`` would pull in
torch through the package ``__init__`` files, and these tools run on CPU boxes
without it (``cpu_test_requirements.txt``). So the module is loaded by path.
"""

import importlib.util
import os
import sys
from types import ModuleType

_NAME = "rdna_eplb_policy"
_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    *([os.pardir] * 3),
    "vllm",
    "model_executor",
    "layers",
    "fused_moe",
    "rdna_eplb_policy.py",
)


def _load() -> ModuleType:
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    spec = importlib.util.spec_from_file_location(_NAME, os.path.normpath(_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


core = _load()
