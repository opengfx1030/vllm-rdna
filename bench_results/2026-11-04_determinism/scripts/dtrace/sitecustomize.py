# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Determinism tracer shim (out of tree). Active only with DTRACE_DIR set.
# Installs a post-import hook that instruments the V2 GPU model runner, then
# chain-loads the venv's own sitecustomize (which this file shadows).
import os
import sys

if os.environ.get("DTRACE_DIR"):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dtrace_hooks  # noqa: E402

        dtrace_hooks.install_import_hook()
    except Exception as e:  # never break the process
        print(f"[dtrace] import hook failed: {e!r}", file=sys.stderr)

_VENV_SC = os.environ.get(
    "DTRACE_CHAIN_SITECUSTOMIZE",
    "/home/chenco_adm/Apps/vllm/venv-7.14.0_0.32.0/lib64/python3.12/site-packages/sitecustomize.py",
)
if _VENV_SC and os.path.exists(_VENV_SC):
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("_venv_sitecustomize", _VENV_SC)
    if _spec is not None and _spec.loader is not None:
        _mod = _ilu.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_mod)
        except Exception as e:
            print(f"[dtrace] chained sitecustomize failed: {e!r}", file=sys.stderr)
