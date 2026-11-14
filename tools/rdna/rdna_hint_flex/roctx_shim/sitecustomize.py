# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Profiling-only shim for tools/rdna/rdna_hint_flex/decode_profile.py. With
# RDNA_ROCTX_SHIM=1 every Python process of a server started under
# `rocprofv3 --selected-regions` maps
#   SIGUSR1 -> roctxProfilerResume(0)   (start recording kernels)
#   SIGUSR2 -> roctxProfilerPause(0)    (stop recording)
# so the driver can open a window on a chosen server state (steady decode)
# instead of a fixed delay from process start. Anything else the venv's own
# sitecustomize does still runs (same chaining as
# tools/monokernel-p0-probe/kineto_shim). Outside rocprofv3 the roctx calls are
# no-ops.
import ctypes
import importlib.util
import os
import signal
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if not _p or os.path.abspath(_p) == _here:
        continue
    _f = os.path.join(_p, "sitecustomize.py")
    if os.path.isfile(_f):
        _spec = importlib.util.spec_from_file_location("_venv_sitecustomize", _f)
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        break

if os.environ.get("RDNA_ROCTX_SHIM") == "1":
    # Load the library at start, not inside the first signal handler: a first
    # load there registers with the tool too late and the first window records
    # nothing.
    _lib = ctypes.CDLL(os.environ["ROCTX_LIB"])
    _lib.roctxProfilerPause.argtypes = [ctypes.c_uint32]
    _lib.roctxProfilerResume.argtypes = [ctypes.c_uint32]

    def _resume(_sig, _frame):
        _lib.roctxProfilerResume(0)

    def _pause(_sig, _frame):
        _lib.roctxProfilerPause(0)

    signal.signal(signal.SIGUSR1, _resume)
    signal.signal(signal.SIGUSR2, _pause)
