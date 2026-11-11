# Profiling-only shim (MonoKernel P0b). A parent process that imported torch
# leaves ROCPROFILER_REGISTER_LIBRARY in the C environment; children (vLLM
# workers) inherit it, rocprofiler-register then never discovers libkineto,
# and torch.profiler records no GPU kernels. Drop it before torch loads, then
# run the venv sitecustomize unchanged.
import os, sys, importlib.util
os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)
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
