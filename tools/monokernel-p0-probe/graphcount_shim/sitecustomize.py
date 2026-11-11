# Measurement-only shim (MonoKernel P0b): log every captured hipGraph's kernel
# nodes, so the launch count of a decode step comes from the graphs the server
# actually replays (no profiler, no queue interception). Put this directory
# first on PYTHONPATH; the venv sitecustomize still runs.
#
# For each torch.cuda.CUDAGraph capture it appends one JSON line to
# $GRAPHCOUNT_DIR/graphs-<pid>.jsonl: {"seq", "nodes", "kernels", "names": {...}}.
# keep_graph=True is forced so the raw graph survives capture_end.
import importlib.util
import os
import sys

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


def _install():
    import collections
    import ctypes
    import json

    import torch

    if not getattr(torch.version, "hip", None):
        return
    out_dir = os.environ.get("GRAPHCOUNT_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    hip = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    hip.hipGraphGetNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    hip.hipGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    hip.hipGraphKernelNodeGetParams.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    hip.hipKernelNameRef.argtypes = [ctypes.c_void_p]
    hip.hipKernelNameRef.restype = ctypes.c_char_p
    hip.hipKernelNameRefByPtr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    hip.hipKernelNameRefByPtr.restype = ctypes.c_char_p

    def kname(func):
        # hipLaunchKernel nodes hold the host stub, module launches (Triton,
        # Tensile) a hipFunction_t; the by-pointer lookup is safe on both.
        nm = hip.hipKernelNameRefByPtr(func, None)
        if not nm:
            nm = hip.hipKernelNameRef(func)
        s = (nm or b"?").decode(errors="replace")
        return s if s.isprintable() else "?"

    class Dim3(ctypes.Structure):
        _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

    class KernelNodeParams(ctypes.Structure):
        _fields_ = [("blockDim", Dim3), ("extra", ctypes.c_void_p), ("func", ctypes.c_void_p),
                    ("gridDim", Dim3), ("kernelParams", ctypes.c_void_p),
                    ("sharedMemBytes", ctypes.c_uint)]

    def _frames():
        f = sys._getframe(2)
        while f is not None:
            yield f
            f = f.f_back

    seq = [0]
    path = os.path.join(out_dir, f"graphs-{os.getpid()}.jsonl")
    cls = torch.cuda.CUDAGraph
    base_init = torch._C._CUDAGraph.__init__
    orig_end = cls.capture_end

    def init(self, keep_graph=False):  # pybind11 takes keep_graph in __init__
        base_init(self, True)

    def capture_end(self):
        orig_end(self)
        try:
            g = ctypes.c_void_p(self.raw_cuda_graph())
            n = ctypes.c_size_t(0)
            hip.hipGraphGetNodes(g, None, ctypes.byref(n))
            arr = (ctypes.c_void_p * n.value)()
            hip.hipGraphGetNodes(g, arr, ctypes.byref(n))
            types, names = collections.Counter(), collections.Counter()
            for node in arr:
                t = ctypes.c_int(-1)
                hip.hipGraphNodeGetType(node, ctypes.byref(t))
                types[t.value] += 1
                if t.value == 0:  # hipGraphNodeTypeKernel
                    p = KernelNodeParams()
                    if hip.hipGraphKernelNodeGetParams(node, ctypes.byref(p)) == 0 and p.func:
                        names[kname(p.func)] += 1
            seq[0] += 1
            ctx = {}
            try:
                from vllm.forward_context import get_forward_context
                fc = get_forward_context()
                ctx = {"mode": str(getattr(fc, "cudagraph_runtime_mode", "")),
                       "batch": str(getattr(fc, "batch_descriptor", ""))}
            except Exception as e:
                ctx = {"ctx_error": repr(e)[:80]}
            caller = [f.f_code.co_name for f in _frames()][:12]
            rec = {"seq": seq[0], "nodes": n.value, "types": dict(types),
                   "kernels": types.get(0, 0), **ctx, "caller": caller,
                   "names": dict(names)}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # never break capture
            with open(path, "a") as f:
                f.write(json.dumps({"seq": -1, "error": repr(e)}) + "\n")

    cls.__init__ = init
    cls.capture_end = capture_end


try:
    _install()
except Exception:
    pass
