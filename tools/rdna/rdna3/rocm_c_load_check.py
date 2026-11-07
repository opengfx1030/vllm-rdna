# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only load check of a ``_rocm_C`` extension (dlopen + registered ops).

``_rocm_C`` registers its ops from static initializers (``TORCH_LIBRARY``), so a
registration whose kernel symbol the build does not define (an op listed outside
the arch block of the source that defines it) shows up only when the library is
loaded: ``vllm/platforms/rocm.py`` swallows the ``ImportError`` and every
``_rocm_C`` op silently disappears. This tool dlopens the library with
``RTLD_NOW`` in a fresh process (no GPU is touched: it never imports vllm, never
calls ``torch.cuda``), then lists the registered ``_rocm_C::`` schemas.

    python tools/rdna/rdna3/rocm_c_load_check.py check <_rocm_C.abi3.so> --out ops.json
    python tools/rdna/rdna3/rocm_c_load_check.py diff base.json new.json

``check`` exits 0 when the library loads, 1 otherwise (the dynamic-loader message,
which names the first undefined symbol, is in the JSON ``error`` field and
``undefined`` lists every undefined non-system symbol ``nm`` reports). Run it with
``HIP_VISIBLE_DEVICES=`` and the ROCm/torch library path of the build
(``tools/rdna/rdna3/build_rocm_c.sh`` does).
"""

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path

_SYSTEM_PREFIXES = (
    "_ZNSt",
    "_ZNKSt",
    "_ZSt",
    "_ZNSs",
    "_ZTVN10__cxxabi",
    "_ZdlPv",
    "_Znwm",
    "_Znam",
    "_ZdaPv",
    "__cxa",
    "__gxx",
    "_Unwind",
    "__hip",
    "hip",
    "__gnu",
    "_ZN3c10",
    "_ZNK3c10",
    "_ZN2at",
    "_ZNK2at",
    "_ZN5torch",
    "_ZNK5torch",
    "_ZN4pybind11",
    "_ZNK4pybind11",
    "_ZN6caffe2",
    "_ZN9__gnu_cxx",
    "_ZNKRSt",
    "_ZNR5torch",
    "_ZTIN3c10",
    "_ZTVN3c10",
    "_ZTVN5torch",
    "_ZTISt",
    "_ZTVSt",
)


def _undefined_project_symbols(so: Path) -> list[str]:
    """Undefined dynamic symbols that look like this project's own functions.

    A heuristic filter on ``nm -D --undefined-only``: names outside libc/libstdc++/
    HIP/torch/c10/pybind11 namespaces. The authoritative answer is the ``dlopen``.
    """
    nm = shutil.which("llvm-nm") or shutil.which("nm")
    if nm is None:
        return []
    out = subprocess.run(
        [nm, "-D", "--undefined-only", str(so)], capture_output=True, text=True
    ).stdout
    names = []
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        if name.startswith(_SYSTEM_PREFIXES) or "@" in name:
            continue
        if name.startswith("_Z"):
            names.append(name)
    return sorted(set(names))


def check(so: Path, out: Path | None) -> int:
    report: dict = {"so": str(so), "size": so.stat().st_size}
    report["undefined"] = _undefined_project_symbols(so)
    try:
        import torch  # loads libtorch / the HIP runtime, no device init
    except Exception as e:  # noqa: BLE001
        report.update(loaded=False, error=f"import torch failed: {e!r}", ops=[])
        return _finish(report, out, 1)
    try:
        ctypes.CDLL(str(so))  # CPython ORs RTLD_NOW into the mode
    except OSError as e:
        report.update(loaded=False, error=str(e), ops=[])
        return _finish(report, out, 1)

    schemas = torch._C._jit_get_all_schemas()
    ops = sorted(s.name for s in schemas if s.name.startswith("_rocm_C::"))
    report.update(loaded=True, error="", ops=ops, op_count=len(ops))
    return _finish(report, out, 0)


def _finish(report: dict, out: Path | None, rc: int) -> int:
    text = json.dumps(report, indent=1, sort_keys=True)
    if out is not None:
        out.write_text(text + "\n")
    print(
        f"{report['so']}: {'LOADS' if report.get('loaded') else 'DOES NOT LOAD'}"
        f" ({report.get('op_count', 0)} _rocm_C ops registered,"
        f" {len(report['undefined'])} undefined project symbols)"
    )
    if not report.get("loaded"):
        print(f"  error: {report.get('error')}")
        for sym in report["undefined"][:20]:
            print(f"  undefined: {sym}")
    return rc


def diff(a: Path, b: Path) -> int:
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    oa, ob = set(ra.get("ops", [])), set(rb.get("ops", []))
    print(f"{a.name}: loaded={ra.get('loaded')} ops={len(oa)}")
    print(f"{b.name}: loaded={rb.get('loaded')} ops={len(ob)}")
    for name in sorted(oa - ob):
        print(f"  only in {a.name}: {name}")
    for name in sorted(ob - oa):
        print(f"  only in {b.name}: {name}")
    return 0 if oa == ob and ra.get("loaded") == rb.get("loaded") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="dlopen a _rocm_C .so and list its ops")
    c.add_argument("so", type=Path)
    c.add_argument("--out", type=Path)
    d = sub.add_parser("diff", help="compare two check reports")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "check":
        return check(args.so, args.out)
    return diff(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
