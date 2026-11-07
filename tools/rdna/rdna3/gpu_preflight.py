#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G0 of the gfx1100 GPU test plan: is HIP 0 the W7800 we think it is?

Run it with ``HIP_VISIBLE_DEVICES=<n>`` set on the same line (gpu_g0_g7.sh does):

    HIP_VISIBLE_DEVICES=0 python tools/rdna/rdna3/gpu_preflight.py \
        --bus cb --arch gfx1100 [--families]

Default mode (no vllm import): exactly one visible GPU, on the expected PCI bus,
of the expected arch, with a full 48 GB class VRAM, and one fp16 matmul runs.
``--families``: imports the vllm ROCm platform (amdsmi is queried by vllm itself)
and checks the mixed-box arch fallback, the family parser with
``VLLM_RDNA3_HIP_OPS=all`` and the registered ``_rocm_C`` ops. It exits non-zero
on the first failed check and prints one JSON line otherwise.
"""

import argparse
import json
import os
import sys


def fail(message: str) -> None:
    print(f"PREFLIGHT FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def check_device_packages(arch: str) -> None:
    """The ROCm wheels ship device code per arch in separate packages."""
    import importlib.metadata as md

    for dist in (f"amd-torch-device-{arch}", f"rocm-sdk-device-{arch}"):
        try:
            md.version(dist)
        except md.PackageNotFoundError:
            fail(
                f"this venv has no {dist}: its torch carries no {arch} kernels and "
                "the first kernel would fail with hipErrorInvalidImage"
            )


def check_device(bus: str, arch: str, min_gib: float) -> dict:
    import torch

    count = torch.accelerator.device_count()
    if count != 1:
        fail(f"HIP_VISIBLE_DEVICES must expose exactly one GPU, got {count}")
    props = torch.cuda.get_device_properties(0)
    if props.pci_bus_id != int(bus, 16):
        fail(f"visible GPU is on PCI bus {props.pci_bus_id:02x}, expected {bus}")
    found_arch = props.gcnArchName.split(":")[0]
    if found_arch != arch:
        fail(f"visible GPU is {found_arch}, expected {arch}")
    total_gib = props.total_memory / 2**30
    if total_gib < min_gib:
        fail(f"visible GPU has {total_gib:.1f} GiB, expected at least {min_gib}")
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    y = x @ x
    torch.accelerator.synchronize()
    if not bool(torch.isfinite(y).all()):
        fail("fp16 matmul on the device produced non-finite values")
    return {
        "bus": f"{props.pci_bus_id:02x}",
        "arch": found_arch,
        "name": props.name,
        "total_gib": round(total_gib, 2),
        "multi_processor_count": props.multi_processor_count,
    }


def check_families(arch: str) -> dict:
    os.environ["VLLM_RDNA3_HIP_OPS"] = "all"
    import torch

    import vllm._custom_ops  # noqa: F401  (loads _C and _rocm_C)
    import vllm.platforms.rocm as rocm
    from vllm.platforms import rdna

    if not rocm._GCN_ARCH.startswith(arch):
        fail(f"vllm resolved the arch {rocm._GCN_ARCH!r}, expected {arch}")
    if not rocm.on_gfx1100():
        fail("on_gfx1100() is False on the W7800")
    families = rdna._rdna3_hip_families()
    dropped = sorted(rdna.RDNA3_HIP_FAMILIES - families)
    schemas = [str(s) for s in torch._C._jit_get_all_schemas()]
    ops = sorted({s.split("(")[0] for s in schemas if s.startswith("_rocm_C::")})
    for needed in ("_rocm_C::moe_gptq_gemm_rdna3", "_rocm_C::gptq_gemm_rdna3"):
        if needed not in ops:
            fail(f"{needed} is not registered ({len(ops)} _rocm_C ops)")
    return {
        "gcn_arch": rocm._GCN_ARCH,
        "families": sorted(families),
        "dropped_families": dropped,
        "rocm_C_ops": len(ops),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bus", required=True, help="expected PCI bus, hex (cb)")
    ap.add_argument("--arch", required=True, help="expected arch (gfx1100)")
    ap.add_argument("--min-gib", type=float, default=44.0)
    ap.add_argument("--families", action="store_true")
    args = ap.parse_args()
    check_device_packages(args.arch)
    report = check_device(args.bus, args.arch, args.min_gib)
    if args.families:
        report.update(check_families(args.arch))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
