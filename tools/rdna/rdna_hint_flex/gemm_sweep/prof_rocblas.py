# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

"""Which Tensile kernel does rocBLAS run for one (shape, M) cell?

  python3 prof_rocblas.py <heur|tuned> <shape> <M> [rows.csv]

Prints one markdown row per kernel name the torch profiler saw in five calls of
``torch.mm(x, w.t())``. ``heur`` leaves TunableOp off (what serving does when its
CSV has no row for the shape); ``tuned`` reads the rows ``sweep.py run`` saved.
(rocprofv3 from the system ROCm cannot attach to the venv's runtime on this box:
"api registration failed with error code 16"; the torch profiler uses the
runtime's own tracer.)
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(mode: str, shape: str, m: int, rows: str | None) -> None:
    import torch  # noqa: PLC0415
    from sweep import SHAPES  # noqa: PLC0415
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    n, k = SHAPES[shape]
    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
    w = torch.randn(n, k, dtype=torch.float16, device="cuda")
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")
    tun = torch.cuda.tunable
    if mode == "tuned":
        tun.set_filename(rows, insert_device_ordinal=False)
        tun.read_file(rows)
        tun.enable(True)
        tun.tuning_enable(False)
    else:
        tun.enable(False)
    for _ in range(2):
        torch.mm(x, w.t(), out=out)
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            torch.mm(x, w.t(), out=out)
        torch.accelerator.synchronize()
    by: dict[str, list[float]] = {}
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = getattr(e, "device_time", None) or getattr(e, "cuda_time", 0.0)
        by.setdefault(e.name, []).append(us)
    for name, ts in sorted(by.items()):
        print(
            f"| {mode} | {shape} | {m} | `{name}` | {len(ts)} | {statistics.median(ts):.1f} |"
        )


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4] if len(sys.argv) > 4 else None,
    )
