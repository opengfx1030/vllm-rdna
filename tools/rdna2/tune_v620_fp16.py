# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt the existing V620 dense probe to FP16 and this SDK's rocBLAS.

Run offline with no inference process. Rows are namespaced by the loaded library
hash; serving only looks up measured rows and never tunes on a request.
"""

import argparse
from pathlib import Path

DENSE_SHAPES = (
    (336, 10240),
    (10240, 320),
    (4096, 2560),
    (2560, 1536),
    (512, 2560),
    (10240, 2560),
    (3584, 2560),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--batch-tokens", nargs="+", type=int, default=[1024, 2048, 4096]
    )
    args = parser.parse_args()
    if any(n <= 0 for n in args.batch_tokens):
        parser.error("Batch token counts must be positive")
    import csv
    import hashlib
    import json
    import os

    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    # Environment overrides the API and is cached on first use. Serving's explicit
    # zeros must not silently disable this offline tuning process.
    for name in (
        "PYTORCH_TUNABLEOP_ENABLED",
        "PYTORCH_TUNABLEOP_TUNING",
        "PYTORCH_TUNABLEOP_FILENAME",
    ):
        os.environ.pop(name, None)
    import torch
    import torch.cuda.tunable as tunable

    torch.accelerator.set_device_index(0)
    torch.manual_seed(620)
    torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")
    libraries = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/librocblas.so" in line
    }
    assert len(libraries) == 1, libraries
    library = libraries.pop()
    with library.open("rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    root = args.output_root
    out = root / ("rocblas-" + digest[:12])
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "tunableop_results0.csv").exists(), "Do not overwrite prior rows"
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                library=str(library),
                sha256=digest,
                torch=torch.__version__,
                hip=torch.version.hip,
                dtype="float16",
                seed=620,
                batch_tokens=args.batch_tokens,
                source="Adapted existing v620-probe-dense-tunable.py",
            )
        )
    )
    tunable.set_filename(
        str(out / "tunableop_results0.csv"), insert_device_ordinal=False
    )
    tunable.set_max_tuning_duration(10)
    tunable.set_max_tuning_iterations(10)
    tunable.set_numerical_check_tolerances(True, 0.01, 0.01)

    def save_rows():
        rows = tunable.get_results()
        assert rows, "Tuning produced no solver rows"
        with (out / "tunableop_results0.csv").open("w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(("Validator", *v) for v in tunable.get_validators())
            writer.writerows(rows)

    def latency(a, w):
        for _ in range(3):
            torch.nn.functional.linear(a, w)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(10):
            torch.nn.functional.linear(a, w)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 10

    for m in args.batch_tokens:
        for n, k in DENSE_SHAPES:
            a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
            w = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.1
            tunable.enable(False)
            assert not tunable.is_enabled()
            reference = torch.nn.functional.linear(a, w)
            before = latency(a, w)
            tunable.enable(True)
            tunable.tuning_enable(True)
            assert tunable.is_enabled() and tunable.tuning_is_enabled()
            actual = torch.nn.functional.linear(a, w)
            tunable.tuning_enable(False)
            assert not tunable.tuning_is_enabled()
            after = latency(a, w)
            torch.testing.assert_close(actual, reference, atol=0.01, rtol=0.01)
            independent = torch.nn.functional.linear(a.float(), w.float())
            print(
                json.dumps(
                    dict(
                        m=m,
                        n=n,
                        k=k,
                        before_ms=before,
                        after_ms=after,
                        speedup=before / after,
                        max_abs=(actual.float() - reference.float()).abs().max().item(),
                        fp32_relative_l2=(
                            (actual.float() - independent).norm() / independent.norm()
                        ).item(),
                    )
                ),
                flush=True,
            )
            del a, w, reference, actual, independent
            save_rows()
    save_rows()
    print(
        json.dumps(
            dict(
                validators=tunable.get_validators(),
                results=tunable.get_results(),
                directory=str(out),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
