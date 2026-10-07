# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend the donor's TunableOp row replay with four-GPU numerical checks."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import csv
    import hashlib
    import json
    import os
    import sysconfig

    import regex as re

    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    for name in (
        "PYTORCH_TUNABLEOP_ENABLED",
        "PYTORCH_TUNABLEOP_TUNING",
        "PYTORCH_TUNABLEOP_FILENAME",
    ):
        os.environ.pop(name, None)
    import torch
    import torch.cuda.tunable as tunable

    directory = args.directory
    provenance = json.loads((directory / "provenance.json").read_text())
    library = provenance.get("library")
    if library is None:
        library = (
            Path(sysconfig.get_path("purelib")) / provenance["library_relative_path"]
        )
    with open(library, "rb") as binary:
        assert hashlib.file_digest(binary, "sha256").hexdigest() == provenance["sha256"]
    source = directory / "tunableop_results0.csv"
    rows = [r for r in csv.reader(source.open()) if r[0] == "GemmTunableOp_Half_TN"]
    assert len(rows) >= 21, len(rows)
    tunable.set_filename(
        str(directory / "qualification-results.csv"), insert_device_ordinal=False
    )
    tunable.tuning_enable(False)
    tunable.enable(True)
    assert tunable.is_enabled() and not tunable.tuning_is_enabled()
    assert tunable.read_file(str(source))
    torch.manual_seed(620)
    for device in range(4):
        torch.accelerator.set_device_index(device)
        for op, sig, sol, _ in rows:
            match = re.fullmatch(r"tn_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)", sig)
            assert match, sig
            n, m, k, lda, ldb, ldc = map(int, match.groups())
            assert (lda, ldb, ldc) == (k, k, n)
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
            w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
            actual = torch.nn.functional.linear(x, w)
            expected = torch.nn.functional.linear(x.float(), w.float())
            relative_l2 = float((actual.float() - expected).norm() / expected.norm())
            assert relative_l2 < 0.002, (device, sig, relative_l2)
            torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
            del x, w, actual, expected
        print(json.dumps(dict(device=device, rows_passed=len(rows))), flush=True)
    for rank in (1, 2, 3):
        target = directory / f"tunableop_results{rank}.csv"
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), target
        else:
            target.write_bytes(source.read_bytes())
    (directory / "QUALIFIED").write_text(
        f"Four V620s: all {len(rows)} FP16 TN shapes matched independent FP32 "
        "calculations.\n"
    )


if __name__ == "__main__":
    main()
