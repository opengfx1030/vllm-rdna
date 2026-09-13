# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prevent cache/batch changes from silently bypassing qualified V620 tuning."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/rdna2/check_v620_tuning.py"
ROWS = ROOT / "tunableop/rocblas-c27e2252cc7a"


class TuningCoverageTests(unittest.TestCase):
    def check(self, rows, block, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--rows-template",
                str(rows / "tunableop_results.csv"),
                "--block-size",
                str(block),
                "--max-num-batched-tokens",
                "4096",
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def test_qualified_tables_cover_both_cache_grids(self):
        for block in (800, 1024):
            result = self.check(ROWS, block)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_aligned_shape_warns_and_strict_checks_fail(self):
        for block, tokens in ((800, 4000), (1024, 3072)):
            with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as folder:
                target = Path(folder)
                for rank in range(4):
                    source = ROWS / f"tunableop_results{rank}.csv"
                    with source.open() as stream:
                        rows = list(csv.reader(stream))
                    if rank == 2:
                        rows = [
                            row
                            for row in rows
                            if not (
                                row[0] == "GemmTunableOp_Half_TN"
                                and row[1].split("_")[2] == str(tokens)
                            )
                        ]
                    with (target / source.name).open("w") as stream:
                        csv.writer(stream).writerows(rows)
                result = self.check(target, block)
                self.assertEqual(result.returncode, 0)
                self.assertIn("WARNING", result.stderr)
                self.assertIn(f"rank 2: {tokens} tokens", result.stderr)
                strict = self.check(target, block, "--strict")
                self.assertEqual(strict.returncode, 2)

    def test_unknown_library_disables_tuning_without_stopping_service(self):
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder) / "librocblas.so"
            library.write_bytes(b"different library build")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; export PYTORCH_TUNABLEOP_ENABLED=1; '
                    "export PYTORCH_TUNABLEOP_FILENAME=stale.csv; "
                    'configure_v620_tunableop "$2" "$3"; '
                    'printf "%s:%s" "$PYTORCH_TUNABLEOP_ENABLED" '
                    '"${PYTORCH_TUNABLEOP_FILENAME-unset}"',
                    "check-tuning",
                    str(ROOT / "tools/rdna2/tunableop_env.sh"),
                    str(library),
                    str(ROWS.parent),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "0:unset")
            self.assertIn("WARNING", result.stderr)

    def test_last_cli_batch_override_is_checked(self):
        result = self.check(ROWS, 1024, "--max-num-batched-tokens", "5120", "--strict")
        self.assertEqual(result.returncode, 2)
        self.assertIn("5120 tokens", result.stderr)


if __name__ == "__main__":
    unittest.main()
