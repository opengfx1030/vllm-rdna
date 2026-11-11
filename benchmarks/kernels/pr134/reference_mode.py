# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Print the pairing mode of a completed qualification arm.

Modes:
  strict-pass     - raw status PASS, no corrected or fatal kernel events.
  diagnostic-only - raw status FAIL whose only kernel events are corrected-link
                    noise (kernel: "It has been corrected by h/w"); usable as a
                    pairing basis for an explicitly diagnostic-only candidate.
  unusable        - anything else (fatal events, failed cleanup, missing files).
"""

import json
import sys
from pathlib import Path


def mode(reference: Path) -> str:
    status = json.loads((reference / "status.json").read_text())
    qualification = json.loads((reference / "qualification.json").read_text())
    if status.get("cleanup") != "PASS":
        return "unusable"
    if qualification["kernel_events"]["fatal_events"]:
        return "unusable"
    if status.get("status") == "PASS":
        return "strict-pass"
    if qualification["qualification_mode"] == "diagnostic-only":
        return "diagnostic-only"
    return "unusable"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: reference_mode.py <reference-dir>")
    try:
        print(mode(Path(sys.argv[1])))
    except (OSError, KeyError, ValueError) as error:
        print("unusable")
        print(f"reference mode unreadable: {error}", file=sys.stderr)
