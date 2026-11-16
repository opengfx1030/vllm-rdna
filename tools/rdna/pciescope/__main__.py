# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``python -m pciescope``."""

from __future__ import annotations

import sys

from pciescope.cli import main

if __name__ == "__main__":
    sys.exit(main())
