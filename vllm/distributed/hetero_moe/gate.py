# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve-side master switch for hetero MoE.

The MoE layer inlines the same environment read so a disabled server
never imports the rest of this package. Keep ``ENV_FLAG`` in sync with
that check.
"""

import os
from collections.abc import Mapping

ENV_FLAG = "VLLM_HETERO_MOE"


def hetero_moe_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the server asked for hetero MoE.

    Args:
        environ: Mapping to read. Defaults to the process environment.

    Returns:
        True only when the flag is the string ``1``.
    """
    env = os.environ if environ is None else environ
    return env.get(ENV_FLAG, "0") == "1"
