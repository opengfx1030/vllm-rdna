# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off LSA guard. See ``docs/explore/lsa/README.md``.

``VLLM_LSA`` is read from the server process environment only. Request
bodies, extra fields, and gateways do not carry it.
"""

from vllm.model_executor.layers.lsa.guard import (
    LSAConfigError,
    check_layer_types,
    lsa_mode,
    reset_lsa_state_for_tests,
)

__all__ = [
    "LSAConfigError",
    "check_layer_types",
    "lsa_mode",
    "reset_lsa_state_for_tests",
]
