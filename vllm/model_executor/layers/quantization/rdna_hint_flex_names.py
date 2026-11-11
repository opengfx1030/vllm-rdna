# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``quant_method`` name of rdna_hint_flex checkpoints, in one place.

``RDNA_HINT_FLEX_QUANT_METHOD`` is the canonical name. Earlier names stay in
``RDNA_HINT_FLEX_QUANT_METHOD_ALIASES`` so checkpoints written with them keep
loading (``hadamard_int2``: the name of the first quantizer releases).
No imports here: the quantization registry imports this module at startup.
"""

RDNA_HINT_FLEX_QUANT_METHOD = "rdna_hint_flex"
RDNA_HINT_FLEX_QUANT_METHOD_ALIASES: tuple[str, ...] = ("hadamard_int2",)
RDNA_HINT_FLEX_QUANT_METHOD_NAMES: tuple[str, ...] = (
    RDNA_HINT_FLEX_QUANT_METHOD,
    *RDNA_HINT_FLEX_QUANT_METHOD_ALIASES,
)
