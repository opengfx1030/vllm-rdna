# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate of the switches of the Inkling-2 prefill expert GEMMs (moe.py).

* ``VLLM_INKLING2_PREFILL_BM`` is ON by default and ``=0`` restores the decode-shaped
  kernels (A/B and rollback knob);
* ``VLLM_INKLING2_PREFILL_INT8`` is OFF by default and only means something while the
  dot2 path is on;
* ``VLLM_INKLING2_PREFILL_BM_MIN_TOKENS`` is the token gate;
* a layer whose K does not fit the kernels (TP shards the intermediate size to
  576 / 288) is not eligible and keeps the decode-shaped kernels.

    pytest tests/models/inkling2_loop/test_moe_prefill_flags.py -q
"""

from __future__ import annotations

import types

import pytest

from vllm.models.inkling2_loop.amd import moe as moe_mod

BM, INT8, MIN_TOKENS = (
    "VLLM_INKLING2_PREFILL_BM",
    "VLLM_INKLING2_PREFILL_INT8",
    "VLLM_INKLING2_PREFILL_BM_MIN_TOKENS",
)


def test_defaults_are_dot2_on_w4a8_off():
    bm, int8, gate = moe_mod._prefill_switches({})
    assert bm is True and int8 is False
    assert gate >= 1


@pytest.mark.parametrize(
    "env,want",
    [
        ({BM: "0"}, (False, False)),  # rollback: decode-shaped kernels everywhere
        ({BM: "1"}, (True, False)),
        ({INT8: "1"}, (True, True)),  # W4A8 opt-in on top of the default dot2 path
        ({BM: "1", INT8: "1"}, (True, True)),
        ({BM: "0", INT8: "1"}, (False, False)),  # W4A8 needs the dot2 prefill path
        ({BM: "1", INT8: "0"}, (True, False)),
    ],
)
def test_switches(env, want):
    bm, int8, _ = moe_mod._prefill_switches(env)
    assert (bm, int8) == want


def test_token_gate_override():
    default = moe_mod._prefill_switches({})[2]
    assert moe_mod._prefill_switches({MIN_TOKENS: "33"})[2] == 33
    assert moe_mod._prefill_switches({MIN_TOKENS: str(default)})[2] == default


@pytest.mark.parametrize(
    "fmt,hidden,inter,want",
    [
        ("nvfp4", 2304, 1152, True),  # real shapes, TP = 1
        ("nvfp4", 2304, 576, True),  # TP = 2: K of the down projection 576 (32-aligned)
        ("nvfp4", 2304, 288, True),  # TP = 4: K of the down projection 288 (32-aligned)
        ("nvfp4", 256, 128, True),  # tiny harness
        ("nvfp4", 288, 128, True),
        ("nvfp4", 2304, 48, False),  # not a multiple of 32
        ("mxfp8", 2304, 1152, True),
        ("mxfp8", 2304, 576, True),
        ("mxfp8", 2304, 288, True),
        ("mxfp8", 2304, 48, False),
    ],
)
def test_prefill_supported_alignment(fmt, hidden, inter, want):
    layer = types.SimpleNamespace(
        quant_format=fmt, hidden_dim=hidden, intermediate_dim=inter
    )
    assert moe_mod.Inkling2LoopMoE._prefill_supported(layer) is want
