# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the RDNA2 W4A8 MoE capture gate (no GPU needed)."""

import pytest

from vllm.model_executor.layers.fused_moe.experts import rdna2_w4a16_moe as m


@pytest.mark.parametrize(
    "num_tokens,cap,expected",
    [
        (256, 0, False),  # below W4A8_MOE_MIN_TOKENS even with graphs off
        (511, 0, False),
        (512, 0, True),  # graphs off: min-tokens threshold applies
        (511, 504, False),
        (512, 504, True),  # documented workaround cap
        (512, 512, False),  # default ceiling: 512 is a capture size
        (513, 512, True),
        (1024, 1024, False),  # raised capture size must move the gate
        (1025, 1024, True),
    ],
)
def test_gate_vs_capture_size(num_tokens, cap, expected):
    assert (
        m.w4a8_moe_enabled_for(num_tokens, max_capture_size=cap, capturing=False)
        is expected
    )


def test_gate_never_while_capturing():
    assert not m.w4a8_moe_enabled_for(4096, max_capture_size=512, capturing=True)


def test_gate_reads_live_config(monkeypatch):
    monkeypatch.setattr(m, "_live_max_cudagraph_capture_size", lambda: 1024)
    assert not m.w4a8_moe_enabled_for(1024, capturing=False)
    assert m.w4a8_moe_enabled_for(1025, capturing=False)


def test_live_capture_size_fallback_is_default(monkeypatch):
    import vllm.config

    def boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", boom)
    assert m._live_max_cudagraph_capture_size() == 512
