# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blackwell-only Triton launch tuning must not be applied to a ROCm gfx1030
(see conftest.py)."""

import importlib

import pytest
import torch

# -- vllm/v1/attention/ops/triton_unified_attention.py: tuned_large_head -----


def _launch_unified_attention(monkeypatch) -> dict:
    """Call unified_attention on CPU tensors with the Triton launch replaced by
    a spy, and return the keyword arguments the kernel would have got."""
    tua = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    launched: dict = {}

    class _Launcher:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launched.update(kwargs, grid=grid)

            return launch

    monkeypatch.setattr(tua, "kernel_unified_attention", _Launcher())
    head_size, num_tokens, block_size = 256, 64, 16
    num_q_heads, num_kv_heads = 16, 4
    q = torch.zeros(num_tokens, num_q_heads, head_size, dtype=torch.float16)
    kv = torch.zeros(4, block_size, num_kv_heads, head_size, dtype=torch.float16)
    tua.unified_attention(
        q=q,
        k=kv,
        v=kv,
        out=torch.zeros_like(q),
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32),
        max_seqlen_q=num_tokens,
        seqused_k=torch.tensor([num_tokens], dtype=torch.int32),
        max_seqlen_k=num_tokens,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    assert launched, "the kernel was not launched"
    return launched


def test_unified_attention_large_head_keeps_decode_tiling_on_gfx1030(
    gfx1030, monkeypatch
):
    # BLOCK_M=32 (BLOCK_Q=8 at 4 queries per KV head) is the B200 retune for
    # head_size 256; the gfx10x branch calls a wider BLOCK_M "uniformly worse".
    launched = _launch_unified_attention(monkeypatch)
    assert launched["BLOCK_M"] == 16
    assert launched["BLOCK_Q"] == 4
    assert launched["TILE_SIZE"] == 16
    assert launched["num_warps"] == 4
    assert launched["num_stages"] == 1


def test_unified_attention_large_head_retune_still_applies_on_sm100(sm100, monkeypatch):
    launched = _launch_unified_attention(monkeypatch)
    assert launched["BLOCK_M"] == 32
    assert launched["BLOCK_Q"] == 8
    assert launched["TILE_SIZE"] == 128
    assert launched["num_warps"] == 8
    assert launched["num_stages"] == 2


# -- vllm/model_executor/layers/mamba/ops/replayssm_config.py ----------------


@pytest.mark.parametrize(
    ("cuda", "capability", "expected_config"),
    [
        pytest.param(False, (10, 3), (16, 1, 64, 128, 2), id="gfx1030"),
        pytest.param(True, (10, 0), (64, 1, 32, 64, 2), id="sm100"),
    ],
)
def test_replayssm_blackwell_launch_config_is_cuda_only(
    patch_platform, cuda, capability, expected_config
):
    patch_platform(cuda, capability)
    from vllm.model_executor.layers.mamba.ops import replayssm_config

    replayssm_config._is_blackwell.cache_clear()
    try:
        config = replayssm_config.get_replayssm_config("mamba2_output_only", dstate=128)
    finally:
        # Do not leak the answer for this fake platform into other tests.
        replayssm_config._is_blackwell.cache_clear()
    assert config == expected_config
