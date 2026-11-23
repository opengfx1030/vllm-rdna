# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-equivalence tests for ``vllm.models.inkling2_loop.amd.blocks``.

Every module is compared against the corresponding class of
``transformers.models.inkling2_loop.modeling_inkling2_loop`` on identical seeded
random weights (copied with ``load_state_dict(strict=True)``) and inputs, on CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as ref_mod
from transformers.models.inkling2_loop.configuration_inkling2_loop import (
    Inkling2LoopTextConfig as RefTextConfig,
)

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    """Load a module by file path so the venv's installed ``vllm`` cannot shadow it."""
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


blocks = _load("_i2l_blocks_under_test", "vllm/models/inkling2_loop/amd/blocks.py")
our_configs = _load("_i2l_configs_under_test", "vllm/models/inkling2_loop/configs.py")

HIDDEN = 48
LOOP_START, LOOP_END = 6, 18
NUM_LOOP_LAYERS = LOOP_END - LOOP_START
GATE_RANK = 8
EPS = 1e-6

FP32_TOL = 1e-5
BF16_TOL = 2e-2

DTYPES = [torch.float32, torch.bfloat16]


def _tol(dtype: torch.dtype) -> float:
    return FP32_TOL if dtype == torch.float32 else BF16_TOL


def _ref_config() -> RefTextConfig:
    cfg = RefTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=24,
        loop_config={
            "core_start": LOOP_START,
            "core_end": LOOP_END,
            "num_passes": 3,
            "gate_rank": GATE_RANK,
        },
        rms_norm_eps=EPS,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        log_scaling_n_floor=8192,
        log_scaling_alpha=0.1,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _our_config():
    return our_configs.Inkling2LoopTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=24,
        loop_config={
            "core_start": LOOP_START,
            "core_end": LOOP_END,
            "num_passes": 3,
            "gate_rank": GATE_RANK,
        },
        rms_norm_eps=EPS,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        log_scaling_n_floor=8192,
        log_scaling_alpha=0.1,
    )


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(
                torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype)
            )


def _randn(shape, seed: int, dtype=torch.float32, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(dtype)


def _sync(ours: torch.nn.Module, ref: torch.nn.Module) -> None:
    """Same keys, shapes and dtypes, then a strict state-dict copy."""
    ref_sd, our_sd = ref.state_dict(), ours.state_dict()
    assert list(our_sd) == list(ref_sd)
    for k in ref_sd:
        assert our_sd[k].shape == ref_sd[k].shape, k
        assert our_sd[k].dtype == ref_sd[k].dtype, k
    ours.load_state_dict(ref_sd, strict=True)


def _max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    return (a.float() - b.float()).abs().max().item()


# --------------------------------------------------------------------------- norms


@pytest.mark.parametrize("dtype", DTYPES)
def test_unweighted_rms_norm(dtype):
    ref = ref_mod.Inkling2LoopUnweightedRMSNorm(eps=1e-5)
    ours = blocks.UnweightedRMSNorm(eps=1e-5)
    _sync(ours, ref)
    x = _randn((2, 5, HIDDEN), 0, dtype, scale=3.0)
    assert _max_err(ours(x), ref(x)) <= _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm(dtype):
    ref = ref_mod.Inkling2LoopRMSNorm(HIDDEN, eps=EPS).to(dtype)
    _randomize_(ref, 1)
    ours = blocks.Inkling2RMSNorm(HIDDEN, eps=EPS, dtype=dtype)
    _sync(ours, ref)
    x = _randn((2, 5, HIDDEN), 2, dtype, scale=3.0)
    assert _max_err(ours(x), ref(x)) <= _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_normed_embedding(dtype):
    vocab, pad = 100, 3
    ref = ref_mod.Inkling2LoopNormedEmbedding(vocab, HIDDEN, pad, EPS).to(dtype)
    _randomize_(ref, 3)
    ours = blocks.NormedEmbedding(vocab, HIDDEN, pad, EPS, dtype=dtype)
    _sync(ours, ref)
    ids = torch.randint(0, vocab, (3, 7), generator=torch.Generator().manual_seed(4))
    ids[0, 0] = pad
    assert _max_err(ours(ids), ref(ids)) <= _tol(dtype)


# ------------------------------------------------------------------- message gate


@pytest.mark.parametrize("dtype", DTYPES)
def test_message_gate(dtype):
    ref = ref_mod.Inkling2LoopMessageGate(_ref_config())
    ref.down_proj.to(dtype)  # fp32 gate projections stay fp32, as in the reference
    _randomize_(ref, 5)
    ours = blocks.MessageGate(_our_config(), dtype=dtype)
    _sync(ours, ref)
    assert ours.down_proj.weight.dtype == dtype
    assert ours.forget_gate_proj.weight.dtype == torch.float32
    assert ours.message_gate_proj.weight.dtype == torch.float32

    hidden = _randn((2, 6, HIDDEN), 6, dtype, scale=2.0)
    message = _randn((2, 6, HIDDEN), 7, dtype, scale=0.5)
    out_ref, out_ours = ref(hidden, message), ours(hidden, message)
    assert out_ours.dtype == torch.float32
    assert _max_err(out_ours, out_ref) <= _tol(dtype)

    # leading dims are opaque: flat (tokens, H) gives the same rows
    flat = ours(hidden.reshape(-1, HIDDEN), message.reshape(-1, HIDDEN))
    assert _max_err(flat, out_ref.reshape(-1, HIDDEN)) <= _tol(dtype)


# ------------------------------------------------------- attention residual blocks


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_queries", [1, 3])
@pytest.mark.parametrize("num_blocks", list(range(8)))
def test_block_attention_residual(num_blocks, num_queries, dtype):
    ref = ref_mod.Inkling2LoopBlockAttentionResidual(
        _ref_config(), num_queries=num_queries
    )
    _randomize_(ref, 10 + num_queries)
    ours = blocks.BlockAttentionResidual(_our_config(), num_queries=num_queries)
    _sync(ours, ref)
    assert ours.query.dtype == torch.float32

    B, S = 2, 5
    block_states = _randn((B, S, num_blocks, HIDDEN), 20 + num_blocks, dtype, scale=2.0)
    partial = _randn((B, S, HIDDEN), 30 + num_blocks, dtype, scale=2.0)
    out_ref = ref(block_states, partial)
    out_ours = ours(block_states, partial)
    assert isinstance(out_ours, tuple) and len(out_ours) == num_queries == len(out_ref)
    for o, r in zip(out_ours, out_ref):
        assert o.shape == (B, S, HIDDEN)
        assert _max_err(o, r) <= _tol(dtype)

    # flat-token layout (T, n, H): same rows as the batched reference
    flat = ours(
        block_states.reshape(B * S, num_blocks, HIDDEN), partial.reshape(B * S, HIDDEN)
    )
    for o, r in zip(flat, out_ref):
        assert _max_err(o, r.reshape(B * S, HIDDEN)) <= _tol(dtype)


def test_block_attention_residual_single_state_is_identity_of_normed_partial():
    """With no stacked blocks the softmax is over one entry: output = rms-normed
    partial."""
    ours = blocks.BlockAttentionResidual(_our_config())
    _randomize_(ours, 40)
    partial = _randn((2, 4, HIDDEN), 41)
    empty = torch.empty(2, 4, 0, HIDDEN)
    (out,) = ours(empty, partial)
    expected = blocks.UnweightedRMSNorm()(partial)
    assert (out - expected).abs().max().item() <= FP32_TOL


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("loop_layer_idx", list(range(NUM_LOOP_LAYERS)))
def test_message_attention_residual(loop_layer_idx, dtype):
    ref = ref_mod.Inkling2LoopMessageAttentionResidual(_ref_config())
    _randomize_(ref, 50)
    ours = blocks.MessageAttentionResidual(_our_config())
    _sync(ours, ref)
    assert ours.query.shape == (NUM_LOOP_LAYERS, 4, HIDDEN)
    assert ours.query.dtype == torch.float32

    B, S, n = 2, 5, 1 + loop_layer_idx % 4
    block_states = _randn((B, S, n, HIDDEN), 60 + loop_layer_idx, dtype, scale=2.0)
    partial = _randn((B, S, HIDDEN), 70 + loop_layer_idx, dtype, scale=2.0)
    out_ref = ref(block_states, partial, loop_layer_idx)
    out_ours = ours(block_states, partial, loop_layer_idx)
    assert out_ours.shape == (B, S, 4, HIDDEN)
    assert _max_err(out_ours, out_ref) <= _tol(dtype)


# ------------------------------------------------------------- activations / clips


@pytest.mark.parametrize("dtype", DTYPES)
def test_softcap(dtype):
    ref = ref_mod.Inkling2LoopSoftcap(4.0)
    ours = blocks.Softcap(4.0)
    _sync(ours, ref)
    x = _randn((3, 17), 80, dtype, scale=20.0)
    assert _max_err(ours(x), ref(x)) <= _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_softcapped_swiglu(dtype):
    ref = ref_mod.Inkling2LoopSoftcappedSwiGLU(_ref_config())
    ours = blocks.SoftcappedSwiGLU(_our_config())
    _sync(ours, ref)
    gate = _randn((4, 33), 81, dtype, scale=15.0)
    up = _randn((4, 33), 82, dtype, scale=60.0)
    out_ref, out_ours = ref(gate, up), ours(gate, up)
    assert out_ours.dtype == dtype
    assert _max_err(out_ours, out_ref) <= _tol(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_clip(dtype):
    ref = ref_mod.Inkling2LoopL2Clip(8.0)
    ours = blocks.L2Clip(8.0)
    _sync(ours, ref)
    # mix of vectors below and above the threshold
    x = _randn((2, 6, 4, 16), 83, dtype, scale=1.0)
    x[:, ::2] = x[:, ::2] * 10
    out_ref = ref(x)
    assert _max_err(ours(x), out_ref) <= _tol(dtype)
    norms = torch.linalg.vector_norm(out_ref.float(), dim=-1)
    assert norms.max() <= 8.0 * (1 + (1e-2 if dtype == torch.bfloat16 else 1e-5))


@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_l2_clip(dtype):
    ref = ref_mod.Inkling2LoopScaledL2Clip(4, 16, 8.0)
    _randomize_(ref, 84)
    ours = blocks.ScaledL2Clip(4, 16, 8.0)
    _sync(ours, ref)
    x = _randn((2, 6, 4, 16), 85, dtype, scale=3.0)
    pos = torch.arange(6)[None].expand(2, -1)
    assert _max_err(ours(x), ref(x)) <= _tol(dtype)
    assert _max_err(ours(x, pos), ref(x, pos)) <= _tol(dtype)


POSITIONS = [0, 1, 100, 8190, 8191, 8192, 8193, 10_000, 65_535, 100_000, 1_048_575]


@pytest.mark.parametrize("dtype", DTYPES)
def test_log_scale_large_positions(dtype):
    ref = ref_mod.Inkling2LoopLogScale(_ref_config())
    ours = blocks.LogScale(_our_config())
    _sync(ours, ref)
    P = len(POSITIONS)
    x = _randn((2, P, 4, 16), 86, dtype)
    pos = torch.tensor(POSITIONS)[None].expand(2, -1).contiguous()
    out_ref, out_ours = ref(x, pos), ours(x, pos)
    assert _max_err(out_ours, out_ref) <= _tol(dtype)
    # positions below the floor are untouched, above it the scale grows
    below = torch.tensor([i for i, p in enumerate(POSITIONS) if p + 1 <= 8192])
    assert _max_err(out_ours[:, below], x[:, below]) <= 1e-6
    assert (out_ours[:, -1].float().abs() > x[:, -1].float().abs() * 1.1).any()


@pytest.mark.parametrize("dtype", DTYPES)
def test_log_scaled_l2_clip(dtype):
    ref = ref_mod.Inkling2LoopLogScaledL2Clip(_ref_config(), 4, 16, 8.0)
    _randomize_(ref, 87)
    ours = blocks.LogScaledL2Clip(_our_config(), 4, 16, 8.0)
    _sync(ours, ref)
    assert list(ours.state_dict()) == ["scale"]
    P = len(POSITIONS)
    x = _randn((2, P, 4, 16), 88, dtype, scale=3.0)
    pos = torch.tensor(POSITIONS)[None].expand(2, -1).contiguous()
    assert _max_err(ours(x, pos), ref(x, pos)) <= _tol(dtype)


# ----------------------------------------------------------------- relative distance


@pytest.mark.parametrize("q_len", [1, 5, 16])
def test_relative_distance_matches_reference_without_cache(q_len):
    embeds = torch.zeros(1, q_len, HIDDEN)
    for is_sliding in (False, True):
        ref = ref_mod.get_relative_distance(embeds, None, is_sliding)
        pos = torch.arange(q_len)
        assert torch.equal(blocks.relative_distance(pos, pos), ref)


def test_relative_distance_with_offsets():
    q = torch.arange(3) + 10
    kv = torch.arange(13)
    out = blocks.relative_distance(q, kv)
    assert out.shape == (3, 13)
    assert out[0, 10].item() == 0 and out[2, 0].item() == 12 and out[0, 12].item() == -2
