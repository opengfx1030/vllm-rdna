# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference-equivalence tests for ``vllm.models.inkling2_loop.amd.blocks``.

Each building block (flat ``[T, ...]`` tokens) is compared with the real class
from ``transformers.models.inkling2_loop.modeling_inkling2_loop`` run at
``B=1, S=T`` on a tiny config, with weights copied by name from seeded random
values.  ``test_*_detects_perturbation`` proves the comparisons can fail.
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

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    """Load by file path so an installed ``vllm`` cannot shadow the worktree."""
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


blocks = _load("_i2l_blocks_flat_under_test", "vllm/models/inkling2_loop/amd/blocks.py")
our_configs = _load(
    "_i2l_configs_flat_under_test", "vllm/models/inkling2_loop/configs.py"
)

HIDDEN = 128
HEADS, HEAD_DIM = 4, 32
GATE_RANK = 16
N_FLOOR = 16
ALPHA = 0.1
ROUTE_SCALE = 2.5
NUM_EXPERTS, TOP_K = 8, 2
LOOP_START, LOOP_END = 2, 4
T = 24
DTYPES = [torch.float32, torch.float16]


def _ref_config() -> RefTextConfig:
    cfg = RefTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=HEADS,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=2,
        swa_head_dim=HEAD_DIM,
        d_rel=8,
        rel_extent=16,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=64,
        route_scale=ROUTE_SCALE,
        n_layers_per_attn_res_block=2,
        log_scaling_n_floor=N_FLOOR,
        log_scaling_alpha=ALPHA,
        loop_config={
            "core_start": LOOP_START,
            "core_end": LOOP_END,
            "num_passes": 2,
            "gate_rank": GATE_RANK,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def _randn(*shape, seed: int, dtype=torch.float32, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(dtype)


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    """Seeded random values for every floating parameter (queries are zero-init)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                v = torch.randn(p.shape, generator=gen) * 0.3
                p.copy_(v.to(p.dtype))


def _copy_by_name(ours: torch.nn.Module, ref: torch.nn.Module) -> None:
    """Identical parameter names, shapes and dtypes, then a strict state copy."""
    ref_sd, our_sd = ref.state_dict(), ours.state_dict()
    assert set(ref_sd) == set(our_sd), (sorted(ref_sd), sorted(our_sd))
    for k, v in ref_sd.items():
        assert our_sd[k].shape == v.shape, k
        assert our_sd[k].dtype == v.dtype, (k, our_sd[k].dtype, v.dtype)
    ours.load_state_dict(ref_sd, strict=True)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _check(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(a.float(), b.float(), rtol=1e-5, atol=1e-5)
    else:
        assert _rel_err(a, b) <= 2e-3, _rel_err(a, b)


def _differs(a: torch.Tensor, b: torch.Tensor) -> bool:
    return not torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------------- norms


@pytest.mark.parametrize("dtype", DTYPES)
def test_unweighted_rms_norm(dtype):
    ref = ref_mod.Inkling2LoopUnweightedRMSNorm(eps=1e-5)
    ours = blocks.UnweightedRMSNorm(eps=1e-5)
    x = _randn(T, HIDDEN, seed=1, dtype=dtype, scale=3.0)
    out = ours(x)
    assert out.dtype == dtype
    _check(out, ref(x[None])[0], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inkling2_rms_norm(dtype):
    ref = ref_mod.Inkling2LoopRMSNorm(HIDDEN, eps=1e-6).to(dtype)
    _randomize_(ref, 2)
    ours = blocks.Inkling2RMSNorm(HIDDEN, eps=1e-6, dtype=dtype)
    _copy_by_name(ours, ref)
    x = _randn(T, HIDDEN, seed=3, dtype=dtype, scale=3.0)
    out = ours(x)
    assert out.dtype == dtype
    _check(out, ref(x[None])[0], dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_softcap(dtype):
    ref = ref_mod.Inkling2LoopSoftcap(4.0)
    ours = blocks.Softcap(4.0)
    x = _randn(T, HIDDEN, seed=4, dtype=dtype, scale=10.0)
    out = ours(x)
    assert out.dtype == dtype
    _check(out, ref(x[None])[0], dtype)
    assert out.float().abs().max() <= 4.0 + 1e-3


# ------------------------------------------------------------------- message gate


def _message_gate_pair(dtype: torch.dtype):
    cfg = _ref_config()
    ref = ref_mod.Inkling2LoopMessageGate(cfg)
    _randomize_(ref, 10)
    ref.down_proj.to(dtype)  # follows the model dtype; the gate projections stay fp32
    ours = blocks.MessageGate(cfg, dtype=dtype)
    _copy_by_name(ours, ref)
    return ref, ours


@pytest.mark.parametrize("dtype", DTYPES)
def test_message_gate(dtype):
    ref, ours = _message_gate_pair(dtype)
    assert ours.down_proj.weight.shape == (GATE_RANK, 2 * HIDDEN)
    assert ours.forget_gate_proj.weight.shape == (HIDDEN, GATE_RANK)
    assert ours.forget_gate_proj.weight.dtype == torch.float32
    assert ours.message_gate_proj.weight.dtype == torch.float32
    h = _randn(T, HIDDEN, seed=11, dtype=dtype, scale=2.0)
    m = _randn(T, HIDDEN, seed=12, dtype=dtype, scale=2.0)
    out = ours(h, m)
    assert out.shape == (T, HIDDEN) and out.dtype == torch.float32
    _check(out, ref(h[None], m[None])[0], dtype)


def test_message_gate_detects_perturbation():
    ref, ours = _message_gate_pair(torch.float32)
    h = _randn(T, HIDDEN, seed=11, scale=2.0)
    m = _randn(T, HIDDEN, seed=12, scale=2.0)
    want = ref(h[None], m[None])[0]
    _check(ours(h, m), want, torch.float32)
    with torch.no_grad():
        ours.message_gate_proj.weight[3, 5] += 0.5
    assert _differs(ours(h, m), want)
    with pytest.raises(AssertionError):
        _check(ours(h, m), want, torch.float32)


# ------------------------------------------------------- block attention residual


def _block_pair(num_queries: int):
    cfg = _ref_config()
    ref = ref_mod.Inkling2LoopBlockAttentionResidual(cfg, num_queries=num_queries)
    _randomize_(ref, 20 + num_queries)
    ours = blocks.BlockAttentionResidual(cfg, num_queries=num_queries)
    _copy_by_name(ours, ref)
    return ref, ours


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_blocks", [0, 1, 3])
@pytest.mark.parametrize("num_queries", [1, 3])
def test_block_attention_residual(num_queries, num_blocks, dtype):
    ref, ours = _block_pair(num_queries)
    shape = (HIDDEN,) if num_queries == 1 else (num_queries, HIDDEN)
    assert ours.query.shape == shape and ours.query.dtype == torch.float32
    states = _randn(T, num_blocks, HIDDEN, seed=30 + num_blocks, dtype=dtype, scale=2.0)
    partial = _randn(T, HIDDEN, seed=40 + num_blocks, dtype=dtype, scale=2.0)
    got = ours(states, partial)
    want = ref(states[None], partial[None])
    assert isinstance(got, tuple) and len(got) == len(want) == num_queries
    for g, w in zip(got, want, strict=True):
        assert g.shape == (T, HIDDEN) and g.dtype == torch.float32
        _check(g, w[0], dtype)


@pytest.mark.parametrize("num_queries", [1, 3])
def test_block_attention_residual_detects_perturbation(num_queries):
    ref, ours = _block_pair(num_queries)
    states = _randn(T, 3, HIDDEN, seed=31, scale=2.0)
    partial = _randn(T, HIDDEN, seed=41, scale=2.0)
    want = ref(states[None], partial[None])
    with torch.no_grad():
        ours.query.view(-1)[7] += 1.0
    got = ours(states, partial)
    assert any(_differs(g, w[0]) for g, w in zip(got, want, strict=True))


# ----------------------------------------------------- message attention residual


def _message_pair():
    cfg = _ref_config()
    ref = ref_mod.Inkling2LoopMessageAttentionResidual(cfg)
    _randomize_(ref, 50)
    ours = blocks.MessageAttentionResidual(cfg)
    _copy_by_name(ours, ref)
    return ref, ours


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("loop_layer_idx", [0, 1])
def test_message_attention_residual(loop_layer_idx, dtype):
    ref, ours = _message_pair()
    assert ours.query.shape == (LOOP_END - LOOP_START, 4, HIDDEN)
    states = _randn(T, 2, HIDDEN, seed=60, dtype=dtype, scale=2.0)
    partial = _randn(T, HIDDEN, seed=61, dtype=dtype, scale=2.0)
    got = ours(states, partial, loop_layer_idx)
    want = ref(states[None], partial[None], loop_layer_idx)[0]
    assert got.shape == (T, 4, HIDDEN) and got.dtype == torch.float32
    _check(got, want, dtype)
    # slot order is (query, key, value, mlp): every slot matches individually
    for g, w in zip(got.unbind(-2), want.unbind(-2), strict=True):
        _check(g, w, dtype)
    # the two loop layers use different queries (so the index is really exercised)
    other = ours(states, partial, 1 - loop_layer_idx)
    assert _differs(other, got)


def test_message_attention_residual_detects_perturbation():
    ref, ours = _message_pair()
    states = _randn(T, 2, HIDDEN, seed=60, scale=2.0)
    partial = _randn(T, HIDDEN, seed=61, scale=2.0)
    want = ref(states[None], partial[None], 1)[0]
    _check(ours(states, partial, 1), want, torch.float32)
    with torch.no_grad():
        ours.query[1, 2, 9] += 1.0  # slot 2 of loop layer 1 only
    got = ours(states, partial, 1)
    assert _differs(got, want)
    assert not _differs(got[:, 0], want[:, 0]) and _differs(got[:, 2], want[:, 2])
    with pytest.raises(AssertionError):
        _check(got, want, torch.float32)


# ------------------------------------------------------------- clips / log scale


def _heads_input(
    seed: int, dtype: torch.dtype, heads: int = HEADS, dim: int = HEAD_DIM
):
    """Per-vector magnitudes spread over >1 decade so some vectors get clipped."""
    x = _randn(T, heads, dim, seed=seed, scale=1.0)
    mag = torch.exp(_randn(T, heads, 1, seed=seed + 1, scale=1.0))
    return (x * mag).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_clip(dtype):
    threshold = 4.0
    ref = ref_mod.Inkling2LoopL2Clip(threshold)
    x = _heads_input(70, dtype)
    out = blocks.l2_clip(x, threshold)
    assert out.dtype == dtype and out.shape == x.shape
    _check(out, ref(x[None])[0], dtype)
    norms_in = torch.linalg.vector_norm(x.float(), dim=-1)
    norms_out = torch.linalg.vector_norm(out.float(), dim=-1)
    assert (norms_in > threshold).any() and (norms_in < threshold).any()
    assert norms_out.max() <= threshold * 1.01
    short = norms_in < threshold * 0.99
    torch.testing.assert_close(norms_out[short], norms_in[short], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_l2_clip(dtype):
    threshold = 4.0
    ref = ref_mod.Inkling2LoopScaledL2Clip(HEADS, HEAD_DIM, threshold)
    _randomize_(ref, 71)
    x = _heads_input(72, dtype)
    out = blocks.scaled_l2_clip(x, ref.scale.detach().clone(), threshold)
    assert out.dtype == dtype
    _check(out, ref(x[None])[0], dtype)


POSITIONS = [0, 14, 15, 16, 100]


@pytest.mark.parametrize("dtype", DTYPES)
def test_log_scale(dtype):
    ref = ref_mod.Inkling2LoopLogScale(_ref_config())
    x = _heads_input(73, dtype)[: len(POSITIONS)]
    pos = torch.tensor(POSITIONS, dtype=torch.int64)
    out = blocks.log_scale(x, pos, ALPHA, N_FLOOR)
    assert out.dtype == dtype
    _check(out, ref(x[None], pos[None])[0], dtype)
    ratio = out.float() / x.float()
    # tau == 1 up to and including position n_floor - 1, > 1 afterwards
    for i, p in enumerate(POSITIONS):
        tau = 1.0 + ALPHA * torch.log(torch.tensor(max((p + 1) / N_FLOOR, 1.0))).item()
        torch.testing.assert_close(
            ratio[i], torch.full_like(ratio[i], tau), rtol=2e-3, atol=2e-3
        )
    assert ratio[3].mean() > 1.0 and ratio[2].mean() == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_log_scaled_l2_clip(dtype):
    threshold = 4.0
    ref = ref_mod.Inkling2LoopLogScaledL2Clip(_ref_config(), HEADS, HEAD_DIM, threshold)
    _randomize_(ref, 74)
    # a position set straddling n_floor, repeated to fill T tokens
    pos = torch.tensor((POSITIONS * 5)[:T], dtype=torch.int64)
    x = _heads_input(75, dtype)
    out = blocks.log_scaled_l2_clip(
        x, ref.scale.detach().clone(), pos, threshold, ALPHA, N_FLOOR
    )
    assert out.dtype == dtype
    _check(out, ref(x[None], pos[None])[0], dtype)


def test_log_scale_detects_perturbation():
    ref = ref_mod.Inkling2LoopLogScale(_ref_config())
    x = _heads_input(73, torch.float32)[: len(POSITIONS)]
    pos = torch.tensor(POSITIONS, dtype=torch.int64)
    want = ref(x[None], pos[None])[0]
    _check(blocks.log_scale(x, pos, ALPHA, N_FLOOR), want, torch.float32)
    # a wrong floor changes the result at positions >= 14 only
    got = blocks.log_scale(x, pos, ALPHA, N_FLOOR + 8)
    assert _differs(got, want)
    assert not _differs(got[:3], want[:3]) and _differs(got[3:], want[3:])


# ------------------------------------------------------------------- MoE helpers


def _router_pair(seed: int = 80):
    cfg = _ref_config()
    ref = ref_mod.Inkling2LoopTopKRouter(cfg)
    _randomize_(ref, seed)
    with torch.no_grad():
        ref.e_score_correction_bias.copy_(_randn(NUM_EXPERTS, seed=seed + 1, scale=0.8))
    return cfg, ref


def _by_id(weights: torch.Tensor, ids: torch.Tensor):
    ids_sorted, order = ids.long().sort(dim=-1)
    return weights.gather(1, order), ids_sorted


def test_route_topk():
    cfg, ref = _router_pair()
    hidden = _randn(T * 3, HIDDEN, seed=82, scale=1.0)
    logits_ref, w_ref, ids_ref = ref(hidden)
    assert ref.e_score_correction_bias.abs().sum() > 0
    w, ids = blocks.route_topk(
        logits_ref, ref.e_score_correction_bias, TOP_K, cfg.route_scale
    )
    assert w.dtype == torch.float32 and w.shape == (T * 3, TOP_K)
    assert ids.dtype == torch.int32 and ids.shape == (T * 3, TOP_K)
    w_s, ids_s = _by_id(w, ids)
    w_ref_s, ids_ref_s = _by_id(w_ref, ids_ref)
    assert torch.equal(ids_s, ids_ref_s)
    torch.testing.assert_close(w_s, w_ref_s, rtol=1e-5, atol=1e-5)
    # normalise-then-scale: every row sums to route_scale
    torch.testing.assert_close(
        w.sum(-1), torch.full((T * 3,), cfg.route_scale), rtol=1e-5, atol=1e-5
    )
    # the bias participates in selection (not in the weights): dropping it must
    # change the chosen experts for at least one token, so the equality above is
    # a real check of the bias handling
    _, ids_nobias = blocks.route_topk(
        logits_ref,
        torch.zeros_like(ref.e_score_correction_bias),
        TOP_K,
        cfg.route_scale,
    )
    assert not torch.equal(ids_nobias.long().sort(-1).values, ids_s.long())


def test_route_topk_detects_perturbation():
    cfg, ref = _router_pair()
    hidden = _randn(T * 3, HIDDEN, seed=82, scale=1.0)
    logits_ref, w_ref, ids_ref = ref(hidden)
    bias = ref.e_score_correction_bias.clone()
    bias[2] += 0.5  # perturbed bias
    w, ids = blocks.route_topk(logits_ref, bias, TOP_K, cfg.route_scale)
    w_s, ids_s = _by_id(w, ids)
    w_ref_s, ids_ref_s = _by_id(w_ref, ids_ref)
    assert not torch.equal(ids_s, ids_ref_s)
    # wrong order of scale/normalise would also be caught: sum != route_scale
    unnormalised = torch.sigmoid(logits_ref).gather(1, ids_ref) * cfg.route_scale
    assert _differs(unnormalised, w_ref)


@pytest.mark.parametrize("dtype", DTYPES)
def test_softcapped_swiglu(dtype):
    cfg = _ref_config()
    ref = ref_mod.Inkling2LoopSoftcappedSwiGLU(cfg)
    gate = _randn(T, 64, seed=90, dtype=dtype, scale=8.0)
    up = _randn(T, 64, seed=91, dtype=dtype, scale=40.0)
    out = blocks.softcapped_swiglu(gate, up, cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
    assert out.dtype == torch.float32 and out.shape == gate.shape
    want = ref(gate[None], up[None])[0]
    assert want.dtype == dtype
    if dtype == torch.float32:
        _check(out, want, dtype)
    else:
        # the reference rounds the same fp32 value to fp16 at the end
        _check(out.to(dtype), want, dtype)
        assert _rel_err(out, want) <= 2e-3
    # the caps are active for these magnitudes
    assert up.float().abs().max() > cfg.mlp_up_softcap
    # the module form returns the activation dtype
    assert blocks.SoftcappedSwiGLU(cfg)(gate, up).dtype == dtype


def test_softcapped_swiglu_detects_perturbation():
    cfg = _ref_config()
    ref = ref_mod.Inkling2LoopSoftcappedSwiGLU(cfg)
    gate = _randn(T, 64, seed=90, scale=8.0)
    up = _randn(T, 64, seed=91, scale=40.0)
    want = ref(gate[None], up[None])[0]
    out = blocks.softcapped_swiglu(gate, up, cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
    _check(out, want, torch.float32)
    wrong = blocks.softcapped_swiglu(gate, up, cfg.mlp_gate_softcap, 20.0)
    assert _differs(wrong, want)


# ---------------------------------------------------------------- our own config


def test_modules_accept_our_config():
    """The blocks only read attribute names shared by both config classes."""
    cfg = our_configs.Inkling2LoopTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=6,
        n_layers_per_attn_res_block=2,
        loop_config={
            "core_start": LOOP_START,
            "core_end": LOOP_END,
            "num_passes": 2,
            "gate_rank": GATE_RANK,
        },
    )
    assert blocks.MessageAttentionResidual(cfg).query.shape == (2, 4, HIDDEN)
    gate = blocks.MessageGate(cfg, dtype=torch.float16)
    assert gate.down_proj.weight.shape == (GATE_RANK, 2 * HIDDEN)
    assert blocks.BlockAttentionResidual(cfg, num_queries=3).query.shape == (3, HIDDEN)
