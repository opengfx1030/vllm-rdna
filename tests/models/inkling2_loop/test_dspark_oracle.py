# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fp64 oracle of the Inkling-2 DSpark draft (``tools/rdna/inkling2/
dspark_check.py``) against independent re-statements of the reference, and the draft
config conversion.  CPU only; the real-checkpoint tests skip without
``~/hfcache/inkling2-mini``.

The HIP draft is held to this oracle in ``tests/kernels/rocm/
test_inkling2_dspark_rdna2.py`` and ``test_dspark_model.py``, so the checks here are
about the *meaning* of the oracle: every test re-derives the quantity another way
(complex rotation for RoPE, the reference's bf16 torch ops for the gate, a masked
full-context softmax for the window) instead of calling the code under test twice.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
CKPT = Path("~/hfcache/inkling2-mini").expanduser()
MTP = CKPT / "mtp.safetensors"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oracle = _load("dspark_check")

needs_real = pytest.mark.skipif(
    not (MTP.exists() and (CKPT / "config.json").exists()),
    reason=f"real Inkling-2-Mini checkpoint not found at {CKPT}",
)

TINY = oracle.DraftConfig(
    hidden_size=64,
    num_layers=3,
    num_heads=6,
    num_kv_heads=2,
    head_dim=16,
    intermediate_size=96,
    markov_rank=8,
    block_size=8,
    window=40,
    local_layer_ids=(0, 2),
    rope_theta=10000.0,
)
VOCAB = 300


@pytest.fixture(scope="module")
def tiny():
    return oracle.random_draft(TINY, VOCAB, seed=3)


@pytest.fixture(scope="module")
def lm_head():
    g = torch.Generator().manual_seed(4)
    return (torch.randn(VOCAB, TINY.hidden_size, generator=g) * 0.2).double()


def _case(d, n_ctx: int, seed: int = 0):
    cfg = d.cfg
    taps = oracle.synthetic_taps(n_ctx, cfg, seed)
    pos = torch.arange(n_ctx)
    g = torch.Generator().manual_seed(seed + 100)
    x0 = torch.randn(cfg.hidden_size, generator=g).double()
    return taps, pos, x0


# --------------------------------------------------------------------------- RoPE


def test_rope_is_a_complex_rotation():
    """Neox RoPE of ``(x1, x2)`` is the complex product ``(x1 + i x2) e^{i a}``."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 3, 32, generator=g).double()
    pos = torch.tensor([0, 1, 17, 1000, 123456])
    out = oracle.rope(x, pos, 1e6)
    j = torch.arange(16, dtype=torch.float64)
    ang = pos.double()[:, None] * (1e6 ** (-2 * j / 32))[None, :]
    z = (
        torch.complex(x[..., :16], x[..., 16:])
        * torch.polar(torch.ones_like(ang), ang)[:, None, :]
    )
    ref = torch.cat([z.real, z.imag], dim=-1)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=0)


def test_rope_fp32_table_loses_precision_at_long_context():
    """The reference's fp32 angle table drifts ~``pos * 6e-8`` rad; ours is exact."""
    pos = torch.tensor([1_000_000])
    cos_e, sin_e = oracle.rope_tables(pos, 128, 1e6, "exact")
    cos_f, sin_f = oracle.rope_tables(pos, 128, 1e6, "fp32")
    err = ((cos_e - cos_f).abs().max()).item()
    assert 1e-3 < err < 0.2  # drift of the highest frequencies, visible
    cos_e, sin_e = oracle.rope_tables(torch.tensor([100]), 128, 1e6, "exact")
    cos_f, sin_f = oracle.rope_tables(torch.tensor([100]), 128, 1e6, "fp32")
    assert (cos_e - cos_f).abs().max().item() < 1e-4


# --------------------------------------------------------------------------- mixing


def test_mix_rms_scale_invariance_and_softmax_weights(tiny):
    """The mixed tap is normalized: scaling every tap leaves it unchanged (up to
    ``eps``) and ``alpha`` rows are probability vectors."""
    taps = oracle.synthetic_taps(7, TINY, 5)
    alpha = tiny.alpha
    assert torch.allclose(alpha.sum(-1), torch.ones(TINY.num_layers + 1).double())
    w = tiny.layer(0, "dflare_norm.weight")
    a = oracle.mix_rms(taps, alpha[0], w, 1e-30)
    b = oracle.mix_rms(taps * 37.0, alpha[0], w, 1e-30)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    # unit RMS before the gain
    rms = (a / w).square().mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- attention


def test_block_attention_is_bidirectional(tiny):
    """A later block row's input changes an earlier row's output (the block is not
    causal): after the first layer row 0 already depends on row 7's ``mask_embed``."""
    taps, pos, x0 = _case(tiny, 20)
    ctx = oracle.context_kv(tiny, taps, pos)
    base_in = oracle.block_input(tiny, x0)
    h0 = oracle.block_hidden(tiny, ctx, pos, base_in, 20)
    bumped = base_in.clone()
    bumped[7] += 1.0
    h1 = oracle.block_hidden(tiny, ctx, pos, bumped, 20)
    assert (h1[0] - h0[0]).abs().max() > 1e-6


@pytest.mark.parametrize("local", [True, False])
def test_window_is_a_mask_over_the_context(tiny, local):
    """Local layers ignore context older than ``window`` positions before the block;
    perturbing it changes the output only if some layer is global."""
    d = oracle.Draft(
        dataclasses.replace(
            TINY, local_layer_ids=(0, 1, 2) if local else (0, 2), window=24
        ),
        tiny.tensors,
    )
    n = 64
    taps, pos, x0 = _case(d, n)
    h_ref = oracle.block_hidden(
        d, oracle.context_kv(d, taps, pos), pos, oracle.block_input(d, x0), n
    )
    old = taps.clone()
    old[: n - d.cfg.window] += 5.0  # positions < start - window
    h_old = oracle.block_hidden(
        d, oracle.context_kv(d, old, pos), pos, oracle.block_input(d, x0), n
    )
    moved = (h_old - h_ref).abs().max().item()
    assert (moved < 1e-12) == local
    # the boundary row (position start - window) is visible
    edge = taps.clone()
    edge[n - d.cfg.window] += 5.0
    h_edge = oracle.block_hidden(
        d, oracle.context_kv(d, edge, pos), pos, oracle.block_input(d, x0), n
    )
    assert (h_edge - h_ref).abs().max() > 1e-6


def test_local_layer_sees_one_window_of_positions_before_the_block(tiny):
    """Key positions ``[start - W, start)`` -- not ``(start - W, start]`` and not
    ``W + B`` -- are the context a local layer reads (upstream ``window = W + block``
    counts the block's own keys)."""
    d = oracle.Draft(
        dataclasses.replace(TINY, local_layer_ids=(0, 1, 2), window=24), tiny.tensors
    )
    n = 64
    taps, pos, x0 = _case(d, n)
    ctx = oracle.context_kv(d, taps, pos)
    full = oracle.block_hidden(d, ctx, pos, oracle.block_input(d, x0), n)
    keep = pos >= n - 24
    cut = [(k[keep], v[keep]) for k, v in ctx]
    cut_h = oracle.block_hidden(d, cut, pos[keep], oracle.block_input(d, x0), n)
    torch.testing.assert_close(full, cut_h, atol=1e-12, rtol=0)
    assert int(keep.sum()) == 24


# --------------------------------------------------------------------------- Markov


def test_fold_is_the_unfolded_head(tiny, lm_head):
    """``lm_head @ hidden_out(latent) == fold_weight @ latent + fold_bias``."""
    fold_w, fold_b = oracle.fold_head(tiny, lm_head)
    g = torch.Generator().manual_seed(9)
    latent = torch.tanh(torch.randn(TINY.markov_rank, generator=g)).double()
    unfolded = lm_head @ (
        tiny["markov_head.hidden_out.weight"] @ latent
        + tiny["markov_head.hidden_out.bias"]
    )
    torch.testing.assert_close(fold_w @ latent + fold_b, unfolded, atol=1e-12, rtol=0)


def test_markov_gate_rounding_matches_the_reference_ops():
    """With ``bf16_gate`` the oracle's gate is the reference's chain of bf16 torch ops
    (upstream ``test_markov_head_fold_and_softcapped_logits``): bit-identical states
    and latents."""
    g = torch.Generator().manual_seed(11)
    R = 8
    raw = torch.randn(32, 3 * R, generator=g).mul(4)
    state = torch.randn(32, R, generator=g).bfloat16()
    gate_raw, cand_raw, out_raw = raw.bfloat16().chunk(3, dim=-1)
    gate = torch.sigmoid(gate_raw)
    ref_state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
    ref_latent = torch.tanh(out_raw)

    def r16(x):
        return x.float().bfloat16().double()

    gr, cr, orr = raw.bfloat16().double().chunk(3, dim=-1)
    st = state.double()
    gt = r16(torch.sigmoid(gr))
    mine = r16(r16(gt * st) + r16(r16(1.0 - gt) * r16(torch.tanh(cr))))
    assert torch.equal(mine.bfloat16(), ref_state)
    assert torch.equal(r16(torch.tanh(orr)).bfloat16(), ref_latent)


def test_markov_chain_matches_a_direct_unrolled_reference(tiny, lm_head):
    """Teacher-forced chain against a second, step-by-step transcription."""
    c = TINY
    R, H = c.markov_rank, c.hidden_size
    g = torch.Generator().manual_seed(12)
    h = torch.randn(c.num_steps, H, generator=g).double()
    f = torch.randn(H, generator=g).double()
    tokens = torch.tensor([5, 17, 255, 0, 299, 42, 8])
    trace = oracle.markov_chain(tiny, h, f, 77, lm_head, tokens)

    W = tiny["markov_head.rnn_proj.weight"]
    b = tiny["markov_head.rnn_proj.bias"]
    w1 = tiny["markov_head.markov_w1.weight"]
    ho_w, ho_b = (
        tiny["markov_head.hidden_out.weight"],
        tiny["markov_head.hidden_out.bias"],
    )
    state, prev = torch.zeros(R).double(), 77
    for i in range(c.num_steps):
        x = torch.cat([state, w1[prev], h[i], f])  # [2R + 2H]
        raw = b + W @ x  # one fused matvec instead of ctx + recurrent halves
        gate = torch.sigmoid(raw[:R])
        state = gate * state + (1 - gate) * torch.tanh(raw[R : 2 * R])
        latent = torch.tanh(raw[2 * R :])
        logits = 30 * torch.tanh(
            (lm_head @ h[i] + lm_head @ (ho_w @ latent + ho_b)) / 30
        )
        torch.testing.assert_close(trace.logits[i], logits, atol=1e-12, rtol=0)
        torch.testing.assert_close(trace.states[i], state, atol=1e-12, rtol=0)
        prev = int(tokens[i])


def test_markov_chain_follows_the_greedy_chain_without_teacher(tiny, lm_head):
    g = torch.Generator().manual_seed(13)
    h = torch.randn(TINY.num_steps, TINY.hidden_size, generator=g).double()
    f = torch.randn(TINY.hidden_size, generator=g).double()
    free = oracle.markov_chain(tiny, h, f, 3, lm_head)
    forced = oracle.markov_chain(tiny, h, f, 3, lm_head, free.tokens)
    assert torch.equal(free.tokens, forced.tokens)
    torch.testing.assert_close(free.logits, forced.logits, atol=0, rtol=0)
    assert (free.logits.abs() <= TINY.softcap).all()
    # the chain really depends on the previous token (a different anchor moves step 0)
    other = oracle.markov_chain(tiny, h, f, 4, lm_head, free.tokens)
    assert (other.logits[0] - free.logits[0]).abs().max() > 1e-9


def test_end_to_end_proposal_shapes(tiny, lm_head):
    taps, pos, x0 = _case(tiny, 50)
    h, trace = oracle.propose(tiny, taps, pos, x0, 9, lm_head)
    assert h.shape == (TINY.block_size, TINY.hidden_size)
    assert trace.logits.shape == (TINY.num_steps, VOCAB)
    assert torch.isfinite(h).all() and torch.isfinite(trace.logits).all()


# ------------------------------------------------------------------- real weights


@needs_real
def test_real_checkpoint_names_and_dtypes():
    from safetensors import safe_open

    cfg = json.loads((CKPT / "config.json").read_text())
    dc = oracle.DraftConfig.from_mtp_config(cfg["mtp_config"])
    with safe_open(str(MTP), framework="pt") as f:
        assert sorted(f.keys()) == oracle.expected_names(dc.num_layers)  # noqa: SIM118
        for name in f.keys():  # noqa: SIM118
            assert f.get_slice(name).get_dtype() == "BF16", name
        shapes = {n: tuple(f.get_slice(n).get_shape()) for n in f.keys()}  # noqa: SIM118
    H, R = dc.hidden_size, dc.markov_rank
    assert shapes["fuse_weight"] == (dc.num_layers + 1, dc.num_taps)
    assert shapes["markov_head.rnn_proj.weight"] == (3 * R, 2 * R + 2 * H)
    assert shapes["markov_head.markov_w1.weight"][1] == R
    assert shapes["layers.0.self_attn.q_proj.weight"] == (dc.num_heads * dc.head_dim, H)
    assert "layers.0.self_attn.v_norm_ctx.weight" not in shapes


@needs_real
def test_real_weights_survive_the_fp16_conversion():
    """bf16 -> fp16 is exact for every normal-range weight; nothing overflows."""
    cfg = json.loads((CKPT / "config.json").read_text())
    dc = oracle.DraftConfig.from_mtp_config(cfg["mtp_config"])
    d = oracle.load_draft(MTP, dc)
    rep = oracle.fp16_weight_report(d)
    assert rep["overflow"] == 0
    assert rep["max_abs"] < 100.0
    assert rep["max_rel"] == 0.0
    assert rep["flushed_to_zero"] < 1000  # of ~3.2e8 weights, all below 3e-8


@needs_real
def test_real_oracle_proposal_is_finite_and_softcapped():
    cfg = json.loads((CKPT / "config.json").read_text())
    dc = oracle.DraftConfig.from_mtp_config(
        cfg["mtp_config"], cfg["text_config"]["final_logit_softcapping"]
    )
    d = oracle.load_draft(MTP, dc)
    vocab = cfg["text_config"]["unpadded_vocab_size"]
    emb, norm, unembed = oracle.read_target_head(CKPT, vocab, [1234])
    x = oracle.rms_norm(emb[0], norm, dc.eps)
    taps = oracle.synthetic_taps(120, dc, 1)
    h, trace = oracle.propose(d, taps, torch.arange(120), x, 1234, unembed)
    assert torch.isfinite(h).all() and torch.isfinite(trace.logits).all()
    assert trace.logits.abs().max() <= dc.softcap
    assert int(trace.tokens.max()) < vocab
    # the fold, with the real head
    fold_w, fold_b = oracle.fold_head(d, unembed)
    latent = trace.latents[0]
    unfolded = unembed @ (
        d["markov_head.hidden_out.weight"] @ latent + d["markov_head.hidden_out.bias"]
    )
    torch.testing.assert_close(fold_w @ latent + fold_b, unfolded, atol=1e-9, rtol=0)
