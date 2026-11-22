# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical parity of the dense-state Inkling-2 attention stack against the
HF reference (``transformers.models.inkling2_loop``), CPU only.

Our modules are loaded by file path: they have no vllm imports, and an
installed ``vllm`` would otherwise shadow ``vllm.models.inkling2_loop``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.inkling2_loop")

from transformers.models.inkling2_loop import modeling_inkling2_loop as ref_mod  # noqa: E402
from transformers.models.inkling2_loop.configuration_inkling2_loop import (  # noqa: E402
    Inkling2LoopTextConfig as RefTextConfig,
)

_ROOT = Path(__file__).resolve().parents[3] / "vllm" / "models" / "inkling2_loop"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


attn = _load(
    "_inkling2_loop_amd_dense_attention_under_test",
    _ROOT / "amd" / "dense_attention.py",
)
OurTextConfig = _load(
    "_inkling2_loop_configs_under_test", _ROOT / "configs.py"
).Inkling2LoopTextConfig

SLIDING_LAYER = 0  # layer_types[0] == "hybrid_sliding"
FULL_LAYER = 5  # layer_types[5] == "hybrid"
TOL = 1e-5


def _configs():
    ref_cfg = RefTextConfig()
    ref_cfg._attn_implementation = "eager"
    return ref_cfg, OurTextConfig()


def _randomize(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("norm.scale"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
            elif name.endswith("conv1d.weight") or name.endswith("static_weight"):
                p.copy_(0.3 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
            elif name.endswith("relative_position_bias.proj"):
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype))
            else:  # linear weights
                p.copy_(0.02 * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _build(kind: str, dtype: torch.dtype = torch.float32, seed: int = 0):
    ref_cfg, our_cfg = _configs()
    if kind == "sliding":
        ref = ref_mod.Inkling2LoopSlidingAttention(ref_cfg, SLIDING_LAYER)
        ours = attn.SlidingAttention(our_cfg, SLIDING_LAYER)
        window = ref_cfg.sliding_window_size
    else:
        ref = ref_mod.Inkling2LoopFullAttention(ref_cfg, FULL_LAYER)
        ours = attn.FullAttention(our_cfg, FULL_LAYER)
        window = None
    assert ref.is_sliding == (kind == "sliding")
    _randomize(ref, seed)
    if dtype != torch.float32:
        ref = ref.to(dtype)
        ours = ours.to(dtype)
    ref.eval()
    ours.eval()
    ours.load_state_dict(ref.state_dict(), strict=True)
    # identical keys, shapes and dtypes
    ref_sd, our_sd = ref.state_dict(), ours.state_dict()
    assert list(ref_sd) == list(our_sd)
    for k, v in ref_sd.items():
        assert our_sd[k].shape == v.shape and our_sd[k].dtype == v.dtype, k
    return ref, ours, window


@torch.no_grad()
def _reference(
    ref, kind, window, hidden, positions, key_hidden=None, value_hidden=None
):
    """Reference forward over a whole sequence ``hidden`` ``[S, H]`` without cache."""
    seq = hidden.shape[0]
    idx = torch.arange(seq)
    distance = idx[:, None] - idx[None, :]
    allowed = distance >= 0
    if window is not None:
        allowed &= distance < window
    mask = torch.zeros(seq, seq, dtype=hidden.dtype).masked_fill(
        ~allowed, torch.finfo(hidden.dtype).min
    )[None, None]
    h = hidden[None]
    pos = positions[None]
    if kind == "sliding":
        out, _ = ref(
            h,
            mask,
            distance,
            key_hidden_states=None if key_hidden is None else key_hidden[None],
            value_hidden_states=None if value_hidden is None else value_hidden[None],
            position_ids=pos,
        )
    else:
        out, _ = ref(
            h,
            h if key_hidden is None else key_hidden[None],
            h if value_hidden is None else value_hidden[None],
            mask,
            distance,
            position_ids=pos,
        )
    return out[0]


@torch.no_grad()
def _run_chunks(ours, hidden, positions, sizes, key_hidden=None, value_hidden=None):
    assert sum(sizes) == hidden.shape[0]
    state = ours.new_state(dtype=hidden.dtype)
    outs, start = [], 0
    for n in sizes:
        sl = slice(start, start + n)
        outs.append(
            ours(
                hidden[sl],
                positions[sl],
                state,
                key_hidden_states=None if key_hidden is None else key_hidden[sl],
                value_hidden_states=None if value_hidden is None else value_hidden[sl],
            )
        )
        start += n
    assert state.num_tokens == hidden.shape[0]
    return torch.cat(outs), state


def _inputs(seq, seed=1, offset=0, n=1):
    gen = torch.Generator().manual_seed(seed)
    xs = [torch.randn(seq, 2304, generator=gen) for _ in range(n)]
    pos = torch.arange(seq) + offset
    return (*xs, pos) if n > 1 else (xs[0], pos)


def _err(a, b):
    return (a - b).abs().max().item()


@pytest.mark.parametrize("kind", ["sliding", "full"])
@pytest.mark.parametrize("seq", [1, 7, 64, 1100])
def test_full_sequence_prefill(kind, seq):
    ref, ours, window = _build(kind)
    x, pos = _inputs(seq)
    expected = _reference(ref, kind, window, x, pos)
    with torch.no_grad():
        stateless = ours(x, pos)  # state=None -> stateless prefill
        got, _ = _run_chunks(ours, x, pos, [seq])
        batched = ours(x[None], pos[None])  # [1, S, H] input form
    assert got.shape == expected.shape
    assert _err(got, expected) <= TOL
    assert _err(stateless, expected) <= TOL
    assert _err(batched[0], expected) <= TOL


@pytest.mark.parametrize("kind", ["sliding", "full"])
def test_chunked_prefill(kind):
    ref, ours, window = _build(kind)
    x, pos = _inputs(1100, seed=2)
    expected = _reference(ref, kind, window, x, pos)
    got, state = _run_chunks(ours, x, pos, [37, 500, 563])
    assert _err(got, expected) <= TOL
    if window is not None:  # sliding state keeps only window - 1 tokens
        assert state.k.shape[2] == window - 1 and state.v.shape[2] == window - 1
    else:
        assert state.k.shape[2] == 1100


@pytest.mark.parametrize("kind", ["sliding", "full"])
def test_token_by_token_decode(kind):
    ref, ours, window = _build(kind)
    x, pos = _inputs(160, seed=3)
    expected = _reference(ref, kind, window, x, pos)
    got, _ = _run_chunks(ours, x, pos, [1] * 160)
    assert _err(got, expected) <= TOL


@pytest.mark.parametrize("kind", ["sliding", "full"])
def test_prefill_then_decode_past_window(kind):
    ref, ours, window = _build(kind)
    x, pos = _inputs(1100, seed=4)
    expected = _reference(ref, kind, window, x, pos)
    got, _ = _run_chunks(ours, x, pos, [1030] + [1] * 70)
    assert _err(got, expected) <= TOL
    got, _ = _run_chunks(ours, x, pos, [3, 1, 1, 700, 1, 393, 1])
    assert _err(got, expected) <= TOL


@pytest.mark.parametrize("kind", ["sliding", "full"])
@pytest.mark.parametrize("sizes", [[64], [10, 1, 1, 30, 22], [1] * 64])
def test_distinct_key_value_hidden_states(kind, sizes):
    ref, ours, window = _build(kind, seed=5)
    x, kx, vx, pos = _inputs(64, seed=6, n=3)
    expected = _reference(ref, kind, window, x, pos, kx, vx)
    # the distinct inputs must matter
    assert _err(expected, _reference(ref, kind, window, x, pos)) > 1e-3
    got, _ = _run_chunks(ours, x, pos, sizes, kx, vx)
    assert _err(got, expected) <= TOL


@pytest.mark.parametrize("kind", ["sliding", "full"])
@pytest.mark.parametrize("sizes", [[64], [1] * 64, [20, 1, 43]])
def test_log_scaling_beyond_n_floor(kind, sizes):
    # Positions 8190.. cross log_scaling_n_floor (8192): log scale is 1 for
    # position + 1 <= 8192 and > 1 afterwards. Attention indices stay 0-based.
    ref, ours, window = _build(kind, seed=7)
    x, pos = _inputs(64, seed=8, offset=8190)
    expected = _reference(ref, kind, window, x, pos)
    got, _ = _run_chunks(ours, x, pos, sizes)
    assert _err(got, expected) <= TOL
    # the offset positions change the full layer's output (log scaling active) and
    # not the sliding layer's (no log scaling there)
    shifted = _reference(ref, kind, window, x, pos - 8190)
    if kind == "full":
        # only the tokens past the floor differ, by tau - 1 ~ 1e-4 * (pos - 8191)
        assert _err(expected, shifted) > 20 * TOL
        assert _err(got, shifted) > 20 * TOL
    else:
        assert _err(expected, shifted) <= TOL


def test_log_scaling_far_positions():
    ref, ours, window = _build("full", seed=9)
    x, pos = _inputs(40, seed=10, offset=100_000)
    expected = _reference(ref, "full", window, x, pos)
    got, _ = _run_chunks(ours, x, pos, [17, 1, 22])
    assert _err(got, expected) <= TOL
    assert _err(got, _reference(ref, "full", window, x, pos - 100_000)) > 1e-2


@pytest.mark.parametrize("kind", ["sliding", "full"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision(kind, dtype):
    ref, ours, window = _build(kind, dtype=dtype, seed=11)
    x, pos = _inputs(130, seed=12)
    x = x.to(dtype)
    expected = _reference(ref, kind, window, x, pos).float()
    got, _ = _run_chunks(ours, x, pos, [100, 1, 29])
    assert got.dtype == dtype
    scale = expected.abs().max().item()
    assert _err(got.float(), expected) <= 2e-2 * scale


@pytest.mark.parametrize(
    "seq,sizes", [(50, [50]), (50, [7, 1, 1, 21, 20]), (9, [1] * 9)]
)
def test_standalone_short_convolution(seq, sizes):
    ref_cfg, _ = _configs()
    k = ref_cfg.conv_kernel_size
    channels = 192
    ref = ref_mod.Inkling2LoopShortConvolution(channels, k, layer_idx=0, conv_idx=2)
    ours = attn.ShortConvolution(channels, k)
    gen = torch.Generator().manual_seed(13)
    with torch.no_grad():
        ref.conv1d.weight.copy_(
            0.3 * torch.randn(ref.conv1d.weight.shape, generator=gen)
        )
    ours.load_state_dict(ref.state_dict(), strict=True)
    assert ours.conv1d.weight.dtype == ref.conv1d.weight.dtype == torch.float32
    assert list(ours.state_dict()) == ["conv1d.weight"]

    x = torch.randn(seq, channels, generator=gen)
    with torch.no_grad():
        expected = ref(x[None])[0]
        state = ours.new_state()
        outs, start = [], 0
        for n in sizes:
            outs.append(ours(x[start : start + n], state=state))
            start += n
        got = torch.cat(outs)
        assert _err(ours(x), expected) <= TOL
        assert _err(ours(x[None])[0], expected) <= TOL
    assert _err(got, expected) <= TOL
    assert state.tail.shape == (1, channels, k - 1)
    assert torch.equal(state.tail, x[-(k - 1) :].T[None])


def test_standalone_short_convolution_low_precision_input():
    k = 4
    ref = ref_mod.Inkling2LoopShortConvolution(32, k, layer_idx=0, conv_idx=0).to(
        torch.float32
    )
    ours = attn.ShortConvolution(32, k)
    gen = torch.Generator().manual_seed(14)
    with torch.no_grad():
        ref.conv1d.weight.copy_(
            0.3 * torch.randn(ref.conv1d.weight.shape, generator=gen)
        )
        ours.load_state_dict(ref.state_dict(), strict=True)
        x = torch.randn(1, 20, 32, generator=gen).to(torch.bfloat16)
        out = ours(x)
        assert out.dtype == torch.bfloat16
        assert _err(out.float(), ref(x).float()) <= 1e-6
