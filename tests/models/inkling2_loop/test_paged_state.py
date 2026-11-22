# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests of the paged Inkling-2 state code against the dense-state oracle.

``conv_taps`` + ``StaticSconv``, ``kv_write`` / ``kv_read`` and ``Inkling2Attention``
(``vllm.models.inkling2_loop.amd``) run on torch CPU tensors behind the engine emulation
of ``paged_harness.py`` (per-group block tables, chunked prefill / decode / mixed
batches, sliding-window block freeing, NaN-poisoned pages).  The oracle is
``dense_attention.py`` (plain torch, dense per-sequence state, pinned to the HF
reference by ``test_attention.py``).  No transformers build with ``inkling2_loop``
is needed.

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE pytest tests/models/inkling2_loop/test_paged_state.py -q
"""

from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("vllm.models.inkling2_loop.amd.state")

from paged_harness import (  # noqa: E402
    PagedWorld,
    fake_vllm_config,
    patched,
    schedule,
    text_config,
)

from vllm.models.inkling2_loop.amd import attention as attn_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import dense_attention as dense  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402

UNIT_TOL = 1e-4  # fp32; the paged and dense paths differ by op ordering only


def _report(what: str, err: float) -> None:
    print(f"[paged-state] {what}: max|diff|={err:.3e}")


@pytest.mark.parametrize("stream", [state_mod.STREAM_K, state_mod.STREAM_ATTN])
@pytest.mark.parametrize("conv_bs", [4, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_conv_taps_matches_dense_short_conv(stream, conv_bs, seed):
    cfg = text_config()
    nkv, D, H, k = cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, 4
    channels = (nkv * D, nkv * D, H, H)
    fake = fake_vllm_config("none", None)
    with patched(state_mod, "get_current_vllm_config", lambda: fake):
        owner = state_mod.Inkling2ConvOwner(
            num_kv_heads=nkv, stream_channels=channels, kernel_size=k, prefix="t.conv"
        )
    world = PagedWorld(
        [(owner, "conv")], {"conv": (conv_bs, k)}, torch.float32, seed=seed
    )
    C = channels[stream]
    gen = torch.Generator().manual_seed(seed)
    paged_conv = attn_mod.StaticSconv(C, k)
    dense_conv = dense.ShortConvolution(C, k)
    with torch.no_grad():
        dense_conv.conv1d.weight.copy_(0.5 * torch.randn(C, 1, k, generator=gen))
        paged_conv.conv1d.weight.copy_(dense_conv.conv1d.weight)
    lengths = [37, 11, 70]
    xs = [torch.randn(n, C, generator=gen) for n in lengths]
    with torch.no_grad():
        want = [dense_conv(x) for x in xs]
    got: list[list[torch.Tensor]] = [[] for _ in lengths]
    for step in schedule(random.Random(seed), lengths):
        work = [(r, b - a) for r, a, b in step]
        md = world.metadata(work)["t.conv"]
        x = torch.cat([xs[r][a:b] for r, a, b in step])
        pos = world.positions(work)
        with torch.no_grad():
            taps = state_mod.conv_taps(owner, stream, x, pos, md)
            assert taps.shape == (k, x.shape[0], C)
            y = paged_conv.forward_taps(taps)
        world.commit(work)
        at = 0
        for r, a, b in step:
            got[r].append(y[at : at + b - a])
            at += b - a
    for r in range(len(lengths)):
        err = float((torch.cat(got[r]) - want[r]).abs().max())
        _report(f"conv_taps stream={stream} bs={conv_bs} request {r}", err)
        assert err < UNIT_TOL


@pytest.mark.parametrize("stream", [state_mod.STREAM_K, state_mod.STREAM_ATTN])
@pytest.mark.parametrize("conv_bs", [4, 8])
@pytest.mark.parametrize("hit_first", [False, True])
def test_conv_taps_of_a_prefix_hit_on_blocks_written_in_the_same_step(
    stream, conv_bs, hit_first
):
    """The scheduler registers the blocks of the tokens it schedules before they are
    computed, so a request of the batch can hit blocks that another request writes in
    this very step: the conv history of its first tokens is read from those blocks, so
    ``conv_taps`` has to write the inputs of the step before it reads."""
    cfg = text_config()
    nkv, D, H, k = cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, 4
    channels = (nkv * D, nkv * D, H, H)
    fake = fake_vllm_config("none", None)
    with patched(state_mod, "get_current_vllm_config", lambda: fake):
        owner = state_mod.Inkling2ConvOwner(
            num_kv_heads=nkv, stream_channels=channels, kernel_size=k, prefix="t.conv"
        )
    world = PagedWorld([(owner, "conv")], {"conv": (conv_bs, k)}, torch.float32)
    C = channels[stream]
    gen = torch.Generator().manual_seed(0)
    paged_conv = attn_mod.StaticSconv(C, k)
    dense_conv = dense.ShortConvolution(C, k)
    with torch.no_grad():
        dense_conv.conv1d.weight.copy_(0.5 * torch.randn(C, 1, k, generator=gen))
        paged_conv.conv1d.weight.copy_(dense_conv.conv1d.weight)
    n, hit = 70, 64
    x_all = torch.randn(n, C, generator=gen)
    with torch.no_grad():
        want = dense_conv(x_all)
    world.fork_prefix(0, 1, hit, in_flight=True)
    work = [(0, n), (1, n - hit)]
    if hit_first:
        work.reverse()
    first = {0: 0, 1: hit}  # first position each request computes
    md = world.metadata(work)["t.conv"]
    x = torch.cat([x_all[first[r] :] for r, _ in work])
    with torch.no_grad():
        taps = state_mod.conv_taps(owner, stream, x, world.positions(work), md)
        y = paged_conv.forward_taps(taps)
    at = 0
    for r, count in work:
        err = float((y[at : at + count] - want[first[r] :]).abs().max())
        _report(f"same-step hit stream={stream} bs={conv_bs} request {r}", err)
        assert err < UNIT_TOL
        at += count


@pytest.mark.parametrize("window", [None, 16])
@pytest.mark.parametrize("kv_bs", [4, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_kv_write_read_round_trip(window, kv_bs, dtype):
    nkv, D = 2, 32
    fake = fake_vllm_config("none", None, dtype)
    with patched(state_mod, "get_current_vllm_config", lambda: fake):
        owner = state_mod.Inkling2KVOwner(
            num_kv_heads=nkv, head_dim=D, sliding_window=window, prefix="t.kv"
        )
    world = PagedWorld([(owner, "g")], {"g": (kv_bs, window)}, dtype, seed=3)
    gen = torch.Generator().manual_seed(7)
    lengths = [70, 23, 9]
    ks = [torch.randn(n, nkv, D, generator=gen) for n in lengths]
    vs = [torch.randn(n, nkv, D, generator=gen) for n in lengths]
    for step in schedule(random.Random(4), lengths):
        work = [(r, b - a) for r, a, b in step]
        md = world.metadata(work)["t.kv"]
        k = torch.cat([ks[r][a:b] for r, a, b in step])
        v = torch.cat([vs[r][a:b] for r, a, b in step])
        state_mod.kv_write(owner, k, v, md)
        world.commit(work)
        for i, (r, a, b) in enumerate(step):
            lo = max(0, a - window + 1) if window else 0  # what _attend reads
            kk, vv = state_mod.kv_read(owner, md, i, lo, b)
            assert kk.dtype == dtype
            assert torch.equal(kk, ks[r][lo:b].to(dtype))
            assert torch.equal(vv, vs[r][lo:b].to(dtype))


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
                p.copy_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _build_attention(kind: str, kv_bs: int, seed: int, conv_bs: int | None = None):
    cfg = text_config()
    layer = 0 if kind == "sliding" else 1
    assert cfg.is_sliding_layer(layer) == (kind == "sliding")
    oracle = (dense.SlidingAttention if kind == "sliding" else dense.FullAttention)(
        cfg, layer
    )
    _randomize(oracle, seed)
    fake = fake_vllm_config("none", None, torch.float32)
    with patched(state_mod, "get_current_vllm_config", lambda: fake):
        paged = attn_mod.Inkling2Attention(cfg, layer)
        nkv, D, H = paged.num_kv_heads, paged.head_dim, cfg.hidden_size
        kv_owner = state_mod.Inkling2KVOwner(
            num_kv_heads=nkv, head_dim=D, sliding_window=paged.window, prefix="t.kv"
        )
        k = cfg.sconv_kernel_size
        conv_owner = state_mod.Inkling2ConvOwner(
            num_kv_heads=nkv,
            stream_channels=(nkv * D, nkv * D, H, H),
            kernel_size=k,
            prefix="t.conv",
        )
    paged.load_state_dict(oracle.state_dict(), strict=True)  # same names / shapes
    paged.eval()
    world = PagedWorld(
        [(kv_owner, "kv"), (conv_owner, "conv")],
        {"kv": (kv_bs, paged.window), "conv": (conv_bs or k, k)},
        torch.float32,
        seed=seed,
    )
    return cfg, oracle, paged, kv_owner, conv_owner, world


@pytest.mark.parametrize("kind", ["sliding", "full"])
@pytest.mark.parametrize("kv_bs", [4, 64])
@pytest.mark.parametrize("seed", [0, 1])
def test_paged_attention_matches_dense_attention(kind, kv_bs, seed):
    cfg, oracle, paged, kv_owner, conv_owner, world = _build_attention(
        kind, kv_bs, seed
    )
    gen = torch.Generator().manual_seed(100 + seed)
    lengths = [70, 40, 5]  # crosses window 16 and rel_extent 32
    hs = [torch.randn(n, cfg.hidden_size, generator=gen) for n in lengths]
    with torch.no_grad():
        want = [oracle(h, torch.arange(len(h))) for h in hs]
    got: list[list[torch.Tensor]] = [[] for _ in lengths]
    for step in schedule(random.Random(seed), lengths):
        work = [(r, b - a) for r, a, b in step]
        mds = world.metadata(work)
        h = torch.cat([hs[r][a:b] for r, a, b in step])
        pos = world.positions(work)
        with torch.no_grad():
            out = paged(h, h, h, pos, kv_owner, conv_owner, mds["t.kv"], mds["t.conv"])
        world.commit(work)
        at = 0
        for r, a, b in step:
            got[r].append(out[at : at + b - a])
            at += b - a
    for r in range(len(lengths)):
        err = float((torch.cat(got[r]) - want[r]).abs().max())
        _report(f"attention {kind} bs={kv_bs} request {r}", err)
        assert err < UNIT_TOL


def test_attention_keeps_query_key_value_inputs_apart():
    """``hq`` / ``hk`` / ``hv`` (the message-augmented inputs of the looped layers)
    feed the q / k / v streams separately, like the oracle's
    ``key_hidden_states`` / ``value_hidden_states``."""
    cfg, oracle, paged, kv_owner, conv_owner, world = _build_attention("full", 16, 5)
    gen = torch.Generator().manual_seed(9)
    hq, hk, hv = (torch.randn(30, cfg.hidden_size, generator=gen) for _ in range(3))
    with torch.no_grad():
        want = oracle(
            hq, torch.arange(30), key_hidden_states=hk, value_hidden_states=hv
        )
    work = [(0, 30)]
    mds = world.metadata(work)
    with torch.no_grad():
        got = paged(
            hq,
            hk,
            hv,
            world.positions(work),
            kv_owner,
            conv_owner,
            mds["t.kv"],
            mds["t.conv"],
        )
    assert float((got - want).abs().max()) < UNIT_TOL
