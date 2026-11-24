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


def _conv_fixture(stream: int, conv_bs: int, drafts: int, seed: int):
    """A conv owner behind the engine emulation, the paged short conv of ``stream`` and
    its dense oracle with the same random weights, and the stream width."""
    cfg = text_config()
    nkv, D, H, k = cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, 4
    channels = (nkv * D, nkv * D, H, H)
    fake = fake_vllm_config(
        "none",
        None,
        torch.float32,
        block_size=conv_bs,
        num_speculative_tokens=drafts,
    )
    with patched(state_mod, "get_current_vllm_config", lambda: fake):
        owner = state_mod.Inkling2ConvOwner(
            num_kv_heads=nkv, stream_channels=channels, kernel_size=k, prefix="t.conv"
        )
    assert owner.ring_rows == k + drafts
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
    return owner, world, paged_conv, dense_conv, C, gen


@pytest.mark.parametrize("stream", [state_mod.STREAM_K, state_mod.STREAM_ATTN])
@pytest.mark.parametrize("conv_bs", [4, 8, 16])
@pytest.mark.parametrize("drafts", [0, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_conv_taps_matches_dense_short_conv(stream, conv_bs, drafts, seed):
    owner, world, paged_conv, dense_conv, C, gen = _conv_fixture(
        stream, conv_bs, drafts, seed
    )
    k = owner.kernel_size
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
@pytest.mark.parametrize("hit_blocks", [1, 2, 5])
def test_conv_taps_of_a_prefix_hit(stream, conv_bs, hit_blocks):
    """A hit starts on blocks another request computed (a step earlier; the manager
    defers same-step hits): its first tokens take the conv history from the ring rows
    at the end of the last hit block, which the owner wrote when it computed them."""
    owner, world, paged_conv, dense_conv, C, gen = _conv_fixture(stream, conv_bs, 0, 0)
    n, hit = 70, hit_blocks * conv_bs
    x_all = torch.randn(n, C, generator=gen)
    with torch.no_grad():
        want = dense_conv(x_all)
    cut = min(13, hit)
    for a, b in ((0, cut), (cut, hit)):  # the owner computes the hit in two chunks
        if b <= a:
            continue
        work = [(0, b - a)]
        md = world.metadata(work)["t.conv"]
        with torch.no_grad():
            state_mod.conv_taps(owner, stream, x_all[a:b], world.positions(work), md)
        world.commit(work)
    world.fork_prefix(0, 1, hit)
    work = [(1, n - hit)]
    md = world.metadata(work)["t.conv"]
    with torch.no_grad():
        taps = state_mod.conv_taps(
            owner, stream, x_all[hit:], world.positions(work), md
        )
        y = paged_conv.forward_taps(taps)
    err = float((y - want[hit:]).abs().max())
    _report(f"prefix hit stream={stream} bs={conv_bs} blocks={hit_blocks}", err)
    assert err < UNIT_TOL


def _verify_steps(stream, conv_bs, drafts, ring_drafts, seed):
    """Accepted-prefix semantics: prompt, then verify steps of ``1 + drafts`` tokens
    of which a random prefix is accepted; the rejected tokens carry junk inputs and
    are recomputed by the next step.  Returns the worst error of the accepted
    outputs against the dense short conv of the accepted sequence."""
    owner, world, paged_conv, dense_conv, C, gen = _conv_fixture(
        stream, conv_bs, ring_drafts, seed
    )
    rng = random.Random(seed)
    N = 90
    truth = torch.randn(N, C, generator=gen)
    with torch.no_grad():
        want = dense_conv(truth)
    worst, pos, first = 0.0, 0, True
    while pos < N:
        n_tok = 9 if first else 1 + drafts
        accepted = n_tok if first else rng.randint(1, n_tok)
        accepted = min(accepted, N - pos)
        x = torch.cat(
            [truth[pos : pos + accepted], 10 * torch.randn(n_tok - accepted, C)]
        )
        work = [(0, n_tok)]
        md = world.metadata(work)["t.conv"]
        with torch.no_grad():
            taps = state_mod.conv_taps(owner, stream, x, world.positions(work), md)
            y = paged_conv.forward_taps(taps)
        err = float((y[:accepted] - want[pos : pos + accepted]).abs().max())
        worst = max(worst, err)
        world.commit(work, accepted={0: accepted})
        pos += accepted
        first = False
    return worst


@pytest.mark.parametrize("stream", [state_mod.STREAM_K, state_mod.STREAM_MLP])
@pytest.mark.parametrize("conv_bs", [4, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conv_taps_after_rejected_draft_tokens(stream, conv_bs, seed):
    worst = _verify_steps(stream, conv_bs, drafts=4, ring_drafts=4, seed=seed)
    _report(f"rejected drafts stream={stream} bs={conv_bs}", worst)
    assert worst < UNIT_TOL


@pytest.mark.parametrize("conv_bs", [8, 16])
def test_a_ring_without_rows_for_the_drafts_loses_the_history(conv_bs):
    """Negative control: rows ``p % 4`` of a no-draft ring (``S = k < conv_bs``)
    are overwritten by the rejected tokens, so the same scenario must break
    (``S = k + drafts`` is what keeps the accepted prefix intact)."""
    worst = _verify_steps(state_mod.STREAM_K, conv_bs, drafts=4, ring_drafts=0, seed=0)
    assert worst > 1e-2, worst


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
    fake = fake_vllm_config(
        "none", None, torch.float32, block_size=conv_bs or cfg.sconv_kernel_size
    )
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
