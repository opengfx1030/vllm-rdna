# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel K2 with an expert map: the placement as data (gfx10x).

``mono_k2_rdna2`` takes an optional ``expert_map`` (global expert id -> row of
the rank's weights, -1 = not on this rank) in place of the contiguous block
``id - expert_base``, and an optional ``ids_out`` that receives the routed
global ids for the EPLB load counters. One Flash-Next TP4 / EP4 rank, the shapes
and helpers of ``test_amd_mono_rdna2_kernels``:

* a permuted placement with its matching map computes the same FFN as the
  linear placement (bit for bit: the routed sums are fixed point, so the row an
  expert sits in only reorders exact integer additions) and what the module path
  computes under the same map (the kernel suite's tolerance: 5e-3 per row except
  fp32 routing near-ties);
* an entry of -1 contributes nothing, like a non-owned expert of the linear
  placement;
* ``ids_out`` holds the routed ids of every row and feeds ``LayerStats.count``;
* neither option is the call of today (same bits, any width);
* a captured graph reads the live map and weights, and its ``ids_out`` feeds the
  counters on every replay.
"""

from types import SimpleNamespace

import pytest
import torch

from . import test_amd_mono_rdna2_kernels as kt
from .test_amd_mono_rdna2_kernels import (
    DEV,
    EL,
    EPS,
    HC,
    INTER,
    ISE,
    ND,
    TIE_GAP,
    TOPK,
    E,
    H,
    R,
)

pytestmark = kt.pytestmark

PAYLOAD = ("w13", "s13", "z13", "w2", "s2", "z2")
# every kernel width (MT 4, 8, 12, 24) at its edges
ROWS = (1, 3, 4, 5, 8, 9, 12, 13, 24)
OUTPUTS = ("res_out", "out", "inj_out")
# K2 rounds the router logits to half before its top-k (spacing 1e-3 .. 2e-3 at
# |logit| 1 .. 4): on a row whose top-10/11 gap is within two such spacings its
# ids can differ from the module path's, which compares fp32 logits
ID_TIE_GAP = 2.5e-3


@pytest.fixture(scope="module")
def ctx():
    """The kernel suite's layer (seed 11: the data of its ``ffn_layer``)."""
    gen = torch.Generator(device=DEV).manual_seed(11)
    L, G = kt._make_layer(gen, keep_golden=False)
    nb = int(torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(H, R, ND, E, ISE, INTER))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=DEV)
    return SimpleNamespace(L=L, G=G, ws=ws)


# ---- placements ------------------------------------------------------------


def _linear_map(base=0):
    """The contiguous block [base, base + EL) in rows 0..EL-1, -1 elsewhere."""
    emap = torch.full((E,), -1, dtype=torch.int32)
    emap[base : base + EL] = torch.arange(EL, dtype=torch.int32)
    return emap.to(DEV)


def _permuted(L, seed, base=0):
    """The block at ``base`` with its rows permuted: row s of the returned
    layer holds the weights of expert base + perm[s]; the map sends that id to
    s. The same function as the linear placement of ``L`` at ``base``."""
    perm = torch.randperm(EL, generator=torch.Generator().manual_seed(seed))
    emap = torch.full((E,), -1, dtype=torch.int32)
    emap[base + perm] = torch.arange(EL, dtype=torch.int32)
    rows = perm.to(DEV)
    Lp = dict(L)
    for name in PAYLOAD:
        Lp[name] = L[name][rows].contiguous()
    return Lp, emap.to(DEV)


def _scattered(seed):
    """A rank owning 128 random experts of the 512 (no block at all): the map
    sends owned[s] to row s of the layer's own weights."""
    owned = torch.randperm(E, generator=torch.Generator().manual_seed(seed))[:EL]
    emap = torch.full((E,), -1, dtype=torch.int32)
    emap[owned] = torch.arange(EL, dtype=torch.int32)
    return emap.to(DEV)


# ---- launches and references -----------------------------------------------


def _launch(L, ws, res, blk, inj, *tail, base=0):
    """One K2 launch through the op, the arguments of ``k2.run``; ``tail`` is
    (expert_map, ids_out) or a prefix of it. Returns (res_out, out, inj_out)."""
    res_out = torch.empty_like(res)
    inj_out = torch.empty((res.shape[0], HC), dtype=res.dtype, device=res.device)
    out = torch.empty_like(blk)
    torch.ops._rocm_C.mono_k2_rdna2(
        res,
        blk,
        inj.contiguous(),
        L["norm_w"],
        L["wd"],
        L["wu"],
        L["wr"],
        L["se_w1"],
        L["se_w2"],
        L["se_wg"],
        L["w13"],
        L["s13"],
        L["z13"],
        L["w2"],
        L["s2"],
        L["z2"],
        res_out,
        inj_out,
        out,
        ws,
        base,
        1,
        0,
        EPS,
        2000.0,
        False,
        False,
        False,
        *tail,
    )
    return res_out, out, inj_out


def _assert_same_bits(got, expected, label):
    for name, a, b in zip(OUTPUTS, got, expected, strict=True):
        kt._assert_repeat_identical(a, b, f"{label}: {name}")


def _assert_no_fault(ws):
    torch.accelerator.synchronize()
    err = torch.ops._rocm_C.mono_k2_rdna2_error(ws).tolist()
    assert err[0] == 0, f"sticky record {err[:3]}"


def _check_module(ctx, L, emap, got, res, blk, inj):
    """K2 against the module path under the same placement. Rows that differ
    must be fp32 routing near-ties (top-10/11 gap < 1e-3), as in the kernel
    suite's ``_check_k2``."""
    ro, out, io = got
    o_ref, ro_ref, io_ref = kt._today_ffn(L, res, blk, inj, emap=emap)
    torch.accelerator.synchronize()
    rowd = (out.float() - o_ref.float()).abs().max(1).values
    gaps = kt._golden_gaps(ctx.G, res, blk, inj)
    bad = [m for m in range(res.shape[0]) if rowd[m] >= 5e-3 and gaps[m] >= TIE_GAP]
    assert not bad, f"rows {bad}: max|out-ref| {rowd.max().item():.2e}"
    assert torch.equal(ro, ro_ref)
    assert (io.float() - io_ref.float()).abs().max().item() < 5e-3


# ---- (a) permuted placements ------------------------------------------------


@pytest.mark.parametrize("base", [0, 200], ids=["block-0", "block-200"])
@pytest.mark.parametrize("M", ROWS)
def test_permuted_placement_matches_linear(ctx, M, base):
    res, blk, inj = kt._inputs(M, 100 + M)
    linear = _launch(ctx.L, ctx.ws, res, blk, inj, base=base)
    Lp, emap = _permuted(ctx.L, seed=7 + M, base=base)
    permuted = _launch(Lp, ctx.ws, res, blk, inj, emap)
    _assert_same_bits(permuted, linear, f"M={M} base={base}")
    _check_module(ctx, Lp, emap, permuted, res, blk, inj)
    _assert_no_fault(ctx.ws)


@pytest.mark.parametrize("M", (3, 8, 12, 24))
def test_scattered_ownership_matches_module_path(ctx, M):
    """No block at all: the placement is any 128 of the 512 experts."""
    res, blk, inj = kt._inputs(M, 100 + M)
    emap = _scattered(seed=3)
    got = _launch(ctx.L, ctx.ws, res, blk, inj, emap)
    _check_module(ctx, ctx.L, emap, got, res, blk, inj)
    _assert_no_fault(ctx.ws)


# ---- (b) -1 entries --------------------------------------------------------


@pytest.mark.parametrize("M", ROWS)
def test_minus_one_is_a_non_owned_expert(ctx, M):
    """The linear block as a map (-1 everywhere else) is the linear placement."""
    base = 100
    res, blk, inj = kt._inputs(M, 100 + M)
    linear = _launch(ctx.L, ctx.ws, res, blk, inj, base=base)
    mapped = _launch(ctx.L, ctx.ws, res, blk, inj, _linear_map(base))
    _assert_same_bits(mapped, linear, f"M={M}")
    _assert_no_fault(ctx.ws)


@pytest.mark.parametrize("M", (8, 24))
def test_unmapped_routed_expert_contributes_nothing(ctx, M):
    res, blk, inj = kt._inputs(M, 100 + M)
    emap = _linear_map(0)
    ids = torch.zeros((M, TOPK), dtype=torch.int32, device=DEV)
    full = _launch(ctx.L, ctx.ws, res, blk, inj, emap, ids)
    routed = ids.cpu()
    owned = sorted({int(e) for e in routed.flatten() if e < EL})
    assert len(owned) >= 2, f"rows route to {len(owned)} local experts"
    gone = owned[0]
    hole = emap.clone()
    hole[gone] = -1
    got = _launch(ctx.L, ctx.ws, res, blk, inj, hole)
    hit = (routed == gone).any(1)
    d = (got[1].float() - full[1].float()).abs().max(1).values.cpu()
    # rows that never routed to it keep their output (the K split of the
    # fixed-point sums follows the used-expert count: fp16-level tolerance)
    assert (d[~hit] < 5e-3).all(), f"unaffected rows moved: {d[~hit].tolist()}"
    # rows that did lose its term
    assert hit.any() and (d[hit] > 1e-3).all(), f"hit rows barely moved: {d[hit]}"
    _check_module(ctx, ctx.L, hole, got, res, blk, inj)
    _assert_no_fault(ctx.ws)


# ---- (c) ids_out ------------------------------------------------------------


@pytest.mark.parametrize("M", ROWS)
def test_ids_out_is_the_routed_ids(ctx, M):
    res, blk, inj = kt._inputs(M, 100 + M)
    emap = _scattered(seed=5)  # ids are global ids whatever the rank owns
    ids_out = torch.full((24, TOPK), -7, dtype=torch.int32, device=DEV)
    got = _launch(ctx.L, ctx.ws, res, blk, inj, emap, ids_out)
    *_, tids, _ = kt._today_ffn(ctx.L, res, blk, inj, return_routes=True, emap=emap)
    torch.accelerator.synchronize()
    mine, ref = ids_out[:M].cpu(), tids.cpu().to(torch.int32)
    gaps = kt._golden_gaps(ctx.G, res, blk, inj)
    for m in range(M):
        assert len(set(mine[m].tolist())) == TOPK, f"row {m}: {mine[m].tolist()}"
        if gaps[m] >= ID_TIE_GAP:
            assert sorted(mine[m].tolist()) == sorted(ref[m].tolist()), f"row {m}"
    assert (mine >= 0).all() and (mine < E).all()
    # K2's slot order is expert order (the sums are fixed point)
    assert (mine[:, 1:] > mine[:, :-1]).all()
    assert (ids_out[M:] == -7).all(), "rows past M must stay untouched"
    # and the FFN is the same as without ids_out
    _assert_same_bits(got, _launch(ctx.L, ctx.ws, res, blk, inj, emap), f"M={M}")
    _assert_no_fault(ctx.ws)


def test_ids_out_and_expert_map_are_independent(ctx):
    M = 8
    res, blk, inj = kt._inputs(M, 100 + M)
    default = _launch(ctx.L, ctx.ws, res, blk, inj)
    ids_only = torch.zeros((M, TOPK), dtype=torch.int32, device=DEV)
    _assert_same_bits(
        _launch(ctx.L, ctx.ws, res, blk, inj, None, ids_only), default, "ids only"
    )
    both = torch.zeros((M, TOPK), dtype=torch.int32, device=DEV)
    _assert_same_bits(
        _launch(ctx.L, ctx.ws, res, blk, inj, _linear_map(0), both), default, "both"
    )
    _assert_same_bits(
        _launch(ctx.L, ctx.ws, res, blk, inj, _linear_map(0)), default, "map only"
    )
    assert torch.equal(ids_only, both)
    _assert_no_fault(ctx.ws)


# ---- (d) the default call ---------------------------------------------------


@pytest.mark.parametrize("M", range(1, 25))
def test_default_call_is_unchanged(ctx, M):
    """Neither option is the Stage 0 kernel: ``k2.run`` of a layer without map
    mode, the op with explicit Nones and the identity map are the same bits."""
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    res, blk, inj = kt._inputs(M, 100 + M)
    default = _launch(ctx.L, ctx.ws, res, blk, inj)
    mono = SimpleNamespace(
        t=ctx.L,
        ws=ctx.ws,
        base=0,
        eps=EPS,
        th=ctx.L,
        eps_a=EPS,
        is_mtp=False,
        dense_mtp=False,
    )
    _assert_same_bits(k2.run(mono, res, blk, inj), default, f"k2.run M={M}")
    _assert_same_bits(
        _launch(ctx.L, ctx.ws, res, blk, inj, None, None), default, f"None M={M}"
    )
    _assert_same_bits(
        _launch(ctx.L, ctx.ws, res, blk, inj, _linear_map(0)), default, f"id M={M}"
    )
    _assert_no_fault(ctx.ws)


# ---- arguments the launch refuses -------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        "map-dtype",
        "map-shape",
        "map-device",
        "ids-dtype",
        "ids-width",
        "ids-rows",
        "ids-stride",
    ],
)
def test_bad_arguments_are_refused_before_launch(ctx, case):
    M = 4
    res, blk, inj = kt._inputs(M, 1)
    emap = _linear_map(0)
    ids = torch.zeros((M, TOPK), dtype=torch.int32, device=DEV)
    if case == "map-dtype":
        emap = emap.long()
    elif case == "map-shape":
        emap = emap[: E - 1].contiguous()
    elif case == "map-device":
        emap = emap.cpu()
    elif case == "ids-dtype":
        ids = ids.long()
    elif case == "ids-width":
        ids = torch.zeros((M, TOPK - 1), dtype=torch.int32, device=DEV)
    elif case == "ids-rows":
        ids = torch.zeros((M - 1, TOPK), dtype=torch.int32, device=DEV)
    else:
        ids = torch.zeros((TOPK, M), dtype=torch.int32, device=DEV).t()
    with pytest.raises(RuntimeError, match="expert_map must be|ids_out must be"):
        _launch(ctx.L, ctx.ws, res, blk, inj, emap, ids)
    _assert_no_fault(ctx.ws)


# ---- graphs and the counters -------------------------------------------------


def _stats_layer(emap, name="model.layers.3.mlp.experts"):
    """The attributes of a RoutedExperts the counters register a layer by."""
    cfg = SimpleNamespace(ep_size=4, ep_rank=0, tp_rank=0, dp_rank=0)
    return SimpleNamespace(
        layer_name=name,
        moe_config=cfg,
        global_num_experts=E,
        expert_map=emap,
        expert_placement_strategy="linear",
    )


def _map_mono(ctx, L, layer):
    """The MonoLayer fields ``k2.run`` reads in map mode (layout._setup_map)."""
    return SimpleNamespace(
        t=L,
        ws=ctx.ws,
        base=0,
        eps=EPS,
        th=L,
        eps_a=EPS,
        is_mtp=False,
        dense_mtp=False,
        map_mode=True,
        experts=layer,
        ids_out=torch.zeros((24, TOPK), dtype=torch.int32, device=DEV),
    )


def _capture(k2, mono, res, blk, inj):
    """Warm up on a side stream, then capture one ``k2.run``; (graph, outputs)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        k2.run(mono, res, blk, inj)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = k2.run(mono, res, blk, inj)
    return graph, got


@pytest.mark.parametrize("M", (3, 8, 24))
def test_k2_ids_feed_the_counters_eager_and_in_graph(ctx, M, tmp_path, monkeypatch):
    from vllm.model_executor.layers.fused_moe import rdna_eplb_stats as es
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    stats = es.EplbStats(str(tmp_path), decode_max=24, ring=64)
    monkeypatch.setattr(k2, "EPLB_STATS", stats)
    Lp, emap = _permuted(ctx.L, seed=11)
    layer = _stats_layer(emap)
    mono = _map_mono(ctx, Lp, layer)
    res, blk, inj = kt._inputs(M, 100 + M)

    def routed():  # the ids K2 wrote last, as per-expert pair counts
        return torch.bincount(mono.ids_out[:M].flatten().long().cpu(), minlength=E)

    def counted():  # decode-phase pairs per expert of the layer's counters
        torch.accelerator.synchronize()
        return ls.state[: 2 * E].view(2, E)[0].cpu()

    got = k2.run(mono, res, blk, inj)  # eager: registers the layer, counts once
    ls = stats.layer_stats(layer, mono.ids_out[:1])
    assert ls is not None
    _check_module(ctx, Lp, emap, got, res, blk, inj)
    expect = routed()
    assert torch.equal(counted(), expect)

    graph, got = _capture(k2, mono, res, blk, inj)  # the warm-up counted once more
    expect = expect + routed()
    assert torch.equal(counted(), expect), "capture must record, not count"
    for t, n in zip((res, blk, inj), kt._inputs(M, 200 + M), strict=True):
        t.copy_(n)
    for replay in range(2):  # new inputs: the ids and the counters follow
        graph.replay()
        expect = expect + routed()
        assert torch.equal(counted(), expect), f"replay {replay}"
    _check_module(ctx, Lp, emap, got, res, blk, inj)
    assert ls.state[ls.off.ordinal].item() == 4
    _assert_no_fault(ctx.ws)


@pytest.mark.parametrize("M", (8, 24))
def test_graph_reads_the_live_map_and_weights(ctx, M, monkeypatch):
    """A rebalancer's commit rewrites map and weight rows in place; the graph
    holds addresses only, so its next replay must follow."""
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    counts: list = []
    collector = SimpleNamespace(
        layer_stats=lambda layer, ids: SimpleNamespace(count=counts.append)
    )
    monkeypatch.setattr(k2, "EPLB_STATS", collector)
    Lp, emap = _permuted(ctx.L, seed=13)
    mono = _map_mono(ctx, Lp, SimpleNamespace(expert_map=emap))
    res, blk, inj = kt._inputs(M, 100 + M)
    graph, got = _capture(k2, mono, res, blk, inj)
    graph.replay()
    torch.accelerator.synchronize()
    before = [x.clone() for x in got]
    ids_before = mono.ids_out[:M].clone()
    assert counts and counts[-1].data_ptr() == mono.ids_out.data_ptr()

    host = emap.cpu()
    owned = sorted({int(e) for e in ids_before.cpu().flatten() if host[e] >= 0})
    assert len(owned) >= 2, f"rows route to {len(owned)} owned experts"
    a, b = owned[0], owned[1]
    ra, rb = int(host[a]), int(host[b])

    def swap_rows():
        for name in PAYLOAD:
            w = Lp[name]
            tmp = w[ra].clone()
            w[ra].copy_(w[rb])
            w[rb].copy_(tmp)

    swap_rows()  # weights and map move together: the same function
    emap[a], emap[b] = rb, ra
    graph.replay()
    torch.accelerator.synchronize()
    _assert_same_bits(got, before, "after a committed swap")
    assert torch.equal(mono.ids_out[:M], ids_before)

    swap_rows()  # the rows move, the map does not: a different function
    graph.replay()
    torch.accelerator.synchronize()
    assert not torch.equal(got[1], before[1]), "the graph did not read the weights"

    emap[a], emap[b] = ra, rb  # the map follows: the start state again
    graph.replay()
    torch.accelerator.synchronize()
    _assert_same_bits(got, before, "after restoring")
    _assert_no_fault(ctx.ws)
