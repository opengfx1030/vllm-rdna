# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused attention-chain kernels of Inkling-2-Mini
(``csrc/rocm/inkling2_fuse_attn_rdna2.cu``) against the launches they replace.

The fused kernels copy the arithmetic of the shipped ones, so every comparison is
BIT-exact (``torch.equal`` on the raw bits, -0.0 / NaN payloads included):

* ``inkling2_fuse_attn_res_conv`` == ``inkling2_sconv_static(write_state=True)`` then
  ``inkling2_glue_attn_res`` (outputs, partial / appended states, ``x1_out`` and the
  conv state after the step)
* ``inkling2_fuse_msgs_fill`` == one ``inkling2_glue_attn_res`` per layer

and, independently of the shipped kernels, the conv + attn_res math against an fp64
torch oracle.  ``I2_FUSE_JIT=<path to a .cu>`` runs the tests on a JIT build of that
file (``tools/rdna/inkling2/fuse_attn_jit.py``): how the hand mutants of
``tools/rdna/inkling2/fuse_attn_mutants.py`` are run.
"""

from __future__ import annotations

import hashlib
import math
import os
import pathlib
import sys

import pytest
import torch

from vllm import _custom_ops as ref_ops
from vllm.platforms.rocm import on_gfx10x

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_gfx10x()),
    reason="needs gfx1030",
)

DEV = "cuda"
EPS = 1e-6
H = 2304  # hidden size of Inkling-2-Mini
NKV = 8  # kv heads: the short convs of the residual stream have H / 8 = 288 channels
WIDTH = H // NKV
COLS = 832  # state row: k 128 | v 128 | attn 288 | mlp 288
OFFSET = 256  # post-attention stream
S_MAX = 10  # block states of the persistent buffer (n_blocks + append + slack)


class _FuseOps:
    """The ops under test: the installed ``_rocm_C`` build, or a JIT build of
    ``I2_FUSE_JIT`` (hand mutants)."""

    def __init__(self) -> None:
        source = os.environ.get("I2_FUSE_JIT", "")
        root = pathlib.Path(__file__).resolve().parents[3]
        if source or not hasattr(torch.ops._rocm_C, "inkling2_fuse_attn_res_conv"):
            sys.path.insert(0, str(root / "tools"))
            from rdna.inkling2 import fuse_attn_jit

            path = pathlib.Path(source) if source else fuse_attn_jit.SOURCE
            tag = hashlib.md5(path.read_bytes()).hexdigest()[:10]
            mod = fuse_attn_jit.build("i2_fuse_test_" + tag, path)
            self.attn_res_conv = mod.attn_res_conv
            self.msgs_fill = mod.msgs_fill
        else:
            self.attn_res_conv = torch.ops._rocm_C.inkling2_fuse_attn_res_conv
            self.msgs_fill = torch.ops._rocm_C.inkling2_fuse_msgs_fill


ops = _FuseOps()


def bits(t: torch.Tensor) -> torch.Tensor:
    """The raw bits of a tensor (fp32 -> int32, fp16 -> int16, int32 as is)."""
    t = t.contiguous().view(-1)
    return t.view({4: torch.int32, 2: torch.int16}[t.element_size()]).cpu()


def assert_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    g, w = bits(got), bits(want)
    assert g.shape == w.shape, f"{what}: shapes {g.shape} vs {w.shape}"
    bad = (g != w).nonzero()
    assert bad.numel() == 0, f"{what}: {bad.shape[0]} of {g.numel()} words differ"


class World:
    """One layer's paged conv state and the decode metadata of ``R`` single-token
    requests (+ ``pad`` graph padding tokens that the metadata maps to no request)."""

    def __init__(self, R, pad, md_short, bs, ring, seed, seq_lens=None, null_reqs=()):
        g = torch.Generator().manual_seed(seed)
        self.R, self.pad, self.bs, self.ring = R, pad, bs, ring
        self.T = R + pad
        if seq_lens is None:
            # 1..3 (start of a request: history short of the 3 taps), a block edge,
            # the first row of the next block and longer contexts
            pool = [1, 2, 3, 4, 5, bs - 1, bs, bs + 1, bs + 2, 2 * bs + 1, 3 * bs - 1]
            seq_lens = [
                pool[int(i)] for i in torch.randint(0, len(pool), (R,), generator=g)
            ]
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        nb = max(1, -(-int(self.seq_lens.max()) // bs))
        n_pages = R * nb + 1  # page 0 is the null block
        ids = 1 + torch.randperm(R * nb, generator=g)
        self.table = ids.view(R, nb).to(torch.int32)
        for r in null_reqs:  # dummy batch of a graph capture: zeroed block table
            self.table[r] = 0
        self.cache0 = torch.randn(n_pages, NKV, ring, COLS, generator=g).mul(2.0).half()
        self.qsl = torch.arange(R + 1, dtype=torch.int32)
        t_md = self.T - md_short
        seq_idx = torch.full((t_md,), -1, dtype=torch.int32)
        seq_idx[: min(R, t_md)] = torch.arange(min(R, t_md), dtype=torch.int32)
        self.seq_idx = seq_idx
        self.x = torch.randn(self.T, H, generator=g).mul(1.5).half()
        self.w = torch.randn(H, 1, 4, generator=g).mul(0.7)

    def meta(self):
        return (
            self.table.to(DEV),
            self.seq_idx.to(DEV),
            self.qsl.to(DEV),
            self.seq_lens.to(DEV),
        )


def make_states(T, n_blocks, with_append, seed):
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randn(T, S_MAX, H, generator=g).mul(2.0)
    return blocks.to(DEV)


def rand_f32(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).mul(scale).to(DEV)


def ref_chain(
    world,
    cache,
    x16,
    conv_w,
    blocks,
    n_blocks,
    x0,
    query,
    out,
    *,
    norm_w,
    partial_out,
    append,
    x1_out,
):
    """The two shipped launches: conv (+ state write) into ``x1_out``, then attn_res."""
    table, seq_idx, qsl, seq_lens = world.meta()
    ref_ops.inkling2_sconv_static(
        x16, conv_w, x1_out, cache, table, seq_idx, qsl, seq_lens, OFFSET, NKV,
        world.bs, True,
    )  # fmt: skip
    ref_ops.inkling2_glue_attn_res(
        blocks, n_blocks, x0 if x0 is not None else x1_out, query, out,
        x1=x1_out if x0 is not None else None, partial_out=partial_out,
        append=append, norm_w=norm_w, eps=EPS, norm_eps=EPS,
    )  # fmt: skip


def run_fused(
    world,
    cache,
    x16,
    conv_w,
    blocks,
    n_blocks,
    x0,
    query,
    out,
    *,
    norm_w,
    partial_out,
    append,
    x1_out,
):
    table, seq_idx, qsl, seq_lens = world.meta()
    ops.attn_res_conv(
        blocks, n_blocks, x0, x16, conv_w, cache, table, seq_idx, qsl, seq_lens,
        OFFSET, NKV, world.bs, x1_out, partial_out, append, query, norm_w, out,
        EPS, EPS,
    )  # fmt: skip


GEOMETRIES = [(64, 3), (4, 3), (16, 10)]  # (conv block size, ring rows)
WORLDS = [(1, 0, 0), (3, 0, 0), (8, 3, 0), (5, 3, 3), (16, 0, 0)]  # R, pad, md_short


@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize("R,pad,md_short", WORLDS)
@pytest.mark.parametrize("Q,norm", [(1, True), (1, False), (3, True), (3, False)])
@pytest.mark.parametrize("n_blocks", [0, 1, 3, 7])
def test_conv_plus_x0(bs, ring, R, pad, md_short, Q, norm, n_blocks):
    """Partial state = x0 + conv(x): the attention-input attn_res of the next visit
    (Q = 1 / 3, norm / raw) and the non-boundary MLP-input attn_res (Q = 1)."""
    world = World(R, pad, md_short, bs, ring, seed=11 + R)
    T = world.T
    x16 = world.x.to(DEV)
    cw = world.w.to(DEV)
    blocks_a = make_states(T, n_blocks, False, 5)
    blocks_b = blocks_a.clone()
    x0 = rand_f32((T, H), 6)
    query = rand_f32((Q, H), 7, 0.5)
    nw = (rand_f32((Q, H), 8, 0.3) + 1.0).half() if norm else None
    odt = torch.float16 if norm else torch.float32
    out_a = torch.zeros(T, Q, H, dtype=odt, device=DEV)
    out_b = torch.zeros_like(out_a)
    po_a = torch.zeros(T, H, device=DEV)
    po_b = torch.zeros_like(po_a)
    cache_a = world.cache0.to(DEV)
    cache_b = world.cache0.to(DEV)
    x1_a = torch.zeros(T, H, device=DEV)
    ref_chain(world, cache_a, x16, cw, blocks_a, n_blocks, x0, query, out_a,
              norm_w=nw, partial_out=po_a, append=None, x1_out=x1_a)  # fmt: skip
    run_fused(world, cache_b, x16, cw, blocks_b, n_blocks, x0, query, out_b,
              norm_w=nw, partial_out=po_b, append=None, x1_out=None)  # fmt: skip
    torch.accelerator.synchronize()
    assert_bits(out_b, out_a, "out")
    assert_bits(po_b, po_a, "partial_out")
    assert_bits(cache_b, cache_a, "conv state")
    assert_bits(blocks_b, blocks_a, "block states")


@pytest.mark.parametrize("bs,ring", GEOMETRIES[:2])
@pytest.mark.parametrize("R,pad,md_short", WORLDS)
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("n_blocks", [0, 2, 5])
def test_conv_only_boundary(bs, ring, R, pad, md_short, norm, n_blocks):
    """Block boundary: the conv result alone is the partial state, the layer input
    is appended as a block state and the conv result is kept (``x1_out`` = attn_c)."""
    world = World(R, pad, md_short, bs, ring, seed=21 + R)
    T = world.T
    x16 = world.x.to(DEV)
    cw = world.w.to(DEV)
    blocks_a = make_states(T, n_blocks, True, 9)
    blocks_b = blocks_a.clone()
    hid = rand_f32((T, H), 10)
    query = rand_f32((1, H), 11, 0.5)
    nw = (rand_f32((1, H), 12, 0.3) + 1.0).half() if norm else None
    odt = torch.float16 if norm else torch.float32
    out_a = torch.zeros(T, 1, H, dtype=odt, device=DEV)
    out_b = torch.zeros_like(out_a)
    cache_a = world.cache0.to(DEV)
    cache_b = world.cache0.to(DEV)
    c_a = torch.full((T, H), 3.0, device=DEV)
    c_b = torch.full((T, H), 3.0, device=DEV)
    ref_chain(world, cache_a, x16, cw, blocks_a, n_blocks, None, query, out_a,
              norm_w=nw, partial_out=None, append=hid, x1_out=c_a)  # fmt: skip
    run_fused(world, cache_b, x16, cw, blocks_b, n_blocks, None, query, out_b,
              norm_w=nw, partial_out=None, append=hid, x1_out=c_b)  # fmt: skip
    torch.accelerator.synchronize()
    assert_bits(out_b, out_a, "out")
    assert_bits(c_b, c_a, "x1_out (attn_c)")
    assert_bits(cache_b, cache_a, "conv state")
    assert_bits(blocks_b, blocks_a, "block states (appended slot)")


@pytest.mark.parametrize("Q", [1, 3])
@pytest.mark.parametrize("nw_dtype", [torch.float16, torch.float32])
def test_norm_weight_dtypes(nw_dtype, Q):
    """fp16 and fp32 norm weights (Q = 1 keeps them in registers)."""
    world = World(4, 1, 0, 64, 3, seed=31)
    T = world.T
    x16, cw = world.x.to(DEV), world.w.to(DEV)
    blocks_a = make_states(T, 2, False, 3)
    blocks_b = blocks_a.clone()
    x0 = rand_f32((T, H), 4)
    query = rand_f32((Q, H), 5, 0.5)
    nw = (rand_f32((Q, H), 6, 0.3) + 1.0).to(nw_dtype)
    out_a = torch.zeros(T, Q, H, dtype=torch.float16, device=DEV)
    out_b = torch.zeros_like(out_a)
    po_a = torch.zeros(T, H, device=DEV)
    po_b = torch.zeros_like(po_a)
    cache_a, cache_b = world.cache0.to(DEV), world.cache0.to(DEV)
    ref_chain(
        world,
        cache_a,
        x16,
        cw,
        blocks_a,
        2,
        x0,
        query,
        out_a,
        norm_w=nw,
        partial_out=po_a,
        append=None,
        x1_out=torch.zeros(T, H, device=DEV),
    )
    run_fused(
        world,
        cache_b,
        x16,
        cw,
        blocks_b,
        2,
        x0,
        query,
        out_b,
        norm_w=nw,
        partial_out=po_b,
        append=None,
        x1_out=None,
    )
    torch.accelerator.synchronize()
    assert_bits(out_b, out_a, "out")
    assert_bits(po_b, po_a, "partial_out")


def conv_oracle(world, x16, cw, cache0):
    """fp64 oracle of the post-conv: (sum_lag x[t - lag] w[3 - lag] + x[t]) / sqrt(5);
    returns ([T, H] fp64, the conv state a decode step leaves)."""
    out = torch.zeros(world.T, H, dtype=torch.float64)
    cache = cache0.clone()
    w = cw.cpu().double().view(H, 4)
    x = x16.cpu().double()
    for t in range(world.T):
        if t >= world.seq_idx.numel() or int(world.seq_idx[t]) < 0:
            continue
        r = int(world.seq_idx[t])
        p1 = int(world.seq_lens[r])
        p = p1 - 1
        taps = [x[t]]
        for lag in range(1, 4):
            sp = p - lag
            if sp < 0:
                taps.append(torch.zeros(H, dtype=torch.float64))
                continue
            blk = int(world.table[r, sp // world.bs])
            row = cache0[blk, :, sp % world.ring, OFFSET : OFFSET + WIDTH]  # [NKV, W]
            taps.append(row.double().reshape(H))
        acc = sum(taps[lag] * w[:, 3 - lag] for lag in range(4))
        out[t] = (acc + taps[0]) / math.sqrt(5.0)
        blk = int(world.table[r, p // world.bs])
        if blk > 0:
            cache[blk, :, p % world.ring, OFFSET : OFFSET + WIDTH] = (
                x[t].view(NKV, WIDTH).half()
            )
    return out, cache


@pytest.mark.parametrize("bs,ring", GEOMETRIES)
@pytest.mark.parametrize("R,pad,md_short", WORLDS)
def test_conv_vs_fp64_oracle(bs, ring, R, pad, md_short):
    """The conv inside the kernel (boundary shape: x1_out = the conv result) and the
    state it leaves against an independent fp64 torch implementation."""
    world = World(R, pad, md_short, bs, ring, seed=41 + R)
    T = world.T
    x16, cw = world.x.to(DEV), world.w.to(DEV)
    want, want_cache = conv_oracle(world, world.x, world.w, world.cache0)
    blocks = make_states(T, 0, True, 2)
    hid = rand_f32((T, H), 3)
    query = rand_f32((1, H), 4, 0.5)
    out = torch.zeros(T, 1, H, device=DEV)
    cache = world.cache0.to(DEV)
    c = torch.full((T, H), float("nan"), device=DEV)
    run_fused(world, cache, x16, cw, blocks, 0, None, query, out, norm_w=None,
              partial_out=None, append=hid, x1_out=c)  # fmt: skip
    torch.accelerator.synchronize()
    torch.testing.assert_close(c.cpu().double(), want, rtol=2e-6, atol=2e-6)
    assert_bits(cache, want_cache.to(DEV), "conv state vs oracle")


def attn_res_oracle(states, query, norm_w, eps, norm_eps):
    """fp64 block attention residual: states [S, H] -> out [Q, H] (or normed)."""
    s = states.double()
    r = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    n = s * r
    sc = (query.double() @ n.T) * (H**-0.5)
    p = torch.softmax(sc, dim=-1)
    out = p @ n
    if norm_w is not None:
        rms = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + norm_eps)
        out = out * rms * norm_w.double()
    return out


@pytest.mark.parametrize("norm", [True, False])
def test_attn_res_part_vs_fp64_oracle(norm):
    """End to end against an fp64 oracle of conv + residual add + block attention."""
    world = World(3, 1, 0, 64, 3, seed=51)
    T = world.T
    n_blocks = 3
    x16, cw = world.x.to(DEV), world.w.to(DEV)
    conv, _ = conv_oracle(world, world.x, world.w, world.cache0)
    blocks = make_states(T, n_blocks, False, 2)
    x0 = rand_f32((T, H), 3)
    query = rand_f32((1, H), 4, 0.5)
    nw = (rand_f32((1, H), 5, 0.3) + 1.0).half() if norm else None
    out = torch.zeros(
        T, 1, H, dtype=torch.float16 if norm else torch.float32, device=DEV
    )
    po = torch.zeros(T, H, device=DEV)
    cache = world.cache0.to(DEV)
    run_fused(world, cache, x16, cw, blocks.clone(), n_blocks, x0, query, out,
              norm_w=nw, partial_out=po, append=None, x1_out=None)  # fmt: skip
    torch.accelerator.synchronize()
    for t in range(world.R):  # real rows
        part = x0[t].cpu().double() + conv[t]
        states = torch.cat([blocks[t, :n_blocks].cpu().double(), part[None]], dim=0)
        want = attn_res_oracle(
            states, query.cpu(), None if nw is None else nw.cpu(), EPS, EPS
        )
        torch.testing.assert_close(po[t].cpu().double(), part, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            out[t, 0].cpu().double(), want[0],
            rtol=2e-3 if norm else 2e-5, atol=2e-3 if norm else 2e-5,
        )  # fmt: skip


@pytest.mark.parametrize("L", [1, 3, 12])
@pytest.mark.parametrize("T", [1, 2, 5, 16])
@pytest.mark.parametrize("n_prev", [1, 2, 5, 7, 11])
def test_msgs_fill(L, T, n_prev):
    """One launch for the messages of ``L`` layers == ``L`` shipped attn_res calls."""
    g = torch.Generator().manual_seed(61 + L)
    s_max = max(n_prev, 1) + 1
    blocks = torch.randn(T, s_max, H, generator=g).mul(2.0).to(DEV)
    x0 = torch.randn(T, H, generator=g).to(DEV)
    x1 = torch.randn(T, H, generator=g).to(DEV)
    queries = torch.randn(L, 4, H, generator=g).mul(0.5).to(DEV)
    out = torch.full((T, L, 4, H), float("nan"), device=DEV)
    ops.msgs_fill(blocks, n_prev, x0, x1, queries, out, EPS)
    torch.accelerator.synchronize()
    for j in range(L):
        want = torch.zeros(T, 4, H, device=DEV)
        ref_ops.inkling2_glue_attn_res(
            blocks, n_prev, x0, queries[j], want, x1=x1, eps=EPS
        )
        torch.accelerator.synchronize()
        assert_bits(out[:, j], want, f"messages of layer {j}")


def test_msgs_fill_strided_batch():
    """The decode graph fills the first T rows of a larger persistent buffer."""
    T, L, n_prev = 3, 4, 3
    g = torch.Generator().manual_seed(71)
    blocks = torch.randn(T, 4, H, generator=g).to(DEV)
    x0 = torch.randn(T, H, generator=g).to(DEV)
    x1 = torch.randn(T, H, generator=g).to(DEV)
    queries = torch.randn(L, 4, H, generator=g).mul(0.5).to(DEV)
    big = torch.full((16, L, 4, H), 5.0, device=DEV)
    ops.msgs_fill(blocks, n_prev, x0, x1, queries, big[:T], EPS)
    torch.accelerator.synchronize()
    assert bool((big[T:] == 5.0).all()), "rows past T were written"
    for j in range(L):
        want = torch.zeros(T, 4, H, device=DEV)
        ref_ops.inkling2_glue_attn_res(
            blocks, n_prev, x0, queries[j], want, x1=x1, eps=EPS
        )
        assert_bits(big[:T, j], want, f"layer {j}")


@pytest.mark.parametrize("boundary", [False, True])
def test_null_block_requests(boundary):
    """Requests with a zeroed block table (the dummy batches of a graph capture and of
    the memory profile) write nothing: block 0 is the null page, not a state page."""
    world = World(5, 2, 0, 16, 3, seed=81, null_reqs=(1, 3))
    T = world.T
    x16, cw = world.x.to(DEV), world.w.to(DEV)
    blocks_a = make_states(T, 2, boundary, 3)
    blocks_b = blocks_a.clone()
    x0 = None if boundary else rand_f32((T, H), 4)
    hid = rand_f32((T, H), 5)
    query = rand_f32((1, H), 6, 0.5)
    nw = (rand_f32((1, H), 7, 0.3) + 1.0).half()
    out_a = torch.zeros(T, 1, H, dtype=torch.float16, device=DEV)
    out_b = torch.zeros_like(out_a)
    po_a = None if boundary else torch.zeros(T, H, device=DEV)
    po_b = None if boundary else torch.zeros(T, H, device=DEV)
    cache_a, cache_b = world.cache0.to(DEV), world.cache0.to(DEV)
    c_a, c_b = torch.zeros(T, H, device=DEV), torch.zeros(T, H, device=DEV)
    ref_chain(
        world,
        cache_a,
        x16,
        cw,
        blocks_a,
        2,
        x0,
        query,
        out_a,
        norm_w=nw,
        partial_out=po_a,
        x1_out=c_a,
        append=hid if boundary else None,
    )
    run_fused(
        world,
        cache_b,
        x16,
        cw,
        blocks_b,
        2,
        x0,
        query,
        out_b,
        norm_w=nw,
        partial_out=po_b,
        x1_out=c_b if boundary else None,
        append=hid if boundary else None,
    )
    torch.accelerator.synchronize()
    assert_bits(out_b, out_a, "out")
    assert_bits(cache_b, cache_a, "conv state (page 0 must stay untouched)")
    assert_bits(blocks_b, blocks_a, "block states")
    if boundary:
        assert_bits(c_b, c_a, "x1_out")
