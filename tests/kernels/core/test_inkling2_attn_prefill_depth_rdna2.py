# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Depth battery of the Inkling-2-Mini gfx1030 prefill attention (the tile kernel).

``inkling2_attention`` (``csrc/rocm/inkling2_attn_rdna2.cu``) runs every request with
two or more query tokens through ``i2_tile_kernel``.  This file qualifies that kernel
at the depths of a long-context prefill (chunks of 1024 / 257 / 33 / 2 / 1 query rows
over 1k .. 144k keys, 24 q / 8 kv heads, sinks, relative bias, sliding and full
layers) and qualifies a *variant* of it: a restructured tile kernel must be
**bit-identical** to the shipped one on every world, not merely within the tolerance.

Two kinds of check, both on the same worlds:

* the fp64 oracle (``chunked_oracle_core``: the oracle of
  ``test_inkling2_attn_rdna2.py`` restructured to run at 144k keys on the GPU, pinned to
  it on small worlds by a CPU test) bounds the error of the shipped op and of the
  variant;
* the variant's output *and* split-K workspace (``part_o`` / ``part_ml``, so a
  difference that the combine kernel would round away shows) must equal the
  shipped op's bit for bit.

The worlds are "needle" worlds, because at 100k keys a random world averages the values
of the keys and every output is ~1e-2 whatever the kernel does.  Here a few keys per
request (``needle_positions``: the first and the last visible position, both sides of
the window and the relative-bias edge of the first and last query row, page / tile
edges, key 0, random depths) are aligned with the queries and score ~17 logits above the
bulk, so the output of a query row is the mix of the needles it may see.  A key that is
wrongly visible, wrongly masked, read from the wrong page / slot, or given the wrong
bias moves the output by far more than the tolerance
(``test_needle_worlds_are_sensitive``).  The paged cache is adversarial as in the
attention suite: unreferenced pages, slots past the sequence end, pages outside every
window and the keys before the window of the first row are NaN.

The op under test: the installed ``_rocm_C`` build, or a JIT build
(``tools/rdna/inkling2/attn_jit.py``).  Environment:

``I2_ATTN_SHIPPED_JIT``
    ``path[@MACRO=value,...]``: the shipped reference is a JIT build of this file (the
    9285bf8523 ``.cu`` when the tree already holds the variant) instead of the installed
    op, or of the repo file when ``_rocm_C`` has no ``inkling2_attention``.
``I2_ATTN_FORCE_JIT``
    ``1``: ignore the installed ``_rocm_C`` and JIT-build the repo file (a stale
    ``_rocm_C`` build next to a newer ``.cu``).
``I2_ATTN_VARIANT_JIT``
    ``path[@MACRO=value,...]``: the variant is a JIT build of this file (this is how
    hand mutants and work-in-progress kernels are qualified;
    ``I2_ATTN_VARIANT_TILE_IMPL``, default 1, is its trailing ``tile_impl`` argument
    when it has one).  Unset: the variant is the shipped op with ``tile_impl=1`` when
    the op takes a ``tile_impl`` argument, and the bit-identity tests skip otherwise.
``I2_ATTN_DEPTH_MAX_L``
    cap on the key count of the depth cases (default 150000).
``I2_ATTN_ORACLE_DEVICE``
    ``cuda`` (default when available) or ``cpu``.

Run: ``pytest tests/kernels/core/test_inkling2_attn_prefill_depth_rdna2.py`` on a
gfx1030 GPU (the oracle and world tests run everywhere).
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import math
import os
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[3]


def _load_attention_suite():
    """The fp64 oracle suite of the attention core (a sibling directory without an
    ``__init__``: loaded by path, under a name of its own)."""
    path = REPO / "tests" / "kernels" / "attention" / "test_inkling2_attn_rdna2.py"
    spec = importlib.util.spec_from_file_location("i2_attn_suite", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_attention_suite()
AttnWorld = base.AttnWorld
HEAD_DIM = base.HEAD_DIM
D_REL = base.D_REL
WINDOW = base.WINDOW
REL_EXTENT = base.REL_EXTENT
KV_BLOCK = base.KV_BLOCK
LOG_ALPHA, Q_CLIP, REL_CLIP = base.LOG_ALPHA, base.Q_CLIP, base.REL_CLIP

MAX_L = int(os.environ.get("I2_ATTN_DEPTH_MAX_L", "150000"))
SPIKE = 12.0  # the needle key is SPIKE x the unit direction of its kv head
TILE_KEYS = 64  # keys per tile of the kernel


def _on_gfx1030() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx1030" in torch.cuda.get_device_properties(0).gcnArchName


requires_gpu = pytest.mark.skipif(not _on_gfx1030(), reason="needs a gfx1030 GPU")


# ---------------------------------------------------------------------------
# the ops under test
# ---------------------------------------------------------------------------
class AttnOp:
    """``inkling2_attention`` with its optional trailing ``tile_impl`` argument.

    ``kind`` is ``wrapper`` (``vllm._custom_ops``, ``tile_impl`` is a keyword) or
    ``jit`` (a pybind function, positional); ``tile_impl`` None leaves it out.
    """

    def __init__(self, label: str, fn, kind: str, tile_impl: int | None):
        self.label, self.fn, self.kind, self.tile_impl = label, fn, kind, tile_impl

    def __call__(self, *args) -> None:
        if self.tile_impl is None:
            self.fn(*args)
        elif self.kind == "wrapper":
            self.fn(*args, tile_impl=self.tile_impl)
        else:
            self.fn(*args, self.tile_impl)

    def __repr__(self) -> str:
        return f"AttnOp({self.label})"


def _jit_op(label: str, spec: str, tile_impl: int, wrapped: bool = False) -> AttnOp:
    """A JIT build of ``spec``.  ``wrapped``: when the build has the ``tile_impl``
    argument and ``vllm._custom_ops.inkling2_attention`` takes it, the op is the
    *wrapper* with the build behind it (the wrapper's ``tile_impl`` policy and default
    are under test too)."""
    sys.path.insert(0, str(REPO / "tools"))
    from rdna.inkling2 import attn_jit

    mod = attn_jit.build(spec)
    if mod.NARGS > 24 and wrapped:
        import vllm._custom_ops as ops

        if "tile_impl" in inspect.signature(ops.inkling2_attention).parameters:
            torch.ops._rocm_C.inkling2_attention = mod.inkling2_attention
            return AttnOp(label, ops.inkling2_attention, "wrapper", tile_impl)
    return AttnOp(
        label, mod.inkling2_attention, "jit", tile_impl if mod.NARGS > 24 else None
    )


def _installed_wrapper():
    import vllm._custom_ops as ops

    if hasattr(torch.ops._rocm_C, "inkling2_attention"):
        return ops.inkling2_attention
    return None


@functools.cache
def shipped_op() -> AttnOp:
    spec = os.environ.get("I2_ATTN_SHIPPED_JIT", "")
    force = bool(spec or os.environ.get("I2_ATTN_FORCE_JIT"))
    wrapper = None if force else _installed_wrapper()
    if wrapper is None:
        return _jit_op("shipped (jit)", spec, 0, wrapped=True)
    takes = "tile_impl" in inspect.signature(wrapper).parameters
    return AttnOp("shipped", wrapper, "wrapper", 0 if takes else None)


@functools.cache
def variant_op() -> AttnOp | None:
    spec = os.environ.get("I2_ATTN_VARIANT_JIT", "")
    if spec:
        impl = int(os.environ.get("I2_ATTN_VARIANT_TILE_IMPL", "1"))
        return _jit_op("variant (jit)", spec, impl)
    ship = shipped_op()
    if ship.tile_impl is None:
        return None
    return AttnOp("variant", ship.fn, ship.kind, 1)


def _require_variant() -> AttnOp:
    """The variant, or a skip (a failure with ``I2_ATTN_REQUIRE_VARIANT=1``): a
    battery without a variant checks the shipped op only."""
    var = variant_op()
    if var is None:
        msg = "no variant: set I2_ATTN_VARIANT_JIT, or use an op with tile_impl"
        if os.environ.get("I2_ATTN_REQUIRE_VARIANT"):
            pytest.fail(msg)
        pytest.skip(msg)
    return var


def _ops():
    from vllm import _custom_ops as ops

    return ops


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------
def needle_positions(
    L: int,
    ql: int,
    window: int,
    rel_extent: int,
    block: int,
    extra: int,
    gen,
    block_starts: bool = False,
) -> list[int]:
    """Key positions that decide the output of the rows of a chunk (q_len ``ql`` over
    ``L`` keys): the edges that a mask / bias / page / tile off-by-one would cross."""
    p0, p1 = L - ql, L - 1  # position of the first / last query row
    cand = {0, 1, 63, 64, L - 1, L - 2, p0, p0 + 1}
    for base_pos in (p0, p1):
        for d in (-1, 0, 1):  # distance rel_extent - 1 / rel_extent / rel_extent + 1
            cand.add(base_pos - rel_extent + d)
        if window:  # first visible key of the row is base - window + 1
            for d in (-1, 0, 1, 2):
                cand.add(base_pos - window + d)
        for unit in (block, TILE_KEYS):
            t = base_pos // unit * unit
            cand |= {t - 1, t, t + unit - 1, t + unit}
    # the first row of every 16-row block of the CTA grid: a variant that tests the tile
    # against the first row of its block (the mask skip, the bias skip) is wrong by one
    # key when the block starts at 62 / 63 (mod 64)
    starts = list(range(0, ql, 16)) if block_starts else []
    for st in starts:
        cand |= {p0 + st, p0 + st + 1}
    mid = L // 2 // TILE_KEYS * TILE_KEYS
    cand |= {mid - 1, mid, mid + TILE_KEYS - 1, mid + TILE_KEYS}
    lo = max(0, p0 - window) if window else 0
    if extra and L - lo > 0:
        cand |= set((lo + torch.randint(0, L - lo, (extra,), generator=gen)).tolist())
    return sorted(j for j in cand if 0 <= j < L)


class DepthWorld(AttnWorld):
    """Needle world (see the module docstring) with a paged cache of ``block``-token
    pages; the attributes of ``AttnWorld`` (so its device arguments and the prepared
    ``q`` / ``rel`` of the oracle are reused)."""

    def __init__(
        self,
        q_lens,
        seq_lens,
        *,
        nkv=8,
        g=3,
        window=WINDOW,
        rel_extent=REL_EXTENT,
        block=KV_BLOCK,
        seed=0,
        needles=True,
        extra_needles=6,
        block_starts=False,
        with_gate=True,
        n_floor=96,
    ):
        gen = torch.Generator().manual_seed(seed)
        D = HEAD_DIM
        self.q_lens, self.seq_lens = list(q_lens), list(seq_lens)
        self.nkv, self.g, self.nh = nkv, g, nkv * g
        self.window, self.rel_extent, self.block = window, rel_extent, block
        self.log_scale = window == 0
        self.n_floor = n_floor
        self.R, self.T = len(q_lens), sum(q_lens)
        self.qsl = [0]
        for ql in q_lens:
            self.qsl.append(self.qsl[-1] + ql)
        pos = []
        for ql, L in zip(q_lens, seq_lens):
            pos += list(range(L - ql, L)) if L > 0 else [0] * ql
        self.positions = torch.tensor(pos, dtype=torch.int64)

        nb = [-(-max(L, 0) // block) for L in seq_lens]
        self.num_pages = sum(nb) + 6
        pool = base.PagePool(gen, self.num_pages)
        self.table = torch.zeros(self.R, max(nb + [1]) + 2, dtype=torch.int32)
        u = F.normalize(base._randn(gen, nkv, D), dim=-1)
        self.k_dense: list[torch.Tensor | None] = []
        self.v_dense: list[torch.Tensor | None] = []
        self.needles: list[list[int]] = []
        self.cache = torch.full(
            (self.num_pages, nkv, block, 2 * D), float("nan"), dtype=torch.float16
        )
        for r, (ql, L) in enumerate(zip(q_lens, seq_lens)):
            if L <= 0:
                self.k_dense.append(None)
                self.v_dense.append(None)
                self.needles.append([])
                continue
            k = base._randn(gen, L, nkv, D)
            idx = needle_positions(
                L, ql, window, rel_extent, block, extra_needles, gen, block_starts
            )
            idx = idx if needles else []
            if idx:
                n = torch.tensor(idx)
                k[n] = SPIKE * u + base._randn(gen, len(idx), nkv, D, scale=0.3)
            self.needles.append(idx)
            k, v = k.half(), base._randn(gen, L, nkv, D).half()
            self.k_dense.append(k)
            self.v_dense.append(v)
            pages = pool.take(nb[r])
            lo_first = max(0, L - ql - window + 1) if window else 0
            for b in range(nb[r]):
                if b < lo_first // block:
                    continue  # fully outside every window: the null block (NaN)
                self.table[r, b] = pages[b]
                # the keys below the window of the first row (in the page that holds
                # its start) stay NaN too: no row reads them
                lo, hi = max(b * block, lo_first), min(L, (b + 1) * block)
                at = lo - b * block
                self.cache[pages[b], :, at : at + hi - lo, :D] = k[lo:hi].transpose(
                    0, 1
                )
                self.cache[pages[b], :, at : at + hi - lo, D:] = v[lo:hi].transpose(
                    0, 1
                )

        T, nh = self.T, self.nh
        mag = torch.tensor([0.6, 1.0, 3.0])[torch.randint(0, 3, (T, nh), generator=gen)]
        noise = F.normalize(base._randn(gen, T, nh, D), dim=-1)
        ug = u.repeat_interleave(g, 0)[None]  # head h -> kv head h // g
        dirs = F.normalize(0.7 * ug + 0.71 * noise, dim=-1)
        self.q_raw = (dirs * mag[..., None] * math.sqrt(D)).half().view(T, nh * D)
        self.rel_raw = (
            (base._randn(gen, T, nh, D_REL) * (mag[..., None] * 2.0))
            .half()
            .view(T, nh * D_REL)
        )
        self.sink_raw = (base._randn(gen, T, nh) + 5.0).half()
        self.gate_raw = (
            base._randn(gen, T, nh * D, scale=1.5).half() if with_gate else None
        )
        self.q_scale = 0.7 + 0.6 * base._rand(gen, nh, D)
        self.rel_scale = 0.7 + 0.6 * base._rand(gen, nh, D_REL)
        # rows past rel_extent are garbage that must never be used
        self.proj_t = base._randn(gen, rel_extent + 8, D_REL, scale=0.3).half()
        self.seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        self.qsl_t = torch.tensor(self.qsl, dtype=torch.int32)

    def oracle_core(self) -> torch.Tensor:
        out, self.mag = _chunked_oracle(self)
        return out

    def oracle(self) -> torch.Tensor:
        out = super().oracle()
        mag = self.mag
        if self.gate_raw is not None:
            mag = mag * torch.sigmoid(self.gate_raw.double().view(self.T, self.nh, -1))
        self.mag_gated = mag.view(self.T, self.nh * HEAD_DIM)
        return out


# ---------------------------------------------------------------------------
# the oracle at depth
# ---------------------------------------------------------------------------
def _oracle_device() -> str:
    want = os.environ.get("I2_ATTN_ORACLE_DEVICE", "")
    if want:
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def chunked_oracle_core(w: AttnWorld, **kw) -> torch.Tensor:
    """Pre-gate attention output ``[T, nh, D]`` in float64 (see ``_chunked_oracle``)."""
    return _chunked_oracle(w, **kw)[0]


def _chunked_oracle(
    w: AttnWorld,
    *,
    device: str | None = None,
    window: int | None = None,
    rel_extent: int | None = None,
    q_block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-gate attention output ``[T, nh, D]`` in float64 and its magnitude
    ``sum_j p_j |v_j| / den`` (the scale of the fp16 rounding of the probabilities):
    ``AttnWorld.oracle_core`` (softmax over the visible keys and the sink, relative bias
    ``proj[dist] . rel`` for ``0 <= dist < rel_extent``) in blocks of ``q_block`` query
    rows, only over the key range a block can see, the bias only over the columns it
    reaches: 144k keys in seconds.  ``window`` / ``rel_extent`` override the world's
    (sensitivity tests)."""
    dev = device or _oracle_device()
    win = w.window if window is None else window
    R = w.rel_extent if rel_extent is None else rel_extent
    q = w.prepped_q().double()
    rel = w.prepped_rel().double()
    sink = w.sink_raw.double()
    proj = w.proj_t[:R].double().to(dev)
    out = torch.zeros(w.T, w.nh, HEAD_DIM, dtype=torch.float64)
    mag = torch.zeros_like(out)
    scale = HEAD_DIM**-0.5
    for r, (ql, L) in enumerate(zip(w.q_lens, w.seq_lens)):
        if L <= 0 or ql == 0:
            continue
        s = w.qsl[r]
        kd, vd = w.k_dense[r], w.v_dense[r]
        assert kd is not None and vd is not None
        for kvh in range(w.nkv):
            kk = kd[:, kvh].to(dev).double()
            vv = vd[:, kvh].to(dev).double()
            for i0 in range(0, ql, q_block):
                i1 = min(ql, i0 + q_block)
                pq = torch.arange(L - ql + i0, L - ql + i1, device=dev)
                lo = max(0, int(pq[0]) - win + 1) if win else 0
                hi = int(pq[-1]) + 1
                j = torch.arange(lo, hi, device=dev)
                dist = pq[:, None] - j[None, :]  # [rows, keys]
                vis = dist >= 0
                if win:
                    vis &= dist < win
                # the bias only reaches the columns with dist < R for some row
                c0 = max(0, int(pq[0]) - R + 1 - lo)
                d_c = dist[:, c0:]
                idx = d_c.clamp(0, R - 1)
                for gi in range(w.g):
                    h = kvh * w.g + gi
                    qh = q[s + i0 : s + i1, h].to(dev)
                    sc = (qh @ kk[lo:hi].T) * scale
                    rl = rel[s + i0 : s + i1, h].to(dev) @ proj.T  # [rows, R]
                    bias = torch.gather(rl, 1, idx).masked_fill(
                        (d_c < 0) | (d_c >= R), 0.0
                    )
                    sc[:, c0:] += bias
                    sc = sc.masked_fill(~vis, float("-inf"))
                    sk = sink[s + i0 : s + i1, h].to(dev)
                    m = torch.maximum(sc.max(1).values, sk)
                    p = torch.exp(sc - m[:, None])
                    den = p.sum(1) + torch.exp(sk - m)
                    out[s + i0 : s + i1, h] = ((p @ vv[lo:hi]) / den[:, None]).cpu()
                    mag[s + i0 : s + i1, h] = (
                        (p @ vv[lo:hi].abs()) / den[:, None]
                    ).cpu()
    return out, mag


def test_chunked_oracle_matches_the_attention_suite_oracle():
    """The restructured oracle is the oracle of ``test_inkling2_attn_rdna2.py``."""
    small = [(3, 3), (17, 80), (64, 1088), (33, 1300), (1, 70), (5, 2300)]
    deep = [(64, 1088), (33, 1300), (5, 2300), (257, 2600)]
    for reqs, sharp, window, rel_extent, nkv, g in [
        (small, False, WINDOW, 1024, 2, 3),
        (small, False, 0, 200, 2, 3),
        (small, False, 0, 1024, 4, 1),
        (small, False, WINDOW, 64, 2, 4),
        (deep, True, WINDOW, 1024, 2, 3),
        (deep, True, 0, 1000, 2, 2),
    ]:
        w = AttnWorld(
            [q for q, _ in reqs],
            [s for _, s in reqs],
            nkv=nkv,
            g=g,
            window=window,
            rel_extent=rel_extent,
            sharp=sharp,
            seed=3,
        )
        torch.testing.assert_close(
            chunked_oracle_core(w, device="cpu", q_block=16),
            w.oracle_core(),
            rtol=1e-12,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# world properties (CPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("block", [16, 64, 128])
def test_needle_world_is_adversarial(window, block):
    q, L = 33, 3000
    w = DepthWorld([q], [L], window=window, block=block, nkv=2, seed=1)
    assert len(w.needles[0]) >= 12
    p0 = L - q
    if window:
        edge = p0 - window
        assert {edge, edge + 1, edge + 2} <= set(w.needles[0])
    assert {L - 1, p0, 0, 1} <= set(w.needles[0])
    nb = -(-L // block)
    used = set(w.table[0, :nb].tolist())
    nulls = int((w.table[0, :nb] == 0).sum())
    lo_first = max(0, p0 - window + 1) if window else 0
    assert nulls == lo_first // block
    # pages nobody references and the slots past the sequence end are NaN
    free = [p for p in range(w.num_pages) if p not in used]
    assert free and torch.isnan(w.cache[free]).all()
    last = int(w.table[0, nb - 1])
    assert L % block and torch.isnan(w.cache[last, :, L - (nb - 1) * block :]).all()
    first = int(w.table[0, lo_first // block])
    assert torch.isnan(w.cache[first, :, : lo_first % block]).all()
    for p in used - {0, last, first}:
        assert torch.isfinite(w.cache[p]).all()
    if first != last:
        assert torch.isfinite(w.cache[first, :, lo_first % block :]).all()


@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_needle_worlds_are_sensitive(window):
    """The worlds make the output a mix of the needles: an off-by-one in the window
    mask, in the relative-bias reach, or a dropped needle position moves the oracle
    output by far more than the tolerance of ``check_subject`` (so the battery has
    teeth)."""
    q, L = 33, 5000
    w = DepthWorld([q], [L], window=window, nkv=2, seed=2)
    ref, mag = _chunked_oracle(w, device="cpu")
    tol = ATOL + PROB_EPS * mag
    trials = {"rel_extent-1": dict(rel_extent=w.rel_extent - 1)}
    if window:
        trials["window-1"] = dict(window=window - 1)
        trials["window+1"] = dict(window=window + 1)
    for name, kw in trials.items():
        moved = chunked_oracle_core(w, device="cpu", **kw)
        ratio = ((moved - ref).abs() / tol).max().item()
        assert ratio > 20, f"{name}: the oracle moves only {ratio:.1f} x the tolerance"
    # dropping every needle of the first query row's window edge (hide them)
    kd = w.k_dense[0].clone()
    for j in w.needles[0]:
        kd[j] = 0.0
    saved, w.k_dense[0] = w.k_dense[0], kd
    try:
        gone = chunked_oracle_core(w, device="cpu")
    finally:
        w.k_dense[0] = saved
    assert ((gone - ref).abs() / tol).max().item() > 100


# ---------------------------------------------------------------------------
# running the op
# ---------------------------------------------------------------------------
def run_op(
    op: AttnOp,
    w: AttnWorld,
    *,
    splits=None,
    strided=False,
    token_major=True,
    layers=1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output ``[T, nh * D]`` fp16 and the workspace (both NaN-poisoned before the
    launch: an unwritten row or a read of a row that was never written shows)."""
    a = w.device_args(strided=strided, token_major=token_major, layers=layers)
    out = torch.full(
        (w.T, w.nh * HEAD_DIM), float("nan"), dtype=torch.float16, device="cuda"
    )
    max_q = max(w.q_lens)
    if splits is None:  # the shipped policy
        splits = _ops().inkling2_attention_splits(
            w.R, w.nkv, w.window, max(w.seq_lens), max_q
        )
    ws = torch.full(
        (_ops().inkling2_attention_workspace_numel(w.R, w.T, w.nh, *splits),),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    op(
        a["q"], a["rel"], a["sink"], a["gate"], a["q_scale"], a["rel_scale"],
        a["rel_proj_t"], a["positions"], a["kv_cache"], a["block_table"],
        a["seq_lens"], a["query_start_loc"], out, ws,
        w.window, w.rel_extent, w.log_scale, LOG_ALPHA, float(w.n_floor),
        Q_CLIP, REL_CLIP, max_q, splits[0], splits[1],
    )  # fmt: skip
    torch.accelerator.synchronize()
    return out, ws


def assert_bit_identical(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Equal bit patterns (NaN payloads included) of two same-shaped tensors."""
    assert got.shape == want.shape and got.dtype == want.dtype
    width = {2: torch.int16, 4: torch.int32}[got.element_size()]
    bad = got.view(width) != want.view(width)
    if bad.any():
        n = int(bad.sum())
        first = int(bad.flatten().nonzero()[0])
        a, b = got.flatten()[first].item(), want.flatten()[first].item()
        where = divmod(first, got.shape[-1]) if got.dim() == 2 else (first,)
        pytest.fail(
            f"{what}: {n} of {got.numel()} elements differ bitwise; first at {where}: "
            f"variant {a!r} shipped {b!r}"
        )


# The fp16 probabilities of the kernel carry a relative rounding error of 2^-11 each;
# summed against V (and rounded once more to fp16 at the output) the error of an output
# element scales with the magnitude sum_j p_j |v_j| / den of the element, whatever its
# value (an output that cancels, ~1e-2 next to a magnitude of ~1, is as wrong by 1e-3
# as one that does not).  The shipped kernel's worst error over the depth cases
# (q 1..1024, L 1k..144k, sliding and full) is 3.6e-3 of the magnitude; the bound is
# 2^-7 = 7.8e-3 of it, a needle that is weighted wrongly by more than a few per cent of
# the output is outside.  ``I2_ATTN_PROB_EPS`` overrides the factor (calibration).
PROB_EPS = float(os.environ.get("I2_ATTN_PROB_EPS", 2**-7))
ATOL = 1e-4


def check_subject(
    op: AttnOp, out: torch.Tensor, w: DepthWorld, ref: torch.Tensor, what: str
) -> None:
    """``out`` against the fp64 oracle: |err| <= ATOL + PROB_EPS * magnitude."""
    got = out.cpu().double()
    label = f"{op.label} {what}"
    assert torch.isfinite(got).all(), (
        f"{label}: non-finite output (unwritten row or a read of poisoned memory)"
    )
    err = (got - ref).abs()
    bound = ATOL + PROB_EPS * w.mag_gated
    worst = (err / bound).max().item()
    print(
        f"inkling2-attn-depth {label}: max_abs_err={err.max().item():.3e} "
        f"max_err/magnitude={(err / w.mag_gated.clamp(min=1e-3)).max().item():.3e} "
        f"worst_err/bound={worst:.3f}"
    )
    if worst > 1.0:
        t, c = divmod(int((err / bound).argmax()), got.shape[1])
        pytest.fail(
            f"{label}: err {err.max().item():.3e} exceeds the bound (token {t}, "
            f"head {c // HEAD_DIM}, dim {c % HEAD_DIM}): got {got[t, c].item():.5f} "
            f"ref {ref[t, c].item():.5f} magnitude {w.mag_gated[t, c].item():.3f}"
        )


def run_case(w: DepthWorld, what: str, *, splits=None, oracle=True, **kw) -> None:
    """One world: the shipped op and the variant (when there is one) against the
    oracle, and the variant against the shipped op bit for bit (output and
    workspace)."""
    ship, var = shipped_op(), variant_op()
    ref = w.oracle() if oracle else None
    out_s, ws_s = run_op(ship, w, splits=splits, **kw)
    if ref is not None:
        check_subject(ship, out_s, w, ref, what)
    else:
        assert torch.isfinite(out_s.cpu()).all(), f"{what}: shipped output not finite"
    if var is None:
        return
    out_v, ws_v = run_op(var, w, splits=splits, **kw)
    if ref is not None:
        check_subject(var, out_v, w, ref, what)
    assert_bit_identical(out_v, out_s, f"{what}: output")
    assert_bit_identical(ws_v, ws_s, f"{what}: split-K workspace")


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("q", [2, 33, 257, 1024])
def test_default_tile_impl_is_the_shipped_result(window, q):
    """What the wrapper picks when ``tile_impl`` is left out (the policy and the
    ``VLLM_INKLING2_PREFILL_ATTN_TILE`` default) is bit-identical to the shipped
    tile kernel: serving never sees a different number."""
    ship = shipped_op()
    if ship.kind != "wrapper" or ship.tile_impl is None:
        pytest.skip("the op under test has no wrapper with a tile_impl default")
    w = DepthWorld([q, 1], [4097, 2049], window=window, seed=q + 5)
    out_d, ws_d = run_op(AttnOp("default", ship.fn, "wrapper", None), w)
    out_s, ws_s = run_op(ship, w)
    assert_bit_identical(out_d, out_s, f"default q={q} window={window}: output")
    assert_bit_identical(ws_d, ws_s, f"default q={q} window={window}: workspace")


@requires_gpu
def test_variant_is_configured():
    """Reports whether the bit-identity half of the battery runs at all."""
    var = _require_variant()
    assert var.fn is not shipped_op().fn or var.tile_impl != shipped_op().tile_impl


# ---------------------------------------------------------------------------
# the depth battery
# ---------------------------------------------------------------------------
CHUNKS = [1024, 257, 33, 2, 1]
DEPTHS = [1025, 1088, 4999, 16383, 16384, 16385, 40961, 65537, 100353, 131073, 144001]


def _depth_cases():
    for window, kind in ((WINDOW, "sliding"), (0, "full")):
        for q in CHUNKS:
            for L in DEPTHS:
                if q > L or L > MAX_L:
                    continue
                yield pytest.param(window, q, L, id=f"{kind}-q{q}-L{L}")


@requires_gpu
@pytest.mark.parametrize("window,q,L", list(_depth_cases()))
def test_prefill_depth(window, q, L):
    """24 q / 8 kv heads, one request: a chunk of ``q`` rows ending at ``L`` keys."""
    w = DepthWorld([q], [L], window=window, seed=L % 9973 + q)
    run_case(w, f"window={window} q={q} L={L}")


def _edge_cases():
    # the first row of the chunk sits on / next to a window edge (p0 - window + 1 is
    # a multiple of 64 or one off), a page edge, a tile edge
    for q in (1024, 257, 33):
        for delta in (-1, 0, 1, 62, 63, 64, 65):
            yield pytest.param(
                WINDOW, q, q + WINDOW + delta, id=f"sliding-edge-q{q}-d{delta}"
            )
        for base_len in (3 * 64, 16 * 64, 255 * 64):
            for delta in (-1, 0, 1):
                L = base_len + q + delta
                yield pytest.param(0, q, L, id=f"full-edge-q{q}-L{L}")


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("q", [256, 300])
@pytest.mark.parametrize("residue", range(16))
def test_prefill_block_start_residues(window, q, residue):
    """The CTA grid splits the chunk into blocks of 16 rows whose first row sits at
    ``L - q + 16 i``: sweep ``(L - q) % 16`` over every value, so that a block starts at
    every position mod 64 (a tile test one key off is exact except when a *full* block
    starts at 62 / 63 mod 64), with a partial last block (q = 300) and without
    (q = 256)."""
    L = 3000 + residue
    w = DepthWorld([q], [L], window=window, seed=residue + q, block_starts=True)
    run_case(w, f"block starts window={window} q={q} L={L}")


@requires_gpu
@pytest.mark.parametrize("window,q,L", list(_edge_cases()))
def test_prefill_window_page_and_tile_edges(window, q, L):
    w = DepthWorld([q], [L], window=window, seed=L + q)
    run_case(w, f"edge window={window} q={q} L={L}")


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("splits_tile", [1, 2, 5, 16, None])
def test_prefill_depth_split_k(window, splits_tile):
    """The split-K partitions of the tile range (each split restarts the online
    softmax, so the first / last tile of every split is an edge of its own)."""
    L = min(65537, MAX_L)
    w = DepthWorld([1024, 257, 33], [L, L // 2 + 1, 16385], window=window, seed=21)
    splits = None if splits_tile is None else (4, splits_tile)
    run_case(w, f"split-k window={window} splits_tile={splits_tile}", splits=splits)


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_prefill_mixed_batch_at_depth(window):
    """Prefill chunks, decode rows, a padded and an empty request in one launch."""
    reqs = [
        (1024, 16385),
        (1, 40961),
        (257, 65537),
        (33, 16384),
        (1, 1025),
        (2, 1088),
        (1, 0),  # padded request
        (0, 17),  # empty request
    ]
    reqs = [(q, L) for q, L in reqs if L <= MAX_L]
    w = DepthWorld([q for q, _ in reqs], [L for _, L in reqs], window=window, seed=22)
    run_case(w, f"mixed window={window}")


# ---------------------------------------------------------------------------
# geometry matrix (moderate depth)
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("rel_extent", [64, 100, 200, 960, 1000, 1024, 1088, 5000])
def test_prefill_relative_bias_reach(window, rel_extent):
    """``dist >= rel_extent`` adds nothing (a variant may skip those tiles' bias): odd
    reaches, shorter than a tile, shorter / longer than the window, longer than the
    context."""
    w = DepthWorld(
        [1024, 257, 33],
        [8193, 4097, 1300],
        window=window,
        rel_extent=rel_extent,
        seed=23,
    )
    run_case(w, f"rel_extent={rel_extent} window={window}")


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("nkv,g", [(8, 3), (4, 3), (2, 3), (8, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("token_major", [True, False], ids=["bhc", "hbc"])
def test_prefill_head_geometry(window, nkv, g, token_major):
    """GQA 24/8 (the model), the TP=2 / TP=4 shards and the other group sizes."""
    w = DepthWorld(
        [257, 33, 1024], [4097, 1300, 5000], nkv=nkv, g=g, window=window, seed=24
    )
    run_case(w, f"nkv={nkv} g={g} window={window}", token_major=token_major)


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
@pytest.mark.parametrize("block", [16, 32, 64, 128])
def test_prefill_block_sizes(window, block):
    """Cache blocks of 16 .. 128 tokens: the 64-key tile may or may not be one page
    (a variant that stages whole pages has a different path below 64)."""
    w = DepthWorld(
        [1024, 257, 33], [8193, 4097, 1300], window=window, block=block, seed=25
    )
    run_case(w, f"block={block} window={window}")


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_prefill_block_outermost_layers(window):
    """Engine ``BL*`` layouts: other layers' pages sit between this layer's blocks."""
    w = DepthWorld([257, 33, 1024], [4097, 1300, 5000], window=window, seed=26)
    run_case(w, f"block-outermost window={window}", layers=3)


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_prefill_no_gate_strided(window):
    """No gate, q / rel / sink as column slices of one fused buffer."""
    w = DepthWorld([257, 33], [4097, 1300], window=window, with_gate=False, seed=27)
    run_case(w, f"no gate strided window={window}", strided=True)


@requires_gpu
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["sliding", "full"])
def test_prefill_random_world(window):
    """No needles: plain random keys, the shipped test's tolerance."""
    w = DepthWorld(
        [1024, 257, 33], [8193, 4097, 1300], window=window, needles=False, seed=28
    )
    run_case(w, f"random window={window}")
