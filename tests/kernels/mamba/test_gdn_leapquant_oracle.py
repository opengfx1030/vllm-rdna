# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The safety checks below follow the list in the LeapQuant RFC, vllm-project/vllm
# #64725 (lihaoyang-amd): section 2.2 (NaN, determinism, flush race) and
# section 5 (numerical safety, request isolation). The RFC has no code; every
# test here is original work for this fork.
"""CPU regression tests for the LeapQuant FP32 oracle (no GPU needed).

The oracle in vllm/model_executor/layers/mamba/ops/leapquant_gdn.py is the
executable specification of the HIP kernels. These tests pin two things:

* The fold (``leapquant_store_state``, the first half of every flush) must
  give a finite, accurate blob for degenerate states: zero, tiny, huge, and
  exactly low rank (the RFC prototype produced NaN there).
* A block that a request has not written yet holds the leftovers of another
  request. Its scales, smoothing factors, compensators, records and
  cursor/pending header must never reach the first read of the new request.

The GPU mirrors of these tests are in test_gdn_leapquant_safety_rdna2.py.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops import leapquant_gdn as lq

K = V = 128
H, HV = 2, 4
SERVING = lq.LeapQuantLayout(
    K, V, window=32, num_comp=4, max_query_len=3, power_iters=2
)
INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

# A dense random state is stored with about 6e-3 relative error (INT8 per row).
INT8_BOUND = 1e-2
# A state of rank <= num_comp is carried almost entirely by the FP16
# compensators; the INT8 residual only holds their rounding error.
LOW_RANK_BOUND = 1e-4
# FP16 rounding of orthonormal compensator rows (5e-4 relative per entry).
ORTHO_TOL = 2e-3
# ``ranged_elems`` spans 34 decades element by element. The largest entry of a
# row sets the INT8 step, so this state is INT8-limited: the error is 1.56e-2
# with the old and the new fold alike (it is not a NaN or compensator effect).
DEGENERATE_BOUND = {"ranged_elems": 2e-2}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.double() - b.double()).norm().item()
    den = b.double().norm().item()
    return num / den if den > 0 else (0.0 if num == 0 else float("inf"))


def _fold(S: torch.Tensor, layout=SERVING):
    """Cold fold of ``S [n, V, K]`` into zeroed blobs; returns (blob, state)."""
    blob = torch.zeros(S.shape[0], layout.nbytes, dtype=torch.uint8)
    lq.leapquant_store_state(layout, blob, S)
    return blob, lq.leapquant_materialize(layout, blob)


def _low_rank(rank: int, scale: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(V, rank, generator=g)
    b = torch.randn(K, rank, generator=g)
    return ((a @ b.T) * scale)[None]


def _integer_low_rank(rank: int, seed: int = 0) -> torch.Tensor:
    """Exactly representable, exactly rank-``rank`` state: the Gram-Schmidt
    residual of a dependent column can be exactly zero."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-3, 4, (V, rank), generator=g).float()
    b = torch.randint(-3, 4, (K, rank), generator=g).float()
    return (a @ b.T)[None]


def _dense(scale: float, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, V, K, generator=g) * scale


def _comp_gram(layout, blob):
    vw = layout.views(blob)
    q = vw["comp_u"].float()[0]  # [r, V]
    return q @ q.T


def _assert_orthonormal_or_zero(gram: torch.Tensor) -> None:
    diag = torch.diagonal(gram)
    live = diag > 0.5
    assert bool(((diag - 1).abs() < ORTHO_TOL)[live].all()), diag
    assert bool((diag[~live].abs() < ORTHO_TOL).all()), diag
    off = gram - torch.diag(diag)
    assert float(off.abs().max()) < ORTHO_TOL, f"compensators not orthogonal: {gram}"


# --------------------------------------------------------------------------
# Item 1: NaN guard and numerics of the fold.
# --------------------------------------------------------------------------


def test_zero_state_folds_to_zero_blob_state():
    blob, got = _fold(torch.zeros(1, V, K))
    vw = SERVING.views(blob)
    assert torch.equal(got, torch.zeros_like(got))
    for name in ("z", "bscale", "comp_k", "comp_u"):
        assert not vw[name].float().abs().any(), name
    assert bool(torch.isfinite(vw["csmooth"]).all())
    assert (vw["header"][..., :2] == 0).all()


@pytest.mark.parametrize("exp", [-8, -14, -20, -30, -38, -42])
@pytest.mark.parametrize("kind", ["dense", "rank1"])
def test_tiny_state_is_finite_and_int8_accurate(kind, exp):
    scale = 10.0**exp
    S = _dense(scale) if kind == "dense" else _low_rank(1, scale)
    _, got = _fold(S)
    assert torch.isfinite(got).all()
    assert _rel(got, S) < INT8_BOUND


@pytest.mark.parametrize("exp", [6, 12, 18, 30, 36, 37])
@pytest.mark.parametrize("kind", ["dense", "rank1"])
def test_huge_state_is_finite_and_int8_accurate(kind, exp):
    scale = 10.0**exp
    S = _dense(scale) if kind == "dense" else _low_rank(1, scale)
    _, got = _fold(S)
    assert torch.isfinite(got).all()
    assert _rel(got, S) < INT8_BOUND


@pytest.mark.parametrize("kind", ["dense", "rank1"])
def test_state_at_the_float_limit_is_finite_and_int8_accurate(kind):
    """The column sum of |R| overflows here. The old fold made it inf and the
    smoothing factor gave 0 * inf = NaN in every read of the block."""
    base = _dense(1.0) if kind == "dense" else _low_rank(1, 1.0)
    S = base / base.abs().max() * 3.0e38
    blob, got = _fold(S)
    assert torch.isfinite(got).all()
    assert _rel(got, S) < INT8_BOUND
    _assert_orthonormal_or_zero(_comp_gram(SERVING, blob))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_exactly_low_rank_state_is_carried_by_the_compensators(rank, scale, seed):
    """Rank <= r: columns past the rank are rounding noise. They must come out
    orthogonal to the others; if they are only normalized they are half parallel
    to them, the dominant direction is counted twice and the INT8 residual
    carries a state-sized error."""
    S = _low_rank(rank, scale, seed)
    blob, got = _fold(S)
    assert torch.isfinite(got).all()
    assert _rel(got, S) < LOW_RANK_BOUND
    _assert_orthonormal_or_zero(_comp_gram(SERVING, blob))


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_exactly_representable_low_rank_state(rank):
    """Integer entries: dependent columns can cancel to exactly zero."""
    S = _integer_low_rank(rank)
    blob, got = _fold(S)
    assert torch.isfinite(got).all()
    assert _rel(got, S) < LOW_RANK_BOUND
    _assert_orthonormal_or_zero(_comp_gram(SERVING, blob))


@pytest.mark.parametrize("rank", [5, 8, 32])
def test_rank_above_num_comp_stays_int8_accurate(rank):
    S = _low_rank(rank, 1.0)
    blob, got = _fold(S)
    assert _rel(got, S) < INT8_BOUND
    _assert_orthonormal_or_zero(_comp_gram(SERVING, blob))


def _degenerate_states() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    sparse = torch.zeros(1, V, K)
    sparse[0, 5, 9] = 3.0
    rows = torch.zeros(1, V, K)
    rows[0, :4] = torch.randn(4, K, generator=g)
    const = torch.full((1, V, K), 0.25)
    row_scale = 10.0 ** (torch.rand(V, generator=g) * 12 - 6)  # 1e-6 .. 1e6
    ranged_rows = torch.randn(1, V, K, generator=g) * row_scale[None, :, None]
    ranged_elems = torch.randn(1, V, K, generator=g) * 10.0 ** (
        torch.rand(1, V, K, generator=g) * 34 - 30
    )  # 1e-30 .. 1e4, element by element
    outlier = _dense(1.0) + 3e4 * (
        torch.randn(1, V, 1, generator=g) * torch.randn(1, 1, K, generator=g)
    )
    return {
        "single_entry": sparse,
        "four_rows": rows,
        "constant": const,
        "ranged_rows": ranged_rows,
        "ranged_elems": ranged_elems,
        "outlier_3e4": outlier,
    }


@pytest.mark.parametrize("name", list(_degenerate_states()))
def test_dynamic_range_and_sparse_states_are_finite(name):
    S = _degenerate_states()[name]
    blob, got = _fold(S)
    assert torch.isfinite(got).all(), name
    assert _rel(got, S) < DEGENERATE_BOUND.get(name, INT8_BOUND), name
    _assert_orthonormal_or_zero(_comp_gram(SERVING, blob))


def test_degenerate_heads_do_not_disturb_their_neighbours():
    """One launch folds many heads: a zero, tiny or rank-1 head must not change
    the head next to it."""
    dense = _dense(1.0, seed=11)
    mix = torch.cat(
        [torch.zeros(1, V, K), _low_rank(1, 1e-30), _low_rank(1, 1.0), dense]
    )
    _, got_mix = _fold(mix)
    _, got_alone = _fold(dense)
    assert torch.isfinite(got_mix).all()
    assert _rel(got_mix[3], got_alone[0]) < 1e-5


def test_orthonormalize_returns_zero_or_unit_columns():
    """Unit-level contract of the Gram-Schmidt step every fold iteration uses."""
    g = torch.Generator().manual_seed(3)
    base = torch.randn(V, 1, generator=g)
    cases = [
        torch.zeros(V, 4),
        base.expand(V, 4).clone() * torch.tensor([1.0, 2.0, -3.0, 0.5]),
        torch.randn(V, 4, generator=g),
        torch.randn(V, 4, generator=g) * 1e-30,
        torch.randn(V, 4, generator=g) * 1e30,
        torch.randn(V, 4, generator=g) * torch.tensor([1e-30, 1.0, 1e30, 1e-3]),
    ]
    for Y in cases:
        Q = lq._orthonormalize(Y[None])[0]
        assert torch.isfinite(Q).all()
        gram = Q.T @ Q
        _assert_orthonormal_or_zero(gram)


def test_orthonormalize_is_invariant_under_power_of_two_scaling():
    """Max-abs scaling makes the result independent of the column magnitude."""
    g = torch.Generator().manual_seed(4)
    Y = torch.randn(1, V, 4, generator=g)
    base = lq._orthonormalize(Y)
    for e in (-100, -30, 30, 100):
        assert torch.equal(lq._orthonormalize(Y * 2.0**e), base), e


def test_flush_of_an_exactly_rank_one_trajectory():
    """The live path: identical keys make the state exactly rank 1; the window
    fills with records and the flush folds them. No NaN, no drift from FP32."""
    layout = lq.LeapQuantLayout(
        K, V, window=16, num_comp=4, max_query_len=1, power_iters=2
    )
    n = 1
    g = torch.Generator().manual_seed(5)
    A_log = torch.log(torch.rand(HV, generator=g) * 0.5 + 0.1)
    dt_bias = torch.zeros(HV)
    k_fixed = torch.randn(n, 1, H, K, generator=g)
    q = torch.randn(n, 1, H, K, generator=g)
    blob = torch.zeros(n, HV, layout.nbytes, dtype=torch.uint8)
    S_ref = torch.zeros(n, HV, V, K)
    errs = []
    for step in range(40):  # 2.5 windows: two flushes
        v = torch.randn(n, 1, HV, V, generator=g)
        a = torch.randn(n, 1, HV, generator=g) - 3
        b = torch.randn(n, 1, HV, generator=g)
        lq.leapquant_normalize(layout, blob, None)
        o = lq.leapquant_decode(
            layout, blob, q.half(), k_fixed.half(), v.half(), a.half(), b.half(),
            A_log, dt_bias, K**-0.5,
        )  # fmt: skip
        qn = lq.l2norm(q.half()).repeat_interleave(HV // H, 2) * K**-0.5
        kn = lq.l2norm(k_fixed.half()).repeat_interleave(HV // H, 2)
        gg, beta = lq.gdn_gates(a.half(), b.half(), A_log, dt_bias)
        S_ref, o_ref, _ = lq.gdn_step_fp32(
            S_ref, qn[:, 0], kn[:, 0], v[:, 0].half(), gg[:, 0], beta[:, 0]
        )
        assert torch.isfinite(o).all()
        errs.append(_rel(o[:, 0], o_ref))
    lq.leapquant_normalize(layout, blob, None)
    assert bool((layout.views(blob)["header"][..., 0] < 16).all())
    # Measured on CPU: worst error 5.4e-4 with this fold, 1.6e-3 with the old
    # fold (its dependent compensator columns were not orthogonal).
    assert max(errs) < 1e-3, f"worst output error {max(errs):.2e}"


# --------------------------------------------------------------------------
# Item 3: stale state of a recycled block must not reach the first read.
# --------------------------------------------------------------------------

LAYOUT_T3 = SERVING
CAP = LAYOUT_T3.capacity


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    A_log = torch.log(torch.rand(HV, generator=g) * 0.99 + 0.01)
    dt_bias = torch.randn(HV, generator=g) * 0.1
    return A_log, dt_bias


def _tokens(n, t, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, t, H, K, generator=g)
    k = torch.randn(n, t, H, K, generator=g)
    v = torch.randn(n, t, HV, V, generator=g)
    a = torch.randn(n, t, HV, generator=g) - 3
    b = torch.randn(n, t, HV, generator=g)
    return [x.half() for x in (q, k, v, a, b)]


def _previous_tenant(n: int, seed: int) -> torch.Tensor:
    """A block as another request left it: stored, decoded for a while, with a
    non-zero cursor, pending records, records of every row and a live boundary."""
    layout = LAYOUT_T3
    A_log, dt_bias = _params(seed)
    g = torch.Generator().manual_seed(seed)
    blob = torch.zeros(n, HV, layout.nbytes, dtype=torch.uint8)
    lq.leapquant_store_state(
        layout, blob, torch.randn(n, HV, V, K, generator=g) * 3.0 + 7.0
    )
    nacc = None
    for step in range(layout.window + 5):  # the ring has been filled and refolded
        q, k, v, a, b = _tokens(n, 3, seed * 1000 + step)
        lq.leapquant_normalize(layout, blob, nacc)
        lq.leapquant_decode(layout, blob, q, k, v, a, b, A_log, dt_bias, K**-0.5)
        nacc = torch.full((n,), 2, dtype=torch.int32)
    hdr = layout.views(blob)["header"]
    assert bool((hdr[..., 0] > 0).any()) and bool((hdr[..., 1] == 3).all())
    return blob


def _nan_leftovers(blob: torch.Tensor) -> torch.Tensor:
    """NaN in every float section and garbage in every int section."""
    out = blob.clone()
    vw = LAYOUT_T3.views(out)
    for name in ("bscale", "csmooth", "rec_g"):
        vw[name].fill_(float("nan"))
    for name in ("comp_k", "comp_u", "rec_k", "rec_u"):
        vw[name].fill_(float("nan"))
    vw["z"].fill_(-128)
    vw["header"][..., 2:] = INT32_MIN
    return out


def _leftovers(kind: str, n: int) -> torch.Tensor:
    if kind == "random_bytes":
        g = torch.Generator().manual_seed(21)
        return torch.randint(
            0, 256, (n, HV, LAYOUT_T3.nbytes), generator=g, dtype=torch.uint8
        )
    if kind == "previous_tenant":
        return _previous_tenant(n, seed=2)
    if kind == "previous_tenant_nan":
        return _nan_leftovers(_previous_tenant(n, seed=3))
    if kind == "valid_looking_header":
        # Cursor and pending that pass the range check, on random bytes: the
        # worst case for a store that forgets to reset them.
        blob = _leftovers("random_bytes", n)
        hdr = LAYOUT_T3.views(blob)["header"]
        hdr[..., 0] = LAYOUT_T3.window - 1
        hdr[..., 1] = CAP - (LAYOUT_T3.window - 1)
        return blob
    raise AssertionError(kind)


LEFTOVERS = [
    "random_bytes",
    "previous_tenant",
    "previous_tenant_nan",
    "valid_looking_header",
]


def _meaningful(layout, blob: torch.Tensor) -> dict[str, torch.Tensor]:
    """Everything a later read depends on: boundary, compensators, the
    ``cursor + pending`` live records of each head and the first two header
    words. Rows past a head's live count are leftovers and are not compared."""
    vw = layout.views(blob)
    out = {
        name: vw[name].clone()
        for name in ("z", "bscale", "csmooth", "comp_k", "comp_u")
    }
    hdr = vw["header"][..., :2].clone()
    out["header"] = hdr
    live = hdr[..., 0] + hdr[..., 1]  # [..., HV]
    row_live = torch.arange(layout.capacity) < live[..., None]  # [..., HV, cap]
    out["rec_k"] = torch.where(row_live[..., None], vw["rec_k"], 0)
    out["rec_u"] = torch.where(row_live[..., None], vw["rec_u"], 0)
    out["rec_g"] = torch.where(row_live, vw["rec_g"], 0.0)
    return out


def _assert_same(layout, got: torch.Tensor, want: torch.Tensor) -> None:
    a, b = _meaningful(layout, got), _meaningful(layout, want)
    for name in a:
        assert torch.equal(a[name], b[name]), name


def _first_chunk(layout, blob: torch.Tensor, S: torch.Tensor, prefill_pass: bool):
    """The first chunk of a request on its block. The serving prefill path
    normalizes and materializes the block first (both read the leftovers; the
    rows without an initial state are zeroed afterwards), then stores."""
    if prefill_pass:
        lq.leapquant_normalize(layout, blob, None)
        lq.leapquant_materialize(layout, blob)
    lq.leapquant_store_state(layout, blob, S)


@pytest.mark.parametrize("prefill_pass", [False, True])
@pytest.mark.parametrize("kind", LEFTOVERS)
def test_store_overwrites_everything_a_later_read_depends_on(kind, prefill_pass):
    layout = LAYOUT_T3
    n = 2
    g = torch.Generator().manual_seed(31)
    S = torch.randn(n, HV, V, K, generator=g) * 0.5
    dirty = _leftovers(kind, n)
    clean = torch.zeros_like(dirty)
    _first_chunk(layout, dirty, S, prefill_pass)
    _first_chunk(layout, clean, S, prefill_pass)
    _assert_same(layout, dirty, clean)
    hdr = layout.views(dirty)["header"]
    assert (hdr[..., :2] == 0).all()
    got = lq.leapquant_materialize(layout, dirty)
    assert torch.isfinite(got).all()
    assert torch.equal(got, lq.leapquant_materialize(layout, clean))


@pytest.mark.parametrize("prefill_pass", [False, True])
@pytest.mark.parametrize("kind", LEFTOVERS)
def test_first_decode_after_store_never_replays_leftover_records(kind, prefill_pass):
    """After the prefill store the cursor is 0: the leftover records, whatever
    they hold, are dead. Decode, the next normalize and the window fold that
    follows must match a block that started clean."""
    layout = LAYOUT_T3
    n = 2
    A_log, dt_bias = _params(5)
    g = torch.Generator().manual_seed(32)
    S = torch.randn(n, HV, V, K, generator=g) * 0.5
    dirty = _leftovers(kind, n)
    clean = torch.zeros_like(dirty)
    for blob in (dirty, clean):
        _first_chunk(layout, blob, S, prefill_pass)
    nacc = None
    for step in range(layout.window + 4):  # past the first fold
        q, k, v, a, b = _tokens(n, 3, 600 + step)
        for blob in (dirty, clean):
            lq.leapquant_normalize(layout, blob, nacc)
        o_dirty = lq.leapquant_decode(
            layout, dirty, q, k, v, a, b, A_log, dt_bias, K**-0.5
        )
        o_clean = lq.leapquant_decode(
            layout, clean, q, k, v, a, b, A_log, dt_bias, K**-0.5
        )
        assert torch.isfinite(o_dirty).all()
        assert torch.equal(o_dirty, o_clean), step
        nacc = torch.tensor([1 + step % 3, 3], dtype=torch.int32)
    lq.leapquant_normalize(layout, dirty, nacc)
    lq.leapquant_normalize(layout, clean, nacc)
    _assert_same(layout, dirty, clean)


def test_fresh_zero_block_is_the_zero_state():
    """A zeroed block with no store is the zero state: its first decode equals
    FP32 decode from zeros, and normalize leaves it zero."""
    layout = LAYOUT_T3
    n = 2
    A_log, dt_bias = _params(6)
    blob = torch.zeros(n, HV, layout.nbytes, dtype=torch.uint8)
    q, k, v, a, b = _tokens(n, 3, 77)
    lq.leapquant_normalize(layout, blob, None)
    assert not blob.any()
    o = lq.leapquant_decode(layout, blob, q, k, v, a, b, A_log, dt_bias, K**-0.5)
    qn = lq.l2norm(q).repeat_interleave(HV // H, 2) * K**-0.5
    kn = lq.l2norm(k).repeat_interleave(HV // H, 2)
    g_, beta = lq.gdn_gates(a, b, A_log, dt_bias)
    S = torch.zeros(n, HV, V, K)
    for t in range(3):
        S, o_ref, _ = lq.gdn_step_fp32(
            S, qn[:, t], kn[:, t], v[:, t], g_[:, t], beta[:, t]
        )
        assert _rel(o[:, t], o_ref) < 2e-3


@pytest.mark.parametrize("poison", ["nan", "inf", "huge"])
def test_records_beyond_cursor_are_never_read(poison):
    """Rows >= cursor of the ring hold a previous window (or another request).
    Normalize, materialize and decode read rows < cursor only."""
    layout = LAYOUT_T3
    n = 3
    A_log, dt_bias = _params(7)
    clean = _previous_tenant(n, seed=4)
    lq.leapquant_normalize(layout, clean, torch.tensor([2, 3, 1], dtype=torch.int32))
    cursor = layout.views(clean)["header"][..., 0]
    assert int(cursor.min()) > 0 and int(cursor.max()) < layout.capacity
    dirty = clean.clone()
    vw = layout.views(dirty)
    value = {"nan": float("nan"), "inf": float("inf"), "huge": 6e4}[poison]
    for i in range(n):
        for h in range(HV):
            c = int(cursor[i, h])
            for name in ("rec_k", "rec_u"):
                vw[name][i, h, c:] = value
            vw["rec_g"][i, h, c:] = value
    assert torch.equal(
        lq.leapquant_materialize(layout, dirty), lq.leapquant_materialize(layout, clean)
    )
    q, k, v, a, b = _tokens(n, 3, 88)
    o_dirty = lq.leapquant_decode(layout, dirty, q, k, v, a, b, A_log, dt_bias, K**-0.5)
    o_clean = lq.leapquant_decode(layout, clean, q, k, v, a, b, A_log, dt_bias, K**-0.5)
    assert torch.isfinite(o_dirty).all()
    assert torch.equal(o_dirty, o_clean)


INVALID_HEADERS = [
    (-1, 0),
    (0, -1),
    (CAP + 1, 0),
    (CAP, 1),
    (3, CAP - 2),
    (INT32_MAX, INT32_MAX),
    (INT32_MIN, 5),
    (5, INT32_MIN),
    (INT32_MAX, 0),
]


@pytest.mark.parametrize("cursor,pending", INVALID_HEADERS)
def test_out_of_range_header_reads_as_empty(cursor, pending):
    """Stale header words that fail the range check mean cursor = pending = 0
    for materialize, normalize and decode, as in ``lq_header_ok``."""
    layout = LAYOUT_T3
    n = 2
    A_log, dt_bias = _params(8)
    g = torch.Generator().manual_seed(41)
    base = torch.zeros(n, HV, layout.nbytes, dtype=torch.uint8)
    lq.leapquant_store_state(layout, base, torch.randn(n, HV, V, K, generator=g) * 0.4)
    reference = base.clone()
    bad = base.clone()
    bad_vw = layout.views(bad)
    bad_vw["header"][..., 0] = cursor
    bad_vw["header"][..., 1] = pending
    for name in ("rec_k", "rec_u", "rec_g"):
        bad_vw[name].fill_(float("nan"))
    # materialize
    assert torch.equal(
        lq.leapquant_materialize(layout, bad),
        lq.leapquant_materialize(layout, reference),
    )
    # normalize (with and without an accepted count) resets the header
    for nacc in (None, torch.tensor([3, 3], dtype=torch.int32)):
        b2, r2 = bad.clone(), reference.clone()
        lq.leapquant_normalize(layout, b2, nacc)
        lq.leapquant_normalize(layout, r2, nacc)
        hdr = layout.views(b2)["header"]
        assert (hdr[..., :2] == 0).all()
        assert torch.equal(
            lq.leapquant_materialize(layout, b2), lq.leapquant_materialize(layout, r2)
        )
    # decode reads the boundary only
    q, k, v, a, b = _tokens(n, 3, 91)
    o_bad = lq.leapquant_decode(layout, bad, q, k, v, a, b, A_log, dt_bias, K**-0.5)
    o_ref = lq.leapquant_decode(
        layout, reference, q, k, v, a, b, A_log, dt_bias, K**-0.5
    )
    assert torch.isfinite(o_bad).all()
    assert torch.equal(o_bad, o_ref)


def test_copy_slot_hands_out_only_its_own_blob():
    """The align pre-copy rewrites the whole destination: the destination's
    leftovers (any section) are not in the copy."""
    layout = LAYOUT_T3
    src = _previous_tenant(1, seed=9)[0]
    for bias in (0, 1, 2):
        copy = lq.leapquant_copy_slot(layout, src, bias)
        assert torch.equal(layout.views(copy)["z"], layout.views(src)["z"]), (
            "boundary must come from the source"
        )
        hdr = layout.views(copy)["header"]
        src_hdr = layout.views(src)["header"]
        assert (hdr[..., 1] == 0).all()
        want = src_hdr[..., 0] + torch.clamp(src_hdr[..., 1], max=bias + 1)
        assert torch.equal(hdr[..., 0], want)
