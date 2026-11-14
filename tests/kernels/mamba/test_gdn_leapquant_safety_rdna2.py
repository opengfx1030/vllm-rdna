# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The safety checks below follow the list in the LeapQuant RFC, vllm-project/vllm
# #64725 (lihaoyang-amd): section 2.2 (NaN, determinism, flush race), section 4
# item 11 (unit tests) and section 5 (numerical safety, request isolation). The
# RFC has no code; every test here is original work for this fork.
"""Safety tests for the LeapQuant GDN HIP kernels (gdn_lq_*_rdna2) on gfx103x.

STATUS: written without GPU access. None of these tests has run on a GPU yet;
the same checks run on the CPU oracle in test_gdn_leapquant_oracle.py. The
pending-GPU list is in docs/rdna2/leapquant-gdn-numerics-and-isolation-2026-11-14.md.

* Fold: zero, tiny, huge and exactly low-rank states give finite, accurate
  blobs with orthonormal compensators, and a degenerate head does not change
  its neighbours.
* Determinism: the same inputs, 200 times, with matmuls on a second stream,
  give bitwise equal outputs and blobs (the MonoKernel too).
* Stale state: a recycled block holds the bytes of another request. Nothing in
  them (scales, smoothing, compensators, records, cursor/pending) may change
  what the new request reads.
* Flush race: many (block, head) pairs with a full window, MTP accepted counts,
  and one grouped normalize over all layers must equal per-layer launches.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops import leapquant_gdn as lq

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx103" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="RDNA2 (gfx103x) GPU required",
)

DEVICE = "cuda"
K = V = 128
H, HV = 2, 6
WINDOW, R, ITERS = 16, 4, 2
INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
N_REPEATS = 200  # determinism: runs of the same inputs
GROUPED_REPEATS = 25  # flush race: grouped vs per-layer launches

# Error bounds shared with the CPU oracle tests.
INT8_BOUND = 1e-2  # dense state: INT8 per row
LOW_RANK_BOUND = 1e-4  # rank <= num_comp: the compensators carry the state
ORTHO_TOL = 2e-3  # fp16 rounding of the orthonormal compensator rows
RANGED_ELEMS_BOUND = 2e-2  # 34 decades per element: INT8-limited


# --------------------------------------------------------------------------
# Launch helpers (signatures as in test_gdn_leapquant_rdna2.py).
# --------------------------------------------------------------------------


def _layout(max_query_len: int, *, window: int = WINDOW) -> lq.LeapQuantLayout:
    return lq.LeapQuantLayout(
        K,
        V,
        window=window,
        num_comp=R,
        max_query_len=max_query_len,
        power_iters=ITERS,
    )


def _ops():
    return torch.ops._rocm_C


def _normalize(L, blob, idx, num_acc, layer_offsets=None):
    args = (blob, idx, num_acc, V, L.num_comp, L.capacity, L.window, L.power_iters)
    if layer_offsets is None:
        _ops().gdn_lq_normalize_rdna2(*args)
    else:
        _ops().gdn_lq_normalize_rdna2(*args, layer_offsets)


def _store(L, state, blob, idx):
    _ops().gdn_lq_store_rdna2(
        state, blob, idx, L.num_comp, L.capacity, L.window, L.power_iters
    )


def _materialize(L, blob, idx, n, poison=0.0):
    out = torch.full((n, HV, V, K), poison, dtype=torch.float32, device=blob.device)
    _ops().gdn_lq_materialize_rdna2(blob, out, idx, L.num_comp, L.capacity, L.window)
    return out


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    A_log = torch.log(torch.rand(HV, generator=g) * 0.99 + 0.01).to(DEVICE)
    dt_bias = (torch.randn(HV, generator=g) * 0.1).to(DEVICE)
    return A_log, dt_bias


def _tokens(n, t, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, t, H, K, generator=g)
    k = torch.randn(n, t, H, K, generator=g)
    v = torch.randn(n, t, HV, V, generator=g)
    a = torch.randn(n, t, HV, generator=g) - 3
    b = torch.randn(n, t, HV, generator=g)
    return [x.half().to(DEVICE) for x in (q, k, v, a, b)]


def _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias, *, poison=0.0):
    n, t = q.shape[:2]
    out = torch.full((1, n * t, HV, V), poison, dtype=torch.float16, device=DEVICE)
    cu = torch.arange(0, n * t + 1, t, dtype=torch.int32, device=DEVICE)
    _ops().gdn_lq_decode_rdna2(
        q.reshape(1, n * t, H, K),
        k.reshape(1, n * t, H, K),
        v.reshape(1, n * t, HV, V),
        a.reshape(n * t, HV),
        b.reshape(n * t, HV),
        A_log,
        dt_bias,
        out,
        blob,
        cu,
        idx,
        K**-0.5,
        L.num_comp,
        L.capacity,
        L.window,
    )
    return out.view(n, t, HV, V)


def _accepted(n, gen, lo=1, hi=3):
    """MTP-style accepted counts: draft tokens accepted per request, lo..hi."""
    return torch.randint(lo, hi + 1, (n,), generator=gen, dtype=torch.int32).to(DEVICE)


def _const(n, value):
    return torch.full((n,), value, dtype=torch.int32, device=DEVICE)


def _rel(a, b):
    # float64: the norm of a 1e30 state overflows float32.
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


# --------------------------------------------------------------------------
# Blob builders and comparison helpers.
# --------------------------------------------------------------------------


def _new_blob(L, n, seed):
    """One layer: block 0 is the NULL block (filled with 7), blocks 1..n hold
    n requests stored from a random state. Returns (blob, state indices)."""
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    blob[0].fill_(7)
    g = torch.Generator().manual_seed(seed)
    S = (torch.randn(n, HV, V, K, generator=g) * 0.5).to(DEVICE)
    idx = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
    _store(L, S, blob, idx)
    return blob, idx


def _advance(L, blob, idx, params, *, steps, seed, accepts):
    """MTP steps: normalize with the counts accepted by the step before, then
    verify 3 tokens. ``accepts(step)`` gives the counts the next step commits.
    Returns the counts the next normalize must commit."""
    n = idx.numel()
    acc = None
    for step in range(steps):
        _normalize(L, blob, idx, acc)
        _decode(L, blob, idx, *_tokens(n, 3, seed=seed + step), *params)
        acc = accepts(step)
    return acc


def _meaningful(L, blob):
    """Everything a later read depends on: boundary, compensators, the first
    ``cursor + pending`` records of each head and header words 0 and 1. Rows
    past a head's live count are leftovers; header words 2 and 3 are not part
    of the contract."""
    vw = L.views(blob)
    out = {
        name: vw[name].clone()
        for name in ("z", "bscale", "csmooth", "comp_k", "comp_u")
    }
    hdr = vw["header"][..., :2].clone()
    out["header"] = hdr
    live = hdr[..., 0] + hdr[..., 1]  # [..., HV]
    row_live = torch.arange(L.capacity, device=blob.device) < live[..., None]
    out["rec_k"] = torch.where(row_live[..., None], vw["rec_k"], 0)
    out["rec_u"] = torch.where(row_live[..., None], vw["rec_u"], 0)
    out["rec_g"] = torch.where(row_live, vw["rec_g"], 0.0)
    return out


def _poison_dead_rows(L, blob, value):
    """Overwrite the record rows no read may reach (rows >= cursor + pending)."""
    vw = L.views(blob)
    live = vw["header"][..., 0].long() + vw["header"][..., 1].long()
    dead = torch.arange(L.capacity, device=blob.device) >= live[..., None]
    vw["rec_k"].masked_fill_(dead[..., None], value)
    vw["rec_u"].masked_fill_(dead[..., None], value)
    vw["rec_g"].masked_fill_(dead, value)


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _assert_bitwise(got, want, what):
    """Recursive bit-exact comparison (NaN payloads included); names the first
    difference."""
    if isinstance(want, dict):
        assert got.keys() == want.keys(), what
        for key in want:
            _assert_bitwise(got[key], want[key], f"{what}.{key}")
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want), what
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_bitwise(g, w, f"{what}[{i}]")
    else:
        assert got.shape == want.shape and got.dtype == want.dtype, what
        a, b = _bits(got), _bits(want)
        assert torch.equal(a, b), (
            f"{what}: {int((a != b).sum())} of {a.numel()} bytes differ"
        )


class _Load:
    """Matmuls on a second stream, so the kernels under test share the compute
    units with unrelated work and block scheduling changes from run to run."""

    def __init__(self, size=2048, depth=6):
        self.stream = torch.cuda.Stream()
        self.a = torch.randn(size, size, dtype=torch.float16, device=DEVICE)
        self.b = torch.randn(size, size, dtype=torch.float16, device=DEVICE)
        self.depth = depth

    def kick(self):
        # No event links this stream to the main one: the work overlaps.
        with torch.cuda.stream(self.stream):
            c = self.a
            for _ in range(self.depth):
                c = (c @ self.b).clamp_(-1, 1)

    def wait(self):
        self.stream.synchronize()


# --------------------------------------------------------------------------
# 1. Fold of degenerate states.
# --------------------------------------------------------------------------


def _low_rank(rank, scale, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(V, rank, generator=g)
    b = torch.randn(K, rank, generator=g)
    return (a @ b.T) * scale


def _integer_low_rank(rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(-3, 4, (V, rank), generator=g).float()
    b = torch.randint(-3, 4, (K, rank), generator=g).float()
    return a @ b.T


def _fold_cases():
    """Name -> (state [V, K], relative error bound; None = must stay zero)."""
    g = torch.Generator().manual_seed(7)
    single = torch.zeros(V, K)
    single[5, 9] = 3.0
    four_rows = torch.zeros(V, K)
    four_rows[:4] = torch.randn(4, K, generator=g)
    row_scale = 10.0 ** (torch.rand(V, generator=g) * 12 - 6)
    cases = {
        "zero": (torch.zeros(V, K), None),
        "single_entry": (single, INT8_BOUND),
        "four_rows": (four_rows, INT8_BOUND),
        "constant": (torch.full((V, K), 0.25), INT8_BOUND),
        "ranged_rows": (
            torch.randn(V, K, generator=g) * row_scale[:, None],
            INT8_BOUND,
        ),
        "ranged_elems": (
            torch.randn(V, K, generator=g)
            * 10.0 ** (torch.rand(V, K, generator=g) * 34 - 30),
            RANGED_ELEMS_BOUND,
        ),
        "outlier_3e4": (
            torch.randn(V, K, generator=g)
            + 3e4 * (torch.randn(V, 1, generator=g) * torch.randn(1, K, generator=g)),
            INT8_BOUND,
        ),
    }
    for exp in (-30, -12, 12, 30, 36, 37):
        cases[f"dense_1e{exp}"] = (
            torch.randn(V, K, generator=g) * 10.0**exp,
            INT8_BOUND,
        )
    cases["rank1_1e37"] = (_low_rank(1, 1e37), INT8_BOUND)
    dense_max = torch.randn(V, K, generator=g)
    for kind, base in (("dense", dense_max), ("rank1", _low_rank(1, 1.0))):
        cases[f"{kind}_at_float_max"] = (base / base.abs().max() * 3.0e38, INT8_BOUND)
    for rank in (1, 2, 3, 4):
        for scale in (1e-6, 1e-3, 1.0, 1e3):
            cases[f"rank{rank}_{scale:g}"] = (_low_rank(rank, scale), LOW_RANK_BOUND)
    for rank in (1, 2, 3):
        cases[f"integer_rank{rank}"] = (_integer_low_rank(rank), LOW_RANK_BOUND)
    return cases


def _assert_orthonormal_or_zero(rows):
    """``rows`` [r, V]: every row is zero or unit, all pairs are orthogonal."""
    gram = rows.float() @ rows.float().T
    diag = torch.diagonal(gram)
    live = diag > 0.5
    assert bool(((diag - 1).abs() < ORTHO_TOL)[live].all()), gram
    assert bool((diag[~live].abs() < ORTHO_TOL).all()), gram
    off = gram - torch.diag(diag)
    assert float(off.abs().max()) < ORTHO_TOL, gram


def test_fold_of_degenerate_states_is_finite_accurate_and_orthonormal():
    """The RFC prototype produced NaN for exactly low-rank states. The HIP fold
    mirrors the oracle guard: scale by the max |entry|, two sweeps, drop
    dependent columns. States near the float limit also cover the smoothing sum,
    which overflowed to inf and made 0 * inf = NaN in every later read."""
    L = _layout(1)
    cases = _fold_cases()
    names = list(cases)
    n = len(names)
    S = torch.stack([cases[nm][0] for nm in names])
    S = S[:, None].expand(n, HV, V, K).contiguous().to(DEVICE)
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    idx = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
    _store(L, S, blob, idx)
    got = _materialize(L, blob, idx, n, poison=float("nan"))
    vw = L.views(blob[1:])
    for i, name in enumerate(names):
        state, bound = cases[name]
        assert torch.isfinite(got[i]).all(), name
        for sect in ("bscale", "csmooth", "comp_k", "comp_u"):
            assert torch.isfinite(vw[sect][i].float()).all(), (name, sect)
        assert bool((vw["bscale"][i] >= 0).all()), name
        # All heads fold the same state: the fold is batch-invariant.
        assert torch.equal(got[i], got[i, :1].expand_as(got[i])), name
        if bound is None:
            assert bool((got[i] == 0).all()), name
        else:
            err = _rel(got[i, 0], state.to(DEVICE))
            assert err < bound, f"{name}: relative error {err:.3e} >= {bound}"
        for h in range(HV):
            _assert_orthonormal_or_zero(vw["comp_u"][i, h])


def test_degenerate_heads_do_not_change_their_neighbours():
    """One launch folds many heads; a zero, tiny or rank-1 head is bitwise
    invisible to the dense head next to it."""
    L = _layout(1)
    g = torch.Generator().manual_seed(11)
    dense = torch.randn(V, K, generator=g)
    neighbours = [
        torch.zeros(V, K),
        _low_rank(1, 1e-30),
        _low_rank(1, 1.0),
        _integer_low_rank(2),
    ]

    def fold(states):
        n = len(states)
        S = torch.stack(states)[:, None].expand(n, HV, V, K).contiguous()
        blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
        idx = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
        _store(L, S.to(DEVICE), blob, idx)
        return _meaningful(L, blob[1:])

    mixed = fold(neighbours + [dense])
    alone = fold([dense])
    last = {name: part[-1:] for name, part in mixed.items()}
    _assert_bitwise(last, alone, "dense head next to degenerate heads")


def test_flush_of_an_exactly_rank_one_trajectory():
    """Identical keys make the state exactly rank 1; the window fills with
    records and each flush folds it. No NaN, no drift from the FP32 state."""
    L = _layout(1)
    A_log, dt_bias = _params(5)
    g = torch.Generator().manual_seed(5)
    k_fixed = torch.randn(1, 1, H, K, generator=g).half().to(DEVICE)
    q = torch.randn(1, 1, H, K, generator=g).half().to(DEVICE)
    blob = torch.zeros(2, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    idx = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    S_ref = torch.zeros(1, HV, V, K, device=DEVICE)
    qn = lq.l2norm(q).repeat_interleave(HV // H, 2) * K**-0.5
    kn = lq.l2norm(k_fixed).repeat_interleave(HV // H, 2)
    errs = []
    for _ in range(40):  # 2.5 windows: two flushes
        v = torch.randn(1, 1, HV, V, generator=g).half().to(DEVICE)
        a = (torch.randn(1, 1, HV, generator=g) - 3).half().to(DEVICE)
        b = torch.randn(1, 1, HV, generator=g).half().to(DEVICE)
        _normalize(L, blob, idx, None)
        o = _decode(L, blob, idx, q, k_fixed, v, a, b, A_log, dt_bias)
        gg, beta = lq.gdn_gates(a, b, A_log, dt_bias)
        S_ref, o_ref, _ = lq.gdn_step_fp32(
            S_ref, qn[:, 0], kn[:, 0], v[:, 0], gg[:, 0], beta[:, 0]
        )
        assert torch.isfinite(o).all()
        errs.append(_rel(o[:, 0], o_ref))
    # The CPU oracle gives 5.4e-4 for this trajectory.
    assert max(errs) < 2e-3, f"worst output error {max(errs):.2e}"


# --------------------------------------------------------------------------
# 2. Determinism under load.
# --------------------------------------------------------------------------

POISONS = (float("nan"), 1234.0, float("inf"), -5.0)


def _trajectory(L, params, n, it):
    """Store -> WINDOW // 2 MTP steps -> two more steps with random accepted
    counts -> materialize, on a fresh zero blob. Every output buffer is
    poisoned with a value that changes with ``it``, so an unwritten element
    differs between repetitions. Returns everything a caller could observe."""
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    blob[0].fill_(7)
    g = torch.Generator().manual_seed(1)
    S = (torch.randn(n, HV, V, K, generator=g) * 0.5).to(DEVICE)
    idx = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
    gen = torch.Generator().manual_seed(9)
    _store(L, S, blob, idx)
    outs, folds = [], 0
    last = L.views(blob[1:])["header"][..., 0].clone()
    for step in range(WINDOW // 2 + 2):
        # Window - 2 committed + 3 pending after the first WINDOW // 2 steps:
        # accepting 2 or 3 folds, accepting 1 does not.
        if step == 0:
            acc = None
        elif step < WINDOW // 2:
            acc = _const(n, 2)
        else:
            acc = _accepted(n, gen)
        _normalize(L, blob, idx, acc)
        cursor = L.views(blob[1:])["header"][..., 0]
        folds += int((cursor < last).sum())
        last = cursor.clone()
        outs.append(
            _decode(
                L, blob, idx, *_tokens(n, 3, seed=200 + step), *params,
                poison=POISONS[(it + step) % len(POISONS)],
            )
        )  # fmt: skip
    _normalize(L, blob, idx, _accepted(n, gen))
    state = _materialize(L, blob, idx, n, poison=POISONS[it % len(POISONS)])
    return {
        "outs": outs,
        "state": state,
        "blob": blob,
        "folds": torch.tensor([folds]),
    }


def test_store_normalize_decode_are_bitwise_reproducible_under_load():
    """Folds (store: cold start, normalize: warm start), replays and decodes on
    identical inputs, N_REPEATS times, with matmuls competing for the compute
    units. Any atomic-order, unwritten-element or shared-memory race shows up
    as a bit that changes between runs."""
    L = _layout(3)
    params = _params(3)
    n = 16
    reference = _trajectory(L, params, n, 0)
    assert bool(torch.isfinite(reference["state"]).all())
    assert int(reference["folds"]) >= n, "each request must flush at least once"
    load = _Load()
    for it in range(1, N_REPEATS):
        load.kick()
        _assert_bitwise(_trajectory(L, params, n, it), reference, f"repeat {it}")
    load.wait()


SLOTS_MONO = 24


def _mono_case(window):
    """MonoKernel inputs: 4 MTP sequences of 3 tokens (one is NULL/padded)."""
    L = _layout(3, window=window)
    n, t = 4, 3
    cd = 2 * H * K + HV * V
    g = torch.Generator().manual_seed(6)
    perm = torch.randperm(SLOTS_MONO - 1, generator=g)[: n * t] + 1
    sidx = perm.int().view(n, t).to(DEVICE)
    sidx[1] = 0  # null / padded sequence
    return {
        "L": L,
        "n": n,
        "t": t,
        "params": _params(5),
        "conv_w": (torch.randn(cd, 4, generator=g) * 0.3).half().to(DEVICE),
        "conv_b": (torch.randn(cd, generator=g) * 0.1).half().to(DEVICE),
        "norm_w": (1.0 + torch.randn(V, generator=g) * 0.1).half().to(DEVICE),
        "sidx": sidx,
        "home": sidx[:, 0].contiguous(),
        "qsl": torch.arange(0, n * t + 1, t, dtype=torch.int32, device=DEVICE),
        "scratch": torch.zeros(
            int(_ops().mono_gdn_rdna2_scratch_bytes(64, HV, H)),
            dtype=torch.uint8,
            device=DEVICE,
        ),
    }


def _mono_inputs(case, step):
    """Fresh qkvz, ba and conv state for one step (a copy on every call)."""
    g = torch.Generator().manual_seed(60 + step)
    cd = 2 * H * K + HV * V

    def rn(*s, std):
        return (torch.randn(*s, generator=g) * std).half().to(DEVICE)

    m = case["n"] * case["t"]
    return (
        rn(m, cd + HV * V, std=0.5),
        rn(m, 2 * HV, std=1.0),
        rn(SLOTS_MONO, cd, 3 + case["t"] - 1, std=0.5),
    )


def _mono_nacc(case, step):
    if step == 0:
        return _const(case["n"], 1)  # fresh blobs hold no pending records
    return torch.tensor([3, 1, 2, 1], dtype=torch.int32, device=DEVICE)


def _mono_step(case, blob, step, y_poison=0.0):
    """normalize, then one MonoKernel layer on LeapQuant blobs: (y, conv)."""
    L, nacc = case["L"], _mono_nacc(case, step)
    qkvz, ba, conv = _mono_inputs(case, step)
    y = torch.full(
        (case["n"] * case["t"], HV * V), y_poison, dtype=torch.float16, device=DEVICE
    )
    _normalize(L, blob, case["home"], nacc)
    A_log, dt_bias = case["params"]
    _ops().mono_gdn_lq_rdna2(
        qkvz, ba, case["conv_w"], case["conv_b"], conv, blob,
        A_log.float(), dt_bias.float(), case["norm_w"], case["sidx"],
        nacc, case["qsl"], y, case["scratch"], H, K**-0.5, 1e-6, 0,
        L.num_comp, L.capacity, L.window,
    )  # fmt: skip
    return y, conv


def _mono_run(case, blob, steps, it):
    """``steps`` MonoKernel layers; returns the outputs and the final blob."""
    ys, convs = [], []
    for step in range(steps):
        y, conv = _mono_step(
            case, blob, step, y_poison=POISONS[(it + step) % len(POISONS)]
        )
        ys.append(y)
        convs.append(conv)
    return {"ys": ys, "convs": convs, "blob": blob}


def _mono_stored_blob(case, base):
    """``base`` blocks, with the live home blocks stored from a random state."""
    live = case["home"][case["home"] > 0]
    g = torch.Generator().manual_seed(8)
    S = (torch.randn(live.numel(), HV, V, K, generator=g) * 0.3).to(DEVICE)
    _store(case["L"], S, base, live)
    return base


def test_mono_gdn_lq_is_bitwise_reproducible_under_load():
    """Same as above for the MonoKernel, which fuses conv, gates and the
    LeapQuant replay in one persistent kernel. The scratch buffer is reused
    between runs, as in serving."""
    case = _mono_case(WINDOW)
    L = case["L"]
    zeros = torch.zeros(SLOTS_MONO, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    base = _mono_stored_blob(case, zeros)
    reference = _mono_run(case, base.clone(), 4, 0)
    for y in reference["ys"]:
        assert bool(torch.isfinite(y).all())
    load = _Load()
    for it in range(1, N_REPEATS):
        load.kick()
        _assert_bitwise(_mono_run(case, base.clone(), 4, it), reference, f"repeat {it}")
    load.wait()


# --------------------------------------------------------------------------
# 3. Stale state of a recycled block.
# --------------------------------------------------------------------------
# GDN blocks are not zeroed when the allocator hands them out. The serving
# contract (gdn_attn.py: a request without a prior state runs as a prefill;
# the prefill zeroes its initial state and stores) overwrites every section a
# read depends on before the first decode. These tests hold the kernels to the
# contract, bit for bit.

LEFTOVERS = (
    "random_bytes",
    "previous_tenant",
    "previous_tenant_nan",
    "valid_looking_header",
)


def _leftover_blob(L, nblocks, kind, seed):
    """Bytes a recycled block can hold before its new request's first store."""
    if kind in ("random_bytes", "valid_looking_header"):
        g = torch.Generator().manual_seed(seed)
        blob = torch.randint(
            0, 256, (nblocks, HV, L.nbytes), dtype=torch.uint8, generator=g
        ).to(DEVICE)
        if kind == "valid_looking_header":
            hdr = L.views(blob)["header"]
            hdr[..., 0] = L.window - 1
            hdr[..., 1] = 2
        return blob
    # A request that ran on this block and left mid-window: real boundary,
    # compensators and records, header with cursor + pending records.
    blob, idx = _new_blob(L, nblocks - 1, seed)
    gen = torch.Generator().manual_seed(seed)
    _advance(
        L, blob, idx, _params(seed), steps=WINDOW // 2 - 1, seed=40 * seed,
        accepts=lambda step: _accepted(nblocks - 1, gen),
    )  # fmt: skip
    if kind == "previous_tenant_nan":
        vw = L.views(blob)
        for name in ("bscale", "csmooth", "comp_k", "comp_u"):
            vw[name].fill_(float("nan"))
        for name in ("rec_k", "rec_u", "rec_g"):
            vw[name].fill_(float("nan"))
    return blob


def _tenant_trace(L, blob, idx, S, params, prefill_pass, steps=2 * WINDOW // 3 + 2):
    """A new request on recycled blocks: the prefill pass (optional), the store,
    then MTP steps through the first flush. Returns what the request can
    observe."""
    n = idx.numel()
    gen = torch.Generator().manual_seed(77)
    if prefill_pass:
        # The serving prefill path normalizes and materializes the leftovers
        # first and zeroes the rows without an initial state afterwards.
        _normalize(L, blob, idx, None)
        _materialize(L, blob, idx, n)
    _store(L, S, blob, idx)
    trace = [_meaningful(L, blob[1:])]
    acc = None
    folded = False
    last = L.views(blob[1:])["header"][..., 0].clone()
    for step in range(steps):
        _normalize(L, blob, idx, acc)
        cursor = L.views(blob[1:])["header"][..., 0]
        folded |= bool((cursor < last).any())
        last = cursor.clone()
        out = _decode(L, blob, idx, *_tokens(n, 3, seed=400 + step), *params)
        trace.append({"out": out, "blob": _meaningful(L, blob[1:])})
        acc = _accepted(n, gen)
    _normalize(L, blob, idx, acc)
    trace.append(_materialize(L, blob, idx, n))
    assert folded, "the trace never crossed a window flush"
    return trace


@pytest.mark.parametrize("prefill_pass", [False, True])
@pytest.mark.parametrize("kind", LEFTOVERS)
def test_new_tenant_is_independent_of_the_leftovers_of_a_recycled_block(
    kind, prefill_pass
):
    """Store, first decode, the flush that follows and every later read equal
    those of a block that started from zeros, bit for bit. With prefill_pass
    the leftovers also go through normalize and materialize before the store,
    as in the serving prefill path."""
    L = _layout(3)
    params = _params(3)
    n = 4
    g = torch.Generator().manual_seed(5)
    S = (torch.randn(n, HV, V, K, generator=g) * 0.5).to(DEVICE)
    idx = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
    clean = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    dirty = _leftover_blob(L, n + 1, kind, seed=5)
    _assert_bitwise(
        _tenant_trace(L, dirty, idx, S, params, prefill_pass),
        _tenant_trace(L, clean, idx, S, params, prefill_pass),
        kind,
    )


def test_first_decode_on_a_fresh_zero_block_without_a_store():
    """A zeroed block (a masked first chunk that stored zeros, or an allocator
    that zeroes) is the zero state: the first decode matches FP32 from zeros."""
    L = _layout(3)
    params = _params(4)
    n = 3
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    idx = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
    tokens = _tokens(n, 3, seed=12)
    _normalize(L, blob, idx, None)
    got = _decode(L, blob, idx, *tokens, *params)
    q, k, v, a, b = (x.cpu() for x in tokens)
    zeros = torch.zeros(n, HV, L.nbytes, dtype=torch.uint8)
    ref = lq.leapquant_decode(
        L, zeros, q, k, v, a, b, params[0].cpu(), params[1].cpu(), K**-0.5
    )
    assert _rel(got.cpu(), ref) < 2e-3
    _normalize(L, blob, idx, _const(n, 2))
    assert torch.isfinite(_materialize(L, blob, idx, n)).all()
    hdr = L.views(blob[1:])["header"]
    assert bool((hdr[..., 0] == 2).all()) and bool((hdr[..., 1] == 0).all())


def test_mono_gdn_lq_on_a_fresh_zero_block_matches_fp32_mono():
    """The MonoKernel's first step on zeroed blocks (no store) equals the FP32
    MonoKernel started from a zero state."""
    case = _mono_case(WINDOW)
    L, t = case["L"], case["t"]
    blob = torch.zeros(SLOTS_MONO, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
    y, conv = _mono_step(case, blob, 0)
    qkvz, ba, conv_ref = _mono_inputs(case, 0)
    A_log, dt_bias = case["params"]
    ssm = torch.zeros(SLOTS_MONO, HV, V, K, device=DEVICE)
    y_ref = torch.empty_like(y)
    _ops().mono_gdn_rdna2(
        qkvz, ba, case["conv_w"], case["conv_b"], conv_ref, ssm,
        A_log.float(), dt_bias.float(), case["norm_w"], case["sidx"],
        _mono_nacc(case, 0), case["qsl"], y_ref, case["scratch"],
        H, K**-0.5, 1e-6, 0,
    )  # fmt: skip
    torch.accelerator.synchronize()
    rows = (case["home"] > 0).repeat_interleave(t)
    assert bool((y[~rows] == 0).all())
    assert _rel(y[rows], y_ref[rows]) < 2e-3
    assert torch.equal(conv, conv_ref)


@pytest.mark.parametrize("kind", LEFTOVERS)
def test_mono_gdn_lq_after_a_store_is_independent_of_leftovers(kind):
    runs = []
    for dirty in (False, True):
        case = _mono_case(WINDOW)  # own scratch: only the blob leftovers differ
        L = case["L"]
        base = (
            _leftover_blob(L, SLOTS_MONO, kind, seed=6)
            if dirty
            else torch.zeros(SLOTS_MONO, HV, L.nbytes, dtype=torch.uint8, device=DEVICE)
        )
        blob = _mono_stored_blob(case, base)
        run = _mono_run(case, blob, 4, 0)
        live = case["home"][case["home"] > 0].long()
        run["blob"] = _meaningful(L, blob[live])
        runs.append(run)
    _assert_bitwise(runs[1], runs[0], kind)


# Header words a previous tenant can leave behind that fail the range check
# 0 <= cursor, 0 <= pending, cursor + pending <= capacity (lq_header_ok). The
# last ones would index far outside the record ring if a kernel trusted them.
CAP = _layout(3).capacity
INVALID_HEADERS = [
    (-1, 0),
    (0, -1),
    (CAP + 1, 0),
    (0, CAP + 1),
    (CAP, 1),
    (WINDOW, CAP),
    (INT32_MIN, INT32_MIN),
    (INT32_MAX, 0),
    (0, INT32_MAX),
    (INT32_MAX, INT32_MAX),
]


def _read_pipeline(L, blob, idx, params):
    """Normalize -> decode -> normalize -> materialize on blocks 1..n."""
    n = idx.numel()
    gen = torch.Generator().manual_seed(3)
    _normalize(L, blob, idx, _accepted(n, gen))
    out = _decode(L, blob, idx, *_tokens(n, 3, seed=500), *params)
    _normalize(L, blob, idx, _accepted(n, gen))
    return {
        "out": out,
        "state": _materialize(L, blob, idx, n),
        "blob": _meaningful(L, blob[1 : n + 1]),
    }


@pytest.mark.parametrize("cursor,pending", INVALID_HEADERS)
def test_header_that_fails_the_range_check_reads_as_empty(cursor, pending):
    """normalize, decode and materialize read such a header as cursor = pending
    = 0 and index no record outside the ring. The record rows of the dirty
    block hold NaN, so a read of any of them shows; the blocks next to it stay
    as they were."""
    L = _layout(3)
    params = _params(5)
    n = 3
    blob, idx = _new_blob(L, n, seed=9)
    guard = torch.randint(
        0, 256, (2, HV, L.nbytes), dtype=torch.uint8,
        generator=torch.Generator().manual_seed(1),
    ).to(DEVICE)  # fmt: skip
    blob = torch.cat([blob, guard])  # blocks n + 1 and n + 2: never addressed
    guard_before = blob[n + 1 :].clone()
    clean, dirty = blob.clone(), blob.clone()
    vw = L.views(dirty[1 : n + 1])
    vw["header"][..., 0] = cursor
    vw["header"][..., 1] = pending
    for name in ("rec_k", "rec_u", "rec_g"):
        vw[name].fill_(float("nan"))
    _assert_bitwise(
        _read_pipeline(L, dirty, idx, params),
        _read_pipeline(L, clean, idx, params),
        f"header ({cursor}, {pending})",
    )
    assert torch.equal(dirty[n + 1 :], guard_before)
    assert bool((dirty[0] == 7).all())


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), 6.0e4])
def test_records_past_the_cursor_are_never_read(poison):
    """Rows >= cursor + pending hold the previous window or another request.
    normalize, decode and materialize read the rows below cursor (and the
    pending rows that normalize commits) only."""
    L = _layout(3)
    params = _params(6)
    n = 8
    blob, idx = _new_blob(L, n, seed=13)
    gen = torch.Generator().manual_seed(4)
    _advance(
        L, blob, idx, params, steps=WINDOW // 2 + 1, seed=900,
        accepts=lambda step: _accepted(n, gen),
    )  # fmt: skip
    clean, dirty = blob.clone(), blob.clone()
    _poison_dead_rows(L, dirty[1:], poison)
    _assert_bitwise(
        _read_pipeline(L, dirty, idx, params),
        _read_pipeline(L, clean, idx, params),
        f"poison {poison}",
    )


def _precopy_lq(blob, block_table, dst_col, src_col, token_bias, tiles=4):
    """Run the align pre-copy kernel with one LeapQuant state tensor."""
    dev = blob.device
    i64 = dict(dtype=torch.int64, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    _ops().precopy_mamba_align_fused_rdna2(
        dst_col,
        src_col,
        token_bias,
        torch.tensor([block_table.data_ptr()], **i64),
        block_table.stride(0),
        torch.tensor([blob.data_ptr()], **i64),
        torch.tensor([blob.stride(0)], **i64),
        torch.tensor([1], **i32),
        torch.tensor([blob[0].numel()], **i64),
        torch.tensor([-1], **i32),
        torch.tensor([0], **i32),
        torch.tensor([blob.shape[1]], **i32),
        torch.tensor([blob.shape[2] - lq.LEAPQUANT_HEADER_BYTES], **i64),
        None,
        dst_col.shape[0],
        False,
        tiles,
    )


@pytest.mark.parametrize("kind", LEFTOVERS)
def test_align_precopy_overwrites_the_leftovers_of_its_destination(kind):
    """The destination block of a prefix-cache copy is recycled too. The copy
    must equal the oracle's copy of the source slot in every byte, whatever the
    destination held (NaN patterns and a valid-looking header included)."""
    L = _layout(3)
    params = _params(11)
    n = 2
    blob, idx = _new_blob(L, n, seed=15)
    _advance(
        L, blob, idx, params, steps=4, seed=40, accepts=lambda step: _const(n, 3)
    )  # fmt: skip
    blob = torch.cat([blob, _leftover_blob(L, n, kind, seed=16)])  # blocks n+1..2n
    block_table = torch.zeros(n, 4, dtype=torch.int32, device=DEVICE)
    block_table[:, 0] = torch.arange(1, n + 1, dtype=torch.int32)
    block_table[:, 1] = torch.arange(n + 1, 2 * n + 1, dtype=torch.int32)
    src_before = blob[1 : n + 1].clone()
    bias = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    _precopy_lq(
        blob, block_table,
        dst_col=torch.ones(n, dtype=torch.int32, device=DEVICE),
        src_col=torch.zeros(n, dtype=torch.int32, device=DEVICE),
        token_bias=bias,
    )  # fmt: skip
    assert torch.equal(blob[1 : n + 1], src_before), "source home block modified"
    for i in range(n):
        want = lq.leapquant_copy_slot(L, src_before[i].cpu(), int(bias[i]))
        assert torch.equal(blob[n + 1 + i].cpu(), want), (kind, i)


# --------------------------------------------------------------------------
# 4. Flush race.
# --------------------------------------------------------------------------


def _offsets(blobs):
    base = blobs[0].data_ptr()
    return torch.tensor(
        [b.data_ptr() - base for b in blobs], dtype=torch.int64, device=DEVICE
    )


def _running_layers(L, layers, n, *, mixed, params):
    """``layers`` blobs of n running requests each. Without ``mixed`` every
    request has window - 2 committed and 3 pending records (accepting 2 or 3
    reaches the window); with it the cursors differ from request to request."""
    gen = torch.Generator().manual_seed(33)
    blobs, idx = [], None
    for layer in range(layers):
        blob, idx = _new_blob(L, n, seed=50 + layer)
        _advance(
            L, blob, idx, params, steps=L.window // 2, seed=1000 * layer,
            accepts=(
                (lambda step: _accepted(n, gen))
                if mixed
                else (lambda step: _const(n, 2))
            ),
        )  # fmt: skip
        blobs.append(blob)
    return blobs, idx, gen


@pytest.mark.parametrize("load", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("window", [16, 32])
def test_grouped_flush_of_full_windows_equals_per_layer_launches(window, mixed, load):
    """Many (block, head) pairs reach the window in the same launch, with
    MTP-style accepted counts. One grouped normalize over every layer (the
    layer_offsets path) must give, bit for bit, what one launch per layer
    gives, every time."""
    L = _layout(3, window=window)
    params = _params(21)
    n, layers = 32, 6
    blobs, idx, gen = _running_layers(L, layers, n, mixed=mixed, params=params)
    acc = _accepted(n, gen, 1 if mixed else 2, 3)
    # The test must reach the flush path, on many pairs.
    will_fold = 0
    for blob in blobs:
        hdr = L.views(blob[1:])["header"]
        commit = torch.minimum(acc[:, None].expand_as(hdr[..., 1]), hdr[..., 1])
        full = (hdr[..., 0] + commit) >= window
        will_fold += int(full.sum())
        if not mixed:
            assert bool(full.all()), "every pair should reach the window"
    assert will_fold >= layers * n * HV // 8, f"only {will_fold} pairs flush"
    loader = _Load() if load else None
    for rep in range(GROUPED_REPEATS):
        if loader is not None:
            loader.kick()
        want = [b.clone() for b in blobs]
        for b in want:
            _normalize(L, b, idx, acc)
        got = [b.clone() for b in blobs]
        _normalize(L, got[0], idx, acc, layer_offsets=_offsets(got))
        for layer in range(layers):
            _assert_bitwise(got[layer], want[layer], f"repeat {rep} layer {layer}")
    if loader is not None:
        loader.wait()
    for blob in blobs:  # the NULL block is never written
        assert bool((blob[0] == 7).all())


@pytest.mark.parametrize("window", [16, 32])
def test_batched_decode_equals_singleton_launches(window):
    """A decode launch over many requests gives each request exactly what a
    launch over that request alone gives: no request reads or writes another's
    blob. The cursors differ, and some requests flushed just before."""
    L = _layout(3, window=window)
    params = _params(23)
    n = 24
    gen = torch.Generator().manual_seed(8)
    blob, idx = _new_blob(L, n, seed=61)
    acc = _advance(
        L, blob, idx, params, steps=window // 2 + 2, seed=700,
        accepts=lambda step: _accepted(n, gen),
    )  # fmt: skip
    _normalize(L, blob, idx, acc)
    tokens = _tokens(n, 3, seed=800)
    batched_blob = blob.clone()
    batched = _decode(L, batched_blob, idx, *tokens, *params)
    single_blob = blob.clone()
    for i in range(n):
        part = [x[i : i + 1] for x in tokens]
        one = _decode(L, single_blob, idx[i : i + 1].contiguous(), *part, *params)
        _assert_bitwise(batched[i : i + 1], one, f"request {i}")
    _assert_bitwise(batched_blob, single_blob, "blobs after the batched decode")
