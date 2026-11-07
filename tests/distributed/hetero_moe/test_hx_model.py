# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the hx reference model and protocol simulator (hx_model.py).

CPU only (numpy): the modules are loaded by path, so the tests run without torch (run
them with ``--noconftest`` where the repo conftests cannot be imported).

* ``dispatch_ref``: hand-computed cases, three independent statements of the rule
  (loop, bitmap and the oracle below) on thousands of random cases, plan errors.
* ``combine_ref`` / ``toy_moe``: fixed summation order, NaN abort output, and the
  partial-sum identity (the pairs of a step partition exactly).
* ``HxSim``: every protocol claim of the design on the simulator: exact outputs, slot
  parity, fences, wrap safety, timeout tick counts, header mismatches, faults.
"""

import importlib
import importlib.util
import itertools
import sys
import types
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG = "vllm.distributed.hetero_moe"


def _load(*names):
    """Import hetero_moe modules by file path under their dotted names. They need
    the standard library (and numpy) only, but ``import vllm`` needs torch: where
    it fails, empty stand-ins for the parent packages are registered."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import vllm  # noqa: F401
    except Exception:
        for key in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[key]
        parent = None
        for dotted in ("vllm", "vllm.distributed", PKG):
            stub = types.ModuleType(dotted)
            stub.__path__ = []
            sys.modules[dotted] = stub
            if parent is not None:
                setattr(parent, dotted.rpartition(".")[2], stub)
            parent = stub
    else:
        importlib.import_module(PKG)
    loaded = []
    for name in names:
        dotted = f"{PKG}.{name}"
        path = ROOT / "vllm" / "distributed" / "hetero_moe" / f"{name}.py"
        mod = sys.modules.get(dotted)
        if mod is None or getattr(mod, "__file__", None) != str(path):
            spec = importlib.util.spec_from_file_location(dotted, path)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mod
            spec.loader.exec_module(mod)
        setattr(sys.modules[PKG], name, mod)
        loaded.append(mod)
    return loaded[0] if len(loaded) == 1 else loaded


hx, M = _load("hx_proto", "hx_model")

NSRV, E, H, K, INTER, ROWS_CAP = 4, 16, 16, 4, 8, 8
TPM = 1000  # ticks per ms of the simulator
WAIT = hx.HX_DEFAULT_WAIT_MS * TPM  # ticks of one in-step wait budget
PLAN = "00000000000000a1"  # plan id of the simulator


def half(a):
    return np.asarray(a).astype(np.float16)


def bits(a):
    return np.ascontiguousarray(a).view(np.uint16)


def same(a, b):
    """Bit-for-bit equality of two fp16 arrays (NaN patterns included)."""
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def is_nan(a):
    return bool((bits(a) == hx.HX_NAN_F16).all())


def u64(mem, off):
    return int.from_bytes(mem[off : off + 8], "little")


def check_record(word, phase, peer, ms, seq=None, *, live=False):
    rec = hx.decode_record(word)
    assert rec.valid and rec.live == live, rec
    assert (rec.phase, rec.peer, rec.ms) == (phase, peer, ms), rec
    if seq is not None:
        assert rec.seq_lo == seq & hx.U32_MAX, (rec, seq)
    return rec


# ===========================================================================
# dispatch_ref
# ===========================================================================
HAND_SLOT = np.array([3, 0, 1, -1, 2, -1, -1, -1])
HAND_NCOLD = np.array([1, 1, 2, 2, 1, 2, 1, 0])
HAND_COLD = np.array(
    [[0, -1], [1, -1], [0, 1], [1, 0], [0, -1], [0, 1], [1, -1], [-1, -1]]
)
HAND_IDS = np.array([[0, 4, 3, -1], [2, 3, 5, 9], [1, 1, 6, 4]])
N1 = [-1, -1, -1, -1]
# (kmax, owned, hub, srv0, srv1) of HAND_IDS at layer 1 on 2 servers. Resident experts
# are 0, 1, 2 and 4 with slots 3, 0, 1, 2; the pair (row r, expert e) of a spilled or
# cold expert goes to cold_srv[e][(r + 1) % n_cold[e]].
HAND_CASES = [
    (
        2,
        None,
        [N1, [2, -1, -1, -1], [1, 1, -1, -1]],
        [[0, 4, 3, -1], [-1, -1, 5, -1], [-1, -1, -1, 4]],
        [N1, [-1, 3, -1, -1], [-1, -1, 6, -1]],
    ),
    (
        0,
        None,
        [N1, N1, N1],
        [[0, 4, 3, -1], [2, -1, 5, -1], [-1, -1, -1, 4]],
        [N1, [-1, 3, -1, -1], [1, 1, 6, -1]],
    ),
    (
        3,
        None,
        [[-1, 4, -1, -1], [2, -1, -1, -1], [1, 1, -1, 4]],
        [[0, -1, 3, -1], [-1, -1, 5, -1], N1],
        [N1, [-1, 3, -1, -1], [-1, -1, 6, -1]],
    ),
    (
        100,
        None,
        [[0, 4, -1, -1], [2, -1, -1, -1], [1, 1, -1, 4]],
        [[-1, -1, 3, -1], [-1, -1, 5, -1], N1],
        [N1, [-1, 3, -1, -1], [-1, -1, 6, -1]],
    ),
    (  # expert 2 is not owned: its pair is dropped and its slot is not touched
        2,
        [1, 1, 0, 1, 1, 1, 1, 1],
        [[-1, 4, -1, -1], N1, [1, 1, -1, 4]],
        [[0, -1, 3, -1], [-1, -1, 5, -1], N1],
        [N1, [-1, 3, -1, -1], [-1, -1, 6, -1]],
    ),
]


@pytest.mark.parametrize("form", ["loop", "bitmap"])
@pytest.mark.parametrize(("kmax", "owned", "hub", "srv0", "srv1"), HAND_CASES)
def test_dispatch_hand_computed(form, kmax, owned, hub, srv0, srv1):
    got_hub, got_srv, errors = M.dispatch_ref(
        None, HAND_IDS, None, 1, HAND_SLOT, HAND_NCOLD, HAND_COLD, kmax, 2, owned,
        form=form,
    )  # fmt: skip
    assert got_hub.dtype == np.int32 and got_srv.dtype == np.int16
    assert got_hub.tolist() == hub
    assert got_srv.tolist() == [srv0, srv1]
    assert errors == 0


def rule_oracle(ids, layer, slot_map, n_cold, cold_srv, kmax, nsrv, owned):
    """A third statement of the rule: the kept slots are the ``kmax`` lowest touched
    slots (sorted-set slice); no rank counting, no popcount."""
    rows, topk = ids.shape
    n_exp = len(slot_map)
    pairs = [(r, c, int(ids[r, c])) for r in range(rows) for c in range(topk)]
    valid = [
        (r, c, e)
        for r, c, e in pairs
        if 0 <= e < n_exp and (owned is None or owned[e] == 1)
    ]
    kept = sorted({int(slot_map[e]) for _, _, e in valid if slot_map[e] >= 0})[:kmax]
    hub = np.full((rows, topk), -1, np.int32)
    srv = np.full((nsrv, rows, topk), -1, np.int16)
    errors = 0
    for r, c, e in valid:
        if slot_map[e] in kept:
            hub[r, c] = e
            continue
        n = int(n_cold[e])
        s = int(cold_srv[e][(r + layer) % n]) if n > 0 else -1
        if 0 <= s < nsrv:
            srv[s, r, c] = e
        else:
            errors += 1
    return hub, srv, errors


def random_case(rng, rows, topk, *, plan_errors=False):
    n_exp = int(rng.choice([8, 16, 64, 256, 512]))
    nsrv = int(rng.choice([1, 2, 4, 8]))
    maxc = int(rng.integers(1, 9))
    n_res = int(rng.integers(0, min(n_exp, 40) + 1))
    slot_map = np.full(n_exp, -1, np.int64)
    resident = rng.choice(n_exp, n_res, replace=False)
    slot_map[resident] = rng.choice(512, n_res, replace=False)
    n_cold = rng.integers(0 if plan_errors else 1, maxc + 1, n_exp)
    cold_srv = rng.integers(
        -1 if plan_errors else 0, nsrv + (2 if plan_errors else 0), (n_exp, maxc)
    )
    ids = rng.integers(-2, n_exp + 3, (rows, topk))  # includes -1 and ids >= E
    if n_res:
        pick = rng.random((rows, topk)) < 0.5
        ids[pick] = rng.choice(resident, int(pick.sum()))
    owned = rng.integers(0, 2, n_exp) if rng.random() < 0.4 else None
    return {
        "x": None,
        "ids": ids,
        "w": None,
        "layer": int(rng.integers(0, 64)),
        "slot_map": slot_map,
        "n_cold": n_cold,
        "cold_srv": cold_srv,
        "kmax": int(rng.choice([0, 1, 7, 63, 512])),
        "nsrv": nsrv,
        "owned": owned,
    }


def oracle_of(case):
    return rule_oracle(
        case["ids"], case["layer"], case["slot_map"], case["n_cold"],
        case["cold_srv"], case["kmax"], case["nsrv"], case["owned"],
    )  # fmt: skip


def assert_three_ways(case):
    want = oracle_of(case)
    for form in ("loop", "bitmap"):
        hub, srv, errors = M.dispatch_ref(**case, form=form)
        assert np.array_equal(hub, want[0]), form
        assert np.array_equal(srv, want[1]), form
        assert errors == want[2], form
        assert hub.dtype == np.int32 and srv.dtype == np.int16, form
    return want


def test_dispatch_forms_and_oracle_agree_on_2000_random_cases():
    rng = np.random.default_rng(2024)
    seen_rows, kept, spilled, invalid = set(), 0, 0, 0
    for i in range(2000):
        rows = 2048 if i % 160 == 0 else (1, 3, 24, 48)[i % 4]
        topk = 4 if rows == 2048 else int(rng.choice([1, 4, 10, 16]))
        case = random_case(rng, rows, topk)
        hub, srv, errors = assert_three_ways(case)
        assert errors == 0  # n_cold >= 1 and every copy names a server
        ids = case["ids"]
        seen_rows.add(rows)
        kept += int((hub >= 0).sum())
        spilled += int((srv >= 0).sum())
        invalid += int(((ids < 0) | (ids >= len(case["slot_map"]))).sum())
    assert seen_rows == {1, 3, 24, 48, 2048}
    assert kept > 5000 and spilled > 5000 and invalid > 5000  # every branch is hit


def test_dispatch_plan_error_batch_loop_equals_bitmap_equals_oracle():
    rng = np.random.default_rng(77)
    total_errors = 0
    for i in range(500):
        rows = (1, 3, 24, 48)[i % 4]
        case = random_case(rng, rows, int(rng.choice([1, 4, 10, 16])), plan_errors=True)
        total_errors += assert_three_ways(case)[2]
    assert total_errors > 500


def test_dispatch_partition_invariants():
    """Every valid owned pair has exactly one home; spilled pairs go to their copy."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        rows = int(rng.choice([1, 3, 24]))
        case = random_case(rng, rows, int(rng.choice([4, 10, 16])))
        hub, srv, errors = M.dispatch_ref(**case, form="bitmap")
        ids, owned, slot_map = case["ids"], case["owned"], case["slot_map"]
        homes = (hub >= 0).astype(int) + (srv >= 0).sum(axis=0)
        in_range = (ids >= 0) & (ids < len(slot_map))
        valid = in_range & (
            True if owned is None else owned[np.where(in_range, ids, 0)] == 1
        )
        assert errors == 0
        assert np.array_equal(homes, valid.astype(int))
        assert np.array_equal(hub[hub >= 0], ids[hub >= 0])
        for s in range(case["nsrv"]):
            assert np.array_equal(srv[s][srv[s] >= 0], ids[srv[s] >= 0])
        for r, c in zip(*np.nonzero(srv >= 0)[1:]):
            s = int(np.nonzero(srv[:, r, c] >= 0)[0][0])
            e = int(ids[r, c])
            n = int(case["n_cold"][e])
            assert s == case["cold_srv"][e][(r + case["layer"]) % n]
        kept_slots = {int(slot_map[e]) for e in hub[hub >= 0]}
        assert len(kept_slots) <= case["kmax"]
        spilled_res = {int(slot_map[e]) for e in srv[srv >= 0] if slot_map[e] >= 0}
        assert all(a < b for a in kept_slots for b in spilled_res)  # lowest slots stay


@pytest.mark.parametrize("form", ["loop", "bitmap"])
def test_dispatch_plan_errors_mask_the_pair_and_count_it(form):
    cold = HAND_COLD.copy()
    cold[6] = [5, -1]  # server 5 does not exist on a 2-server island
    cold[3] = [-1, 0]  # the first copy of expert 3 is unset
    ids = np.array([[7, 1, 6, 3, 0], [3, 3, 3, 3, 3]])  # e7 has n_cold == 0
    hub, srv, errors = M.dispatch_ref(
        None, ids, None, 0, HAND_SLOT, HAND_NCOLD, cold, 100, 2, form=form
    )
    assert errors == 3  # row 0: e7 (no copy), e6 (server 5), e3 (copy -1)
    assert hub.tolist() == [[-1, 1, -1, -1, 0], [-1] * 5]
    assert srv[0].tolist() == [[-1] * 5, [3] * 5]  # row 1: (1 + 0) % 2 -> copy 1
    assert (srv[1] == -1).all()


def test_dispatch_non_owned_pairs_are_dropped_never_errors():
    ncold = HAND_NCOLD.copy()
    ncold[7] = 0
    owned = np.ones(8, np.int64)
    owned[7] = 0
    ids = np.array([[7, 7, 1, 9]])
    for form in ("loop", "bitmap"):
        hub, srv, errors = M.dispatch_ref(
            None, ids, None, 0, HAND_SLOT, ncold, HAND_COLD, 100, 2, owned, form=form
        )
        assert errors == 0 and hub.tolist() == [[-1, -1, 1, -1]] and (srv == -1).all()
        # the same pairs with an owner are a plan error
        _, _, errors = M.dispatch_ref(
            None, ids, None, 0, HAND_SLOT, ncold, HAND_COLD, 100, 2, None, form=form
        )
        assert errors == 2


def test_dispatch_accepts_missing_x_w_owned_and_any_int_dtype():
    want = M.dispatch_ref(
        None, HAND_IDS, None, 1, HAND_SLOT, HAND_NCOLD, HAND_COLD, 2, 2
    )
    x = np.zeros((3, 16), np.float16)
    w = np.zeros((3, 4), np.float32)
    for dtype in (np.int16, np.int32, np.int64):
        for form in ("loop", "bitmap"):
            got = M.dispatch_ref(
                x, HAND_IDS.astype(dtype), w, 1, HAND_SLOT.astype(np.int32),
                HAND_NCOLD.astype(np.int32), HAND_COLD.astype(np.int32), 2, 2,
                np.ones(8, np.uint8), form=form,
            )  # fmt: skip
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
            assert got[2] == 0


def test_dispatch_stacked_tables_select_the_layer():
    other_slot = np.roll(HAND_SLOT, 3)
    slot = np.stack([HAND_SLOT, other_slot])
    ncold = np.stack([HAND_NCOLD, HAND_NCOLD])
    cold = np.stack([HAND_COLD, HAND_COLD])
    owned = np.ones((2, 8), np.uint8)
    for form in ("loop", "bitmap"):
        got = M.dispatch_ref(
            None, HAND_IDS, None, 1, slot, ncold, cold, 2, 2, owned, form=form
        )
        want = M.dispatch_ref(
            None, HAND_IDS, None, 1, other_slot, HAND_NCOLD, HAND_COLD, 2, 2,
            form=form,
        )  # fmt: skip
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
        first = M.dispatch_ref(
            None, HAND_IDS, None, 0, slot, ncold, cold, 2, 2, owned, form=form
        )
        assert not np.array_equal(first[0], got[0])  # layer 0 uses slot_map[0]
        with pytest.raises(ValueError, match="layers"):
            M.dispatch_ref(
                None, HAND_IDS, None, 2, slot, ncold, cold, 2, 2, owned, form=form
            )


BASE_CALL = {
    "x": None, "ids": HAND_IDS, "w": None, "layer": 1, "slot_map": HAND_SLOT,
    "n_cold": HAND_NCOLD, "cold_srv": HAND_COLD, "kmax": 2, "nsrv": 2, "owned": None,
}  # fmt: skip


@pytest.mark.parametrize(
    "override",
    [
        {"ids": HAND_IDS.astype(np.float32)},
        {"ids": HAND_IDS[0]},
        {"ids": np.zeros((2, 17), np.int32)},
        {"ids": np.zeros((2, 0), np.int32)},
        {"layer": 64},
        {"layer": -1},
        {"nsrv": 0},
        {"nsrv": 9},
        {"kmax": -1},
        {"slot_map": np.array([0, 0, 1, -1, 2, -1, -1, -1])},  # shared slot
        {"slot_map": np.array([512, 0, 1, -1, 2, -1, -1, -1])},
        {"slot_map": np.array([-2, 0, 1, -1, 2, -1, -1, -1])},
        {"slot_map": np.zeros(513, np.int64) - 1, "n_cold": np.ones(513, np.int64)},
        {"n_cold": np.array([3, 1, 2, 2, 1, 2, 1, 0])},  # wider than cold_srv
        {"n_cold": np.ones(7, np.int64)},
        {"cold_srv": np.zeros((8, 9), np.int64)},
        {"cold_srv": np.zeros((8, 0), np.int64)},
        {"owned": np.array([2, 1, 1, 1, 1, 1, 1, 1])},
        {"owned": np.ones(7, np.int64)},
        {"x": np.zeros((5, 4), np.float16)},
        {"w": np.zeros((3, 3), np.float32)},
        {"form": "fast"},
    ],
)
def test_dispatch_rejects_malformed_inputs(override):
    with pytest.raises(ValueError):
        M.dispatch_ref(**{**BASE_CALL, **override})


# ===========================================================================
# combine_ref and toy_moe
# ===========================================================================
def crafted_partials(rows=3):
    """Partials whose sum depends on the order: +big and -big cancel, and the first
    tiny one is below the fp32 resolution of ``big`` (2**-9), so it survives only if
    it is added after the cancellation."""
    rng = np.random.default_rng(0)
    big = (np.float32(16384) + 256 * np.arange(rows * H, dtype=np.float32)).reshape(
        rows, H
    )
    big = half(big)

    def tiny():
        return half(2.0**-12 * (1 + 0.5 * rng.random((rows, H))))

    return big, [tiny(), -big, tiny(), tiny()]  # hub, servers 0..3


def sum_in_order(hub, parts, order):
    acc = hub.astype(np.float32)
    for s in order:
        acc = acc + parts[s].astype(np.float32)
    return acc.astype(np.float16)


def test_combine_ref_sums_hub_first_then_servers_left_to_right_in_fp32():
    hub, parts = crafted_partials()
    got = M.combine_ref(hub, parts)
    scalar = np.empty(hub.shape, np.float16)
    for i, j in itertools.product(range(hub.shape[0]), range(hub.shape[1])):
        acc = np.float32(hub[i, j])
        for p in parts:  # one fp32 add at a time, as the kernel does
            acc = np.float32(acc + np.float32(p[i, j]))
        scalar[i, j] = np.float16(acc)
    assert got.dtype == np.float16
    assert same(got, scalar)
    # the order matters here: tiny[0] is lost when added to big, kept after -big
    assert not same(got, sum_in_order(hub, parts, [1, 0, 2, 3]))
    assert not same(got, sum_in_order(hub, parts, [3, 2, 1, 0]))
    expected_tail = half(parts[2].astype(np.float32) + parts[3].astype(np.float32))
    assert same(got, expected_tail)


def test_combine_ref_edge_cases():
    hub = half(np.arange(6).reshape(2, 3))
    assert same(M.combine_ref(hub, []), hub)  # no servers: the hub partial
    big = half(np.full((2, 3), 60000.0))
    with np.errstate(all="raise"):  # an fp16 overflow is inf, not a warning
        assert np.isinf(M.combine_ref(big, [big])).all()


def test_combine_ref_aborted_is_nan_whatever_the_partials():
    hub, parts = crafted_partials()
    out = M.combine_ref(hub, parts, aborted=True)
    assert out.dtype == np.float16 and out.shape == hub.shape
    assert (bits(out) == 0x7E00).all() and np.isnan(out).all()
    assert (bits(M.combine_ref(hub, [], aborted=True)) == 0x7E00).all()


def test_combine_ref_rejects_wrong_dtypes_and_shapes():
    hub = half(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        M.combine_ref(hub.astype(np.float32), [])
    with pytest.raises(ValueError):
        M.combine_ref(hub, [hub.astype(np.float32)])
    with pytest.raises(ValueError):
        M.combine_ref(hub, [half(np.zeros((2, 4)))])


def moe_weights(rng, n_exp=E):
    W13 = (rng.standard_normal((n_exp, 2 * INTER, H)) * 0.3).astype(np.float32)
    W2 = (rng.standard_normal((n_exp, H, INTER)) * 0.3).astype(np.float32)
    return W13, W2


def test_toy_moe_matches_a_float64_scalar_reference():
    rng = np.random.default_rng(1)
    W13, W2 = moe_weights(rng)
    x = rng.standard_normal((5, H)).astype(np.float32)
    ids = rng.integers(-2, E + 2, (5, K))
    w = rng.random((5, K)).astype(np.float32)
    ref = np.zeros((5, H))
    for r, k in itertools.product(range(5), range(K)):
        e = ids[r, k]
        if not 0 <= e < E:
            continue
        h = W13[e].astype(np.float64) @ x[r].astype(np.float64)
        gate, up = h[:INTER], h[INTER:]
        act = gate / (1.0 + np.exp(-gate)) * up  # silu(gate) * up
        ref[r] += w[r, k] * (W2[e].astype(np.float64) @ act)
    got = M.toy_moe(x, ids, w, W13, W2)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_toy_moe_masked_pairs_contribute_exactly_zero():
    rng = np.random.default_rng(2)
    W13, W2 = moe_weights(rng)
    x = rng.standard_normal((3, H)).astype(np.float32)
    w = np.full((3, K), np.nan, np.float32)  # even a NaN weight is masked out
    for bad in (-1, E, E + 5, -7):
        out = M.toy_moe(x, np.full((3, K), bad), w, W13, W2)
        assert out.shape == (3, H) and (out == 0).all()
    ids = np.array([[3, -1, 3, E]] * 3)
    w = rng.random((3, K)).astype(np.float32)
    once = M.toy_moe(x, np.array([[3, -1, -1, -1]] * 3), w, W13, W2)
    twice = M.toy_moe(x, ids, w, W13, W2)
    assert not np.array_equal(once, twice)  # the second copy of expert 3 counts
    w2 = w.copy()
    w2[:, 1] = 1e30
    w2[:, 3] = 1e30
    assert np.array_equal(M.toy_moe(x, ids, w2, W13, W2), twice)


def test_toy_moe_is_linear_in_the_weights():
    rng = np.random.default_rng(3)
    W13, W2 = moe_weights(rng)
    x = rng.standard_normal((4, H)).astype(np.float32)
    ids = np.stack([rng.permutation(E)[:K] for _ in range(4)])
    w = rng.random((4, K)).astype(np.float32)
    np.testing.assert_allclose(
        M.toy_moe(x, ids, 2.0 * w, W13, W2), 2.0 * M.toy_moe(x, ids, w, W13, W2),
        rtol=1e-5, atol=1e-6,
    )  # fmt: skip


def test_partial_sum_identity_pairs_partition_exactly():
    """The hub and server partials sum to the dense MoE of the valid owned pairs."""
    rng = np.random.default_rng(9)
    W13, W2 = moe_weights(rng)
    for _ in range(60):
        rows = int(rng.integers(1, 9))
        slot_map = np.full(E, -1, np.int64)
        resident = rng.choice(E, 6, replace=False)
        slot_map[resident] = rng.permutation(6)
        n_cold = rng.integers(1, 4, E)
        cold = rng.integers(0, NSRV, (E, 3))
        ids = rng.integers(-2, E + 3, (rows, K))
        owned = rng.integers(0, 2, E) if rng.random() < 0.5 else None
        x = rng.standard_normal((rows, H)).astype(np.float16)
        w = rng.random((rows, K)).astype(np.float32)
        hub, srv, errors = M.dispatch_ref(
            x, ids, w, int(rng.integers(0, 64)), slot_map, n_cold, cold,
            int(rng.choice([0, 1, 3, 100])), NSRV, owned,
        )  # fmt: skip
        assert errors == 0
        ok = (ids >= 0) & (ids < E)
        if owned is not None:
            ok &= owned[np.where(ok, ids, 0)] == 1
        full = M.toy_moe(x, np.where(ok, ids, -1), w, W13, W2)
        parts = [M.toy_moe(x, hub, w, W13, W2)]
        parts += [M.toy_moe(x, srv[s], w, W13, W2) for s in range(NSRV)]
        np.testing.assert_allclose(sum(parts), full, rtol=1e-4, atol=1e-5)
        union = np.maximum(hub, srv.max(axis=0))  # disjoint: at most one is >= 0
        assert np.array_equal(union, np.where(ok, ids, -1))


# ===========================================================================
# HxSim: environment, tasks and oracles
# ===========================================================================
class Env(NamedTuple):
    layers: int
    steps: int
    rows: int
    bucket: int
    kmax: int
    tables: object
    W13: np.ndarray
    W2: np.ndarray
    data: list  # data[step][layer] = (x fp16, ids int32, w fp32)


def make_tables(rng, layers, n_res=8, maxc=3):
    slot_map = np.full((layers, E), -1, np.int32)
    n_cold = rng.integers(1, maxc + 1, (layers, E)).astype(np.int32)
    cold = np.full((layers, E, maxc), -1, np.int32)
    for lay in range(layers):
        slot_map[lay, rng.permutation(E)[:n_res]] = rng.permutation(n_res)
        for e in range(E):
            cold[lay, e, : n_cold[lay, e]] = rng.permutation(NSRV)[: n_cold[lay, e]]
    return M.Tables(slot_map, n_cold, cold)


def make_env(layers=6, steps=1, rows=3, kmax=2, seed=0, bucket=None):
    rng = np.random.default_rng(seed)
    tables = make_tables(rng, layers)
    W13 = (rng.standard_normal((layers, E, 2 * INTER, H)) * 0.3).astype(np.float32)
    W2 = (rng.standard_normal((layers, E, H, INTER)) * 0.3).astype(np.float32)
    data = []
    for _ in range(steps):
        per_step = []
        for _ in range(layers):
            x = rng.standard_normal((rows, H)).astype(np.float16)
            ids = np.stack([rng.permutation(E)[:K] for _ in range(rows)]).astype(
                np.int32
            )
            per_step.append((x, ids, rng.random((rows, K)).astype(np.float32)))
        data.append(per_step)
    return Env(
        layers, steps, rows, rows if bucket is None else bucket, kmax, tables, W13, W2,
        data,
    )  # fmt: skip


def make_sim(env, **kw):
    kw.setdefault("kmax", env.kmax)
    return M.HxSim(env.tables, nsrv=NSRV, rows_cap=ROWS_CAP, topk=K, hidden=H, **kw)


def partial_sum(env, layer, x, ids, w):
    """The fp16 partial of one process (what it would push or keep)."""
    return half(M.toy_moe(x, ids, w, env.W13[layer], env.W2[layer]))


def expected(env, step, layer):
    """Bit-exact output of a healthy layer: dispatch_ref + toy_moe + combine_ref."""
    x, ids, w = env.data[step][layer]
    t = env.tables
    hub_ids, srv_ids, _ = M.dispatch_ref(
        x, ids, w, layer, t.slot_map, t.n_cold, t.cold_srv, env.kmax, NSRV, t.owned
    )
    parts = [partial_sum(env, layer, x, srv_ids[s], w) for s in range(NSRV)]
    return M.combine_ref(partial_sum(env, layer, x, hub_ids, w), parts)


def full_moe(env, step, layer):
    """The dense MoE over all valid owned pairs (the island's mathematical output)."""
    x, ids, w = env.data[step][layer]
    owned = env.tables.owned
    if owned is not None:
        ids = np.where(owned[layer][ids] == 1, ids, -1)
    return M.toy_moe(x, ids, w, env.W13[layer], env.W2[layer])


def assert_exact(env, hub_out):
    assert hub_out
    for (step, layer), (_, _, out) in hub_out.items():
        assert same(out, expected(env, step, layer)), (step, layer)


def finish(sim, out, step, item, overlap, marks):
    layer, y_hub = item
    if overlap:
        yield from sim.delay(overlap)
    t0 = sim.now
    if marks is not None:
        marks.append(t0)
    result = yield from sim.combine(layer, y_hub)
    out[(step, layer)] = (t0, sim.now, result)


def hub_task(
    sim, env, out, *, depth=1, overlap=0, stop=None, empty=(), marks=None, hook=None
):
    """The hub program: ``depth`` requests in flight (1: in order), ``overlap`` ticks
    of other work before each combine, ``stop=(step, layer)``: dies before dispatching
    that layer, ``empty``: steps without layers, ``hook(step)`` at each step start."""
    for step in range(env.steps):
        if hook is not None:
            hook(step)
        yield from sim.begin_step(env.rows, env.bucket)
        if step in empty:
            continue
        pending = []
        for layer in range(env.layers):
            if stop == (step, layer):
                return
            x, ids, w = env.data[step][layer]
            hub_ids = yield from sim.dispatch(layer, x, ids, w)
            pending.append((layer, partial_sum(env, layer, x, hub_ids, w)))
            while len(pending) >= depth:
                yield from finish(sim, out, step, pending.pop(0), overlap, marks)
        while pending:
            yield from finish(sim, out, step, pending.pop(0), overlap, marks)


def srv_task(
    sim, env, s, out, *, compute=10, stop=None, skip=(), epoch0=0, desc=None,
    marks=None, begin_delay=0,
):  # fmt: skip
    """Server ``s``: ``compute`` ticks (int or ``f(step, layer)``) between the request
    and the push, ``stop=(step, layer)``: dies after receiving that request, ``skip``:
    steps the server never begins, ``desc``: a step descriptor (or ``f(step)``),
    ``begin_delay``: idle ticks before each begin."""
    for step in range(env.steps):
        if step in skip:
            continue
        yield from sim.delay(begin_delay)
        t0 = sim.now
        if desc is None:
            step_desc = M.StepDesc(epoch0 + step + 1, env.rows, env.bucket)
        else:
            step_desc = desc(step) if callable(desc) else desc
        yield from sim.srv_begin(s, step_desc)
        out[(s, step, "begin")] = (t0, sim.now)
        for layer in range(env.layers):
            t0 = sim.now
            if marks is not None:
                marks.append(t0)
            ids = yield from sim.srv_wait(s, layer, env.rows)
            out[(s, step, layer)] = (t0, sim.now)
            if stop == (step, layer):
                return
            yield from sim.delay(compute(step, layer) if callable(compute) else compute)
            x, w = sim.srv_read(s, layer, env.rows)
            y = partial_sum(env, layer, x, ids, w)
            yield from sim.srv_push(s, layer, env.rows, y)


def run_island(sim, env, *, hub=None, srv=None, all_srv=None, extra=None):
    """Run the hub and the four servers; per-server options override ``all_srv``.
    Returns the hub log ``{(step, layer): (t0, t1, out)}`` and the server log
    ``{(s, step, layer | "begin"): (t0, t1)}`` (the wait of that layer / the begin)."""
    hub_out: dict = {}
    srv_out: dict = {}
    tasks = {"hub": hub_task(sim, env, hub_out, **(hub or {}))}
    for s in range(NSRV):
        opts = {**(all_srv or {}), **(srv or {}).get(s, {})}
        tasks[f"srv{s}"] = srv_task(sim, env, s, srv_out, **opts)
    tasks.update(extra or {})
    sim.run(tasks)
    return hub_out, srv_out


# ===========================================================================
# HxSim: healthy operation
# ===========================================================================
@pytest.mark.parametrize("order", ["adversarial", "fifo"])
@pytest.mark.parametrize("nslots", [1, 2, 4])
def test_in_order_run_is_exact_and_equals_the_dense_moe(nslots, order):
    env = make_env(layers=6, steps=2)
    sim = make_sim(env, nslots=nslots, order=order)
    hub_out, _ = run_island(sim, env)
    assert len(hub_out) == 12
    assert_exact(env, hub_out)
    for (step, layer), (_, _, out) in hub_out.items():
        dense = full_moe(env, step, layer)
        np.testing.assert_allclose(out.astype(np.float32), dense, rtol=2e-2, atol=2e-2)
    assert not any(c.aborted for c in sim.ctxs)
    assert not any(c.record or c.live for c in sim.ctxs)


def test_status_words_count_steps_layers_and_pairs():
    env = make_env(layers=6, steps=2)
    sim = make_sim(env)
    run_island(sim, env)
    st = sim.hub.st
    assert st[hx.HX_ST_EPOCH] == 2 and st[hx.HX_ST_STEPS] == 2
    assert st[hx.HX_ST_DISPATCHES] == 12 and st[hx.HX_ST_COMBINES] == 12
    assert st[hx.HX_ST_PAIRS_HUB] + st[hx.HX_ST_PAIRS_REMOTE] == 2 * 6 * 3 * K
    assert st[hx.HX_ST_PAIRS_HUB] > 0 and st[hx.HX_ST_PAIRS_REMOTE] > 0
    assert st[hx.HX_ST_RECORD] == 0 and st[hx.HX_ST_LIVE] == 0
    for c in sim.srv:
        assert c.st[hx.HX_ST_EPOCH] == c.last_epoch == 2
        assert c.st[hx.HX_ST_STEPS] == 2 and (c.cur_T, c.cur_bucket) == (3, 3)


@pytest.mark.parametrize("nslots", [2, 4])
def test_buffers_hold_the_last_request_and_reply_of_each_slot(nslots):
    """Slot parity: layer l lives in slot l % nslots, flags included."""
    env = make_env(layers=6, steps=2)
    sim = make_sim(env, nslots=nslots)
    run_island(sim, env)
    sim.settle()
    hdr = hx.unpack_hdr(sim.inbox[0])
    assert hdr == hx.Hdr(hx.HX_MAGIC, 2, 3, 3, 1, 1, 0, hx.plan_hash(PLAN), 0, 2)
    t, ibx, rpl = env.tables, sim.ibx, sim.rpl
    for j in range(nslots):
        last = max(layer for layer in range(6) if layer % nslots == j)
        x, ids, w = env.data[1][last]
        _, srv_ids, _ = M.dispatch_ref(
            x, ids, w, last, t.slot_map, t.n_cold, t.cold_srv, env.kmax, NSRV
        )
        for s in range(NSRV):
            seq = hx.seq_of(2, last)
            assert u64(sim.inbox[s], ibx.req_flag_off(j)) == seq
            assert u64(sim.reply, rpl.flag_off(s, j)) == seq
            mem, base = sim.inbox[s], ibx.slot_off(j)
            got_x = np.frombuffer(mem, "<f2", 3 * H, base).reshape(3, H)
            got_ids = np.frombuffer(mem, "<i2", 3 * K, base + ibx.ids_off)
            got_w = np.frombuffer(mem, "<f4", 3 * K, base + ibx.w_off)
            assert same(got_x, x)
            assert np.array_equal(got_ids.reshape(3, K), srv_ids[s])
            assert np.array_equal(got_w.reshape(3, K), w)
            y = np.frombuffer(sim.reply, "<f2", 3 * H, rpl.y_off(s, j))
            want = partial_sum(env, last, x, srv_ids[s], w)
            assert same(y.reshape(3, H), want)


@pytest.mark.parametrize(
    ("nslots", "depth", "safe"),
    [
        (1, 1, True),  # in order: a reply implies the request was consumed
        (1, 2, False),
        (1, 3, False),
        (2, 2, True),  # two micro-batches (requests) in flight on two slots
        (2, 3, False),
        (2, 4, False),
        (4, 2, True),
        (4, 4, True),
        (4, 5, False),
    ],
)
def test_pipeline_depth_must_not_exceed_the_slot_count(nslots, depth, safe):
    env = make_env(layers=8, seed=3)
    sim = make_sim(env, nslots=nslots)
    hub_out, _ = run_island(sim, env, hub={"depth": depth}, all_srv={"compute": 100})
    exact = [same(out, expected(env, *key)) for key, (_, _, out) in hub_out.items()]
    assert len(exact) == 8
    assert all(exact) if safe else not all(exact)
    assert not any(c.aborted for c in sim.ctxs)  # an overwrite is silent


@pytest.mark.parametrize("removed", [set(), {"dispatch"}, {"push"}])
def test_fences_are_required_against_an_adversarial_link(removed):
    env = make_env(layers=6, seed=1)
    sim = make_sim(env, fences=M.FENCES_ALL - removed, order="adversarial")
    hub_out, _ = run_island(sim, env, all_srv={"compute": 0})
    exact = [same(out, expected(env, *key)) for key, (_, _, out) in hub_out.items()]
    assert all(exact) if not removed else not any(exact)


@pytest.mark.parametrize("removed", [{"hdr", "dispatch", "push"}, {"dispatch"}])
def test_in_order_link_needs_no_fence(removed):
    """Control run: with program order kept by the link the same code is exact."""
    env = make_env(layers=6, seed=1)
    sim = make_sim(env, fences=M.FENCES_ALL - removed, order="fifo")
    hub_out, _ = run_island(sim, env, all_srv={"compute": 0})
    assert_exact(env, hub_out)


def test_missing_header_fence_is_a_bad_magic_on_the_first_step():
    env = make_env(layers=2)
    sim = make_sim(env, fences=M.FENCES_ALL - {"hdr"})
    run_island(sim, env)
    for c in sim.srv:  # hdr_seq overtook the (still zero) header body
        check_record(c.record, hx.HX_PH_HDR, hx.HX_MM_MAGIC, 0, seq=1)
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 0, 2000, hx.seq_of(1, 0))


def test_missing_header_fence_is_a_skipped_epoch_on_later_steps():
    env = make_env(layers=2, steps=2)
    sim = make_sim(env)

    def drop_hdr_fence(step):
        if step == 1:
            sim.fences = M.FENCES_ALL - {"hdr"}

    hub_out, _ = run_island(sim, env, hub={"hook": drop_hdr_fence})
    for layer in range(2):  # step 0 had its fence
        assert same(hub_out[(0, layer)][2], expected(env, 0, layer))
    for c in sim.srv:  # the stale body still says epoch 1 under hdr_seq == 2
        check_record(c.record, hx.HX_PH_HDR, hx.HX_MM_SKIPPED, 0, seq=2)
    assert all(is_nan(hub_out[(1, layer)][2]) for layer in range(2))


# ===========================================================================
# HxSim: wrap safety
# ===========================================================================
def run_epochs(env, epoch0, **kw):
    sim = make_sim(env, epoch0=epoch0, **kw)
    hub_out, _ = run_island(sim, env, all_srv={"epoch0": epoch0})
    return sim, hub_out


def test_three_steps_across_the_epoch_wrap_equal_the_unwrapped_run():
    """48 layers x 3 steps x 4 servers starting at epoch 2**58 - 2: the sequence
    numbers cross 2**64 between steps 0 and 1."""
    env = make_env(layers=48, steps=3, seed=5)
    _, base = run_epochs(env, 0)
    sim, wrapped = run_epochs(env, 2**58 - 2)
    assert len(base) == len(wrapped) == 3 * 48
    for key, (_, _, out) in base.items():
        assert same(out, wrapped[key][2]), key
        assert same(out, expected(env, *key)), key
    assert sim.hub.epoch == 2**58 + 1 and sim.srv[3].epoch == 2**58 + 1
    assert not any(c.aborted for c in sim.ctxs)
    sim.settle()  # the final flags are small numbers again
    final = hx.seq_of(2**58 + 1, 47)
    assert final == 112 and u64(sim.reply, sim.rpl.flag_off(0, 47 % 2)) == final
    assert hx.seq_of(2**58 - 1, 47) > final  # numerically the flags went down


def test_a_plain_greater_equal_compare_breaks_after_the_wrap():
    env = make_env(layers=48, steps=3, seed=5)
    _, wrapped = run_epochs(env, 2**58 - 2, reached=lambda flag, seq: flag >= seq)
    bad = sorted(
        k for k, (_, _, out) in wrapped.items() if not same(out, expected(env, *k))
    )
    assert bad and bad[0][0] >= 1  # step 0 (below 2**64) is fine, the rest is stale


# ===========================================================================
# HxSim: timeouts and the abort state machine
# ===========================================================================
def test_server_death_times_out_the_hub_then_every_server_and_stays_aborted():
    env = make_env(layers=8, steps=2, seed=4)
    sim = make_sim(env)
    hub_out, srv_out = run_island(
        sim, env, hub={"overlap": 500}, srv={2: {"stop": (0, 3)}}
    )
    for layer in range(3):  # healthy until the death
        assert same(hub_out[(0, layer)][2], expected(env, 0, layer))
    # hub: the wait for server 2 of layer 3 runs the full budget plus one tick
    t0, t1, out = hub_out[(0, 3)]
    assert t1 - t0 == WAIT + 1 and is_nan(out)
    rec = check_record(sim.hub.record, hx.HX_PH_COMBINE, 2, 2000, hx.seq_of(1, 3))
    assert not rec.live and rec.layer == 3 and rec.epoch == 1
    text = hx.describe_record(sim.hub.record, "hub")
    assert "peer server 2" in text and "2000 ms" in text and "layer 3" in text
    # later layers and the whole next step cost nothing and stay NaN
    for key, (a, b, out) in hub_out.items():
        if key > (0, 3):
            assert b == a and is_nan(out), key
    assert len([k for k in hub_out if k[0] == 1]) == 8
    st = sim.hub.st
    assert st[hx.HX_ST_EPOCH] == 1 and st[hx.HX_ST_STEPS] == 1  # begin_step was a no-op
    assert st[hx.HX_ST_DISPATCHES] == 4 and st[hx.HX_ST_COMBINES] == 3
    # survivors: the request of layer 4 never comes; each times out on its own clock
    for s in (0, 1, 3):
        c = sim.srv[s]
        t0, t1 = srv_out[(s, 0, 4)]
        assert t1 - t0 == WAIT + 1
        check_record(c.record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 4))
        assert "peer hub 0" in hx.describe_record(c.record, "server")
        later = [(s, 0, layer) for layer in range(5, 8)] + [(s, 1, "begin")]
        later += [(s, 1, layer) for layer in range(8)]
        for key in later:
            assert srv_out[key][0] == srv_out[key][1], key  # aborted: no waiting
        assert c.st[hx.HX_ST_EPOCH] == 1 and c.st[hx.HX_ST_STEPS] == 1
    assert not sim.srv[2].aborted and sim.srv[2].record == 0  # the dead one is silent
    # sticky: the first claimer wins, nothing overwrites the record
    before = sim.hub.record
    assert sim.claim(sim.hub, hx.HX_PH_HOST, 0, 0, 0) is False
    assert sim.hub.record == before


def test_hub_death_times_out_every_server_at_the_request_wait():
    env = make_env(layers=6, steps=2)
    sim = make_sim(env)
    hub_out, srv_out = run_island(sim, env, hub={"stop": (0, 2)})
    assert sorted(hub_out) == [(0, 0), (0, 1)]
    assert_exact(env, hub_out)
    for s in range(NSRV):
        t0, t1 = srv_out[(s, 0, 2)]
        assert t1 - t0 == WAIT + 1
        check_record(sim.srv[s].record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 2))
        for layer in (3, 4, 5):
            assert srv_out[(s, 0, layer)][0] == srv_out[(s, 0, layer)][1]
        assert srv_out[(s, 1, "begin")][0] == srv_out[(s, 1, "begin")][1]
    assert not sim.hub.aborted


@pytest.mark.parametrize(("begin_wait_ms", "ms"), [(30_000, 30_000), (600_000, 0xFFFF)])
def test_begin_wait_has_its_own_bound(begin_wait_ms, ms):
    env = make_env(layers=2)
    sim = make_sim(env, begin_wait_ms=begin_wait_ms)
    srv_out: dict = {}
    sim.run({"srv0": srv_task(sim, env, 0, srv_out)})  # no hub at all
    t0, t1 = srv_out[(0, 0, "begin")]
    assert t1 - t0 == begin_wait_ms * TPM + 1
    rec = check_record(sim.srv[0].record, hx.HX_PH_SRV_BEGIN, 0, ms, seq=1)
    assert rec.saturated == (ms == 0xFFFF) and rec.layer is None and rec.epoch == 1
    assert sim.srv[0].live  # a long wait refreshed the live word on the way


def test_in_step_wait_is_bounded_by_wait_ms_not_by_begin_wait_ms():
    env = make_env(layers=2)
    sim = make_sim(env, wait_ms=1500, begin_wait_ms=30_000)
    _, srv_out = run_island(sim, env, hub={"stop": (0, 0)})  # begin_step, no dispatch
    for s in range(NSRV):
        t0, t1 = srv_out[(s, 0, 0)]
        assert t1 - t0 == 1500 * TPM + 1
        check_record(sim.srv[s].record, hx.HX_PH_SRV_WAIT, 0, 1500, hx.seq_of(1, 0))


def late_run(compute_s0):
    """Server 0, the first one the hub waits for, computes ``compute_s0`` ticks at
    layer 3 (the hub starts waiting for it as soon as it starts the combine)."""
    env = make_env(layers=4, seed=2)
    sim = make_sim(env)
    slow = {0: {"compute": lambda step, layer: compute_s0 if layer == 3 else 10}}
    hub_out, _ = run_island(sim, env, srv=slow)
    return env, sim, hub_out


def test_late_peer_in_time_gives_identical_bytes_and_the_boundary_is_exact():
    env, _, base = late_run(10)
    assert_exact(env, base)
    _, _, cal = late_run(1000)  # calibrate: ticks until server 0's flag is visible
    arrival = cal[(0, 3)][1] - cal[(0, 3)][0]
    # visible exactly budget + 1 ticks after the wait began: the flag wins the tick
    _, sim, on_time = late_run(1000 + WAIT + 1 - arrival)
    t0, t1, _ = on_time[(0, 3)]
    assert t1 - t0 == WAIT + 1
    assert not any(c.aborted for c in sim.ctxs)
    for key in base:
        assert same(on_time[key][2], base[key][2]), key
    # one tick later the hub has timed out on the same tick count and aborted
    _, sim, late = late_run(1000 + WAIT + 2 - arrival)
    t0, t1, out = late[(0, 3)]
    assert t1 - t0 == WAIT + 1 and is_nan(out)
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 0, 2000, hx.seq_of(1, 3))
    for layer in range(3):
        assert same(late[(0, layer)][2], base[(0, layer)][2])


def test_a_reply_that_arrives_after_the_abort_is_harmless():
    _, sim, hub_out = late_run(WAIT + 50_000)
    assert is_nan(hub_out[(0, 3)][2]) and sim.hub.aborted
    record = sim.hub.record
    assert record == hx.encode_record(hx.HX_PH_COMBINE, 0, hx.seq_of(1, 3), 2000)
    sim.settle()  # the late y and flag land in the hub's buffers
    assert u64(sim.reply, sim.rpl.flag_off(0, 3 % 2)) == hx.seq_of(1, 3)
    assert sim.hub.record == record and sim.hub.st[hx.HX_ST_COMBINES] == 3
    assert not sim.srv[0].aborted  # the late server finished its last layer


def lose_only(src=None, dst=None, kind=None, layer=None):
    """A ``lose`` predicate that drops the stores matching every given field."""
    want = (src, dst, kind, layer)
    return lambda *got: all(w is None or w == g for w, g in zip(want, got))


def test_lost_reply_flag_times_out_the_hub_on_that_server():
    env = make_env(layers=6, seed=6)
    sim = make_sim(env)
    sim.lose = lose_only(src="srv1", kind="reply_flag", layer=3)
    hub_out, _ = run_island(sim, env)
    for layer in range(3):
        assert same(hub_out[(0, layer)][2], expected(env, 0, layer))
    assert is_nan(hub_out[(0, 3)][2])
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 1, 2000, hx.seq_of(1, 3))
    for s in range(NSRV):  # nobody is told: every server waits for layer 4 in vain
        check_record(sim.srv[s].record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 4))


def test_lost_request_flag_times_out_one_server_and_the_hub():
    env = make_env(layers=6, seed=6)
    sim = make_sim(env)
    sim.lose = lose_only(dst="srv2", kind="req_flag", layer=5)
    hub_out, _ = run_island(sim, env)
    check_record(sim.srv[2].record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 5))
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 2, 2000, hx.seq_of(1, 5))
    assert not any(sim.srv[s].aborted for s in (0, 1, 3))  # layer 5 is the last
    for layer in range(5):
        assert same(hub_out[(0, layer)][2], expected(env, 0, layer))


def test_lost_header_publication_times_out_that_server_at_the_begin_bound():
    env = make_env(layers=2, seed=6)
    sim = make_sim(env, begin_wait_ms=3000)
    sim.lose = lose_only(dst="srv3", kind="hdr_seq")
    _, srv_out = run_island(sim, env)
    t0, t1 = srv_out[(3, 0, "begin")]
    assert t1 - t0 == 3000 * TPM + 1
    check_record(sim.srv[3].record, hx.HX_PH_SRV_BEGIN, 0, 3000, seq=1)
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 3, 2000, hx.seq_of(1, 0))
    for s in (0, 1, 2):  # the hub aborted after layer 0: layer 1 never arrives
        check_record(sim.srv[s].record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 1))


# ===========================================================================
# HxSim: header checks
# ===========================================================================
CHECK_ORDER = [  # the order in which a server tests the header fields
    hx.HX_MM_MAGIC,
    hx.HX_MM_OPCODE,
    hx.HX_MM_VERSION,
    hx.HX_MM_PLAN,
    hx.HX_MM_EPOCH,
    hx.HX_MM_T,
    hx.HX_MM_BUCKET,
]


def break_header(sim, env, fields):
    """Make the hub's header (or the host descriptor) wrong in all of ``fields``."""
    desc = {"epoch": 1, "T": env.rows, "bucket": env.bucket}
    for field in fields:
        if field == hx.HX_MM_MAGIC:
            sim.hub.magic = 0xBAD
        elif field == hx.HX_MM_OPCODE:
            sim.hub.opcode = 2
        elif field == hx.HX_MM_VERSION:
            sim.hub.version = 2
        elif field == hx.HX_MM_PLAN:
            sim.hub.plan = hx.plan_hash("00000000000000b2")
        elif field == hx.HX_MM_EPOCH:
            desc["epoch"] = 7
        elif field == hx.HX_MM_T:
            desc["T"] = env.rows - 1
        elif field == hx.HX_MM_BUCKET:
            desc["bucket"] = env.bucket + 5
    return M.StepDesc(**desc)


@pytest.mark.parametrize(
    "fields",
    [[f] for f in CHECK_ORDER]
    + [[a, b] for a, b in itertools.combinations(CHECK_ORDER, 2)]
    + [CHECK_ORDER],
)
def test_header_mismatch_names_the_first_failing_field(fields):
    env = make_env(layers=2)
    sim = make_sim(env)
    desc = break_header(sim, env, fields)
    hub_out, srv_out = run_island(sim, env, all_srv={"desc": desc})
    first = min(fields, key=CHECK_ORDER.index)
    for s in range(NSRV):
        c = sim.srv[s]
        check_record(c.record, hx.HX_PH_HDR, first, 0, seq=1)
        t0, t1 = srv_out[(s, 0, "begin")]
        assert t1 - t0 < 100  # reported as soon as the header is visible
        assert f"field {hx.HX_MISMATCH_NAMES[first]}," in hx.describe_record(
            c.record, "server"
        )
        assert c.st[hx.HX_ST_EPOCH] == 0 and c.st[hx.HX_ST_STEPS] == 0  # not adopted
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 0, 2000, hx.seq_of(1, 0))
    assert all(is_nan(out) for _, _, out in hub_out.values())


@pytest.mark.parametrize(
    ("wrong_epoch", "wrong_plan", "first"),
    [
        (None, False, hx.HX_MM_SKIPPED),
        (9, False, hx.HX_MM_SKIPPED),  # skipped is checked before the epoch
        (None, True, hx.HX_MM_PLAN),  # and after the plan
    ],
)
def test_skipped_epoch_sits_between_the_plan_and_the_epoch_checks(
    wrong_epoch, wrong_plan, first
):
    """The servers never begin step 1: at step 2 they expect epoch 2 and find 3."""
    env = make_env(layers=2, steps=3)
    sim = make_sim(env)

    def desc(step):
        epoch = wrong_epoch if wrong_epoch and step == 2 else step + 1
        return M.StepDesc(epoch, env.rows, env.bucket)

    def hook(step):
        if wrong_plan and step == 2:
            sim.hub.plan = hx.plan_hash("00000000000000b2")

    servers = {"skip": (1,), "desc": desc, "begin_delay": 1000}
    hub_out, _ = run_island(
        sim, env, hub={"empty": (1,), "hook": hook}, all_srv=servers
    )
    for layer in range(2):  # step 0 ran normally
        assert same(hub_out[(0, layer)][2], expected(env, 0, layer))
    for c in sim.srv:
        check_record(c.record, hx.HX_PH_HDR, first, 0, seq=2)
        assert c.st[hx.HX_ST_EPOCH] == 1  # the step-2 header was not adopted


def test_srv_wait_widens_ids_and_masks_what_is_out_of_range():
    env = make_env(layers=2)
    sim = make_sim(env)
    ibx, got = sim.ibx, {}
    raw = np.array(
        [[600, -7, 3, 15], [16, -1, 0, 32767], [-32768, 5, 5, 5]], dtype="<i2"
    )

    def hub():
        yield from sim.begin_step(3, 3)
        sim.write("srv0", ibx.slot_off(0) + ibx.ids_off, raw.tobytes())
        sim.write("srv0", ibx.req_flag_off(0), hx.seq_of(1, 0).to_bytes(8, "little"))

    def server():
        yield from sim.srv_begin(0, M.StepDesc(1, 3, 3))
        got["ids"] = yield from sim.srv_wait(0, 0, 3)

    sim.run({"hub": hub(), "srv0": server()})
    assert got["ids"].dtype == np.int32
    assert got["ids"].tolist() == [[-1, -1, 3, 15], [-1, -1, 0, -1], [-1, 5, 5, 5]]


# ===========================================================================
# HxSim: plan errors, ownership, kmax
# ===========================================================================
def env_with_plan_error():
    """Expert 5 of layer 2 has no hub slot and no cold copy; one pair asks for it."""
    env = make_env(layers=5, seed=9)
    t = env.tables
    slot_map, n_cold = t.slot_map.copy(), t.n_cold.copy()
    slot_map[2, 5], n_cold[2, 5] = -1, 0
    x, ids, w = env.data[0][2]
    ids = np.where(ids == 5, 6, ids)
    ids[1, 2] = 5
    env.data[0][2] = (x, ids, w)
    return env._replace(tables=M.Tables(slot_map, n_cold, t.cold_srv))


def test_plan_error_claims_phase_7_but_still_publishes_the_layer():
    env = env_with_plan_error()
    sim = make_sim(env)
    hub_out, srv_out = run_island(sim, env)
    for layer in range(2):
        assert same(hub_out[(0, layer)][2], expected(env, 0, layer))
    rec = check_record(sim.hub.record, hx.HX_PH_PLAN, 0, 0, hx.seq_of(1, 2))
    assert rec.layer == 2 and "plan error" in hx.describe_record(sim.hub.record, "hub")
    for layer in (2, 3, 4):
        t0, t1, out = hub_out[(0, layer)]
        assert t1 == t0 and is_nan(out)  # aborted at entry: no waiting
    st = sim.hub.st
    assert st[hx.HX_ST_DISPATCHES] == 3 and st[hx.HX_ST_COMBINES] == 2
    sim.settle()
    for s in range(NSRV):  # layer 2 went out: every server answered it ...
        assert u64(sim.reply, sim.rpl.flag_off(s, 0)) == hx.seq_of(1, 2)
        # ... and then waited for layer 3, which the aborted hub never sends
        check_record(sim.srv[s].record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 3))


def test_the_plan_error_pair_is_masked_in_the_published_ids():
    env = env_with_plan_error()
    t = env.tables
    x, ids, w = env.data[0][2]
    hub_ids, srv_ids, errors = M.dispatch_ref(
        x, ids, w, 2, t.slot_map, t.n_cold, t.cold_srv, env.kmax, NSRV
    )
    assert errors == 1 and hub_ids[1, 2] == -1 and (srv_ids[:, 1, 2] == -1).all()


def test_non_owned_pairs_are_dropped_without_an_abort():
    env = make_env(layers=5, seed=9)
    t = env.tables
    owned = np.ones(t.slot_map.shape, np.uint8)
    owned[2, 5] = 0
    owned[3, 1] = 0
    env = env._replace(tables=M.Tables(t.slot_map, t.n_cold, t.cold_srv, owned))
    sim = make_sim(env)
    hub_out, _ = run_island(sim, env)
    assert_exact(env, hub_out)
    assert not any(c.aborted for c in sim.ctxs)
    for (step, layer), (_, _, out) in hub_out.items():  # the dense MoE skips them too
        np.testing.assert_allclose(
            out.astype(np.float32), full_moe(env, step, layer), rtol=2e-2, atol=2e-2
        )
    pairs = sim.hub.st[hx.HX_ST_PAIRS_HUB] + sim.hub.st[hx.HX_ST_PAIRS_REMOTE]
    dropped = sum(
        int((owned[layer][env.data[0][layer][1]] == 0).sum()) for layer in range(5)
    )
    assert dropped > 0 and pairs == 5 * 3 * K - dropped


def test_kmax_may_depend_on_the_step_size():
    env = make_env(layers=3, rows=3)
    sim = make_sim(env, kmax=lambda rows: 0 if rows == 3 else 100)
    hub_out, _ = run_island(sim, env)
    assert sim.hub.st[hx.HX_ST_PAIRS_HUB] == 0  # T = 3: nothing stays on the hub
    assert sim.hub.st[hx.HX_ST_PAIRS_REMOTE] == 3 * 3 * K
    for (step, layer), (_, _, out) in hub_out.items():
        np.testing.assert_allclose(
            out.astype(np.float32), full_moe(env, step, layer), rtol=2e-2, atol=2e-2
        )
    assert not any(c.aborted for c in sim.ctxs)


# ===========================================================================
# HxSim: combine order, interleavings, host pokes, live word
# ===========================================================================
@pytest.mark.parametrize("perm", list(itertools.permutations(range(NSRV))))
def test_combine_order_is_fixed_whatever_the_arrival_order(perm):
    """Servers answer in the order ``perm``; the sum is still hub, 0, 1, 2, 3."""
    hub_part, parts = crafted_partials()
    fixed = M.combine_ref(hub_part, parts)
    env = make_env(layers=1)
    sim = make_sim(env)
    got = {}
    x, ids, w = env.data[0][0]

    def hub():
        yield from sim.begin_step(3, 3)
        yield from sim.dispatch(0, x, ids, w)
        got["out"] = yield from sim.combine(0, hub_part)

    def server(s):
        yield from sim.srv_begin(s, M.StepDesc(1, 3, 3))
        yield from sim.srv_wait(s, 0, 3)
        yield from sim.delay(10 + 3000 * perm.index(s))
        yield from sim.srv_push(s, 0, 3, parts[s])

    sim.run({"hub": hub(), **{f"srv{s}": server(s) for s in range(NSRV)}})
    assert same(got["out"], fixed)
    arrival = sum_in_order(hub_part, parts, perm)  # what an arrival-order sum gives
    assert same(arrival, fixed) == (perm in ((0, 1, 2, 3), (0, 1, 3, 2)))


@pytest.mark.parametrize("order", ["adversarial", "fifo"])
def test_the_scheduler_interleaving_never_changes_the_result(order):
    env = make_env(layers=4, steps=2, seed=8)

    def one_run(seed):
        def compute(step, layer):
            return 7 * (layer + 1) + 13 * seed % 29

        sim = make_sim(env, order=order, seed=seed)
        hub_out, _ = run_island(sim, env, all_srv={"compute": compute})
        assert_exact(env, hub_out)
        assert not any(c.aborted for c in sim.ctxs)
        return sim.now, [bits(out).tobytes() for *_, out in hub_out.values()]

    runs = {seed: one_run(seed) for seed in range(8)}
    assert len({tuple(out) for _, out in runs.values()}) == 1  # same bytes
    assert len({now for now, _ in runs.values()}) > 1  # the schedules did differ
    assert one_run(3) == runs[3]  # the same seed replays the same run


def stalled_combine_run(env, host):
    """A combine waiting for dead server 2 at (0, 3), with a host task alongside."""
    sim = make_sim(env)
    marks: list = []
    hub_out, _ = run_island(
        sim, env, hub={"overlap": 500, "marks": marks},
        srv={2: {"stop": (0, 3)}}, extra={"host": host(sim, marks)},
    )  # fmt: skip
    return sim, hub_out


def test_host_poke_ends_a_combine_wait_at_the_next_eighth_tick():
    env = make_env(layers=4)
    log = {}

    def host(sim, marks):
        while len(marks) < 4:  # the 4th combine is the stalled one (layer 3)
            yield from sim.delay(10)
        yield from sim.delay(marks[3] + 500_013 - sim.now)
        log["poke"] = sim.now
        sim.host_poke("hub")

    sim, hub_out = stalled_combine_run(env, host)
    t0, t1, out = hub_out[(0, 3)]
    assert log["poke"] == t0 + 500_013
    assert t1 - t0 == 500_016  # the poll that reads the word is every 8th
    check_record(sim.hub.record, hx.HX_PH_HOST, 2, 500, hx.seq_of(1, 3))
    assert is_nan(out) and sim.hub.aborted
    assert "host abort poke" in hx.describe_record(sim.hub.record, "hub")


def test_host_poke_ends_a_server_request_wait():
    env = make_env(layers=4)
    sim = make_sim(env)
    marks: list = []
    log: dict = {}

    def host():
        while len(marks) < 3:  # server 0 waits for layer 2, which never comes
            yield from sim.delay(10)
        yield from sim.delay(marks[2] + 100_003 - sim.now)
        sim.host_poke("srv0")
        log["poke"] = sim.now

    _, srv_out = run_island(
        sim,
        env,
        hub={"stop": (0, 2)},
        srv={0: {"marks": marks}},
        extra={"host": host()},
    )
    t0, t1 = srv_out[(0, 0, 2)]
    assert t1 - t0 == 100_008 and log["poke"] == t0 + 100_003
    check_record(sim.srv[0].record, hx.HX_PH_HOST, 0, 100, hx.seq_of(1, 2))
    for s in (1, 2, 3):  # the others were not poked and time out normally
        check_record(sim.srv[s].record, hx.HX_PH_SRV_WAIT, 0, 2000, hx.seq_of(1, 2))


def test_a_poke_set_before_the_wait_fires_at_the_first_poll_boundary():
    env = make_env(layers=2)
    sim = make_sim(env)
    sim.host_poke("hub")
    hub_out, _ = run_island(sim, env)
    t0, t1, out = hub_out[(0, 0)]
    assert t1 - t0 == 8 and is_nan(out)
    check_record(sim.hub.record, hx.HX_PH_HOST, 0, 0, hx.seq_of(1, 0))


def test_a_ready_flag_beats_a_pending_poke():
    env = make_env(layers=3)
    sim = make_sim(env)
    sim.host_poke("hub")
    hub_out, _ = run_island(sim, env, hub={"overlap": 500})  # replies are in already
    assert_exact(env, hub_out)
    assert not sim.hub.aborted


def test_live_record_is_refreshed_every_second_and_never_cleared():
    env = make_env(layers=4)
    samples = {}

    def host(sim, marks):
        while len(marks) < 4:
            yield from sim.delay(10)
        t0 = marks[3]
        for name, dt in (
            ("before_1s", 999_990),
            ("after_1s", 1_000_005),
            ("before_2s", 1_999_990),
            ("after_2s", 2_000_005),
        ):
            yield from sim.delay(t0 + dt - sim.now)
            samples[name] = sim.hub.live

    sim, hub_out = stalled_combine_run(env, host)
    seq = hx.seq_of(1, 3)
    assert samples["before_1s"] == 0
    check_record(samples["after_1s"], hx.HX_PH_COMBINE, 2, 1000, seq, live=True)
    check_record(samples["before_2s"], hx.HX_PH_COMBINE, 2, 1000, seq, live=True)
    check_record(samples["after_2s"], hx.HX_PH_COMBINE, 2, 2000, seq, live=True)
    assert "still waiting" in hx.describe_record(samples["after_1s"], "hub")
    check_record(sim.hub.record, hx.HX_PH_COMBINE, 2, 2000, seq)  # final, not live
    assert sim.hub.live == samples["after_2s"]  # the abort leaves the live word alone


# ===========================================================================
# HxSim: scheduler and argument checks
# ===========================================================================
def tiny_sim():
    return make_sim(make_env(layers=1))


def test_run_returns_the_task_results_and_delay_advances_the_clock():
    sim = tiny_sim()

    def task(n):
        yield from sim.delay(n)
        return sim.now

    assert sim.run({"a": task(100), "b": task(40)}) == {"a": 100, "b": 40}
    assert sim.now == 100
    assert sim.run({}) == {}


def test_run_stops_a_task_that_never_finishes():
    sim = tiny_sim()

    def spin():
        while True:
            yield None

    with pytest.raises(RuntimeError, match="no convergence"):
        sim.run({"spin": spin()}, max_turns=1000)


GOOD_DISPATCH = {
    "x": np.zeros((3, H), np.float16),
    "ids": np.zeros((3, K), np.int32),
    "w": np.zeros((3, K), np.float32),
}


@pytest.mark.parametrize(
    "override",
    [
        {"x": np.zeros((3, H), np.float32)},  # x is fp16
        {"w": np.zeros((3, K), np.float64)},  # w is fp32
        {"ids": np.zeros((3, K + 1), np.int32)},  # K is fixed by the buffers
        {"ids": np.zeros(K, np.int32)},  # not [T, K]
        {"ids": np.zeros((ROWS_CAP + 1, K), np.int32)},  # more rows than rows_cap
        {"ids": np.zeros((2, K), np.int32)},  # begin_step announced 3 rows
    ],
)
def test_dispatch_rejects_bad_arguments(override):
    sim = tiny_sim()
    args = {**GOOD_DISPATCH, **override}

    def task():
        yield from sim.begin_step(3, 3)
        yield from sim.dispatch(0, args["x"], args["ids"], args["w"])

    with pytest.raises(ValueError):
        sim.run({"hub": task()})


@pytest.mark.parametrize("rows", [0, ROWS_CAP + 1])
def test_begin_step_rejects_a_bad_row_count(rows):
    sim = tiny_sim()
    with pytest.raises(ValueError):
        sim.run({"hub": sim.begin_step(rows, 0)})


def test_constructor_and_pokes_reject_bad_arguments():
    env = make_env(layers=1)
    with pytest.raises(ValueError):
        make_sim(env, order="random")
    with pytest.raises(ValueError):
        make_sim(env, fences={"dispatch", "bogus"})
    sim = make_sim(env)
    for call in (
        lambda: sim.host_poke("hub", 0),
        lambda: sim.host_poke("hub", 1 << 64),
        lambda: sim.write("hub", sim.rpl.total_bytes - 4, bytes(8)),
        lambda: sim.write("hub", -1, bytes(8)),
    ):
        with pytest.raises(ValueError):
            call()
