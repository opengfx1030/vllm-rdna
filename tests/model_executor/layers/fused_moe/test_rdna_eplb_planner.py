# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the online EPLB planner (numpy, a real helper process, no GPU).

Counters and ring records are synthesised exactly as ``rdna_eplb_stats`` lays them out
(``offsets()``, ring words of ``tools/rdna/eplb/summary.py``), so ``plan`` and the
helper process see what the worker would hand them.
"""

import dataclasses
import os
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from vllm.model_executor.layers.fused_moe import rdna_eplb_planner as planner
from vllm.model_executor.layers.fused_moe.rdna_eplb_planner import (
    PlannerClient,
    PlannerSpec,
    Snapshot,
    plan,
    ring_samples,
)

E, R, NB = 64, 4, 3
EDGES = (1, 2, 4)
ROWS = 2  # valid rows of every synthetic decode call: band 1
WORDS = 32
EXPERT_BYTES = 2_534_400


def make_spec(
    layers: int = 2,
    ring_records: int = 1 << 14,
    max_swaps: int = 8,
    objective: str = "touch",
    **kw,
) -> PlannerSpec:
    off_touch = 2 * E
    off_calls = off_touch + NB * E
    off_rows = off_calls + NB + 1
    off_ordinal = off_rows + 2
    return PlannerSpec(
        layers=layers,
        experts=E,
        ranks=R,
        nb=NB,
        edges=EDGES,
        state_words=off_ordinal + 2,
        off_touch=off_touch,
        off_calls=off_calls,
        off_rows=off_rows,
        off_ordinal=off_ordinal,
        ring_records=ring_records,
        ring_words=WORDS,
        max_swaps=max_swaps,
        objective=objective,
        expert_bytes=EXPERT_BYTES,
        **kw,
    )


def linear_owner(layers: int) -> np.ndarray:
    return np.tile((np.arange(E) // (E // R)).astype(np.uint8), (layers, 1))


class Sim:
    """The device counters and the ring of ``rdna_eplb_stats`` for synthetic calls."""

    def __init__(self, spec: PlannerSpec, owner: np.ndarray, seed: int = 0) -> None:
        self.spec, self.owner = spec, owner
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((spec.layers, spec.state_words), np.int64)
        self.ring = np.zeros((spec.ring_records, WORDS), np.int32)
        self.gseq = 0
        self.truth: list[list[np.ndarray]] = [[] for _ in range(spec.layers)]

    def _record(self, layer: int, phase: int, per_rank: np.ndarray) -> None:
        sp, st = self.spec, self.state[layer]
        rec = np.zeros(WORDS, np.int32)
        rec[0], rec[1], rec[2], rec[6] = layer, ROWS, ROWS, phase
        rec[7] = st[sp.off_ordinal]
        rec[8 + phase * 8 : 8 + phase * 8 + R] = per_rank
        if sp.ring_records:
            self.ring[self.gseq % sp.ring_records] = rec
        self.gseq += 1
        st[sp.off_ordinal] += 1

    def decode(self, calls: int, p: list[np.ndarray]) -> None:
        """``calls`` decode calls per layer; ``p[l][e]`` = P(expert touched)."""
        sp = self.spec
        for _ in range(calls):
            for layer in range(sp.layers):
                touched = self.rng.random(E) < p[layer]
                st = self.state[layer]
                st[sp.off_touch + 1 * E : sp.off_touch + 2 * E] += touched
                st[sp.off_calls + 1] += 1
                per_rank = np.bincount(
                    self.owner[layer], weights=touched, minlength=R
                ).astype(np.int32)
                self.truth[layer].append(per_rank)
                self._record(layer, 0, per_rank)

    def prefill(self, calls: int, lam: list[np.ndarray]) -> None:
        """``calls`` prefill calls per layer; ``lam[l][e]`` = mean pairs of expert e."""
        sp = self.spec
        for _ in range(calls):
            for layer in range(sp.layers):
                pairs = self.rng.poisson(lam[layer]).astype(np.int64)
                st = self.state[layer]
                st[E : 2 * E] += pairs
                st[sp.off_calls + NB] += 1
                per_rank = np.bincount(
                    self.owner[layer], weights=pairs, minlength=R
                ).astype(np.int32)
                self._record(layer, 1, per_rank)

    def snapshot(self, before: Snapshot | None = None) -> Snapshot:
        return Snapshot(
            self.state.copy(), self.owner.copy(), self.gseq, self.ring.copy(), before
        )


def hot(rate: float, base: float = 0.05, n: int = 8) -> np.ndarray:
    """Eight experts on rank 0 are touched at ``rate``; the rest at ``base``."""
    p = np.full(E, base)
    p[:n] = rate
    return p


def chain(spec, windows, seed=0, owner=None):
    """Snapshots after each window of ``(calls, p-per-layer)``; ``s[-2].before`` set."""
    owner = linear_owner(spec.layers) if owner is None else owner
    sim = Sim(spec, owner, seed)
    snaps = [sim.snapshot()]
    for calls, p in windows:
        sim.decode(calls, p)
        snaps.append(sim.snapshot(snaps[-1]))
    return sim, snaps


def modelled_slowest(owner_row: np.ndarray, p: np.ndarray) -> float:
    return float(np.bincount(owner_row, weights=p, minlength=R).max())


def apply(owner: np.ndarray, swaps) -> np.ndarray:
    out = owner.copy()
    for layer, a, b in swaps:
        out[layer][a], out[layer][b] = out[layer][b], out[layer][a]
    return out


# ---- plan --------------------------------------------------------------------------


def test_persistent_overload_gets_swaps_that_lower_the_slowest_rank():
    spec = make_spec(layers=2, max_swaps=8)
    p = [hot(0.9), np.full(E, 0.3)]  # layer 0: rank 0 overloaded; layer 1: balanced
    _, s = chain(spec, [(400, p), (400, p)])
    owner = linear_owner(2)
    swaps, report = plan(s[1], s[2], owner, spec)
    assert swaps and len(swaps) <= spec.max_swaps
    assert {layer for layer, _, _ in swaps} == {0}
    seen: set[int] = set()
    for layer, a, b in swaps:
        assert owner[layer][a] != owner[layer][b]
        assert a not in seen and b not in seen
        seen.update((a, b))
    after = apply(owner, swaps)
    assert modelled_slowest(after[0], p[0]) < 0.8 * modelled_slowest(owner[0], p[0])
    assert np.array_equal(after[1], owner[1])
    entry = report["layers"][0]
    assert entry["reason"] == "rebalance" and entry["swaps"] == len(swaps)
    assert entry["pred_max_load"] < entry["cur_max_load"]
    assert report["reason"] == "rebalance"
    assert report["layers"][1]["reason"] == "gain below min_rel_gain"
    # deterministic: the same input gives the same plan and report
    assert plan(s[1], s[2], owner, spec) == (swaps, report)


def test_max_swaps_caps_the_plan_and_the_best_layer_goes_first():
    spec = make_spec(layers=2, max_swaps=2)
    p = [hot(0.5), hot(0.9)]  # layer 1 gains more
    _, s = chain(spec, [(400, p), (400, p)])
    swaps, report = plan(s[1], s[2], linear_owner(2), spec)
    assert len(swaps) == 2
    assert {layer for layer, _, _ in swaps} == {1}
    assert report["layers"][0]["reason"] in ("plan is full", "rebalance")
    wide = make_spec(layers=2, max_swaps=64)
    both, _ = plan(s[1], s[2], linear_owner(2), wide)
    assert {layer for layer, _, _ in both} == {0, 1}
    assert both[0][0] == 1  # the layer with the best net gain comes first


@pytest.mark.parametrize("seed", range(4))
def test_stationary_noise_only_input_keeps_the_placement(seed):
    spec = make_spec(layers=2)
    p = [np.full(E, 0.3)] * 2
    _, s = chain(spec, [(400, p), (400, p)], seed=seed)
    swaps, report = plan(s[1], s[2], linear_owner(2), spec)
    assert swaps == []
    assert report["reason"] == "keep the placement"


def test_first_and_second_snapshot_only_prime_the_baseline():
    spec = make_spec()
    p = [hot(0.9)] * 2
    _, s = chain(spec, [(400, p), (400, p)])
    owner = linear_owner(2)
    swaps, report = plan(None, s[0], owner, spec)
    assert swaps == [] and "first snapshot" in report["reason"]
    swaps, report = plan(s[0], s[1], owner, spec)  # one window: no holdout possible
    assert swaps == [] and "two windows" in report["reason"]


def test_holdout_rejects_a_one_window_transient():
    spec = make_spec(layers=1)
    flat, skewed = [np.full(E, 0.3)], [hot(0.9)]
    owner = linear_owner(1)
    _, s = chain(spec, [(400, flat), (400, skewed)])
    swaps, report = plan(s[1], s[2], owner, spec)
    assert swaps == []
    assert report["layers"][0]["reason"] == "holdout gain below min_rel_gain"
    _, s = chain(spec, [(400, skewed), (400, skewed)])  # the same skew in both windows
    assert plan(s[1], s[2], owner, spec)[0]


def test_a_commit_in_the_window_skips_the_layer_until_two_clean_windows():
    spec = make_spec(layers=1)
    p = [hot(0.9)]
    _, s = chain(spec, [(400, p), (400, p)])
    moved = linear_owner(1)
    moved[0][0], moved[0][63] = 3, 0
    s[1].owner = moved  # the placement changed between the first two snapshots
    swaps, report = plan(s[1], s[2], s[2].owner, spec)
    assert swaps == []
    assert "placement changed" in report["layers"][0]["reason"]


def test_prefill_pairs_objective_balances_pair_counts():
    spec = make_spec(layers=1, objective="pairs")
    lam = [np.where(np.arange(E) < 8, 40.0, 2.0)]
    owner = linear_owner(1)
    sim = Sim(spec, owner, seed=2)
    snaps = [sim.snapshot()]
    for _ in range(2):
        sim.prefill(300, lam)
        snaps.append(sim.snapshot(snaps[-1]))
    swaps, report = plan(snaps[1], snaps[2], owner, spec)
    assert swaps and report["layers"][0]["reason"] == "rebalance"
    after = apply(owner, swaps)
    assert modelled_slowest(after[0], lam[0]) < 0.8 * modelled_slowest(owner[0], lam[0])


# ---- ring decode ---------------------------------------------------------------------


def test_ring_decode_after_wrap_returns_the_newest_records_in_order():
    spec = make_spec(layers=2, ring_records=50)
    sim = Sim(spec, linear_owner(2), seed=3)
    sim.decode(100, [hot(0.9), hot(0.5)])  # 200 records through a 50-slot ring
    assert sim.gseq == 200 and sim.gseq % 50 == 0
    sim.decode(3, [hot(0.9), hot(0.5)])  # gseq no longer a multiple of the ring
    for layer in range(2):
        got = ring_samples(spec, sim.ring, sim.gseq, layer, 0, 10**6)
        want = np.array(sim.truth[layer][-25:], dtype=np.float64)  # 25 calls per layer
        assert got.shape == (25, R) and np.array_equal(got, want)
        part = ring_samples(spec, sim.ring, sim.gseq, layer, 90, 100)
        assert np.array_equal(
            part, np.array(sim.truth[layer][90:100], dtype=np.float64)
        )


def test_ring_shorter_than_the_history_and_empty_rings_decode_without_error():
    spec = make_spec(layers=1, ring_records=50)
    sim = Sim(spec, linear_owner(1), seed=4)
    assert ring_samples(spec, sim.ring, 0, 0, 0, 10).shape == (0, R)  # nothing yet
    sim.decode(20, [hot(0.9)])  # ring not full: slots beyond gseq hold zeros
    got = ring_samples(spec, sim.ring, sim.gseq, 0, 0, 10**6)
    assert np.array_equal(got, np.array(sim.truth[0], dtype=np.float64))
    none = make_spec(layers=1, ring_records=0)
    assert ring_samples(none, np.zeros((0, WORDS), np.int32), 7, 0, 0, 10).shape == (
        0,
        R,
    )


def test_plan_uses_what_a_wrapped_ring_still_holds_and_skips_when_too_little():
    p = [hot(0.9), np.full(E, 0.3)]
    # 2 layers x 400 calls kept per layer: the whole new window and 100 old calls
    spec = make_spec(layers=2, ring_records=800)
    _, s = chain(spec, [(300, p), (300, p)])
    swaps, report = plan(s[1], s[2], linear_owner(2), spec)
    assert swaps and report["layers"][0]["samples"][0] == 100
    # a ring that kept only the new window cannot back the holdout
    tiny = make_spec(layers=2, ring_records=600)
    _, s = chain(tiny, [(300, p), (300, p)])
    swaps, report = plan(s[1], s[2], linear_owner(2), tiny)
    assert swaps == []
    assert report["layers"][0]["reason"] == "too few ring records in a window"
    none = make_spec(layers=2, ring_records=0)
    _, s = chain(none, [(300, p), (300, p)])
    assert plan(s[1], s[2], linear_owner(2), none)[0] == []


def test_spec_validates_and_round_trips_through_json():
    spec = make_spec()
    assert PlannerSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError):
        dataclasses.replace(spec, objective="both")
    with pytest.raises(ValueError):
        dataclasses.replace(spec, edges=(1, 2))


# ---- the helper process ----------------------------------------------------------


def shm_names(client: PlannerClient) -> list[str]:
    return [b.name for b in client._blocks]


def gone(name: str) -> bool:
    try:
        shared_memory.SharedMemory(name=name).close()
    except FileNotFoundError:
        return True
    return False


def load(client: PlannerClient, snap: Snapshot) -> None:
    client.state[:] = snap.state
    client.owner[:] = snap.owner
    client.ring[:] = snap.ring


def wait_answer(client: PlannerClient, times: list[float] | None = None, limit=30.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        t0 = time.perf_counter()
        got = client.poll()
        if times is not None:
            times.append(time.perf_counter() - t0)
        if got is not None:
            return got
        time.sleep(0.005)
    raise TimeoutError("no answer from the planner helper")


@pytest.fixture
def three_windows():
    spec = make_spec(layers=2, ring_records=1 << 12)
    p = [hot(0.9), np.full(E, 0.3)]
    _, s = chain(spec, [(300, p), (300, p)])
    return spec, s


def test_client_round_trip_matches_the_in_process_plan(three_windows):
    spec, s = three_windows
    client = PlannerClient(spec)
    names = shm_names(client)
    try:
        client.start()
        assert client.alive()
        want, _ = plan(s[1], s[2], linear_owner(2), spec)
        assert want
        for i, snap in enumerate(s):  # baseline, one window, two windows
            load(client, snap)
            assert client.poll() is None  # nothing submitted: never blocks
            assert client.submit(step=100 * i, gseq=snap.gseq)
            assert client.pending and not client.submit(1, 1)  # one in flight
            got = wait_answer(client)
            assert not client.pending and client.answered_step == 100 * i
            assert got == (want if i == 2 else [])
        report = client.last_report
        assert report is not None and report["layers"]["0"]["reason"] == "rebalance"
        # the helper copied the blocks on submit: overwrite them, ask again, same answer
        client.state[:] = 0
        assert client.submit(step=300, gseq=0)
        assert wait_answer(client) is not None
        assert client.alive()
    finally:
        client.close()
    assert all(gone(n) for n in names)
    assert not client.alive() and client.poll() == []
    client.close()  # idempotent


def test_helper_has_not_imported_torch_or_vllm(three_windows):
    spec, _ = three_windows
    client = PlannerClient(spec)
    try:
        client.start()
        assert client.helper_modules() == []
        assert client.alive()
    finally:
        client.close()


def test_submit_and_poll_are_cheap_and_never_block(three_windows):
    spec, s = three_windows
    client = PlannerClient(spec)
    try:
        client.start()
        load(client, s[2])
        submit_t: list[float] = []
        poll_t: list[float] = []
        for step in range(20):
            t0 = time.perf_counter()
            sent = client.submit(step, s[2].gseq)
            submit_t.append(time.perf_counter() - t0)
            assert sent
            wait_answer(client, poll_t)
        for sample in (submit_t, poll_t):
            assert sorted(sample)[int(0.9 * len(sample))] < 1e-3  # loose: ~1 ms
    finally:
        client.close()


def test_a_dead_helper_is_reported_and_never_raises(three_windows):
    spec, s = three_windows
    client = PlannerClient(spec)
    names = shm_names(client)
    try:
        client.start()
        load(client, s[2])
        assert client.submit(1, s[2].gseq)
        assert client._proc is not None
        client._proc.kill()
        client._proc.wait()
        assert client.poll() == []  # death surfaces as "keep the placement"
        assert not client.alive()
        assert client.submit(2, s[2].gseq) is False
        assert client.poll() == []
        assert client.last_error
    finally:
        client.close()
    assert all(gone(n) for n in names)


def test_helper_exits_when_the_parent_pipe_closes(three_windows):
    spec, _ = three_windows
    client = PlannerClient(spec)
    try:
        client.start()
        proc = client._proc
        assert proc is not None and proc.stdin is not None
        proc.stdin.close()  # what the kernel does when the parent dies
        assert proc.wait(timeout=20) == 0
        assert not client.alive()
    finally:
        client.close()


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs procfs")
def test_helper_is_niced_and_runs_the_file_by_path(three_windows):
    spec, _ = three_windows
    client = PlannerClient(spec)
    try:
        client.start()
        proc = client._proc
        assert proc is not None
        cmd: list[bytes] = []
        for _ in range(100):  # the kernel fills cmdline once the exec has finished
            with open(f"/proc/{proc.pid}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
            if len(cmd) > 1:
                break
            time.sleep(0.05)
        assert os.path.abspath(planner.__file__).encode() in cmd
        # field 19 of /proc/<pid>/stat is the nice value (set before the first line)
        for _ in range(200):
            with open(f"/proc/{proc.pid}/stat") as f:
                stat = f.read().rsplit(")", 1)[1].split()
            if int(stat[16]) == 19:
                break
            time.sleep(0.05)
        assert int(stat[16]) == 19
    finally:
        client.close()
