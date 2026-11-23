# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online EPLB: N simulated ranks in lockstep over a rendezvous all-reduce.

Every rank runs in its own thread and meets the others inside the fake collective,
so a rank that schedules a collective the others do not (the failure that would
hang a real TP group) breaks the barrier and fails the test.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import rdna_eplb_online as eo
from vllm.model_executor.layers.fused_moe import rdna_eplb_stats as es

E = 16
LAYERS = 3
TOP_K = 2
# (name, row shape, dtype): 616 int32 words per expert, three 1 KiB chunks.
SHAPES = (
    ("w13_weight_packed", (4, 64), torch.int32),
    ("w13_weight_scale", (2, 64), torch.float16),
    ("w13_qzeros", (2, 8), torch.int32),
    ("w2_weight_packed", (8, 32), torch.int32),
    ("w2_weight_scale", (2, 16), torch.float16),
    ("w2_qzeros", (2, 4), torch.int32),
)


class Bus:
    """Rendezvous all-reduce of ``n`` threads; shapes must agree."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.inputs: list[torch.Tensor | None] = [None] * n
        self.gather = threading.Barrier(n)
        self.release = threading.Barrier(n)
        self.sizes: list[int] = []

    def reduce(self, rank: int, tensor: torch.Tensor) -> torch.Tensor:
        self.inputs[rank] = tensor
        if self.gather.wait(timeout=30) == 0:
            self.sizes.append(tensor.numel())
        inputs = [x for x in self.inputs if x is not None]
        assert len({(tuple(x.shape), x.dtype) for x in inputs}) == 1
        out = torch.stack([x.to(torch.int64) for x in inputs]).sum(0).to(tensor.dtype)
        self.release.wait(timeout=30)
        return out

    def abort(self) -> None:
        self.gather.abort()
        self.release.abort()


class Port:
    def __init__(self, bus: Bus, rank: int, device: torch.device) -> None:
        self.bus, self.rank, self.world_size, self.device = bus, rank, bus.n, device

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.bus.reduce(self.rank, tensor)


class Module(SimpleNamespace):
    """Weak-referenceable stand-in of a RoutedExperts module."""


def payload(layer: int, expert: int, index: int, shape, dtype: torch.dtype):
    gen = torch.Generator().manual_seed(7919 * layer + 31 * expert + 1009 * index)
    if dtype == torch.int32:
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=gen, dtype=dtype)
    return torch.rand(shape, generator=gen).to(dtype)


def make_rank(rank: int, ranks: int, ring: int = 64):
    device = torch.device("cpu")
    stats = es.EplbStats(None, kernel="torch", ring=ring)
    local = E // ranks
    mine = range(rank * local, (rank + 1) * local)
    modules = []
    for pos in range(LAYERS):
        module = Module()
        for i, (name, shape, dtype) in enumerate(SHAPES):
            rows = [payload(pos, e, i, shape, dtype) for e in mine]
            setattr(module, name, torch.stack(rows).to(device))
        expert_map = torch.full((E,), -1, dtype=torch.int32)
        expert_map[mine.start : mine.stop] = torch.arange(local, dtype=torch.int32)
        module.expert_map = expert_map.to(device)
        module.moe_config = SimpleNamespace(
            ep_size=ranks, ep_rank=rank, tp_rank=rank, dp_rank=0
        )
        module.global_num_experts = E
        module.layer_name = f"model.layers.{pos}.mlp.experts"
        module.expert_placement_strategy = "linear"
        ids = torch.zeros((1, TOP_K), dtype=torch.int32, device=device)
        assert stats.layer_stats(module, ids) is not None
        modules.append(module)
    return stats, modules


def make_group(ranks: int, cfg: eo.OnlineConfig, planner=None, ring: int = 64):
    bus = Bus(ranks)
    group = []
    for rank in range(ranks):
        stats, modules = make_rank(rank, ranks, ring)
        ctl = eo.OnlineEplb(
            cfg,
            collective=Port(bus, rank, torch.device("cpu")),
            stats=stats,
            planner=planner if rank == 0 else None,
        )
        group.append((ctl, stats, modules))
    return bus, group


def run(bus: Bus, ctls, steps: int, before=None) -> None:
    errors: list[BaseException] = []

    def work(rank: int, ctl) -> None:
        try:
            for _ in range(steps):
                if before is not None:
                    before(rank, ctl)
                ctl.step()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            bus.abort()

    threads = [threading.Thread(target=work, args=(r, c)) for r, c in enumerate(ctls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "a rank hangs"
    if errors:
        raise errors[0]


def placement(group) -> np.ndarray:
    """Rank of every expert of every layer, proven by the maps and the rows."""
    ranks = len(group)
    owner = np.full((LAYERS, E), -1)
    for rank, (_, _stats, modules) in enumerate(group):
        for pos, module in enumerate(modules):
            slots = module.expert_map.cpu().numpy()
            local = np.flatnonzero(slots >= 0)
            assert sorted(slots[local]) == list(range(E // ranks))
            for e in local:
                assert owner[pos, e] == -1, f"expert {e} of layer {pos} is on two ranks"
                owner[pos, e] = rank
                for i, (name, shape, dtype) in enumerate(SHAPES):
                    want = payload(pos, int(e), i, shape, dtype)
                    got = getattr(module, name)[slots[e]].cpu()
                    assert torch.equal(got, want), (pos, int(e), name)
    assert (owner >= 0).all(), "an expert is on no rank"
    for _, stats, _ in group:
        for pos, layer in enumerate(stats.layers):
            assert np.array_equal(layer.owner_host.numpy(), owner[pos])
            assert np.array_equal(layer.owner.cpu().numpy(), owner[pos])
    return owner


LINEAR = np.tile(np.repeat(np.arange(4), E // 4), (LAYERS, 1))
FAST = dict(mailbox_steps=4, chunk_kb=1, decision_steps=8, max_swaps=4)


@pytest.mark.parametrize("words", [0, 1, 31, 8190, 8191, 8192, 16389, 638400])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"),
        ),
    ],
)
def test_vote_checksum_preserves_modular_contract_and_other_vote_fields(words, device):
    values = [-(2**31), 2**31 - 1, -1, 0, 1, 2**30, -(2**30)]
    data = (values * (-(-words // len(values))))[:words]
    expected = (
        sum(v * (i % eo.CHECK_MOD + 1) for i, v in enumerate(data)) & eo.CHECK_MASK
    )
    ctl = eo.OnlineEplb(eo.OnlineConfig())
    ctl._check = (
        torch.arange(words, dtype=torch.int64, device=device) % eo.CHECK_MOD + 1
    ).to(torch.int32)
    ctl._products = torch.empty(words, dtype=torch.int32, device=device)
    ctl._checksum_sum = torch.empty((), dtype=torch.int32, device=device)
    vote = torch.full((eo.VOTE_WORDS,), -17, dtype=torch.int32, device=device)
    ctl._checksum_into(
        torch.tensor(data, dtype=torch.int32, device=device),
        words,
        vote[eo.V_SENT_A : eo.V_SENT_A + 1],
    )
    want = torch.full_like(vote, -17)
    want[eo.V_SENT_A] = expected
    assert torch.equal(vote, want)


def test_chaos_swaps_keep_every_expert_on_exactly_one_rank():
    cfg = eo.OnlineConfig(chaos=3, chaos_seed=5, **FAST)
    bus, group = make_group(4, cfg)
    ctls = [g[0] for g in group]
    run(bus, ctls, 600)
    assert not any(c.off for c in ctls)
    assert ctls[0].committed >= 10
    assert len({(c.plans, c.committed, c.collectives) for c in ctls}) == 1
    owner = placement(group)
    assert not np.array_equal(owner, LINEAR)
    # Equal cardinality survives any number of swaps.
    assert (np.stack([(owner == r).sum(1) for r in range(4)]) == E // 4).all()
    # One collective per step at most, and a chunk never exceeds its budget.
    assert len(bus.sizes) <= 600
    assert max(bus.sizes) <= max(2 * cfg.chunk_words, eo.HEADER + 6 * cfg.max_swaps)


def test_idle_balancer_costs_one_small_collective_per_mailbox():
    cfg = eo.OnlineConfig(**FAST)
    bus, group = make_group(4, cfg)
    run(bus, [g[0] for g in group], 41)
    assert bus.sizes == [eo.HEADER + eo.SWAP_WORDS * cfg.max_swaps] * 10
    assert np.array_equal(placement(group), LINEAR)


def test_one_rank_error_switches_all_ranks_off_at_the_same_step():
    cfg = eo.OnlineConfig(chaos=2, **FAST)
    bus, group = make_group(4, cfg)
    # Rank 2 loses a tensor: its setup fails, the others are fine.
    del group[2][2][1].w2_weight_packed
    del group[2][2][1].w2_weight_scale
    del group[2][2][1].w2_qzeros
    ctls = [g[0] for g in group]
    run(bus, ctls, 50)
    assert all(c.off for c in ctls)
    assert {c.t for c in ctls} == {cfg.mailbox_steps}
    assert len(bus.sizes) == 1  # the mailbox that carried the error, nothing after
    for rank in (0, 1, 3):
        for module in group[rank][2]:
            assert (module.expert_map.numpy() >= 0).sum() == E // 4


class Scribbler(eo.OnlineEplb):
    """A rank whose receive buffer is damaged before the vote."""

    def _vote_and_commit(self, swap):
        self._recv[3] += 1
        super()._vote_and_commit(swap)


def test_damaged_transfer_fails_the_vote_and_keeps_the_placement():
    cfg = eo.OnlineConfig(chaos=1, chaos_seed=1, **FAST)
    bus = Bus(4)
    group = []
    for rank in range(4):
        stats, modules = make_rank(rank, 4)
        ctl = Scribbler(
            cfg, collective=Port(bus, rank, torch.device("cpu")), stats=stats
        )
        group.append((ctl, stats, modules))
    ctls = [g[0] for g in group]
    run(bus, ctls, 80)
    assert all(c.off for c in ctls)
    assert len({c.t for c in ctls}) == 1
    assert all(c.committed == 0 for c in ctls)
    assert np.array_equal(placement(group), LINEAR)


class FakePlanner:
    """The PlannerClient surface, answering once with a fixed plan."""

    def __init__(self, stats: es.EplbStats, answer) -> None:
        layer = stats.layers[0]
        self.state = np.zeros((LAYERS, layer.off.size), dtype=np.int64)
        self.ring = np.zeros((stats.ring_n, es.WORDS), dtype=np.int32)
        self.owner = np.zeros((LAYERS, E), dtype=np.uint8)
        self.answer = answer
        self.submitted: list[tuple[int, int]] = []
        self.polls = 0
        self.closed = False

    def start(self) -> None:
        pass

    def submit(self, step: int, gseq: int) -> None:
        self.submitted.append((step, gseq))

    def poll(self):
        self.polls += 1
        if self.polls < 3:
            return None
        answer, self.answer = self.answer, []
        return answer

    def close(self) -> None:
        self.closed = True


def group_with_planner(answer, **cfg_kwargs):
    cfg = eo.OnlineConfig(**{**FAST, **cfg_kwargs})
    bus = Bus(4)
    group = []
    planner = None
    for rank in range(4):
        stats, modules = make_rank(rank, 4)
        if rank == 0:
            planner = FakePlanner(stats, answer)
            for i, layer in enumerate(stats.layers):
                planner.owner[i] = layer.owner_host.numpy()
                # Counters the snapshot must carry over verbatim.
                layer.state += torch.arange(layer.state.numel()) * (i + 1)
        ctl = eo.OnlineEplb(
            cfg,
            collective=Port(bus, rank, torch.device("cpu")),
            stats=stats,
            planner=planner if rank == 0 else None,
        )
        group.append((ctl, stats, modules))
    return bus, group, planner


def test_planner_answer_travels_through_the_mailbox_and_is_executed():
    # Layer 1: expert 0 (rank 0) <-> expert 15 (rank 3); layer 2: 5 <-> 9.
    bus, group, planner = group_with_planner([(1, 0, 15), (2, 5, 9)])
    ctls = [g[0] for g in group]
    run(bus, ctls, 120)
    assert not any(c.off for c in ctls)
    assert [c.committed for c in ctls] == [2] * 4
    want = LINEAR.copy()
    want[1, 0], want[1, 15] = 3, 0
    want[2, 5], want[2, 9] = 2, 1
    assert np.array_equal(placement(group), want)
    assert np.array_equal(planner.owner, want)
    # The snapshot is the counters, layer by layer, and it was submitted.
    stats = group[0][1]
    for i, layer in enumerate(stats.layers):
        assert np.array_equal(planner.state[i], layer.state.numpy())
    assert len(planner.submitted) >= 1


@pytest.mark.parametrize(
    "answer",
    [
        [(1, 0, 1)],  # both on rank 0
        [(9, 0, 15)],  # no such layer
        [(1, 0, 99)],  # no such expert
        [(1, 0, 15), (1, 15, 3)],  # an expert twice in one plan
    ],
)
def test_invalid_planner_answer_is_never_posted(answer):
    bus, group, _ = group_with_planner(answer)
    ctls = [g[0] for g in group]
    run(bus, ctls, 80)
    assert not any(c.off for c in ctls)
    assert all(c.plans == 0 for c in ctls)
    assert np.array_equal(placement(group), LINEAR)


def test_plan_against_the_placement_is_refused_by_the_vote():
    bus, group, _ = group_with_planner([])
    ctls = [g[0] for g in group]
    # A forged plan: expert 0 of layer 0 is on rank 0, the plan says rank 1.
    forged = np.zeros(eo.HEADER + eo.SWAP_WORDS, dtype=np.int32)
    forged[eo.M_SEQ], forged[eo.M_SWAPS] = 1, 1
    forged[eo.HEADER :] = (0, 0, 15, 1, 3, group[0][0].cfg.chunk_words)
    ctls[0]._outbox = forged
    run(bus, ctls, 80)
    assert all(c.off for c in ctls)
    assert len({c.t for c in ctls}) == 1
    assert np.array_equal(placement(group), LINEAR)


def test_single_rank_and_flag_off_do_nothing(monkeypatch):
    bus = Bus(1)
    stats, _ = make_rank(0, 1)
    ctl = eo.OnlineEplb(
        eo.OnlineConfig(chaos=1, **FAST),
        collective=Port(bus, 0, torch.device("cpu")),
        stats=stats,
    )
    for _ in range(20):
        ctl.step()
    assert ctl.off and bus.sizes == []
    monkeypatch.delenv(es.ONLINE_ENV, raising=False)
    monkeypatch.delenv(es.STATS_DIR_ENV, raising=False)
    assert eo.OnlineEplb.from_env() is None
    assert es.EplbStats.from_env() is None


def test_flag_on_enables_the_counters_without_snapshot_files(monkeypatch, tmp_path):
    monkeypatch.delenv(es.STATS_DIR_ENV, raising=False)
    monkeypatch.setenv(es.ONLINE_ENV, "1")
    monkeypatch.setenv(eo.CHUNK_ENV, "32")
    monkeypatch.chdir(tmp_path)
    stats = es.EplbStats.from_env()
    assert stats is not None and stats.directory is None
    ctl = eo.OnlineEplb.from_env()
    assert ctl is not None and ctl.cfg.chunk_words == 32 * 256
    stats.tick()
    stats.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mailbox_steps": 0},
        {"mailbox_steps": 1},
        {"chunk_every": 256},
        {"mailbox_steps": 4, "chunk_every": 8},
        {"chunk_kb": 0},
        {"chunk_kb": 8192},
        {"max_swaps": 0},
        {"objective": "speed"},
        {"chaos": -1},
    ],
)
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        eo.OnlineConfig(**kwargs)


def test_skewed_traffic_is_rebalanced_by_the_real_planner_process():
    """End to end: counters -> shared memory -> helper process -> mailbox -> swaps.

    Experts 0..3 (all on rank 0 under the linear placement) are six times hotter
    than the others, so rank 0 touches far more experts per call than its peers.
    """
    import time

    hot = 4
    cfg = eo.OnlineConfig(mailbox_steps=8, chunk_kb=1, decision_steps=48, max_swaps=4)
    # Two decision windows of ring records must fit: 3 layers x ~55 steps each.
    bus, group = make_group(4, cfg, ring=2048)
    ctls = [g[0] for g in group]
    weights = torch.ones(E)
    weights[:hot] = 6.0

    def traffic(rank: int, ctl) -> None:
        gen = torch.Generator().manual_seed(ctl.t)  # the same ids on every rank
        ids = torch.multinomial(
            weights.expand(3, E), TOP_K, replacement=False, generator=gen
        ).to(torch.int32)
        for layer in group[rank][1].layers:
            layer.count(ids)
        if rank == 0:
            time.sleep(0.001)  # wall time for the helper process

    def hot_per_rank() -> np.ndarray:
        owner = np.stack([s.owner_host.numpy() for s in group[0][1].layers])
        return np.stack([(owner[:, :hot] == r).sum(1) for r in range(4)], 1)

    assert hot_per_rank().max() == hot
    try:
        # One plan is capped at max_swaps, so the layers are served over
        # several decisions.
        for _ in range(40):
            run(bus, ctls, 100, traffic)
            assert not any(c.off for c in ctls)
            if (hot_per_rank()[:, 0] < hot).all():
                break
        assert ctls[0].committed >= 3, "the planner never rebalanced"
        assert len({(c.plans, c.committed) for c in ctls}) == 1
        placement(group)
        # Every layer got hot experts off rank 0.
        assert (hot_per_rank()[:, 0] < hot).all(), hot_per_rank()
    finally:
        for ctl in ctls:
            ctl.close()


def _gfx1030_moe() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from vllm import _custom_ops  # noqa: F401  (loads _rocm_C)
        from vllm.platforms.rocm import on_gfx10x

        return on_gfx10x() and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2")
    except Exception:
        return False


@pytest.mark.skipif(not _gfx1030_moe(), reason="needs gfx1030 and the RDNA2 MoE op")
@pytest.mark.parametrize("M", [3, 17])
def test_swapped_resident_experts_serve_the_same_moe_in_eager_and_replay(
    M, monkeypatch
):
    """Two ranks on one GPU: the real kernels follow the rewritten slots and map.

    The graphs are captured before any swap, as in serving: they must read the
    new weights and the new ``expert_map`` through the pointers they captured.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.rdna2_moe_resident import (
        apply_resident,
        prepare_resident_layer,
    )
    from vllm.platforms import current_platform
    from vllm.utils import gpu_sync_debug as sync

    vote_and_commit = eo.OnlineEplb._vote_and_commit

    def checked_vote_and_commit(self, swap):
        with sync._checked_region("error"):
            return vote_and_commit(self, swap)

    monkeypatch.setattr(sync, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(eo.OnlineEplb, "_vote_and_commit", checked_vote_and_commit)

    torch.manual_seed(23)
    dev = torch.device("cuda")
    ranks, local, experts, K, N_inter, top_k, group = 2, 4, 8, 128, 256, 2, 128
    dense: list[list[torch.Tensor]] = [[], []]  # [w13, w2][expert]
    bus = Bus(ranks)
    group_ = []
    for rank in range(ranks):
        module = torch.nn.Module()
        for which, (name, n, k) in enumerate(
            (("w13", 2 * N_inter, K), ("w2", K, N_inter))
        ):
            codes = torch.randint(0, 16, (local, n, k), dtype=torch.uint8, device=dev)
            weight = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous()
            scales = (
                torch.rand(local, n, k // group, device=dev, dtype=torch.float16) * 0.01
            )
            full = ((codes.float() - 8) * scales.repeat_interleave(group, -1)).half()
            dense[which].extend(full.unbind(0))
            module.register_parameter(
                name + "_qweight", torch.nn.Parameter(weight, requires_grad=False)
            )
            module.register_parameter(
                name + "_scales", torch.nn.Parameter(scales, requires_grad=False)
            )
        prepare_resident_layer(module, group)
        module.activation = MoEActivation.SILU
        module.apply_router_weight_on_input = False
        module.global_num_experts = experts
        expert_map = torch.full((experts,), -1, dtype=torch.int32)
        expert_map[rank * local : (rank + 1) * local] = torch.arange(local)
        module.expert_map = expert_map.to(dev)
        module.moe_config = SimpleNamespace(
            ep_size=ranks, ep_rank=rank, tp_rank=rank, dp_rank=0
        )
        module.layer_name = "model.layers.0.mlp.experts"
        module.expert_placement_strategy = "linear"
        stats = es.EplbStats(None, kernel="torch", ring=64)
        ids0 = torch.zeros((1, top_k), dtype=torch.int32, device=dev)
        assert stats.layer_stats(module, ids0) is not None
        cfg = eo.OnlineConfig(
            mailbox_steps=4, chunk_kb=8, decision_steps=8, max_swaps=4, chaos=2
        )
        ctl = eo.OnlineEplb(cfg, collective=Port(bus, rank, dev), stats=stats)
        group_.append((ctl, stats, module))

    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    topk_ids = torch.stack(
        [torch.randperm(experts, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(M, top_k, device=dev), dim=-1)

    def reference() -> torch.Tensor:
        ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
        for m in range(M):
            for slot in range(top_k):
                e = int(topk_ids[m, slot])
                first = torch.nn.functional.linear(
                    x[m : m + 1].float(), dense[0][e].float()
                ).half()
                gate, up = first.float().chunk(2, -1)
                act = (torch.nn.functional.silu(gate) * up).half()
                second = torch.nn.functional.linear(act.float(), dense[1][e].float())
                ref[m] += second.squeeze(0) * topk_weights[m, slot]
        return ref

    def eager() -> torch.Tensor:
        return sum(
            apply_resident(g[2], x, topk_weights, topk_ids).float() for g in group_
        )

    torch.testing.assert_close(eager(), reference(), atol=0.001, rtol=0.02)
    graphs, captured = [], []
    for _, _, module in group_:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured.append(apply_resident(module, x, topk_weights, topk_ids))
        graphs.append(graph)

    before = [g[2].expert_map.clone() for g in group_]
    run(bus, [g[0] for g in group_], 300)
    ctls = [g[0] for g in group_]
    assert not any(c.off for c in ctls)
    assert ctls[0].committed >= 4 and ctls[0].committed == ctls[1].committed
    assert any(not torch.equal(b, g[2].expert_map) for b, g in zip(before, group_))
    owned = torch.stack([g[2].expert_map >= 0 for g in group_]).sum(0)
    assert bool((owned == 1).all())

    torch.testing.assert_close(eager(), reference(), atol=0.001, rtol=0.02)
    for factor in (1.0, 0.5, -1.0):
        x.mul_(factor)
        for graph in graphs:
            graph.replay()
        current_platform.synchronize()
        replayed = sum(c.float() for c in captured)
        torch.testing.assert_close(replayed, reference(), atol=0.001, rtol=0.02)


def _has_op(name: str) -> bool:
    return _gfx1030_moe() and hasattr(torch.ops._rocm_C, name)


@pytest.mark.skipif(not _gfx1030_moe(), reason="needs gfx1030 and the RDNA2 MoE op")
@pytest.mark.parametrize("w4a8", [False, True])
@pytest.mark.parametrize("M", [4, 32])
def test_swapped_k_first_experts_serve_the_same_moe(M, w4a8):
    """The K-first layout (W4A16 and W4A8 activations) under swapped experts.

    Each expert's partial is computed by the same kernel from the same bytes
    wherever it lives, so the two ranks' sum only differs by the fp16 order of
    the accumulation inside a rank.
    """
    if w4a8 and not _has_op("moe_w4a8_gemm_rdna2"):
        pytest.skip("moe_w4a8_gemm_rdna2 is not built")
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
        _rdna2_fused_moe,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    torch.manual_seed(29)
    dev = torch.device("cuda")
    ranks, local, experts, K, N_inter, top_k, group = 2, 4, 8, 512, 256, 4, 64

    def packed(k: int, n: int) -> torch.Tensor:
        q = torch.randint(0, 16, (experts, k, n), dtype=torch.int32, device=dev)
        out = torch.zeros(experts, k // 8, n, dtype=torch.int32, device=dev)
        for i in range(8):
            out |= (q[:, i::8, :] & 0xF) << (i * 4)
        for e in range(experts):
            row = out[e].contiguous()
            ops.gptq_shuffle(row, 4)
            out[e] = row
        return out

    def scales(groups: int, n: int) -> torch.Tensor:
        return (0.05 * torch.rand((experts, groups, n), device=dev) + 0.01).half()

    def qzeros(groups: int, n: int) -> torch.Tensor:
        zeros = torch.full(
            (groups, n), scalar_types.uint4b8.bias - 1, dtype=torch.int32, device=dev
        )
        one = pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, 1)
        return one.unsqueeze(0).expand(experts, -1, -1).contiguous()

    full = {
        "w13_weight_packed": packed(K, 2 * N_inter),
        "w2_weight_packed": packed(N_inter, K),
        "w13_weight_scale": scales(K // group, 2 * N_inter),
        "w2_weight_scale": scales(N_inter // group, K),
        "w13_qzeros": qzeros(K // group, 2 * N_inter),
        "w2_qzeros": qzeros(N_inter // group, K),
    }
    bus = Bus(ranks)
    group_ = []
    for rank in range(ranks):
        module = Module()
        for name, tensor in full.items():
            setattr(module, name, tensor[rank * local : (rank + 1) * local].clone())
        module.rdna2_w1_buf = torch.zeros(
            M * top_k, 2 * N_inter, dtype=torch.float16, device=dev
        )
        module.rdna2_act_buf = torch.empty(
            M * top_k, N_inter, dtype=torch.float16, device=dev
        )
        module.rdna2_empty_tw = torch.empty(0, device=dev)
        expert_map = torch.full((experts,), -1, dtype=torch.int32)
        expert_map[rank * local : (rank + 1) * local] = torch.arange(local)
        module.expert_map = expert_map.to(dev)
        module.moe_config = SimpleNamespace(
            ep_size=ranks, ep_rank=rank, tp_rank=rank, dp_rank=0
        )
        module.global_num_experts = experts
        module.layer_name = "model.layers.0.mlp.experts"
        module.expert_placement_strategy = "linear"
        stats = es.EplbStats(None, kernel="torch", ring=64)
        ids0 = torch.zeros((1, top_k), dtype=torch.int32, device=dev)
        assert stats.layer_stats(module, ids0) is not None
        cfg = eo.OnlineConfig(
            mailbox_steps=4, chunk_kb=32, decision_steps=8, max_swaps=4, chaos=2
        )
        ctl = eo.OnlineEplb(cfg, collective=Port(bus, rank, dev), stats=stats)
        group_.append((ctl, stats, module))

    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    topk_ids = torch.stack(
        [torch.randperm(experts, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.softmax(torch.randn(M, top_k, device=dev), dim=-1)

    def forward() -> torch.Tensor:
        return sum(
            _rdna2_fused_moe(
                x,
                topk_w,
                topk_ids,
                layer=g[2],
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
                global_num_experts=experts,
                expert_map=g[2].expert_map,
                w4a8=w4a8,
            ).float()
            for g in group_
        )

    before = forward()
    assert torch.isfinite(before).all() and float(before.abs().mean()) > 0
    maps = [g[2].expert_map.clone() for g in group_]
    ctls = [g[0] for g in group_]
    run(bus, ctls, 200)
    assert not any(c.off for c in ctls) and ctls[0].committed >= 4
    assert any(not torch.equal(m, g[2].expert_map) for m, g in zip(maps, group_))
    # Every expert's bytes are where the maps say, bit for bit.
    for name, tensor in full.items():
        for _, _, module in group_:
            slots = module.expert_map
            for e in torch.nonzero(slots >= 0).flatten().tolist():
                assert torch.equal(getattr(module, name)[int(slots[e])], tensor[e])
    after = forward()
    torch.testing.assert_close(after, before, atol=0.02, rtol=0.01)


@pytest.mark.skipif(not _gfx1030_moe(), reason="needs gfx1030")
def test_mailbox_error_disables_balancer_under_strict_sync_check(monkeypatch):
    """The intentional mailbox handshake must not kill a sync-checked worker."""
    from vllm.utils import gpu_sync_debug as sync

    device = torch.device("cuda")
    cfg = eo.OnlineConfig()
    # A one-participant rendezvous exercises the real GPU mailbox transfers;
    # the error word follows the same coordinated-disable path as a peer error.
    ctl = eo.OnlineEplb(cfg, collective=Port(Bus(1), 0, device))
    ctl._mail_host = torch.zeros(
        eo.HEADER + eo.SWAP_WORDS * cfg.max_swaps, dtype=torch.int32
    )
    ctl._mail = torch.zeros_like(ctl._mail_host, device=device)
    ctl._error = "setup failed"
    ctl.t = cfg.mailbox_steps
    monkeypatch.setattr(sync, "_SYNC_CHECK_MODE", "error")
    with sync._checked_region("error"):
        ctl._mailbox()
    assert ctl.off
    assert ctl.collectives == 1
    assert ctl._mail.cpu()[eo.M_ERRORS].item() == 1
