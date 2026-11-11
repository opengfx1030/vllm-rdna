# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU exactness of the hierarchical rdna_ar all-reduce (VLLM_RDNA_AR_HIER).

Two islands of G ranks (TP8 on the 2 x 4 V620 box: HIP 1-4 behind root port
c0, HIP 6-9 behind root port 00; TP4 with two forced islands of two). The
result of every algorithm of a hier group must be bit-equal on every rank to
the CPU reference of the island-grouped arithmetic: each island's payloads
are summed in rank order in fp32 and rounded to the tensor dtype, then the two
partials are added in fp32 and rounded once more. That holds for the flat
exchange with that arithmetic (flat-isl), for the 3-phase hier algorithm with
any number of owners and any block count, for the size dispatch from the hier
threshold up, for misaligned views and ragged lengths, for adversarial payloads
(order-sensitive magnitudes, NaN / Inf / signed zeros / subnormals, fp16 island
overflow), and for CUDA-graph replays with new inputs. The flat algorithm of the
same instance keeps the old rank-order numerics, and so does the size dispatch
below the hier threshold. Integer-valued payloads must also
equal RCCL's all_reduce, and the error of random payloads against an fp64
reference stays within the rounding bound of the arithmetic.

``VLLM_RDNA_AR_HIER_TEST_ITERS`` (default 2) scales the random loops,
``VLLM_RDNA_AR_HIER_TEST_SOAK`` (default 600) the back-to-back soak
(closed-form integer payloads, device-side check, a GEMM on a side stream when
``VLLM_RDNA_AR_HIER_TEST_LOAD=1``), ``VLLM_RDNA_AR_HIER_TEST_MAX_KB`` (default
20480) the largest payload, ``VLLM_RDNA_AR_TEST_REQUIRE_RCCL=1`` turns an
unavailable RCCL group into a failure.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms.rdna import on_rdna2

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ALGO_FLAT, ALGO_FLAT_ISL, ALGO_HIER = 0, 1, 2
MIN_KB = 32  # hier threshold the test configures: below it the legacy flat one-shot
# Payload bytes: decode (M x 2560 x 2 B at M = 1..24), mixed, 1-20 MiB, plus
# lengths that are not a multiple of the 16-byte unit (added elements below).
BYTES = (
    4096,
    8192,
    15360,
    30720,
    61440,
    122880,
    262144,
    1 << 20,
    5 << 20,
    20 << 20,
)
RAGGED = (0, 1, 3, 7)


def _has_hier() -> bool:
    try:
        import vllm._custom_ops  # noqa: F401  (registers _rocm_C)

        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("rdna_ar_hier_init" in str(s) for s in schemas)


def _shards(world: int, n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """[world, n] payloads whose magnitudes make the accumulation order and
    both roundings matter."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(world, n, generator=g)
    x = x * torch.where(torch.rand(world, n, generator=g) < 0.02, 300.0, 1.0)
    return x.to(dtype)


def _flat_reference(shards: torch.Tensor) -> torch.Tensor:
    """The legacy flat sum: rank order, fp32, one rounding."""
    acc = torch.zeros(shards.shape[1], dtype=torch.float32)
    for r in range(shards.shape[0]):
        acc += shards[r].float()
    return acc.to(shards.dtype)


def _island_partials(shards: torch.Tensor, islands: list[int]) -> list[torch.Tensor]:
    """Each island's payloads summed in rank order in fp32, rounded to dtype."""
    parts = []
    for k in (0, 1):
        acc = torch.zeros(shards.shape[1], dtype=torch.float32)
        for r in range(shards.shape[0]):
            if islands[r] == k:
                acc += shards[r].float()
        parts.append(acc.to(shards.dtype))
    return parts


def _island_reference(shards: torch.Tensor, islands: list[int]) -> torch.Tensor:
    p0, p1 = _island_partials(shards, islands)
    return (p0.float() + p1.float()).to(shards.dtype)


def _same(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Bit-equal, with every NaN equal to every NaN (payloads may differ)."""
    got, want = got.cpu(), want.cpu()
    if got.dtype != want.dtype or got.shape != want.shape:
        return False
    nan = got.isnan()
    if not torch.equal(nan, want.isnan()):
        return False
    return torch.equal(got.masked_fill(nan, 0), want.masked_fill(nan, 0))


def _rccl_group(world: int):
    try:
        pg = dist.new_group(list(range(world)), backend="nccl")
        t = torch.ones(8, device="cuda")
        dist.all_reduce(t, group=pg)
        torch.accelerator.synchronize()
        assert float(t[0]) == float(world)
        return pg
    except Exception as e:  # RCCL not usable in this environment
        print(f"rccl group unavailable: {e!r}", flush=True)
        return None


def _configs(world: int) -> list[tuple[str, int, int, int]]:
    """(name, algo, owners, blocks) the exactness checks rotate through."""
    g = world // 2
    cfgs = [("flat-isl", ALGO_FLAT_ISL, 0, 0), ("hier", ALGO_HIER, 0, 0)]
    for x in sorted({g, max(g // 2, 1), 1}):
        for blocks in (0, 1, 3, 32):
            cfgs.append((f"hier-x{x}-b{blocks}", ALGO_HIER, x, blocks))
    return cfgs


def _worker(
    rank: int, world: int, port: int, cache_root: str, islands_spec: str
) -> None:
    max_kb = int(os.environ.get("VLLM_RDNA_AR_HIER_TEST_MAX_KB", "20480"))
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        VLLM_CACHE_ROOT=cache_root,
        VLLM_RDNA_AR_MAX_KB=str(max_kb),
        VLLM_RDNA_AR_ONESHOT_KB=str(max_kb),
        VLLM_RDNA_AR_HIER_MAX_KB=str(max_kb),
        VLLM_RDNA_AR_HIER="1",
        VLLM_RDNA_AR_HIER_ISLANDS=islands_spec,
        VLLM_RDNA_AR_WAIT_MS="30000",
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
    )

    comm = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not comm.disabled
    islands = comm.hier_islands
    assert islands is not None, "hier did not turn on (wrong topology?)"
    assert sorted(islands.count(k) for k in (0, 1)) == [world // 2] * 2
    assert islands[0] == 0
    ops = comm._ops
    iters = int(os.environ.get("VLLM_RDNA_AR_HIER_TEST_ITERS", "2"))
    soak = int(os.environ.get("VLLM_RDNA_AR_HIER_TEST_SOAK", "600"))
    island_t = torch.tensor(islands, dtype=torch.int64)

    def configure(
        owners: int, blocks: int, algo: int = -1, min_kb: int = MIN_KB
    ) -> None:
        dist.barrier()
        ops.rdna_ar_hier_init(
            comm.handle, island_t, owners, min_kb, algo, blocks, max_kb
        )
        dist.barrier()

    def run(x: torch.Tensor, algo: int, blocks: int = 0) -> torch.Tensor:
        return ops.rdna_ar_all_reduce_algo(comm.handle, x, algo, blocks)

    sizes = [
        (b // torch.empty((), dtype=dt).element_size() + extra, dt)
        for dt in DTYPES
        for b in BYTES
        for extra in (RAGGED if b <= (1 << 20) else RAGGED[:1] + RAGGED[-1:])
    ]
    sizes = [
        (n, dt)
        for n, dt in sizes
        if n * torch.empty((), dtype=dt).element_size() <= max_kb * 1024
    ]
    cfgs = _configs(world)
    pending: list[tuple[torch.Tensor, torch.Tensor, tuple]] = []
    checked = 0

    def flush() -> None:
        nonlocal checked
        torch.accelerator.synchronize()
        for got, ref, what in pending:
            assert _same(got, ref), what
            checked += 1
        pending.clear()

    # --- every algorithm / config against the island-grouped reference ------
    seed = 0
    cur_owners = -1
    for it in range(iters):
        for si, (n, dt) in enumerate(sizes):
            seed += 1
            if it > 0 and n * torch.empty((), dtype=dt).element_size() > (1 << 20):
                continue  # one pass of the multi-MiB payloads (CPU reference)
            shards = _shards(world, n, dt, seed)
            ref = _island_reference(shards, islands)
            # a 16-byte aligned tensor, or a view one element in
            if (it + si) % 3 == 2:
                base = torch.zeros(n + 8, dtype=dt, device="cuda")
                inp = base[1 : n + 1]
                inp.copy_(shards[rank])
            else:
                inp = shards[rank].cuda()
            assert comm.should_use(inp), (n, dt)
            # rotate the configs: all of them over the sizes, a few per size
            for ci in range(3):
                name, algo, owners, blocks = cfgs[(it * 7 + si * 3 + ci) % len(cfgs)]
                if owners != cur_owners:
                    flush()
                    configure(owners, 0)
                    cur_owners = owners
                pending.append((run(inp, algo, blocks), ref, (it, n, dt, name, rank)))
            flush()
            # the dispatch of the instance: the legacy flat one-shot (rank-order
            # sum) below the threshold, the 3-phase algorithm (island sums) above
            configure(0, 0)
            cur_owners = 0
            flat_ref = _flat_reference(shards)
            want = ref if n * inp.element_size() >= MIN_KB * 1024 else flat_ref
            pending.append((comm.all_reduce(inp), want, (it, n, dt, "dispatch", rank)))
            # the legacy flat algorithm keeps its numerics
            pending.append((run(inp, ALGO_FLAT), flat_ref, (it, n, dt, "flat")))
            flush()
    flush()
    dist.barrier()

    # --- adversarial payloads ------------------------------------------------
    configure(0, 0)
    for dt in DTYPES:
        n = 8192 + 5
        big = float(torch.finfo(dt).max) / 2
        pats: dict[str, torch.Tensor] = {}
        g = torch.Generator().manual_seed(77)
        # magnitudes that make the summation order visible
        mag = (torch.rand(world, n, generator=g) * 12 - 6).exp2()
        sign = torch.where(torch.rand(world, n, generator=g) < 0.5, -1.0, 1.0)
        pats["order"] = (mag * sign).to(dt)
        pats["zeros"] = torch.zeros(world, n, dtype=dt)
        pats["negzero"] = torch.full((world, n), -0.0, dtype=dt)
        pats["same"] = torch.full((world, n), 1.5, dtype=dt)
        tiny = torch.full((world, n), float(torch.finfo(dt).smallest_normal), dtype=dt)
        pats["subnormal"] = (tiny * 0.25) * sign.to(dt)
        # an island sums past the dtype's range while the total does not
        over = torch.full((world, n), big, dtype=dt)
        for r in range(world):
            over[r] = big if islands[r] == 0 else -big
        pats["island-overflow"] = over
        spec = pats["order"].clone()
        spec[:, 0::7] = float("nan")
        spec[0, 1::7] = float("inf")
        spec[1, 1::7] = float("-inf")
        spec[2, 2::7] = float("inf")
        spec[:, 3::7] = -0.0
        spec[:, 4::7] = 0.0
        spec[1, 5::7] = float("nan")
        pats["specials"] = spec
        for name, shards in pats.items():
            ref = _island_reference(shards, islands)
            inp = shards[rank].cuda()
            for algo, nb in ((ALGO_FLAT_ISL, 0), (ALGO_HIER, 0), (ALGO_HIER, 3)):
                got = run(inp, algo, nb)
                torch.accelerator.synchronize()
                assert _same(got, ref), (name, dt, algo, nb, rank)
    dist.barrier()

    # --- integer payloads equal RCCL; random payloads stay within the bound ---
    pg = _rccl_group(world)
    if pg is None:
        assert os.environ.get("VLLM_RDNA_AR_TEST_REQUIRE_RCCL") != "1", "no RCCL"
    else:
        for it in range(iters):
            for n, dt in sizes[:: max(1, len(sizes) // 24)]:
                seed += 1
                g = torch.Generator().manual_seed(seed)
                vals = torch.randint(-6, 7, (world, n), generator=g).to(dt)
                x = vals[rank].cuda()
                theirs = x.clone()
                dist.all_reduce(theirs, group=pg)
                for algo in (ALGO_FLAT_ISL, ALGO_HIER):
                    ours = run(x, algo)
                    torch.accelerator.synchronize()
                    assert torch.equal(ours, theirs), (it, n, dt, algo, rank)
        dist.barrier()
    errs: dict[str, list[float]] = {}
    for dt in (torch.float16, torch.bfloat16):
        eps = float(torch.finfo(dt).eps)
        for n in (30720, 1 << 20):
            seed += 1
            shards = _shards(world, n, dt, seed)
            exact = shards.double().sum(0)
            p0, p1 = (p.double() for p in _island_partials(shards, islands))
            fin = _island_reference(shards, islands).double()
            # one rounding of each island partial and of their sum
            slack = 2 * world * 2.0**-24 * shards.double().abs().sum(0)
            tiny = 3 * float(torch.finfo(dt).smallest_normal) * eps
            bound = (eps / 2) * (p0.abs() + p1.abs() + fin.abs()) * 1.01 + slack + tiny
            for algo in (ALGO_FLAT_ISL, ALGO_HIER):
                got = run(shards[rank].cuda(), algo).cpu().double()
                err = (got - exact).abs()
                assert bool((err <= bound).all()), (dt, n, algo, float(err.max()))
                errs.setdefault(f"{dt}-{algo}", []).append(float(err.max()))
            if pg is not None:
                theirs = shards[rank].cuda()
                dist.all_reduce(theirs, group=pg)
                errs.setdefault(f"{dt}-rccl", []).append(
                    float((theirs.cpu().double() - exact).abs().max())
                )
    dist.barrier()

    # --- CUDA graph: capture once, replay with new inputs --------------------
    for n, dt in (
        (7680, torch.float16),
        (61440, torch.float16),
        (30720, torch.bfloat16),
    ):
        for name, algo, owners, blocks in (
            ("flat-isl", ALGO_FLAT_ISL, 0, 0),
            ("hier", ALGO_HIER, 0, 0),
            ("hier-x1", ALGO_HIER, 1, 0),
            ("hier-x2-b3", ALGO_HIER, 2, 3),
        ):
            configure(owners, 0)
            g_in = torch.zeros(n, dtype=dt, device="cuda")
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):  # warm up outside the graph
                run(g_in, algo, blocks)
            torch.accelerator.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run(g_in, algo, blocks)
            for replay in range(max(10, iters)):
                seed += 1
                shards = _shards(world, n, dt, seed)
                g_in.copy_(shards[rank])
                graph.replay()
                torch.accelerator.synchronize()
                assert _same(out, _island_reference(shards, islands)), (
                    n,
                    dt,
                    name,
                    replay,
                )
            del graph
    dist.barrier()

    # --- soak: back-to-back, mixed algorithms and sizes, closed-form check ---
    configure(0, 0)
    side = torch.cuda.Stream()
    load = os.environ.get("VLLM_RDNA_AR_HIER_TEST_LOAD") == "1"
    if load:
        a = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
        c = torch.empty_like(a)
    plan = [
        (n, dt, algo)
        for n, dt in (
            (2048, torch.float16),
            (7683, torch.float16),
            (30720, torch.float16),
            (61440, torch.bfloat16),
            (123000, torch.float16),
            (65536, torch.float32),
            (300000, torch.float16),
        )
        for algo in (ALGO_FLAT_ISL, ALGO_HIER, -1, ALGO_FLAT)
    ]
    plan = [
        p
        for p in plan
        if p[0] * torch.empty((), dtype=p[1]).element_size() <= max_kb * 1024
    ]
    idx = torch.arange(max(p[0] for p in plan), device="cuda", dtype=torch.int64)
    bad = torch.zeros((), dtype=torch.int64, device="cuda")
    for it in range(soak):
        n, dt, algo = plan[it % len(plan)]
        mul, off = 5 + 2 * (it % 7), 3 + it % 5
        x = ((idx[:n] * mul + rank * off) % 17).to(dt)
        want = torch.zeros(n, dtype=torch.float32, device="cuda")
        for r in range(world):
            want += ((idx[:n] * mul + r * off) % 17).float()
        if load:
            with torch.cuda.stream(side):
                torch.mm(a, a, out=c)
        out = ops.rdna_ar_all_reduce_algo(comm.handle, x, algo, 0)
        bad += (out.float() != want).any().long()
        if it % 256 == 255:
            assert int(bad) == 0, (it, n, dt, algo, rank)
    torch.accelerator.synchronize()
    assert int(bad) == 0, (soak, rank)
    dist.barrier()

    assert int(ops.rdna_ar_timeout_info(comm.handle)) == 0  # no abort
    if rank == 0:
        print(
            f"rdna_ar hier world {world} islands {islands}: {checked * world} "
            f"exact results, {soak * world} soak results; max abs error vs fp64 "
            f"(fp16/bf16 flat-isl, hier, rccl): {errs}",
            flush=True,
        )
    dist.destroy_process_group()


@pytest.mark.skipif(not _has_hier(), reason="rdna_ar hier all-reduce not built")
@pytest.mark.parametrize(
    ("world", "port", "islands"),
    [
        (4, 29551, "0,1|2,3"),
        (4, 29553, "0,2|1,3"),
        (8, 29555, ""),
        (8, 29557, "0,2,4,6|1,3,5,7"),
        (6, 29559, "0,1,2|3,4,5"),
    ],
)
def test_rdna_ar_hier_is_exact(tmp_path, world, port, islands):
    if not (
        torch.cuda.is_available()
        and torch.accelerator.device_count() >= world
        and on_rdna2()
    ):
        pytest.skip(f"needs {world} gfx10x GPUs")
    mp.spawn(
        _worker, args=(world, port, str(tmp_path), islands), nprocs=world, join=True
    )
