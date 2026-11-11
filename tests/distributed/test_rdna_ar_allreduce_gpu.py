# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU exactness of the gfx10x one-shot all-reduce (VLLM_RDNA_AR).

The result must be bit-equal to the fixed-order fp32 sum (rank 0 .. W-1,
cast to the tensor dtype) on every rank, for every payload up to the one-shot
gate: 16-byte multiples (vector finisher), odd lengths (scalar tail), and
views that are not 16-byte aligned (scalar finisher). Integer-valued payloads
must also equal RCCL's all_reduce. Collectives are issued back to back (no
host sync between them) so the staging parity and sequence handshake are
stressed, then once more inside a CUDA graph with new inputs per replay.

``VLLM_RDNA_AR_TEST_ITERS`` (default 8) scales the random loops,
``VLLM_RDNA_AR_TEST_RCCL_ITERS`` (default 3) the RCCL comparison, and
``VLLM_RDNA_AR_TEST_REQUIRE_RCCL=1`` turns an unavailable RCCL group into a
failure instead of a skipped comparison.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms.rdna import on_rdna2

GATE_KB = 256
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Element counts: ones that are not a multiple of the 16-byte vector, the
# decode capture sizes (M = 3/6/12/24 tokens x 2560 hidden), and the gate.
SIZES = (
    1,
    3,
    7,
    8,
    9,
    15,
    16,
    17,
    100,
    2560,
    5120,
    7680,
    7681,
    15360,
    30720,
    39424,
    61440,
    65536,
    98304,
    131071,
    131072,
)
BATCH = 48


def _has_rdna_ar() -> bool:
    try:
        import vllm._custom_ops  # noqa: F401  (registers _rocm_C)

        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("rdna_ar_all_reduce" in str(s) for s in schemas)


def _shards(world: int, n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """[world, n] payloads whose magnitudes make both the fp32 accumulation
    order and the final rounding to dtype matter."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(world, n, generator=g)
    x = x * torch.where(torch.rand(world, n, generator=g) < 0.02, 300.0, 1.0)
    return x.to(dtype)


def _reference(shards: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(shards.shape[1], dtype=torch.float32)
    for r in range(shards.shape[0]):
        acc += shards[r].float()
    return acc.to(shards.dtype)


def _fits(n: int, dtype: torch.dtype) -> bool:
    return n * torch.empty((), dtype=dtype).element_size() <= GATE_KB * 1024


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


def _worker(rank: int, world: int, port: int, cache_root: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        VLLM_CACHE_ROOT=cache_root,
        VLLM_RDNA_AR_MAX_KB=str(GATE_KB),
        VLLM_RDNA_AR_ONESHOT_KB=str(GATE_KB),
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
    )

    comm = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not comm.disabled
    iters = int(os.environ.get("VLLM_RDNA_AR_TEST_ITERS", "8"))
    rccl_iters = int(os.environ.get("VLLM_RDNA_AR_TEST_RCCL_ITERS", "3"))

    # --- bit-equal to the fixed-order fp32 reference, back-to-back --------
    cases = [(n, dt) for dt in DTYPES for n in SIZES if _fits(n, dt)]
    pending: list[tuple[torch.Tensor, torch.Tensor, tuple]] = []
    checked = 0

    def flush() -> None:
        nonlocal checked
        torch.accelerator.synchronize()
        for got, ref, what in pending:
            assert got.dtype == ref.dtype and got.shape == ref.shape, what
            assert torch.equal(got.cpu(), ref), what
            checked += 1
        pending.clear()

    seed = 0
    for it in range(iters):
        for n, dt in cases:
            seed += 1
            shards = _shards(world, n, dt, seed)
            ref = _reference(shards)
            # Alternate a fresh 16-byte aligned tensor with a view that starts
            # one element in (misaligned for every dtype here).
            if (it + n) % 3 == 2:
                base = torch.zeros(n + 8, dtype=dt, device="cuda")
                inp = base[1 : n + 1]
                inp.copy_(shards[rank])
            else:
                inp = shards[rank].cuda()
            assert comm.should_use(inp), (n, dt)
            pending.append((comm.all_reduce(inp), ref, (it, n, dt, rank)))
            if len(pending) >= BATCH:
                flush()
    flush()
    dist.barrier()

    # --- integer-valued payloads equal RCCL's all_reduce --------------------
    pg = _rccl_group(world)
    if pg is None:
        assert os.environ.get("VLLM_RDNA_AR_TEST_REQUIRE_RCCL") != "1", "no RCCL"
    else:
        for it in range(rccl_iters):
            for n, dt in cases:
                seed += 1
                g = torch.Generator().manual_seed(seed)
                vals = torch.randint(-6, 7, (world, n), generator=g).to(dt)
                x = vals[rank].cuda()
                ours = comm.all_reduce(x)
                theirs = x.clone()
                dist.all_reduce(theirs, group=pg)
                torch.accelerator.synchronize()
                assert torch.equal(ours, theirs), (it, n, dt, rank)
                assert torch.equal(ours.cpu(), _reference(vals)), (it, n, dt)
        dist.barrier()

    # --- soak: many back-to-back collectives against a device-side reference
    # (RCCL all-gather of the inputs is a pure copy, summed in rank order in
    # fp32 on the GPU, so no host round trip per collective).
    soak = int(os.environ.get("VLLM_RDNA_AR_TEST_SOAK", "0"))
    if soak and pg is not None:
        gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        for it in range(soak):
            n, dt = cases[it % len(cases)]
            x = torch.randn(n, device="cuda", generator=gen)
            big = torch.rand(n, device="cuda", generator=gen) < 0.02
            x = (x * torch.where(big, 300.0, 1.0)).to(dt)
            ours = comm.all_reduce(x)
            gathered = torch.empty(world * n, dtype=dt, device="cuda")
            dist.all_gather_into_tensor(gathered, x, group=pg)
            acc = torch.zeros(n, dtype=torch.float32, device="cuda")
            for r in range(world):
                acc += gathered[r * n : (r + 1) * n].float()
            bad += (ours != acc.to(dt)).any().long()
            if it % 1024 == 1023:
                assert int(bad) == 0, (it, n, dt, rank)
        assert int(bad) == 0, (soak, rank)
        dist.barrier()

    # --- CUDA graph: capture once, replay with new inputs -------------------
    for n, dt in (
        (7680, torch.float16),
        (61440, torch.float16),
        (30720, torch.bfloat16),
    ):
        g_in = torch.zeros(n, dtype=dt, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):  # warm up outside the graph
            comm.all_reduce(g_in)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = comm.all_reduce(g_in)
        for replay in range(max(10, iters)):
            seed += 1
            shards = _shards(world, n, dt, seed)
            g_in.copy_(shards[rank])
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out.cpu(), _reference(shards)), (n, dt, replay)

    assert int(comm._ops.rdna_ar_timeout_info(comm.handle)) == 0  # no abort
    if rank == 0:
        print(
            f"rdna_ar all_reduce: {checked * world} exact results"
            f" + {soak * world} soak results",
            flush=True,
        )
    dist.destroy_process_group()


@pytest.mark.skipif(not _has_rdna_ar(), reason="rdna_ar all-reduce not built")
@pytest.mark.parametrize(("world", "port"), [(2, 29541), (4, 29543)])
def test_rdna_ar_allreduce_is_exact(tmp_path, world, port):
    if not (
        torch.cuda.is_available()
        and torch.accelerator.device_count() >= world
        and on_rdna2()
    ):
        pytest.skip(f"needs {world} gfx10x GPUs")
    mp.spawn(_worker, args=(world, port, str(tmp_path)), nprocs=world, join=True)
