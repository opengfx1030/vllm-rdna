# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity of the gfx10x one-shot all-gather and vocab argmax (VLLM_RDNA_AR).

The all-gather must be bit-equal to ``all_gather_into_tensor`` on the last dim.
The argmax must equal ``torch.argmax`` over the gathered per-rank values:
ties go to the lowest rank and NaN wins. Both run inside CUDA graphs on the
decode path, so replay with new inputs is checked too.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms.rdna import on_rdna2

VOCAB_SHARD = 1024


def _has_rdna_ar() -> bool:
    try:
        import vllm._custom_ops  # noqa: F401  (registers _rocm_C)

        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("rdna_ar_argmax" in str(s) for s in schemas)


def _shards(world: int, shape: tuple[int, ...], dtype: torch.dtype, seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g).to(dtype) for _ in range(world)]


def _argmax_case(world: int, rows: int, dtype: torch.dtype, seed: int):
    """Per-rank (values, global indices) and the reference winner per row."""
    g = torch.Generator().manual_seed(seed)
    vals = (torch.rand(world, rows, generator=g) * 0.5).to(dtype)
    idx = torch.arange(world)[:, None] * VOCAB_SHARD + torch.randint(
        1, VOCAB_SHARD, (world, rows), generator=g
    )
    if rows >= 3:
        vals[:, 1] = 0.9  # exact tie on every rank -> lowest rank's index
        vals[world // 2, 2] = float("nan")  # NaN wins, as in torch.argmax
    winners = vals.float().T.argmax(dim=-1)  # [rows] over ranks, first max
    expected = idx.T.gather(1, winners[:, None])[:, 0]
    return vals, idx, expected


def _worker(rank: int, world: int, port: int, cache_root: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), VLLM_CACHE_ROOT=cache_root
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
    )

    comm = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not comm.disabled

    # --- all-gather: bit-equal to the reference layout, outputs stay valid ---
    kept = []
    step = 0
    for shape in ((1, 64), (3, 97), (8, 640), (5, 128), (1, 2), (4, 2048)):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            step += 1
            shards = _shards(world, shape, dtype, step)
            inp = shards[rank].cuda()
            assert comm.should_all_gather(inp), (shape, dtype)
            kept.append((comm.all_gather(inp), torch.cat(shards, dim=-1)))
    torch.accelerator.synchronize()
    for got, ref in kept:
        assert got.shape == ref.shape
        assert torch.equal(got.cpu(), ref)

    # --- argmax: lowest-rank ties, NaN wins, every dtype ---
    kept_am = []
    for rows in (1, 3, 8, 64, 1024):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            step += 1
            vals, idx, expected = _argmax_case(world, rows, dtype, step)
            v, i = vals[rank].cuda(), idx[rank].cuda()
            assert comm.should_argmax(v, i), (rows, dtype)
            kept_am.append((comm.argmax(v, i), expected))
    torch.accelerator.synchronize()
    for got, ref in kept_am:
        assert got.dtype == torch.int64
        assert torch.equal(got.cpu(), ref)

    # --- eligibility edges: fall back instead of mishandling ---
    big = torch.zeros(64 * 1024, dtype=torch.float16, device="cuda")
    assert not comm.should_all_gather(big)
    assert not comm.should_all_gather(torch.zeros(4, 8, device="cuda").t())
    assert not comm.should_all_gather(torch.zeros(0, 8, device="cuda"))
    v = torch.zeros(8, dtype=torch.float16, device="cuda")
    assert not comm.should_argmax(v, torch.zeros(8, dtype=torch.int32, device="cuda"))
    assert not comm.should_argmax(v, torch.zeros(7, dtype=torch.int64, device="cuda"))
    assert not comm.should_argmax(
        torch.zeros(9000, dtype=torch.float16, device="cuda"),
        torch.zeros(9000, dtype=torch.int64, device="cuda"),
    )  # 9000 * 8 B > 64 KiB one-shot gate

    # --- CUDA graph: capture once, replay with new inputs ---
    g_vals = torch.zeros(8, dtype=torch.float16, device="cuda")
    g_idx = torch.zeros(8, dtype=torch.int64, device="cuda")
    g_in = torch.zeros(3, 97, dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # warm up outside the graph
        comm.argmax(g_vals, g_idx)
        comm.all_gather(g_in)
    torch.accelerator.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_am = comm.argmax(g_vals, g_idx)
        out_ag = comm.all_gather(g_in)
    for replay in range(3):
        vals, idx, expected = _argmax_case(world, 8, torch.float16, 900 + replay)
        shards = _shards(world, (3, 97), torch.float16, 950 + replay)
        g_vals.copy_(vals[rank])
        g_idx.copy_(idx[rank])
        g_in.copy_(shards[rank])
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out_am.cpu(), expected)
        assert torch.equal(out_ag.cpu(), torch.cat(shards, dim=-1))

    assert int(comm._ops.rdna_ar_timeout_info(comm.handle)) == 0  # no abort
    dist.destroy_process_group()


@pytest.mark.skipif(not _has_rdna_ar(), reason="rdna_ar gather/argmax not built")
@pytest.mark.parametrize(("world", "port"), [(2, 29531), (4, 29533)])
def test_rdna_ar_gather_argmax_match_reference(tmp_path, world, port):
    if not (
        torch.cuda.is_available()
        and torch.accelerator.device_count() >= world
        and on_rdna2()
    ):
        pytest.skip(f"needs {world} gfx10x GPUs")
    mp.spawn(_worker, args=(world, port, str(tmp_path)), nprocs=world, join=True)
