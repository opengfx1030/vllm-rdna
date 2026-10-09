# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""2-GPU test of the gfx10x one-shot all-reduce kernel (VLLM_RDNA_AR).

A result must stay valid after later collectives: models keep all-reduce
outputs alive (the embedding all-reduce output becomes the decoder residual),
so an output buffer shared across calls corrupts them.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms.rdna import on_rdna2


def _has_rdna_ar() -> bool:
    try:
        import vllm._custom_ops  # noqa: F401  (registers _rocm_C)

        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("rdna_ar_init" in str(s) for s in schemas)


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
    expect = []
    kept = []
    for step, numel in enumerate((12 * 2048, 2048, 12 * 2048, 8)):
        g = torch.Generator().manual_seed(step)
        full = torch.randn(world, numel, generator=g).half()
        inp = full[rank].cuda()
        assert comm.should_use(inp)
        kept.append(comm.all_reduce(inp))
        expect.append(full.float().sum(0))
    torch.accelerator.synchronize()
    for out, ref in zip(kept, expect):
        torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-3)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.accelerator.device_count() >= 2
        and on_rdna2()
    ),
    reason="needs 2 gfx10x GPUs",
)
@pytest.mark.skipif(not _has_rdna_ar(), reason="rdna_ar op not built")
def test_rdna_ar_outputs_survive_later_calls(tmp_path):
    mp.spawn(_worker, args=(2, 29517, str(tmp_path)), nprocs=2, join=True)


def _late_peer_worker(
    rank: int, world: int, port: int, cache_root: str, late_s: float, wait_ms: str
) -> None:
    """Rank 1 reaches the all-reduce ``late_s`` seconds after rank 0, as when
    one rank JIT-compiles a kernel on the first request."""
    import time

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), VLLM_CACHE_ROOT=cache_root
    )
    if wait_ms:
        os.environ["VLLM_RDNA_AR_WAIT_MS"] = wait_ms
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    import vllm._custom_ops  # noqa: F401  (registers _rocm_C)
    from vllm.distributed.device_communicators.rdna_all_reduce import (
        RdnaOneShotAllReduce,
        describe_abort,
    )

    comm = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not comm.disabled
    inp = torch.full((4096,), float(rank + 1), dtype=torch.float16, device="cuda")
    comm.all_reduce(inp)  # warm, both on time
    torch.accelerator.synchronize()
    dist.barrier()
    if rank == 1:
        time.sleep(late_s)
    out = comm.all_reduce(inp)
    torch.accelerator.synchronize()
    code = int(comm._ops.rdna_ar_timeout_info(comm.handle))
    result = (code, bool((out == world * (world + 1) / 2).all()))
    gathered: list = [None] * world
    dist.all_gather_object(gathered, result)
    if wait_ms:
        # A wait past the configured bound still aborts with a clear record
        # carrying the measured wall-clock wait (16 ms units).
        assert gathered[0][0] != 0, gathered
        msg = describe_abort(gathered[0][0], 0)
        assert "peer rank 1's flag never arrived" in msg
        waited = int(msg.split("~")[1].split(" ms")[0])
        assert int(wait_ms) - 16 <= waited <= int(wait_ms) + 200, msg
    else:
        assert all(c == 0 and ok for c, ok in gathered), [
            (describe_abort(c, r) if c else "ok", ok)
            for r, (c, ok) in enumerate(gathered)
        ]
        # The on-time rank recorded the late peer with the real wait (~late_s).
        if rank == 0:
            from vllm.distributed.device_communicators.rdna_all_reduce import (
                describe_late,
            )

            late = describe_late(int(comm._ops.rdna_ar_slow_info(comm.handle)), 0)
            waited = int(late.split("~")[1].split(" ms")[0])
            assert late_s * 1000 - 500 <= waited <= late_s * 1000 + 1500, late
    dist.destroy_process_group()


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.accelerator.device_count() >= 2
        and on_rdna2()
    ),
    reason="needs 2 gfx10x GPUs",
)
@pytest.mark.skipif(not _has_rdna_ar(), reason="rdna_ar op not built")
@pytest.mark.parametrize("late_s,wait_ms", [(5.0, ""), (2.0, "300")])
def test_rdna_ar_late_peer(tmp_path, late_s, wait_ms):
    """A peer that is legitimately seconds late must not wedge the collective
    (default bound); an explicit small VLLM_RDNA_AR_WAIT_MS still aborts."""
    mp.spawn(
        _late_peer_worker,
        args=(2, 29519, str(tmp_path), late_s, wait_ms),
        nprocs=2,
        join=True,
    )
