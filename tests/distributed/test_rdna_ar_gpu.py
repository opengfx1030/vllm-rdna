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
