# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU exactness of the 16-byte push of the gfx10x one-shot all-reduce.

``VLLM_RDNA_AR_VPUSH=1`` makes the one-shot kernel push its payload to the
peers with 16-byte stores instead of 2-byte ones (large payloads are bound by
the PCIe write size). The finisher and the staging protocol are unchanged, so
the result must stay bit-equal to the fixed-order fp32 sum (rank 0 .. W-1, cast
to the tensor dtype), and equal to the scalar-push instance, for payloads up to
the TP2 prefill all-reduce (2048 x 5120 fp16 = 20 MiB): 16-byte multiples, odd
lengths (the scalar tail of the push), and views that are not 16-byte aligned
(the scalar push). The gate is raised to 32 MiB, so the staging slots are 32 MiB.

    pytest -s tests/distributed/test_rdna_ar_vpush_gpu.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms.rdna import on_rdna2

GATE_KB = 32768
# (elements, dtype): the 16-byte multiples and odd tails of every dtype, then
# the sizes that matter: 1 MiB, 10 MiB (a 1024-token chunk) and 20 MiB (2048).
CASES = (
    (1, torch.float16),
    (9, torch.float16),
    (4099, torch.float16),
    (65537, torch.float16),
    (524288, torch.float16),
    (2621441, torch.float16),
    (5242880, torch.float16),
    (10485760, torch.float16),
    (1048577, torch.bfloat16),
    (1048576, torch.float32),
    (262147, torch.float32),
)


def _has_rdna_ar() -> bool:
    try:
        import vllm._custom_ops  # noqa: F401  (registers _rocm_C)

        schemas = torch._C._jit_get_all_schemas()
    except Exception:
        return False
    return any("rdna_ar_all_reduce" in str(s) for s in schemas)


def _shards(world: int, n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(world, n, generator=g)
    x = x * torch.where(torch.rand(world, n, generator=g) < 0.02, 300.0, 1.0)
    return x.to(dtype)


def _reference(shards: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(shards.shape[1], dtype=torch.float32)
    for r in range(shards.shape[0]):
        acc += shards[r].float()
    return acc.to(shards.dtype)


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

    # The flag is read when an instance is created.
    os.environ["VLLM_RDNA_AR_VPUSH"] = "0"
    scalar = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    os.environ["VLLM_RDNA_AR_VPUSH"] = "1"
    vec = RdnaOneShotAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not scalar.disabled and not vec.disabled

    seed = 0
    checked = 0
    for it in range(2):
        for n, dt in CASES:
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
            assert scalar.should_use(inp) and vec.should_use(inp), (n, dt)
            got_s = scalar.all_reduce(inp)
            got_v = vec.all_reduce(inp)
            torch.accelerator.synchronize()
            assert torch.equal(got_s.cpu(), ref), ("scalar", it, n, dt, rank)
            assert torch.equal(got_v.cpu(), ref), ("vpush", it, n, dt, rank)
            checked += 1
    assert int(vec._ops.rdna_ar_timeout_info(vec.handle)) == 0  # no abort
    if rank == 0:
        print(f"rdna_ar vpush: {checked * world} exact results", flush=True)
    dist.destroy_process_group()


@pytest.mark.skipif(not _has_rdna_ar(), reason="rdna_ar all-reduce not built")
def test_rdna_ar_vpush_is_exact(tmp_path):
    if not (
        torch.cuda.is_available()
        and torch.accelerator.device_count() >= 2
        and on_rdna2()
    ):
        pytest.skip("needs 2 gfx10x GPUs")
    mp.spawn(_worker, args=(2, 29553, str(tmp_path)), nprocs=2, join=True)
