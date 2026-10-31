# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow packed-page transport; these tests do not claim live request paging."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def packed_caches(tensor, *, groups=2):
    page_bytes = tensor.shape[1] * tensor.element_size()
    return CanonicalKVCaches(
        [CanonicalKVCacheTensor(tensor, page_bytes)],
        [[CanonicalKVCacheRef(0, page_bytes)] for _ in range(groups)],
    )


@pytest.fixture
def mock_handler_worker(monkeypatch):
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", False)
    monkeypatch.setattr(gpu_worker, "SingleDirectionOffloadingHandler", MagicMock())
    return CPUOffloadingWorker(
        packed_caches(torch.zeros(8, 32, dtype=torch.int8)), 1, 2
    )


def test_physical_page_wrappers_use_one_full_group(mock_handler_worker):
    worker = mock_handler_worker
    assert worker.supports_physical_page_snapshots
    spec = CPULoadStoreSpec([1, 0])
    worker.submit_physical_page_store(1, [2, 6], spec)
    args = worker._store_handler.transfer_async.call_args.args
    assert args[0] == 1
    assert args[1].block_ids.tolist() == [2, 6]
    assert args[1].group_sizes == [2, 0]
    assert args[1].block_indices == [0, 0]
    assert args[2] is spec
    worker.submit_physical_page_load(2, spec, [1, 5])
    args = worker._load_handler.transfer_async.call_args.args
    assert args[1] is spec
    assert args[2].block_ids.tolist() == [1, 5]


@pytest.mark.parametrize("pages", [[], [0], [-1], [8], [2, 2], [1.5], [True]])
def test_physical_page_rejects_invalid_gpu_ids(mock_handler_worker, pages):
    with pytest.raises(ValueError):
        mock_handler_worker.submit_physical_page_store(1, pages, CPULoadStoreSpec([0]))
    mock_handler_worker._store_handler.transfer_async.assert_not_called()


@pytest.mark.parametrize("chunks", [[], [0], [0, 0], [-1, 0], [0, 2]])
def test_physical_page_rejects_invalid_cpu_ids(mock_handler_worker, chunks):
    with pytest.raises(ValueError):
        mock_handler_worker.submit_physical_page_load(
            1, CPULoadStoreSpec(chunks), [2, 6]
        )
    mock_handler_worker._load_handler.transfer_async.assert_not_called()


@pytest.mark.parametrize(
    "layout", ["chunked", "fragment", "multiple", "strided", "mapped", "empty"]
)
def test_physical_page_capability_rejects_unsupported_layout(monkeypatch, layout):
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", False)
    monkeypatch.setattr(gpu_worker, "SingleDirectionOffloadingHandler", MagicMock())
    caches = packed_caches(torch.zeros(8, 32, dtype=torch.int8))
    chunks = 1
    if layout == "chunked":
        chunks = 2
    elif layout == "fragment":
        caches.group_data_refs[1][0].page_size_bytes = 16
    elif layout == "multiple":
        caches.tensors *= 2
    elif layout == "mapped":
        caches.group_data_refs[0][0].mapping = MagicMock()
    elif layout == "empty":
        caches.group_data_refs = []
    else:
        caches = packed_caches(torch.zeros(16, 32, dtype=torch.int8)[::2])
    worker = CPUOffloadingWorker(caches, chunks, 2)
    assert not worker.supports_physical_page_snapshots
    with pytest.raises(ValueError, match="full packed"):
        worker.submit_physical_page_store(1, [1], CPULoadStoreSpec([0]))


@pytest.mark.parametrize("canonical_layout", [False, True])
def test_physical_page_capability_rejects_shared_region(monkeypatch, canonical_layout):
    monkeypatch.setattr(gpu_worker, "PIN_MEMORY", False)
    monkeypatch.setattr(gpu_worker, "SingleDirectionOffloadingHandler", MagicMock())
    monkeypatch.setattr(gpu_worker, "_canonical_block_sizes", lambda *args: [32])
    region = MagicMock()
    cpu = torch.zeros(2, 32, dtype=torch.int8)
    region.create_next_worker_view.return_value = cpu
    region.create_next_canonical_view.return_value = cpu
    worker = CPUOffloadingWorker(
        packed_caches(torch.zeros(8, 32, dtype=torch.int8)),
        1,
        2,
        mmap_region=region,
        canonical_layout=canonical_layout,
    )
    assert not worker.supports_physical_page_snapshots
    with pytest.raises(ValueError, match="full packed"):
        worker.submit_physical_page_load(1, CPULoadStoreSpec([0]), [1])


def test_physical_page_submission_failure_preserves_result(mock_handler_worker):
    worker = mock_handler_worker
    worker._store_handler.transfer_async.return_value = False
    worker._load_handler.transfer_async.return_value = False
    spec = CPULoadStoreSpec([0])
    assert not worker.submit_physical_page_store(1, [1], spec)
    assert not worker.submit_physical_page_load(2, spec, [2])


@pytest.fixture
def physical_config():
    # Keep this qualification test runnable with --noconftest as well as the
    # normal suite. Import lazily so CPU contract checks need no config setup.
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HIP device")
@pytest.mark.parametrize("policy", ["lru", "arc"])
@torch.inference_mode()
def test_physical_page_allocator_backed_gpu_roundtrip(physical_config, policy):
    # Each key is private to this request/snapshot generation, not a prefix hash.
    ctx = ReqContext(req_id="physical-roundtrip")
    keys = [
        make_offload_key(f"snapshot:engine1:req1:attempt1:{i}".encode(), 0)
        for i in range(2)
    ]
    manager = CPUOffloadingManager(num_chunks=2, cache_policy=policy)
    tensor = torch.zeros((8, 1024), dtype=torch.int8, device="cuda")
    tensor[2].fill_(37)
    tensor[6].fill_(-91)
    expected = tensor[[2, 6]].clone()
    worker = CPUOffloadingWorker(packed_caches(tensor), 1, 2)
    lease = None
    try:
        output = manager.prepare_store(keys, ctx)
        assert output is not None and output.keys_to_store == keys
        assert worker.submit_physical_page_store(1, [2, 6], output.store_spec)
        worker.wait({1})
        results = worker.get_finished()
        assert len(results) == 1 and results[0].success and results[0].job_id == 1
        assert results[0].transfer_size == 2048
        lease = manager.complete_store_and_retain(keys, ctx)
        tensor.fill_(11)
        pressure_key = make_offload_key(b"other-private-snapshot", 0)
        assert manager.prepare_store([pressure_key], ctx) is None
        load_spec = manager.prepare_load(keys, ctx)
        assert worker.submit_physical_page_load(2, load_spec, [1, 5])
        worker.wait({2})
        results = worker.get_finished()
        assert len(results) == 1 and results[0].success and results[0].job_id == 2
        assert results[0].transfer_size == 2048
        manager.complete_load(keys, ctx)
        assert torch.equal(tensor[[1, 5]], expected)
        assert torch.all(tensor[[2, 6]] == 11)
        assert manager.prepare_store([pressure_key], ctx) is None
        assert manager.release_lease(lease)
        lease = None
        assert manager.prepare_store([pressure_key], ctx) is not None
    finally:
        worker.shutdown()
        if lease is not None:
            manager.release_lease(lease)
