# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.worker import worker_base
from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase

pytestmark = pytest.mark.skip_global_cleanup


class ExtensionA:
    events: list[str]

    def init_worker_extension(self):
        self.events.append("init-a")

    def destroy_worker_extension(self):
        self.events.append("destroy-a")


class ExtensionB: ...


class FailingExtension:
    def destroy_worker_extension(self):
        raise RuntimeError("destroy failed")


@pytest.fixture
def make_wrapper(monkeypatch):
    # Fresh worker class per test, since extension injection mutates its bases.
    class Worker(WorkerBase):
        def __init__(self, **_kwargs):
            self.events: list[str] = []

        def init_device(self):
            self.events.append("device")

        def shutdown(self):
            self.events.append("shutdown")

    classes = {
        cls.__name__: cls for cls in (Worker, ExtensionA, ExtensionB, FailingExtension)
    }
    monkeypatch.setattr(worker_base, "resolve_obj_by_qualname", classes.__getitem__)
    monkeypatch.setattr(
        worker_base,
        "worker_receiver_cache_from_config",
        lambda *_args, **_kwargs: None,
    )

    def make(extension: str) -> WorkerWrapperBase:
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                worker_cls="Worker", worker_extension_cls=extension
            ),
            enable_trace_function_call_for_thread=lambda: None,
        )
        wrapper = WorkerWrapperBase()
        wrapper.init_worker([{"vllm_config": config}])
        wrapper.init_device()
        return wrapper

    return make


def test_worker_extension_hooks_follow_configured_class(make_wrapper):
    for extension, expected in (
        ("ExtensionA", ["device", "init-a", "destroy-a", "shutdown"]),
        ("ExtensionB", ["device", "shutdown"]),
        ("ExtensionA", ["device", "init-a", "destroy-a", "shutdown"]),
    ):
        wrapper = make_wrapper(extension)
        wrapper.shutdown()
        assert wrapper.worker.events == expected


def test_failing_destroy_hook_is_logged_not_raised(make_wrapper, caplog_vllm):
    wrapper = make_wrapper("FailingExtension")
    wrapper.shutdown()
    assert wrapper.worker.events == ["device", "shutdown"]
    assert "Worker extension destroy hook failed" in caplog_vllm.text


def test_triton_cache_artifacts_do_not_clobber_other_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    files = []
    for rank in (5, 6):
        worker_base._configure_worker_triton_cache(rank)
        directory = Path(os.environ["TRITON_CACHE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "same_kernel.amdgcn"
        pending = directory / "pending"
        pending.write_text(f"assembly for rank {rank}")
        pending.replace(artifact)
        files.append(artifact)

    assert files[0].read_text() == "assembly for rank 5"
    assert files[1].read_text() == "assembly for rank 6"


def test_triton_worker_cache_reuses_artifacts_and_respects_new_root(
    tmp_path, monkeypatch
):
    first_root = tmp_path / "first"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(first_root))
    worker_base._configure_worker_triton_cache(5)
    directory = Path(os.environ["TRITON_CACHE_DIR"])
    directory.mkdir(parents=True)
    artifact = directory / "kernel.hsaco"
    artifact.write_bytes(b"compiled target/draft kernel")

    worker_base._configure_worker_triton_cache(5)
    assert (
        Path(os.environ["TRITON_CACHE_DIR"]) / artifact.name
    ).read_bytes() == b"compiled target/draft kernel"

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "second"))
    worker_base._configure_worker_triton_cache(5)
    assert not (Path(os.environ["TRITON_CACHE_DIR"]) / artifact.name).exists()
    assert artifact.read_bytes() == b"compiled target/draft kernel"
