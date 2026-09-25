# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay of CUDAGraphWrapper must be input-dependent.

Eager-backend piecewise graphs on gfx1030 replayed capture-time activations
(greedy decode collapsed to the warmup token "duct") because replay ignored
runtime tensors whose data_ptr differed from capture. These tests drive the
shipped CUDAGraphWrapper.copy_runtime_inputs_into_static helper and, when a
GPU is present, the wrapper capture/replay path itself.
"""

import pytest
import torch
import torch.nn as nn

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform


def test_eager_piecewise_splits_tp_collectives():
    """RCCL all_reduce must not sit inside a captured eager GraphModule."""
    cfg = CompilationConfig(
        backend="eager",
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        mode=CompilationMode.VLLM_COMPILE,
    )
    cfg.set_splitting_ops_for_v1(all2all_backend="allgather_reducescatter")
    assert "vllm::tensor_model_parallel_all_reduce" in (cfg.splitting_ops or [])
    assert "vllm::unified_attention_with_output" in (cfg.splitting_ops or [])


def test_rocm_full_executes_as_piecewise():
    """HIP FULL decode executes piecewise graphs; NVIDIA FULL does not."""
    from vllm.v1.worker.gpu.cudagraph_utils import rocm_full_executes_as_piecewise

    assert rocm_full_executes_as_piecewise(CUDAGraphMode.PIECEWISE) is False
    if current_platform.is_rocm():
        assert rocm_full_executes_as_piecewise(CUDAGraphMode.FULL) is True
    else:
        assert rocm_full_executes_as_piecewise(CUDAGraphMode.FULL) is False


def test_rocm_inductor_fpp_splits_tp_collectives():
    """ROCm inductor FULL_AND_PIECEWISE must also split TP collectives."""
    cfg = CompilationConfig(
        backend="inductor",
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        mode=CompilationMode.VLLM_COMPILE,
    )
    cfg.set_splitting_ops_for_v1(all2all_backend="allgather_reducescatter")
    if current_platform.is_rocm():
        assert "vllm::tensor_model_parallel_all_reduce" in (cfg.splitting_ops or [])
    assert "vllm::unified_attention_with_output" in (cfg.splitting_ops or [])


def test_should_copy_and_wrap_eager_piecewise_graphmodules():
    """backend=eager + PIECEWISE must wrap GraphModules (not skip-wrap)."""
    from vllm.compilation.backends import (
        should_copy_cudagraph_inputs,
        wrap_with_cudagraph_if_needed,
    )
    from vllm.compilation.piecewise_backend import RangeEntry
    from vllm.config.utils import Range

    cfg = CompilationConfig(
        backend="eager",
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )
    assert should_copy_cudagraph_inputs(cfg)

    class FakeGM(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class FakePW:
        def __init__(self) -> None:
            self.range_entries = {
                Range(start=4, end=4): RangeEntry(
                    compile_range=Range(start=4, end=4),
                    compiled=True,
                    runnable=FakeGM(),
                )
            }

    vllm_config = VllmConfig(compilation_config=cfg)
    pw = FakePW()
    wrapped = wrap_with_cudagraph_if_needed(
        pw, vllm_config, cfg, is_first_graph=True, is_last_graph=True
    )
    assert wrapped is pw
    runnable = next(iter(pw.range_entries.values())).runnable
    assert isinstance(runnable, CUDAGraphWrapper)
    assert runnable.runnable.__class__ is FakeGM


def test_copy_runtime_inputs_into_static_is_input_dependent():
    static = [torch.zeros(2, 4), torch.ones(3)]
    first = [
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.tensor([1.0, 2.0, 3.0]),
    ]
    assert CUDAGraphWrapper.copy_runtime_inputs_into_static(first, static)
    assert torch.equal(static[0], first[0])
    assert torch.equal(static[1], first[1])

    second = [
        torch.arange(8, 16, dtype=torch.float32).reshape(2, 4),
        torch.tensor([9.0, 8.0, 7.0]),
    ]
    assert CUDAGraphWrapper.copy_runtime_inputs_into_static(second, static)
    assert torch.equal(static[0], second[0])
    assert torch.equal(static[1], second[1])
    assert not torch.equal(static[0], first[0])


def test_copy_runtime_inputs_rejects_shape_mismatch():
    static = [torch.zeros(2, 4)]
    runtime = [torch.zeros(8)]
    assert CUDAGraphWrapper.copy_runtime_inputs_into_static(runtime, static) is False
    assert torch.equal(static[0], torch.zeros(2, 4))


def test_clone_and_copy_tree_is_input_dependent():
    a = torch.arange(4, dtype=torch.float32)
    b = torch.ones(2)
    cloned = CUDAGraphWrapper._clone_tree(((a, b),))
    assert cloned[0][0].data_ptr() != a.data_ptr()
    assert torch.equal(cloned[0][0], a)
    src = ((torch.arange(4, 8, dtype=torch.float32), torch.full((2,), 9.0)),)
    assert CUDAGraphWrapper._copy_tree(src, cloned)
    assert torch.equal(cloned[0][0], src[0][0])
    assert not torch.equal(cloned[0][0], a)


def test_collect_input_tensors_walks_nested_tuples():
    t0 = torch.arange(4, dtype=torch.float32)
    t1 = torch.ones(2)
    t2 = torch.zeros(3)
    collected = CUDAGraphWrapper._collect_input_tensors(
        ((t0, t1),),
        {"x": {"y": t2}},
    )
    assert collected[0] is t0
    assert collected[1] is t1
    assert collected[2] is t2


def test_copy_nested_runtime_inputs_is_input_dependent():
    static = [torch.zeros(4), torch.zeros(2)]
    first = CUDAGraphWrapper._collect_input_tensors(
        ((torch.arange(4, dtype=torch.float32), torch.ones(2)),),
        {},
    )
    assert CUDAGraphWrapper.copy_runtime_inputs_into_static(first, static)
    assert torch.equal(static[0], first[0])
    assert torch.equal(static[1], first[1])
    second = CUDAGraphWrapper._collect_input_tensors(
        ((torch.arange(4, 8, dtype=torch.float32), torch.full((2,), 7.0)),),
        {},
    )
    assert CUDAGraphWrapper.copy_runtime_inputs_into_static(second, static)
    assert torch.equal(static[0], second[0])
    assert not torch.equal(static[0], first[0])


def test_mrope_get_positions_contiguous_per_capture_size():
    """Dummy extra column makes stride max_tokens+1; inductor wants (N, 1)."""
    from vllm.v1.worker.gpu.mm.rope import RopeState

    rs = RopeState(
        num_dims=3,
        has_delta=True,
        max_num_reqs=4,
        max_num_tokens=2048,
        max_model_len=4096,
        device=torch.device("cpu"),
    )
    raw = rs.positions[:, :8]
    assert raw.stride() == (2049, 1)
    p8 = rs.get_positions(8)
    assert p8.shape == (3, 8)
    assert p8.is_contiguous()
    assert p8.stride() == (8, 1)
    assert rs.get_positions(8).data_ptr() == p8.data_ptr()
    p4 = rs.get_positions(4)
    assert p4.data_ptr() == p8.data_ptr()
    p2048 = rs.get_positions(2048)
    assert p2048.shape == (3, 2048)
    assert p2048.stride() == (2048, 1)
    assert p2048.data_ptr() == p8.data_ptr()


def test_clone_activations_makes_index_views_contiguous():
    """RoPE positions are (3, 8) views of a (3, 2049) buffer; inductor
    asserts stride (8, 1) at capture size 8."""
    buf = torch.zeros(3, 2049, dtype=torch.int64)
    view = buf[:, :8]
    assert view.shape == (3, 8)
    assert view.stride() == (2049, 1)
    cloned = CUDAGraphWrapper._clone_activations(view, num_tokens=8)
    assert cloned.shape == (3, 8)
    assert cloned.is_contiguous()
    assert cloned.stride() == (8, 1)
    cloned.fill_(7)
    assert not torch.equal(view, cloned)


def test_copy_and_call_stages_contiguous_runtime_shape():
    from vllm.compilation.backends import make_copy_and_call

    buf = torch.zeros(3, 2049, dtype=torch.int64)
    runtime = buf[:, :8]
    runtime.copy_(torch.arange(24, dtype=torch.int64).reshape(3, 8))
    staged = [None]
    wrapped = make_copy_and_call([0], staged, lambda *a: a[0])
    out = wrapped(runtime)
    assert out.shape == (3, 8)
    assert out.is_contiguous()
    assert out.stride() == (8, 1)
    assert torch.equal(out, runtime.contiguous())
    out2 = wrapped(runtime + 1)
    assert torch.equal(out2, (runtime + 1).contiguous())
    assert out2.data_ptr() == out.data_ptr()


def test_clone_activations_skips_large_buffers():
    num_tokens = 4
    act = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    huge = torch.zeros(2048, 2048)
    cloned = CUDAGraphWrapper._clone_activations((act, huge), num_tokens)
    assert cloned[0].data_ptr() != act.data_ptr()
    assert torch.equal(cloned[0], act)
    assert cloned[1] is huge


def test_gemma_rms_norm_is_opaque_custom_op():
    assert hasattr(torch.ops.vllm, "gemma_rms_norm")
    assert hasattr(torch.ops.vllm, "gemma_fused_add_rms_norm")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="gemma_rms_norm is registered on the GPU dispatch key",
)
def test_gemma_rms_norm_torch_compile_matches_eager(default_vllm_config):
    torch.manual_seed(0)
    device = current_platform.device_type
    layer = GemmaRMSNorm(16).to(device)
    x = torch.randn(4, 16, device=device)
    eager = layer(x)
    compiled = torch.compile(layer, backend="eager")
    out = compiled(x)
    torch.testing.assert_close(out, eager)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA/HIP required for CUDAGraphWrapper capture/replay",
)
def test_cudagraph_wrapper_replay_follows_new_inputs():
    device = current_platform.device_type
    stream = torch.cuda.Stream()

    class Scale(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    model = Scale().to(device)
    wrapper = CUDAGraphWrapper(model, vllm_config, runtime_mode=CUDAGraphMode.FULL)
    desc = BatchDescriptor(num_tokens=4)

    capture_buf = torch.ones(2, 4, device=device)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        wrapper(capture_buf)

    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=desc,
    ):
        captured = wrapper(capture_buf).clone()

    replay_buf = torch.arange(8, dtype=capture_buf.dtype, device=device).reshape(2, 4)
    assert replay_buf.data_ptr() != capture_buf.data_ptr()
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=desc,
    ):
        replayed = wrapper(replay_buf)
    stream.synchronize()

    torch.testing.assert_close(replayed, replay_buf * 2)
    assert not torch.allclose(replayed, captured)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA/HIP required for CUDAGraphWrapper capture/replay",
)
def test_cudagraph_wrapper_replay_follows_nested_tuple_inputs():
    device = current_platform.device_type
    stream = torch.cuda.Stream()

    class AddPair(nn.Module):
        def forward(self, xs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return xs[0] + xs[1]

    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    model = AddPair().to(device)
    wrapper = CUDAGraphWrapper(model, vllm_config, runtime_mode=CUDAGraphMode.FULL)
    desc = BatchDescriptor(num_tokens=4)

    a = torch.ones(2, 4, device=device)
    b = torch.full((2, 4), 3.0, device=device)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        wrapper((a, b))

    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=desc,
    ):
        captured = wrapper((a, b)).clone()

    a2 = torch.arange(8, dtype=a.dtype, device=device).reshape(2, 4)
    b2 = torch.full((2, 4), 10.0, device=device)
    assert a2.data_ptr() != a.data_ptr()
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=desc,
    ):
        replayed = wrapper((a2, b2))
    stream.synchronize()

    torch.testing.assert_close(replayed, a2 + b2)
    assert not torch.allclose(replayed, captured)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA/HIP required for breakable CUDA graph capture",
)
def test_breakable_full_eager_break_reads_replay_forward_context():
    """ROCm FULL graphs re-run GDN/FA as eager segments. Those segments
    must see the *replay* forward context (current attn_metadata), not
    capture-time dummy metadata. Missing context is the FPP4/FPP6
    first-token-then-garbage failure."""
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )
    from vllm.forward_context import get_forward_context

    seen: list[object] = []
    device = current_platform.device_type
    x = torch.ones(4, device=device)
    y = torch.empty_like(x)

    @eager_break_during_capture
    def gdn_like(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        seen.append(get_forward_context().attn_metadata)
        out.copy_(inp * 2)
        return out

    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cap = BreakableCUDAGraphCapture()
        with set_forward_context(
            attn_metadata="capture",
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            with cap:
                y.copy_(x)
                gdn_like(x, y)
        assert seen == ["capture"]
        assert cap.num_eager_breaks == 1

        x.copy_(torch.arange(4, dtype=x.dtype, device=device))
        seen.clear()
        with set_forward_context(
            attn_metadata="replay",
            vllm_config=vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        ):
            cap.replay()
        stream.synchronize()
    assert seen == ["replay"]
    torch.testing.assert_close(y, x * 2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA/HIP required for CUDAGraphWrapper capture/replay",
)
def test_eager_piecewise_replay_does_not_return_capture_output(
    monkeypatch,
):
    """PIECEWISE + backend=eager must replay, not skip or return warmup.

    Skip-replay calls the runnable again (correct values, no graph). Stale
    replay returns capture-time output (the "duct" failure). Both fail here:
    the runnable must not run on the second PIECEWISE call, and the result
    must follow the new tensor.
    """
    monkeypatch.setenv("VLLM_CG_SKIP_REPLAY", "0")
    device = current_platform.device_type
    stream = torch.cuda.Stream()
    calls = {"n": 0}

    class Scale(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return x * 2

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            backend="eager",
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    model = Scale().to(device)
    wrapper = CUDAGraphWrapper(
        model, vllm_config, runtime_mode=CUDAGraphMode.PIECEWISE
    )
    desc = BatchDescriptor(num_tokens=4)

    capture_buf = torch.ones(2, 4, device=device)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        wrapper(capture_buf)

    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=desc,
    ):
        captured = wrapper(capture_buf).clone()
    calls_after_capture = calls["n"]

    replay_buf = torch.arange(8, dtype=capture_buf.dtype, device=device).reshape(2, 4)
    assert replay_buf.data_ptr() != capture_buf.data_ptr()
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=desc,
    ):
        replayed = wrapper(replay_buf)
    stream.synchronize()

    assert calls["n"] == calls_after_capture, (
        "PIECEWISE eager replay re-entered the runnable (skip-replay); "
        f"calls {calls_after_capture} -> {calls['n']}"
    )
    torch.testing.assert_close(replayed, replay_buf * 2)
    assert not torch.allclose(replayed, captured)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA/HIP required for CUDAGraphWrapper capture/replay",
)
def test_eager_piecewise_replay_multi_op_module(monkeypatch):
    """HIP replay of a multi-op eager module must still follow new inputs."""
    monkeypatch.setenv("VLLM_CG_SKIP_REPLAY", "0")
    device = current_platform.device_type
    stream = torch.cuda.Stream()
    calls = {"n": 0}

    class Deep(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            y = x
            for _ in range(16):
                y = y * 1.01 + 0.001
            return y

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            backend="eager",
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    model = Deep().to(device)
    wrapper = CUDAGraphWrapper(
        model, vllm_config, runtime_mode=CUDAGraphMode.PIECEWISE
    )
    desc = BatchDescriptor(num_tokens=4)
    capture_buf = torch.ones(2, 4, device=device)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        wrapper(capture_buf)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=desc,
    ):
        captured = wrapper(capture_buf).clone()
    n_cap = calls["n"]
    replay_buf = torch.arange(8, dtype=capture_buf.dtype, device=device).reshape(2, 4)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=desc,
    ):
        replayed = wrapper(replay_buf)
    stream.synchronize()
    expect = replay_buf
    for _ in range(16):
        expect = expect * 1.01 + 0.001
    assert calls["n"] == n_cap
    torch.testing.assert_close(replayed, expect, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(replayed, captured)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA/HIP required for CUDAGraphWrapper capture/replay",
)
def test_eager_piecewise_replay_follows_inplace_same_ptr(monkeypatch):
    """Capture aliases must not freeze warmup tokens on same-ptr replay.

    27B eager-piecewise embed graphs saw input_ids=[0,1,0,1,...] because
    the wrapper captured the caller's tensor and _copy_tree no-op'd.
    """
    monkeypatch.setenv("VLLM_CG_SKIP_REPLAY", "0")
    device = current_platform.device_type
    stream = torch.cuda.Stream()
    calls = {"n": 0}

    class Scale(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return x * 2

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            backend="eager",
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
    )
    model = Scale().to(device)
    wrapper = CUDAGraphWrapper(
        model, vllm_config, runtime_mode=CUDAGraphMode.PIECEWISE
    )
    desc = BatchDescriptor(num_tokens=4)
    buf = torch.ones(2, 4, device=device)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        wrapper(buf)
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=desc,
    ):
        captured = wrapper(buf).clone()
    n_cap = calls["n"]

    buf.copy_(torch.arange(8, dtype=buf.dtype, device=device).reshape(2, 4))
    with torch.cuda.stream(stream), set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=desc,
    ):
        replayed = wrapper(buf)
    stream.synchronize()

    assert calls["n"] == n_cap
    torch.testing.assert_close(replayed, buf * 2)
    assert not torch.allclose(replayed, captured)
