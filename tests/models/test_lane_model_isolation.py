# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-lane model-side state isolation (S6).

The two-lane engine runs one thread per lane. Every buffer a model writes
during ``forward`` must therefore be per-lane; only weights, the KV/mamba
tensors and the (read-only) PLE table are shared. These tests pin that
contract on the helpers the Qwen4Exp model uses.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.lane as lane_module
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.models.qwen4_exp.amd.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
)
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEPinnedHostEmbedding,
)
from vllm.v1.lane import Lane, lane_scope

# GPU tests name the device they own (see tests/v1/worker/test_lane_runner.py).
_TEST_DEVICE_INDEX = os.environ.get("VLLM_TEST_DEVICE")
HAS_GPU = _TEST_DEVICE_INDEX is not None and torch.cuda.is_available()


@pytest.fixture
def lane_mode(monkeypatch: pytest.MonkeyPatch):
    """Enable lane mode for one test without leaking it to its neighbours."""
    monkeypatch.setattr(lane_module, "_lane_mode", True)


def _fake_model(num_buffers: int) -> SimpleNamespace:
    return SimpleNamespace(
        _mtp_hidden_buffers=[torch.zeros(4, 2) for _ in range(num_buffers)],
    )


def _mtp_buffer(model):
    return Qwen4ExpModel.get_lane_mtp_hidden_buffer(model)


def _prefetch_buffer(ple, create: bool = True):
    return Qwen4ExpPLEPinnedHostEmbedding._lane_prefetch_buffer(ple, create)


def _fake_ple() -> SimpleNamespace:
    """Attributes ``_lane_prefetch_buffer`` touches, without a real table."""
    return SimpleNamespace(
        _prefetch_buffers={},
        _prefetch_alloc_lock=threading.Lock(),
        _prefetch_rows=8,
        _num_ngram_heads=1,
        embedding_dim=4,
        embedding_method=SimpleNamespace(lookup_dtype=lambda _self: torch.float32),
        _uva_weight=torch.zeros(0),
    )


def test_lane_scope_is_thread_local_and_restores() -> None:
    assert lane_module.current_lane() == Lane.DECODE
    with lane_scope(Lane.PREFILL):
        assert lane_module.current_lane() == Lane.PREFILL
    assert lane_module.current_lane() == Lane.DECODE

    seen: list[int] = []

    def worker() -> None:
        seen.append(int(lane_module.current_lane()))
        with lane_scope(Lane.PREFILL):
            seen.append(int(lane_module.current_lane()))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == [int(Lane.DECODE), int(Lane.PREFILL)]


def test_mtp_hidden_buffer_is_keyed_by_lane(lane_mode: None) -> None:
    model = _fake_model(2)
    decode_buffer = _mtp_buffer(model)
    with lane_scope(Lane.PREFILL):
        prefill_buffer = _mtp_buffer(model)
    assert decode_buffer is not prefill_buffer
    assert decode_buffer.data_ptr() != prefill_buffer.data_ptr()

    # Lane D keeps today's object, so a captured graph stays valid.
    assert _mtp_buffer(model) is model._mtp_hidden_buffers[0]


def test_mtp_hidden_buffer_requires_one_per_lane() -> None:
    """Flag off allocates one buffer; lane 1 must fail loudly, not alias D."""
    model = _fake_model(1)
    assert _mtp_buffer(model) is model._mtp_hidden_buffers[0]
    with lane_scope(Lane.PREFILL), pytest.raises(RuntimeError, match="lane 1"):
        _mtp_buffer(model)


def test_no_mtp_buffer_returns_none() -> None:
    model = SimpleNamespace(_mtp_hidden_buffers=[])
    assert _mtp_buffer(model) is None


class _Backbone:
    """Stands in for the compiled Qwen4ExpModel: records the buffer it gets."""

    get_lane_mtp_hidden_buffer = Qwen4ExpModel.get_lane_mtp_hidden_buffer

    def __init__(self, num_buffers: int) -> None:
        self._mtp_hidden_buffers = [torch.zeros(4, 2) for _ in range(num_buffers)]
        self.buffers_seen: list[torch.Tensor | None] = []

    def __call__(self, *args, mtp_hidden_buffer=None, **kwargs) -> torch.Tensor:
        self.buffers_seen.append(mtp_hidden_buffer)
        return torch.zeros(1)


def _assert_lane_buffers_passed(backbone: _Backbone, forward) -> None:
    forward()
    with lane_scope(Lane.PREFILL):
        forward()
    decode_buffer, prefill_buffer = backbone._mtp_hidden_buffers
    assert len(backbone.buffers_seen) == 2
    assert backbone.buffers_seen[0] is decode_buffer
    assert backbone.buffers_seen[1] is prefill_buffer


def test_causal_lm_passes_the_calling_lanes_mtp_buffer(lane_mode: None) -> None:
    backbone = _Backbone(2)
    model = SimpleNamespace(model=backbone)
    positions = torch.zeros(1, dtype=torch.long)
    _assert_lane_buffers_passed(
        backbone, lambda: Qwen4ExpForCausalLM.forward(model, None, positions)
    )


def test_vl_model_passes_the_calling_lanes_mtp_buffer(lane_mode: None) -> None:
    backbone = _Backbone(2)
    model = SimpleNamespace(language_model=SimpleNamespace(model=backbone))
    positions = torch.zeros(1, dtype=torch.long)
    _assert_lane_buffers_passed(
        backbone,
        lambda: Qwen4ExpForConditionalGeneration.forward(model, None, positions),
    )


def test_compiled_forward_never_selects_the_lane_buffer() -> None:
    """Without Dynamo's guards (vLLM drops them), a lane read traced in the
    compiled forward is evaluated once, on lane D: lane P would write D's
    buffer."""
    names = Qwen4ExpModel.forward.__code__.co_names
    assert "get_lane_mtp_hidden_buffer" not in names
    assert "current_lane" not in names


@pytest.mark.skipif(not HAS_GPU, reason="Requires a GPU")
def test_compiled_module_writes_the_buffer_each_lane_passes(lane_mode: None) -> None:
    """The fix's mechanism under vLLM compilation: lane P runs the code
    compiled for lane D, and each call writes the buffer its caller passed."""
    from torch import nn

    from vllm.compilation.counter import compilation_counter
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context

    device = torch.device("cuda", int(_TEST_DEVICE_INDEX or 0))

    @support_torch_compile(dynamic_arg_dims={"x": 0})
    class Writer(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()

        def forward(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            y = x * 2 + 1
            out[: y.shape[0]].copy_(y)
            return y

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.NONE
        )
    )
    with set_current_vllm_config(config):
        writer = Writer(vllm_config=config)
    buffers = [torch.zeros(8, 4, device=device) for _ in range(2)]
    with set_current_vllm_config(config), set_forward_context({}, config):
        writer(torch.full((4, 4), 1.0, device=device), buffers[0])
        compiled = compilation_counter.clone()
        with lane_scope(Lane.PREFILL):
            writer(torch.full((4, 4), 5.0, device=device), buffers[1])
    assert compilation_counter == compiled
    assert buffers[0][:4].cpu().eq(3.0).all()
    assert buffers[1][:4].cpu().eq(11.0).all()


def test_ple_prefetch_buffer_is_per_lane_and_stable(lane_mode: None) -> None:
    ple = _fake_ple()

    decode_buffer = _prefetch_buffer(ple)
    assert decode_buffer is not None
    assert decode_buffer is _prefetch_buffer(ple)

    with lane_scope(Lane.PREFILL):
        prefill_buffer = _prefetch_buffer(ple)
        assert prefill_buffer is not None
        # Same lane, same object; different lane, different storage.
        assert prefill_buffer is _prefetch_buffer(ple)
    assert prefill_buffer is not decode_buffer
    assert prefill_buffer.data_ptr() != decode_buffer.data_ptr()
    assert sorted(ple._prefetch_buffers) == [int(Lane.DECODE), int(Lane.PREFILL)]


def test_ple_prefetch_buffer_from_a_thread_does_not_alias(lane_mode: None) -> None:
    """The lane thread's default lane is DECODE until it enters a scope."""
    ple = _fake_ple()
    _prefetch_buffer(ple)

    seen: list[int] = []

    def worker() -> None:
        with lane_scope(Lane.PREFILL):
            seen.append(_prefetch_buffer(ple).data_ptr())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == [ple._prefetch_buffers[int(Lane.PREFILL)].data_ptr()]
    assert seen[0] != ple._prefetch_buffers[int(Lane.DECODE)].data_ptr()


def test_ple_prefetch_rows_are_bounded() -> None:
    check = Qwen4ExpPLEPinnedHostEmbedding._check_prefetch_rows
    buffer = torch.zeros(8, 1, 4)
    check(None, buffer, torch.zeros(8, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="holds 8 rows"):
        check(None, buffer, torch.zeros(9, 1, dtype=torch.long))


def test_gdn_decode_arenas_stay_off_the_prefill_lane(lane_mode: None) -> None:
    """Lane P runs the GDN decode kernel on the paged state (R-S6-1).

    The arenas keep D's captured decode graphs' state pointers stable. A
    one-token prefill on lane P may run the decode kernel; it binds the paged
    state and leaves D's arenas unallocated and unwritten.
    """
    gdn = QwenGatedDeltaNetAttention
    layer = SimpleNamespace(_conv_state_arena=None, _ssm_state_arena=None)
    conv_state, ssm_state = torch.zeros(2, 3), torch.zeros(2, 4)
    metadata = SimpleNamespace(use_state_arenas=True)
    with lane_scope(Lane.PREFILL):
        ensured = gdn._ensure_gdn_state_arenas(layer, conv_state, ssm_state)
        bound = gdn._bind_decode_state_arenas(layer, conv_state, ssm_state, metadata, 1)
        gdn._unbind_decode_state_arenas(layer, conv_state, ssm_state, metadata, 1)
    assert ensured[0] is conv_state and ensured[1] is ssm_state
    assert bound[0] is conv_state and bound[1] is ssm_state
    assert layer._conv_state_arena is None and layer._ssm_state_arena is None


def test_workspace_scratch_is_per_lane() -> None:
    """GDN checkpoints, QSA split-KV and segmented-attention scratch all go
    through the workspace manager, keyed by ``(ubatch, lane)``."""
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        use_workspace_lane,
    )

    init_workspace_manager(torch.device("cpu"), 1, 2)
    manager = current_workspace_manager()
    spec = ((4,), torch.float32)
    with use_workspace_lane(int(Lane.DECODE)):
        (decode_buf,) = manager.get_simultaneous(spec)
    with use_workspace_lane(int(Lane.PREFILL)):
        (prefill_buf,) = manager.get_simultaneous(spec)
    assert decode_buf.data_ptr() != prefill_buf.data_ptr()
    with use_workspace_lane(int(Lane.DECODE)):
        assert manager.get_simultaneous(spec)[0].data_ptr() == decode_buf.data_ptr()
