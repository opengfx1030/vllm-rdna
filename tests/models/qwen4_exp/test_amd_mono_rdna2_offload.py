# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-pool configuration and split K2 parity under eviction and replay."""

import ast
import functools
import importlib.util
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_PAYLOAD = ("w13", "s13", "z13", "w2", "s2", "z2")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def options():
    return _load(
        "rdna2_offload_options_test",
        _ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/__init__.py",
    ).Options


@pytest.mark.parametrize("slots", [0, 1, 128, 256])
@pytest.mark.parametrize("policy", ["lfu_lru", "lru"])
def test_options_preserve_offload_capacity_and_policy(options, slots, policy):
    opts = options.from_dict(
        {
            "expert_cache_slots": slots,
            "expert_cache_policy": policy,
            "expert_cache_decay": 7,
        }
    )
    assert (opts.expert_cache_slots, opts.expert_cache_policy) == (slots, policy)
    assert opts.expert_cache_decay == 7


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"expert_cache_slots": -1}, "expert_cache_slots"),
        ({"expert_cache_slots": 257}, "expert_cache_slots"),
        ({"expert_cache_slots": True}, "integer"),
        ({"expert_cache_slots": 1.5}, "integer"),
        ({"expert_cache_slots": "32"}, "integer"),
        ({"expert_cache_policy": "random"}, "expert_cache_policy"),
        ({"expert_cache_decay": 0}, "positive"),
        ({"expert_cache_decay": -1}, "positive"),
        ({"expert_cache_decay": False}, "integer"),
        ({"expert_cache_decay": 1.5}, "integer"),
        ({"expert_cache_copy": "unknown"}, "expert_cache_copy"),
        ({"expert_cache_slot": 32}, "unknown"),
    ],
)
def test_options_reject_unsafe_or_misspelled_pool_configuration(
    options, extra, message
):
    with pytest.raises(ValueError, match=message):
        options.from_dict(extra)


def test_offload_remains_disabled_by_default(options):
    assert options.from_dict({}).expert_cache_slots == 0
    assert options.from_dict({}).expert_cache_copy == "tensor"


@pytest.mark.parametrize(
    "rdna, monokernel, extra, expected",
    [
        (True, True, {"rocm_mono_decode": {"expert_cache_slots": 32}}, True),
        (False, True, {"rocm_mono_decode": {"expert_cache_slots": 32}}, False),
        (True, False, {"rocm_mono_decode": {"expert_cache_slots": 32}}, False),
        (True, True, {"rocm_mono_decode": {"expert_cache_slots": 0}}, False),
        (True, True, {"rocm_mono_decode": {}}, False),
        (True, True, {"rocm_mono_decode": None}, False),
        (True, True, None, False),
    ],
)
def test_breakable_default_is_scoped_to_rdna_expert_cache(
    monkeypatch, rdna, monokernel, extra, expected
):
    source = ast.parse((_ROOT / "vllm/config/vllm.py").read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "VllmConfig"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_uses_breakable_cudagraph_by_default"
    )
    monkeypatch.setitem(
        sys.modules, "vllm.platforms.rdna", SimpleNamespace(on_rdna_family=lambda: rdna)
    )
    namespace: dict[str, Any] = {
        "default_breakable_cudagraph_architectures": lambda: {"LegacyModel"}
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "vllm.py", "exec"),
        namespace,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["Qwen4ExpForCausalLM"]),
        kernel_config=SimpleNamespace(monokernel=monokernel),
        additional_config=extra,
    )
    default = namespace[method.name]
    assert default(config) is expected
    config.model_config.architectures = ["LegacyModel"]
    assert default(config)  # Preserve existing architecture defaults.
    config.model_config = None
    assert not default(config)


@pytest.mark.parametrize("publish", [False, True])
def test_piecewise_wrapper_preserves_opted_in_replay_context(publish):
    source = ast.parse((_ROOT / "vllm/compilation/breakable_cudagraph.py").read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "BreakableCUDAGraphWrapper"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "_capture")
    ]
    for method in methods:
        method.returns = None
        for argument in (*method.args.args, *method.args.kwonlyargs):
            argument.annotation = None

    class Wrapper:
        _all_instances = set()

    class Capture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    offloader = SimpleNamespace(
        sync_prev_onload=lambda: None, join_after_forward=lambda: None
    )
    namespace: dict[str, Any] = {
        "BreakableCUDAGraphWrapper": Wrapper,
        "BreakableCUDAGraphCapture": Capture,
        "current_platform": SimpleNamespace(get_global_graph_pool=lambda: "pool"),
        "logger": Mock(),
        "logging": SimpleNamespace(DEBUG=10),
        "set_graph_pool_id": lambda pool: None,
        "gc": SimpleNamespace(isenabled=lambda: False),
        "get_offloader": lambda: offloader,
        "validate_cudagraph_capturing_enabled": lambda: None,
        "weak_ref_tensors": lambda output: output,
    }
    exec(
        compile(ast.Module(body=[*methods], type_ignores=[]), "breakable.py", "exec"),
        namespace,
    )
    wrapper: Any = Wrapper()
    namespace["__init__"](
        wrapper,
        lambda: "output",
        SimpleNamespace(compilation_config=object()),
        publish_replay_context=publish,
    )
    wrapper._collect_tensor_addresses = lambda args, kwargs: []
    entry = SimpleNamespace(batch_descriptor="shape")
    assert namespace["_capture"](wrapper, entry, (), {}) == "output"
    assert entry.capture.kwargs == {"pool": "pool", "publish_replay_context": publish}


def test_ple_short_conv_eager_boundary_reads_current_replay_metadata():
    torch = pytest.importorskip("torch")
    ple_source = ast.parse(
        (_ROOT / "vllm/models/qwen4_exp/amd/ple_layer.py").read_text()
    )
    ple = next(
        node
        for node in ple_source.body
        if isinstance(node, ast.FunctionDef) and node.name == "qwen4_exp_ple_short_conv"
    )
    decorator_source = ast.parse(
        (_ROOT / "vllm/compilation/breakable_cudagraph.py").read_text()
    )
    decorator = next(
        node
        for node in decorator_source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "eager_break_during_capture"
    )
    context = SimpleNamespace(cudagraph_runtime_mode="piecewise", bias=1)
    seen = []

    def short_conv(inputs):
        seen.append(context.bias)
        return inputs + context.bias

    layer = SimpleNamespace(_short_conv=short_conv)
    context.no_compile_layers = {"ple": layer}

    class Capture:
        _capturing = True

        def __init__(self):
            self.segments = []

        def add_eager(self, fn):
            self._capturing = False
            result = fn()
            self.segments.append(fn)
            self._capturing = True
            return result

    capture = Capture()
    namespace: dict[str, Any] = {
        "F": object,
        "Any": object,
        "functools": functools,
        "torch": torch,
        "on_rdna_family": lambda: True,
        "is_breakable_cudagraph_enabled": lambda: True,
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "BreakableCUDAGraphCapture": SimpleNamespace(current=lambda: capture),
    }
    exec(
        compile(ast.Module(body=[decorator, ple], type_ignores=[]), "ple.py", "exec"),
        namespace,
    )
    inputs = torch.ones((3, 4))
    output = torch.zeros_like(inputs)
    namespace["qwen4_exp_ple_short_conv"](inputs, output, "ple")
    assert len(capture.segments) == 1
    torch.testing.assert_close(output, inputs + 1)
    context = SimpleNamespace(
        cudagraph_runtime_mode="piecewise", bias=9, no_compile_layers={"ple": layer}
    )
    inputs.fill_(2)
    capture._capturing = False
    capture.segments[0]()
    assert seen == [1, 9]
    torch.testing.assert_close(output, inputs + 9)


def test_production_offload_reuses_graphs_but_replans_changing_demand(monkeypatch):
    """Execute the production orchestrator against a recording GPU boundary.

    Graph capture records calls without executing them. Replay consumes the
    current route/map contents, exercising the orchestration rather than a
    separately written implementation of its launch loop.
    """
    policy_cls = _load(
        "rdna2_offload_policy_integration_test",
        _ROOT / "vllm/model_executor/layers/fused_moe/rdna_expert_cache_policy.py",
    ).ExpertCachePolicy
    events = []

    class Tensor:
        def __init__(self, shape, dtype="half", device="cuda", role=None):
            self.shape, self.dtype, self.device, self.role = shape, dtype, device, role
            self.values = []
            self.slots = 0  # a map's real slots are below it

        def contiguous(self):
            return self

        def is_contiguous(self):
            return True

        def copy_(self, source, non_blocking=False):
            self.values = source.values.copy()
            return self

        def masked_fill_(self, mask, value):
            topk = self.shape[1]
            for index in range(len(self.values)):
                if mask[index // topk, 0]:
                    self.values[index] = value
            return self

        def data_ptr(self):
            return id(self)

        def cpu(self):
            return self

        def reshape(self, *shape):
            return self

        def tolist(self):
            return self.values.copy()

    class NoncontiguousInjection(Tensor):
        def contiguous(self):
            raise AssertionError("use stable injection staging, not a temporary copy")

        def is_contiguous(self):
            return False

    capture = None

    class Graph:
        def __init__(self):
            self.calls = []

        def replay(self):
            for args in self.calls:
                execute(args)

    @contextmanager
    def graph_context(graph, stream):
        nonlocal capture
        capture = graph
        try:
            yield
        finally:
            capture = None

    def execute(args):
        stage, maps = args[23], args[22]
        events.append(f"2{maps.role}" if stage == 2 else stage)
        if stage == 1:
            assert all(tensor.is_contiguous() for tensor in args[:3])
            args[20].values = args[0].values.copy()
            args[18].values = [100 + sum(args[2].values)]
        elif stage == 2:
            # A launch runs the experts mapped to a real slot; the dummy slot
            # and -1 add nothing, whichever routes reach them.
            real = [0 <= slot < maps.slots for slot in maps.values]
            args[18].values[0] += sum(
                expert + 1
                for expert in args[20].values
                if 0 <= expert < len(real) and real[expert]
            )

    def native_op(*args):
        if capture is not None:
            capture.calls.append(args)
        else:
            execute(args)

    class Cache:
        def __init__(self, capacity=2, experts=8):
            self.policy = policy_cls(experts, capacity)
            self.slot_count = capacity
            self.dummy_slot = capacity if capacity < experts else None
            self.absent = -1 if self.dummy_slot is None else self.dummy_slot
            self.experts = experts
            self.k2_payload = {name: Tensor((1,)) for name in _PAYLOAD}
            self.slot_map, self.resident_map, self.miss_map = (
                Tensor((experts,), "int32", role=role) for role in "srm"
            )
            for tensor in (self.slot_map, self.resident_map, self.miss_map):
                tensor.slots = capacity  # real slots are below it
            self.slot_map.values = [-1] * experts
            self.resident_map.values = [self.absent] * experts
            self.miss_map.values = [self.absent] * experts

        def start_tracking(self):
            events.append("track")

        def ensure(self, experts, by_residency=False):
            lease = self.policy.acquire(experts)
            slots = {expert: slot for expert, slot, _ in lease.bindings}
            loads = {expert for expert, _, miss in lease.bindings if miss}
            if not by_residency:
                self.slot_map.values = [
                    slots.get(expert, -1) for expert in range(self.experts)
                ]
            lease.commit()
            mapping = self.policy.mapping
            self.resident_map.values = [
                mapping.get(expert, self.absent) for expert in range(self.experts)
            ]
            if loads:
                self.miss_map.values = [
                    mapping[expert] if expert in loads else self.absent
                    for expert in range(self.experts)
                ]
            events.append("acquire")
            return lease

        def release(self, lease):
            lease.release()
            events.append("release")

    class Event:
        def record(self):
            events.append("drain")

        def synchronize(self):
            events.append("drained")

    active_boundary = object()
    fake_torch = SimpleNamespace(
        empty_like=lambda tensor: Tensor(tensor.shape, tensor.dtype, tensor.device),
        empty=lambda shape, dtype, device="cpu", pin_memory=False: Tensor(
            shape, dtype, device
        ),
        int32="int32",
        float32="float32",
        cuda=SimpleNamespace(
            CUDAGraph=Graph, graph=graph_context, current_stream=lambda: 0, Event=Event
        ),
        ops=SimpleNamespace(_rocm_C=SimpleNamespace(mono_k2_offload_rdna2=native_op)),
    )
    runtime_name = "vllm.model_executor.layers.fused_moe.rdna_expert_cache"
    forward_context = SimpleNamespace(is_padding=None, additional_kwargs={})
    context_available = True
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            is_forward_context_available=lambda: context_available,
            get_forward_context=lambda: forward_context,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        runtime_name,
        SimpleNamespace(is_capturing_expert_work=lambda: False, PAYLOAD_NAMES=_PAYLOAD),
    )
    source = ast.parse(
        (_ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/k2.py").read_text()
    )
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("run_offloaded", "_padding")
    ]
    for function in functions:
        function.decorator_list = []
    namespace: dict[str, Any] = {
        "torch": fake_torch,
        "OPTS": SimpleNamespace(k2_num_wg=0),
        "offload_trace": SimpleNamespace(ENABLED=False),
        "BreakableCUDAGraphCapture": SimpleNamespace(current=lambda: active_boundary),
        "_PADDING_KEY": "rdna_offload_padding",
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "k2.py", "exec"),
        namespace,
    )
    run = namespace["run_offloaded"]
    names = ("norm_w", "wd", "wu", "wr", "se_w1", "se_w2", "se_wg", *_PAYLOAD)
    mono = SimpleNamespace(
        expert_cache=Cache(),
        t={name: Tensor((1,)) for name in names},
        offload_ws=Tensor((1,)),
        base=0,
        eps=1e-6,
    )
    inputs = (
        NoncontiguousInjection((1, 10240)),
        NoncontiguousInjection((1, 2560)),
        NoncontiguousInjection((1, 4)),
    )
    inputs[0].values = [0, 1, 2, 3, 0]
    first = run(mono, *inputs)
    assert first[1].values == [100 + 1 + 2 + 3 + 4 + 1]
    graphs = 3  # prefix, pass on the slot map, finish: no pass can be split
    assert len(mono._offload_graphs) == graphs
    replacement_hidden = NoncontiguousInjection((1, 10240))
    replacement_hidden.values = [3, 2, 4, 3]
    replacement_attention = NoncontiguousInjection((1, 2560))
    replacement_attention.values = [9]
    replacement_injection = NoncontiguousInjection((1, 4))
    replacement_injection.values = [1, 2, 3, 4]
    assert replacement_injection.data_ptr() != inputs[2].data_ptr()
    inputs = (replacement_hidden, replacement_attention, replacement_injection)
    second = run(mono, *inputs)
    assert second == first  # Stable caller-visible storage across replay.
    assert second[1].values == [110 + 4 + 3 + 5 + 4]
    assert len(mono._offload_graphs) == graphs
    assert len(mono._offload_inputs) == 2
    assert next(
        graph for key, graph in mono._offload_graphs.items() if key[0] == 1
    ).calls[0][1].values == [9]
    assert mono.expert_cache.policy.epoch == 2  # Observe once, not per pass.
    # A demand that may outgrow the cache is planned once the host has read it:
    # nothing is launched before, and the passes mask in the experts one after
    # the other.
    call = [1, "drain", "drained", "acquire", "2s", "release"]
    call += ["acquire", "2s", "release", 3]
    assert events == call * 2
    active_boundary = None
    context_available = False  # Standalone adapters have no ForwardContext.
    run(mono, *inputs)
    assert len(mono._offload_graphs) == graphs  # Eager calls do not capture.
    assert mono.expert_cache.policy.epoch == 3
    # Only the first row is real. Padding routes must neither fill slots nor
    # inflate frequency, even though the GPU route buffer retains all rows.
    real_torch = pytest.importorskip("torch")
    context_available = True
    forward_context.is_padding = real_torch.tensor([False, True, True])
    padded_inputs = (Tensor((3, 10240)), Tensor((3, 2560)), Tensor((3, 4)))
    padded_inputs[0].values = [0] * 10 + [6] * 10 + [7] * 10
    previous = mono.expert_cache.policy.frequencies
    result = run(mono, *padded_inputs)
    frequency = mono.expert_cache.policy.frequencies
    assert frequency[0] == previous[0] + 10
    assert frequency[6:] == previous[6:]
    assert {0, 6, 7} & set(mono.expert_cache.policy.mapping) == {0}
    assert result[1].values == [100 + 10]  # Only the real row's routes ran.

    # A demand that cannot outgrow the cache is one pass whose experts are
    # resident or loaded: the resident ones run while the host reads the
    # routing, the loads once they have landed, and each launch counts the
    # whole demand.
    forward_context.is_padding = None

    def owner_of(cache):
        return SimpleNamespace(
            expert_cache=cache, t=mono.t, offload_ws=Tensor((1,)), base=0, eps=1e-6
        )

    def routed(owner, *experts):
        hidden = Tensor((1, 10240))
        hidden.values = list(experts)
        return run(owner, hidden, Tensor((1, 2560)), Tensor((1, 4)))[1].values

    single = owner_of(Cache(10, experts=16))

    early = ["track", 1, "drain", "2r", "drained"]
    load_then_run = early + ["acquire", "2m", "release", 3]
    hit_only = early + ["acquire", "release", 3]
    nothing_local = early + [3]
    evicting = tuple(range(3, 13))
    for experts, expected, sequence in (
        ((0, 1), 1 + 2, load_then_run),  # cold: every expert is loaded
        ((1, 0), 2 + 1, hit_only),  # every expert resident: no second launch
        ((0, 2), 1 + 3, load_then_run),  # one hit, one load
        (evicting, sum(range(4, 14)), load_then_run),  # evicts all earlier experts
        ((2, 3), 3 + 4, load_then_run),  # 3 hit, 2 reloaded
        ((3, 3, 2), 4 + 4 + 3, hit_only),
        ((-1, 20), 0, nothing_local),  # padding and other ranks' experts only
    ):
        events.clear()
        assert routed(single, *experts) == [100 + expected], experts
        assert events == sequence, experts
    assert single.expert_cache.slot_map.values == [-1] * 16  # no pass mask needed
    # With no dummy slot (every expert fits) or a demand that may outgrow the
    # cache, the passes run on the slot map once the routing is known.
    for cache in (Cache(16, experts=16), Cache(9, experts=16)):
        events.clear()
        assert routed(owner_of(cache), 4, 5) == [100 + 5 + 6]
        assert events == [1, "drain", "drained", "acquire", "2s", "release", 3]


@pytest.mark.parametrize("kernel", ["k1", "k2"])
def test_native_fake_outputs_are_row_major_for_noncontiguous_inputs(kernel):
    torch = pytest.importorskip("torch")
    hidden = torch.empty((10240, 3)).t()
    attention = torch.empty((2560, 3)).t()
    injection = torch.empty((3, 8))[:, ::2]
    source = ast.parse(
        (_ROOT / f"vllm/models/qwen4_exp/amd/mono/rdna2/{kernel}.py").read_text()
    )
    fake = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == f"_{kernel}_fake"
    )
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=[fake], type_ignores=[]), f"{kernel}.py", "exec"),
        namespace,
    )
    output = namespace[f"_{kernel}_fake"](hidden, attention, injection, "layer")
    assert [tuple(value.shape) for value in output] == [(3, 10240), (3, 2560), (3, 4)]
    assert all(value.is_contiguous() for value in output)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("publish_context", [None, False, True])
def test_breakable_replay_only_publishes_opted_in_context_and_restores_previous(
    fail, publish_context
):
    """Run the production replay method without importing GPU runtime modules."""
    source = ast.parse((_ROOT / "vllm/compilation/breakable_cudagraph.py").read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "BreakableCUDAGraphCapture"
    )
    replay = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "replay"
    )
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    namespace: dict[str, Any] = {"Any": object}
    exec(
        compile(
            ast.Module(body=[init, replay], type_ignores=[]),
            "breakable_cudagraph.py",
            "exec",
        ),
        namespace,
    )
    tls = threading.local()
    previous: Any = object()
    tls.active = previous
    capture = SimpleNamespace(_tls=tls, capturing_segment=False)
    if publish_context is None:
        namespace["__init__"](capture)
    else:
        namespace["__init__"](capture, publish_replay_context=publish_context)
    assert capture.publish_replay_context is bool(publish_context)
    visited = []

    def eager_segment():
        assert tls.active is (capture if publish_context else previous)
        if publish_context:
            assert not tls.active.capturing_segment
        visited.append(True)
        if fail:
            raise RuntimeError("copy failure")

    capture.segments = [eager_segment]
    if fail:
        with pytest.raises(RuntimeError, match="copy failure"):
            namespace["replay"](capture)
    else:
        namespace["replay"](capture)
    assert visited == [True]
    assert tls.active is previous


@pytest.fixture(scope="module", params=[128, 256], ids=["tp4-ep4", "tp2-ep2"])
def native_case(request):
    torch = pytest.importorskip("torch")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a gfx1030 GPU and the rebuilt split K2 op")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx1030":
        pytest.skip("split K2 qualification targets gfx1030")
    import vllm._custom_ops  # noqa: F401

    if not hasattr(torch.ops._rocm_C, "mono_k2_offload_rdna2"):
        pytest.skip("rebuild _rocm_C with mono_k2_offload_rdna2")
    helper = _load(
        "rdna2_offload_kernel_helpers_test",
        Path(__file__).with_name("test_amd_mono_rdna2_kernels.py"),
    )
    local_experts = request.param
    # Reuse the resident suite's native packing and randomized payloads.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(helper, "EL", local_experts)
        patch.setattr(helper, "ISE", 160 if local_experts == 128 else 320)
        generator = torch.Generator(device="cuda").manual_seed(411)
        layer, golden = helper._make_layer(generator, keep_golden=False)
    helper.offload_golden = golden
    return torch, helper, layer, local_experts


def _buffers(torch, helper, layer, rows, slots, local_experts):
    shape = (rows, helper.H)
    workspace_bytes = torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(
        helper.H, helper.R, helper.ND, helper.E, layer["se_w2"].shape[1], helper.INTER
    )
    payload = {name: torch.empty_like(layer[name][:slots]) for name in _PAYLOAD}
    return {
        "payload": payload,
        "res_out": torch.empty(
            rows, helper.HC * helper.H, device="cuda", dtype=torch.half
        ),
        "inj_out": torch.empty(rows, helper.HC, device="cuda", dtype=torch.half),
        "out": torch.empty(shape, device="cuda", dtype=torch.half),
        "ws": torch.zeros(workspace_bytes + 8192, device="cuda", dtype=torch.uint8),
        "ids": torch.empty(rows, helper.TOPK, device="cuda", dtype=torch.int32),
        "weights": torch.empty(rows, helper.TOPK, device="cuda", dtype=torch.float32),
        "map": torch.full((local_experts,), -1, device="cuda", dtype=torch.int32),
    }


def _args(layer, inputs, buffers):
    return (
        *inputs,
        *(
            layer[name]
            for name in ("norm_w", "wd", "wu", "wr", "se_w1", "se_w2", "se_wg")
        ),
        *(buffers["payload"][name] for name in _PAYLOAD),
        *(buffers[name] for name in ("res_out", "inj_out", "out", "ws")),
    )


def _stage(torch, layer, inputs, buffers, stage, base):
    torch.ops._rocm_C.mono_k2_offload_rdna2(
        *_args(layer, inputs, buffers),
        buffers["ids"],
        buffers["weights"],
        buffers["map"],
        stage,
        base,
        1,
        0,
        1e-6,
        2000.0,
    )


def _full_reference(torch, helper, layer, inputs, local_experts, base):
    buffers = _buffers(
        torch, helper, layer, inputs[0].shape[0], local_experts, local_experts
    )
    buffers["payload"] = {name: layer[name] for name in _PAYLOAD}
    buffers["map"].copy_(torch.arange(local_experts, device="cuda", dtype=torch.int32))
    if local_experts == 128:
        # TP4 compares against the independent, unchanged resident K2 entry.
        torch.ops._rocm_C.mono_k2_rdna2(
            *_args(layer, inputs, buffers),
            base,
            1,
            0,
            helper.EPS,
            2000.0,
            True,
            False,
            False,
        )
    else:
        # Resident K2 supports 128 experts only. TP2's oracle is one full
        # 256-expert split pass, compared with capacity-three eviction passes.
        for stage in (1, 2, 3):
            _stage(torch, layer, inputs, buffers, stage, base)
    return tuple(buffers[name].clone() for name in ("res_out", "out", "inj_out"))


def _demand(torch, buffers, base):
    local_experts = buffers["map"].numel()
    return list(
        dict.fromkeys(
            expert - base
            for expert in buffers["ids"].reshape(-1).cpu().tolist()
            if base <= expert < base + local_experts
        )
    )


def _fill_pass(torch, layer, buffers, experts):
    buffers["map"].fill_(-1)
    for slot, expert in enumerate(experts):
        for name in _PAYLOAD:
            buffers["payload"][name][slot].copy_(layer[name][expert])
        buffers["map"][expert] = slot


def _assert_outputs(torch, buffers, expected):
    assert torch.equal(buffers["res_out"], expected[0])
    torch.testing.assert_close(buffers["out"], expected[1], rtol=0, atol=2e-3)
    torch.testing.assert_close(buffers["inj_out"], expected[2], rtol=0, atol=2e-3)
    assert torch.ops._rocm_C.mono_k2_rdna2_error(buffers["ws"])[0].item() == 0


def _assert_independent_tp2(torch, helper, layer, inputs, buffers, base):
    """Compare with unfused HC/router/shared-expert/ordinary HIP MoE ops."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(helper, "EL", 256)
        out, residual, injection, ids, _ = helper._today_ffn(
            layer, *inputs, return_routes=True, expert_base=base
        )
    torch.accelerator.synchronize()
    assert bool(torch.isfinite(out).all()), "independent reference must remain finite"
    if inputs[0].shape[0] >= 3:
        assert bool(((ids >= base + 128) & (ids < base + 256)).any()), (
            "TP2 fixture must execute experts outside the old 128-expert limit"
        )
    assert torch.equal(buffers["res_out"], residual)
    torch.testing.assert_close(buffers["inj_out"], injection, rtol=0, atol=5e-3)
    # Keep the existing resident suite's fp32-router near-tie allowance:
    # different GEMM reduction paths may swap the 10th/11th expert only when
    # their logit gap is below 1e-3.
    gaps = helper._golden_gaps(helper.offload_golden, *inputs)
    differences = (buffers["out"].float() - out.float()).abs().amax(dim=1).tolist()
    bad = [
        row
        for row, (difference, gap) in enumerate(zip(differences, gaps))
        if difference >= 5e-3 and gap >= helper.TIE_GAP
    ]
    assert not bad, (
        f"TP2 independent reference mismatch rows={bad}, differences={differences}"
    )


@pytest.mark.parametrize("rows", [1, 3, 8, 12, 24])
def test_split_k2_eviction_passes_match_full_payload(native_case, rows):
    torch, helper, layer, local_experts = native_case
    base = 512 - local_experts  # Exercise nonzero rank-local ownership offsets.
    inputs = helper._inputs(rows, 500 + rows)
    expected = _full_reference(torch, helper, layer, inputs, local_experts, base)
    buffers = _buffers(torch, helper, layer, rows, 3, local_experts)
    _stage(torch, layer, inputs, buffers, 1, base)
    demand = _demand(torch, buffers, base)
    if rows >= 3:
        assert len(demand) > 3, "fixture must exercise more than one expert pass"
    for start in range(0, len(demand), 3):
        _fill_pass(torch, layer, buffers, demand[start : start + 3])
        _stage(torch, layer, inputs, buffers, 2, base)
    _stage(torch, layer, inputs, buffers, 3, base)
    _assert_outputs(torch, buffers, expected)
    if local_experts == 256:
        _assert_independent_tp2(torch, helper, layer, inputs, buffers, base)


@pytest.mark.parametrize("native_case", [256], indirect=True, ids=["tp2-ep2"])
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("parts", [2, 3])
def test_demand_split_over_launches_that_each_count_it_all_is_one_full_pass(
    native_case, rows, parts
):
    """An all-zero slot keeps a demand split over several launches bit-exact.

    A launch sizes its K slices by the number of experts it runs and rounds
    every slice on its own, so a demand split by leaving experts out of a launch
    differs from one full pass in the last bit. Mapping the experts that other
    launches run onto a zero slot keeps every launch's count at the whole
    demand, and the zero slot adds exactly nothing to the fixed-point sums.
    """
    torch, helper, layer, local_experts = native_case
    base = 512 - local_experts
    inputs = helper._inputs(rows, 700 + rows)
    expected = _full_reference(torch, helper, layer, inputs, local_experts, base)
    slots = 96
    buffers = _buffers(torch, helper, layer, rows, slots + 1, local_experts)
    dummy = slots
    for name in _PAYLOAD:
        buffers["payload"][name][dummy].zero_()
    _stage(torch, layer, inputs, buffers, 1, base)
    demand = _demand(torch, buffers, base)
    assert parts <= len(demand) <= slots
    for slot, expert in enumerate(demand):
        for name in _PAYLOAD:
            buffers["payload"][name][slot].copy_(layer[name][expert])
    for part in range(parts):
        buffers["map"].fill_(dummy)
        for slot, expert in enumerate(demand):
            if slot % parts == part:
                buffers["map"][expert] = slot
        _stage(torch, layer, inputs, buffers, 2, base)
    _stage(torch, layer, inputs, buffers, 3, base)
    for name, want in zip(("res_out", "out", "inj_out"), expected):
        assert torch.equal(buffers[name], want), name
    assert torch.ops._rocm_C.mono_k2_rdna2_error(buffers["ws"])[0].item() == 0


def _offload_mono(torch, helper, layer, slots):
    """One layer's production offload state: a real cache and native workspaces."""
    from vllm.model_executor.layers.fused_moe.rdna_expert_cache import (
        PAYLOAD_NAMES,
        PackedExpertCache,
    )

    host = {
        target: layer[source].detach().cpu().contiguous()
        for target, source in zip(PAYLOAD_NAMES, _PAYLOAD)
    }
    cache = PackedExpertCache(host, slots, torch.device("cuda"))
    tensors = dict(layer)
    tensors.update(
        {
            source: cache.payload[target]
            for target, source in zip(PAYLOAD_NAMES, _PAYLOAD)
        }
    )

    def workspace(shared_width):
        size = torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(
            helper.H, helper.R, helper.ND, helper.E, shared_width, helper.INTER
        )
        return torch.ops._rocm_C.mono_k2_rdna2_workspace_alloc(
            size + 8192, torch.accelerator.current_device_index()
        )

    return SimpleNamespace(
        expert_cache=cache,
        t=tensors,
        offload_ws=workspace(320),
        ws=workspace(160),
        base=256,
        eps=helper.EPS,
        layer=SimpleNamespace(layer_idx=0),
    )


@pytest.mark.parametrize("native_case", [256], indirect=True, ids=["tp2-ep2"])
@pytest.mark.parametrize(("rows", "slots"), [(1, 96), (3, 96), (8, 96), (3, 24)])
def test_resident_pass_runs_a_single_pass_demand_as_one_full_pass(
    native_case, monkeypatch, rows, slots
):
    """Hit, partial-hit and cold layers equal one pass over the demand, bit for bit.

    A demand that cannot outgrow the cache runs in two launches, the resident
    experts before the host has read the routing and the loads after them, each
    counting the whole demand through the cache's zero slot. A larger demand is
    planned once the host knows it, and runs as one pass when it fits.
    """
    torch, helper, layer, local_experts = native_case
    from vllm.models.qwen4_exp.amd.mono.rdna2 import OPTS, k2

    monkeypatch.setattr(OPTS, "timing", False)
    monkeypatch.setattr(OPTS, "k2_num_wg", 0)
    mono = _offload_mono(torch, helper, layer, slots)
    cache = mono.expert_cache
    loads, exact = [], 0
    for seed in (1300, 1300, 1301, 1302, 1301, 1303, 1303, 1304, 1302):
        inputs = helper._inputs(rows, seed)
        misses, passes = cache.expert_misses, cache.committed_passes
        output = k2.run_offloaded(mono, *inputs)
        torch.accelerator.synchronize()
        loads.append(cache.expert_misses - misses)
        expected = _full_reference(
            torch, helper, layer, inputs, local_experts, mono.base
        )
        assert torch.equal(output[0], expected[0])
        if cache.committed_passes - passes > 1:
            # A demand larger than the cache runs in several passes.
            for value, want in zip(output[1:], expected[1:]):
                torch.testing.assert_close(value, want, rtol=0, atol=2e-3)
        else:
            exact += 1
            assert torch.equal(output[1], expected[1]), f"seed={seed}"
            assert torch.equal(output[2], expected[2]), f"seed={seed}"
        assert torch.ops._rocm_C.mono_k2_rdna2_error(mono.offload_ws)[0].item() == 0
    assert loads[0] > 0 and loads[1] == 0, loads  # cold, then fully resident
    assert exact >= 6, "the fixture must exercise single-pass demands"
    assert loads[6] == 0, loads  # the same demand again is fully resident
    if slots == 96 and rows <= 3:
        assert loads[4] == 0, loads  # nothing was evicted in between
    elif slots < 96:
        # The small cache evicts: later demands mix hits and loads.
        assert 0 < cache.expert_misses < cache.expert_demands
        assert sum(loads) > loads[0], loads
    # Neither the loads nor the launches ever write the zero slot.
    assert all(not t[cache.dummy_slot].any() for t in cache.k2_payload.values())


@pytest.mark.parametrize("rows", [3, 12, 24])
def test_split_stage_graphs_read_new_routes_and_slot_contents(native_case, rows):
    torch, helper, layer, local_experts = native_case
    base = 512 - local_experts
    inputs = helper._inputs(rows, 601)
    buffers = _buffers(torch, helper, layer, rows, 3, local_experts)
    # Warm specialization and buffers before capturing each fixed-shape stage.
    for stage in (1, 2, 3):
        _stage(torch, layer, inputs, buffers, stage, base)
    torch.accelerator.synchronize()
    graphs = {}
    for stage in (1, 2, 3):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _stage(torch, layer, inputs, buffers, stage, base)
        graphs[stage] = graph

    def pointers():
        return tuple(
            tensor.data_ptr()
            for tensor in (
                *(buffers[name] for name in ("ids", "weights", "map", "ws", "out")),
                *(buffers["payload"][name] for name in _PAYLOAD),
            )
        )

    initial_pointers = pointers()
    prior_routes = None
    changed = False
    for seed in (602, 603, 602):
        for destination, source in zip(inputs, helper._inputs(rows, seed)):
            destination.copy_(source)
        expected = _full_reference(torch, helper, layer, inputs, local_experts, base)
        graphs[1].replay()
        routes = buffers["ids"].cpu().clone()
        changed |= prior_routes is not None and not torch.equal(routes, prior_routes)
        prior_routes = routes
        demand = _demand(torch, buffers, base)
        assert len(demand) > 3
        # Copies and changing host plans remain outside the captured graphs.
        # Reverse demand order to alter expert-to-slot placement across passes.
        demand.reverse()
        for start in range(0, len(demand), 3):
            _fill_pass(torch, layer, buffers, demand[start : start + 3])
            graphs[2].replay()
        graphs[3].replay()
        _assert_outputs(torch, buffers, expected)
        assert initial_pointers == pointers()
    assert changed, "graph replay must consume changed routing, not capture-time IDs"


@pytest.mark.parametrize("native_case", [256], indirect=True, ids=["tp2-ep2"])
def test_production_graph_orchestration_with_stale_capture_inputs(
    native_case, monkeypatch
):
    """Real host-bank copies and nested stage graphs must obey outer producers.

    Capture records the producer copy without executing it. Its destination is
    deliberately NaN when the expert eager break runs during capture; replay
    must instead observe the valid producer output, including across two row
    buckets using the same native scratch allocation.
    """
    torch, helper, layer, local_experts = native_case
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.qwen4_exp.amd.mono.rdna2 import OPTS, k2

    monkeypatch.setattr(OPTS, "timing", False)
    monkeypatch.setattr(OPTS, "k2_num_wg", 0)
    mono = _offload_mono(torch, helper, layer, 96)
    cache = mono.expert_cache

    def assert_control_clear(phase):
        for name in ("ws", "offload_ws"):
            record = torch.ops._rocm_C.mono_k2_rdna2_error(getattr(mono, name)).tolist()
            assert record[0] == 0, f"{phase}: {name} sticky fault {record}"

    def assert_one_full_pass(inputs, outputs):
        expected = _full_reference(
            torch, helper, layer, inputs, local_experts, mono.base
        )
        assert all(torch.equal(got, want) for got, want in zip(outputs, expected))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captures = []
    with torch.cuda.stream(stream):
        for rows in (6, 3):
            values = helper._inputs(rows, 800 + rows)
            source = tuple(
                torch.empty(
                    (value.shape[1], rows), dtype=value.dtype, device=value.device
                )
                .t()
                .copy_(value)
                for value in values
            )
            assert all(not value.is_contiguous() for value in source)
            # Warm native specializations and stable cache buffers eagerly.
            eager = k2.run_offloaded(mono, *source)
            torch.accelerator.synchronize()
            assert_control_clear(f"eager rows={rows}")
            _assert_independent_tp2(
                torch,
                helper,
                layer,
                tuple(value.contiguous() for value in source),
                dict(zip(("res_out", "out", "inj_out"), eager)),
                mono.base,
            )
            assert_one_full_pass(tuple(value.contiguous() for value in source), eager)
            produced_hidden = torch.full_like(source[0], float("nan"))
            torch.accelerator.synchronize()
            outer = BreakableCUDAGraphCapture(publish_replay_context=True)
            with outer:
                produced_hidden.copy_(source[0])
                output = k2.run_offloaded(mono, produced_hidden, source[1], source[2])
            torch.accelerator.synchronize()
            assert outer.num_eager_breaks > 0
            assert_control_clear(f"capture rows={rows}")
            captures.append(outer)  # Retain both descriptors and graph pools.
            for seed in (900 + rows, 1000 + rows):
                for destination, value in zip(source, helper._inputs(rows, seed)):
                    destination.copy_(value)
                produced_hidden.fill_(float("nan"))
                outer.replay()
                torch.accelerator.synchronize()
                assert_control_clear(f"replay rows={rows} seed={seed}")
                _assert_independent_tp2(
                    torch,
                    helper,
                    layer,
                    tuple(value.contiguous() for value in source),
                    dict(zip(("res_out", "out", "inj_out"), output)),
                    mono.base,
                )
                assert_one_full_pass(
                    tuple(value.contiguous() for value in source), output
                )
                assert all(bool(torch.isfinite(value).all()) for value in output)
        assert len(captures) == 2
        # A padded M=3 bucket must plan exactly its one real row after cache
        # churn. Native routes and graph shapes remain M=3 throughout.
        from collections import Counter

        import vllm.forward_context as forward_context_module

        context = SimpleNamespace(
            is_padding=torch.tensor([False, True, True], device="cuda"),
            additional_kwargs={},
        )
        monkeypatch.setattr(
            forward_context_module, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(
            forward_context_module, "get_forward_context", lambda: context
        )
        for seed, bank in ((1103, range(96)), (1203, range(96, 192))):
            lease = cache.ensure(bank)
            cache.release(lease)
            inputs = helper._inputs(3, seed)
            previous = cache.policy.frequencies
            result = k2.run_offloaded(mono, *inputs)
            torch.accelerator.synchronize()
            assert_control_clear(f"padded rows=3 seed={seed}")
            _assert_independent_tp2(
                torch,
                helper,
                layer,
                tuple(value[:1].contiguous() for value in inputs),
                dict(
                    zip(("res_out", "out", "inj_out"), (value[:1] for value in result))
                ),
                mono.base,
            )
            routes = mono._offload_buffers[(3, inputs[0].dtype, inputs[0].device)][3]
            demand = Counter(
                expert - mono.base
                for expert in routes[0].cpu().tolist()
                if mono.base <= expert < mono.base + local_experts
            )
            assert tuple(
                current - old
                for current, old in zip(cache.policy.frequencies, previous)
            ) == tuple(demand[expert] for expert in range(local_experts))
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("native_case", [128], indirect=True, ids=["tp4-geometry"])
@pytest.mark.parametrize("kernel", ["k1", "k2"])
def test_native_activation_adapters_pack_noncontiguous_inputs(native_case, kernel):
    torch, helper, layer, _ = native_case
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k1, k2

    size = torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(
        helper.H, helper.R, helper.ND, helper.E, 160, helper.INTER
    )
    mono = SimpleNamespace(
        t=layer,
        th=layer,
        ws=torch.ops._rocm_C.mono_k2_rdna2_workspace_alloc(
            size + 8192, torch.accelerator.current_device_index()
        ),
        base=384,
        eps=helper.EPS,
        eps_a=helper.EPS,
        is_mtp=False,
        dense_mtp=False,
    )
    inputs = helper._inputs(3, 1111)
    run = k1.run if kernel == "k1" else k2.run
    expected = tuple(value.clone() for value in run(mono, *inputs))
    strided = tuple(
        torch.empty((value.shape[1], 3), dtype=value.dtype, device=value.device)
        .t()
        .copy_(value)
        for value in inputs
    )
    assert all(not value.is_contiguous() for value in strided)
    actual = run(mono, *strided)
    for value, reference in zip(actual, expected):
        assert value.is_contiguous()
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    record = torch.ops._rocm_C.mono_k2_rdna2_error(mono.ws).tolist()
    assert record[0] == 0, f"{kernel}: sticky fault {record}"


@pytest.fixture
def offload_trace(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("VLLM_RDNA_EXPERT_TRACE_DIR", str(tmp_path))
    module = _load(
        "rdna2_offload_trace_test",
        _ROOT / "vllm/models/qwen4_exp/amd/mono/rdna2/offload_trace.py",
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    context = SimpleNamespace(
        attn_metadata={
            "layer.attn": SimpleNamespace(
                num_prefills=0,
                seq_lens=torch.tensor([41, 42, 43]),
                query_start_loc=torch.tensor([0, 1, 2, 3]),
            )
        },
        is_padding=torch.tensor([False, True, False]),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            get_forward_context=lambda: context,
        ),
    )
    return module, context


@pytest.mark.parametrize("gate", ["disabled", "capture", "raw", "unarmed"])
def test_offload_trace_gates_before_touching_inputs(offload_trace, monkeypatch, gate):
    module, _ = offload_trace
    capture = None
    if gate == "disabled":
        module.ENABLED = False
    elif gate == "capture":
        capture = SimpleNamespace(replaying=False)
    elif gate == "raw":
        monkeypatch.setattr(
            module.torch.cuda, "is_current_stream_capturing", lambda: True
        )
    else:
        monkeypatch.setenv(
            "VLLM_RDNA_EXPERT_TRACE_ARM_FILE", "/nonexistent/offload-arm"
        )
    assert module.begin(None, None, None, None, capture) is None
    assert module._COUNTS == {}


@pytest.mark.parametrize("gate", ["warmup", "prefill", "padding", "limit"])
def test_offload_trace_skips_nonreal_decode(offload_trace, monkeypatch, gate):
    module, context = offload_trace
    torch = module.torch
    if gate == "warmup":
        context.attn_metadata = None
    elif gate == "prefill":
        context.attn_metadata["layer.attn"].num_prefills = 1
    elif gate == "padding":
        context.is_padding.fill_(True)
    else:
        monkeypatch.setenv("VLLM_RDNA_EXPERT_TRACE_MAX_CALLS", "0")
    mono = SimpleNamespace(layer=SimpleNamespace(layer_idx=2))
    value = torch.arange(12).reshape(3, 4)
    assert module.begin(mono, value, value, value, None) is None
    assert module._COUNTS == {}


def test_offload_trace_replay_receipt_owns_only_real_rows(
    offload_trace, monkeypatch, tmp_path
):
    module, _ = offload_trace
    torch = module.torch
    arm = tmp_path / "armed"
    arm.touch()
    monkeypatch.setenv("VLLM_RDNA_EXPERT_TRACE_ARM_FILE", str(arm))
    monkeypatch.setenv("VLLM_RDNA_EXPERT_TRACE_MAX_CALLS", "1")
    monkeypatch.setenv("VLLM_RDNA_EXPERT_TRACE_MAX_ROWS", "1")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    # HIP's raw capture query can remain true at a safe Breakable replay hook.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    mono = SimpleNamespace(layer=SimpleNamespace(layer_idx=2))
    value = torch.arange(12).reshape(3, 4)
    original = value.clone()
    receipt = module.begin(mono, value, value, value, SimpleNamespace(replaying=True))
    value.add_(100)
    module.finish(receipt, value, value, value, value, value)
    saved = torch.load(tmp_path / "rank3-layer2-call000000.pt", weights_only=True)
    assert saved["version"] == 1 and saved["call"] == 0
    assert saved["rank"] == 3 and saved["layer"] == 2
    assert saved["rows"].tolist() == [0] and saved["padded_rows"] == 3
    assert torch.equal(saved["tensors"]["residual"], original[:1])
    assert torch.equal(saved["tensors"]["output"], value[:1])
    assert saved["metadata"]["layer.attn"]["seq_lens"].tolist() == [41, 42, 43]
    assert module.begin(mono, value, value, value, None) is None
    assert not list(tmp_path.glob("*.tmp"))


def test_offload_gdn_trace_snapshots_exact_bound_state(
    offload_trace, monkeypatch, tmp_path
):
    module, _ = offload_trace
    torch = module.torch
    monkeypatch.setenv("VLLM_RDNA_GDN_TRACE_MAX_CALLS", "1")
    monkeypatch.setenv("VLLM_RDNA_GDN_TRACE_LAYER", "1")
    monkeypatch.setenv("VLLM_RDNA_GDN_TRACE_SEQ_LEN", "41")
    monkeypatch.setenv("VLLM_RDNA_EXPERT_TRACE_MAX_CALLS", "0")
    layer = SimpleNamespace(prefix="model.layers.1.linear_attn")
    md = SimpleNamespace(spec_sequence_masks=None)
    value = torch.arange(12).reshape(3, 4)
    conv = torch.arange(20).reshape(5, 4)
    ssm = torch.arange(40, dtype=torch.float32).reshape(5, 2, 4)
    original_conv, original_ssm = conv.clone(), ssm.clone()
    indices = torch.tensor([3, 0, 1], dtype=torch.int32)
    monkeypatch.setenv("VLLM_RDNA_GDN_TRACE_SEQ_LEN", "42")
    assert module.begin_gdn(layer, md, value, value, conv, ssm, indices, None) is None
    assert module._COUNTS == {}
    monkeypatch.setenv("VLLM_RDNA_GDN_TRACE_SEQ_LEN", "41")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    receipt = module.begin_gdn(
        layer, md, value, value, conv, ssm, indices, SimpleNamespace(replaying=True)
    )
    conv.add_(10)
    ssm.add_(20)
    module.finish_gdn(receipt, value, conv, ssm)
    saved = torch.load(tmp_path / "gdn-rank0-layer1-call000000.pt", weights_only=True)
    tensors = saved["tensors"]
    assert saved["kind"] == "gdn" and saved["rows"].tolist() == [0, 2]
    assert tensors["state_indices"].tolist() == [3, 1]
    assert torch.equal(tensors["conv_before"], original_conv[[3, 1]])
    assert torch.equal(tensors["ssm_before"], original_ssm[[3, 1]])
    assert torch.equal(tensors["conv_after"], conv[[3, 1]])
    assert torch.equal(tensors["ssm_after"], ssm[[3, 1]])
    assert torch.equal(tensors["mixed_qkvz"], value[[0, 2]])
    assert module.begin_gdn(layer, md, value, value, conv, ssm, indices, None) is None
    assert (
        module.begin_gdn(
            SimpleNamespace(prefix="model.layers.2.linear_attn"),
            md,
            value,
            value,
            conv,
            ssm,
            indices,
            None,
        )
        is None
    )
