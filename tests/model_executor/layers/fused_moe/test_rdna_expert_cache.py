# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-bank contracts without loading the platform-dependent vLLM package."""

import ast
import builtins
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cache_module(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    directory = root / "vllm/model_executor/layers/fused_moe"
    policy_name = "vllm.model_executor.layers.fused_moe.rdna_expert_cache_policy"
    for name, path in (
        (policy_name, directory / "rdna_expert_cache_policy.py"),
        ("rdna_expert_cache_storage_test", directory / "rdna_expert_cache.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def _bank(module, experts=256):
    return {
        name: torch.arange(experts * 6, dtype=torch.int32).reshape(experts, 2, 3)
        + offset * 10000
        for offset, name in enumerate(module.PAYLOAD_NAMES)
    }


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("history", [(0, 1, 2), (2, 1, 0)])
def test_cached_reduction_rounds_once_after_all_passes(
    cache_module, monkeypatch, capacity, history
):
    """FP16 group sums lose the half-unit; integer accumulation preserves it."""
    source = (
        Path(__file__).resolve().parents[4]
        / "vllm/model_executor/layers/quantization/compressed_tensors"
        / "compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_cached_moe"
    )
    function.decorator_list = []
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.rdna_expert_cache",
        cache_module,
    )
    contributions = torch.tensor(
        [[2048 << 32], [-(2048 << 32)], [1 << 31]], dtype=torch.int64
    )

    def run_fixed_pass(x, weights, remapped, *, routed_accumulator, **kwargs):
        active = remapped.flatten().ge(0)
        assert routed_accumulator.dtype == torch.int64
        routed_accumulator[:, :-1].add_(contributions[active].sum(0))
        return routed_accumulator

    def cast(accumulator, output):
        output.copy_((accumulator[:, :-1].float() / (1 << 32)).to(output.dtype))

    namespace = {
        "torch": torch,
        "_rdna2_fused_moe": run_fixed_pass,
        "ops": SimpleNamespace(moe_fixedpoint_to_half_rdna2=cast),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    cache = cache_module.PackedExpertCache(_bank(cache_module, 3), capacity, "cpu")
    for expert in history:
        cache.release(cache.ensure([expert]))
    layer = SimpleNamespace(
        _rdna_expert_cache=cache, expert_map=None, apply_router_weight_on_input=False
    )
    x = torch.zeros(1, 1, dtype=torch.half)
    ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    result = torch.empty_like(x)
    namespace["_apply_cached_moe"](result, layer, x, torch.ones(1, 3), ids, None)
    assert result.item() == 0.5


def test_six_payloads_move_together_and_slots_keep_storage(cache_module):
    bank = _bank(cache_module)
    cache = cache_module.PackedExpertCache(bank, 2, "cpu")
    pointers = {name: t.data_ptr() for name, t in cache.payload.items()}
    for experts in ([255, 2], [2, 9], [17, 255]):
        lease = cache.ensure(experts)
        for expert, slot, _ in lease.bindings:
            for name in bank:
                assert torch.equal(cache.payload[name][slot], bank[name][expert])
        assert {name: t.data_ptr() for name, t in cache.payload.items()} == pointers
        assert cache.slot_map.ge(0).sum().item() == 2
        cache.release(lease)
    assert cache.logical_num_experts == 256
    assert cache.slot_count == 2
    expected = sum(t.numel() * t.element_size() for t in bank.values())
    assert cache.host_bytes == expected + 256 * 4
    # Two slots and the kernel's all-zero dummy slot.
    assert cache.device_bytes == expected * 3 // 256 + 256 * 4


def test_hit_has_no_payload_copy_and_map_excludes_inactive_residents(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module), 2, "cpu")
    first = cache.ensure([3, 5])
    cache.release(first)
    copied = cache.copied_bytes
    second = cache.ensure([5])
    assert cache.copied_bytes == copied
    assert cache.slot_map[3].item() == -1
    assert cache.slot_map[5].item() >= 0
    assert cache.policy.mapping[3] >= 0
    cache.release(second)
    maps = cache.map_uploads
    third = cache.ensure([5])
    assert cache.map_uploads == maps
    cache.release(third)
    assert (cache.expert_demands, cache.expert_misses, cache.committed_passes) == (
        4,
        2,
        3,
    )


def test_global_local_slot_mapping_preserves_padding_and_pass_coverage(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 4), 2, "cpu")
    expert_map = torch.tensor([-1, 0, -1, 1, 2, -1, 3], dtype=torch.int32)
    routed = torch.tensor([[1, 3, 4], [6, 1, -1], [0, 8, 3]])
    local = cache.local_ids(routed, expert_map)
    assert local.tolist() == [[0, 1, 2], [3, 0, -1], [-1, -1, 1]]
    ids = local.flatten().tolist()
    cache.policy.observe(ids)
    appearances = torch.zeros_like(local)
    for experts in cache.policy.plan_passes(ids):
        lease = cache.ensure(experts)
        remapped = cache.remap(local, lease)
        appearances += remapped.ge(0)
        for expert, slot, _ in lease.bindings:
            assert torch.equal(remapped.eq(slot), local.eq(expert))
        cache.release(lease)
    assert torch.equal(appearances, local.ge(0).long())
    assert cache.policy.frequencies == (2, 2, 1, 1)


def test_failed_copy_invalidates_victim_before_retry(cache_module, monkeypatch):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 3), 1, "cpu")
    first = cache.ensure([0])
    cache.release(first)
    original = cache._copy

    def fail(bindings):
        original(bindings)
        raise RuntimeError("copy failed after overwriting the victim")

    monkeypatch.setattr(cache, "_copy", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        cache.ensure([1])
    assert cache.policy.mapping == {}
    monkeypatch.setattr(cache, "_copy", original)
    retry = cache.ensure([0])
    assert retry.bindings == ((0, 0, True),)
    cache.release(retry)


def test_kernel_payload_ends_in_an_all_zero_dummy_slot_nothing_else_touches(
    cache_module,
):
    bank = _bank(cache_module, 8)
    cache = cache_module.PackedExpertCache(bank, 3, "cpu")
    assert cache.dummy_slot == 3
    for name in bank:
        storage = cache.k2_payload[name]
        assert storage.shape[0] == 4 and cache.payload[name].shape[0] == 3
        assert cache.payload[name].data_ptr() == storage.data_ptr()
        assert not storage[3].any()
    for experts in ([0, 1, 2], [5, 6, 7], [1, 2, 3]):
        cache.release(cache.ensure(experts))
        assert not any(t[3].any() for t in cache.k2_payload.values())
    # A cache that holds every expert has no room for one and needs no padding.
    for slots in (8, 20):
        full = cache_module.PackedExpertCache(bank, slots, "cpu")
        assert full.slot_count == 8 and full.dummy_slot is None
        for name in bank:
            assert full.k2_payload[name].shape == full.payload[name].shape
            assert full.k2_payload[name].data_ptr() == full.payload[name].data_ptr()


def _constant_qzero_bank(module, experts=24):
    bank = _bank(module, experts)
    for name in module.QZERO_NAMES:
        bank[name].fill_(module.QZERO_VALUE)
    return bank


def _learned_bytes(module, bank):
    return sum(
        bank[name][0].numel() * bank[name][0].element_size()
        for name in module.PAYLOAD_NAMES
        if name not in module.QZERO_NAMES
    )


def test_constant_qzeros_copy_only_the_four_learned_payloads(cache_module):
    bank = _constant_qzero_bank(cache_module, 16)
    cache = cache_module.PackedExpertCache(bank, 4, "cpu", constant_qzeros=True)
    assert cache.constant_qzeros is True
    learned = _learned_bytes(cache_module, bank)
    full = sum(t[0].numel() * t[0].element_size() for t in bank.values())
    zeros = sum(
        bank[name][0].numel() * bank[name][0].element_size()
        for name in cache_module.QZERO_NAMES
    )
    assert full == learned + zeros and cache.copied_bytes == 0
    expected = 0
    for experts in ([0, 1, 2, 3], [2, 9], [3, 12, 13], [0], [5, 6, 7, 8]):
        before = cache.expert_misses
        lease = cache.ensure(experts)
        expected += (cache.expert_misses - before) * learned
        assert cache.copied_bytes == expected
        for expert, slot, _ in lease.bindings:
            for name in bank:
                assert torch.equal(cache.payload[name][slot], bank[name][expert])
        cache.release(lease)
    assert expected == cache.copied_bytes == cache.expert_misses * learned
    baseline = cache_module.PackedExpertCache(bank, 4, "cpu")
    baseline_lease = baseline.ensure([0, 1, 2, 3])
    baseline.release(baseline_lease)
    assert baseline.copied_bytes == 4 * full


@pytest.mark.parametrize("seed", range(2))
def test_constant_qzeros_every_slot_matches_the_baseline_through_reuse(
    cache_module, seed
):
    rng = random.Random(seed)
    experts, slots = 12, 3
    bank = _constant_qzero_bank(cache_module, experts)
    enabled = cache_module.PackedExpertCache(bank, slots, "cpu", constant_qzeros=True)
    baseline = cache_module.PackedExpertCache(bank, slots, "cpu")
    dummy = enabled.dummy_slot
    assert dummy == slots == baseline.dummy_slot
    seen: set[int] = set()
    demands = [[0, 1, 2]] + [
        rng.sample(range(experts), rng.randint(1, slots)) for _ in range(80)
    ]
    for demand in demands:
        enabled_lease = enabled.ensure(demand)
        baseline_lease = baseline.ensure(demand)
        assert enabled_lease.bindings == baseline_lease.bindings
        assert torch.equal(enabled.slot_map, baseline.slot_map)
        seen.update(slot for _, slot, _ in enabled_lease.bindings)
        for name in cache_module.PAYLOAD_NAMES:
            assert torch.equal(enabled.payload[name], baseline.payload[name])
            assert torch.equal(enabled.k2_payload[name], baseline.k2_payload[name])
        for name in cache_module.QZERO_NAMES:
            assert not enabled.k2_payload[name][dummy].any()
        enabled.release(enabled_lease)
        baseline.release(baseline_lease)
    assert seen == set(range(slots))


@pytest.mark.parametrize("invalid", ["dtype", "value"])
def test_constant_qzeros_refuse_a_host_zero_point_that_is_not_constant(
    cache_module, invalid
):
    bank = _constant_qzero_bank(cache_module, 8)
    if invalid == "dtype":
        bank["w2_qzeros"] = bank["w2_qzeros"].to(torch.int64)
    else:
        bank["w13_qzeros"][6, 0, 0] = cache_module.QZERO_VALUE - 1
    with pytest.raises(ValueError, match="qzeros"):
        cache_module.PackedExpertCache(bank, 2, "cpu", constant_qzeros=True)
    # The default cache still accepts the bank and copies all six payloads.
    assert cache_module.PackedExpertCache(bank, 2, "cpu").constant_qzeros is False


def test_a_constant_qzeros_miss_leaves_the_slot_zero_point_row_alone(cache_module):
    bank = _constant_qzero_bank(cache_module, 8)
    cache = cache_module.PackedExpertCache(bank, 1, "cpu", constant_qzeros=True)
    marker = cache_module.QZERO_VALUE ^ 0x0F
    for name in cache_module.QZERO_NAMES:
        cache.payload[name][0].fill_(marker)
    lease = cache.ensure([3])
    assert lease.bindings == ((3, 0, True),)
    for name in cache_module.PAYLOAD_NAMES:
        if name in cache_module.QZERO_NAMES:
            assert cache.payload[name][0].eq(marker).all()
        else:
            assert torch.equal(cache.payload[name][0], bank[name][3])
    cache.release(lease)


def test_constant_qzeros_reject_the_fixed_six_copy_native_path(cache_module):
    bank = _constant_qzero_bank(cache_module, 8)
    with pytest.raises(ValueError, match="expert_cache_copy=native"):
        cache_module.PackedExpertCache(
            bank, 2, "cpu", copy_mode="native", constant_qzeros=True
        )
    with pytest.raises(ValueError, match="boolean"):
        cache_module.PackedExpertCache(bank, 2, "cpu", constant_qzeros=1)


def test_constant_qzeros_option_is_opt_in_and_paired_with_tensor_copies(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "mono_rdna2_options_test",
        root / "vllm/models/qwen4_exp/amd/mono/rdna2/__init__.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "mono_rdna2_options_test", module)
    spec.loader.exec_module(module)
    assert module.Options().expert_cache_constant_qzeros is False
    opts = module.Options.from_dict(
        {"expert_cache_slots": 115, "expert_cache_constant_qzeros": True}
    )
    assert opts.expert_cache_constant_qzeros is True
    assert opts.expert_cache_copy == "tensor"
    for extra, message in (
        ({"expert_cache_constant_qzeros": 1}, "boolean"),
        ({"expert_cache_constant_qzeros": "true"}, "boolean"),
        (
            {"expert_cache_constant_qzeros": True, "expert_cache_copy": "native"},
            "tensor",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            module.Options.from_dict(extra)


@pytest.mark.parametrize("seed", range(4))
def test_residency_maps_follow_every_kind_of_pass(cache_module, seed):
    """The first pass reads ``resident_map`` before the host plans the loads."""
    rng = random.Random(seed)
    cache = cache_module.PackedExpertCache(_bank(cache_module, 24), 5, "cpu")
    dummy = cache.dummy_slot
    assert dummy == 5
    cache.start_tracking()
    for _ in range(200):
        experts = rng.sample(range(24), rng.randint(1, 5))
        by_residency = rng.random() < 0.5
        before = dict(cache.policy.mapping)
        pass_map = cache.slot_map.tolist()
        lease = cache.ensure(experts, by_residency=by_residency)
        loads = {expert for expert, _, miss in lease.bindings if miss}
        assert loads == {expert for expert in experts if expert not in before}
        if by_residency:
            assert cache.slot_map.tolist() == pass_map
        else:
            # slot_map masks in exactly what this pass runs.
            assert {e for e in range(24) if cache.slot_map[e] >= 0} == set(experts)
        assert cache.resident_map.tolist() == [
            cache.policy.mapping.get(expert, dummy) for expert in range(24)
        ]
        if loads:
            assert cache.miss_map.tolist() == [
                cache.policy.mapping[expert] if expert in loads else dummy
                for expert in range(24)
            ]
        cache.release(lease)


def test_the_two_maps_of_a_demand_run_each_expert_once_and_count_it_twice(
    cache_module,
):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 16), 6, "cpu")
    dummy = cache.dummy_slot
    cache.start_tracking()
    cache.release(cache.ensure([1, 2, 3], by_residency=True))
    demand = [2, 9, 3, 12, 1]
    first = cache.resident_map.tolist()  # what the pass before the host plan reads
    lease = cache.ensure(demand, by_residency=True)
    second = cache.miss_map.tolist()  # what the pass after the loads reads
    final = cache.resident_map.tolist()
    run_first = {e for e in demand if first[e] != dummy}
    run_second = {e for e in demand if second[e] != dummy}
    assert run_first == {1, 2, 3}
    assert run_second == {9, 12}
    # Both passes count the whole demand, with the real slot or the dummy one.
    assert all(first[e] >= 0 and second[e] >= 0 for e in demand)
    assert all(first[e] == final[e] for e in run_first)
    assert all(second[e] == final[e] for e in run_second)
    cache.release(lease)


def test_a_pass_on_the_residency_maps_needs_tracking(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 8), 3, "cpu")
    with pytest.raises(RuntimeError, match="start_tracking"):
        cache.ensure([1], by_residency=True)
    assert cache.policy.mapping == {}


def test_tracking_starts_from_the_current_residency(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 8), 3, "cpu")
    dummy = cache.dummy_slot
    cache.release(cache.ensure([1, 4]))
    assert cache.resident_map.tolist() == [dummy] * 8  # nothing reads it yet
    assert cache.resident_uploads == 0
    cache.start_tracking()
    cache.start_tracking()
    assert cache.resident_uploads == 1
    assert cache.resident_map.tolist() == [
        cache.policy.mapping.get(expert, dummy) for expert in range(8)
    ]
    cache.release(cache.ensure([6]))
    assert cache.resident_map[6].item() == cache.policy.mapping[6]
    uploads = cache.resident_uploads
    cache.release(cache.ensure([6]))  # a hit changes no residency
    assert cache.resident_uploads == uploads


def test_a_cache_without_a_dummy_slot_marks_absent_experts_unmapped(cache_module):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 4), 4, "cpu")
    assert cache.dummy_slot is None
    cache.start_tracking()
    assert cache.resident_map.tolist() == [-1] * 4
    cache.release(cache.ensure([2], by_residency=True))
    assert cache.resident_map.tolist() == [-1, -1, 0, -1]


def test_failed_load_forgets_the_overwritten_resident(cache_module, monkeypatch):
    cache = cache_module.PackedExpertCache(_bank(cache_module, 3), 1, "cpu")
    dummy = cache.dummy_slot
    assert dummy == 1
    cache.start_tracking()
    cache.release(cache.ensure([0]))
    assert cache.resident_map.tolist() == [0, dummy, dummy]
    original = cache._copy

    def fail(bindings):
        original(bindings)
        raise RuntimeError("copy failed after overwriting the victim")

    monkeypatch.setattr(cache, "_copy", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        cache.ensure([1])
    assert cache.resident_map.tolist() == [dummy] * 3
    monkeypatch.setattr(cache, "_copy", original)
    cache.release(cache.ensure([0]))
    assert cache.resident_map.tolist() == [0, dummy, dummy]


def test_host_preparation_keeps_shuffle_bounded_to_one_expert(cache_module):
    experts = 5
    packed = torch.arange(experts * 16 * 8, dtype=torch.int32).reshape(experts, 16, 8)
    scales = torch.arange(experts * 16 * 8, dtype=torch.float16).reshape(experts, 16, 8)
    layer = SimpleNamespace(
        w13_weight_packed=packed,
        w2_weight_packed=packed.clone(),
        w13_weight_scale=scales,
        w2_weight_scale=scales.clone(),
    )
    calls = []

    def shuffle(row, bits):
        assert row.shape == (8, 16)
        assert row.is_contiguous()
        calls.append(bits)
        row.add_(1)

    host = cache_module.prepare_host_payload(layer, 8, "cpu", shuffle)
    assert calls == [4] * (experts * 2)
    assert torch.equal(host["w13_weight_packed"], packed.transpose(1, 2) + 1)
    assert torch.equal(host["w2_weight_scale"], scales.transpose(1, 2))
    assert host["w13_qzeros"].shape == (experts, 8, 2)
    assert host["w13_qzeros"].eq(0x77777777).all()
    assert set(host) == set(cache_module.PAYLOAD_NAMES)
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in host.values())
    assert packed[0, 0, 0].item() == 0


@pytest.mark.parametrize(
    ("active", "raw_capture", "expected"),
    [
        (None, False, False),
        (None, True, True),
        (SimpleNamespace(capturing_segment=False), True, False),
        (SimpleNamespace(capturing_segment=True), False, True),
    ],
)
def test_capture_guard_distinguishes_eager_breaks_from_raw_graphs(
    cache_module, monkeypatch, active, raw_capture, expected
):
    original = builtins.__import__

    def import_breakable(name, *args, **kwargs):
        if name == "vllm.compilation.breakable_cudagraph":
            return SimpleNamespace(
                BreakableCUDAGraphCapture=SimpleNamespace(current=lambda: active)
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_breakable)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: raw_capture)
    assert cache_module.is_capturing_expert_work() is expected


def _loading_config(monkeypatch, slots):
    config = SimpleNamespace(
        additional_config={
            "rocm_mono_decode": {
                "expert_cache_slots": slots,
                "expert_cache_policy": "lru",
                "expert_cache_decay": 64,
            }
        },
        kernel_config=SimpleNamespace(monokernel=True),
        model_config=SimpleNamespace(
            architectures=["Qwen4ExpForConditionalGeneration"]
        ),
    )
    modules = {
        "vllm.config": SimpleNamespace(get_current_vllm_config_or_none=lambda: config),
        "vllm.platforms": SimpleNamespace(
            current_platform=SimpleNamespace(is_rocm=lambda: True)
        ),
        "vllm.platforms.rocm": SimpleNamespace(on_gfx10x=lambda: True),
    }
    original = builtins.__import__

    def import_config(name, *args, **kwargs):
        return modules[name] if name in modules else original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_config)
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cuda:0"))
    cls = type("CompressedTensorsWNA16RDNA2MoEMethod", (), {})
    method = cls()
    method.symmetric = True
    method.num_bits = 4
    method.moe = SimpleNamespace(
        in_dtype=torch.float16,
        has_bias=False,
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(enable_eplb=False),
    )
    return config, method


def test_loading_context_stages_only_enabled_native_experts(cache_module, monkeypatch):
    config, method = _loading_config(monkeypatch, 2)
    layer = SimpleNamespace()
    with cache_module.cpu_weight_context(method, layer):
        assert torch.empty(1).device.type == "cpu"
    assert layer._rdna_expert_cache_slots == 2
    assert layer._rdna_expert_cache_device == torch.device("cuda:0")
    assert layer._rdna_expert_cache_policy == "lru"
    assert layer._rdna_expert_cache_decay == 64
    assert layer._rdna_expert_cache_constant_qzeros is False
    assert method.requires_device_loading is False
    config.additional_config["rocm_mono_decode"]["expert_cache_slots"] = 0
    untouched = SimpleNamespace()
    with cache_module.cpu_weight_context(method, untouched):
        pass
    assert vars(untouched) == {}


def test_loading_context_propagates_constant_qzeros(cache_module, monkeypatch):
    config, method = _loading_config(monkeypatch, 2)
    knobs = config.additional_config["rocm_mono_decode"]
    knobs["expert_cache_constant_qzeros"] = True
    layer = SimpleNamespace()
    with cache_module.cpu_weight_context(method, layer):
        assert torch.empty(1).device.type == "cpu"
    assert layer._rdna_expert_cache_constant_qzeros is True
    knobs["expert_cache_constant_qzeros"] = 1
    rejected = SimpleNamespace()
    with (
        pytest.raises(ValueError, match="true or false"),
        cache_module.cpu_weight_context(method, rejected),
    ):
        pytest.fail("a non-boolean constant qzeros flag must not begin staging")
    assert vars(rejected) == {}


@pytest.mark.parametrize("unsupported", ["symmetric", "dtype", "eplb", "model"])
def test_loading_context_rejects_unsupported_expert_contracts(
    cache_module, monkeypatch, unsupported
):
    config, method = _loading_config(monkeypatch, 2)
    if unsupported == "symmetric":
        method.symmetric = False
    elif unsupported == "dtype":
        method.moe.in_dtype = torch.bfloat16
    elif unsupported == "eplb":
        method.moe.moe_parallel_config.enable_eplb = True
    else:
        config.model_config.architectures = ["OtherMoEForCausalLM"]
    layer = SimpleNamespace()
    with pytest.raises(ValueError), cache_module.cpu_weight_context(method, layer):
        pytest.fail("unsupported expert loading must not begin")
    assert vars(layer) == {}


@pytest.mark.parametrize("copy_mode", ["tensor", "native"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_device_slot_reuse_waits_for_previous_consumer(cache_module, copy_mode):
    if copy_mode == "native":
        if torch.version.hip is None:
            pytest.skip("native expert copies require HIP")
        import vllm._custom_ops  # noqa: F401

    bank = _bank(cache_module, 4)
    bank = {
        name: value.half() if "scale" in name else value for name, value in bank.items()
    }
    cache = cache_module.PackedExpertCache(bank, 1, "cuda", copy_mode=copy_mode)
    assert all(t.is_pinned() for t in cache.host_payload.values())
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    copies = []
    maps = []
    pointers = {name: t.data_ptr() for name, t in cache.payload.items()}
    experts = (0, 1, 3, 0, 2, 1, 1)
    for index, expert in enumerate(experts):
        with torch.cuda.stream(streams[index % 2]):
            lease = cache.ensure([expert])
            copies.append(
                {name: value[0].clone() for name, value in cache.payload.items()}
            )
            maps.append(cache.slot_map.clone())
            cache.release(lease)
    for stream in streams:
        stream.synchronize()
    for result, mapping, expert in zip(copies, maps, experts, strict=True):
        for name, value in result.items():
            assert torch.equal(value.cpu(), bank[name][expert])
        expected = torch.full((4,), -1, dtype=torch.int32)
        expected[expert] = 0
        assert torch.equal(mapping.cpu(), expected)
    assert {name: t.data_ptr() for name, t in cache.payload.items()} == pointers
    assert cache.native_copy_calls == (6 if copy_mode == "native" else 0)
    assert 0 <= cache.batch_copy_calls <= cache.native_copy_calls


@pytest.mark.parametrize("copy_mode", ["unknown", "native"])
def test_cpu_cache_rejects_unsupported_copy_modes(cache_module, copy_mode):
    with pytest.raises(ValueError):
        cache_module.PackedExpertCache(
            _bank(cache_module, 4), 1, "cpu", copy_mode=copy_mode
        )


def test_worker_copy_control_requires_quiescence_and_keeps_pool_storage(
    cache_module, monkeypatch
):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "expert_cache_worker_test", root / "tools/rdna/mono/expert_cache_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = cache_module.PackedExpertCache(_bank(cache_module, 4), 2, "cpu")
    cached_module = SimpleNamespace(_rdna_expert_cache=cache)
    model = SimpleNamespace(modules=lambda: [cached_module, cached_module])
    worker = module.ExpertCacheWorker()
    worker.get_model = lambda: model
    synchronizations = []
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: synchronizations.append(True)
    )
    pointers = {name: value.data_ptr() for name, value in cache.payload.items()}
    lease = cache.ensure([0, 1])
    with pytest.raises(RuntimeError, match="released leases"):
        worker.rdna_expert_cache_control("tensor")
    assert not synchronizations
    cache.release(lease)
    result = worker.rdna_expert_cache_control("tensor")
    assert result["caches"] == 1
    assert result["expert_demands"] == result["expert_misses"] == 2
    assert result["copy_modes"] == ["tensor"]
    assert synchronizations == [True]
    assert pointers == {name: value.data_ptr() for name, value in cache.payload.items()}
    assert cache.policy.mapping == {0: 0, 1: 1}
    options = SimpleNamespace(timing=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.qwen4_exp.amd.mono.rdna2",
        SimpleNamespace(OPTS=options),
    )
    result = worker.rdna_expert_cache_control(finite_checks="on")
    assert result["debug_timing"] is options.timing is True
    assert cache.policy.mapping == {0: 0, 1: 1}
    assert pointers == {name: value.data_ptr() for name, value in cache.payload.items()}
    worker.rdna_expert_cache_control(finite_checks="off")
    assert options.timing is False
    worker.model_runner = SimpleNamespace(_mono_mtp_failstop=True)
    failstop = SimpleNamespace(_USE_TP_COLLECTIVE=True)
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.amd.mono.rdna2.failstop", failstop
    )
    for mode in ("wait", "off", "collective"):
        result = worker.rdna_expert_cache_control(post_step_checks=mode)
        assert result["post_step_checks"] == mode
        assert worker.model_runner._mono_mtp_failstop is (mode != "off")
        assert failstop._USE_TP_COLLECTIVE is (mode != "wait")
    assert pointers == {name: value.data_ptr() for name, value in cache.payload.items()}


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires HIP"
)
@pytest.mark.parametrize("invalid", ["source", "slot", "duplicate", "shape"])
def test_native_copy_validates_entire_pass_before_writing(cache_module, invalid):
    import vllm._custom_ops  # noqa: F401

    cache = cache_module.PackedExpertCache(_bank(cache_module, 4), 2, "cuda")
    for payload in cache.payload.values():
        payload.fill_(-1)
    sources = list(cache.host_payload.values())
    destinations = list(cache.payload.values())
    experts, slots = [0, 1], [0, 1]
    if invalid == "source":
        experts[-1] = 4
    elif invalid == "slot":
        slots[-1] = 2
    elif invalid == "duplicate":
        slots[-1] = 0
    else:
        destinations[-1] = torch.empty((2, 2, 4), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops._rocm_C.rdna2_expert_cache_copy(sources, destinations, experts, slots)
    torch.accelerator.synchronize()
    assert all(payload.eq(-1).all().item() for payload in cache.payload.values())


@pytest.mark.parametrize("max_calls", [2, 10])
def test_worker_trace_replays_exact_misses_from_initial_cache_state(
    cache_module, max_calls
):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "expert_trace_worker_test", root / "tools/rdna/mono/expert_cache_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = cache_module.PackedExpertCache(_bank(cache_module, 4), 2, "cpu")

    def demand(ids):
        cache.policy.observe(ids)
        for experts in cache.policy.plan_passes(ids):
            cache.release(cache.ensure(experts))

    demand([3, 3, 1])
    original = cache.policy.observe
    cached_module = SimpleNamespace(_rdna_expert_cache=cache)
    worker = module.ExpertCacheWorker()
    worker.get_model = lambda: SimpleNamespace(
        named_modules=lambda: [("model.layers.0.experts", cached_module)]
    )
    worker.rdna_expert_cache_trace("start", max_calls=max_calls, layers=[0])
    calls = [[2, 2, -1], [0, 1, 2], [3], [], [1, 3]]
    for ids in calls:
        demand(ids)
    record = worker.rdna_expert_cache_trace("stop")["layers"][0]
    assert cache.policy.observe == original
    assert record["seen_calls"] == len(calls)
    assert record["ids"] == calls[:max_calls]
    assert record["final_misses"] == cache.expert_misses

    replay = cache_module.ExpertCachePolicy(
        record["local_experts"],
        record["capacity"],
        record["policy"],
        record["decay_interval"],
    )
    replay.epoch = record["initial_epoch"]
    replay._frequency = record["initial_frequencies"][:]
    replay._last_used = record["initial_last_used"][:]
    replay._expert_slots = record["initial_mapping"].copy()
    for expert, slot in replay.mapping.items():
        replay._slot_experts[slot] = expert
    misses = record["initial_misses"]
    for index, ids in enumerate(record["ids"]):
        assert record["misses_before"][index] == misses
        replay.observe(ids)
        for experts in replay.plan_passes(ids):
            lease = replay.acquire(experts)
            misses += sum(miss for _, _, miss in lease.bindings)
            lease.commit()
            lease.release()
    assert misses == record["captured_end_misses"]
    seen = record["seen_calls"]
    demand([0])
    assert record["seen_calls"] == seen


def test_worker_rejects_native_switch_for_constant_qzeros(cache_module):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "expert_constant_worker_test", root / "tools/rdna/mono/expert_cache_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = cache_module.PackedExpertCache(
        _constant_qzero_bank(cache_module), 2, "cpu", constant_qzeros=True
    )
    cached_module = SimpleNamespace(_rdna_expert_cache=cache)
    worker = module.ExpertCacheWorker()
    worker.get_model = lambda: SimpleNamespace(modules=lambda: [cached_module])
    with pytest.raises(ValueError, match="constant expert qzeros require tensor"):
        worker.rdna_expert_cache_control("native")
    assert cache._copy_op is None


def test_worker_trace_byte_cap_keeps_an_exact_complete_forward_prefix(
    cache_module, monkeypatch
):
    import json

    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "expert_bounded_worker_test", root / "tools/rdna/mono/expert_cache_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_TRACE_BYTES_LIMIT", 60)
    caches = [
        cache_module.PackedExpertCache(_bank(cache_module, 4), 1, "cpu")
        for _ in range(2)
    ]
    worker = module.ExpertCacheWorker()
    worker.get_model = lambda: SimpleNamespace(
        named_modules=lambda: [
            (f"model.layers.{index}.experts", SimpleNamespace(_rdna_expert_cache=cache))
            for index, cache in enumerate(caches)
        ]
    )
    worker.rdna_expert_cache_trace("start", max_calls=128)
    for step in range(3):
        for cache in caches:
            ids = [step] + [-1] * 9
            cache.policy.observe(ids)
            for experts in cache.policy.plan_passes(ids):
                cache.release(cache.ensure(experts))
    result = worker.rdna_expert_cache_trace("stop")
    assert result["raw_bytes"] == 60
    assert result["byte_limit_reached"]
    assert [len(record["ids"]) for record in result["layers"]] == [2, 1]
    assert [record["captured_end_misses"] for record in result["layers"]] == [2, 1]
    assert [cache.expert_misses for cache in caches] == [3, 3]

    spec = importlib.util.spec_from_file_location(
        "expert_prefetch_replay_test", root / "tools/rdna/mono/ram10_prefetch.py"
    )
    assert spec is not None and spec.loader is not None
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    response = json.loads(json.dumps({"results": [result]}))
    measured = replay.replay(response, warmup_calls=0)
    assert measured["native_replay_exact"]
    assert measured["scored_calls_per_layer"] == 1
    assert measured["unscored_tail_calls"] == 1
    assert measured["scored"]["demand_misses"] == 2


def test_worker_reports_physical_kv_storage_without_double_counting_views():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "kv_memory_worker_test", root / "tools/rdna/mono/expert_cache_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packed_page = torch.zeros(64, dtype=torch.uint8)
    recurrent_page = torch.zeros(7, dtype=torch.int64)
    worker = module.ExpertCacheWorker()
    worker.model_runner = SimpleNamespace(
        kv_caches=[
            packed_page[:10],
            packed_page[12:],
            recurrent_page[1:],
            packed_page.view(8, 8),
        ]
    )
    measured = worker.rdna_cache_memory()
    assert measured["kv_cache_storage_bytes"] == 120
    assert measured["kv_cache_storages"] == 2
    assert measured["kv_cache_views"] == 4
    assert measured["kv_cache_devices"] == ["cpu"]
