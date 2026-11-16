# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership, checkpoint arithmetic and paged-state contracts for unequal GDN."""

from dataclasses import FrozenInstanceError
from math import prod
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def gdn(monkeypatch):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import gdn_parallel, linear

    # Real Parameters, loaders and linear apply; only discovery/GEMM dispatch
    # are CPU seams, following test_qsa_parallel.
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *args, **kwargs: lambda layer, x, weight, bias: F.linear(
            x, weight, bias
        ),
    )
    return gdn_parallel


@pytest.fixture
def dense_gdn(gdn, monkeypatch):
    from vllm.model_executor.layers import rdna_dense_int8

    monkeypatch.setattr(rdna_dense_int8, "_ENABLED", True)
    monkeypatch.setenv("VLLM_RDNA_DENSE_INT8_MIN_ROWS", "64")
    monkeypatch.setattr(gdn.current_platform, "is_rocm", lambda: True)
    return rdna_dense_int8


def _geometry(gdn, rank, dim=4, ba=False):
    partition = gdn.gdn_head_partition(16, 48, dim, dim, rank, 3)
    if ba:
        sizes = (48, 48)
        slices = ((partition.value_start, partition.value_count),) * 2
    else:
        sizes = (16 * dim, 16 * dim, 48 * dim, 48 * dim)
        slices = (
            (partition.key_start * dim, partition.key_dim),
            (partition.key_start * dim, partition.key_dim),
            (partition.value_start * dim, partition.value_dim),
            (partition.value_start * dim, partition.value_dim),
        )
    return partition, sizes, slices


def _rows(sizes, slices):
    offset = 0
    rows = []
    for size, (start, count) in zip(sizes, slices):
        rows.append(torch.arange(offset + start, offset + start + count))
        offset += size
    return torch.cat(rows)


def _load_projection(layer, source, sizes, packing):
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    root = torch.nn.Module()
    root.proj = layer
    components = source.split(sizes)
    if packing == "fused":
        weights, mapper = [("proj.weight", source)], None
    elif packing == "qkv_z":
        weights = [
            ("qkv.weight", torch.cat(components[:3])),
            ("z.weight", components[3]),
        ]
        mapper = WeightsMapper(
            orig_to_new_stacked={
                "qkv.weight": ("proj.weight", (0, 1, 2)),
                "z.weight": ("proj.weight", 3),
            }
        )
    else:
        weights = [
            (f"component{i}.weight", value) for i, value in enumerate(components)
        ]
        mapper = WeightsMapper(
            orig_to_new_stacked={
                f"component{i}.weight": ("proj.weight", i) for i in range(len(sizes))
            }
        )
    assert AutoWeightsLoader(root).load_weights(weights, mapper=mapper) == {
        "proj.weight"
    }


def _quantize_global(source):
    weight = source.half()
    scale = weight.abs().amax(dim=1).float().clamp_min(1e-8) / 127
    codes = torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


def test_partition_owns_six_five_five_whole_key_groups_without_padding(gdn):
    key_owners = torch.zeros(16, dtype=torch.int64)
    value_owners = torch.zeros(48, dtype=torch.int64)
    for rank, expected in enumerate(((0, 6, 0, 18), (6, 5, 18, 15), (11, 5, 33, 15))):
        p = gdn.gdn_head_partition(16, 48, 128, 128, rank, 3)
        assert (p.key_start, p.key_count, p.value_start, p.value_count) == expected
        assert p.head_k_dim == p.head_v_dim == 128
        assert p.value_start == 3 * p.key_start
        assert p.value_count == 3 * p.key_count
        assert (p.key_dim, p.value_dim, p.conv_dim) == (
            (768, 2304, 3840) if rank == 0 else (640, 1920, 3200)
        )
        key_owners[p.key_start : p.key_start + p.key_count] += 1
        value_owners[p.value_start : p.value_start + p.value_count] += 1
        with pytest.raises(FrozenInstanceError):
            p.key_count = 99
    assert torch.equal(key_owners, torch.ones_like(key_owners))
    assert torch.equal(value_owners, torch.ones_like(value_owners))
    p = gdn.gdn_head_partition(16, 48, 128, 128, 0, 1)
    assert (p.key_start, p.key_count, p.value_start, p.value_count) == (0, 16, 0, 48)


@pytest.mark.parametrize(
    "args",
    [
        (0, 48, 128, 128, 0, 3),
        (16, 47, 128, 128, 0, 3),
        (16, 8, 128, 128, 0, 3),
        (2, 6, 128, 128, 0, 3),
        (16, 48, 0, 128, 0, 3),
        (16, 48, 128, -1, 0, 3),
        (16, 48, 128, 128, -1, 3),
        (16, 48, 128, 128, 3, 3),
        (16, 48, 128, 128, 0, 0),
    ],
)
def test_invalid_group_geometry_is_rejected(gdn, args):
    with pytest.raises((ValueError, TypeError)):
        gdn.gdn_head_partition(*args)


@pytest.mark.parametrize("packing", ["fused", "qkv_z", "components"])
def test_qkvz_rows_and_arithmetic_reconstruct_checkpoint_once(gdn, packing):
    source = torch.arange(512 * 16, dtype=torch.float32).reshape(512, 16) / 1001
    x = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 31
    expected = F.linear(x, source)
    reconstructed = torch.empty_like(expected)
    owners = torch.zeros(512, dtype=torch.int64)
    for rank in range(3):
        _, sizes, slices = _geometry(gdn, rank)
        layer = gdn.GDNPackedColumnParallelLinear(16, sizes, slices)
        _load_projection(layer, source, sizes, packing)
        rows = _rows(sizes, slices)
        torch.testing.assert_close(layer.weight, source[rows])
        actual, bias = layer(x)
        assert bias is None
        torch.testing.assert_close(actual, expected[:, rows])
        reconstructed[:, rows] = actual
        owners[rows] += 1
    assert torch.equal(owners, torch.ones_like(owners))
    torch.testing.assert_close(reconstructed, expected)


@pytest.mark.parametrize("packing", ["fused", "components"])
def test_ba_remains_component_major_for_unequal_value_owners(gdn, packing):
    source = torch.arange(96 * 16, dtype=torch.float32).reshape(96, 16) / 503
    x = torch.arange(48, dtype=torch.float32).reshape(3, 16) / 47
    reconstructed = torch.empty(3, 96)
    owners = torch.zeros(96, dtype=torch.int64)
    for rank, count in enumerate((36, 30, 30)):
        _, sizes, slices = _geometry(gdn, rank, ba=True)
        layer = gdn.GDNPackedColumnParallelLinear(16, sizes, slices)
        _load_projection(layer, source, sizes, packing)
        rows = _rows(sizes, slices)
        assert layer.weight.shape == (count, 16)
        torch.testing.assert_close(layer.weight, source[rows])
        actual, _ = layer(x)
        reconstructed[:, rows] = actual
        owners[rows] += 1
    assert torch.equal(owners, torch.ones_like(owners))
    torch.testing.assert_close(reconstructed, F.linear(x, source))


@pytest.mark.parametrize("reduce_results", [False, True])
def test_output_owns_matching_value_columns_and_reduces_once(
    gdn, monkeypatch, reduce_results
):
    generator = torch.Generator().manual_seed(73)
    source = torch.randn(7, 192, generator=generator)
    x = torch.randn(2, 3, 192, generator=generator)
    total = F.linear(x, source)
    contributions = []
    owners = torch.zeros(192, dtype=torch.int64)
    for rank in range(3):
        p, _, _ = _geometry(gdn, rank)
        start, count = p.value_start * 4, p.value_dim
        layer = gdn.GDNOutputParallelLinear(192, 7, p, reduce_results=reduce_results)
        layer.weight.weight_loader(layer.weight, source)
        torch.testing.assert_close(layer.weight, source[:, start : start + count])
        local_x = x[..., start : start + count]
        contribution = F.linear(local_x, source[:, start : start + count])
        calls = []

        def reduce(value, calls=calls, contribution=contribution):
            calls.append(value.clone())
            torch.testing.assert_close(value, contribution)
            return total.clone()

        monkeypatch.setattr(gdn, "tensor_model_parallel_all_reduce", reduce)
        actual, bias = layer(local_x)
        assert bias is None
        assert len(calls) == int(reduce_results)
        torch.testing.assert_close(actual, total if reduce_results else contribution)
        contributions.append(contribution)
        owners[start : start + count] += 1
    assert torch.equal(owners, torch.ones_like(owners))
    torch.testing.assert_close(torch.stack(contributions).sum(0), total)


@pytest.mark.parametrize("per_value", [False, True])
def test_conv_and_value_parameters_load_each_original_element_once(gdn, per_value):
    source = (
        torch.arange(48, dtype=torch.float32)
        if per_value
        else torch.arange(320 * 4, dtype=torch.float32).reshape(320, 1, 4)
    )
    restored = torch.empty_like(source)
    owners = torch.zeros(source.shape[0], dtype=torch.int64)
    for rank in range(3):
        p, sizes, slices = _geometry(gdn, rank)
        sizes, slices = (
            ((48,), ((p.value_start, p.value_count),))
            if per_value
            else (sizes[:3], slices[:3])
        )
        rows = _rows(sizes, slices)
        parameter = torch.nn.Parameter(
            torch.empty_like(source[rows]), requires_grad=False
        )
        loader = gdn.gdn_sharded_weight_loader(sizes, slices)
        loader(parameter, source)
        torch.testing.assert_close(parameter, source[rows])
        restored[rows] = parameter.detach()
        owners[rows] += 1
    assert torch.equal(owners, torch.ones_like(owners))
    assert torch.equal(restored, source)


@pytest.mark.parametrize(
    "shard,shape",
    [
        (None, (511, 16)),
        (None, (512, 15)),
        (None, (512, 16, 1)),
        ((0, 1, 2), (319, 16)),
        ((0, 1, 2), (320, 15)),
        (3, (191, 16)),
        (4, (192, 16)),
        ((0, 1, 4), (320, 16)),
    ],
)
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_packed_malformed_sources_are_rejected_before_any_write(
    gdn, rank, shard, shape
):
    _, sizes, slices = _geometry(gdn, rank)
    layer = gdn.GDNPackedColumnParallelLinear(16, sizes, slices)
    layer.weight.data.fill_(37)
    before = layer.weight.detach().clone()
    with pytest.raises((ValueError, TypeError)):
        layer.weight.weight_loader(layer.weight, torch.ones(shape), shard)
    assert torch.equal(layer.weight, before)


@pytest.mark.parametrize(
    "kind,shape",
    [
        ("conv", (319, 1, 4)),
        ("conv", (320, 4)),
        ("conv", (320, 1, 3)),
        ("value", (47,)),
        ("value", (48, 1)),
        ("output", (6, 192)),
        ("output", (7, 191)),
        ("output", (7, 192, 1)),
    ],
)
def test_other_malformed_sources_preserve_parameters_and_statistics(gdn, kind, shape):
    p, sizes, slices = _geometry(gdn, 1)
    if kind == "output":
        layer = gdn.GDNOutputParallelLinear(192, 7, p)
        parameter = layer.weight
        loader = parameter.weight_loader
    else:
        sizes, slices, local_shape = (
            ((48,), ((p.value_start, p.value_count),), (15,))
            if kind == "value"
            else (sizes[:3], slices[:3], (p.conv_dim, 1, 4))
        )
        parameter = torch.nn.Parameter(torch.empty(local_shape), requires_grad=False)
        loader = gdn.gdn_sharded_weight_loader(sizes, slices)
    parameter.data.fill_(37)
    before = parameter.detach().clone()
    with pytest.raises((ValueError, TypeError)):
        loader(parameter, torch.ones(shape))
    assert torch.equal(parameter, before)
    if kind == "output":
        assert not hasattr(layer, "_rdna_i8_row_absmax")


@pytest.mark.parametrize("kind", ["packed", "output"])
def test_non_dense_quantization_is_rejected_without_substitution(gdn, kind):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Unsupported:
        def create_weights(self, *args, **kwargs):
            UnquantizedLinearMethod().create_weights(*args, **kwargs)

    quant = SimpleNamespace(
        get_quant_method=lambda *args, **kwargs: Unsupported(),
        online_quantization_config=None,
    )
    p, sizes, slices = _geometry(gdn, 0)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        if kind == "packed":
            gdn.GDNPackedColumnParallelLinear(16, sizes, slices, quant_config=quant)
        else:
            gdn.GDNOutputParallelLinear(192, 7, p, quant_config=quant)


@pytest.mark.parametrize("packing", ["fused", "qkv_z"])
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_qkvz_and_ba_bytes_match_global_quantize_then_slice(
    gdn, dense_gdn, packing, source_dtype
):
    generator = torch.Generator().manual_seed(127)
    for ba in (False, True):
        source = torch.randn(96 if ba else 16384, 32, generator=generator).to(
            source_dtype
        )
        source[0].zero_()
        source[1].fill_(1e-6)
        expected_codes, expected_scale = _quantize_global(source)
        for rank in range(3):
            _, sizes, slices = _geometry(gdn, rank, dim=128, ba=ba)
            layer = gdn.GDNPackedColumnParallelLinear(32, sizes, slices).half()
            _load_projection(layer, source, sizes, "components" if ba else packing)
            assert dense_gdn._eligible(layer)
            if ba:
                assert layer.weight.shape[0] == (36 if rank == 0 else 30)
            rows = _rows(sizes, slices)
            codes, scales = dense_gdn._quantize_weight(layer.weight)
            assert torch.equal(codes, expected_codes[rows])
            assert torch.equal(scales, expected_scale[rows])
            x = torch.arange(64, dtype=torch.float16).reshape(2, 32) / 127
            actual = dense_gdn.linear_released(x, codes, scales, None)
            reference_weight = (
                expected_codes[rows].float() * expected_scale[rows, None].float()
            ).half()
            torch.testing.assert_close(actual, F.linear(x, reference_weight))


@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_output_uses_full_fp16_k_absmax_and_reload_replaces_it(
    gdn, dense_gdn, source_dtype
):
    generator = torch.Generator().manual_seed(193)
    source = torch.randn(80, 6144, generator=generator).to(source_dtype)
    source[:, 2304:4224] *= 19
    source[0].zero_()
    for rank in range(3):
        p, _, _ = _geometry(gdn, rank, dim=128)
        layer = gdn.GDNOutputParallelLinear(6144, 80, p).half()
        start, count = p.value_start * 128, p.value_dim
        for loaded in (source, source.flip(1).contiguous() * 3):
            layer.weight.weight_loader(layer.weight, loaded)
            assert dense_gdn._eligible(layer)
            expected_codes, expected_scale = _quantize_global(loaded)
            codes, scales = dense_gdn._quantize_weight(
                layer.weight, layer._rdna_i8_row_absmax
            )
            assert torch.equal(codes, expected_codes[:, start : start + count])
            assert torch.equal(scales, expected_scale)
            assert torch.equal(
                layer._rdna_i8_row_absmax, loaded.half().abs().amax(dim=1).float()
            )
        before = layer.weight.detach().clone()
        statistics = layer._rdna_i8_row_absmax.clone()
        with pytest.raises((ValueError, TypeError)):
            layer.weight.weight_loader(layer.weight, source[:, :count])
        assert torch.equal(layer.weight, before)
        assert torch.equal(layer._rdna_i8_row_absmax, statistics)


def test_int8_unaligned_local_width_keeps_real_unquantized_fallback(gdn, dense_gdn):
    p = gdn.gdn_head_partition(16, 48, 3, 3, 1, 3)
    layer = gdn.GDNOutputParallelLinear(144, 80, p, reduce_results=False).half()
    source = torch.arange(80 * 144, dtype=torch.float16).reshape(80, 144) / 8192
    layer.weight.weight_loader(layer.weight, source)
    assert not dense_gdn._eligible(layer)
    assert not hasattr(layer, "_rdna_i8_row_absmax")
    x = torch.ones(2, 45, dtype=torch.float16)
    actual, _ = layer(x)
    torch.testing.assert_close(actual, F.linear(x, source[:, 54:99]))


@pytest.fixture
def gdn_state_layer(gdn, monkeypatch):
    from vllm.config import CompilationConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm.model_executor.layers.mamba.gdn import base, qwen_gdn_linear_attn
    from vllm.models.qwen4_exp.amd import model as amd_model

    class UnusedChunkBackend(torch.nn.Module):
        gdn_prefill_backend = "triton"

    # No attention kernels execute here. Keep the actual layer constructor,
    # projections, state calculator, MambaSpec and raw-storage binder.
    monkeypatch.setattr(qwen_gdn_linear_attn, "ChunkGatedDeltaRule", UnusedChunkBackend)
    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(qwen_gdn_linear_attn.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(qwen_gdn_linear_attn, "on_rdna_family", lambda: True)
    monkeypatch.setattr(qwen_gdn_linear_attn, "on_rdna2", lambda: True)
    monkeypatch.setattr(qwen_gdn_linear_attn, "MONO_GDN_HOOK", None)
    config = SimpleNamespace(
        hidden_size=16,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16, hf_text_config=config),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
            mamba_block_size=16,
            mamba_page_size_padded=None,
            mamba_cache_mode="none",
            use_replayssm=False,
            replayssm_state_dtype="auto",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=3),
        quant_config=None,
        kernel_config=SimpleNamespace(linear_backend="auto"),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        num_speculative_tokens=2,
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn, "get_current_vllm_config", lambda: runtime
    )
    monkeypatch.setattr(base, "get_tensor_model_parallel_world_size", lambda: 3)

    def make(
        rank,
        layout,
        prefix=None,
        *,
        head_dim=128,
        state_dtype="auto",
        use_replayssm=False,
    ):
        config.linear_key_head_dim = config.linear_value_head_dim = head_dim
        runtime.cache_config.replayssm_state_dtype = state_dtype
        runtime.cache_config.use_replayssm = use_replayssm
        monkeypatch.setattr(base, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(mamba_utils, "get_conv_state_layout", lambda: layout)
        # Each constructor represents a different worker, whose compilation
        # registry may legitimately contain the same layer name.
        runtime.compilation_config = CompilationConfig(mode=0, custom_ops=["none"])
        with set_current_vllm_config(runtime):
            layer = qwen_gdn_linear_attn.QwenGatedDeltaNetAttention(
                config,
                runtime,
                prefix=prefix or f"model.layers.{rank}.linear_attn",
                unequal_tp=True,
            )
        return layer, runtime

    return make, amd_model.Qwen4ExpForCausalLM


@pytest.mark.parametrize(
    "overrides",
    [
        {"head_dim": 3},
        {"head_dim": 64},
        {"state_dtype": "leapquant"},
        {"use_replayssm": True},
    ],
)
def test_unsupported_unequal_cache_geometry_fails_before_projection_allocation(
    gdn_state_layer, monkeypatch, overrides
):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    make_layer, _ = gdn_state_layer

    def allocation_must_not_run(*args, **kwargs):
        raise AssertionError("unsupported geometry reached projection allocation")

    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "GDNPackedColumnParallelLinear",
        allocation_must_not_run,
    )
    with pytest.raises(ValueError, match="K=V=128"):
        make_layer(0, "SD", **overrides)


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_real_state_binding_uses_local_geometry_inside_shared_maximum_page(
    gdn_state_layer, layout
):
    make_layer, model_class = gdn_state_layer
    from vllm.v1.kv_cache_interface import (
        KVCacheLayout,
        KVCacheTensor,
        MambaSpec,
        create_kv_cache_views,
    )

    for rank, (conv_dim, heads) in enumerate(((3840, 18), (3200, 15), (3200, 15))):
        layer, runtime = make_layer(rank, layout)
        assert (
            layer.num_k_heads,
            layer.num_v_heads,
            layer.key_dim,
            layer.value_dim,
            layer.conv_dim,
        ) == (16, 48, 2048, 6144, 10240)
        assert (layer.local_conv_dim, layer.local_num_v_heads) == (conv_dim, heads)
        assert layer.conv1d.weight.shape == (conv_dim, 1, 4)
        assert layer.A_log.shape == layer.dt_bias.shape == (heads,)
        conv_shape = (5, conv_dim) if layout == "SD" else (conv_dim, 5)
        assert layer.get_state_shape() == (conv_shape, (heads, 128, 128))
        envelope = model_class.get_gdn_mamba_state_shape_from_config(runtime)
        assert envelope == ((5, 3840) if layout == "SD" else (3840, 5), (18, 128, 128))
        maximum_page = MambaSpec(
            block_size=16, shapes=envelope, dtypes=(torch.float16, torch.float32)
        ).page_size_bytes
        runtime.cache_config.mamba_page_size_padded = maximum_page
        spec = layer.get_kv_cache_spec(runtime)
        assert isinstance(spec, MambaSpec)
        assert spec.shapes == envelope
        assert spec.page_size_bytes == maximum_page
        assert spec.real_page_size_bytes == maximum_page
        assert spec.num_speculative_blocks == 2
        assert not spec.tp_replicated
        guard, blocks = 64, 3
        raw = torch.full((2 * guard + blocks * maximum_page,), 85, dtype=torch.int8)
        tensor = KVCacheTensor(
            size=raw.numel(),
            layers=[layer.prefix],
            layer_stride=blocks * maximum_page,
            block_stride=maximum_page,
            offset=guard,
        )
        views = create_kv_cache_views(raw, spec, blocks, KVCacheLayout.LBHNC, tensor)
        layer.bind_kv_cache(views[0])
        expected = raw.clone()
        offset = guard + maximum_page
        for index, (state, shape, dtype) in enumerate(
            zip(layer.kv_cache, layer.get_state_shape(), spec.dtypes)
        ):
            assert state.shape == (blocks, *shape)
            assert state.dtype == dtype
            assert state.stride(0) * state.element_size() == maximum_page
            value = rank + index + 1
            state[1].fill_(value)
            content = torch.full(shape, value, dtype=dtype).view(torch.int8).reshape(-1)
            nbytes = prod(shape) * state.element_size()
            expected[offset : offset + nbytes] = content
            offset += nbytes
        # Entire sentinel block 0, unscheduled block 2, guard bytes and the
        # smaller owners' unowned tail bytes must remain byte-for-byte intact.
        assert torch.equal(raw, expected)


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("cache_mode", ["none", "all", "align"])
def test_same_layer_tp3_planner_allocates_canonical_pages_but_binds_local_states(
    gdn_state_layer, monkeypatch, layout, cache_mode
):
    """Catch the centralized planner rejecting unequal specs for one TP layer.

    Equal padded byte counts alone cannot satisfy its same-layer equality
    guard; canonical declarations must also survive allocation and local binding.
    """
    from dataclasses import replace

    from vllm.v1.core.kv_cache_planning import (
        get_global_kv_cache_groups,
        get_kv_cache_configs,
    )
    from vllm.v1.kv_cache_interface import KVCacheLayout
    from vllm.v1.worker import utils as worker_utils

    make_layer, model_class = gdn_state_layer
    name = "model.layers.0.linear_attn"
    workers = [make_layer(rank, layout, prefix=name) for rank in range(3)]
    layers = [layer for layer, _ in workers]
    runtime = workers[0][1]
    envelope = model_class.get_gdn_mamba_state_shape_from_config(runtime)
    runtime.cache_config.mamba_cache_mode = cache_mode
    runtime.cache_config.num_gpu_blocks_override = None
    runtime.cache_config.prefix_cache_retention_interval = None
    runtime.cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout.LBHNC
    runtime.model_config.original_max_model_len = 16
    runtime.model_config.max_model_len = 16
    runtime.model_config.supports_kv_cache_grouping = False
    runtime.scheduler_config.disable_hybrid_kv_cache_manager = False
    runtime.parallel_config.decode_context_parallel_size = 1
    runtime.attention_config = SimpleNamespace(hisparse_config=None)
    runtime.num_prefill_lookahead_tokens = 0
    runtime.speculative_config.use_eagle_block_drop = lambda: False

    specs = [{name: layer.get_kv_cache_spec(runtime)} for layer in layers]
    page_size = specs[0][name].page_size_bytes
    runtime.cache_config.mamba_page_size_padded = page_size
    specs = [{name: layer.get_kv_cache_spec(runtime)} for layer in layers]
    blocks = 5  # null block + up to four resident MTP2 state blocks
    memory = [blocks * page_size] * 3

    # Reproduce the old consumer failure with genuine local ownership and
    # identical padded page sizes, not a mocked planner result.
    local_specs = [
        {name: replace(spec[name], shapes=layer.get_state_shape())}
        for layer, spec in zip(layers, specs)
    ]
    with pytest.raises(AssertionError, match="same layer are different across workers"):
        get_global_kv_cache_groups(runtime, local_specs)

    groups = get_global_kv_cache_groups(runtime, specs)
    plans = get_kv_cache_configs(runtime, specs, memory, groups)
    assert len(plans) == 3
    # The constructor's ROCm discovery seams must not warm GPU workspaces
    # when the real allocator is deliberately given a CPU device.
    monkeypatch.setattr(worker_utils.current_platform, "is_rocm", lambda: False)
    for rank, (layer, spec_by_name, plan) in enumerate(zip(layers, specs, plans)):
        spec = spec_by_name[name]
        assert spec.shapes == envelope
        assert spec.dtypes == (torch.float16, torch.float32)
        assert spec.block_size == 16
        assert spec.mamba_cache_mode == cache_mode
        assert spec.num_speculative_blocks == 2
        assert not spec.tp_replicated
        assert plan.num_blocks == blocks
        assert plan.kv_tp_replicas == 1
        (tensor,) = plan.kv_cache_tensors
        assert tensor.layers == [name]
        assert tensor.block_stride == page_size
        caches = worker_utils.allocate_kv_cache(
            plan, torch.device("cpu"), KVCacheLayout.LBHNC
        )
        pages = caches[name]
        assert pages.dtype == torch.int8
        assert pages.shape == (blocks, 1, 1, page_size)
        pages.fill_(85)
        expected = pages.clone()
        layer.bind_kv_cache(pages)
        conv_dim, heads = ((3840, 18), (3200, 15), (3200, 15))[rank]
        conv_shape = (5, conv_dim) if layout == "SD" else (conv_dim, 5)
        assert layer.get_state_shape() == (conv_shape, (heads, 128, 128))
        offset = 0
        for index, (state, shape, dtype) in enumerate(
            zip(layer.kv_cache, layer.get_state_shape(), spec.dtypes)
        ):
            assert state.shape == (blocks, *shape)
            assert state.dtype == dtype
            assert state.stride(0) * state.element_size() == tensor.block_stride
            assert (
                state.untyped_storage().data_ptr() == pages.untyped_storage().data_ptr()
            )
            value = rank + index + 1
            state[1].fill_(value)
            content = torch.full(shape, value, dtype=dtype).view(torch.int8).reshape(-1)
            nbytes = prod(shape) * state.element_size()
            expected[1, 0, 0, offset : offset + nbytes] = content
            offset += nbytes
        # Includes all other blocks and the smaller owners' unowned page tail.
        assert torch.equal(pages, expected)


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_local_state_copy_specs_and_arenas_preserve_unowned_page_bytes(
    gdn_state_layer, layout
):
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
    )
    from vllm.v1.attention.backends.gdn_attn import (
        alloc_gdn_state_arenas,
        gather_gdn_state_arenas,
        scatter_gdn_state_arenas,
    )

    make_layer, _ = gdn_state_layer
    for rank in range(3):
        layer, runtime = make_layer(rank, layout)
        spec = layer.get_kv_cache_spec(runtime)
        page_size = spec.page_size_bytes
        pages = torch.full((5, 1, 1, page_size), 85, dtype=torch.int8)
        layer.bind_kv_cache(pages)
        conv, ssm = layer.kv_cache
        conv[1].fill_(rank + 1)
        ssm[1].fill_(rank + 2)
        expected = pages.clone()

        # Copy lengths are the owner's natural state size, not the common
        # planner page stride; SD additionally advances by accepted tokens.
        for accepted in (1, 2, 3) if layout == "SD" else (1,):
            conv_copy = get_conv_copy_spec(conv, [1, 2, 3], 0, accepted)
            source = conv[1, accepted - 1 :] if layout == "SD" else conv[1]
            assert conv_copy.start_addr == source.data_ptr()
            assert conv_copy.num_elements == source.numel()
            temporal_copy = get_temporal_copy_spec(ssm, [1, 2, 3], 0, accepted)
            assert temporal_copy.start_addr == ssm[accepted].data_ptr()
            assert temporal_copy.num_elements == ssm[accepted].numel()
            assert temporal_copy.num_elements * ssm.element_size() < page_size

        conv_arena, ssm_arena = alloc_gdn_state_arenas(
            2,
            conv.shape[1:],
            ssm.shape[1:],
            conv.dtype,
            ssm.dtype,
            torch.device("cpu"),
        )
        slots = torch.tensor([1, 2], dtype=torch.int32)
        gather_gdn_state_arenas(conv, ssm, conv_arena, ssm_arena, slots, 2)
        torch.testing.assert_close(conv_arena[1:3], conv[slots])
        torch.testing.assert_close(ssm_arena[1:3], ssm[slots])
        conv_arena[1].fill_(rank + 3)
        ssm_arena[1].fill_(rank + 4)
        scatter_gdn_state_arenas(conv, ssm, conv_arena, ssm_arena, slots, 2)
        offset = 0
        for state, value in ((conv, rank + 3), (ssm, rank + 4)):
            content = (
                torch.full(state.shape[1:], value, dtype=state.dtype)
                .view(torch.int8)
                .reshape(-1)
            )
            expected[1, 0, 0, offset : offset + content.numel()] = content
            offset += content.numel()
        assert torch.equal(pages, expected)
        assert not conv_arena[0].any()
        assert not ssm_arena[0].any()
