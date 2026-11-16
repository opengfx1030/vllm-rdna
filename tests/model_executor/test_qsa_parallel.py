# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership/loader arithmetic; no distributed or device initialization."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def qsa(monkeypatch):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear, qsa_parallel

    # Keep the actual linear parameters, loaders and apply method; replace only
    # backend selection and distributed discovery, as in the CPU fuser tests.
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *args, **kwargs: lambda layer, x, weight, bias: F.linear(
            x, weight, bias
        ),
    )
    return qsa_parallel


def _sources(hidden=16, heads=8, kv=2, dim=4):
    # Distinct values for both halves of each per-head [Q, gate] pair.
    rows = 2 * (heads + kv) * dim
    return (
        torch.arange(1, rows * hidden + 1, dtype=torch.float32).reshape(rows, hidden)
        / 101
    )


def _row_ids(partition, heads=8, kv=2, dim=4):
    q = torch.arange(
        2 * partition.q_start * dim, 2 * (partition.q_start + partition.q_count) * dim
    )
    k = torch.arange(
        2 * heads * dim + partition.kv_start * dim,
        2 * heads * dim + (partition.kv_start + partition.kv_count) * dim,
    )
    v = k + kv * dim
    return torch.cat((q, k, v))


def _load_qkv(layer, source, separate):
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    root = torch.nn.Module()
    root.proj = layer
    if separate:
        q, k, v = source.split((64, 8, 8))
        weights = [("q.weight", q), ("k.weight", k), ("v.weight", v)]
        mapper = WeightsMapper(
            orig_to_new_stacked={
                f"{shard}.weight": ("proj.weight", shard) for shard in ("q", "k", "v")
            }
        )
    else:
        weights, mapper = [("proj.weight", source)], None
    assert AutoWeightsLoader(root).load_weights(weights, mapper=mapper) == {
        "proj.weight"
    }


@pytest.mark.parametrize("ordinal", [0, 1, 2])
@pytest.mark.parametrize("separate", [False, True])
def test_rotating_qkv_reconstructs_original_head_gate_and_kv_rows(
    qsa, ordinal, separate
):
    source = _sources()
    x = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 31
    expected = F.linear(x, source)
    reconstructed = torch.empty_like(expected)
    ownership = torch.zeros(source.shape[0], dtype=torch.int64)
    for rank in range(3):
        partition = qsa.qsa_head_partition(8, 2, ordinal, rank, 3)
        slot = (rank - ordinal) % 3
        assert (
            partition.q_start,
            partition.q_count,
            partition.kv_start,
            partition.kv_count,
        ) == ((slot * 4, 4, slot, 1) if slot < 2 else (8, 0, 2, 0))
        layer = qsa.QSAQKVParallelLinear(16, 8, 2, 4, partition=partition)
        _load_qkv(layer, source, separate)
        rows = _row_ids(partition)
        torch.testing.assert_close(layer.weight, source[rows])
        local, bias = layer(x)
        assert bias is None
        torch.testing.assert_close(local, expected[:, rows])
        reconstructed[:, rows] = local
        ownership[rows] += 1
        if slot == 2:
            assert sum(p.numel() for p in layer.parameters()) == 0
            assert sum(b.numel() for b in layer.buffers()) == 0
    assert torch.equal(ownership, torch.ones_like(ownership))
    torch.testing.assert_close(reconstructed, expected)


@pytest.mark.parametrize("ordinal", [0, 1, 2])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_output_owns_original_columns_and_reduces_exactly_once(
    qsa, monkeypatch, ordinal, reduce_results
):
    from vllm.model_executor.models.utils import AutoWeightsLoader

    generator = torch.Generator().manual_seed(51)
    source = torch.randn(7, 32, generator=generator)
    x = torch.randn(2, 3, 32, generator=generator)
    total = F.linear(x, source)
    locals_ = []
    for rank in range(3):
        partition = qsa.qsa_head_partition(8, 2, ordinal, rank, 3)
        start, width = partition.q_start * 4, partition.q_count * 4
        layer = qsa.QSAOutputParallelLinear(
            32, 7, 4, partition=partition, reduce_results=reduce_results
        )
        assert AutoWeightsLoader(layer).load_weights([("weight", source)]) == {"weight"}
        torch.testing.assert_close(layer.weight, source[:, start : start + width])
        local_x = x[..., start : start + width]
        contribution = F.linear(local_x, source[:, start : start + width])
        calls = []

        def reduce(value, calls=calls, contribution=contribution):
            calls.append(value.clone())
            torch.testing.assert_close(value, contribution)
            return total.clone()

        monkeypatch.setattr(qsa, "tensor_model_parallel_all_reduce", reduce)
        actual, bias = layer(local_x)
        assert bias is None
        torch.testing.assert_close(actual, total if reduce_results else contribution)
        assert len(calls) == int(reduce_results)
        if not reduce_results:
            locals_.append(actual)
        if not width:
            assert sum(p.numel() for p in layer.parameters()) == 0
            assert sum(b.numel() for b in layer.buffers()) == 0
    if not reduce_results:
        torch.testing.assert_close(torch.stack(locals_).sum(0), total)


def test_empty_owners_consume_keys_without_gemm_or_stale_output(qsa, monkeypatch):
    partition = qsa.qsa_head_partition(8, 2, 0, 2, 3)
    qkv = qsa.QSAQKVParallelLinear(16, 8, 2, 4, partition=partition)
    output = qsa.QSAOutputParallelLinear(
        32, 7, 4, partition=partition, reduce_results=False
    )
    _load_qkv(qkv, _sources(), separate=True)
    output.weight.weight_loader(output.weight, torch.ones(7, 32))

    def unexpected(*args, **kwargs):
        pytest.fail("empty owner attempted a GEMM or reduction")

    monkeypatch.setattr(qkv.quant_method, "apply", unexpected)
    monkeypatch.setattr(output.quant_method, "apply", unexpected)
    monkeypatch.setattr(qsa, "tensor_model_parallel_all_reduce", unexpected)
    assert qkv(torch.ones(2, 16))[0].shape == (2, 0)
    first, _ = output(torch.empty(2, 0))
    assert torch.count_nonzero(first) == 0
    first.fill_(99)
    second, _ = output(torch.empty(2, 0))
    assert torch.count_nonzero(second) == 0
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize(
    "shard,shape",
    [
        (None, (79, 16)),
        (None, (80, 15)),
        (None, (80, 16, 1)),
        ("q", (63, 16)),
        ("k", (7, 16)),
        ("v", (8, 15)),
        ("gate", (64, 16)),
        (0, (64, 16)),
    ],
)
def test_qkv_rejects_malformed_source_before_mutation(qsa, rank, shard, shape):
    layer = qsa.QSAQKVParallelLinear(
        16, 8, 2, 4, partition=qsa.qsa_head_partition(8, 2, 0, rank, 3)
    )
    layer.weight.data.fill_(37)
    before = layer.weight.detach().clone()
    with pytest.raises((ValueError, TypeError)):
        if shard is None:
            layer.weight.weight_loader(layer.weight, torch.ones(shape))
        else:
            layer.weight.weight_loader(layer.weight, torch.ones(shape), shard)
    torch.testing.assert_close(layer.weight, before)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("shape", [(6, 32), (7, 31), (7, 32, 1)])
def test_output_rejects_partial_source_before_mutation(qsa, rank, shape):
    layer = qsa.QSAOutputParallelLinear(
        32, 7, 4, partition=qsa.qsa_head_partition(8, 2, 0, rank, 3)
    )
    layer.weight.data.fill_(37)
    before = layer.weight.detach().clone()
    with pytest.raises((ValueError, TypeError)):
        layer.weight.weight_loader(layer.weight, torch.ones(shape))
    torch.testing.assert_close(layer.weight, before)


@pytest.mark.parametrize(
    "args",
    [
        (0, 2, 0, 0, 3),
        (7, 2, 0, 0, 3),
        (8, 1, 0, 0, 3),
        (8, 2, -1, 0, 3),
        (8, 2, 0, -1, 3),
        (8, 2, 0, 3, 3),
        (8, 2, 0, 0, 2),
        (8, 2, 0, 0, 1),
    ],
)
def test_invalid_ownership_is_rejected(qsa, args):
    with pytest.raises((ValueError, TypeError)):
        qsa.qsa_head_partition(*args)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("kind", ["qkv", "output"])
def test_non_unquantized_method_is_rejected_without_substitution(qsa, kind, rank):
    class Unsupported:
        def create_weights(self, *args, **kwargs):
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod

            UnquantizedLinearMethod().create_weights(*args, **kwargs)

    quant = SimpleNamespace(
        get_quant_method=lambda *args, **kwargs: Unsupported(),
        online_quantization_config=None,
    )
    partition = qsa.qsa_head_partition(8, 2, 0, rank, 3)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        if kind == "qkv":
            qsa.QSAQKVParallelLinear(
                16, 8, 2, 4, partition=partition, quant_config=quant
            )
        else:
            qsa.QSAOutputParallelLinear(
                32, 7, 4, partition=partition, quant_config=quant
            )


@pytest.mark.parametrize("ordinal", [0, 1, 2])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_codes_and_fp16_scales_match_global_quantize_then_slice(
    qsa, monkeypatch, ordinal, source_dtype, separate
):
    from vllm.model_executor.layers import rdna_dense_int8 as dense

    monkeypatch.setattr(dense, "_ENABLED", True)
    monkeypatch.setenv("VLLM_RDNA_DENSE_INT8_MIN_ROWS", "64")
    monkeypatch.setattr(qsa.current_platform, "is_rocm", lambda: True)
    generator = torch.Generator().manual_seed(127)
    qkv_source = torch.randn(80, 16, generator=generator).to(source_dtype)
    out_source = torch.randn(80, 32, generator=generator).to(source_dtype)
    out_source[:, 16:] *= 9
    out_source[0].zero_()

    def quantize_global(source):
        weight = source.half()
        scale = weight.abs().amax(dim=1).float().clamp_min(1e-8) / 127
        codes = (
            torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
        )
        return codes, scale.half()

    expected_qkv, expected_qkv_scale = quantize_global(qkv_source)
    expected_out, expected_out_scale = quantize_global(out_source)
    for rank in range(3):
        partition = qsa.qsa_head_partition(8, 2, ordinal, rank, 3)
        qkv = qsa.QSAQKVParallelLinear(16, 8, 2, 4, partition=partition).half()
        output = qsa.QSAOutputParallelLinear(32, 80, 4, partition=partition).half()
        _load_qkv(qkv, qkv_source, separate)
        output.weight.weight_loader(output.weight, out_source)
        if not partition.q_count:
            assert not dense._eligible(qkv) and not dense._eligible(output)
            assert not hasattr(output, "_rdna_i8_row_absmax")
            continue
        # Local QKV has only 40 rows; eligibility retains the global 80 rows.
        assert dense._eligible(qkv) and dense._eligible(output)
        qkv_codes, qkv_scale = dense._quantize_weight(qkv.weight)
        rows = _row_ids(partition)
        assert torch.equal(qkv_codes, expected_qkv[rows])
        assert torch.equal(qkv_scale, expected_qkv_scale[rows])
        codes, scale = dense._quantize_weight(output.weight, output._rdna_i8_row_absmax)
        start, width = partition.q_start * 4, partition.q_count * 4
        assert torch.equal(codes, expected_out[:, start : start + width])
        assert torch.equal(scale, expected_out_scale)
