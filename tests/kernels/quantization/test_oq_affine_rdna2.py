# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native compact affine arithmetic, routing and graph ownership.

References retain original integer codes and independently construct a little-
endian byte bitstream. CPU FP32 tensor operations separate reconstruction's
multiply/add; no native unpacker or production adapter is used by the oracle.
Run only after the campaign's GPU safety admission, on an isolated gfx1030.
"""

import gc

import numpy as np
import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms.rdna import on_rdna2
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    reset_workspace_manager,
)

pytestmark = pytest.mark.skipif(not on_rdna2(), reason="native RDNA2 affine kernels")


def _pack(q, bits):
    values = q.numpy().astype(np.uint16)
    bitplanes = (values[..., None] >> np.arange(bits, dtype=np.uint16)) & 1
    bitplanes = bitplanes.astype(np.uint8)
    stream = np.packbits(
        bitplanes.reshape(*values.shape[:-1], -1), axis=-1, bitorder="little"
    )
    padding = (-stream.shape[-1]) % 4
    stream = np.pad(stream, [(0, 0)] * (stream.ndim - 1) + [(0, padding)])
    return torch.from_numpy(stream.copy().view(np.int32)).cuda()


def _make_weights(shape, bits=6, group=64, seed=1):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randint(1 << bits, shape, generator=generator, dtype=torch.int64)
    metadata_shape = (*shape[:-1], (shape[-1] + group - 1) // group)
    scales = (torch.randn(metadata_shape, generator=generator) * 0.003).bfloat16()
    biases = (torch.randn(metadata_shape, generator=generator) * 0.02).bfloat16()
    # Exercise zero and negative scales, not just the usual positive quantizer.
    scales.reshape(-1)[::13] = 0
    w = _dequant(q, scales, biases, group)
    return (_pack(q, bits), scales.cuda(), biases.cuda()), w


def _dequant(q, scales, biases, group):
    groups = torch.arange(q.shape[-1]) // group
    product = scales.float()[..., groups] * q.float()
    affine = product + biases.float()[..., groups]
    return affine.half()


def _dense_check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.1)
    error = torch.linalg.vector_norm(actual.cpu().float() - expected.float())
    norm = torch.linalg.vector_norm(expected.float())
    assert error / norm.clamp_min(1e-30) < 0.001


def _empty():
    return torch.empty(0, dtype=torch.int32, device="cuda")


def _map(rank):
    result = torch.full((512,), -1, dtype=torch.int32)
    # Every rank has two real compact experts, globally separated.
    result[rank] = 0
    result[rank + 256] = 1
    return result.cuda()


def _project(x, weights, ids, mapping, n, divisor, bm=4, tn=8, aligned=False):
    if aligned:
        alignment = moe_align_block_size(
            ids, bm, 512, expert_map=mapping, ignore_invalid_experts=True
        )
    else:
        alignment = (_empty(), _empty(), _empty())
    out = torch.full((ids.numel(), n), float("nan"), device="cuda")
    ops.moe_oq_affine_gemm_rdna2(
        x, *weights, ids, mapping, *alignment, out, 6, 64, divisor, bm, tn
    )
    return out


def _reference_projection(x, w, ids, mapping, divisor):
    x, ids, mapping = x.cpu().float(), ids.cpu().flatten(), mapping.cpu()
    out = torch.zeros((ids.numel(), w.shape[1]), dtype=torch.float32)
    for route, global_id in enumerate(ids.tolist()):
        if 0 <= global_id < mapping.numel():
            local = int(mapping[global_id])
            if 0 <= local < w.shape[0]:
                out[route] = x[route // divisor] @ w[local].float().T
    return out


@pytest.mark.parametrize(
    "bits,group", [(6, 32), (6, 64), (6, 128), (8, 32), (8, 64), (8, 128)]
)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_embedding_all_codes_boundaries_and_exact_nonowner_zero(bits, group, id_dtype):
    k = 160  # Published PLE row: 30 words and five groups for affine6/group32.
    q = (torch.arange(5 * k).reshape(5, k) % (1 << bits)).long()
    shape = (5, (k + group - 1) // group)
    scales = torch.ones(shape, dtype=torch.bfloat16)
    biases = torch.zeros(shape, dtype=torch.bfloat16)
    scales[0] = -0.03125
    scales[1] = 0
    biases[1] = 0.75
    scales[2] = 2.0**-25  # Narrowing BF16 metadata to FP16 loses this scale.
    scales[3] = 2.0**-130  # Genuine subnormal BF16, widened exactly.
    biases[3] = -(2.0**-130)
    # Huge BF16 metadata cancels exactly; FP16 metadata would instead give NaN.
    q[4] = 1
    scales[4] = 65536
    biases[4] = -65536
    ids = torch.tensor(
        [[0, 1, 2], [3, 4, -1], [5, -2, 2]], device="cuda", dtype=id_dtype
    )
    out = torch.full((*ids.shape, k), float("nan"), dtype=torch.float16, device="cuda")
    ops.oq_affine_embedding_rdna2(
        ids, _pack(q, bits), scales.cuda(), biases.cuda(), out, bits, group, k
    )
    expected = torch.zeros((*ids.shape, k), dtype=torch.float16)
    w = _dequant(q, scales, biases, group)
    for position in np.ndindex(*ids.shape):
        row = int(ids.cpu()[position])
        if 0 <= row < 5:
            expected[position] = w[row]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    # Nonowner +0 includes affine bias and has no negative-zero sign bit.
    assert torch.equal(
        out.cpu()[ids.cpu() < 0].view(torch.int16),
        torch.zeros_like(out.cpu()[ids.cpu() < 0].view(torch.int16)),
    )


@pytest.mark.parametrize("bits,group", [(6, 32), (6, 64), (6, 128), (8, 128)])
@pytest.mark.parametrize(
    "start,local_k", [(0, 640), (80, 80), (160, 80), (3, 77), (559, 81), (124, 21)]
)
def test_dense_global_k_word_offsets_and_segment_output(bits, group, start, local_k):
    weights, w = _make_weights((13, 640), bits, group)
    # start124 is word-aligned for affine8 but its first16 weights cross
    # group128; packed-word alignment must not imply shared affine metadata.
    codes, scales, biases = weights
    word0, word1 = (start * bits) // 32, ((start + local_k) * bits + 31) // 32
    g0, g1 = start // group, (start + local_k + group - 1) // group
    x = (torch.randn(7, local_k) * 0.1).half().cuda()
    out = torch.full((7, 19), 7, dtype=torch.float16, device="cuda")
    ops.oq_affine_linear_rdna2(
        x,
        codes[:, word0:word1].contiguous(),
        scales[:, g0:g1].contiguous(),
        biases[:, g0:g1].contiguous(),
        out,
        bits,
        group,
        start,
        (start * bits) % 32,
        g0,
        3,
        4,
        8,
    )
    _dense_check(
        out[:, 3:16], (x.cpu().float() @ w[:, start : start + local_k].float().T).half()
    )
    assert torch.all(out[:, :3] == 7) and torch.all(out[:, 16:] == 7)


@pytest.mark.parametrize(
    "m,bm,tn",
    [
        (1, 1, 4),
        (2, 1, 8),
        (3, 1, 16),
        (33, 2, 4),
        (33, 2, 8),
        (65, 4, 4),
        (65, 4, 8),
        (67, 8, 4),
        (67, 8, 8),
    ],
)
def test_dense_all_finite_tiles(m, bm, tn):
    weights, w = _make_weights((41, 640), seed=30)
    x = (torch.randn(m, 640) * 0.05).half().cuda()
    out = torch.empty((m, 41), device="cuda", dtype=torch.float16)
    ops.oq_affine_linear_rdna2(x, *weights, out, 6, 64, 0, 0, 0, 0, bm, tn)
    _dense_check(out, (x.cpu().float() @ w.float().T).half())


@pytest.mark.parametrize("m", [255, 256, 257])
@pytest.mark.parametrize(
    "bits,group",
    [
        (6, 32),
        (6, 64),
        (6, 128),
        (8, 32),
        (8, 64),
        (8, 128),
    ],
)
@pytest.mark.parametrize("start,local_k", [(3, 77), (124, 21)])
def test_dense_staged_boundary_local_shards_and_output_offset(
    m,
    bits,
    group,
    start,
    local_k,
):
    weights, w = _make_weights((13, 640), bits, group, seed=108)
    codes, scales, biases = weights
    word0 = start * bits // 32
    word1 = ((start + local_k) * bits + 31) // 32
    g0, g1 = start // group, (start + local_k + group - 1) // group
    local = (
        codes[:, word0:word1].contiguous(),
        scales[:, g0:g1].contiguous(),
        biases[:, g0:g1].contiguous(),
    )
    x = (torch.randn(m, local_k) * 0.1).half().cuda()
    out = torch.full((m, 19), 7, dtype=torch.float16, device="cuda")
    manager = current_workspace_manager()
    if m < 256:
        # A cold, locked manager cannot allocate: fused calls need no scratch.
        manager.lock()

    def launch():
        ops.oq_affine_linear_rdna2(
            x,
            *local,
            out,
            bits,
            group,
            start,
            start * bits % 32,
            g0,
            3,
            8,
            8,
        )

    launch()
    if m >= 256:
        (scratch,) = manager.get_simultaneous(((13, local_k), torch.float16))
        # Reconstruction is exact even though GEMM's reduction order may differ.
        torch.testing.assert_close(
            scratch.cpu(),
            w[:, start : start + local_k],
            rtol=0,
            atol=0,
        )
        pointer = scratch.data_ptr()
        manager.lock()
    for _ in range(2):
        x.normal_(std=0.1)
        launch()
        _dense_check(
            out[:, 3:16],
            (x.cpu().float() @ w[:, start : start + local_k].float().T).half(),
        )
        assert torch.all(out[:, :3] == 7) and torch.all(out[:, 16:] == 7)
    if m >= 256:
        (reused,) = manager.get_simultaneous(((13, local_k), torch.float16))
        assert reused.data_ptr() == pointer


def test_dense_compiled_callable_crosses_staged_boundary_without_recompile():
    weights, w = _make_weights((13, 80), seed=109)
    counter = CompileCounterWithBackend("inductor")

    def project(x):
        out = torch.empty(
            (x.shape[0], 19),
            dtype=torch.float16,
            device=x.device,
        )
        ops.oq_affine_linear_rdna2(
            x,
            *weights,
            out,
            6,
            64,
            0,
            0,
            0,
            3,
            8,
            8,
        )
        return out[:, 3:16]

    compiled = torch.compile(
        project,
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )
    # Compile without any manager: tracing and small-M execution need no scratch.
    reset_workspace_manager()
    for m in (7, 257, 255, 256, 33, 7):
        if m == 257:
            init_workspace_manager(torch.device("cuda"), num_ubatches=1)
            manager = current_workspace_manager()
            manager.get_simultaneous(((13, 80), torch.float16))
            manager.lock()
        x = (torch.randn(m, 80) * 0.1).half().cuda()
        torch._dynamo.mark_dynamic(x, 0, min=2, max=1024)
        _dense_check(compiled(x), (x.cpu().float() @ w.float().T).half())
    assert counter.frame_count == 1


def test_reconstruction_is_not_factored_affine_dot():
    # w16=1 for all q, whereas q*s+b before FP16 rounding is not exactly 1.
    q = torch.full((1, 64), 3, dtype=torch.int64)
    q[:, 32:] = 1
    scales = torch.full((1, 2), 2.0**-14, dtype=torch.bfloat16)
    biases = torch.ones((1, 2), dtype=torch.bfloat16)
    x = torch.ones((1, 64), dtype=torch.float16, device="cuda")
    x[:, 32:] = -1
    out = torch.empty((1, 1), dtype=torch.float16, device="cuda")
    ops.oq_affine_linear_rdna2(
        x, _pack(q, 6), scales.cuda(), biases.cuda(), out, 6, 32, 0, 0, 0, 0, 4, 8
    )
    expected = (x.cpu().float() @ _dequant(q, scales, biases, 32).float().T).half()
    factored = (
        (
            x.cpu().float()
            * (
                q.float() * scales.float().repeat_interleave(32, -1)
                + biases.float().repeat_interleave(32, -1)
            )
        ).sum()
    ).half()
    assert expected.item() != factored.item()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tn", [4, 8, 16])
def test_direct_invalid_and_nonlocal_routes_never_substitute_expert_zero(
    rank, id_dtype, tn
):
    weights, w = _make_weights((2, 31, 640), seed=rank + 10)
    ids = torch.tensor(
        [[rank, rank + 256, -1, 512, 2**31 - 1, (rank + 1) % 8]],
        device="cuda",
        dtype=id_dtype,
    )
    mapping = _map(rank)
    x = (torch.randn(1, 640) * 0.1).half().cuda()
    actual = _project(x, weights, ids, mapping, 31, ids.size(1), tn=tn)
    expected = _reference_projection(x, w, ids, mapping, ids.size(1))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.1)
    assert torch.count_nonzero(actual[2:]) == 0


@pytest.mark.parametrize("bm,tn", [(2, 4), (2, 8), (4, 4), (4, 8), (8, 4), (8, 8)])
def test_aligned_projection_device_count_padding_and_scatter(bm, tn):
    weights, w = _make_weights((2, 37, 640), seed=3)
    mapping = _map(3)
    ids = torch.tensor(
        [3, 259, -1, 512, 4] * 67, device="cuda", dtype=torch.int32
    ).reshape(67, 5)
    x = (torch.randn(67, 640) * 0.1).half().cuda()
    actual = _project(x, weights, ids, mapping, 37, 5, bm, tn, True)
    expected = _reference_projection(x, w, ids, mapping, 5)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.1)
    assert torch.count_nonzero(actual.reshape(67, 5, 37)[:, 2:]) == 0


@pytest.mark.parametrize("tokens", [2, 67])
def test_full_moe_real_projection_shapes_ordered_fp32_reduction(tokens):
    # Real published gate/up K2560,N1280 and down K640,N2560; bounded E2.
    first, w1 = _make_weights((2, 1280, 2560), seed=10)
    second, w2 = _make_weights((2, 2560, 640), seed=11)
    mapping = _map(0)
    ids = torch.tensor(
        [0, 256, 1, 0, -1, 256, 511, 0, 256, 1] * tokens,
        device="cuda",
        dtype=torch.int64,
    ).reshape(tokens, 10)
    x = (torch.randn(tokens, 2560) * 0.01).half().cuda()
    router = torch.linspace(-0.2, 0.3, tokens * 10, device="cuda").reshape(tokens, 10)
    gate_up = _project(x, first, ids, mapping, 1280, 10, aligned=tokens > 32)
    act = torch.empty((tokens * 10, 640), dtype=torch.float16, device="cuda")
    ops.oq_silu_mul_rdna2(gate_up, act)
    down = _project(act, second, ids, mapping, 2560, 1, aligned=tokens > 32)
    out = torch.empty((tokens, 2560), dtype=torch.float16, device="cuda")
    ops.oq_moe_reduce_rdna2(down.view(tokens, 10, 2560), router, out, True)
    first_ref = _reference_projection(x, w1, ids, mapping, 10)
    gate, up = first_ref.chunk(2, dim=-1)
    act_ref = (gate / (1 + torch.exp(-gate)) * up).half()
    down_ref = _reference_projection(act_ref, w2, ids, mapping, 1).view(
        tokens, 10, 2560
    )
    expected = torch.zeros((tokens, 2560), dtype=torch.float32)
    for slot in range(10):
        expected = expected + down_ref[:, slot] * router.cpu()[:, slot, None]
    expected = expected.half()
    assert torch.isfinite(out).all()
    difference = torch.linalg.vector_norm(out.cpu().float() - expected.float())
    assert (
        difference / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30) < 0.01
    )


@pytest.mark.parametrize("multiply", [False, True])
def test_reduce_cancellation_is_ascending_slot_order(multiply):
    slots = torch.tensor(
        [[[2**24, -(2**24)], [1, -1], [-(2**24), 2**24], [0.5, -0.5]]],
        dtype=torch.float32,
        device="cuda",
    )
    weights = torch.tensor([[1, -1, 1, -1]], dtype=torch.float32, device="cuda")
    out = torch.empty((1, 2), dtype=torch.float16, device="cuda")
    ops.oq_moe_reduce_rdna2(slots, weights, out, multiply)
    expected = torch.zeros((1, 2), dtype=torch.float32)
    for slot in range(4):
        value = slots.cpu()[:, slot]
        if multiply:
            value = value * weights.cpu()[:, slot, None]
        expected = expected + value
    torch.testing.assert_close(out.cpu(), expected.half(), rtol=0, atol=0)


def test_embedding_nonfinite_metadata_keeps_declared_ieee_behavior():
    q = torch.tensor([[0, 1, 2, 3] * 8], dtype=torch.int64)
    scales = torch.tensor([[float("inf")]], dtype=torch.bfloat16, device="cuda")
    biases = torch.zeros_like(scales)
    ids = torch.tensor([0, -1], dtype=torch.int32, device="cuda")
    out = torch.empty((2, 32), dtype=torch.float16, device="cuda")
    ops.oq_affine_embedding_rdna2(ids, _pack(q, 6), scales, biases, out, 6, 32, 32)
    expected = _dequant(q, scales.cpu(), biases.cpu(), 32)
    torch.testing.assert_close(
        out[0].cpu(), expected[0], rtol=0, atol=0, equal_nan=True
    )
    assert torch.equal(out[1], torch.zeros_like(out[1]))


def test_mutation_aliases_and_missing_ep_map_are_rejected():
    weights, _ = _make_weights((8, 64))
    x = torch.ones((1, 64), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        ops.oq_affine_linear_rdna2(x, *weights, x, 6, 64, 0, 0, 0, 0, 4, 8)
    expert_weights, _ = _make_weights((2, 8, 64))
    ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        _project(x, expert_weights, ids, _empty(), 8, 1)


def test_graphs_share_pool_without_owning_eager_or_filler_storage():
    weights, _ = _make_weights((2, 37, 640), seed=50)
    mapping = _map(1)
    sizes = (67, 12)
    static = {}
    for tokens in sizes:
        x = (torch.randn(tokens, 640) * 0.1).half().cuda()
        ids = torch.tensor(
            [1, 257, 2, -1] * tokens, dtype=torch.int32, device="cuda"
        ).reshape(tokens, 4)
        static[tokens] = (x, ids)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for tokens in sizes:
            _project(
                *static[tokens][:1],
                weights,
                static[tokens][1],
                mapping,
                37,
                4,
                aligned=tokens > 32,
            )
    torch.cuda.current_stream().wait_stream(side)
    graphs, outputs, pool = {}, {}, None
    for tokens in sizes:
        graphs[tokens] = torch.cuda.CUDAGraph()
        x, ids = static[tokens]
        with torch.cuda.graph(graphs[tokens], pool=pool):
            outputs[tokens] = _project(
                x, weights, ids, mapping, 37, 4, aligned=tokens > 32
            )
        pool = graphs[tokens].pool()
    torch.cuda.synchronize()
    gc.collect()
    before = torch.cuda.memory_allocated()
    for tokens in (1, 5, 19, 24, 31):
        x = torch.ones((tokens, 640), dtype=torch.float16, device="cuda")
        ids = torch.ones((tokens, 4), dtype=torch.int32, device="cuda")
        eager = _project(x, weights, ids, mapping, 37, 4)
        del eager, x, ids
    torch.cuda.synchronize()
    gc.collect()
    assert torch.cuda.memory_allocated() == before
    fillers = [
        torch.full((tokens * 4, width), 7, dtype=dtype, device="cuda")
        for tokens in (1, 12, 19, 67)
        for width, dtype in ((37, torch.float32), (640, torch.float16))
    ]
    for rep, tokens in enumerate((12, 67, 12, 67)):
        x, ids = static[tokens]
        x.copy_(torch.randn_like(x) * 0.1)
        new_routes = torch.tensor(
            [257, 2, -1, 1] * tokens, dtype=torch.int32, device="cuda"
        ).reshape_as(ids)
        if rep % 2:
            new_routes[:, 0] = 1
        ids.copy_(new_routes)
        expected = _project(x, weights, ids, mapping, 37, 4, aligned=tokens > 32)
        outputs[tokens].fill_(float("nan"))
        graphs[tokens].replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[tokens], expected, rtol=0, atol=0)
    for filler in fillers:
        assert torch.all(filler == 7)


def test_dense_lookup_activation_and_reduce_graph_mutations():
    weights, w = _make_weights((16, 64), seed=82)
    ids = torch.tensor([0, 15, -1, 16], dtype=torch.int64, device="cuda")
    lookup = torch.empty((4, 64), dtype=torch.float16, device="cuda")
    x = torch.randn((4, 64), device="cuda").half()
    dense = torch.empty((4, 16), dtype=torch.float16, device="cuda")
    act = torch.empty((4, 8), dtype=torch.float16, device="cuda")
    gate_up = torch.randn((4, 16), device="cuda")
    slots = torch.randn((2, 2, 8), device="cuda")
    router = torch.randn((2, 2), device="cuda")
    result = torch.empty((2, 8), dtype=torch.float16, device="cuda")

    def launch():
        ops.oq_affine_embedding_rdna2(ids, *weights, lookup, 6, 64, 64)
        ops.oq_affine_linear_rdna2(x, *weights, dense, 6, 64, 0, 0, 0, 0, 4, 8)
        ops.oq_silu_mul_rdna2(gate_up, act)
        ops.oq_moe_reduce_rdna2(slots, router, result, True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        x.normal_(std=0.1)
        gate_up.normal_(std=0.2)
        slots.normal_(std=0.1)
        router.normal_(std=0.1)
        ids.copy_(torch.tensor([-1, 2, 14, 100], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected_lookup = torch.zeros_like(lookup, device="cpu")
        expected_lookup[1] = w[2]
        expected_lookup[2] = w[14]
        torch.testing.assert_close(lookup.cpu(), expected_lookup, rtol=0, atol=0)
        _dense_check(dense, (x.cpu().float() @ w.float().T).half())
        gate, up = gate_up.cpu().chunk(2, dim=-1)
        torch.testing.assert_close(
            act.cpu(),
            (gate / (1 + torch.exp(-gate)) * up).half(),
            rtol=0.001,
            atol=0.001,
        )
        expected = torch.zeros((2, 8))
        for slot in range(2):
            expected = expected + slots.cpu()[:, slot] * router.cpu()[:, slot, None]
        torch.testing.assert_close(result.cpu(), expected.half(), rtol=0, atol=0)


def test_direct_invalid_positive_and_negative_map_entries_read_no_weights():
    weights, _ = _make_weights((2, 31, 640), seed=93)
    mapping = _map(0)
    mapping[0] = 2  # Exactly E: must not dereference the first nonlocal expert.
    mapping[256] = 2**31 - 1
    mapping[1] = -2
    ids = torch.tensor([[0, 256, 1]], dtype=torch.int64, device="cuda")
    x = torch.ones((1, 640), dtype=torch.float16, device="cuda")
    actual = _project(x, weights, ids, mapping, 31, 3)
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(12345)
    reset_workspace_manager()
    init_workspace_manager(torch.device("cuda"), num_ubatches=1)
    try:
        yield
    finally:
        reset_workspace_manager()
