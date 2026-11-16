# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real native-affine consumers, independently packed sources and bounded oracles.

The EP8 tests serialize all eight ownership maps on one device. The separately
selected TP8 test needs eight physical GPUs; it does not emulate a TP group.
Largest source component: 600 MiB CPU; local compact MoE: 243.75 MiB GPU.
No reference expands all 512 experts. Compact source shards are built once per
module and mapped one component at a time for all EP owners and graph checks.
"""

import faulthandler
import gc
import hashlib
import json
from math import prod
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.utils import multi_gpu_test
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.mlx_affine import MlxAffineConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.mlx_affine_loading import load_mlx_affine_weights
from vllm.platforms.rdna import on_rdna2
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    reset_workspace_manager,
    use_workspace_lane,
)

pytestmark = pytest.mark.skipif(not on_rdna2(), reason="native RDNA2 affine consumers")


@pytest.fixture(autouse=True)
def _workspace():
    reset_workspace_manager()
    init_workspace_manager(
        torch.device("cuda"),
        num_ubatches=1,
        num_lanes=2,
    )
    try:
        yield
    finally:
        reset_workspace_manager()


def _config(root, modules):
    """Bind real construction to a config-pinned protected metadata view.

    modules maps original prefixes to (leading shape, logical K, bits, group).
    The iterable loader, rather than a safetensors reader, supplies payloads.
    """
    root = Path(root)
    root.mkdir(parents=True)
    quant = {
        "quant_method": "mlx_affine",
        "format_version": 1,
        "module_specs": {
            name: {"bits": bits, "group_size": group, "mode": "affine"}
            for name, (_, _, bits, group) in modules.items()
        },
    }
    config = MlxAffineConfig.from_config(quant)
    tensors, offset = {}, 0
    for name, (leading, k, bits, group) in modules.items():
        for component, dtype, columns, itemsize in (
            ("weight", "U32", (k * bits + 31) // 32, 4),
            ("scales", "BF16", (k + group - 1) // group, 2),
            ("biases", "BF16", (k + group - 1) // group, 2),
        ):
            shape = [*leading, columns]
            size = prod(shape) * itemsize
            tensors[f"{name}.{component}"] = {
                "file": "model.safetensors",
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
                "runtime_name": f"{name}.{component}".replace(
                    ".switch_mlp.", ".experts."
                ),
                "layout": "identity",
            }
            offset += size
    manifest = {
        "format_version": 1,
        "view": "target",
        "source_revision": "e171af86f499f1855b0fb71d781105e8dd609610",
        "module_specs_sha256": config.specs.sha256,
        "tensors": tensors,
    }
    original = root.parent / (root.name + ".safetensors")
    original.write_bytes(b"iterable source: payloads supplied by the test")
    (root / "model.safetensors").symlink_to("../" + original.name)
    raw = json.dumps(manifest).encode()
    (root / "mlx-affine-manifest.json").write_bytes(raw)
    config.maybe_update_config(
        str(root),
        {
            "quantization_config": quant,
            "num_hidden_layers": 48,
            "mlx_affine": {
                "manifest": "mlx-affine-manifest.json",
                "view": "target",
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            },
        },
    )
    return config


def _codes(rows, k, bits, seed, expert, row_start=0):
    # Every integer code and every word/group boundary occurs. Expert identity
    # affects both integer codes and metadata, exposing wrong ownership copies.
    r = np.arange(row_start, row_start + rows, dtype=np.uint32)[:, None]
    c = np.arange(k, dtype=np.uint32)[None, :]
    return ((c + 7 * r + 11 * seed + 3 * expert) % (1 << bits)).astype(np.uint16)


def _pack_cpu(q, bits):
    planes = ((q[..., None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    stream = np.packbits(planes.reshape(*q.shape[:-1], -1), axis=-1, bitorder="little")
    stream = np.pad(stream, [(0, 0)] * (stream.ndim - 1) + [(0, -stream.shape[-1] % 4)])
    return torch.from_numpy(stream.copy().view(np.uint32))


def _metadata(rows, k, group, seed, expert, row_start=0):
    r = torch.arange(row_start, row_start + rows)[:, None]
    g = torch.arange((k + group - 1) // group)[None, :]
    index = r * 3 + g + seed * 7 + expert
    scales = (((index % 11).float() - 5) * 0.0005).bfloat16()
    biases = (((index % 13).float() - 6) * 0.002).bfloat16()
    # BF16 subnormal and values that vanish if metadata is narrowed to FP16.
    scales[index % 17 == 0] = 2.0**-130
    scales[index % 19 == 0] = 2.0**-25
    scales[index % 23 == 0] = 0
    return scales, biases


def _sources(modules):
    for seed, (name, (leading, k, bits, group)) in enumerate(modules.items()):
        experts, rows = leading if len(leading) == 2 else (1, leading[0])
        for component in ("weight", "scales", "biases"):
            columns = (
                (k * bits + 31) // 32
                if component == "weight"
                else (k + group - 1) // group
            )
            tensor = torch.empty(
                *leading,
                columns,
                dtype=torch.uint32 if component == "weight" else torch.bfloat16,
                device="cpu",
            )
            for expert in range(experts):
                target = tensor[expert] if len(leading) == 2 else tensor
                # Packing uses <=64 rows, never an expanded expert collection.
                for start in range(0, rows, 64):
                    stop = min(start + 64, rows)
                    if component == "weight":
                        q = _codes(
                            stop - start,
                            k,
                            bits,
                            seed,
                            expert,
                            row_start=start,
                        )
                        target[start:stop].copy_(_pack_cpu(q, bits))
                        del q
                    else:
                        metadata = _metadata(
                            stop - start,
                            k,
                            group,
                            seed,
                            expert,
                            row_start=start,
                        )
                        target[start:stop].copy_(metadata[component == "biases"])
                        del metadata
            yield f"{name}.{component}", tensor
            del target, tensor


def _weight(modules, name, expert=0):
    leading, k, bits, group = modules[name]
    rows = leading[-1]
    seed = list(modules).index(name)
    q = torch.from_numpy(_codes(rows, k, bits, seed, expert)).float()
    scales, biases = _metadata(rows, k, group, seed, expert)
    groups = torch.arange(k) // group
    # Explicit separate FP32 multiply and add, followed by FP16 rounding.
    product = q * scales.float()[:, groups]
    return (product + biases.float()[:, groups]).half()


class _Consumers(nn.Module):
    def __init__(self, quant_config, **layers):
        super().__init__()
        self.quant_config = quant_config
        for name, layer in layers.items():
            self.add_module(name, layer)


def _shard_sources(paths):
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            # Safetensors handles expose keys(), not mapping iteration.
            for name in shard.keys():  # noqa: SIM118
                yield name, shard.get_tensor(name)


def _load(model, modules, weights):
    def ordinary(weights):
        assert list(weights) == [], "all sources in this fixture are compact"
        return set()

    loaded = load_mlx_affine_weights(model, weights, ordinary)
    assert loaded == set(dict(model.named_parameters()))
    receipt = model._mlx_affine_load_receipt
    assert receipt.compact_components == len(modules) * 3
    assert receipt.max_copy_tile_bytes <= 1024 * 1024


def _dense_close(actual, expected):
    actual = actual.cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.1)
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    assert error / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30) < 0.001


def _dense_modules():
    return {
        "layer.mlp.gate_proj": ((64,), 640, 6, 64),
        "layer.mlp.up_proj": ((128,), 640, 8, 128),
        "layer.down_proj": ((33,), 640, 6, 128),
        "layer.attn.q_proj": ((128,), 640, 6, 64),
        "layer.attn.k_proj": ((16,), 640, 8, 32),
        "layer.attn.v_proj": ((16,), 640, 6, 128),
        "embed_tokens": ((59,), 160, 6, 32),
    }


def _dense_consumers(config, *, disable_tp):
    kwargs = dict(params_dtype=torch.float16, quant_config=config)
    with torch.device("cuda"):
        return _Consumers(
            config,
            merged=MergedColumnParallelLinear(
                640,
                [64, 128],
                bias=False,
                prefix="layer.mlp.gate_up_proj",
                disable_tp=disable_tp,
                return_bias=False,
                **kwargs,
            ),
            row=RowParallelLinear(
                640,
                33,
                bias=False,
                reduce_results=False,
                prefix="layer.down_proj",
                disable_tp=disable_tp,
                return_bias=False,
                **kwargs,
            ),
            qkv=QKVParallelLinear(
                640,
                8,
                16,
                2,
                bias=False,
                prefix="layer.attn.qkv_proj",
                disable_tp=disable_tp,
                return_bias=False,
                **kwargs,
            ),
            vocab=VocabParallelEmbedding(
                59,
                160,
                prefix="embed_tokens",
                padding_size=64,
                disable_tp=disable_tp,
                reduce_results=False,
                **kwargs,
            ),
        )


def _check_dense_consumers(model, modules, rank, tp, *, token_counts=(1, 33)):
    for tokens in token_counts:
        x = (torch.arange(tokens * 640).reshape(tokens, 640) % 17 - 8).half() / 64
        # Alternating signs and zero sums exercise cancellation, not only norms.
        x[:, 1::2].neg_()
        merged = torch.cat(
            [
                _weight(modules, name)[rank * (rows // tp) : (rank + 1) * (rows // tp)]
                for name, rows in (
                    ("layer.mlp.gate_proj", 64),
                    ("layer.mlp.up_proj", 128),
                )
            ]
        )
        _dense_close(model.merged(x.cuda()), (x.float() @ merged.float().T).half())
        k0, k1 = rank * (640 // tp), (rank + 1) * (640 // tp)
        row = _weight(modules, "layer.down_proj")[:, k0:k1]
        _dense_close(
            model.row(x[:, k0:k1].contiguous().cuda()),
            (x[:, k0:k1].float() @ row.float().T).half(),
        )
        q0, q1 = rank * (128 // tp), (rank + 1) * (128 // tp)
        q = _weight(modules, "layer.attn.q_proj")[q0:q1]
        kv_rank = rank // 4 if tp == 8 else 0
        kv_width = 8 if tp == 8 else 16
        kv0, kv1 = kv_rank * kv_width, (kv_rank + 1) * kv_width
        kv = [
            _weight(modules, f"layer.attn.{name}_proj")[kv0:kv1] for name in ("k", "v")
        ]
        qkv = torch.cat([q, *kv])
        _dense_close(model.qkv(x.cuda()), (x.float() @ qkv.float().T).half())
    indices = model.vocab.shard_indices
    ids = torch.tensor(
        [
            0,
            7,
            8,
            15,
            16,
            31,
            32,
            34,
            55,
            56,
            58,
            indices.org_vocab_start_index,
            indices.org_vocab_end_index - 1,
        ],
        device="cuda",
    )
    actual = model.vocab(ids).cpu()
    expected = _weight(modules, "embed_tokens")[ids.cpu()]
    if tp > 1:
        owner = (ids.cpu() >= indices.org_vocab_start_index) & (
            ids.cpu() < indices.org_vocab_end_index
        )
        expected[~owner] = 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Identity activations expose reconstruction, independent of dot reduction.
    basis = torch.eye(640, dtype=torch.float16, device="cuda")
    reconstructed = model.merged(basis).cpu().T
    torch.testing.assert_close(reconstructed, merged, rtol=0, atol=0)
    reconstructed_qkv = model.qkv(basis).cpu().T
    torch.testing.assert_close(reconstructed_qkv, qkv, rtol=0, atol=0)
    local_basis = torch.eye(640 // tp, dtype=torch.float16, device="cuda")
    reconstructed_row = model.row(local_basis).cpu().T
    torch.testing.assert_close(reconstructed_row, row, rtol=0, atol=0)


@pytest.mark.parametrize(
    "additional_config",
    [
        {},
        {
            "rdna_oq_affine": {
                "decode_tile_n": 16,
                "prefill_tile_n": 4,
                "prefill_block_m": 8,
                "dense_prefill_block_m": 2,
                "chunk_tokens": 128,
            }
        },
    ],
    ids=["default", "independent-dense-tile"],
)
@torch.inference_mode()
def test_real_merged_row_qkv_vocab_consumers(
    tmp_path,
    default_vllm_config,
    additional_config,
):
    modules = _dense_modules()
    with set_current_vllm_config(VllmConfig(additional_config=additional_config)):
        model = _dense_consumers(_config(tmp_path / "native", modules), disable_tp=True)
    _load(model, modules, _sources(modules))
    # Cover both runtime dispatch boundaries, including partial prefill tiles,
    # with defaults and independent dense/routed reuse settings.
    _check_dense_consumers(
        model,
        modules,
        0,
        1,
        token_counts=(1, 32, 33, 255, 256, 257, 1023),
    )


@torch.inference_mode()
def test_real_dense_changed_input_graphs_share_scratch_across_layers(
    tmp_path,
    default_vllm_config,
):
    modules = _dense_modules()
    model = _dense_consumers(_config(tmp_path / "native", modules), disable_tp=True)
    _load(model, modules, _sources(modules))
    reference = (
        torch.cat(
            [_weight(modules, f"layer.mlp.{name}_proj") for name in ("gate", "up")]
        ),
        _weight(modules, "layer.down_proj"),
        torch.cat(
            [_weight(modules, f"layer.attn.{name}_proj") for name in ("q", "k", "v")]
        ),
    )
    static = {
        lane: (torch.randn(256 + lane, 640) * 0.05).half().cuda() for lane in (0, 1)
    }

    def launch(x):
        # Mixed 6/8-bit segments and different N reuse the same transient buffer.
        return model.merged(x), model.row(x), model.qkv(x)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for lane, x in static.items():
            with use_workspace_lane(lane):
                launch(x)
    torch.cuda.current_stream().wait_stream(side)
    manager = current_workspace_manager()
    scratch = {}
    for lane in static:
        with use_workspace_lane(lane):
            (scratch[lane],) = manager.get_simultaneous(
                ((128, 640), torch.float16),
            )
    assert scratch[0].data_ptr() != scratch[1].data_ptr()
    manager.lock()
    graphs, outputs, pool = {}, {}, None
    for lane, x in static.items():
        graphs[lane] = torch.cuda.CUDAGraph()
        with use_workspace_lane(lane), torch.cuda.graph(graphs[lane], pool=pool):
            outputs[lane] = launch(x)
        pool = graphs[lane].pool()
    filler = torch.full((257, 192), 7, dtype=torch.float16, device="cuda")
    for iteration, lane in enumerate((1, 0, 1, 0), 1):
        x = static[lane]
        x.normal_(mean=iteration * 0.001, std=0.05)
        expected = tuple((x.cpu().float() @ w.float().T).half() for w in reference)
        # Eager calls overwrite the same scratch between capture and replay.
        for eager_lane in (lane, 1 - lane):
            with use_workspace_lane(eager_lane):
                eager = launch(-static[eager_lane])
            del eager
        for out in outputs[lane]:
            out.fill_(float("nan"))
        graphs[lane].replay()
        torch.cuda.synchronize()
        for actual, wanted in zip(outputs[lane], expected):
            _dense_close(actual, wanted)
        with use_workspace_lane(lane):
            (reused,) = manager.get_simultaneous(((128, 640), torch.float16))
            assert reused.data_ptr() == scratch[lane].data_ptr()
        assert torch.all(filler == 7)


@pytest.mark.parametrize("rank", range(8))
@torch.inference_mode()
def test_real_column_tp8_owned_rows(tmp_path, default_vllm_config, rank):
    modules = {"projection": ((64,), 640, 8, 128)}
    config = _config(tmp_path / "native", modules)
    with torch.device("cuda"):
        layer = ColumnParallelLinear(
            640,
            64,
            bias=False,
            params_dtype=torch.float16,
            quant_config=config,
            prefix="projection",
            tp_rank=rank,
            tp_size=8,
            return_bias=False,
        )
    model = _Consumers(config, projection=layer)
    _load(model, modules, _sources(modules))
    x = torch.eye(640, dtype=torch.float16, device="cuda")
    torch.testing.assert_close(
        layer(x).cpu().T,
        _weight(modules, "projection")[rank * 8 : (rank + 1) * 8],
        rtol=0,
        atol=0,
    )


def _tp8_worker(rank, port, root):
    faulthandler.enable()
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=8,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=8)
        try:
            init_workspace_manager(torch.device("cuda", rank), num_ubatches=1)
            modules = _dense_modules()
            model = _dense_consumers(
                _config(Path(root) / f"rank{rank}", modules),
                disable_tp=False,
            )
            with torch.inference_mode():
                _load(model, modules, _sources(modules))
                _check_dense_consumers(model, modules, rank, 8)
        finally:
            # Finish local device work, then rendezvous without a new GPU collective.
            torch.cuda.synchronize(rank)
            get_world_group().barrier()
            reset_workspace_manager()
            cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=8)
def test_real_tp8_row_global_groups_qkv_replication_and_vocab(tmp_path):
    """Real K640/local80 group128 shards and replicated KV on eight GPUs.

    Local projections deliberately disable collectives. This is not a TP8
    collective correctness test; PLE's distributed suite covers that separately.
    """
    torch.multiprocessing.spawn(
        _tp8_worker,
        args=(get_open_port(), str(tmp_path)),
        nprocs=8,
        join=True,
    )


def _moe_modules():
    prefix = "layer.mlp.switch_mlp"
    return {
        f"{prefix}.gate_proj": ((512, 640), 2560, 6, 64),
        f"{prefix}.up_proj": ((512, 640), 2560, 6, 64),
        f"{prefix}.down_proj": ((512, 2560), 640, 6, 64),
    }


def _moe(root, rank, source_shards):
    modules = _moe_modules()
    config = _config(root, modules)
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=1,
        dp_rank=0,
        ep_size=8,
        ep_rank=rank,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=512,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=64,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device="cuda",
        in_dtype=torch.float16,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
    )
    with torch.device("cuda"):
        manager = ExpertMapManager(
            max_num_batched_tokens=256,
            top_k=10,
            global_num_experts=512,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
            layer_name="layer.mlp.experts",
        )
        layer = RoutedExperts(
            layer_name="layer.mlp.experts",
            params_dtype=torch.float16,
            moe_config=moe_config,
            quant_config=config,
            expert_map_manager=manager,
        )
    model = _Consumers(config, experts=layer)
    _load(model, modules, _shard_sources(source_shards))
    expected_map = torch.full((512,), -1, dtype=torch.int32)
    expected_map[rank * 64 : (rank + 1) * 64] = torch.arange(64, dtype=torch.int32)
    torch.testing.assert_close(layer.expert_map.cpu(), expected_map, rtol=0, atol=0)
    return layer, modules


@pytest.fixture(scope="module")
def moe_source_shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("compact-moe-sources")
    paths = []
    for index, (name, tensor) in enumerate(_sources(_moe_modules())):
        path = root / f"{index:02d}.safetensors"
        save_file({name: tensor}, str(path))
        paths.append(path)
        del tensor
    return tuple(paths)


@pytest.fixture(scope="module", params=range(8), ids=lambda rank: f"ep{rank}")
def owned_moe(request, tmp_path_factory, moe_source_shards):
    with set_current_vllm_config(VllmConfig()):
        with torch.inference_mode():
            root = tmp_path_factory.mktemp(f"ep{request.param}") / "native"
            layer, modules = _moe(root, request.param, moe_source_shards)
        yield layer, modules, request.param
    del layer
    gc.collect()
    torch.cuda.synchronize()


def _inputs(tokens, rank, variant=0):
    start = rank * 64
    routes = [
        start,
        start + 63,
        (start + 64) % 512,
        -1,
        512,
        start,
        start + 63,
        -7,
        2**31 - 1,
        start,
    ]
    ids = torch.tensor(routes, dtype=torch.int64).repeat(tokens, 1)
    if variant:
        ids = ids.roll(variant, dims=1)
        ids[::2, 0] = start + 1
    # A whole row with no owned routes must be exactly zero, including biases.
    if tokens > 1:
        ids[-1] = torch.tensor([-1, 512, -7, 1024, (start + 64) % 512] * 2)
    gen = torch.Generator().manual_seed(700 + tokens + variant)
    x = (torch.randn(tokens, 2560, generator=gen) * 0.04).half()
    x[:, 1::2].neg_()
    router = torch.tensor(
        [0.3, -0.2, 0.1, 0, 1, -0.1, 0.25, 0.1, -0.2, 0.2],
        dtype=torch.float32,
    ).repeat(tokens, 1)
    return x.cuda(), router.cuda(), ids.cuda()


def _moe_reference(modules, x, router, ids, rank):
    x, router, ids = x.cpu(), router.cpu(), ids.cpu()
    slots = torch.zeros(*ids.shape, 2560, dtype=torch.float32)
    # Only selected, locally owned experts are reconstructed, one at a time.
    for expert in sorted(set(ids.flatten().tolist())):
        if not rank * 64 <= expert < (rank + 1) * 64:
            continue
        token, slot = (ids == expert).nonzero(as_tuple=True)
        gate_w = _weight(modules, "layer.mlp.switch_mlp.gate_proj", expert)
        up_w = _weight(modules, "layer.mlp.switch_mlp.up_proj", expert)
        gate = x[token].float() @ gate_w.float().T
        up = x[token].float() @ up_w.float().T
        del gate_w, up_w
        act = (gate / (1 + torch.exp(-gate)) * up).half()
        down_w = _weight(modules, "layer.mlp.switch_mlp.down_proj", expert)
        slots[token, slot] = act.float() @ down_w.float().T
        del down_w, act
    output = torch.zeros(x.shape, dtype=torch.float32)
    for slot in range(10):
        product = slots[:, slot] * router[:, slot, None]
        output = output + product
    return output.half()


def _moe_close(actual, expected):
    assert torch.isfinite(actual).all()
    difference = torch.linalg.vector_norm(actual.cpu().float() - expected.float())
    norm = torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
    assert difference / norm < 0.01


@pytest.mark.parametrize("tokens", [2, 33])
@torch.inference_mode()
def test_real_routed_experts_full_fp32_ordered_moe(owned_moe, tokens):
    layer, modules, rank = owned_moe
    x, router, ids = _inputs(tokens, rank)
    actual = layer.forward_modular(x, router, ids)
    _moe_close(actual, _moe_reference(modules, x, router, ids, rank))
    assert torch.count_nonzero(actual[-1]) == 0


@torch.inference_mode()
def test_real_routed_experts_graph_replay_and_eager_scratch(
    tmp_path, moe_source_shards
):
    with set_current_vllm_config(VllmConfig()):
        layer, modules = _moe(tmp_path / "native", 0, moe_source_shards)
        sizes = (33, 12)
        static = {tokens: _inputs(tokens, 0) for tokens in sizes}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for values in static.values():
                layer.forward_modular(*values)
        torch.cuda.current_stream().wait_stream(side)
        graphs, outputs, pool = {}, {}, None
        for tokens in sizes:
            graphs[tokens] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[tokens], pool=pool):
                outputs[tokens] = layer.forward_modular(*static[tokens])
            pool = graphs[tokens].pool()
        torch.cuda.synchronize()
        gc.collect()
        before = torch.cuda.memory_allocated()
        for tokens in (1, 3, 5, 19, 24, 31, 34, 41):
            x, router, ids = _inputs(tokens, 0)
            eager = layer.forward_modular(x, router, ids)
            del eager, x, router, ids
        torch.cuda.synchronize()
        gc.collect()
        assert torch.cuda.memory_allocated() == before
        fillers = [
            torch.full((tokens * 10, width), 7, dtype=dtype, device="cuda")
            for tokens in (1, 12, 19, 33, 41)
            for width, dtype in (
                (1280, torch.float32),
                (640, torch.float16),
                (2560, torch.float32),
            )
        ]
        for iteration, tokens in enumerate((12, 33, 12, 33), 1):
            new = _inputs(tokens, 0, iteration)
            for destination, source in zip(static[tokens], new):
                destination.copy_(source)
            expected = _moe_reference(modules, *new, 0)
            eager = layer.forward_modular(*new)
            outputs[tokens].fill_(float("nan"))
            graphs[tokens].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[tokens], eager, rtol=0, atol=0)
            _moe_close(outputs[tokens], expected)
            del eager, new
        for filler in fillers:
            assert torch.all(filler == 7)


def _op_arguments(kind, device="cuda"):
    def empty(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    half, fp32, i32, bf16 = torch.float16, torch.float32, torch.int32, torch.bfloat16
    codes, scales = empty((8, 12), i32), empty((8, 1), bf16)
    biases = torch.ones_like(scales) / 64
    if kind in ("linear", "linear_staged"):
        tokens = 256 if kind == "linear_staged" else 2
        return (
            ops.oq_affine_linear_rdna2,
            [
                empty((tokens, 64), half),
                codes,
                scales,
                biases,
                empty((tokens, 8), half),
                6,
                64,
                0,
                0,
                0,
                0,
                8,
                8,
            ],
            4,
        )
    if kind == "embedding":
        return (
            ops.oq_affine_embedding_rdna2,
            [
                empty((2,), i32),
                codes,
                scales,
                biases,
                empty((2, 64), half),
                6,
                64,
                64,
            ],
            4,
        )
    if kind == "moe":
        mapping = torch.full((512,), -1, dtype=i32, device=device)
        mapping[:64] = torch.arange(64, dtype=i32, device=device)
        return (
            ops.moe_oq_affine_gemm_rdna2,
            [
                empty((2, 64), half),
                empty((64, 8, 12), i32),
                empty((64, 8, 1), bf16),
                empty((64, 8, 1), bf16),
                empty((2, 10), i32),
                mapping,
                empty((0,), i32),
                empty((0,), i32),
                empty((0,), i32),
                empty((20, 8), fp32),
                6,
                64,
                10,
                1,
                8,
            ],
            9,
        )
    if kind == "silu":
        return ops.oq_silu_mul_rdna2, [empty((20, 16), fp32), empty((20, 8), half)], 1
    return (
        ops.oq_moe_reduce_rdna2,
        [empty((2, 10, 8), fp32), empty((2, 10), fp32), empty((2, 8), half), True],
        2,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "linear",
        "linear_staged",
        "embedding",
        "moe",
        "silu",
        "reduce",
    ],
)
@pytest.mark.parametrize("fault", ["shape", "dtype", "device"])
def test_public_native_ops_fake_and_real_reject_bad_output_contract(kind, fault):
    for fake in (False, True):
        mode = FakeTensorMode() if fake else torch.inference_mode()
        with mode:
            op, args, index = _op_arguments(kind)
            output = args[index]
            shape = tuple(output.shape)
            dtype, device = output.dtype, output.device
            if fault == "shape":
                shape = (shape[0] + 1, *shape[1:])
            elif fault == "dtype":
                dtype = torch.float64
            else:
                device = "cpu"
            args[index] = torch.empty(shape, dtype=dtype, device=device)
            with pytest.raises(RuntimeError):
                op(*args)
