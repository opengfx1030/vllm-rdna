# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the hx expert-server model on a tiny synthetic checkpoint.

The checkpoint is written with ``safetensors`` in the layout of the Flash-Next
W4A16 checkpoint (per expert ``gate_proj``/``up_proj``/``down_proj`` with
``weight_packed`` int32 [N, K/8], ``weight_scale`` bf16 [N, K/g] and
``weight_shape``) at tiny geometry. The HIP ops are replaced: ``gptq_shuffle`` by
an injected fake and ``_rdna2_fused_moe`` by a recorder or by a torch emulation
that reads the K-first tensors the loader produced, independently of the module's
own reference.
"""

import ast
import importlib
import inspect
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

import regex as re  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from vllm.distributed.hetero_moe import server_model as sm  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types  # noqa: E402

GROUP = 128
EXPERTS = 6
LAYERS = 3
TOP_K = 3
# Held experts per layer: ragged, one empty layer, 4 of 6.
SETS = [[1, 4], [], [0, 2, 3, 5]]
GEOMETRIES = [  # (checkpoint key prefix, hidden, inter)
    pytest.param(("model.language_model.", 256, 128), id="lm-prefix-256x128"),
    pytest.param(("model.", 384, 256), id="model-prefix-384x256"),
    pytest.param(("", 256, 128), id="bare-prefix-256x128"),
]


@pytest.fixture(autouse=True, scope="module")
def few_cpu_threads():
    """Tiny tensors: on a loaded many-core box, OpenMP fan-out costs 100x its gain."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, 4))
    yield
    torch.set_num_threads(before)


def prod():
    return importlib.import_module(sm.PRODUCTION_MODULE)


def shard_file(shard: int) -> str:
    return f"model-{shard + 1:05d}-of-00002.safetensors"


def write_checkpoint(
    root: Path,
    prefix: str,
    hidden: int,
    inter: int,
    *,
    drop: tuple[int, int, str] | None = None,
    scale_groups: int | None = None,
):
    """Tiny two-shard checkpoint; returns {(layer, expert): {proj: (words, scales)}}."""
    gen = torch.Generator().manual_seed(0)
    shards: list[dict[str, torch.Tensor]] = [{}, {}]
    weight_map: dict[str, str] = {}
    raw: dict[tuple[int, int], dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            shard = (layer * EXPERTS + expert) % 2
            entry = {}
            for proj, n, k in (
                ("gate_proj", inter, hidden),
                ("up_proj", inter, hidden),
                ("down_proj", hidden, inter),
            ):
                packed = torch.randint(
                    -(2**31), 2**31, (n, k // 8), dtype=torch.int32, generator=gen
                )
                # Multiples of 2**-8: exact in bf16 and in fp16.
                steps = torch.randint(
                    1, 9, (n, scale_groups or k // GROUP), generator=gen
                )
                scale = (steps.float() * 2.0**-8).to(torch.bfloat16)
                entry[proj] = (packed, scale)
                base = f"{prefix}layers.{layer}.mlp.experts.{expert}.{proj}."
                for kind, value in (
                    ("weight_packed", packed),
                    ("weight_scale", scale),
                    ("weight_shape", torch.tensor([n, k], dtype=torch.int64)),
                ):
                    if drop == (layer, expert, f"{proj}.{kind}"):
                        continue
                    shards[shard][base + kind] = value
                    weight_map[base + kind] = shard_file(shard)
            raw[(layer, expert)] = entry
    unrelated = {  # present in the real index; the loader must never read them
        f"{prefix}layers.0.mlp.gate.weight": (EXPERTS, hidden),
        "mtp.layers.0.mlp.experts.gate_up_proj": (2, 2 * inter, hidden),
        "mtp.layers.0.mlp.experts.down_proj": (2, hidden, inter),
    }
    for name, shape in unrelated.items():
        shards[0][name] = torch.zeros(shape, dtype=torch.bfloat16)
        weight_map[name] = shard_file(0)
    for i, tensors in enumerate(shards):
        save_file(tensors, str(root / shard_file(i)), metadata={"format": "pt"})
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (root / sm.INDEX_NAME).write_text(json.dumps(index))
    return raw


@pytest.fixture(scope="module", params=GEOMETRIES)
def ckpt(request, tmp_path_factory):
    prefix, hidden, inter = request.param
    root = tmp_path_factory.mktemp("ckpt")
    raw = write_checkpoint(root, prefix, hidden, inter)
    return SimpleNamespace(root=root, raw=raw, hidden=hidden, inter=inter)


class Shuffle:
    """Fake ``gptq_shuffle``: records each call, optionally edits in place."""

    def __init__(self, fn=None):
        self.calls = []
        self.fn = fn

    def __call__(self, tensor, bits):
        self.calls.append((tuple(tensor.shape), tensor.dtype, bits))
        if self.fn is not None:
            self.fn(tensor)


def make_model(ckpt, shuffle=None, sets=SETS, **kwargs):
    kwargs.setdefault("num_experts", EXPERTS)
    kwargs.setdefault("top_k", TOP_K)
    kwargs.setdefault("max_capture_rows", 2)
    return sm.ExpertServerModel(
        ckpt.root,
        sets,
        "cpu",
        group_size=GROUP,
        shuffle=Shuffle() if shuffle is None else shuffle,
        **kwargs,
    )


@pytest.fixture
def loaded(ckpt):
    model = make_model(ckpt)
    model.load()
    return model


def call_inputs(model, rows, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, model.hidden, generator=gen).to(torch.float16)
    ids = torch.randint(-1, model.num_experts, (rows, model.top_k), generator=gen)
    w = torch.rand(rows, model.top_k, generator=gen)
    return x, ids.to(torch.int32), w


# --- name resolution ---------------------------------------------------------


@pytest.mark.parametrize("prefix", ["model.language_model.", "model.", ""])
def test_expert_tensor_names_resolve_through_the_index(prefix):
    weight_map = {}
    for layer in (1, 10):
        for expert in (1, 10, 100):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("weight_packed", "weight_scale", "weight_shape"):
                    name = f"{prefix}layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
                    weight_map[name] = "shard"
    weight_map["mtp.layers.1.mlp.experts.1.gate_proj.weight_packed"] = "mtp"
    weight_map["mtp.layers.0.mlp.experts.gate_up_proj"] = "mtp"
    weight_map[f"{prefix}layers.1.mlp.gate.weight"] = "shard"

    names = sm.ExpertServerModel.expert_tensor_names(weight_map, 1, 10)

    base = f"{prefix}layers.1.mlp.experts.10."
    assert names == {
        "gate_packed": base + "gate_proj.weight_packed",
        "gate_scale": base + "gate_proj.weight_scale",
        "up_packed": base + "up_proj.weight_packed",
        "up_scale": base + "up_proj.weight_scale",
        "down_packed": base + "down_proj.weight_packed",
        "down_scale": base + "down_proj.weight_scale",
    }


def test_expert_tensor_names_reject_missing_and_ambiguous():
    full = {
        f"model.layers.0.mlp.experts.0.{p}.{k}": "s"
        for p in ("gate_proj", "up_proj", "down_proj")
        for k in ("weight_packed", "weight_scale")
    }
    del full["model.layers.0.mlp.experts.0.up_proj.weight_scale"]
    with pytest.raises(KeyError, match="up_scale"):
        sm.ExpertServerModel.expert_tensor_names(full, 0, 0)
    with pytest.raises(KeyError, match="layer 7 expert 3"):
        sm.ExpertServerModel.expert_tensor_names(full, 7, 3)
    full["model.layers.0.mlp.experts.0.up_proj.weight_scale"] = "s"
    full["model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale"] = "s"
    with pytest.raises(ValueError, match="listed twice"):
        sm.ExpertServerModel.expert_tensor_names(full, 0, 0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "model.language_model.layers.7.mlp.experts.12.down_proj.weight_scale",
            (7, 12, "down_scale"),
        ),
        ("layers.0.mlp.experts.0.up_proj.weight_packed", (0, 0, "up_packed")),
        ("model.layers.0.mlp.experts.0.up_proj.weight_shape", None),
        ("model.layers.0.mlp.experts.0.up_proj.weight_packed.x", None),
        ("mtp.layers.0.mlp.experts.0.gate_proj.weight_packed", None),
        ("model.xlayers.0.mlp.experts.0.gate_proj.weight_packed", None),
        ("model.layers.0.mlp.shared_expert.gate_proj.weight", None),
    ],
)
def test_parse_expert_key(name, expected):
    assert sm.parse_expert_key(name) == expected


# --- constructor: geometry, validation, local_map ---------------------------


def test_geometry_comes_from_the_tensors(ckpt):
    model = make_model(ckpt)
    assert (model.hidden, model.inter) == (ckpt.hidden, ckpt.inter)
    assert model.num_layers == LAYERS


def test_local_map_for_ragged_and_empty_sets(ckpt):
    model = make_model(ckpt)
    expected = [
        [-1, 0, -1, -1, 1, -1],
        [-1] * 6,
        [0, -1, 1, 2, -1, 3],
    ]
    assert [m.tolist() for m in model.local_map] == expected
    for table in model.local_map:
        assert table.dtype == torch.int32 and table.shape == (EXPERTS,)
        assert table.device.type == "cpu" and table.is_contiguous()


@pytest.mark.parametrize(
    ("sets", "message"),
    [
        ([[2, 1], [], []], "ascending"),
        ([[1, 1], [], []], "ascending"),
        ([[0, EXPERTS], [], []], "in \\[0, 6\\)"),
        ([[-1, 0], [], []], "in \\[0, 6\\)"),
        ([[], [], []], "nothing to serve"),
    ],
)
def test_constructor_rejects_bad_expert_sets(ckpt, sets, message):
    with pytest.raises(ValueError, match=message):
        make_model(ckpt, sets=sets)


def test_constructor_fails_fast_on_missing_tensor_and_bad_shape(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    write_checkpoint(missing, "model.", 256, 128, drop=(0, 1, "up_proj.weight_scale"))
    with pytest.raises(KeyError, match="layer 0 expert 1.*up_scale"):
        make_model(SimpleNamespace(root=missing), sets=[[1, 4], [], []])

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    write_checkpoint(wrong, "model.", 256, 128, scale_groups=3)
    with pytest.raises(ValueError, match="expected BF16/F16/F32"):
        make_model(SimpleNamespace(root=wrong), sets=[[1], [], []])


def test_constructor_reads_no_weights(ckpt, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("safe_open before load()")

    monkeypatch.setattr(sm, "safe_open", refuse)
    make_model(ckpt)


# --- load: stacking order, scale conversion, shuffle, qzeros, scratch -------


def test_load_stacks_k_first_gate_then_up(ckpt, loaded):
    inter, hidden = ckpt.inter, ckpt.hidden
    for layer, held in enumerate(SETS):
        holder = loaded.layer_module(layer)
        if not held:
            assert holder is None
            continue
        assert holder.w13_weight_packed.shape == (len(held), hidden // 8, 2 * inter)
        assert holder.w2_weight_packed.shape == (len(held), inter // 8, hidden)
        for slot, expert in enumerate(held):
            raw = ckpt.raw[(layer, expert)]
            gate, up, down = (
                raw[p][0].tolist() for p in ("gate_proj", "up_proj", "down_proj")
            )
            # Row j of w13 holds K words 8j..8j+7 of every output column n:
            # columns [0, inter) are gate, [inter, 2*inter) are up.
            w13 = [
                [gate[n][j] for n in range(inter)] + [up[n][j] for n in range(inter)]
                for j in range(hidden // 8)
            ]
            w2 = [[down[n][j] for n in range(hidden)] for j in range(inter // 8)]
            assert holder.w13_weight_packed[slot].tolist() == w13
            assert holder.w2_weight_packed[slot].tolist() == w2


def test_load_converts_bf16_scales_to_k_first_fp16(ckpt, loaded):
    inter, hidden = ckpt.inter, ckpt.hidden
    holder = loaded.layer_module(2)
    assert holder.w13_weight_scale.dtype == torch.float16
    assert holder.w2_weight_scale.dtype == torch.float16
    for slot, expert in enumerate(SETS[2]):
        raw = ckpt.raw[(2, expert)]
        gate, up, down = (
            raw[p][1].tolist() for p in ("gate_proj", "up_proj", "down_proj")
        )
        s13 = [
            [gate[n][g] for n in range(inter)] + [up[n][g] for n in range(inter)]
            for g in range(hidden // GROUP)
        ]
        s2 = [[down[n][g] for n in range(hidden)] for g in range(inter // GROUP)]
        assert holder.w13_weight_scale[slot].tolist() == s13
        assert holder.w2_weight_scale[slot].tolist() == s2
    # reading the bf16 bits as fp16 would give different numbers
    gate_scale = ckpt.raw[(2, SETS[2][0])]["gate_proj"][1]
    misread = gate_scale.view(torch.int16).view(torch.float16).t()
    assert not torch.equal(misread, holder.w13_weight_scale[0][:, :inter])


def test_shuffle_runs_once_per_expert_matrix_on_the_stacked_weights(ckpt):
    fake = Shuffle(fn=lambda t: t.add_(1))
    model = make_model(ckpt, shuffle=fake)
    model.load()
    n_experts = sum(len(s) for s in SETS)
    inter, hidden = ckpt.inter, ckpt.hidden
    w13_calls = [c for c in fake.calls if c[0] == (hidden // 8, 2 * inter)]
    w2_calls = [c for c in fake.calls if c[0] == (inter // 8, hidden)]
    assert len(w13_calls) == n_experts and len(w2_calls) == n_experts
    assert all(c[1] == torch.int32 and c[2] == 4 for c in fake.calls)
    reference = make_model(ckpt)
    reference.load()
    for layer in (0, 2):
        for name in ("w13_weight_packed", "w2_weight_packed"):
            got = getattr(model.layer_module(layer), name)
            want = getattr(reference.layer_module(layer), name) + 1
            assert torch.equal(got, want)


def test_load_synthesizes_symmetric_qzeros_and_shares_scratch(ckpt, loaded):
    first, last = loaded.layer_module(0), loaded.layer_module(2)
    groups13, groups2 = ckpt.hidden // GROUP, ckpt.inter // GROUP
    for holder, n in ((first, 2), (last, 4)):
        assert holder.w13_qzeros.shape == (n, groups13, 2 * ckpt.inter // 8)
        assert holder.w2_qzeros.shape == (n, groups2, ckpt.hidden // 8)
        # stored 7 per nibble: the kernel adds 1, zero point 8 = uint4b8 bias
        assert (holder.w13_qzeros == 0x77777777).all()
        assert (holder.w2_qzeros == 0x77777777).all()
    for name in ("rdna2_w1_buf", "rdna2_act_buf", "rdna2_empty_tw"):
        assert getattr(first, name) is getattr(last, name)
    assert first.rdna2_w1_buf.shape == (128, 2 * ckpt.inter)  # 2 * 3 rows: floor
    assert first.rdna2_act_buf.shape == (128, ckpt.inter)


def test_production_scratch_rows_follow_max_capture_rows(ckpt):
    model = make_model(ckpt, top_k=10, max_capture_rows=48)
    model.load()
    assert model.layer_module(0).rdna2_w1_buf.shape == (480, 2 * ckpt.inter)


def test_progress_reports_every_layer_in_order(ckpt):
    seen = []
    make_model(ckpt).load(progress=lambda layer, n: seen.append((layer, n)))
    assert seen == [(0, 3), (1, 3), (2, 3)]


def test_load_opens_one_file_per_shard_and_reads_in_offset_order(ckpt, monkeypatch):
    real = sm.safe_open
    opened, reads = [], []

    class Spy:
        def __init__(self, path, **kwargs):
            self.path = path
            self.inner = real(path, **kwargs)
            opened.append(path)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

        def get_tensor(self, name):
            reads.append((Path(self.path).name, name))
            return self.inner.get_tensor(name)

    model = make_model(ckpt)
    monkeypatch.setattr(sm, "safe_open", Spy)
    model.load()

    assert sorted(Path(p).name for p in opened) == [shard_file(0), shard_file(1)]
    weight_map = json.loads((ckpt.root / sm.INDEX_NAME).read_text())["weight_map"]
    wanted = {
        name
        for layer, held in enumerate(SETS)
        for expert in held
        for name in sm.ExpertServerModel.expert_tensor_names(
            weight_map, layer, expert
        ).values()
    }
    assert len(reads) == len(wanted) == 6 * sum(len(s) for s in SETS)
    assert {name for _, name in reads} == wanted

    offsets = {}
    for i in (0, 1):
        data = (ckpt.root / shard_file(i)).read_bytes()
        (size,) = struct.unpack("<Q", data[:8])
        for name, info in json.loads(data[8 : 8 + size]).items():
            if name != "__metadata__":
                offsets[(shard_file(i), name)] = info["data_offsets"][0]
    layer_of = re.compile(r"layers\.(\d+)\.")
    per_layer: dict[int, list[tuple[str, int]]] = {}
    for shard, name in reads:
        layer = int(layer_of.search(name).group(1))
        per_layer.setdefault(layer, []).append((shard, offsets[(shard, name)]))
    assert sorted(per_layer) == [0, 2]
    for order in per_layer.values():
        assert order == sorted(order)


def test_failed_load_leaves_the_model_unusable_and_restores_the_op(ckpt):
    ops = prod().ops
    original = ops.gptq_shuffle

    def boom(tensor, bits):
        raise RuntimeError("shuffle exploded")

    model = make_model(ckpt, shuffle=boom)
    with pytest.raises(RuntimeError, match="shuffle exploded"):
        model.load()
    assert ops.gptq_shuffle is original
    with pytest.raises(RuntimeError, match="has not completed"):
        model.moe(0, *call_inputs(model, 2))
    assert model.layer_module(0) is None


def test_default_shuffle_is_the_custom_op_and_an_injected_one_is_restored(
    ckpt, monkeypatch
):
    import vllm._custom_ops as ops

    default = Shuffle()
    monkeypatch.setattr(ops, "gptq_shuffle", default)
    model = sm.ExpertServerModel(
        ckpt.root, SETS, "cpu", num_experts=EXPERTS, top_k=TOP_K, group_size=GROUP
    )
    model.load()
    assert len(default.calls) == 2 * sum(len(s) for s in SETS)

    injected = Shuffle()
    default.calls.clear()
    make_model(ckpt, shuffle=injected).load()
    assert len(injected.calls) == 2 * sum(len(s) for s in SETS)
    assert default.calls == [] and ops.gptq_shuffle is default


def test_production_module_is_imported_lazily():
    imported = []
    for node in ast.parse(inspect.getsource(sm)).body:
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported  # the scan sees the real imports
    assert not any("compressed_tensors" in name for name in imported)
    assert not any(name.startswith("vllm._custom_ops") for name in imported)


# --- memory accounting ------------------------------------------------------


def test_expert_bytes_match_the_production_geometry():
    assert sm.expert_bytes(2560, 640, 128) == (2_534_400, 19_200)


@pytest.mark.parametrize(("top_k", "capture", "rows"), [(3, 2, 128), (10, 48, 480)])
def test_memory_bytes_is_experts_scratch_and_maps(ckpt, top_k, capture, rows):
    model = make_model(ckpt, top_k=top_k, max_capture_rows=capture)
    maps = LAYERS * EXPERTS * 4
    assert model.memory_bytes() == maps  # nothing loaded yet
    model.load()
    h, i = ckpt.hidden, ckpt.inter
    per_expert = (
        (h // 8) * 2 * i * 4  # w13 words
        + (i // 8) * h * 4  # w2 words
        + (h // GROUP) * 2 * i * 2  # w13 scales, fp16
        + (i // GROUP) * h * 2  # w2 scales, fp16
        + (h // GROUP) * (2 * i // 8) * 4  # w13 qzeros
        + (i // GROUP) * (h // 8) * 4  # w2 qzeros
    )
    scratch = rows * 2 * i * 2 + rows * i * 2  # shared once, not per layer
    assert model.memory_bytes() == 6 * per_expert + scratch + maps
    assert sum(sm.expert_bytes(h, i, GROUP)) == per_expert


# --- dequantization reference -----------------------------------------------


def test_dequant_reference_of_a_hand_packed_word():
    nibbles = [0, 1, 2, 3, 4, 5, 6, 15]
    word = sum(n << (4 * i) for i, n in enumerate(nibbles))
    assert word == 0xF6543210
    packed = torch.tensor([[word - 2**32]], dtype=torch.int32)  # int32: negative
    scale = torch.tensor([[0.5]], dtype=torch.bfloat16)
    got = sm.dequantize_checkpoint_weight(packed, scale, group_size=8)
    assert got.dtype == torch.float32
    assert got.tolist() == [[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 3.5]]

    two = torch.tensor([[word - 2**32, 0x8]], dtype=torch.int32)
    scales = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16)
    got = sm.dequantize_checkpoint_weight(two, scales, group_size=8)
    assert got[0, :8].tolist() == [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 3.5]
    # word 1: nibble 8 then seven zeros -> (8 - 8) * 2 = 0, then (0 - 8) * 2 = -16
    assert got[0, 8:].tolist() == [0.0] + [-16.0] * 7


def test_dequant_reference_agrees_with_the_repo_packer():
    assert sm.UINT4B8_BIAS == scalar_types.uint4b8.bias == 8
    assert scalar_types.uint4b8.size_bits == 4
    gen = torch.Generator().manual_seed(3)
    values = torch.randint(0, 16, (5, 256), dtype=torch.int32, generator=gen)
    words = pack_quantized_values_into_int32(values, scalar_types.uint4b8, packed_dim=1)
    assert torch.equal(sm.unpack_int4_words(words), values)
    scales = torch.rand(5, 2, generator=gen).to(torch.bfloat16)
    want = (values.float() - 8) * scales.float().repeat_interleave(128, dim=1)
    assert torch.equal(sm.dequantize_checkpoint_weight(words, scales, 128), want)
    with pytest.raises(ValueError, match="does not match"):
        sm.dequantize_checkpoint_weight(words, scales, 64)


# --- moe() wiring -----------------------------------------------------------


def test_moe_calls_the_production_chain_with_the_contract_arguments(
    loaded, monkeypatch
):
    module = prod()
    signature = inspect.signature(module._rdna2_fused_moe)
    calls = []

    def recorder(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)  # fails if the signature drifted
        bound.apply_defaults()
        calls.append(bound.arguments)
        return torch.full((args[0].shape[0], loaded.hidden), 3, dtype=torch.float16)

    monkeypatch.setattr(module, "_rdna2_fused_moe", recorder)
    x, ids, w = call_inputs(loaded, 5)

    out = loaded.moe(2, x, ids, w)

    (a,) = calls
    assert a["hidden_states"] is x and a["topk_ids"] is ids and a["topk_weights"] is w
    assert a["layer"] is loaded.layer_module(2)
    assert a["activation"] is MoEActivation.SILU
    assert a["apply_router_weight_on_input"] is False
    assert a["global_num_experts"] == EXPERTS
    assert a["expert_map"] is loaded.local_map[2]
    assert a["w4a8"] is False and a["zero_v2"] is False
    assert a["ignore_invalid_experts"] is True
    assert a["routed_accumulator"] is None
    # the chain's own fresh result is returned as is
    assert out.shape == (5, loaded.hidden) and out.dtype == torch.float16
    assert (out == 3).all() and out.data_ptr() not in (x.data_ptr(), ids.data_ptr())
    assert loaded.moe(2, x, ids, w).data_ptr() != out.data_ptr()


def test_a_layer_without_local_experts_returns_zeros_without_the_chain(
    loaded, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("chain launched")

    monkeypatch.setattr(prod(), "_rdna2_fused_moe", refuse)
    x, ids, w = call_inputs(loaded, 4)
    out = loaded.moe(1, x, ids, w)
    assert out.shape == (4, loaded.hidden) and out.dtype == torch.float16
    assert not out.any() and out.data_ptr() != x.data_ptr()


def test_moe_rejects_inputs_the_kernels_would_misread(loaded, monkeypatch):
    monkeypatch.setattr(prod(), "_rdna2_fused_moe", lambda *a, **k: a[0])
    x, ids, w = call_inputs(loaded, 3)
    loaded.moe(0, x, ids, w)
    with pytest.raises(ValueError, match="topk_ids"):
        loaded.moe(0, x, ids.to(torch.int64), w)
    with pytest.raises(ValueError, match="topk_ids"):
        loaded.moe(0, x, ids[:, :2].contiguous(), w)
    with pytest.raises(ValueError, match="topk_w"):
        loaded.moe(0, x, ids, w.to(torch.float16))
    with pytest.raises(ValueError, match="x must"):
        loaded.moe(0, x.float(), ids, w)
    with pytest.raises(ValueError, match="x must"):
        loaded.moe(0, x[:, :-8], ids, w)
    with pytest.raises(ValueError, match="x must"):
        loaded.moe(0, x.t().contiguous().t(), ids, w)
    with pytest.raises(IndexError):
        loaded.moe(LAYERS, x, ids, w)


def test_moe_and_selftest_need_a_completed_load_and_load_runs_once(ckpt):
    model = make_model(ckpt)
    with pytest.raises(RuntimeError, match="has not completed"):
        model.moe(0, *call_inputs(model, 1))
    with pytest.raises(RuntimeError, match="has not completed"):
        model.known_answer_selftest()
    model.load()
    with pytest.raises(RuntimeError, match="already called"):
        model.load()


# --- known-answer selftest against a torch emulation of the chain -----------


def make_emulator(variant="ok", scale=1.0):
    """The production chain in torch, reading the K-first tensors of the layer."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)

    def k_rows(words):  # [n, K/8, N] -> [n, K, N]: nibble i of word j is K index 8j+i
        n, a, b = words.shape
        nib = (words.unsqueeze(2) >> shifts.view(1, 1, 8, 1)) & 0xF
        return nib.reshape(n, a * 8, b)

    def n_cols(words):  # [n, G, N/8] -> [n, G, N]
        n, g, c = words.shape
        nib = (words.unsqueeze(3) >> shifts.view(1, 1, 1, 8)) & 0xF
        return nib.reshape(n, g, c * 8)

    def dequant(packed, scales, qzeros):
        q = k_rows(packed).float()
        zero = n_cols(qzeros).float() + 1  # GPTQ v1: the kernel adds 1 to stored zeros
        g = q.shape[1] // scales.shape[1]
        s = scales.float().repeat_interleave(g, 1)
        return (q - zero.repeat_interleave(g, 1)) * s

    def run(
        hidden_states,
        topk_weights,
        topk_ids,
        layer,
        activation,
        apply_router_weight_on_input,
        global_num_experts,
        expert_map,
        w4a8=False,
        zero_v2=False,
        ignore_invalid_experts=False,
        routed_accumulator=None,
    ):
        assert activation is MoEActivation.SILU and ignore_invalid_experts
        t, k = topk_ids.shape
        out = torch.zeros(t, layer.w2_weight_packed.shape[2])
        if variant == "zeros":
            return out.half()
        w13 = dequant(layer.w13_weight_packed, layer.w13_weight_scale, layer.w13_qzeros)
        w2 = dequant(layer.w2_weight_packed, layer.w2_weight_scale, layer.w2_qzeros)
        n, _, two_inter = w13.shape
        inter = two_inter // 2
        for r in range(t):
            for c in range(k):
                expert = int(topk_ids[r, c])
                if variant == "no_map":  # global id used as the local slot
                    slot = expert if 0 <= expert < n else -1
                else:
                    in_range = 0 <= expert < global_num_experts
                    slot = int(expert_map[expert]) if in_range else -1
                if slot < 0 and variant == "no_mask":  # masked pairs run slot 0
                    slot = 0
                if slot < 0:
                    continue
                h = hidden_states[r].float() @ w13[slot]
                gate, up = h[:inter], h[inter:]
                if variant == "swap":
                    gate, up = up, gate
                act = torch.nn.functional.silu(gate) * up
                out[r] += topk_weights[r, c] * (act @ w2[slot])
        return (out * scale).half()

    return run


def test_selftest_passes_on_a_correct_chain_and_reports_errors(loaded, monkeypatch):
    monkeypatch.setattr(prod(), "_rdna2_fused_moe", make_emulator())
    result = loaded.known_answer_selftest()
    assert result["layers"] == [0, 2]  # the empty layer is skipped
    assert result["max_abs_ref"] > 0.05  # the check has signal to discriminate
    assert result["max_abs_err"] < 0.01 * result["max_abs_ref"]
    assert result["rows"] == 6 and result["moe_ms"] >= 0 and result["ref_ms"] >= 0
    for layer, record in result["per_layer"].items():
        held = SETS[layer]
        assert {held[0], held[-1]} <= set(record["experts"]) <= set(held)
        assert record["tolerance"] > sm.SELFTEST_ABS_FLOOR
    one = loaded.known_answer_selftest(layers=[2], experts_per_layer=1)
    assert one["layers"] == [2] and len(one["per_layer"][2]["experts"]) == 1


def test_selftest_is_deterministic_per_seed(loaded, monkeypatch):
    monkeypatch.setattr(prod(), "_rdna2_fused_moe", make_emulator())
    a = loaded.known_answer_selftest(seed=7)
    b = loaded.known_answer_selftest(seed=7)
    c = loaded.known_answer_selftest(seed=8)
    for key in ("max_abs_err", "max_abs_ref"):
        assert a[key] == b[key]
    assert a["max_abs_ref"] != c["max_abs_ref"]


def test_selftest_inputs_cover_masking_and_every_selected_expert(loaded, monkeypatch):
    seen = []
    emulator = make_emulator()

    def spy(hidden, weights, ids, layer, *args, **kwargs):
        seen.append((ids.clone(), weights.clone()))
        return emulator(hidden, weights, ids, layer, *args, **kwargs)

    monkeypatch.setattr(prod(), "_rdna2_fused_moe", spy)
    loaded.known_answer_selftest(layers=[2], experts_per_layer=3, seed=5)
    (ids, weights), (decode_ids, _) = seen  # the 6-row batch, then its first 3 rows
    held = set(SETS[2])
    assert ids.shape == (6, TOP_K) and torch.equal(decode_ids, ids[:3])
    assert torch.allclose(weights.sum(-1), torch.ones(6))
    selected = {int(e) for e in ids[0]}
    assert len(selected) == 3 and selected <= held  # row 0 holds every selected one
    assert (ids == -1).any(), "no masked id"
    assert any(e >= 0 and e not in held for e in ids.flatten().tolist()), (
        "no foreign id"
    )
    assert not any(e in held for e in ids[3].tolist()), (
        "row 3 must hold no local expert"
    )
    for row in ids.tolist():  # nothing but selected experts, masked and foreign ids
        assert all(e < 0 or e in selected or e not in held for e in row)


@pytest.mark.parametrize("variant", ["zeros", "swap", "no_map", "no_mask"])
def test_selftest_catches_a_broken_chain(loaded, monkeypatch, variant):
    monkeypatch.setattr(prod(), "_rdna2_fused_moe", make_emulator(variant))
    with pytest.raises(RuntimeError, match="selftest failed"):
        loaded.known_answer_selftest()


def test_selftest_catches_unmasked_pairs_in_a_single_layer(loaded, monkeypatch):
    # Layer 0 holds no expert 0: running masked pairs as slot 0 is only visible
    # through the -1 entries and the ids of experts held elsewhere.
    monkeypatch.setattr(prod(), "_rdna2_fused_moe", make_emulator("no_mask"))
    with pytest.raises(RuntimeError, match="layer 0"):
        loaded.known_answer_selftest(layers=[0])


@pytest.mark.parametrize(
    ("scale", "passes"), [(1.005, True), (0.99, True), (1.1, False)]
)
def test_selftest_tolerance_boundary(loaded, monkeypatch, scale, passes):
    monkeypatch.setattr(prod(), "_rdna2_fused_moe", make_emulator(scale=scale))
    if passes:
        loaded.known_answer_selftest()
    else:
        with pytest.raises(RuntimeError, match="max abs error"):
            loaded.known_answer_selftest()


def test_selftest_catches_non_finite_output_and_rejects_bad_arguments(
    loaded, monkeypatch
):
    def nan_chain(hidden, *args, **kwargs):
        return torch.full((hidden.shape[0], loaded.hidden), float("nan")).half()

    monkeypatch.setattr(prod(), "_rdna2_fused_moe", nan_chain)
    with pytest.raises(RuntimeError, match="non-finite"):
        loaded.known_answer_selftest(layers=[0])
    with pytest.raises(ValueError, match="holds no expert"):
        loaded.known_answer_selftest(layers=[1])
    with pytest.raises(ValueError, match="at least 1"):
        loaded.known_answer_selftest(experts_per_layer=0)


# --- smoke CLI helpers ------------------------------------------------------


def smoke():
    return importlib.import_module("tools.rdna.hetero_moe.server_model_smoke")


def test_smoke_expert_sets_take_the_first_n_or_a_stride():
    mod = smoke()
    first = mod.build_expert_sets([0, 2], 4, num_layers=3, num_experts=512, stride=1)
    assert first == [[0, 1, 2, 3], [], [0, 1, 2, 3]]
    strided = mod.build_expert_sets([1], 3, num_layers=2, num_experts=512, stride=100)
    assert strided == [[], [0, 100, 200]]
    with pytest.raises(ValueError, match="outside"):
        mod.build_expert_sets([0], 7, num_layers=1, num_experts=512, stride=100)
    with pytest.raises(ValueError, match="layer 5"):
        mod.build_expert_sets([5], 1, num_layers=2, num_experts=512, stride=1)


def test_smoke_batches_use_held_experts_and_honour_local_frac(loaded):
    mod = smoke()
    held = set(SETS[2])
    x, ids, w, pairs, touched = mod.make_batch(loaded, 2, 8, 1.0, seed=3)
    assert x.shape == (8, loaded.hidden) and x.dtype == torch.float16
    assert ids.shape == (8, TOP_K) and ids.dtype == torch.int32
    assert w.dtype == torch.float32 and torch.allclose(w.sum(-1), torch.ones(8))
    assert all(len(set(row)) == TOP_K and set(row) <= held for row in ids.tolist())
    assert pairs == 8 * TOP_K and 3 <= touched <= len(held)
    again = mod.make_batch(loaded, 2, 8, 1.0, seed=3)
    assert torch.equal(again[1], ids) and torch.equal(again[0], x)

    none = mod.make_batch(loaded, 2, 8, 0.0, seed=3)
    assert (none[1] == -1).all() and none[3] == 0 and none[4] == 0

    # layer 0 holds 2 experts for top_k 3: the third column stays masked
    half = mod.make_batch(loaded, 0, 64, 0.5, seed=3)[1]
    assert set(half.flatten().tolist()) == {-1, 1, 4}
    assert (half[:, 2] == -1).all() and 0 < int((half >= 0).sum()) < 128


def test_smoke_cli_defaults_help_and_cpu_refusal(capsys):
    mod = smoke()
    args = mod.parse_args(["--model-dir", "/x"])
    assert (args.layers, args.rows, args.device) == ([0, 23, 47], [3, 24], "cuda:0")
    with pytest.raises(SystemExit) as exc:
        mod.parse_args(["--help"])
    assert exc.value.code == 0
    assert "--experts-per-layer" in capsys.readouterr().out
    assert mod.main(["--model-dir", "/nonexistent", "--device", "cpu"]) == 2
    assert "needs a CUDA/ROCm device" in capsys.readouterr().err
