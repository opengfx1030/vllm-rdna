# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gates of the tensor-parallel Inkling-2 serving model (TP = 2, 4).

Three layers, none of which needs a GPU:

* the sharding arithmetic (``parallel.py``: partition of heads / experts / vocabulary,
  the refusal of degrees the kernels cannot serve) and the checkpoint slicing
  (``loader.local_slice``: exact reassembly of every tensor, NVFP4 and MXFP8 experts
  included, views not copies);
* the loaded weights of the real serving code (``load_weights`` of
  ``Inkling2LoopTextForCausalLM`` on N ranks): every rank's parameters and repacked
  expert buffers reassemble to the TP=1 model bit for bit;
* the whole model on N in-process ranks (threads that meet at every collective, the sum
  taken in rank order like ``rdna_ar``): the paged serving model of ``test_paged_model``
  with the experts sharded exactly as the kernels' are, against the transformers
  reference, over chunked prefill, decode and mixed batches; every rank returns the
  same logits bit for bit.

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_tensor_parallel.py -q
"""

from __future__ import annotations

import copy
import threading
import types

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)
pytest.importorskip("vllm.models.inkling2_loop.amd.model")

from paged_harness import (  # noqa: E402
    fake_vllm_config,
    owners_of,
    patched,
    text_config,
)
from test_paged_model import (  # noqa: E402
    FP32_TOL,
    PROMPT_LENS,
    PagedModel,
    _batch_plan,
    _load_tool,
    _prompts,
    make,
    ref,
)

from vllm.models.inkling2_loop.amd import glue as glue_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import (
    loader,  # noqa: E402
    parallel,  # noqa: E402
)
from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import moe as moe_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402
from vllm.models.inkling2_loop.configs import (  # noqa: E402
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
)

# ---------------------------------------------------------------------------
# sharding arithmetic
# ---------------------------------------------------------------------------


def test_partition_helpers():
    tp = parallel.TensorParallel(4, 2)
    assert (
        tp.enabled and tp.local(24, "heads") == 6 and tp.span(24, "heads") == (12, 18)
    )
    x = torch.arange(24).view(4, 6)
    assert torch.equal(tp.narrow(x, 0, "rows"), x[2:3])
    assert torch.equal(parallel.TensorParallel(2, 1).narrow(x, 1, "cols"), x[:, 3:])
    assert not parallel.NO_TP.enabled and parallel.NO_TP.all_reduce(x) is x
    assert parallel.NO_TP.all_gather(x) is x
    with pytest.raises(ValueError, match="not divisible"):
        tp.local(10, "heads")
    with pytest.raises(ValueError, match="outside"):
        parallel.TensorParallel(2, 2)
    with pytest.raises(NotImplementedError):
        tp.all_reduce(x)  # the base class has no group to talk to


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_lm_head_shards_tile_the_vocabulary(tp):
    vocab = 200058  # padded to a multiple of 64 (200064), then split evenly
    shards = [parallel.lm_head_shard(vocab, tp, r) for r in range(tp)]
    assert {rows for _, rows, _ in shards} == {200064 // tp if tp > 1 else vocab}
    covered = 0
    for lo, rows, take in shards:
        assert lo == covered or (tp == 1 and lo == 0)
        assert 0 <= take <= rows
        covered += take
    assert covered == vocab
    # only the last shard is padded
    assert [take < rows for _, rows, take in shards] == [False] * (tp - 1) + [tp > 1]


def test_real_model_is_supported_at_1_2_4_only():
    cfg = Inkling2LoopTextConfig()  # the real Inkling-2-Mini: 24 / 8 heads, I = 1152
    for size in (1, 2, 4):
        parallel.check_supported(cfg, parallel.TensorParallel(size, 0))
    for size, needles in (
        (3, ["8 KV heads"]),
        (8, ["3 q heads per rank", "MoE intermediate size 1152 / 8"]),
    ):
        with pytest.raises(ValueError, match="cannot run with") as err:
            parallel.check_supported(cfg, parallel.TensorParallel(size, 0))
        for needle in needles:
            assert needle in str(err.value)


def test_hidden_size_must_pack_into_the_conv_state():
    """The post-attention / post-MLP conv streams (``hidden_size`` channels) are packed
    head-major over the rank's KV heads, ``width % 4 == 0`` for the kernels."""
    cfg = Inkling2LoopTextConfig()
    for size in (2, 4):
        parallel.check_supported(cfg, parallel.TensorParallel(size, 0))
    for hidden, size in ((2306, 2), (2306, 4)):  # 2306 / 4, 2306 / 2: not whole rows
        bad = copy.copy(cfg)
        bad.hidden_size = hidden
        with pytest.raises(ValueError, match="head-major"):
            parallel.check_supported(bad, parallel.TensorParallel(size, 0))


# ---------------------------------------------------------------------------
# checkpoint slicing
# ---------------------------------------------------------------------------


def _ranks(size: int):
    return [parallel.TensorParallel(size, r) for r in range(size)]


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("dst,dim", sorted(loader.ATTENTION_SPLIT_DIM.items()))
def test_attention_slices_reassemble_exactly_and_are_views(dst, dim, size):
    shape = [3, 5, 7]
    shape[dim] = 8 * size  # the sharded dimension, whatever the tensor's rank
    t = torch.randn(*shape)
    name = f"model.layers.3.{dst}"
    parts = [loader.local_slice(name, t, tp) for tp in _ranks(size)]
    assert torch.equal(torch.cat(parts, dim=dim), t)
    assert all(p.shape[dim] == 8 for p in parts)
    assert all(
        p.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for p in parts
    )
    with pytest.raises(ValueError, match="not divisible"):
        loader.local_slice(
            name, t.narrow(dim, 0, 8 * size - 1), parallel.TensorParallel(size, 0)
        )


@pytest.mark.parametrize(
    "name",
    [
        "model.embed_tokens.weight",
        "model.layers.3.input_layernorm.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.layers.3.mlp.w13_weight_amax",
        "model.layers.3.mlp.w2_weight_outer_scale_denominator",
        "model.layers.3.post_attention_convolution.conv1d.weight",
        "model.layers.3.self_attn.relative_position_bias.proj",
        "model.layers.3.q_message_gate.down_proj.weight",
        "model.layers.3.attention_input.query",
    ],
)
def test_replicated_tensors_are_returned_as_is(name):
    t = torch.randn(8, 12)
    for tp in _ranks(2) + _ranks(4):
        assert loader.local_slice(name, t, tp) is t


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("trailing", [(), (24,), (12,)])
def test_w13_slices_deinterleave_to_the_rank_gate_and_up(size, trailing):
    """The checkpoint rows are ``[g0, u0, g1, u1, ...]``: rank ``r`` keeps units
    ``[r I_l, (r+1) I_l)`` and its ``[gate_r; up_r]`` (the model layout) are the
    corresponding slices of the full de-interleaved ``[gate; up]``."""
    experts, inter = 3, 8 * size
    w13 = torch.randn(experts, 2 * inter, *trailing)
    full = torch.empty_like(w13)
    model_mod.copy_gate_up_deinterleaved(full, w13)
    gate, up = full[:, :inter], full[:, inter:]
    assert torch.equal(gate, w13[:, 0::2]) and torch.equal(up, w13[:, 1::2])
    ranks = []
    for tp in _ranks(size):
        piece = loader.local_slice("model.layers.0.mlp.w13_weight", w13, tp)
        assert piece.shape[1] == 2 * (inter // size)
        local = torch.empty_like(piece.contiguous())
        model_mod.copy_gate_up_deinterleaved(local, piece)
        ranks.append(local)
    il = inter // size
    assert torch.equal(torch.cat([r[:, :il] for r in ranks], dim=1), gate)
    assert torch.equal(torch.cat([r[:, il:] for r in ranks], dim=1), up)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("name", ["w2_weight", "w2_weight_scale"])
def test_w2_slices_split_the_intermediate_dimension(size, name):
    t = torch.randn(3, 16, 8 * size)  # [E, H, I...] (packed bytes or scale blocks)
    parts = [
        loader.local_slice(f"model.layers.0.mlp.{name}", t, tp) for tp in _ranks(size)
    ]
    assert torch.equal(torch.cat(parts, dim=2), t)


def test_tp1_slicing_is_the_identity():
    t = torch.randn(4, 8)
    for dst in (*loader.ATTENTION_SPLIT_DIM, "mlp.w13_weight", "mlp.w2_weight"):
        assert loader.local_slice(f"model.layers.1.{dst}", t, parallel.NO_TP) is t


# ---------------------------------------------------------------------------
# an in-process tensor-parallel group
# ---------------------------------------------------------------------------


class ThreadGroup:
    """``size`` threads that meet at every collective."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots: list[torch.Tensor | None] = [None] * size

    def exchange(self, rank: int, x: torch.Tensor) -> list[torch.Tensor]:
        self.slots[rank] = x.detach().clone()  # the owner may reuse its buffer at once
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()  # everyone has read before the next collective writes
        return values  # type: ignore[return-value]

    def run(self, fn):
        """``fn(rank)`` on every rank concurrently; results in rank order."""
        self.barrier.reset()  # a failed earlier run leaves it broken
        results: list = [None] * self.size
        errors: list[BaseException] = []

        def target(rank: int) -> None:
            try:
                results[rank] = fn(rank)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                self.barrier.abort()  # release the ranks waiting for this one

        threads = [threading.Thread(target=target, args=(r,)) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        if errors:
            raise (real or errors)[0]
        return results


class ThreadTP(parallel.TensorParallel):
    def __init__(self, group: ThreadGroup, rank: int) -> None:
        super().__init__(group.size, rank)
        self.group = group

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for part in self.group.exchange(self.rank, x):  # rank order, like rdna_ar
            total = part.clone() if total is None else total + part
        return total

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cat(self.group.exchange(self.rank, x), dim=dim)


_LOCAL = threading.local()


def _metadata_of_this_rank(prefix):
    return _LOCAL.getter(prefix)


@pytest.fixture(autouse=True)
def _per_thread_metadata(monkeypatch):
    """Every rank has its own caches and block tables: the metadata getter of the model
    code is looked up per thread."""
    monkeypatch.setattr(model_mod, "get_metadata", _metadata_of_this_rank)


class RankModel(PagedModel):
    """One rank of a thread group: the paged serving model with this rank's weights
    and this rank's share of the (exact, dense) experts; ``kwargs`` select another
    device / expert kernels / path (``PagedModel``), e.g. the HIP serving path of
    ``test_serving_path_gpu.py``."""

    def __init__(
        self, ckpt, tensors, dtype, tp: parallel.TensorParallel, **kwargs
    ) -> None:
        with patched(model_mod, "current_tensor_parallel", lambda: tp):
            super().__init__(ckpt, tensors, dtype, **kwargs)

    def _install_dense_experts(self, tensors) -> None:
        tp = self.lm.tp
        for i, layer in enumerate(self.lm.model.layers):
            up, down = ref.dequant_experts(tensors, f"model.llm.layers.{i}.mlp.experts")
            inter = up.shape[1] // 2
            lo, hi = tp.span(inter, "intermediate")
            up = torch.cat([up[:, lo:hi], up[:, inter + lo : inter + hi]], dim=1)
            down = down[:, :, lo:hi]
            cfg = copy.copy(self.cfg)
            cfg.intermediate_size = hi - lo
            experts = moe_mod.ExpertsReference(
                cfg, dtype=torch.float32, device=self.device
            )
            experts.gate_up_proj.data.copy_(up)
            experts.down_proj.data.copy_(down)
            layer.mlp.forward = self._dense_forward(layer.mlp, experts)

    @torch.no_grad()
    def step(self, work, metadata=True):
        spec = [(r, len(t)) for r, t in work]
        input_ids = torch.tensor(
            [x for _, t in work for x in t], dtype=torch.long, device=self.device
        )
        assert self.world is not None
        positions = self.world.positions(spec)
        mds = self.world.metadata(spec) if metadata else {}
        _LOCAL.getter = (lambda prefix: mds[prefix]) if metadata else (lambda p: None)
        hidden = self.lm(input_ids, positions)
        logits = self.lm.compute_logits(hidden).float().cpu()
        self.world.commit(spec)
        out, at = {}, 0
        for r, n in spec:
            out[r] = logits[at : at + n]
            at += n
        return out


# the default tiny model has 4 / 2 heads: 8 / 4 heads shard to 2 / 1 per rank at TP=4
TINY = {"num_heads": 8, "num_kv_heads": 4}


@pytest.fixture(scope="module", autouse=True)
def _threads():
    ref.configure_threads(8)


def _variant(name: str, tmp_path_factory, **spec_args):
    if name == "tiny1":
        spec = make.TinySpec(**TINY, **spec_args)
    else:
        spec = make.TinySpec(
            **TINY,
            num_hidden_layers=8,
            loop_start=2,
            loop_end=6,
            loop_num_passes=3,
            **spec_args,
        )
    path = tmp_path_factory.mktemp(name)
    make.write_checkpoint(path, spec, seed=0)
    return types.SimpleNamespace(
        name=name,
        path=path,
        tensors=ref.read_checkpoint(path),
        ref=ref.build_reference(path),
        groups={},  # tp size -> (ThreadGroup, [RankModel])
        logits={},
    )


@pytest.fixture(scope="module", params=["tiny1", "tiny8"])
def variant(request, tmp_path_factory):
    return _variant(request.param, tmp_path_factory)


@pytest.fixture(scope="module")
def stable(tmp_path_factory):
    """``tiny1`` with the router's selection bias scaled up until the top-k choice is
    far from any tie: a perturbation as small as an fp16 rounding flips no expert."""
    return _variant("tiny1", tmp_path_factory, router_bias_scale=100.0)


def _group(variant, size: int):
    """``size`` ranks with weights loaded (once per variant); fresh caches."""
    if size not in variant.groups:
        group = ThreadGroup(size)
        models = [
            RankModel(variant.path, variant.tensors, torch.float32, ThreadTP(group, r))
            for r in range(size)
        ]
        variant.groups[size] = (group, models)
    group, models = variant.groups[size]
    for m in models:
        m.reset()
    return group, models


def _want(variant, ids) -> torch.Tensor:
    key = tuple(ids)
    if key not in variant.logits:
        variant.logits[key] = ref.full_logits(variant.ref, ids)
    return variant.logits[key]


def _same_on_every_rank(per_rank: list[torch.Tensor]) -> torch.Tensor:
    for other in per_rank[1:]:
        assert torch.equal(per_rank[0], other), "ranks disagree"
    return per_rank[0]


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def _tp1_model(variant) -> PagedModel:
    if "tp1" not in variant.groups:
        variant.groups["tp1"] = PagedModel(variant.path, variant.tensors, torch.float32)
    return variant.groups["tp1"]


def _reassemble(
    name: str, full: torch.Tensor, parts: list[torch.Tensor]
) -> torch.Tensor:
    """The tensor the ranks' ``parts`` add up to: concatenated along the one dimension
    that shrinks by the tensor-parallel size (the fused gate / up buffers of the
    experts hold ``[gate; up]`` along it, so their halves are concatenated apart)."""
    size = len(parts)
    if parts[0].shape == full.shape:
        for p in parts:
            assert torch.equal(p, parts[0]), f"{name}: replicated tensor differs"
        return parts[0]
    dims = [d for d in range(full.dim()) if parts[0].shape[d] * size == full.shape[d]]
    assert len(dims) == 1, (name, full.shape, parts[0].shape)
    d = dims[0]
    if "w13_" in name:
        half = parts[0].shape[d] // 2
        gate = torch.cat([p.narrow(d, 0, half) for p in parts], dim=d)
        up = torch.cat([p.narrow(d, half, half) for p in parts], dim=d)
        return torch.cat([gate, up], dim=d)
    return torch.cat(parts, dim=d)


@pytest.mark.parametrize("size", [2, 4])
def test_rank_weights_reassemble_to_the_tp1_model(variant, size):
    """``load_weights`` on N ranks: every parameter and every repacked expert buffer
    (NVFP4 layers and the MXFP8 one) is this rank's slice of the TP=1 tensor."""
    _, ranks = _group(variant, size)
    full = _tp1_model(variant).lm
    full_state = {**dict(full.named_parameters()), **dict(full.named_buffers())}
    rank_states = [
        {**dict(m.lm.named_parameters()), **dict(m.lm.named_buffers())} for m in ranks
    ]
    sharded = []
    for name, want in full_state.items():
        parts = [s[name].detach() for s in rank_states]
        if name in ("lm_head.weight", "model.embed_tokens.weight"):
            # vocabulary shards, the last one zero padded
            got = torch.cat(parts, dim=0)
            assert torch.equal(got[: want.shape[0]], want)
            assert not got[want.shape[0] :].any()
            sharded.append(name)
            continue
        got = _reassemble(name, want.detach(), parts)
        assert torch.equal(got, want.detach()), name
        if parts[0].shape != want.shape:
            sharded.append(name)
    for s in rank_states:
        assert set(s) == set(full_state)
    # everything that should be sharded is: per layer the 15 attention tensors, 4 expert
    # buffers (qweight / scales of w13 and w2) and the globals of w13 / w2 are not
    kinds = {n.rsplit(".", 1)[-1] for n in sharded}
    assert {"weight", "scale", "static_weight", "w13_qweight", "w2_scales"} <= kinds
    assert all(
        ".mlp.w13_globals" not in n and ".mlp.w2_globals" not in n for n in sharded
    )


def test_rank_parameters_shrink_exactly_where_the_loader_slices(variant):
    """The model's own shapes and ``loader.local_slice`` agree on what is sharded."""
    _, ranks = _group(variant, 2)
    full = dict(_tp1_model(variant).lm.named_parameters())
    local = dict(ranks[1].lm.named_parameters())
    tp = ranks[1].lm.tp
    for name, p in full.items():
        if name in ("lm_head.weight", "model.embed_tokens.weight"):
            continue
        sliced = loader.local_slice(name, p.detach(), tp)
        assert tuple(sliced.shape) == tuple(local[name].shape), name
        assert (sliced.shape != p.shape) == (local[name].shape != p.shape), name


@pytest.mark.parametrize("size", [2, 4])
def test_embedding_rows_come_from_the_owner_and_sum_exactly(variant, size):
    """The vocabulary-sharded table gives the TP=1 embedding of every id: the first and
    last row of each shard, the ids around the shard boundaries and the table's end."""
    group, ranks = _group(variant, size)
    full = _tp1_model(variant).lm.model
    rows = ranks[0].lm.model.embed_tokens.weight.shape[0]
    top = full.embed_tokens.weight.shape[0] - 1
    assert rows * size >= top + 1 and rows < top + 1  # the table is really split
    edges = {0, 1, top - 1, top}
    for k in range(1, size):
        edges |= {k * rows - 1, k * rows, k * rows + 1}
    ids = torch.tensor(sorted(edges) + [7, 100, 321], dtype=torch.long)
    got = _same_on_every_rank(
        group.run(lambda r: ranks[r].lm.model.embed_input_ids(ids))
    )
    assert torch.equal(got, full.embed_input_ids(ids))


def test_kv_and_conv_owners_hold_only_the_rank_heads(variant):
    _, ranks = _group(variant, 4)
    full = _tp1_model(variant).lm
    for rank in ranks:
        for a, b in zip(full.model.layers, rank.lm.model.layers):
            for kv_a, kv_b in zip(a.kv_owners, b.kv_owners):
                assert kv_b.num_kv_heads * 4 == kv_a.num_kv_heads
            for c_a, c_b in zip(a.conv_owners, b.conv_owners):
                assert c_b.num_kv_heads * 4 == c_a.num_kv_heads
                # the K / V streams shrink with the heads, the two replicated streams
                # (post-attention / post-MLP) keep their channels, spread over fewer
                # heads
                (_, wk_a), (_, wk_b) = c_a.stream_ranges[0], c_b.stream_ranges[0]
                assert wk_a == wk_b  # one head's head_dim
                assert c_b.stream_ranges[2][1] == 4 * c_a.stream_ranges[2][1]


# ---------------------------------------------------------------------------
# the model on N ranks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("n", PROMPT_LENS)
def test_tp_prefill_matches_reference(variant, size, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    group, ranks = _group(variant, size)
    got = _same_on_every_rank(group.run(lambda r: ranks[r].step([(0, ids)])[0]))
    assert got.shape[-1] == 500  # the real vocabulary: the padded shard tail is cut
    err = float((got - _want(variant, ids)).abs().max())
    print(f"[tp] {variant.name} tp={size} prefill {n}: max|dlogit|={err:.3e}")
    assert err < FP32_TOL


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("chunk", [1, 7, 33])
def test_tp_chunked_prefill_matches_reference(variant, size, chunk):
    ids = _prompts()[2]
    group, ranks = _group(variant, size)

    def run(r):
        return ranks[r].run(0, ids, chunk)

    got = _same_on_every_rank(group.run(run))
    err = float((got - _want(variant, ids)).abs().max())
    print(f"[tp] {variant.name} tp={size} chunk {chunk}: max|dlogit|={err:.3e}")
    assert err < FP32_TOL


@pytest.mark.parametrize("size", [2, 4])
def test_tp_cached_decode_matches_reference(variant, size):
    ids = _prompts()[2]
    group, ranks = _group(variant, size)
    split = len(ids) - 8

    def run(r):
        rows = [ranks[r].step([(0, ids[:split])])[0][-1:]]
        for tok in ids[split : len(ids) - 1]:
            rows.append(ranks[r].step([(0, [tok])])[0])
        return torch.cat(rows)

    got = _same_on_every_rank(group.run(run))
    want = _want(variant, ids)[split - 1 : len(ids) - 1]
    err = float((got - want).abs().max())
    print(f"[tp] {variant.name} tp={size} cached decode: max|dlogit|={err:.3e}")
    assert err < FP32_TOL


@pytest.mark.parametrize("size", [2, 4])
def test_tp_batch_of_requests_matches_reference(variant, size):
    p5, p40, p70 = _prompts()
    group, ranks = _group(variant, size)
    plan = _batch_plan(p5, p40, p70)

    def run(r):
        fed: dict[int, list[int]] = {0: [], 1: [], 2: []}
        got: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
        for step in plan:
            out = ranks[r].step([(q, list(t)) for q, t in step])
            for q, t in step:
                fed[q] += t
                got[q].append(out[q])
        return fed, {q: torch.cat(v) for q, v in got.items()}

    results = group.run(run)
    fed = results[0][0]
    for q in (0, 1, 2):
        got = _same_on_every_rank([res[1][q] for res in results])
        err = float((got - _want(variant, fed[q])).abs().max())
        print(f"[tp] {variant.name} tp={size} batch request {q}: max|dlogit|={err:.3e}")
        assert err < FP32_TOL


@pytest.mark.parametrize("size", [2, 4])
def test_tp_forward_without_metadata_is_finite(variant, size):
    """Profiling / dummy runs have no attention metadata: the MoE collectives still run
    on every rank, the attention ones are skipped by every rank alike."""
    group, ranks = _group(variant, size)
    outs = group.run(lambda r: ranks[r].step([(0, _prompts()[1])], metadata=False)[0])
    assert torch.isfinite(_same_on_every_rank(outs)).all()


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("block", [1, 9])
def test_tp_logits_of_many_rows_gather_in_blocks(variant, size, block, monkeypatch):
    """Above ``LOGITS_GATHER_ROWS`` rows the head shards are gathered a block of rows
    at a time into the final buffer: the same logits as one gather, tail included."""
    monkeypatch.setattr(model_mod, "LOGITS_GATHER_ROWS", block)
    ids = _prompts()[2]
    group, ranks = _group(variant, size)
    got = _same_on_every_rank(group.run(lambda r: ranks[r].step([(0, ids)])[0]))
    assert got.shape == (len(ids), 500)  # the padded shard tail is cut
    err = float((got - _want(variant, ids)).abs().max())
    print(f"[tp] {variant.name} tp={size} gather block {block}: max|dlogit|={err:.3e}")
    assert err < FP32_TOL


def test_tp_degree_other_than_the_checkpoint_heads_is_refused(variant):
    group = ThreadGroup(8)
    with pytest.raises(ValueError, match="cannot run with"):
        RankModel(variant.path, variant.tensors, torch.float32, ThreadTP(group, 0))


def test_draft_model_refuses_tensor_parallelism(variant):
    _, ranks = _group(variant, 2)
    with pytest.raises(NotImplementedError, match="DSpark"):
        ranks[0].lm.set_aux_hidden_state_layers((1,))


# ---------------------------------------------------------------------------
# greedy sampling over a vocabulary-sharded head
# ---------------------------------------------------------------------------


class _FakeHead:
    """A CPU stand-in for ``LMHeadPack`` that follows the contract of
    ``inkling2_linear_lm_head`` (partial maxima over ``nparts`` contiguous chunks of the
    rank's valid columns, global ids through ``col_base``, ``-inf`` padding)."""

    def __init__(self, weight, n_valid, col_base):
        self.weight, self.n_valid, self.col_base = weight, n_valid, col_base

    @property
    def rows(self):
        return self.weight.shape[0]

    def __call__(self, x, logits, part_val, part_idx, softcap=30.0, variant=-1):
        z = torch.tanh((x.float() @ self.weight.float().T) / softcap) * softcap
        z[:, self.n_valid :] = float("-inf")
        nparts = part_val.shape[1]
        chunk = -(-self.n_valid // nparts)
        part_val.fill_(float("-inf"))
        part_idx.fill_(self.col_base)
        for p in range(nparts):
            lo, hi = p * chunk, min((p + 1) * chunk, self.n_valid)
            if lo >= hi:
                continue
            v, i = z[:, lo:hi].max(dim=1)  # first index of the maximum
            part_val[:, p] = v
            part_idx[:, p] = (self.col_base + lo + i).to(torch.int32)
        if logits is not None:
            logits.copy_(z)


def _reduce(part_val, part_idx, out):
    """``inkling2_linear_argmax_reduce``: the largest value, the lowest id on a tie."""
    assert part_val.is_contiguous() and part_idx.is_contiguous()
    best = part_val.max(dim=1, keepdim=True).values
    ids = torch.where(
        part_val == best, part_idx.to(torch.int64), torch.iinfo(torch.int64).max
    )
    out.copy_(ids.min(dim=1).values)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("rows", [1, 3, 5, 17])
def test_greedy_token_ids_gathers_the_argmax_partials(monkeypatch, size, rows):
    from vllm import _custom_ops as ops

    monkeypatch.setattr(ops, "inkling2_linear_argmax_reduce", _reduce)
    vocab, hidden = 500, 16
    gen = torch.Generator().manual_seed(rows)
    weight = torch.randn(vocab, hidden, generator=gen)
    x = torch.randn(rows, hidden, generator=gen)
    # the best id of row 0 appears twice, in two shards: the lower id has to win
    weight[300] = weight[7] = 8 * x[0] / x[0].norm()
    full = torch.tanh((x @ weight.T) / 30.0) * 30.0
    want = full.argmax(dim=1)
    assert int(want[0]) == 7
    group = ThreadGroup(size)

    def run(r):
        tp = ThreadTP(group, r)
        lo, rows_r, take = parallel.lm_head_shard(vocab, size, r)
        shard = torch.zeros(rows_r, hidden)
        shard[:take] = weight[lo : lo + take]
        me = types.SimpleNamespace(
            tp=tp,
            _softcap=30.0,
            _argmax_parts=24,
            _lm_head_pack=lambda: _FakeHead(shard, take, lo),
        )
        return model_mod.Inkling2LoopTextForCausalLM.greedy_token_ids(me, x)

    got = _same_on_every_rank(group.run(run))
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# the fp16 conv ring at TP > 1
# ---------------------------------------------------------------------------


def _ring_logits(model, ids, chunk: int, conv_dtype) -> torch.Tensor:
    """Logits of ``ids`` fed ``chunk`` tokens per step with the conv rings in
    ``conv_dtype``: one rank (``PagedModel``) or a thread group ``(group, ranks)``."""
    if isinstance(model, PagedModel):
        return model.reset(conv_dtype=conv_dtype).run(0, ids, chunk)
    group, ranks = model

    def run(r):
        return ranks[r].reset(conv_dtype=conv_dtype).run(0, ids, chunk)

    return _same_on_every_rank(group.run(run))


SUM_ORDER_TOL = 5e-4  # fp32 compute, N ranks against one rank: 1.4e-4 at most seen


def _conv_owners(model: PagedModel) -> list:
    return [o for o, g in owners_of(model.lm.model.layers) if g == "conv"]


def _replicated_ring(owner, stream: int) -> torch.Tensor:
    """The ring of a replicated stream (post-attention / post-MLP) as
    ``[blocks, rows, channels]``: the same channel order whatever the rank's head
    count."""
    o0, w = owner.stream_ranges[stream]
    ring = owner.kv_cache[..., o0 : o0 + w]  # [blocks, H, rows, w]
    return ring.permute(0, 2, 1, 3).reshape(ring.shape[0], ring.shape[2], -1)


def _rounding_flips(a: PagedModel, b: PagedModel) -> float:
    """Fraction of the replicated-stream ring entries of two fp32-ring runs (the same
    blocks and positions) whose fp16 rounding differs."""
    flips = total = 0
    for oa, ob in zip(_conv_owners(a), _conv_owners(b)):
        for stream in (state_mod.STREAM_ATTN, state_mod.STREAM_MLP):
            x, y = _replicated_ring(oa, stream), _replicated_ring(ob, stream)
            seen = torch.isfinite(x) & torch.isfinite(y)
            flips += int((x.half()[seen] != y.half()[seen]).sum())
            total += int(seen.sum())
    return flips / total


@pytest.mark.parametrize("chunk", [1, 7])
@pytest.mark.parametrize("size", [2, 4])
def test_tp_fp16_conv_ring_only_rounds_the_history(stable, size, chunk):
    """The engine keeps the conv rings in fp16.  Where TP=1 feeds the conv fp16 GEMM
    outputs (stored exactly), a TP rank feeds the fp32 sum of the o_proj partials: the
    current token's taps are exact and its history comes back from the ring rounded.
    With fp32 compute (nothing else rounds) on a router-stable model:

    * fp32 rings: N ranks and one rank equal each other and transformers up to the fp32
      summation order: the shards, the replicated streams and the ring addressing add
      nothing;
    * fp16 rings: the rounding costs the same on N ranks as on one, and what N ranks
      and one rank still differ by is less than that cost: the fp32 summation order
      moves a stored value by a few ulps, which flips its fp16 rounding for the ~1% of
      the entries that sit near a boundary.
    """
    ids = _prompts()[2]
    one, tp = _tp1_model(stable), _group(stable, size)
    exact = _ring_logits(tp, ids, chunk, torch.float32)
    one_exact = _ring_logits(one, ids, chunk, torch.float32)
    flips = _rounding_flips(tp[1][0], one)
    rounded = _ring_logits(tp, ids, chunk, torch.float16)
    one_rounded = _ring_logits(one, ids, chunk, torch.float16)

    def err(a, b):
        return float((a - b).abs().max())

    ring, one_ring = err(rounded, exact), err(one_rounded, one_exact)
    print(
        f"[tp-ring] {stable.name} tp={size} chunk {chunk}: max|dlogit| fp32 ring vs "
        f"transformers {err(exact, _want(stable, ids)):.3e}; tp{size} fp16 ring vs "
        f"tp{size} fp32 ring {ring:.3e}; tp1 fp16 ring vs tp1 fp32 ring "
        f"{one_ring:.3e}; tp{size} vs tp1 with fp16 rings "
        f"{err(rounded, one_rounded):.3e}, with fp32 rings "
        f"{err(exact, one_exact):.3e}; fp16 rounding of the stored replicated "
        f"streams differs in {flips:.2e} of the entries"
    )
    assert err(exact, _want(stable, ids)) < SUM_ORDER_TOL
    assert err(exact, one_exact) < SUM_ORDER_TOL
    assert 4 * SUM_ORDER_TOL < ring < 0.1
    assert 4 * SUM_ORDER_TOL < one_ring < 0.1
    assert ring < 2 * one_ring and one_ring < 2 * ring
    assert err(rounded, one_rounded) < max(ring, one_ring)
    assert 0 < flips < 0.05


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------

HIP_SHAPE = {"head_dim": 128, "d_rel": 16, "num_heads": 24, "num_kv_heads": 8}


@pytest.mark.parametrize("size", [1, 2, 4])
def test_workspace_is_sized_for_the_rank_share(size):
    cfg = text_config(**HIP_SHAPE)
    ws = glue_mod.Inkling2GlueWorkspace(cfg, 48, "cpu", tp_size=size)
    inter = cfg.intermediate_size // size
    assert inter == ws.I
    assert ws.f16["gu"].numel() == ws.moe_chunk * ws.K * 2 * inter
    assert ws.f16["act"].numel() == ws.moe_chunk * ws.K * inter
    v = ws.bind(37)
    assert v.o_out.shape == (37, cfg.hidden_size)
    # the o_proj partial of a K-shard is fp32 (summed by the all-reduce); TP=1 keeps the
    # fp16 output that feeds the short convolution directly
    assert v.o_out.dtype == (torch.float32 if size > 1 else torch.float16)
    assert (
        glue_mod.attention_workspace_numel(24 // size, 8 // size) <= ws.attn_ws.numel()
    )
    with pytest.raises(ValueError, match="not divisible"):
        glue_mod.Inkling2GlueWorkspace(cfg, 48, "cpu", tp_size=3)


# ---------------------------------------------------------------------------
# the caches of a rank
# ---------------------------------------------------------------------------

# Pages of one layer visit of the real Inkling-2-Mini (fp16, 64-token blocks) per TP
# degree, as (K|V page, conv ring page, conv row width): the K|V page follows the
# rank's kv heads (8 / tp); a conv row holds, per head, the rank's k and v channels
# (128 each) next to the replicated post-attention and post-MLP streams (2304
# channels each) spread over the rank's heads, and the page is 3 ring rows of it.
REAL_PAGES = {
    1: (262_144, 39_936, 832),
    2: (131_072, 33_792, 1408),
    4: (65_536, 30_720, 2560),
}


def _real_rank(size: int, rank: int = 0, drafts: int = 0):
    """The model of rank ``rank`` of ``size`` of the real Inkling-2-Mini (meta device)
    and the vLLM config slice it was built with."""
    fake = fake_vllm_config(
        "real-model", Inkling2LoopConfig(), num_speculative_tokens=drafts
    )
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with (
            patched(state_mod, "get_current_vllm_config", lambda: fake),
            patched(loader, "read_expert_formats", lambda m, r, n: ["nvfp4"] * n),
            torch.device("meta"),
        ):
            lm = model_mod.Inkling2LoopTextForCausalLM(
                vllm_config=fake, serving=False, tp=parallel.TensorParallel(size, rank)
            )
    finally:
        torch.set_default_dtype(old)
    return lm, fake


def _specs_of(lm, fake):
    """``(kv specs, conv specs)`` of every layer visit, in model order."""
    kv, conv = [], []
    for layer in lm.model.layers:
        kv += [o.get_kv_cache_spec(fake) for o in layer.kv_owners]
        conv += [o.get_kv_cache_spec(fake) for o in layer.conv_owners]
    return kv, conv


@pytest.mark.parametrize("drafts", [0, 7])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_real_rank_cache_specs_and_pages(size, drafts):
    """The owners of a rank of the real model declare the pages the engine allocates:
    K|V pages of the rank's kv heads, a conv ring page of the rank's own row width
    (not the TP=1 row divided by the degree)."""
    from vllm.models.inkling2_loop.amd.conv_state import Inkling2ConvSpec
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    cfg = Inkling2LoopConfig().text_config
    kv_page, conv_page, row = REAL_PAGES[size]
    nkv = cfg.num_key_value_heads // size
    ring = cfg.sconv_kernel_size - 1 + drafts
    # the pinned numbers follow from the config
    assert kv_page == 64 * nkv * 2 * cfg.head_dim * 2
    assert row == 2 * cfg.head_dim + 2 * cfg.hidden_size // nkv
    assert conv_page == nkv * 3 * row * 2

    seen = []
    for rank in sorted({0, size - 1}):
        lm, fake = _real_rank(size, rank, drafts)
        kv, conv = _specs_of(lm, fake)
        assert len(kv) == 48 and len(conv) == 48
        assert sum(isinstance(s, SlidingWindowSpec) for s in kv) == 40
        assert sum(isinstance(s, FullAttentionSpec) for s in kv) == 8
        for spec in kv:
            assert spec.num_kv_heads == nkv and spec.head_size == cfg.head_dim
            assert spec.page_size_bytes == kv_page
        for spec in conv:
            assert isinstance(spec, Inkling2ConvSpec)
            assert spec.num_kv_heads == nkv and spec.head_size == row
            assert spec.block_size == spec.tokens_per_state == 64
            assert spec.state_content_bytes == ring * row * 2
            assert spec.page_size_bytes == nkv * ring * row * 2
            if not drafts:
                assert spec.page_size_bytes == conv_page
        for layer in lm.model.layers:
            for owner in layer.conv_owners:
                assert owner.ring_rows == ring and owner.row_width == row
                # k and v are the rank's heads (one head's head_dim of each per
                # row), the two replicated streams share the heads
                widths = [w for _, w in owner.stream_ranges]
                assert widths == [cfg.head_dim] * 2 + [cfg.hidden_size // nkv] * 2
        seen.append((kv, conv))
    for kv, conv in seen[1:]:  # every rank holds the same pages
        assert (kv, conv) == seen[0]


def lm_owners(fake) -> dict:
    """Cache owners of a model built under ``fake``, by registered layer name."""
    return dict(fake.compilation_config.static_forward_context)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    """The engine config of the real Inkling-2-Mini at ``max_model_len`` 262,144 and
    1,024-token chunks as the planner tool ``kv_plan.py`` builds it for a V620 (no
    GPU, no weights), and the tool itself.  Building the config registers the repo's
    config classes with ``AutoConfig``; that registration is dropped again, the
    transformers reference tests of the session load transformers' own classes."""
    from transformers import CONFIG_MAPPING

    kv_plan = _load_tool("kv_plan")
    ckpt = tmp_path_factory.mktemp("real-config")
    hf_config = Inkling2LoopConfig()
    hf_config.architectures = ["Inkling2LoopForConditionalGeneration"]
    hf_config.save_pretrained(ckpt)
    args = types.SimpleNamespace(
        ckpt=str(ckpt), max_num_batched_tokens=1024, max_num_seqs=2, layout="BLNHC"
    )
    with (
        patched(CONFIG_MAPPING, "_extra_content", dict(CONFIG_MAPPING._extra_content)),
        kv_plan.v620_platform(),
    ):
        cfg = kv_plan.engine_config(args, 262144)
    return types.SimpleNamespace(tool=kv_plan, cfg=cfg)


@pytest.mark.parametrize("drafts", [0, 7])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_planner_tool_derives_the_pages_the_rank_owners_declare(plan, size, drafts):
    """``kv_plan.py`` builds the rank's model itself and widens the conv ring of its
    specs by the speculative tokens: the result is what the owners declare when the
    engine runs with a speculative config.  (Dividing the TP=1 specs by the degree
    gives 19,968 B instead of 33,792 B conv pages at TP=2.)"""
    kv_plan = plan.tool
    with patched(loader, "read_expert_formats", lambda m, r, n: ["nvfp4"] * n):
        tool = kv_plan.layer_specs(plan.cfg, size)
    lm, fake = _real_rank(size, 0, drafts)
    own = {name: o.get_kv_cache_spec(fake) for name, o in lm_owners(fake).items()}
    assert set(tool) == set(own)
    for name, spec in tool.items():
        assert kv_plan.with_drafts(spec, drafts) == own[name], name


@pytest.mark.parametrize("drafts", [0, 7])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_planner_groups_hold_the_rank_pages(plan, size, drafts):
    """One pool block holds the 8 layers of a full-attention group; the conv layers
    (a page of ``3 + drafts`` rows each) fill as many blocks as their pages need, so a
    rank's conv state is split into ``48 / (block // page)`` groups: 1 / 2 / 3 at
    TP = 1 / 2 / 4 (4 / 6 / 10 with 7 draft tokens), not one at every degree."""
    from vllm.v1.core import kv_cache_planning as planning

    kv_plan, cfg = plan.tool, plan.cfg
    text = cfg.model_config.hf_config.text_config
    kv_page, _, row = REAL_PAGES[size]
    nkv = text.num_key_value_heads // size
    conv_page = nkv * (text.sconv_kernel_size - 1 + drafts) * row * 2
    with patched(loader, "read_expert_formats", lambda m, r, n: ["nvfp4"] * n):
        specs = kv_plan.layer_specs(cfg, size)
    builder, groups = kv_plan.rank_groups(cfg, specs, drafts)
    pool = planning._pool_bytes_per_block_from_builder(builder, cfg, groups)
    assert pool == 8 * kv_page
    n_conv = _cdiv(48, pool // conv_page)
    assert (
        n_conv
        == {(1, 0): 1, (2, 0): 2, (4, 0): 3, (1, 7): 4, (2, 7): 6, (4, 7): 10}[
            size, drafts
        ]
    )
    rows = kv_plan.group_rows(cfg, groups)
    kinds = [kind for kind, _, _ in rows]
    assert (
        kinds.count("SlidingWindowSpec") == 5 and kinds.count("FullAttentionSpec") == 1
    )
    conv = [r for r in rows if r[0] == "Inkling2ConvSpec"]
    assert len(conv) == n_conv == len(kinds) - 6
    assert sum(layers for _, layers, _ in conv) == 48
    assert max(layers for _, layers, _ in conv) * conv_page <= pool
    # blocks one request of ``max_model_len`` needs per group: the window (or the conv
    # taps) plus the tokens in flight, a block more for the partial one; all of it for
    # the full-attention group
    length, flight = cfg.model_config.max_model_len, cfg.max_in_flight_tokens
    window = text.sliding_window_size - 1
    for kind, layers, blocks in rows:
        if kind == "SlidingWindowSpec":
            assert blocks == _cdiv(min(window + flight, length), 64) + 1
        elif kind == "Inkling2ConvSpec":
            taps = text.sconv_kernel_size - 1
            assert blocks == _cdiv(min(taps + flight, length), 64) + 1
        else:
            assert blocks == length // 64 and layers == 8


@pytest.mark.parametrize("drafts", [0, 7])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_allocated_pages_bind_to_the_rank_owners(plan, size, drafts):
    """The runner's allocation of the planner's config (the real ``allocate_kv_cache``,
    on CPU): every layer gets a view of its owner's shape, a conv owner's ring comes
    out as ``[blocks, kv heads, rows, row width]``, and inside one group the pages of
    the layers of a block do not overlap and stay inside the block."""
    from vllm.v1.core import kv_cache_planning as planning
    from vllm.v1.worker.utils import allocate_kv_cache

    cfg = plan.cfg
    lm, fake = _real_rank(size, 0, drafts)
    owners = lm_owners(fake)
    # the owners' own specs (their ring already has the rows for the drafts)
    specs = {name: o.get_kv_cache_spec(fake) for name, o in owners.items()}
    groups = planning.get_kv_cache_groups(cfg, specs)
    builder = planning.get_kv_cache_config_builder(cfg)
    blocks = 5
    kv_cfg = builder.get_kv_cache_config_from_groups(cfg, groups, num_blocks=blocks)
    layout = cfg.cache_config.get_resolved_kv_cache_layout()
    views = allocate_kv_cache(
        kv_cfg,
        torch.device("cpu"),
        layout,
        None,
        allocate=lambda n: torch.zeros(n, dtype=torch.int8),
    )
    assert set(views) == set(owners)
    text = cfg.model_config.hf_config.text_config
    nkv = text.num_key_value_heads // size
    for name, view in views.items():
        owner = owners[name]
        owner.kv_cache = view
        assert view.dtype == torch.float16 and view.shape[:2] == (blocks, nkv)
        if isinstance(owner, state_mod.Inkling2ConvOwner):
            assert owner.state.shape == (
                blocks,
                nkv,
                text.sconv_kernel_size - 1 + drafts,
                REAL_PAGES[size][2],
            )
        else:
            assert view.shape[2:] == (64, 2 * text.head_dim)
    pool = kv_cfg.kv_cache_tensors[0].size // blocks
    for group in kv_cfg.kv_cache_groups:
        for block in range(1, blocks):
            spans = []
            for name in group.layer_names:
                v = views[name]
                base = v.untyped_storage().data_ptr()
                first = (v.storage_offset() + block * v.stride(0)) * v.element_size()
                extent = 1 + sum(
                    (n - 1) * s for n, s in zip(v.shape[1:], v.stride()[1:])
                )
                spans.append((first, first + extent * v.element_size(), base))
            spans.sort()
            assert spans[0][0] >= block * pool and spans[-1][1] <= (block + 1) * pool
            assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), group
