# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The serving Inkling-2 DSpark draft (``vllm/models/inkling2_loop/amd/dspark.py``).

CPU: how the checkpoint is packed and loaded, what the loader refuses, the config
checks, the attention metadata and the KV owners.

gfx1030 (``_rocm_C`` with the ``inkling2_dspark`` ops): the whole draft -- tap store,
context K/V precompute, three bidirectional layers over a shuffled paged cache, the
recurrent Markov head with the folded LM head -- driven the way the V2 runner drives
it (``tools/rdna/inkling2/dspark_check.py``: ``HipDraft``) against the float64 oracle
of the same file, on random weights of the real shapes and on the real
``mtp.safetensors`` with the real target head (skipped without
``~/hfcache/inkling2-mini``).  The oracle is teacher-forced with the draft's own
tokens, so every step's logits are compared on the same Markov inputs.  A CUDA graph
of the block pass replays bit-exact against the eager pass on new data.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
CKPT = Path("~/hfcache/inkling2-mini").expanduser()
MTP = CKPT / "mtp.safetensors"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oracle = sys.modules.get("dspark_check") or _load("dspark_check")

TINY = oracle.DraftConfig(
    hidden_size=64,
    num_layers=3,
    num_heads=6,
    num_kv_heads=2,
    head_dim=16,
    intermediate_size=96,
    markov_rank=8,
    block_size=8,
    window=40,
    local_layer_ids=(0, 2),
    rope_theta=10000.0,
)
VOCAB, PADDED = 300, 304


def _h(t: torch.Tensor) -> torch.Tensor:
    """What an fp16 parameter holds for the bf16 checkpoint value ``t``."""
    return t.bfloat16().half().double()


@pytest.fixture(scope="module")
def draft():
    return oracle.round_to_bf16(oracle.random_draft(TINY, PADDED, seed=3))


@pytest.fixture()
def model(draft):
    return oracle.build_model(draft, VOCAB, PADDED, 4, 4096, "cpu")


@pytest.fixture(scope="module")
def mod():
    from vllm.models.inkling2_loop.amd import dspark

    return dspark


# ---------------------------------------------------------------------------
# CPU: loading
# ---------------------------------------------------------------------------
def test_loader_packs_every_tensor_where_the_kernels_read_it(model, draft):
    m = model.model
    h = TINY.hidden_size
    for i, layer in enumerate(m.layers):

        def g(name, i=i):
            return draft.layer(i, name)

        expected = {
            "qkv_w": torch.cat(
                [g(f"self_attn.{p}_proj.weight") for p in ("q", "k", "v")]
            ),
            "kvctx_w": torch.cat(
                [g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")]
            ),
            "o_w": g("self_attn.o_proj.weight"),
            "down_w": g("mlp.down_proj.weight"),
            "q_norm_w": g("self_attn.q_norm.weight"),
            "k_norm_w": g("self_attn.k_norm.weight"),
            "kctx_norm_w": g("self_attn.k_norm_ctx.weight"),
            "input_ln_w": g("input_layernorm.weight"),
            "post_ln_w": g("post_attention_layernorm.weight"),
            # gate and up rows alternate: row 2j gate_j, row 2j + 1 up_j
            "gate_up_w": torch.stack(
                [g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 1
            ).reshape(-1, h),
        }
        for name, ref in expected.items():
            got = getattr(layer, name)
            assert got.dtype == torch.float16, name
            assert torch.equal(got.double(), _h(ref)), (i, name)
        assert torch.equal(m.dflare_w[i].double(), _h(g("dflare_norm.weight")))
    assert m.fuse_weight.dtype == torch.float32
    assert torch.equal(m.fuse_weight.double(), draft["fuse_weight"].bfloat16().double())
    assert torch.equal(m.norm_w.double(), _h(draft["norm.weight"]))
    assert torch.equal(m.mask_embed.double(), _h(draft["mask_embed"]))
    head = m.markov_head
    w = draft["markov_head.rnn_proj.weight"]
    r = TINY.markov_rank
    assert torch.equal(
        head.dflare_w[0].double(), _h(draft["markov_head.dflare_norm.weight"])
    )
    assert torch.equal(head.w1.double(), _h(draft["markov_head.markov_w1.weight"]))
    assert torch.equal(head.rnn_proj_w.double(), _h(w))
    assert head.rnn_bias.dtype == torch.float32
    assert torch.equal(
        head.rnn_bias.double(), draft["markov_head.rnn_proj.bias"].bfloat16().double()
    )
    # the derived layouts the recurrence reads: [state; w1[prev]] transposed, then
    # the h and f_rnn column blocks as contiguous GEMM weights
    assert torch.equal(head.rnn_wt.double(), _h(w[:, : 2 * r]).t())
    assert torch.equal(head.rnn_wh.double(), _h(w[:, 2 * r : 2 * r + h]))
    assert torch.equal(head.rnn_wf.double(), _h(w[:, 2 * r + h :]))
    assert head.rnn_wt.is_contiguous() and head.rnn_wf.is_contiguous()
    hidden_out = draft["markov_head.hidden_out.weight"]
    assert torch.equal(head.hidden_out_t.double(), _h(hidden_out).t())
    assert torch.equal(
        head.hidden_out_b.double(), _h(draft["markov_head.hidden_out.bias"])
    )


def test_every_checkpoint_name_routes_to_exactly_the_parameters(model, draft):
    names = oracle.expected_names(TINY.num_layers)
    assert sorted(draft.tensors) == names
    dest = {model._route(n)[0] for n in names}
    assert dest == {n for n, _ in model.named_parameters()}
    assert (
        model.load_weights((k, v.bfloat16()) for k, v in draft.tensors.items()) == dest
    )


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("missing", ValueError, "incomplete or repeats"),
        ("duplicate", ValueError, "incomplete or repeats"),
        ("unknown", KeyError, "unmapped"),
        ("layer", KeyError, "no layer 3"),
        ("shape", ValueError, "does not fit"),
        ("dtype", ValueError, "expected bf16"),
        ("overflow", ValueError, "does not fit fp16"),
    ],
)
def test_loader_refuses_wrong_checkpoints(model, draft, case, error, match):
    pick = "layers.1.mlp.down_proj.weight"
    items = [(k, v.bfloat16()) for k, v in draft.tensors.items()]
    if case == "missing":
        items = [kv for kv in items if kv[0] != pick]
    elif case == "duplicate":
        items.append(next(kv for kv in items if kv[0] == pick))
    elif case == "unknown":
        items.append(
            ("layers.0.self_attn.rotary_emb.inv_freq", torch.zeros(8).bfloat16())
        )
    elif case == "layer":
        items.append(("layers.3.mlp.down_proj.weight", torch.zeros(4, 4).bfloat16()))
    elif case == "shape":
        items = [
            (k, torch.zeros(5, 7).bfloat16() if k == pick else v) for k, v in items
        ]
    elif case == "dtype":
        items = [(k, v.float() if k == pick else v) for k, v in items]
    elif case == "overflow":
        big = draft[pick].bfloat16().clone()
        big[0, 0] = 1e6  # a finite bf16 value above the fp16 maximum
        items = [(k, big if k == pick else v) for k, v in items]
    with pytest.raises(error, match=match):
        model.load_weights(iter(items))


def test_fp16_report_counts_what_the_conversion_flushes(mod):
    report = mod._Fp16Report()
    src = torch.tensor([1.0, 1e-9, 0.0, 1e-12, 6e-8, -3.0], dtype=torch.bfloat16)
    dst = torch.empty(6, dtype=torch.float16)
    report.convert("x", src, dst)
    # 1e-9 and 1e-12 are below half the smallest fp16 subnormal; 6e-8 is not; 0 stays
    assert (report.flushed, report.values, report.max_abs) == (2, 6, 3.0)
    assert dst[0] == 1.0 and dst[5] == -3.0 and dst[4] != 0


# ---------------------------------------------------------------------------
# CPU: config, metadata, owners
# ---------------------------------------------------------------------------
def test_the_draft_checks_its_block_width_and_context_length(draft):
    with pytest.raises(ValueError, match="num_speculative_tokens = 7"):
        oracle.build_model(draft, VOCAB, PADDED, 4, 4096, "cpu", False, num_steps=6)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        oracle.build_model(draft, VOCAB, PADDED, 4, 2_000_000, "cpu", False)


def test_attention_metadata_builder_references_the_persistent_buffers(mod):
    from vllm.v1.attention.backend import AttentionCGSupport

    builder = mod.Inkling2DraftMetadataBuilder(None, ["a"], None, torch.device("cpu"))
    common = types.SimpleNamespace(
        block_table_tensor=torch.zeros(4, 8, dtype=torch.int32),
        slot_mapping=torch.arange(64, dtype=torch.int64),
        seq_lens=torch.full((4,), 9, dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=16,
    )
    md = builder.build(0, common)
    assert md.block_table is common.block_table_tensor
    assert md.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
    assert md.slot_mapping.shape == (16,) and md.seq_lens.shape == (2,)
    assert md.seq_lens.data_ptr() == common.seq_lens.data_ptr()
    assert builder.get_cudagraph_support(None, None) == AttentionCGSupport.UNIFORM_BATCH
    assert builder.supports_draft_decode_metadata_update
    builder.update_draft_decode_metadata(md)  # nothing to refresh: the buffers are live
    assert md.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
    # capture reads every request: one key each, written into the live buffer
    builder.build_for_cudagraph_capture(common)
    assert common.seq_lens.tolist() == [1, 1, 9, 9]


def test_kv_owners_name_window_and_backend_the_three_layers(model, mod):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    assert model.get_draft_kv_cache_layer_names() == [
        f"model.layers.{i}.self_attn.attn" for i in range(3)
    ]
    assert model.get_draft_attn_causal() == [False] * 3
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=16))
    specs = [layer.kv_owner.get_kv_cache_spec(cfg) for layer in model.model.layers]
    for i, spec in enumerate(specs):
        if i in TINY.local_layer_ids:
            assert isinstance(spec, SlidingWindowSpec), i
            assert spec.sliding_window == TINY.window
        else:
            assert type(spec) is FullAttentionSpec, i
        assert (spec.block_size, spec.num_kv_heads, spec.head_size) == (16, 2, 16)
        assert spec.dtype == torch.float16
    for layer in model.model.layers:
        backend = layer.kv_owner.get_attn_backend()
        assert backend.get_builder_cls() is mod.Inkling2DraftMetadataBuilder


def test_speculator_config_resolves_to_the_inkling2_speculator():
    from vllm.models.inkling2_loop.speculator_config import (
        Inkling2DSparkSpeculatorConfig,
    )
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    cls = resolve_obj_by_qualname(Inkling2DSparkSpeculatorConfig.speculator_class)
    assert cls.__name__ == "Inkling2DSparkSpeculator"
    assert issubclass(cls, DSparkSpeculator)


# ---------------------------------------------------------------------------
# gfx1030
# ---------------------------------------------------------------------------
def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    if "gfx1030" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    return hasattr(torch.ops._rocm_C, "inkling2_dspark_argmax")


requires_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx1030 GPU with the inkling2_dspark ops"
)
needs_real = pytest.mark.skipif(
    not (MTP.exists() and (CKPT / "config.json").exists()),
    reason=f"real Inkling-2-Mini checkpoint not found at {CKPT}",
)

REAL = oracle.DraftConfig()  # the shapes of mtp.safetensors
SYN_VOCAB, SYN_PADDED = 3000, 3072

# Largest deviation from the float64 oracle that the fp16 draft may show.  The bounds
# are about 10x the errors observed on the shapes below (docs/rdna2/inkling2-dspark.md
# lists them: hidden rel. L2 <= 1.3e-3, logits <= 2.4e-2, K / V <= 1.4e-2 for
# magnitudes up to 30): loose enough for another GPU / compiler; a wrong window or tap
# is expected to move them by orders of magnitude [INFERENCE: not mutated here].
TOL = {
    "k_rel": 4e-3,  # context K / V cache, relative to the largest oracle magnitude
    "hidden_rel_l2": 1.5e-2,
    "f_rnn_abs": 2e-2,
    "logit_abs": 0.25,  # softcapped logits: |z| <= 30
    "same_math_rel_l2": 3e-3,  # two runs of the same math, different summation order
}


def _synthetic():
    """A random draft of the real shapes plus a random target head."""
    d = oracle.round_to_bf16(oracle.random_draft(REAL, SYN_PADDED, seed=5))
    g = torch.Generator().manual_seed(11)
    lm = torch.randn(SYN_VOCAB, REAL.hidden_size, generator=g, dtype=torch.float64)
    lm *= 0.125  # logits of std ~6 on unit-scale hidden rows
    anchors = [17, 901, 2222, 5, 1234, 777, 42, 2999]
    emb = {
        a: 0.5 * torch.randn(REAL.hidden_size, generator=g, dtype=torch.float64)
        for a in anchors
    }
    norm = 1.0 + 0.2 * torch.randn(REAL.hidden_size, generator=g, dtype=torch.float64)
    return d, lm, emb, norm, anchors


@pytest.fixture(scope="module")
def synthetic_hip():
    if not _gpu_ready():
        pytest.skip("needs a gfx1030 GPU with the inkling2_dspark ops")
    d, lm, emb, norm, anchors = _synthetic()
    hip = oracle.HipDraft(
        d, lm, emb, norm, padded_vocab=SYN_PADDED, max_reqs=40, page_size=16
    )
    return hip, anchors


def _contexts(lengths, anchors, seed=0):
    return [
        oracle.Context(
            oracle.synthetic_taps(n, REAL, seed + i), anchors[i % len(anchors)]
        )
        for i, n in enumerate(lengths)
    ]


def _check(hip, res, *, tokens_exact=False):
    cache = oracle.cache_error(hip, hip.world)
    rep = oracle.compare(hip, res)
    print("cache", cache, "\ndraft", rep)
    scale = cache["kv_max_magnitude"]
    assert cache["k_max_abs"] <= TOL["k_rel"] * scale, cache
    assert cache["v_max_abs"] <= TOL["k_rel"] * scale, cache
    assert rep["hidden_rel_l2"] <= TOL["hidden_rel_l2"], rep
    assert rep["f_rnn_max_abs"] <= TOL["f_rnn_abs"], rep
    assert rep["logit_max_abs"] <= TOL["logit_abs"], rep
    # the draft's token is the oracle's argmax up to the logit error (a near tie)
    assert rep["token_gap"] <= 2 * TOL["logit_abs"], rep
    if tokens_exact:
        assert rep["token_mismatch"] == 0, rep
    return rep


@requires_gpu
def test_hip_draft_matches_the_oracle_on_random_weights(synthetic_hip):
    """Contexts shorter than, just past and far past the 1024-key window."""
    hip, anchors = synthetic_hip
    res = hip.propose(_contexts([37, 1100, 2100], anchors))
    assert res.tokens.shape == (3, 7) and res.logits.shape == (3, 7, SYN_VOCAB)
    assert torch.isfinite(res.hidden).all() and torch.isfinite(res.logits).all()
    _check(hip, res)


@requires_gpu
@pytest.mark.parametrize("page_size", [16, 64])
def test_page_size_does_not_change_the_answer(synthetic_hip, page_size):
    hip, anchors = synthetic_hip
    ctx = _contexts([300, 1030], anchors, seed=7)
    old = hip.page_size
    hip.page_size = 16
    base = hip.propose(ctx)
    hip.page_size = page_size
    other = hip.propose(ctx)
    hip.page_size = old
    # the same math on different pages: the attention reads the same keys, only the
    # split-KV partition (hence the fp32 summation order) can differ
    gap = (base.hidden - other.hidden).norm() / base.hidden.norm()
    assert gap <= TOL["same_math_rel_l2"], float(gap)
    assert torch.equal(base.tokens, other.tokens)


@requires_gpu
def test_requests_beyond_one_head_chunk_match_the_oracle(synthetic_hip):
    """40 requests: two Markov-head chunks (32 + 8) and the dense GEMM designs."""
    hip, anchors = synthetic_hip
    lengths = [20 + 3 * i for i in range(40)]
    res = hip.propose(_contexts(lengths, anchors, seed=21))
    assert res.logits.shape[0] == 40
    _check(hip, res)


@requires_gpu
def test_padded_requests_leave_the_real_ones_untouched(synthetic_hip):
    hip, anchors = synthetic_hip
    ctx = _contexts([50, 400], anchors, seed=3)
    base = hip.propose(ctx, capture_logits=False)
    padded = hip.propose(ctx, capture_logits=False, pad=2)
    w = hip.world
    rows = 4 * hip.mod.BLOCK
    hidden = hip.model._scratch.out[:rows].float().cpu()
    assert torch.isfinite(hidden).all(), "padding rows must stay finite"
    # the split-KV partition follows the batch width, so only the summation order of
    # the real rows may differ
    gap = (base.hidden - padded.hidden).norm() / base.hidden.norm()
    assert gap <= TOL["same_math_rel_l2"], float(gap)
    assert torch.equal(base.tokens, padded.tokens)
    assert w.seq_lens.tolist()[2:] == [0, 0]


@requires_gpu
def test_f_rnn_reads_the_last_committed_row_after_rejections(synthetic_hip):
    hip, anchors = synthetic_hip
    ctx = _contexts([30, 64, 9], anchors, seed=4)
    w = hip.prepare(ctx)
    rejected = [0, 3, 8]
    out = torch.zeros(
        hip.max_reqs, REAL.hidden_size, dtype=torch.float16, device="cuda"
    )
    for dtype in (torch.int32, torch.int64):  # the runner's buffer is int32
        out.zero_()
        num_rejected = torch.tensor(rejected, dtype=dtype, device="cuda")
        hip.model.compute_f_rnn(w.stored, w.query_start_loc, num_rejected, out, 3)
        for r, x in enumerate(ctx):
            row = x.taps.shape[0] - 1 - rejected[r]
            ref = oracle.f_rnn(hip.d, w.used_taps[r][row])
            err = (out[r].double().cpu() - ref).abs().max()
            assert err <= TOL["f_rnn_abs"], (dtype, r, float(err))


@requires_gpu
def test_block_pass_replays_in_a_hip_graph(synthetic_hip):
    """The whole draft step (backbone + Markov head) records into one graph, and a
    replay on new taps / anchors equals the eager pass bit for bit."""
    hip, anchors = synthetic_hip
    first = _contexts([40, 300], anchors, seed=31)
    second = _contexts([40, 300], anchors[2:] + anchors[:2], seed=32)
    w = hip.prepare(first)
    hip.context_pass(w)
    hip.block_pass(w)  # warm-up: lazy init outside the capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hip.block_pass(w)
    hip.refill(w, second)
    hip.context_pass(w)
    eager_hidden = hip.block_pass(w).clone()
    eager_tokens = w.spec.draft_tokens.clone()
    captured.zero_()
    w.spec.draft_tokens.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    rows = w.num_reqs * hip.mod.BLOCK
    assert torch.equal(captured[:rows], eager_hidden[:rows])
    assert torch.equal(w.spec.draft_tokens, eager_tokens)
    assert (eager_tokens[:2] != 0).any()


@requires_gpu
@needs_real
def test_hip_draft_matches_the_oracle_on_the_real_checkpoint():
    import json

    cfg_json = json.loads((CKPT / "config.json").read_text())
    text = cfg_json["text_config"]
    cfg = oracle.DraftConfig.from_mtp_config(
        cfg_json["mtp_config"], text.get("final_logit_softcapping", 30.0)
    )
    d = oracle.load_draft(MTP, cfg)
    anchors = [1234, 5678, 90210]
    emb, norm, unembed = oracle.read_target_head(
        CKPT, text["unpadded_vocab_size"], anchors
    )
    hip = oracle.HipDraft(
        d,
        unembed,
        dict(zip(anchors, emb)),
        norm,
        padded_vocab=d["markov_head.markov_w1.weight"].shape[0],
        max_reqs=4,
    )
    res = hip.propose(
        [
            oracle.Context(oracle.synthetic_taps(n, cfg, i), a)
            for i, (n, a) in enumerate(zip([300, 1500, 90], anchors))
        ]
    )
    assert res.logits.shape == (3, 7, text["unpadded_vocab_size"])
    _check(hip, res)
