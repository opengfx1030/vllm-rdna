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
    # derive() frees the repacked checkpoint originals once rnn_wt/wh/wf and
    # hidden_out_t are built.
    assert head.rnn_proj_w.numel() == 0 and head.hidden_out_w.numel() == 0
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


@pytest.mark.parametrize(
    ("schedule", "max_seqs", "expected"),
    [
        (None, 8, 7),  # no schedule: every drafted token can be verified
        ([[1, 1, 3], [2, 8, 0]], 8, 3),
        ([[1, 8, 5]], 8, 5),
        ([[1, 2, 2], [3, 8, 6]], 2, 2),  # batch sizes past max_num_seqs never occur
        ([[1, 2, 2], [3, 4, 6]], 8, 6),  # the last range carries on to max_num_seqs
        ([[1, 8, 0]], 8, 1),  # the head still needs one step to run
        ([[1, 8, 9]], 8, 7),  # the block has seven tokens
    ],
)
def test_the_draft_runs_as_many_markov_steps_as_the_schedule_verifies(
    mod, schedule, max_seqs, expected
):
    cfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            num_speculative_tokens_per_batch_size=schedule
        ),
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_seqs),
    )
    assert mod.scheduled_draft_tokens(cfg, 7) == expected


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


def _ids_file(tmp_path, ids):
    import numpy as np

    path = tmp_path / "draft_vocab.npy"
    np.save(path, np.asarray(ids))
    return str(path)


def test_draft_vocab_gathers_the_rows_of_the_listed_ids(model, tmp_path):
    m, h = model.model.markov_head, TINY.hidden_size
    g = torch.Generator().manual_seed(1)
    lm = torch.randn(VOCAB, h, generator=g).half()
    m.fold_w = torch.randn(VOCAB, TINY.markov_rank, generator=g).half()
    m.fold_b = torch.randn(VOCAB, generator=g)
    model.lm_head = types.SimpleNamespace(weight=lm)
    ids = [0, 3, 4, 70, 299]
    got = model._derive_draft_vocab(_ids_file(tmp_path, ids))
    sel = torch.tensor(ids)
    assert torch.equal(got.ids, sel)
    assert got.local_idx.dtype == torch.int32
    assert torch.equal(got.local_idx, sel.to(torch.int32))
    assert not hasattr(got, "weight")
    assert torch.equal(got.fold_w, m.fold_w[sel])
    assert torch.equal(got.fold_b, m.fold_b[sel])
    assert got.fold_w.is_contiguous()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_draft_vocab_shards_across_tp_ranks(draft, tmp_path, tp_size):
    from vllm.models.inkling2_loop.amd.parallel import TensorParallel, lm_head_shard

    # Vocab padded to multiple of 64 divisible by tp_size (320 = 64 * 5; 256 = 64 * 4).
    vocab, padded = 250, 256
    g = torch.Generator().manual_seed(7)
    full_lm = torch.randn(vocab, TINY.hidden_size, generator=g).half()
    full_fw = torch.randn(vocab, TINY.markov_rank, generator=g).half()
    full_fb = torch.randn(vocab, generator=g)

    # One list spanning all ranks, and one list concentrated on rank 0 (empty shards).
    for ids in ([0, 17, 63, 64, 127, 128, 191, 192, 249], [0, 5, 12]):
        path = _ids_file(tmp_path, ids)
        collected_ids = []
        for rank in range(tp_size):
            lo, rows, take = lm_head_shard(vocab, tp_size, rank)
            m = oracle.build_model(draft, vocab, padded, 4, 4096, "cpu", False)
            m._tp = TensorParallel(tp_size, rank)
            m._head_first, m._head_rows, m._head_take = lo, rows, take
            lm_shard = torch.zeros(rows, TINY.hidden_size, dtype=torch.float16)
            fw_shard = torch.zeros(rows, TINY.markov_rank, dtype=torch.float16)
            fb_shard = torch.zeros(rows, dtype=torch.float32)
            lm_shard[:take].copy_(full_lm[lo : lo + take])
            fw_shard[:take].copy_(full_fw[lo : lo + take])
            fb_shard[:take].copy_(full_fb[lo : lo + take])
            m.lm_head = types.SimpleNamespace(weight=lm_shard)
            m.model.markov_head.fold_w = fw_shard
            m.model.markov_head.fold_b = fb_shard

            got = m._derive_draft_vocab(path)
            expected = [i for i in ids if lo <= i < lo + take]
            sel = torch.tensor(expected, dtype=torch.int64)
            assert torch.equal(got.ids, sel)
            assert got.local_idx.dtype == torch.int32
            assert torch.equal(got.local_idx, (sel - lo).to(torch.int32))
            assert not hasattr(got, "weight")
            if expected:
                assert torch.equal(
                    lm_shard.index_select(0, got.local_idx.long()), full_lm[sel]
                )
                assert torch.equal(got.fold_w, full_fw[sel])
                assert torch.equal(got.fold_b, full_fb[sel])
            else:
                assert got.fold_w.shape == (0, TINY.markov_rank)
                assert got.fold_b.shape == (0,)
            collected_ids.extend(expected)
        assert collected_ids == ids


def test_greedy_draft_vocab_keeps_only_subset_folded_head_and_sizes_scratch(
    draft, tmp_path, monkeypatch
):
    from vllm.models.inkling2_loop.amd import dspark as mod
    from vllm.models.inkling2_loop.amd import parallel
    from vllm.models.inkling2_loop.amd import state as state_mod

    ids = [0, 3, 5, 11, 19, 42, 77, 100, 199, 250, 299]  # 11 ids -> pad 16
    path = _ids_file(tmp_path, ids)
    hf = oracle.draft_hf_config(draft, VOCAB, PADDED)

    # Greedy + draft_vocab: scratch is sized to active_vocab, full fold_w/b freed,
    # and attach_target frees hidden_out_t / hidden_out_b.
    cfg_greedy = oracle.fake_vllm_config(hf, TINY.num_steps, 4, 4096)
    cfg_greedy.speculative_config.speculator_options = {"draft_vocab": path}
    monkeypatch.setattr(state_mod, "get_current_vllm_config", lambda: cfg_greedy)
    monkeypatch.setattr(mod, "current_tensor_parallel", lambda: parallel.NO_TP)
    with torch.device("cpu"):
        m_greedy = mod.Inkling2DSparkModel(vllm_config=cfg_greedy)
    m_greedy.load_weights((k, v.bfloat16()) for k, v in draft.tensors.items())
    assert m_greedy._scratch.base.shape[1] == 16
    assert m_greedy._scratch.logits.shape[1] == 16

    g = torch.Generator().manual_seed(9)
    lm = torch.randn(VOCAB, TINY.hidden_size, generator=g).half()
    target = types.SimpleNamespace(
        model=types.SimpleNamespace(
            embed_norm=types.SimpleNamespace(weight=torch.ones(TINY.hidden_size).half())
        )
    )
    m_greedy.lm_head = types.SimpleNamespace(weight=lm)
    m_greedy.model.embed_tokens = types.SimpleNamespace(weight=lm)
    # Reference full fold before attach_target frees hidden_out_t/b
    ref_fw = (lm.float() @ m_greedy.model.markov_head.hidden_out_t.float().t()).to(
        torch.float16
    )
    ref_fb = lm.float() @ m_greedy.model.markov_head.hidden_out_b.float()
    m_greedy.attach_target(target)
    head_g = m_greedy.model.markov_head
    assert head_g.fold_w.numel() == 0 and head_g.fold_b.numel() == 0
    assert head_g.hidden_out_t.numel() == 0 and head_g.hidden_out_b.numel() == 0
    assert m_greedy._draft_vocab is not None
    sel = torch.tensor(ids, dtype=torch.int64)
    assert torch.equal(m_greedy._draft_vocab.fold_w, ref_fw[sel])
    assert torch.allclose(m_greedy._draft_vocab.fold_b, ref_fb[sel], atol=1e-5)

    # Probabilistic sampling keeps full-vocab fold_w / fold_b and full-vocab scratch.
    cfg_prob = oracle.fake_vllm_config(hf, TINY.num_steps, 4, 4096)
    cfg_prob.speculative_config.draft_sample_method = "probabilistic"
    cfg_prob.speculative_config.speculator_options = {"draft_vocab": path}
    monkeypatch.setattr(state_mod, "get_current_vllm_config", lambda: cfg_prob)
    with torch.device("cpu"):
        m_prob = mod.Inkling2DSparkModel(vllm_config=cfg_prob)
    m_prob.load_weights((k, v.bfloat16()) for k, v in draft.tensors.items())
    assert m_prob._scratch.base.shape[1] == -(-VOCAB // 8) * 8
    m_prob.lm_head = types.SimpleNamespace(weight=lm)
    m_prob.model.embed_tokens = types.SimpleNamespace(weight=lm)
    m_prob.attach_target(target)
    assert m_prob.model.markov_head.fold_w.shape == (VOCAB, TINY.markov_rank)
    assert m_prob.model.markov_head.fold_b.shape == (VOCAB,)


def test_per_batch_size_markov_steps_tracks_dynamic_schedule(draft, monkeypatch):
    from vllm.models.inkling2_loop.amd import dspark as mod
    from vllm.models.inkling2_loop.amd import parallel
    from vllm.models.inkling2_loop.amd import state as state_mod

    hf = oracle.draft_hf_config(draft, VOCAB, PADDED)
    cfg = oracle.fake_vllm_config(hf, TINY.num_steps, 16, 4096)
    cfg.speculative_config.num_speculative_tokens_per_batch_size = [
        [1, 1, 7],
        [2, 2, 2],
        [3, 16, 0],
    ]
    monkeypatch.setattr(state_mod, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(mod, "current_tensor_parallel", lambda: parallel.NO_TP)
    with torch.device("cpu"):
        m = mod.Inkling2DSparkModel(vllm_config=cfg)
    assert m.steps_for_reqs(1) == 7
    assert m.steps_for_reqs(2) == 2
    assert m.steps_for_reqs(3) == 1
    assert m.steps_for_reqs(16) == 1


@pytest.mark.parametrize(
    ("max_seqs", "target_cap", "max_draft_reqs", "expected_tokens"),
    [
        (2, 4, 2, [8, 16]),
        (4, 8, 2, [8, 16]),
        (4, 8, 4, [8, 16, 24, 32]),
        (16, 32, 2, [8, 16]),
    ],
)
def test_dflash_cudagraph_manager_captures_all_draft_reqs_above_target_cap(
    monkeypatch, max_seqs, target_cap, max_draft_reqs, expected_tokens
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as cg_mod
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

    monkeypatch.setattr(
        cg_mod,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(cg_mod.current_platform, "get_global_graph_pool", lambda: None)

    cc = types.SimpleNamespace(
        prefill_cudagraph_capture_token_sizes=[],
        decode_cudagraph_capture_request_sizes=[1],
        max_cudagraph_capture_size=target_cap,
        max_decode_cudagraph_capture_tokens=target_cap,
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        use_breakable_cudagraph=False,
    )
    vllm_cfg = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=max_seqs, max_num_batched_tokens=1024
        ),
        compilation_config=cc,
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1
        ),
        cache_config=types.SimpleNamespace(use_kda_recoverssm=False),
        kernel_config=types.SimpleNamespace(monokernel=False),
        additional_config={},
        uniform_decode_query_len=1,
    )
    mgr = DFlashCudaGraphManager(
        vllm_cfg,
        torch.device("cpu"),
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_lens={8: max_draft_reqs},
    )
    # Target compilation_config must not be mutated.
    assert cc.max_cudagraph_capture_size == target_cap
    full_descs = mgr._capture_descs.get(CUDAGraphMode.FULL, [])
    assert sorted(d.num_tokens for d in full_descs) == expected_tokens
    mgr._graphs_captured = True
    for r in range(1, max_draft_reqs + 1):
        desc = mgr.dispatch(
            num_reqs=r,
            num_tokens=r * 8,
            uniform_token_count=8,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_tokens >= r * 8


@pytest.mark.parametrize(
    ("schedule", "max_seqs", "expected_max_reqs", "expected_mode"),
    [
        (None, 16, 16, "FULL_DECODE_ONLY"),
        ([[1, 2, 2], [3, 16, 0]], 16, 2, "FULL_DECODE_ONLY"),
        ([[1, 2, 7], [3, 16, 0]], 16, 2, "FULL_DECODE_ONLY"),
        ([[1, 4, 2], [5, 16, 0]], 16, 4, "FULL_DECODE_ONLY"),
        ([[1, 16, 0]], 16, 0, "NONE"),
    ],
)
def test_dflash_cudagraph_manager_caps_capture_to_dynamic_schedule(
    monkeypatch, schedule, max_seqs, expected_max_reqs, expected_mode
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_mod

    captured: dict = {}

    class _DummyMgr:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_lens):
            captured["mode"] = cudagraph_mode
            captured["decode_query_lens"] = decode_query_lens

    monkeypatch.setattr(dflash_mod, "DFlashCudaGraphManager", _DummyMgr)

    spec_cfg = types.SimpleNamespace(
        num_speculative_tokens_per_batch_size=schedule,
        uses_dynamic_speculative_decoding=lambda: schedule is not None,
    )
    stub = types.SimpleNamespace(
        _speculator_name="DFlash",
        vllm_config=types.SimpleNamespace(speculative_config=spec_cfg),
        device=torch.device("cpu"),
        max_num_reqs=max_seqs,
        num_speculative_steps=7,
        num_query_per_req=8,
        attn_cg_support=types.SimpleNamespace(
            min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
            min_cg_attn_backend="TEST",
        ),
    )
    stub._max_draft_reqs = lambda: dflash_mod.DFlashSpeculator._max_draft_reqs(stub)
    assert stub._max_draft_reqs() == expected_max_reqs
    dflash_mod.DFlashSpeculator.init_cudagraph_manager(
        stub, CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert captured["mode"] == getattr(CUDAGraphMode, expected_mode)
    assert captured["decode_query_lens"] == {8: max(expected_max_reqs, 1)}


@pytest.mark.parametrize(
    "ids",
    [[], [5, 5, 6], [7, 3], [-1, 4], [4, 300], [[1, 2], [3, 4]]],
    ids=["empty", "repeat", "descending", "negative", "past the vocabulary", "2-d"],
)
def test_draft_vocab_refuses_lists_that_are_not_ascending_ids(model, tmp_path, ids):
    model.lm_head = types.SimpleNamespace(weight=torch.zeros(VOCAB, TINY.hidden_size))
    with pytest.raises(ValueError, match="strictly ascending token ids"):
        model._derive_draft_vocab(_ids_file(tmp_path, ids))


def test_the_step_cost_follows_the_draft_vocabulary(tmp_path):
    from vllm.models.inkling2_loop.speculator_config import (
        _V620_STEP_COST,
        Inkling2DSparkSpeculatorConfig,
    )

    spec = types.SimpleNamespace(
        speculator_options={
            "adaptive_k": {"decay": 0.5},
            "draft_vocab": _ids_file(tmp_path, list(range(2000))),
        },
        draft_model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(vocab_size=20000)
        ),
    )
    policy = Inkling2DSparkSpeculatorConfig(spec).make_num_tokens_policy([0, 7])
    assert policy is not None
    cost = policy.cost
    assert cost.draft_ms < _V620_STEP_COST.draft_ms - 2.0
    assert cost.draft_token_ms < _V620_STEP_COST.draft_token_ms
    assert cost.forward_ms == _V620_STEP_COST.forward_ms


def test_adaptive_verification_is_refused_with_a_pointer_to_adaptive_k():
    from vllm.models.inkling2_loop.speculator_config import (
        Inkling2DSparkSpeculatorConfig,
    )

    spec = types.SimpleNamespace(
        speculator_options=None, enable_adaptive_verification=True
    )
    with pytest.raises(ValueError, match="adaptive_k"):
        Inkling2DSparkSpeculatorConfig.validate_options(spec)
    spec.enable_adaptive_verification = False
    spec.speculator_options = {"adaptive_k": {}, "draft_vocab": "ids.npy"}
    Inkling2DSparkSpeculatorConfig.validate_options(spec)
    spec.speculator_options = {"adaptive_k": {}}
    cfg = Inkling2DSparkSpeculatorConfig(spec)
    assert cfg.uses_adaptive_num_tokens
    policy = cfg.make_num_tokens_policy([0, 7, 7])
    assert policy is not None and policy.choose(["r"]) >= 1
    spec.speculator_options = {"adaptive_k": False}
    assert not Inkling2DSparkSpeculatorConfig(spec).uses_adaptive_num_tokens
    spec.speculator_options = {"adaptive_k": {"decay": 0.5, "draft_ms": 2.3}}
    custom = Inkling2DSparkSpeculatorConfig(spec).make_num_tokens_policy([0, 7])
    assert custom is not None
    assert custom.decay == 0.5 and custom.cost.draft_ms == 2.3
    for bad in ({"draft_vocab": 3}, {"adaptive_k": {"nope": 1}}, {"other": 1}):
        spec.speculator_options = bad
        with pytest.raises(ValueError):
            Inkling2DSparkSpeculatorConfig.validate_options(spec)


def test_adaptive_step_cost_selects_tp_table_and_scales_vocab(tmp_path):
    """Adaptive K picks the measured TP=1/2/4 V620 step-cost model and scales
    draft_vocab head savings inversely with TP."""
    import numpy as np

    from vllm.models.inkling2_loop.speculator_config import (
        _V620_STEP_COST,
        _V620_TP2_STEP_COST,
        _V620_TP4_STEP_COST,
        Inkling2DSparkSpeculatorConfig,
    )

    ids_path = tmp_path / "vocab.npy"
    np.save(ids_path, np.arange(32768, dtype=np.int32))
    spec = types.SimpleNamespace(
        speculator_options={"adaptive_k": {}},
        enable_adaptive_verification=False,
        draft_model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(vocab_size=200058)
        ),
    )
    cfg = Inkling2DSparkSpeculatorConfig(spec)
    p1 = cfg.make_num_tokens_policy([0, 7], tensor_parallel_size=1)
    p2 = cfg.make_num_tokens_policy([0, 7], tensor_parallel_size=2)
    p4 = cfg.make_num_tokens_policy([0, 7], tensor_parallel_size=4)
    assert p1 is not None and p1.cost == _V620_STEP_COST
    assert p2 is not None and p2.cost == _V620_TP2_STEP_COST
    assert p4 is not None and p4.cost == _V620_TP4_STEP_COST

    # With draft_vocab, savings halve at TP=2 and quarter at TP=4.
    spec.speculator_options = {"adaptive_k": {}, "draft_vocab": str(ids_path)}
    v1 = cfg.make_num_tokens_policy([0, 7], tensor_parallel_size=1)
    v2 = cfg.make_num_tokens_policy([0, 7], tensor_parallel_size=2)
    v4 = cfg.make_num_tokens_policy([0, 7], tensor_parallel_size=4)
    assert v1 is not None and v2 is not None and v4 is not None
    drop1 = _V620_STEP_COST.draft_ms - v1.cost.draft_ms
    drop2 = _V620_TP2_STEP_COST.draft_ms - v2.cost.draft_ms
    drop4 = _V620_TP4_STEP_COST.draft_ms - v4.cost.draft_ms
    assert pytest.approx(drop2, rel=1e-6) == drop1 / 2.0
    assert pytest.approx(drop4, rel=1e-6) == drop1 / 4.0


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
@pytest.mark.parametrize("steps", [1, 3, 6])
def test_the_markov_chain_stops_at_the_largest_verified_token(synthetic_hip, steps):
    """A schedule that verifies at most ``steps`` tokens makes the draft run that many
    Markov steps: they are the full chain's first tokens bit for bit (which the tests
    above hold to the oracle), the later columns are never written, and the backbone
    is untouched."""
    hip, anchors = synthetic_hip
    ctx = _contexts([37, 300, 1100], anchors, seed=13)
    full = hip.propose(ctx, capture_logits=False)
    assert full.tokens[:, steps:].all(), "a full chain writes every column"
    full_steps = hip.model._markov_steps
    hip.model._markov_steps = steps
    try:
        cut = hip.propose(ctx, capture_logits=False)
    finally:
        hip.model._markov_steps = full_steps
    assert torch.equal(cut.tokens[:, :steps], full.tokens[:, :steps])
    assert not cut.tokens[:, steps:].any(), "the unused columns keep their zeros"
    assert torch.equal(cut.hidden, full.hidden)


@requires_gpu
def test_draft_vocab_restricts_the_greedy_chain(synthetic_hip, tmp_path):
    """With every id listed the chain is the full chain bit for bit; with a subset
    it drafts only listed ids, and its first token is the full chain's argmax over
    them (the later tokens continue from different ones)."""
    import numpy as np

    hip, anchors = synthetic_hip
    model = hip.model
    ctx = _contexts([37, 300, 1100], anchors, seed=17)
    full = hip.propose(ctx)
    ids = np.sort(np.random.default_rng(0).choice(SYN_VOCAB, 700, replace=False))
    try:
        for listed in (np.arange(SYN_VOCAB), ids):
            model._draft_vocab_path = _ids_file(tmp_path, listed.astype(np.int32))
            model.refresh_derived_weights()
            ranked = hip.propose(ctx, capture_logits=False)
            if len(listed) == SYN_VOCAB:
                assert torch.equal(ranked.tokens, full.tokens)
    finally:
        model._draft_vocab_path = None
        model.refresh_derived_weights()
    allowed = torch.from_numpy(ids).long()
    assert torch.isin(ranked.tokens, allowed).all()
    first = full.logits[:, 0, :][:, allowed].argmax(-1)
    assert torch.equal(ranked.tokens[:, 0], allowed[first])
    assert torch.equal(hip.propose(ctx, capture_logits=False).tokens, full.tokens)


@requires_gpu
@pytest.mark.parametrize("tp_size", [2, 4])
def test_draft_vocab_tp_argmax_matches_single_rank(synthetic_hip, tmp_path, tp_size):
    """TP=2 / TP=4 with draft_vocab (including an empty-shard case) reproduces the
    TP=1 draft_vocab argmax bit for bit, including lowest-id tie breaking."""
    import numpy as np

    from vllm.models.inkling2_loop.amd.parallel import TensorParallel, lm_head_shard

    hip, _ = synthetic_hip
    model = hip.model
    g = torch.Generator().manual_seed(29)
    # 5 rows, including an exact cross-rank tie on row 4.
    full_logits = torch.randn(5, SYN_VOCAB, generator=g, dtype=torch.float16).cuda()

    # Case 1: ids spread across all ranks; Case 2: ids only on rank 0 (empty shards).
    spread_ids = np.sort(
        np.random.default_rng(7).choice(SYN_VOCAB, 700, replace=False)
    ).astype(np.int32)
    rank0_ids = np.arange(0, 128, 3, dtype=np.int32)
    # Force an exact tie between the lowest and highest id in spread_ids on row 4.
    full_logits[4, int(spread_ids[0])] = 25.0
    full_logits[4, int(spread_ids[-1])] = 25.0

    for listed in (spread_ids, rank0_ids):
        allowed = torch.from_numpy(listed).long().cuda()
        ref = allowed[full_logits[:, allowed].argmax(dim=-1)]

        # Collect each rank's partial (value, global_id) as _argmax_across_ranks does.
        partials: list[torch.Tensor] = []
        for rank in range(tp_size):
            lo, rows, take = lm_head_shard(SYN_VOCAB, tp_size, rank)
            mask = (allowed >= lo) & (allowed < lo + take)
            rank_ids = allowed[mask]
            ranked = hip.mod._DraftVocab(
                rank_ids,
                (rank_ids - lo).to(torch.int32),
                torch.empty((rank_ids.numel(), 0), dtype=torch.float16, device="cuda"),
                torch.empty((rank_ids.numel(),), dtype=torch.float32, device="cuda"),
            )
            if rank_ids.numel() > 0:
                model._scratch.logits[:5, : rank_ids.numel()].copy_(
                    full_logits[:, rank_ids]
                )
            # Run the per-rank half of _argmax_across_ranks via a stub all_gather.
            part = torch.empty(10, dtype=torch.float32, device="cuda")
            if rank_ids.numel() == 0:
                part[:5].fill_(float("-inf"))
                part[5:10].view(torch.int32).fill_(0x7FFFFFFF)
            else:
                torch.ops._rocm_C.inkling2_dspark_argmax_partial(
                    model._scratch.logits, part, 5, rank_ids.numel(), 0, ranked.ids
                )
            partials.append(part.view(10, 1))

        gathered = torch.cat(partials, dim=-1).contiguous()
        # Verify the actual _argmax_across_ranks helper end-to-end on rank 0.
        orig_tp = model._tp
        try:

            class _FakeTP(TensorParallel):
                def __init__(self, size, rank, expect0, out):
                    super().__init__(size, rank)
                    self._expect0 = expect0
                    self._out = out

                def all_gather(self, x, dim=-1):
                    assert torch.equal(x, self._expect0)
                    return self._out

            model._tp = _FakeTP(tp_size, 0, partials[0], gathered)
            lo0, _, take0 = lm_head_shard(SYN_VOCAB, tp_size, 0)
            r0_ids = allowed[(allowed >= lo0) & (allowed < lo0 + take0)]
            if r0_ids.numel() > 0:
                model._scratch.logits[:5, : r0_ids.numel()].copy_(
                    full_logits[:, r0_ids]
                )
            ranked0 = hip.mod._DraftVocab(
                r0_ids,
                (r0_ids - lo0).to(torch.int32),
                torch.empty((r0_ids.numel(), 0), dtype=torch.float16, device="cuda"),
                torch.empty((r0_ids.numel(),), dtype=torch.float32, device="cuda"),
            )
            got = model._argmax_across_ranks(5, ranked0)
        finally:
            model._tp = orig_tp
        assert torch.equal(got, ref)


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
