# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU gate of the native HIP serving path of Inkling-2 against its torch reference.

The same serving model class runs twice on the GPU behind the engine emulation of
``paged_harness.py``: ``path="reference"`` is the eager torch path (what the CPU tests
gate against transformers), ``path="hip"`` the kernels of
``csrc/rocm/inkling2_*_rdna2.cu`` from persistent buffers
(``Inkling2LoopTextForCausalLM(serving=True)``).  Both use the gfx1030 expert GEMMs, so
everything but the expert kernels differs.

Checkpoints: ``make_tiny_checkpoint.py`` with the attention shapes of the kernels (head
dim 128, d_rel 16, six q heads over two kv heads); the 8-layer variant loops layers 2:6
three times, so message gates are exercised.  The random tiny models are
router-chaotic under fp16 (a near-tie in the top-k expert choice flips and moves the
logits by O(1), see ``docs/rdna2/inkling2-port-spec.md``), so the strict tests use
``*-stable`` checkpoints whose router bias is scaled up until the top-k choice is far
from any tie: there the two paths pick the same experts and must agree to fp16
rounding.  The free-routing checkpoints only print statistics.

* layer level, teacher forced: every layer visit of a step of the reference path is
  replayed on the HIP layer from the recorded inputs (block states, messages, partial
  block), so errors cannot compound; the router choices are compared as well;
* model level: single prefill, chunked prefill, decode, batches, random schedules,
  windows, prefix-cache hits (also on blocks the owner writes in the same step),
  DSpark taps, ``inputs_embeds``, greedy argmax, no metadata, no allocation / host sync.

    HIP_VISIBLE_DEVICES=2 PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_serving_path_gpu.py -q -s
"""

from __future__ import annotations

import contextlib
import gc
import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)
if not torch.cuda.is_available():
    pytest.skip("needs a ROCm GPU", allow_module_level=True)

from paged_harness import patched, schedule  # noqa: E402
from test_paged_model import (  # noqa: E402
    HIT,
    PROMPT_LENS,
    PagedModel,
    _batch_plan,
    _dlogp,
    _fp16_check,
    _prompts,
    _want,
    make,
    ref,
)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402

if not hasattr(torch.ops._rocm_C, "inkling2_attention"):
    pytest.skip("this build has no Inkling-2 HIP kernels", allow_module_level=True)

SHAPE = {"head_dim": 128, "d_rel": 16, "num_heads": 6, "num_kv_heads": 2}
STABLE_BIAS = 100.0  # scale of the router's selection bias of the ``*-stable`` models
KV_BS = [16, 64]  # the attention kernels take power-of-two pages of at least 16 keys
REL_TOL = 1e-2  # per-token relative error of a layer visit (non-flipped routes)


def _variant(name: str, tmp_path_factory, bias_scale: float):
    shape = {**SHAPE, "router_bias_scale": bias_scale}
    if name.startswith("tiny1"):
        spec = make.TinySpec(**shape)
    else:
        spec = make.TinySpec(
            num_hidden_layers=8, loop_start=2, loop_end=6, loop_num_passes=3, **shape
        )
    path = tmp_path_factory.mktemp(name)
    make.write_checkpoint(path, spec, seed=0)
    return SimpleNamespace(
        name=name,
        path=path,
        tensors=ref.read_checkpoint(path),
        ref=ref.build_reference(path),  # transformers, fp32, CPU
        models={},  # "reference" / "hip" -> PagedModel (weights loaded once)
        logits={},  # prompt -> reference logits of every position
    )


@pytest.fixture(scope="module", params=["tiny1", "tiny8"])
def hv(request, tmp_path_factory):
    """Router-stable HIP-shaped checkpoints (strict tests)."""
    return _variant(f"{request.param}-stable", tmp_path_factory, STABLE_BIAS)


@pytest.fixture(scope="module", params=["tiny1", "tiny8"])
def hv_free(request, tmp_path_factory):
    """Free-routing HIP-shaped checkpoints (statistics)."""
    return _variant(request.param, tmp_path_factory, 1.0)


def _pm(v, path: str, **world) -> PagedModel:
    if path not in v.models:
        v.models[path] = PagedModel(
            v.path,
            v.tensors,
            torch.float16,
            device="cuda",
            experts="kernel",
            path=path,
        )
    return v.models[path].reset(**world)


# -- teacher-forced layer parity -------------------------------------------------------
@contextlib.contextmanager
def _recording(pm: PagedModel):
    """Inputs, outputs and router choice of every layer visit run inside."""
    records: list[SimpleNamespace] = []
    hooks = []
    for i, layer in enumerate(pm.lm.model.layers):

        def pre(_m, args, i=i):
            hidden, blocks, messages, _positions, visit = args
            records.append(
                SimpleNamespace(
                    layer=i,
                    visit=visit,
                    hidden=hidden.clone(),
                    blocks=blocks.clone(),
                    messages=None if messages is None else messages.clone(),
                    route=None,
                    out=None,
                )
            )

        def post(_m, _args, out):
            records[-1].out = out.clone()

        def route(_m, _args, out):
            records[-1].route = out[2].clone()

        hooks += [
            layer.register_forward_pre_hook(pre),
            layer.register_forward_hook(post),
            layer.mlp.gate.register_forward_hook(route),
        ]
    try:
        yield records
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def _replay(pm: PagedModel, work, records):
    """Run the recorded visits on the HIP layers of ``pm`` (own caches): per visit the
    fp32 output ``[T, H]`` and the router's top-k ids."""
    spec = [(r, len(t)) for r, t in work]
    assert pm.world is not None
    positions = pm.world.positions(spec)
    mds = pm.world.metadata(spec)
    ws = pm.lm.model.glue_ws
    T = positions.shape[0]
    v = ws.bind(T)
    outs, routes = [], []
    with patched(model_mod, "get_metadata", lambda prefix: mds[prefix]):
        for rec in records:
            layer = pm.lm.model.layers[rec.layer]
            n = rec.blocks.shape[1]
            if n:
                v.blk[:, :n].copy_(rec.blocks)
            has_msgs = rec.messages is not None
            if has_msgs:
                v.msgs.copy_(rec.messages)
            x0, x1, _ = layer.forward_hip(
                ws, v, rec.hidden, None, n, has_msgs, positions, rec.visit, None
            )
            outs.append(x0 + x1)
            routes.append(ws.topk_ids[:T].clone())
    pm.world.commit(spec)
    return outs, routes


def _teacher_forced(v, steps, what: str, kv_bs: int = 16):
    """Reference path forward over ``steps`` (each ``[(request, tokens)]``) with the HIP
    layers replaying every visit; returns (worst relative error, flipped router rows,
    router rows, visits)."""
    ref_pm = _pm(v, "reference", kv_block_size=kv_bs)
    hip = _pm(v, "hip", kv_block_size=kv_bs)
    worst, flips, rows, visits = 0.0, 0, 0, 0
    for work in steps:
        with _recording(ref_pm) as records:
            ref_pm.step(work)
        outs, routes = _replay(hip, work, records)
        for rec, out, route in zip(records, outs, routes):
            assert torch.isfinite(out).all(), f"{what}: layer {rec.layer} not finite"
            want = torch.sort(rec.route, -1).values
            flipped = (want != torch.sort(route.long(), -1).values).any(-1)
            flips += int(flipped.sum())
            rows += flipped.numel()
            keep = ~flipped
            if keep.any():
                a, b = out[keep], rec.out[keep]
                worst = max(worst, float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max()))
            visits += 1
    print(
        f"[hip-layer] {v.name} {what}: {visits} visits, worst per-token relative error "
        f"{worst:.3e}, router rows differing {flips}/{rows}"
    )
    return worst, flips, rows, visits


def _scenarios():
    p5, p40, p70 = _prompts()
    return {
        "prefill 70": [[(0, p70)]],
        "chunked 70 / 16": [[(0, p70[a : a + 16])] for a in range(0, 70, 16)],
        "prefill then decode": [[(0, p70[:62])], *[[(0, [t])] for t in p70[62:]]],
        "batch": [
            [(r, list(t)) for r, t in step] for step in _batch_plan(p5, p40, p70)
        ],
    }


@pytest.mark.parametrize("scenario", list(_scenarios()))
def test_every_layer_visit_matches_the_reference_path(hv, scenario):
    worst, flips, _rows, visits = _teacher_forced(hv, _scenarios()[scenario], scenario)
    assert visits > 0
    assert flips == 0, "router-stable checkpoints must route identically"
    assert worst < REL_TOL


# -- model level (stable routing): HIP vs transformers and vs the reference path ------
@pytest.mark.parametrize("n", PROMPT_LENS)
def test_prefill_matches_the_references(hv, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    got = _pm(hv, "hip").step([(0, ids)])[0]
    assert got.shape[-1] == 500
    _fp16_check(hv, f"hip prefill {n}", got, ids)
    other = _pm(hv, "reference").step([(0, ids)])[0]
    d = _dlogp(got, other)
    print(
        f"[hip] {hv.name} prefill {n}: mean|dlogp| vs reference path "
        f"{d.abs().mean():.3e}"
    )
    assert float(d.abs().mean()) < 0.05


@pytest.mark.parametrize("kv_bs", KV_BS)
@pytest.mark.parametrize("chunk", [1, 7, 16, 33])
def test_chunked_prefill(hv, kv_bs, chunk):
    ids = _prompts()[2]
    got = _pm(hv, "hip", kv_block_size=kv_bs).run(0, ids, chunk)
    _fp16_check(hv, f"hip chunk {chunk} kv_bs {kv_bs}", got, ids)


def test_cached_decode(hv):
    ids = _prompts()[2]
    split = len(ids) - 8
    pm = _pm(hv, "hip")
    rows = [pm.step([(0, ids[:split])])[0][-1:]]
    for tok in ids[split : len(ids) - 1]:
        rows.append(pm.step([(0, [tok])])[0])
    _fp16_check(hv, "hip cached decode", torch.cat(rows), ids, first=split - 1)


def test_batch_of_requests(hv):
    p5, p40, p70 = _prompts()
    seqs = {0: p5 + [7], 1: p40, 2: p70}
    pm = _pm(hv, "hip")
    got: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    fed: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for step in _batch_plan(p5, p40, p70):
        out = pm.step([(r, list(t)) for r, t in step])
        for r, t in step:
            fed[r] += t
            got[r].append(out[r])
    for r in (0, 1, 2):
        assert fed[r] == seqs[r]
        _fp16_check(hv, f"hip batch request {r}", torch.cat(got[r]), seqs[r])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kv_bs", KV_BS)
def test_random_schedule(hv, seed, kv_bs):
    prompts = _prompts()
    pm = _pm(hv, "hip", kv_block_size=kv_bs, seed=seed)
    got: list[list[torch.Tensor]] = [[], [], []]
    for step in schedule(random.Random(seed), [len(p) for p in prompts]):
        out = pm.step([(r, prompts[r][a:b]) for r, a, b in step])
        for r, _a, _b in step:
            got[r].append(out[r])
    for r in range(3):
        _fp16_check(
            hv,
            f"hip random {seed} kv_bs {kv_bs} request {r}",
            torch.cat(got[r]),
            prompts[r],
        )


@pytest.mark.parametrize("kv_bs", KV_BS)
@pytest.mark.parametrize("chunk", [16, 33])
def test_long_sequence_past_every_window(hv, kv_bs, chunk):
    """200 tokens: the windowed groups free their blocks behind the window; chunked
    prefill and single-token steps never read a freed block (the harness poisons them,
    a read makes the output non-finite).  Teacher forced per layer visit: a router
    near-tie that one fp16 pipeline resolves differently from the other would change
    every later token of a free-running comparison (that chaos is what the free-routing
    statistics and the end-to-end checks report)."""
    gen = torch.Generator().manual_seed(99)
    ids = torch.randint(0, 500, (200,), generator=gen).tolist()
    plan = [(0, 70, chunk), (70, 150, 1), (150, 196, chunk), (196, 200, 1)]
    steps = [
        [(0, ids[start : min(start + size, end)])]
        for first, end, size in plan
        for start in range(first, end, size)
    ]
    what = f"200 tokens chunk {chunk} kv_bs {kv_bs}"
    worst, flips, rows, visits = _teacher_forced(hv, steps, what, kv_bs)
    assert visits > 0
    assert flips * 200 <= rows, "more than 0.5% of the router rows differ"
    assert worst < REL_TOL


def test_request_reuse_after_finish_does_not_leak_state(hv):
    p5, p40, _ = _prompts()
    pm = _pm(hv, "hip", seed=5)
    pm.step([(0, p40)])
    assert pm.world is not None
    pm.world.finish(0)
    _fp16_check(hv, "hip recycled blocks", pm.step([(1, p5)])[1], p5)


# -- prefix caching -----------------------------------------------------------------
@pytest.mark.parametrize("kv_bs", KV_BS)
@pytest.mark.parametrize("same_tail", [True, False])
@pytest.mark.parametrize("chunk", [None, 4])
def test_prefix_cache_hit(hv, kv_bs, same_tail, chunk):
    ids = _prompts()[2]
    seqs = {
        0: list(ids),
        1: ids[:HIT] + (ids[HIT:] if same_tail else [3, 1, 4, 1, 5, 9]),
    }
    pm = _pm(hv, "hip", kv_block_size=kv_bs)
    pm.step([(0, ids[:HIT])])
    assert pm.world is not None
    pm.world.fork_prefix(0, 1, HIT)
    tail = len(ids) - HIT
    got: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for a in range(0, tail, chunk or tail):
        end = a + (chunk or tail)
        out = pm.step([(r, seqs[r][HIT + a : HIT + end]) for r in (0, 1)])
        for r in (0, 1):
            got[r].append(out[r])
    for tokens in ((11, 13), (12, 14)):
        out = pm.step([(r, [tokens[r]]) for r in (0, 1)])
        for r in (0, 1):
            seqs[r].append(tokens[r])
            got[r].append(out[r])
    for r in (0, 1):
        _fp16_check(
            hv,
            f"hip prefix hit kv_bs {kv_bs} request {r}",
            torch.cat(got[r]),
            seqs[r],
            first=HIT,
        )


def test_prefix_cache_hit_outlives_the_owner(hv):
    p5, p40, p70 = _prompts()
    pm = _pm(hv, "hip", kv_block_size=16, seed=3)
    pm.step([(0, p70[:HIT])])
    assert pm.world is not None
    pm.world.fork_prefix(0, 1, HIT)
    pm.world.finish(0)
    out = pm.step([(1, p70[HIT:]), (2, p40), (3, p5)])
    for r, ids in ((1, p70), (2, p40), (3, p5)):
        _fp16_check(
            hv,
            f"hip hit outlives owner, request {r}",
            out[r],
            ids,
            first=HIT if r == 1 else 0,
        )


@pytest.mark.parametrize("kv_bs", KV_BS)
@pytest.mark.parametrize("hit_first", [False, True])
def test_prefix_cache_hit_on_blocks_written_in_the_same_step(hv, kv_bs, hit_first):
    """The scheduler registers the blocks of the tokens it schedules before they are
    computed, so a request of the same batch can hit blocks the owner only writes in
    this very step.  The conv / K / V history reads of the hit's first tokens race with
    those writes inside one kernel launch; the serving path launches those ops twice
    (``conv_passes``) and the second launch must see all of them, in either batch
    order.  Run repeatedly: the CTA order of a launch is not deterministic."""
    ids = _prompts()[2]
    pm = _pm(hv, "hip", kv_block_size=kv_bs)
    assert pm.world is not None
    work = [(0, ids), (1, ids[HIT:])]
    for rep in range(8):
        pm.reset(kv_block_size=kv_bs, seed=rep)
        assert pm.world is not None
        pm.world.fork_prefix(0, 1, HIT, in_flight=True)
        out = pm.step(work[::-1] if hit_first else work)
        _fp16_check(hv, f"hip same-step hit kv_bs {kv_bs} request 0", out[0], ids)
        _fp16_check(
            hv, f"hip same-step hit kv_bs {kv_bs} request 1", out[1], ids, first=HIT
        )
        # the hit and its owner compute the same tokens from the same state
        assert torch.allclose(out[1], out[0][HIT:], atol=5e-3, rtol=0), rep


# -- everything else around the layers -------------------------------------------
def test_without_metadata_the_forward_is_finite(hv):
    """Profiling / dummy runs have no attention metadata."""
    out = _pm(hv, "hip").step([(0, _prompts()[1])], metadata=False)[0]
    assert torch.isfinite(out).all()


def _step_embeds(pm: PagedModel, work):
    """``PagedModel.step`` feeding ``embed_input_ids`` rows as ``inputs_embeds`` (the
    path of the vision mixin)."""
    spec = [(r, len(t)) for r, t in work]
    ids = torch.tensor([x for _, t in work for x in t], dtype=torch.long, device="cuda")
    assert pm.world is not None
    positions = pm.world.positions(spec)
    mds = pm.world.metadata(spec)
    with (
        patched(model_mod, "get_metadata", lambda prefix: mds[prefix]),
        torch.no_grad(),
    ):
        hidden = pm.lm(None, positions, inputs_embeds=pm.lm.embed_input_ids(ids))
        logits = pm.lm.compute_logits(hidden).float().cpu()
    pm.world.commit(spec)
    return logits


def test_inputs_embeds_path_equals_the_input_ids_path(hv):
    ids = _prompts()[1]
    pm = _pm(hv, "hip")
    plain = pm.step([(0, ids)])[0]
    pm.reset()
    assert torch.equal(_step_embeds(pm, [(0, ids)]), plain)


def test_embedding_matches_the_reference_path(hv):
    ids = torch.randint(0, 500, (37,), device="cuda")
    got = _pm(hv, "hip").lm.embed_input_ids(ids)
    want = _pm(hv, "reference").lm.embed_input_ids(ids)
    assert got.dtype == want.dtype == torch.float16
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-3, atol=1e-3)


def test_rms_norm_sum_out_widens_without_a_weight():
    """``Inkling2Model.forward_hip`` widens ``inputs_embeds`` into the fp32 stream with
    ``rms_norm(x, sum_out=...)``; that must be an exact fp16 -> fp32 copy."""
    x = torch.randn(33, 256, device="cuda", dtype=torch.float16)
    junk = torch.empty_like(x)
    wide = torch.empty(33, 256, device="cuda", dtype=torch.float32)
    ops.inkling2_glue_rms_norm(x, junk, sum_out=wide, eps=1e-6)
    assert torch.equal(wide, x.float())


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 16, 17, 40])
def test_lm_head_logits_and_greedy_ids(hv, rows):
    lm = _pm(hv, "hip").lm
    gen = torch.Generator(device="cuda").manual_seed(rows)
    h = (torch.randn(rows, 256, device="cuda", generator=gen) * 2).half()
    logits = lm.compute_logits(h)
    want = torch.tanh((h.float() @ lm.lm_head.weight.float().T) / 30.0) * 30.0
    assert logits.shape == (rows, 500) and logits.dtype == torch.float32
    torch.testing.assert_close(logits, want, rtol=1e-4, atol=1e-3)
    ids = lm.greedy_token_ids(h)
    assert ids.dtype == torch.int64 and ids.shape == (rows,)
    assert torch.equal(ids, logits.argmax(-1))


def test_seq_idx_matches_the_torch_construction():
    scratch = state_mod.SeqIdxScratch(512, "cuda")
    rng = random.Random(0)
    for _ in range(25):
        reqs = rng.randint(1, 9)
        lens = [rng.choice([0, 1, 1, 2, 7, 33]) for _ in range(reqs)]
        lens[rng.randrange(reqs)] = rng.randint(1, 40)  # at least one token
        tokens = sum(lens)
        qsl = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32
        )
        common = SimpleNamespace(
            num_reqs=reqs,
            num_actual_tokens=tokens,
            query_start_loc=qsl.cuda(),
            query_start_loc_cpu=qsl,
            seq_lens=torch.tensor(lens, dtype=torch.int32, device="cuda") + 5,
            block_table_tensor=torch.zeros(reqs, 4, dtype=torch.int32, device="cuda"),
            slot_mapping=torch.zeros(tokens, dtype=torch.int64, device="cuda"),
            max_query_len=max(lens),
            max_seq_len=max(lens) + 5,
        )
        got = state_mod.make_metadata(common, scratch).seq_idx
        want = state_mod.make_metadata(common, None).seq_idx
        assert torch.equal(got.cpu(), want.cpu()), lens


def test_dspark_taps_match_the_reference_path(hv):
    pm_hip, pm_ref = _pm(hv, "hip"), _pm(hv, "reference")
    cfg = pm_hip.cfg
    visits = cfg.num_hidden_layers + (cfg.loop_num_passes - 1) * (
        cfg.loop_layer_index_end - cfg.loop_layer_index_start
    )
    ids = (0, 3, visits - 1)
    taps = {}
    for name, pm in (("hip", pm_hip), ("reference", pm_ref)):
        pm.lm.set_aux_hidden_state_layers(ids)
        plain = pm.lm.forward

        def tapped(*args, _plain=plain, _name=name, **kwargs):
            hidden, aux = _plain(*args, **kwargs)
            taps[_name] = [t.clone() for t in aux]
            return hidden

        pm.lm.forward = tapped
        try:
            pm.step([(0, _prompts()[1])])
        finally:
            del pm.lm.forward
            pm.lm.set_aux_hidden_state_layers(())
        pm.reset()
    assert len(taps["hip"]) == len(taps["reference"]) == len(ids) + 1
    for k, (a, b) in enumerate(zip(taps["hip"], taps["reference"])):
        assert a.dtype == torch.float32 and a.shape == b.shape
        rel = float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())
        print(f"[hip-taps] {hv.name} tap {k}: worst per-token relative error {rel:.3e}")
        # the layer-level test bounds one visit at 1.3e-3; the fp32 stream accumulates
        visit = ids[k] if k < len(ids) else visits  # the last tap follows every visit
        assert rel < 2e-3 * (visit + 1), k
    # plain forward after the taps were switched off: a bare tensor again
    out = _pm(hv, "hip").step([(0, _prompts()[0])])[0]
    assert torch.isfinite(out).all()


def test_forward_allocates_nothing_and_never_syncs(hv):
    """The serving forward is HIP-graph capturable: no allocation, no host
    synchronisation (a sync debug mode that raises turns any ``.item()`` /
    ``.tolist()`` / ``nonzero`` into an error)."""
    pm = _pm(hv, "hip")
    ids = _prompts()[1]
    spec = [(0, len(ids))]
    input_ids = torch.tensor(ids, dtype=torch.long, device="cuda")
    assert pm.world is not None
    positions = pm.world.positions(spec)
    mds = pm.world.metadata(spec)
    with (
        patched(model_mod, "get_metadata", lambda prefix: mds[prefix]),
        torch.no_grad(),
    ):
        pm.lm(input_ids, positions)  # warm up: memoised views, lazy module state
        torch.accelerator.synchronize()
        before = torch.accelerator.memory_stats()["allocation.all.allocated"]
        torch.cuda.set_sync_debug_mode("error")
        try:
            pm.lm(input_ids, positions)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        after = torch.accelerator.memory_stats()["allocation.all.allocated"]
    assert after == before, f"{after - before} device allocations in one forward"


def test_the_serving_path_has_no_torch_fallback(hv):
    pm = _pm(hv, "hip")
    assert pm.lm.serving and pm.lm.model.serving
    # the checkpoint-layout weights were repacked and freed: the reference forward
    # cannot run, so a stray call fails instead of silently computing in torch
    layer = pm.lm.model.layers[0]
    assert not hasattr(layer.self_attn.q_proj.linear, "weight")
    ids = torch.tensor(_prompts()[0], dtype=torch.long, device="cuda")
    with pytest.raises(AttributeError):
        pm.lm.model.forward_reference(ids, torch.arange(len(ids), device="cuda"))


def _owned_storages(model: torch.nn.Module, exclude: set[int]) -> dict[int, tuple]:
    """Device storages the model owns, ``pointer -> (bytes, shape, dtype)``: every
    tensor reachable through the attributes of its modules (parameters, buffers, the
    stacked / packed weights and the pack objects holding them), except the KV / conv
    cache owners (the harness's world fills them) and the storages in ``exclude`` (the
    shared workspace)."""
    skip = (state_mod.Inkling2KVOwner, state_mod.Inkling2ConvOwner, type)
    found: dict[int, tuple] = {}
    seen: set[int] = set()
    todo: list[object] = [model]
    while todo:
        obj = todo.pop()
        if id(obj) in seen or isinstance(obj, skip):
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            if obj.is_cuda and storage.data_ptr() not in exclude:
                found[storage.data_ptr()] = (
                    storage.nbytes(),
                    tuple(obj.shape),
                    obj.dtype,
                )
        elif isinstance(obj, dict):
            todo.extend(obj.values())
        elif isinstance(obj, list | tuple):
            todo.extend(obj)
        elif hasattr(obj, "__dict__"):
            todo.extend(vars(obj).values())
    return found


def test_serving_weights_are_held_once(hv):
    """Repacking replaces the weights, it does not copy them next to the originals:
    the serving model owns no more device memory than the reference model (the
    persistent workspace aside)."""

    def build(path: str) -> PagedModel:
        gc.collect()
        return PagedModel(
            hv.path,
            hv.tensors,
            torch.float16,
            device="cuda",
            experts="kernel",
            path=path,
        )

    ref_pm, hip_pm = build("reference"), build("hip")
    ws = hip_pm.lm.model.glue_ws
    ws_ptrs = {t.untyped_storage().data_ptr() for t in ws.tensors()}
    ref_own = _owned_storages(ref_pm.lm, set())
    hip_own = _owned_storages(hip_pm.lm, ws_ptrs)
    ref_bytes = sum(b for b, _, _ in ref_own.values())
    hip_bytes = sum(b for b, _, _ in hip_own.values())
    net: dict[tuple, int] = {}
    for sign, own in ((1, hip_own), (-1, ref_own)):
        for key in own.values():
            net[key] = net.get(key, 0) + sign
    listing = "; ".join(
        f"{n:+d}x {k[1]} {k[2]} ({k[0]} B)"
        for k, n in sorted(net.items(), key=lambda kn: -abs(kn[0][0] * kn[1]))
        if n
    )[:1600]
    print(
        f"[hip-memory] {hv.name}: reference model {ref_bytes / 2**20:.2f} MiB, HIP "
        f"model {hip_bytes / 2**20:.2f} MiB (+ workspace {ws.nbytes() / 2**20:.2f} "
        f"MiB); tensors the HIP model owns more (+) / fewer (-) of than the reference "
        f"model: {listing}"
    )
    # the packs replace the dense weights one for one; what is left are the small
    # stacked fp16 / fp32 copies (layer-norm weights, gate stacks, the router) which
    # weigh relatively more in a tiny model than in the real one, where
    # ``real_check`` logs "Model loading took" within 0.01 GiB of the reference
    assert hip_bytes <= 1.10 * ref_bytes, listing
    del ref_pm, hip_pm


# -- free routing: statistics only ------------------------------------------------
def test_free_routing_statistics(hv_free):
    """Router-chaotic checkpoints: print the numbers the end-to-end checks report.  A
    near-tie of the top-k choice flips differently in the two paths, so the strict
    comparisons are the stable-routing tests above; this one needs finite logits."""
    for ids in _prompts():
        want = _want(hv_free, ids)
        stats = {}
        for path in ("reference", "hip"):
            got = _pm(hv_free, path).step([(0, ids)])[0]
            assert torch.isfinite(got).all()
            d = _dlogp(got, want)
            stats[path] = (
                float(d[-1].abs().mean()),
                float(d.abs().mean()),
                int(got[-1].argmax()) == int(want[-1].argmax()),
            )
        print(
            f"[hip-free] {hv_free.name} n={len(ids)}: last-position mean|dlogp| "
            f"reference path {stats['reference'][0]:.3e} hip "
            f"{stats['hip'][0]:.3e}; all positions {stats['reference'][1]:.3e} / "
            f"{stats['hip'][1]:.3e}; argmax equal "
            f"{stats['reference'][2]} / {stats['hip'][2]}"
        )


@pytest.mark.parametrize("top_k", [2, 8])
def test_run_to_run_difference_by_experts_per_token(top_k, tmp_path_factory):
    """The expert GEMMs (``moe_lowp_dot2_rdna2.cu``, shared by both paths) add each
    token's expert outputs into the pre-zeroed output with a packed fp16 CAS atomic
    add: the sum is order dependent at fp16 precision beyond two experts per token
    (``a + b`` commutes) and the real checkpoint takes eight.  Five runs of the same
    prompt on the same weights: bitwise equal for top-2; for top-8, with the expert
    choice fixed by the bias, the runs differ by rounding noise only (measured 0.20 ..
    0.24 on a logit scale of 19.6, the sum order amplified through 16 layer visits),
    bounded at 5% (a race would show O(1) differences); the figure is printed."""
    spec = make.TinySpec(
        num_hidden_layers=8,
        loop_start=2,
        loop_end=6,
        loop_num_passes=3,
        n_routed_experts=16,
        num_experts_per_tok=top_k,
        router_bias_scale=1e4,  # the expert choice is the bias alone: nothing flips it
        **SHAPE,
    )
    path = tmp_path_factory.mktemp(f"topk{top_k}")
    make.write_checkpoint(path, spec, seed=0)
    pm = PagedModel(
        path,
        ref.read_checkpoint(path),
        torch.float16,
        device="cuda",
        experts="kernel",
        path="hip",
    )
    ids = torch.randint(0, 500, (70,), generator=torch.Generator().manual_seed(5))
    runs = [pm.reset(kv_block_size=16).run(0, ids.tolist(), 70) for _ in range(5)]
    worst = max((r - runs[0]).abs().max().item() for r in runs[1:])
    scale = runs[0].abs().max().item()
    print(
        f"[determinism] top-{top_k}: max |logit difference| between five identical "
        f"runs {worst:.3e} (logit scale {scale:.3f})"
    )
    assert torch.isfinite(runs[0]).all()
    if top_k == 2:
        assert worst == 0.0
    else:
        assert worst < 5 * REL_TOL * scale
