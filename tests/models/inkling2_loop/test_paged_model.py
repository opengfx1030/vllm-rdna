# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate for the *paged* Inkling-2 serving model against the transformers reference.

The code under test is the serving code itself -
``Inkling2LoopTextForCausalLM`` (``load_weights`` included), ``Inkling2Model``,
the attention / conv / KV state code and
``Inkling2MetadataBuilder`` - on torch CPU tensors, behind the engine emulation of
``paged_harness.py`` (per-group block tables with their own block size, sliding-window
block freeing and null blocks, chunked prefill, decode, mixed batches, NaN-poisoned
pages).  The gfx1030 expert GEMMs are the only replaced part: the experts of the
reference (exact fp32 dequantisation of the checkpoint) stand in for the NVFP4 /
MXFP8 kernels.

The reference is ``tools/rdna/inkling2/tiny_reference.py`` (transformers
``inkling2_loop``, fp32, CPU) on the checkpoints of ``make_tiny_checkpoint.py``: the
default 6-layer model and the 8-layer variant (loop 2:6 x3, two attention-residual
blocks inside the loop, so the message slots are observable).  Logits of **every**
position are compared.

fp16 is the serving dtype.  The random tiny models are router-chaotic: a near-tie in the
top-k expert choice flips under fp16 rounding and moves the logits by O(1), and which
decisions flip depends on the chunking and the batch (the GPU engine shows the same, see
``docs/rdna2/inkling2-port-spec.md``).  The ``test_fp16_*`` tests therefore impose the
expert choice of the fp32 run by position (``PagedModel.forced_routes``), also for
chunked prefill, mixed batches and cached decode: a state or metadata bug then shows up
in the logits, a router flip cannot.

    env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES= \
        PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_paged_model.py -q
"""

from __future__ import annotations

import contextlib
import importlib.util
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)
pytest.importorskip("vllm.models.inkling2_loop.amd.model")

from paged_harness import (  # noqa: E402
    PagedWorld,
    fake_vllm_config,
    owners_of,
    patched,
    schedule,
)

from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import moe as moe_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402
from vllm.models.inkling2_loop.configs import Inkling2LoopConfig  # noqa: E402

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2"
PROMPT_LENS = (5, 40, 70)
# fp32 logits; paged and reference differ by op ordering only (see the [paged] lines)
FP32_TOL = 2e-3


def _load_tool(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


make = _load_tool("make_tiny_checkpoint")
ref = _load_tool("tiny_reference")


def _prompts(vocab: int = 500) -> list[list[int]]:
    gen = torch.Generator().manual_seed(1234)
    return [torch.randint(0, vocab, (n,), generator=gen).tolist() for n in PROMPT_LENS]


class PagedModel:
    """The serving model + the emulated engine around it.  CPU by default;
    ``device="cuda"`` with ``experts="kernel"`` runs the gfx1030 expert kernels
    (``experts="exact"``: the dense fp32 experts of the reference, on ``device``).

    ``path``: ``"reference"`` is the eager torch path (the oracle), ``"hip"`` the native
    HIP kernels of the serving path (needs a GPU, ``experts="kernel"`` and a checkpoint
    with the attention shapes of the kernels, see ``glue.check_kernel_shapes``)."""

    def __init__(
        self,
        ckpt: Path,
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device="cpu",
        experts: str = "exact",
        path: str = "reference",
        num_speculative_tokens: int = 0,
    ) -> None:
        assert experts in ("exact", "kernel") and path in ("reference", "hip")
        assert path == "reference" or experts == "kernel"
        self.device = torch.device(device)
        self.path = path
        hf_config = Inkling2LoopConfig.from_pretrained(str(ckpt))
        fake = fake_vllm_config(
            ckpt, hf_config, num_speculative_tokens=num_speculative_tokens
        )
        self.max_tokens = fake.scheduler_config.max_num_batched_tokens
        old = torch.get_default_dtype()
        torch.set_default_dtype(dtype)  # the engine builds the model under its dtype
        try:
            with (
                patched(state_mod, "get_current_vllm_config", lambda: fake),
                torch.device(self.device),
            ):
                self.lm = model_mod.Inkling2LoopTextForCausalLM(
                    vllm_config=fake, serving=path == "hip"
                )
        finally:
            torch.set_default_dtype(old)
        self.cfg = self.lm.config
        self.dtype = dtype
        with torch.no_grad():
            loaded = self.lm.load_weights(iter(tensors.items()))
        assert "lm_head.weight" in loaded and "model.message_input.query" in loaded
        if experts == "exact":
            self._install_dense_experts(tensors)
        self.world: PagedWorld | None = None
        # the step being run: (request, first row, end row, first position), and how
        # often each layer's router ran in it (= the layer visit)
        self._segments: list[tuple[int, int, int, int]] = []
        self._visits: dict[int, int] = defaultdict(int)

    def reset(
        self,
        kv_block_size: int = 64,
        conv_block_size: int | None = None,
        num_blocks: int = 128,
        seed: int = 0,
        conv_dtype=None,
    ) -> PagedModel:
        """New, NaN-filled caches (the weights are kept); ``conv_dtype``: the dtype of
        the conv rings (the engine stores them in the model dtype, whatever the dtype
        of the conv inputs)."""
        k = self.cfg.sconv_kernel_size
        self.world = PagedWorld(
            owners_of(self.lm.model.layers),
            {
                "swa": (kv_block_size, self.cfg.sliding_window_size),
                "full": (kv_block_size, None),
                "conv": (conv_block_size or k, k),
            },
            self.dtype,
            num_blocks=num_blocks,
            seed=seed,
            device=self.device,
            hip=self.path == "hip",
            max_tokens=self.max_tokens,
            conv_dtype=conv_dtype,
        )
        return self

    def _install_dense_experts(self, tensors: dict[str, torch.Tensor]) -> None:
        """Exact dequantised experts instead of the gfx1030 kernels."""
        for i, layer in enumerate(self.lm.model.layers):
            experts = moe_mod.ExpertsReference(
                self.cfg, dtype=torch.float32, device=self.device
            )
            up, down = ref.dequant_experts(tensors, f"model.llm.layers.{i}.mlp.experts")
            experts.gate_up_proj.data.copy_(up)
            experts.down_proj.data.copy_(down)
            layer.mlp.forward = self._dense_forward(layer.mlp, experts)

    @staticmethod
    def _dense_forward(moe, experts):
        def forward(hidden: torch.Tensor) -> torch.Tensor:
            _, weights, ids = moe.gate(hidden)
            return experts(hidden.float(), ids, weights).to(hidden.dtype)

        return forward

    @torch.no_grad()
    def step(
        self,
        work: list[tuple[int, list[int]]],
        metadata: bool = True,
        accepted: dict[int, int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """One engine step.  ``work = [(request, new_token_ids)]``; returns the
        logits ``[n, vocab]`` of every new token of every request.  ``accepted[r]``:
        only that many of the tokens of request ``r`` stay computed (a verify step
        whose draft tokens were rejected)."""
        spec = [(r, len(t)) for r, t in work]
        input_ids = torch.tensor(
            [x for _, t in work for x in t], dtype=torch.long, device=self.device
        )
        assert self.world is not None
        positions = self.world.positions(spec)
        self._segments, at = [], 0
        for r, n in spec:
            self._segments.append((r, at, at + n, self.world.computed.get(r, 0)))
            at += n
        self._visits = defaultdict(int)
        mds = self.world.metadata(spec) if metadata else {}
        getter = (lambda prefix: mds[prefix]) if metadata else (lambda prefix: None)
        with patched(model_mod, "get_metadata", getter):
            hidden = self.lm(input_ids, positions)
        logits_ = self.lm.compute_logits(hidden)
        assert logits_ is not None
        logits = logits_.float().cpu()
        self.world.commit(spec, accepted)
        out, at = {}, 0
        for r, n in spec:
            out[r] = logits[at : at + n]
            at += n
        return out

    def run(self, req: int, ids: list[int], chunk: int) -> torch.Tensor:
        """Feed ``ids`` of one request ``chunk`` tokens per step; returns the logits
        ``[len(ids), vocab]``."""
        return torch.cat(
            [
                self.step([(req, ids[at : at + chunk])])[req]
                for at in range(0, len(ids), chunk)
            ]
        )

    @contextlib.contextmanager
    def recording_routes(self):
        """Yields ``{(layer, n-th call): top-k expert ids [T, k]}`` filled by the
        forwards run inside the block."""
        routes: dict[tuple[int, int], torch.Tensor] = {}
        calls: dict[int, int] = defaultdict(int)

        def hook(i):
            def record(module, args, out):
                routes[(i, calls[i])] = out[2].clone()
                calls[i] += 1

            return record

        hooks = [
            layer.mlp.gate.register_forward_hook(hook(i))
            for i, layer in enumerate(self.lm.model.layers)
        ]
        try:
            yield routes
        finally:
            for h in hooks:
                h.remove()

    @contextlib.contextmanager
    def forced_routes(self, routes: dict[int, dict[tuple[int, int], torch.Tensor]]):
        """Router decisions (which experts) of the requests in ``routes`` taken from
        ``routes[request][(layer, visit)]`` (top-k ids ``[T, k]`` of the whole
        sequence, indexed by position, whatever the chunking / batching); the weights
        stay this model's own sigmoid scores of those experts.  Other requests keep
        their own choice."""

        def forced(i, gate):
            def forward(hidden):
                flat = hidden.float().reshape(-1, gate.hidden_dim)
                logits = torch.nn.functional.linear(flat, gate.weight)
                scores = torch.sigmoid(logits)
                visit = self._visits[i]
                self._visits[i] += 1
                idx = torch.topk(
                    scores + gate.e_score_correction_bias,
                    gate.top_k,
                    dim=-1,
                    sorted=False,
                ).indices
                for req, a, b, pos0 in self._segments:
                    if req in routes:
                        table = routes[req][(i, visit)]
                        idx[a:b] = table[pos0 : pos0 + (b - a)].to(idx.device)
                w = scores.gather(1, idx)
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
                return logits, w * gate.routed_scaling_factor, idx

            return forward

        saved = []
        for i, layer in enumerate(self.lm.model.layers):
            saved.append((layer.mlp.gate, layer.mlp.gate.forward))
            layer.mlp.gate.forward = forced(i, layer.mlp.gate)
        try:
            yield
        finally:
            for gate, forward in saved:
                gate.forward = forward


@pytest.fixture(scope="module", autouse=True)
def _threads():
    ref.configure_threads(8)


@pytest.fixture(scope="module", params=["tiny1", "tiny8"])
def variant(request, tmp_path_factory):
    if request.param == "tiny1":
        spec, seed = make.TinySpec(), 0
    else:
        spec = make.TinySpec(
            num_hidden_layers=8, loop_start=2, loop_end=6, loop_num_passes=3
        )
        seed = 0  # make_tiny_checkpoint.py's default: the checkpoint of the GPU runs
    path = tmp_path_factory.mktemp(request.param)
    make.write_checkpoint(path, spec, seed=seed)
    return types.SimpleNamespace(
        name=request.param,
        path=path,
        tensors=ref.read_checkpoint(path),
        ref=ref.build_reference(path),
        models={},  # dtype -> PagedModel (weights loaded once)
        logits={},  # prompt -> reference logits of every position
        routes={},  # sequence -> expert choice of the fp32 run
    )


def _model(variant, dtype=torch.float32, **world) -> PagedModel:
    if dtype not in variant.models:
        variant.models[dtype] = PagedModel(variant.path, variant.tensors, dtype)
    return variant.models[dtype].reset(**world)


def _want(variant, ids) -> torch.Tensor:
    key = tuple(ids)
    if key not in variant.logits:
        variant.logits[key] = ref.full_logits(variant.ref, ids)
    return variant.logits[key]


def _err(variant, what: str, got: torch.Tensor, want: torch.Tensor) -> float:
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.isfinite(got).all(), (
        "non-finite logits (stale / unwritten cache read?)"
    )
    err = float((got - want).abs().max())
    print(f"[paged] {variant.name} {what}: max|dlogit|={err:.3e}")
    return err


@pytest.mark.parametrize("n", PROMPT_LENS)
def test_single_step_prefill_matches_reference(variant, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    got = _model(variant).step([(0, ids)])[0]
    assert got.shape[-1] == 500  # exactly the real vocabulary
    assert _err(variant, f"prefill {n}", got, _want(variant, ids)) < FP32_TOL


@pytest.mark.parametrize("kv_bs", [4, 16, 64])
@pytest.mark.parametrize("chunk", [1, 3, 7, 16, 33])
def test_chunked_prefill_matches_reference(variant, kv_bs, chunk):
    ids = _prompts()[2]  # 70 tokens: crosses the window (16) and rel_extent (32)
    got = _model(variant, kv_block_size=kv_bs).run(0, ids, chunk)
    err = _err(variant, f"chunk {chunk} kv_bs {kv_bs}", got, _want(variant, ids))
    assert err < FP32_TOL


def test_cached_decode_matches_reference(variant):
    ids = _prompts()[2]
    pm = _model(variant)
    split = len(ids) - 8
    rows = [pm.step([(0, ids[:split])])[0][-1:]]
    for tok in ids[split : len(ids) - 1]:
        rows.append(pm.step([(0, [tok])])[0])
    got = torch.cat(rows)  # logits that follow positions split-1 .. len-2
    full = _want(variant, ids)
    assert (
        _err(variant, "decode vs uncached", got, full[split - 1 : len(ids) - 1])
        < FP32_TOL
    )
    cached = ref.teacher_forced_decode(variant.ref, ids, 8)  # HF cache path
    assert _err(variant, "decode vs HF cache", got[1:], cached[:-1]) < FP32_TOL


def test_greedy_decode_matches_reference(variant):
    ids = _prompts()[2]
    want_ids, want_logits = ref.greedy_cached(variant.ref, ids, 8)
    pm = _model(variant)
    logits = pm.step([(0, ids)])[0][-1:]
    got_ids, rows = [], []
    for step in range(8):
        rows.append(logits)
        got_ids.append(int(logits[0].argmax()))
        if step + 1 < 8:
            logits = pm.step([(0, [got_ids[-1]])])[0]
    assert got_ids == want_ids
    assert _err(variant, "greedy decode", torch.cat(rows), want_logits) < FP32_TOL


@pytest.mark.parametrize("kv_bs", [4, 64])
@pytest.mark.parametrize("chunk", [16, 33])
def test_long_sequence_past_every_window_matches_reference(variant, kv_bs, chunk):
    """200 tokens: the windowed groups (16 keys, 4 conv taps) free their blocks behind
    the window, the 64-token ones from position 79 and 143 on.  Chunked prefill and
    single-token decode steps (also exactly where a block becomes free: positions 79 and
    143) never read a freed block (the harness poisons them)."""
    gen = torch.Generator().manual_seed(99)
    ids = torch.randint(0, 500, (200,), generator=gen).tolist()
    pm = _model(variant, kv_block_size=kv_bs)
    plan = [(0, 70, chunk), (70, 150, 1), (150, 196, chunk), (196, 200, 1)]
    rows = []
    for first, end, size in plan:
        for start in range(first, end, size):
            rows.append(pm.step([(0, ids[start : min(start + size, end)])])[0])
    what = f"200 tokens chunk {chunk} kv_bs {kv_bs}"
    assert _err(variant, what, torch.cat(rows), _want(variant, ids)) < FP32_TOL


def _batch_plan(p5, p40, p70):
    """Prefill chunks, single-token steps and a whole prefill of different requests in
    one batch, in changing batch order (request 0 ends up as ``p5 + [7]``)."""
    return [
        [(1, p40[:16]), (2, p70[:50]), (0, p5)],
        [(2, p70[50:51]), (1, p40[16:33])],
        [(1, p40[33:]), (2, p70[51:52]), (0, [7])],
        [(2, p70[52:])],
    ]


def test_batch_of_requests_matches_reference(variant):
    """Prefill chunks, single-token steps and a whole prefill of different requests
    in one batch, in changing batch order; each equals its own uncached reference."""
    p5, p40, p70 = _prompts()
    pm = _model(variant)
    plan = _batch_plan(p5, p40, p70)
    fed: dict[int, list[int]] = {0: [], 1: [], 2: []}
    got: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    for step in plan:
        out = pm.step([(r, list(t)) for r, t in step])
        for r, t in step:
            fed[r] += t
            got[r].append(out[r])
    for r in (0, 1, 2):
        err = _err(
            variant, f"batch request {r}", torch.cat(got[r]), _want(variant, fed[r])
        )
        assert err < FP32_TOL


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kv_bs", [4, 64])
def test_random_schedule_matches_reference(variant, seed, kv_bs):
    prompts = _prompts()
    pm = _model(variant, kv_block_size=kv_bs, seed=seed)
    got: list[list[torch.Tensor]] = [[], [], []]
    for step in schedule(random.Random(seed), [len(p) for p in prompts]):
        out = pm.step([(r, prompts[r][a:b]) for r, a, b in step])
        for r, _a, _b in step:
            got[r].append(out[r])
    for r in range(3):
        err = _err(
            variant,
            f"random {seed} kv_bs {kv_bs} request {r}",
            torch.cat(got[r]),
            _want(variant, prompts[r]),
        )
        assert err < FP32_TOL


def test_request_reuse_after_finish_does_not_leak_state(variant):
    p5, p40, _ = _prompts()
    pm = _model(variant, seed=5)
    pm.step([(0, p40)])
    assert pm.world is not None
    pm.world.finish(0)
    got = pm.step([(1, p5)])[1]  # recycled (poisoned) blocks
    assert _err(variant, "recycled blocks", got, _want(variant, p5)) < FP32_TOL


def test_forward_without_metadata_is_finite(variant):
    """Profiling / dummy runs have no attention metadata."""
    out = _model(variant).step([(0, _prompts()[1])], metadata=False)[0]
    assert torch.isfinite(out).all()


def _dlogp(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(got.float(), -1) - torch.log_softmax(want.float(), -1)


def _expert_choice(variant, ids) -> dict[tuple[int, int], torch.Tensor]:
    """Top-k expert ids of every (layer, visit) of the fp32 run over all of ``ids``."""
    key = tuple(ids)
    if key not in variant.routes:
        pm32 = _model(variant, dtype=torch.float32)
        with pm32.recording_routes() as routes:
            pm32.step([(0, ids)])
        variant.routes[key] = routes
    return variant.routes[key]


def _fp16_check(variant, what: str, got, ids, first: int = 0) -> None:
    """``got``: fp16 logits of the positions ``first ..`` of ``ids``.  With the expert
    choice of the fp32 run they agree to ~1e-2 (rounding only)."""
    want = _want(variant, ids)[first : first + len(got)]
    assert torch.isfinite(got).all()
    d = _dlogp(got, want)
    print(
        f"[paged] {variant.name} fp16 {what}: mean|dlogp| {float(d.abs().mean()):.3e}"
        f" (last position {float(d[-1].abs().mean()):.3e})"
    )
    assert float(d.abs().mean()) < 0.05
    assert int(got[-1].argmax()) == int(want[-1].argmax())


def test_fp16_serving_dtype_matches_reference_given_the_same_expert_choice(variant):
    """Serving dtype: fp16 weights / activations (experts exact).  The only thing
    fp16 rounding can do on top of the fp32 run is flip near-tie top-k router
    decisions, which a random tiny model amplifies (see the free-routing line);
    with the fp32 run's expert choice the logits agree to ~1e-2."""
    prompts = _prompts()
    free, forced = [], []
    for ids in prompts:
        routes = _expert_choice(variant, ids)
        pm = _model(variant, dtype=torch.float16)
        free.append(pm.step([(0, ids)])[0])
        pm.reset()
        with pm.forced_routes({0: routes}):
            forced.append(pm.step([(0, ids)])[0])
    for ids, f, g in zip(prompts, free, forced):
        want = _want(variant, ids)
        mean_free = float(_dlogp(f, want)[-1].abs().mean())
        d = _dlogp(g, want)
        print(
            f"[paged] {variant.name} fp16 n={len(ids)}: last-pos mean|dlogp| "
            f"free routing {mean_free:.3e}, expert choice of the fp32 run "
            f"{float(d[-1].abs().mean()):.3e} "
            f"(all positions {float(d.abs().mean()):.3e})"
        )
        assert torch.isfinite(f).all() and torch.isfinite(g).all()
        assert float(d.abs().mean()) < 0.05
        assert int(g[-1].argmax()) == int(want[-1].argmax())


@pytest.mark.parametrize("kv_bs", [4, 64])
@pytest.mark.parametrize("chunk", [7, 16])
@pytest.mark.parametrize("n", [40, 70])
def test_fp16_chunked_prefill_given_the_same_expert_choice(variant, kv_bs, chunk, n):
    """Chunked prefill in the serving dtype: free routing flips near-ties differently
    from the unchunked run (the chunk shapes change the rounding), the expert choice
    of the fp32 run removes that and leaves rounding noise only."""
    ids = _prompts()[PROMPT_LENS.index(n)]
    pm = _model(variant, dtype=torch.float16, kv_block_size=kv_bs)
    with pm.forced_routes({0: _expert_choice(variant, ids)}):
        got = pm.run(0, ids, chunk)
    _fp16_check(variant, f"chunk {chunk} kv_bs {kv_bs} n={n}", got, ids)


def test_fp16_batch_given_the_same_expert_choice(variant):
    """Requests of different prompts in one batch share no state (the expert choice
    of every request is its own fp32 run's)."""
    p5, p40, p70 = _prompts()
    seqs = {0: p5 + [7], 1: p40, 2: p70}
    pm = _model(variant, dtype=torch.float16)
    fed: dict[int, list[int]] = {0: [], 1: [], 2: []}
    got: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    routes = {r: _expert_choice(variant, ids) for r, ids in seqs.items()}
    with pm.forced_routes(routes):
        for step in _batch_plan(p5, p40, p70):
            out = pm.step([(r, list(t)) for r, t in step])
            for r, t in step:
                fed[r] += t
                got[r].append(out[r])
    for r in (0, 1, 2):
        assert fed[r] == seqs[r]
        _fp16_check(variant, f"batch request {r}", torch.cat(got[r]), seqs[r])


def test_fp16_cached_decode_given_the_same_expert_choice(variant):
    ids = _prompts()[2]
    split = len(ids) - 8
    pm = _model(variant, dtype=torch.float16)
    with pm.forced_routes({0: _expert_choice(variant, ids)}):
        rows = [pm.step([(0, ids[:split])])[0][-1:]]
        for tok in ids[split : len(ids) - 1]:
            rows.append(pm.step([(0, [tok])])[0])
    _fp16_check(variant, "cached decode", torch.cat(rows), ids, first=split - 1)


def test_forced_routes_follow_the_positions_of_each_request(variant):
    """The forcing the fp16 tests rely on: the router rows of a request get the expert
    choice of their own request and positions, whatever the chunking or the requests
    around them."""
    p5, p40, _ = _prompts()
    seqs = {0: p5 + [7], 1: p40}
    pm = _model(variant, dtype=torch.float16)
    experts = pm.lm.model.layers[0].mlp.gate.weight.shape[0]
    # not the model's own choice, so that the forcing cannot pass by accident
    table = {
        r: {key: (ids + 1) % experts for key, ids in _expert_choice(variant, s).items()}
        for r, s in seqs.items()
    }
    plan = [[(1, p40[:16]), (0, p5)], [(1, p40[16:33])], [(0, [7]), (1, p40[33:])]]
    checked = 0
    with pm.forced_routes(table):
        for work in plan:
            with pm.recording_routes() as got:
                pm.step([(r, list(t)) for r, t in work])
            for req, a, b, pos0 in pm._segments:
                for (layer, visit), ids in got.items():
                    want = table[req][(layer, visit)][pos0 : pos0 + (b - a)]
                    assert torch.equal(ids[a:b], want), (req, layer, visit)
                    checked += 1
    assert checked > 0


HIT = 64  # a multiple of every group's block size, like the engine's hybrid alignment


@pytest.mark.parametrize("kv_bs", [4, 16, 64])
@pytest.mark.parametrize("same_tail", [True, False])
@pytest.mark.parametrize("chunk", [None, 4])
def test_prefix_cache_hit_matches_the_reference(variant, kv_bs, same_tail, chunk):
    """A request whose first 64 tokens are a prefix-cache hit (blocks computed by
    another request and shared, the null block before the window of the windowed groups)
    computes only the rest (in one step or in chunks), then decodes like a request that
    computed everything itself; the request that owns the blocks continues in the same
    batch."""
    ids = _prompts()[2]
    seqs = {
        0: list(ids),
        1: ids[:HIT] + (ids[HIT:] if same_tail else [3, 1, 4, 1, 5, 9]),
    }
    pm = _model(variant, kv_block_size=kv_bs)
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
    for tokens in ((11, 13), (12, 14)):  # decode, both requests in one batch
        out = pm.step([(r, [tokens[r]]) for r in (0, 1)])
        for r in (0, 1):
            seqs[r].append(tokens[r])
            got[r].append(out[r])
    for r in (0, 1):
        want = _want(variant, seqs[r])[HIT:]
        err = _err(
            variant, f"prefix hit kv_bs {kv_bs} request {r}", torch.cat(got[r]), want
        )
        assert err < FP32_TOL


def test_prefix_cache_hit_outlives_the_owner(variant):
    """The shared blocks stay alive when the request that computed them finishes, and
    are not handed to the next request."""
    p5, p40, p70 = _prompts()
    pm = _model(variant, kv_block_size=16, seed=3)
    pm.step([(0, p70[:HIT])])
    assert pm.world is not None
    pm.world.fork_prefix(0, 1, HIT)
    pm.world.finish(0)
    out = pm.step([(1, p70[HIT:]), (2, p40), (3, p5)])  # new requests recycle blocks
    for r, ids in ((1, p70), (2, p40), (3, p5)):
        want = _want(variant, ids)[HIT if r == 1 else 0 :]
        assert (
            _err(variant, f"hit outlives its owner, request {r}", out[r], want)
            < FP32_TOL
        )


@pytest.mark.parametrize("kv_bs", [4, 16, 64])
def test_prefix_cache_hit_on_blocks_the_owner_completed_inside_a_chunk(variant, kv_bs):
    """The owner computes all 70 tokens in one step, so the blocks below ``HIT`` end in
    the middle of its chunk: the conv state a hit resumes from (the last three
    positions of the last block it reuses) is what the owner's block-end rows wrote.
    A request that would hit blocks written in the very same step is deferred to the
    next step by ``Inkling2ConvStateManager`` (``test_conv_state_manager.py``: the
    conv history of its first tokens would be read before the owner writes it), so the
    hit is scheduled one step later; it computes the rest and decodes like a request
    that computed everything itself."""
    ids = _prompts()[2]
    pm = _model(variant, kv_block_size=kv_bs)
    seqs = {0: list(ids), 1: list(ids)}
    got = {0: [pm.step([(0, ids)])[0]], 1: []}
    assert pm.world is not None
    pm.world.fork_prefix(0, 1, HIT)
    got[1].append(pm.step([(1, ids[HIT:])])[1])
    for tokens in ((11, 13), (12, 14)):  # decode, both requests in one batch
        out = pm.step([(r, [tokens[r]]) for r in (0, 1)])
        for r in (0, 1):
            seqs[r].append(tokens[r])
            got[r].append(out[r])
    for r, first in ((0, 0), (1, HIT)):
        want = _want(variant, seqs[r])[first:]
        what = f"deferred hit kv_bs {kv_bs} request {r}"
        assert _err(variant, what, torch.cat(got[r]), want) < FP32_TOL
