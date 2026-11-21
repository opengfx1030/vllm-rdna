# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate for the *paged* Inkling-2 serving model against the transformers reference.

The code under test is the serving code itself -
``Inkling2LoopForConditionalGeneration`` (``load_weights`` included), ``Inkling2Model``,
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
    (``experts="exact"``: the dense fp32 experts of the reference, on ``device``)."""

    def __init__(
        self,
        ckpt: Path,
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device="cpu",
        experts: str = "exact",
    ) -> None:
        assert experts in ("exact", "kernel")
        self.device = torch.device(device)
        hf_config = Inkling2LoopConfig.from_pretrained(str(ckpt))
        fake = fake_vllm_config(ckpt, hf_config)
        old = torch.get_default_dtype()
        torch.set_default_dtype(dtype)  # the engine builds the model under its dtype
        try:
            with (
                patched(state_mod, "get_current_vllm_config", lambda: fake),
                torch.device(self.device),
            ):
                self.lm = model_mod.Inkling2LoopForConditionalGeneration(
                    vllm_config=fake
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

    def reset(
        self,
        kv_block_size: int = 64,
        conv_block_size: int | None = None,
        num_blocks: int = 128,
        seed: int = 0,
    ) -> PagedModel:
        """New, NaN-filled caches (the weights are kept)."""
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
        self, work: list[tuple[int, list[int]]], metadata: bool = True
    ) -> dict[int, torch.Tensor]:
        """One engine step.  ``work = [(request, new_token_ids)]``; returns the
        logits ``[n, vocab]`` of every new token of every request."""
        spec = [(r, len(t)) for r, t in work]
        input_ids = torch.tensor(
            [x for _, t in work for x in t], dtype=torch.long, device=self.device
        )
        assert self.world is not None
        positions = self.world.positions(spec)
        mds = self.world.metadata(spec) if metadata else {}
        getter = (lambda prefix: mds[prefix]) if metadata else (lambda prefix: None)
        with patched(model_mod, "get_metadata", getter):
            hidden = self.lm(input_ids, positions)
        logits_ = self.lm.compute_logits(hidden)
        assert logits_ is not None
        logits = logits_.float().cpu()
        self.world.commit(spec)
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
    def forced_routes(self, routes):
        """Router decisions (which experts) taken from ``routes``; the weights stay
        this model's own sigmoid scores of those experts."""
        calls: dict[int, int] = defaultdict(int)
        saved = []

        def forced(i, gate):
            def forward(hidden):
                flat = hidden.float().reshape(-1, gate.hidden_dim)
                logits = torch.nn.functional.linear(flat, gate.weight)
                scores = torch.sigmoid(logits)
                idx = routes[(i, calls[i])].to(hidden.device)
                calls[i] += 1
                w = scores.gather(1, idx)
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
                return logits, w * gate.routed_scaling_factor, idx

            return forward

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
        seed = 3
    path = tmp_path_factory.mktemp(request.param)
    make.write_checkpoint(path, spec, seed=seed)
    return types.SimpleNamespace(
        name=request.param,
        path=path,
        tensors=ref.read_checkpoint(path),
        ref=ref.build_reference(path),
        models={},  # dtype -> PagedModel (weights loaded once)
        logits={},  # prompt -> reference logits of every position
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


def test_batch_of_requests_matches_reference(variant):
    """Prefill chunks, single-token steps and a whole prefill of different requests
    in one batch, in changing batch order; each equals its own uncached reference."""
    p5, p40, p70 = _prompts()
    pm = _model(variant)
    plan = [
        [(1, p40[:16]), (2, p70[:50]), (0, p5)],
        [(2, p70[50:51]), (1, p40[16:33])],
        [(1, p40[33:]), (2, p70[51:52]), (0, [7])],
        [(2, p70[52:])],
    ]
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


def test_fp16_serving_dtype_matches_reference_given_the_same_expert_choice(variant):
    """Serving dtype: fp16 weights / activations (experts exact).  The only thing
    fp16 rounding can do on top of the fp32 run is flip near-tie top-k router
    decisions, which a random tiny model amplifies (see the free-routing line);
    with the fp32 run's expert choice the logits agree to ~1e-2."""
    prompts = _prompts()
    free, forced = [], []
    for ids in prompts:
        pm32 = _model(variant, dtype=torch.float32)
        with pm32.recording_routes() as routes:
            pm32.step([(0, ids)])
        pm = _model(variant, dtype=torch.float16)
        free.append(pm.step([(0, ids)])[0])
        pm.reset()
        with pm.forced_routes(routes):
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
