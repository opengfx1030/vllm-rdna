# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit, kernel, prefix-cache & HIP-graph tests for Inkling-2 LoopCD (2610.02185).

Covers:
1. CPU config parsing & reference math oracles (``LoopCDParams``,
   ``read_loopcd_env_defaults``, ``parse_loopcd_params``,
   ``loopcd_blend_reference``, ``loopcd_logits_reference``) — runs everywhere,
   with no GPU and no ``transformers.models.inkling2_loop`` dependency.
2. GPU native RDNA2 HIP kernel parity & 0-ULP fast-exit checks:
   - ``inkling2_glue_loopcd_blend`` vs ``loopcd_blend_reference``
   - ``inkling2_loopcd_logits`` vs ``loopcd_logits_reference`` (fp64, incl. V=200058
     and sliced non-contiguous row stride 201024)
   - ``inkling2_loopcd_prepare_hidden`` across prefill, chunked prefill, decode,
     DSpark verify, mode gates, ``blend_blocks=False``, and graph padding tails
   - ``inkling2_glue_loopcd_readout`` for ``gate_flag=0`` (0-ULP untouched) and
     ``gate_flag=1`` (bit-identical to ``inkling2_glue_attn_res``).
3. Model-level CPU/GPU parity, 0-ULP ``alpha=0.0`` baseline gate, prefix-cache
   invariance on shared blocks, and single-capture ``FULL`` HIP-graph replay
   across baseline, mixed-batch ``LoopCD-Hidden``, and adaptive ``LoopCD-Logits``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parents[2] / "tools" / "rdna" / "inkling2"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from paged_harness import (  # noqa: E402
    PagedWorld,
    cdiv,
    fake_vllm_config,
    owners_of,
    patched,
)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.inkling2_loop.amd import loopcd as loopcd_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import moe as moe_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402
from vllm.models.inkling2_loop.amd.loopcd import (  # noqa: E402
    LoopCDParams,
    loopcd_blend_reference,
    loopcd_logits_reference,
    parse_loopcd_params,
    read_loopcd_env_defaults,
)
from vllm.models.inkling2_loop.configs import Inkling2LoopConfig  # noqa: E402

_NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm GPU"
)

HIP_SHAPE = {
    "head_dim": 128,
    "d_rel": 16,
    "num_heads": 6,
    "num_kv_heads": 2,
}
STABLE_SPEC_KWARGS = {
    "num_hidden_layers": 8,
    "loop_start": 2,
    "loop_end": 6,
    "loop_num_passes": 3,
    # Scale router biases so every layer's k vs k+1 bias gap exceeds 1.0
    # (layer 3 is 0.316 at 100.0 -> 1.58 at 500.0 > max sigmoid swing 1.0),
    # guaranteeing zero fp16 router flips between reference and HIP paths.
    "router_bias_scale": 500.0,
    "num_experts_per_tok": 2,
    **HIP_SHAPE,
}


def _load_tool(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_safetensors() -> tuple[Any, Any]:
    pytest.importorskip("safetensors")
    return _load_tool("make_tiny_checkpoint"), _load_tool("tiny_reference")


def _require_loopcd_hip_ops() -> None:
    for attr in (
        "inkling2_glue_loopcd_blend",
        "inkling2_glue_loopcd_readout",
        "inkling2_loopcd_prepare_hidden",
        "inkling2_loopcd_logits",
    ):
        if not hasattr(ops, attr):
            pytest.skip(f"vllm._custom_ops.{attr} is not available in this build")
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "inkling2_attention"
    ):
        pytest.skip("this build has no Inkling-2 HIP kernels")


def _prompts(vocab: int = 500) -> tuple[list[int], list[int], list[int]]:
    gen = torch.Generator().manual_seed(1234)
    return tuple(  # type: ignore[return-value]
        torch.randint(0, vocab, (n,), generator=gen).tolist() for n in (8, 40, 64)
    )


def _dlogp(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(got.float(), -1) - torch.log_softmax(want.float(), -1)


# ---------------------------------------------------------------------------
# Self-contained PagedModel & FULL HIP-Graph Rig (no transformers.inkling2_loop)
# ---------------------------------------------------------------------------


class TinyPagedModel:
    """Self-contained paged harness around ``Inkling2LoopTextForCausalLM`` that
    does not require ``transformers.models.inkling2_loop``."""

    def __init__(
        self,
        ckpt: Path,
        tensors: dict[str, torch.Tensor],
        dtype: torch.dtype = torch.float16,
        *,
        device: str = "cpu",
        experts: str = "exact",
        path: str = "reference",
        num_speculative_tokens: int = 0,
        max_num_batched_tokens: int = 256,
    ) -> None:
        assert experts in ("exact", "kernel") and path in ("reference", "hip")
        assert path == "reference" or experts == "kernel"
        self.device = torch.device(device)
        self.path = path
        self.dtype = dtype
        hf_config = Inkling2LoopConfig.from_pretrained(str(ckpt))
        fake = fake_vllm_config(
            ckpt,
            hf_config,
            dtype=dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            num_speculative_tokens=num_speculative_tokens,
        )
        self.max_tokens = fake.scheduler_config.max_num_batched_tokens
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with (
                patched(state_mod, "get_current_vllm_config", lambda: fake),
                torch.device(self.device),
            ):
                self.lm = model_mod.Inkling2LoopTextForCausalLM(
                    vllm_config=fake, serving=(path == "hip")
                )
        finally:
            torch.set_default_dtype(old_dtype)
        self.cfg = self.lm.config
        with torch.no_grad():
            loaded = self.lm.load_weights(iter(tensors.items()))
        assert "lm_head.weight" in loaded and "model.message_input.query" in loaded
        if experts == "exact":
            _, ref_tool = _require_safetensors()
            for i, layer in enumerate(self.lm.model.layers):
                dense = moe_mod.ExpertsReference(
                    self.cfg, dtype=torch.float32, device=self.device
                )
                up, down = ref_tool.dequant_experts(
                    tensors, f"model.llm.layers.{i}.mlp.experts"
                )
                dense.gate_up_proj.data.copy_(up)
                dense.down_proj.data.copy_(down)
                layer.mlp.forward = self._dense_moe_forward(layer.mlp, dense)
        self.world: PagedWorld | None = None

    @staticmethod
    def _dense_moe_forward(moe: Any, dense_experts: Any):
        def forward(hidden: torch.Tensor) -> torch.Tensor:
            _, weights, ids = moe.gate(hidden)
            return dense_experts(hidden.float(), ids, weights).to(hidden.dtype)

        return forward

    def reset(
        self,
        *,
        kv_block_size: int = 16,
        conv_block_size: int | None = None,
        num_blocks: int = 128,
        seed: int = 0,
    ) -> TinyPagedModel:
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
            hip=(self.path == "hip"),
            max_tokens=self.max_tokens,
        )
        return self

    @torch.no_grad()
    def step_with_hidden(
        self,
        work: list[tuple[int, list[int]]],
        *,
        accepted: dict[int, int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        spec = [(r, len(t)) for r, t in work]
        input_ids = torch.tensor(
            [x for _, t in work for x in t], dtype=torch.long, device=self.device
        )
        assert self.world is not None
        positions = self.world.positions(spec)
        mds = self.world.metadata(spec)
        with patched(model_mod, "get_metadata", lambda prefix: mds[prefix]):
            hidden = self.lm(input_ids, positions)
            assert isinstance(hidden, torch.Tensor)
            hidden_copy = hidden.clone()
            logits_ = self.lm.compute_logits(hidden)
        assert logits_ is not None
        logits = logits_.float().cpu()
        self.world.commit(spec, accepted)
        out: dict[int, torch.Tensor] = {}
        at = 0
        for r, n in spec:
            out[r] = logits[at : at + n]
            at += n
        return hidden_copy, out

    @torch.no_grad()
    def step(
        self,
        work: list[tuple[int, list[int]]],
        *,
        accepted: dict[int, int] | None = None,
    ) -> dict[int, torch.Tensor]:
        _, out = self.step_with_hidden(work, accepted=accepted)
        return out


def _cache_bits(pm: TinyPagedModel) -> list[torch.Tensor]:
    assert pm.world is not None
    return [
        owner.kv_cache.view(torch.int16).clone()
        for g in pm.world.groups.values()
        for owner in g.owners
    ]


def _restore_cache_bits(pm: TinyPagedModel, saved: list[torch.Tensor]) -> None:
    assert pm.world is not None
    owners = [owner for g in pm.world.groups.values() for owner in g.owners]
    assert len(owners) == len(saved)
    for owner, bits in zip(owners, saved):
        owner.kv_cache.view(torch.int16).copy_(bits)


class LoopCDGraphRig:
    """Single-capture FULL HIP-graph harness mirroring ``ModelCudaGraphManager``."""

    MAX_REQS = 8
    MAX_TOKENS = 64

    def __init__(self, pm: TinyPagedModel) -> None:
        assert pm.world is not None
        self.pm = pm
        self.world = pm.world
        dev = pm.device
        self.ids = torch.zeros(self.MAX_TOKENS, dtype=torch.long, device=dev)
        self.pos = torch.zeros(self.MAX_TOKENS, dtype=torch.long, device=dev)
        self.qsl = torch.zeros(self.MAX_REQS + 1, dtype=torch.int32, device=dev)
        self.seq_lens = torch.zeros(self.MAX_REQS, dtype=torch.int32, device=dev)
        self.tables: dict[str, torch.Tensor] = {}
        self.slots: dict[str, torch.Tensor] = {}
        self.scratch: dict[str, state_mod.SeqIdxScratch] = {}
        for g in self.world.groups.values():
            width = cdiv(self.world.max_len, g.bs) + 1
            self.tables[g.name] = torch.zeros(
                self.MAX_REQS, width, dtype=torch.int32, device=dev
            )
            self.slots[g.name] = torch.full(
                (self.MAX_TOKENS,), -1, dtype=torch.int64, device=dev
            )
            self.scratch[g.name] = state_mod.SeqIdxScratch(self.MAX_TOKENS, dev)
        self.hidden = torch.zeros(
            self.MAX_TOKENS, pm.cfg.hidden_size, dtype=torch.float16, device=dev
        )
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mds: dict[str, state_mod.Inkling2Metadata] = {}

    @staticmethod
    def dummy_batch(reqs: int, q: int) -> SimpleNamespace:
        n = reqs * q
        return SimpleNamespace(
            ids=[0] * n,
            pos=[0] * n,
            qsl=[i * q for i in range(reqs + 1)],
            seq_lens=[q] * reqs,
            tables={},
            slots={},
        )

    def real_batch(self, work: list[tuple[int, list[int]]]) -> SimpleNamespace:
        world = self.world
        starts: list[int] = [0]
        ids: list[int] = []
        pos: list[int] = []
        seq_lens: list[int] = []
        for r, toks in work:
            p0 = world.computed.get(r, 0)
            ids += toks
            pos += range(p0, p0 + len(toks))
            seq_lens.append(p0 + len(toks))
            starts.append(starts[-1] + len(toks))
        tables: dict[str, list[list[int]]] = {}
        slots: dict[str, list[int]] = {}
        for g in world.groups.values():
            width = cdiv(world.max_len, g.bs) + 1
            rows, sl = [], []
            for r, toks in work:
                p0 = world.computed.get(r, 0)
                g.tables.setdefault(r, [])
                g.remove_skipped(r, p0)
                g.allocate(r, p0 + len(toks))
                rows.append(g.row(r, width, p0))
                t = g.tables[r]
                sl += [
                    t[p // g.bs] * g.bs + p % g.bs for p in range(p0, p0 + len(toks))
                ]
            tables[g.name], slots[g.name] = rows, sl
        return SimpleNamespace(
            ids=ids, pos=pos, qsl=starts, seq_lens=seq_lens, tables=tables, slots=slots
        )

    def load(
        self,
        b: SimpleNamespace,
        reqs: int,
        tokens: int,
        *,
        max_query_len: int,
        max_seq_len: int,
    ) -> None:
        n = len(b.ids)
        self.ids.zero_()
        self.ids[:n] = torch.tensor(b.ids, dtype=torch.long)
        self.pos.zero_()
        self.pos[:n] = torch.tensor(b.pos, dtype=torch.long)
        qsl = b.qsl + [n] * (reqs - (len(b.qsl) - 1))
        self.qsl.fill_(n)
        self.qsl[: len(qsl)] = torch.tensor(qsl, dtype=torch.int32)
        self.seq_lens.zero_()
        self.seq_lens[: len(b.seq_lens)] = torch.tensor(b.seq_lens, dtype=torch.int32)
        self.mds = {}
        for g in self.world.groups.values():
            table, slot = self.tables[g.name], self.slots[g.name]
            table.zero_()
            rows = b.tables.get(g.name, [])
            if rows:
                table[: len(rows)] = torch.tensor(rows, dtype=torch.int32)
            slot.fill_(-1)
            if b.slots.get(g.name):
                slot[:n] = torch.tensor(b.slots[g.name], dtype=torch.int64)
            common = SimpleNamespace(
                num_reqs=reqs,
                num_actual_tokens=tokens,
                query_start_loc=self.qsl,
                query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
                seq_lens=self.seq_lens,
                block_table_tensor=table[:reqs],
                slot_mapping=slot[:tokens],
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
            )
            md = state_mod.make_metadata(common, self.scratch[g.name])
            for owner in g.owners:
                self.mds[owner.prefix] = md

    @torch.no_grad()
    def forward(self, tokens: int) -> torch.Tensor:
        with patched(model_mod, "get_metadata", lambda prefix: self.mds[prefix]):
            out = self.pm.lm(self.ids[:tokens], self.pos[:tokens])
            assert isinstance(out, torch.Tensor)
            return out

    @torch.no_grad()
    def capture(self, reqs: int, q: int) -> None:
        tokens = reqs * q
        self.load(
            self.dummy_batch(reqs, q),
            reqs,
            tokens,
            max_query_len=q,
            max_seq_len=self.world.max_len,
        )
        self.hidden[:tokens] = self.forward(tokens)  # warm up views/packs
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.hidden[:tokens] = self.forward(tokens)
        self.graphs[(reqs, q)] = graph

    @torch.no_grad()
    def run_step_no_commit(
        self,
        batch: SimpleNamespace,
        reqs: int,
        q: int,
        *,
        replay: bool,
        graph_trace_ops: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = reqs * q
        n = len(batch.ids)
        self.load(
            batch,
            reqs,
            tokens,
            max_query_len=q,
            max_seq_len=self.world.max_len,
        )
        if replay:
            self.graphs[(reqs, q)].replay()
        else:
            ws = self.pm.lm.model.glue_ws
            if graph_trace_ops and ws is not None and ws.has_loop:
                old_h, old_l = ws.loopcd_hidden_active, ws.loopcd_logits_active
                ws.loopcd_hidden_active = True
                ws.loopcd_logits_active = True
                try:
                    self.hidden[:tokens] = self.forward(tokens)
                finally:
                    ws.loopcd_hidden_active = old_h
                    ws.loopcd_logits_active = old_l
            else:
                self.hidden[:tokens] = self.forward(tokens)
        h_out = self.hidden[:n].clone()
        logits = self.pm.lm.compute_logits(self.hidden[:n])
        assert logits is not None
        return h_out, logits.float().clone()


# ===========================================================================
# 1. CPU Config & Reference Oracle Unit Tests
# ===========================================================================

_LOOPCD_ENV_VARS = (
    "VLLM_INKLING2_LOOPCD_ALPHA",
    "VLLM_INKLING2_LOOPCD_MODE",
    "VLLM_INKLING2_LOOPCD_ADAPTIVE",
    "VLLM_INKLING2_LOOPCD_BETA",
    "VLLM_INKLING2_LOOPCD_BLEND_BLOCKS",
    "VLLM_INKLING2_LOOPCD_WEAK_PASS",
)


def _clear_loopcd_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _LOOPCD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_loopcd_params_defaults_and_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_loopcd_env(monkeypatch)
    p = LoopCDParams()
    assert p.alpha == 0.0
    assert p.mode == "hidden"
    assert p.adaptive is False
    assert math.isclose(p.beta, 0.1, rel_tol=1e-7)
    assert p.blend_blocks is True
    assert p.weak_pass == -1
    assert p.enabled is False
    assert p.mode_code == loopcd_mod.LOOPCD_MODE_OFF == 0
    assert math.isclose(p.log_beta, math.log(0.1), rel_tol=1e-6)

    p_hid = LoopCDParams(alpha=0.25, mode="hidden")
    assert p_hid.enabled is True
    assert p_hid.mode_code == loopcd_mod.LOOPCD_MODE_HIDDEN == 1

    p_zero_beta = LoopCDParams(alpha=0.25, mode="logits", beta=0.0)
    assert p_zero_beta.enabled is True
    assert p_zero_beta.mode_code == loopcd_mod.LOOPCD_MODE_LOGITS == 2
    assert math.isinf(p_zero_beta.log_beta) and p_zero_beta.log_beta < 0

    env_def = read_loopcd_env_defaults()
    assert env_def == p


def test_read_loopcd_env_defaults_and_parse_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_loopcd_env(monkeypatch)
    monkeypatch.setenv("VLLM_INKLING2_LOOPCD_ALPHA", "0.35")
    monkeypatch.setenv("VLLM_INKLING2_LOOPCD_MODE", "logits")
    monkeypatch.setenv("VLLM_INKLING2_LOOPCD_ADAPTIVE", "1")
    monkeypatch.setenv("VLLM_INKLING2_LOOPCD_BETA", "0.05")
    monkeypatch.setenv("VLLM_INKLING2_LOOPCD_BLEND_BLOCKS", "0")
    monkeypatch.setenv("VLLM_INKLING2_LOOPCD_WEAK_PASS", "0")

    defaults = read_loopcd_env_defaults()
    assert math.isclose(defaults.alpha, 0.35, rel_tol=1e-6)
    assert defaults.mode == "logits"
    assert defaults.adaptive is True
    assert math.isclose(defaults.beta, 0.05, rel_tol=1e-6)
    assert defaults.blend_blocks is False
    assert defaults.weak_pass == 0
    assert defaults.enabled is True
    assert defaults.mode_code == 2

    # None / empty extra_args preserves defaults.
    assert parse_loopcd_params(None, defaults=defaults) == defaults
    assert parse_loopcd_params({}, defaults=defaults) == defaults

    # Per-request extra_args override individual fields cleanly.
    over = parse_loopcd_params(
        {
            "loopcd_alpha": 0.2,
            "loopcd_mode": "hidden",
            "loopcd_adaptive": "false",
            "loopcd_beta": 0.0,
            "loopcd_blend_blocks": "true",
            "loopcd_weak_pass": -1,
        },
        defaults=defaults,
    )
    assert math.isclose(over.alpha, 0.2, rel_tol=1e-6)
    assert over.mode == "hidden"
    assert over.adaptive is False
    assert over.beta == 0.0 and math.isinf(over.log_beta) and over.log_beta < 0
    assert over.blend_blocks is True
    assert over.weak_pass == -1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
    ],
)
def test_parse_loopcd_bool_coercions(
    monkeypatch: pytest.MonkeyPatch, raw: Any, expected: bool
) -> None:
    _clear_loopcd_env(monkeypatch)
    p = parse_loopcd_params(
        {
            "loopcd_alpha": 0.25,
            "loopcd_adaptive": raw,
            "loopcd_blend_blocks": raw,
        }
    )
    assert p.adaptive is expected
    assert p.blend_blocks is expected


@pytest.mark.parametrize(
    "bad_extra",
    [
        {"loopcd_alpha": -0.1},
        {"loopcd_alpha": float("nan")},
        {"loopcd_alpha": float("inf")},
        {"loopcd_mode": "invalid_mode"},
        {"loopcd_mode": "off"},
        {"loopcd_beta": -0.01},
        {"loopcd_beta": 1.5},
        {"loopcd_beta": float("nan")},
        {"loopcd_weak_pass": -2},
    ],
)
def test_parse_loopcd_params_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, bad_extra: dict[str, Any]
) -> None:
    _clear_loopcd_env(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        parse_loopcd_params(bad_extra)


def _unpack_blend_ref(
    res: Any,
    x0_in: torch.Tensor,
    x1_in: torch.Tensor | None,
    blk_in: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Normalize ``loopcd_blend_reference`` return shape ``(x0, x1, blk)`` or
    ``(h, blk)`` into ``(combined_h, x1_or_none, blk_or_none)``."""
    if isinstance(res, tuple) and len(res) == 3:
        rx0, rx1, rblk = res
        rh = rx0 if rx1 is None else (rx0 + rx1)
        return rh, rx1, rblk
    if isinstance(res, tuple) and len(res) == 2:
        rh, rblk = res
        return rh, None, rblk
    if res is None:
        rh = x0_in if x1_in is None else (x0_in + x1_in)
        return rh, x1_in, blk_in
    raise AssertionError(f"Unexpected loopcd_blend_reference return: {type(res)}")


@pytest.mark.parametrize("with_x1", [False, True])
@pytest.mark.parametrize("num_blk", [0, 3])
def test_loopcd_blend_reference_exact_math_and_zero_ulp(
    with_x1: bool, num_blk: int
) -> None:
    gen = torch.Generator().manual_seed(42 + int(with_x1) * 7 + num_blk)
    T, H = 6, 64
    x0 = torch.randn(T, H, generator=gen, dtype=torch.float32)
    x0[0, 0] = -0.0  # signed-zero bit-pattern check on omega == 0.0
    x1 = torch.randn(T, H, generator=gen, dtype=torch.float32) if with_x1 else None
    prev_x0 = torch.randn(T, H, generator=gen, dtype=torch.float32)
    prev_x1 = torch.randn(T, H, generator=gen, dtype=torch.float32)
    blk = (
        torch.randn(T, num_blk, H, generator=gen, dtype=torch.float32)
        if num_blk > 0
        else None
    )
    prev_blk = (
        torch.randn(T, num_blk, H, generator=gen, dtype=torch.float32)
        if num_blk > 0
        else None
    )
    omega = torch.tensor([0.0, 0.25, 0.0, 0.5, 0.0, 0.75], dtype=torch.float32)
    omega_blk = torch.tensor([0.0, 0.25, 0.0, 0.0, 0.0, 0.4], dtype=torch.float32)

    x0_in = x0.clone()
    x1_in = None if x1 is None else x1.clone()
    blk_in = None if blk is None else blk.clone()
    res = loopcd_blend_reference(
        x0_in, x1_in, prev_x0, prev_x1, blk_in, prev_blk, omega, omega_blk=omega_blk
    )
    rh, rx1, rblk = _unpack_blend_ref(res, x0_in, x1_in, blk_in)

    inactive = omega == 0.0
    active = ~inactive
    if isinstance(res, tuple) and len(res) == 3:
        assert torch.equal(res[0][inactive], x0[inactive])
        if x1 is not None and rx1 is not None:
            assert torch.equal(rx1[inactive], x1[inactive])
    else:
        orig_h = x0 if x1 is None else (x0 + x1)
        assert torch.equal(rh[inactive], orig_h[inactive])

    h_final = x0.double() + (0.0 if x1 is None else x1.double())
    h_prev = prev_x0.double() + prev_x1.double()
    w = omega.double().unsqueeze(-1)
    want_h = (h_final + w * (h_final - h_prev)).float()
    torch.testing.assert_close(rh[active], want_h[active], atol=1e-6, rtol=1e-6)

    if num_blk > 0:
        assert blk is not None and prev_blk is not None and rblk is not None
        blk_inactive = omega_blk == 0.0
        blk_active = ~blk_inactive
        assert torch.equal(rblk[blk_inactive], blk[blk_inactive])
        wb = omega_blk.double().view(T, 1, 1)
        want_blk = (blk.double() + wb * (blk.double() - prev_blk.double())).float()
        torch.testing.assert_close(
            rblk[blk_active], want_blk[blk_active], atol=1e-6, rtol=1e-6
        )


def test_loopcd_logits_reference_zero_ulp_cutoff_and_adaptive_damping() -> None:
    # 1) 0-ULP bit-identity when omega == 0.0
    gen = torch.Generator().manual_seed(99)
    z_final = torch.randn(4, 37, generator=gen, dtype=torch.float32)
    z_final[0, 3] = -0.0
    z_first = torch.randn(4, 37, generator=gen, dtype=torch.float32)
    omega = torch.tensor([0.0, 0.35, 0.0, 0.6], dtype=torch.float32)
    log_beta = torch.full((4,), math.log(0.1), dtype=torch.float32)
    adaptive = torch.tensor([False, False, True, True], dtype=torch.bool)

    out = loopcd_logits_reference(
        z_final.clone(), z_first.clone(), omega, log_beta, adaptive
    )
    assert torch.equal(out[0], z_final[0])
    assert torch.equal(out[2], z_final[2])

    # 2) Plausibility cutoff -inf masking & exact fixed contrastive differences
    row1 = z_final[1].double()
    row1_first = z_first[1].double()
    cutoff1 = row1.max() + math.log(0.1)
    plausible1 = row1 >= cutoff1
    assert torch.equal(torch.isneginf(out[1]), ~plausible1)
    assert torch.isfinite(out[1][plausible1]).all()
    idx = torch.nonzero(plausible1, as_tuple=False).squeeze(-1)
    assert idx.numel() >= 2
    u, v = int(idx[0]), int(idx[1])
    got_diff = float(out[1, u].double() - out[1, v].double())
    want_diff = float(
        1.35 * (row1[u] - row1[v]) - 0.35 * (row1_first[u] - row1_first[v])
    )
    assert math.isclose(got_diff, want_diff, rel_tol=1e-5, abs_tol=1e-5)

    # beta = 0.0 (log_beta = -inf) keeps all finite entries finite
    out_no_cut = loopcd_logits_reference(
        z_final.clone(),
        z_first.clone(),
        torch.full((4,), 0.4, dtype=torch.float32),
        torch.full((4,), float("-inf"), dtype=torch.float32),
        torch.zeros(4, dtype=torch.bool),
    )
    assert torch.isfinite(out_no_cut).all()

    # 3) Adaptive (1 - p_max) damping:
    #    Row 0: sharp near-deterministic distribution -> p_max ~ 1 -> omega_eff ~ 0
    #    Row 1: exact 2-way uniform tie -> p_max = 0.5 -> omega_eff = 0.5 * omega_0
    z_f = torch.tensor(
        [
            [30.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, -100.0, -100.0],
        ],
        dtype=torch.float32,
    )
    z_w = torch.tensor(
        [
            [2.0, 4.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    omega_0 = 0.4
    adap_out = loopcd_logits_reference(
        z_f.clone(),
        z_w.clone(),
        torch.tensor([omega_0, omega_0], dtype=torch.float32),
        torch.tensor([float("-inf"), math.log(0.1)], dtype=torch.float32),
        torch.tensor([True, True], dtype=torch.bool),
    )
    # Sharp row: out[0, 0] - out[0, 1] = (1 + w_eff)*30 - w_eff*(-2) = 30 + 32*w_eff
    sharp_w_eff = (float(adap_out[0, 0] - adap_out[0, 1]) - 30.0) / 32.0
    assert abs(sharp_w_eff) < 1e-6, f"Expected omega_eff ~ 0, got {sharp_w_eff}"

    # 2-way tie row: z_f[1, 0] == z_f[1, 1] == 10, z_w[1, 0] - z_w[1, 1] == -2
    # => out[1, 0] - out[1, 1] = +2 * omega_eff, with omega_eff = 0.5 * 0.4 = 0.2
    tie_w_eff = float(adap_out[1, 0] - adap_out[1, 1]) / 2.0
    assert math.isclose(tie_w_eff, 0.5 * omega_0, rel_tol=1e-6, abs_tol=1e-6)
    assert torch.isneginf(adap_out[1, 2]) and torch.isneginf(adap_out[1, 3])


def test_cpu_reference_model_loopcd_modes_and_zero_ulp(tmp_path: Path) -> None:
    make, ref_tool = _require_safetensors()
    spec = make.TinySpec(
        num_hidden_layers=8,
        loop_start=2,
        loop_end=6,
        loop_num_passes=3,
        router_bias_scale=100.0,
    )
    make.write_checkpoint(tmp_path, spec, seed=0)
    tensors = ref_tool.read_checkpoint(tmp_path)
    pm = TinyPagedModel(
        tmp_path,
        tensors,
        torch.float16,
        device="cpu",
        experts="exact",
        path="reference",
    )
    ids = _prompts()[0]

    pm.reset(kv_block_size=16)
    pm.lm.clear_loopcd()
    h_base, log_base_map = pm.step_with_hidden([(0, ids)])
    log_base = log_base_map[0]
    bits_base = _cache_bits(pm)

    for mode in ("hidden", "logits"):
        pm.reset(kv_block_size=16)
        pm.lm.set_loopcd(LoopCDParams(alpha=0.0, mode=mode))
        h_zero, log_zero_map = pm.step_with_hidden([(0, ids)])
        assert torch.equal(h_zero, h_base)
        assert torch.equal(log_zero_map[0], log_base)
        for a, b in zip(_cache_bits(pm), bits_base):
            assert torch.equal(a, b)

    # Non-zero alpha in both hidden and logits modes changes output logits.
    for params in (
        LoopCDParams(alpha=0.35, mode="hidden", blend_blocks=True),
        LoopCDParams(alpha=0.35, mode="hidden", blend_blocks=False),
        LoopCDParams(alpha=0.35, mode="logits", adaptive=False, weak_pass=-1),
        LoopCDParams(alpha=0.35, mode="logits", adaptive=True, weak_pass=0),
    ):
        pm.reset(kv_block_size=16)
        pm.lm.set_loopcd(params)
        moved = pm.step([(0, ids)])[0]
        assert not torch.equal(moved, log_base)
    pm.lm.clear_loopcd()


# ===========================================================================
# 2. GPU Kernel-Level Parity & 0-ULP Tests
# ===========================================================================


def _call_loopcd_logits_op(
    z_final: torch.Tensor,
    z_first: torch.Tensor,
    omega: torch.Tensor,
    log_beta: torch.Tensor,
    adaptive_bool: list[bool],
    *,
    stats_scratch: torch.Tensor | None = None,
) -> None:
    adap = torch.tensor(adaptive_bool, dtype=torch.bool, device=z_final.device)
    ops.inkling2_loopcd_logits(
        z_final,
        z_first,
        omega,
        log_beta,
        adap,
        stats_scratch=stats_scratch,
    )


def _call_prepare_hidden_op(
    omega_out: torch.Tensor,
    omega_blk_out: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    req_alpha: torch.Tensor,
    req_mode: torch.Tensor,
    blend_blk_vals: list[bool],
    num_tokens_padded: int,
) -> None:
    req_blend = torch.tensor(
        [1 if b else 0 for b in blend_blk_vals],
        dtype=torch.int32,
        device=omega_out.device,
    )
    ops.inkling2_loopcd_prepare_hidden(
        omega_out,
        omega_blk_out,
        idx_mapping,
        query_start_loc,
        seq_lens,
        prefill_len,
        req_alpha,
        req_mode,
        req_blend,
        num_tokens_padded,
    )


def _call_loopcd_readout_op(
    blocks: torch.Tensor,
    n_blocks: int,
    x0: torch.Tensor,
    query: torch.Tensor,
    norm_w: torch.Tensor,
    out_2d: torch.Tensor,
    *,
    x1: torch.Tensor | None = None,
    gate_flag: torch.Tensor | None = None,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> None:
    ops.inkling2_glue_loopcd_readout(
        blocks,
        n_blocks,
        x0,
        query if query.dim() == 2 else query.unsqueeze(0),
        norm_w if norm_w.dim() == 2 else norm_w.unsqueeze(0),
        out_2d.unsqueeze(1),
        x1=x1,
        gate_flag=gate_flag,
        eps=eps,
        norm_eps=norm_eps,
    )


@_NEEDS_GPU
@pytest.mark.parametrize("T", [1, 2, 5, 16, 32])
@pytest.mark.parametrize("H", [256, 2304])
@pytest.mark.parametrize("P", [0, 1, 3])
@pytest.mark.parametrize("with_x1", [False, True])
@pytest.mark.parametrize("blend_blocks", [True, False])
def test_gpu_loopcd_blend_parity_and_zero_ulp(
    T: int, H: int, P: int, with_x1: bool, blend_blocks: bool
) -> None:
    _require_loopcd_hip_ops()
    gen = torch.Generator().manual_seed(1000 + T * 31 + H + P * 7 + int(with_x1))
    x0_cpu = torch.randn(T, H, generator=gen, dtype=torch.float32)
    x0_cpu[0, 0] = -0.0
    x1_cpu = torch.randn(T, H, generator=gen, dtype=torch.float32) if with_x1 else None
    prev_x0_cpu = torch.randn(T, H, generator=gen, dtype=torch.float32)
    prev_x1_cpu = torch.randn(T, H, generator=gen, dtype=torch.float32)

    pattern = [0.0, 0.3, 0.0, 0.5]
    omega_vals = [pattern[i % len(pattern)] for i in range(T)]
    omega_cpu = torch.tensor(omega_vals, dtype=torch.float32)
    if blend_blocks:
        blk_pattern = [0.0, 0.3, 0.2, 0.0]
        omega_blk_cpu = torch.tensor(
            [blk_pattern[i % len(blk_pattern)] for i in range(T)], dtype=torch.float32
        )
    else:
        omega_blk_cpu = torch.zeros(T, dtype=torch.float32)

    if P > 0:
        # Exercise non-trivial outer stride (slice of [T, P + 2, H])
        # matching v.blk[:, num_pre + 1 : n].
        full_blk_cpu = torch.randn(T, P + 2, H, generator=gen, dtype=torch.float32)
        prev_blk_cpu = torch.randn(T, P, H, generator=gen, dtype=torch.float32)
        blk_ref_in = full_blk_cpu[:, 1 : 1 + P].clone()
    else:
        full_blk_cpu = None
        prev_blk_cpu = None
        blk_ref_in = None

    ref_res = loopcd_blend_reference(
        x0_cpu.clone(),
        None if x1_cpu is None else x1_cpu.clone(),
        prev_x0_cpu,
        prev_x1_cpu,
        blk_ref_in,
        prev_blk_cpu,
        omega_cpu,
        omega_blk=omega_blk_cpu,
    )
    want_h, want_x1, want_blk = _unpack_blend_ref(
        ref_res,
        x0_cpu,
        x1_cpu,
        blk_ref_in,
    )

    x0_gpu = x0_cpu.cuda()
    x1_gpu = None if x1_cpu is None else x1_cpu.cuda()
    prev_x0_gpu = prev_x0_cpu.cuda()
    prev_x1_gpu = prev_x1_cpu.cuda()
    omega_gpu = omega_cpu.cuda()
    omega_blk_gpu = omega_blk_cpu.cuda()
    if P > 0:
        assert full_blk_cpu is not None and prev_blk_cpu is not None
        full_blk_gpu = full_blk_cpu.cuda()
        blk_gpu = full_blk_gpu[:, 1 : 1 + P]
        prev_blk_gpu = prev_blk_cpu.cuda()
    else:
        full_blk_gpu = None
        blk_gpu = None
        prev_blk_gpu = None

    ops.inkling2_glue_loopcd_blend(
        x0_gpu,
        prev_x0_gpu,
        prev_x1_gpu,
        omega_gpu,
        x1=x1_gpu,
        blk=blk_gpu,
        prev_blk=prev_blk_gpu,
        omega_blk=omega_blk_gpu,
    )

    inactive = omega_cpu == 0.0
    active = ~inactive
    assert torch.equal(x0_gpu.cpu()[inactive], x0_cpu[inactive])
    if x1_gpu is not None and x1_cpu is not None:
        assert torch.equal(x1_gpu.cpu()[inactive], x1_cpu[inactive])

    got_h = x0_gpu.cpu() if x1_gpu is None else (x0_gpu.cpu() + x1_gpu.cpu())
    if active.any():
        torch.testing.assert_close(got_h[active], want_h[active], atol=1e-6, rtol=1e-6)
        if want_x1 is not None and x1_gpu is not None:
            torch.testing.assert_close(
                x1_gpu.cpu()[active], want_x1[active], atol=1e-6, rtol=1e-6
            )

    if P > 0:
        assert (
            blk_gpu is not None
            and full_blk_gpu is not None
            and full_blk_cpu is not None
        )
        assert want_blk is not None
        # Surrounding block slots outside [1 : 1 + P] must stay 0-ULP untouched.
        assert torch.equal(full_blk_gpu[:, 0].cpu(), full_blk_cpu[:, 0])
        assert torch.equal(full_blk_gpu[:, 1 + P].cpu(), full_blk_cpu[:, 1 + P])
        blk_inactive = omega_blk_cpu == 0.0
        blk_active = ~blk_inactive
        assert torch.equal(
            blk_gpu.cpu()[blk_inactive], full_blk_cpu[:, 1 : 1 + P][blk_inactive]
        )
        if blk_active.any():
            torch.testing.assert_close(
                blk_gpu.cpu()[blk_active],
                want_blk[blk_active],
                atol=1e-6,
                rtol=1e-6,
            )


@_NEEDS_GPU
@pytest.mark.parametrize("B", [1, 2, 4, 8, 17])
@pytest.mark.parametrize("V", [500, 200058])
@pytest.mark.parametrize("beta", [0.0, 0.05, 0.1, 0.5])
def test_gpu_loopcd_logits_parity_and_zero_ulp(B: int, V: int, beta: float) -> None:
    _require_loopcd_hip_ops()
    gen = torch.Generator(device="cuda").manual_seed(
        2000 + B * 19 + (V % 97) + int(beta * 1000)
    )
    z_final = torch.randn(B, V, generator=gen, device="cuda", dtype=torch.float32) * 4.0
    z_first = torch.randn(B, V, generator=gen, device="cuda", dtype=torch.float32) * 4.0
    z_final[0, 0] = -0.0

    log_b = float("-inf") if beta == 0.0 else math.log(beta)
    if math.isfinite(log_b):
        # Keep non-tie test points >1e-4 away from exact fp32 threshold ties so -inf
        # mask comparison is deterministic across all 200,058 columns.
        row_max = z_final.max(dim=-1, keepdim=True).values
        thresh = row_max + log_b
        near_tie = (z_final - thresh).abs() < 2e-4
        z_final = torch.where(near_tie, z_final + 1e-3, z_final)

    omega_pat = [0.0, 0.25, 0.0, 0.5]
    adap_pat = [False, True, False, True]
    omega_vals = [omega_pat[i % 4] for i in range(B)]
    adap_vals = [adap_pat[i % 4] for i in range(B)]
    omega = torch.tensor(omega_vals, dtype=torch.float32, device="cuda")
    log_beta = torch.full((B,), log_b, dtype=torch.float32, device="cuda")

    z_orig = z_final.clone()
    z_gpu = z_final.clone()
    _call_loopcd_logits_op(z_gpu, z_first, omega, log_beta, adap_vals)

    want = loopcd_logits_reference(
        z_orig.double().cpu(),
        z_first.double().cpu(),
        omega.double().cpu(),
        log_beta.double().cpu(),
        torch.tensor(adap_vals, dtype=torch.bool),
    ).float()
    got = z_gpu.cpu()

    inactive = torch.tensor([w == 0.0 for w in omega_vals], dtype=torch.bool)
    assert torch.equal(got[inactive], z_orig.cpu()[inactive])

    active = ~inactive
    if active.any():
        got_act = got[active]
        want_act = want[active]
        assert torch.equal(torch.isneginf(got_act), torch.isneginf(want_act))
        finite = torch.isfinite(want_act)
        assert torch.isfinite(got_act[finite]).all()
        torch.testing.assert_close(
            got_act[finite], want_act[finite], atol=1e-5, rtol=1e-5
        )


@_NEEDS_GPU
def test_gpu_loopcd_logits_non_contiguous_sliced_stride() -> None:
    """Sliced ``z[:, :200058]`` view of padded ``[B, 201024]`` logits."""
    _require_loopcd_hip_ops()
    B, V, V_PAD = 4, 200058, 201024
    gen = torch.Generator(device="cuda").manual_seed(7777)
    z_final_pad = (
        torch.randn(B, V_PAD, generator=gen, device="cuda", dtype=torch.float32) * 3.5
    )
    z_first_pad = (
        torch.randn(B, V_PAD, generator=gen, device="cuda", dtype=torch.float32) * 3.5
    )
    tail_sentinel = z_final_pad[:, V:].clone()

    z_final = z_final_pad[:, :V]
    z_first = z_first_pad[:, :V]
    assert not z_final.is_contiguous() and z_final.stride(0) == V_PAD

    log_b = math.log(0.1)
    thresh = z_final.max(dim=-1, keepdim=True).values + log_b
    z_final.copy_(torch.where((z_final - thresh).abs() < 2e-4, z_final + 1e-3, z_final))
    z_orig = z_final.clone()

    omega = torch.tensor([0.0, 0.25, 0.0, 0.5], dtype=torch.float32, device="cuda")
    log_beta = torch.full((B,), log_b, dtype=torch.float32, device="cuda")
    adap_vals = [False, True, False, True]

    _call_loopcd_logits_op(z_final, z_first, omega, log_beta, adap_vals)

    # Padding columns [200058 : 201024] must remain 0-ULP untouched.
    assert torch.equal(z_final_pad[:, V:], tail_sentinel)
    # Inactive rows [0, 2] must remain 0-ULP untouched.
    assert torch.equal(z_final[[0, 2]], z_orig[[0, 2]])

    want = loopcd_logits_reference(
        z_orig.double().cpu(),
        z_first.double().cpu(),
        omega.double().cpu(),
        log_beta.double().cpu(),
        torch.tensor(adap_vals, dtype=torch.bool),
    ).float()
    got = z_final.cpu()
    for r in (1, 3):
        assert torch.equal(torch.isneginf(got[r]), torch.isneginf(want[r]))
        fin = torch.isfinite(want[r])
        torch.testing.assert_close(got[r][fin], want[r][fin], atol=1e-5, rtol=1e-5)


@_NEEDS_GPU
def test_gpu_loopcd_prepare_hidden_all_schedule_geometries() -> None:
    """Single-chunk prefill, non-final & final chunked prefill, decode, DSpark
    verify, ``mode != 1``, ``blend_blocks=False``, and CUDA-graph padding tail."""
    _require_loopcd_hip_ops()
    dev = torch.device("cuda")
    num_slots = 16
    prefill_len = torch.zeros(num_slots, dtype=torch.int32, device=dev)
    req_alpha = torch.zeros(num_slots, dtype=torch.float32, device=dev)
    req_mode = torch.zeros(num_slots, dtype=torch.int32, device=dev)
    blend_blk = [True] * num_slots

    # Populate distinct state table slots (non-identity idx_mapping):
    # req 0 -> slot 3: single-chunk prefill (P=12, seq_len=12 == prefill_len=12)
    # req 1 -> slot 9: non-final chunked prefill (qlen=8, seq_len=8 < prefill_len=16)
    # req 2 -> slot 1: final chunked prefill (qlen=8, seq_len=16 == prefill_len=16)
    # req 3 -> slot 12: decode (qlen=1, seq_len=17 > prefill_len=16)
    # req 4 -> slot 5: DSpark verify (qlen=4, seq_len=20 > prefill_len=16)
    # req 5 -> slot 7: mode=2 (logits) on decode -> 0.0
    # req 6 -> slot 2: mode=0 (off) on decode -> 0.0
    # req 7 -> slot 14: blend_blocks=False on DSpark verify (qlen=3)
    #   -> omega=0.4, omega_blk=0.0
    specs = [
        # (slot, qlen, seq_len, p_len, alpha, mode, b_blk, want_w, want_wb)
        (3, 12, 12, 12, 0.35, 1, True, [0.0] * 11 + [0.35], [0.0] * 11 + [0.35]),
        (9, 8, 8, 16, 0.45, 1, True, [0.0] * 8, [0.0] * 8),
        (1, 8, 16, 16, 0.45, 1, True, [0.0] * 7 + [0.45], [0.0] * 7 + [0.45]),
        (12, 1, 17, 16, 0.25, 1, True, [0.25], [0.25]),
        (5, 4, 20, 16, 0.30, 1, True, [0.30] * 4, [0.30] * 4),
        (7, 1, 25, 10, 0.50, 2, True, [0.0], [0.0]),
        (2, 2, 12, 10, 0.50, 0, True, [0.0, 0.0], [0.0, 0.0]),
        (14, 3, 19, 16, 0.40, 1, False, [0.40] * 3, [0.0] * 3),
    ]
    idx_list: list[int] = []
    qsl_list: list[int] = [0]
    seq_list: list[int] = []
    want_omega: list[float] = []
    want_omega_blk: list[float] = []

    for slot, qlen, slen, plen, alpha, mode, b_blk, ww, wwb in specs:
        prefill_len[slot] = plen
        req_alpha[slot] = alpha
        req_mode[slot] = mode
        blend_blk[slot] = b_blk
        idx_list.append(slot)
        qsl_list.append(qsl_list[-1] + qlen)
        seq_list.append(slen)
        want_omega.extend(ww)
        want_omega_blk.extend(wwb)

    actual_tokens = qsl_list[-1]
    pad_tail = 11
    num_tokens_padded = actual_tokens + pad_tail
    want_omega.extend([0.0] * pad_tail)
    want_omega_blk.extend([0.0] * pad_tail)

    # Pre-poison output buffers (including entries beyond num_tokens_padded)
    omega_out = torch.full(
        (num_tokens_padded + 8,), 99.0, dtype=torch.float32, device=dev
    )
    omega_blk_out = torch.full(
        (num_tokens_padded + 8,), 88.0, dtype=torch.float32, device=dev
    )
    idx_mapping = torch.tensor(idx_list, dtype=torch.int32, device=dev)
    qsl = torch.tensor(qsl_list, dtype=torch.int32, device=dev)
    seq_lens = torch.tensor(seq_list, dtype=torch.int32, device=dev)

    _call_prepare_hidden_op(
        omega_out,
        omega_blk_out,
        idx_mapping,
        qsl,
        seq_lens,
        prefill_len,
        req_alpha,
        req_mode,
        blend_blk,
        num_tokens_padded,
    )

    exp_w = torch.tensor(want_omega, dtype=torch.float32)
    exp_wb = torch.tensor(want_omega_blk, dtype=torch.float32)
    assert torch.equal(omega_out[:num_tokens_padded].cpu(), exp_w)
    assert torch.equal(omega_blk_out[:num_tokens_padded].cpu(), exp_wb)
    # Entries past num_tokens_padded stay untouched.
    assert (omega_out[num_tokens_padded:].cpu() == 99.0).all()
    assert (omega_blk_out[num_tokens_padded:].cpu() == 88.0).all()


@_NEEDS_GPU
@pytest.mark.parametrize("T", [1, 3, 8, 17])
@pytest.mark.parametrize("H", [256, 2304])
@pytest.mark.parametrize("n_blocks", [0, 1, 3])
@pytest.mark.parametrize("with_x1", [False, True])
def test_gpu_loopcd_readout_gate_zero_ulp_and_attn_res_exact(
    T: int, H: int, n_blocks: int, with_x1: bool
) -> None:
    _require_loopcd_hip_ops()
    gen = torch.Generator(device="cuda").manual_seed(
        5000 + T * 13 + H + n_blocks * 5 + int(with_x1)
    )
    blocks = torch.randn(
        T, max(n_blocks, 1), H, generator=gen, device="cuda", dtype=torch.float32
    )
    x0 = torch.randn(T, H, generator=gen, device="cuda", dtype=torch.float32)
    x1 = (
        torch.randn(T, H, generator=gen, device="cuda", dtype=torch.float32)
        if with_x1
        else None
    )
    query = torch.randn(1, H, generator=gen, device="cuda", dtype=torch.float32)
    norm_w = (
        torch.randn(1, H, generator=gen, device="cuda", dtype=torch.float32)
        .abs()
        .add_(0.2)
        .half()
    )

    # 1) gate_flag = 0 -> out stays 0-ULP untouched
    sentinel = torch.linspace(-17.5, 23.25, T * H, device="cuda", dtype=torch.float32)
    sentinel = sentinel.half().view(T, H).contiguous()
    out_gated_off = sentinel.clone()
    flag_zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    _call_loopcd_readout_op(
        blocks,
        n_blocks,
        x0,
        query,
        norm_w,
        out_gated_off,
        x1=x1,
        gate_flag=flag_zero,
    )
    assert torch.equal(out_gated_off, sentinel)

    # 2) gate_flag = 1 -> bit-identical to inkling2_glue_attn_res
    out_gated_on = sentinel.clone()
    flag_one = torch.ones(1, dtype=torch.int32, device="cuda")
    _call_loopcd_readout_op(
        blocks,
        n_blocks,
        x0,
        query,
        norm_w,
        out_gated_on,
        x1=x1,
        gate_flag=flag_one,
    )

    out_ref = torch.empty(T, 1, H, dtype=torch.float16, device="cuda")
    ops.inkling2_glue_attn_res(
        blocks,
        n_blocks,
        x0,
        query,
        out_ref,
        x1=x1,
        norm_w=norm_w,
        eps=1e-6,
        norm_eps=1e-6,
    )
    assert torch.equal(out_gated_on, out_ref.squeeze(1))


# ===========================================================================
# 3. GPU Model-Level Parity, 0-ULP Gate, Prefix-Cache & FULL HIP-Graph Replay
# ===========================================================================


@pytest.fixture(scope="module")
def gpu_tiny8_bundle(tmp_path_factory: pytest.TempPathFactory):
    if not torch.cuda.is_available():
        pytest.skip("needs a ROCm GPU")
    _require_loopcd_hip_ops()
    make, ref_tool = _require_safetensors()
    spec = make.TinySpec(**STABLE_SPEC_KWARGS)
    ckpt = tmp_path_factory.mktemp("tiny8-loopcd-stable")
    make.write_checkpoint(ckpt, spec, seed=0)
    tensors = ref_tool.read_checkpoint(ckpt)
    hip_pm = TinyPagedModel(
        ckpt,
        tensors,
        torch.float16,
        device="cuda",
        experts="kernel",
        path="hip",
    )
    ref_pm = TinyPagedModel(
        ckpt,
        tensors,
        torch.float16,
        device="cuda",
        experts="kernel",
        path="reference",
    )
    return SimpleNamespace(ckpt=ckpt, tensors=tensors, hip=hip_pm, ref=ref_pm)


@_NEEDS_GPU
def test_gpu_model_zero_ulp_baseline_gate(gpu_tiny8_bundle: SimpleNamespace) -> None:
    """``alpha=0.0`` in BOTH ``LoopCD-Hidden`` and ``LoopCD-Logits`` is 0-ULP
    bit-identical (hidden, logits, and KV/conv pages) to ``clear_loopcd``."""
    pm: TinyPagedModel = gpu_tiny8_bundle.hip
    ws = pm.lm.model.glue_ws
    assert ws is not None
    assert ws.loopcd_omega is not None
    assert ws.loopcd_logits_flag is not None
    assert ws.loopcd_weak_h is not None
    for attr in (
        "loopcd_omega",
        "loopcd_omega_blk",
        "loopcd_logits_flag",
        "loopcd_weak_h",
    ):
        assert hasattr(ws, attr), f"missing workspace buffer {attr}"

    prompt = _prompts()[1]  # 40 tokens
    decode_toks = [13, 27]

    def run_arm(params: LoopCDParams | None):
        pm.reset(kv_block_size=16, seed=0)
        if params is None:
            pm.lm.clear_loopcd()
        else:
            pm.lm.set_loopcd(params)
        h_p, l_p = pm.step_with_hidden([(0, prompt)])
        h_d0, l_d0 = pm.step_with_hidden([(0, [decode_toks[0]])])
        h_d1, l_d1 = pm.step_with_hidden([(0, [decode_toks[1]])])
        bits = _cache_bits(pm)
        pm.lm.clear_loopcd()
        return (h_p, h_d0, h_d1), (l_p[0], l_d0[0], l_d1[0]), bits

    base_h, base_l, base_bits = run_arm(None)
    for mode in ("hidden", "logits"):
        got_h, got_l, got_bits = run_arm(LoopCDParams(alpha=0.0, mode=mode))
        for a, b in zip(got_h, base_h):
            assert torch.equal(a, b), f"{mode} alpha=0.0 hidden differs"
        for a, b in zip(got_l, base_l):
            assert torch.equal(a, b), f"{mode} alpha=0.0 logits differ"
        for i, (a, b) in enumerate(zip(got_bits, base_bits)):
            assert torch.equal(a, b), f"{mode} alpha=0.0 cache page {i} differs"


@_NEEDS_GPU
@pytest.mark.parametrize(
    "params",
    [
        LoopCDParams(alpha=0.35, mode="hidden", blend_blocks=True),
        LoopCDParams(alpha=0.35, mode="hidden", blend_blocks=False),
        LoopCDParams(alpha=0.30, mode="logits", adaptive=False, beta=0.1, weak_pass=-1),
        LoopCDParams(alpha=0.30, mode="logits", adaptive=True, beta=0.1, weak_pass=-1),
        LoopCDParams(alpha=0.30, mode="logits", adaptive=False, beta=0.1, weak_pass=0),
        LoopCDParams(alpha=0.30, mode="logits", adaptive=True, beta=0.1, weak_pass=0),
    ],
)
def test_gpu_model_reference_vs_hip_parity(
    gpu_tiny8_bundle: SimpleNamespace, params: LoopCDParams
) -> None:
    """``serving=False`` (``forward_reference``) vs ``serving=True`` (``forward_hip``)
    agree within fp16 tolerance across LoopCD-Hidden and LoopCD-Logits variants,
    and ``alpha > 0`` genuinely moves logits away from ``alpha = 0``."""
    hip_pm: TinyPagedModel = gpu_tiny8_bundle.hip
    ref_pm: TinyPagedModel = gpu_tiny8_bundle.ref
    ids = _prompts()[1]  # 40 tokens

    hip_pm.reset(kv_block_size=16, seed=0)
    hip_pm.lm.clear_loopcd()
    hip_base = torch.cat(
        [hip_pm.step([(0, ids[:32])])[0], hip_pm.step([(0, [ids[32]])])[0]], dim=0
    )

    outs: dict[str, torch.Tensor] = {}
    for tag, pm in (("hip", hip_pm), ("ref", ref_pm)):
        pm.reset(kv_block_size=16, seed=0)
        pm.lm.set_loopcd(params)
        p_out = pm.step([(0, ids[:32])])[0]
        d_out = pm.step([(0, [ids[32]])])[0]
        outs[tag] = torch.cat([p_out, d_out], dim=0)
        pm.lm.clear_loopcd()

    got, want = outs["hip"], outs["ref"]
    # Verify LoopCD actually moved the output away from baseline.
    finite_base = torch.isfinite(got) & torch.isfinite(hip_base)
    assert finite_base.any()
    assert float((got[finite_base] - hip_base[finite_base]).abs().max()) > 1e-3

    # Compare distributions across reference and HIP serving paths.
    # Under LoopCD-Logits with beta > 0, implausible tokens are masked to -inf
    # in both paths (so raw log_softmax subtraction would form (-inf) - (-inf) = NaN).
    logp_got = torch.log_softmax(got.float(), dim=-1)
    logp_want = torch.log_softmax(want.float(), dim=-1)
    p_got = torch.softmax(got.float(), dim=-1)
    p_want = torch.softmax(want.float(), dim=-1)
    assert torch.isfinite(p_got).all() and torch.isfinite(p_want).all()
    mean_tv = float((0.5 * (p_got - p_want).abs().sum(dim=-1)).mean())
    assert mean_tv < 0.02, f"{params}: mean TV={mean_tv:.3e}"

    both_finite = torch.isfinite(got) & torch.isfinite(want)
    assert both_finite.any()
    if params.mode == "hidden":
        assert both_finite.all()
    mean_dlogp = float((logp_got[both_finite] - logp_want[both_finite]).abs().mean())
    assert mean_dlogp < 0.05, f"{params}: mean|dlogp|={mean_dlogp:.3e}"
    top2 = want.topk(2, dim=-1).values
    clear = torch.isfinite(top2[:, 1]) & ((top2[:, 0] - top2[:, 1]) > 0.6)
    if clear.any():
        assert torch.equal(got.argmax(-1)[clear], want.argmax(-1)[clear])


@_NEEDS_GPU
def test_gpu_prefix_cache_invariance_with_loopcd_hidden(
    gpu_tiny8_bundle: SimpleNamespace,
) -> None:
    """Prefilling 40 tokens (``kv_block_size=16``) with ``LoopCD-Hidden`` ``alpha=0.5``
    and ``omega = [0.0]*39 + [0.5]`` leaves the 2 full cached blocks (tokens 0..31)
    and a forked ``alpha=0.0`` request bit-identical (``torch.equal``) to a cold
    ``alpha=0.0`` run."""
    pm: TinyPagedModel = gpu_tiny8_bundle.hip
    ids = _prompts()[1]  # 40 tokens -> 2 full 16-token blocks (0..31) + 8 tail tokens
    assert len(ids) == 40
    tail_req1 = [5, 19, 41, 83, 12, 77, 9, 31]
    decode_steps = [17, 29]

    def run_scenario(loopcd_on_owner: bool):
        pm.reset(kv_block_size=16, conv_block_size=16, seed=0)
        assert pm.world is not None
        if loopcd_on_owner:
            omega = torch.tensor([0.0] * 39 + [0.5], dtype=torch.float32, device="cuda")
            pm.lm.set_loopcd(
                LoopCDParams(alpha=0.5, mode="hidden", blend_blocks=True),
                omega=omega,
                omega_blk=omega,
            )
        else:
            pm.lm.clear_loopcd()

        owner_prefill = pm.step([(0, ids)])[0]

        # Extract raw 16-bit cache contents of the 2 full blocks (tokens 0..31)
        block_snapshots: dict[str, list[torch.Tensor]] = {}
        for gname, g in pm.world.groups.items():
            full_blk_ids = g.tables[0][:2]
            assert len(full_blk_ids) == 2 and 0 not in full_blk_ids
            block_snapshots[gname] = [
                owner.kv_cache[full_blk_ids].view(torch.int16).clone()
                for owner in g.owners
            ]

        # Fork the 32 cached prefix tokens to request 1 running with alpha=0.0
        pm.world.fork_prefix(0, 1, 32)
        pm.lm.clear_loopcd()
        fork_outs = [pm.step([(1, tail_req1)])[1]]
        for tok in decode_steps:
            fork_outs.append(pm.step([(1, [tok])])[1])
        return owner_prefill, block_snapshots, torch.cat(fork_outs, dim=0)

    warm_owner, warm_blocks, warm_fork_logits = run_scenario(loopcd_on_owner=True)
    cold_owner, cold_blocks, cold_fork_logits = run_scenario(loopcd_on_owner=False)

    # Owner's first 39 token logits have omega=0.0 -> bit-identical, token 39 differs!
    assert torch.equal(warm_owner[:39], cold_owner[:39])
    assert not torch.equal(warm_owner[39], cold_owner[39])

    # All KV + conv cache pages of the 2 shared full blocks are 0-ULP bit-identical!
    for gname in warm_blocks:
        for idx, (wb, cb) in enumerate(zip(warm_blocks[gname], cold_blocks[gname])):
            assert torch.equal(wb, cb), f"group {gname} owner {idx} block cache differs"

    # Forked alpha=0.0 request logits (tail prefill + decode) are 0-ULP bit-identical!
    assert torch.equal(warm_fork_logits, cold_fork_logits)


@_NEEDS_GPU
def test_gpu_full_hip_graph_capture_once_and_replay_all_modes(
    gpu_tiny8_bundle: SimpleNamespace,
) -> None:
    """Capture ONE ``FULL`` HIP graph with ``alpha=0.0`` (``ws.loopcd_omega`` zeros,
    ``ws.loopcd_logits_flag = 0``), and without recapturing verify bit-exact ``B == C``
    replay for:
    1. ``alpha=0.0`` vs eager ``B`` and ``clear_loopcd()`` baseline,
    2. ``LoopCD-Hidden`` mixed batch ``omega = [0.0, 0.35, 0.0, 0.5]``,
    3. ``LoopCD-Logits`` (``flag = 1``, ``alpha = 0.25``, ``adaptive = True``).
    """
    pm: TinyPagedModel = gpu_tiny8_bundle.hip
    ws = pm.lm.model.glue_ws
    assert ws is not None
    assert ws.loopcd_omega is not None
    assert ws.loopcd_logits_flag is not None
    assert ws.loopcd_weak_h is not None

    gen = torch.Generator().manual_seed(2026)
    prefill_lens = (20, 30, 16, 24)
    seqs = {
        r: torch.randint(0, 500, (n + 4,), generator=gen).tolist()
        for r, n in enumerate(prefill_lens)
    }

    pm.reset(kv_block_size=16, seed=0)
    rig = LoopCDGraphRig(pm)
    pm.lm.clear_loopcd()
    pm.step([(r, seqs[r][: prefill_lens[r]]) for r in range(4)])

    # Capture ONE FULL graph at (reqs=4, q=1) with alpha=0.0
    pm.lm.set_loopcd(LoopCDParams(alpha=0.0, mode="hidden"))
    assert float(ws.loopcd_omega[:4].abs().max().item()) == 0.0
    assert int(ws.loopcd_logits_flag.item()) == 0
    rig.capture(4, 1)
    assert list(rig.graphs.keys()) == [(4, 1)]

    # Prepare the real 4-request decode batch once and snapshot cache bits
    decode_work = [(r, [seqs[r][prefill_lens[r]]]) for r in range(4)]
    batch = rig.real_batch(decode_work)
    saved_caches = _cache_bits(pm)

    # --- 1. Replay with alpha=0.0 vs eager B and clear_loopcd() baseline ---
    _restore_cache_bits(pm, saved_caches)
    pm.lm.clear_loopcd()
    base_h, base_logits = rig.run_step_no_commit(
        batch, 4, 1, replay=False, graph_trace_ops=False
    )

    for mode in ("hidden", "logits"):
        _restore_cache_bits(pm, saved_caches)
        pm.lm.set_loopcd(LoopCDParams(alpha=0.0, mode=mode))
        eager0_h, eager0_logits = rig.run_step_no_commit(batch, 4, 1, replay=False)

        _restore_cache_bits(pm, saved_caches)
        pm.lm.set_loopcd(LoopCDParams(alpha=0.0, mode=mode))
        graph0_h, graph0_logits = rig.run_step_no_commit(batch, 4, 1, replay=True)

        assert torch.equal(graph0_h, eager0_h)
        assert torch.equal(graph0_h, base_h)
        assert torch.equal(graph0_logits, eager0_logits)
        assert torch.equal(graph0_logits, base_logits)

    # --- 2. Replay with LoopCD-Hidden mixed batch omega = [0.0, 0.35, 0.0, 0.5] ---
    omega_mix = torch.tensor([0.0, 0.35, 0.0, 0.5], dtype=torch.float32, device="cuda")
    hid_params = LoopCDParams(alpha=0.5, mode="hidden", blend_blocks=True)

    _restore_cache_bits(pm, saved_caches)
    pm.lm.set_loopcd(hid_params, omega=omega_mix, omega_blk=omega_mix)
    eager_hid_h, eager_hid_logits = rig.run_step_no_commit(batch, 4, 1, replay=False)

    _restore_cache_bits(pm, saved_caches)
    pm.lm.set_loopcd(hid_params, omega=omega_mix, omega_blk=omega_mix)
    graph_hid_h, graph_hid_logits = rig.run_step_no_commit(batch, 4, 1, replay=True)

    assert torch.equal(graph_hid_h, eager_hid_h)
    assert torch.equal(graph_hid_logits, eager_hid_logits)
    # Inactive rows [0, 2] match baseline bit-for-bit; active rows [1, 3] move!
    assert torch.equal(graph_hid_logits[[0, 2]], base_logits[[0, 2]])
    assert not torch.equal(graph_hid_logits[1], base_logits[1])
    assert not torch.equal(graph_hid_logits[3], base_logits[3])

    # --- 3. Replay with LoopCD-Logits (flag=1, alpha=0.25, adaptive=True) ---
    logits_params = LoopCDParams(
        alpha=0.25, mode="logits", adaptive=True, beta=0.1, weak_pass=-1
    )

    _restore_cache_bits(pm, saved_caches)
    pm.lm.set_loopcd(logits_params)
    assert int(ws.loopcd_logits_flag.item()) == 1
    ws.loopcd_weak_h[:4].fill_(-777.0)
    eager_cd_h, eager_cd_logits = rig.run_step_no_commit(batch, 4, 1, replay=False)
    eager_weak_h = ws.loopcd_weak_h[:4].clone()

    _restore_cache_bits(pm, saved_caches)
    pm.lm.set_loopcd(logits_params)
    assert int(ws.loopcd_logits_flag.item()) == 1
    ws.loopcd_weak_h[:4].fill_(888.0)
    graph_cd_h, graph_cd_logits = rig.run_step_no_commit(batch, 4, 1, replay=True)
    graph_weak_h = ws.loopcd_weak_h[:4].clone()

    assert torch.isfinite(graph_weak_h).all()
    assert torch.equal(graph_weak_h, eager_weak_h)
    assert torch.equal(graph_cd_h, eager_cd_h)
    assert torch.equal(graph_cd_logits, eager_cd_logits)
    assert not torch.equal(graph_cd_logits, base_logits)

    # Confirm no extra graph was captured.
    assert list(rig.graphs.keys()) == [(4, 1)]
    pm.lm.clear_loopcd()
