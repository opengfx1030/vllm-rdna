# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--monokernel`` next to the W4A8 MoE prefill (``VLLM_RDNA2_W4A8_SDOT4=1``).

The two never serve the same step and read one set of packed expert weights:
load only sets a Python flag for W4A8, K2 serves at most 8 rows, the modules
9-24 rows on the W4A16 GEMM, and the W4A8 GEMM starts at 512 tokens. These CPU
checks pin that down: the switch accepts W4A8 and still refuses the expert
cache with it, load leaves the weights and buffers identical with W4A8 on or
off, K2 binds the tensors the W4A8 GEMM reads, and the MoE dispatch hands the
W4A8 and the W4A16 GEMM the very same tensors.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.experts import rdna2_w4a16_moe as moe
from vllm.models.qwen4_exp.amd import mono_decode_rdna2 as md
from vllm.models.qwen4_exp.amd.mono.rdna2 import OPTS, Options, layout

CACHE_REFUSAL = r"^--monokernel expert cache .*W4A8 MoE prefill .*\.$"
EXPERT_TENSORS = (
    ("w13_weight_packed", "w13_weight_scale", "w13_qzeros"),
    ("w2_weight_packed", "w2_weight_scale", "w2_qzeros"),
)


def _config(knobs=None, tp=None):
    extra = {} if knobs is None else {"rocm_mono_decode": knobs}
    arch = "Qwen4ExpForConditionalGeneration"
    cfg = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=True),
        model_config=SimpleNamespace(architectures=[arch]),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        use_v2_model_runner=True,
        additional_config=extra,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    if tp is not None:  # what check_config reads for the expert cache
        cfg.parallel_config.tensor_parallel_size = tp
        cfg.compilation_config.cudagraph_mode = None
        cfg.compilation_config.splitting_ops = []
    return cfg


@pytest.fixture
def deployment(monkeypatch):
    """A deployment the kernels take: gfx10x, every op built, W4A8 off."""
    from vllm import envs

    monkeypatch.setattr(md, "_on_gfx10x", lambda: True)
    monkeypatch.setattr(md, "ops_refusal", lambda: None)
    monkeypatch.setattr(envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    monkeypatch.setenv("VLLM_RAISE_ON_LOGIT_NANS", "1")
    monkeypatch.delenv(moe.W4A8_MOE_ENV_VAR, raising=False)
    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: False)
    saved = Options()
    saved.assign(OPTS)
    yield
    OPTS.assign(saved)


def _w4a8_on(monkeypatch):
    monkeypatch.setenv(moe.W4A8_MOE_ENV_VAR, "1")
    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: True)


class CompressedTensorsWNA16RDNA2MoEMethod:  # name-matched stand-in
    def __init__(self, w4a8=False, symmetric=True):
        self.symmetric = symmetric
        self._zero_v2 = not symmetric
        self._w4a8 = w4a8


class _Experts(nn.Module):
    def __init__(self, qm):
        super().__init__()
        self.quant_method = qm


class _Layer(nn.Module):
    """The parts of a Qwen4Exp decoder layer the switch looks at."""

    def __init__(self, qm):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _Experts(qm)


# ---- the switch -----------------------------------------------------------


def test_w4a8_is_accepted_next_to_k2(deployment, monkeypatch):
    _w4a8_on(monkeypatch)
    cfg = _config()
    OPTS.assign(md.options(cfg))
    layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod(w4a8=True))
    assert md.moe_refusal(layer) is None
    names = md.layer_hook(layer, cfg, "model.layers.3", True)
    assert names == ("model.layers.3.mono_rdna2",) * 2
    assert not cfg.compilation_config.static_forward_context[names[0]].is_mtp


def test_expert_cache_without_w4a8_is_accepted_per_layer(deployment):
    cfg = _config(knobs={"expert_cache_slots": 32})
    OPTS.assign(md.options(cfg))
    layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    assert md.moe_refusal(layer) is None


def test_expert_cache_refuses_w4a8_per_layer(deployment, monkeypatch):
    _w4a8_on(monkeypatch)
    cfg = _config(knobs={"expert_cache_slots": 32})
    OPTS.assign(md.options(cfg))
    layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod(w4a8=True))
    assert "VLLM_RDNA2_W4A8_SDOT4" in md.moe_refusal(layer)
    with pytest.raises(ValueError, match=CACHE_REFUSAL):
        md.layer_hook(layer, cfg, "model.layers.3", True)
    assert not cfg.compilation_config.static_forward_context


def test_expert_cache_refuses_w4a8_at_the_api_server(deployment, monkeypatch):
    cache = {"expert_cache_slots": 32}
    md.check_config(_config(knobs=cache, tp=4))  # the cache alone
    monkeypatch.setenv(moe.W4A8_MOE_ENV_VAR, "0")
    md.check_config(_config(knobs=cache, tp=4))
    monkeypatch.setenv(moe.W4A8_MOE_ENV_VAR, "1")
    md.check_config(_config())  # W4A8 alone
    with pytest.raises(ValueError, match=CACHE_REFUSAL):
        md.check_config(_config(knobs=cache, tp=4))


def test_no_row_cap_reaches_the_w4a8_gate():
    """K1 / K2 / GDN never serve a step the W4A8 GEMM does, whatever the knobs."""
    gate = moe.W4A8_MOE_MIN_TOKENS
    assert layout.MAX_ROWS == md._MAX_ROWS
    assert max(md._MAX_ROWS, md._GDN_MAX_ROWS) < gate
    huge = 10**6
    o = md.options(
        _config(
            knobs={
                "k1_max_rows": huge,
                "k2_max_rows": huge,
                "mtp_k1_max_rows": huge,
                "mtp_k2_max_rows": huge,
                "gdn_max_rows": huge,
            }
        )
    )
    caps = (o.k1_max_rows, o.k2_max_rows, o.mtp_k1_max_rows, o.mtp_k2_max_rows)
    assert max(*caps, o.gdn_max_rows) < gate


def test_boot_log_names_where_w4a8_takes_over(monkeypatch):
    lines = []
    monkeypatch.setattr(
        md, "logger", SimpleNamespace(info=lambda fmt, *a: lines.append(fmt % a))
    )
    o = md.options(_config())
    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: False)
    assert md._w4a8_part() == ""
    md._log_parts(o)
    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: True)
    md._log_parts(o)
    off, on = lines
    assert "W4A8" not in off  # without W4A8 the line is what it always was
    assert on == off + f", {md._w4a8_part()}"
    assert f"from {moe.W4A8_MOE_MIN_TOKENS} tokens" in on


# ---- K2 binds what the W4A8 GEMM reads --------------------------------------


def _target_layer(*, w4a8, symmetric=True):
    """A Flash-Next TP4/EP4 decoder layer on the meta device, as K2 resolves it."""

    def meta(shape, dtype=torch.float16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def hc():
        return SimpleNamespace(
            hc_count=4,
            use_combine=True,
            hc_norm=SimpleNamespace(weight=meta((10240,))),
            input_mix_weight_down_block_inject=SimpleNamespace(
                weight=meta((336, 10240)), weight_i8=None
            ),
            input_mix_weight_up=SimpleNamespace(
                weight=meta((10240, 320)), weight_i8=None
            ),
            config=SimpleNamespace(rms_norm_eps=1e-6),
        )

    experts = SimpleNamespace(
        quant_method=CompressedTensorsWNA16RDNA2MoEMethod(w4a8, symmetric),
        w13_weight_packed=meta((128, 320, 1280), torch.int32),
        w13_weight_scale=meta((128, 20, 1280)),
        w13_qzeros=meta((128, 20, 160), torch.int32),
        w2_weight_packed=meta((128, 80, 2560), torch.int32),
        w2_weight_scale=meta((128, 5, 2560)),
        w2_qzeros=meta((128, 5, 320), torch.int32),
        expert_map=torch.arange(128, dtype=torch.int32),
        top_k=10,
        renormalize=True,
    )
    mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=meta((512, 2560))),
        experts=experts,
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=meta((320, 2560))),
            down_proj=SimpleNamespace(weight=meta((2560, 160))),
            expert_gate=SimpleNamespace(weight=meta((1, 2560))),
        ),
    )
    return SimpleNamespace(
        mlp=mlp, mlp_hyper_connection=hc(), attn_hyper_connection=hc()
    )


@pytest.fixture
def mono_layout(deployment, monkeypatch):
    monkeypatch.setattr(layout, "workspace", lambda device, scope: object())
    monkeypatch.setattr(layout, "ZEROS", {})


@pytest.mark.parametrize("w4a8", [False, True], ids=["w4a16", "w4a8"])
def test_k2_binds_the_tensors_the_w4a8_gemm_reads(mono_layout, w4a8):
    layer = _target_layer(w4a8=w4a8)
    mono = layout.MonoLayer(layer)
    assert mono.setup(), mono.reason
    assert mono.ready and mono.k1_ready
    experts = layer.mlp.experts
    for key, attr in (
        ("w13", "w13_weight_packed"),
        ("s13", "w13_weight_scale"),
        ("z13", "w13_qzeros"),
        ("w2", "w2_weight_packed"),
        ("s2", "w2_weight_scale"),
        ("z2", "w2_qzeros"),
    ):
        assert mono.t[key] is getattr(experts, attr), key  # no copy, no repack


def test_k2_still_refuses_asymmetric_experts(mono_layout):
    mono = layout.MonoLayer(_target_layer(w4a8=True, symmetric=False))
    assert not mono.setup()
    assert "asymmetric" in mono.reason


# ---- load changes nothing but the flag ------------------------------------------


def _rdna2_method_module():
    pytest.importorskip("compressed_tensors")
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_wna16_rdna2 as wiring,
    )

    return wiring


def _checkpoint_layer(seed, experts=3, hidden=64, inter=32, group=32):
    """Compressed-tensors layout as loaded: packed [E, N, K/8], scales [E, N, G]."""
    gen = torch.Generator().manual_seed(seed)
    layer = nn.Module()
    for name, n, k in (
        ("w13", 2 * inter, hidden),
        ("w2", hidden, inter),
    ):
        packed = torch.randint(0, 2**30, (experts, n, k // 8), generator=gen).int()
        scale = torch.rand(experts, n, k // group, generator=gen).half()
        for suffix, tensor in (("weight_packed", packed), ("weight_scale", scale)):
            layer.register_parameter(
                f"{name}_{suffix}", nn.Parameter(tensor, requires_grad=False)
            )
    return layer


def _load(wiring, monkeypatch, *, w4a8):
    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: w4a8)
    monkeypatch.setattr(
        wiring.ops, "gptq_shuffle", lambda t, bits: t.bitwise_xor_(0x5), raising=False
    )
    method = object.__new__(wiring.CompressedTensorsWNA16RDNA2MoEMethod)
    method.group_size = 32
    method.symmetric = True
    layer = _checkpoint_layer(seed=7)
    method.process_weights_after_loading(layer)
    return method, layer


def test_load_leaves_weights_and_buffers_identical_with_w4a8(monkeypatch):
    wiring = _rdna2_method_module()
    off_method, off = _load(wiring, monkeypatch, w4a8=False)
    on_method, on = _load(wiring, monkeypatch, w4a8=True)
    assert off_method._w4a8 is False
    assert on_method._w4a8 is True
    others = (
        {k: v for k, v in vars(m).items() if k != "_w4a8"}
        for m in (off_method, on_method)
    )
    assert next(others) == next(others)  # group size, symmetry, zero convention

    off_params, on_params = dict(off.named_parameters()), dict(on.named_parameters())
    assert off_params.keys() == on_params.keys()
    assert {n for pair in EXPERT_TENSORS for n in pair} <= set(off_params)
    for name, p in off_params.items():
        q = on_params[name]
        assert (p.dtype, p.shape, p.stride()) == (q.dtype, q.shape, q.stride()), name
        assert torch.equal(p, q), name
    # the scratch K2 does not read: same names, shapes and dtypes
    scratch = {k for k, v in vars(off).items() if isinstance(v, torch.Tensor)}
    assert scratch == {k for k, v in vars(on).items() if isinstance(v, torch.Tensor)}
    assert scratch == {"rdna2_w1_buf", "rdna2_act_buf", "rdna2_empty_tw"}
    for name in scratch:
        a, b = getattr(off, name), getattr(on, name)
        assert (a.dtype, a.shape, a.device) == (b.dtype, b.shape, b.device), name


# ---- dispatch: one layout, two GEMMs ----------------------------------------------


@pytest.fixture
def gemm_calls(monkeypatch):
    """Record the two MoE GEMMs (kind, weights, scales, zeros) instead of running
    them; everything around them (routing, activation) is stubbed."""
    wiring = _rdna2_method_module()
    calls = []

    def recorder(kind):
        def gemm(a, c, w, s, z, *rest, **kw):
            calls.append((kind, w, s, z))

        return gemm

    for op, kind in (("moe_w4a8_gemm_rdna2", "w4a8"), ("moe_gptq_gemm_rdna2", "w4a16")):
        monkeypatch.setattr(wiring.ops, op, recorder(kind), raising=False)
    monkeypatch.setattr(wiring, "_ROUTE_ZERO", False)
    monkeypatch.setattr(wiring, "_DETERMINISTIC", False)
    ids = (torch.zeros(1, dtype=torch.int32),) * 3
    monkeypatch.setattr(wiring, "moe_align_block_size", lambda *a, **k: ids)
    monkeypatch.setattr(wiring, "apply_moe_activation", lambda *a, **k: None)
    return wiring, calls


def _expert_layer():
    f16, i32 = torch.float16, torch.int32
    return SimpleNamespace(
        w13_weight_packed=torch.zeros(4, 8, 64, dtype=i32),
        w13_weight_scale=torch.zeros(4, 2, 64, dtype=f16),
        w13_qzeros=torch.zeros(4, 2, 8, dtype=i32),
        w2_weight_packed=torch.zeros(4, 4, 64, dtype=i32),
        w2_weight_scale=torch.zeros(4, 1, 64, dtype=f16),
        w2_qzeros=torch.zeros(4, 1, 8, dtype=i32),
        rdna2_w1_buf=torch.zeros(16, 64, dtype=f16),
        rdna2_act_buf=torch.zeros(16, 32, dtype=f16),
        rdna2_empty_tw=torch.empty(0),
    )


def _fused_moe(wiring, layer, tokens, *, w4a8):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    wiring._rdna2_fused_moe(
        torch.zeros(tokens, 64, dtype=torch.float16),
        torch.ones(tokens, 2),
        torch.zeros(tokens, 2, dtype=torch.int32),
        layer,
        MoEActivation.SILU,
        False,
        4,
        None,
        w4a8=w4a8,
    )


@pytest.mark.parametrize("tokens", [3, 8, 24, 511, 512, 2048], ids=lambda n: f"{n}tok")
def test_w4a8_gemm_starts_at_the_gate_on_the_same_tensors(gemm_calls, tokens):
    wiring, calls = gemm_calls
    layer = _expert_layer()
    _fused_moe(wiring, layer, tokens, w4a8=True)
    kind = "w4a8" if tokens >= moe.W4A8_MOE_MIN_TOKENS else "w4a16"
    assert [c[0] for c in calls] == [kind, kind]  # w13, then w2
    for (_, *got), names in zip(calls, EXPERT_TENSORS):
        assert all(g is getattr(layer, n) for g, n in zip(got, names))

    # W4A8 switched off: the same call reads the same tensors with the W4A16 GEMM
    on = [c[1:] for c in calls]
    calls.clear()
    _fused_moe(wiring, layer, tokens, w4a8=False)
    assert [c[0] for c in calls] == ["w4a16", "w4a16"]
    for got, want in zip([c[1:] for c in calls], on):
        assert all(g is w for g, w in zip(got, want))
