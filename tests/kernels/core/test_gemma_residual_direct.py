# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_RDNA_GEMMA_RESIDUAL_DIRECT: GemmaRMSNorm(x, residual) on gfx10x.

Flag on: the fp16 (1 + w) scale is cached in a buffer refreshed by the weight
loader and torch.ops._C.fused_add_rms_norm runs in place on x and residual.
The reference is the flag-off path (functional opaque op, IR clones).
GPU tests skip off gfx10x; the gating tests run anywhere.
"""

import pytest
import torch

from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    _gemma_residual_direct,
)
from vllm.platforms import current_platform

ENV = "VLLM_RDNA_GEMMA_RESIDUAL_DIRECT"
HIDDEN = 5120
EPS = 1e-6


def _on_gfx10x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


requires_gfx10x = pytest.mark.skipif(
    not _on_gfx10x(), reason="needs a gfx10x ROCm device"
)


@pytest.fixture(autouse=True)
def vllm_c_priority():
    """gfx10x serves the IR norm ops through vllm_c (rocm.py ir_op_priority); a
    bare test config would pick the native impl, which rounds differently."""
    if not _on_gfx10x():
        yield
        return
    from vllm import ir

    with (
        ir.ops.rms_norm.set_priority(["vllm_c", "native"]),
        ir.ops.fused_add_rms_norm.set_priority(["vllm_c", "native"]),
    ):
        yield


def _real_like_weight(gen: torch.Generator, hidden: int = HIDDEN) -> torch.Tensor:
    # Gemma-style gamma (scale = 1 + w): mostly small, a few channels far from 0.
    w = torch.randn(hidden, generator=gen, device="cpu") * 0.3
    idx = torch.randperm(hidden, generator=gen)[:16]
    w[idx] = torch.randn(16, generator=gen) * 3.0
    return w.to(torch.float16)


def _real_like_act(
    gen: torch.Generator, rows: int, hidden: int = HIDDEN
) -> torch.Tensor:
    # Residual-stream-like: unit bulk plus a few massive-activation channels.
    a = torch.randn(rows, hidden, generator=gen)
    a[:, :4] *= 60.0
    return a.to(torch.float16)


def _make_norm(
    monkeypatch, direct: bool, weight: torch.Tensor, device: str = "cuda"
) -> GemmaRMSNorm:
    monkeypatch.setenv(ENV, "1" if direct else "0")
    with torch.device(device):
        layer = GemmaRMSNorm(weight.numel(), eps=EPS).to(torch.float16)
    # Load through the parameter's own loader, as AutoWeightsLoader does.
    loader = getattr(layer.weight, "weight_loader", None)
    if loader is None:
        layer.weight.data.copy_(weight)
    else:
        loader(layer.weight, weight)
    return layer


def test_flag_off_has_no_state(default_vllm_config, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    layer = GemmaRMSNorm(64)
    assert not layer.rdna_residual_direct
    assert not hasattr(layer, "rdna_scale16")
    assert not hasattr(layer.weight, "weight_loader")


def test_flag_requires_gfx10x(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert _gemma_residual_direct() == _on_gfx10x()
    monkeypatch.setenv(ENV, "0")
    assert not _gemma_residual_direct()


@requires_gfx10x
@pytest.mark.parametrize("rows", [1, 2, 8, 24, 2048])
@torch.inference_mode()
def test_bit_exact_vs_functional(default_vllm_config, monkeypatch, rows):
    gen = torch.Generator().manual_seed(rows)
    w = _real_like_weight(gen)
    ref = _make_norm(monkeypatch, False, w)
    new = _make_norm(monkeypatch, True, w)
    assert new.rdna_residual_direct and new._rdna_scale_ready
    x = _real_like_act(gen, rows).cuda()
    r = _real_like_act(gen, rows).cuda()

    ref_out, ref_res = ref(x.clone(), r.clone())

    x_in, r_in = x.clone(), r.clone()
    out, res = new(x_in, r_in)
    assert out.data_ptr() == x_in.data_ptr()  # in place
    assert res.data_ptr() == r_in.data_ptr()
    assert torch.equal(out, ref_out)
    assert torch.equal(res, ref_res)


@requires_gfx10x
@torch.inference_mode()
def test_not_ready_falls_back_without_mutation(default_vllm_config, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with torch.device("cuda"):
        layer = GemmaRMSNorm(HIDDEN, eps=EPS).to(torch.float16)
    layer.weight.data.normal_(0.0, 0.3)  # a loader that bypasses weight_loader
    assert layer.rdna_residual_direct and not layer._rdna_scale_ready
    gen = torch.Generator().manual_seed(3)
    x = _real_like_act(gen, 4).cuda()
    r = _real_like_act(gen, 4).cuda()
    x0, r0 = x.clone(), r.clone()
    out, res = layer(x, r)
    assert torch.equal(x, x0) and torch.equal(r, r0)  # functional op: inputs kept
    ref = _make_norm(monkeypatch, False, layer.weight.data.cpu())
    ref_out, ref_res = ref(x0.clone(), r0.clone())
    assert torch.equal(out, ref_out) and torch.equal(res, ref_res)


@requires_gfx10x
@torch.inference_mode()
def test_weight_reload_refreshes_scale_in_place(default_vllm_config, monkeypatch):
    gen = torch.Generator().manual_seed(11)
    w_a, w_b = _real_like_weight(gen), _real_like_weight(gen)
    new = _make_norm(monkeypatch, True, w_a)
    ptr = new.rdna_scale16.data_ptr()  # a captured graph reads this address
    x = _real_like_act(gen, 3).cuda()
    r = _real_like_act(gen, 3).cuda()
    out_a, _ = new(x.clone(), r.clone())

    new.weight.weight_loader(new.weight, w_b)  # reload
    assert new.rdna_scale16.data_ptr() == ptr
    ref = _make_norm(monkeypatch, False, w_b)
    ref_out, ref_res = ref(x.clone(), r.clone())
    out_b, res_b = new(x.clone(), r.clone())
    assert torch.equal(out_b, ref_out) and torch.equal(res_b, ref_res)
    assert not torch.equal(out_a, out_b)


@requires_gfx10x
@pytest.mark.parametrize("layout", ["row_strided", "misaligned"])
@torch.inference_mode()
def test_off_fast_path_layouts_match(default_vllm_config, monkeypatch, layout):
    # Not contiguous, or not 16-byte aligned: the in-place kernel would pick a
    # different reduction width than the clones the functional path feeds it,
    # so these go through the functional op and are copied back.
    gen = torch.Generator().manual_seed(5)
    w = _real_like_weight(gen)
    ref = _make_norm(monkeypatch, False, w)
    new = _make_norm(monkeypatch, True, w)
    x_src = _real_like_act(gen, 6).cuda()
    r = _real_like_act(gen, 6).cuda()
    if layout == "row_strided":
        buf = torch.empty(6, 2 * HIDDEN, dtype=torch.float16, device="cuda")
        x_view = buf[:, :HIDDEN]
        assert not x_view.is_contiguous()
    else:
        buf = torch.empty(6 * HIDDEN + 1, dtype=torch.float16, device="cuda")
        x_view = buf[1:].view(6, HIDDEN)
        assert x_view.is_contiguous() and x_view.data_ptr() % 16 != 0
    x_view.copy_(x_src)
    ref_out, ref_res = ref(x_src.clone(), r.clone())
    r_in = r.clone()
    out, res = new(x_view, r_in)
    assert out.data_ptr() == x_view.data_ptr()  # still in place for the caller
    assert res.data_ptr() == r_in.data_ptr()
    assert torch.equal(out, ref_out) and torch.equal(res, ref_res)


@requires_gfx10x
@torch.inference_mode()
def test_3d_input_uses_functional_path(default_vllm_config, monkeypatch):
    gen = torch.Generator().manual_seed(9)
    w = _real_like_weight(gen)
    new = _make_norm(monkeypatch, True, w)
    x = _real_like_act(gen, 4).cuda().view(2, 2, HIDDEN)
    r = _real_like_act(gen, 4).cuda().view(2, 2, HIDDEN)
    x0, r0 = x.clone(), r.clone()
    out, res = new(x, r)
    assert torch.equal(x, x0) and torch.equal(r, r0)  # not mutated
    assert out.shape == x.shape and res.shape == r.shape


@requires_gfx10x
@torch.inference_mode()
def test_aliased_x_and_residual_raises(default_vllm_config, monkeypatch):
    gen = torch.Generator().manual_seed(2)
    new = _make_norm(monkeypatch, True, _real_like_weight(gen))
    x = _real_like_act(gen, 2).cuda()
    with pytest.raises(RuntimeError, match="alias"):
        new(x, x)


def _chain(layers_in, layers_post, final, mats, emb):
    """Mirror Qwen3NextDecoderLayer/Model: layer 0 norms without a residual and
    aliases it to the embedding, later calls add in place."""
    hidden, residual = emb, None
    for ln_in, ln_post, m in zip(layers_in, layers_post, mats):
        if residual is None:
            residual = hidden
            hidden = ln_in(hidden)
        else:
            hidden, residual = ln_in(hidden, residual)
        hidden = hidden @ m
        hidden, residual = ln_post(hidden, residual)
        hidden = hidden @ m
    hidden, _ = final(hidden, residual)
    return hidden


@requires_gfx10x
@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_decoder_chain_bit_exact_and_alias_behaviour(
    default_vllm_config, monkeypatch, compiled
):
    gen = torch.Generator().manual_seed(21)
    n_layers, rows = 3, 5
    ws = [_real_like_weight(gen) for _ in range(2 * n_layers + 1)]
    mats = [
        (torch.randn(HIDDEN, HIDDEN, generator=gen) * (HIDDEN**-0.5))
        .to(torch.float16)
        .cuda()
        for _ in range(n_layers)
    ]
    emb = _real_like_act(gen, rows).cuda()

    def build(direct):
        norms = [_make_norm(monkeypatch, direct, w) for w in ws]
        return norms[0:-1:2], norms[1::2][:n_layers], norms[-1]

    ref_in, ref_post, ref_final = build(False)
    ref_emb = emb.clone()
    ref = _chain(ref_in, ref_post, ref_final, mats, ref_emb)
    assert torch.equal(ref_emb, emb)  # functional path never touches the embedding

    new_in, new_post, new_final = build(True)
    fn = _chain
    if compiled:
        fn = torch.compile(_chain, backend="inductor", dynamic=False)
    new_emb = emb.clone()
    out = fn(new_in, new_post, new_final, mats, new_emb)
    assert torch.equal(out, ref)
    # Eager in-place: the embedding buffer is the residual stream afterwards,
    # which is why Qwen3NextModel.forward clones caller-owned inputs_embeds and
    # an aux-layer-0 capture before layer 0.
    if not compiled:
        assert not torch.equal(new_emb, emb)
