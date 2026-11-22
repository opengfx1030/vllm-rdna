# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini MoE block: reference equivalence (CPU) and kernel checks (GPU).

* CPU: our block with :class:`ExpertsReference` == transformers' reference
  ``Inkling2LoopSparseMoeBlock`` (fp32, <= 1e-5); the NVFP4 / MXFP8 repack
  helpers produce exactly the kernel layout.
* GPU (gfx1030, skipped without ``_rocm_C`` kernels): the kernel-backed blocks
  on random checkpoint-layout NVFP4 / MXFP8 experts vs. the fp32 reference block
  on torch-dequantised weights.

The production module and our config are loaded by file path so the test also
works where an installed ``vllm`` shadows the worktree package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


moe = _load("_i2_moe_under_test", "vllm/models/inkling2_loop/amd/moe.py")
ours_cfg = _load("_i2_cfg_under_test", "vllm/models/inkling2_loop/configs.py")

ref_cfg_mod = pytest.importorskip(
    "transformers.models.inkling2_loop.configuration_inkling2_loop"
)
ref_mod = pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop"
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def make_configs(hidden: int, inter: int, experts: int, top_k: int):
    kw = dict(
        hidden_size=hidden,
        n_routed_experts=experts,
        num_experts_per_tok=top_k,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    ours = ours_cfg.Inkling2LoopTextConfig(intermediate_size=inter, **kw)
    ref = ref_cfg_mod.Inkling2LoopTextConfig(moe_intermediate_size=inter, **kw)
    return ours, ref


# ---------------------------------------------------------------------------
# CPU: reference equivalence
# ---------------------------------------------------------------------------

CPU_CFGS = [(64, 32, 16, 4), (128, 64, 128, 8)]


def _randomise(block, gen: torch.Generator) -> None:
    """Deterministic non-trivial weights for any block exposing the reference
    state-dict keys."""
    with torch.no_grad():
        for name, p in block.named_parameters():
            p.copy_(
                torch.randn(p.shape, generator=gen, dtype=p.dtype)
                * (0.5 if "gate.weight" in name else 0.05)
            )
        # a spread-out selection bias so the expert load is uneven
        block.gate.e_score_correction_bias.copy_(
            torch.randn(block.gate.e_score_correction_bias.shape, generator=gen) * 0.5
        )


@pytest.mark.parametrize("hidden, inter, experts, top_k", CPU_CFGS)
@pytest.mark.parametrize("shape", [(1,), (3,), (37,), (2, 5)])
def test_block_matches_reference(hidden, inter, experts, top_k, shape):
    ours_c, ref_c = make_configs(hidden, inter, experts, top_k)
    ref = ref_mod.Inkling2LoopSparseMoeBlock(ref_c).eval()
    ours = moe.Inkling2LoopSparseMoeBlock(ours_c).eval()

    gen = torch.Generator().manual_seed(1234)
    _randomise(ref, gen)

    # identical state-dict keys, shapes and dtypes
    ref_sd, our_sd = ref.state_dict(), ours.state_dict()
    assert list(our_sd) == list(ref_sd)
    for k in ref_sd:
        assert our_sd[k].shape == ref_sd[k].shape, k
        assert our_sd[k].dtype == ref_sd[k].dtype, k
    ours.load_state_dict(ref_sd, strict=True)

    x = torch.randn(*shape, hidden, generator=gen)
    with torch.no_grad():
        want = ref(x if x.dim() == 3 else x.unsqueeze(0)).reshape(
            x.shape
        )  # ref needs [B, S, H]
        got = ours(x)
        # uneven expert loads (and idle experts for tiny batches) are exercised
        _, _, idx = ours.gate(x.reshape(-1, hidden))
        counts = torch.bincount(idx.flatten(), minlength=experts)
        if x.numel() // hidden > 1 and experts == 16:
            assert counts.max() != counts.min()
        if x.numel() // hidden <= 3 and experts == 128:
            assert (counts == 0).any()
    assert got.shape == want.shape and got.dtype == want.dtype
    err = (got - want).abs().max().item()
    assert err <= 1e-5, err


@pytest.mark.parametrize("hidden, inter, experts, top_k", CPU_CFGS)
def test_router_matches_reference(hidden, inter, experts, top_k):
    ours_c, ref_c = make_configs(hidden, inter, experts, top_k)
    ref = ref_mod.Inkling2LoopTopKRouter(ref_c)
    ours = moe.Inkling2LoopTopKRouter(ours_c)
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(ref.weight.shape, generator=gen) * 0.5)
        ref.e_score_correction_bias.copy_(torch.randn(experts, generator=gen))
    ours.load_state_dict(ref.state_dict(), strict=True)
    assert ours.weight.dtype == torch.float32
    assert ours.e_score_correction_bias.dtype == torch.float32
    x = torch.randn(19, hidden, generator=gen).to(torch.bfloat16)
    with torch.no_grad():
        r_logits, r_w, r_i = ref(x)
        o_logits, o_w, o_i = ours(x)
    # topk(sorted=False) order is unspecified: compare per-expert weights
    dense_r = torch.zeros(19, experts).scatter_(1, r_i, r_w)
    dense_o = torch.zeros(19, experts).scatter_(1, o_i, o_w)
    assert o_logits.dtype == torch.float32
    assert (o_logits - r_logits).abs().max().item() <= 1e-5
    assert (dense_o - dense_r).abs().max().item() <= 1e-5
    assert torch.equal(torch.sort(o_i, dim=1).values, torch.sort(r_i, dim=1).values)
    # selected, normalised sigmoid scores * route_scale sum to route_scale
    assert torch.allclose(o_w.sum(-1), torch.full((19,), 8.0), atol=1e-5)


def test_softcapped_swiglu_matches_reference():
    ours_c, ref_c = make_configs(64, 32, 16, 4)
    act = ref_mod.Inkling2LoopSoftcappedSwiGLU(ref_c)
    gen = torch.Generator().manual_seed(3)
    gate = torch.randn(11, 32, generator=gen) * 30  # exercises both softcaps
    up = torch.randn(11, 32, generator=gen) * 60
    for dt in (torch.float32, torch.bfloat16):
        want = act(gate.to(dt), up.to(dt))
        got = moe.softcapped_swiglu(gate.to(dt), up.to(dt), 4.0, 25.0)
        assert got.dtype == dt
        assert (got.float() - want.float()).abs().max().item() <= 1e-5


# ---------------------------------------------------------------------------
# Quantised-weight helpers (checkpoint layout) shared by CPU and GPU tests
# ---------------------------------------------------------------------------


def rand_nvfp4(num_e: int, n: int, k: int, gen: torch.Generator):
    """Random NVFP4 tensors in CHECKPOINT layout."""
    w = torch.randint(0, 256, (num_e, n, k // 2), dtype=torch.uint8, generator=gen)
    exp = torch.randint(5, 9, (num_e, n, k // 16), generator=gen)  # scales 2^-2 .. 2^1
    man = torch.randint(0, 8, (num_e, n, k // 16), generator=gen)
    s = ((exp << 3) | man).to(torch.uint8)
    amax = 10.0 + 10.0 * torch.rand(num_e, generator=gen)
    den = torch.tensor([2688.0])
    return w, s, amax, den


def dequant_nvfp4_ckpt(w, s, amax, den) -> torch.Tensor:
    """Value = e2m1 * e4m3 * (amax / den), block 16 along K -> fp32 [E, N, K]."""
    num_e, n, kd2 = w.shape
    wi = w.to(torch.int64)
    nib = torch.stack((wi & 0xF, wi >> 4), dim=-1).reshape(
        num_e, n, kd2 * 2
    )  # low = even k
    mag = E2M1.to(w.device)[nib & 0x7]
    val = torch.where(nib & 0x8 != 0, -mag, mag)
    sc = s.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=-1)
    glob = (amax.float() / den.float()).view(num_e, 1, 1)
    return val * sc * glob


def rand_mxfp8(num_e: int, n: int, k: int, gen: torch.Generator):
    sign = torch.randint(0, 2, (num_e, n, k), generator=gen)
    exp = torch.randint(3, 10, (num_e, n, k), generator=gen)
    man = torch.randint(0, 8, (num_e, n, k), generator=gen)
    w = ((sign << 7) | (exp << 3) | man).to(torch.uint8)  # never the NaN code
    s = (121 + torch.randint(0, 3, (num_e, n, k // 32), generator=gen)).to(torch.uint8)
    return w, s


def dequant_mxfp8_ckpt(w, s) -> torch.Tensor:
    """Value = e4m3 * 2^(ue8m0 - 127), block 32 along K -> fp32 [E, N, K]."""
    val = w.view(torch.float8_e4m3fn).to(torch.float32)
    sc = torch.pow(2.0, s.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    return val * sc


def kernel_layout_dequant_nvfp4(q, s, g) -> torch.Tensor:
    """Dequantise the repacked tensors as the kernel reads them -> [E, K, N]."""
    words = q.view(torch.int32).to(torch.int64) & 0xFFFFFFFF  # [E, K/8, N]
    num_e, k8, n = words.shape
    vals = torch.empty(num_e, k8 * 8, n)
    for i in range(8):
        nib = (words >> (4 * i)) & 0xF
        mag = E2M1[nib & 0x7]
        vals[:, i::8, :] = torch.where(nib & 0x8 != 0, -mag, mag)
    sc = s.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=1)
    return vals * sc * g.view(num_e, 1, 1)


def kernel_layout_dequant_mxfp8(q, s) -> torch.Tensor:
    words = q.view(torch.int32).to(torch.int64) & 0xFFFFFFFF  # [E, K/4, N]
    num_e, k4, n = words.shape
    vals = torch.empty(num_e, k4 * 4, n)
    for i in range(4):
        byte = ((words >> (8 * i)) & 0xFF).to(torch.uint8)
        vals[:, i::4, :] = byte.view(torch.float8_e4m3fn).to(torch.float32)
    sc = torch.pow(2.0, s.to(torch.float32) - 127.0).repeat_interleave(32, dim=1)
    return vals * sc


# ---------------------------------------------------------------------------
# CPU: repack layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_e, n, k", [(3, 16, 64), (2, 24, 96)])
def test_repack_nvfp4_layout(num_e, n, k):
    gen = torch.Generator().manual_seed(11)
    w, s, amax, den = rand_nvfp4(num_e, n, k, gen)
    q, bs, bg = moe.repack_nvfp4(w, s, amax, den)
    assert q.dtype == torch.uint32 and tuple(q.shape) == (num_e, k // 8, n)
    assert bs.dtype == torch.uint8 and tuple(bs.shape) == (num_e, k // 16, n)
    assert bg.dtype == torch.float32 and tuple(bg.shape) == (num_e,)
    assert torch.equal(bg, amax / den)
    want = dequant_nvfp4_ckpt(w, s, amax, den).transpose(1, 2)  # [E, K, N]
    assert torch.equal(kernel_layout_dequant_nvfp4(q, bs, bg), want)
    # float8 scale tensors are accepted and give identical bytes
    q2, bs2, _ = moe.repack_nvfp4(w, s.view(torch.float8_e4m3fn), amax, den)
    assert torch.equal(q2.view(torch.int32), q.view(torch.int32)) and torch.equal(
        bs2, bs
    )


@pytest.mark.parametrize("num_e, n, k", [(3, 16, 64), (2, 24, 96)])
def test_repack_mxfp8_layout(num_e, n, k):
    gen = torch.Generator().manual_seed(12)
    w, s = rand_mxfp8(num_e, n, k, gen)
    q, bs = moe.repack_mxfp8(w, s)
    assert q.dtype == torch.uint32 and tuple(q.shape) == (num_e, k // 4, n)
    assert bs.dtype == torch.uint8 and tuple(bs.shape) == (num_e, k // 32, n)
    want = dequant_mxfp8_ckpt(w, s).transpose(1, 2)
    assert torch.equal(kernel_layout_dequant_mxfp8(q, bs), want)
    q2, bs2 = moe.repack_mxfp8(w.view(torch.float8_e4m3fn), s)
    assert torch.equal(q2.view(torch.int32), q.view(torch.int32)) and torch.equal(
        bs2, bs
    )


def test_kernel_experts_load_checkpoint_cpu():
    """``load_checkpoint`` fills the buffers with exactly the repacked tensors."""
    ours_c, _ = make_configs(64, 32, 4, 2)
    gen = torch.Generator().manual_seed(13)
    w13 = rand_nvfp4(4, 64, 64, gen)
    w2 = rand_nvfp4(4, 64, 32, gen)
    experts = moe.ExpertsNvfp4(ours_c)
    experts.load_checkpoint(*w13, *w2)
    q, s, g = moe.repack_nvfp4(*w13)
    assert torch.equal(experts.w13_qweight, q.view(torch.int32))
    assert torch.equal(experts.w13_scales, s) and torch.equal(experts.w13_globals, g)
    q, s, g = moe.repack_nvfp4(*w2)
    assert torch.equal(experts.w2_qweight, q.view(torch.int32))

    m13 = rand_mxfp8(4, 64, 64, gen)
    m2 = rand_mxfp8(4, 64, 32, gen)
    experts8 = moe.ExpertsMxfp8(ours_c)
    experts8.load_checkpoint(*m13, *m2)
    q, s = moe.repack_mxfp8(*m13)
    assert torch.equal(experts8.w13_qweight, q.view(torch.int32))
    assert torch.equal(experts8.w13_scales, s)
    assert set(dict(experts8.named_buffers())) == {
        "w13_qweight",
        "w13_scales",
        "w2_qweight",
        "w2_scales",
    }


def test_block_size_m_heuristic():
    f = moe._block_size_m
    assert f(1, 8, 128) == 1
    assert f(16, 8, 128) == 1
    assert f(32, 8, 128) == 2
    assert f(64, 8, 128) == 4
    assert f(1024, 8, 128) == 8


# ---------------------------------------------------------------------------
# GPU: kernel-backed experts vs fp32 reference on dequantised weights
# ---------------------------------------------------------------------------


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import vllm._custom_ops  # noqa: F401  (loads _rocm_C)
    except Exception:
        return False
    rocm = getattr(torch.ops, "_rocm_C", None)
    return rocm is not None and all(
        hasattr(rocm, n) for n in ("moe_nvfp4_gemm_rdna2", "moe_mxfp8_gemm_rdna2")
    )


gpu_only = pytest.mark.skipif(
    not _gpu_ready(), reason="needs gfx1030 with moe_nvfp4/mxfp8_gemm_rdna2 kernels"
)

# (H, I, E, top_k); the last two are the per-rank intermediate sizes of the real
# checkpoint at TP=2 and TP=4 (1152 / 2, 1152 / 4: K tiles of 256 with a 64 / 32 tail)
GPU_CFGS = [
    (256, 128, 8, 2),
    (2304, 1152, 16, 8),
    (2304, 576, 16, 8),
    (2304, 288, 16, 8),
]
TOKENS = [1, 4, 8, 64]


def _gpu_blocks(fmt: str, cfg, seed: int):
    """(kernel block, fp32 reference block with dequantised weights), on GPU."""
    hidden, inter, experts, top_k = cfg
    ours_c, _ = make_configs(hidden, inter, experts, top_k)
    gen = torch.Generator().manual_seed(seed)
    if fmt == "nvfp4":
        w13 = rand_nvfp4(experts, 2 * inter, hidden, gen)
        w2 = rand_nvfp4(experts, hidden, inter, gen)
        dq13, dq2 = dequant_nvfp4_ckpt(*w13), dequant_nvfp4_ckpt(*w2)
        kernel_cls = moe.ExpertsNvfp4
    else:
        w13 = rand_mxfp8(experts, 2 * inter, hidden, gen)
        w2 = rand_mxfp8(experts, hidden, inter, gen)
        dq13, dq2 = dequant_mxfp8_ckpt(*w13), dequant_mxfp8_ckpt(*w2)
        kernel_cls = moe.ExpertsMxfp8

    router_w = torch.randn(experts, hidden, generator=gen) * 0.05
    router_b = torch.randn(experts, generator=gen) * 0.3

    def set_router(block):
        with torch.no_grad():
            block.gate.weight.copy_(router_w)
            block.gate.e_score_correction_bias.copy_(router_b)

    kern = moe.Inkling2LoopSparseMoeBlock(ours_c, kernel_cls, device="cuda").cuda()
    kern.experts.load_checkpoint(*[t.cuda() for t in (*w13, *w2)])
    set_router(kern)

    ref = moe.Inkling2LoopSparseMoeBlock(
        ours_c, moe.ExpertsReference, device="cuda"
    ).cuda()
    with torch.no_grad():
        ref.experts.gate_up_proj.copy_(dq13)
        ref.experts.down_proj.copy_(dq2)
    set_router(ref)
    return kern.eval(), ref.eval(), hidden, top_k, experts


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).abs().max() / want.abs().max()).item()


def _check(
    fmt: str,
    cfg,
    m: int,
    tol: float,
    *,
    chunk: int | None = None,
    block_size_m: int | None = None,
) -> float:
    kern, ref, hidden, top_k, experts = _gpu_blocks(fmt, cfg, seed=21)
    if chunk is not None:
        kern.experts.max_tokens_per_call = chunk
    gen = torch.Generator().manual_seed(100 + m)
    x = torch.randn(m, hidden, generator=gen).cuda().to(torch.float16)
    with torch.no_grad():
        want = ref(x.float())  # same routing input values as the fp16 kernel path
        if block_size_m is None:
            got = kern(x)
        else:
            _, w, idx = kern.gate(x)
            got = kern.experts(x, idx, w, block_size_m=block_size_m)
    torch.accelerator.synchronize()
    assert got.dtype == torch.float16 and got.shape == x.shape
    err = _rel_err(got, want)
    print(f"{fmt} cfg={cfg} M={m} chunk={chunk} bs={block_size_m}: rel err {err:.3e}")
    assert err <= tol, err
    return err


@gpu_only
@pytest.mark.parametrize("cfg", GPU_CFGS)
@pytest.mark.parametrize("m", TOKENS)
def test_nvfp4_kernel_block_matches_dequant_reference(cfg, m):
    _check("nvfp4", cfg, m, 3e-2)


@gpu_only
@pytest.mark.parametrize("cfg", GPU_CFGS)
@pytest.mark.parametrize("m", TOKENS)
def test_mxfp8_kernel_block_matches_dequant_reference(cfg, m):
    _check("mxfp8", cfg, m, 1e-2)


@gpu_only
@pytest.mark.parametrize("fmt, tol", [("nvfp4", 3e-2), ("mxfp8", 1e-2)])
@pytest.mark.parametrize("block_size_m", [1, 2, 4, 8])
def test_kernel_block_size_m(fmt, tol, block_size_m):
    _check(fmt, GPU_CFGS[0], 8, tol, block_size_m=block_size_m)


@gpu_only
@pytest.mark.parametrize("fmt, tol", [("nvfp4", 3e-2), ("mxfp8", 1e-2)])
def test_kernel_token_chunking(fmt, tol):
    """64 tokens through 24-token chunks (last chunk is partial)."""
    _check(fmt, GPU_CFGS[0], 64, tol, chunk=24)
