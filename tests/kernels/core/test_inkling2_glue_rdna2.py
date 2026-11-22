# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the Inkling-2-Mini glue kernels (csrc/rocm/inkling2_glue_rdna2.cu).

Every op is compared against an fp64 reference written from the formulas of
``vllm/models/inkling2_loop/amd/blocks.py`` / ``moe.py`` on ragged token counts,
strided views, tiny and huge activations and softmax / softcap saturation.  The
whole layer-visit sequences are additionally compared against the torch modules
of ``blocks.py`` themselves.

Tolerances.  fp32 outputs agree to ~1e-5 (only the order of the fp32 reductions
differs).  An fp16 output is compared with the correctly rounded fp64 value in
two ways: every element must be within one fp16 ulp of it plus an absolute floor
(``check_f16`` ``atol``), and at most ``max_mismatch`` of the elements may
differ from it.  The floor exists because the error of an fp32 pipeline does
not shrink with the magnitude of an individual output element: an element near
zero can be many of *its own* ulps off while being 1e-7 off in absolute terms.
The mismatch fraction is what makes a systematic error (a truncating instead of
a round-to-nearest conversion, a 1e-4 relative bias) visible even though the
floor and one ulp are both loose.  ``GLUE_TEST_STATS=1`` prints the measured
errors of every comparison (used to calibrate the bounds).
"""

import os
import types
import zlib

import pytest
import torch

from vllm import _custom_ops as _vllm_ops
from vllm.models.inkling2_loop.amd import blocks
from vllm.platforms.rocm import on_gfx10x

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and on_gfx10x()
        and hasattr(torch.ops._rocm_C, "inkling2_glue_attn_res")
    ),
    reason="needs gfx1030 and a build with csrc/rocm/inkling2_glue_rdna2.cu",
)


class _JitGlueOps:
    """``I2_GLUE_JIT=<path to a .cu>``: the ops the decode-latency variants change
    (gate_down, gate_mix, moe_route) run from a JIT build of that source -- this is how
    the mutants of the kernels are checked (``tools/rdna/inkling2/glue_jit.py``); every
    other attribute is the installed build's.  ``VLLM_INKLING2_FUSE_GLUE`` selects the
    variants inside the build."""

    def __init__(self, source: str):
        import hashlib
        import pathlib
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
        from rdna.inkling2 import glue_jit

        tag = hashlib.md5(pathlib.Path(source).read_bytes()).hexdigest()[:10]
        self._mod = glue_jit.build("i2_glue_test_" + tag, source)

    def inkling2_glue_msg_gate_down(self, h, m, wd, d, eps=1e-6):
        self._mod.msg_gate_down(h, m, wd, d, eps)

    def inkling2_glue_msg_gate_mix(
        self,
        h,
        m,
        d,
        wf,
        wg,
        out,
        *,
        norm_w=None,
        scratch=None,
        partials=None,
        counters=None,
        norm_eps=1e-6,
    ):
        self._mod.msg_gate_mix(
            h, m, d, wf, wg, norm_w, out, scratch, partials, counters, norm_eps
        )

    def inkling2_glue_moe_route(
        self,
        x,
        gate_w,
        bias,
        logits,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        post_pad,
        counters,
        block_size_m,
        route_scale,
        *,
        zero1=None,
        zero2=None,
    ):
        self._mod.moe_route(
            x,
            gate_w,
            bias,
            logits,
            topk_weights,
            topk_ids,
            sorted_ids,
            expert_ids,
            post_pad,
            counters,
            zero1,
            zero2,
            block_size_m,
            route_scale,
        )

    def __getattr__(self, name):
        return getattr(_vllm_ops, name)


ops = (
    _JitGlueOps(os.environ["I2_GLUE_JIT"])
    if os.environ.get("I2_GLUE_JIT")
    else _vllm_ops
)

DEV = "cuda"
EPS = 1e-6
PAD = 8
PAD_FILL = 777.0  # exactly representable in fp16


@pytest.fixture(autouse=True)
def _inference_mode():
    # the ops have no autograd formulas; module parameters used as inputs must
    # not require grad
    with torch.inference_mode():
        yield


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _seed(name: str) -> int:
    return zlib.crc32(name.encode())


def _randn(g, *shape, mag=1.0, dtype=torch.float32):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * mag).to(dtype)


def _padded(t: torch.Tensor):
    """``t`` as a GPU view inside rows that are PAD elements wider.

    Returns ``(view, backing)``; the padding holds PAD_FILL so out-of-bounds
    writes are detected with ``_pad_intact``.
    """
    *lead, last = t.shape
    buf = torch.full((*lead, last + PAD), PAD_FILL, dtype=t.dtype)
    buf[..., :last] = t
    buf = buf.to(DEV)
    return buf[..., :last], buf


def _pad_intact(buf: torch.Tensor, last: int) -> bool:
    return bool((buf[..., last:] == PAD_FILL).all())


def _f16_ulp(ref: torch.Tensor) -> torch.Tensor:
    """fp16 spacing at ``ref`` (fp64), including the subnormal range."""
    _, e = torch.frexp(ref.abs().clamp_min(2.0**-14))
    return torch.ldexp(torch.ones_like(ref), e - 11)


STATS = os.environ.get("GLUE_TEST_STATS") == "1"
# floor of an fp32 pipeline over ~2k terms, in fp32 ulps of the largest output
F32_PIPELINE_ULPS = 16
# fp16 rounding of the projection input flips about one of its 4608 elements per
# row between the kernel and the fp64 reference; d moves by |w| * ulp(x) <= 2e-4
D_ATOL = 5e-4
# a flipped fp16 element of d (one step, <= 2e-3) moves a gate logit by about
# |w| * 2e-3 ~ 6e-5 and a gated, normalised output by up to ~1e-3: one fp16 ulp
# at unit scale, allowed on top of the one ulp bound in the visit tests
VISIT_ATOL = 2e-3


def check_f16(got, ref, what, *, atol=None, max_mismatch=5e-3):
    """``got`` (fp16) vs the fp64 ``ref``.

    Args:
        got: kernel output, fp16.
        ref: reference, any float dtype (compared as fp64).
        what: label of the comparison for the failure message.
        atol: absolute floor added to the one fp16 ulp bound; defaults to
            ``F32_PIPELINE_ULPS`` fp32 ulps of ``max |ref|``.
        max_mismatch: largest fraction of elements allowed to differ from
            ``fp16(ref)``.

    """
    assert got.dtype == torch.float16, what
    got = got.detach().cpu().double()
    ref = ref.detach().cpu().double()
    assert got.shape == ref.shape, (what, got.shape, ref.shape)
    assert torch.isfinite(got).all(), f"{what}: non finite output"
    if atol is None:
        atol = (
            F32_PIPELINE_ULPS
            * 2.0**-24
            * (ref.abs().max().item() if ref.numel() else 0.0)
        )
    err = (got - ref.to(torch.float16).double()).abs()
    # a neighbour across a binade boundary is one ulp of the larger binade
    excess = err - torch.maximum(_f16_ulp(ref), _f16_ulp(got))
    worst = excess.max().item() if err.numel() else 0.0
    frac = (err > 0).double().mean().item() if err.numel() else 0.0
    if STATS:
        print(
            f"[stat] {what}: max|err| {err.max().item() if err.numel() else 0:.3e} "
            f"excess over 1ulp {worst:.3e} (atol {atol:.3e}) mismatch {frac:.4%} "
            f"(limit {max_mismatch:.2%}) ref max {ref.abs().max().item():.3e}",
            flush=True,
        )
    assert worst <= atol, f"{what}: {worst:.3e} beyond one fp16 ulp (atol {atol:.1e})"
    assert frac <= max_mismatch, f"{what}: {frac:.4%} elements differ from fp16(ref)"


def check_f32(got, ref, what, rtol=2e-5, atol=2e-5):
    got = got.detach().cpu().double()
    ref = ref.detach().cpu().double()
    assert got.shape == ref.shape, (what, got.shape, ref.shape)
    assert torch.isfinite(got).all(), f"{what}: non finite output"
    bound = atol + rtol * ref.abs()
    bad = (got - ref).abs() - bound
    if STATS:
        print(
            f"[stat] {what}: max|err| {(got - ref).abs().max().item():.3e} "
            f"(bound atol {atol:.1e} rtol {rtol:.1e}) "
            f"ref max {ref.abs().max().item():.3e}",
            flush=True,
        )
    assert bool((bad <= 0).all()), (
        f"{what}: max |err| {(got - ref).abs().max().item():.3e} "
        f"(ref max {ref.abs().max().item():.3e}), "
        f"{int((bad > 0).sum())} of {ref.numel()} outside tolerance"
    )


def _unw_norm(x, eps=EPS):
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _w_norm(x, w, eps=EPS):
    """``Inkling2RMSNorm`` in fp64 (weight applied after the normalisation)."""
    return w.double() * _unw_norm(x, eps)


# ---------------------------------------------------------------------------
# attn_res
# ---------------------------------------------------------------------------
def ref_attn_res(blocks_, nb, x0, x1, append, query, eps, norm_w, norm_eps):
    states = [blocks_[:, j].double() for j in range(nb)]
    partial = x0.double() + (x1.double() if x1 is not None else 0.0)
    if append is not None:
        states.append(append.double())
    states.append(partial)
    normed = _unw_norm(torch.stack(states, 1), eps)  # [T, NS, H]
    scores = torch.einsum("qh,tsh->tqs", query.double(), normed)
    scores = scores * normed.shape[-1] ** -0.5
    out = torch.einsum("tqs,tsh->tqh", torch.softmax(scores, -1), normed)
    if norm_w is not None:
        out = _w_norm(out, norm_w, norm_eps)
    return out, partial


def _case(**kw):
    base = dict(
        T=1, H=2304, Q=1, nb=3, x1=False, popt=False, app=False, norm=None,
        rows=1, mag=1.0, qmag=1.0, strided=False, eps=EPS,
    )  # fmt: skip
    base.update(kw)
    return base


ATTN_RES_CASES = {
    # sliding-layer prologue with the lazy residual add and fused layer norm
    "sliding_prologue_t1": _case(x1=True, popt=True, norm="f16"),
    "sliding_prologue_t17": _case(T=17, nb=5, x1=True, popt=True, norm="f16"),
    # full layers: three queries, three layer-norm weight rows
    "full_prologue_t1": _case(Q=3, x1=True, popt=True, norm="f16", rows=3),
    "full_prologue_t3_f32w": _case(T=3, Q=3, nb=4, norm="f32", rows=3),
    "full_prologue_t1024": _case(
        T=1024, Q=3, nb=5, x1=True, popt=True, norm="f16", rows=3
    ),
    # mlp_input of a block boundary layer: the layer input is appended
    "boundary_mid_t1": _case(nb=2, x1=True, popt=True, app=True, norm="f16"),
    "boundary_mid_t17": _case(T=17, nb=4, app=True, popt=True, norm="f16"),
    # no stored states at all (layer 1 prologue)
    "first_state_only": _case(nb=0, norm="f16"),
    "first_state_append": _case(T=3, nb=0, app=True, popt=True, norm="f32"),
    # message residual (queries q, k, v, mlp): raw fp32 output
    "messages_t1": _case(Q=4, nb=1, x1=True),
    "messages_t17": _case(T=17, Q=4, nb=2, x1=True),
    "messages_t1024": _case(T=1024, Q=4, nb=1, x1=True),
    # raw fp32 outputs of the other query counts
    "raw_q1": _case(T=3, Q=1, nb=4),
    "raw_q3": _case(T=17, Q=3, nb=4, x1=True),
    # state counts around the register-resident limit and the maximum
    "many_states_48": _case(T=3, H=256, Q=3, nb=48, x1=True, norm="f16", rows=3),
    "max_states_64": _case(T=2, H=256, Q=4, nb=62, app=True, x1=True),
    "six_states_boundary": _case(T=5, H=256, Q=1, nb=5, norm="f16"),
    # hidden sizes: tiny model, ragged vector counts, one / four float4 per thread
    "tiny_h256": _case(T=17, H=256, Q=3, nb=2, x1=True, norm="f16", rows=3),
    "ragged_h100": _case(T=5, H=100, Q=1, nb=3, norm="f16"),
    "ragged_h36": _case(T=3, H=36, Q=4, nb=2),
    "h3072": _case(T=3, H=3072, Q=3, nb=4, norm="f16", rows=3),
    "h4096": _case(T=3, H=4096, Q=3, nb=4, norm="f16", rows=3),
    # tiny activations (eps dominates), huge activations, saturated softmax
    "tiny_activations": _case(T=17, Q=3, nb=3, x1=True, norm="f16", mag=1e-3),
    "huge_activations": _case(T=17, Q=3, nb=3, x1=True, norm="f16", mag=1e15),
    "saturated_softmax": _case(T=17, Q=4, nb=4, qmag=60.0),
    "saturated_softmax_norm": _case(T=3, Q=3, nb=4, qmag=60.0, norm="f16", rows=3),
    "other_eps": _case(T=5, Q=1, nb=2, mag=1e-3, eps=1e-4, norm="f16"),
    # strided rows / slots
    "strided_all": _case(
        T=17, Q=3, nb=3, x1=True, popt=True, app=True, norm="f16", rows=3,
        strided=True,
    ),
    "strided_raw": _case(T=17, Q=4, nb=2, x1=True, strided=True),
}  # fmt: skip


@pytest.mark.parametrize("name", list(ATTN_RES_CASES))
def test_attn_res(name):
    c = ATTN_RES_CASES[name]
    T, H, Q, nb = c["T"], c["H"], c["Q"], c["nb"]
    g = _gen(_seed(name))
    S = nb + (1 if c["app"] else 0) + 2
    mag = c["mag"]
    # states of different magnitude, like the real block states
    scale = (0.5 + 1.5 * torch.rand(1, S, 1, generator=g)) * mag
    blocks_c = _randn(g, T, S, H) * scale
    x0_c = _randn(g, T, H, mag=mag)
    x1_c = _randn(g, T, H, mag=mag) if c["x1"] else None
    app_c = _randn(g, T, H, mag=mag) if c["app"] else None
    query_c = _randn(g, Q, H, mag=c["qmag"])
    nw_c: torch.Tensor | None = None
    if c["norm"]:
        nw_c = 1.0 + 0.3 * _randn(g, c["rows"], H)
        if c["norm"] == "f16":
            nw_c = nw_c.to(torch.float16)

    if nb > 0 or c["app"]:
        if c["strided"]:
            blocks_g, blocks_buf = _padded(blocks_c)
        else:
            blocks_g, blocks_buf = blocks_c.to(DEV), None
    else:
        blocks_g, blocks_buf = torch.empty(0, device=DEV), None
    blocks_before = blocks_g.clone()
    x0_g, x0_buf = _padded(x0_c) if c["strided"] else (x0_c.to(DEV), None)
    x1_g = x1_c.to(DEV) if x1_c is not None else None
    app_g = app_c.to(DEV) if app_c is not None else None
    query_g = query_c.to(DEV)
    nw_g = nw_c.to(DEV) if nw_c is not None else None
    out_dtype = torch.float16 if c["norm"] else torch.float32
    out_c = torch.full((T, Q, H), PAD_FILL, dtype=out_dtype)
    out_g, out_buf = _padded(out_c) if c["strided"] else (out_c.to(DEV), None)
    po_g = torch.full((T, H), -1.0, device=DEV) if c["popt"] else None

    ops.inkling2_glue_attn_res(
        blocks_g, nb, x0_g, query_g, out_g, x1=x1_g, partial_out=po_g,
        append=app_g, norm_w=nw_g, eps=c["eps"], norm_eps=EPS,
    )  # fmt: skip
    torch.accelerator.synchronize()

    ref, partial = ref_attn_res(
        blocks_c, nb, x0_c, x1_c, app_c, query_c, c["eps"], nw_c, EPS
    )
    if c["norm"]:
        check_f16(out_g, ref, name)
    else:
        check_f32(out_g, ref, name, rtol=2e-5, atol=2e-5)
    if po_g is not None:
        assert torch.equal(po_g.cpu(), partial.float()), "partial_out"
    if nb > 0 or c["app"]:
        expect = blocks_before.clone()
        if c["app"]:
            expect[:, nb] = app_g
        assert torch.equal(blocks_g, expect), "block states modified"
    if blocks_buf is not None:
        assert _pad_intact(blocks_buf, H), "blocks padding overwritten"
    if x0_buf is not None:
        assert _pad_intact(x0_buf, H), "x0 padding overwritten"
    if out_buf is not None:
        assert _pad_intact(out_buf, H), "out padding overwritten"


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("Q", [3, 4])
def test_attn_res_stream_major_out(norm, Q):
    """``out`` may be the [T, Q, H] view of a stream-major [Q, T, H] buffer."""
    g = _gen(51 + Q)
    T, H, nb = 17, 2304, 3
    blocks_g = _randn(g, T, nb, H).to(DEV)
    x0, x1 = _randn(g, T, H).to(DEV), _randn(g, T, H).to(DEV)
    query = _randn(g, Q, H).to(DEV)
    kw = {}
    dt = torch.float32
    if norm:
        kw["norm_w"] = (1.0 + 0.3 * _randn(g, Q, H)).to(torch.float16).to(DEV)
        dt = torch.float16
    ref = torch.empty(T, Q, H, dtype=dt, device=DEV)
    ops.inkling2_glue_attn_res(blocks_g, nb, x0, query, ref, x1=x1, **kw)
    sm = torch.full((Q, T, H), -5.0, dtype=dt, device=DEV)
    out = sm.permute(1, 0, 2)
    ops.inkling2_glue_attn_res(blocks_g, nb, x0, query, out, x1=x1, **kw)
    torch.accelerator.synchronize()
    assert torch.equal(out, ref)


def test_attn_res_rejects_bad_arguments():
    g = _gen(1)
    T, H = 3, 256
    x0 = _randn(g, T, H).to(DEV)
    q1 = _randn(g, 1, H).to(DEV)
    out = torch.empty(T, 1, H, device=DEV)
    blk = _randn(g, T, 4, H).to(DEV)
    f = ops.inkling2_glue_attn_res
    with pytest.raises(RuntimeError, match="Q must be"):
        f(blk, 2, x0, _randn(g, 2, H).to(DEV), torch.empty(T, 2, H, device=DEV))
    with pytest.raises(RuntimeError, match="states exceed"):
        f(torch.empty(T, 70, H, device=DEV), 70, x0, q1, out)
    with pytest.raises(RuntimeError, match="unit stride"):
        f(blk, 2, x0, q1, torch.empty(T, 1, 2 * H, device=DEV)[:, :, ::2])
    with pytest.raises(RuntimeError, match="must be"):
        f(blk, 2, x0.half(), q1, out)
    with pytest.raises(RuntimeError, match="multiples of 4"):
        f(blk, 2, torch.empty(T, H + 2, device=DEV)[:, 2:], q1, out)
    with pytest.raises(RuntimeError, match="must be torch.float16|must be Half"):
        f(blk, 2, x0, q1, out, norm_w=torch.ones(1, H, device=DEV))


# ---------------------------------------------------------------------------
# message gate
# ---------------------------------------------------------------------------
def _gate_weights(g, G, R, H, wscale=(0.02, 0.05)):
    """Module-layout weights: down [G, R, 2H], forget / gain [G, H, R]."""
    wd = _randn(g, G, R, 2 * H, mag=wscale[0]).to(torch.float16)
    wf = _randn(g, G, H, R, mag=wscale[1])
    wg = _randn(g, G, H, R, mag=wscale[1])
    return wd, wf, wg


def _pack_gate(w_mod: torch.Tensor, fp16: bool) -> torch.Tensor:
    """Kernel layout of forget / gain weights from ``nn.Linear`` ``[G, H, R]``."""
    G, H, R = w_mod.shape
    if not fp16:
        return w_mod.transpose(1, 2).contiguous().to(DEV)  # [G, R, H] fp32
    w = w_mod.to(torch.float16).reshape(G, H, R // 2, 2)  # [G, H, R/2, 2]
    return w.permute(0, 2, 1, 3).contiguous().to(DEV)  # [G, R/2, H, 2]


def ref_gate_down(h, m, wd, eps=EPS):
    G = m.shape[1]
    hh = h.double().expand(-1, G, -1)
    x = torch.cat([_unw_norm(hh, eps), _unw_norm(m, eps)], -1)  # [T, G, 2H]
    x = x.to(torch.float16).double()  # the projection input is fp16
    return torch.einsum("tgk,grk->gtr", x, wd.double())  # [G, T, R]


def ref_gate_mix(h, m, d, wf_eff, wg_eff, norm_w, norm_eps=EPS):
    G = m.shape[1]
    hh = h.double().expand(-1, G, -1).transpose(0, 1)  # [G, T, H]
    mm = m.double().transpose(0, 1)
    f = torch.tanh(torch.einsum("gtr,ghr->gth", d.double(), wf_eff))
    gg = torch.tanh(torch.einsum("gtr,ghr->gth", d.double(), wg_eff))
    out = (1.0 - f) * hh + gg * mm
    if norm_w is not None:
        out = _w_norm(out, norm_w.double().unsqueeze(1), norm_eps)  # [G, 1, H]
    return out


def _gate_case(**kw):
    base = dict(
        T=1, G=3, SH=3, R=64, H=2304, w16=True, norm=True, nwdt="f16", rows=3,
        mag=1.0, strided=False,
    )  # fmt: skip
    base.update(kw)
    return base


GATE_CASES = {
    "qkv_full_t1": _gate_case(),
    "qkv_full_t1_f32w": _gate_case(w16=False),
    "qkv_sliding_t1": _gate_case(SH=1, rows=1),
    "mlp_t1": _gate_case(G=1, SH=1, rows=1),
    "qkv_full_t3": _gate_case(T=3),
    "qkv_full_t3_f32w": _gate_case(T=3, w16=False),
    "qkv_full_t2_f32w": _gate_case(T=2, w16=False),
    "qkv_full_t4_f32w": _gate_case(T=4, w16=False),
    "qkv_sliding_t4_f32w": _gate_case(T=4, SH=1, rows=1, w16=False),
    "mlp_t1_f32w": _gate_case(G=1, SH=1, rows=1, w16=False),
    "mlp_t4_f32w": _gate_case(T=4, G=1, SH=1, rows=1, w16=False),
    "raw_fp32_out_t3_f32w": _gate_case(T=3, norm=False, w16=False),
    "raw_fp32_out_t1_f32w": _gate_case(T=1, norm=False, w16=False),
    "qkv_full_t5_f32w": _gate_case(T=5, w16=False),
    "qkv_full_t4_f32w_huge": _gate_case(T=4, w16=False, mag=1e8),
    "qkv_full_t4_f32w_tiny": _gate_case(T=4, w16=False, mag=1e-3),
    "mlp_t1_f32w_tiny": _gate_case(G=1, SH=1, rows=1, w16=False, mag=1e-3),
    "qkv_sliding_t17": _gate_case(T=17, SH=1, rows=1),
    "qkv_full_t17_f32w": _gate_case(T=17, w16=False),
    "mlp_t17": _gate_case(T=17, G=1, SH=1, rows=1),
    "mlp_t17_f32w": _gate_case(T=17, G=1, SH=1, rows=1, w16=False),
    "qkv_full_t1024": _gate_case(T=1024),
    "mlp_t1024_f32w": _gate_case(T=1024, G=1, SH=1, rows=1, w16=False),
    "qkv_full_t64": _gate_case(T=64),
    "qkv_full_t100": _gate_case(T=100),
    "qkv_full_t100_f32w": _gate_case(T=100, w16=False),
    "mlp_t100": _gate_case(T=100, G=1, SH=1, rows=1),
    "mlp_t300": _gate_case(T=300, G=1, SH=1, rows=1),
    "raw_fp32_out": _gate_case(T=17, norm=False),
    "raw_fp32_out_f32w": _gate_case(T=17, norm=False, w16=False),
    "tiny_h256_r16": _gate_case(T=17, R=16, H=256),
    "tiny_h256_r16_f32w": _gate_case(T=3, R=16, H=256, w16=False),
    "ragged_h100": _gate_case(T=5, R=16, H=100),
    "odd_chunks_h2432": _gate_case(T=17, R=16, H=2432),
    "odd_chunks_h2432_f32w": _gate_case(T=5, R=16, H=2432, w16=False),
    "ragged_h36_t3": _gate_case(T=3, R=8, H=36, w16=False),
    "rank_256": _gate_case(T=3, R=256, H=512),
    "tiny_activations": _gate_case(T=17, mag=1e-3),
    "huge_activations": _gate_case(T=17, mag=1e8),
    "norm_w_f32": _gate_case(T=17, nwdt="f32"),
    "strided_messages": _gate_case(T=17, strided=True),
}  # fmt: skip


@pytest.mark.parametrize("name", list(GATE_CASES))
def test_msg_gate(name):
    c = GATE_CASES[name]
    T, G, SH, R, H = c["T"], c["G"], c["SH"], c["R"], c["H"]
    g = _gen(_seed(name))
    h_c = _randn(g, T, SH, H, mag=c["mag"])
    m_c = _randn(g, T, G, H, mag=c["mag"])
    wd_c, wf_c, wg_c = _gate_weights(g, G, R, H)
    nw_c = (1.0 + 0.3 * _randn(g, c["rows"], H)) if c["norm"] else None
    if nw_c is not None and c["nwdt"] == "f16":
        nw_c = nw_c.to(torch.float16)

    if c["strided"]:
        # messages are the [T, 4, H] tensor of MessageAttentionResidual;
        # the gates take its first G streams
        m4 = _randn(g, T, 4, H, mag=c["mag"])
        m_c = m4[:, :G].contiguous()
        m_g = m4.to(DEV)[:, :G]
        h_g, h_buf = _padded(h_c)
        assert m_g.stride() == (4 * H, H, 1)
    else:
        m_g, h_g, h_buf = m_c.to(DEV), h_c.to(DEV), None
    wd_g = wd_c.to(DEV)
    d_g = torch.empty(G, T, R, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_msg_gate_down(h_g, m_g, wd_g, d_g, EPS)

    d_ref = ref_gate_down(h_c, m_c, wd_c)
    check_f16(d_g, d_ref, f"{name}: down projection", atol=D_ATOL, max_mismatch=0.10)

    wf_g, wg_g = _pack_gate(wf_c, c["w16"]), _pack_gate(wg_c, c["w16"])
    wf_eff = (wf_c.to(torch.float16) if c["w16"] else wf_c).double()
    wg_eff = (wg_c.to(torch.float16) if c["w16"] else wg_c).double()
    out_dt = torch.float16 if c["norm"] else torch.float32
    out_g = torch.empty(G, T, H, dtype=out_dt, device=DEV)
    kw = {}
    if c["norm"]:
        assert nw_c is not None
        n_tiles = T  # upper bound of ceil(T / TB)
        kw = dict(
            norm_w=nw_c.to(DEV),
            scratch=torch.empty(G, T, H, device=DEV),
            partials=torch.empty(G, T, (H + 127) // 128, device=DEV),
            counters=torch.zeros(G * n_tiles, dtype=torch.int32, device=DEV),
        )
    ops.inkling2_glue_msg_gate_mix(h_g, m_g, d_g, wf_g, wg_g, out_g, **kw)
    torch.accelerator.synchronize()
    if c["norm"]:
        assert not kw["counters"].any(), "counters not left at zero"

    # the mix is checked on the kernel's own fp16 d: exact input, no rounding flips
    ref = ref_gate_mix(h_c, m_c, d_g.cpu(), wf_eff, wg_eff, nw_c)
    if c["norm"]:
        check_f16(out_g, ref, f"{name}: mix", max_mismatch=1e-2)
    else:
        check_f32(out_g, ref, f"{name}: mix", rtol=5e-5, atol=5e-5 * c["mag"])
    if h_buf is not None:
        assert _pad_intact(h_buf, H)


def test_msg_gate_rejects_bad_arguments():
    g = _gen(2)
    T, G, R, H = 3, 3, 16, 256
    h = _randn(g, T, G, H).to(DEV)
    m = _randn(g, T, G, H).to(DEV)
    wd = _randn(g, G, R, 2 * H).to(torch.float16).to(DEV)
    d = torch.empty(G, T, R, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_msg_gate_down(h, m, wd, d)
    wf = _pack_gate(_randn(g, G, H, R), True)
    out = torch.empty(G, T, H, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError, match="needs norm_w"):
        ops.inkling2_glue_msg_gate_mix(h, m, d, wf, wf, out)
    with pytest.raises(RuntimeError, match="scratch must be"):
        ops.inkling2_glue_msg_gate_mix(
            h, m, d, wf, wf, out, norm_w=torch.ones(G, H, device=DEV)
        )
    with pytest.raises(RuntimeError, match="contiguous"):
        ops.inkling2_glue_msg_gate_down(h, m, wd.transpose(1, 2), d)
    with pytest.raises(RuntimeError, match="must be torch.float32|must be Float"):
        ops.inkling2_glue_msg_gate_down(h.half(), m, wd, d)
    # the down projection stages whole rows in registers: H <= 4096
    wide = 4104
    hw = _randn(g, 1, 1, wide).to(DEV)
    wdw = torch.zeros(1, 2, 2 * wide, dtype=torch.float16, device=DEV)
    dw = torch.empty(1, 1, 2, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError, match="H <= 4096"):
        ops.inkling2_glue_msg_gate_down(hw, hw, wdw, dw)


# ---------------------------------------------------------------------------
# rms_norm / embed_norm
# ---------------------------------------------------------------------------
def _rms_case(**kw):
    base = dict(
        T=17, H=2304, xdt="f32", odt="f16", w="f16", x1=False, sum_out=False,
        rnd=False, mag=1.0, eps=EPS, strided=False,
    )  # fmt: skip
    base.update(kw)
    return base


RMS_CASES = {
    "layer0_input_norm": _rms_case(T=1, w="f16", xdt="f32", odt="f16"),
    "layer0_t1024": _rms_case(T=1024, w="f16", xdt="f32", odt="f16"),
    "final_norm_lazy_add": _rms_case(x1=True, sum_out=True),
    "final_norm_t1": _rms_case(T=1, x1=True),
    "unweighted": _rms_case(T=3, w=None, odt="f32"),
    "weight_f32": _rms_case(T=3, w="f32", odt="f32"),
    "fp16_input": _rms_case(xdt="f16", odt="f32", rnd=True),
    "embed_path_f16_in_f32_out": _rms_case(T=3, xdt="f16", odt="f32", rnd=True),
    "round_flag": _rms_case(T=17, odt="f32", rnd=True, w="f32"),
    "no_round_f32": _rms_case(T=17, odt="f32", rnd=False, w="f32"),
    "tiny_h256": _rms_case(T=5, H=256),
    "ragged_h36": _rms_case(T=3, H=36, odt="f32"),
    "h4096": _rms_case(T=3, H=4096),
    "tiny_activations": _rms_case(mag=1e-4, odt="f32", w="f32"),
    "huge_activations": _rms_case(mag=1e15, odt="f32", w="f32"),
    "other_eps": _rms_case(mag=1e-3, eps=1e-4, odt="f32", w="f32"),
    "strided": _rms_case(x1=True, sum_out=True, strided=True),
}


def _tdt(s):
    return {"f16": torch.float16, "f32": torch.float32}[s]


@pytest.mark.parametrize("name", list(RMS_CASES))
def test_rms_norm(name):
    c = RMS_CASES[name]
    T, H = c["T"], c["H"]
    g = _gen(_seed(name))
    x_c = _randn(g, T, H, mag=c["mag"], dtype=_tdt(c["xdt"]))
    x1_c = _randn(g, T, H, mag=c["mag"]) if c["x1"] else None
    w_c = None
    if c["w"]:
        w_c = (1.0 + 0.3 * _randn(g, H)).to(_tdt(c["w"]))
    x_g, x_buf = _padded(x_c) if c["strided"] else (x_c.to(DEV), None)
    x1_g = x1_c.to(DEV) if x1_c is not None else None
    out_c = torch.full((T, H), PAD_FILL, dtype=_tdt(c["odt"]))
    out_g, out_buf = _padded(out_c) if c["strided"] else (out_c.to(DEV), None)
    so_g = torch.full((T, H), -1.0, device=DEV) if c["sum_out"] else None
    ops.inkling2_glue_rms_norm(
        x_g, out_g, weight=w_c.to(DEV) if w_c is not None else None, x1=x1_g,
        sum_out=so_g, eps=c["eps"], round_fp16=c["rnd"],
    )  # fmt: skip
    torch.accelerator.synchronize()

    v = x_c.double() + (x1_c.double() if x1_c is not None else 0.0)
    ref = _unw_norm(v, c["eps"])
    if w_c is not None:
        ref = w_c.double() * ref
    if c["odt"] == "f16":
        check_f16(out_g, ref, name)
    elif c["rnd"]:
        # an fp32 buffer that holds fp16 values (embedding / inputs_embeds path)
        assert torch.equal(out_g, out_g.half().float()), "not fp16-rounded"
        check_f16(out_g.half(), ref, name)
    else:
        check_f32(out_g, ref, name, rtol=2e-5, atol=2e-5)
    if so_g is not None:
        assert torch.equal(so_g.cpu(), x_c.float() + x1_c), "sum_out"
    if out_buf is not None:
        assert _pad_intact(out_buf, H) and _pad_intact(x_buf, H)


@pytest.mark.parametrize("idt", [torch.int32, torch.int64])
@pytest.mark.parametrize("wdt", ["f16", "f32"])
@pytest.mark.parametrize("odt", ["f16", "f32"])
def test_embed_norm(idt, wdt, odt):
    g = _gen(5)
    V, H, T = 4099, 2304, 33
    table = _randn(g, V, H, mag=0.05).to(torch.float16)
    w = (1.0 + 0.3 * _randn(g, H)).to(_tdt(wdt))
    ids = torch.randint(0, V, (T,), generator=g, dtype=torch.int64)
    ids[0], ids[1], ids[2] = 0, V - 1, V // 2
    # out-of-range ids are clamped (never read out of bounds)
    ids[3], ids[4] = -7, V + 123
    out = torch.empty(T, H, dtype=_tdt(odt), device=DEV)
    ops.inkling2_glue_embed_norm(
        ids.to(idt).to(DEV), table.to(DEV), w.to(DEV), out, EPS
    )
    torch.accelerator.synchronize()
    rows = table[ids.clamp(0, V - 1)]
    ref = w.double() * _unw_norm(rows)
    if odt == "f16":
        check_f16(out, ref, "embed_norm")
    else:
        assert torch.equal(out, out.half().float())
        check_f16(out.half(), ref, "embed_norm")


def test_embed_norm_strided_table_and_odd_vocab():
    g = _gen(6)
    V, H, T = 1000, 256, 5
    table_c = _randn(g, V, H, mag=0.1).to(torch.float16)
    table, buf = _padded(table_c)
    w = torch.ones(H, dtype=torch.float16, device=DEV)
    out = torch.empty(T, H, dtype=torch.float16, device=DEV)
    ids = torch.tensor([999, 0, 500, 1, 998], dtype=torch.int32, device=DEV)
    ops.inkling2_glue_embed_norm(ids, table, w, out, EPS)
    check_f16(out, _unw_norm(table_c[ids.cpu().long()]), "strided table")


# ---------------------------------------------------------------------------
# embed_lookup: this rank's rows of a vocabulary-sharded embedding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("idt", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "first,rows", [(0, 4099), (1, 1), (1000, 1500), (4098, 1), (3000, 1099)]
)
def test_embed_lookup_shard(idt, first, rows):
    """A shard returns its rows bit for bit and zeros for every other id; ids are
    clamped to the vocabulary before the ownership test (like embed_norm)."""
    g = _gen(40 + first)
    total, H, T = 4099, 2304, 48
    table_c = _randn(g, total, H, mag=0.05).to(torch.float16)
    # a view of the full table: a read past the end of the shard would see the next row
    table = table_c.to(DEV)
    shard = table[first : first + rows]
    ids = torch.randint(0, total, (T,), generator=g, dtype=torch.int64)
    edge = [0, total - 1, first, first + rows - 1, first - 1, first + rows]
    edge += [-7, total + 123]
    ids[: len(edge)] = torch.tensor(edge)
    out = torch.full((T, H), PAD_FILL, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_embed_lookup(ids.to(idt).to(DEV), shard, out, first, total)
    torch.accelerator.synchronize()
    clamped = ids.clamp(0, total - 1)
    own = (clamped >= first) & (clamped < first + rows)
    zero = torch.zeros((), dtype=torch.float16)
    want = torch.where(own[:, None], table_c[clamped], zero)
    assert torch.equal(out.cpu(), want)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_embed_lookup_shards_add_up_to_the_full_gather(size):
    """What the all-reduce relies on: the owner's row plus zeros is the row, also with
    a zero-padded last shard."""
    g = _gen(50)
    total, H, T = 4099, 256, 64
    per = -(-total // size)
    table_c = _randn(g, total, H, mag=0.5).to(torch.float16)
    padded = torch.zeros(per * size, H, dtype=torch.float16)
    padded[:total] = table_c
    padded = padded.to(DEV)
    ids = torch.randint(0, total, (T,), generator=g, dtype=torch.int64)
    ids[:3] = torch.tensor([0, total - 1, per])
    ids = ids.to(DEV)
    acc = torch.zeros(T, H, dtype=torch.float16, device=DEV)
    for r in range(size):
        part = torch.full((T, H), PAD_FILL, dtype=torch.float16, device=DEV)
        ops.inkling2_glue_embed_lookup(
            ids, padded[r * per : (r + 1) * per], part, r * per, total
        )
        acc += part
    torch.accelerator.synchronize()
    assert torch.equal(acc.cpu(), table_c[ids.cpu()])


@pytest.mark.parametrize("odt", ["f16", "f32"])
def test_embed_lookup_then_rms_norm_is_embed_norm(odt):
    """The tensor-parallel embedding (lookup, all-reduce, norm) is bit-identical to
    the fused TP=1 op: same normalisation code on the same rows."""
    g = _gen(52)
    V, H, T = 4099, 2304, 33
    table = _randn(g, V, H, mag=0.05).to(torch.float16).to(DEV)
    w = (1.0 + 0.3 * _randn(g, H)).to(torch.float16).to(DEV)
    ids = torch.randint(0, V, (T,), generator=g, dtype=torch.int64).to(DEV)
    fused = torch.empty(T, H, dtype=_tdt(odt), device=DEV)
    ops.inkling2_glue_embed_norm(ids, table, w, fused, EPS)
    rows = torch.empty(T, H, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_embed_lookup(ids, table, rows, 0, V)
    split = torch.empty(T, H, dtype=_tdt(odt), device=DEV)
    ops.inkling2_glue_rms_norm(rows, split, weight=w, eps=EPS, round_fp16=True)
    torch.accelerator.synchronize()
    assert torch.equal(fused, split)


def test_embed_lookup_strided_rows_and_empty_batch():
    g = _gen(51)
    total, H = 1000, 256
    table_c = _randn(g, total, H, mag=0.1).to(torch.float16)
    table, tbuf = _padded(table_c)
    out, obuf = _padded(torch.full((5, H), PAD_FILL, dtype=torch.float16))
    ids = torch.tensor([999, 0, 500, 1, 998], dtype=torch.int32, device=DEV)
    ops.inkling2_glue_embed_lookup(ids, table, out, 0, total)
    torch.accelerator.synchronize()
    assert torch.equal(out.cpu(), table_c[ids.cpu().long()])
    assert _pad_intact(obuf, H) and _pad_intact(tbuf, H)
    ops.inkling2_glue_embed_lookup(ids[:0], table, out[:0], 0, total)  # T = 0
    torch.accelerator.synchronize()


def test_embed_lookup_rejects_bad_arguments():
    H = 256
    ids = torch.zeros(3, dtype=torch.int32, device=DEV)
    table = torch.zeros(10, H, dtype=torch.float16, device=DEV)
    out = torch.zeros(3, H, dtype=torch.float16, device=DEV)
    with pytest.raises(RuntimeError, match="table must be"):
        ops.inkling2_glue_embed_lookup(ids, table.float(), out, 0, 10)
    with pytest.raises(RuntimeError, match="out must be"):
        ops.inkling2_glue_embed_lookup(ids, table, out[:2], 0, 10)
    with pytest.raises(RuntimeError, match="vocab_total"):
        ops.inkling2_glue_embed_lookup(ids, table, out, 0, 0)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        ops.inkling2_glue_embed_lookup(ids, table[:, :4], out[:, :4], 0, 10)


# ---------------------------------------------------------------------------
# swiglu
# ---------------------------------------------------------------------------
def ref_swiglu(gate_up, gc, uc):
    gate, up = gate_up.double().chunk(2, dim=-1)
    return (
        (torch.tanh(gate / gc) * gc) * torch.sigmoid(gate) * (torch.tanh(up / uc) * uc)
    )


@pytest.mark.parametrize("R", [1, 3, 17, 1024, 8192])
@pytest.mark.parametrize("inter", [128, 1152])
@pytest.mark.parametrize("mag", [1.0, 30.0, 4000.0])
def test_swiglu(R, inter, mag):
    g = _gen(R * 7 + inter + int(mag))
    gu_c = _randn(g, R, 2 * inter, mag=mag).to(torch.float16)
    out = torch.empty(R, inter, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_swiglu(gu_c.to(DEV), out, 4.0, 25.0)
    torch.accelerator.synchronize()
    what = f"swiglu R={R} I={inter} mag={mag}"
    check_f16(out, ref_swiglu(gu_c, 4.0, 25.0), what)


def test_swiglu_saturation_and_extremes():
    inter = 128
    vals = torch.tensor(
        [0.0, -0.0, 1e-8, -1e-8, 0.5, -0.5, 3.9, 4.0, 4.1, 24.9, 25.0, 26.0, 88.0,
         -88.0, 100.0, -100.0, 1000.0, -1000.0, 30000.0, -30000.0, 65504.0, -65504.0],
        dtype=torch.float32,
    )  # fmt: skip
    n = vals.numel()
    gate = vals.repeat_interleave(n)  # all (gate, up) pairs
    up = vals.repeat(n)
    pad = (-gate.numel()) % inter
    gate = torch.cat([gate, torch.zeros(pad)])
    up = torch.cat([up, torch.zeros(pad)])
    rows = gate.numel() // inter
    gu = torch.cat([gate.view(rows, inter), up.view(rows, inter)], -1)
    gu = gu.to(torch.float16)
    out = torch.empty(rows, inter, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_swiglu(gu.to(DEV), out, 4.0, 25.0)
    torch.accelerator.synchronize()
    check_f16(out, ref_swiglu(gu, 4.0, 25.0), "swiglu extremes")
    other = torch.empty_like(out)
    ops.inkling2_glue_swiglu(gu.to(DEV), other, 3.0, 7.0)
    check_f16(other, ref_swiglu(gu, 3.0, 7.0), "swiglu other caps")


# ---------------------------------------------------------------------------
# moe_route
# ---------------------------------------------------------------------------
def ref_align(ids: torch.Tensor, E: int, BM: int):
    """Stable ``moe_align_block_size`` layout of ``ids`` [T, K]."""
    flat = ids.flatten().long()
    numel = flat.numel()
    max_padded = numel + E * (BM - 1)
    if numel < E:
        max_padded = min(numel * BM, max_padded)
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32)
    expert_ids = torch.full((-(-max_padded // BM),), -1, dtype=torch.int32)
    pos = blk = 0
    for e in range(E):
        idx = (flat == e).nonzero().flatten()
        n = idx.numel()
        nb = -(-n // BM)
        sorted_ids[pos : pos + n] = idx.to(torch.int32)
        expert_ids[blk : blk + nb] = e
        pos += nb * BM
        blk += nb
    return sorted_ids, expert_ids, pos


def _route_case(**kw):
    base = dict(
        T=1, E=128, K=8, BM=1, w16=True, H=2304, scale=2.5, zero=True,
        ties=False, mag=1.0, hot=False,
    )  # fmt: skip
    base.update(kw)
    return base


ROUTE_CASES = {
    "decode_t1_f16w": _route_case(),
    "decode_t1_f32w": _route_case(w16=False),
    "t2": _route_case(T=2, BM=2),
    "t3_f32w": _route_case(T=3, BM=2, w16=False),
    "t4_f32w_bm1": _route_case(T=4, BM=1, w16=False),
    "t2_f32w_bm1": _route_case(T=2, BM=1, w16=False),
    "e255_k16_f32w": _route_case(T=33, E=255, K=16, H=512, BM=8, w16=False),
    "h2304_e128_t1_f32w_hot": _route_case(T=1, BM=1, w16=False, hot=True),
    "t4": _route_case(T=4, BM=4),
    "t17": _route_case(T=17, BM=4),
    "t17_f32w": _route_case(T=17, BM=8, w16=False),
    "t64": _route_case(T=64, BM=8),
    "t1024": _route_case(T=1024, BM=8),
    "t1024_f32w": _route_case(T=1024, BM=8, w16=False),
    "t300_ragged_neg": _route_case(T=300, BM=8),
    "t200_f32w_ragged_neg": _route_case(T=200, BM=8, w16=False),
    "tiny_e8_k2": _route_case(T=1, E=8, K=2, H=256),
    "tiny_e8_k2_t17": _route_case(T=17, E=8, K=2, H=256, BM=4, w16=False),
    "tiny_e8_k2_t5": _route_case(T=5, E=8, K=2, H=256, BM=2),
    "fewer_entries_than_experts": _route_case(T=1, E=128, K=2, BM=8),
    "e3_k3_all_experts": _route_case(T=7, E=3, K=3, H=64, BM=4),
    "e255_k16": _route_case(T=33, E=255, K=16, H=512, BM=8),
    "e33_ragged_waves": _route_case(T=9, E=33, K=5, H=128, BM=2, w16=False),
    "ties_pick_lowest_index": _route_case(T=17, E=64, K=8, BM=4, ties=True),
    "ties_f32w": _route_case(T=3, E=64, K=8, BM=2, ties=True, w16=False),
    "hot_experts_t300": _route_case(T=300, BM=8, hot=True),
    "hot_single_expert_t500": _route_case(T=500, K=1, BM=8, hot=True),
    "hot_experts_e255_t40": _route_case(T=40, E=255, K=16, H=512, BM=4, hot=True),
    "big_logits_saturate_sigmoid": _route_case(T=17, mag=40.0),
    "tiny_logits": _route_case(T=17, mag=1e-4),
    "no_zero_buffers": _route_case(T=17, zero=False),
    "route_scale_one": _route_case(T=3, scale=1.0),
}  # fmt: skip


def _route_inputs(c, g):
    T, E, H = c["T"], c["E"], c["H"]
    x_c = _randn(g, T, H, mag=c["mag"]).to(torch.float16)
    w_c = _randn(g, E, H, mag=1.0 / H**0.5)
    bias_c = 0.1 * _randn(g, E)
    if c["ties"]:
        # identical expert rows and zero bias: exact ties everywhere
        w_c = w_c[:8].repeat(E // 8, 1)
        bias_c = torch.zeros(E)
    if c["hot"]:
        bias_c[: c["K"]] += 100.0  # every token picks the same K experts
    if c["w16"]:
        w_c = w_c.to(torch.float16)
    return x_c, w_c, bias_c


def _route_buffers(c, zero_val=1.0):
    T, E, K, BM = c["T"], c["E"], c["K"], c["BM"]
    max_padded, max_blocks, n_counters = ops.inkling2_glue_moe_route_sizes(T, E, K, BM)
    return dict(
        logits=torch.full((T, E), float("nan"), device=DEV),
        topk_weights=torch.full((T, K), float("nan"), device=DEV),
        topk_ids=torch.full((T, K), -5, dtype=torch.int32, device=DEV),
        sorted_ids=torch.full((max_padded,), -7, dtype=torch.int32, device=DEV),
        expert_ids=torch.full((max_blocks,), -7, dtype=torch.int32, device=DEV),
        post_pad=torch.full((1,), -7, dtype=torch.int32, device=DEV),
        counters=torch.zeros(n_counters, dtype=torch.int32, device=DEV),
    )


def _check_route(c, x_c, w_c, bias_c, b, what):
    T, E, K, BM = c["T"], c["E"], c["K"], c["BM"]
    logits = b["logits"].cpu()
    ref_logits = x_c.double() @ w_c.double().T
    check_f32(logits, ref_logits, f"{what}: logits", rtol=2e-5, atol=2e-5)

    ids = b["topk_ids"].cpu().long()
    assert ids.min() >= 0 and ids.max() < E, f"{what}: id out of range"
    assert all(len(set(r.tolist())) == K for r in ids), f"{what}: duplicate experts"
    # valid top-k of the kernel's own scores (near ties may legitimately swap)
    scores = torch.sigmoid(logits)  # fp32, like the kernel
    biased = (scores + bias_c).double()
    sel = biased.gather(1, ids)
    mask = torch.ones(T, E, dtype=torch.bool).scatter(1, ids, False)
    worst_out = biased.masked_fill(~mask, -float("inf")).max(1).values
    tol = 1e-6 * biased.abs().max(1).values.clamp_min(1.0)
    assert bool((sel.min(1).values >= worst_out - tol).all()), f"{what}: not top-k"
    if c["ties"]:
        # exact ties: the lowest expert indices win, in selection order
        order = torch.sort(biased, dim=1, descending=True, stable=True).indices[:, :K]
        assert torch.equal(ids, order), f"{what}: tie order"

    w = b["topk_weights"].cpu().double()
    chosen = scores.double().gather(1, ids)
    ref_w = chosen / (chosen.sum(1, keepdim=True) + 1e-20) * c["scale"]
    check_f32(w, ref_w, f"{what}: weights", rtol=2e-5, atol=1e-6)

    s_ref, e_ref, post_ref = ref_align(b["topk_ids"].cpu(), E, BM)
    assert torch.equal(b["sorted_ids"].cpu(), s_ref), f"{what}: sorted_ids"
    assert torch.equal(b["expert_ids"].cpu(), e_ref), f"{what}: expert_ids"
    assert int(b["post_pad"].item()) == post_ref, f"{what}: post_pad"
    assert not b["counters"].any(), f"{what}: counters not left at zero"


@pytest.mark.parametrize("name", list(ROUTE_CASES))
def test_moe_route(name):
    c = ROUTE_CASES[name]
    g = _gen(_seed(name))
    x_c, w_c, bias_c = _route_inputs(c, g)
    b = _route_buffers(c)
    zeros: dict[str, torch.Tensor] = {}
    if c["zero"]:
        z1 = torch.ones(c["T"] * c["K"] * 2 * 128, dtype=torch.float16, device=DEV)
        z2 = torch.ones(c["T"] * c["H"] + 4096, dtype=torch.float16, device=DEV)
        guards = (z1[-8:].clone(), z2[-8:].clone())
        zeros = dict(zero1=z1[:-8], zero2=z2[:-8])
    ops.inkling2_glue_moe_route(
        x_c.to(DEV), w_c.to(DEV), bias_c.to(DEV), b["logits"], b["topk_weights"],
        b["topk_ids"], b["sorted_ids"], b["expert_ids"], b["post_pad"],
        b["counters"], c["BM"], c["scale"], **zeros,
    )  # fmt: skip
    torch.accelerator.synchronize()
    _check_route(c, x_c, w_c, bias_c, b, name)
    if c["zero"]:
        assert not zeros["zero1"].any() and not zeros["zero2"].any(), (
            "zero fill incomplete"
        )
        assert torch.equal(z1[-8:], guards[0]) and torch.equal(z2[-8:], guards[1]), (
            "zero fill ran past the buffer"
        )


def test_moe_route_rejects_bad_arguments():
    g = _gen(3)
    c = _route_case(T=4, E=16, K=2, H=64, BM=2)
    x_c, w_c, bias_c = _route_inputs(c, g)
    b = _route_buffers(c)
    args = [x_c.to(DEV), w_c.to(DEV), bias_c.to(DEV), b["logits"], b["topk_weights"],
            b["topk_ids"], b["sorted_ids"], b["expert_ids"], b["post_pad"],
            b["counters"], 2, 1.0]  # fmt: skip
    ops.inkling2_glue_moe_route(*args)  # baseline is accepted
    bad = list(args)
    bad[9] = torch.zeros(1, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="counters needs"):
        ops.inkling2_glue_moe_route(*bad)
    bad = list(args)
    bad[6] = torch.zeros(3, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError, match="sorted_ids needs"):
        ops.inkling2_glue_moe_route(*bad)
    bad = list(args)
    bad[10] = 3
    with pytest.raises(RuntimeError, match="block_size_m"):
        ops.inkling2_glue_moe_route(*bad)
    big = _route_case(T=1025, E=128, K=8, H=64, BM=8)
    xb, wb, bb = _route_inputs(big, g)
    b2 = _route_buffers(big)
    with pytest.raises(RuntimeError, match="at most"):
        ops.inkling2_glue_moe_route(
            xb.to(DEV), wb.to(DEV), bb.to(DEV), b2["logits"], b2["topk_weights"],
            b2["topk_ids"], b2["sorted_ids"], b2["expert_ids"], b2["post_pad"],
            b2["counters"], 8, 1.0,
        )  # fmt: skip


def test_moe_route_nan_row_keeps_ids_valid():
    """Inf / NaN activations must not make the experts GEMM read out of range."""
    g = _gen(4)
    c = _route_case(T=5, E=128, K=8, BM=4)
    x_c, w_c, bias_c = _route_inputs(c, g)
    x_c[2, 7] = float("nan")  # every logit of token 2 becomes NaN
    b = _route_buffers(c)
    ops.inkling2_glue_moe_route(
        x_c.to(DEV), w_c.to(DEV), bias_c.to(DEV), b["logits"], b["topk_weights"],
        b["topk_ids"], b["sorted_ids"], b["expert_ids"], b["post_pad"],
        b["counters"], c["BM"], c["scale"],
    )  # fmt: skip
    torch.accelerator.synchronize()
    ids = b["topk_ids"].cpu().long()
    assert ((ids >= 0) & (ids < c["E"])).all(), "id out of range"
    assert not b["counters"].any(), "counters not left at zero"
    ok = [0, 1, 3, 4]  # the other tokens are unaffected: still a valid top-k
    biased = (torch.sigmoid(b["logits"].cpu()[ok]) + bias_c).double()
    sel = biased.gather(1, ids[ok])
    rest = biased.scatter(1, ids[ok], -float("inf")).max(1).values
    assert bool((sel.min(1).values >= rest - 1e-6).all()), "healthy tokens broken"


def test_moe_route_repeated_launches_do_not_race():
    """The last-arriver CTAs (top-k, align) must never miss an update."""
    c = _route_case(T=17, E=128, K=8, BM=4)
    n = 150
    g = _gen(11)
    xs, ws, biases = [], [], []
    for _ in range(n):
        x_c, w_c, bias_c = _route_inputs(c, g)
        xs.append(x_c)
        ws.append(w_c)
        biases.append(bias_c)
    bufs = [_route_buffers(c) for _ in range(n)]
    counters = torch.zeros_like(bufs[0]["counters"])  # shared: must self reset
    for i in range(n):
        b = bufs[i]
        ops.inkling2_glue_moe_route(
            xs[i].to(DEV), ws[i].to(DEV), biases[i].to(DEV), b["logits"],
            b["topk_weights"], b["topk_ids"], b["sorted_ids"], b["expert_ids"],
            b["post_pad"], counters, c["BM"], c["scale"],
        )  # fmt: skip
    torch.accelerator.synchronize()
    assert not counters.any()
    for i in range(n):
        bufs[i]["counters"] = counters
        _check_route(c, xs[i], ws[i], biases[i], bufs[i], f"launch {i}")


# ---------------------------------------------------------------------------
# whole layer visits against the torch modules of blocks.py
# ---------------------------------------------------------------------------
def _cfg(H, R):
    return types.SimpleNamespace(
        hidden_size=H, rms_norm_eps=EPS, loop_gate_rank=R,
        loop_layer_index_start=6, loop_layer_index_end=18,
    )  # fmt: skip


def _fill(module, g, std):
    """Random parameters that are exactly bf16 (the checkpoint dtype).

    The kernels hold fp16 copies of the gate weights; a bf16 value converts to
    fp16 without rounding, so the module path and the kernels see equal weights.
    """
    for p in module.parameters():
        p.data.copy_(_randn(g, *p.shape, mag=std).to(torch.bfloat16).to(p.dtype))


@pytest.mark.parametrize("T", [1, 17])
@pytest.mark.parametrize("w16", [True, False])
@pytest.mark.parametrize("H,R", [(2304, 64), (256, 16)])
@pytest.mark.parametrize("full", [True, False])
def test_loop_visit_attention_prologue_matches_modules(T, w16, H, R, full):
    """Messages -> prologue -> message gates -> layer norms (pass >= 1 visit)."""
    g = _gen(T * 31 + H + int(full))
    cfg = _cfg(H, R)
    nb, nb_prev = 3, 1
    bar = blocks.BlockAttentionResidual(cfg, 3 if full else 1).to(DEV)
    mar = blocks.MessageAttentionResidual(cfg).to(DEV)
    gates = [blocks.MessageGate(cfg, dtype=torch.float16).to(DEV) for _ in range(3)]
    norms = [
        blocks.Inkling2RMSNorm(H, EPS, dtype=torch.float16).to(DEV) for _ in range(3)
    ]
    for mod in (bar, mar):
        _fill(mod, g, 1.0)  # queries with a non trivial softmax
    for mod in gates:
        _fill(mod, g, 0.03)
    for n in norms:
        n.weight.data.copy_((1 + 0.3 * _randn(g, H)).to(torch.float16))
    j = 5

    hid = _randn(g, T, H).to(DEV)
    blk = _randn(g, T, nb, H).to(DEV)
    prev_blk = _randn(g, T, nb_prev, H).to(DEV)
    prev_hid = _randn(g, T, H).to(DEV)
    attn = _randn(g, T, H).to(DEV)  # x1: the lazily added second summand

    with torch.no_grad():
        # ---- torch reference (blocks.py, exactly what model.py does)
        partial = hid + attn
        msgs = mar(prev_blk, prev_hid, j)  # [T, 4, H]
        qm, km, vm, _ = msgs.unbind(1)
        if full:
            qh, kh, vh = bar(blk, partial)
            hq = norms[0](gates[0](qh, qm)).to(torch.float16)
            hk = norms[1](gates[1](kh, km)).to(torch.float16)
            hv = norms[2](gates[2](vh, vm)).to(torch.float16)
        else:
            h = bar(blk, partial)[0]
            hq = norms[0](gates[0](h, qm)).to(torch.float16)
            hk = norms[0](gates[1](h, km)).to(torch.float16)
            hv = norms[0](gates[2](h, vm)).to(torch.float16)
        ref = torch.stack([hq, hk, hv])

    # ---- kernel sequence
    Q = 3 if full else 1
    msgs_k = torch.empty(T, 4, H, device=DEV)
    ops.inkling2_glue_attn_res(
        prev_blk, nb_prev, prev_hid, mar.query[j].float(), msgs_k, eps=EPS
    )
    raw = torch.empty(T, Q, H, device=DEV)
    hidden_sum = torch.empty(T, H, device=DEV)
    ops.inkling2_glue_attn_res(
        blk, nb, hid, bar.query.float().view(Q, H), raw, x1=attn,
        partial_out=hidden_sum, eps=EPS,
    )  # fmt: skip
    assert torch.equal(hidden_sum, hid + attn)
    wd = torch.stack([m.down_proj.weight.data for m in gates])  # [3, R, 2H]
    wf = torch.stack([m.forget_gate_proj.weight.data for m in gates])  # [3, H, R]
    wg = torch.stack([m.message_gate_proj.weight.data for m in gates])
    if w16:
        wf_k = _pack_gate(wf.cpu(), True)
        wg_k = _pack_gate(wg.cpu(), True)
    else:
        wf_k, wg_k = _pack_gate(wf.cpu(), False), _pack_gate(wg.cpu(), False)
    nw = (
        torch.stack([n.weight.data for n in norms])
        if full
        else norms[0].weight.data[None]
    )
    d = torch.empty(3, T, R, dtype=torch.float16, device=DEV)
    out = torch.empty(3, T, H, dtype=torch.float16, device=DEV)
    h_in = raw if full else raw  # [T, Q, H]: Q == 3 streams or 1 broadcast
    ops.inkling2_glue_msg_gate_down(h_in, msgs_k[:, :3], wd, d, EPS)
    ops.inkling2_glue_msg_gate_mix(
        h_in, msgs_k[:, :3], d, wf_k, wg_k, out, norm_w=nw,
        scratch=torch.empty(3, T, H, device=DEV),
        partials=torch.empty(3, T, (H + 127) // 128, device=DEV),
        counters=torch.zeros(3 * T, dtype=torch.int32, device=DEV),
        norm_eps=EPS,
    )  # fmt: skip
    torch.accelerator.synchronize()
    # the module path rounds d to fp16 from a differently ordered fp32 sum, so a
    # few of its elements sit one fp16 step away from the kernel's: bound the
    # absolute error of the gates instead of counting ulps
    check_f16(out, ref.double(), "visit prologue", atol=VISIT_ATOL, max_mismatch=0.2)


@pytest.mark.parametrize("T", [1, 17])
@pytest.mark.parametrize("with_messages", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("H,R", [(2304, 64), (256, 16)])
def test_loop_visit_mlp_input_matches_modules(T, with_messages, boundary, H, R):
    """Attention output -> mlp_input -> [mlp gate] -> post_attention_layernorm.

    A block boundary layer appends its input and uses the attention output as
    the partial state; the other layers attend over ``hidden + attention``.
    """
    g = _gen(T * 17 + H + 3 * int(boundary))
    cfg = _cfg(H, R)
    nb = 4
    bar = blocks.BlockAttentionResidual(cfg).to(DEV)
    gate = blocks.MessageGate(cfg, dtype=torch.float16).to(DEV)
    norm = blocks.Inkling2RMSNorm(H, EPS, dtype=torch.float16).to(DEV)
    _fill(bar, g, 1.0)
    _fill(gate, g, 0.03)
    norm.weight.data.copy_((1 + 0.3 * _randn(g, H)).to(torch.float16))
    hid = _randn(g, T, H).to(DEV)
    attn = _randn(g, T, H).to(DEV)
    blk = _randn(g, T, nb + 2, H).to(DEV)
    mm = _randn(g, T, H).to(DEV)
    with torch.no_grad():
        if boundary:
            states, partial = torch.cat([blk[:, :nb], hid.unsqueeze(1)], 1), attn
        else:
            states, partial = blk[:, :nb], hid + attn
        h = bar(states, partial)[0]
        if with_messages:
            h = gate(h, mm)
        ref = norm(h).to(torch.float16)

    blk_k = blk.clone()
    qry = bar.query.detach().float().view(1, H)
    x0, extra = (attn, dict(append=hid)) if boundary else (hid, dict(x1=attn))
    residual = None if boundary else torch.empty(T, H, device=DEV)
    if with_messages:
        raw = torch.empty(T, 1, H, device=DEV)
        ops.inkling2_glue_attn_res(
            blk_k, nb, x0, qry, raw, partial_out=residual, eps=EPS, **extra
        )
        d = torch.empty(1, T, R, dtype=torch.float16, device=DEV)
        out = torch.empty(1, T, H, dtype=torch.float16, device=DEV)
        wd = gate.down_proj.weight.data[None]
        wf = _pack_gate(gate.forget_gate_proj.weight.data[None].cpu(), True)
        wg = _pack_gate(gate.message_gate_proj.weight.data[None].cpu(), True)
        mmv = mm.view(T, 1, H)
        ops.inkling2_glue_msg_gate_down(raw, mmv, wd, d, EPS)
        ops.inkling2_glue_msg_gate_mix(
            raw, mmv, d, wf, wg, out, norm_w=norm.weight.data[None],
            scratch=torch.empty(1, T, H, device=DEV),
            partials=torch.empty(1, T, (H + 127) // 128, device=DEV),
            counters=torch.zeros(T, dtype=torch.int32, device=DEV), norm_eps=EPS,
        )  # fmt: skip
        got = out[0]
        atol, mismatch = VISIT_ATOL, 0.25
    else:
        got = torch.empty(T, 1, H, dtype=torch.float16, device=DEV)
        ops.inkling2_glue_attn_res(
            blk_k, nb, x0, qry, got, partial_out=residual,
            norm_w=norm.weight.data[None], eps=EPS, norm_eps=EPS, **extra,
        )  # fmt: skip
        got = got[:, 0]
        atol, mismatch = None, 0.02
    torch.accelerator.synchronize()
    if boundary:
        assert torch.equal(blk_k[:, nb], hid), "boundary append"
    else:
        assert torch.equal(blk_k, blk), "block states modified"
        assert torch.equal(residual, hid + attn), "residual"
    check_f16(got, ref.double(), "mlp input", atol=atol, max_mismatch=mismatch)


def test_router_matches_blocks_route_topk():
    g = _gen(21)
    c = _route_case(T=64, E=128, K=8, BM=4, w16=False)
    x_c, w_c, bias_c = _route_inputs(c, g)
    b = _route_buffers(c)
    ops.inkling2_glue_moe_route(
        x_c.to(DEV), w_c.to(DEV), bias_c.to(DEV), b["logits"], b["topk_weights"],
        b["topk_ids"], b["sorted_ids"], b["expert_ids"], b["post_pad"], b["counters"],
        c["BM"], 2.5,
    )  # fmt: skip
    torch.accelerator.synchronize()
    logits = (x_c.float().to(DEV) @ w_c.to(DEV).T).cpu()  # the torch router
    w_ref, ids_ref = blocks.route_topk(logits, bias_c, 8, 2.5)
    got_ids = b["topk_ids"].cpu()
    assert torch.equal(got_ids.sort(1).values, ids_ref.sort(1).values)
    got = b["topk_weights"].cpu().gather(1, got_ids.sort(1).indices)
    want = w_ref.gather(1, ids_ref.long().sort(1).indices)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)


def test_swiglu_matches_blocks():
    g = _gen(22)
    gu = _randn(g, 40, 2 * 1152, mag=6.0).to(torch.float16).to(DEV)
    out = torch.empty(40, 1152, dtype=torch.float16, device=DEV)
    ops.inkling2_glue_swiglu(gu, out, 4.0, 25.0)
    gate, up = gu.chunk(2, -1)
    ref = blocks.softcapped_swiglu(gate, up, 4.0, 25.0).to(torch.float16)
    check_f16(out, ref.double(), "swiglu vs blocks")


# ---------------------------------------------------------------------------
# copy_states (snapshots of the persistent block states)
# ---------------------------------------------------------------------------
def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32).cpu()


@pytest.mark.parametrize(
    "T,P,H,slots,lo,dst_slots,dst_lo",
    [
        (1, 1, 2304, 6, 2, 2, 1),  # the prev-states slice blocks[:, 2:3]
        (17, 1, 2304, 6, 2, 1, 0),
        (5, 3, 256, 8, 3, 4, 1),
        (3, 4, 36, 4, 0, 5, 1),  # H not a multiple of 128
    ],
)
def test_copy_states_3d(T, P, H, slots, lo, dst_slots, dst_lo):
    g = _gen(T * 7 + P)
    src_full = _randn(g, T, slots, H, mag=3.0)
    src_full[0, lo, 0] = float("nan")  # a copy must not compute on the data
    src_full[0, lo, 1] = -0.0
    src_full[0, lo, 2] = float("inf")
    src_full = src_full.to(DEV)
    src_before = src_full.clone()
    dst_full = torch.full((T, dst_slots, H), -123.0, device=DEV)  # sentinel
    ops.inkling2_glue_copy_states(
        src_full[:, lo : lo + P], dst_full[:, dst_lo : dst_lo + P]
    )
    torch.accelerator.synchronize()
    assert torch.equal(_bits(src_full), _bits(src_before)), "source modified"
    assert torch.equal(
        _bits(dst_full[:, dst_lo : dst_lo + P]), _bits(src_full[:, lo : lo + P])
    )
    other = torch.ones(dst_slots, dtype=torch.bool)
    other[dst_lo : dst_lo + P] = False
    assert bool((dst_full.cpu()[:, other] == -123.0).all()), "wrote outside dst"


def test_copy_states_2d_padded_rows():
    g = _gen(8)
    T, H, ld = 9, 2304, 2304 + 8
    src_full = _randn(g, T, ld).to(DEV)
    dst_full = torch.full((T, ld + 16), -1.0, device=DEV)
    ops.inkling2_glue_copy_states(src_full[:, :H], dst_full[:, :H])
    torch.accelerator.synchronize()
    assert torch.equal(_bits(dst_full[:, :H]), _bits(src_full[:, :H]))
    assert bool((dst_full.cpu()[:, H:] == -1.0).all()), "wrote the row padding"


def test_copy_states_rejects_bad_arguments():
    g = _gen(5)
    a = _randn(g, 4, 2, 64).to(DEV)
    b = torch.empty_like(a)
    with pytest.raises(RuntimeError, match="one shape"):
        ops.inkling2_glue_copy_states(a, b[:, :1])
    with pytest.raises(RuntimeError, match="must be torch.float32|must be Float"):
        ops.inkling2_glue_copy_states(a.half(), b)
    with pytest.raises(RuntimeError, match="alias"):
        ops.inkling2_glue_copy_states(a, a)
    with pytest.raises(RuntimeError, match="unit stride"):
        ops.inkling2_glue_copy_states(a.transpose(1, 2), b.transpose(1, 2))
    a62 = _randn(g, 4, 2, 62).to(DEV)
    with pytest.raises(RuntimeError, match="multiple"):
        ops.inkling2_glue_copy_states(a62, torch.empty_like(a62))
    base = torch.empty(4 * 2 * 64 + 1, device=DEV)
    with pytest.raises(RuntimeError, match="byte aligned"):
        ops.inkling2_glue_copy_states(base[1:].view(4, 2, 64), b)


# ---------------------------------------------------------------------------
# graph capture and concurrency of the last-arriver patterns
# ---------------------------------------------------------------------------
def test_gate_mix_repeated_launches_do_not_race():
    g = _gen(31)
    T, G, R, H = 5, 3, 64, 2304
    wd_c, wf_c, wg_c = _gate_weights(g, G, R, H)
    wd, wf, wg = wd_c.to(DEV), _pack_gate(wf_c, True), _pack_gate(wg_c, True)
    nw_c = (1.0 + 0.3 * _randn(g, G, H)).to(torch.float16)
    counters = torch.zeros(G * T, dtype=torch.int32, device=DEV)
    n = 120
    cases = []
    for _ in range(n):
        h_c, m_c = _randn(g, T, G, H), _randn(g, T, G, H)
        h_g, m_g = h_c.to(DEV), m_c.to(DEV)
        d = torch.empty(G, T, R, dtype=torch.float16, device=DEV)
        out = torch.empty(G, T, H, dtype=torch.float16, device=DEV)
        ops.inkling2_glue_msg_gate_down(h_g, m_g, wd, d, EPS)
        ops.inkling2_glue_msg_gate_mix(
            h_g, m_g, d, wf, wg, out, norm_w=nw_c.to(DEV),
            scratch=torch.empty(G, T, H, device=DEV),
            partials=torch.empty(G, T, (H + 127) // 128, device=DEV),
            counters=counters, norm_eps=EPS,
        )  # fmt: skip
        cases.append((h_c, m_c, d, out))
    torch.accelerator.synchronize()
    assert not counters.any()
    wf_eff, wg_eff = wf_c.to(torch.float16).double(), wg_c.to(torch.float16).double()
    for i, (h_c, m_c, d, out) in enumerate(cases):
        ref = ref_gate_mix(h_c, m_c, d.cpu(), wf_eff, wg_eff, nw_c)
        check_f16(out, ref, f"gate mix launch {i}", max_mismatch=1e-2)


def test_visit_sequence_graph_capture_and_replay():
    """The glue sequence of one visit is capturable and replays on new data."""
    g = _gen(41)
    T, H, R, E, K, BM, inter = 3, 2304, 64, 128, 8, 2, 1152
    cfg = _cfg(H, R)
    bar = blocks.BlockAttentionResidual(cfg, 3).to(DEV)
    _fill(bar, g, 1.0)
    qry = bar.query.detach().float().view(3, H)
    wd_c, wf_c, wg_c = _gate_weights(g, 3, R, H)
    wd, wf, wg = wd_c.to(DEV), _pack_gate(wf_c, True), _pack_gate(wg_c, True)
    nw = (1.0 + 0.3 * _randn(g, 3, H)).to(torch.float16).to(DEV)
    gw = _randn(g, E, H, mag=1.0 / H**0.5).to(torch.float16).to(DEV)
    bias = (0.1 * _randn(g, E)).to(DEV)
    max_padded, max_blocks, n_counters = ops.inkling2_glue_moe_route_sizes(T, E, K, BM)

    blk = _randn(g, T, 6, H).to(DEV)
    hid = torch.zeros(T, H, device=DEV)
    x1 = torch.zeros(T, H, device=DEV)
    msgs = torch.zeros(T, 4, H, device=DEV)
    raw = torch.empty(T, 3, H, device=DEV)
    hsum = torch.empty(T, H, device=DEV)
    d = torch.empty(3, T, R, dtype=torch.float16, device=DEV)
    out = torch.empty(3, T, H, dtype=torch.float16, device=DEV)
    scratch = torch.empty(3, T, H, device=DEV)
    partials = torch.empty(3, T, (H + 127) // 128, device=DEV)
    gcnt = torch.zeros(3 * T, dtype=torch.int32, device=DEV)
    x16 = torch.empty(T, H, dtype=torch.float16, device=DEV)
    logits = torch.empty(T, E, device=DEV)
    tw = torch.empty(T, K, device=DEV)
    ti = torch.empty(T, K, dtype=torch.int32, device=DEV)
    sid = torch.empty(max_padded, dtype=torch.int32, device=DEV)
    eid = torch.empty(max_blocks, dtype=torch.int32, device=DEV)
    pp = torch.empty(1, dtype=torch.int32, device=DEV)
    rcnt = torch.zeros(n_counters, dtype=torch.int32, device=DEV)
    gu = torch.empty(T * K, 2 * inter, dtype=torch.float16, device=DEV)
    act = torch.empty(T * K, inter, dtype=torch.float16, device=DEV)
    z1 = torch.ones(T * K * 2 * inter, dtype=torch.float16, device=DEV)

    def visit():
        ops.inkling2_glue_attn_res(blk, 4, hid, qry, raw, x1=x1, partial_out=hsum)
        ops.inkling2_glue_msg_gate_down(raw, msgs[:, :3], wd, d, EPS)
        ops.inkling2_glue_msg_gate_mix(
            raw, msgs[:, :3], d, wf, wg, out, norm_w=nw, scratch=scratch,
            partials=partials, counters=gcnt,
        )  # fmt: skip
        x16.copy_(out[0])  # stands in for the attention / mlp stages
        ops.inkling2_glue_moe_route(
            x16, gw, bias, logits, tw, ti, sid, eid, pp, rcnt, BM, 2.5, zero1=z1
        )
        ops.inkling2_glue_swiglu(gu, act, 4.0, 25.0)

    visit()  # warm up (also loads the kernels outside the capture)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        visit()
    for step in range(3):
        hid.copy_(_randn(g, T, H).to(DEV))
        x1.copy_(_randn(g, T, H).to(DEV))
        msgs.copy_(_randn(g, T, 4, H).to(DEV))
        gu.copy_(_randn(g, T * K, 2 * inter, mag=5.0).to(torch.float16).to(DEV))
        z1.fill_(1.0)
        graph.replay()
        torch.accelerator.synchronize()
        # eager run on the same inputs must match the replay bit for bit
        snap = [t.clone() for t in (out, ti, tw, sid, eid, act, hsum)]
        z1.fill_(1.0)
        visit()
        torch.accelerator.synchronize()
        for a, b2, nm in zip(
            snap,
            (out, ti, tw, sid, eid, act, hsum),
            ["out", "ti", "tw", "sid", "eid", "act", "hsum"],
        ):
            assert torch.equal(a, b2), f"replay {step}: {nm} differs from eager"
        assert not z1.any()
        assert not gcnt.any() and not rcnt.any()
        ref_s, ref_e, ref_p = ref_align(ti.cpu(), E, BM)
        assert torch.equal(sid.cpu(), ref_s) and torch.equal(eid.cpu(), ref_e)
        check_f16(act, ref_swiglu(gu.cpu(), 4.0, 25.0), "graph swiglu")
