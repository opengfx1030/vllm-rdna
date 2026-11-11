# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp RDNA2 MonoKernel parts against today's op sequences (gfx10x).

K1 / K2 at widths 1-24 (eager and graph replay, NaN fail-stop on a sticky
error), the fused GDN decode core (output, conv state, ssm state; non-spec and
MTP-verify layouts), and the QSA part (bounded indexer, HIP sparse decode,
fused norm + RoPE, 3-row store). Ported from tools/rdna/mono/{check_op,
k1_bench,gdn_fused_bench,qsa_bench,qsa_normrope_check,qsa_store3_check}.py,
same tolerances. Shapes are one Flash-Next TP4 / EP4 rank.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


def _gfx10x_with_ops() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        return False
    from vllm.models.qwen4_exp.amd import mono_decode_rdna2 as md

    return md.ops_refusal() is None


pytestmark = pytest.mark.skipif(
    not _gfx10x_with_ops(), reason="needs gfx10x with the MonoKernel ops built"
)

DEV = "cuda"

# ---- K1 / K2 -------------------------------------------------------------

H, HC, R, ND, E, EL, TOPK, INTER, ISE, GROUP = (
    2560,
    4,
    320,
    336,
    512,
    128,
    10,
    640,
    160,
    128,
)
EPS = 1e-6
TIE_GAP = 1e-3  # fp32 routing near-tie: top-10/11 logit gap below this


def _make_layer(gen, keep_golden):
    """Random weights in the layouts the kernels read; (kernel dict, golden)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _synthesize_rdna3_qzeros,
    )

    def rn(*s, std):
        return torch.randn(*s, generator=gen, device=DEV) * std

    L = {
        "norm_w": rn(HC * H, std=0.1).half(),
        "wd": rn(ND, HC * H, std=(HC * H) ** -0.5).half(),
        "wu": rn(HC * H, R, std=R**-0.5 * 2).half(),
        "wr": rn(E, H, std=H**-0.5 * 2).half(),
        "se_w1": rn(2 * ISE, H, std=H**-0.5).half(),
        "se_w2": rn(H, ISE, std=ISE**-0.5).half(),
        "se_wg": rn(H, std=H**-0.5).half(),
    }

    def q4(K, N):
        q = torch.randint(
            0, 16, (EL, K, N), generator=gen, device=DEV, dtype=torch.int32
        )
        s = (
            torch.rand(EL, K // GROUP, N, generator=gen, device=DEV) * 0.01 + 0.005
        ).half()
        packed = torch.zeros(EL, K // 8, N, dtype=torch.int32, device=DEV)
        for j in range(8):
            packed |= (q[:, j::8, :] & 0xF) << (4 * j)
        for e in range(EL):
            pe = packed[e].contiguous()
            ops.gptq_shuffle(pe, 4)
            packed[e] = pe
        z = _synthesize_rdna3_qzeros(K // GROUP, N, DEV)
        z = z.unsqueeze(0).expand(EL, -1, -1).contiguous()
        dense = None
        if keep_golden:
            dense = (q.float() - 8.0) * s.float().repeat_interleave(GROUP, dim=1)
        return packed, s.contiguous(), z, dense

    L["w13"], L["s13"], L["z13"], g13 = q4(H, 2 * INTER)
    L["w2"], L["s2"], L["z2"], g2 = q4(INTER, H)
    G = {
        k: v.float()
        for k, v in L.items()
        if v.dtype == torch.half and k not in ("s13", "s2")
    }
    G["w13"], G["w2"] = g13, g2
    return L, G


def _golden_gaps(G, res, blk, inj):
    """fp32 router logit gap between the 10th and 11th expert per token."""
    res, blk, inj = res.float(), blk.float(), inj.float()
    M = res.shape[0]
    ro = (
        res.view(M, HC, H) + blk[:, None, :] * (2 * torch.sigmoid(inj / HC))[:, :, None]
    )
    xn = ro * torch.rsqrt(ro.pow(2).mean(-1, keepdim=True) + EPS)
    xn = (xn * (1 + G["norm_w"].view(HC, H))).reshape(M, HC * H)
    dai = xn @ G["wd"].t()
    gate = F.silu(dai[:, :R] / HC) @ G["wu"].t()
    b = (torch.sigmoid(gate).view(M, HC, H) * xn.view(M, HC, H)).mean(1)
    srt = (b @ G["wr"].t()).sort(-1, descending=True).values
    return (srt[:, TOPK - 1] - srt[:, TOPK]).tolist()


def _today_ffn(L, res, blk, inj, *, return_routes=False):
    """Today's decode FFN block on one EP rank (expert base 0)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers import rdna_ops  # noqa: F401
    from vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe import (
        _resolve_block_size_m,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

    M = res.shape[0]
    emap = torch.full((E,), -1, dtype=torch.int32, device=DEV)
    emap[:EL] = torch.arange(EL, dtype=torch.int32, device=DEV)
    res_out, xn = hc_combine_norm(res, blk, inj, L["norm_w"], EPS, HC)
    b, dai = torch.ops.vllm.rdna_hc_mix(
        xn, L["wd"], None, None, L["wu"], None, None, R, HC
    )
    inj_out = dai[:, R : R + HC].contiguous()
    se = torch.ops.vllm.rdna_shared_expert(
        b, L["se_w1"], None, None, L["se_w2"], None, None, L["se_wg"]
    )
    logits = rocm_unquantized_gemm_impl(b, L["wr"], None)
    tw, tids, _ = fused_topk(b, logits, TOPK, True)
    bsm = _resolve_block_size_m(M)
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        tids, bsm, EL, emap, ignore_invalid_experts=True
    )
    w1_out = torch.zeros(M * TOPK, 2 * INTER, dtype=torch.half, device=DEV)
    twf = tw.reshape(-1).float()
    ops.moe_gptq_gemm_rdna2(
        b,
        w1_out,
        L["w13"],
        L["s13"],
        L["z13"],
        twf.new_empty(0),
        sorted_ids,
        expert_ids,
        ntp,
        TOPK,
        bsm,
        False,
        0,
    )
    g, u = w1_out.chunk(2, dim=-1)
    routed = torch.zeros(M, H, dtype=torch.half, device=DEV)
    ops.moe_gptq_gemm_rdna2(
        F.silu(g) * u,
        routed,
        L["w2"],
        L["s2"],
        L["z2"],
        twf,
        sorted_ids,
        expert_ids,
        ntp,
        1,
        bsm,
        True,
        TOPK,
    )
    result = (routed + se, res_out, inj_out)
    if return_routes:
        return (*result, tids, tw)
    return result


@pytest.fixture(scope="module")
def ffn_layer():
    gen = torch.Generator(device=DEV).manual_seed(11)
    L, G = _make_layer(gen, keep_golden=True)
    nb = int(torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(H, R, ND, E, ISE, INTER))
    ws = torch.zeros(nb + 8192, dtype=torch.uint8, device=DEV)
    mono = SimpleNamespace(
        t=L,
        ws=ws,
        base=0,
        eps=EPS,
        th=L,
        eps_a=EPS,
        is_mtp=False,
        dense_mtp=False,
    )
    return L, G, mono


def _inputs(M, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (
        torch.randn(M, HC * H, generator=g, device=DEV).half(),
        (torch.randn(M, H, generator=g, device=DEV) * 0.5).half(),
        torch.randn(M, HC, generator=g, device=DEV).half(),
    )


def _check_k2(L, G, got, res, blk, inj):
    ro, out, io = got  # k2.run: (residual, partial FFN output, injection)
    o_ref, ro_ref, io_ref = _today_ffn(L, res, blk, inj)
    torch.accelerator.synchronize()
    rowd = (out.float() - o_ref.float()).abs().max(1).values
    gaps = _golden_gaps(G, res, blk, inj)
    # rows that differ must be fp32 routing near-ties (top-10/11 gap < 1e-3)
    bad = [m for m in range(res.shape[0]) if rowd[m] >= 5e-3 and gaps[m] >= TIE_GAP]
    assert not bad, f"rows {bad}: max|out-ref| {rowd.max().item():.2e}"
    assert torch.equal(ro, ro_ref)
    assert (io.float() - io_ref.float()).abs().max().item() < 5e-3


@pytest.mark.parametrize("M", range(1, 25))
def test_k2_matches_today(ffn_layer, M):
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    L, G, mono = ffn_layer
    res, blk, inj = _inputs(M, 100 + M)
    _check_k2(L, G, k2.run(mono, res, blk, inj), res, blk, inj)
    # graph capture + replay with new inputs copied in
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        k2.run(mono, res, blk, inj)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        got = k2.run(mono, res, blk, inj)
    for t, n in zip((res, blk, inj), _inputs(M, 200 + M)):
        t.copy_(n)
    g.replay()
    _check_k2(L, G, got, res, blk, inj)
    err = torch.ops._rocm_C.mono_k2_rdna2_error(mono.ws).tolist()
    assert err[0] == 0, f"sticky record {err[:3]}"


def test_mtp_control_reset_is_captured_and_clears_sticky_error(ffn_layer):
    """MTP's graph memset must clear a previous error before the fused launch."""
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    L, G, mono = ffn_layer
    mtp = SimpleNamespace(
        t=L,
        ws=mono.ws,
        base=mono.base,
        eps=mono.eps,
        is_mtp=True,
        dense_mtp=False,
    )
    res, blk, inj = _inputs(4, 704)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = k2.run(mtp, res, blk, inj)

    # Model a previous failed launch's sticky header. This tests reset/replay,
    # not an actual barrier timeout or worker recovery.
    mono.ws.view(torch.int32)[2].fill_(1)
    torch.accelerator.synchronize()
    graph.replay()
    _check_k2(L, G, got, res, blk, inj)
    err = torch.ops._rocm_C.mono_k2_rdna2_error(mono.ws).tolist()
    assert err[0] == 0, f"MTP replay did not clear the sticky record: {err[:3]}"


@pytest.fixture(scope="module")
def router_boundary_layer():
    """Fixed fixture used to guard the quantized-target router FP16 boundary."""
    gen = torch.Generator(device=DEV).manual_seed(50124)
    layer, _ = _make_layer(gen, keep_golden=False)
    return layer


@pytest.mark.parametrize(
    ("M", "first_delta", "second_delta"),
    ((3, 200_000, 100_000), (7, 100_000, 200_000), (24, 100_000, 200_000)),
)
def test_quantized_target_router_half_boundary_matches_native(
    router_boundary_layer, M, first_delta, second_delta
):
    """The quantized target Mono route IDs must match native FP16 router logits.

    The layer seed and changed input seed deltas pin the historical failures.
    Every graph replay is also checked against a fresh eager candidate and
    today's module path; a near tie is not an exemption for different IDs.
    """
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    L = router_boundary_layer
    ws_bytes = int(
        torch.ops._rocm_C.mono_k2_rdna2_workspace_bytes(H, R, ND, E, ISE, INTER)
    )
    ws = torch.zeros(ws_bytes + 8192, dtype=torch.uint8, device=DEV)
    mono = SimpleNamespace(
        t=L,
        ws=ws,
        base=0,
        eps=EPS,
        is_mtp=False,
        dense_mtp=False,
    )

    def check(candidate, res, blk, inj, label):
        actual, actual_residual, actual_injection, actual_ids, _ = _today_ffn(
            L, res, blk, inj, return_routes=True
        )
        torch.accelerator.synchronize()
        error = torch.ops._rocm_C.mono_k2_rdna2_error(ws).tolist()
        assert error[0] == 0, f"{label}: sticky record {error[:3]}"

        # Native fused_topk and Mono may use different slot ordering; selected
        # expert IDs must match exactly as sets for every row.
        # Ctrl has eight 32-bit words, eight 64-bit timestamps and the
        # wg_done[10][256] 64-bit debug array before topk_id. The standalone
        # C++ topk helper is not registered as a production Torch operator.
        topk_offset = 8 * 4 + 8 * 8 + 10 * 256 * 8
        mono_ids = ws.narrow(0, topk_offset, M * TOPK * 4).view(torch.int32)
        mono_ids = mono_ids.view(M, TOPK).sort(dim=-1).values
        actual_ids = actual_ids.sort(dim=-1).values
        assert torch.equal(mono_ids, actual_ids), (
            f"{label}: router expert IDs differ; "
            f"Mono={mono_ids.tolist()} native={actual_ids.tolist()}"
        )

        got_residual, got_ffn, got_injection = candidate
        max_ffn_error = (got_ffn.float() - actual.float()).abs().max().item()
        assert max_ffn_error < 5e-3, f"{label}: FFN max abs error {max_ffn_error:.6g}"
        assert torch.equal(got_residual, actual_residual), f"{label}: residual differs"
        injection_error = (
            (got_injection.float() - actual_injection.float()).abs().max().item()
        )
        assert injection_error < 5e-3, (
            f"{label}: injection max abs error {injection_error:.6g}"
        )

    # Start with the same base input family as the strict scratch sweep.
    res, blk, inj = _inputs(M, 50124 + M)
    check(k2.run(mono, res, blk, inj), res, blk, inj, "initial eager")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = k2.run(mono, res, blk, inj)

    for delta in (first_delta, second_delta):
        changed = _inputs(M, 50124 + delta + M)
        for dst, src in zip((res, blk, inj), changed, strict=True):
            dst.copy_(src)
        torch.accelerator.synchronize()

        eager_candidate = k2.run(mono, res, blk, inj)
        check(eager_candidate, res, blk, inj, f"fresh eager seed+{delta}")
        graph.replay()
        check(graph_outputs, res, blk, inj, f"graph replay seed+{delta}")
        for graph_tensor, eager_tensor in zip(
            graph_outputs, eager_candidate, strict=True
        ):
            replay_error = (
                (graph_tensor.float() - eager_tensor.float()).abs().max().item()
            )
            assert replay_error < 5e-3, (
                f"graph/eager path mismatch at seed+{delta}: {replay_error:.6g}"
            )


@pytest.mark.parametrize("M", range(1, 25))
def test_k1_matches_today(ffn_layer, M):
    from vllm.model_executor.layers import rdna_ops  # noqa: F401
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k1
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

    L, _, mono = ffn_layer
    res, blk, inj = _inputs(M, 50 + M)

    def today():
        ro, xn = hc_combine_norm(res, blk, inj, L["norm_w"], EPS, HC)
        b, dai = torch.ops.vllm.rdna_hc_mix(
            xn, L["wd"], None, None, L["wu"], None, None, R, HC
        )
        return ro, b, dai[:, R : R + HC]

    def check(got):
        ro, bl, io = got
        ro_r, bl_r, io_r = today()
        torch.accelerator.synchronize()
        assert (bl.float() - bl_r.float()).abs().max().item() < 5e-3
        assert torch.equal(ro, ro_r)
        assert (io.float() - io_r.float()).abs().max().item() < 5e-3

    check(k1.run(mono, res, blk, inj))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        k1.run(mono, res, blk, inj)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        got = k1.run(mono, res, blk, inj)
    for t, n in zip((res, blk, inj), _inputs(M, 150 + M)):
        t.copy_(n)
    g.replay()
    check(got)


def test_mtp_k1_runs_when_dense_k2_is_unsupported(monkeypatch):
    """MTP K1 must remain usable when its dense FFN makes K2 fall back."""
    from vllm import forward_context
    from vllm.model_executor.layers import rdna_ops  # noqa: F401
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k1
    from vllm.models.qwen4_exp.amd.mono.rdna2.layout import MonoLayer
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

    gen = torch.Generator(device=DEV).manual_seed(704)
    L, _ = _make_layer(gen, keep_golden=False)

    def hc():
        return SimpleNamespace(
            hc_count=HC,
            use_combine=True,
            input_mix_weight_down_block_inject=SimpleNamespace(
                weight=L["wd"], weight_i8=None
            ),
            input_mix_weight_up=SimpleNamespace(weight=L["wu"], weight_i8=None),
            hc_norm=SimpleNamespace(weight=L["norm_w"]),
            config=SimpleNamespace(rms_norm_eps=EPS),
        )

    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            gate=object(), experts=SimpleNamespace(quant_method=SimpleNamespace())
        ),
        mlp_hyper_connection=hc(),
        attn_hyper_connection=hc(),
    )
    mono = MonoLayer(layer, is_mtp=True)
    assert not mono.setup()  # Unsupported dense MTP FFN disables K2.
    assert mono.setup_k1()

    context = SimpleNamespace(no_compile_layers={"mtp.layers.0": mono})
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(k1.OPTS, "k1", True)
    monkeypatch.setattr(k1, "MODES", {"eager", "full"})
    is_capturing = torch.cuda.is_current_stream_capturing
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(k1, "mode", lambda: "eager")

    res, blk, inj = _inputs(4, 705)

    def today():
        ro, xn = hc_combine_norm(res, blk, inj, L["norm_w"], EPS, HC)
        b, dai = torch.ops.vllm.rdna_hc_mix(
            xn, L["wd"], None, None, L["wu"], None, None, R, HC
        )
        return ro, b, dai[:, R : R + HC]

    def check(got):
        ro, block_input, injection = got
        ro_ref, block_input_ref, injection_ref = today()
        torch.accelerator.synchronize()
        assert torch.equal(ro, ro_ref)
        assert (block_input.float() - block_input_ref.float()).abs().max().item() < 5e-3
        assert (injection.float() - injection_ref.float()).abs().max().item() < 5e-3

    check(k1._k1(res, blk, inj, "mtp.layers.0"))

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", is_capturing)
    monkeypatch.setattr(k1, "mode", lambda: "full")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = k1._k1(res, blk, inj, "mtp.layers.0")
    for tensor, new in zip((res, blk, inj), _inputs(4, 706)):
        tensor.copy_(new)
    graph.replay()
    check(got)


def test_k2_fail_stop_writes_nan(ffn_layer):
    """A sticky error record makes the next launch write NaN, not stale data."""
    from vllm.models.qwen4_exp.amd.mono.rdna2 import k2

    L, _, mono = ffn_layer
    ws = torch.zeros_like(mono.ws)
    fs = SimpleNamespace(t=L, ws=ws, base=0, eps=EPS, is_mtp=False, dense_mtp=False)
    res, blk, inj = _inputs(3, 7)
    ws[8:12].copy_(torch.tensor([1], dtype=torch.int32).view(torch.uint8))
    out = k2.run(fs, res, blk, inj)[1]
    torch.accelerator.synchronize()
    assert bool(torch.isnan(out).all())


# ---- GDN -----------------------------------------------------------------

GH, GHV, GK, GV = 4, 12, 128, 128
CD = 2 * GH * GK + GHV * GV
VD = GHV * GV
NQ = CD + VD
SLOTS = 96
TSPEC = 3  # MTP2 verify: 1 + 2 drafts per sequence


def _gdn_reference(L, qkvz, ba, conv, ssm, sidx, nacc, qsl, eps, sig):
    """Today's region from the in_proj outputs to the out_proj input."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    M = qkvz.shape[0]
    mixed, z = qkvz.split([CD, VD], dim=-1)
    b, a = (t.contiguous() for t in ba.chunk(2, dim=-1))
    core = torch.zeros(M, GHV, GV, dtype=torch.half, device=DEV)
    if nacc is None:
        mixed = causal_conv1d_update(
            mixed,
            conv,
            L["convw"],
            L["convb"],
            "silu",
            conv_state_indices=sidx,
            validate_data=False,
        )
        torch.ops._rocm_C.gdn_decode_rdna2(
            mixed,
            a,
            b,
            L["a_log"],
            L["dtb"],
            core.unsqueeze(1),
            ssm,
            sidx,
            GK**-0.5,
            True,
        )
    else:
        N = sidx.shape[0]
        mixed = causal_conv1d_update(
            mixed,
            conv,
            L["convw"],
            L["convb"],
            "silu",
            conv_state_indices=sidx[:, 0][:N],
            num_accepted_tokens=nacc,
            query_start_loc=qsl,
            max_query_len=sidx.size(-1),
            validate_data=False,
        )
        q, k, v = torch.split(mixed, [GH * GK, GH * GK, VD], dim=-1)
        q, k, v = (t.contiguous().view(1, M, -1, GK) for t in (q, k, v))
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=L["a_log"],
            a=a,
            b=b,
            dt_bias=L["dtb"],
            q=q,
            k=k,
            v=v,
            initial_state=ssm,
            inplace_final_state=True,
            cu_seqlens=qsl,
            ssm_state_indices=sidx,
            num_accepted_tokens=nacc,
            use_qk_l2norm_in_kernel=True,
        )
        core.copy_(o.view(M, GHV, GV))
    x2 = core.reshape(-1, GV).contiguous()
    y = torch.empty_like(x2)
    torch.ops._rocm_C.gated_rms_norm(
        y, x2, z.reshape(M, GHV, GV).reshape(-1, GV).contiguous(), L["gnw"], eps, sig
    )
    return y.view(M, VD)


GDN_CASES = [(False, M) for M in (1, 3, 6, 12, 24)] + [
    (True, M) for M in (3, 6, 12, 24)
]


@pytest.mark.parametrize("layout", ["DS", "SD"])
@pytest.mark.parametrize("spec, M", GDN_CASES)
@pytest.mark.parametrize("sig", [0, 1])
def test_gdn_fused_matches_today(layout, spec, M, sig):
    gen = torch.Generator(device=DEV).manual_seed(100 + M + 7 * spec + sig)

    def rn(*s, std):
        return (torch.randn(*s, generator=gen, device=DEV) * std).half()

    L = dict(
        convw=rn(CD, 4, std=0.3),
        convb=rn(CD, std=0.1),
        gnw=(1.0 + rn(GV, std=0.1).float()).half(),
        a_log=(torch.rand(GHV, generator=gen, device=DEV) * 2.0 - 1.0).float(),
        dtb=(torch.randn(GHV, generator=gen, device=DEV) * 0.5).float(),
    )
    qkvz = rn(M, NQ, std=0.5)
    ba = rn(M, 2 * GHV, std=1.0)
    empty = torch.empty(0, dtype=torch.int32, device=DEV)
    if not spec:
        sidx = (torch.randperm(SLOTS - 1, generator=gen, device=DEV)[:M] + 1).int()
        if M >= 3:
            sidx[1] = 0  # null slot
        nacc = qsl = None
    else:
        N = M // TSPEC
        sidx = torch.randperm(SLOTS - 1, generator=gen, device=DEV)[: N * TSPEC] + 1
        sidx = sidx.int().view(N, TSPEC).contiguous()
        nacc = torch.randint(1, TSPEC + 1, (N,), generator=gen, device=DEV).int()
        if N >= 2:
            sidx[1] = 0  # padded sequence
        qsl = torch.arange(0, M + 1, TSPEC, device=DEV, dtype=torch.int32)
    conv_len = 3 + (TSPEC - 1 if spec else 0)
    conv = rn(SLOTS, CD, conv_len, std=0.5)
    if layout == "SD":  # (slot, len, dim) storage, (slot, dim, len) view
        conv = conv.transpose(1, 2).contiguous().transpose(1, 2)
    ssm = (torch.randn(SLOTS, GHV, GV, GK, generator=gen, device=DEV) * 0.1).float()
    conv_r, ssm_r = conv.clone(), ssm.clone()
    eps = 1e-6
    y_r = _gdn_reference(L, qkvz.clone(), ba, conv_r, ssm_r, sidx, nacc, qsl, eps, sig)
    scratch = torch.zeros(
        int(torch.ops._rocm_C.mono_gdn_rdna2_scratch_bytes(64, GHV, GH)),
        dtype=torch.uint8,
        device=DEV,
    )
    y = torch.empty(M, VD, dtype=torch.half, device=DEV)
    torch.ops._rocm_C.mono_gdn_rdna2(
        qkvz,
        ba,
        L["convw"],
        L["convb"],
        conv,
        ssm,
        L["a_log"],
        L["dtb"],
        L["gnw"],
        sidx,
        empty if nacc is None else nacc,
        empty if qsl is None else qsl,
        y,
        scratch,
        GH,
        GK**-0.5,
        eps,
        sig,
    )
    torch.accelerator.synchronize()
    if spec:
        assert nacc is not None
        valid = sidx[torch.arange(M // TSPEC, device=DEV), nacc - 1] > 0
        valid = valid.repeat_interleave(TSPEC)
    else:
        valid = sidx > 0
    d_y = (y[valid].float() - y_r[valid].float()).abs().max().item()
    assert d_y / max(y_r[valid].float().abs().max().item(), 1e-6) < 1e-2
    if (~valid).any():
        assert bool((y[~valid] == 0).all())
    assert (conv[1:].float() - conv_r[1:].float()).abs().max().item() < 1e-3
    assert (ssm[1:] - ssm_r[1:]).abs().max().item() < 1e-3
    assert bool((scratch[: 64 * (GHV * 12 + GH * 4)] == 0).all())  # counters reset


# ---- QSA -----------------------------------------------------------------

MAXLEN, RATIO, QTOPK, PAGE = 262144, 4, 2048, 64
IH, ID = 4, 128  # indexer heads / dim
QH, KVH, HD = 6, 1, 256  # attention heads per rank, kv heads, head dim


def _qsa_build(gen, n_req, ctx):
    live_c = (ctx // RATIO + PAGE - 1) // PAGE
    live_k = (ctx + PAGE - 1) // PAGE
    ccache = (
        torch.randn(n_req * live_c + 1, PAGE, 1, ID, generator=gen, device=DEV) * 0.5
    ).half()
    kv = (
        torch.randn(n_req * live_k + 1, PAGE, 2, KVH, HD, generator=gen, device=DEV)
        * 0.5
    ).half()
    ctab = torch.full(
        (n_req, MAXLEN // RATIO // PAGE), -1, dtype=torch.int32, device=DEV
    )
    ktab = torch.full((n_req, MAXLEN // PAGE), -1, dtype=torch.int32, device=DEV)
    for r in range(n_req):
        ctab[r, :live_c] = torch.arange(
            1 + r * live_c, 1 + (r + 1) * live_c, device=DEV
        )
        ktab[r, :live_k] = torch.arange(
            1 + r * live_k, 1 + (r + 1) * live_k, device=DEV
        )
    return ccache, kv, ctab, ktab


@pytest.mark.parametrize("ctx", [1024, 16384])
@pytest.mark.parametrize("M", [1, 3, 8, 24])
def test_qsa_decode_matches_triton(monkeypatch, ctx, M):
    """Bounded indexer: same selected set (host bound and in-graph early exit);
    all-padding tile skip: same output; HIP sparse decode vs Triton."""
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as mq
    from vllm.models.qwen4_exp.amd.ops import qsa as Q

    gen = torch.Generator(device=DEV).manual_seed(5 + M + ctx)
    T = 3 if M % 3 == 0 and M > 1 else 1  # verify rows per sequence
    n_req = M // T
    ccache, kv, ctab, ktab = _qsa_build(gen, n_req, ctx)
    seq = torch.full((n_req,), ctx, dtype=torch.int32, device=DEV)
    t2r = torch.arange(n_req, device=DEV, dtype=torch.int32).repeat_interleave(T)
    pos = (ctx - T + torch.arange(M, device=DEV) % T).long()
    qi = (torch.randn(M, IH, ID, generator=gen, device=DEV) * 0.5).half()
    qa = (torch.randn(M, QH, HD, generator=gen, device=DEV) * 0.5).half()
    kc, vc = kv[:, :, 0], kv[:, :, 1]
    out_w = QTOPK + RATIO - 1

    def select(on, host_bound=True):
        monkeypatch.setattr(Q, "MONO_QSA", mq if on else None)
        return Q.qsa_select_paged_tokens(
            qi,
            ccache,
            ctab,
            t2r,
            pos,
            seq,
            QTOPK,
            RATIO,
            torch.empty(M, out_w, dtype=torch.int32, device=DEV),
            max_seq_len=ctx if on and host_bound else None,
        )

    i0, i1, i2 = select(False), select(True), select(True, host_bound=False)
    for i in (i1, i2):
        assert torch.equal(i0.sort(1).values, i.sort(1).values)
    # the Triton attention with and without the all-padding tile skip
    monkeypatch.setattr(Q, "MONO_QSA", None)
    o0 = Q.qsa_sparse_paged_attention(qa, kc, vc, i0, ktab, t2r, torch.empty_like(qa))
    monkeypatch.setattr(Q, "MONO_QSA", SimpleNamespace(sparse_decode=lambda *a: False))
    o1 = Q.qsa_sparse_paged_attention(qa, kc, vc, i1, ktab, t2r, torch.empty_like(qa))
    torch.accelerator.synchronize()
    assert (o0.float() - o1.float()).abs().max().item() < 1e-3
    # the HIP sparse decode (from mq.MIN_ROWS rows)
    oh = torch.empty_like(qa)
    took = mq.sparse_decode(qa, kc, vc, QH // KVH, i0, ktab, t2r, oh)
    assert took == (M >= mq.MIN_ROWS)
    if took:
        torch.accelerator.synchronize()
        rel = (o0.float() - oh.float()).abs().max().item() / max(
            o0.float().abs().max().item(), 1e-6
        )
        assert rel < 5e-3
        assert bool((mq._COUNTERS[torch.accelerator.current_device_index()] == 0).all())


def test_mono_qsa_sparse_decode_precedes_the_hip_gate(monkeypatch):
    """An installed mono part takes the fused decode ahead of the generic HIP
    sparse attention (both HIP); a mono refusal still falls through to it."""
    from vllm.models.qwen4_exp.amd.ops import qsa as Q

    gen = torch.Generator(device=DEV).manual_seed(31)
    _ccache, kv, _ctab, ktab = _qsa_build(gen, 1, 1024)
    M = 3
    q = (torch.randn(M, QH, HD, generator=gen, device=DEV) * 0.5).half()
    kc, vc = kv[:, :, 0], kv[:, :, 1]
    idx = torch.zeros(M, QTOPK + RATIO - 1, dtype=torch.int32, device=DEV)
    t2r = torch.zeros(M, dtype=torch.int32, device=DEV)

    mono_calls: list[int] = []
    hip_calls: list[int] = []
    mono_take = True

    def mono_sparse_decode(_q, _k, _v, group_size, *_args):
        mono_calls.append(group_size)
        return mono_take

    def hip_sparse(*args):
        hip_calls.append(1)
        return args[-1]

    monkeypatch.setattr(Q.qsa_rdna2, "qsa_use_rdna2", lambda dtype=None: True)
    monkeypatch.setattr(Q.qsa_rdna2, "qsa_sparse_paged_attention", hip_sparse)
    monkeypatch.setattr(
        Q, "MONO_QSA", SimpleNamespace(sparse_decode=mono_sparse_decode)
    )

    Q.qsa_sparse_paged_attention(q, kc, vc, idx, ktab, t2r, torch.empty_like(q))
    assert mono_calls == [QH // KVH]
    assert hip_calls == []

    mono_take = False
    Q.qsa_sparse_paged_attention(q, kc, vc, idx, ktab, t2r, torch.empty_like(q))
    assert mono_calls == [QH // KVH, QH // KVH]
    assert hip_calls == [1]


FLASH_NEXT_ROPE = dict(
    head_size=256,
    max_position=262144,
    rope_parameters={
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
)


@pytest.mark.parametrize("heads", [4, 1])
@pytest.mark.parametrize("n", [1, 3, 8, 24, 2048])
def test_qsa_norm_rope_matches_today(default_vllm_config, heads, n):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.amd.indexer_qsa import apply_qsa_rmsnorm, apply_qsa_rope
    from vllm.models.qwen4_exp.amd.mono.rdna2.qsa import norm_rope

    torch.manual_seed(heads * 10000 + n)
    rope = get_rope(**FLASH_NEXT_ROPE, dtype=torch.float16).to(DEV)
    norm = GemmaRMSNorm(128, eps=1e-6).to(device=DEV, dtype=torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device=DEV) * 0.1)
    x = torch.randn(n, heads, 128, device=DEV, dtype=torch.float16)
    pos = torch.randint(0, 20000, (3, n), device=DEV)
    ref = apply_qsa_rope(
        rope, pos, apply_qsa_rmsnorm(norm, x.reshape(-1, 128)).reshape_as(x)
    )
    fused = norm_rope(norm, rope, pos, x)
    assert fused is not None
    assert (ref.float() - fused.float()).abs().max().item() < 4e-3


@pytest.mark.parametrize("backend", ["hip", "triton"])
@pytest.mark.parametrize("n", [1, 3, 24, 2048])
def test_qsa_store_rows3_is_bitwise_three_stores(n, backend, monkeypatch):
    """One launch (HIP mono_qsa_store3_rdna2, or the Triton fallback) leaves the
    three indexer caches bit-identical to three qsa_store_cache_rows calls, PAD
    (-1) slots included, with the serve's layouts (rope rows int64 in a view of
    an fp16 page)."""
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as mono_qsa
    from vllm.models.qwen4_exp.amd.mono.rdna2.qsa import store_rows3
    from vllm.models.qwen4_exp.amd.ops import qsa_rdna2
    from vllm.models.qwen4_exp.amd.ops.qsa import qsa_store_cache_rows

    if backend == "hip":
        if not mono_qsa._has_hip_store3():
            pytest.skip("mono_qsa_store3_rdna2 not built")
    else:
        monkeypatch.setattr(mono_qsa, "_has_hip_store3", lambda: False)
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", False)

    g = torch.Generator(device=DEV).manual_seed(3 + n)
    pages, page = 64, 64
    comp = torch.randn(pages, page, 1, 128, generator=g, device=DEV).half()
    raw = torch.randn(pages, page, 1, 128, generator=g, device=DEV).half()
    rope_store = torch.zeros(pages, page, 1, 12, device=DEV, dtype=torch.float16)
    c_slots = torch.randperm(pages * page, generator=g, device=DEV)[:n].long()
    c_slots[::3] = -1  # most tokens close no group
    r_slots = torch.randperm(pages * page, generator=g, device=DEV)[:n].long()
    crow = torch.randn(n, 1, 128, generator=g, device=DEV).half()
    rrow = torch.randn(n, 1, 128, generator=g, device=DEV).half()
    prow = torch.randint(0, 1 << 40, (3, n), generator=g, device=DEV).t().unsqueeze(1)
    a = [t.clone() for t in (comp, raw, rope_store)]
    qsa_store_cache_rows(a[0], c_slots, crow)
    qsa_store_cache_rows(a[1], r_slots, rrow)
    qsa_store_cache_rows(a[2].view(torch.int64)[..., :3], r_slots, prow)
    b = [t.clone() for t in (comp, raw, rope_store)]
    assert store_rows3(
        b[0], c_slots, crow, b[1], r_slots, rrow, b[2].view(torch.int64)[..., :3], prow
    )
    torch.accelerator.synchronize()
    for x, y in zip(a, b):  # bitwise: int64 rows read as fp16 can be NaN patterns
        assert torch.equal(x.view(torch.int16), y.view(torch.int16))


@pytest.mark.parametrize("n", [1, 3, 24, 2048])
def test_qsa_norm_rope_store3_matches_two_launches(default_vllm_config, n):
    """mono_qsa_k_store_rdna2 (norm+rope fused with the three indexer stores)
    leaves the caches as norm_rope + store_rows3 would: the compressed keys to
    rounding (summation order), the raw and RoPE rows bitwise, PAD (-1)
    compressed slots untouched, with the serve's layouts (first positions
    [n, 3] int64, rope rows int64 in a view of an fp16 page)."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as mono_qsa

    if not mono_qsa._has_hip_k_store():
        pytest.skip("mono_qsa_k_store_rdna2 not built")
    g = torch.Generator(device=DEV).manual_seed(11 + n)
    rope = get_rope(**FLASH_NEXT_ROPE, dtype=torch.float16).to(DEV)
    norm = GemmaRMSNorm(128, eps=1e-6).to(device=DEV, dtype=torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, generator=g, device=DEV) * 0.1)
    pages, page = 64, 64
    comp = torch.randn(pages, page, 1, 128, generator=g, device=DEV).half()
    raw = torch.randn(pages, page, 1, 128, generator=g, device=DEV).half()
    rope_store = torch.zeros(pages, page, 1, 12, device=DEV, dtype=torch.float16)
    c_slots = torch.randperm(pages * page, generator=g, device=DEV)[:n].long()
    c_slots[::3] = -1  # most tokens close no group
    r_slots = torch.randperm(pages * page, generator=g, device=DEV)[:n].long()
    pooled = torch.randn(n, 1, 128, generator=g, device=DEV).half()
    first = torch.randint(0, 20000, (n, 3), generator=g, device=DEV)
    rrow = torch.randn(n, 1, 128, generator=g, device=DEV).half()
    prow = torch.randint(0, 1 << 40, (3, n), generator=g, device=DEV).t().unsqueeze(1)

    a = [t.clone() for t in (comp, raw, rope_store)]
    normalized = mono_qsa.norm_rope(norm, rope, first.transpose(0, 1), pooled)
    assert normalized is not None
    assert mono_qsa.store_rows3(
        a[0],
        c_slots,
        normalized,
        a[1],
        r_slots,
        rrow,
        a[2].view(torch.int64)[..., :3],
        prow,
    )
    b = [t.clone() for t in (comp, raw, rope_store)]
    assert mono_qsa.norm_rope_store3(
        norm,
        rope,
        first,
        pooled,
        b[0],
        c_slots,
        b[1],
        r_slots,
        rrow,
        b[2].view(torch.int64)[..., :3],
        prow,
    )
    torch.accelerator.synchronize()
    assert (a[0].float() - b[0].float()).abs().max().item() < 4e-3
    untouched = torch.ones(pages * page, dtype=torch.bool, device=DEV)
    untouched[c_slots[c_slots >= 0]] = False
    assert torch.equal(b[0].view(-1, 128)[untouched], comp.view(-1, 128)[untouched])
    for x, y in zip(a[1:], b[1:]):
        assert torch.equal(x.view(torch.int16), y.view(torch.int16))


# (waves, tile slots, lanes per key, unroll, q in LDS): the policy's shapes
# (strided small-tile, one tile per block with whole-row loads), a wide-unroll
# one and the LDS-q group kernel
_MQA_SHAPES = [
    (4, 8, 4, 1, False),
    (8, 0, 16, 2, False),
    (4, 0, 8, 4, False),
    (8, 0, 8, 1, True),
]


@pytest.mark.parametrize("group", [1, 2, 3, 4, None])
@pytest.mark.parametrize("shape", _MQA_SHAPES)
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("rows", [1, 3, 8, 24, 130])
def test_qsa_mqa_matches_triton(group, shape, width, rows):
    """mono_qsa_mqa_rdna2 (HIP indexer scoring, early exit) gives the Triton
    kernel's visible_blocks and its logits on [0, visible), for any group
    size and kernel shape (group None: mqa_config's own choice) against any
    batch layout: runs of `width` rows per request at consecutive positions,
    ragged contexts, an unmapped page (-1) and a row with no request."""
    import math

    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as mono_qsa
    from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops

    if not mono_qsa._has_hip_mqa():
        pytest.skip("mono_qsa_mqa_rdna2 not built")
    g = torch.Generator(device=DEV).manual_seed(rows * 13 + width * 5 + (group or 0))
    ratio, page, cap = 4, 64, 4096
    reqs = (rows + width - 1) // width
    ctx = torch.randint(1, cap * ratio, (reqs,), generator=g, device=DEV)
    live = (ctx // ratio + page - 1) // page
    table = torch.full((reqs, cap // page), -1, dtype=torch.int32, device=DEV)
    used = 1
    for r in range(reqs):
        n = int(live[r])
        table[r, :n] = torch.arange(used, used + n, device=DEV)
        used += n
    table[0, 0] = -1  # an unmapped page
    k_cache = torch.randn(used, page, 1, 128, generator=g, device=DEV).half()
    q = torch.randn(rows, 4, 128, generator=g, device=DEV).half()
    t2r = (torch.arange(rows, device=DEV) // width).int()
    if rows > 2:
        t2r[-1] = -1  # a row with no request
    off = width - 1 - (torch.arange(rows, device=DEV) % width)
    pos = (ctx[t2r.clamp(min=0).long()] - 1 - off).clamp(min=0)
    seq = ctx.int()

    old = qsa_ops.MONO_QSA
    try:
        qsa_ops.MONO_QSA = SimpleNamespace(mqa_paged=lambda *a: False)
        ref_l, ref_v = qsa_ops.qsa_mqa_paged(
            q, k_cache, table, t2r, pos, seq, ratio, num_columns=cap
        )
        if group is None:
            if shape != _MQA_SHAPES[0]:
                pytest.skip("the policy picks its own shape")
            mono_qsa.DECODE_WIDTH = width
        else:
            waves, tiles, lanes, unroll, q_lds = shape
            if q_lds and group == 1:
                pytest.skip("LDS q is built for groups of 2-4")
            mono_qsa.FORCE = (group, waves, tiles, max(lanes, group), unroll, q_lds)
        qsa_ops.MONO_QSA = mono_qsa
        got_l, got_v = qsa_ops.qsa_mqa_paged(
            q, k_cache, table, t2r, pos, seq, ratio, num_columns=cap
        )
    finally:
        qsa_ops.MONO_QSA = old
        mono_qsa.FORCE = None
        mono_qsa.DECODE_WIDTH = 1
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=0)
    scale = 1.0 / math.sqrt(128)
    for r in range(rows):
        n = int(ref_v[r])
        torch.testing.assert_close(
            got_l[r, :n], ref_l[r, :n], rtol=2e-3, atol=2e-3 * scale * 16
        )


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("n_req", [1, 5, 8])
def test_qsa_epilogue_matches_today(default_vllm_config, width, n_req):
    """mono_qsa_epilogue_rdna2 (one launch) leaves q, the compressed key
    cache and the raw-key / RoPE-position ring as today's chain does
    (norm_rope q, qsa_compress_groups, norm_rope k, store_rows3): q and the
    compressed keys to rounding, the pooled input and both rings bitwise,
    with the serve's ring (key + int64 RoPE tail in one fp16 page, size
    ratio * cdiv(ratio + nspec, ratio)), MRoPE positions, groups whose
    members come from the ring and from this step's rows, and PAD compressed
    slots."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa as mono_qsa
    from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
    from vllm.models.qwen4_exp.common.qsa_cache import canonical_qsa_rope_positions

    if not mono_qsa._has_hip_epilogue():
        pytest.skip("mono_qsa_epilogue_rdna2 not built")
    g = torch.Generator(device=DEV).manual_seed(7 * n_req + width)
    ratio = 4
    ring = ratio * -(-(ratio + width - 1) // ratio)
    rope = get_rope(**FLASH_NEXT_ROPE, dtype=torch.float16).to(DEV)
    qn = GemmaRMSNorm(128, eps=1e-6).to(device=DEV, dtype=torch.float16)
    kn = GemmaRMSNorm(128, eps=1e-6).to(device=DEV, dtype=torch.float16)
    with torch.no_grad():
        qn.weight.copy_(torch.randn(128, generator=g, device=DEV) * 0.1)
        kn.weight.copy_(torch.randn(128, generator=g, device=DEV) * 0.1)

    n = n_req * width
    ctx = torch.randint(ratio + width, 5000, (n_req,), generator=g, device=DEV)
    ctx[0] = ratio * 37  # request 0's last row closes a group
    t2r = (torch.arange(n, device=DEV) // width).int()
    off = width - 1 - (torch.arange(n, device=DEV) % width)
    end = (ctx[t2r.long()] - 1 - off).long()
    qsl = torch.arange(0, n + 1, width, device=DEV).int()
    blocks = 2 * n_req + 1
    state_table = torch.randperm(blocks, generator=g, device=DEV)[:n_req]
    state_table = state_table.int().view(-1, 1)
    r_slots = state_table[t2r.long(), 0].long() * ring + end % ring
    pages, page = 64, 64
    c_slots = torch.randperm(pages * page, generator=g, device=DEV)[:n].long()
    c_slots[(end + 1) % ratio != 0] = -1  # tokens that close no group
    # [3, n] MRoPE positions: t = logical position, h / w offset
    pos = torch.stack((end, end + 3, end + 7))

    kv = torch.randn(blocks, ring, 1, 140, generator=g, device=DEV).half()
    rope_ring = kv[..., 128:].view(torch.int64)
    rope_ring.copy_(torch.randint(0, 100000, rope_ring.shape, generator=g, device=DEV))
    comp = torch.randn(pages, page, 1, 128, generator=g, device=DEV).half()
    qk = torch.randn(n, 640, generator=g, device=DEV).half()
    raw_md = SimpleNamespace(
        block_table=state_table,
        token_to_req=t2r,
        query_start_loc=qsl,
        logical_positions=end,
        slot_mapping=r_slots,
        max_query_len=width,
    )
    comp_md = SimpleNamespace(slot_mapping=c_slots)

    # today's chain
    kv_a, comp_a = kv.clone(), comp.clone()
    q_a = mono_qsa.norm_rope(qn, rope, pos, qk[:, :512].view(n, 4, 128))
    token_k = qk[:, 512:].reshape(n, 1, 128)
    rows = canonical_qsa_rope_positions(pos)
    pooled, first = qsa_ops.qsa_compress_groups_with_ratio(
        token_k,
        rows,
        kv_a[..., :128],
        state_table,
        t2r,
        qsl,
        end,
        c_slots,
        ratio,
        kv_a[..., 128:].view(torch.int64),
    )
    normed = mono_qsa.norm_rope(kn, rope, first.transpose(0, 1), pooled)
    old = mono_qsa._has_hip_store3
    try:
        mono_qsa._has_hip_store3 = lambda: False
        assert mono_qsa.store_rows3(
            comp_a,
            c_slots,
            normed,
            kv_a[..., :128],
            r_slots,
            token_k,
            kv_a[..., 128:].view(torch.int64),
            rows,
        )
    finally:
        mono_qsa._has_hip_store3 = old

    # one launch
    kv_b, comp_b = kv.clone(), comp.clone()
    q_b = mono_qsa.epilogue(
        qn,
        kn,
        rope,
        qk,
        pos,
        raw_md,
        comp_md,
        ratio,
        comp_b,
        kv_b[..., :128],
        kv_b[..., 128:].view(torch.int64),
    )
    assert q_b is not None
    torch.testing.assert_close(q_b, q_a, rtol=0, atol=4e-3)
    torch.testing.assert_close(comp_b, comp_a, rtol=0, atol=4e-3)
    assert torch.equal(kv_b.view(torch.int16), kv_a.view(torch.int16))
    assert (c_slots >= 0).any()


@pytest.mark.parametrize("n", [1, 3, 24, 2048])
@pytest.mark.parametrize("kind", ["key", "rope"])
def test_qsa_hip_store_cache_rows_matches_triton(n, kind):
    """The HIP QSA path's store (qsa_store_cache_rows_rdna2, VLLM_RDNA_QSA_HIP=1)
    writes the same cache bytes as the Triton store: dense fp16 key rows, and
    the serve's transposed int64 RoPE rows into a padded fp16 page view, with
    PAD (-1) slots skipped."""
    from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
    from vllm.models.qwen4_exp.amd.ops import qsa_rdna2
    from vllm.models.qwen4_exp.common.qsa_cache import canonical_qsa_rope_positions

    if not hasattr(torch.ops._rocm_C, "qsa_store_cache_rows_rdna2"):
        pytest.skip("qsa_store_cache_rows_rdna2 not built")
    g = torch.Generator(device=DEV).manual_seed(n * 2 + (kind == "rope"))
    pages, page = 64, 64
    slots = torch.randperm(pages * page, generator=g, device=DEV)[:n].long()
    slots[::5] = -1
    if kind == "key":
        store = torch.randn(pages, page, 1, 140, generator=g, device=DEV).half()
        rows = torch.randn(n, 1, 128, generator=g, device=DEV).half()

        def view(t):
            return t[..., :128]
    else:
        store = torch.randn(pages, page, 1, 140, generator=g, device=DEV).half()
        pos = torch.randint(0, 1 << 40, (3, n), generator=g, device=DEV)
        rows = canonical_qsa_rope_positions(pos)  # [n, 1, 3], stride(-1) = n

        def view(t):
            return t[..., 128:].view(torch.int64)

    if qsa_rdna2.qsa_use_rdna2():
        pytest.skip("run with VLLM_RDNA_QSA_HIP unset (Triton reference)")
    a, b = store.clone(), store.clone()
    qsa_ops.qsa_store_cache_rows(view(a), slots, rows)
    qsa_rdna2.qsa_store_cache_rows_compat(view(b), slots, rows)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
