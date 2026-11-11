#!/usr/bin/env python3
"""E1' standalone: mono_gdn_rdna2 (conv update + delta rule + gated RMSNorm,
two launches) against today's GDN decode region of one Qwen4Exp linear-attention
layer on one TP rank, from the in_proj outputs (qkvz, ba) to the out_proj input.

  non-spec  qkvz.split / ba.chunk -> zeros(core) -> causal_conv1d_update ->
            gdn_decode_rdna2 -> RMSNormGated (gated_rms_norm, contiguous copies)
  verify    causal_conv1d_update (num_accepted, query_start_loc) ->
            rearrange -> fused_sigmoid_gating_delta_rule_update -> RMSNormGated

Compares y, the conv state and the ssm state (null slot 0 excluded), then times
both regions in a graph over 16 distinct layers (cold Infinity Cache).

  PYTHONPATH=<tree> python3 gdn_fused_bench.py [--widths 1,3,6,12,24]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_bench as A  # noqa: E402

H, HV, K, V = 4, 12, 128, 128
CD = 2 * H * K + HV * V  # 2560 conv channels
VD = HV * V              # 1536
NQ = CD + VD             # 4096
SLOTS = 96
TSPEC = 3                # MTP2 verify: 1 + 2 drafts per sequence


def build_ext():
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.environ.get("MONO_SRC_DIR", os.path.join(here, "../../../csrc/rocm"))
    return load(
        name="gdn_fused_ext",
        sources=[os.path.join(src, "mono_gdn_rdna2.cu"),
                 os.path.join(here, "gdn_fused_binding.cpp")],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1030", "-ffp-contract=fast"]
        + os.environ.get("GDN_FUSED_CFLAGS", "").split(),
        build_directory=os.environ.get("MONO_BUILD_DIR"),
        verbose=False,
    )


def make_layer(gen, dev):
    def rn(*s, std):
        return (torch.randn(*s, generator=gen, device=dev) * std).half()
    return dict(
        convw=rn(CD, 4, std=0.3), convb=rn(CD, std=0.1),
        gnw=(1.0 + rn(V, std=0.1).float()).half(),
        alog=(torch.rand(HV, generator=gen, device=dev) * 2.0 - 1.0).float(),
        dtb=(torch.randn(HV, generator=gen, device=dev) * 0.5).float(),
    )


def make_states(gen, dev, layout, clen):
    conv = (torch.randn(SLOTS, CD, clen, generator=gen, device=dev) * 0.5).half()
    if layout == "SD":  # (slot, len, dim) storage, (slot, dim, len) view
        conv = conv.transpose(1, 2).contiguous().transpose(1, 2)
    ssm = (torch.randn(SLOTS, HV, V, K, generator=gen, device=dev) * 0.1).float()
    return conv, ssm


def make_inputs(gen, dev, M):
    qkvz = (torch.randn(M, NQ, generator=gen, device=dev) * 0.5).half()
    ba = (torch.randn(M, 2 * HV, generator=gen, device=dev) * 1.0).half()
    return qkvz, ba


def make_meta(gen, dev, M, spec, null_row):
    """non-spec: sidx [M]; verify: sidx [N, 3], nacc [N], qsl [N + 1]."""
    if not spec:
        sidx = (torch.randperm(SLOTS - 1, generator=gen, device=dev)[:M] + 1).int()
        if null_row and M >= 3:
            sidx[1] = 0
        return sidx, None, None
    N = M // TSPEC
    sidx = (torch.randperm(SLOTS - 1, generator=gen, device=dev)[:N * TSPEC] + 1).int()
    sidx = sidx.view(N, TSPEC).contiguous()
    nacc = torch.randint(1, TSPEC + 1, (N,), generator=gen, device=dev).int()
    if null_row and N >= 2:
        sidx[1] = 0  # padded sequence
    qsl = torch.arange(0, M + 1, TSPEC, device=dev, dtype=torch.int32)
    return sidx, nacc, qsl


def reference(L, qkvz, ba, conv, ssm, sidx, nacc, qsl, eps, sig):
    """Today's region from the in_proj outputs to the out_proj input."""
    from vllm import _custom_ops  # noqa: F401
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    M = qkvz.shape[0]
    mixed, z = qkvz.split([CD, VD], dim=-1)
    z = z.reshape(M, HV, V)
    b, a = ba.chunk(2, dim=-1)
    b, a = b.contiguous(), a.contiguous()
    core = torch.zeros(M, HV, V, dtype=torch.half, device=qkvz.device)
    if nacc is None:
        mixed = causal_conv1d_update(mixed, conv, L["convw"], L["convb"], "silu",
                                     conv_state_indices=sidx, validate_data=False)
        torch.ops._rocm_C.gdn_decode_rdna2(mixed, a, b, L["alog"], L["dtb"],
                                           core.unsqueeze(1), ssm, sidx, K ** -0.5, True)
    else:
        N = sidx.shape[0]
        mixed = causal_conv1d_update(mixed, conv, L["convw"], L["convb"], "silu",
                                     conv_state_indices=sidx[:, 0][:N],
                                     num_accepted_tokens=nacc, query_start_loc=qsl,
                                     max_query_len=sidx.size(-1), validate_data=False)
        q, k, v = torch.split(mixed, [H * K, H * K, VD], dim=-1)
        q, k, v = (t.contiguous().view(1, M, -1, K) for t in (q, k, v))
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=L["alog"], a=a, b=b, dt_bias=L["dtb"], q=q, k=k, v=v,
            initial_state=ssm, inplace_final_state=True, cu_seqlens=qsl,
            ssm_state_indices=sidx, num_accepted_tokens=nacc,
            use_qk_l2norm_in_kernel=True)
        core.copy_(o.view(M, HV, V))
    x2 = core.reshape(-1, V).contiguous()
    y = torch.empty_like(x2)
    torch.ops._rocm_C.gated_rms_norm(y, x2, z.reshape(-1, V).contiguous(), L["gnw"], eps, sig)
    return y.view(M, VD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,3,6,12,24")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--layout", default="DS,SD")
    a = ap.parse_args()
    ext = build_ext()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(31)
    eps = 1e-6
    widths = [int(x) for x in a.widths.split(",")]
    sb = int(ext.scratch_bytes(max(widths), HV, H))
    scratch = torch.zeros(sb, dtype=torch.uint8, device=dev)
    ok = True
    L0 = make_layer(gen, dev)
    empty = torch.empty(0, dtype=torch.int32, device=dev)

    def fused(L, qkvz, ba, conv, ssm, sidx, nacc, qsl, y, sig):
        ext.gdn_fused(qkvz, ba, L["convw"], L["convb"], conv, ssm, L["alog"], L["dtb"],
                      L["gnw"], sidx, empty if nacc is None else nacc,
                      empty if qsl is None else qsl, y, scratch, H, K ** -0.5, eps, sig)

    cases = [(False, M) for M in widths] + [(True, M) for M in widths if M % TSPEC == 0]
    for layout in a.layout.split(","):
        for spec, M in cases:
            for sig in (0, 1):
                gt = torch.Generator(device=dev).manual_seed(100 + M + 7 * spec + sig)
                qkvz, ba = make_inputs(gt, dev, M)
                sidx, nacc, qsl = make_meta(gt, dev, M, spec, null_row=True)
                conv, ssm = make_states(gt, dev, layout, 3 + (TSPEC - 1 if spec else 0))
                conv_r, ssm_r = conv.clone(), ssm.clone()
                y_r = reference(L0, qkvz.clone(), ba,  # the conv update writes in place
                                conv_r, ssm_r, sidx, nacc, qsl, eps, sig)
                y = torch.empty(M, VD, dtype=torch.half, device=dev)
                fused(L0, qkvz, ba, conv, ssm, sidx, nacc, qsl, y, sig)
                torch.cuda.synchronize()
                if spec:
                    valid = (sidx[torch.arange(M // TSPEC, device=dev), nacc - 1] > 0)
                    valid = valid.repeat_interleave(TSPEC)
                else:
                    valid = sidx > 0
                d_y = (y[valid].float() - y_r[valid].float()).abs().max().item()
                rel = d_y / max(y_r[valid].float().abs().max().item(), 1e-6)
                null_ok = bool((y[~valid] == 0).all().item()) if (~valid).any() else True
                d_cs = (conv[1:].float() - conv_r[1:].float()).abs().max().item()
                d_ss = (ssm[1:] - ssm_r[1:]).abs().max().item()
                good = rel < 1e-2 and d_cs < 1e-3 and d_ss < 1e-3 and null_ok
                ok &= good
                print(f"{layout} {'verify ' if spec else 'nonspec'} M={M:2d} sig={sig} "
                      f"y max {d_y:.2e} (rel {rel:.1e}) null {'ok' if null_ok else 'BAD'} "
                      f"conv_state {d_cs:.1e} ssm_state {d_ss:.1e} {'ok' if good else 'BAD'}")
    clean = bool((scratch[:64 * (HV * 12 + H * 4)] == 0).all().item())
    ok &= clean
    print("norm scratch zero after calls:", clean)

    # timing: 16 distinct layers (states + inputs), graph-captured, DS layout
    layers = [make_layer(gen, dev) for _ in range(a.layers)]
    for spec, M in cases:
        gt = torch.Generator(device=dev).manual_seed(200 + M + 7 * spec)
        ins = [make_inputs(gt, dev, M) for _ in layers]
        sts = [make_states(gt, dev, "DS", 3 + (TSPEC - 1 if spec else 0)) for _ in layers]
        sidx, nacc, qsl = make_meta(gt, dev, M, spec, null_row=False)
        ys = [torch.empty(M, VD, dtype=torch.half, device=dev) for _ in layers]
        t_f = A.graph_time(lambda: [fused(layers[i], *ins[i], *sts[i], sidx, nacc, qsl, ys[i], 0)
                                    for i in range(len(layers))], a.iters) / len(layers)
        t_r = A.graph_time(lambda: [reference(layers[i], *ins[i], *sts[i], sidx, nacc, qsl,
                                              eps, 0) for i in range(len(layers))],
                           a.iters) / len(layers)
        print(f"{'verify ' if spec else 'nonspec'} M={M:2d} today {t_r:6.1f} us  "
              f"fused {t_f:6.1f} us  x{t_r / t_f:.2f}")
    print("OK" if ok else "BAD")


if __name__ == "__main__":
    main()
