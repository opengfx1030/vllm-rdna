# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aron Hsiao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx10x decode ports from leapdragon/vllm-rdna2-qwen (fd8d939ae, a80aad336).

CPU: the opt-in dispatch predicates (VLLM_RDNA_DECODE_M16, VLLM_RDNA_MOE_V2,
VLLM_RDNA_MOE_ALIAS). GPU (gfx10x, ops built): gemv_i8_rdna2 at M = 1..16,
the fused glue ops at M = 16 (must equal two M = 8 calls), and
moe_skinny_int4_decode_v2 against a torch reference and v1, with an EP
expert_map and the residual epilogue. Ported from his tools/rdna2/
{skinny_m16,glue_m16,moe_v2}_test.py.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import rdna_ops


@pytest.mark.parametrize(
    ("m16", "n", "expected"),
    [
        (False, 1, True),
        (False, 8, True),
        (False, 9, False),
        (False, 16, False),
        (True, 8, True),
        (True, 12, False),
        (True, 16, True),
        (True, 17, False),
        (True, 0, False),
    ],
)
def test_fused_decode_rows(monkeypatch, m16, n, expected):
    monkeypatch.setattr(rdna_ops, "_DECODE_M16", m16)
    assert rdna_ops._fused_ok(n) is expected


def test_moe_alias_is_opt_in(monkeypatch):
    from vllm.model_executor.layers.fused_moe import modular_kernel
    from vllm.platforms import rocm

    monkeypatch.setattr(rocm, "on_gfx10x", lambda: True)
    monkeypatch.delenv("VLLM_RDNA_MOE_ALIAS", raising=False)
    assert modular_kernel._rdna_moe_alias() is False
    monkeypatch.setenv("VLLM_RDNA_MOE_ALIAS", "1")
    assert modular_kernel._rdna_moe_alias() is True
    monkeypatch.setattr(rocm, "on_gfx10x", lambda: False)
    assert modular_kernel._rdna_moe_alias() is False


def _gpu_ops(name: str):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("ROCm only")
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("gfx10x only")
    import vllm._custom_ops  # noqa: F401  (loads _rocm_C)

    if not hasattr(torch.ops._rocm_C, name):
        pytest.skip(f"{name} not built")
    return torch.ops._rocm_C


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    err = (a.float() - b.float()).abs().max()
    return (err / b.float().abs().max().clamp_min(1e-6)).item()


def _i8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = w.abs().amax(dim=1).float().clamp_min(1e-8) / 127.0
    q = torch.round(w.float() / s[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), s.half().contiguous()


@pytest.mark.parametrize("m", [1, 4, 8, 9, 12, 15, 16])
def test_gemv_i8_rows_up_to_16(m):
    o = _gpu_ops("gemv_i8_rdna2")
    torch.manual_seed(0)
    k, n = 2560, 1536
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    q, s = _i8(w)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    ref = F.linear(x, (q.float() * s.float()[:, None]).half())
    assert _rel(o.gemv_i8_rdna2(x, q, s, None), ref) < 1e-2


@pytest.mark.parametrize("quant", [True, False])
def test_fused_glue_m16_equals_two_m8(quant):
    o = _gpu_ops("rdna_gemv_act")
    torch.manual_seed(0)
    h, r, hc, inter = 2560, 320, 4, 160
    dev, f16 = "cuda", torch.float16
    weights = dict(
        d=torch.randn(336, hc * h, device=dev, dtype=f16) * 0.01,
        u=torch.randn(hc * h, r, device=dev, dtype=f16) * 0.05,
        a=torch.randn(2 * inter, h, device=dev, dtype=f16) * 0.02,
        b=torch.randn(h, inter, device=dev, dtype=f16) * 0.05,
    )
    p = {k: (_i8(v) if quant else (v.contiguous(), None)) for k, v in weights.items()}
    wg = (torch.randn(h, device=dev, dtype=f16) * 0.02).contiguous()
    xn = torch.randn(16, hc * h, device=dev, dtype=f16)
    x = torch.randn(16, h, device=dev, dtype=f16)
    lora = torch.randn(16, r, device=dev, dtype=f16)
    act = torch.randn(16, inter, device=dev, dtype=f16)

    def rows(t, sl):
        return t[sl].contiguous()

    calls = {
        "gemv_act": lambda sl: o.rdna_gemv_act(
            rows(xn, sl), p["d"][0], p["d"][1], r, 1.0 / hc
        ),
        "hc_up_gate_mix": lambda sl: o.rdna_hc_up_gate_mix(
            rows(lora, sl), p["u"][0], p["u"][1], rows(xn, sl), hc
        ),
        "se_gate_up_silu": lambda sl: o.rdna_se_gate_up_silu(
            rows(x, sl), p["a"][0], p["a"][1]
        ),
        "se_down_gated": lambda sl: o.rdna_se_down_gated(
            rows(act, sl), p["b"][0], p["b"][1], rows(x, sl), wg
        ),
    }
    for name, f in calls.items():
        y16 = f(slice(0, 16))
        y8 = torch.cat([f(slice(0, 8)), f(slice(8, 16))])
        assert torch.isfinite(y16).all(), name
        torch.testing.assert_close(y16, y8, atol=0, rtol=0, msg=name)


def _pack_sequential_int4(q: torch.Tensor) -> torch.Tensor:
    sh = q.shape
    w = q.reshape(*sh[:-1], sh[-1] // 8, 8).to(torch.int64)
    out = torch.zeros(*sh[:-1], sh[-1] // 8, dtype=torch.int64, device=q.device)
    for j in range(8):
        out |= w[..., j] << (4 * j)
    return out.to(torch.int32).contiguous()


@pytest.mark.parametrize("m", [1, 3, 8, 16])
@pytest.mark.parametrize(
    ("ids_dtype", "w_dtype"),
    [(torch.int64, torch.float32), (torch.int32, torch.float16)],
)
def test_moe_skinny_v2_matches_reference(m, ids_dtype, w_dtype):
    o = _gpu_ops("moe_skinny_int4_decode_v2")
    torch.manual_seed(0)
    dev = "cuda"
    e_glob, e_loc, h, inter, g, topk = 64, 16, 2560, 640, 128, 10
    q13 = torch.randint(0, 16, (e_loc, 2 * inter, h), device=dev, dtype=torch.uint8)
    q2 = torch.randint(0, 16, (e_loc, h, inter), device=dev, dtype=torch.uint8)
    s13 = (torch.rand(e_loc, 2 * inter, h // g, device=dev) * 0.01 + 0.002).half()
    s2 = (torch.rand(e_loc, h, inter // g, device=dev) * 0.01 + 0.002).half()
    w13, w2 = _pack_sequential_int4(q13), _pack_sequential_int4(q2)
    d13 = (q13.float() - 8) * s13.float().repeat_interleave(g, dim=2)
    d2 = (q2.float() - 8) * s2.float().repeat_interleave(g, dim=2)
    emap = torch.full((e_glob,), -1, dtype=torch.int32, device=dev)
    emap[torch.randperm(e_glob, device=dev)[:e_loc]] = torch.arange(
        e_loc, dtype=torch.int32, device=dev
    )

    x = torch.randn(m, h, device=dev, dtype=torch.float16) * 0.5
    ids = torch.stack([torch.randperm(e_glob, device=dev)[:topk] for _ in range(m)]).to(
        ids_dtype
    )
    tw = torch.softmax(torch.randn(m, topk, device=dev), -1).to(w_dtype)
    res = torch.randn(m, h, device=dev, dtype=torch.float16) * 0.1
    ref = torch.zeros(m, h, device=dev)
    for mi in range(m):
        for j in range(topk):
            e = int(emap[int(ids[mi, j])])
            if e < 0:
                continue
            gu = d13[e] @ x[mi].float()
            a = (F.silu(gu[:inter]) * gu[inter:]).half().float()
            ref[mi] += float(tw[mi, j]) * (d2[e] @ a)

    act = torch.empty(m, topk, inter, device=dev, dtype=torch.float16)
    out_v1 = torch.empty(m, h, device=dev, dtype=torch.float16)
    out_v2 = torch.empty_like(out_v1)
    out_res = torch.empty_like(out_v1)
    o.moe_skinny_int4_decode(x, w13, s13, w2, s2, tw, ids, act, out_v1, g, emap)
    o.moe_skinny_int4_decode_v2(
        x, w13, s13, w2, s2, tw, ids, act, out_v2, g, emap, None
    )
    o.moe_skinny_int4_decode_v2(
        x, w13, s13, w2, s2, tw, ids, act, out_res, g, emap, res
    )
    assert _rel(out_v1, ref) < 2e-2
    assert _rel(out_v2, ref) < 2e-2
    assert torch.isfinite(out_res).all()
    assert _rel(out_res, ref + res.float()) < 2e-2
