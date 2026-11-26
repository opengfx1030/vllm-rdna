# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate of the Inkling-2 deterministic expert path (moe.py).

``VLLM_RDNA2_MOE_DETERMINISTIC=1`` (the WNA16 MoE switch, same spelling) must
make the stock NVFP4 / MXFP8 routed experts run-to-run bit-identical:

* the flag is resolved once, with the WNA16 truth values (``1``/``true``) and
  is a no-op when unset or off;
* with the flag on, every chunk (even a 2-token decode step) runs the
  prefill-BM w13 (plain stores, one fp32 accumulator per routed row) and a
  BM w2 with ``output_topk=0`` (plain stores) followed by the fp32 top-k sum in
  ascending k -- no decode kernel, so no packed-fp16 CAS atomics;
* the output does not depend on how ``sorted_ids`` ordered the routed rows
  (two runs, permuted routing layout, bit-identical output);
* with the flag off the shipped dispatch is unchanged (decode kernels, fused
  w2 top-k epilogue, no deterministic scratch on the workspace);
* a shape the BM kernels cannot run under the flag is refused, not silently
  run nondeterministically, as is the offloaded expert path (not wired);
* with the variable unset the module resolves the switch off, so the default
  serving dispatch is exactly the shipped one (the flag is opt-in).

    pytest tests/models/inkling2_loop/test_moe_deterministic.py -q
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.inkling2_loop.amd import moe as moe_mod

SWITCH = "VLLM_RDNA2_MOE_DETERMINISTIC"
HIDDEN, INTER, EXPERTS, TOP_K = 32, 32, 4, 2


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,want",
    [
        ({}, False),  # unset: no-op
        ({SWITCH: "0"}, False),
        ({SWITCH: ""}, False),
        ({SWITCH: "yes"}, False),  # only the WNA16 truth values
        ({SWITCH: "1"}, True),
        ({SWITCH: "true"}, True),
        ({SWITCH: "TRUE"}, True),
    ],
)
def test_switch_truth_values(env, want):
    assert moe_mod._deterministic_switch(env) is want


def test_switch_is_the_wna16_variable():
    # one variable for both MoE implementations: the compressed-tensors WNA16
    # path (vllm.envs.VLLM_RDNA2_MOE_DETERMINISTIC) and the stock Inkling-2 path
    # must never disagree about its name or its truth values
    with open(moe_mod.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert 'env.get("VLLM_RDNA2_MOE_DETERMINISTIC", "0")' in src
    assert "_DETERMINISTIC = _deterministic_switch(os.environ)" in src


def test_flag_unset_resolves_to_the_shipped_dispatch():
    # the default path is what ships: without the variable the module resolves
    # the switch off (and the flag-off dispatch test below pins the call tuple)
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    code = (
        "from vllm.models.inkling2_loop.amd import moe;"
        "print(moe._DETERMINISTIC, moe._deterministic_switch({}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "False False"


# ---------------------------------------------------------------------------
# a fake serving layer: tensors on CPU, the HIP ops replaced
# ---------------------------------------------------------------------------


def _views(tokens: int, hidden: int, inter: int, top_k: int):
    n = tokens * top_k
    tw = torch.zeros(n, dtype=torch.float32)
    return types.SimpleNamespace(
        sorted_ids=torch.zeros(n, dtype=torch.int32),
        expert_ids=torch.zeros(n, dtype=torch.int32),
        gu=torch.zeros(n, 2 * inter, dtype=torch.float16),
        act=torch.zeros(n, inter, dtype=torch.float16),
        topk_w=tw.view(tokens, top_k),
        topk_w_flat=tw,
        topk_ids=torch.zeros(tokens, top_k, dtype=torch.int32),
        logits=torch.zeros(tokens, EXPERTS, dtype=torch.float32),
    )


def _workspace(tokens: int, hidden: int, inter: int, top_k: int):
    ws = types.SimpleNamespace(
        moe_chunk=8,
        post_pad=torch.zeros(1, dtype=torch.int32),
        route_counters=torch.zeros(tokens + 1, dtype=torch.int32),
        no_weights=torch.zeros(0, dtype=torch.float32),
    )
    views: dict[tuple[int, int], types.SimpleNamespace] = {}

    def moe_views(tc: int, bm: int):
        key = (tc, bm)
        if key not in views:
            views[key] = _views(tc, hidden, inter, top_k)
        return views[key]

    ws.moe_views = moe_views
    return ws


class _Recorder:
    def __init__(self, hidden, inter, top_k, permute):
        self.calls = []
        self.hidden, self.inter, self.top_k, self.permute = (
            hidden,
            inter,
            top_k,
            permute,
        )

    def route(
        self, x, router_w, bias, logits, topk_w, topk_ids, sorted_ids, *args, **kwargs
    ):
        tokens = x.shape[0]
        n = tokens * self.top_k
        self.route_call = {"zero1": kwargs.get("zero1"), "zero2": kwargs.get("zero2")}
        tw = torch.arange(1, n + 1, dtype=torch.float32).reshape(tokens, self.top_k)
        topk_w.copy_(tw / tw.sum(dim=1, keepdim=True))
        order = torch.arange(n)
        if self.permute:
            order = order.flip(0)
        sorted_ids.fill_(n)  # sentinel = beyond every real routed id
        sorted_ids[:n] = order

    def cta(self, expert_ids, post_pad, num_experts, block_size_m):
        return torch.zeros(2, 4, dtype=torch.int32)

    def swiglu(self, gu, act, gate_cap, up_cap):
        act.copy_(gu[:, : self.inter] + gu[:, self.inter :])

    def gemm(
        self,
        stage,
        a,
        c,
        tw,
        sorted_ids,
        expert_ids,
        post_pad,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
        cta_info=None,
    ):
        self.calls.append(
            {
                "stage": stage,
                "mul": mul_topk_weight,
                "out_topk": output_topk,
                "bm": cta_info is not None,
                "shape": tuple(c.shape),
            }
        )
        if stage == "w13":
            rows = torch.arange(c.shape[0], dtype=torch.float32)
            c.copy_((rows % 11).unsqueeze(1).to(torch.float16))
        else:
            # plain store per routed id, exactly like the BM kernel with
            # output_topk == 0: a row's value does not depend on where the
            # router placed it.
            for rid in sorted_ids.tolist():
                if rid < c.shape[0]:
                    c[rid].fill_(float(rid) + float(a[rid, 0]))


def _layer(hidden=HIDDEN, inter=INTER):
    layer = types.SimpleNamespace(
        quant_format="nvfp4",
        top_k=TOP_K,
        hidden_dim=hidden,
        intermediate_dim=inter,
        num_experts=EXPERTS,
        prefix="fake",
        gate=types.SimpleNamespace(
            routed_scaling_factor=1.0,
            e_score_correction_bias=torch.zeros(EXPERTS),
        ),
        router_w=torch.zeros(EXPERTS, hidden),
        gate_softcap=4.0,
        up_softcap=25.0,
        expert_cache=None,
    )
    layer.max_tokens_per_call = 1024
    layer._default_block_size_m = lambda tc, k, max_query_len=0: 1
    layer._prefill_supported = lambda: hidden % 32 == 0 and inter % 32 == 0
    return layer


def _run(monkeypatch, det, *, permute=False, tokens=2, hidden=HIDDEN, inter=INTER):
    monkeypatch.setattr(moe_mod, "_DETERMINISTIC", det)
    rec = _Recorder(hidden, inter, TOP_K, permute)
    monkeypatch.setattr(ops, "inkling2_glue_moe_route", rec.route)
    monkeypatch.setattr(ops, "moe_prefill_cta_rdna2", rec.cta)
    monkeypatch.setattr(ops, "inkling2_glue_swiglu", rec.swiglu)
    layer = _layer(hidden, inter)
    layer._gemm = rec.gemm

    def fused(
        gate_up,
        c,
        tw,
        sorted_ids,
        expert_ids,
        post_pad,
        top_k,
        block_size_m,
        mul_topk_weight,
        output_topk,
    ):
        # the shipped decode path's fused SwiGLU w2 (FUSE_GLUE, chunks <= 16):
        # one decode launch with the atomic top-k epilogue
        assert mul_topk_weight is True and output_topk == TOP_K
        rec.calls.append(
            {
                "stage": "w2_swiglu",
                "mul": True,
                "out_topk": output_topk,
                "bm": False,
                "shape": tuple(c.shape),
            }
        )

    layer._gemm_swiglu = fused
    ws = _workspace(tokens, hidden, inter, TOP_K)
    x = torch.zeros(tokens, hidden, dtype=torch.float16)
    out = torch.zeros(tokens, hidden, dtype=torch.float16)
    moe_mod.Inkling2LoopMoE.forward_hip(layer, x, ws, out)
    return rec, ws, out


def test_flag_on_is_deterministic_and_order_independent(monkeypatch):
    rec_a, ws_a, out_a = _run(monkeypatch, True, permute=False)
    rec_b, ws_b, out_b = _run(monkeypatch, True, permute=True)

    # 2 tokens < the prefill token gate: the flag still forces the BM kernels
    assert rec_a.route_call["zero1"] is None  # BM w13 stores every routed row
    w13, w2 = rec_a.calls
    assert w13["stage"] == "w13" and w13["bm"] and w13["mul"] is False
    assert w13["out_topk"] == 0
    assert w2["stage"] == "w2" and w2["bm"] and w2["mul"] is False
    assert w2["out_topk"] == 0  # plain stores, no atomic top-k reduction
    assert w2["shape"] == (2 * TOP_K, HIDDEN)
    assert hasattr(ws_a, "_moe_det_rows")

    assert torch.equal(out_a, out_b)  # permuted routed rows: same bits
    # and the output is the fp32 ascending-k sum of what the w2 plain stores
    # wrote, not an fp16 atomic accumulation
    views = ws_a.moe_views(2, 8)
    rows = ws_a._moe_det_rows[: 2 * TOP_K]
    y = rows.view(2, TOP_K, HIDDEN).float() * views.topk_w.unsqueeze(-1)
    ref = y[:, 0]
    for j in range(1, TOP_K):
        ref = ref + y[:, j]
    assert torch.equal(out_a, ref.to(torch.float16))


def test_flag_off_keeps_the_shipped_dispatch(monkeypatch):
    rec, ws, _ = _run(monkeypatch, False)
    assert rec.route_call["zero1"] is not None  # router zero-fills the w13 output
    w13 = rec.calls[0]
    assert w13["stage"] == "w13" and not w13["bm"] and w13["mul"] is False
    assert w13["out_topk"] == 0
    if moe_mod.FUSE_GLUE and moe_mod._FUSE_SWIGLU_MAX_TOKENS >= 2:
        # FUSE_GLUE default: the SwiGLU runs in the decode w2's activation load
        assert [c["stage"] for c in rec.calls] == ["w13", "w2_swiglu"]
        assert rec.calls[1]["mul"] is True and rec.calls[1]["out_topk"] == TOP_K
    else:
        w2 = rec.calls[1]
        assert w2["stage"] == "w2" and not w2["bm"]
        assert w2["mul"] is True and w2["out_topk"] == TOP_K
    assert not hasattr(ws, "_moe_det_rows")  # no-op when unset


def test_flag_on_refuses_shapes_without_the_bm_kernels(monkeypatch):
    with pytest.raises(RuntimeError, match="needs the prefill-BM expert GEMMs"):
        _run(monkeypatch, True, hidden=16, inter=24)  # 24 is not a multiple of 32


def test_flag_on_refuses_the_offload_path(monkeypatch):
    monkeypatch.setattr(moe_mod, "_DETERMINISTIC", True)
    layer = _layer()
    layer.expert_cache = object()  # any non-None cache: the offload dispatch
    layer._gemm = lambda *a, **k: None
    ws = _workspace(2, HIDDEN, INTER, TOP_K)
    x = torch.zeros(2, HIDDEN, dtype=torch.float16)
    out = torch.zeros(2, HIDDEN, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="offloaded expert path"):
        moe_mod.Inkling2LoopMoE.forward_hip(layer, x, ws, out)


def test_flag_on_uses_the_bm_kernels_for_big_chunks_too(monkeypatch):
    rec, ws, out = _run(monkeypatch, True, tokens=8, hidden=32, inter=32)
    assert rec.calls[0]["bm"] and rec.calls[1]["bm"]
    assert rec.calls[1]["out_topk"] == 0
    assert out.shape == (8, 32)
