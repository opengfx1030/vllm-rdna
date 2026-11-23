# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level gate of the prefill expert GEMMs (``csrc/rocm/moe_prefill_bm_rdna2.cu``).

The HIP serving model of ``test_serving_path_gpu.py`` (router-stable tiny checkpoints:
NVFP4 experts in every layer but the last, MXFP8 in the last; hidden 256, intermediate
128) runs the same prompts with the prefill kernels

* off (``VLLM_INKLING2_PREFILL_BM=0``: every chunk on the decode-shaped kernels),
* on (the default: fp16 dot2 32-row kernels, here with the token gate lowered so that
  prompts of 5 / 40 / 70 tokens take the path), and
* on with ``VLLM_INKLING2_PREFILL_INT8=1`` (W4A8 NVFP4 layers, experimental).

The routing is stable (the checkpoints scale the router's selection bias up until the
top-k choice is far from any tie), so the logits of every configuration differ by
accumulation order and rounding only.  The call counters of the ops prove that the path
under test ran (and that the flag, the token gate and the fallbacks really switch it),
so the comparisons are not vacuous.

The error against the fp32 transformers reference (exact experts) is compared PAIRED
over checkpoints and prompts, not against a single-seed bound.  At the tiny shapes
(K 256 / 128) both kernels sit on the fp16 output rounding floor (fp64 relative L2 of
the two equal to 0.1 % in each of 36 format x stage x token cases), so their logits
differ by rounding noise.  This small network amplifies it through the router: of 120
(checkpoint, prompt) pairs (tiny1 + tiny8, 10 checkpoints, 6 prompts each) five have
ONE top-k expert set that differs between the two runs (one token of 560 to 1120
token-visits) and an error ratio of 1.7 to 7; the other 115 pairs have a median ratio
of 0.998 (tiny1) and 1.007 (tiny8), a geometric sd of 0.04 and 0.06 and a range of
0.83 to 1.17.  The runs are bitwise repeatable here, so there is no repeat-to-repeat
spread to read the noise from; the spread over checkpoints is the floor.  The gate is
therefore the pairs whose router decisions agree: their median ratio (at most 1.04; 99 %
bootstrap bound of the measured pairs 1.013 tiny1, 1.018 tiny8) and the one-sided 99 %
bound of their geometric mean (at most 1.06; measured 1.008 and 1.027), and at most a
fifth of the pairs may differ in a router decision (measured 1 and 4 of 48).  A
systematic loss of 5 % fails the median on both; the fp64-oracle tests of the kernel
file are the sensitive accuracy gate.

    HIP_VISIBLE_DEVICES=2 PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_moe_prefill_bm_gpu.py -q -s
"""

from __future__ import annotations

import contextlib
import math
import statistics
import types

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)
if not torch.cuda.is_available():
    pytest.skip("needs a ROCm GPU", allow_module_level=True)

from test_paged_model import (  # noqa: E402
    PROMPT_LENS,
    _dlogp,
    _fp16_check,
    _prompts,
    _want,
)
from test_serving_path_gpu import (  # noqa: E402
    SHAPE,
    STABLE_BIAS,
    _pm,
    _variant,
    make,
    ref,
)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.inkling2_loop.amd import moe as moe_mod  # noqa: E402

if not hasattr(torch.ops._rocm_C, "moe_prefill_cta_rdna2"):
    pytest.skip("this build has no moe_prefill_bm_rdna2.cu", allow_module_level=True)

COUNTED = (
    "inkling2_glue_moe_route",
    "moe_prefill_cta_rdna2",
    "moe_prefill_act_quant_rdna2",
    "moe_nvfp4_bm_rdna2",
    "moe_nvfp4_bm_i8_rdna2",
    "moe_mxfp8_bm_rdna2",
    "moe_nvfp4_gemm_rdna2",
    "moe_mxfp8_gemm_rdna2",
    "moe_nvfp4_gemm_swiglu_rdna2",
    "moe_mxfp8_gemm_swiglu_rdna2",
)
# the decode-shaped expert GEMMs: w13 / w2 (and, for chunks of <= 16 tokens, w2 with
# the SwiGLU fused into its activation load: VLLM_INKLING2_FUSE_GLUE, on by default)
DECODE_GEMMS = (
    "moe_nvfp4_gemm_rdna2",
    "moe_mxfp8_gemm_rdna2",
    "moe_nvfp4_gemm_swiglu_rdna2",
    "moe_mxfp8_gemm_swiglu_rdna2",
)


def _decode_gemms(c):
    return sum(c[n] for n in DECODE_GEMMS)


PAIRED_SEEDS = 8  # checkpoints (weights) of the paired comparison; 6 prompts each
# error ratio (dot2 prefill / decode kernels) of the pairs whose router decisions agree:
# median at most MEDIAN_RATIO, one-sided 99 % bound of the geometric mean at most
# MEAN_RATIO; at most MAX_FLIPPED of the pairs may differ in a router decision
MEDIAN_RATIO = 1.04
MEAN_RATIO = 1.06
MAX_FLIPPED = 0.2
Z99 = 2.4  # one-sided 99 % quantile of Student t (43 to 47 degrees of freedom)
INT8_TOL = 1.0  # W4A8 (experimental, quality-changing) must only stay bounded here


@pytest.fixture(scope="module", params=["tiny1", "tiny8"])
def hv(request, tmp_path_factory):
    """Router-stable HIP-shaped checkpoints (the strict tests of the serving path)."""
    return _variant(f"{request.param}-stable", tmp_path_factory, STABLE_BIAS)


@contextlib.contextmanager
def _config(bm: bool, int8: bool = False, min_tokens: int = 1):
    """The module switches of moe.py (read at call time) + call counters of the ops."""
    counts = dict.fromkeys(COUNTED, 0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(moe_mod, "_PREFILL_BM", bm)
        mp.setattr(moe_mod, "_PREFILL_INT8", bm and int8)
        mp.setattr(moe_mod, "_PREFILL_BM_MIN_TOKENS", min_tokens)
        for name in COUNTED:
            real = getattr(ops, name)

            def counted(*args, _real=real, _name=name, **kwargs):
                counts[_name] += 1
                return _real(*args, **kwargs)

            mp.setattr(ops, name, counted)
        yield counts


def _prefill(hv, ids, **config):
    with _config(**config) as counts:
        logits = _pm(hv, "hip").step([(0, ids)])[0]
    assert torch.isfinite(logits).all()
    return logits, counts


def _stats(what, got, want):
    d = _dlogp(got, want)
    print(
        f"[prefill-bm] {what}: mean|dlogp| {float(d.abs().mean()):.3e} "
        f"max {float(d.abs().max()):.3e}"
    )
    return d


def _error(what, hv, ids, got):
    """Mean |dlogp| of ``got`` against the fp32 transformers reference."""
    m = float(_dlogp(got, _want(hv, ids)).abs().mean())
    print(f"[prefill-bm] {what}: mean|dlogp| vs the fp32 reference {m:.3e}")
    return m


@pytest.mark.parametrize("n", PROMPT_LENS)
def test_flag_off_runs_only_the_decode_kernels(hv, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    _, c = _prefill(hv, ids, bm=False, min_tokens=1)
    gemms = _decode_gemms(c)
    assert c["inkling2_glue_moe_route"] > 0
    assert gemms == 2 * c["inkling2_glue_moe_route"]
    for name in COUNTED[1:6]:
        assert c[name] == 0, name


@pytest.mark.parametrize("n", PROMPT_LENS)
def test_dot2_prefill_runs_on_every_moe_visit(hv, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    off, c_off = _prefill(hv, ids, bm=False)
    on, c = _prefill(hv, ids, bm=True)
    routes = c["inkling2_glue_moe_route"]
    assert routes == c_off["inkling2_glue_moe_route"] > 0
    # every MoE visit took the prefill path: one work list, two GEMMs
    assert c["moe_prefill_cta_rdna2"] == routes
    assert c["moe_nvfp4_bm_rdna2"] + c["moe_mxfp8_bm_rdna2"] == 2 * routes
    assert c["moe_mxfp8_bm_rdna2"] > 0 and c["moe_nvfp4_bm_rdna2"] > 0
    assert _decode_gemms(c) == 0
    assert c["moe_nvfp4_bm_i8_rdna2"] == c["moe_prefill_act_quant_rdna2"] == 0
    _stats(f"{hv.name} n={n} dot2 against the decode kernels", on, off)
    _error(f"{hv.name} n={n} decode kernels", hv, ids, off)
    _error(f"{hv.name} n={n} dot2 prefill kernels", hv, ids, on)
    # absolute: mean |dlogp| < 0.05 and the same argmax at the last position
    _fp16_check(hv, f"dot2 prefill n={n}", on, ids)


def _seeded_variant(kind, seed, tmp_path_factory):
    """``_variant`` of a router-stable tiny checkpoint with the weights of ``seed``."""
    shape = {**SHAPE, "router_bias_scale": STABLE_BIAS}
    if kind == "tiny1":
        spec = make.TinySpec(**shape)
    else:
        spec = make.TinySpec(
            num_hidden_layers=8, loop_start=2, loop_end=6, loop_num_passes=3, **shape
        )
    path = tmp_path_factory.mktemp(f"{kind}-seed{seed}")
    make.write_checkpoint(path, spec, seed=seed)
    return types.SimpleNamespace(
        name=f"{kind}-stable-seed{seed}",
        path=path,
        tensors=ref.read_checkpoint(path),
        ref=ref.build_reference(path),
        models={},
        logits={},
    )


@contextlib.contextmanager
def _recorded_routes():
    """The top-k expert sets (ids sorted per token) of every route launch run inside."""
    sets = []
    real = ops.inkling2_glue_moe_route

    def record(x, router_w, bias, logits, topk_w, topk_ids, *args, **kwargs):
        out = real(x, router_w, bias, logits, topk_w, topk_ids, *args, **kwargs)
        sets.append(topk_ids[: x.shape[0]].clone().sort(dim=1).values)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ops, "inkling2_glue_moe_route", record)
        yield sets


@pytest.mark.parametrize("kind", ["tiny1", "tiny8"])
def test_dot2_prefill_is_not_worse_than_the_decode_kernels(kind, tmp_path_factory):
    """Paired over ``PAIRED_SEEDS`` checkpoints x (the 3 standard prompts + 3 random
    ones): median and one-sided 99 % bound of the error ratio of the pairs whose router
    decisions agree (module docstring)."""
    ratios, flipped, total = [], 0, 0
    for seed in range(PAIRED_SEEDS):
        v = _seeded_variant(kind, seed, tmp_path_factory)
        gen = torch.Generator().manual_seed(7000 + seed)
        prompts = list(_prompts()) + [
            torch.randint(0, 500, (n,), generator=gen).tolist() for n in PROMPT_LENS
        ]
        for ids in prompts:
            with _recorded_routes() as routes_off:
                off, _ = _prefill(v, ids, bm=False)
            with _recorded_routes() as routes_on:
                on, c = _prefill(v, ids, bm=True)
            assert c["moe_prefill_cta_rdna2"] == c["inkling2_glue_moe_route"] > 0
            assert len(routes_on) == len(routes_off) > 0
            total += 1
            if any(not torch.equal(a, b) for a, b in zip(routes_off, routes_on)):
                flipped += 1  # a discrete router decision differs: not an accuracy pair
                continue
            want = _want(v, ids)
            e_off = float(_dlogp(off, want).abs().mean())
            e_on = float(_dlogp(on, want).abs().mean())
            ratios.append(e_on / e_off)
        v.models.clear()
        del v
    logs = [math.log(r) for r in ratios]
    mean = sum(logs) / len(logs)
    bound = mean + Z99 * statistics.stdev(logs) / math.sqrt(len(logs))
    median = statistics.median(ratios)
    print(
        f"[prefill-bm] {kind} paired, {len(ratios)} pairs with equal router decisions "
        f"(+ {flipped} of {total} with one decision flipped): median error ratio "
        f"{median:.4f}, geometric mean {math.exp(mean):.4f}, 99 % upper bound "
        f"{math.exp(bound):.4f}, dot2 worse in {sum(r > 1 for r in ratios)}"
    )
    assert flipped <= MAX_FLIPPED * total
    assert median <= MEDIAN_RATIO
    assert bound <= math.log(MEAN_RATIO)


@pytest.mark.parametrize("n", PROMPT_LENS)
def test_token_gate_keeps_small_chunks_on_the_decode_kernels(hv, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    _, c = _prefill(hv, ids, bm=True, min_tokens=n + 1)
    assert c["moe_prefill_cta_rdna2"] == 0
    assert _decode_gemms(c) == 2 * (c["inkling2_glue_moe_route"])
    _, c = _prefill(hv, ids, bm=True, min_tokens=n)
    assert c["moe_prefill_cta_rdna2"] == c["inkling2_glue_moe_route"] > 0


@pytest.mark.parametrize("n", PROMPT_LENS)
def test_int8_prefill_is_opt_in_and_bounded(hv, n):
    ids = _prompts()[PROMPT_LENS.index(n)]
    on, c = _prefill(hv, ids, bm=True, int8=True)
    routes = c["inkling2_glue_moe_route"]
    assert c["moe_prefill_cta_rdna2"] == routes
    # NVFP4 layers: W4A8 (quantise the activations, then the int8 kernel); MXFP8: dot2
    assert c["moe_nvfp4_bm_i8_rdna2"] == c["moe_prefill_act_quant_rdna2"] > 0
    assert c["moe_nvfp4_bm_rdna2"] == 0
    assert c["moe_nvfp4_bm_i8_rdna2"] + c["moe_mxfp8_bm_rdna2"] == 2 * routes
    assert torch.isfinite(on).all()
    assert _error(f"{hv.name} n={n} W4A8", hv, ids, on) < INT8_TOL


def test_int8_needs_the_dot2_prefill_path(hv):
    ids = _prompts()[2]
    _, c = _prefill(hv, ids, bm=False, int8=True)
    assert c["moe_nvfp4_bm_i8_rdna2"] == c["moe_prefill_cta_rdna2"] == 0
    assert c["moe_nvfp4_gemm_rdna2"] > 0


def test_ineligible_k_falls_back_to_the_decode_kernels(hv):
    """TP shards the intermediate size (576 / 288 at the real shapes): K not a multiple
    of 128 keeps the NVFP4 layers on the decode kernels."""
    ids = _prompts()[2]
    layers = [
        m
        for m in hv.models["hip"].lm.modules()
        if isinstance(m, moe_mod.Inkling2LoopMoE)
    ]
    assert layers
    old = [m.intermediate_dim for m in layers]
    try:
        for m in layers:
            m.intermediate_dim = (
                96  # w2 K = 96: no multiple of 128 (NVFP4), of 32 (MXFP8) yes
            )
        _, c = _prefill(hv, ids, bm=True)
    finally:
        for m, v in zip(layers, old):
            m.intermediate_dim = v
    assert c["moe_nvfp4_bm_rdna2"] == 0
    assert c["moe_nvfp4_gemm_rdna2"] > 0
    assert c["moe_mxfp8_bm_rdna2"] > 0  # K = 96 is a multiple of 32
