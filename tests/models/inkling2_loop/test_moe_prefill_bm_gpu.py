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
accumulation order and rounding only.  The default path must be at least as close to
the fp32 transformers reference (exact experts) as the decode-shaped kernels are.  The
call counters of the ops prove that the path under test ran (and that the flag, the
token gate and the fallbacks really switch it), so the comparison is not vacuous.

    HIP_VISIBLE_DEVICES=2 PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_moe_prefill_bm_gpu.py -q -s
"""

from __future__ import annotations

import contextlib

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
from test_serving_path_gpu import STABLE_BIAS, _pm, _variant  # noqa: E402

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
# the decode-shaped expert GEMMs: w13 / w2 (and, for chunks of <= 16 tokens, w2 with the SwiGLU fused into its
# activation load: VLLM_INKLING2_FUSE_GLUE, on by default)
DECODE_GEMMS = (
    "moe_nvfp4_gemm_rdna2",
    "moe_mxfp8_gemm_rdna2",
    "moe_nvfp4_gemm_swiglu_rdna2",
    "moe_mxfp8_gemm_swiglu_rdna2",
)


def _decode_gemms(c):
    return sum(c[n] for n in DECODE_GEMMS)


NOT_WORSE = 1.1  # error against the reference: at most this x the decode kernels'
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
def test_dot2_prefill_is_not_worse_than_the_decode_kernels(hv, n):
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
    e_off = _error(f"{hv.name} n={n} decode kernels", hv, ids, off)
    e_on = _error(f"{hv.name} n={n} dot2 prefill kernels", hv, ids, on)
    _fp16_check(hv, f"dot2 prefill n={n}", on, ids)
    assert e_on <= NOT_WORSE * e_off + 1e-3


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
