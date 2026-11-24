# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HI2 Inkling expert kernels on gfx1030 vs the fp32 reference (GPU only).

The fixture is a tiny stacked HI2 directory made on the CPU by
``tools/inkling_hi2/make_tiny_hi2.py`` (the quantizer's own pipeline on the fork's
tiny checkpoint).  The test loads it through the module's HI2 collector, runs the
production grouped GEMMs (``rdna_hint_flex_moe_transform_rdna2`` +
``..._gemm_rdna2`` + ``inkling2_glue_swiglu``) and compares the routed output with
``hi2.reference_moe`` (float64 decode, fp32 matmuls) at fp16 kernel tolerance.

    cd $TREE
    HI2_TINY_DIR=$HOME/w4a8_runs/inkling-hi2-runtime/tiny/hi2 \\
    HIP_VISIBLE_DEVICES=<leased pair> \\
    python -m pytest tests/models/inkling2_loop/test_hi2_kernel_gpu.py -q

No GPU is touched unless ``torch.cuda.is_available()`` and the HI2 ops exist; the
module skips otherwise (this is the exact command for the first GPU session).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from vllm.models.inkling2_loop.amd import glue, hi2, moe
from vllm.models.inkling2_loop.configs import Inkling2LoopTextConfig

_ROOT = Path(__file__).resolve().parents[3]

_HI2_OPS = (
    "rdna_hint_flex_moe_transform_rdna2",
    "rdna_hint_flex_moe_gemm_rdna2",
    "inkling2_glue_swiglu",
    "inkling2_glue_moe_route",
    "moe_align_block_size",
)
_TINY = os.environ.get("HI2_TINY_DIR")


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    if not _TINY or not (Path(_TINY) / "model.safetensors.index.json").exists():
        return False
    try:
        import vllm._custom_ops  # noqa: F401  (loads _rocm_C)
    except Exception:
        return False
    rocm = getattr(torch.ops, "_rocm_C", None)
    return rocm is not None and all(hasattr(rocm, n) for n in _HI2_OPS)


pytestmark = pytest.mark.skipif(
    not _gpu_ready(),
    reason="needs gfx1030 with the HI2 MoE kernels and HI2_TINY_DIR",
)
DIR = Path(_TINY) if _TINY else Path()
_FIELDS = (
    "codes",
    "code_off",
    "blk_params",
    "param_off",
    "row_consts",
    "row_nbits",
    "row_block",
    "had",
    "col_scale",
    "part_off",
)


def _index() -> dict:
    return json.loads((DIR / "model.safetensors.index.json").read_text())["weight_map"]


def _tensor(name: str) -> torch.Tensor:
    from safetensors import safe_open

    idx = _index()
    with safe_open(str(DIR / idx[name]), "pt") as f:
        return f.get_tensor(name)


def _config():
    text = json.loads((DIR / "config.json").read_text())["text_config"]
    return Inkling2LoopTextConfig(**text)


def _build(layer: int, tp_size: int = 1, tp_rank: int = 0):
    """A loaded ``Inkling2LoopMoE`` of one tiny layer on this rank."""
    cfg = _config()
    m = moe.Inkling2LoopMoE(
        cfg, "hi2", prefix=f"t{layer}", tp_size=tp_size, tp_rank=tp_rank
    ).cuda()
    prefix = f"model.language_model.layers.{layer}.mlp"
    for group in ("gate_up", "down"):
        for field in _FIELDS:
            key = f"{prefix}.experts.{group}.hi2_{field}"
            if key in _index():
                m.load_hi2_field(group, field, _tensor(key))
    with torch.no_grad():
        m.gate.weight.copy_(_tensor(f"{prefix}.gate.weight").float())
        m.gate.e_score_correction_bias.copy_(
            _tensor(f"{prefix}.gate.e_score_correction_bias").float()
        )
    m.process_weights_after_loading()
    return m


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()


def _oracle(m, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, weights, ids = m.gate(x)
        return hi2.reference_moe(
            x,
            ids,
            weights,
            m._hi2_weights("w13"),
            m._hi2_weights("w2"),
            m.gate_softcap,
            m.up_softcap,
            experts=m.num_experts,
        )


@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_grouped_kernels_match_the_reference(tokens):
    m = _build(0)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(tokens, m.hidden_dim, generator=gen).cuda().to(torch.float16)
    want = _oracle(m, x)
    got = m(x)
    torch.accelerator.synchronize()
    assert got.dtype == torch.float16 and got.shape == x.shape
    err = _rel_err(got, want)
    print(f"HI2 M={tokens}: rel err {err:.3e}")
    assert err <= 5e-2, err


def test_tp2_ranks_sum_to_the_single_rank_output():
    m1 = _build(0)
    m0 = _build(0, tp_size=2, tp_rank=0)
    m_1 = _build(0, tp_size=2, tp_rank=1)
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(16, m1.hidden_dim, generator=gen).cuda().to(torch.float16)
    with torch.no_grad():
        want = m1(x)
        got = m0(x).float() + m_1(x).float()
    torch.accelerator.synchronize()
    err = _rel_err(got, want)
    print(f"HI2 TP=2 vs TP=1: rel err {err:.3e}")
    assert err <= 5e-2, err


def test_hip_path_matches_the_eager_kernel_and_does_not_allocate():
    m = _build(0)
    max_tokens = 64
    ws = glue.Inkling2GlueWorkspace(_config(), max_tokens, "cuda", hi2=True)
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(64, m.hidden_dim, generator=gen).cuda().to(torch.float16)
    out = torch.empty_like(x)
    with torch.no_grad():
        want = _oracle(m, x)
        m.prepare_hip()
        for _ in range(2):  # warm the memoised views and the route counters
            m.forward_hip(x, ws, out)
        torch.accelerator.synchronize()
        before = torch.accelerator.memory_allocated(out.device)
        m.forward_hip(x, ws, out)
        torch.accelerator.synchronize()
        after = torch.accelerator.memory_allocated(out.device)
    err = _rel_err(out, want)
    print(f"HI2 HIP path: rel err {err:.3e}, alloc delta {after - before} B")
    assert err <= 5e-2, err
    assert after == before, f"forward_hip allocated {after - before} bytes"
