# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``Inkling2LoopMoE`` (serving module) vs the transformers reference block.

GPU test (gfx1030 with the ``moe_nvfp4_gemm_rdna2`` / ``moe_mxfp8_gemm_rdna2``
kernels).  Random NVFP4 / MXFP8 experts are generated in the *checkpoint* layout
(N-first), loaded through the module's own loading path
(``load_expert_group`` / ``load_gate_bias`` / ``process_weights_after_loading``)
and run on fp16 activations.  The reference is the REAL
``Inkling2LoopSparseMoeBlock`` from transformers on CPU in fp32, fed with the
weights dequantised independently here (E2M1 / E4M3 tables come from the kernel
test, MXFP8 goes through ``torch.float8_e4m3fn``).

Run (box): ``HIP_VISIBLE_DEVICES=1 pytest tests/models/test_inkling2_loop_moe.py``.

The production module and our config are loaded by file path so the test also
works where an installed ``vllm`` shadows the worktree package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


moe = _load("_i2_moe_serving_under_test", "vllm/models/inkling2_loop/amd/moe.py")
ours_cfg = _load("_i2_moe_serving_cfg", "vllm/models/inkling2_loop/configs.py")

ref_cfg_mod = pytest.importorskip(
    "transformers.models.inkling2_loop.configuration_inkling2_loop"
)
ref_mod = pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop"
)


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

E, H, INTER, TOP_K = 8, 256, 128, 2
TOKENS = [1, 3, 17, 64]
TOL = 3e-3  # max |got - want| / max |want|, fp16 kernels vs fp32 reference
SEED = 5


def _kernel_test_helpers():
    """E2M1 / E4M3 decoders of the kernel test (independent of the module)."""
    return _load(
        "_i2_kernel_test_helpers",
        "tests/kernels/quantization/test_rdna2_moe_lowp.py",
    )


# ---------------------------------------------------------------------------
# Checkpoint-layout generators and independent dequantisation
# ---------------------------------------------------------------------------


def rand_nvfp4(n: int, k: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """``weight U8 [E, n, k/2]``, ``weight_scale E4M3 [E, n, k/16]``, ``amax [E]``,
    ``outer_scale_denominator [1]`` -- global = amax / 2688."""
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, generator=gen)
    exp = torch.randint(5, 9, (E, n, k // 16), generator=gen)  # scales 2^-2 .. 2^1
    man = torch.randint(0, 8, (E, n, k // 16), generator=gen)
    s = ((exp << 3) | man).to(torch.uint8).view(torch.float8_e4m3fn)
    amax = 60.0 + 60.0 * torch.rand(E, generator=gen)  # value std ~ 0.1
    return {
        "weight": w,
        "weight_scale": s,
        "weight_amax": amax,
        "weight_outer_scale_denominator": torch.tensor([2688.0]),
    }


def rand_mxfp8(n: int, k: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    sign = torch.randint(0, 2, (E, n, k), generator=gen)
    exp = torch.randint(3, 10, (E, n, k), generator=gen)
    man = torch.randint(0, 8, (E, n, k), generator=gen)
    w = ((sign << 7) | (exp << 3) | man).to(torch.uint8).view(torch.float8_e4m3fn)
    s = (121 + torch.randint(0, 3, (E, n, k // 32), generator=gen)).to(torch.uint8)
    return {"weight": w, "weight_scale": s}


def dequant_nvfp4(t: dict[str, torch.Tensor]) -> torch.Tensor:
    """Checkpoint layout -> fp32 ``[E, N, K]``, through the kernel test's
    K-first word layout and decoders (a different path than the module's)."""
    h = _kernel_test_helpers()
    w = t["weight"].to(torch.int64)
    num_e, n, kd2 = w.shape
    k = kd2 * 2
    code = torch.stack((w & 0xF, w >> 4), dim=-1).reshape(num_e, n, k)  # low = even k
    code = code.transpose(1, 2)  # [E, K, N]
    packed = torch.zeros(num_e, k // 8, n, dtype=torch.int64)
    for i in range(8):
        packed |= code[:, i::8, :] << (4 * i)
    scales = t["weight_scale"].view(torch.uint8).transpose(1, 2).contiguous()
    glob = t["weight_amax"] / t["weight_outer_scale_denominator"]
    return h.dequant_nvfp4(packed, scales, glob).transpose(1, 2).contiguous()


def dequant_mxfp8(t: dict[str, torch.Tensor]) -> torch.Tensor:
    val = t["weight"].to(torch.float32)
    sc = torch.pow(2.0, t["weight_scale"].to(torch.float32) - 127.0)
    return val * sc.repeat_interleave(32, dim=-1)


FORMATS = {
    "nvfp4": (rand_nvfp4, dequant_nvfp4),
    "mxfp8": (rand_mxfp8, dequant_mxfp8),
}


# ---------------------------------------------------------------------------
# Module / reference construction
# ---------------------------------------------------------------------------


def make_configs():
    kw = dict(
        hidden_size=H,
        n_routed_experts=E,
        num_experts_per_tok=TOP_K,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    ours = ours_cfg.Inkling2LoopTextConfig(intermediate_size=INTER, **kw)
    ref = ref_cfg_mod.Inkling2LoopTextConfig(moe_intermediate_size=INTER, **kw)
    return ours, ref


def build_case(fmt: str, seed: int = SEED):
    """Checkpoint tensors + router tensors + dequantised fp32 weights."""
    rand, dequant = FORMATS[fmt]
    gen = torch.Generator().manual_seed(seed)
    w13 = rand(2 * INTER, H, gen)
    w2 = rand(H, INTER, gen)
    group = {f"w13_{k}": v for k, v in w13.items()}
    group.update({f"w2_{k}": v for k, v in w2.items()})
    return {
        "group": group,
        "dq13": dequant(w13),  # [E, 2I, H]
        "dq2": dequant(w2),  # [E, H, INTER]
        "router_w": torch.randn(E, H, generator=gen) * 0.06,
        "router_b": torch.randn(E, generator=gen) * 0.3,
    }


def build_module(fmt: str, case: dict, device: str = "cuda"):
    ours, _ = make_configs()
    with torch.device(device):
        mod = moe.Inkling2LoopMoE(ours, fmt, prefix="layers.0.mlp")
    mod.load_expert_group(case["group"])
    with torch.no_grad():
        mod.gate.weight.copy_(case["router_w"])
    mod.load_gate_bias(case["router_b"])
    mod.process_weights_after_loading()
    return mod.eval()


def build_reference(case: dict):
    _, ref_c = make_configs()
    ref = ref_mod.Inkling2LoopSparseMoeBlock(ref_c).eval()
    with torch.no_grad():
        ref.experts.gate_up_proj.copy_(case["dq13"])
        ref.experts.down_proj.copy_(case["dq2"])
        ref.gate.weight.copy_(case["router_w"])
        ref.gate.e_score_correction_bias.copy_(case["router_b"])
    return ref


def run_case(fmt: str, num_tokens: int, perturb_expert: int | None = None):
    """-> (relative error, selected expert id sets equal).

    ``perturb_expert`` doubles every ``w13`` block scale of that expert in the
    tensors handed to the module only (the reference keeps the true weights)."""
    case = build_case(fmt)
    group = dict(case["group"])
    if perturb_expert is not None:
        s = group["w13_weight_scale"].clone()
        view = s.view(torch.uint8)
        view[perturb_expert] += 8 if fmt == "nvfp4" else 1  # exponent + 1 == x2
        group["w13_weight_scale"] = s
    mod_case = dict(case, group=group)
    mod = build_module(fmt, mod_case)
    ref = build_reference(case)

    gen = torch.Generator().manual_seed(100 + num_tokens)
    x = torch.randn(num_tokens, H, generator=gen).to(torch.float16)
    with torch.no_grad():
        got = mod(x.cuda())
        torch.accelerator.synchronize()
        want = ref(x.float().unsqueeze(0)).squeeze(0)
        _, _, ids_got = mod.gate(x.cuda())
        _, _, ids_want = ref.gate(x.float())
    assert got.dtype == torch.float16 and got.shape == x.shape
    assert torch.isfinite(got).all()
    rel = ((got.float().cpu() - want).abs().max() / want.abs().max()).item()
    ids_equal = torch.equal(
        torch.sort(ids_got.cpu(), dim=1).values, torch.sort(ids_want, dim=1).values
    )
    return rel, ids_equal


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@gpu_only
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_matches_reference(fmt, num_tokens):
    rel, ids_equal = run_case(fmt, num_tokens)
    print(f"{fmt} T={num_tokens}: rel err {rel:.3e}, ids equal {ids_equal}")
    assert ids_equal, "selected expert sets differ from the reference router"
    assert rel <= TOL, rel


@gpu_only
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_detects_perturbed_expert_scale(fmt):
    """The comparison must fail when one expert's scales are wrong."""
    clean, _ = run_case(fmt, 64)
    bad, ids_equal = run_case(fmt, 64, perturb_expert=3)
    print(f"{fmt}: clean rel {clean:.3e}, perturbed rel {bad:.3e}")
    assert ids_equal  # routing is unaffected; only the expert output moves
    assert clean <= TOL
    assert bad > 10 * TOL, bad


@gpu_only
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("block_size_m", [1, 2, 4, 8])
def test_every_block_size_m(fmt, block_size_m):
    case = build_case(fmt)
    mod = build_module(fmt, case)
    ref = build_reference(case)
    x = torch.randn(17, H, generator=torch.Generator().manual_seed(9)).to(torch.float16)
    with torch.no_grad():
        _, w, idx = mod.gate(x.cuda())
        got = moe._ExpertsKernel.forward(mod, x.cuda(), idx, w, block_size_m)
        want = ref(x.float().unsqueeze(0)).squeeze(0)
    rel = ((got.float().cpu() - want).abs().max() / want.abs().max()).item()
    assert rel <= TOL, rel


@gpu_only
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_token_chunking(fmt):
    """64 tokens through 24-token chunks (last chunk partial) == one call."""
    case = build_case(fmt)
    mod = build_module(fmt, case)
    x = torch.randn(64, H, generator=torch.Generator().manual_seed(3)).to(torch.float16)
    with torch.no_grad():
        whole = mod(x.cuda())
        mod.max_tokens_per_call = 24
        chunked = mod(x.cuda())
    ref = build_reference(case)
    with torch.no_grad():
        want = ref(x.float().unsqueeze(0)).squeeze(0)
    rel = ((chunked.float().cpu() - want).abs().max() / want.abs().max()).item()
    assert rel <= TOL, rel
    # different block_size_m per chunk changes only atomic accumulation order
    assert (chunked.float() - whole.float()).abs().max() <= TOL * whole.abs().max()


@gpu_only
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_parameter_layout_and_repack_frees_checkpoint_tensors(fmt):
    ours, _ = make_configs()
    with torch.device("cuda"):
        mod = moe.Inkling2LoopMoE(ours, fmt, prefix="layers.0.mlp")
    p = dict(mod.named_parameters())
    assert p["gate.weight"].shape == (E, H) and p["gate.weight"].dtype == torch.float32
    bias = p["gate.e_score_correction_bias"]
    assert bias.shape == (E,) and bias.dtype == torch.float32
    f8 = torch.float8_e4m3fn
    if fmt == "nvfp4":
        want = {
            "w13_weight": ((E, 2 * INTER, H // 2), torch.uint8),
            "w13_weight_scale": ((E, 2 * INTER, H // 16), f8),
            "w13_weight_amax": ((E,), torch.float32),
            "w13_weight_outer_scale_denominator": ((1,), torch.float32),
            "w2_weight": ((E, H, INTER // 2), torch.uint8),
            "w2_weight_scale": ((E, H, INTER // 16), f8),
            "w2_weight_amax": ((E,), torch.float32),
            "w2_weight_outer_scale_denominator": ((1,), torch.float32),
        }
    else:
        want = {
            "w13_weight": ((E, 2 * INTER, H), f8),
            "w13_weight_scale": ((E, 2 * INTER, H // 32), torch.uint8),
            "w2_weight": ((E, H, INTER), f8),
            "w2_weight_scale": ((E, H, INTER // 32), torch.uint8),
        }
    for name, (shape, dtype) in want.items():
        assert p[name].shape == shape and p[name].dtype == dtype, name
    assert set(p) == set(want) | {"gate.weight", "gate.e_score_correction_bias"}

    # loading validates names, shapes and dtypes
    case = build_case(fmt)
    bad = dict(case["group"])
    bad.pop("w2_weight_scale")
    with pytest.raises(ValueError):
        mod.load_expert_group(bad)
    bad = dict(case["group"])
    bad["w13_weight_scale"] = bad["w13_weight_scale"].view(torch.uint8).to(torch.int16)
    with pytest.raises(ValueError):
        mod.load_expert_group(bad)
    with pytest.raises(RuntimeError):  # forward before repack
        mod(torch.zeros(1, H, dtype=torch.float16, device="cuda"))

    mod.load_expert_group(case["group"])
    mod.process_weights_after_loading()
    assert set(dict(mod.named_parameters())) == {
        "gate.weight",
        "gate.e_score_correction_bias",
    }
    bufs = dict(mod.named_buffers())
    words = 8 if fmt == "nvfp4" else 4
    assert bufs["w13_qweight"].shape == (E, H // words, 2 * INTER)
    assert bufs["w2_qweight"].shape == (E, INTER // words, H)
    assert mod.state_dict().keys() == {"gate.weight", "gate.e_score_correction_bias"}
    if fmt == "nvfp4":
        want_g = case["group"]["w13_weight_amax"] / 2688.0
        assert torch.equal(bufs["w13_globals"].cpu(), want_g)
