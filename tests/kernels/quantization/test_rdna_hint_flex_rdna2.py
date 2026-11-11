# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rdna_hint_flex (HadamardINT2-FP16 spec v3.2) weight support.

The reference below is a port of ``tools/hi2/hi2/format.py`` (the quantizer's
ground truth for the bit layout): LSB-first word-aligned code packing, the
8-bit block parameters decoded through the three fp16 row constants, the
order-16 Walsh-Hadamard on flagged groups, the smoothing vector, and
``reference_linear`` in float64.

CPU tests (run anywhere): the host helpers in
``vllm/model_executor/layers/quantization/utils/rdna_hint_flex_utils.py`` (dequantize,
row/column slicing for tensor parallelism, concatenation for fused layers,
validation, the dequantized fallback) against the reference.

GPU tests (gfx1030 with the ``_rocm_C`` rdna_hint_flex ops): the dequant kernel is
bit-exact against the reference decode; the activation transform and the
GEMV (decode) and dequant+GEMM (prefill) paths of ``rdna_hint_flex_linear_rdna2`` match
``reference_linear`` to fp16 accuracy.

Run ``pytest tests/kernels/quantization/test_rdna_hint_flex_rdna2.py``.
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu

GROUP = 16


# ---- reference (port of tools/hi2/hi2/format.py) -------------------------
def ref_hadamard16(dtype=torch.float64) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < GROUP:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(GROUP)).to(dtype)


def ref_fwht_groups(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    xg = x.double().reshape(*x.shape[:-1], k // GROUP, GROUP)
    rot = xg @ ref_hadamard16()
    out = torch.where(flags.bool().reshape(-1, 1), rot, xg)
    return out.reshape(x.shape)


def ref_decode_params(lo_u8, step_u8, lo_min, lo_scale, step_scale):
    lo = (
        (lo_u8.double() * lo_scale.double() + lo_min.double()).float().to(torch.float16)
    )
    step = (step_u8.float() * step_scale.float()).to(torch.float16)
    return lo, step


def ref_pack_codes(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    per_word = 32 // nbits
    n = codes.shape[-1]
    c = codes.to(torch.int64).reshape(*codes.shape[:-1], n // per_word, per_word)
    word = torch.zeros(c.shape[:-1], dtype=torch.int64)
    for i in range(per_word):
        word |= c[..., i] << (nbits * i)
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)


def ref_unpack_codes(words: torch.Tensor, nbits: int) -> torch.Tensor:
    per_word = 32 // nbits
    w = words.to(torch.int64) & 0xFFFFFFFF
    mask = (1 << nbits) - 1
    c = torch.stack([(w >> (nbits * i)) & mask for i in range(per_word)], dim=-1)
    return c.reshape(*words.shape[:-1], -1)


def ref_dequantize_stored(t: dict) -> torch.Tensor:
    """Row by row, straight from the spec: float64 ``[R, K]`` of fp16 values."""
    r = t["row_nbits"].numel()
    k = t["had"].numel() * GROUP
    out = torch.zeros(r, k, dtype=torch.float64)
    for i in range(r):
        nb, bs = int(t["row_nbits"][i]), int(t["row_block"][i])
        c0, c1 = int(t["code_off"][i]), int(t["code_off"][i + 1])
        p0 = int(t["param_off"][i])
        codes = ref_unpack_codes(t["codes"][c0:c1], nb).reshape(k // bs, bs)
        p = t["blk_params"][2 * p0 : 2 * (p0 + k // bs)].reshape(-1, 2)
        rc = t["row_consts"][i]
        lo, step = ref_decode_params(p[:, 0], p[:, 1], rc[0], rc[1], rc[2])
        w = lo.double().unsqueeze(-1) + codes.double() * step.double().unsqueeze(-1)
        out[i] = w.to(torch.float16).double().reshape(-1)
    return out


def ref_linear(x: torch.Tensor, t: dict) -> torch.Tensor:
    xd = x.double()
    if t.get("col_scale") is not None:
        xd = xd / t["col_scale"].double()
    return ref_fwht_groups(xd, t["had"]) @ ref_dequantize_stored(t).T


# ---- random rdna_hint_flex tensors ---------------------------------------------------
def make_hi2(
    r: int,
    k: int,
    nbits=(2, 4, 8),
    blocks=(16, 32, 64),
    had: str = "mixed",
    smooth: bool = True,
    seed: int = 0,
) -> dict:
    """Random tensor in the checkpoint layout (spec 2.4), built with the
    reference packer. Row widths and block sizes are drawn per row."""
    g = torch.Generator().manual_seed(seed)
    nb = torch.tensor(nbits)[torch.randint(len(nbits), (r,), generator=g)]
    bl = torch.tensor(blocks)[torch.randint(len(blocks), (r,), generator=g)]
    words, params = [], []
    code_off, param_off = [0], [0]
    for i in range(r):
        n, b = int(nb[i]), int(bl[i])
        codes = torch.randint(0, 2**n, (k,), generator=g)
        words.append(ref_pack_codes(codes, n))
        p = torch.randint(0, 256, (k // b, 2), generator=g, dtype=torch.int64)
        p[torch.rand(k // b, generator=g) < 0.05, 1] = 0  # degenerate blocks
        params.append(p.to(torch.uint8).reshape(-1))
        code_off.append(code_off[-1] + k * n // 32)
        param_off.append(param_off[-1] + k // b)
    span = 0.02 + 0.08 * torch.rand(r, generator=g)
    levels = (2**nb - 1).double()
    consts = torch.stack(
        [
            -span / 2,  # lo_min
            span / 2 / 255,  # lo_scale
            (span / levels / 255 * 2).float(),  # step_scale
        ],
        dim=1,
    ).to(torch.float16)
    if had == "all":
        flags = torch.ones(k // GROUP, dtype=torch.uint8)
    elif had == "none":
        flags = torch.zeros(k // GROUP, dtype=torch.uint8)
    else:
        flags = (torch.rand(k // GROUP, generator=g) < 0.5).to(torch.uint8)
    t = {
        "codes": torch.cat(words),
        "code_off": torch.tensor(code_off, dtype=torch.int32),
        "blk_params": torch.cat(params),
        "param_off": torch.tensor(param_off, dtype=torch.int32),
        "row_consts": consts,
        "row_nbits": nb.to(torch.uint8),
        "row_block": bl.to(torch.uint8),
        "had": flags,
    }
    if smooth:
        t["col_scale"] = (0.25 + 1.5 * torch.rand(k, generator=g)).to(torch.float16)
    return t


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ---- CPU: host helpers ----------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1])
def test_utils_dequantize_bit_exact(seed):
    t = make_hi2(24, 256, seed=seed)
    hu.validate(t)
    got = hu.dequantize_stored(t)
    assert got.dtype == torch.float16
    assert torch.equal(got.double(), ref_dequantize_stored(t))


def test_utils_slicing_and_concat():
    t = make_hi2(20, 384, seed=3)
    w = ref_dequantize_stored(t)
    rows = hu.select_rows(t, 5, 17)
    hu.validate(rows)
    assert torch.equal(ref_dequantize_stored(rows), w[5:17])
    cols = hu.select_cols(t, 128, 320)
    hu.validate(cols)
    assert torch.equal(ref_dequantize_stored(cols), w[:, 128:320])
    assert torch.equal(cols["had"], t["had"][8:20])
    assert torch.equal(cols["col_scale"], t["col_scale"][128:320])
    both = hu.concat_rows([hu.select_rows(t, 0, 7), hu.select_rows(t, 7, 20)])
    hu.validate(both)
    assert torch.equal(ref_dequantize_stored(both), w)
    # Row-parallel TP: the two column halves add up to the full product.
    x = torch.randn(3, 384, dtype=torch.float64)
    halves = [hu.select_cols(t, 0, 192), hu.select_cols(t, 192, 384)]
    y = ref_linear(x[:, :192], halves[0]) + ref_linear(x[:, 192:], halves[1])
    assert rel_err(y, ref_linear(x, t)) < 1e-12


def test_utils_validate_rejects_bad_offsets():
    t = make_hi2(4, 128, seed=4)
    bad = dict(t)
    bad["code_off"] = t["code_off"].clone()
    bad["code_off"][2] += 1
    with pytest.raises(ValueError):
        hu.validate(bad)
    bad = dict(t)
    bad["row_block"] = torch.full_like(t["row_block"], 128)
    with pytest.raises(ValueError):
        hu.validate(bad)
    with pytest.raises(ValueError):
        hu.select_cols(t, 16, 64)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("had", ["none", "all", "mixed"])
def test_utils_fallback_linear(smooth, had):
    t = make_hi2(16, 256, had=had, smooth=smooth, seed=5)
    x = torch.randn(5, 256, dtype=torch.float32)
    xt = hu.act_transform(x, t) if hu.needs_transform(t) else x
    y = xt.double() @ hu.dequantize_stored(t).double().T
    assert rel_err(y, ref_linear(x, t)) < 1e-6


def test_config_module_selection():
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.layers.quantization.rdna_hint_flex import (
        RdnaHintFlexConfig,
    )
    from vllm.model_executor.layers.quantization.rdna_hint_flex_names import (
        RDNA_HINT_FLEX_QUANT_METHOD,
        RDNA_HINT_FLEX_QUANT_METHOD_NAMES,
    )

    for name in RDNA_HINT_FLEX_QUANT_METHOD_NAMES:
        assert name in QUANTIZATION_METHODS
        assert get_quantization_config(name) is RdnaHintFlexConfig
    cfg = RdnaHintFlexConfig.from_config(
        {"quant_method": RDNA_HINT_FLEX_QUANT_METHOD, "version": "3.2", "mode": "flex"}
    )
    assert cfg.get_name() == RDNA_HINT_FLEX_QUANT_METHOD
    names = []
    for m in ("q_proj", "k_proj", "v_proj"):
        base = f"model.language_model.layers.0.self_attn.{m}"
        names += [f"{base}.hi2_{f}" for f in hu.FIELDS + ("col_scale",)]
    names.append("model.visual.blocks.0.attn.qkv.weight")
    cfg.set_checkpoint_tensor_names(names)
    cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    fields = cfg._module_fields("language_model.model.layers.0.self_attn.qkv_proj")
    assert all(f is not None and "col_scale" in f for f in fields)
    assert cfg._module_fields("visual.blocks.0.attn.qkv") == [None]
    with pytest.raises(ValueError):
        RdnaHintFlexConfig(version="3.1")


# ---- CPU: I8 compute mode reference (spec 6.2) ----------------------------
@pytest.mark.parametrize("smooth", [False, True])
def test_i8_reference(smooth):
    t = make_hi2(32, 512, had="mixed", smooth=smooth, seed=21)
    x = torch.randn(4, 512, dtype=torch.float16)
    xt = hu.act_transform_f32(x, t)
    ref = ref_fwht_groups(
        x.double() / (t["col_scale"].double() if smooth else 1.0), t["had"]
    )
    assert rel_err(xt, ref) < 1e-6
    xq, sx, sumq = hu.act_quant_i8(x, t)
    assert xq.dtype == torch.int8 and int(xq.abs().max()) <= 127
    assert torch.equal(sumq, xq.reshape(4, 32, 16).to(torch.int32).sum(-1))
    # Group-wise int8 adds well under 1% output error (spec 5).
    y = hu.i8_linear_reference(xq, sx, sumq, t)
    assert rel_err(y, ref_linear(x, t)) < 1e-2


# ---- GPU: RDNA2 kernels ---------------------------------------------------
def _kernels_available() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return (
        on_gfx10x()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "rdna_hint_flex_linear_rdna2")
    )


gfx1030_only = pytest.mark.skipif(
    not _kernels_available(), reason="needs gfx1030 and the _rocm_C rdna_hint_flex ops"
)


def _dev(t: dict) -> dict:
    d = {f: hu.canonicalize(t)[f].cuda() for f in hu.FIELDS}
    s = t.get("col_scale")
    d["col_scale"] = (
        s.cuda() if s is not None else torch.empty(0, dtype=torch.float16).cuda()
    )
    return d


def _linear(x: torch.Tensor, d: dict, transform: bool, max_m: int = 8, mode: int = 0):
    from vllm import _custom_ops as ops

    return ops.rdna_hint_flex_linear_rdna2(
        x,
        d["col_scale"],
        d["had"],
        d["codes"],
        d["code_off"],
        d["blk_params"],
        d["param_off"],
        d["row_consts"],
        d["row_nbits"],
        d["row_block"],
        transform,
        max_m,
        mode,
    )


@gfx1030_only
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dequant_kernel_bit_exact(seed):
    from vllm import _custom_ops as ops

    t = make_hi2(72, 512, seed=seed)
    d = _dev(t)
    out = torch.empty(72, 512, dtype=torch.float16, device="cuda")
    ops.rdna_hint_flex_dequant_rdna2(
        d["codes"],
        d["code_off"],
        d["blk_params"],
        d["param_off"],
        d["row_consts"],
        d["row_nbits"],
        d["row_block"],
        out,
    )
    torch.accelerator.synchronize()
    assert torch.equal(out.cpu().double(), ref_dequantize_stored(t))


@gfx1030_only
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("had", ["none", "all", "mixed"])
def test_act_transform_kernel(smooth, had):
    from vllm import _custom_ops as ops

    t = make_hi2(1, 1024, had=had, smooth=smooth, seed=7)
    d = _dev(t)
    x = torch.randn(6, 1024, dtype=torch.float16)
    out = torch.empty(6, 1024, dtype=torch.float16, device="cuda")
    ops.rdna_hint_flex_act_transform_rdna2(x.cuda(), d["col_scale"], d["had"], out)
    xd = x.double()
    if smooth:
        xd = xd / t["col_scale"].double()
    ref = ref_fwht_groups(xd, t["had"])
    # One fp16 rounding of an fp32 result.
    assert rel_err(out, ref) < 1e-3
    assert (out.cpu().double() - ref).abs().max() <= 1e-3 * ref.abs().max() + 1e-6


@gfx1030_only
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 9, 16, 37])
@pytest.mark.parametrize(
    "nbits,blocks",
    [((2,), (32,)), ((4,), (64,)), ((8,), (16,)), ((2, 4, 8), (16, 32, 64))],
)
@pytest.mark.parametrize(
    "had,smooth", [("all", True), ("none", False), ("mixed", True)]
)
def test_linear_kernel_vs_reference(m, nbits, blocks, had, smooth):
    r, k = 200, 1024
    t = make_hi2(r, k, nbits=nbits, blocks=blocks, had=had, smooth=smooth, seed=m)
    d = _dev(t)
    x = torch.randn(m, k, dtype=torch.float16)
    y = _linear(x.cuda(), d, hu.needs_transform(t))
    torch.accelerator.synchronize()
    assert y.shape == (m, r) and y.dtype == torch.float16
    ref = ref_linear(x, t)
    # Error sources: fp16 transformed activation, fp16 output, fp32 dot.
    assert rel_err(y, ref) < 2e-3, (m, nbits, blocks, had, smooth)


@gfx1030_only
@pytest.mark.parametrize("m", [1, 4, 8, 12])
def test_decode_and_prefill_paths_agree(m):
    """Same input through the GEMV and the dequant+GEMM paths."""
    t = make_hi2(96, 768, seed=11)
    d = _dev(t)
    x = torch.randn(m, 768, dtype=torch.float16).cuda()
    y_gemv = _linear(x, d, True, max_m=64)
    y_gemm = _linear(x, d, True, max_m=0)
    torch.accelerator.synchronize()
    assert rel_err(y_gemv, y_gemm) < 1e-3


@gfx1030_only
def test_gemv_strided_output():
    from vllm import _custom_ops as ops

    t = make_hi2(64, 512, had="none", smooth=False, seed=13)
    d = _dev(t)
    x = torch.randn(3, 512, dtype=torch.float16)
    big = torch.zeros(3, 100, dtype=torch.float16, device="cuda")
    out = big[:, 20:84]
    ops.rdna_hint_flex_gemv_rdna2(
        x.cuda(),
        d["codes"],
        d["code_off"],
        d["blk_params"],
        d["param_off"],
        d["row_consts"],
        d["row_nbits"],
        d["row_block"],
        out,
    )
    torch.accelerator.synchronize()
    assert rel_err(out, ref_linear(x, t)) < 2e-3
    assert not big[:, :20].any() and not big[:, 84:].any()


@gfx1030_only
@pytest.mark.parametrize(
    "had,smooth", [("all", True), ("none", False), ("mixed", True)]
)
def test_act_quant_kernel_bit_exact(had, smooth):
    from vllm import _custom_ops as ops

    t = make_hi2(1, 2048, had=had, smooth=smooth, seed=31)
    d = _dev(t)
    x = torch.randn(7, 2048, dtype=torch.float16)
    x[0, :16] = 0  # an all-zero group (sx = 0)
    xq = torch.empty(7, 2048, dtype=torch.int8, device="cuda")
    sx = torch.empty(7, 128, dtype=torch.float16, device="cuda")
    sumq = torch.empty(7, 128, dtype=torch.int32, device="cuda")
    tr = hu.needs_transform(t)
    ops.rdna_hint_flex_act_quant_rdna2(
        x.cuda(), d["col_scale"], d["had"], tr, xq, sx, sumq
    )
    torch.accelerator.synchronize()
    rq, rs, rsum = hu.act_quant_i8(x, t, tr)
    assert torch.equal(sx.cpu(), rs)
    assert torch.equal(xq.cpu(), rq)
    assert torch.equal(sumq.cpu(), rsum)


@gfx1030_only
@pytest.mark.parametrize("m", [1, 3, 8, 13])
@pytest.mark.parametrize(
    "nbits,blocks",
    [((2,), (16,)), ((4,), (32,)), ((8,), (64,)), ((2, 4, 8), (16, 32, 64))],
)
def test_i8_linear_vs_reference(m, nbits, blocks):
    r, k = 160, 1024
    t = make_hi2(r, k, nbits=nbits, blocks=blocks, seed=40 + m)
    d = _dev(t)
    x = torch.randn(m, k, dtype=torch.float16)
    xq, sx, sumq = hu.act_quant_i8(x, t)
    ref = hu.i8_linear_reference(xq, sx, sumq, t)
    y_all = _linear(x.cuda(), d, True, mode=2)
    y_dec = _linear(x.cuda(), d, True, max_m=16, mode=1)
    torch.accelerator.synchronize()
    # Integer dots are exact; fp32 accumulation and the fp16 output remain.
    assert rel_err(y_all, ref) < 1e-3
    assert rel_err(y_dec, ref) < 1e-3
    # The accuracy cost of I8 against the exact product stays small.
    assert rel_err(y_all, ref_linear(x, t)) < 1e-2


# ---- CPU: stacked expert layout (tools/hi2 --moe-layout stacked) ----------
def _write_stack(experts: list[list[dict]]) -> dict:
    """What the writer stores for one group: the experts' parts concatenated
    in expert-major order, rebased offsets, ``had [E, K/16]``, ``col_scale
    [E, K]`` (all or none) and ``part_off``."""
    flat = [p for e in experts for p in e]
    t = hu.concat_rows(flat, check_transform=False)
    out = {f: t[f].clone() for f in hu.FIELDS if f != "had"}
    out["had"] = torch.stack([e[0]["had"] for e in experts])
    if experts[0][0].get("col_scale") is not None:
        out["col_scale"] = torch.stack([e[0]["col_scale"] for e in experts])
    rows = [hu.num_rows(p) for p in flat]
    out["part_off"] = torch.tensor([0] + rows, dtype=torch.int64).cumsum(0).int()
    return out


def _moe_experts(e, hidden, inter, seed, smooth_down=True):
    gate, up, down = [], [], []
    for i in range(e):
        g = make_hi2(inter, hidden, seed=seed + 10 * i, smooth=True)
        u = make_hi2(inter, hidden, seed=seed + 10 * i + 1, smooth=True)
        u["had"], u["col_scale"] = g["had"], g["col_scale"]
        d = make_hi2(hidden, inter, seed=seed + 10 * i + 2, smooth=smooth_down)
        gate.append(g)
        up.append(u)
        down.append(d)
    return gate, up, down


def _moe_layer(e_local, hidden, inter_part, tp=1, rank=0, local=None, e_global=None):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.rdna_hint_flex_moe import (
        RdnaHintFlexMoEMethod,
    )

    layer = torch.nn.Module()
    method = RdnaHintFlexMoEMethod(None, 0)
    method.create_weights(layer, e_local, hidden, inter_part, torch.float16)
    layer.moe_config = SimpleNamespace(tp_size=tp, tp_rank=rank)
    layer.global_num_experts = e_global or e_local
    local = local or {g: g for g in range(e_local)}
    layer._map_global_expert_id_to_local_expert_id = lambda g: local.get(g, -1)
    return layer, method


def _load_per_expert(layer, method, gate, up, down, local, stages=("w13", "w2")):
    for shard, tensors in (("w1", gate), ("w3", up), ("w2", down)):
        stage = "w2" if shard == "w2" else "w13"
        if stage not in stages:
            continue
        for g, t in enumerate(tensors):
            if g in local:
                for f, v in t.items():
                    method.absorb(layer, v, f"{stage}_weight.hi2_{f}", shard, local[g])


def _load_stacked(layer, method, gate, up, down, stages=("w13", "w2")):
    groups = {
        "w13": ("gate_up", [[a, b] for a, b in zip(gate, up)]),
        "w2": ("down", [[d] for d in down]),
    }
    for stage in stages:
        group, experts = groups[stage]
        for f, v in _write_stack(experts).items():
            method.absorb_stack(layer, v, group, f)


def _assert_same_stages(a, b):
    for stage in ("w13", "w2"):
        ta = getattr(a, f"_rdna_hint_flex_{stage}")
        tb = getattr(b, f"_rdna_hint_flex_{stage}")
        assert set(ta) == set(tb)
        for f in ta:
            assert ta[f].dtype == tb[f].dtype, (stage, f)
            assert torch.equal(ta[f], tb[f]), (stage, f)
        assert getattr(a, f"_rdna_hint_flex_{stage}_tr") == getattr(
            b, f"_rdna_hint_flex_{stage}_tr"
        )


@pytest.mark.parametrize(
    "tp,rank,local,smooth_down",
    [
        (1, 0, None, True),
        (1, 0, None, False),
        (2, 0, None, True),
        (2, 1, None, False),
        (1, 0, {2: 0, 3: 1}, True),  # expert parallel: experts 2, 3 local
        (2, 1, {0: 1, 3: 0}, True),  # EP with a non-monotonic local order
    ],
)
def test_moe_stacked_layout_equals_per_expert(tp, rank, local, smooth_down):
    """Loading the stacked group gives exactly the stage tensors of the
    per-expert checkpoint (TP=1 direct, TP>1 / EP through part_off)."""
    e, hidden, inter = 4, 128, 128
    gate, up, down = _moe_experts(e, hidden, inter * tp, 7, smooth_down)
    local = local or {g: g for g in range(e)}
    a, ma = _moe_layer(len(local), hidden, inter, tp, rank, local, e)
    _load_per_expert(a, ma, gate, up, down, local)
    ma.process_weights_after_loading(a)
    b, mb = _moe_layer(len(local), hidden, inter, tp, rank, local, e)
    _load_stacked(b, mb, gate, up, down)
    mb.process_weights_after_loading(b)
    _assert_same_stages(a, b)
    if tp == 1 and len(local) == e:
        assert b._rdna_hint_flex_w13["had"].shape == (e, hidden // 16)
        if not smooth_down:
            assert b._rdna_hint_flex_w2["col_scale"].numel() == 0


def test_moe_stacked_and_per_expert_mixed():
    """A checkpoint can stack one stage and keep the other per expert (the
    writer leaves a group per expert when it cannot stack it)."""
    e, hidden, inter = 4, 128, 64
    gate, up, down = _moe_experts(e, hidden, inter, 11)
    for i in range(1, e, 2):  # mixed smoothing: the writer keeps down per expert
        down[i].pop("col_scale")
    local = {g: g for g in range(e)}
    a, ma = _moe_layer(e, hidden, inter)
    _load_per_expert(a, ma, gate, up, down, local)
    ma.process_weights_after_loading(a)
    b, mb = _moe_layer(e, hidden, inter)
    _load_stacked(b, mb, gate, up, down, stages=("w13",))
    _load_per_expert(b, mb, gate, up, down, local, stages=("w2",))
    mb.process_weights_after_loading(b)
    _assert_same_stages(a, b)


def test_moe_stacked_layout_rejects_bad_stacks():
    e, hidden, inter = 2, 128, 64
    gate, up, down = _moe_experts(e, hidden, inter, 3)
    for mutate, msg in (
        (lambda s: s.__setitem__("part_off", s["part_off"][:-1]), "part_off"),
        (lambda s: s.pop("part_off"), "lacks"),
        (lambda s: s.__setitem__("col_scale", s["col_scale"][:1]), "col_scale"),
        (None, "experts"),  # a stack of one expert for a two-expert layer
    ):
        b, mb = _moe_layer(e, hidden, inter)
        if mutate is None:
            st = _write_stack([[gate[0], up[0]]])
        else:
            st = _write_stack([[a, c] for a, c in zip(gate, up)])
            mutate(st)
        for f, v in st.items():
            mb.absorb_stack(b, v, "gate_up", f)
        _load_stacked(b, mb, gate, up, down, stages=("w2",))
        with pytest.raises(ValueError, match=msg):
            mb.process_weights_after_loading(b)


# ---- GPU: routed experts on a tiny synthetic checkpoint --------------------
def _synthetic_experts(e: int, hidden: int, inter: int, seed: int):
    """Per expert: gate/up sharing one transform, down with its own, mixed
    row widths and blocks (the layout tools/hi2 would write per expert)."""
    gate, up, down = [], [], []
    for i in range(e):
        g = make_hi2(inter, hidden, seed=seed + 10 * i, smooth=True)
        u = make_hi2(inter, hidden, seed=seed + 10 * i + 1, smooth=True)
        u["had"], u["col_scale"] = g["had"], g["col_scale"]
        d = make_hi2(hidden, inter, seed=seed + 10 * i + 2, smooth=i % 2 == 0)
        gate.append(g)
        up.append(u)
        down.append(d)
    return gate, up, down


class _Experts(torch.nn.Module):
    pass


@gfx1030_only
@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("tokens", [1, 3, 9, 40, 300])
def test_moe_vs_dense_reference(tokens, mode):
    """Up to 16 tokens: per-pair GEMV; above (F16 mode): pairs grouped by
    expert through the fused decode-in-GEMM kernel."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.rdna_hint_flex_moe import (
        RdnaHintFlexMoEMethod,
        moe_reference,
    )

    e, hidden, inter, top_k = 4, 256, 128, 2
    gate, up, down = _synthetic_experts(e, hidden, inter, seed=100)
    with torch.device("cuda"):
        layer = _Experts()
        method = RdnaHintFlexMoEMethod(None, mode)
        method.create_weights(layer, e, hidden, inter, torch.float16)
    layer.moe_config = SimpleNamespace(tp_size=1, tp_rank=0)
    layer.activation = "silu"
    for shard, tensors in (("w1", gate), ("w3", up), ("w2", down)):
        param = "w2_weight" if shard == "w2" else "w13_weight"
        for i, t in enumerate(tensors):
            for f, v in t.items():
                method.absorb(layer, v, f"{param}.hi2_{f}", shard, i)
    method.process_weights_after_loading(layer)

    g = torch.Generator().manual_seed(tokens)
    x = (torch.randn(tokens, hidden, generator=g) * 0.5).to(torch.float16)
    ids = torch.stack([torch.randperm(e, generator=g)[:top_k] for _ in range(tokens)])
    if tokens > 1:
        ids[1, 1] = -1  # a pair whose expert is not on this rank
    w = torch.softmax(torch.randn(tokens, top_k, generator=g), -1)
    out = method.apply(layer, x.cuda(), w.cuda(), ids.cuda())
    torch.accelerator.synchronize()
    ref = moe_reference(x, ids, w, gate, up, down)
    assert out.shape == (tokens, hidden)
    # fp16 activations between the stages (transform, gate/up, SiLU product);
    # I8 adds the per-16-column int8 activation error of spec 5.
    assert rel_err(out, ref) < (1e-2 if mode == 0 else 3e-2)


@gfx1030_only
@pytest.mark.parametrize("m,r,k", [(64, 256, 512), (200, 300, 1024), (129, 128, 2048)])
def test_fused_gemm_vs_reference(m, r, k):
    from vllm import _custom_ops as ops

    t = make_hi2(r, k, seed=60 + m)
    d = _dev(t)
    x = torch.randn(m, k, dtype=torch.float16)
    xt = hu.act_transform(x, t).cuda()
    out = torch.empty(m, r, dtype=torch.float16, device="cuda")
    ops.rdna_hint_flex_gemm_rdna2(xt, d, out)
    torch.accelerator.synchronize()
    assert rel_err(out, ref_linear(x, t)) < 2e-3


@gfx1030_only
@pytest.mark.parametrize("smooth", [False, True])
def test_act_quant64_kernel(smooth):
    from vllm import _custom_ops as ops

    t = make_hi2(1, 1024, had="mixed", smooth=smooth, seed=70)
    d = _dev(t)
    x = torch.randn(5, 1024, dtype=torch.float16)
    x[1, :64] = 0  # an all-zero group
    xq = torch.empty(5, 1024, dtype=torch.int8, device="cuda")
    sx = torch.empty(5, 16, dtype=torch.float32, device="cuda")
    sq = torch.empty(5, 64, dtype=torch.float32, device="cuda")
    ops.rdna_hint_flex_act_quant64_rdna2(
        x.cuda(), d["col_scale"], d["had"], True, xq, sx, sq
    )
    torch.accelerator.synchronize()
    rq, rs, rsq = hu.act_quant64(x, t)
    assert torch.equal(sx.cpu(), rs)
    assert torch.equal(xq.cpu(), rq)
    assert torch.allclose(sq.cpu(), rsq, rtol=1e-6, atol=0)


@gfx1030_only
@pytest.mark.parametrize("tile", [0, 1])
@pytest.mark.parametrize("m", [1, 48, 130, 257])
@pytest.mark.parametrize(
    "nbits,blocks",
    [((2,), (64,)), ((4,), (32,)), ((8,), (16,)), ((2, 4, 8), (16, 32, 64))],
)
def test_i8gemm_vs_reference(m, nbits, blocks, tile):
    from vllm import _custom_ops as ops

    r, k = 200, 1024
    t = make_hi2(r, k, nbits=nbits, blocks=blocks, seed=80 + m)
    d = _dev(t)
    x = torch.randn(m, k, dtype=torch.float16)
    xq, sx, sq = hu.act_quant64(x, t)
    out = torch.empty(m, r, dtype=torch.float16, device="cuda")
    ops.rdna_hint_flex_i8gemm_rdna2(xq.cuda(), sx.cuda(), sq.cuda(), d, out, tile)
    torch.accelerator.synchronize()
    assert rel_err(out, hu.i8g64_reference(xq, sx, t)) < 1e-3
    # Per-64 int8 activations stay within the spec 5 error budget.
    assert rel_err(out, ref_linear(x, t)) < 2e-2
