# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling-2-Mini weight-only quantized decode linears on RDNA2 (gfx1030).

``inkling2_qlinear_grouped`` / ``inkling2_qlinear_lm_head`` (int8 group 64, int4 group 32 /
64 weights, fp16 activations, fp32 accumulation, 1 to 4 token rows) and the offline
quantizer / packer of ``qlinear_pack.py``.

The kernel reference is the fp64 matmul with the dequantized weights ``q * scale`` (the
exact weights the kernels multiply with), so the only error left is the fp32 accumulation
order: 2e-5 of the output rms.  The checks themselves live in
``tools/rdna/inkling2/qlinear_check.py`` (which the hand mutants of ``qlinear_mutants.py``
run against: every mutant must fail them); this file runs them over the instances of the
serving build.  The quantizer and packer tests need no GPU.

Run ``pytest tests/kernels/quantization/test_inkling2_qlinear_rdna2.py``.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from vllm.models.inkling2_loop.amd import linear_pack as lp
from vllm.models.inkling2_loop.amd import qlinear_pack as qp
from vllm.platforms import current_platform

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "tools" / "rdna" / "inkling2")
)
import qlinear_check as qc  # noqa: E402

_had_mp_flag = "VLLM_ENABLE_V1_MULTIPROCESSING" in os.environ
import gptq_quant as gq  # noqa: E402  (sets that variable at import time)

if not _had_mp_flag:
    os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)

_HAVE_OPS = (
    current_platform.is_rocm()
    and hasattr(torch.ops, "_rocm_C")
    and hasattr(torch.ops._rocm_C, "inkling2_qlinear_grouped")
    and hasattr(torch.ops._rocm_C, "inkling2_qlinear_lm_head")
)
if _HAVE_OPS:
    from vllm.platforms.rocm import on_gfx10x

    _HAVE_OPS = on_gfx10x()
gfx1030_only = pytest.mark.skipif(
    not _HAVE_OPS, reason="Requires gfx1030 with the inkling2_qlinear ops"
)

# (bits, group, rows per tile, chunks per iteration, waves): the instances of the serving
# build (attention input pack, o_proj) of every format
INSTANCES = sorted(
    {
        (bits, group, *cfg)
        for bits, group in qp.SERVING_FORMATS
        for cfg in (qp.ATTN_IN_CONFIG[bits], qp.O_PROJ_CONFIG[bits])
    }
)
SEGMENTS = [
    (2304, [300, 77, 24, 130], [0, 0, 1, 2]),  # attention input like: 3 inputs
    (3072, [100], [0]),  # o_proj like
]


@pytest.fixture(autouse=True)
def _no_failures():
    qc.FAILS.clear()
    yield
    assert not qc.FAILS, qc.FAILS


def _fits(bits, group, R, cpi, K):
    ck = qp.CK[bits]
    gc = group // ck
    return not (K % (ck * (32 // R)) or (K // (ck * (32 // R))) % gc or cpi % gc)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 2, 3, 4])
@pytest.mark.parametrize("K,rows,seg_x", SEGMENTS)
@pytest.mark.parametrize("bits,group,R,cpi,wv", INSTANCES)
def test_grouped_matches_dequantized_fp64(bits, group, R, cpi, wv, K, rows, seg_x, M):
    if not _fits(bits, group, R, cpi, K):
        pytest.skip("shape not tiled by this instance")
    be = qc.OpBackend()
    tag = f"w{bits}g{group} R={R} cpi={cpi} M={M} K={K}"
    qc.check_grouped(qp, be, bits, group, R, cpi, wv, M, K, rows, seg_x, tag)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 3, 4])
@pytest.mark.parametrize("bits,group,R,cpi,wv", INSTANCES)
def test_every_weight_element_is_read_exactly(bits, group, R, cpi, wv, M):
    """Unit-vector activations: the output is the dequantized weight element, bit for bit
    (a wrong nibble / byte selector, scale pair or lane map changes a value)."""
    K = 3072
    if not _fits(bits, group, R, cpi, K):
        pytest.skip("shape not tiled by this instance")
    tag = f"w{bits}g{group} R={R} cpi={cpi} M={M}"
    qc.check_onehot(qp, qc.OpBackend(), bits, group, R, cpi, wv, M, K, 37, tag, 3)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 2, 4])
@pytest.mark.parametrize("bits,group,R,cpi,wv", INSTANCES)
def test_epilogues_masks_and_fp16_outputs(bits, group, R, cpi, wv, M):
    if not _fits(bits, group, R, cpi, 2304):
        pytest.skip("shape not tiled by this instance")
    tag = f"w{bits}g{group} R={R} cpi={cpi} M={M}"
    qc.check_epilogue(qp, qc.OpBackend(), bits, group, R, cpi, wv, M, 2304, tag)


@gfx1030_only
@pytest.mark.parametrize("M", [1, 2, 3, 4])
@pytest.mark.parametrize("bits,group", sorted(qp.SERVING_FORMATS))
def test_lm_head_logits_argmax_ties_and_mask(bits, group, M):
    """Softcapped logits, the ``-inf`` vocabulary mask, the fused argmax partials with
    ties inside a wave, across waves and across tiles of a lane (the lowest vocabulary id
    wins), ``col_base``, and the partials through ``inkling2_linear_argmax_reduce``."""
    tag = f"lm_head w{bits}g{group} M={M}"
    qc.check_lm_head(
        qp, qc.OpBackend(), bits, group, qp.LM_HEAD_CPI, M, 2304, 40007, 64, tag
    )


@gfx1030_only
def test_hybrid_packs_run_the_right_kernel():
    """Serving packs: the quantized kernels up to ``MAX_ROWS`` rows, the fp16 pack above,
    every output against the fp64 oracle and bit-exact against the pack that must run."""
    qc.check_production_packs(qp, qc.OpBackend())


@gfx1030_only
def test_rejects_bad_arguments():
    dev = "cuda"
    w = torch.randn(64, 2304, device=dev).half()
    wq, sc = qp.pack_weight(w, 8, 64, 8, 8)
    x = torch.randn(5, 2304, device=dev).half()
    out = torch.empty(5, 64, device=dev, dtype=torch.float16)
    from vllm import _custom_ops as ops

    with pytest.raises(RuntimeError):  # more than 4 rows
        ops.inkling2_qlinear_grouped(
            [x], wq, sc, [out], [0], [64], [], [], [], 8, 64, 8, 8, 8
        )
    x4 = x[:4].contiguous()
    out4 = out[:4].contiguous()
    with pytest.raises(RuntimeError):  # no instance of this format
        ops.inkling2_qlinear_grouped(
            [x4], wq, sc, [out4], [0], [64], [], [], [], 8, 32, 8, 8, 8
        )
    with pytest.raises(RuntimeError):  # a scale tensor of another shape
        ops.inkling2_qlinear_grouped(
            [x4],
            wq,
            sc[:-1].contiguous(),
            [out4],
            [0],
            [64],
            [],
            [],
            [],
            8,
            64,
            8,
            8,
            8,
        )
    ops.inkling2_qlinear_grouped(
        [x4], wq, sc, [out4], [0], [64], [], [], [], 8, 64, 8, 8, 8
    )


# --- the quantizer and the packer (CPU) ---------------------------------------------


@pytest.mark.parametrize(
    "spec,want",
    [
        ("w8", (8, 64, False)),
        ("w8g64", (8, 64, False)),
        ("W4", (4, 32, True)),
        ("w4g32", (4, 32, True)),
        ("w4g32n", (4, 32, False)),
        ("w4g64", (4, 64, True)),
    ],
)
def test_parse_format(spec, want):
    assert qp.parse_format(spec) == want


@pytest.mark.parametrize("spec", ["", "8", "w3", "w8g128", "w4g128", "w4g16", "int8"])
def test_parse_format_rejects(spec):
    with pytest.raises(ValueError):
        qp.parse_format(spec)


@pytest.mark.parametrize("bits,group", sorted(qp.SERVING_FORMATS))
@pytest.mark.parametrize("clip", [False, True])
def test_quantizer_error_bounds(bits, group, clip):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(96, 512, generator=g) * (0.01 + torch.rand(96, 1, generator=g))
    w[3] = 0.0  # an all-zero row: the scale floor keeps it finite
    w[5, 7] = 40.0  # an outlier
    q, s = qp.quantize_rows(w, bits, group, clip)
    deq = qp.dequantize_rows(q, s, group)
    qmax = 2 ** (bits - 1) - 1
    assert int(q.min()) >= -qmax - 1 and int(q.max()) <= qmax
    assert bool(torch.isfinite(s).all()) and float(s.float().min()) >= 2.0**-14
    err = (deq - w).abs().view(96, -1, group)
    amax = w.abs().view(96, -1, group).amax(-1, keepdim=True)
    # round-to-nearest: at most half a step, the step being scale * 1 (clipped groups clamp
    # the largest values instead: bounded by the clip ratio)
    step = s.float().unsqueeze(-1)
    if clip:
        assert bool((err <= 0.5 * step + (1 - 0.65) * amax + 1e-6).all())
    else:
        assert bool((err <= 0.5 * step + 1e-6).all())
    # the clip search never makes a group worse than the plain absmax scale
    if clip:
        q0, s0 = qp.quantize_rows(w, bits, group, False)
        e0 = ((qp.dequantize_rows(q0, s0, group) - w) ** 2).view(96, -1, group).sum(-1)
        e1 = (err**2).sum(-1)
        assert bool((e1 <= e0 + 1e-9).all())


@pytest.mark.parametrize("bits,group", sorted(qp.SERVING_FORMATS))
@pytest.mark.parametrize("R,cpi", [(8, 8), (8, 4), (4, 4), (32, 8)])
def test_packed_weights_are_a_permutation(bits, group, R, cpi):
    """The tile layout loses and duplicates nothing: the multiset of stored values equals
    the multiset of the quantized values (zero point added), padding rows are zero values."""
    K = 3072
    ck = qp.CK[bits]
    gc = group // ck
    if not _fits(bits, group, R, cpi, K):
        pytest.skip("shape not tiled")
    g = torch.Generator().manual_seed(1)
    w = torch.randn(70, K, generator=g) * 0.02  # 70 rows: padded to a multiple of R
    wq, sc = qp.pack_weight(w, bits, group, R, cpi)
    q, s = qp.quantize_rows(w, bits, group)
    zp = 1 << (bits - 1)
    u = (q.to(torch.int16) + zp).flatten()
    pad_rows = (-70) % R
    padded = torch.cat([u, torch.full((pad_rows * K,), zp, dtype=torch.int16)])
    if bits == 8:
        stored = wq.flatten().to(torch.int16)
    else:
        b = wq.flatten().to(torch.int32)
        stored = torch.stack([b & 15, b >> 4], -1).flatten().to(torch.int16)
    assert stored.numel() == padded.numel()
    assert torch.equal(torch.sort(stored).values, torch.sort(padded).values)
    # scales: one fp16 per group (and per padding row group), padded to whole iterations
    assert sc.dtype == torch.float16
    assert sc.numel() >= (70 + pad_rows) * (K // group)


def test_quantize_linear_pack_keeps_segments():
    g = torch.Generator().manual_seed(2)
    parts = [
        (
            "a",
            torch.randn(300, 2304, generator=g).half(),
            0,
            torch.float16,
            lp.ACT_NONE,
            1.0,
        ),
        (
            "b",
            torch.randn(24, 2304, generator=g).half(),
            1,
            torch.float16,
            lp.ACT_NONE,
            1.0,
        ),
        (
            "c",
            torch.randn(77, 2304, generator=g).half(),
            0,
            torch.float16,
            lp.ACT_NONE,
            1.0,
        ),
    ]
    pack = lp._pack(parts, torch.float16)
    q = qp.quantize_linear_pack(pack, "w8", qp.ATTN_IN_CONFIG[8])
    assert q.seg_n == pack.seg_n and q.seg_x == pack.seg_x
    assert q.seg_act == pack.seg_act and q.seg_valid == pack.seg_valid
    # every segment is padded to whole tiles on its own
    tiles = sum(-(-n // q.rows_per_tile) for n in pack.seg_n)
    assert q.weight.shape[0] == tiles
    assert q.nbytes < pack.weight.numel() * 2 * 0.55  # about half the bytes of fp16


def test_levers_are_off_by_default(monkeypatch):
    for name in ("LM_HEAD", "ATTN_IN", "O_PROJ", "ATTN"):
        monkeypatch.delenv(f"VLLM_INKLING2_{name}_QUANT", raising=False)
    w = torch.randn(64, 2304).half()
    pack = lp._pack([("o", w, 0, torch.float16, lp.ACT_NONE, 1.0)], torch.float16)
    assert qp.maybe_quantize_attn_in(pack) is pack
    assert qp.maybe_quantize_o_proj(pack) is pack
    head = lp.LMHeadPack(w, 64, 0)
    assert qp.maybe_quantize_lm_head(head) is head
    monkeypatch.setenv("VLLM_INKLING2_ATTN_QUANT", "off")
    assert qp.maybe_quantize_o_proj(pack) is pack


def _correlated_inputs(K, tokens, seed):
    g = torch.Generator().manual_seed(seed)
    scale = (torch.rand(K, generator=g) ** 2 * 3).unsqueeze(0)
    x = torch.randn(tokens, K, generator=g) @ (torch.randn(K, K, generator=g) * scale)
    return x * 0.1


@pytest.mark.parametrize("bits,group", sorted(qp.SERVING_FORMATS))
def test_gptq_with_an_identity_hessian_is_round_to_nearest(bits, group):
    """No input correlation, nothing to compensate: the solver returns exactly the integers
    and scales of the round-to-nearest quantizer (same clip search)."""
    g = torch.Generator().manual_seed(4)
    w = torch.randn(64, 512, generator=g) * 0.02
    q, s = gq.gptq(w, torch.eye(512), bits, group, clip=bits == 4)
    q0, s0 = qp.quantize_rows(w, bits, group, bits == 4)
    assert torch.equal(q, q0) and torch.equal(s, s0)


@pytest.mark.parametrize("bits,group", sorted(qp.SERVING_FORMATS))
def test_gptq_beats_round_to_nearest_on_the_output_error(bits, group):
    x = _correlated_inputs(512, 3000, 0)
    g = torch.Generator().manual_seed(5)
    w = torch.randn(96, 512, generator=g) * 0.02
    q, s = gq.gptq(w, x.t() @ x, bits, group, clip=bits == 4)
    qmax = 2 ** (bits - 1) - 1
    assert q.dtype == torch.int8 and s.dtype == torch.float16
    assert int(q.min()) >= -qmax - 1 and int(q.max()) <= qmax
    q0, s0 = qp.quantize_rows(w, bits, group, bits == 4)

    def out_err(wh):  # the energy of the output error on the calibration inputs
        return float((((w - wh) @ x.t()) ** 2).sum() / ((w @ x.t()) ** 2).sum())

    e_gptq = out_err(qp.dequantize_rows(q, s, group))
    e_rtn = out_err(qp.dequantize_rows(q0, s0, group))
    assert (
        e_gptq < 0.85 * e_rtn
    )  # about 1.6x better on this input; 1.0x without compensation
    assert gq.proxy_error(
        w, qp.dequantize_rows(q, s, group), x.t() @ x
    ) == pytest.approx(e_gptq, rel=2e-3)


@pytest.mark.parametrize("bits,group", sorted(qp.SERVING_FORMATS))
def test_gptq_is_independent_of_the_block_size(bits, group):
    """The lazy block updates are exact: one block of all the columns (every compensation
    inside the block) and blocks of 128 (the rest propagated between blocks) give the same
    integers up to a rare rounding tie."""
    x = _correlated_inputs(512, 3000, 0)
    g = torch.Generator().manual_seed(5)
    w = torch.randn(96, 512, generator=g) * 0.02
    h = x.t() @ x
    q1, s1 = gq.gptq(w, h, bits, group, bits == 4, blocksize=128)
    q2, s2 = gq.gptq(w, h, bits, group, bits == 4, blocksize=512)
    assert float((q1 != q2).float().mean()) < 0.005
    assert float((s1 != s2).float().mean()) < 0.005


def test_calibrated_file_drives_the_pack(tmp_path, monkeypatch):
    """The integers of ``VLLM_INKLING2_QUANT_FILE`` end up in the tile-packed weight."""
    g = torch.Generator().manual_seed(6)
    parts = [
        (
            "a",
            torch.randn(40, 2304, generator=g).half(),
            0,
            torch.float16,
            lp.ACT_NONE,
            1.0,
        ),
        (
            "b",
            torch.randn(24, 2304, generator=g).half(),
            1,
            torch.float16,
            lp.ACT_NONE,
            1.0,
        ),
        (
            "c",
            torch.randn(17, 2304, generator=g).half(),
            0,
            torch.float16,
            lp.ACT_NONE,
            1.0,
        ),
    ]
    pack = lp._pack(parts, torch.float16)
    cfg = qp.ATTN_IN_CONFIG[4]
    monkeypatch.delenv("VLLM_INKLING2_QUANT_FILE", raising=False)
    rtn = qp.quantize_linear_pack(
        pack, "w4g32", cfg, layer=7
    )  # no file: round to nearest
    weights, lo = {}, 0
    for name, n in zip(pack.names, pack.seg_n):
        q, s = qp.quantize_rows(pack.weight[lo : lo + n], 4, 32, True)
        weights[f"7.{name}"] = {"q": q, "s": s}
        lo += n
    path = tmp_path / "calibrated.pt"

    def load(layer):
        torch.save({"bits": 4, "group": 32, "weights": weights}, path)
        monkeypatch.setenv("VLLM_INKLING2_QUANT_FILE", str(path))
        monkeypatch.setattr(qp, "_QUANT_FILE", None)
        return qp.quantize_linear_pack(pack, "w4g32", cfg, layer=layer)

    same = load(
        7
    )  # the file holds the round-to-nearest integers: the pack is unchanged
    assert torch.equal(same.weight, rtn.weight) and torch.equal(same.scale, rtn.scale)
    q = weights["7.b"]["q"]
    q[5, 9] = 3 if int(q[5, 9]) != 3 else 4  # one integer of one segment
    one = load(7)
    assert int((one.weight != rtn.weight).sum()) == 1  # exactly that nibble's byte
    assert torch.equal(one.scale, rtn.scale)
    other = qp.quantize_linear_pack(
        pack, "w4g32", cfg, layer=8
    )  # layers not in the file
    assert torch.equal(other.weight, rtn.weight)
    with pytest.raises(ValueError, match="holds w4g32"):  # another format is an error
        qp.quantize_linear_pack(pack, "w8", qp.ATTN_IN_CONFIG[8], layer=7)
    weights["7.b"]["q"] = q[:, :64].contiguous()
    with pytest.raises(ValueError, match="calibrated"):  # a wrong shape is an error
        load(7)
