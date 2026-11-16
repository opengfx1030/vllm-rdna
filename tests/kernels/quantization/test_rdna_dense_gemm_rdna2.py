# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Dense fp16 prefill GEMM of the HI2 path (``rdna_hint_flex_dense_gemm_rdna2``).

``out[M, N] = x[M, K] w[N, K]^T`` with fp16 operands, fp32 accumulation
(``v_dot2_f32_f16``) and one fp16 rounding, the layout of ``F.linear``: the GEMM
that follows the rdna_hint_flex dequant pass above ``DECODE_MAX_M`` (docs/rdna2/
hi2-prefill-gemm2-2026-11-15.md). Selected by ``RDNA_HINT_FLEX_PREFILL_GEMM=hip``.

CPU tests (run anywhere): the dispatch-table header is well formed, the table
generator's pick rule and design list, the environment switch and what
``RdnaHintFlexLinearMethod.apply`` hands the op (nothing extra when off).

GPU tests (gfx1030 with the ``_rocm_C`` dense GEMM op): every design against an
fp32 reference and against rocBLAS at the five Qwen3.8-27B HI2 TP2 shapes for
M in {1, 27, 512, 700, 1023, 1024, 1031, 1500, 2046, 2048} (the error table is
printed with ``-s`` and written to ``$RDNA_DENSE_GEMM_ERR_CSV`` when set),
tails (N off the tile grid, tiny N), strided operands, the fallback for what a
design cannot run, the table-driven auto mode, and the composite
``rdna_hint_flex_linear_rdna2`` with the switch on and off (off = bit-identical
to dequant + ``torch.mm``). They skip cleanly without a V620-class GPU.

Run ``pytest tests/kernels/quantization/test_rdna_dense_gemm_rdna2.py``.
"""

import csv
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import regex as re
import torch

REPO = Path(__file__).resolve().parents[3]
SWEEP_DIR = REPO / "tools" / "rdna" / "rdna_hint_flex" / "gemm_sweep"
TABLE_CUH = REPO / "csrc" / "rocm" / "rdna_hint_flex_dense_gemm_table.cuh"
sys.path.insert(0, str(SWEEP_DIR))
gen: Any = importlib.import_module("gen_dispatch")

# Qwen3.8-27B HI2 TP2, per rank: (N, K); o_proj has the out_proj shape.
TP2_SLOTS = {
    "gate_up": (17408, 5120),
    "down": (5120, 8704),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "out_proj": (5120, 3072),
}
M_LIST = (1, 27, 512, 700, 1023, 1024, 1031, 1500, 2046, 2048)
NUM_DESIGNS = len(gen.DESIGNS)
REL_FRO_TOL = 1e-3  # vs fp32 (rocBLAS itself ~2e-4: the fp16 output rounding)


# ---- CPU: the table, the generator, the switch ------------------------------
def parse_table() -> dict[tuple[int, int], dict]:
    text = TABLE_CUH.read_text()
    anchors = {
        name: [(int(m), int(d)) for m, d in re.findall(r"\{(\d+), (-?\d+)\}", body)]
        for name, body in re.findall(
            r"kAnchors_(\d+_\d+)\[\] = \{(.*?)\n\};", text, flags=re.S
        )
    }
    slots = {}
    for n, k, lo, hi, cnt, name in re.findall(
        r"\{(\d+), (\d+), (\d+), (\d+), (\d+), kAnchors_(\d+_\d+)\}", text
    ):
        a = anchors[name]
        assert name == f"{n}_{k}" and len(a) == int(cnt)
        slots[(int(n), int(k))] = {"lo": int(lo), "hi": int(hi), "anchors": a}
    return slots


def test_table_header_is_well_formed():
    slots = parse_table()
    assert set(slots) == set(TP2_SLOTS.values())
    for (n, k), s in slots.items():
        ms = [m for m, _ in s["anchors"]]
        assert ms == sorted(set(ms)) and ms, (n, k)
        assert s["lo"] <= ms[0] and s["hi"] >= ms[-1]
        assert all(-1 <= d < NUM_DESIGNS for _, d in s["anchors"])
        bk = {0: 32, 1: 32, 2: 64, 3: 32}
        assert k % max(bk[d] for _, d in s["anchors"] if d >= 0) == 0


def test_generator_design_list_matches_kernel():
    assert gen.kernel_design_names() == gen.DESIGNS


def _rows(kind, variant, n, k, m, us):
    return {
        "kind": kind, "variant": variant, "shape": "x", "N": n, "K": k, "M": m, "us": us,
    }  # fmt: skip


def test_generator_pick_rule(tmp_path):
    d0, d3 = gen.DESIGNS[0], gen.DESIGNS[3]
    rows = [
        # M=512: d3 is 8 % faster than the better rocBLAS row -> design 3
        _rows("rocblas_heur", "default", 64, 64, 512, 110.0),
        _rows("rocblas_tuned", "tunableop", 64, 64, 512, 100.0),
        _rows("dense", d0, 64, 64, 512, 104.0),
        _rows("dense", d3, 64, 64, 512, 92.0),
        # M=700: best design only 0.5 % faster: stays on rocBLAS
        _rows("rocblas_heur", "default", 64, 64, 700, 200.0),
        _rows("dense", d0, 64, 64, 700, 199.0),
        # M=1024: slower: rocBLAS; a "hip" row from the production bench counts as well
        _rows("rocblas_heur", "default", 64, 64, 1024, 300.0),
        _rows("hip", d3, 64, 64, 1024, 310.0),
        # a design row without any rocBLAS row of its cell is ignored
        _rows("dense", d0, 64, 64, 2048, 10.0),
    ]
    p = tmp_path / "a.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    slots = gen.build(gen.load([p]), 1.01, 0.75, 2.0)
    got = {m: d for m, d, _, _ in slots[(64, 64)]}
    assert got == {512: 3, 700: -1, 1024: -1}
    # later files override earlier rows of the same (kind, variant, N, K, M)
    p2 = tmp_path / "b.csv"
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerow(_rows("dense", d0, 64, 64, 700, 150.0))
    got = {
        m: d for m, d, _, _ in gen.build(gen.load([p, p2]), 1.01, 0.75, 2.0)[(64, 64)]
    }
    assert got[700] == 0


def test_generator_composite_veto(tmp_path):
    d0 = gen.DESIGNS[0]

    def write(name, rows):
        p = tmp_path / name
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return p

    kernel = write(
        "k.csv",
        [
            _rows("rocblas_heur", "default", 64, 64, 512, 110.0),
            _rows("hip", d0, 64, 64, 512, 100.0),
            _rows("rocblas_heur", "default", 64, 64, 1024, 210.0),
            _rows("hip", d0, 64, 64, 1024, 200.0),
        ],
    )
    # the whole composite op: rocBLAS wins at M=512 (the dequant pass leaves the
    # scratch hot), the kernel still wins at M=1024
    comp = write(
        "c.csv",
        [
            _rows("rocblas_heur", "default", 64, 64, 512, 130.0),
            _rows("hip", "prefill_gemm=1", 64, 64, 512, 131.0),
            _rows("rocblas_heur", "default", 64, 64, 1024, 230.0),
            _rows("hip", "prefill_gemm=1", 64, 64, 1024, 215.0),
        ],
    )
    veto = gen.load_veto([comp], 1.01)
    assert veto == {(64, 64, 512)}
    slots = gen.build(gen.load([kernel]), 1.01, 0.75, 2.0, veto)
    assert {m: d for m, d, _, _ in slots[(64, 64)]} == {512: -1, 1024: 0}


def test_env_switch_values(monkeypatch):
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf

    monkeypatch.delenv("RDNA_HINT_FLEX_PREFILL_GEMM", raising=False)
    assert rhf._prefill_gemm_from_env() == 0
    for v, want in (
        ("", 0),
        ("0", 0),
        ("off", 0),
        ("rocblas", 0),
        ("hip", 4),
        (" HIP ", 4),
        ("hip-notail", 1),
    ):
        monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", v)
        assert rhf._prefill_gemm_from_env() == want, v
    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "triton")
    with pytest.raises(ValueError, match="RDNA_HINT_FLEX_PREFILL_GEMM"):
        rhf._prefill_gemm_from_env()


def _fake_segment():
    z = torch.zeros(1)
    return {
        "transform": False, "col_scale": z, "had": z, "codes": z, "code_off": z,
        "blk_params": z, "param_off": z, "row_consts": z, "row_nbits": z, "row_block": z,
    }  # fmt: skip


@pytest.mark.parametrize(
    "env, mode", [(None, 0), ("off", 0), ("hip", 4), ("hip-notail", 1)]
)
def test_apply_hands_the_op_the_switch(monkeypatch, env, mode):
    """What apply() passes to the op: nothing extra when off, the prefill_gemm mode
    with the switch on."""
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf

    if env is None:
        monkeypatch.delenv("RDNA_HINT_FLEX_PREFILL_GEMM", raising=False)
    else:
        monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", env)
    ns = types.SimpleNamespace(
        rdna_hint_flex_linear_rdna2=object(), rdna_hint_flex_dense_gemm_rdna2=object()
    )
    monkeypatch.setattr(rhf, "_rdna2_kernels_available", lambda: True)
    monkeypatch.setattr(torch.ops, "_rocm_C", ns, raising=False)
    method = rhf.RdnaHintFlexLinearMethod(False)
    assert method.use_kernels and method.prefill_gemm == mode

    seen: list[tuple] = []

    def spy(*args):
        seen.append(args)
        return torch.zeros(args[0].shape[0], 4)

    ns.rdna_hint_flex_linear_rdna2 = spy
    layer = types.SimpleNamespace(_rdna_hint_flex_segments=[_fake_segment()])
    method.apply(layer, torch.zeros(3, 64))
    (args,) = seen
    # decode_max_m, mode, silu_mul are always there; prefill_gemm only when on
    assert len(args) == 14 + bool(mode), len(args)
    if mode:
        assert args[-1] == mode


def test_switch_off_without_the_op(monkeypatch):
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf

    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "hip")
    monkeypatch.setattr(rhf, "_rdna2_kernels_available", lambda: True)
    monkeypatch.setattr(
        torch.ops,
        "_rocm_C",
        types.SimpleNamespace(rdna_hint_flex_linear_rdna2=object()),
        raising=False,
    )
    assert rhf.RdnaHintFlexLinearMethod(False).prefill_gemm == 0


def test_switch_ignored_off_rdna2(monkeypatch):
    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf

    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "hip")
    monkeypatch.setattr(rhf, "_rdna2_kernels_available", lambda: False)
    method = rhf.RdnaHintFlexLinearMethod(False)
    assert not method.use_kernels and method.prefill_gemm == 0


# ---- GPU --------------------------------------------------------------------
def _gfx1030_with_op() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return (
        on_gfx10x()
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1030")
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "rdna_hint_flex_dense_gemm_rdna2")
    )


gfx1030_only = pytest.mark.skipif(
    not _gfx1030_with_op(),
    reason="needs a gfx1030 GPU and the _rocm_C rdna_hint_flex_dense_gemm_rdna2 op",
)

DESIGN_BK = {0: 32, 1: 32, 2: 64, 3: 32, 4: 64, 5: 64}


def errs(c: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = c.float() - ref
    return float(d.abs().max()), float(d.norm() / ref.norm().clamp_min(1e-30))


_ERR_ROWS: list[dict] = []


@pytest.fixture(scope="module", autouse=True)
def _emit_error_table():
    yield
    if not _ERR_ROWS:
        return
    head = "slot,N,K,M,design,max_abs,rel_fro,rb_max_abs,rb_rel_fro"
    lines = [
        ",".join(
            str(r[c]) for c in head.split(",")
        ) for r in _ERR_ROWS
    ]  # fmt: skip
    print("\n" + head + "\n" + "\n".join(lines))
    path = os.environ.get("RDNA_DENSE_GEMM_ERR_CSV")
    if path:
        Path(path).write_text(head + "\n" + "\n".join(lines) + "\n")


@gfx1030_only
@pytest.mark.parametrize("slot", list(TP2_SLOTS))
def test_designs_match_fp32_and_rocblas(slot):
    from vllm import _custom_ops as ops

    n, k = TP2_SLOTS[slot]
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).half()
    for m in M_LIST:
        x = torch.randn(m, k, device="cuda", generator=g).half()
        ref = x.float() @ w.float().T
        rb = x @ w.T
        rb_abs, rb_rel = errs(rb, ref)
        outs = {}
        for design in range(NUM_DESIGNS):
            if k % DESIGN_BK[design]:
                continue
            out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
            ops.rdna_hint_flex_dense_gemm_rdna2(x, w, out, design)
            abs_e, rel_e = errs(out, ref)
            _ERR_ROWS.append(
                dict(slot=slot, N=n, K=k, M=m, design=design,
                     max_abs=f"{abs_e:.5f}", rel_fro=f"{rel_e:.3e}",
                     rb_max_abs=f"{rb_abs:.5f}", rb_rel_fro=f"{rb_rel:.3e}")
            )  # fmt: skip
            assert torch.isfinite(out).all(), (slot, m, design)
            assert rel_e <= REL_FRO_TOL, (slot, m, design, rel_e)
            assert rel_e <= 1.5 * rb_rel + 1e-4, (slot, m, design, rel_e, rb_rel)
            assert abs_e <= max(2 * rb_abs, 0.02), (slot, m, design, abs_e, rb_abs)
            outs[design] = out
        # designs differ only by summation order inside fp32 + one fp16 rounding
        first = next(iter(outs.values()))
        for design, out in outs.items():
            assert (out.float() - first.float()).abs().max() <= 0.02, (slot, m, design)
        # auto mode: the table's design (bit-identical to the forced one) or rocBLAS
        auto = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
        ops.rdna_hint_flex_dense_gemm_rdna2(x, w, auto)
        pick = ops.rdna_hint_flex_dense_gemm_pick_rdna2(m, n, k)
        assert -1 <= pick < NUM_DESIGNS
        want = rb if pick < 0 else outs[pick]
        assert torch.equal(auto, want), (slot, m, pick)


@gfx1030_only
@pytest.mark.parametrize(
    "n, k",
    [(130, 64), (8, 64), (4104, 192), (257, 128), (17409, 5120), (5121, 3072)],
)
@pytest.mark.parametrize("m", [1, 49, 127, 129, 1000])
def test_tails_off_the_tile_grid(n, k, m):
    from vllm import _custom_ops as ops

    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).half()
    x = torch.randn(m, k, device="cuda", generator=g).half()
    ref = x.float() @ w.float().T
    for design in range(NUM_DESIGNS):
        if k % DESIGN_BK[design]:
            continue
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
        ops.rdna_hint_flex_dense_gemm_rdna2(x, w, out, design)
        _, rel_e = errs(out, ref)
        assert torch.isfinite(out).all() and rel_e <= REL_FRO_TOL, (n, k, m, design)


@gfx1030_only
def test_strided_operands():
    """A row stride above K (16-byte aligned) on x, w and out is honoured."""
    from vllm import _custom_ops as ops

    m, n, k = 300, 1100, 256
    g = torch.Generator(device="cuda").manual_seed(2)
    xb = torch.randn(m, k + 64, device="cuda", generator=g).half()
    wb = (torch.randn(n, k + 32, device="cuda", generator=g) * 0.05).half()
    x, w = xb[:, :k], wb[:, :k]
    ref = x.float() @ w.float().T
    for design in range(NUM_DESIGNS):
        ob = torch.full((m, n + 40), float("nan"), device="cuda", dtype=torch.float16)
        out = ob[:, :n]
        ops.rdna_hint_flex_dense_gemm_rdna2(x, w, out, design)
        assert torch.isfinite(out).all() and errs(out, ref)[1] <= REL_FRO_TOL, design
        assert torch.isnan(ob[:, n:]).all(), "wrote past the row"


@gfx1030_only
def test_unsupported_problems_fall_back_or_raise():
    from vllm import _custom_ops as ops

    g = torch.Generator(device="cuda").manual_seed(3)
    # K not a multiple of the design's K step: forced raises, auto uses rocBLAS
    m, n, k = 64, 256, 96  # 96 % 64 != 0 (design 2); % 32 == 0 (designs 0, 1, 3)
    x = torch.randn(m, k, device="cuda", generator=g).half()
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).half()
    out = torch.empty(m, n, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="cannot run"):
        ops.rdna_hint_flex_dense_gemm_rdna2(x, w, out, 2)
    ops.rdna_hint_flex_dense_gemm_rdna2(x, w, out, -1)
    assert torch.equal(out, x @ w.T)
    # a pointer off 16 bytes: forced raises, the table path stays correct
    xb = torch.randn(m * k + 1, device="cuda", generator=g).half()
    xm = xb[1:].view(m, k)
    ref = xm.float() @ w.float().T
    with pytest.raises(RuntimeError, match="cannot run"):
        ops.rdna_hint_flex_dense_gemm_rdna2(xm, w, out, 0)
    ops.rdna_hint_flex_dense_gemm_rdna2(xm, w, out, -1)
    assert errs(out, ref)[1] <= REL_FRO_TOL
    # empty M is a no-op
    ops.rdna_hint_flex_dense_gemm_rdna2(x[:0], w, out[:0], 0)
    # dtype and shape contracts
    with pytest.raises(RuntimeError, match="fp16"):
        ops.rdna_hint_flex_dense_gemm_rdna2(x.float(), w.float(), out.float(), 0)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ops.rdna_hint_flex_dense_gemm_rdna2(x, w[:10], out, 0)


@gfx1030_only
def test_pick_outside_the_validated_set_is_rocblas():
    from vllm import _custom_ops as ops

    pick = ops.rdna_hint_flex_dense_gemm_pick_rdna2
    assert pick(2048, 12345, 5120) == -1  # not one of the five slots
    assert pick(1, 17408, 5120) == -1  # below the table's M range
    assert pick(0, 17408, 5120) == -1
    assert pick(2048, 17408, 5120) >= 0  # a slot, the main prefill chunk


def _hi2(n: int, k: int, seed: int) -> dict:
    sys.path.insert(0, str(REPO / "tools" / "rdna" / "rdna_hint_flex"))
    from bench_kernel import make

    t = make(n, k, int4_frac=0.5, seed=seed)
    return {f: v.cuda() for f, v in t.items()}


@gfx1030_only
@pytest.mark.parametrize("m", [49, 512, 700, 1031])
def test_composite_switch_off_is_dequant_plus_mm_and_on_matches(m):
    from vllm import _custom_ops as ops

    n, k = TP2_SLOTS["out_proj"]
    t = _hi2(n, k, seed=4)
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(m, k, device="cuda", generator=g).half()
    lin = (t["col_scale"], t["had"], t["codes"], t["code_off"], t["blk_params"],
           t["param_off"], t["row_consts"], t["row_nbits"], t["row_block"], False, 48)  # fmt: skip
    off = ops.rdna_hint_flex_linear_rdna2(x, *lin)
    off0 = ops.rdna_hint_flex_linear_rdna2(x, *lin, 0, False, 0)
    wd = torch.empty(n, k, device="cuda", dtype=torch.float16)
    ops.rdna_hint_flex_dequant_rdna2(
        t["codes"], t["code_off"], t["blk_params"], t["param_off"],
        t["row_consts"], t["row_nbits"], t["row_block"], wd,
    )  # fmt: skip
    ref = x @ wd.T
    assert torch.equal(off, ref) and torch.equal(off0, ref)  # flag off: unchanged
    on = ops.rdna_hint_flex_linear_rdna2(x, *lin, 0, False, 1)
    _, rel = errs(on, ref.float())
    assert rel <= REL_FRO_TOL, rel
    picked = ops.rdna_hint_flex_dense_gemm_pick_rdna2(m, n, k) >= 0
    if picked:
        # the HIP kernel ran: same values to fp16 rounding, not the same bits
        assert not torch.equal(on, ref)
    else:
        assert torch.equal(on, ref)  # table says rocBLAS: identical call


@gfx1030_only
@pytest.mark.parametrize("m", [130, 1025, 1031, 1045, 1072, 1073, 1100, 2049])
def test_composite_tail_split(m):
    """prefill_gemm 2 (GEMV tail), 3 (tail designs), 4 (hybrid): the r = M % 128
    tail rows run apart from the whole tile rows. Every row matches the fp32
    product of the activation and the dequantized weights; where no split is
    possible (no table design at M - r, r = 0, r above the mode's limit) the
    result is bit-identical to mode 1."""
    from vllm import _custom_ops as ops

    n, k = TP2_SLOTS["out_proj"]
    t = _hi2(n, k, seed=6)
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(m, k, device="cuda", generator=g).half()
    lin = (t["col_scale"], t["had"], t["codes"], t["code_off"], t["blk_params"],
           t["param_off"], t["row_consts"], t["row_nbits"], t["row_block"], False, 48)  # fmt: skip
    wd = torch.empty(n, k, device="cuda", dtype=torch.float16)
    ops.rdna_hint_flex_dequant_rdna2(
        t["codes"], t["code_off"], t["blk_params"], t["param_off"],
        t["row_consts"], t["row_nbits"], t["row_block"], wd,
    )  # fmt: skip
    ref = x.float() @ wd.float().T
    one = ops.rdna_hint_flex_linear_rdna2(x, *lin, 0, False, 1)
    r = m % 128
    can_split = r > 0 and ops.rdna_hint_flex_dense_gemm_pick_rdna2(m - r, n, k) >= 0
    for mode, r_max in ((2, 48), (3, 64), (4, 64)):
        got = ops.rdna_hint_flex_linear_rdna2(x, *lin, 0, False, mode)
        assert torch.isfinite(got).all(), (m, mode)
        assert errs(got, ref)[1] <= REL_FRO_TOL, (m, mode)
        if can_split and r <= r_max:
            assert errs(got[m - r :], ref[m - r :])[1] <= REL_FRO_TOL, (m, mode)
        else:
            assert torch.equal(got, one), (m, mode)
