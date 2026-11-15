# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Dense fp16 prefill GEMM designs of ``tools/rdna/rdna_hint_flex/gemm_sweep``.

``C[M, N] = A[M, K] W[N, K]^T`` with fp16 operands, fp32 accumulation
(``v_dot2_f32_f16``) and an fp16 result: the GEMM that follows the HI2 dequant
kernel in the prefill (docs/rdna2/hi2-prefill-gemm-2026-11-15.md).

CPU tests (run anywhere): the design-name grammar round-trips for every preset,
the validity rules reject what the kernel's static_asserts reject, and
``report.py`` builds its tables from rows.

GPU test (gfx1030 with hipcc): two designs are compiled into a temporary
directory and compared with an fp32 reference and with ``torch.mm`` (rocBLAS) on
the production N x K, odd M, M below one tile, N off the tile grid, tiny N, a
strided activation matrix and a strided output. It skips cleanly without a GPU,
without hipcc and on any other architecture.

Run ``pytest tests/kernels/quantization/test_rdna_prefill_gemm_sweep.py``.
"""

import csv
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

SWEEP_DIR = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "rdna"
    / "rdna_hint_flex"
    / "gemm_sweep"
)
sys.path.insert(0, str(SWEEP_DIR))
sweep: Any = importlib.import_module("sweep")
report: Any = importlib.import_module("report")

PRESETS = [
    "base", "diag", "diag2", "tiles", "core", "occ", "knobs", "rast", "splitk", "lw8", "swz", "rot", "t168", "fin", "best256", "rast2", "rast3", "rast4", "memdiag",
]  # fmt: skip


# ---- CPU: the design space ------------------------------------------------
@pytest.mark.parametrize("name", PRESETS)
def test_preset_names_round_trip(name):
    variants = sweep.preset(name)
    assert variants, name
    for v in variants:
        assert sweep.parse_name(v.name) == v
        assert v.valid(), v.name


def test_literal_names_and_inline_specs_load():
    v = sweep.Variant(bm=256, bn=128, tm=16, tn=8, bk=32, nbuf=2, pad=8)
    (got,) = sweep.load_variants(v.name)
    assert got == v
    (inline,) = sweep.load_variants("bm=256,bn=128,tm=16,tn=8,bk=32,nbuf=2,pad=8")
    assert inline == v
    assert sweep.parse_name("not-a-design") is None


@pytest.mark.parametrize(
    "kw, ok",
    [
        ({}, True),
        ({"bm": 128, "bn": 128, "tm": 16, "tn": 8, "lw": 8}, True),
        (
            {"bm": 128, "bn": 128, "tm": 3, "tn": 8},
            False,
        ),  # micro-tile must divide the tile
        ({"bm": 32, "bn": 32, "tm": 8, "tn": 8}, False),  # 16 threads: under one wave
        (
            {"bm": 256, "bn": 256, "tm": 8, "tn": 8, "nbuf": 2, "pad": 8},
            False,
        ),  # LDS over 64 KiB
        ({"bk": 12}, False),  # BK must be a multiple of 8 halves
        ({"pad": 4}, False),  # rows must stay 16-byte aligned
        ({"swz": 1, "pad": 8}, False),  # the swizzle replaces the padding
        ({"swz": 1, "pad": 0}, True),
        ({"brot": 1}, True),
        ({"brot": 2, "lw": 8}, False),  # rotation under the fragment width
        ({"brot": 1, "glw": 8}, False),  # rotation works on 16-byte chunks
    ],
)
def test_validity_rules(kw, ok):
    assert sweep.Variant(**kw).valid() is ok


def test_defines_carry_every_knob():
    v = sweep.Variant(tm=16, tn=8, lw=8, brot=1, swz=0, ku=2)
    d = dict(
        x[len("-DCFG_") :].split("=", 1) for x in v.defines() if x.startswith("-DCFG_")
    )
    assert (d["TM"], d["TN"], d["LW"], d["BROT"], d["KU"]) == ("16", "8", "8", "1", "2")
    assert d["NAME"].strip('"') == v.name


# ---- CPU: report.py -------------------------------------------------------
def _write_rows(path: Path) -> None:
    fields = ["phase", "kind", "variant", "shape", "N", "K", "M", "us", "us_min", "tflops", "vs_rb",
              "max_abs", "rel_fro", "rel_fp32", "vgpr", "lds", "scratch", "threads", "blocks_per_wgp",
              "waves_per_simd", "gfx_mhz", "power_w", "note"]  # fmt: skip
    rows = []

    def add(kind, variant, shape, m, us):
        rows.append(dict.fromkeys(fields, "") | {"kind": kind, "variant": variant, "shape": shape, "N": 17408,
                        "K": 5120, "M": m, "us": us, "us_min": us, "tflops": 2 * m * 17408 * 5120 / us / 1e6,
                        "vgpr": 128, "lds": 32768, "waves_per_simd": 4})  # fmt: skip

    for m, heur, tuned, dense in (
        (512, 4000.0, 4100.0, 3900.0),
        (2048, 19000.0, 16500.0, 16000.0),
    ):
        add("rocblas_heur", "heuristic", "gate_up", m, heur)
        add("rocblas_tuned", "tuned", "gate_up", m, tuned)
        add("fused_ref", "gemm2", "gate_up", m, heur * 1.1)
        add("dense", "128x128k32t8x8n2p8r0g16b0w0s1m0u1", "gate_up", m, dense)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_report_tables(tmp_path):
    p = tmp_path / "rows.csv"
    _write_rows(p)
    cs = report.cells(report.load([str(p)]))
    name = "128x128k32t8x8n2p8r0g16b0w0s1m0u1"
    ok = lambda v: "m0u" in v  # noqa: E731
    assert "gate_up" in report.reference_table(cs)
    assert name in report.per_cell_best(cs, ok)
    # the oracle compares against the faster rocBLAS row: 3900 vs 4000 (1.026x), 16000 vs 16500 (1.031x)
    text = report.oracle_summary(cs, ok, bar=1.10, floor=0.97)
    assert "cells 2" in text and "> 1.03x: 1" in text and "within 3 %: 1" in text
    matrix = report.matrix_table(cs, [name])
    assert "3900 (" in matrix and "16000 (" in matrix


# ---- GPU: correctness against fp32 and rocBLAS ----------------------------
DESIGNS = [
    sweep.Variant(),  # 128x128 tile, 8x8 per thread, BK 32, 2 LDS stages, 8 halves of padding
    sweep.Variant(
        bm=128, bn=128, tm=16, tn=8, lw=8
    ),  # 16x8 per thread, 8-byte LDS fragments
]


def _gpu_or_skip() -> str:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx1030"):
        pytest.skip(f"gfx1030 only (this is {props.gcnArchName})")
    hipcc = os.environ.get("HIPCC") or shutil.which("hipcc")
    if not hipcc:
        pytest.skip("hipcc not found (set HIPCC)")
    assert hipcc is not None
    return hipcc


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    hipcc = _gpu_or_skip()
    saved = (sweep.OUT, sweep.HIPCC)
    sweep.OUT, sweep.HIPCC = tmp_path_factory.mktemp("prefill_gemm"), hipcc
    try:
        for v in DESIGNS:
            name, rc, msg = sweep.compile_one(v, [])
            assert rc == 0, f"{name}: {msg}"
        yield [sweep.Lib(v) for v in DESIGNS]
    finally:
        sweep.OUT, sweep.HIPCC = saved


def _launch(lib, a, w, c, m, n, k, lda, ldc):
    ws = sweep.lib_ws(torch, lib, m, n)
    rc = lib.lib.variant_launch(
        a.data_ptr(), w.data_ptr(), c.data_ptr(), m, n, k, lda, k, ldc,
        ws.data_ptr() if ws is not None else None, torch.cuda.current_stream().cuda_stream,
    )  # fmt: skip
    assert rc == 0
    torch.accelerator.synchronize()


def _check(c, ref32, rocblas):
    assert not torch.isnan(c).any()
    rel = float((c.float() - ref32).norm() / ref32.norm().clamp_min(1e-30))
    assert rel <= sweep.CHECK_REL_TOL, f"relative error vs fp32: {rel:.2e}"
    # both results are rounded to fp16: at most two half-ulps apart
    assert (
        float((c.float() - rocblas.float()).abs().max())
        <= 2**-9 * float(ref32.abs().max()) + 1e-6
    )


@pytest.mark.parametrize("m", [1, 7, 49, 64, 65, 127, 129, 700, 1500])
@pytest.mark.parametrize(
    "n,k", [(5120, 3072), (17408, 5120), (4104, 192), (130, 64), (8, 64)]
)
def test_matches_fp32_and_rocblas(libs, n, k, m):
    gen = torch.Generator(device="cuda").manual_seed(m * 7919 + n)
    x = torch.randn(m, k, dtype=torch.float16, device="cuda", generator=gen)
    w = torch.randn(n, k, dtype=torch.float16, device="cuda", generator=gen) * 0.05
    ref32 = x.float() @ w.float().t()
    rocblas = x @ w.t()
    for lib in libs:
        c = torch.full((m, n), float("nan"), dtype=torch.float16, device="cuda")
        _launch(lib, x, w, c, m, n, k, k, n)
        _check(c, ref32, rocblas)


def test_strided_and_offset_operands(libs):
    """Lda > K, ldc > N and operands that start 16 bytes into their allocation."""
    m, n, k = 333, 1000, 256
    gen = torch.Generator(device="cuda").manual_seed(3)
    big_a = torch.randn(
        m * (k + 64) + 8, dtype=torch.float16, device="cuda", generator=gen
    )
    a = big_a[8:].as_strided((m, k), (k + 64, 1))
    w = torch.randn(n, k, dtype=torch.float16, device="cuda", generator=gen) * 0.05
    ref32 = a.float() @ w.float().t()
    rocblas = a @ w.t()
    for lib in libs:
        big_c = torch.full(
            (m, n + 40), float("nan"), dtype=torch.float16, device="cuda"
        )
        c = big_c[:, :n]
        _launch(lib, a, w, c, m, n, k, k + 64, n + 40)
        _check(c, ref32, rocblas)
        assert torch.isnan(big_c[:, n:]).all(), "wrote outside the N columns"


def test_k_not_a_multiple_of_bk_is_refused(libs):
    for lib in libs:
        k = lib.v.bk + 8
        a = torch.zeros(4, k, dtype=torch.float16, device="cuda")
        w = torch.zeros(8, k, dtype=torch.float16, device="cuda")
        c = torch.zeros(4, 8, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="rc=-2"):
            lib.launch(a, w, c, None)
