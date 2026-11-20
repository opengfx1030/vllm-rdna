# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
import subprocess
import sys

import pybase64 as base64
import pytest
import torch

from vllm.model_executor.layers.quantization.rdna_hint_flex import RdnaHintFlexConfig
from vllm.model_executor.layers.quantization.utils import rdna_hint_flex_utils as hu
from vllm.platforms import current_platform


def codebook_config():
    # Published depth-16 coefficients (Table 6), with an independently known
    # symmetric table. This tests the stream/ABI, not learned table quality.
    encoded = base64.b64encode(bytes(v % 256 for v in range(-63, 1))).decode()
    return {
        "name": "cptc",
        "revision": 1,
        "depth": 16,
        "tap_q": 6,
        "layout": "mirror_int8",
        "shift": 0,
        "block": "row_block",
        "tables": {
            "2": {
                "a0": 0x7AE6,
                "a1": 0xC8B3,
                "table_scale": 0.0313,
                "tbl_b64": encoded,
            },
            "4": {
                "a0": 0x6A78,
                "a1": 0x85CF,
                "table_scale": 0.0313,
                "tbl_b64": encoded,
            },
        },
    }


def packed_tensor(k=128, rows=6):
    pattern = ((2, 16), (4, 16), (2, 32), (4, 32), (2, 64), (4, 64))
    bits = torch.tensor(
        [pattern[i % len(pattern)][0] for i in range(rows)], dtype=torch.uint8
    )
    blocks = torch.tensor(
        [pattern[i % len(pattern)][1] for i in range(rows)], dtype=torch.uint8
    )
    words, scales, co, po = [], [], [0], [0]
    gen = torch.Generator().manual_seed(719)
    unpacked = []
    for nb, bs in zip(bits.tolist(), blocks.tolist()):
        codes = torch.randint(0, 1 << nb, (k,), generator=gen, dtype=torch.int64)
        unpacked.append(codes)
        packed = torch.zeros(k * nb // 32, dtype=torch.int64)
        for j in range(32 // nb):
            packed |= codes[j :: 32 // nb] << (j * nb)
        words.append(packed.to(torch.int32))
        scales.append(torch.linspace(0, 0.25, k // bs).half())
        co.append(co[-1] + packed.numel())
        po.append(po[-1] + k // bs)
    return {
        "codes": torch.cat(words),
        "code_off": torch.tensor(co, dtype=torch.int32),
        "scale": torch.cat(scales),
        "param_off": torch.tensor(po, dtype=torch.int32),
        "row_nbits": bits,
        "row_block": blocks,
        "had": torch.zeros(k // 16, dtype=torch.uint8),
    }, unpacked


def independent_decode(t, unpacked, metadata):
    """Python big-int circular stream, independent of vector/window helpers."""
    out = torch.empty(len(unpacked), unpacked[0].numel(), dtype=torch.float16)
    for r, codes in enumerate(unpacked):
        nb, bs = int(t["row_nbits"][r]), int(t["row_block"][r])
        entry = metadata["tables"][str(nb)]
        raw = base64.b64decode(entry["tbl_b64"])
        table = torch.tensor(
            [v if v < 128 else v - 256 for v in raw], dtype=torch.float64
        )
        table_scale = float(torch.tensor(entry["table_scale"], dtype=torch.float16))
        table = (table * table_scale).half()
        for start in range(0, codes.numel(), bs):
            stream = sum(int(codes[start + j]) << (j * nb) for j in range(bs))
            wrapped = stream | (stream << (bs * nb))
            scale = float(t["scale"][int(t["param_off"][r]) + start // bs])
            for j in range(bs):
                q = (wrapped >> (j * nb)) & 65535
                index = (((entry["a0"] * q) & 65535) >> 10) + (
                    ((entry["a1"] * q) & 65535) >> 10
                )
                value = float(table[index] if index < 64 else -table[126 - index])
                out[r, start + j] = value * scale
    return out


def test_cyclic_decode_matches_independent_bitstream():
    t, unpacked = packed_tensor()
    metadata = codebook_config()
    book = hu.CptcCodebooks.from_config(metadata)
    hu.validate(t, cptc=book)
    actual = hu.dequantize_stored(t, book)
    assert torch.equal(actual, independent_decode(t, unpacked, metadata))


def test_tp_and_fused_partitions_preserve_cyclic_blocks():
    t, _ = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    full = hu.dequantize_stored(t, book)
    for k0 in (0, 64):
        part = hu.select_cols(t, k0, k0 + 64)
        hu.validate(part, 64, book)
        assert torch.equal(hu.dequantize_stored(part, book), full[:, k0 : k0 + 64])
    merged = hu.concat_rows([hu.select_rows(t, 0, 2), hu.select_rows(t, 2, 6)])
    hu.validate(merged, cptc=book)
    assert torch.equal(hu.dequantize_stored(merged, book), full)


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t["tables"]["2"].update(tbl_b64="not base64"),
        lambda t: t["tables"]["4"].update(table_scale=float("nan")),
        lambda t: t["tables"]["4"].update(a0=65536),
        lambda t: t.update(depth=15),
        lambda t: t.update(layout="unknown"),
        lambda t: t.update(unknown=1),
        lambda t: t["tables"]["2"].update(unknown=1),
        lambda t: t["tables"]["2"].update(table_scale=1e-12),
        lambda t: t["tables"]["2"].update(
            tbl_b64=base64.b64encode(b"\x80" + bytes(63)).decode()
        ),
    ],
)
def test_malformed_codebooks_rejected_before_device_load(change):
    metadata = codebook_config()
    change(metadata)
    with pytest.raises(ValueError):
        hu.CptcCodebooks.from_config(metadata)


def test_shared_codebook_depth_must_support_four_bit_states():
    metadata = codebook_config()
    metadata["depth"] = 7
    for entry in metadata["tables"].values():
        entry.update(a0=1024, a1=512)
    with pytest.raises(ValueError, match="depth"):
        hu.CptcCodebooks.from_config(metadata)


def test_codec_version_and_compute_mode_cannot_silently_misdecode(monkeypatch):
    monkeypatch.delenv("VLLM_RDNA_HINT_FLEX_COMPUTE_MODE", raising=False)
    metadata = codebook_config()
    with pytest.raises(ValueError, match="explicit CPTC"):
        RdnaHintFlexConfig(version="3.3")
    with pytest.raises(ValueError, match="version 3.3"):
        RdnaHintFlexConfig(cptc=metadata)
    with pytest.raises(ValueError, match="fp16"):
        RdnaHintFlexConfig(version="3.3", cptc=metadata, compute_mode="i8")
    t, _ = packed_tensor()
    with pytest.raises(ValueError, match="metadata must agree"):
        hu.validate(t)


def test_invalid_scale_and_offsets_rejected():
    t, _ = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    bad = copy.deepcopy(t)
    bad["scale"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        hu.validate(bad, cptc=book)
    bad = copy.deepcopy(t)
    bad["param_off"][2] += 1
    with pytest.raises(ValueError, match="param_off"):
        hu.validate(bad, cptc=book)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="visible RDNA2 HIP device required",
)
@pytest.mark.parametrize("m", [1, 3, 4, 7, 8, 24, 48, 49, 129])
def test_hip_cptc_compressed_linear_matches_reference(m):
    from vllm import _custom_ops as ops
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("gfx1030 implementation")
    assert hasattr(torch.ops._rocm_C, "rdna_hint_flex_cptc_linear_rdna2")
    t, _ = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    expected_w = hu.dequantize_stored(t, book).cuda()
    d = {key: value.cuda() for key, value in t.items()}
    coeff, tables = book.coefficients.cuda(), book.tables.cuda()
    args = (
        d["codes"],
        d["code_off"],
        d["scale"],
        d["param_off"],
        d["row_nbits"],
        d["row_block"],
    )
    dequant = ops.rdna_hint_flex_cptc_dequant_rdna2(*args, coeff, tables, 128)
    assert torch.equal(dequant, expected_w)
    gen = torch.Generator().manual_seed(19)
    # Row-strided input exercises correct addressing without a GPU-side copy.
    x = torch.randn(m, 129, generator=gen, dtype=torch.float16).cuda()[:, :128]
    y = ops.rdna_hint_flex_cptc_linear_rdna2(
        x,
        torch.empty(0, device="cuda", dtype=torch.float16),
        d["had"],
        *args,
        coeff,
        tables,
        False,
        48,
    )
    torch.testing.assert_close(
        y.float(), x.float() @ expected_w.float().T, atol=0.003, rtol=0.003
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kernel")
def test_hip_cptc_zero_and_subnormal_scale_decode():
    from vllm import _custom_ops as ops
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("gfx1030 implementation")
    t, unpacked = packed_tensor()
    values = torch.tensor([0.0, 2.0**-24, 2.0**-20, 2.0**-10]).half()
    t["scale"] = values.repeat((t["scale"].numel() + 3) // 4)[: t["scale"].numel()]
    metadata = codebook_config()
    expected = independent_decode(t, unpacked, metadata)
    assert (
        (expected.abs() > 0) & (expected.abs() < torch.finfo(torch.float16).tiny)
    ).any()
    book = hu.CptcCodebooks.from_config(metadata)
    d = {key: value.cuda() for key, value in t.items()}
    buffers = (
        d["codes"],
        d["code_off"],
        d["scale"],
        d["param_off"],
        d["row_nbits"],
        d["row_block"],
    )
    actual = ops.rdna_hint_flex_cptc_dequant_rdna2(
        *buffers, book.coefficients.cuda(), book.tables.cuda(), 128
    )
    assert torch.equal(actual.cpu(), expected)


# ---- CPTC GEMV v2 / cptc-fused prefill (mode 5) --------------------------

needs_gfx1030 = pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="visible RDNA2 HIP device required",
)


def _require_native_cptc():
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("gfx1030 implementation")
    assert hasattr(torch.ops._rocm_C, "rdna_hint_flex_cptc_linear_rdna2")


def cptc_cuda(t, book):
    d = {key: value.cuda() for key, value in t.items()}
    return d, book.coefficients.cuda(), book.tables.cuda()


def cptc_linear(
    x,
    d,
    coeff,
    tables,
    *,
    transform=False,
    decode_max_m=48,
    silu_mul=False,
    prefill_gemm=0,
):
    from vllm import _custom_ops as ops

    col_scale = d.get("col_scale")
    if col_scale is None:
        col_scale = torch.empty(0, dtype=torch.float16, device=x.device)
    return ops.rdna_hint_flex_cptc_linear_rdna2(
        x,
        col_scale,
        d["had"],
        d["codes"],
        d["code_off"],
        d["scale"],
        d["param_off"],
        d["row_nbits"],
        d["row_block"],
        coeff,
        tables,
        transform,
        decode_max_m,
        silu_mul,
        prefill_gemm,
    )


def cptc_reference(t, unpacked, x):
    """fp64 matmul against the big-int decode, independent of the kernels."""
    weight = independent_decode(t, unpacked, codebook_config()).double()
    return x.double() @ weight.T


def assert_cptc_close(actual, expected):
    torch.testing.assert_close(
        actual.float().cpu(), expected.float(), atol=3e-3, rtol=3e-3
    )


def silu_and_mul(gate_up):
    out = torch.empty(
        gate_up.shape[0], gate_up.shape[1] // 2, dtype=gate_up.dtype,
        device=gate_up.device,
    )  # fmt: skip
    torch.ops._C.silu_and_mul(out, gate_up)
    return out


def _fake_cptc_ops(monkeypatch, *, kernels):
    import types

    from vllm.model_executor.layers.quantization import rdna_hint_flex as rhf

    ns = types.SimpleNamespace(
        rdna_hint_flex_linear_rdna2=object(),
        rdna_hint_flex_cptc_linear_rdna2=object(),
        rdna_hint_flex_dense_gemm_rdna2=object(),
    )
    monkeypatch.setattr(rhf, "_rdna2_kernels_available", lambda: kernels)
    monkeypatch.setattr(torch.ops, "_rocm_C", ns, raising=False)
    return rhf


@pytest.mark.parametrize("kernels", [True, False])
def test_prefill_cptc_fused_requires_cptc(monkeypatch, kernels):
    """The mode-5 env value is rejected for a v3.2 checkpoint."""
    rhf = _fake_cptc_ops(monkeypatch, kernels=kernels)
    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "cptc-fused")
    with pytest.raises(ValueError, match="requires a CPTC"):
        rhf.RdnaHintFlexLinearMethod(False)


def test_prefill_cptc_fused_requires_native_kernels(monkeypatch):
    """A CPTC checkpoint still needs the native kernels, not a silent fallback."""
    import types

    rhf = _fake_cptc_ops(monkeypatch, kernels=False)
    # Skip the earlier rebuilt-extension guard so this asserts the mode-5 check.
    monkeypatch.setattr(
        rhf, "current_platform", types.SimpleNamespace(is_rocm=lambda: False)
    )
    book = hu.CptcCodebooks.from_config(codebook_config())
    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "cptc-fused")
    with pytest.raises(ValueError, match="native RDNA2 CPTC kernels"):
        rhf.RdnaHintFlexLinearMethod(False, 0, book)


def test_prefill_cptc_fused_accepted_with_native_cptc(monkeypatch):
    rhf = _fake_cptc_ops(monkeypatch, kernels=True)
    book = hu.CptcCodebooks.from_config(codebook_config())
    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "cptc-fused")
    method = rhf.RdnaHintFlexLinearMethod(False, 0, book)
    assert method.prefill_gemm == 5


def test_existing_prefill_mode_ignored_without_native_kernels(monkeypatch):
    """Old mode names stay silently ignored off the native kernels."""
    import types

    rhf = _fake_cptc_ops(monkeypatch, kernels=False)
    monkeypatch.setattr(
        rhf, "current_platform", types.SimpleNamespace(is_rocm=lambda: False)
    )
    monkeypatch.setenv("RDNA_HINT_FLEX_PREFILL_GEMM", "hip")
    method = rhf.RdnaHintFlexLinearMethod(False)
    assert method.prefill_gemm == 0


_CPTC_WORKER = r"""
import os
import sys

import torch

from vllm import _custom_ops as ops

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
coeff = payload["coeff"].cuda()
tables = payload["tables"].cuda()


def run_group(group):
    t = {key: value.cuda() for key, value in group["t"].items()}
    col_scale = t.get("col_scale")
    if col_scale is None:
        col_scale = torch.empty(0, dtype=torch.float16, device="cuda")
    transform = group.get("transform", False)
    decode_max_m = group.get("decode_max_m", 48)
    silu_mul = group.get("silu_mul", False)
    prefill_gemm = group.get("prefill_gemm", 0)

    def call(x):
        return ops.rdna_hint_flex_cptc_linear_rdna2(
            x,
            col_scale,
            t["had"],
            t["codes"],
            t["code_off"],
            t["scale"],
            t["param_off"],
            t["row_nbits"],
            t["row_block"],
            coeff,
            tables,
            transform,
            decode_max_m,
            silu_mul,
            prefill_gemm,
        )

    xs = [x.cuda() for x in group["xs"]]
    one_hot = group.get("one_hot")
    one_hot = one_hot.cuda() if one_hot is not None else None
    warm = xs[0] if xs else one_hot
    assert warm is not None
    call(warm)
    torch.accelerator.synchronize()
    switch_to = payload.get("switch_to")
    if switch_to is not None:
        os.environ["RDNA_HINT_FLEX_GEMV"] = switch_to
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        ys = [call(x) for x in xs]
        one_hot_out = call(one_hot) if one_hot is not None else None
        torch.accelerator.synchronize()
    torch.accelerator.synchronize()
    return {
        "name": group["name"],
        "ys": [y.cpu() for y in ys],
        "one_hot": one_hot_out.cpu() if one_hot_out is not None else None,
        "names": [event.key for event in prof.key_averages()],
        "gemv_env": os.environ.get("RDNA_HINT_FLEX_GEMV"),
    }


with torch.no_grad():
    results = [run_group(group) for group in payload["groups"]]
torch.save({"groups": results}, sys.argv[2])
"""


def _run_cptc_worker(tmp_path, payload, *, variant=None):
    """Run the CPTC op in a fresh process so the GEMV switch is read once."""
    in_path = tmp_path / "cptc_in.pt"
    out_path = tmp_path / "cptc_out.pt"
    torch.save(payload, str(in_path))
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("RDNA_HINT_FLEX_GEMV", "RDNA_HINT_FLEX_PREFILL_GEMM")
    }
    if variant is not None:
        env["RDNA_HINT_FLEX_GEMV"] = variant
    proc = subprocess.run(
        [sys.executable, "-c", _CPTC_WORKER, str(in_path), str(out_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return torch.load(str(out_path), map_location="cpu", weights_only=True)


def _normal_and_subnormal_groups():
    """Mixed M metadata plus the zero/subnormal scale set, for one worker."""
    t, unpacked = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    gen = torch.Generator().manual_seed(31)
    xs = [
        torch.randn(m, 128, generator=gen, dtype=torch.float16)
        for m in (1, 3, 8, 24, 49, 64, 129)
    ]
    sub = copy.deepcopy(t)
    values = torch.tensor([0.0, 2.0**-24, 2.0**-20, 2.0**-10]).half()
    sub["scale"] = values.repeat((t["scale"].numel() + 3) // 4)[: t["scale"].numel()]
    sub_x = torch.randn(
        24, 128, generator=torch.Generator().manual_seed(41), dtype=torch.float16
    )
    common = {
        "transform": False,
        "decode_max_m": 1000,
        "silu_mul": False,
        "prefill_gemm": 0,
    }
    groups = [
        {
            "name": "mixed",
            "t": t,
            "xs": xs,
            "one_hot": torch.eye(128, dtype=torch.float16),
            **common,
        },
        {"name": "zero_subnormal", "t": sub, "xs": [sub_x], **common},
    ]
    return book, unpacked, t, sub, xs, sub_x, groups


@needs_gfx1030
def test_hip_cptc_gemv_variants_in_fresh_processes(tmp_path):
    """Fresh processes select each GEMV variant and decode every M case.

    The selector is cached per process, so each variant needs its own process.
    Unset, v1, and auto must stay bit-identical.
    """
    _require_native_cptc()
    book, unpacked, t, sub, xs, sub_x, groups = _normal_and_subnormal_groups()
    payload = {
        "coeff": book.coefficients,
        "tables": book.tables,
        "groups": groups,
    }
    runs = {
        variant: _run_cptc_worker(tmp_path, payload, variant=variant)
        for variant in (None, "v1", "v2", "auto")
    }
    expected_weights = hu.dequantize_stored(t, book).T
    for tag, result in runs.items():
        by_name = {group["name"]: group for group in result["groups"]}
        names = by_name["mixed"]["names"]
        expect_split = tag == "v2"
        assert any("cptc_gemv_split" in name for name in names) == expect_split
        if not expect_split:
            assert any(
                "cptc_gemv" in name and "cptc_gemv_split" not in name for name in names
            ), names
        for y, x in zip(by_name["mixed"]["ys"], xs):
            assert_cptc_close(y, cptc_reference(t, unpacked, x))
        assert torch.equal(by_name["mixed"]["one_hot"], expected_weights)
        for y in by_name["zero_subnormal"]["ys"]:
            torch.testing.assert_close(
                y.float().cpu(),
                cptc_reference(sub, unpacked, sub_x).float(),
                atol=2e-7,
                rtol=3e-3,
            )
    for tag in ("v1", "auto"):
        baseline = runs[None]["groups"][0]
        candidate = runs[tag]["groups"][0]
        for y, y_ref in zip(candidate["ys"], baseline["ys"]):
            assert torch.equal(y, y_ref)
        assert torch.equal(candidate["one_hot"], baseline["one_hot"])


@needs_gfx1030
def test_hip_cptc_split_thresholds_use_legacy_without_a_partition(tmp_path):
    _require_native_cptc()
    book = hu.CptcCodebooks.from_config(codebook_config())
    x = torch.randn(
        3, 64, generator=torch.Generator().manual_seed(53), dtype=torch.float16
    )
    groups = []
    for rows in (4096, 8192):
        t, _ = packed_tensor(k=64, rows=rows)
        groups.append(
            {
                "name": str(rows),
                "t": t,
                "xs": [x],
                "one_hot": torch.eye(64, dtype=torch.float16),
                "transform": False,
                "decode_max_m": 1000,
                "silu_mul": False,
                "prefill_gemm": 0,
            }
        )
    payload = {"coeff": book.coefficients, "tables": book.tables, "groups": groups}
    baseline = _run_cptc_worker(tmp_path, payload, variant="v1")["groups"]
    candidate = _run_cptc_worker(tmp_path, payload, variant="v2")["groups"]
    for old, new in zip(baseline, candidate):
        names = new["names"]
        split = new["name"] == "4096"
        assert any("cptc_gemv_split" in name for name in names) == split, names
        if not split:
            assert any(
                "cptc_gemv" in name and "cptc_gemv_split" not in name for name in names
            ), names
            assert torch.equal(new["ys"][0], old["ys"][0])
        else:
            assert_cptc_close(new["ys"][0], old["ys"][0])
        assert torch.equal(new["one_hot"], old["one_hot"])


@needs_gfx1030
def test_hip_cptc_gemv_switch_is_read_once(tmp_path):
    """A later env change cannot switch the kernel after the first call."""
    _require_native_cptc()
    book, unpacked, t, _, _, _, _ = _normal_and_subnormal_groups()
    x = torch.randn(
        8, 128, generator=torch.Generator().manual_seed(51), dtype=torch.float16
    )
    group = {
        "name": "cached",
        "t": t,
        "xs": [x],
        "transform": False,
        "decode_max_m": 1000,
        "silu_mul": False,
        "prefill_gemm": 0,
    }
    result = _run_cptc_worker(
        tmp_path,
        {
            "coeff": book.coefficients,
            "tables": book.tables,
            "groups": [group],
            "switch_to": "v2",
        },
    )
    names = result["groups"][0]["names"]
    assert result["groups"][0]["gemv_env"] == "v2"
    assert not any("cptc_gemv_split" in name for name in names), names
    assert any(
        "cptc_gemv" in name and "cptc_gemv_split" not in name for name in names
    ), names
    assert_cptc_close(result["groups"][0]["ys"][0], cptc_reference(t, unpacked, x))


@needs_gfx1030
def test_hip_cptc_prefill_fused_kernel_selection(tmp_path):
    """Modes 0/4 use dequant + rocBLAS here; mode 5 must launch fused decode."""
    _require_native_cptc()
    book, _, t, _, _, _, _ = _normal_and_subnormal_groups()
    groups = [
        {
            "name": str(mode),
            "t": t,
            "xs": [],
            "one_hot": torch.eye(128, dtype=torch.float16),
            "transform": False,
            "decode_max_m": 0,
            "silu_mul": False,
            "prefill_gemm": mode,
        }
        for mode in (0, 4, 5)
    ]
    result = _run_cptc_worker(
        tmp_path,
        {"coeff": book.coefficients, "tables": book.tables, "groups": groups},
    )
    expected = hu.dequantize_stored(t, book).T
    for group in result["groups"]:
        fused = group["name"] == "5"
        names = group["names"]
        assert any("cptc_prefill_fused" in name for name in names) == fused, names
        assert any("cptc_dequant" in name for name in names) != fused, names
        assert torch.equal(group["one_hot"], expected)


@needs_gfx1030
@pytest.mark.parametrize("m", [1, 3, 8, 24, 49, 64, 129])
@pytest.mark.parametrize("mode", [4, 5])
def test_hip_cptc_prefill_mode_matches_reference(m, mode):
    """Mode 4 (dequant + rocBLAS at this shape) and in-tile CPTC decode (5)."""
    _require_native_cptc()
    t, unpacked = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    x = torch.randn(
        m, 128, generator=torch.Generator().manual_seed(37), dtype=torch.float16
    )
    d, coeff, tables = cptc_cuda(t, book)
    y = cptc_linear(x.cuda(), d, coeff, tables, decode_max_m=0, prefill_gemm=mode)
    assert_cptc_close(y, cptc_reference(t, unpacked, x))


@needs_gfx1030
def test_hip_cptc_prefill_fused_row_strided_input():
    """Mode 5 reads a narrow [M, K] view without a GPU-side copy."""
    _require_native_cptc()
    t, unpacked = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    x = torch.randn(49, 129, dtype=torch.float16, device="cuda")[:, :128]
    assert x.stride(0) == 129
    d, coeff, tables = cptc_cuda(t, book)
    y = cptc_linear(x, d, coeff, tables, decode_max_m=0, prefill_gemm=5)
    assert_cptc_close(y, cptc_reference(t, unpacked, x.cpu()))


@needs_gfx1030
def test_hip_cptc_prefill_fused_large_shape_and_n_tail():
    """M = 1024 and R = 69 (one full 64-row tile and a 5-row tail)."""
    _require_native_cptc()
    t, unpacked = packed_tensor(k=256, rows=69)
    book = hu.CptcCodebooks.from_config(codebook_config())
    x = torch.randn(1024, 256, dtype=torch.float16)
    d, coeff, tables = cptc_cuda(t, book)
    xd = x.cuda()
    expected = cptc_reference(t, unpacked, x)
    for mode in (4, 5):
        y = cptc_linear(xd, d, coeff, tables, decode_max_m=0, prefill_gemm=mode)
        assert_cptc_close(y, expected)


@needs_gfx1030
@pytest.mark.parametrize("silu_mul", [False, True])
def test_hip_cptc_prefill_fused_transform(silu_mul):
    """Mode 5 consumes the same transformed (and gate|up folded) input."""
    _require_native_cptc()
    k = 128
    t, unpacked = packed_tensor()
    t["had"] = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.uint8)
    t["col_scale"] = torch.linspace(0.5, 1.5, k).half()
    book = hu.CptcCodebooks.from_config(codebook_config())
    hu.validate(t, cptc=book)
    d, coeff, tables = cptc_cuda(t, book)
    x = torch.randn(49, 2 * k if silu_mul else k, dtype=torch.float16) * 2
    xd = x.cuda()
    if silu_mul:
        fused = cptc_linear(
            xd, d, coeff, tables, transform=True, decode_max_m=0,
            silu_mul=True, prefill_gemm=5,
        )  # fmt: skip
        act = silu_and_mul(xd)
        plain = cptc_linear(
            act, d, coeff, tables, transform=True, decode_max_m=0,
            prefill_gemm=5,
        )  # fmt: skip
        assert torch.equal(fused, plain)
        src = act.cpu()
    else:
        plain = cptc_linear(
            xd, d, coeff, tables, transform=True, decode_max_m=0,
            prefill_gemm=5,
        )  # fmt: skip
        src = x
    xt = hu.act_transform(src, t)
    assert_cptc_close(plain, cptc_reference(t, unpacked, xt))


@needs_gfx1030
def test_hip_cptc_zero_subnormal_scales_through_prefill_paths():
    """Zero and subnormal block scales decode identically in modes 4 and 5."""
    _require_native_cptc()
    t, unpacked = packed_tensor()
    values = torch.tensor([0.0, 2.0**-24, 2.0**-20, 2.0**-10]).half()
    t["scale"] = values.repeat((t["scale"].numel() + 3) // 4)[: t["scale"].numel()]
    book = hu.CptcCodebooks.from_config(codebook_config())
    d, coeff, tables = cptc_cuda(t, book)
    x = torch.randn(
        24, 128, generator=torch.Generator().manual_seed(41), dtype=torch.float16
    )
    expected = cptc_reference(t, unpacked, x)

    # Small absolute tolerance: a flush-to-zero subnormal decode is ~1e-6.
    def check(y):
        torch.testing.assert_close(
            y.float().cpu(), expected.float(), atol=2e-7, rtol=3e-3
        )

    for mode in (4, 5):
        y = cptc_linear(x.cuda(), d, coeff, tables, decode_max_m=0, prefill_gemm=mode)
        check(y)


@needs_gfx1030
def test_hip_cptc_fused_prefill_changed_input_graph_replay():
    """The mode-5 kernel has no retained state: capture, change x, replay."""
    _require_native_cptc()
    t, unpacked = packed_tensor()
    book = hu.CptcCodebooks.from_config(codebook_config())
    d, coeff, tables = cptc_cuda(t, book)
    gen = torch.Generator().manual_seed(43)
    inputs = [
        torch.randn(49, 128, generator=gen, dtype=torch.float16) for _ in range(3)
    ]
    xd = inputs[0].cuda()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            cptc_linear(xd, d, coeff, tables, decode_max_m=0, prefill_gemm=5)
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    static = inputs[0].cuda()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cptc_linear(static, d, coeff, tables, decode_max_m=0, prefill_gemm=5)
    for x in inputs:
        static.copy_(x.cuda())
        graph.replay()
        actual = captured.clone()
        assert_cptc_close(actual, cptc_reference(t, unpacked, x))
