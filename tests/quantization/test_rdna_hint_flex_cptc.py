# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

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


def packed_tensor(k=128):
    bits = torch.tensor([2, 4, 2, 4, 2, 4], dtype=torch.uint8)
    blocks = torch.tensor([16, 16, 32, 32, 64, 64], dtype=torch.uint8)
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
