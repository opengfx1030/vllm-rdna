# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the dequant model that the V620 run relies on.

.venv/bin/python -m pytest benchmarks/kernels/w4a16_exact_dequant -q
"""

import numpy as np
import pytest

from benchmarks.kernels.w4a16_exact_dequant import reference as ref


@pytest.mark.parametrize("zero", range(17))
def test_zero_constants_are_exact_fp16_integers(zero):
    """The exact path's bit patterns: 0x6400 | z and 0x6400 | 16 z."""
    assert ref.fp16_from_bits(0x6400 | zero) == 1024 + zero
    assert ref.fp16_from_bits(0x6400 | (zero << 4)) == 1024 + 16 * zero


def test_exact_path_rounds_each_weight_once():
    p = ref.make_problem(1, 256, 512, 128)
    args = (p.q, p.zeros, p.scales, p.group_size)
    once = ref.dequant_exact(*args).astype(np.float16)
    np.testing.assert_array_equal(ref.dequant_exact_fp16(*args), once)


def test_baked_bias_hits_the_low_offsets():
    p = ref.make_problem(1, 512, 2048, 128)
    args = (p.q, p.zeros, p.scales, p.group_size)
    err = np.abs(ref.dequant_baked(*args) - ref.dequant_exact(*args))
    low = np.isin(np.arange(2048) % 8, ref.LOW_OFFSETS)
    assert err[low].mean() > 8 * err[~low].mean()


@pytest.mark.parametrize("scale,zero_survives", [(0.007, False), (0.0078125, True)])
def test_quantized_zero_under_baked_dequant(scale, zero_survives):
    """Predicts test_rdna2_w4a16_quantized_zero_stays_zero on the default
    build: q == z gives exactly 0 only when s * 1032 is an fp16 value."""
    k, n, g = 128, 8, 128
    q = np.full((k, n), 8)
    zeros = np.full((1, n), 8)
    scales = np.full((1, n), scale, dtype=np.float16)
    x = np.ones((1, k), dtype=np.float16)
    out = ref.matmul(x, ref.dequant_baked(q, zeros, scales, g))
    assert bool((out == 0).all()) == zero_survives
    assert (ref.matmul(x, ref.dequant_exact_fp16(q, zeros, scales, g)) == 0).all()


@pytest.mark.parametrize("zeros", ["random", "symmetric"])
@pytest.mark.parametrize("group_size", [32, 128])
def test_output_error_baked_vs_exact(zeros, group_size):
    """On the dense test's distribution the baked bias costs ~2-4 % rel-L2
    (why test_rdna2_w4a16.py allows 5e-2); the exact path fits 5e-3."""
    p = ref.make_problem(16, 512, 2048, group_size, zeros, seed=1)
    args = (p.q, p.zeros, p.scales, p.group_size)
    exact = ref.matmul(p.x, ref.dequant_exact(*args))
    baked = ref.rel_l2(ref.matmul(p.x, ref.dequant_baked(*args)), exact)
    fixed = ref.rel_l2(ref.matmul(p.x, ref.dequant_exact_fp16(*args)), exact)
    assert 1e-2 < baked < 5e-2
    assert fixed < 1e-3
    assert baked > 20 * fixed


def test_shuffle_model_matches_the_kernel_read_order():
    """Pairs read by the 0x000F000F masks are the LOW_OFFSETS K positions."""
    rng = np.random.default_rng(2)
    q = rng.integers(0, 16, size=(64, 16))
    packed = np.zeros((8, 16), dtype=np.uint32)
    for j in range(8):
        packed |= q[j::8].astype(np.uint32) << np.uint32(4 * j)
    shuffled = ref.exllama_shuffle(packed)
    np.testing.assert_array_equal(ref.unpack_shuffled(shuffled), q)
    low_slots = [s for s in range(8) if ref.SLOT_TO_K[s] in ref.LOW_OFFSETS]
    assert low_slots == [0, 2, 4, 6]  # (qa & 0x000F000F) and its >> 8


def test_harness_packs_what_the_kernels_read(monkeypatch):
    """check_ops.device_inputs, decoded the way the kernels decode it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm.model_executor.layers.quantization.utils.quant_utils")
    from benchmarks.kernels.w4a16_exact_dequant import check_ops

    class FakeOps:
        @staticmethod
        def gptq_shuffle(w, g_idx, bits):
            shuffled = ref.exllama_shuffle(w.numpy().view(np.uint32))
            w.copy_(torch.from_numpy(shuffled.view(np.int32)))

    monkeypatch.setattr(check_ops, "_ops", lambda: FakeOps)
    for fmt, zeros, offset in (("uint4", "random", 0), ("uint4b8", "symmetric", 1)):
        p = ref.make_problem(2, 64, 256, 128, zeros, seed=3)
        d = check_ops.device_inputs(p, fmt, device="cpu")
        np.testing.assert_array_equal(
            ref.unpack_shuffled(d["w_q"].numpy().view(np.uint32)), p.q
        )
        stored = ref.unpack_zeros(d["qzeros"].numpy().view(np.uint32))
        np.testing.assert_array_equal(stored + offset, p.zeros)
        assert d["use_v2_format"] == (fmt == "uint4")
