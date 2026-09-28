# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the W4A8 sdot4 explore contract (no torch, no GPU).

These pin the layout and arithmetic the HIP draft relies on, so a kernel
failure on the V620 can be attributed to the kernel rather than the math.

Run from the repo root::

    .venv/bin/python -m pytest benchmarks/kernels/w4a8_sdot4_explore -q
"""

import numpy as np
import pytest

from benchmarks.kernels.w4a8_sdot4_explore import reference as ref

# ---------------------------------------------------------------------------
# Layout: the pack RDNA2W4A16LinearKernel leaves in memory
# ---------------------------------------------------------------------------


def _shuffle_4bit_8_literal(qa: int) -> int:
    """Line-by-line port of shuffle_4bit_8 (gptq/qdq_4.cuh)."""
    qb = 0
    for i in range(4):
        qa0 = qa & 0x0F
        qa1 = (qa & 0xF0) >> 4
        qa >>= 8
        qb |= qa1 << (i * 4 + 16)
        qb |= qa0 << (i * 4)
    return qb


def test_shuffle_matches_exllama_source():
    rng = np.random.default_rng(0)
    words = rng.integers(0, 2**32, size=256, dtype=np.uint64).astype(np.uint32)
    got = ref.exllama_shuffle(words.reshape(-1, 1)).ravel()
    want = np.array([_shuffle_4bit_8_literal(int(w)) for w in words], np.uint32)
    np.testing.assert_array_equal(got, want)


def test_w4a16_dequant_reads_shuffled_pack_in_k_order():
    """The production fp16 dequant must see k0..k7 on the same pack."""
    assert ref.w4a16_dequant_k_order() == tuple(range(8))


def test_shuffle_roundtrip():
    rng = np.random.default_rng(1)
    q = rng.integers(0, 16, size=(64, 24), dtype=np.uint8)
    np.testing.assert_array_equal(
        ref.unpack_shuffled(ref.exllama_shuffle(ref.pack_k_major(q))), q
    )


def test_swar_unpack_yields_a_perm_order():
    rng = np.random.default_rng(2)
    q = rng.integers(0, 16, size=(32, 8), dtype=np.uint8)
    lo, hi = ref.swar_unpack(ref.exllama_shuffle(ref.pack_k_major(q)))
    assert ref.A_PERM == (0, 4, 1, 5, 2, 6, 3, 7)
    lo_b = lo.view(np.uint8).reshape(4, 8, 4)
    hi_b = hi.view(np.uint8).reshape(4, 8, 4)
    for c in range(4):
        for j in range(4):
            np.testing.assert_array_equal(lo_b[c, :, j], q[8 * c + ref.A_PERM[j]])
            np.testing.assert_array_equal(hi_b[c, :, j], q[8 * c + ref.A_PERM[4 + j]])


def test_sign_extension_is_wrong_for_zero_point_packs():
    """The old contract sign-extended nibbles; that never equals ``q - z``."""
    k_idx, n_idx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    q = ((k_idx + n_idx) % 16).astype(np.uint8)  # every value in every slot
    w = ref.exllama_shuffle(ref.pack_k_major(q))
    by_slot = np.stack(
        [q.reshape(2, 8, 16)[:, k_off, :] for k_off in ref.SHUFFLE_SLOTS], -1
    ).astype(np.int64)
    sext = ref.sign_extend_nibbles(w)
    for z in range(17):  # AWQ zeros are 0..15, GPTQv1 effective zeros 1..16
        assert not np.array_equal(sext, by_slot - z), z
    # It is q - 8 only after flipping every nibble's top bit (z == 8 packs).
    flipped = ref.sign_extend_nibbles(w ^ np.uint32(0x88888888))
    np.testing.assert_array_equal(flipped, by_slot - 8)


# ---------------------------------------------------------------------------
# sdot4 and act quant
# ---------------------------------------------------------------------------


def test_sdot4_matches_bruteforce():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2**32, size=512, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, size=512, dtype=np.uint64).astype(np.uint32)
    a[0], b[0] = 0x80808080, 0x80808080  # -128 * -128 lanes
    acc = rng.integers(-(2**20), 2**20, size=512)
    got = ref.sdot4(a, b, acc)
    ab = a.view(np.int8).reshape(-1, 4).astype(np.int64)
    bb = b.view(np.int8).reshape(-1, 4).astype(np.int64)
    np.testing.assert_array_equal(got, acc + (ab * bb).sum(axis=1))
    assert got[0] == acc[0] + 4 * 128 * 128


def test_act_quant_matches_vllm_convention():
    x = np.zeros((3, 16), dtype=np.float16)
    x[0, :5] = [127.0, 2.5, 3.5, -2.5, -127.0]  # inv == 1: ties go to even
    x[2, :3] = [-3.0, 1.0, 0.25]
    a, scale = ref.quantize_per_token(x)
    np.testing.assert_array_equal(a[0, :5], [127, 2, 4, -2, -127])
    np.testing.assert_array_equal(a[1], 0)  # all-zero row
    assert scale[1] == 0
    assert scale[2] == np.float32(3.0) / np.float32(127.0)
    assert a.dtype == np.int8 and scale.dtype == np.float32
    assert np.abs(a.astype(np.int32)).max() <= 127


@pytest.mark.parametrize("m_tile", [8, 16, 32])
def test_tile_layout_matches_kernel_indexing(m_tile):
    """a[(t*K/8 + c)*MT*8 + m*8 + j] == a_perm[t*MT + m, 8c + j]; pads zero."""
    m, k, g = 37, 96, 32
    act = ref.quantize_act(ref.make_problem(m, 8, k, g, "uint4", seed=10).x, g)
    tiled = ref.tile_a(act.a_perm, m_tile)
    tiles = -(-m // m_tile)
    assert tiled.size == tiles * m_tile * k
    for row in (0, m_tile - 1, m - 1):
        t, r = divmod(row, m_tile)
        for c in (0, k // 8 - 1):
            off = (t * (k // 8) + c) * m_tile * 8 + r * 8
            np.testing.assert_array_equal(
                tiled[off : off + 8], act.a_perm[row, 8 * c : 8 * c + 8]
            )
    np.testing.assert_array_equal(ref.untile_a(tiled, m, k, m_tile), act.a_perm)
    assert not tiled.reshape(tiles, k // 8, m_tile, 8)[-1, :, m % m_tile :].any()
    sums = ref.tile_groups(act.asum, m_tile).reshape(tiles, k // g, m_tile)
    np.testing.assert_array_equal(sums[0, :, 1], act.asum[1])


@pytest.mark.parametrize("group_size", ref.SUPPORTED_GROUP_SIZES)
def test_group_sums_survive_a_permutation(group_size):
    x = ref.make_problem(8, 8, 256, group_size, "uint4", seed=4).x
    act = ref.quantize_act(x, group_size)
    np.testing.assert_array_equal(act.asum, ref.group_sums(act.a_perm, group_size))
    np.testing.assert_array_equal(
        act.asum, act.a_i8.astype(np.int64).reshape(8, -1, group_size).sum(2)
    )


# ---------------------------------------------------------------------------
# Kernel replay vs oracle
# ---------------------------------------------------------------------------

CASES = [
    (wt, gs, ks)
    for wt in ref.ELIGIBLE_WEIGHT_TYPES
    for gs in ref.SUPPORTED_GROUP_SIZES
    for ks in ref.K_STEPS
]


@pytest.mark.parametrize("weight_type,group_size,k_step", CASES)
def test_group_partials_are_exact(weight_type, group_size, k_step):
    """SWAR unpack + A_PERM + z*asum fold == sum(a * (q - z)) exactly."""
    p = ref.make_problem(17, 24, 512, group_size, weight_type, seed=5)
    act = ref.quantize_act(p.x, group_size)
    _, partials = ref.emulate_kernel(
        act, p.w_shuf, p.zeros_eff_gn, p.scales_gn, group_size, k_step
    )
    a = act.a_i8.astype(np.int64).reshape(17, -1, group_size)
    w = p.q_kn.astype(np.int64).reshape(-1, group_size, 24) - p.zeros_eff_gn[:, None]
    np.testing.assert_array_equal(partials, np.einsum("mgk,gkn->mgn", a, w))


@pytest.mark.parametrize("weight_type,group_size,k_step", CASES)
def test_f32_path_within_bound(weight_type, group_size, k_step):
    p = ref.make_problem(16, 32, 384, group_size, weight_type, seed=6)
    act = ref.quantize_act(p.x, group_size)
    c, _ = ref.emulate_kernel(
        act, p.w_shuf, p.zeros_eff_gn, p.scales_gn, group_size, k_step
    )
    orc = ref.oracle_w4a8(act, p.q_kn, p.zeros_eff_gn, p.scales_gn, group_size)
    bound = ref.f32_flush_bound(orc.mag, 384 // group_size)
    assert np.all(np.abs(c - orc.c) <= bound)


@pytest.mark.parametrize("split_k", [1, 2, 4])
def test_f16_split_k_within_bound(split_k):
    p = ref.make_problem(16, 32, 512, 32, "uint4", seed=7)
    act = ref.quantize_act(p.x, 32)
    c, _ = ref.emulate_kernel(
        act, p.w_shuf, p.zeros_eff_gn, p.scales_gn, 32, split_k=split_k, out_f16=True
    )
    orc = ref.oracle_w4a8(act, p.q_kn, p.zeros_eff_gn, p.scales_gn, 32)
    bound = ref.f16_output_bound(orc.mag, 16, split_k)
    assert c.dtype == np.float16
    assert np.all(np.abs(c.astype(np.float64) - orc.c) <= bound)


def test_short_loop_body_is_caught():
    """ConfigH lesson: advertising K_STEP=32 while consuming 16 must fail."""
    p = ref.make_problem(16, 16, 256, 128, "uint4b8", seed=8)
    act = ref.quantize_act(p.x, 128)
    c, _ = ref.emulate_kernel(
        act, p.w_shuf, p.zeros_eff_gn, p.scales_gn, 128, 32, dwords_per_step=2
    )
    orc = ref.oracle_w4a8(act, p.q_kn, p.zeros_eff_gn, p.scales_gn, 128)
    assert np.any(np.abs(c - orc.c) > 100 * ref.f32_flush_bound(orc.mag, 2))


@pytest.mark.parametrize("group_size", [32, 128, 4096])
def test_worst_case_partials_stay_exact(group_size):
    """a=-128, q=15, z=0 or 16: no i32 wrap, f32 cast exact below 2**24."""
    m, n, k = 2, 8, group_size
    q = np.full((k, n), 15, dtype=np.uint8)
    for z in (0, 16):
        zeros = np.full((1, n), z, dtype=np.int64)
        a = np.full((m, k), -128, dtype=np.int8)
        act = ref.ActQuant(
            a_i8=a,
            a_perm=ref.permute_a(a),
            scale=np.ones(m, np.float32),
            asum=ref.group_sums(a, k),
        )
        w = ref.exllama_shuffle(ref.pack_k_major(q))
        _, partials = ref.emulate_kernel(
            act, w, zeros, np.ones((1, n), np.float16), k, 32
        )
        want = -128 * (15 - z) * k
        assert np.all(partials == want)
        assert abs(want) < 2**24


# ---------------------------------------------------------------------------
# Rules the docs quote
# ---------------------------------------------------------------------------


def test_budget_numbers_quoted_in_design():
    table = {
        (16, 4, 32): (512, 48, 192, 1.55),
        (16, 4, 128): (2048, 192, 192, 1.92),
        (8, 4, 32): (256, 48, 96, 1.64),
        (32, 2, 32): (512, 24, 192, 1.51),
    }
    for (mt, npt, gs), (sdot4, unpack, flush, ratio) in table.items():
        b = ref.loop_budget(mt, npt, gs)
        assert (b.sdot4, b.unpack, b.flush) == (sdot4, unpack, flush)
        assert round(b.ratio, 2) == ratio
    per_group_a = {(16, 4): (1.43, 1.70, 1.87), (8, 4): (1.52, 1.78, 1.95)}
    for (mt, npt), ratios in per_group_a.items():
        for gs, ratio in zip((32, 64, 128), ratios):
            b = ref.loop_budget(mt, npt, gs, flush_per_output=4)
            assert round(b.ratio, 2) == ratio


def test_split_k_is_group_aligned():
    assert ref.valid_split_ks(2560, 128) == [1, 2, 4, 5, 10]
    assert ref.valid_split_ks(8704, 32) == [1, 2, 4, 8, 16]
    for m, n, k in [(624, 6144, 2560), (2048, 2560, 8704), (2048, 8704, 2560)]:
        for gs in (32, 128):
            split = ref.pick_split_k(m, n, k, gs)
            assert (k // split) % gs == 0
            assert ref.lds_bytes(16, k // split, gs) <= 64 * 1024


def test_eligibility_rules():
    ok = dict(
        weight_type="uint4",
        group_size=32,
        k=2560,
        n=6144,
        has_g_idx=False,
        layer="model.layers.3.mlp.gate_up_proj",
    )
    assert ref.eligibility(**ok) is None
    for key, bad in [
        ("weight_type", "uint8b128"),
        ("has_g_idx", True),
        ("group_size", -1),
        ("group_size", 16),
        ("k", 2560 + 16),
        ("n", 6148),
        ("layer", "model.layers.3.self_attn.qkv_proj"),
    ]:
        assert ref.eligibility(**{**ok, key: bad}) is not None, key


def test_a8_error_grows_with_outliers():
    """Why G1 exists: per-token int8 error is small until outliers appear."""
    errs = []
    for outliers in (0, 8):
        p = ref.make_problem(
            32, 64, 1024, 32, "uint4", seed=9, outlier_channels=outliers
        )
        act = ref.quantize_act(p.x, 32)
        w4a8 = ref.oracle_w4a8(act, p.q_kn, p.zeros_eff_gn, p.scales_gn, 32).c
        w4a16 = ref.w4a16_reference(p.x, p.q_kn, p.zeros_eff_gn, p.scales_gn, 32)
        errs.append(np.linalg.norm(w4a8 - w4a16) / np.linalg.norm(w4a16))
    assert errs[0] < 0.02
    assert errs[1] > 2 * errs[0]


# ---------------------------------------------------------------------------
# Prior art: JartX's RDNA3 fork (see docs/explore/w4a8-sdot4/PRIOR-ART-RDNA3.md)
# ---------------------------------------------------------------------------


def test_rdna2_w4a16_bakes_a_rounded_bias():
    """gfx1030's W4A16 dequant stores s·(−1024 − z) in fp16 (JartX dafcde3).

    The rounding lands on K offsets {0,1,4,5} and costs a few % rel-L2 even on
    zero-mean activations: several times what per-token A8 costs with exact
    integer weights. G1 and G2 have to be read with that in mind.
    """
    p = ref.make_problem(64, 512, 2560, 32, "uint4", seed=3)
    args = (p.q_kn, p.zeros_eff_gn, p.scales_gn, 32)
    exact_w = ref.dequant_weights_f64(*args)
    baked_w = ref.w4a16_rdna2_weights(*args).astype(np.float64)
    err = np.abs(baked_w - exact_w)
    low = np.isin(np.arange(2560) % 8, (0, 1, 4, 5))
    assert err[low].mean() > 8 * err[~low].mean()

    x = p.x.astype(np.float64)
    exact = x @ exact_w

    def rel(c):
        return np.linalg.norm(c - exact) / np.linalg.norm(exact)

    baked = rel(x @ baked_w)
    a8 = rel(ref.oracle_w4a8(ref.quantize_act(p.x, 32), *args).c)
    assert 0.01 < baked < 0.05
    assert a8 < baked / 2


@pytest.mark.parametrize("group_size", [32, 128])
def test_per_group_a_scales_stay_exact_and_bounded(group_size):
    """Per-(token, group) A scales: same integer partials, one extra rounding."""
    m, n, k = 16, 32, 512
    p = ref.make_problem(m, n, k, group_size, "uint4b8", seed=11)
    act = ref.quantize_act(p.x, group_size, per_group_scale=True)
    assert act.scale.shape == (m, k // group_size)
    args = (p.zeros_eff_gn, p.scales_gn, group_size)
    c, partials = ref.emulate_kernel(act, p.w_shuf, *args)
    a = act.a_i8.astype(np.int64).reshape(m, -1, group_size)
    w = p.q_kn.astype(np.int64).reshape(-1, group_size, n) - p.zeros_eff_gn[:, None]
    np.testing.assert_array_equal(partials, np.einsum("mgk,gkn->mgn", a, w))

    orc = ref.oracle_w4a8(act, p.q_kn, *args)
    groups = k // group_size
    assert np.all(np.abs(c - orc.c) <= ref.f32_flush_bound(orc.mag, groups, True))
    c16, _ = ref.emulate_kernel(act, p.w_shuf, *args, split_k=2, out_f16=True)
    bound16 = ref.f16_output_bound(orc.mag, groups, 2, True)
    assert np.all(np.abs(c16.astype(np.float64) - orc.c) <= bound16)


def test_per_group_a_scales_contain_outliers():
    """Why G1 also runs --act-group-size: outliers only coarsen their group."""
    p = ref.make_problem(32, 64, 1024, 32, "uint4", seed=9, outlier_channels=8)
    args = (p.q_kn, p.zeros_eff_gn, p.scales_gn, 32)
    exact = ref.w4a16_reference(p.x, *args)
    errs = []
    for per_group in (False, True):
        act = ref.quantize_act(p.x, 32, per_group_scale=per_group)
        c = ref.oracle_w4a8(act, *args).c
        errs.append(np.linalg.norm(c - exact) / np.linalg.norm(exact))
    assert errs[1] < errs[0] / 2


# ---------------------------------------------------------------------------
# ISA audit of the HIP draft (needs a clang with the AMDGPU backend)
# ---------------------------------------------------------------------------


def _amdgpu_clang() -> str | None:
    import shutil
    import subprocess

    clang = shutil.which("clang")
    if clang is None:
        return None
    targets = subprocess.run(
        [clang, "-print-targets"], capture_output=True, text=True
    ).stdout
    return clang if "amdgcn" in targets else None


@pytest.mark.skipif(_amdgpu_clang() is None, reason="needs clang with amdgcn")
def test_isa_matches_budget():
    """Every sweep config compiles for gfx1030 within the design budget."""
    from benchmarks.kernels.w4a8_sdot4_explore import isa_check

    rows, notes = isa_check.audit(isa_check.compile_asm(_amdgpu_clang()))
    gemm = [r for r in rows if r.budget is not None]
    assert len(gemm) == 3 * len(isa_check.sweep_configs())
    assert [f"{r.config} G={r.group}: {r.failures}" for r in rows if r.failures] == []
    assert sum("probe" in n for n in notes) == 3


@pytest.mark.skipif(_amdgpu_clang() is None, reason="needs clang with amdgcn")
def test_capi_glue_compiles():
    """The ctypes glue parses on both passes and emits every kernel."""
    from benchmarks.kernels.w4a8_sdot4_explore import isa_check

    kinds = isa_check.check_capi(_amdgpu_clang())
    n_cfg = len(isa_check.sweep_configs())
    assert sorted(kinds) == sorted(
        ["gemm"] * 3 * n_cfg + ["act_quant"] * 6 + ["probe"] * 3
    )
