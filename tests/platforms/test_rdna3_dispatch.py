# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1100 (RDNA3) dispatch of the Flash-Next HIP kernel families (CPU only).

The gfx1030 HIP kernels run on gfx1030 by default. On a gfx1100 (Radeon PRO W7800)
device the same call sites keep the Triton / torch / upstream route unless the
family is named in ``VLLM_RDNA3_HIP_OPS``. These tests fake the architecture (no
GPU is touched) and check, for gfx1030 / gfx1100 / gfx942 / not-ROCm:

* the family list parser and ``on_rdna3_hip`` / ``on_rdna2_hip`` truth tables,
* every call site that was widened: its gfx1030 answer is the old one, its
  gfx1100 answer follows the opt-in,
* the mixed-arch handling of the amdsmi query.
"""

import contextlib
import types

import pytest
import torch

from vllm.platforms import current_platform

rdna = pytest.importorskip("vllm.platforms.rdna")
rocm = pytest.importorskip("vllm.platforms.rocm")

_FLAG_NAMES = (
    "_ON_GFX1X",
    "_ON_GFX10X",
    "_ON_GFX11",
    "_ON_GFX1100",
    "_ON_GFX1151",
    "_ON_GFX115X",
    "_ON_GFX12X",
    "_ON_MI3XX",
    "_ON_GFX9",
    "_ON_GFX90A",
    "_ON_GFX942",
    "_ON_GFX950",
    "_ON_GFX1250",
    "_ON_CDNA",
    "_ON_RDNA",
    "_ON_RDNA4",
)
_ARCH_FLAGS = {
    "gfx1030": {"_ON_GFX10X": True},
    "gfx1100": {
        "_ON_GFX1X": True,
        "_ON_GFX11": True,
        "_ON_GFX1100": True,
        "_ON_RDNA": True,
    },
    "gfx942": {
        "_ON_MI3XX": True,
        "_ON_GFX9": True,
        "_ON_GFX942": True,
        "_ON_CDNA": True,
    },
}
_GCN = {
    "gfx1030": "gfx1030",
    "gfx1100": "gfx1100",
    "gfx942": "gfx942:sramecc+:xnack-",
}


@pytest.fixture
def set_arch(monkeypatch):
    """``set_arch(arch, families="", all_hip="1")``: fake the process arch.

    ``arch`` is gfx1030, gfx1100, gfx942 or None (not ROCm). Resets every gate
    cache in vllm.platforms.rdna so each call re-evaluates.
    """

    def _set(arch, families="", all_hip="1"):
        for name in (
            "_ON_RDNA_FAMILY",
            "_ON_RDNA2",
            "_ON_RDNA3",
            "_RDNA3_HIP_FAMILIES",
        ):
            monkeypatch.setattr(rdna, name, None)
        monkeypatch.setattr(current_platform, "is_rocm", lambda: arch is not None)
        for name in _FLAG_NAMES:
            monkeypatch.setattr(rocm, name, False)
        for name, value in _ARCH_FLAGS.get(arch, {}).items():
            monkeypatch.setattr(rocm, name, value)
        monkeypatch.setattr(rocm, "_GCN_ARCH", _GCN.get(arch, ""))
        monkeypatch.setenv("VLLM_RDNA3_HIP_OPS", families)
        monkeypatch.setenv("VLLM_RDNA_ALL_HIP", all_hip)
        _fake_rocm_ops(monkeypatch)

    return _set


def _fake_rocm_ops(monkeypatch, *names, probes=True):
    """Make ``torch.ops._rocm_C`` expose ``names`` (callables) and, unless
    ``probes`` is False, the per-family probe ops of ``vllm.platforms.rdna`` (a
    build that carries the fork's gfx1030 HIP sources)."""
    ops = set(names)
    if probes:
        ops |= set(rdna._FAMILY_PROBE_OP.values())
    ns = types.SimpleNamespace(**{n: (lambda *a, **k: None) for n in ops})
    monkeypatch.setattr(torch.ops, "_rocm_C", ns, raising=False)


# --------------------------------------------------------------------------
# vllm.platforms.rdna
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", set()),
        ("none", set()),
        (" , ", set()),
        ("hc", {"hc"}),
        (" HC , Qsa ", {"hc", "qsa"}),
        ("hc,none", {"hc"}),
        ("all", set(rdna.RDNA3_HIP_FAMILIES)),
        ("gdn,all", set(rdna.RDNA3_HIP_FAMILIES)),
    ],
)
def test_family_list_parser(set_arch, value, expected):
    set_arch("gfx1100", value)
    assert set(rdna._rdna3_hip_families()) == expected


def test_family_list_rejects_unknown_name(set_arch):
    set_arch("gfx1100", "hc,bogus")
    with pytest.raises(ValueError, match="bogus"):
        rdna._rdna3_hip_families()


def test_named_family_missing_from_the_build_is_an_error(set_arch, monkeypatch):
    set_arch("gfx1100", "hc,gdn,v1")
    _fake_rocm_ops(monkeypatch, probes=False)  # a gfx1100-only build
    with pytest.raises(RuntimeError, match=r"gdn \(gdn_decode_rdna2\), v1 \(apply_w"):
        rdna._rdna3_hip_families()


def test_families_without_a_probe_never_need_the_build(set_arch, monkeypatch):
    set_arch("gfx1100", "hc,se,dense,ar")
    _fake_rocm_ops(monkeypatch, probes=False)
    assert set(rdna._rdna3_hip_families()) == {"hc", "se", "dense", "ar"}


def test_all_keeps_only_the_families_the_build_carries(set_arch, monkeypatch):
    set_arch("gfx1100", "all")
    _fake_rocm_ops(monkeypatch, "gdn_decode_rdna2", probes=False)
    expected = set(rdna.RDNA3_HIP_FAMILIES) - {"qsa", "moeu", "v1"}
    assert set(rdna._rdna3_hip_families()) == expected


def test_probe_table_names_only_known_families(set_arch):
    assert set(rdna._FAMILY_PROBE_OP) <= set(rdna.RDNA3_HIP_FAMILIES)


def test_family_list_is_read_once(set_arch, monkeypatch):
    set_arch("gfx1100", "hc")
    assert rdna.on_rdna3_hip("hc")
    monkeypatch.setenv("VLLM_RDNA3_HIP_OPS", "qsa")
    assert rdna.on_rdna3_hip("hc") and not rdna.on_rdna3_hip("qsa")


def test_unknown_family_argument_is_an_error(set_arch):
    set_arch("gfx1100", "all")
    with pytest.raises(ValueError):
        rdna.on_rdna3_hip("nope")


@pytest.mark.parametrize("family", sorted(rdna.RDNA3_HIP_FAMILIES))
@pytest.mark.parametrize("arch", ["gfx1030", "gfx942", None])
def test_on_rdna3_hip_is_false_off_gfx1100(set_arch, arch, family):
    # A malformed value must not matter on another arch: the parser is never
    # reached, so a typo cannot break a gfx1030 server.
    set_arch(arch, "all,bogus")
    assert rdna.on_rdna3_hip(family) is False


@pytest.mark.parametrize("family", sorted(rdna.RDNA3_HIP_FAMILIES))
def test_on_rdna3_hip_gfx1100_follows_the_opt_in(set_arch, family):
    set_arch("gfx1100", "")
    assert rdna.on_rdna3_hip(family) is False
    set_arch("gfx1100", family)
    assert rdna.on_rdna3_hip(family) is True
    for other in rdna.RDNA3_HIP_FAMILIES - {family}:
        assert rdna.on_rdna3_hip(other) is False
    set_arch("gfx1100", "all")
    assert rdna.on_rdna3_hip(family) is True


def test_on_rdna2_hip_unchanged_on_gfx1030(set_arch):
    set_arch("gfx1030", "")
    assert rdna.on_rdna2_hip() is True
    assert rdna.on_rdna2_hip("gdn") is True
    assert rdna.on_rdna2_hip(family=None) is True
    set_arch("gfx1030", "", all_hip="0")
    assert rdna.on_rdna2_hip() is False
    assert rdna.on_rdna2_hip(family=None) is False
    # The gfx1100 opt-in list has no effect on gfx1030.
    set_arch("gfx1030", "all", all_hip="0")
    assert rdna.on_rdna2_hip("v1") is False


def test_on_rdna2_hip_on_gfx1100(set_arch):
    set_arch("gfx1100", "")
    assert rdna.on_rdna2_hip() is False
    set_arch("gfx1100", "gdn")
    assert rdna.on_rdna2_hip("gdn") is True
    assert rdna.on_rdna2_hip() is False  # default family is v1
    set_arch("gfx1100", "v1")
    assert rdna.on_rdna2_hip() is True
    assert rdna.on_rdna2_hip("gdn") is False
    # VLLM_RDNA_ALL_HIP only governs gfx10x.
    set_arch("gfx1100", "v1", all_hip="0")
    assert rdna.on_rdna2_hip() is True


def test_gfx10x_only_meaning_is_kept_for_the_jit_warmup(set_arch):
    """kernel_warmup keeps the Triton JIT warmup on gfx1100 whatever is opted in."""
    set_arch("gfx1100", "all")
    assert rdna.on_rdna2_hip(family=None) is False
    set_arch("gfx942", "all")
    assert rdna.on_rdna2_hip(family=None) is False


def test_on_rdna_family_covers_gfx1100(set_arch):
    for arch, expected in (("gfx1030", True), ("gfx1100", True), ("gfx942", False)):
        set_arch(arch)
        assert rdna.on_rdna_family() is expected
    set_arch(None)
    assert rdna.on_rdna_family() is False


# --------------------------------------------------------------------------
# Call sites
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch, families, expected",
    [
        ("gfx1030", "", True),
        ("gfx1030", "all", True),
        ("gfx1100", "", False),
        ("gfx1100", "gdn,qsa,se,dense,ar,v1", False),
        ("gfx1100", "hc", True),
        ("gfx1100", "all", True),
        ("gfx942", "all", False),
        (None, "all", False),
    ],
)
def test_hyperconnection_fused_gate(set_arch, arch, families, expected):
    hc = pytest.importorskip("vllm.models.qwen4_exp.amd.hyperconnection")
    set_arch(arch, families)
    x = torch.zeros(2, 8, dtype=torch.float16)
    assert hc._rdna_fused_ok(x) is expected
    assert hc._rdna_fused_ok(x.bfloat16()) is False


def test_hyperconnection_gfx1030_never_consults_the_gfx1100_opt_in(
    set_arch, monkeypatch
):
    hc = pytest.importorskip("vllm.models.qwen4_exp.amd.hyperconnection")
    set_arch("gfx1030")

    def boom(family):
        raise AssertionError("gfx1030 must short-circuit on on_gfx10x()")

    monkeypatch.setattr(rdna, "on_rdna3_hip", boom)
    assert hc._rdna_fused_ok(torch.zeros(1, 8, dtype=torch.float16)) is True


def test_hyperconnection_fused_gate_honours_the_env_switch(set_arch, monkeypatch):
    hc = pytest.importorskip("vllm.models.qwen4_exp.amd.hyperconnection")
    set_arch("gfx1100", "hc")
    monkeypatch.setenv("VLLM_RDNA_FUSED_HC", "0")
    assert hc._rdna_fused_ok(torch.zeros(1, 8, dtype=torch.float16)) is False


@pytest.mark.parametrize(
    "arch, families, expected",
    [
        ("gfx1030", "", True),
        ("gfx1100", "", False),
        ("gfx1100", "hc", True),
        ("gfx1100", "gdn", False),
    ],
)
def test_hc_prefill_helper_gate(set_arch, monkeypatch, arch, families, expected):
    hc_rdna2 = pytest.importorskip("vllm.models.qwen4_exp.amd.ops.hc_rdna2")
    set_arch(arch, families)
    monkeypatch.setattr(hc_rdna2, "_HAVE_OPS", True)
    monkeypatch.setattr(hc_rdna2, "VLLM_RDNA_HC_PREFILL_HIP", True)
    assert hc_rdna2.hc_use_rdna2() is expected
    # A build without the gfx1030 sources never routes to the missing ops.
    monkeypatch.setattr(hc_rdna2, "_HAVE_OPS", False)
    assert hc_rdna2.hc_use_rdna2() is False


def test_rocm_op_probe_is_cached_per_name_set(monkeypatch):
    rdna_ops = pytest.importorskip("vllm.model_executor.layers.rdna_ops")
    rdna_ops._have_rocm_ops.cache_clear()
    _fake_rocm_ops(monkeypatch, "rdna_gemv_act")
    assert rdna_ops._have_rocm_ops("rdna_gemv_act") is True
    assert rdna_ops._have_rocm_ops("rdna_gemv_act", "rdna_hc_up_gate_mix") is False
    _fake_rocm_ops(monkeypatch, "rdna_gemv_act", "rdna_hc_up_gate_mix")
    assert rdna_ops._have_rocm_ops("rdna_gemv_act") is True  # cached
    assert rdna_ops._have_rocm_ops("rdna_gemv_act", "rdna_hc_up_gate_mix") is False
    rdna_ops._have_rocm_ops.cache_clear()
    assert rdna_ops._have_rocm_ops("rdna_gemv_act", "rdna_hc_up_gate_mix") is True
    rdna_ops._have_rocm_ops.cache_clear()


@pytest.mark.parametrize(
    "arch, families, have_ops, expected",
    [
        ("gfx1030", "", True, True),
        ("gfx1100", "", True, False),
        ("gfx1100", "qsa", True, True),
        ("gfx1100", "qsa", False, False),
        ("gfx1100", "hc,gdn,se,dense,ar,v1", True, False),
        ("gfx942", "all", True, False),
    ],
)
def test_qsa_hip_gate(set_arch, monkeypatch, arch, families, have_ops, expected):
    qsa_rdna2 = pytest.importorskip("vllm.models.qwen4_exp.amd.ops.qsa_rdna2")
    set_arch(arch, families)
    monkeypatch.setattr(qsa_rdna2, "_HAVE_OPS", have_ops)
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_DECODE_MODE", "auto")
    assert qsa_rdna2.qsa_use_rdna2() is expected
    assert qsa_rdna2.qsa_use_rdna2(torch.float16) is expected
    # The HIP kernels are fp16 only on every arch.
    assert qsa_rdna2.qsa_use_rdna2(torch.bfloat16) is False


def test_qsa_hip_gate_gfx1030_never_consults_the_opt_in(set_arch, monkeypatch):
    qsa_rdna2 = pytest.importorskip("vllm.models.qwen4_exp.amd.ops.qsa_rdna2")
    set_arch("gfx1030")
    monkeypatch.setattr(qsa_rdna2, "_HAVE_OPS", False)  # even without the probe
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_HIP", True)
    monkeypatch.setattr(qsa_rdna2, "VLLM_RDNA_QSA_DECODE_MODE", "auto")
    assert qsa_rdna2.qsa_use_rdna2() is True


@pytest.mark.parametrize(
    "arch, families, have_op, expected",
    [
        ("gfx1030", "", True, "rdna2"),
        ("gfx1100", "", True, "triton"),
        ("gfx1100", "qsa", True, "rdna2"),
        ("gfx1100", "qsa", False, "error"),
        ("gfx942", "all", True, "triton"),
    ],
)
def test_qsa_metadata_builder_choice(
    set_arch, monkeypatch, arch, families, have_op, expected
):
    qsa_cache = pytest.importorskip("vllm.models.qwen4_exp.common.qsa_cache")
    set_arch(arch, families)
    monkeypatch.setattr(qsa_cache, "HAS_TRITON", True)
    _fake_rocm_ops(monkeypatch, *(["build_qsa_metadata_rdna2"] if have_op else []))
    if expected == "error":
        with pytest.raises(RuntimeError, match="build_qsa_metadata_rdna2"):
            qsa_cache._resolve_qsa_metadata_builder()
        return
    builder = qsa_cache._resolve_qsa_metadata_builder()
    want = {
        "rdna2": qsa_cache.build_qsa_metadata_rdna2,
        "triton": qsa_cache.build_qsa_metadata_triton,
    }[expected]
    assert builder is want


@pytest.mark.parametrize(
    "arch, families, have_op, env, expected",
    [
        ("gfx1030", "", True, "1", True),
        ("gfx1030", "", True, "0", False),
        ("gfx1100", "", True, "1", False),
        ("gfx1100", "gdn", True, "1", True),
        ("gfx1100", "gdn", False, "1", False),
        ("gfx1100", "qsa,hc,v1", True, "1", False),
        ("gfx942", "all", True, "1", False),
    ],
)
def test_gdn_prefill_chain_gate(
    set_arch, monkeypatch, arch, families, have_op, env, expected
):
    gdn = pytest.importorskip(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    set_arch(arch, families)
    monkeypatch.setenv("VLLM_GDN_HIP_PREFILL", env)
    _fake_rocm_ops(monkeypatch, *(["gdn_prefill_prep_rdna2"] if have_op else []))
    assert gdn._gdn_prefill_dispatch_available() is expected


def _fake_shared_expert(fp16=True, exl3=False):
    dtype = torch.float16 if fp16 else torch.bfloat16

    def lin():
        return types.SimpleNamespace(weight=torch.zeros(4, 4, dtype=dtype), bias=None)

    return types.SimpleNamespace(
        _exl3_fp16_activation=exl3,
        gate_up_proj=lin(),
        down_proj=lin(),
        expert_gate=lin(),
    )


@pytest.mark.parametrize(
    "arch, families, expected",
    [
        ("gfx1030", "", True),
        ("gfx1030", "all", True),
        ("gfx1100", "", False),
        ("gfx1100", "hc,gdn,qsa,dense,ar,v1", False),
        ("gfx1100", "se", True),
        ("gfx942", "all", False),
    ],
)
def test_shared_expert_fused_gate(set_arch, arch, families, expected):
    qwen2_moe = pytest.importorskip("vllm.model_executor.models.qwen2_moe")
    set_arch(arch, families)
    x = torch.zeros(2, 4, dtype=torch.float16)
    assert qwen2_moe._rdna_shared_expert_ok(_fake_shared_expert(), x) is expected
    # Every arch keeps the weight-dtype / activation-mode / dtype refusals.
    assert qwen2_moe._rdna_shared_expert_ok(_fake_shared_expert(fp16=False), x) is False
    assert qwen2_moe._rdna_shared_expert_ok(_fake_shared_expert(exl3=True), x) is False
    assert (
        qwen2_moe._rdna_shared_expert_ok(_fake_shared_expert(), x.bfloat16()) is False
    )


@pytest.fixture
def se_ops(monkeypatch):
    """The shared-expert HIP ops as recorders on CPU tensors; the fixture value
    is the list of op names the call under test reached."""
    rdna_ops = pytest.importorskip("vllm.model_executor.layers.rdna_ops")
    calls: list[str] = []

    def gate_up(name):
        def op(x, w, *rest):
            calls.append(name)
            rows, half = x.shape[0], w.shape[0] // 2
            return (
                torch.zeros(rows, half, dtype=torch.float16),
                torch.zeros(rows, dtype=torch.float32),
            )

        return op

    def down(name):
        def op(act, w, *rest):
            calls.append(name)
            return torch.zeros(act.shape[0], w.shape[0], dtype=torch.float16)

        return op

    ns = types.SimpleNamespace(
        rdna_se_gate_up_silu=gate_up("wave_gate_up"),
        rdna_se_down_gated=down("wave_down"),
        rdna_se_gate_up_tile=gate_up("tile_gate_up"),
        rdna_se_down_tile=down("tile_down"),
    )
    monkeypatch.setattr(torch.ops, "_rocm_C", ns, raising=False)
    rdna_ops._have_rocm_ops.cache_clear()
    yield calls
    rdna_ops._have_rocm_ops.cache_clear()


def _se_call(rdna_ops, rows, inter=160):
    k, h = 64, 32
    x = torch.zeros(rows, k, dtype=torch.float16)
    w1 = torch.zeros(2 * inter, k, dtype=torch.float16)
    w2 = torch.zeros(h, inter, dtype=torch.float16)
    w_gate = torch.zeros(k, dtype=torch.float16)
    out = rdna_ops._rdna_shared_expert(x, w1, None, None, w2, None, None, w_gate)
    assert out.shape == (rows, h)


@pytest.mark.parametrize(
    "rows, cap, tile_min, down_tile, inter, expected",
    [
        (8, 32, 9, True, 160, ["wave_gate_up", "wave_down"]),
        (24, 32, 9, True, 160, ["tile_gate_up", "tile_down"]),
        (32, 32, 9, False, 160, ["tile_gate_up", "wave_down"]),  # the old routes
        (32, 32, 0, True, 160, ["wave_gate_up", "wave_down"]),
        (33, 32, 9, True, 160, []),  # above the default cap: torch
        (48, 64, 9, True, 160, ["tile_gate_up", "tile_down"]),
        (64, 64, 9, True, 640, ["tile_gate_up", "tile_down"]),
        (65, 64, 9, True, 160, []),
        # the wave kernels stop at 32 rows: above it both tile halves or torch
        (48, 64, 9, False, 160, []),
        (48, 64, 0, True, 160, []),
        # 73 column tiles fit the 128 tickets once (32 rows) but not twice
        (32, 64, 9, True, 1152, ["tile_gate_up", "tile_down"]),
        (48, 64, 9, True, 1152, []),
    ],
)
def test_shared_expert_row_cap_and_tile_gates(
    monkeypatch, se_ops, rows, cap, tile_min, down_tile, inter, expected
):
    rdna_ops = pytest.importorskip("vllm.model_executor.layers.rdna_ops")
    monkeypatch.setattr(rdna_ops, "_SE_DECODE_MAX", cap)
    monkeypatch.setattr(rdna_ops, "_SE_TILE_MIN", tile_min)
    monkeypatch.setattr(rdna_ops, "_SE_DOWN_TILE", down_tile)
    _se_call(rdna_ops, rows, inter)
    assert se_ops == expected


@pytest.mark.parametrize(
    "arch, families, env, expected",
    [
        ("gfx1030", "", "1", True),
        ("gfx1030", "", "0", False),
        ("gfx1100", "", "1", False),
        ("gfx1100", "dense", "1", True),
        ("gfx1100", "dense", "0", False),
        ("gfx1100", "hc,gdn,qsa,se,ar,v1", "1", False),
        ("gfx942", "all", "1", False),
    ],
)
def test_dense_int8_shadow_gate(set_arch, monkeypatch, arch, families, env, expected):
    dense = pytest.importorskip("vllm.model_executor.layers.rdna_dense_int8")
    set_arch(arch, families)
    monkeypatch.setattr(dense, "_ENABLED", None)
    monkeypatch.setenv("VLLM_RDNA_DENSE_INT8", env)
    assert dense.enabled() is expected


@pytest.fixture
def gemm_probe(monkeypatch):
    """Patch the dense GEMV ops of layers/utils.py with recorders."""
    utils = pytest.importorskip("vllm.model_executor.layers.utils")
    calls: list[str] = []

    def rec(name):
        def fn(*args, **kwargs):
            calls.append(name)
            # gemv_f16_rdna2(x, w, bias) and rdna_gemm_smallm(x, w); wvSplitK(w, x, ..)
            x, weight = (args[1], args[0]) if name == "wvSplitK" else args[:2]
            return torch.zeros(x.shape[0], weight.shape[0], dtype=x.dtype)

        return fn

    for name in ("gemv_f16_rdna2", "wvSplitK", "rdna_gemm_smallm"):
        monkeypatch.setattr(utils.ops, name, rec(name), raising=False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 70)
    monkeypatch.setattr(utils, "_have_rdna_smallm", lambda: True)
    monkeypatch.setattr(utils, "_RDNA_SMALLM_MAX_BYTES", 4 << 20)
    return utils, calls


@pytest.mark.parametrize(
    "arch, families, n, expected",
    [
        ("gfx1030", "", 3, "gemv_f16_rdna2"),
        ("gfx1030", "", 24, "rdna_gemm_smallm"),
        ("gfx1100", "", 3, "wvSplitK"),
        ("gfx1100", "dense", 3, "gemv_f16_rdna2"),
        ("gfx1100", "dense", 24, "rdna_gemm_smallm"),
        ("gfx1100", "hc,gdn,qsa,se,ar,v1", 3, "wvSplitK"),
    ],
)
def test_dense_fp16_gemv_dispatch(set_arch, gemm_probe, arch, families, n, expected):
    utils, calls = gemm_probe
    set_arch(arch, families)
    x = torch.zeros(n, 2560, dtype=torch.float16)
    w = torch.zeros(64, 2560, dtype=torch.float16)
    utils.rocm_unquantized_gemm_impl(x, w)
    assert calls[:1] == [expected], calls


def test_dense_gemv_gfx1100_skips_the_smallm_op_when_not_built(
    set_arch, gemm_probe, monkeypatch
):
    utils, calls = gemm_probe
    set_arch("gfx1100", "dense")
    monkeypatch.setattr(utils, "_have_rdna_smallm", lambda: False)
    x = torch.zeros(24, 2560, dtype=torch.float16)
    w = torch.zeros(64, 2560, dtype=torch.float16)
    with contextlib.suppress(Exception):  # the rocBLAS fallback may need a device
        utils.rocm_unquantized_gemm_impl(x, w)
    assert "rdna_gemm_smallm" not in calls


@pytest.mark.parametrize(
    "arch, families, expected",
    [
        ("gfx1030", "", True),
        ("gfx1100", "", False),
        ("gfx1100", "hc,gdn,qsa,se,dense,ar,v1", False),
        ("gfx1100", "moeu", True),
        ("gfx942", "all", False),
    ],
)
def test_unquantized_moe_hip_backend_priority(set_arch, arch, families, expected):
    """The MTP drafter's fp16 experts: HIP backend first only where it is on."""
    unq = pytest.importorskip("vllm.model_executor.layers.fused_moe.oracle.unquantized")
    set_arch(arch, families)
    backends = unq._get_priority_backends(types.SimpleNamespace())
    assert (unq.UnquantizedMoeBackend.RDNA2_HIP in backends) is expected
    if expected:
        assert backends[0] is unq.UnquantizedMoeBackend.RDNA2_HIP
        assert unq.UnquantizedMoeBackend.TRITON in backends  # the fallback stays


@pytest.mark.parametrize(
    "arch, families, have_ops, expected",
    [
        ("gfx1030", "", True, True),
        ("gfx1030", "", False, True),  # gfx10x fails loudly in apply() instead
        ("gfx1100", "", True, False),
        ("gfx1100", "moeu", True, True),
        ("gfx1100", "moeu", False, False),  # a gfx1100-only build lacks the ops
        ("gfx1100", "hc,gdn,qsa,se,dense,ar,v1", True, False),
        ("gfx942", "all", True, False),
    ],
)
def test_unquantized_moe_hip_experts_device_support(
    set_arch, monkeypatch, arch, families, have_ops, expected
):
    experts = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.experts.rdna2_unquantized_moe"
    )
    set_arch(arch, families)
    monkeypatch.setattr(experts, "rdna2_unquantized_moe_available", lambda: have_ops)
    assert experts.Rdna2UnquantizedExperts._supports_current_device() is expected


# --------------------------------------------------------------------------
# Mixed-arch amdsmi handling
# --------------------------------------------------------------------------


@pytest.fixture
def fake_amdsmi(monkeypatch):
    """``fake_amdsmi(targets)``: amdsmi reports one handle per entry.

    An entry that is an Exception instance makes that handle's asic query raise.
    """

    def _install(targets):
        handles = list(range(len(targets)))
        monkeypatch.setattr(rocm, "amdsmi_init", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(
            rocm, "amdsmi_shut_down", lambda *a, **k: None, raising=False
        )
        monkeypatch.setattr(
            rocm, "amdsmi_get_processor_handles", lambda: handles, raising=False
        )

        def asic(handle):
            target = targets[handle]
            if isinstance(target, Exception):
                raise target
            return {"target_graphics_version": target}

        monkeypatch.setattr(rocm, "amdsmi_get_gpu_asic_info", asic, raising=False)

    return _install


def test_amdsmi_homogeneous_box(fake_amdsmi):
    fake_amdsmi(["gfx1100", "gfx1100"])
    assert rocm._query_gcn_arch_from_amdsmi() == "gfx1100"


def test_amdsmi_mixed_box_raises(fake_amdsmi):
    fake_amdsmi(["gfx1100", "gfx1030", "gfx1030", "gfx1100"])
    with pytest.raises(RuntimeError, match="mixed"):
        rocm._query_gcn_arch_from_amdsmi()


def test_amdsmi_skips_a_handle_that_cannot_answer(fake_amdsmi):
    fake_amdsmi([RuntimeError("asleep"), "gfx1030", "gfx1030"])
    assert rocm._query_gcn_arch_from_amdsmi() == "gfx1030"


def test_amdsmi_no_valid_answer_raises(fake_amdsmi):
    fake_amdsmi([RuntimeError("a"), ""])
    with pytest.raises(RuntimeError, match="did not return"):
        rocm._query_gcn_arch_from_amdsmi()


def test_mixed_box_resolves_through_torch(fake_amdsmi, monkeypatch):
    """Mixed archs fall back to torch, which honours the visible-device list."""
    fake_amdsmi(["gfx1100", "gfx1030"])
    props = types.SimpleNamespace(gcnArchName="gfx1100")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: props, raising=False
    )
    assert rocm._get_gcn_arch() == "gfx1100"


def test_homogeneous_box_does_not_touch_torch(fake_amdsmi, monkeypatch):
    fake_amdsmi(["gfx1030", "gfx1030"])

    def boom(device):
        raise AssertionError("torch.cuda must not be initialised")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom, raising=False)
    assert rocm._get_gcn_arch() == "gfx1030"
