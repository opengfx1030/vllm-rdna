# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused HC prefill mix on gfx1030 (csrc/rocm/hc_prefill_rdna2.cu) against the
torch prefill path of ``rdna_hc_mix`` (rocBLAS + Triton silu/gate mix).

The split-K down GEMM reduces its K slices with per-tile arrival tickets
(zero between launches, one pool slot per stream) instead of a zero-filled
atomic accumulator, so a call is exactly two kernels and bitwise repeatable.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

from vllm.platforms import current_platform


def _gfx10x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    return on_gfx10x()


pytestmark = pytest.mark.skipif(not _gfx10x(), reason="gfx1030 HIP kernels")

HC = 4
HIDDEN = 2560
LORA = 320
DOWN_N = LORA + HC + 12  # merged down+inject, padded to 16 rows
M_VALUES = [
    9, 12, 16, 17, 18, 24, 25, 32, 33, 63, 64, 65, 100, 128, 200, 255, 256, 384,
    513, 1030, 2048,
]  # fmt: skip


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    w_down = (torch.randn(DOWN_N, HC * HIDDEN, device="cuda") * 0.02).half()
    w_up = (torch.randn(HC * HIDDEN, LORA, device="cuda") * 0.05).half()
    return w_down, w_up


def _mix(xn, w_down, w_up, mode, monkeypatch):
    from vllm.model_executor.layers import rdna_ops

    monkeypatch.setattr(rdna_ops, "_HC_PREFILL_FUSED", mode)
    return rdna_ops._rdna_hc_mix(xn, w_down, None, None, w_up, None, None, LORA, HC)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("m", M_VALUES)
def test_fused_prefill_matches_torch(weights, mode, m, monkeypatch):
    w_down, w_up = weights
    xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
    ref_out, ref_inj = _mix(xn, w_down, w_up, 0, monkeypatch)
    out, inj = _mix(xn, w_down, w_up, mode, monkeypatch)
    assert out.shape == ref_out.shape and inj.shape == ref_inj.shape
    assert inj.shape == (m, HC) and inj.is_contiguous()
    assert inj.dtype == torch.float16 and out.dtype == torch.float16
    # mode 1 shares the rocBLAS down GEMM; mode 2 accumulates it in fp32
    # (split-K) and rounds once, so the injection may differ by an fp16 ulp.
    torch.testing.assert_close(inj, ref_inj, atol=4e-3, rtol=2e-3)
    torch.testing.assert_close(out, ref_out, atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("m", [256, 1030, 2048])
@pytest.mark.parametrize(
    "knob", ["SPLITS=1", "SPLITS=2", "SPLITS=3", "SPLITS=8", "TARGET=144"]
)
def test_down_plan_env_overrides_match_torch(weights, m, knob, monkeypatch):
    """The M >= 256 down GEMM keeps its result under every split plan."""
    name, value = knob.split("=")
    monkeypatch.setenv(f"VLLM_RDNA_HC_DOWN_{name}", value)
    w_down, w_up = weights
    xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
    ref_out, ref_inj = _mix(xn, w_down, w_up, 0, monkeypatch)
    out, inj = _mix(xn, w_down, w_up, 2, monkeypatch)
    torch.testing.assert_close(inj, ref_inj, atol=4e-3, rtol=2e-3)
    torch.testing.assert_close(out, ref_out, atol=3e-3, rtol=1e-2)


def test_decode_unchanged(weights, monkeypatch):
    w_down, w_up = weights
    xn = torch.randn(4, HC * HIDDEN, device="cuda").half()
    ref = _mix(xn, w_down, w_up, 0, monkeypatch)
    for mode in (1, 2):
        got = _mix(xn, w_down, w_up, mode, monkeypatch)
        torch.testing.assert_close(got[0], ref[0], atol=0, rtol=0)
        torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0)


def test_non_contiguous_falls_back(weights, monkeypatch):
    w_down, w_up = weights
    big = torch.randn(100, 2 * HC * HIDDEN, device="cuda").half()
    xn = big[:, : HC * HIDDEN]
    assert not xn.is_contiguous()
    ref = _mix(xn.contiguous(), w_down, w_up, 0, monkeypatch)
    got = _mix(xn, w_down, w_up, 2, monkeypatch)
    torch.testing.assert_close(got[0], ref[0], atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("mode", [1, 2])
def test_cuda_graph_replay(weights, mode, monkeypatch):
    """Fresh outputs on the current stream: capture once, replay on new data."""
    w_down, w_up = weights
    m = 300
    xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
    _mix(xn, w_down, w_up, mode, monkeypatch)  # warm up outside capture
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, inj = _mix(xn, w_down, w_up, mode, monkeypatch)
    for _ in range(3):
        xn.copy_(torch.randn_like(xn))
        g.replay()
        ref_out, ref_inj = _mix(xn.clone(), w_down, w_up, 0, monkeypatch)
        torch.testing.assert_close(inj, ref_inj, atol=4e-3, rtol=2e-3)
        torch.testing.assert_close(out, ref_out, atol=3e-3, rtol=1e-2)


def test_fused_is_default():
    import os

    from vllm.model_executor.layers import rdna_ops

    if "VLLM_RDNA_HC_PREFILL_FUSED" not in os.environ:
        assert rdna_ops._HC_PREFILL_FUSED == 2
    assert rdna_ops._hc_fused_available()


def test_op_checks_shapes(weights):
    import vllm._custom_ops as ops

    w_down, w_up = weights
    xn = torch.randn(16, HC * HIDDEN, device="cuda").half()
    with pytest.raises(RuntimeError):
        ops.rdna_hc_mix_prefill(xn, w_down, w_up, LORA, 3)
    with pytest.raises(RuntimeError):
        ops.rdna_hc_up_gate_mix_prefill(
            torch.empty(16, DOWN_N, device="cuda"), w_up, xn, LORA, HC
        )


def _hc(xn, w_down, w_up):
    import vllm._custom_ops as ops

    return ops.rdna_hc_mix_prefill(xn, w_down, w_up, LORA, HC)


# 1 and 8 are decode shapes the op accepts directly; 8192 has a single K slice
# (no workspace, no tickets); the rest split K in the narrow (< 256) and wide
# tile configurations.
@pytest.mark.parametrize("m", [1, 8, 9, 18, 64, 255, 256, 513, 2048, 8192])
def test_bitwise_repeatable(weights, m):
    w_down, w_up = weights
    xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
    ref = _hc(xn, w_down, w_up)
    for _ in range(5):
        got = _hc(xn, w_down, w_up)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_interleaved_shapes_rearm_tickets(weights):
    """Tile counts differ per shape; every launch must leave the tickets at 0."""
    w_down, w_up = weights
    ms = [9, 18, 255, 64, 100, 384, 18, 24, 2048, 9]
    xs = [torch.randn(m, HC * HIDDEN, device="cuda").half() for m in ms]
    refs = [_hc(x, w_down, w_up) for x in xs]
    for _ in range(3):
        for x, ref in zip(xs, refs):
            got = _hc(x, w_down, w_up)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_concurrent_streams_do_not_share_tickets(weights):
    w_down, w_up = weights
    xa = torch.randn(18, HC * HIDDEN, device="cuda").half()
    xb = torch.randn(64, HC * HIDDEN, device="cuda").half()
    xc = torch.randn(300, HC * HIDDEN, device="cuda").half()
    refs = [_hc(x, w_down, w_up) for x in (xa, xb, xc)]
    torch.accelerator.synchronize()
    streams = [torch.cuda.Stream() for _ in range(3)]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    runs = []
    for _ in range(40):
        for s, x in zip(streams, (xa, xb, xc)):
            with torch.cuda.stream(s):
                runs.append(_hc(x, w_down, w_up))
    torch.accelerator.synchronize()
    for i, got in enumerate(runs):
        ref = refs[i % 3]
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_graph_with_several_mixes_and_eager_calls_between_replays(weights):
    w_down, w_up = weights
    xs = [torch.randn(m, HC * HIDDEN, device="cuda").half() for m in (18, 64, 300)]
    for x in xs:
        _hc(x, w_down, w_up)  # creates the ticket pool outside capture
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [_hc(x, w_down, w_up) for x in xs]
    for _ in range(3):
        for x in xs:
            x.copy_(torch.randn_like(x))
        g.replay()
        # eager calls on the replay stream use their own slot
        refs = [_hc(x.clone(), w_down, w_up) for x in xs]
        for got, ref in zip(outs, refs):
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_decode_mix_is_two_kernels(weights):
    """No accumulator fill: a decode-size call launches the down GEMM and the
    up/mix kernel and nothing else."""
    from torch.profiler import ProfilerActivity, profile

    w_down, w_up = weights
    for m in (9, 18, 64):
        xn = torch.randn(m, HC * HIDDEN, device="cuda").half()
        _hc(xn, w_down, w_up)
        torch.accelerator.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _hc(xn, w_down, w_up)
            torch.accelerator.synchronize()
        names = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        if not names:
            pytest.skip("profiler returned no kernel events")
        assert len(names) == 2, names
        assert "gemm_tn_splitk_k" in names[0] and "hc_up_mix_prefill_k" in names[1]


# A fresh process: the first HC mix of the process happens inside a graph
# capture (no ticket pool yet -> per-call zeroed tickets), then more captures
# and eager calls on more streams than the pool has slots.
_FRESH_PROCESS = textwrap.dedent(
    """
    import torch
    import vllm._custom_ops as ops

    HC, HIDDEN, LORA, DOWN_N = 4, 2560, 320, 336
    torch.manual_seed(0)
    w_down = (torch.randn(DOWN_N, HC * HIDDEN, device="cuda") * 0.02).half()
    w_up = (torch.randn(HC * HIDDEN, LORA, device="cuda") * 0.05).half()

    def mix(x):
        return ops.rdna_hc_mix_prefill(x, w_down, w_up, LORA, HC)

    def same(a, b):
        return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    # load the module's kernels without touching the ticket pool
    dai = torch.randn(16, DOWN_N, device="cuda").half()
    xw = torch.randn(16, HC * HIDDEN, device="cuda").half()
    ops.rdna_hc_up_gate_mix_prefill(dai, w_up, xw, LORA, HC)
    torch.accelerator.synchronize()

    x = torch.randn(18, HC * HIDDEN, device="cuda").half()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):  # first HC mix of the process, while capturing
        out = mix(x)
    ref = mix(x)  # creates the pool
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g.replay()
        assert same(out, mix(x.clone())), "capture before the pool existed"

    # more eager streams than slots
    xs = [torch.randn(18 + i, HC * HIDDEN, device="cuda").half() for i in range(12)]
    refs = [mix(v) for v in xs]
    torch.accelerator.synchronize()
    streams = [torch.cuda.Stream() for _ in xs]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    runs = []
    for _ in range(10):
        for s, v in zip(streams, xs):
            with torch.cuda.stream(s):
                runs.append(mix(v))
    torch.accelerator.synchronize()
    for i, got in enumerate(runs):
        assert same(got, refs[i % len(xs)]), "eager streams beyond the slots"

    # more capture streams than capture slots
    graphs = []
    for i in range(12):
        xi = torch.randn(24, HC * HIDDEN, device="cuda").half()
        gi = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gi, stream=torch.cuda.Stream()):
            oi = mix(xi)
        graphs.append((gi, xi, oi))
    for gi, xi, oi in graphs:
        xi.copy_(torch.randn_like(xi))
        gi.replay()
        assert same(oi, mix(xi.clone())), "capture streams beyond the slots"
    torch.accelerator.synchronize()
    print("OK")
    """
)


def test_pool_exhaustion_and_first_use_in_capture():
    res = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


# ---------------------------------------------------------------------------
# rdna_gemm_smallm: the same ticketed split-K kernel as a dense fp16 GEMM
# ---------------------------------------------------------------------------
SMALLM_SHAPES = [
    (512, 2560),  # MoE router
    (24, 2560),  # GDN in_proj_ba
    (640, 2560),
    (320, 2560),
    (336, 10240),  # HC down projection
    (2560, 1536),
    (3584, 2560),
    (4, 32),  # one K slab, one tile
    (256, 96),  # fewer K slabs than the pipeline depth
    (512, 224),  # 7 slabs: partial last pipeline group
    (128, 1312),  # 41 slabs split over several K slices
]


def _smallm(x, w):
    import vllm._custom_ops as ops

    return ops.rdna_gemm_smallm(x, w)


@pytest.mark.parametrize("n,k", SMALLM_SHAPES)
@pytest.mark.parametrize("m", [1, 3, 8, 9, 16, 24, 31, 32])
def test_gemm_smallm_matches_fp32_reference(n, k, m):
    torch.manual_seed(n + k + m)
    w = (torch.randn(n, k, device="cuda") * 0.02).half()
    x = torch.randn(m, k, device="cuda").half()
    y = _smallm(x, w)
    assert y.shape == (m, n) and y.dtype == torch.float16 and y.is_contiguous()
    ref = x.float() @ w.float().T
    torch.testing.assert_close(y.float(), ref, atol=2e-2, rtol=2e-3)
    # Deterministic: the K slices are summed in z order by the last block.
    assert torch.equal(y, _smallm(x, w))


def test_gemm_smallm_interleaved_with_hc_rearms_tickets(weights):
    w_down, w_up = weights
    xs = [torch.randn(m, HC * HIDDEN, device="cuda").half() for m in (9, 24, 32)]
    w = (torch.randn(512, 2560, device="cuda") * 0.02).half()
    xr = torch.randn(24, 2560, device="cuda").half()
    hc_ref = [_hc(x, w_down, w_up) for x in xs]
    sm_ref = _smallm(xr, w)
    for _ in range(3):
        for x, ref in zip(xs, hc_ref):
            got = _hc(x, w_down, w_up)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
            assert torch.equal(_smallm(xr, w), sm_ref)


def test_gemm_smallm_graph_replay():
    w = (torch.randn(512, 2560, device="cuda") * 0.02).half()
    x = torch.randn(24, 2560, device="cuda").half()
    _smallm(x, w)  # creates the ticket pool outside capture
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _smallm(x, w)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g.replay()
        assert torch.equal(out, _smallm(x.clone(), w))


def test_gemm_smallm_checks_shapes():
    w = (torch.randn(512, 2560, device="cuda") * 0.02).half()
    with pytest.raises(RuntimeError):  # more than 32 rows
        _smallm(torch.randn(33, 2560, device="cuda").half(), w)
    with pytest.raises(RuntimeError):  # K not a multiple of 32
        _smallm(torch.randn(8, 2552, device="cuda").half(), w[:, :2552].contiguous())
    with pytest.raises(RuntimeError):  # N not a multiple of 4
        _smallm(torch.randn(8, 2560, device="cuda").half(), w[:510].contiguous())
    with pytest.raises(RuntimeError):  # not fp16
        _smallm(torch.randn(8, 2560, device="cuda"), w)
    with pytest.raises(RuntimeError):  # non-contiguous weight
        _smallm(torch.randn(8, 2560, device="cuda").half(), w.T.contiguous().T)


def test_dense_dispatch_routes_small_weights_to_smallm(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers import utils

    calls = []
    real = ops.rdna_gemm_smallm

    def spy(x, w):
        calls.append(tuple(w.shape))
        return real(x, w)

    monkeypatch.setattr(ops, "rdna_gemm_smallm", spy)
    monkeypatch.setattr(utils, "_RDNA_SMALLM_MAX_BYTES", 4 << 20)

    def run(m, n, k, bias=False):
        w = (torch.randn(n, k, device="cuda") * 0.02).half()
        x = torch.randn(m, k, device="cuda").half()
        b = torch.randn(n, device="cuda").half() if bias else None
        return x, w, utils.rocm_unquantized_gemm_impl(x, w, b)

    x, w, y = run(24, 512, 2560)
    assert calls == [(512, 2560)]
    torch.testing.assert_close(y.float(), x.float() @ w.float().T, atol=2e-2, rtol=2e-3)
    x, w, y = run(9, 24, 2560)  # GDN in_proj_ba shape, lowest row count
    assert calls[-1] == (24, 2560)
    for m, n, k, bias in [
        (8, 512, 2560, False),  # gemv range
        (33, 512, 2560, False),  # more rows than the kernel takes
        (24, 10240, 2560, False),  # weight above the ceiling
        (24, 512, 2560, True),  # bias
    ]:
        before = len(calls)
        run(m, n, k, bias)
        assert len(calls) == before, (m, n, k, bias)
    monkeypatch.setattr(utils, "_RDNA_SMALLM_MAX_BYTES", 0)
    before = len(calls)
    run(24, 512, 2560)
    assert len(calls) == before
