# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LeapQuant GDN state kernels (gdn_lq_*_rdna2) against the FP32 oracle in
vllm/model_executor/layers/mamba/ops/leapquant_gdn.py.

The failure modes guarded here are the ones that corrupt serving silently:
records committed for rejected MTP drafts, folds that drift from the FP32
trajectory, NULL-block slots that get written, align-mode block copies that
hand out an uncommitted state, and kernels that do not replay inside a CUDA
graph.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops import leapquant_gdn as lq

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx103" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="RDNA2 (gfx103x) GPU required",
)

K = V = 128
H, HV = 2, 6
WINDOW, R, ITERS = 16, 4, 2


def _layout(max_query_len: int) -> lq.LeapQuantLayout:
    return lq.LeapQuantLayout(
        K,
        V,
        window=WINDOW,
        num_comp=R,
        max_query_len=max_query_len,
        power_iters=ITERS,
    )


def _ops():
    return torch.ops._rocm_C


def _normalize(L, blob, idx, num_acc):
    _ops().gdn_lq_normalize_rdna2(
        blob,
        idx,
        num_acc,
        V,
        R,
        L.capacity,
        WINDOW,
        ITERS,
    )


def _store(L, state, blob, idx):
    _ops().gdn_lq_store_rdna2(state, blob, idx, R, L.capacity, WINDOW, ITERS)


def _materialize(L, blob, idx, n):
    out = torch.empty(n, HV, V, K, dtype=torch.float32, device=blob.device)
    _ops().gdn_lq_materialize_rdna2(blob, out, idx, R, L.capacity, WINDOW)
    return out


def _params(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A_log = torch.log(torch.rand(HV, generator=g) * 0.99 + 0.01).cuda()
    dt_bias = (torch.randn(HV, generator=g) * 0.1).cuda()
    return A_log, dt_bias


def _tokens(n, t, seed, outlier=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n, t, H, K, generator=g)
    k = torch.randn(n, t, H, K, generator=g)
    v = torch.randn(n, t, HV, V, generator=g)
    if outlier is not None:
        v = v + outlier
    a = torch.randn(n, t, HV, generator=g) - 3
    b = torch.randn(n, t, HV, generator=g)
    return [x.half().cuda() for x in (q, k, v, a, b)]


def _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias):
    n, t = q.shape[:2]
    out = torch.empty(1, n * t, HV, V, dtype=torch.float16, device="cuda")
    cu = torch.arange(0, n * t + 1, t, dtype=torch.int32, device="cuda")
    _ops().gdn_lq_decode_rdna2(
        q.reshape(1, n * t, H, K),
        k.reshape(1, n * t, H, K),
        v.reshape(1, n * t, HV, V),
        a.reshape(n * t, HV),
        b.reshape(n * t, HV),
        A_log,
        dt_bias,
        out,
        blob,
        cu,
        idx,
        K**-0.5,
        R,
        L.capacity,
        WINDOW,
    )
    return out.view(n, t, HV, V)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def test_store_materialize_matches_reference():
    L = _layout(1)
    n = 3
    torch.manual_seed(0)
    # Low-rank outliers plus noise, the structure the compensators target.
    S = torch.randn(n, HV, V, K, device="cuda")
    S += (
        40
        * torch.randn(n, HV, V, 1, device="cuda")
        * torch.randn(n, HV, 1, K, device="cuda")
    )
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    blob.random_(0, 255)  # stale bytes must not leak into a fresh store
    idx = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
    _store(L, S, blob, idx)
    got = _materialize(L, blob, idx, n)

    ref_blob = blob[1:].clone()
    lq.leapquant_store_state(L, ref_blob, S)
    ref = lq.leapquant_materialize(L, ref_blob)

    assert _rel(got, S) < 5e-3, "HIP fold loses more than INT8 resolution"
    assert _rel(got, ref) < 5e-3
    assert torch.isfinite(got).all()
    hdr = L.views(blob[1:])["header"]
    assert (hdr[..., :2] == 0).all()


def test_null_block_untouched_and_zero_output():
    L = _layout(3)
    A_log, dt_bias = _params()
    blob = torch.zeros(3, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    blob[0].fill_(7)
    snapshot = blob[0].clone()
    idx = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q, k, v, a, b = _tokens(2, 3, seed=1)
    _normalize(L, blob, idx, torch.tensor([1, 1], dtype=torch.int32, device="cuda"))
    o = _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias)
    assert torch.equal(blob[0], snapshot)
    assert (o[0] == 0).all()
    assert o[1].abs().sum() > 0


def test_decode_matches_reference_on_same_blob():
    L = _layout(3)
    A_log, dt_bias = _params()
    n = 4
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    # Home column of a [n, 1 + num_spec] state-index table: a strided view.
    table = torch.zeros(n, 3, dtype=torch.int32, device="cuda")
    table[:, 0] = torch.arange(1, n + 1, dtype=torch.int32)
    idx = table[:, 0]
    S = torch.randn(n, HV, V, K, device="cuda") * 0.5
    _store(L, S, blob, idx.contiguous())
    # Fill part of the window so the replay path is exercised.
    for step in range(5):
        q, k, v, a, b = _tokens(n, 3, seed=10 + step)
        _normalize(L, blob, idx, torch.full((n,), 2, dtype=torch.int32, device="cuda"))
        _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias)
    _normalize(
        L, blob, idx, torch.tensor([1, 2, 3, 2], dtype=torch.int32, device="cuda")
    )

    ref_blob = blob[1:].clone()
    q, k, v, a, b = _tokens(n, 3, seed=99)
    got = _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias)
    ref = lq.leapquant_decode(L, ref_blob, q, k, v, a, b, A_log, dt_bias, K**-0.5)
    assert _rel(got, ref) < 2e-3
    gv, rv = L.views(blob[1:]), L.views(ref_blob)
    assert torch.equal(gv["header"], rv["header"])
    c = int(gv["header"][0, 0, 0])
    assert _rel(gv["rec_u"][..., c : c + 3, :], rv["rec_u"][..., c : c + 3, :]) < 2e-3
    assert _rel(gv["rec_k"][..., c : c + 3, :], rv["rec_k"][..., c : c + 3, :]) < 2e-3


@pytest.mark.parametrize("spec", [False, True])
def test_trajectory_tracks_fp32_with_rejections(spec):
    """Long decode with folds every window; with MTP-style rejections the
    committed state must follow only the accepted prefix."""
    T = 3 if spec else 1
    L = _layout(T)
    A_log, dt_bias = _params(seed=3)
    n = 2
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    idx = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
    S_ref = torch.zeros(n, HV, V, K, device="cuda")
    outlier = torch.randn(V) * 4
    gen = torch.Generator(device="cpu").manual_seed(5)
    num_acc = None
    errs = []
    for step in range(120):
        q, k, v, a, b = _tokens(n, T, seed=1000 + step, outlier=outlier)
        _normalize(L, blob, idx, num_acc)
        o = _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias)
        acc = (
            torch.randint(1, T + 1, (n,), generator=gen)
            if spec
            else torch.ones(n, dtype=torch.long)
        )
        qn = lq.l2norm(q).repeat_interleave(HV // H, 2) * K**-0.5
        kn = lq.l2norm(k).repeat_interleave(HV // H, 2)
        g, beta = lq.gdn_gates(a, b, A_log, dt_bias)
        for i in range(n):
            Sr = S_ref[i]
            for t in range(int(acc[i])):
                Sr, o_r, _ = lq.gdn_step_fp32(
                    Sr, qn[i, t], kn[i, t], v[i, t], g[i, t], beta[i, t]
                )
                errs.append(_rel(o[i, t], o_r))
            S_ref[i] = Sr
        # Non-spec steps read slot 0: num_accepted None commits one record.
        num_acc = acc.to(torch.int32).cuda() if spec else None
    late = errs[len(errs) // 2 :]
    assert max(late) < 3e-2, f"worst late-token rel err {max(late):.3e}"
    assert sum(late) / len(late) < 6e-3, (
        f"mean late rel err {sum(late) / len(late):.3e}"
    )


def test_normalize_and_decode_replay_in_cuda_graph():
    L = _layout(3)
    A_log, dt_bias = _params()
    n = 2
    blob = torch.zeros(n + 1, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    idx = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
    num_acc = torch.full((n,), 3, dtype=torch.int32, device="cuda")
    q, k, v, a, b = _tokens(n, 3, seed=7)
    eager_blob = blob.clone()

    def step(bl):
        _normalize(L, bl, idx, num_acc)
        return _decode(L, bl, idx, q, k, v, a, b, A_log, dt_bias)

    step(blob)  # warm up / build kernels outside capture
    blob.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(blob)
    blob.zero_()
    eager = []
    for _ in range(8):  # crosses a window boundary: folds inside the graph
        graph.replay()
        eager.append(step(eager_blob))
        torch.accelerator.synchronize()
        assert torch.equal(out, eager[-1])
        assert torch.equal(blob, eager_blob)


def _precopy_lq(blob, block_table, dst_col, src_col, token_bias, tiles=4):
    """Run the align pre-copy kernel with one LeapQuant state tensor."""
    dev = blob.device
    i64 = dict(dtype=torch.int64, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    _ops().precopy_mamba_align_fused_rdna2(
        dst_col,
        src_col,
        token_bias,
        torch.tensor([block_table.data_ptr()], **i64),
        block_table.stride(0),
        torch.tensor([blob.data_ptr()], **i64),
        torch.tensor([blob.stride(0)], **i64),
        torch.tensor([1], **i32),
        torch.tensor([blob[0].numel()], **i64),
        torch.tensor([-1], **i32),
        torch.tensor([0], **i32),
        torch.tensor([blob.shape[1]], **i32),
        torch.tensor([blob.shape[2] - lq.LEAPQUANT_HEADER_BYTES], **i64),
        None,
        dst_col.shape[0],
        False,
        tiles,
    )


def test_align_precopy_commits_accepted_records():
    """Migrating a running home blob to the next block column must produce
    exactly the state a verify with num_accepted = token_bias + 1 reads."""
    L = _layout(3)
    A_log, dt_bias = _params(seed=11)
    n = 2
    blob = torch.zeros(2 * n + 1, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    blob[n + 1 :].random_(0, 255)  # destination blocks hold stale bytes
    block_table = torch.zeros(n, 4, dtype=torch.int32, device="cuda")
    block_table[:, 0] = torch.arange(1, n + 1, dtype=torch.int32)
    block_table[:, 1] = torch.arange(n + 1, 2 * n + 1, dtype=torch.int32)
    home = block_table[:, 0]
    _store(L, torch.randn(n, HV, V, K, device="cuda") * 0.5, blob, home.contiguous())
    for step in range(4):
        q, k, v, a, b = _tokens(n, 3, seed=40 + step)
        _normalize(L, blob, home, torch.full((n,), 3, dtype=torch.int32, device="cuda"))
        _decode(L, blob, home, q, k, v, a, b, A_log, dt_bias)
    src_before = blob[1 : n + 1].clone()

    bias = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    _precopy_lq(
        blob,
        block_table,
        dst_col=torch.ones(n, dtype=torch.int32, device="cuda"),
        src_col=torch.zeros(n, dtype=torch.int32, device="cuda"),
        token_bias=bias,
    )
    assert torch.equal(blob[1 : n + 1], src_before), "source home block modified"
    for i in range(n):
        ref = lq.leapquant_copy_slot(L, src_before[i], int(bias[i]))
        assert torch.equal(blob[n + 1 + i], ref)
    # The migrated blob is the state the verify would have read in place.
    expect = src_before.clone()
    lq.leapquant_normalize(L, expect, bias.long() + 1)
    got = _materialize(L, blob, block_table[:, 1].contiguous(), n)
    assert _rel(got, lq.leapquant_materialize(L, expect)) < 1e-5
