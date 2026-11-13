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


def test_grouped_normalize_matches_per_layer():
    """One launch over a group of layers (layer_offsets) must leave every
    layer's blobs exactly as normalizing each layer on its own does, folds
    and NULL slots included."""
    L = _layout(3)
    A_log, dt_bias = _params()
    n, layers = 3, 3
    idx = torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda")
    blobs = []
    for layer in range(layers):
        blob = torch.zeros(n, HV, L.nbytes, dtype=torch.uint8, device="cuda")
        blob[0].fill_(7)
        _store(L, torch.randn(n, HV, V, K, device="cuda") * 0.5, blob, idx)
        # 8 steps commit 7 * 2 = 14 records; committing 3 (request 0) or 2
        # (request 2) more below reaches the window and folds.
        for step in range(8):
            q, k, v, a, b = _tokens(n, 3, seed=100 * layer + step)
            _normalize(L, blob, idx, torch.full((n,), 2, dtype=torch.int32).cuda())
            _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias)
        blobs.append(blob)
    num_acc = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    ref = [blob.clone() for blob in blobs]
    for blob in ref:
        _normalize(L, blob, idx, num_acc)
    base = blobs[0].data_ptr()
    offsets = torch.tensor(
        [blob.data_ptr() - base for blob in blobs], dtype=torch.int64, device="cuda"
    )
    _ops().gdn_lq_normalize_rdna2(
        blobs[0], idx, num_acc, V, R, L.capacity, WINDOW, ITERS, offsets
    )
    folded = 0
    for got, want in zip(blobs, ref):
        assert torch.equal(got, want)
        cursor = L.views(got[1:])["header"][..., 0]
        folded += int((cursor < 8).sum())
    assert folded > 0, "no fold exercised"


def test_normalize_groups_follow_kv_cache_groups():
    """Hybrid models spread their GDN layers over several KV-cache groups with
    separate block tables, so one request sits at different slots per group.
    The shared normalize launch must cover only its own group's layers: the
    layers normalized group by group in model order must end exactly as
    normalizing every layer on its own at its group's slots."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        share_leapquant_normalize_groups,
    )
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    L = _layout(3)
    A_log, dt_bias = _params()
    n, slots = 2, 5
    names = [f"model.layers.{i}.linear_attn" for i in range(4)]
    # group 0 = layers 0, 2; group 1 = layers 1, 3; request r at different slots
    group_of = {name: i % 2 for i, name in enumerate(names)}
    group_idx = [
        torch.tensor([1, 3], dtype=torch.int32, device="cuda"),
        torch.tensor([4, 2], dtype=torch.int32, device="cuda"),
    ]
    ctx = {}
    for i, name in enumerate(names):
        idx = group_idx[group_of[name]]
        blob = torch.zeros(slots, HV, L.nbytes, dtype=torch.uint8, device="cuda")
        _store(L, torch.randn(n, HV, V, K, device="cuda") * 0.5, blob, idx)
        # 8 steps leave 14 committed + 3 pending records: the commit folds
        for step in range(8):
            q, k, v, a, b = _tokens(n, 3, seed=100 * i + step)
            _normalize(L, blob, idx, torch.full((n,), 2, dtype=torch.int32).cuda())
            _decode(L, blob, idx, q, k, v, a, b, A_log, dt_bias)
        layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
        torch.nn.Module.__init__(layer)
        layer._lq_layout = L
        layer.kv_cache = (None, blob)
        ctx[name] = layer
    groups = [
        KVCacheGroupSpec([n_ for n_ in names if group_of[n_] == g], None)
        for g in range(2)
    ]
    share_leapquant_normalize_groups(names, ctx, groups)
    assert ctx[names[0]]._lq_group is ctx[names[2]]._lq_group
    assert ctx[names[0]]._lq_group is not ctx[names[1]]._lq_group

    num_acc = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    ref = []
    for name in names:
        blob = ctx[name].kv_cache[1].clone()
        _normalize(L, blob, group_idx[group_of[name]], num_acc)
        ref.append(blob)
    for name in names:
        layer = ctx[name]
        layer._lq_normalize(layer.kv_cache[1], group_idx[group_of[name]], num_acc)
    for name, want in zip(names, ref):
        assert torch.equal(ctx[name].kv_cache[1], want), name


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


def _mono_inputs(m, conv_len, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cd = 2 * H * K + HV * V

    def rn(*s, std):
        return (torch.randn(*s, generator=g) * std).half().cuda()

    qkvz = rn(m, cd + HV * V, std=0.5)
    ba = rn(m, 2 * HV, std=1.0)
    conv = rn(SLOTS_MONO, cd, conv_len, std=0.5)
    return qkvz, ba, conv


SLOTS_MONO = 24


@pytest.mark.parametrize("spec", [False, True])
def test_mono_gdn_leapquant_matches_fp32_mono(spec):
    """The MonoKernel GDN on LeapQuant blobs (mono_gdn_lq_rdna2) is the FP32
    MonoKernel (mono_gdn_rdna2) run on the materialized state: same outputs
    and conv state, and every per-token state the FP32 kernel stores is the
    blob with that many records committed. Two steps, so the second replays
    the first one's records."""
    t = 3 if spec else 1
    L = _layout(t)
    n = 4
    cd = 2 * H * K + HV * V
    A_log, dt_bias = _params(5)
    g = torch.Generator(device="cpu").manual_seed(6)
    conv_w = (torch.randn(cd, 4, generator=g) * 0.3).half().cuda()
    conv_b = (torch.randn(cd, generator=g) * 0.1).half().cuda()
    norm_w = (1.0 + torch.randn(V, generator=g) * 0.1).half().cuda()
    perm = torch.randperm(SLOTS_MONO - 1, generator=g)[: n * t] + 1
    sidx = perm.int().view(n, t).cuda() if spec else perm.int().cuda()
    sidx[1] = 0  # null / padded sequence
    home = (sidx[:, 0] if spec else sidx).contiguous()
    blob = torch.zeros(SLOTS_MONO, HV, L.nbytes, dtype=torch.uint8, device="cuda")
    live = home[home > 0]
    _store(L, torch.randn(live.numel(), HV, V, K, device="cuda") * 0.3, blob, live)
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    qsl = torch.arange(0, n * t + 1, t, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(
        int(_ops().mono_gdn_rdna2_scratch_bytes(64, HV, H)),
        dtype=torch.uint8,
        device="cuda",
    )
    nacc = None
    for step in range(2):
        qkvz, ba, conv = _mono_inputs(n * t, 3 + t - 1, seed=60 + step)
        if spec:
            nacc = torch.tensor([3, 1, 2, 1], dtype=torch.int32, device="cuda")
            if step == 0:
                nacc.fill_(1)  # fresh blobs hold no pending records
        _normalize(L, blob, home, nacc)
        # FP32 reference: the committed home state at the slot it reads.
        ssm = torch.zeros(SLOTS_MONO, HV, V, K, device="cuda")
        if nacc is not None:
            read = sidx[torch.arange(n, device="cuda"), nacc.long() - 1]
        else:
            read = sidx
        ssm[read.long()] = _materialize(L, blob, home, n)
        ssm[0] = 0
        conv_ref, y_ref = conv.clone(), torch.empty(n * t, HV * V).half().cuda()
        y = torch.empty_like(y_ref)
        common = (A_log.float(), dt_bias.float(), norm_w, sidx)
        tail = (empty if not spec else nacc, empty if not spec else qsl)
        _ops().mono_gdn_rdna2(
            qkvz, ba, conv_w, conv_b, conv_ref, ssm, *common, *tail, y_ref,
            scratch, H, K**-0.5, 1e-6, 0,
        )  # fmt: skip
        _ops().mono_gdn_lq_rdna2(
            qkvz, ba, conv_w, conv_b, conv, blob, *common, *tail, y,
            scratch, H, K**-0.5, 1e-6, 0, R, L.capacity, WINDOW,
        )  # fmt: skip
        torch.accelerator.synchronize()
        assert torch.equal(conv, conv_ref)
        rows = (home > 0).repeat_interleave(t)
        assert bool((y[~rows] == 0).all())
        assert _rel(y[rows], y_ref[rows]) < 2e-3
        # Each per-token FP32 state is the blob with t + 1 records committed.
        for acc in range(1, t + 1):
            trial = blob.clone()
            _normalize(
                L,
                trial,
                home,
                torch.full((n,), acc, dtype=torch.int32, device="cuda")
                if spec
                else None,
            )
            got = _materialize(L, trial, home, n)
            want = ssm[(sidx[:, acc - 1] if spec else sidx).long()]
            live_rows = home > 0
            assert _rel(got[live_rows], want[live_rows]) < 2e-3, (step, acc)
